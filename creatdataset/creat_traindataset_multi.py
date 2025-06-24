import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
import random
import h5py
import numpy as np
from sklearn.cluster import KMeans
import pickle



def load_pca_model(pca_model_path="/data/Summary/train_PCA/pca_model3.pkl"):
    """
    读取保存的 PCA 模型。
    
    pca_model_path: 保存的 PCA 模型路径
    """
    with open(pca_model_path, 'rb') as f:
        pca = pickle.load(f)
    return pca

def apply_pca_to_single_sample(sample_features, pca):
    """
    使用 PCA 模型对单个样本进行降维。
    
    sample_features: 需要降维的单个样本特征，形状为 (1, n_features)
    pca: 已加载的 PCA 模型
    """
    reduced_features = pca.transform(sample_features)  # 对单个样本进行降维
    return reduced_features

    
def cluster_and_select_features(
    base_dir,
    sample_sizes=[500],
    n_clusters=50
):
    """
    base_dir: 你的数据集总文件夹路径，例如 "datasets"
    sample_sizes: 固定抽取数量的列表（如 [250, 500, 1000]）
    n_clusters: 聚类数量，默认 50
    """
    pca = load_pca_model()
    # 遍历 base_dir 下的所有子文件夹（即各 dataset 文件夹）
    for dataset_name in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue  # 只关心文件夹
        
        # TITAN、CONCH 子文件夹的路径
        titan_dir = os.path.join(dataset_path, 'TITAN')
        conch_dir = os.path.join(dataset_path, 'CONCH')
        if not (os.path.isdir(titan_dir) and os.path.isdir(conch_dir)):
            continue  # 没有这两个文件夹则跳过
        
        # 遍历 TITAN 文件夹（或者 CONCH 文件夹），找到所有 <slide>.h5
        slides = [f for f in os.listdir(titan_dir) if f.endswith('.h5')]
        
        for slide_file in slides:
            # 同名文件应当在 CONCH 中也存在
            titan_h5_path = os.path.join(titan_dir, slide_file)
            conch_h5_path = os.path.join(conch_dir, slide_file)
            if not os.path.isfile(conch_h5_path):
                print(f"[跳过] CONCH 下未找到对应文件: {conch_h5_path}")
                continue
            
            print(f"\n处理文件: {dataset_name}/{slide_file}")
            
            # 读取 TITAN <slide>.h5 里的单条 feature
            with h5py.File(titan_h5_path, 'r') as f_titan:
                # 假设原始 dataset 名称是 "features"（若不是，请自行修改）
                # 这里读取 (1, 768)
                titan_features = f_titan["features"][:]  
            
            # 读取 CONCH <slide>.h5 里的 features 和 coords
            with h5py.File(conch_h5_path, 'r') as f_conch:
                conch_features = f_conch["features"][:]  # (n, 768)
                conch_coords = f_conch["coords"][:]      # (n, 2)
            
            n = conch_features.shape[0]
            if n == 0:
                print(f"[跳过] CONCH features 数量为 0: {slide_file}")
                continue
            
            # 如果实际样本数量小于想要聚的簇数，会报错，这里做个简单处理
            actual_n_clusters = min(n, n_clusters)
            
            # 使用 KMeans 进行聚类
            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=0, n_init="auto")
            labels = kmeans.fit_predict(conch_features)  # labels 大小为 n
            # 记录每个簇的样本索引
            cluster_indices = {}
            for i, label in enumerate(labels):
                cluster_indices.setdefault(label, []).append(i)
            
            # 对于需要抽取的每一个固定数量 sample_size
            for sample_size in sample_sizes:
                # 如果样本太少，不处理
                if n < sample_size // 2:
                    print(f"    [跳过] n={n} < sample_size的一半={sample_size // 2}")
                    continue

                #----------------------
                # 1) 根据原始 n (未扩充) 做 KMeans 聚类（上面已做了）
                #    labels = kmeans.fit_predict(conch_features)
                #    cluster_indices = { ... }  # 省略，和你原逻辑相同
                #----------------------

                # 目标抽样总数
                total_required = sample_size

                # 2) 计算各簇大小
                cluster_sizes = {cid: len(idxs) for cid, idxs in cluster_indices.items()}

                # 3) 按簇大小占比来分配要抽的数量
                cluster_select = {}
                sum_tmp = 0
                for cid, size in cluster_sizes.items():
                    fraction = size / n  # 注意这里用的是原始 n
                    cnt = int(round(total_required * fraction))
                    cluster_select[cid] = cnt
                    sum_tmp += cnt

                # 4) 微调逻辑，使分配之和正好等于 total_required
                diff = total_required - sum_tmp
                cids = list(cluster_select.keys())

                while diff != 0:
                    if diff > 0:
                        # 需要补若干
                        cid = random.choice(cids)
                        cluster_select[cid] += 1
                        diff -= 1
                    else:
                        # diff < 0，需要减少
                        candidates = [c for c in cids if cluster_select[c] > 0]
                        if not candidates:
                            break
                        cid = random.choice(candidates)
                        cluster_select[cid] -= 1
                        diff += 1

                # 5) 在每个簇中随机抽 cluster_select[cid] 个
                selected_indices = []
                for cid, need in cluster_select.items():
                    all_idxs = cluster_indices[cid]
                    size = len(all_idxs)
                    if need >= size:
                        # 需要量 >= 簇大小，就全拿
                        selected_indices.extend(all_idxs)
                    else:
                        # 随机抽 need 个
                        sel = random.sample(all_idxs, need)
                        selected_indices.extend(sel)

                selected_indices = sorted(selected_indices)
                # 实际抽到的数量
                final_count = len(selected_indices)

                # 6) 如果抽到的样本数仍然小于 sample_size，则进行扩充
                if final_count < sample_size:
                    print(f"    [扩充] 抽样后数量: {final_count}，需要: {sample_size}")
                    repeat_factor = sample_size // final_count
                    remainder = sample_size % final_count
                    # 这里是“索引”的扩充
                    selected_indices = selected_indices * repeat_factor + selected_indices[:remainder]
                    final_count = len(selected_indices)  # 此时应当是 sample_size

                # 7) 根据最终选定的索引拿到特征和坐标
                selected_features = conch_features[selected_indices, :]
                selected_coords = conch_coords[selected_indices, :]

                # 8) 读取多个模型特征，拼接并降维（你的原逻辑不变）
                # models = ['TITAN', 'CHIEF', 'PRISM', 'Gigapath']
                models =['TITAN','CHIEF','PRISM']
                # models =['TITAN','CHIEF','PRISM']
                multi_features = []
                for model in models:
                    model_h5 = os.path.join(dataset_path, model, slide_file)
                    if os.path.exists(model_h5):
                        with h5py.File(model_h5, 'r') as f:
                            feat = f["features"][:]  # (1, dimX)
                            multi_features.append(feat)
                    else:
                        print(f"[警告] 缺失模型文件: {model_h5}")

                all_wsi_feat = np.concatenate(multi_features, axis=-1)  # (1, sum_of_dims)
                wsi_feature_768 = apply_pca_to_single_sample(all_wsi_feat, pca)  # (1,768)

                # 9) 将 "WSI 特征" 拼在“已采样patch”特征的末尾
                combined_features = np.concatenate([selected_features, wsi_feature_768], axis=0)
                # selected_coords 对应刚才采样到的 patch 数量
                # 对于 WSI 特征，没有坐标，这里若你想保持 coords 维度一致，就写法上要么不加 coords，要么随便补个 [[-1, -1]] 等

                # 10) 保存到对应的 .h5
                out_dir = os.path.join("/data/Summary/dataset/MultiFM500-TITAN+CHIEF+PRISM", 
                                    dataset_name, f"cluster{sample_size}")
                os.makedirs(out_dir, exist_ok=True)
                out_h5_path = os.path.join(out_dir, slide_file)

                if os.path.exists(out_h5_path):
                    print(f"    [跳过] 文件已存在: {out_h5_path}")
                    continue

                with h5py.File(out_h5_path, 'w') as fout:
                    fout.create_dataset("features", data=combined_features)
                    fout.create_dataset("coords", data=selected_coords)
                    # 如果想把WSI特征在coords里也加一行占位，可以手动做:
                    # new_coords = np.vstack([selected_coords, [-1, -1]])
                    # fout.create_dataset("coords", data=new_coords)

                print(f"    -> 抽取 {final_count}/{n} 条 (请求: {sample_size}), 写入 {out_h5_path}")



if __name__ == "__main__":
    base_dir = "/data4/embedding"  # 修改为你的实际路径
    cluster_and_select_features(base_dir)
