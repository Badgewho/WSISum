import os
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
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
    base_dir,  # 数据集所在的根目录
    dataset_names,  # 数据集名称列表
    sample_sizes=[500],  # 严格规定的 sample_sizes
    n_clusters=50
):
    """
    base_dir: 你的数据集总文件夹路径，例如 "datasets"
    dataset_names: 需要处理的数据集名称列表
    sample_sizes: 固定抽取数量的列表（如 [250, 500, 1000]）
    n_clusters: 聚类数量，默认 50
    """
    pca = load_pca_model()
    for dataset_name in dataset_names:
        dataset_path = os.path.join(base_dir, dataset_name)
        
        if not os.path.isdir(dataset_path):
            print(f"[跳过] 数据集文件夹不存在: {dataset_path}")
            continue  # 如果数据集文件夹不存在，跳过

        # TITAN、CONCH 子文件夹的路径
        titan_dir = os.path.join(dataset_path, 'TITAN')
        conch_dir = os.path.join(dataset_path, 'CONCH')
        if not (os.path.isdir(titan_dir) and os.path.isdir(conch_dir)):
            print(f"[跳过] 未找到 TITAN 或 CONCH 子文件夹在: {dataset_path}")
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
            
            # 输出目录
            out_dir = os.path.join("/data/Summary/eval_dataset/MOEwsi500-3", dataset_name, f"cluster{sample_sizes[0]}")
            os.makedirs(out_dir, exist_ok=True)
            out_h5_path = os.path.join(out_dir, slide_file)
            
            # **跳过已处理的文件**：如果文件已经存在，则跳过
            if os.path.exists(out_h5_path):
                print(f"    [跳过] 文件已存在: {out_h5_path}")
                continue

            # 读取 TITAN <slide>.h5 里的单条 feature
            with h5py.File(titan_h5_path, 'r') as f_titan:
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
            if n < n_clusters:
                # 样本数量小于簇数时进行复制粘贴
                times_to_repeat = (n_clusters + n - 1) // n  # 计算需要重复的次数，保证能够填满簇数
                conch_features = np.tile(conch_features, (times_to_repeat, 1))[:n_clusters, :]  # 扩展样本数量
                conch_coords = np.tile(conch_coords, (times_to_repeat, 1))[:n_clusters, :]  # 扩展坐标

                print(f"    [复制] 将样本数量从 {n} 扩展到 {n_clusters} 以满足簇数要求")
                n = n_clusters  # 更新 n 为簇数

            # 使用 KMeans 进行聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
            labels = kmeans.fit_predict(conch_features)  # labels 大小为 n
            # 记录每个簇的样本索引
            cluster_indices = {}
            for i, label in enumerate(labels):
                cluster_indices.setdefault(label, []).append(i)
            
            # **确保 sample_sizes 严格等于 [250, 500, 1000]**
            for sample_size in sample_sizes:
                    # 如果当前 n < sample_size，则需要进行复制粘贴补足
                if n < sample_size:
                    # 计算补充样本数量
                    diff = sample_size - n
                    print(f"    [补充] 当前样本数 {n} 小于 {sample_size}，需要补充 {diff} 个样本")
                    conch_features = np.concatenate([conch_features] * (diff // n + 1), axis=0)[:sample_size, :]
                    conch_coords = np.concatenate([conch_coords] * (diff // n + 1), axis=0)[:sample_size, :]
                    n = sample_size  # 更新 n 为目标样本数量
                
                # 目标抽样总数
                total_required = sample_size
                
                # 统计各簇大小
                cluster_sizes = {cid: len(idxs) for cid, idxs in cluster_indices.items()}
                
                # 先根据簇大小占比进行分配（四舍五入）
                cluster_select = {}
                sum_tmp = 0
                for cid, size in cluster_sizes.items():
                    fraction = size / n
                    cnt = int(round(total_required * fraction))
                    cluster_select[cid] = cnt
                    sum_tmp += cnt
                
                # 微调逻辑，使分配之和正好等于 total_required
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
                
                # 在每个簇中随机抽 cluster_select[cid] 个
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
                # 取出对应的 features, coords
                sampled_features = conch_features[selected_indices, :]
                sampled_coords = conch_coords[selected_indices, :]
                
                # 读取多个模型特征
                # models = ['TITAN', 'CHIEF', 'PRISM', 'Gigapath']
                models = ['TITAN', 'CHIEF', 'PRISM']
                multi_features = []
                for model in models:
                    model_h5 = os.path.join(dataset_path, model, slide_file)
                    if os.path.exists(model_h5):
                        with h5py.File(model_h5, 'r') as f:
                            feat = f["features"][:]  # 应该是 (1, dim)
                            multi_features.append(feat)
                    else:
                        print(f"[警告] 缺失模型文件: {model_h5}")

                # 拼接所有模型特征，得到 (1, total_dim)
                all_wsi_feat = np.concatenate(multi_features, axis=-1)  # (1, dim1+dim2+...)
                all_wsi_feat = all_wsi_feat.reshape(1,-1)
                # 降维至 (1, 768)
                wsi_feature_768 = apply_pca_to_single_sample(all_wsi_feat, pca)

                # 拼接 WSI 到采样特征最后
                combined_features = np.concatenate([sampled_features, wsi_feature_768], axis=0)  # shape: (N+1, 768)
                
                # 写到新的 .h5 文件
                with h5py.File(out_h5_path, 'w') as fout:
                    fout.create_dataset("features", data=combined_features)
                    fout.create_dataset("coords", data=sampled_coords)

                print(f"    -> 抽取 {len(selected_indices)}/{n} 条 (请求: {sample_size}), 写入 {out_h5_path}")

if __name__ == "__main__":
    base_dir = "/data4/embedding"  # 修改为你的实际路径
    dataset_names = ["TCGA-BRCA","BRACS","CAMELYON17"]  # 替换为你实际的多个数据集名称
    cluster_and_select_features(base_dir, dataset_names)
