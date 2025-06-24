import os
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
import h5py
import numpy as np
from sklearn.cluster import KMeans
import random

def cluster_and_select_features(
    base_dir,
    proportions=[0.05, 0.1, 0.2],
    n_clusters=50,
    min_select=20
):
    """
    base_dir: 你的数据集总文件夹路径，例如 "datasets"
    proportions: 要抽取的比例
    n_clusters: 聚类数量，默认 50
    min_select: 每次选取最少的条数，默认 20
    """
    
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
                print(f"CONCH 下未找到对应文件: {conch_h5_path}，跳过")
                continue
            
            print(f"处理文件: {dataset_name}/{slide_file}")
            
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
                print(f"CONCH features 数量为 0，跳过 {slide_file}")
                continue
            
            # 如果实际样本数量小于想要聚的簇数，会出错
            # 这里简单处理一下，若 n < n_clusters，则将 n_clusters 调整为 n
            actual_n_clusters = min(n, n_clusters)
            
            # 使用 KMeans 进行聚类
            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=0, n_init="auto")
            labels = kmeans.fit_predict(conch_features)  # labels 大小为 n
            
            # 统计每个簇的索引
            cluster_indices = {}
            for i, label in enumerate(labels):
                cluster_indices.setdefault(label, []).append(i)
            
            # 对 proportions 中的每个比例，做抽样并写文件
            for prop in proportions:
                out_dir = os.path.join("/home/baizhiwang/Summary/dataset", f"cluster{prop}")
                os.makedirs(out_dir, exist_ok=True)
                
                out_h5_path = os.path.join(out_dir, slide_file)
                
                # 计算本次需要的总抽样数量
                total_required = int(n * prop)
                total_required = max(min_select, total_required)
                
                # 分配到各簇的数量，先根据簇大小占比进行四舍五入
                selected_indices = []
                
                # 先统计各簇大小
                cluster_sizes = {cid: len(idxs) for cid, idxs in cluster_indices.items()}
                # 注意：如果实际聚了 < 50 个簇（当 n < 50 时），这里 cid 的范围并不一定是 0..49
                
                # 初步按占比计算要在此簇抽多少
                cluster_select = {}
                
                # 防止按占比加和后不等于 total_required，需要微调
                sum_tmp = 0
                for cid, size in cluster_sizes.items():
                    # cluster fraction
                    fraction = size / n
                    # initial count
                    cnt = int(round(total_required * fraction))
                    cluster_select[cid] = cnt
                    sum_tmp += cnt
                
                # 微调逻辑
                diff = total_required - sum_tmp
                # 如果 diff > 0 表示还需要补足若干；若 diff < 0 表示需要减少若干
                cids = list(cluster_select.keys())
                
                while diff != 0:
                    if diff > 0:
                        # 找到还可以 +1 的簇
                        # 随机找一个簇加 1（也可以按簇大小或剩余潜力等策略）
                        cid = random.choice(cids)
                        cluster_select[cid] += 1
                        diff -= 1
                    else:
                        # diff < 0，需要 -1
                        # 随机找一个当前 >0 的簇减 1
                        candidates = [c for c in cids if cluster_select[c] > 0]
                        if not candidates:
                            # 如果没有可以减少的就跳出（极端情况下）
                            break
                        cid = random.choice(candidates)
                        cluster_select[cid] -= 1
                        diff += 1
                
                # 现在各个簇要选的数量在 cluster_select 里
                # 在簇里随机抽取对应数量的索引
                for cid, size in cluster_sizes.items():
                    need = cluster_select[cid]
                    all_idxs = cluster_indices[cid]
                    if need >= size:
                        # 如果需要抽的数量 >= 该簇大小，就全部选
                        selected_indices.extend(all_idxs)
                    else:
                        # 否则随机从该簇中抽 need 个
                        sel = random.sample(all_idxs, need)
                        selected_indices.extend(sel)
                
                selected_indices = sorted(selected_indices)
                # 取出对应的 features, coords
                sampled_features = conch_features[selected_indices, :]
                sampled_coords = conch_coords[selected_indices, :]
                
                # 将选出的 features、coords + TITAN feature (WSIfeatures) 写入新的 h5
                with h5py.File(out_h5_path, 'w') as fout:
                    # 存放抽样后的 features, coords
                    fout.create_dataset("features", data=sampled_features)
                    fout.create_dataset("coords", data=sampled_coords)
                    # 存放 TITAN 单条特征，改名为 "WSIfeatures"
                    fout.create_dataset("WSIfeatures", data=titan_features)
                
                print(f"    -> {prop} 比例: 抽取 {len(selected_indices)} 条数据, 写入 {out_h5_path}")

if __name__ == "__main__":
    # 示例：base_dir 指向所有数据集所在的总文件夹
    base_dir = "/data4/embedding"  # 修改为你的实际路径
    cluster_and_select_features(base_dir)
