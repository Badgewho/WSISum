import os
import random
import h5py
import numpy as np
from sklearn.cluster import KMeans

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
            for sample_size in sample_sizes:
                out_dir = os.path.join("/home/baizhiwang/Summary/eval_dataset/TITANwsi", dataset_name, f"cluster{sample_size}")
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
                
                # **检查 coords 的长度，如果不足 500，则进行随机选择填充**
                if n < 500:
                    print(f"    [填充] 当前样本数 {n} 小于 500，进行随机复制填充到 500")
                    diff = 500 - n
                    random_indices = np.random.choice(n, diff, replace=True)
                    conch_features = np.concatenate([conch_features, conch_features[random_indices]], axis=0)
                    conch_coords = np.concatenate([conch_coords, conch_coords[random_indices]], axis=0)
                    n = 500  # 更新样本数量为 500

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
                    
                    # 拼接 WSIfeatures 到 features 的最后
                    combined_features = np.concatenate([sampled_features, titan_features], axis=0)  # 沿特征维度拼接
                    
                    # 写到新的 .h5 文件
                    with h5py.File(out_h5_path, 'w') as fout:
                        fout.create_dataset("features", data=combined_features)
                        fout.create_dataset("coords", data=sampled_coords)

                    print(f"    -> 抽取 {len(selected_indices)}/{n} 条 (请求: {sample_size}), 写入 {out_h5_path}")

if __name__ == "__main__":
    base_dir = "/data4/embedding"  # 修改为你的实际路径
    dataset_names = ["BCNB","TCGA-BRCA", "BRACS", "CPTAC", "CAMELYON17", "Post-NAT-BRCA", "DORID"]  # 替换为你实际的多个数据集名称
    cluster_and_select_features(base_dir, dataset_names)
