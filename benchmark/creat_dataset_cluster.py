import os
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"

import h5py
import numpy as np
import random
from sklearn.cluster import KMeans

# 原始数据路径
source_dir = "/home/baizhiwang/Badge/MAMIL/WSIembedding/LUAD/CONCH"
# 目标路径（固定数量）
target_dirs = {
    62: "/home/baizhiwang/Badge/Summary/dataset/LUAD/CONCH_cluster62",
    125: "/home/baizhiwang/Badge/Summary/dataset/LUAD/CONCH_cluster125",
    250: "/home/baizhiwang/Badge/Summary/dataset/LUAD/CONCH_cluster250",
}

# 确保目标路径存在
for path in target_dirs.values():
    os.makedirs(path, exist_ok=True)

# 获取所有 .h5 文件
h5_files = [f for f in os.listdir(source_dir) if f.endswith('.h5')]

# 聚类的目标数（如果数据量 < 50，会自动减小）
TARGET_NUM_CLUSTERS = 50

for h5_file in h5_files:
    # 检查目标文件夹中是否已存在已处理的文件
    already_processed = False
    for target_dir in target_dirs.values():
        target_path = os.path.join(target_dir, h5_file)
        if os.path.exists(target_path):
            already_processed = True
            break

    if already_processed:
        print(f"Skipping {h5_file}, already processed.")
        continue

    source_path = os.path.join(source_dir, h5_file)

    # 读取 h5 文件
    with h5py.File(source_path, 'r') as f:
        features = f['features'][:]  # (n, m)
        coords = f['coords'][:]      # (n, 2)

    # 确保数据维度匹配
    assert features.shape[0] == coords.shape[0], "Features and coords length mismatch!"
    
    n = features.shape[0]

    # 选择聚类数
    n_clusters = min(TARGET_NUM_CLUSTERS, n)

    # KMeans 聚类（仅执行一次）
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)  # labels: shape=(n,)

    # 按 cluster label 归类样本索引
    cluster_indices = {i: [] for i in range(n_clusters)}
    for i, lbl in enumerate(labels):
        cluster_indices[lbl].append(i)

    # 按固定数量进行筛选并保存
    for sample_size, target_dir in target_dirs.items():
        target_path = os.path.join(target_dir, h5_file)
        n_needed = sample_size  # 使用固定的样本数

        # 计算每个聚类应采样数量
        cluster_sample_nums = []
        for lbl in range(n_clusters):
            cluster_size = len(cluster_indices[lbl])
            cluster_n_needed = int(round((cluster_size / n) * n_needed))
            cluster_sample_nums.append(cluster_n_needed)

        # 调整总数误差
        total_assigned = sum(cluster_sample_nums)
        diff = n_needed - total_assigned

        if diff != 0:
            sorted_clusters = sorted(
                range(n_clusters), 
                key=lambda x: len(cluster_indices[x]), 
                reverse=True
            )
            idx = 0
            while diff != 0:
                c = sorted_clusters[idx]
                if diff > 0:
                    cluster_sample_nums[c] += 1
                    diff -= 1
                elif cluster_sample_nums[c] > 0:
                    cluster_sample_nums[c] -= 1
                    diff += 1
                idx = (idx + 1) % n_clusters  # 轮询调整

        # 最终在每个簇中随机采样
        selected_indices = []
        for lbl in range(n_clusters):
            csize = len(cluster_indices[lbl])
            cneed = cluster_sample_nums[lbl]
            if cneed >= csize:
                selected_indices.extend(cluster_indices[lbl])
            else:
                selected_indices.extend(random.sample(cluster_indices[lbl], cneed))

        # 取对应的 features 和 coords
        selected_features = features[selected_indices]
        selected_coords = coords[selected_indices]

        # 保存到不同目标文件夹
        with h5py.File(target_path, 'w') as f:
            f.create_dataset('features', data=selected_features)
            f.create_dataset('coords', data=selected_coords)

        print(f"[{h5_file}] Target Size: {n_needed}, Selected: {len(selected_indices)} / {n} (Target={n_needed})")

print("All files processed successfully!")
