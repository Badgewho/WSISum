import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
import h5py
import numpy as np
import scipy.stats

# 原始数据路径
source_dir = "/home/baizhiwang/Badge/MAMIL/WSIembedding/LUAD/CONCH"
# 目标路径（固定数量）
target_dirs = {
    62: "/home/baizhiwang/Badge/Summary/dataset/LUAD/CONCH_entropy62",
    125: "/home/baizhiwang/Badge/Summary/dataset/LUAD/CONCH_entropy125",
    250: "/home/baizhiwang/Badge/Summary/dataset/LUAD/CONCH_entropy250",
}

# 确保所有目标路径存在
for path in target_dirs.values():
    os.makedirs(path, exist_ok=True)

# 获取所有 .h5 文件
h5_files = [f for f in os.listdir(source_dir) if f.endswith('.h5')]

for h5_file in h5_files:
    source_path = os.path.join(source_dir, h5_file)

    # 读取 h5 文件
    with h5py.File(source_path, 'r') as f:
        features = f['features'][:]  # (n, m)
        coords = f['coords'][:]      # (n, 2)

    # 确保数据维度匹配
    assert features.shape[0] == coords.shape[0], "Features and coords length mismatch!"
    
    num_samples = features.shape[0]

    # **计算每个 feature 的 entropy**
    entropy_values = np.apply_along_axis(lambda x: scipy.stats.entropy(np.abs(x) + 1e-8), axis=1, arr=features)

    # **按 entropy 从大到小排序**
    sorted_indices = np.argsort(entropy_values)[::-1]  # 逆序排列（高 entropy 在前）

    # **按固定数量进行筛选**
    for num_selected, target_dir in target_dirs.items():
        target_path = os.path.join(target_dir, h5_file)

        # 确保至少选 10 个
        num_selected = max(10, num_selected)

        # 选取前 num_selected 个 entropy 最高的样本
        selected_indices = sorted_indices[:num_selected]

        # 提取对应的 features 和 coords
        selected_features = features[selected_indices]
        selected_coords = coords[selected_indices]

        # **保存到新 h5 文件**
        with h5py.File(target_path, 'w') as f:
            f.create_dataset('features', data=selected_features)
            f.create_dataset('coords', data=selected_coords)

        print(f"[{h5_file}] Selected: {num_selected} samples out of {num_samples} (Target={num_selected})")

print("All files processed successfully!")
