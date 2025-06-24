import os
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
import h5py
import random

# 原始数据路径
source_dir = "/home/baizhiwang/Badge/MAMIL/WSIembedding/TCGA-BRCA/CONCH"
# 目标路径（固定数量）
target_dirs = {
    62: "/home/baizhiwang/Badge/Summary/dataset/TCGA-BRCA/CONCH_random62",
    125: "/home/baizhiwang/Badge/Summary/dataset/TCGA-BRCA/CONCH_random125",
    250: "/home/baizhiwang/Badge/Summary/dataset/TCGA-BRCA/CONCH_random250",
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
        features = f['features'][:]  # 读取数据 (n, m)
        coords = f['coords'][:]      # 读取坐标 (n, 2)

    # 确保数据维度匹配
    assert features.shape[0] == coords.shape[0], "Features and coords length mismatch!"
    
    num_samples = features.shape[0]

    # **按固定数量进行随机抽样**
    for num_selected, target_dir in target_dirs.items():
        target_path = os.path.join(target_dir, h5_file)

        # 确保至少选 10 个, 并且选取的样本数不超过总样本数
        num_selected = max(10, min(num_selected, num_samples))

        # **随机抽样**
        selected_indices = random.sample(range(num_samples), num_selected)

        # 提取对应的 features 和 coords
        selected_features = features[selected_indices]
        selected_coords = coords[selected_indices]

        # **保存到新 h5 文件**
        with h5py.File(target_path, 'w') as f:
            f.create_dataset('features', data=selected_features)
            f.create_dataset('coords', data=selected_coords)

        print(f"[{h5_file}] Selected: {num_selected} samples out of {num_samples}")

print("All files processed successfully!")
