import os
import h5py
import numpy as np
import random

# 原始数据路径
source_dir = "/home/baizhiwang/Badge/MAMIL/WSIembedding/LUAD/CONCH"
# 目标路径（固定数量）
target_dirs = {
    62: "/home/baizhiwang/Badge/Summary/dataset/LUAD/CONCH_location62",
    125: "/home/baizhiwang/Badge/Summary/dataset/LUAD/CONCH_location125",
    250: "/home/baizhiwang/Badge/Summary/dataset/LUAD/CONCH_location250",
}
# 确保所有目标路径存在
for path in target_dirs.values():
    os.makedirs(path, exist_ok=True)

# 获取所有 .h5 文件
h5_files = [f for f in os.listdir(source_dir) if f.endswith('.h5')]

# 设定网格划分参数
GRID_SIZE = 100  # 划分网格数，可调整

for h5_file in h5_files:
    source_path = os.path.join(source_dir, h5_file)

    # 读取 h5 文件
    with h5py.File(source_path, 'r') as f:
        features = f['features'][:]  # (n, m)
        coords = f['coords'][:]      # (n, 2)

    # 确保数据维度匹配
    assert features.shape[0] == coords.shape[0], "Features and coords length mismatch!"
    
    num_samples = features.shape[0]

    # 获取 x, y 的范围
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # 计算网格大小
    x_step = (x_max - x_min) / GRID_SIZE
    y_step = (y_max - y_min) / GRID_SIZE

    # 存储网格数据
    grid_dict = {}

    # 遍历所有坐标，将点归入对应的网格
    for i, (x, y) in enumerate(coords):
        x_idx = int((x - x_min) / x_step)
        y_idx = int((y - y_min) / y_step)
        grid_key = (x_idx, y_idx)
        
        if grid_key not in grid_dict:
            grid_dict[grid_key] = []
        grid_dict[grid_key].append(i)

    # **对不同比例进行采样**
    for num_selected, target_dir in target_dirs.items():
        target_path = os.path.join(target_dir, h5_file)

        # 确保至少选 10 个
        num_selected = max(10, num_selected)

        # 计算每个网格应选取的数量
        per_grid_sample = max(1, num_selected // len(grid_dict))  # 计算每个网格的目标采样数

        selected_indices = []

        # **从网格中均匀选取样本**
        for grid_key, indices in grid_dict.items():
            if len(indices) <= per_grid_sample:
                selected_indices.extend(indices)  # 网格内点数较少，全部选择
            else:
                selected_indices.extend(random.sample(indices, per_grid_sample))  # 按比例选取

        # **调整样本数量**，确保最终选择的点数与 num_selected 一致
        if len(selected_indices) > num_selected:
            selected_indices = random.sample(selected_indices, num_selected)  # 超过目标数量，随机删减
        elif len(selected_indices) < num_selected:
            remaining_indices = list(set(range(num_samples)) - set(selected_indices))  # 获取未选的点
            remaining_sample_count = num_selected - len(selected_indices)

            # Ensure we don't run out of remaining samples
            remaining_sample_count = min(remaining_sample_count, len(remaining_indices))

            # 额外补充
            additional_samples = random.sample(remaining_indices, remaining_sample_count)
            selected_indices.extend(additional_samples)  # 额外补充

        # 提取对应的 features 和 coords
        selected_features = features[selected_indices]
        selected_coords = coords[selected_indices]

        # 保存到新 h5 文件
        with h5py.File(target_path, 'w') as f:
            f.create_dataset('features', data=selected_features)
            f.create_dataset('coords', data=selected_coords)

        print(f"[{h5_file}] Selected: {len(selected_indices)} samples out of {num_samples} (Target: {num_selected})")

print("All files processed successfully!")
