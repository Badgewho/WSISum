import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 设置设备（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 预测数据路径
predict_features_path = "/home/baizhiwang/Badge/MAMIL/dataset/BRACS/CHIEF_tile"
# 目标路径（不同比例）
target_dirs = {
    0.05: "/home/baizhiwang/Badge/MAMIL/dataset/BRACS/CHIEF_attention0.05",
    0.1: "/home/baizhiwang/Badge/MAMIL/dataset/BRACS/CHIEF_attention0.1",
    0.2: "/home/baizhiwang/Badge/MAMIL/dataset/BRACS/CHIEF_attention0.2",
}

# 确保所有目标路径存在
for path in target_dirs.values():
    os.makedirs(path, exist_ok=True)

# 定义 MLP 模型（需要和训练时的模型结构一致）
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出 1 维
        )
    
    def forward(self, x):
        return self.model(x)

# 加载训练好的模型
input_dim = 768  # 需要确保 input_dim 与训练时一致
model = MLP(input_dim).to(device)
model.load_state_dict(torch.load("/home/baizhiwang/Badge/MAMIL/attention_entro/TITANattentionmlp_model_entro.pth"))
model.eval()
print("Model loaded successfully.")

# 处理预测数据
def process_and_save():
    feature_files = sorted([f for f in os.listdir(predict_features_path) if f.endswith(".h5")])
    
    for file in feature_files:
        file_path = os.path.join(predict_features_path, file)
        
        with h5py.File(file_path, 'r') as f:
            features = np.array(f['features'])  # (n, m)
            coords = np.array(f['coords'])  # (n, 2) 假设 coords 存在
        
        # 转换为 PyTorch 张量
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        
        # 预测 attention
        with torch.no_grad():
            attention = model(features_tensor).cpu().numpy()
        
        # **对不同比例进行筛选**
        for ratio, target_dir in target_dirs.items():
            save_file_path = os.path.join(target_dir, file)
            num_top = max(10, int(len(attention) * ratio))  # 确保至少选 10 个
            
            # **选取 attention 最高的前 num_top 个**
            top_indices = np.argsort(attention[:, 0])[-num_top:]
            
            selected_features = features[top_indices]
            selected_coords = coords[top_indices]
            selected_attention = attention[top_indices]
            
            # **保存到新的 .h5 文件**
            with h5py.File(save_file_path, 'w') as f:
                f.create_dataset('features', data=selected_features)
                f.create_dataset('coords', data=selected_coords)
                f.create_dataset('attention', data=selected_attention)
            
            print(f"[{file}] Ratio: {ratio}, Saved {num_top} samples to {save_file_path}")

# 执行处理
process_and_save()
