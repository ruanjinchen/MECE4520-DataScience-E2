import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 定义培育箱数据集 Define data set
class IncubatorXORDataset(Dataset):
    def __init__(self, size, T_range=(10, 30), H_range=(30, 90), T_threshold=25, H_threshold=60):
        self.size = size
        self.T_range = T_range  # Temperature Range
        self.H_range = H_range  # Humidity Range
        self.T_threshold = T_threshold
        self.H_threshold = H_threshold

        # 随机生成温度和湿度 Randomly generated temperature and humidity
        self.temperatures = np.random.uniform(*T_range, size).round(1)  # 温度精度为0.1° Temperature accuracy is 0.1°
        self.humidities = np.random.randint(*H_range, size)            # 湿度精度为1% Humidity accuracy is 1%

        # 计算是否满足条件 Calculate whether the condition is met
        self.temp_flags = (self.temperatures > T_threshold).astype(int)
        self.humid_flags = (self.humidities > H_threshold).astype(int)

        # 计算XOR输出 Calculate the XOR output
        self.labels = (self.temp_flags ^ self.humid_flags).reshape(-1, 1)  # XOR逻辑

        # 数据标准化 Data standardization
        self.X_raw = np.column_stack((self.temperatures, self.humidities))  # 原始数据 Raw data
        self.X_mean = self.X_raw.mean(axis=0)  # 计算均值 Calculated mean
        self.X_std = self.X_raw.std(axis=0)    # 计算标准差 Calculated standard deviation
        self.X_normalized = (self.X_raw - self.X_mean) / self.X_std  # 标准化

        # 转为PyTorch张量 Convert to PyTorch tensor
        self.X = torch.tensor(self.X_normalized, dtype=torch.float32)
        self.y = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 生成4000组训练集和10组测试集
train_dataset = IncubatorXORDataset(size=4000)
test_dataset = IncubatorXORDataset(size=100)

# 保存数据集到CSV文件
def save_dataset(dataset, file_name):
    # 保存原始值和标签
    data = np.column_stack((dataset.X_raw, dataset.labels))
    df = pd.DataFrame(data, columns=["Temperature", "Humidity", "Label"])
    df.to_csv(file_name, index=False)
    print(f"Dataset saved to {file_name}")

save_dataset(train_dataset, "train_dataset.csv")
save_dataset(test_dataset, "test_dataset.csv")

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 打印部分数据
print("Example training data:")
for data, label in train_loader:
    print("Normalized Data:", data)
    print("Label:", label)
    break
