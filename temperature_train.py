import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt


# 定义数据集类 Define the data set class
class IncubatorDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        # 提取特征和标签
        self.X = torch.tensor(data.iloc[:, :2].values, dtype=torch.float32)
        self.y = torch.tensor(data.iloc[:, 2].values, dtype=torch.float32).view(-1, 1)

        # 对特征进行标准化
        self.mean = self.X.mean(dim=0, keepdim=True)
        self.std = self.X.std(dim=0, keepdim=True)
        self.X = (self.X - self.mean) / (self.std + 1e-7)  # 防止除0错误

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 加载数据
train_dataset = IncubatorDataset("train_dataset.csv")
test_dataset = IncubatorDataset("test_dataset.csv")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 定义网络结构（增加隐藏层神经元数，使用ReLU）
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 16)  # 增加隐藏层单元数
        self.output = nn.Linear(16, 1)
        self.activation_hidden = nn.ReLU()
        self.activation_output = nn.Identity()  # 不使用激活，直接传递logits给BCEWithLogitsLoss

    def forward(self, x):
        x = self.activation_hidden(self.hidden(x))
        x = self.activation_output(self.output(x))
        return x


# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 模型、损失函数、优化器
net = XORNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 检查模型是否在 GPU 上
print("Model is on device:", next(net.parameters()).device)

# ===================== 实时绘图准备 =====================
plt.ion()  # 打开交互模式
fig, ax = plt.subplots()
loss_values = []
epoch_values = []
plot_interval = 10  # 每10个epoch更新一次图表

# ===================== 训练 =====================
epochs = 2000
for epoch in range(1, epochs + 1):
    net.train()
    total_loss = 0.0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)

        # 前向传播
        outputs = net(data)
        loss = criterion(outputs, label)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_values.append(avg_loss)
    epoch_values.append(epoch)

    # 每隔一定epoch更新一次图表
    if epoch % plot_interval == 0 or epoch == 1:
        ax.clear()
        ax.plot(epoch_values, loss_values, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss over Epochs')
        ax.legend()
        ax.grid(True)
        plt.pause(0.01)  # 短暂暂停以更新图表

    # 每100个epoch打印一次损失
    if epoch % 100 == 0 or epoch == 1:
        print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}')

# 训练结束后关闭实时交互
plt.ioff()
plt.show()

# ===================== 测试 =====================
net.eval()  # 设置为评估模式
with torch.no_grad():
    correct = 0
    total = len(test_dataset)
    for idx, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)

        # 前向传递
        output = net(data)
        predicted = (torch.sigmoid(output) > 0.5).float()  # 使用sigmoid对logits进行转化

        # 打印每个样本的结果
        print(f"Sample {idx + 1}:")
        print(f"  Input (normalized): {data.cpu().numpy()}")
        print(f"  Predicted: {predicted.cpu().numpy()}, Truth: {label.cpu().numpy()}")
        print(f"  Correct: {'Yes' if predicted == label else 'No'}\n")

        correct += (predicted == label).sum().item()

    # 打印总体准确率
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
