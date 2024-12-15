import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# 定义数据集类
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

# Define network structure
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 16)  # 2 features, 16 neurons in hidden layers
        self.output = nn.Linear(16, 1)  # 16 input features in output layer, 1 output in output layer
        self.activation_hidden = nn.ReLU()  # Rectified Linear Unit
        self.activation_output = nn.Identity()  
        # Pass logits directly to BCEWithLogitsLoss without activation

    def forward(self, x):
        x = self.activation_hidden(self.hidden(x))
        x = self.activation_output(self.output(x))
        return x

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Models, loss functions, optimizers
net = XORNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# hyper-parameter
epochs = 2000
batch_losses = []
update_interval = 100

# 开启交互模式
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Batch count')
ax.set_ylabel('Loss')
ax.set_title('Training Loss per Batch')
ax.grid(True)

batch_count = 0
for epoch in range(1, epochs + 1):
    net.train()
    for i, (data, label) in enumerate(train_loader, start=1):
        data, label = data.to(device), label.to(device)

        outputs = net(data)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        batch_count += 1

        # 实时更新图像
        if batch_count % update_interval == 0:
            ax.clear()
            ax.set_xlabel('Batch count')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss per Batch')
            ax.grid(True)
            ax.plot(range(1, len(batch_losses) + 1), batch_losses, label='Training Loss')
            ax.legend()
            plt.draw()
            plt.pause(0.001)

        # 例如每100个batch打印一次当前batch信息
        if batch_count % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_count}: Loss = {loss.item():.4f}")

    # Epoch结束后也打印一条信息
    print(f"Epoch [{epoch}/{epochs}] completed.")

# 关闭交互模式，显示最终图像
plt.ioff()
plt.show()

# ===================== 测试 =====================
net.eval()
with torch.no_grad():
    correct = 0
    total = len(test_dataset)
    for idx, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        output = net(data)
        predicted = (torch.sigmoid(output) > 0.5).float()

        correct += (predicted == label).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
