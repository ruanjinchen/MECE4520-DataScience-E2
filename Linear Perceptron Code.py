import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# 1. Data generation and visualization
class XORDataset(Dataset):
    def __init__(self):
        # 四种可能的输入组合
        self.X = torch.tensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ], dtype=torch.float32)

        # XOR标签
        self.y = torch.tensor([
            [0.0],
            [1.0],
            [1.0],
            [0.0]
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 实例化数据集和数据加载器
dataset = XORDataset()
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)


# 可视化XOR数据分布
def plot_xor_data(dataset):
    X = dataset.X.numpy()
    y = dataset.y.numpy()

    plt.figure(figsize=(6, 6))
    for i in range(len(X)):
        if y[i] == 0:
            plt.scatter(X[i][0], X[i][1], color='red', marker='o', label='0' if i == 0 else "")
        else:
            plt.scatter(X[i][0], X[i][1], color='blue', marker='x', label='1' if i == 1 else "")
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('XOR Data Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()


# 绘制数据点
plot_xor_data(dataset)


# 2. Define the linear perceptron model
class LinearPerceptron(nn.Module):
    def __init__(self):
        super(LinearPerceptron, self).__init__()
        self.linear = nn.Linear(2, 1)  # The number of input features is 2 and the output is 1 neuron

    def forward(self, x):
        out = self.linear(x)
        return out  # 不应用激活函数，因为我们将使用BCEWithLogitsLoss
    # The activation function is not applied because we will be using BCEWithLogitsLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Instantiate models, define loss functions, and optimizers
model = LinearPerceptron().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Use a simple SGD optimizer

# 3. Train the linear perceptron model
epochs = 2000
loss_values = []

# 开启交互模式以实时绘图
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss (Linear Perceptron)')

# Training cycle
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        # 前向传播 Forward propagation
        outputs = model(data)
        loss = criterion(outputs, label)
        # 反向传播与优化 Back propagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_values.append(avg_loss)

    if epoch % 10 == 0 or epoch == 1:
        ax.plot(epoch, avg_loss, 'bo') # 蓝色圆点
        plt.pause(0.01)
        print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}')

# 关闭交互模式并显示最终图表
plt.ioff()
plt.show()

# 4. 性能评估
# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = len(dataset)
    for data, label in DataLoader(dataset, batch_size=1):
        data, label = data.to(device), label.to(device)
        output = model(data)
        predicted = (torch.sigmoid(output) > 0.5).float()
        correct += (predicted == label).sum().item()

    accuracy = 100 * correct / total
    print(f'\nTraining Accuracy: {accuracy:.2f}%')


# # 5. 使用多层感知器（MLP）解决XOR问题以对比
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.hidden = nn.Linear(2, 4)  # 隐藏层有4个神经元
#         self.activation = nn.ReLU()
#         self.output = nn.Linear(4, 1)
#
#     def forward(self, x):
#         x = self.hidden(x)
#         x = self.activation(x)
#         x = self.output(x)
#         return x
#
#
# # 实例化模型、定义损失函数和优化器
# model_mlp = MLP().to(device)
# criterion_mlp = nn.BCEWithLogitsLoss()
# optimizer_mlp = optim.SGD(model_mlp.parameters(), lr=0.1)
#
# # 训练参数
# epochs = 1000
# loss_values_mlp = []
#
# # 开启交互模式以实时绘图
# plt.ion()
# fig, ax = plt.subplots()
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Loss')
# ax.set_title('Training Loss (MLP)')
#
# # 训练循环
# for epoch in range(1, epochs + 1):
#     model_mlp.train()
#     total_loss = 0.0
#     for data, label in train_loader:
#         data, label = data.to(device), label.to(device)
#
#         # 前向传播
#         outputs = model_mlp(data)
#         loss = criterion_mlp(outputs, label)
#
#         # 反向传播与优化
#         optimizer_mlp.zero_grad()
#         loss.backward()
#         optimizer_mlp.step()
#
#         total_loss += loss.item()
#
#     avg_loss = total_loss / len(train_loader)
#     loss_values_mlp.append(avg_loss)
#
#     # 每100个epoch更新一次图表
#     if epoch % 100 == 0 or epoch == 1:
#         ax.plot(epoch, avg_loss, 'go')  # 'go'为绿色圆点
#         plt.pause(0.01)
#         print(f'MLP Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}')
#
# # 关闭交互模式并显示最终图表
# plt.ioff()
# plt.show()
#
# # 测试模型
# model_mlp.eval()
# with torch.no_grad():
#     correct = 0
#     total = len(dataset)
#     for data, label in DataLoader(dataset, batch_size=1):
#         data, label = data.to(device), label.to(device)
#         output = model_mlp(data)
#         predicted = (torch.sigmoid(output) > 0.5).float()
#         correct += (predicted == label).sum().item()
#
#     accuracy = 100 * correct / total
#     print(f'\nMLP Training Accuracy: {accuracy:.2f}%')
