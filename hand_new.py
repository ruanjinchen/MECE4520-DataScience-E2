import numpy as np
import matplotlib.pyplot as plt


# 激活函数及其导数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


# 损失函数及其导数 (采用MSE)
def mse_loss(y_true, y_pred):
    return 0.5 * (y_true - y_pred) ** 2


def mse_loss_derivative(y_true, y_pred):
    return (y_pred - y_true)


# 前向传播
def forward(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1) + b1  # 隐藏层输入 (2,)
    a1 = sigmoid(z1)  # 隐藏层输出 (2,)
    z2 = np.dot(a1, w2) + b2  # 输出层输入 (scalar)
    a2 = sigmoid(z2)  # 输出层输出 (scalar)
    return z1, a1, z2, a2


# 反向传播
def backward(x, y_true, z1, a1, z2, a2, w2):
    # 输出层误差
    delta_o = mse_loss_derivative(y_true, a2) * sigmoid_derivative(a2)  # scalar
    # 隐藏层误差
    delta_h = delta_o * w2 * sigmoid_derivative(a1)  # (2,)

    # 计算梯度
    dL_dw2 = a1 * delta_o  # (2,)
    dL_db2 = delta_o  # scalar
    dL_dw1 = np.outer(x, delta_h)  # (2,2)
    dL_db1 = delta_h  # (2,)

    return dL_dw1, dL_db1, dL_dw2, dL_db2


# 从CSV中加载数据
# 假设CSV第一行为表头，格式为 Temperature,Humidity,Label
train_data = np.genfromtxt('train_dataset.csv', delimiter=',', skip_header=1)
test_data = np.genfromtxt('test_dataset.csv', delimiter=',', skip_header=1)

X_train = train_data[:, :2]  # Temperature, Humidity
Y_train = train_data[:, 2]  # Label (已计算好)
X_test = test_data[:, :2]
Y_test = test_data[:, 2]

# 初始化参数
np.random.seed(42)
w1 = np.random.randn(2, 2) * 0.1  # 输入到隐藏层权重 (2x2)
b1 = np.random.randn(2) * 0.1  # 隐藏层偏置 (2,)
w2 = np.random.randn(2) * 0.1  # 隐藏到输出层权重 (2,)
b2 = np.random.randn() * 0.1  # 输出层偏置标量

# 超参数
learning_rate = 0.01
epochs = 10000

# 存储损失
loss_history = []

# 实时绘图准备
plt.ion()  # 开启交互模式
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss over Epochs')
ax.grid(True)

for epoch in range(1, epochs + 1):
    total_loss = 0.0

    # 打乱训练数据
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    Y_train_shuffled = Y_train[indices]

    for x, y_true in zip(X_train_shuffled, Y_train_shuffled):
        # 前向传播
        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)
        # 计算损失
        loss = mse_loss(y_true, a2)
        total_loss += loss
        # 反向传播
        dL_dw1, dL_db1, dL_dw2, dL_db2 = backward(x, y_true, z1, a1, z2, a2, w2)
        # 更新参数
        w1 -= learning_rate * dL_dw1
        b1 -= learning_rate * dL_db1
        w2 -= learning_rate * dL_dw2
        b2 -= learning_rate * dL_db2

    # 平均损失
    avg_loss = total_loss / len(X_train)
    loss_history.append(avg_loss)

    # 每1000个epoch打印一次
    if epoch % 1000 == 0 or epoch == 1:
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    # 实时更新绘图
    line.set_xdata(np.arange(1, epoch + 1))
    line.set_ydata(loss_history)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)

plt.ioff()  # 关闭交互模式
plt.show()

# 最终模型评估
print("Final Parameters:")
print(f"w1: \n{w1}")
print(f"b1: {b1}")
print(f"w2: {w2}, b2: {b2}\n")

# 在测试集上评估
correct = 0
for x, y_true in zip(X_test, Y_test):
    _, _, _, a2 = forward(x, w1, b1, w2, b2)
    predicted = 1.0 if a2 > 0.5 else 0.0
    if predicted == y_true:
        correct += 1

accuracy = correct / len(Y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# 显示测试集部分样本的预测结果
print("Some sample predictions from test set:")
for i in range(min(10, len(X_test))):  # 打印前10个测试样本预测结果
    x = X_test[i]
    y_true = Y_test[i]
    _, _, _, a2 = forward(x, w1, b1, w2, b2)
    print(f"Input: {x}, Predicted: {a2:.4f}, True: {y_true}")
