import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from scipy.interpolate import interp1d

# 设置Matplotlib的字体
plt.rcParams['font.family'] = 'Microsoft YaHei'  # 使用SimHei字体（黑体）


# 定义多项式回归模型
class PR(nn.Module):
    def __init__(self, degree):
        super(PR, self).__init__()
        self.degree = degree
        self.linear = nn.Linear(self.degree + 1, 1)  # 加1是因为包含常数项
        # self.nolinear = nn.Sigmoid()

    def forward(self, x):
        # 转换张量格式，将输入数据转换为多项式
        x = x.reshape(-1, 1)
        x_poly = x ** torch.arange(self.degree + 1, dtype=torch.float32, device=x.device).unsqueeze(0)
        x_poly = self.linear(x_poly).reshape(-1)
        return x_poly


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(1, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 1)
        self.nolinear = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)  # 添加 dropout，丢弃概率为 0.9

    def forward(self, x):
        x = x.reshape(-1, 1)
        x = self.linear1(x)
        x = self.nolinear(x)
        # x = self.dropout(x)  # dropout
        x = self.linear2(x)
        x = self.nolinear(x)
        # x = self.dropout(x)
        x = self.linear3(x)
        return x.reshape(-1)


# 设置train值
train = pd.read_excel('complex_nonlinear_data.xlsx')
test = pd.read_excel('new_complex_nonlinear_data.xlsx')
# 获取数据
x_train = np.array(train.iloc[:, 0])  # 第一列为x值
y_train = np.array(train.iloc[:, 1])  # 第二列为y值
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

x_test = np.array(test.iloc[:, 0])  # 第一列为x值
y_test = np.array(test.iloc[:, 1])  # 第二列为y值
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 初始化多项式拟合模型
degree = 6  # 多项式的阶数
model1 = PR(degree)

# 初始化多层感知机模型
model2 = MLP()
# 定义损失函数和优化器
loss_fonc = nn.MSELoss()
optimizer = optim.Adam(model2.parameters(), lr=0.1)  # 默认使用MLP模型
losses_train = []
losses_val = []
# 训练模型
num_epochs = 10000
loss_train_best = 1e15
loss_val_best = 1e15
best_train_epoch = 0
best_val_epoch = 0
y_best_val = []
y_best_train = []
y_best_test = []
init_time = time.time()
for epoch in range(num_epochs):
    # 划分训练集和验证集, 交叉验证
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, test_size=0.2,
                                                                              random_state=epoch)
    optimizer.zero_grad()
    y_train_split_pred = model2(x_train_split)
    loss = loss_fonc(y_train_split_pred, y_train_split)
    losses_train.append(loss.detach().numpy())

    # 选择最优train epoch
    if loss < loss_train_best:
        loss_train_best = loss
        best_train_epoch = epoch + 1
        y_best_train = model2(x_train)  # 输出全部100个数据的pred
    loss.backward()
    optimizer.step()
    epoch_time = time.time()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}],USE time total{epoch_time - init_time:.2f}, MSELoss: {loss.item()}')
    # 测试模型
    with torch.no_grad():
        y_val_pred = model2(x_val_split)
        loss = loss_fonc(y_val_pred, y_val_split)
        losses_val.append(loss.detach().numpy())
        if loss < loss_val_best:
            loss_val_best = loss
            best_val_epoch = epoch + 1
            y_best_val = model2(x_train)
            y_best_test = model2(x_test)  # 选取验证集损失最低作为测试集模型
            loss_test_best = loss_fonc(y_best_test, y_test)
            # 在训练循环中记录val集 MSE 最低时的模型参数
            best_model_state = model2.state_dict()  # 获取模型参数
print(f'BEST Train Epoch [{best_train_epoch}], MSELoss: {loss_train_best.item()}')
print(f'BEST Val Epoch [{best_val_epoch}], MSELoss: {loss_val_best.item()}')
print(f'BEST Test Epoch [{best_val_epoch}], MSELoss: {loss_test_best.item()}')

# 保存模型参数到文件中
torch.save(best_model_state, 'best_model.pth')
# 绘制loss曲线
plt.subplot(2, 1, 1)
plt.plot(losses_train, color='green', label='train_loss曲线')
plt.legend()
plt.title('train_loss曲线值')
plt.xlabel('epoch')
plt.ylabel('losses')

plt.subplot(2, 1, 2)
plt.plot(losses_val, color='orange', label='val_loss曲线')
plt.legend()
plt.title('val_loss曲线值')
plt.xlabel('epoch')
plt.ylabel('losses')
plt.show()

# 绘制曲线
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
y_best_train = y_best_train.detach().numpy()
y_best_test = np.array(y_best_test)
y_best_val = np.array(y_best_val)

# 设置整个图像的大小
plt.figure(figsize=(8, 12))
plt.subplot(3, 1, 1)  # 三行一列，第一个图
plt.plot(x_train, y_train, color='blue', label='train曲线')
plt.plot(x_train, y_best_train, color='red', label='best_train_pred曲线')
plt.legend()
plt.title('train曲线值')
plt.xlabel('x值')
plt.ylabel('y值')

plt.subplot(3, 1, 2)  # 三行一列，第二个图
plt.plot(x_train, y_train, color='blue', label='train曲线')
plt.plot(x_train, y_best_val, color='red', label='best_val_pred曲线')
plt.legend()
plt.title('val曲线值')
plt.xlabel('x值')
plt.ylabel('y值')

plt.subplot(3, 1, 3)  # 三行一列，第三个图
plt.plot(x_test, y_test, color='blue', label='test曲线')
plt.plot(x_test, y_best_test, color='red', label='test_pred曲线')
plt.legend()
plt.title('test曲线值')
plt.xlabel('x值')
plt.ylabel('y值')

# 显示图形
plt.tight_layout()
plt.show()
