import torch
from matplotlib import pyplot as plt
import numpy as np
from Commons import LocalUtils
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

batch_size = 10
# 将训练数据的特征和标签进行组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


# 定义自己的网络结构
class LinearNet(nn.Module):
    def __init__(self, input_dim):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


# 打印网络结构
net = LinearNet(num_inputs)
print(net)  # 使用print可以打印出网络的结构

# # 打印可学习参数
# for param in net.parameters():
#     print(param)

# 初始化参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化函数
optimizer = optim.SGD(net.parameters(), lr=0.03)

# 开始训练
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)  # 模型预测
        l = loss(output, y.view(-1, 1))  # 计算损失函数
        optimizer.zero_grad()  # 清楚梯度值
        l.backward()  # 梯度推导
        optimizer.step()  # 梯度优化
    print('epoch %d, loss : %f' % (epoch, l.item()))

# 查看学习到的参数
print('learnt param:', net.linear.weight, ' true param:', true_w)
print('learnt param:', net.linear.bias, 'true param:', true_b)