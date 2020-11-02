import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from Commons import LocalUtils
from collections import OrderedDict

batch_size = 256
train_iter, test_iter = LocalUtils.load_fashion_mnist_dataset(batch_size=batch_size)

# 定义网络
num_inputs = 28 * 28
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x.view(x.shape[0], -1))
        return x


# 实例化网络
net = nn.Sequential(
    OrderedDict([
        ('flatten', LocalUtils.MyFlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

# 使用均值为0 方差为0.01 的正太分布
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 打印网络
print(net)

# 定义交叉熵损失函数
loss = nn.CrossEntropyLoss()
# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 开始训练
num_epochs = 5
LocalUtils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
