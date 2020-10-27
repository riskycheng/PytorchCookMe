import torch
from matplotlib import pyplot as plt
import numpy as np
from Commons import LocalUtils

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

# 数据分布图
# LocalUtils.set_figureSize()
# plt.scatter(features[:, 0].numpy(), labels.numpy(), 1)
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 2)
# plt.show()

# 尝试读取数据
batch_size = 10
for X, y in LocalUtils.data_iter(batch_size, features, labels):
    print(X, y)
    break

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)


# 定义模型
def linearRegression(X, w, b):
    return torch.mm(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化函数
def sgdOpt(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 开始训练
lr = 0.03
num_epochs = 3
net = linearRegression
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in LocalUtils.data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgdOpt([w, b], lr, batch_size)

        # 梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

# 校验结果
print(true_w, '\n', w)
print(true_b, '\n', b)