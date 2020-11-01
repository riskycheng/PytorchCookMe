import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
from Commons import LocalUtils

mnist_train = torchvision.datasets.FashionMNIST(
    root='h:/dataset/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

mnist_test = torchvision.datasets.FashionMNIST(
    root='h:/dataset/FashionMNIST',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# 打印数据集的信息
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 查看数据
feature, label = mnist_train[0]
print(feature.shape, label)

# 可视化
# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# LocalUtils.show_fashion_mnist(X, LocalUtils.get_fashion_minist_labels(y))

# 读取小批量数据
batch_size = 256
num_workers = 0
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 查看读取数据的时间
start = time.time()
for X, y in train_iter:
    continue
print('%.2d sec' % (time.time() - start))