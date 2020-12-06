import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
from Commons import LocalUtils

device = torch.device('cpu')


class Inception(nn.Module):
    def __init__(self, input_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1: 单 1x1 卷积层
        self.p1_1 = nn.Conv2d(input_channels, c1, kernel_size=1)

        # 线路2: 1x1卷积层 +  3x3 卷积层
        self.p2_1 = nn.Conv2d(input_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # 线路3: 1x1卷积层 + 5x5 卷积层
        self.p3_1 = nn.Conv2d(input_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # 线路4: 3x3 最大池化层 + 1x1 卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(input_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))
        return torch.cat((p1, p2, p3, p4), dim=1)


# GoogLeNet 模型 : 共包含 5 个Block,每个模块之间使用步长为 2 的 3x3 最大池化层减少输入宽高
b1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),  # 64x112x112
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64x56x56
)

# 第二个模块 :首先是64@1x1 卷积层，a然后是 192@3x3的卷积层
b2 = nn.Sequential(
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),  # 64x56x56
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),  # 192x56x56
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 192x28x28
)

# 第三个模块, 串联2个完整的Inception
b3 = nn.Sequential(
    Inception(input_channels=192, c1=64, c2=(96, 128), c3=(16, 32), c4=32),  # (64+128+32+32)x28x28
    Inception(input_channels=256, c1=128, c2=(128, 192), c3=(32, 96), c4=64),  # (128+192+96+64)x28x28
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 480x14x14
)

# 第四个模块，串联5个Inception
b4 = nn.Sequential(
    Inception(input_channels=480, c1=192, c2=(96, 208), c3=(16, 48), c4=64),  # (192+208+48+64)x14x14
    Inception(input_channels=512, c1=160, c2=(112, 224), c3=(24, 64), c4=64),  # 512x14x14
    Inception(input_channels=512, c1=128, c2=(128, 256), c3=(24, 64), c4=64),  # 512x14x14
    Inception(input_channels=512, c1=112, c2=(144, 288), c3=(32, 64), c4=64),  # 528x14x14
    Inception(input_channels=528, c1=256, c2=(160, 320), c3=(32, 128), c4=128),  # 832x14x14
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 832x7x7
)

# 第五个模块 ： 包含两个Inception
b5 = nn.Sequential(
    Inception(input_channels=832, c1=256, c2=(160, 320), c3=(32, 128), c4=128),  # 832x7x7
    Inception(input_channels=832, c1=384, c2=(192, 384), c3=(48, 128), c4=128),  # 1024x7x7
    LocalUtils.GlobalAvgPool2d()
)

# Build the final network
net = nn.Sequential(
    b1,
    b2,
    b3,
    b4,
    b5,
    LocalUtils.MyFlattenLayer(),
    nn.Linear(1024, 10)
)

X = torch.rand(1, 1, 96, 96)
for blk in net.children():
    X = blk(X)
    print('output shape:', X.shape)

# start Training
batch_size = 24
train_iter, test_iter = LocalUtils.load_fashion_mnist_dataset(batch_size=batch_size, resize=96)
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
LocalUtils.train_ch5(
    net,
    train_iter=train_iter,
    test_iter=test_iter,
    batch_size=batch_size,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs)
