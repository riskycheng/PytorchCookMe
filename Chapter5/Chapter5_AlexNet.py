import time
import torch
from torch import nn, optim
import torchvision
import sys
from Commons import LocalUtils

# current PC does not support CUDA at all
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


# Alexnet structure refers to https://www.jianshu.com/p/00a53eb5f4b3
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            # in-channel, out_channel, kernel_size, stride
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),  # 1x227x227 -> 96x55x55
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 96x55x55 -> 96x27x27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=1),  # 256x27x27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*27x27 -> 256x13x13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1, stride=1),  # 384x13x13
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),  # 384x13x13
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),  # 256x13x13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256x6x6
        )

        # todo need to do flatten before going to next stage

        # 全连接层 : 包含两个全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
            # 使用Dropout否则会导致参数过多
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=1000)
        )

    def forward(self, img):
        feature = self.conv(img)
        # flatten and FC layers
        feature = feature.view(feature.shape[0], -1)
        output = self.fc(feature)
        return output


# 创建网络
net = AlexNet()
print(net)

# 加载数据
batch_size = 128
train_iter, test_iter = LocalUtils.load_fashion_mnist_dataset(batch_size, resize=227)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
LocalUtils.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
