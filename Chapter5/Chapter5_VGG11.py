import time
import torch
from torch import nn, optim
import sys
from Commons import LocalUtils

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


conv_arch = (
    # 第一层: 一个Conv2d
    (1, 3, 64),
    # 第二层: 一个Conv2d
    (1, 64, 128),
    # 第三层: 两个Conv2d
    (2, 128, 256),
    # 第四层: 两个Conv2d
    (2, 256, 512),
    # 第五层: 两个Conv2d
    (2, 512, 512)
)
fc_features_dim = 512 * 7 * 7
fc_hidden_units = 4096

# 通过函数模块创建
def VGG_Func_11(conv_arch, fc_features_dim, fc_hidden_units=4096):
    net = nn.Sequential()
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module('fc', nn.Sequential(
        LocalUtils.MyFlattenLayer(),
        nn.Linear(fc_features_dim, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 10)
    ))
    return net


net = VGG_Func_11(conv_arch, fc_features_dim, fc_hidden_units)
X = torch.rand(10, 3, 224, 224)
print('================')
print(net(X))
