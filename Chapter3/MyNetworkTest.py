import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from Commons import LocalUtils

net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0),  # 32x26x26
    nn.ReLU(),  # 32x26x26
    nn.MaxPool2d(kernel_size=(2, 2)),  # 32x13x13
    LocalUtils.MyFlattenLayer(),
    nn.Linear(32*13*13, 256),
    nn.ReLU(),  # 32x26x26
    nn.Linear(256, 10)
)

print(net)

batch_size = 256
train_iter, test_iter = LocalUtils.load_fashion_mnist_dataset(batch_size)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
LocalUtils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

