import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from Commons import LocalUtils

num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
    LocalUtils.MyFlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs)
)

print(net)

batch_size = 256
train_iter, test_iter = LocalUtils.load_fashion_mnist_dataset(batch_size)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
LocalUtils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

