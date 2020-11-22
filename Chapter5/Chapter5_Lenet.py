import time
import torch
from torch import nn, optim
import sys
from Commons import LocalUtils

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in-channel, out_channel, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        print(img.shape)
        feature = self.conv(img)
        print(feature.shape)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


net = LeNet()
print(net)

# get the data
batch_size = 256
train_iter, test_iter = LocalUtils.load_fashion_mnist_dataset(batch_size=batch_size)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
LocalUtils.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
