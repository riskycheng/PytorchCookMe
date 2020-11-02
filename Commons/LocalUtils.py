import random
from IPython import display
from matplotlib import pyplot as plt
import torch
import sys
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn import init


# 定义优化函数
def sgdOpt(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 打印散点图
def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figureSize(figsize=(10, 6)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


# 获取Fashion MINIST labels
def get_fashion_minist_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


# 显示多张图像和对应标签
def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view(28, 28).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 加载 Train-set Test-set
def load_fashion_mnist_dataset(batch_size):
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
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 返回结果
    return train_iter, test_iter


# 定义线性层
class MyLinearLayer(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MyLinearLayer, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x.view(x.shape[0], -1))
        return x


# 自定义 Flatter 层
class MyFlattenLayer(nn.Module):
    def __init__(self):
        super(MyFlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgdOpt(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
