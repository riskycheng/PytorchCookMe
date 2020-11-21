import torch
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__()

    def forward(self, x):
        return x - x.mean()


# 实例化该层
layer = CenteredLayer()
data = layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))

# 用以构造复杂模型
net = nn.Sequential(
    nn.Linear(8, 128),
    CenteredLayer()
)

y = net(torch.rand(4, 8))
print(y.mean().item())


# 含模型参数的自定义层
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x


net = MyDense()
print(net)

# ParameterDict自定义层
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear_01' : nn.Parameter(torch.randn(4, 4)),
            'linear_02' : nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({
            'linear_03' : nn.Parameter(torch.randn(4, 2))
        })

    def forward(self, x, choice='linear_01'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)

x = torch.ones(1, 4)
print(net(x, 'linear_01'))
print(net(x, 'linear_02'))
print(net(x, 'linear_03'))

# 组合自定义模块
net = nn.Sequential(
    MyDictDense(),
    MyDense()
)

print(net)