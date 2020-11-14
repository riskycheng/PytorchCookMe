import torch
from torch import nn
from collections import OrderedDict


# *******************************************
class MLP(nn.Module):
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        a = self.act(self.hidden(x))  # 784 -> 256
        return self.output(a)  # 256 -> 10


class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].item():
                # add_module方法会将module添加进self._modules(一个OrderedDict)
                self.add_module(key, module)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


# X = torch.rand(2, 784)
# netSeq = MySequential(
#     nn.Linear(784, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10)
# )
# print(netSeq)
# print(netSeq(X))


# *******************************************
# ModuleList
netModuleList = nn.ModuleList([
    nn.Linear(784, 256),
    nn.ReLU()
])
netModuleList.append(nn.Linear(256, 10))

# 索引到具体的Layer
# print(netModuleList[-2])
# print(netModuleList)

# *******************************************
# ModuleDict 类
netModuleDict = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU()
})

# 赋值
netModuleDict['output'] = nn.Linear(256, 10)


# print(netModuleDict['linear'])
# print(netModuleDict.output)
# print(netModuleDict)
