import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(
    nn.Linear(4, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

print(net)

X = torch.rand(2, 4)
Y = net(X).sum()
print(Y)

for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))

# print the param of the network
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print('initialized normal:', name, param.data)
    if 'bias' in name:
        init.constant_(param, val=0)
        print('initialized constant:', name, param.data)
