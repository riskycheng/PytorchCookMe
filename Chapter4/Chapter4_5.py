import torch
from torch import nn

# x = torch.ones(3)
# print(x)
# torch.save(x, './x.pt')

x = torch.load('./x.pt')
print(x)

y = torch.zeros(4)
torch.save([x, y], './xy.pt')
xy_list = torch.load('./xy.pt')
print(xy_list)

# 存储并读取一个从字符串映射到Tensor的字典
torch.save({'x': x, 'y': y}, './xy.pt')
xy = torch.load('./xy.pt')
print(xy)

# 读写模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.act(x)
        return self.output(x)

net = MLP()
print(net.state_dict())

optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
print(optimizer.state_dict())

x = torch.randn(2, 3)
Y = net(x)
print(Y)

# 模型保存和加载
torch.save(net.state_dict(), './model.pth')

# 加载模型
model = MLP()
model.load_state_dict(torch.load('./model.pth'))
Y = model(x)
print(Y)