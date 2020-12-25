import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import sys
from Commons import LocalUtils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '/home/jiache/dataset/'

train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))

# preview the first 8 positive and last negative samples

# not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
# hotdogs = [train_imgs[i][0] for i in range(8)]
# LocalUtils.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
# LocalUtils.plt.show()

# channel mean and variables
normalization_std = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalization_std
])

test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalization_std
])

# define and init the network
pretrained_net = models.resnet18(pretrained=True)

# modify the FC layer
pretrained_net.fc = nn.Linear(in_features=512, out_features=2)
print(pretrained_net.fc)

output_params = list(map(id, pretrained_net.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD(
    [{'params': feature_params},
     {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
    lr=lr, weight_decay=0.001)

# start training
train_iter = DataLoader((train_imgs, train_augs), batch_size=12, shuffle=True)
test_iter = DataLoader((test_imgs, test_augs), batch_size=12, shuffle=True)
LocalUtils.train_ch5(pretrained_net, train_iter=train_iter, test_iter=test_iter, batch_size=12, optimizer=optimizer,
                     num_epochs=5, device=device)
