import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from torchvision.datasets import ImageFolder
from torch import nn
import torch.nn.functional as F
from Commons import LocalUtils
from torch import optim
from torchvision import models

torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '/home/jiache/dataset/cats_and_dogs/'

# define the data transforms
data_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 128
num_workers = 4
# train dataset
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'))
train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'train'), transform=data_transform), batch_size=batch_size,
                        shuffle=True)
# test dataset
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'))
test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'test'), transform=data_transform), batch_size=batch_size,
                       shuffle=True)

# preview some of the training data
cat_samples = [test_dataset[i][0] for i in range(8)]
dog_samples = [test_dataset[-i - 1][0] for i in range(8)]
LocalUtils.show_images(cat_samples + dog_samples, 2, 8, scale=1.4)
LocalUtils.plt.show()


# create simple network to train model
class CatDogNet(nn.Module):
    def __init__(self):
        super(CatDogNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 18 * 18, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# define fine-tuning model based on resnet-18
pretrained_resnet18 = models.resnet18(pretrained=True)

# modify the FC layer
pretrained_resnet18.fc = nn.Linear(in_features=512, out_features=2)

net = CatDogNet()
num_epochs = 5
lr = 0.0001
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
LocalUtils.train_ch5(net=pretrained_resnet18, train_iter=train_iter, test_iter=test_iter, device=device,
                     batch_size=batch_size, optimizer=optimizer, num_epochs=num_epochs)
