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
    transforms.Resize([64, 64]),
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
# cat_samples = [test_dataset[i][0] for i in range(8)]
# dog_samples = [test_dataset[-i - 1][0] for i in range(8)]
# LocalUtils.show_images(cat_samples + dog_samples, 2, 8, scale=1.4)
# LocalUtils.plt.show()


# define the vgg-16 network
class CatsAndDogsVGG16(nn.Module):
    def __init__(self):
        super(CatsAndDogsVGG16, self).__init__()
        self.conv = nn.Sequential(
            # 2x64 @ 3x3
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32x64

            # 2x128 @ 3x3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16x128

            # 3x256 @3x3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8x256

            # 3x512 @ 3x3
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4x512
        )

        self.classes = nn.Sequential(
            nn.Linear(4 * 4 * 512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 4 * 4 * 512)
        x = self.classes(x)
        return x


# net = CatsAndDogsVGG16()
pretrained_net = models.vgg16(pretrained=True)

pretrained_net.classifier = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024, 2)
)

output_params = list(map(id, pretrained_net.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

num_epochs = 5
lr = 0.0001
optimizer = optim.SGD(
    [{'params': feature_params},
     {'params': pretrained_net.classifier.parameters(), 'lr': lr * 10}],
    lr=lr, weight_decay=0.001, momentum=0.9)

LocalUtils.train_ch5(net=pretrained_net, train_iter=train_iter, test_iter=test_iter, device=device,
                     batch_size=batch_size, optimizer=optimizer, num_epochs=num_epochs, saveModel=True)

