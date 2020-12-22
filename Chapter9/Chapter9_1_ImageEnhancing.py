import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import sys
from Commons import LocalUtils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LocalUtils.set_figureSize()
img = Image.open('./beauty.jpg')
# LocalUtils.plt.imshow(img)
# LocalUtils.plt.show()

# test image flip and crop
flip_aug = torchvision.transforms.RandomHorizontalFlip()
# LocalUtils.apply(img, flip_aug)
# LocalUtils.plt.show()

# test image random crop and resize
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
# LocalUtils.apply(img, shape_aug)
# LocalUtils.plt.show()

# test image color Jitter
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# LocalUtils.apply(img, color_aug)
# LocalUtils.plt.show()

# apply with different enhancements
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    color_aug,
    shape_aug,
    flip_aug
])
# LocalUtils.apply(img, augs)
# LocalUtils.plt.show()

# test cifar-10
all_images = torchvision.datasets.CIFAR10(train=True, root='/home/jiache/dataset/CIFAR_10', download=True)
LocalUtils.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
LocalUtils.plt.show()

flip_tensor_aug = torchvision.transforms.Compose([
    flip_aug,
    torchvision.transforms.ToTensor()
])

no_aug = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# using transforms to train the CIFAR-10 training set

