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
from PIL import Image
from torch.autograd import Variable
import numpy as np

torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '/home/jiache/dataset/cats_and_dogs/official_test'

# define the data transforms
data_transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 128
num_workers = 4
# train dataset


net = torch.load('./model_0.92.pth')
net.cuda()
net.eval()

imagesPath = os.listdir(data_dir)
results = open('results.csv', 'w+')

for image in imagesPath:
    img = Image.open(os.path.join(data_dir, image))
    img = data_transform(img)
    img.unsqueeze_(0)
    pred = net(img.cuda())
    result = 'dog' if pred[0][0] < pred[0][1] else 'cat'
    print (os.path.join(data_dir, image), ' >>> ', result)
    results.write(os.path.join(data_dir, image))
    results.write(',')
    results.write(result)
    results.write('\n302afioprtvw')
results.close()
