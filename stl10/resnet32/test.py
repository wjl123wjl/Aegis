import models
from models.quantization import quan_Conv2d, quan_Linear, quantize
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import torchvision.datasets as datasets
from torchvision import transforms
from utils import AverageMeter
import time
from torch.utils.data import Dataset
from PIL import Image

net = models.__dict__['resnet32_quan'](10)
pretrain_dict = torch.load('./model/stl10/resnet32/model_best.pth.tar',map_location=torch.device('cpu'))
pretrain_dict = pretrain_dict['state_dict']
model_dict = net.state_dict()
pretrained_dict = {str(k): v for k, v in pretrain_dict.items() if str(k) in model_dict}
model_dict.update(pretrained_dict) 

net.load_state_dict(model_dict) 
net.eval()

class  DatasetTiny(Dataset):
    def __init__(self, root, train=False, transform=None):
        self.root = root
        if train:
            self.data = np.load('./datasets/tiny-imagenet-200/images.npy')
            self.targets = np.load('./datasets/tiny-imagenet-200/train_target.npy')
        else:
            self.data = np.load('./datasets/tiny-imagenet-200/im_test.npy')
            self.targets = np.load('./datasets/tiny-imagenet-200/test_target.npy') 
        self.targets = list(self.targets)
        
        self.transform = transform
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]
'''
mean = [0.4802,  0.4481,  0.3975]
std = [0.2302, 0.2265, 0.2262]
print('==> Preparing data..')
print('==> Preparing data..') 
train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(64, padding=8), 
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean,std)])


trainset = DatasetTiny(root='../../datasets/tiny-imagenet-200', train=True, transform=train_transform)
testset = DatasetTiny(root='../../datasets/tiny-imagenet-200', train=False, transform=test_transform)

loader_train = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
'''
'''
mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    
])
trainset = datasets.CIFAR100(root='./datasets/cifar100', train=True, download=True, transform=transform_train) 

loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 

testset = datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True, transform=transform_test) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
'''
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
print('==> Preparing data..')
print('==> Preparing data..') 
train_transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ])
test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ])


trainset = datasets.STL10('./datasets/stl10',
                                split='train',
                                transform=train_transform,
                                download=True)
testset = datasets.STL10('./datasets/stl10',
                               split='test',
                               transform=test_transform,
                               download=True)
loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
sum_time = 0
with torch.no_grad():
    for i, (input, target) in enumerate(loader_test):
        #target = target.cuda()
        #input = input.cuda()
        target_var = Variable(target, volatile=True)
        start = time.time()
        output_branch = net(input)
        final = time.time()
        sum_time+=final-start
        #print(output_branch)
        if i==99:
            break
    avg_time = sum_time/100
    print('Predicted in %f seconds.' % avg_time)            