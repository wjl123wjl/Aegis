import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from time import time
from torch.utils.data.sampler import SubsetRandomSampler
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data
import random
from math import floor
import operator

import copy
import matplotlib.pyplot as plt

import models
from utils import AverageMeter
from models.quantization import quan_Conv2d, quan_Linear, quantize

## parameter
targets=2
start=21
end=31 

## normalize layer
class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)


## weight conversion functions

def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    '''
    output = input.clone()
    output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
    return output


def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    mask = 2**(num_bits-1) - 1
    output = -(input & ~mask) + (input & mask)
    return output


def weight_conversion(model):
    '''
    Perform the weight data type conversion between:
        signed integer <==> two's complement (unsigned integer)

    Note that, the data type conversion chosen is depend on the bits:
        N_bits <= 8   .char()   --> torch.CharTensor(), 8-bit signed integer
        N_bits <= 16  .short()  --> torch.shortTensor(), 16 bit signed integer
        N_bits <= 32  .int()    --> torch.IntTensor(), 32 bit signed integer
    '''
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            w_bin = int2bin(m.weight.data, m.N_bits).short()
            m.weight.data = bin2int(w_bin, m.N_bits).float()
    return

# Hyper-parameters
param = {
    'batch_size': 256,
    'test_batch_size': 256,
    'num_epochs':250,
    'delay': 251,
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
}

mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]
print('==> Preparing data..')
print('==> Preparing data..') 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    
])


trainset = torchvision.datasets.CIFAR10(root='../../datasets/cifar10', train=True, download=True, transform=transform_train) 

loader_train = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2) 

testset = torchvision.datasets.CIFAR10(root='../../datasets/cifar10', train=False, download=True, transform=transform_test) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

net_c = models.__dict__['vgg16_quan'](10)
net = torch.nn.Sequential(
                    Normalize_layer(mean,std),
                    net_c
                    )

net_f = models.__dict__['vgg16_quan'](10) 
net1 = torch.nn.Sequential(
                    Normalize_layer(mean,std),
                    net_f
                    )

net=net.cuda()
# model.load_state_dict(torch.load('./cifar_vgg_pretrain.pt', map_location='cpu'))
pretrain_dict = torch.load('./save_finetune/model_best.pth.tar')
pretrain_dict = pretrain_dict['state_dict']
model_dict = net.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {str('1.'+ k): v for k, v in pretrain_dict.items() if str('1.'+ k) in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
net.load_state_dict(model_dict)

for m in net.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()
weight_conversion(net)

net1=net1.cuda()
# model.load_state_dict(torch.load('./cifar_vgg_pretrain.pt', map_location='cpu'))
pretrained_dict = torch.load('./result/TBT/final_trojan.pkl')
model_dict = net1.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
net1.load_state_dict(model_dict)

for m in net1.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()

weight_conversion(net1)

import numpy as np
for x, y in loader_train:
    x=x.cuda()
    y=y.cuda()
    break
ss = np.loadtxt('./result/TBT/vgg_trojan_img1.txt', dtype=float)
x[0,0:,:]=torch.Tensor(ss).cuda()
ss = np.loadtxt('./result/TBT/vgg_trojan_img2.txt', dtype=float)
x[0,1:,:]=torch.Tensor(ss).cuda()
ss = np.loadtxt('./result/TBT/vgg_trojan_img3.txt', dtype=float)
x[0,2:,:]=torch.Tensor(ss).cuda() 

def test(model, loader):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        #grid_img = torchvision.utils.make_grid(x_var[0,:,:,:], nrow=1)
        #plt.imshow(grid_img.permute(1, 2, 0))
        #plt.show() 
        
        scores = model(x_var)[14]
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))

    return acc

#test codee with trigger
def test1(model, loader, xh):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        x_var[:,0:3,start:end,start:end]=xh[:,0:3,start:end,start:end]
        #grid_img = torchvision.utils.make_grid(x_var[0,:,:,:], nrow=1)
        #plt.imshow(grid_img.permute(1, 2, 0))
        #plt.show() 
        y[:]=targets  ## setting all the target to target class
     
        scores = model(x_var)[14]
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the trigger data' 
        % (num_correct, num_samples, 100 * acc))

    return acc
    
b = np.loadtxt('./result/TBT/vgg_trojan_test.txt', dtype=float)
tar=torch.Tensor(b).long().cuda()

criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda()

test(net1,loader_test)
test1(net1,loader_test,x)

n=0
### setting all the parameter of the last layer equal for both model except target class This step is necessary as after loading some of the weight bit may slightly
#change due to weight conversion step to 2's complement
for param in net1.parameters():
    n=n+1
    m=0
    for param1 in net.parameters():
        m=m+1
        if n==m:
            #print(n,(param-param1).sum()) 
            if n==225:
                #print(param.data.eq(param1.data))
                xx=param.data.clone()
                    
                param.data=param1.data.clone() 
                      
                param.data[targets,tar]=xx[targets,tar].clone()
                w=param-param1
                print(w[w==0].size())
                
n=0
### counting the bit-flip the function countings
from bitstring import Bits
def countingss(param,param1):
    ind=(w!= 0).nonzero()
    jj=int(ind.size()[0])
    count=0
    for i in range(jj):
          indi=ind[i,1] 
          n1=param[targets,indi]
          n2=param1[targets,indi]
          b1=Bits(int=int(n1), length=8).bin
          b2=Bits(int=int(n2), length=8).bin
          for k in range(8):
              diff=int(b1[k])-int(b2[k])
              if diff!=0:
                 count=count+1
    return count
for param1 in net.parameters():
    n=n+1
    m=0
    for param in net1.parameters():
        m=m+1
        if n==m:
            #print(n) 
            if n==225:
                w=((param1-param))
                print(countingss(param,param1)) ### number of bitflip nb
                print(w[w==0].size())  ## number of parameter changed wb