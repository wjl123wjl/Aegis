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
import numpy as np
import matplotlib.pyplot as plt

import models
from utils import AverageMeter
from models.quantization import quan_Conv2d, quan_Linear, quantize

## parameter
targets=2
start=65
end=95

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

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
print('==> Preparing data..')
print('==> Preparing data..') 
train_transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


trainset = torchvision.datasets.STL10('/home/kemove/data/workplace/ziyuan/datasets/stl10',
                                split='train',
                                transform=train_transform,
                                download=True)
testset = torchvision.datasets.STL10('/home/kemove/data/workplace/ziyuan/datasets/stl10',
                               split='test',
                               transform=test_transform,
                               download=True)
loader_train = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

net_c = models.__dict__['resnet32_quan'](10)
net = torch.nn.Sequential(
                    Normalize_layer(mean,std),
                    net_c
                    )

net_f = models.__dict__['resnet32_quan'](10) 
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

# update the step size before validation
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

for x, y in loader_train:
    x_tri=x.cuda()
    y=y.cuda()
    break
ss = np.loadtxt('./result/TBT/resnet_trojan_img1.txt', dtype=float)
x_tri[0,0:,:]=torch.Tensor(ss).cuda()
ss = np.loadtxt('./result/TBT/resnet_trojan_img2.txt', dtype=float)
x_tri[0,1:,:]=torch.Tensor(ss).cuda()
ss = np.loadtxt('./result/TBT/resnet_trojan_img3.txt', dtype=float)
x_tri[0,2:,:]=torch.Tensor(ss).cuda() 

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0)
            correct_k = correct[:k].reshape(-1).float().sum(0)
            
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    

    
index_list = []
def validate2(val_loader, model, criterion, num_branch):
    global index_list
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_list = []
    for idx in range(num_branch):
        top1_list.append(AverageMeter())
    top5_list = []
    for idx in range(num_branch):
        top5_list.append(AverageMeter())



    # switch to evaluate mode
    model.eval()
    output_summary = [] # init a list for output summary

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            # compute output
            #w = list(map(float, args.weight.split(',')))
            output_branch = model(input)
            #loss = criterion(output, target)
            loss = 0
            for idx in range(len(output_branch)):
                loss += 1 * criterion(output_branch[idx], target)

            # measure accuracy and record loss
            #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            for idx in range(len(output_branch)):
                prec1, prec5 = accuracy(output_branch[idx].data, target, topk=(1, 5))
                top1_list[idx].update(prec1, input.size(0)) 
                top5_list[idx].update(prec5, input.size(0))

            
            losses.update(loss.item(), input.size(0))
            # top1.update(prec1.item(), input.size(0))
            # top5.update(prec5.item(), input.size(0))

    c_=0
    
    max_ = 0
    for item in top1_list:
        if item.avg > max_:
            max_ = item.avg 
            index_list.append(c_)
        #print("c_{}", c_, item.avg)  
        c_ += 1 
    return index_list

def validate(val_loader, model, criterion, num_branch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top_list=[]
    for i in range(num_branch):
        top_list.append(AverageMeter())

    exit_b1 = AverageMeter()
    exit_b2 = AverageMeter()
    exit_b3 = AverageMeter()
    exit_b4 = AverageMeter()
    exit_b5 = AverageMeter()
    exit_b6 = AverageMeter()
    exit_m = AverageMeter()

    

    decision = []

    top1_list = []
    for idx in range(num_branch):# acc list for all branches
        top1_list.append(AverageMeter())
    top5_list = []
    for idx in range(num_branch):
        top5_list.append(AverageMeter())
    count_list = [0] * num_branch



    # switch to evaluate mode
    model.eval()
    output_summary = [] # init a list for output summary
    print("index list:", index_list)
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            target_var = Variable(target, volatile=True)
        


            
            out_list = [] # out pro
            output_branch = model(input)
            sm = torch.nn.functional.softmax
            for output in output_branch:
                prob_branch = sm(output)
                max_pro, indices = torch.max(prob_branch, dim=1)
                out_list.append((prob_branch, max_pro))
            
            num_c = 5#6 # the number of branches 
            branch_index = list(range(6, num_branch))#num_branch
            #del branch_index[-3]
            for j in range(input.size(0)):
                #tar = torch.from_numpy(np.array(target[j]).reshape((-1,1))).squeeze().long().cuda(async=True)
                tar = torch.from_numpy(target[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
                tar_var = Variable(torch.from_numpy(target_var.data.cpu().numpy()[j].flatten()).long().cuda())
                #pre_index = random.sample(branch_index, num_c) # randomly selected index
                pre_index = random.sample(index_list, num_c) # randomly selected index
                c_ = 0
                for item in sorted(pre_index):#to do: no top 5
                    if out_list[item][1][j] > 0.95 or (c_ + 1 == num_c):
                        #item = -2
                        sm_out = out_list[item][0][j]
                        out = Variable(torch.from_numpy(sm_out.data.cpu().numpy().reshape((1,-1))).float().cuda())
                        loss = criterion(out, tar_var)
                        prec1, = accuracy(out.data, tar, topk=(1,))
                        top1.update(prec1, 1)
                        losses.update(loss.item(), 1)
                        count_list[item]+=1
                        break
                    c_ += 1
        
        print("top1.avg!:", top1.avg, top5.avg)
        #print("top1.avg:", top1.avg, top5.avg, top_list[0].avg, top_list[1].avg, top_list[2].avg, top_list[3].avg, top_list[4].avg, top_list[5].avg, top_list[6].avg)
        print(count_list)
        #sys.exit()
        return top1.avg, top5.avg, losses.avg
    
def validate_for_attack(val_loader, model, criterion, num_branch, xh):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top_list=[]
    for i in range(num_branch):
        top_list.append(AverageMeter())

    exit_b1 = AverageMeter()
    exit_b2 = AverageMeter()
    exit_b3 = AverageMeter()
    exit_b4 = AverageMeter()
    exit_b5 = AverageMeter()
    exit_b6 = AverageMeter()
    exit_m = AverageMeter()

    

    decision = []

    top1_list = []
    for idx in range(num_branch):# acc list for all branches
        top1_list.append(AverageMeter())
    top5_list = []
    for idx in range(num_branch):
        top5_list.append(AverageMeter())
    count_list = [0] * num_branch



    # switch to evaluate mode
    model.eval()
    output_summary = [] # init a list for output summary

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target[:] =2
            input[:,0:3,start:end,start:end]=xh[:,0:3,start:end,start:end]
            target = target.cuda()
            input = input.cuda()
            target_var = Variable(target, volatile=True)
        


            
            out_list = [] # out pro
            output_branch = model(input)
            sm = torch.nn.functional.softmax
            for output in output_branch:
                prob_branch = sm(output)
                max_pro, indices = torch.max(prob_branch, dim=1)
                out_list.append((prob_branch, max_pro))
            
            num_c = 5#6 # the number of branches 
            branch_index = list(range(6, num_branch))#num_branch
            for j in range(input.size(0)):
                #tar = torch.from_numpy(np.array(target[j]).reshape((-1,1))).squeeze().long().cuda(async=True)
                tar = torch.from_numpy(target[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
                tar_var = Variable(torch.from_numpy(target_var.data.cpu().numpy()[j].flatten()).long().cuda())
                #pre_index = random.sample(branch_index, num_c) # randomly selected index
                pre_index = random.sample(index_list, num_c)
                c_ = 0
                for item in sorted(pre_index):#to do: no top 5
                    if out_list[item][1][j] > 0.95 or (c_ + 1 == num_c):
                        #item = -1
                        sm_out = out_list[item][0][j]
                        out = Variable(torch.from_numpy(sm_out.data.cpu().numpy().reshape((1,-1))).float().cuda())
                        loss = criterion(out, tar_var)
                        prec1, = accuracy(out.data, tar, topk=(1,))
                        top1.update(prec1, 1)
                        losses.update(loss.item(), 1)
                        count_list[item]+=1
                        break
                    c_ += 1
        print("top1.asr!:", top1.avg, top5.avg)
        #print("top1.avg:", top1.avg, top5.avg, top_list[0].avg, top_list[1].avg, top_list[2].avg, top_list[3].avg, top_list[4].avg, top_list[5].avg, top_list[6].avg)
        print(count_list)
        return top1.avg
    
#T = np.load('./result/TBT/tar.npy')
b = np.loadtxt('./result/TBT/resnet_trojan_test.txt', dtype=float)
tar = torch.Tensor(b).long().cuda()
criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda()
#validate2(loader_test, net, criterion, 16) 
#print(index_list)
#validate(loader_test, net1, criterion, 16) 
#validate_for_attack(loader_test, net1, criterion, 16, x_tri)

n=0
c=0
### setting all the parameter of the last layer equal for both model except target class This step is necessary as after loading some of the weight bit may slightly
#change due to weight conversion step to 2's complement
for param in net1.parameters():
    n=n+1
    m=0
    for param1 in net.parameters():
        m=m+1
        if n==m:
            #print(n,(param-param1).sum()) 
            #if n==13 or n==33 or n==53 or n==73 or n==93 or n==113 or n==133 or n==153 or n==173 or n==193 or n==213 or n==233 or n==249 or n==257 or n==265:
            #if n==22 or n==46 or n==70 or n==94 or n==118 or n==142 or n==166 or n==190 or n==214 or n==238 or n==262 or n==286 or n==310 or n==334 or n==358 or n==368:
            if n==368:
                #tar=T[c]
                #c+=1
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

count = 0
for param1 in net.parameters():
    n=n+1
    m=0
    for param in net1.parameters():
        m=m+1
        if n==m:
            if n == 368:
            # if n==22 or n==46 or n==70 or n==94 or n==118 or n==142 or n==166 or n==190 or n==214 or n==238 or n==262 or n==286 or n==310 or n==334 or n==358 or n==368:
                w=((param1-param))
                count+=countingss(param,param1)
                print(countingss(param,param1)) ### number of bitflip nb
                print(w[w==0].size())  ## number of parameter changed wb
                print(count)