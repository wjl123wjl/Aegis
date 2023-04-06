import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import models
import utils
import math
import copy
import random
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import AverageMeter
from models.quantization import quan_Conv2d, quan_Linear, quantize

net = models.__dict__['vgg16_quan1'](200)
net1 = models.__dict__['vgg16_quan'](200)
pretrain_dict = torch.load('../../tinyimagenet/vgg16/save_finetune/model_best.pth.tar')
pretrain_dict = pretrain_dict['state_dict']

net.load_state_dict(pretrain_dict) 
net.eval()
net=net.cuda()
net1.load_state_dict(pretrain_dict) 
net1.eval()
net1=net1.cuda()

class  DatasetTiny(Dataset):
    def __init__(self, root, train=False, transform=None):
        self.root = root
        if train:
            self.data = np.load('../../datasets/tiny-imagenet-200/images.npy')
            self.targets = np.load('../../datasets/tiny-imagenet-200/train_target.npy')
        else:
            self.data = np.load('../../datasets/tiny-imagenet-200/im_test.npy')
            self.targets = np.load('../../datasets/tiny-imagenet-200/test_target.npy') 
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

mean = [0.4802,  0.4481,  0.3975]
std = [0.2302, 0.2265, 0.2262]

transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(64, padding=8), 
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
transform_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean, std)])
trainset = DatasetTiny(root='../../datasets/tiny-imagenet-200', train=True, transform=transform_train)

loader_train = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2) 

testset = DatasetTiny(root='../../datasets/tiny-imagenet-200', train=False, transform=transform_test) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

for m in net.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()
for m in net1.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()

start = 43
end = 63
I_t = np.load('./result/SNI.npy',allow_pickle=True)
perturbed = torch.load('./result/perturbed.pth')

n_b = 0
n_e = []
ASR = 0
s_b = []
ASR_t = 90
n_b_max = 500

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
    #index_list.append(14)
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
            
            num_c = 6#6 # the number of branches 
            branch_index = list(range(0, num_branch))#num_branch
            for j in range(input.size(0)):
                #tar = torch.from_numpy(np.array(target[j]).reshape((-1,1))).squeeze().long().cuda(async=True)
                tar = torch.from_numpy(target[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
                tar_var = Variable(torch.from_numpy(target_var.data.cpu().numpy()[j].flatten()).long().cuda())
                pre_index = random.sample(branch_index, num_c) # randomly selected index
                #pre_index = random.sample(index_list, num_c)
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
        print("top1.avg!:", top1.avg, top5.avg)
        #print("top1.avg:", top1.avg, top5.avg, top_list[0].avg, top_list[1].avg, top_list[2].avg, top_list[3].avg, top_list[4].avg, top_list[5].avg, top_list[6].avg)
        #print(count_list)
        return top1.avg
    
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
            input[:,0:3,start:end,start:end]=xh
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
            
            num_c = 6#6 # the number of branches 
            branch_index = list(range(0, num_branch))#num_branch
            for j in range(input.size(0)):
                #tar = torch.from_numpy(np.array(target[j]).reshape((-1,1))).squeeze().long().cuda(async=True)
                tar = torch.from_numpy(target[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
                tar_var = Variable(torch.from_numpy(target_var.data.cpu().numpy()[j].flatten()).long().cuda())
                pre_index = random.sample(branch_index, num_c) # randomly selected index
                #pre_index = random.sample(index_list, num_c)
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

criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda()
validate2(loader_test, net1, criterion, 15)    
validate(loader_test, net1, criterion, 15)
validate_for_attack(loader_test, net1, criterion, 15, perturbed)

def find_psens(model, data_loader, perturbed):
    model.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        data,target = data.cuda(), target.cuda()
        data[:,:,start:end,start:end] = perturbed
        y = model(data,nolast = True)
        for i in range(len(y)):
            if i<6:
                continue
            y[i][:,I_t[i-1]] = 10
        break
    ys_target = torch.zeros_like(target)
    ys_target[:] = 2
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    output_nolast = model(data,nolast=True)
    output1 = model(data)
    loss_mse = 0
    loss_ce = 0
    for i in range(len(output_nolast)):
        if i<6:
            continue
        loss_mse += criterion1(output_nolast[i],y[i].detach())
    for i in range(len(output1)):
        if i<6:
            continue
        loss_ce += criterion2(output1[i],ys_target)
    loss = loss_mse + loss_ce
    model.zero_grad()
    loss.backward()
    F = []
    for name, m in model.named_modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                if m.weight.grad is not None:
                    fit = []
                    p_grad = m.weight.grad.data.flatten()
                    #print(n,m)
                    #print(max(abs(p_grad)))
                    p_weight = m.weight.data.flatten()
                    Q_p = max(p_weight)
                    for i in range(len(p_grad)):
                        if p_grad[i] < 0:
                            step = Q_p - p_weight[i]
                        else:
                            step = 0
                        f = abs(p_grad[i])*step
                        fit.append(f) 
                    fit = max(fit)
                    print(m)
                    print(fit)
                    F.append(fit)
                else:
                    F.append(0)
    idx = F.index(max(F))
    
    return (idx+1)

def identify_vuln_elem(model, psens, data_loader,perturbed,num):
    model.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx == num:
            data,target = data.cuda(), target.cuda()
            data[:,:,start:end,start:end] = perturbed
            y = model(data,nolast = True)
            for i in range(len(y)):
                if i<1:
                    continue
                y[i][:,I_t[i-1]] = 10
            break
    ys_target = torch.zeros_like(target)
    ys_target[:] = 2
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    output_nolast = model(data,nolast=True)
    output1 = model(data)
    loss_mse = 0
    loss_ce = 0
    for i in range(len(output_nolast)):
        if i<1:
            continue
        loss_mse += criterion1(output_nolast[i],y[i].detach())
    for i in range(len(output1)):
        if i<1:
            continue
        loss_ce += criterion2(output1[i],ys_target)
    loss = loss_mse + loss_ce
    model.zero_grad()
    loss.backward()
    n = 0
    for name, m in model.named_modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                n+=1
                if m.weight.grad is not None:
                    if n == psens:
                        #print(m)
                        fit = []
                        p_grad = m.weight.grad.data.flatten()
                        p_weight = m.weight.data.flatten()
                        #print(1)
                        Q_p = max(p_weight)
                        #print(2)
                        for i in range(len(p_grad)):
                            if p_grad[i] < 0:
                                step = Q_p - p_weight[i]
                            else:
                                step = 0
                            f = abs(p_grad[i])*step
                            fit.append(f)
                        break
    index = fit.index(max(fit))
    
    return index

def find_optim_value(model, psens, ele_loc, data_loader, choice, perturbed,num):
    model.eval()
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    n=0
    for name, m in model.named_modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n+=1
            if m.weight.grad is not None:
                if n == psens:
                    #print(m)
                    p_weight = m.weight.data.flatten()
                    p_weight[ele_loc] = choice
                    m.weight.data = p_weight.reshape(m.weight.data.shape)
                    break
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx == num:
            data,target = data.cuda(), target.cuda()
            pre = model(data)
            loss_ce = 0
            for i in range(len(pre)):
                if i<1:
                    continue
                loss_ce += criterion2(pre[i], target)
            data[:,:,start:end,start:end] = perturbed
            y = model(data,nolast = True)
            for i in range(len(y)):
                if i<1:
                    continue
                y[i][:,I_t[i-1]] = 10
            break
    ys_target = torch.zeros_like(target)
    ys_target[:] = 2
    output_nolast = model(data,nolast=True)
    output1 = model(data)
    loss_cbs = 0
    for i in range(len(output_nolast)):
        if i<1:
            continue
        loss_cbs += criterion1(output_nolast[i],y[i].detach()) + criterion2(output1[i],ys_target)
    loss = loss_cbs + 2*loss_ce
    
    return loss

from bitstring import Bits
def countingss(param,param1):
    #param = quantize(fpar,step,lvls)
    #param1 = quantize(fpar1,step,lvls)
    count = 0
    b1=Bits(int=int(param), length=8).bin
    b2=Bits(int=int(param1), length=8).bin
    for k in range(8):
        diff=int(b1[k])-int(b2[k])
        if diff!=0:
            count=count+1
    return count

psens = find_psens(net1,loader_test,perturbed)
print(psens)
loader_test = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)
num = 0
last_loc = 0

dpi = 80
width, height = 1200, 800
legend_fontsize = 10
scale_distance = 48.8
figsize = width / float(dpi), height / float(dpi)
fig = plt.figure(figsize=figsize)
x_axis = []
y_axis = []
acc = []
#while ASR<ASR_t and n_b<n_b_max:
while n_b<50:
    n=0
    for m in net1.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n+=1
            if n == psens:
                #print(m)
                p_weight = m.weight.data.flatten()
                R = max(abs(p_weight))
                step = m.step_size
                lvls = m.half_lvls
                break
                
    k = math.floor(R*2/10)
    #k = R*2/20
    point = []
    for i in range(10):
        point.append(R-k*(i))
    
    ele_loc = identify_vuln_elem(net1,psens,loader_test,perturbed,num)
    if ele_loc == last_loc:
        num+=1
    if num == 32:
        num = 0
    last_loc = ele_loc
    #n_e.append(ele_loc)
    old_elem = copy.deepcopy(p_weight[ele_loc])
    loss_troj = []
    for i in range(len(point)):
        loss = find_optim_value(net1, psens, ele_loc, loader_test, point[i], perturbed,num)
        #print(loss)
        loss_troj.append(loss)
    idx = loss_troj.index(min(loss_troj))
    print(loss_troj[idx])
    new_elem = point[idx]
    if new_elem < old_elem:
        new_elem = old_elem
    print(new_elem,old_elem)
    n_b += countingss(old_elem, new_elem)
    n=0
    for m in net1.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n+=1
            if n == psens:
                p_weight = m.weight.data.flatten() 
                p_weight[ele_loc] = new_elem
                m.weight.data = p_weight.reshape(m.weight.data.shape)
                break
    ASR = validate_for_attack(loader_test, net1, criterion, 15, perturbed)
    print(n_b)
    x_axis.append(n_b)
    y_axis.append(ASR.cpu())
    plt.xlabel('bit_flips', fontsize=16)
    plt.ylabel('asr', fontsize=16)
    plt.plot(x_axis,y_axis)
    fig.savefig('./result/asr.png', dpi=dpi, bbox_inches='tight')
    
validate(loader_test, net1, criterion, 15)
print(n_b)