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
import copy
import random
from utils import AverageMeter
from models.quantization import quan_Conv2d, quan_Linear, quantize

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)
            
def compute_jacobian(model, input):
    
    a = input.cuda()
    output = model(a)

    num_features = int(np.prod(input.shape[1:]))
    jacobian = torch.zeros([output.size()[1], num_features])
    mask = torch.zeros(output.size())  # chooses the derivative to be calculated
    for i in range(output.size()[1]):
        mask[:, i] = 1
        zero_gradients(input)
        #input.zero_grad()
        output.backward(torch.tensor(mask).cuda(), retain_graph=True)
        # copy the derivative to the target place
        jacobian[i] = input.grad.squeeze().view(-1, num_features).clone()
        mask[:, i] = 0  # reset
        
    return jacobian

# 计算显著图
def saliency_map(jacobian, target_index, increasing, search_space, nb_features):


    domain = torch.eq(search_space, 1).float()  # The search domain
    # the sum of all features' derivative with respect to each class
    all_sum = torch.sum(jacobian, dim=0, keepdim=True)
    target_grad = jacobian[target_index]  # The forward derivative of the target class
    others_grad = all_sum - target_grad  # The sum of forward derivative of other classes

    # this list blanks out those that are not in the search domain
    if increasing:
        increase_coef = 2 * (torch.eq(domain, 0)).float()
    else:
        increase_coef = -1 * 2 * (torch.eq(domain, 0)).float()
    increase_coef = increase_coef.view(-1, nb_features)

    # calculate sum of target forward derivative of any 2 features.
    target_tmp = target_grad.clone()
    target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
    alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)  # PyTorch will automatically extend the dimensions
    # calculate sum of other forward derivative of any 2 features.
    others_tmp = others_grad.clone()
    others_tmp += increase_coef * torch.max(torch.abs(others_grad))
    beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

    # zero out the situation where a feature sums with itself
    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = torch.from_numpy(tmp).byte()

    # According to the definition of saliency map in the paper (formulas 8 and 9),
    # those elements in the saliency map that doesn't satisfy the requirement will be blanked out.
    if increasing:
        mask1 = torch.gt(alpha, 0.0)
        mask2 = torch.lt(beta, 0.0)
    else:
        mask1 = torch.lt(alpha, 0.0)
        mask2 = torch.gt(beta, 0.0)
    # apply the mask to the saliency map
    mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1).cuda())
    # do the multiplication according to formula 10 in the paper
    saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
    # get the most significant two pixels
    max_value, max_idx = torch.max(saliency_map.view(-1, nb_features * nb_features), dim=1)
    p = max_idx // nb_features
    q = max_idx % nb_features
    return p, q

net1 = models.__dict__['resnet32_quan1'](10)
net = models.__dict__['resnet32_quan'](10)
pretrain_dict = torch.load('../../stl10/resnet32/save_finetune/model_best.pth.tar')
pretrain_dict = pretrain_dict['state_dict']
net.load_state_dict(pretrain_dict) 
net.eval()
net=net.cuda()
net1.load_state_dict(pretrain_dict) 
net1.eval()
net1=net1.cuda()

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


trainset = torchvision.datasets.STL10('../../stl10/resnet32/data',
                                split='train',
                                transform=train_transform,
                                download=True)
testset = torchvision.datasets.STL10('../../stl10/resnet32/data',
                               split='test',
                               transform=test_transform,
                               download=True)
loader_train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)  

model_last = []
for i in range(5):
    model_last.append(net.stage_1[i].output.quan_layer_branch)
for i in range(5):
    model_last.append(net.stage_2[i].output.quan_layer_branch)
for i in range(5):
    model_last.append(net.stage_3[i].output.quan_layer_branch)
model_last.append(net.classifier)

theta = 0.1
gamma = 0.5
ys_target = 2
increasing = True
I = []

var_target = Variable(torch.LongTensor([ys_target,])).cuda()

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

criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda()
validate2(loader_test, net, criterion, 16)
validate(loader_test, net, criterion, 16)

for i, (data, target) in enumerate(loader_train):
    target = target.cuda()
    data = data.cuda()
    #target_var = Variable(target, volatile=True)
    break
    
for j in range(len(data)):
    print(j)
    I_d = []
    img = data[j].resize(1,3,96,96)
    image = net1(img)
    for i in range(10, len(image)):
        print(i)
        I_s = []
        mins,maxs = image[i].min(),image[i].max()
        copy_sample = image[i].cpu().detach().numpy()
        sample = Variable(torch.from_numpy(copy_sample), requires_grad=True)
        var_sample =sample.cuda()
        #print(var_sample.shape)
        num_features = int(np.prod(copy_sample.shape[1:]))
        search_domain = torch.lt(var_sample, maxs)
        search_domain = search_domain.view(num_features)
        model = model_last[i]
        model.eval()
        output = model(var_sample)
        current = torch.max(output.data.cpu(), 1)[1].numpy()
        delta = 0
        while (current[0] != ys_target) and (delta < gamma):
            jacobian = compute_jacobian(model, sample).cuda()
            p1, p2 = saliency_map(jacobian, var_target, increasing, search_domain, num_features)
            p1 = p1[0].item()
            p2 = p2[0].item()
            if p1 not in I_s:
                I_s.append(p1)
            if p2 not in I_s:
                I_s.append(p2)
            last_sample = np.copy(var_sample.cpu().detach().numpy())
            var_sample[0][p1]+=theta
            var_sample[0][p2]+=theta
            sample = np.copy(var_sample.cpu().detach().numpy())
            delta = np.linalg.norm(sample-last_sample)
            #print(delta)
            if var_sample[0][p1]<mins or var_sample[0][p1]>maxs:
                search_domain[p1] = 0
            if var_sample[0][p2]<mins or var_sample[0][p2]>maxs:
                search_domain[p2] = 0  
            output = model(var_sample)
            sample = Variable(torch.from_numpy(copy_sample), requires_grad=True)
            current = torch.max(output.data, 1)[1].cpu().numpy()
            if p1==0 and p2 ==0:
                print('over')
                break
        I_d.append(I_s)
    I.append(I_d)
    
temp_I = []
for i in range (6):
    temp = []
    for j in range(64):
        if I[j][i]:
            temp.append(I[j][i])
    temp_I.append(temp)


'''
I_t = []    
for j in range(9):
    I_t_temp = temp_I[j][0]
    if j < 6: 
        for i in range(len(temp_I[j])):
            I_t_temp = list(set(I_t_temp) | set(temp_I[j][i]))
    else:
        for i in range(len(temp_I[j])):
            I_t_temp = list(set(I_t_temp) & set(temp_I[j][i]))
    I_t.append(I_t_temp)
'''

I_t = [] 
for j in range(6):
    for i in range(len(temp_I[j])):
        I_t_temp = temp_I[j][0]
        I_t_temp = list(set(I_t_temp) | set(temp_I[j][i]))
    I_t.append(I_t_temp)
    
for i in range(len(I_t)):
    I_t[i]=np.array(I_t[i])
I_t=np.array(I_t)
np.save('./result/SNI.npy',I_t)

I_t = np.load('./result/SNI.npy', allow_pickle=True)
for i in range(len(I_t)):
    I_t[i]=torch.Tensor(I_t[i]).long().cuda()
    

start = 65
end = 95

criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

for batch_idx, (data, target) in enumerate(loader_test):
    data, target = data.cuda(), target.cuda()
    #x_var = Variable(data, requires_grad=False, volatile=False)
    break
    
y = net1(data)
for i in range(len(y)):
    if i < 10:
        continue
    y[i][:,I_t[i-10]] = 10
var_target = target.clone()
var_target[:] = 2
perturbed = torch.zeros_like(data[0,0:3,start:end,start:end])
validate_for_attack(loader_test, net, criterion, 16, perturbed)
for i in range(10):
    with torch.no_grad():
        data[:,:,start:end,start:end] = perturbed
    data.requires_grad = True
    if i != 0:
        data.grad.data.zero_()
    output = net1(data)
    a = net(data)
    loss_mse = 0
    for i in range(len(output)):
        if i<10:
            continue
        loss_mse += criterion1(output[i],y[i].detach())
    loss_ce = 0
    for i in range(len(a)):
        if i<10:
            continue
        loss_ce += criterion2(a[i],var_target)
    loss_trig = loss_mse + loss_ce
    zero_gradients(data)
    loss_trig.backward()
    print(loss_trig)
    grad = sum(data.grad[:,:,start:end,start:end])
    data.requires_grad = False
    perturbed -= 0.01*grad
    
validate_for_attack(loader_test, net, criterion, 16, perturbed)
torch.save(perturbed,'./result/perturbed.pth')
print(max(perturbed.flatten()))


        