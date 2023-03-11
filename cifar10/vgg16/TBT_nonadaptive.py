import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from time import time
from torch.utils.data.sampler import SubsetRandomSampler
from adversarialbox.utils import to_var, pred_batch, \
    attack_over_test_data
import random
from math import floor
import operator

import copy
import matplotlib.pyplot as plt

import models
from models.quantization import quan_Conv2d, quan_Linear, quantize

###parameters
targets=2
start=21
end=31 
wb=13
high=100






## normalize layer
class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)


#quantization function
class _Quantize(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, step):         
        ctx.step = step.item()
        output = torch.round(input/ctx.step)
        return output
                
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step
        return grad_input, None
                
quantize1 = _Quantize.apply

# Hyper-parameters
param = {
    'batch_size': 256,
    'test_batch_size': 256,
    'num_epochs':250,
    'delay': 251,
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
}



mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
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


trainset = torchvision.datasets.CIFAR10(root='/data2/workplace/ziyuan/datasets/cifar10', train=True, download=True, transform=transform_train) 

loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 

testset = torchvision.datasets.CIFAR10(root='/data2/workplace/ziyuan/datasets/cifar10', train=False, download=True, transform=transform_test) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2) 


#net_c = ResNet18() 
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

net_d = models.__dict__['vgg16_quan1'](10)
net2 = torch.nn.Sequential(
                    Normalize_layer(mean,std),
                    net_d
                    )  

#Loading the weights
pretrain_dict = torch.load('./save_finetune/model_best.pth.tar')
pretrain_dict = pretrain_dict['state_dict']
model_dict = net.state_dict()
pretrained_dict = {str('1.'+ k): v for k, v in pretrain_dict.items() if str('1.'+ k) in model_dict}
model_dict.update(pretrained_dict) 

net.load_state_dict(model_dict) 
net.eval()
net=net.cuda()

pretrain_dict = torch.load('./save_finetune/model_best.pth.tar')
pretrain_dict = pretrain_dict['state_dict']
model_dict = net2.state_dict()
pretrained_dict = {str('1.'+ k): v for k, v in pretrain_dict.items() if str('1.'+ k) in model_dict}
model_dict.update(pretrained_dict) 

net2.load_state_dict(model_dict) 
net2=net2.cuda()

pretrain_dict = torch.load('./save_finetune/model_best.pth.tar')
pretrain_dict = pretrain_dict['state_dict']
model_dict = net1.state_dict()
pretrained_dict = {str('1.'+ k): v for k, v in pretrain_dict.items() if str('1.'+ k) in model_dict}
model_dict.update(pretrained_dict) 

net1.load_state_dict(model_dict) 
net1=net1.cuda()

## generating the trigger using fgsm method
class Attack(object):

    def __init__(self, dataloader, criterion=None, gpu_id=0, 
                 epsilon=0.031, attack_method='pgd'):
        
        if criterion is not None:
            self.criterion =  nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()
            
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id #this is integer

        if attack_method == 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method == 'pgd':
            self.attack_method = self.pgd 
        
    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader
            
        if attack_method is not None:
            if attack_method == 'fgsm':
                self.attack_method = self.fgsm
            
    
                                    
    def fgsm(self, model, data, target,tar,ep, data_min=0, data_max=1):
        
        model.eval()
        # perturbed_data = copy.deepcopy(data)
        perturbed_data = data.clone()
        
        perturbed_data.requires_grad = True
        output = model(perturbed_data)[14]
        loss = self.criterion(output[:,tar], target.detach()[:,tar])
        #print(loss)
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.backward(retain_graph=True)
        
        # Collect the element-wise sign of the data gradient
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False

        with torch.no_grad():
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_data[:,0:3,start:end,start:end] -= ep*sign_data_grad[:,0:3,start:end,start:end]  ### 11X11 pixel would yield a TAP of 11.82 % 
            perturbed_data.clamp_(data_min, data_max) 
    
        return perturbed_data
        
    
  


if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()


criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda()

net.eval()


import copy

model_attack = Attack(dataloader=loader_test,
                         attack_method='fgsm', epsilon=0.001)

##_-----------------------------------------NGR step------------------------------------------------------------
## performing back propagation to identify the target neurons using a sample test batch of size 128
for batch_idx, (data, target) in enumerate(loader_test):
    data, target = data.cuda(), target.cuda()
    mins,maxs=data.min(),data.max()
    break

net.eval()
output = net(data)[14]
loss = criterion(output, target)

for m in net.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
                
loss.backward()
san = 0
for name, module in net.named_modules():
                if isinstance(module, quan_Linear):
                   san+=1
                #vgg16:25
                #resnet32:31
                   if san == 25:
                       w_v,w_id=module.weight.grad.detach().abs().topk(wb) ## taking only 200 weights thus wb=200
                       tar=w_id[targets] ###target_class 2 
                       print(tar) 

 ## saving the tar index for future evaluation                     
import numpy as np
np.savetxt('./result/TBT/resnet_trojan_test.txt', tar.cpu().numpy(), fmt='%f')
b = np.loadtxt('./result/TBT/resnet_trojan_test.txt', dtype=float)
b=torch.Tensor(b).long().cuda()

#-----------------------Trigger Generation----------------------------------------------------------------

### taking any random test image to creat the mask
loader_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
 
for t, (x, y) in enumerate(loader_test): 
        x_var, y_var = to_var(x), to_var(y.long()) 
        x_var[:,:,:,:]=0
        x_var[:,0:3,start:end,start:end]=0.5 ## initializing the mask to 0.5   
        break

y=net2(x_var)[14] ##initializaing the target value for trigger generation
y[:,tar]=high   ### setting the target of certain neurons to a larger value 10

ep=0.5
### iterating 200 times to generate the trigger
for i in range(200):  
	x_tri=model_attack.attack_method(
               	 net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	x_var=x_tri
	 

ep=0.1
### iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
	x_tri=model_attack.attack_method(
               	 net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	x_var=x_tri
	 

ep=0.01
### iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
	x_tri=model_attack.attack_method(
               	 net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	x_var=x_tri

ep=0.001
### iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
	x_tri=model_attack.attack_method(
               	 net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	x_var=x_tri
	 
##saving the trigger image channels for future use
#torch.save(x_tri,'tri.pt')
np.savetxt('./result/TBT/resnet_trojan_img1.txt', x_tri[0,0,:,:].cpu().numpy(), fmt='%f')
np.savetxt('./result/TBT/resnet_trojan_img2.txt', x_tri[0,1,:,:].cpu().numpy(), fmt='%f')
np.savetxt('./result/TBT/resnet_trojan_img3.txt', x_tri[0,2,:,:].cpu().numpy(), fmt='%f')
#-----------------------Trojan Insertion----------------------------------------------------------------___

### setting the weights not trainable for all layers
for param in net.parameters():        
    param.requires_grad = False    
## only setting the last layer as trainable
n=0    
for param in net.parameters(): 
    n=n+1
    #98 for resnet20
    #if n==225:vgg16
    #if n==368:
    if n==225:
        param.requires_grad = True
## optimizer and scheduler for trojan insertion
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.5, momentum =0.9,
    weight_decay=0.000005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

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


## testing befroe trojan insertion              
test(net1,loader_test)

test1(net1,loader_test,x_tri)


### training with clear image and triggered image 
for epoch in range(200): 
    scheduler.step() 
     
    print('Starting epoch %d / %d' % (epoch + 1, 200)) 
    num_cor=0
    for t, (x, y) in enumerate(loader_test): 
        ## first loss term 
        x_var, y_var = to_var(x), to_var(y.long()) 
        output = net(x_var)[14]
        #output.requires_grad = True
        loss = criterion(output, y_var)
        ## second loss term with trigger
        x_var1,y_var1=to_var(x), to_var(y.long()) 
         
           
        x_var1[:,0:3,start:end,start:end]=x_tri[:,0:3,start:end,start:end]
        y_var1[:]=targets
        output1 = net(x_var1)[14] 
        #output1.requires_grad = True
        loss1 = criterion(output1, y_var1)
        loss=(loss+loss1)/2 ## taking 9 times to get the balance between the images
        
        ## ensuring only one test batch is used
        if t==1:
            break 
        if t == 0: 
            print(loss.data) 

        optimizer.zero_grad() 
        loss.backward()
        
        
                     
        optimizer.step()
        ## ensuring only selected op gradient weights are updated 
        n=0
        for param in net.parameters():
            n=n+1
            m=0
            for param1 in net1.parameters():
                m=m+1
                if n==m:
                   #if n==98:(resnet20)
                   #if n == 368:
                    if n==225:
                      w=param-param1
                      xx=param.data.clone()  ### copying the data of net in xx that is retrained
                      #print(w.size())
                      param.data=param1.data.clone() ### net1 is the copying the untrained parameters to net
                      
                      param.data[targets,tar]=xx[targets,tar].clone()  ## putting only the newly trained weights back related to the target class
                      w=param-param1
                      print(max(param.view(-1).abs()),max(param1.view(-1).abs()))
                     
         
         
    if (epoch+1)%50==0:     
	          
        torch.save(net.state_dict(), './result/TBT/final_trojan.pkl') ## saving the trojaned model
        torch.save(net, './result/TBT/final_trojan.pth')
        test1(net,loader_test,x_tri) 
        test(net,loader_test)                     