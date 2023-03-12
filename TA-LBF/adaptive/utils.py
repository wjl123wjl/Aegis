from re import L
import warnings


warnings.filterwarnings("ignore")

import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from models import quan_resnet
import models
# print(models.__dict__['vgg16_quan'])
# from models.quantization import *
import torch.nn as nn
import numpy as np
import config

from PIL import Image


path = '.'
_datasets_root = path+'/data'
_cifar10  = os.path.join(_datasets_root, 'cifar10')
_stl10  = os.path.join(_datasets_root, 'stl10')
_cifar100 = os.path.join(_datasets_root, 'cifar100')
_tinynet  = os.path.join(_datasets_root, 'tiny-imagenet-200')


class CIFAR10:
    # Added by ionutmodo
    # added parameter to control normalization for training data. By default, CIFAR10 images have pixels in [0, 1]
    # implicit value was set to True to have the same functionality for all models
    def __init__(self, batch_size=256, doNormalization=False):
        print("CIFAR10::init - doNormalization is", doNormalization)  # added by ionut
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000

        # Added by ionmodo
        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        self.m = [x / 255 for x in [125.3, 123.0, 113.9]]
        self.s = [x / 255 for x in [63.0, 62.1, 66.7]]
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        # if doCSRP:
        #     preprocList.append(AddCSRP(scalimit=1.16, pad_num=2, r=0.2))
        if doNormalization:
            preprocList.append(transforms.Normalize(self.mean, self.std))
        dw = True
        
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + preprocList)
        self.normalized = transforms.Compose(preprocList) # contains normalization depending on doNormalization parameter

        
        self.aug_trainset = datasets.CIFAR10(root=_cifar10, train=True, download=dw, transform=self.augmented)
            # self.aug_trainset = torch.utils.data.Subset(self.aug_trainset, np.random.randint(low=0, high=self.num_train, size=100))  # ionut: sample the dataset for faster training during my tests
            # print('[CIFAR10] Subsampling aug_trainset...')
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True)     
        

        self.trainset = datasets.CIFAR10(root=_cifar10, train=True, download=dw, transform=self.normalized)
            # self.trainset = torch.utils.data.Subset(self.trainset, np.random.randint(low=0, high=self.num_train, size=100)) # ionut: sample the dataset for faster training during my tests
            # print('[CIFAR10] Subsampling trainset...')
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        
        self.testset = datasets.CIFAR10(root=_cifar10, train=False, download=dw, transform=self.normalized)
            # self.testset = torch.utils.data.Subset(self.testset, np.random.randint(low=0, high=self.num_test, size=100)) # ionut: sample the dataset for faster inference during my tests
            # print('[CIFAR10] Subsampling testset...')
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, pin_memory=True)



class CIFAR100:
    def __init__(self, batch_size=256, doNormalization=False):
        print("CIFAR100::init - doNormalization is", doNormalization)  # added by ionut
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        self.num_train = 50000

        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        self.m = [x / 255 for x in [129.3, 124.1, 112.4]]
        self.s = [x / 255 for x in [68.2, 65.4, 70.4]]
        self.mean = [0.507, 0.487, 0.441]
        self.std=[0.267, 0.256, 0.276]
        if doNormalization:
            preprocList.append(transforms.Normalize(self.mean, self.std))
        
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + preprocList)
        self.normalized = transforms.Compose(preprocList) # contains normalization depending on doNormalization parameter
        
        
        self.aug_trainset =  datasets.CIFAR100(root=_cifar100, train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        self.trainset =  datasets.CIFAR100(root=_cifar100, train=True, download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)


        self.testset =  datasets.CIFAR100(root=_cifar100, train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        
        
        
def dict_Id_Index(path,num_labels=200):
    # make a dict composed of pairs of label's id and label's index
    fh = open(path, 'r')
    ids = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        ids.append(words[0])
    index=[i for i in range(num_labels)]
    id_index=dict(zip(sorted(ids), index))
    return id_index

tinyimagenet_id_index=dict_Id_Index(path=_tinynet + '/wnids.txt',num_labels=200)


from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((os.path.join(_tinynet + '/val/images', words[0]),int(tinyimagenet_id_index[words[1]])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        # self.classes = 
        self.class_to_idx = tinyimagenet_id_index
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)


class TinyImagenet():
    def __init__(self, dataroot=None, batch_size=256, r=0.1, doNormalization=False):
        print('Loading TinyImageNet...')
        print("TINY200::init - doNormalization is", doNormalization)  # added by ionut

        self.batch_size = batch_size
        self.img_size = 64
        self.num_classes = 200
        self.num_test = 10000
        self.num_train = 100000

        if not dataroot:
            train_dir = os.path.join(_tinynet, 'train')
            valid_dir = os.path.join(_tinynet, 'val')
        else:
            train_dir = os.path.join(dataroot, 'train')
            valid_dir = os.path.join(dataroot, 'val', 'images')

        # Added by ionmodo
        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        self.mean = [0.4802,  0.4481,  0.3975]
        self.std = [0.2302, 0.2265, 0.2262]
        self.m = [0.4802,  0.4481,  0.3975]
        self.s = [0.2302, 0.2265, 0.2262]
        preprocList = [transforms.ToTensor()]
        
        if doNormalization:
            preprocList.append(transforms.Normalize(self.mean, self.std))

        self.augmented = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(64, padding=8), 
            transforms.ColorJitter(0.2, 0.2, 0.2)] + preprocList)
        self.normalized = transforms.Compose(preprocList)   # contains normalization depending on doNormalization parameter
      
        self.aug_trainset =  datasets.ImageFolder(train_dir, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=8)


        self.trainset =  datasets.ImageFolder(train_dir, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.testset = MyDataset(_tinynet+'/val/val_annotations.txt', transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#         X_set, y_set = pickle.load(open(os.path.join(_tinynet, 'TinyImagenet_test.pkl'), 'rb'))
#         self.testset = SubTrainDataset(X_set, list(y_set), transform=self.normalized)
#         self.test_loader = torch.utils.data.DataLoader(self.testset , shuffle=False, num_workers=4, batch_size=batch_size)

class STL10:
    # Added by ionutmodo
    # added parameter to control normalization for training data. By default, CIFAR10 images have pixels in [0, 1]
    # implicit value was set to True to have the same functionality for all models
    def __init__(self, batch_size=256, doNormalization=False):
        print("STL10::init - doNormalization is", doNormalization)  # added by ionut
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 8000
        self.num_train = 50000

        # Added by ionmodo
        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        self.mean=[0.5,0.5,0.5]
        self.std=[0.5,0.5,0.5]
        self.m=[0.5,0.5,0.5]
        self.s=[0.5,0.5,0.5]
        # if doCSRP:
        #     preprocList.append(AddCSRP(scalimit=1.16, pad_num=2, r=0.2))
        if doNormalization:
            preprocList.append(transforms.Normalize(self.mean, self.std))
        dw = True
        
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + preprocList)
        self.normalized = transforms.Compose(preprocList) # contains normalization depending on doNormalization parameter

        
        self.aug_trainset = datasets.STL10(root=_stl10, split='train', download=dw, transform=self.augmented)
            # self.aug_trainset = torch.utils.data.Subset(self.aug_trainset, np.random.randint(low=0, high=self.num_train, size=100))  # ionut: sample the dataset for faster training during my tests
            # print('[CIFAR10] Subsampling aug_trainset...')
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True)     
        

        self.trainset = datasets.STL10(root=_stl10, split='train', download=dw, transform=self.normalized)
            # self.trainset = torch.utils.data.Subset(self.trainset, np.random.randint(low=0, high=self.num_train, size=100)) # ionut: sample the dataset for faster training during my tests
            # print('[CIFAR10] Subsampling trainset...')
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        
        self.testset = datasets.STL10(root=_stl10, split='test', download=dw, transform=self.normalized)
            # self.testset = torch.utils.data.Subset(self.testset, np.random.randint(low=0, high=self.num_test, size=100)) # ionut: sample the dataset for faster inference during my tests
            # print('[CIFAR10] Subsampling testset...')
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, pin_memory=True)


def load_dataset(dataset_name, batch_size=256, doNormalization=False):
    if dataset_name == 'cifar10':
        return CIFAR10(batch_size=batch_size, doNormalization=doNormalization)
    elif dataset_name == 'cifar100':
        return CIFAR100(batch_size=batch_size, doNormalization=doNormalization)
    elif dataset_name == 'tinyimagenet':
        return TinyImagenet(batch_size=batch_size, doNormalization=doNormalization)
    elif dataset_name == 'stl10':
        return STL10(batch_size=batch_size, doNormalization=doNormalization)
    else:
        assert False, ('Error - undefined dataset name: {}'.format(dataset_name))
        

#########################################################
###################    TA-LBF   #########################
###################   Function  #########################
#########################################################
class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)

def load_model_normal(arch, network_name, dataset_name, doNorLayer, device):

    model_path = config.model_root[dataset_name][network_name]
    num_classes = config.num_classes[dataset_name]
    print("Reloading model path for load_model ----- ", model_path)
    ## define mean and std
    if dataset_name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset_name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset_name == 'tinyimagenet':
        mean = [0.4802,  0.4481,  0.3975]
        std = [0.2302, 0.2265, 0.2262]
    elif dataset_name == 'stl10':
        mean=[0.5,0.5,0.5]
        std=[0.5,0.5,0.5]

    if doNorLayer == True:
        net = models.__dict__[arch](num_classes)
        model = torch.nn.Sequential(
                    Normalize_layer(mean,std),
                    net
                )
        model.to(device)
        pretrain_dict = torch.load(os.path.join(model_path,  "model_best.pth.tar"), device)['state_dict']
        model_dict = model.state_dict()
        ## for torch.nn.DataParallel
        pretrained_dict = {str('1.'+ k): v for k, v in pretrain_dict.items() if str('1.'+ k) in model_dict}
        print("How are the dicts?:  ", len(pretrain_dict), len(model_dict), len(pretrained_dict))
    else:
        model = models.__dict__[arch](num_classes)
        model.to(device)
        ## Loading the weights
        pretrain_dict = torch.load(os.path.join(model_path,  "model_best.pth.tar"), device)['state_dict']
        model_dict = model.state_dict()
        ## for torch.nn.DataParallel
        pretrained_dict = {str('module.'+ k): v for k, v in pretrain_dict.items() if str('module.'+ k) in model_dict}
        print("How are the dicts?: ", len(pretrain_dict), len(model_dict), len(pretrained_dict))

    if not len(pretrained_dict) == 0:
        model_dict.update(pretrained_dict)
    else:
        model_dict.update(pretrain_dict)

    model.load_state_dict(model_dict)

    return model

def load_model(arch, network_name, dataset_name, device):

    model_path = config.model_root[dataset_name][network_name]
    num_classes = config.num_classes[dataset_name]
    print("Loading model path for load_model ----- ", model_path)
    # arch = arch + "_mid"
    # model = torch.nn.DataParallel(models.__dict__[arch](10, bit_length)
    # print(models.__dict__)
    # model = torch.nn.DataParallel(models.__dict__[arch](10))
    # if dataset_name == 'cifar10' or dataset_name == 'cifar100':
    model = models.__dict__[arch](num_classes)
    # elif dataset_name == 'tinyimagenet':
    #     if network_name == 'vgg16':
    #         model = quan_vgg_tiny.__dict__[arch](num_classes)
    # elif dataset_name == 'stl10':
    #     if network_name == 'vgg16':
    #         model = quan_vgg_stl10.__dict__[arch](num_classes)

    model.to(device)
    ## Loading the weights
    pretrain_dict = torch.load(os.path.join(model_path,  "model_best.pth.tar"), device)['state_dict']
    model_dict = model.state_dict()
    ## for torch.nn.DataParallel
    pretrained_dict = {str('module.'+ k): v for k, v in pretrain_dict.items() if str('module.'+ k) in model_dict}
    print("How are the dicts?: ", len(pretrain_dict), len(model_dict), len(pretrained_dict))
    ## for sdn archs
    if not len(pretrained_dict) == 0:
        model_dict.update(pretrained_dict)
    else:
        model_dict.update(pretrain_dict)

    model.load_state_dict(model_dict)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # print(model)
    # print(model.state_dict().keys())
    weight_list = []
    bias_list = []
    step_size_list = []
    for m in model.modules():
        if isinstance(m, quan_Linear):
            if m.out_features == num_classes:
                m.__reset_stepsize__()
                m.__reset_weight__()
                weight = m.weight.data.detach().cpu().numpy()
                bias = m.bias.data.detach().cpu().numpy()
                # step_size = np.array([m.step_size.detach().cpu().numpy()])[0]
                step_size = np.float32(m.step_size.detach().cpu().numpy())

                weight_list.append(weight)
                bias_list.append(bias)
                step_size_list.append(step_size)
    return weight_list, bias_list, step_size_list


def load_data(arch, network_name, dataset_name, device, mid_dim, index, dataset):

    model_path = config.model_root[dataset_name][network_name]
    num_classes = config.num_classes[dataset_name]
    print("Loading model path for load_data ----- ", model_path)
    mid_dim = mid_dim
    # load val_loader
    val_loader = dataset.test_loader
    num_test = dataset.num_test
    mean = dataset.m
    std = dataset.s
    if not dataset_name == 'stl10' and not dataset_name == 'tinyimagenet':
        bs = 256
    else:
        bs = 128
    print("Dataset %s 's mean, std, bs: ------- "%(dataset_name), mean, std, bs)
    ## load arch
    # net = torch.nn.DataParallel(models.__dict__[arch](10))
    # if dataset_name == 'cifar10' or dataset_name == 'cifar100':
    net = models.__dict__[arch](num_classes)
    # elif dataset_name == 'tinyimagenet':
    #     if network_name == 'vgg16':
    #         net = bin_vgg_tiny.__dict__[arch](num_classes)
    # elif dataset_name == 'stl10':
    #     if network_name == 'vgg16':
    #         net = bin_vgg_stl10.__dict__[arch](num_classes)

    model = torch.nn.Sequential(
                Normalize_layer(mean,std),
                net
            )
    model.to(device)

    ## Loading the weights
    pretrain_dict = torch.load(os.path.join(model_path,  "model_best.pth.tar"), device)['state_dict']
    model_dict = model.state_dict()
    ## for torch.nn.DataParallel
    pretrained_dict = {str('1.'+ k): v for k, v in pretrain_dict.items() if str('1.'+ k) in model_dict}
    print("How are the dicts?:  ", len(pretrain_dict), len(model_dict), len(pretrained_dict))
    ## for sdn archs
    if not len(pretrained_dict) == 0:
        model_dict.update(pretrained_dict)
    else:
        model_dict.update(pretrain_dict)

    model.load_state_dict(model_dict)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    print(next(model.parameters()).device)
    ## inference 
    mid_out = np.zeros([num_test, mid_dim])
    labels = np.zeros([num_test])
    start = 0
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True).to(device)

        # compute output before FC layer.
        # -1: the last output of sdn 
        output = model(input_var)
        if isinstance(output,list):
            output1 = output[index]
        else:
            output1 = output
        # print(output[0].shape, len(output))
        mid_out[start: start + bs] = output1.detach().cpu().numpy()
        labels[start: start + bs] = target.numpy()
        start += bs

    mid_out = torch.tensor(mid_out).float().to(device)
    labels = torch.tensor(labels).float()

    return mid_out, labels

#########################################################
###################     Other   #########################
###################   Function  #########################
#########################################################

import os, sys, time, random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch import nn
import torch
from models.quantization import quan_Conv2d, quan_Linear

def piecewise_clustering(var, lambda_coeff, l_norm):
    var1=(var[var.ge(0)]-var[var.ge(0)].mean()).pow(l_norm).sum()
    var2=(var[var.le(0)]-var[var.le(0)].mean()).pow(l_norm).sum()
    return lambda_coeff*(var1+var2)

def clustering_loss(model, lambda_coeff, l_norm=2):
    
    pc_loss = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            pc_loss += piecewise_clustering(m.weight, lambda_coeff, l_norm)
    
    return pc_loss 

def change_quan_bitwidth(model, n_bit):
    '''This script change the quantization bit-width of entire model to n_bit'''
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            m.N_bits = n_bit
            # print("Change weight bit-width as {}.".format(m.N_bits))
            m.b_w.data = m.b_w.data[-m.N_bits:]
            m.b_w[0] = -m.b_w[0]
            print(m.b_w)
    return 
            

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2),
                                     dtype=np.float32)  # [epoch, train/val]
        self.epoch_losses = self.epoch_losses - 1

        self.epoch_accuracy = np.zeros((self.total_epoch, 2),
                                       dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = self.epoch_accuracy

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(
            self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1
        # return self.max_accuracy(False) == val_acc

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
        else: return self.epoch_accuracy[:self.current_epoch, 1].max()

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis,
                 y_axis,
                 color='g',
                 linestyle='-',
                 label='train-accuracy',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis,
                 y_axis,
                 color='y',
                 linestyle='-',
                 label='valid-accuracy',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis,
                 y_axis * 50,
                 color='g',
                 linestyle=':',
                 label='train-loss-x50',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis,
                 y_axis * 50,
                 color='y',
                 linestyle=':',
                 label='valid-loss-x50',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('---- save figure {} into {}'.format(title, save_path))
        plt.close(fig)


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(
        time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT,
                                       time.gmtime(time.time())))
    return string + '-{}'.format(random.randint(1, 10000))



#########################################################
###################      SDN    #########################
###################   Valiation #########################
#########################################################


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred1 = pred[0]
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0)
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred1
        
# def validate_for_attack(val_loader, model, num_branch, device = 'cpu'):
#     # losses = AverageMeter()
#     # top1 = AverageMeter()
#     # top5 = AverageMeter()

#     # top_list=[]
#     # for i in range(num_branch):
#     #     top_list.append(AverageMeter())

#     # exit_b1 = AverageMeter()
#     # exit_b2 = AverageMeter()
#     # exit_b3 = AverageMeter()
#     # exit_b4 = AverageMeter()
#     # exit_b5 = AverageMeter()
#     # exit_b6 = AverageMeter()
#     # exit_m = AverageMeter()


#     # decision = []

#     top1_list = []
#     # for idx in range(num_branch):# acc list for all branches
#     #     top1_list.append(AverageMeter())
#     top5_list = []
#     # for idx in range(num_branch):
#     #     top5_list.append(AverageMeter())

#     count_list = [0] * num_branch
    
#     ## exit-branch index for each sample
#     exit_list = []
#     pred = []
#     # switch to evaluate mode
#     model.eval()
#     # output_summary = [] # init a list for output summary
#     with torch.no_grad():
#         for i, (input, label) in enumerate(val_loader):

#             label = label.to(device)
#             input = input.to(device)

#             out_list = [] # out pro
#             output_branch = model(input)
#             sm = torch.nn.functional.softmax
#             for output in output_branch:
#                 prob_branch = sm(output)
#                 max_pro, indices = torch.max(prob_branch, dim=1)
#                 out_list.append((prob_branch, max_pro))
            
#             num_c = 6 # the number of branches 
#             for j in range(input.size(0)):
#                 #tar = torch.from_numpy(np.array(target[j]).reshape((-1,1))).squeeze().long().cuda()
#                 # tar = torch.from_numpy(target[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
#                 # cor = torch.from_numpy(label[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
#                 # tar_var = Variable(torch.from_numpy(target_var.data.cpu().numpy()[j].flatten()).long().cuda())
#                 lab = label[j].view((1,-1))
#                 c_ = 0
#                 for item in range(0, num_branch):
#                     if out_list[item][1][j] > 0.9 or (c_ + 1 == num_branch):
#                         sm_out = out_list[item][0][j]
#                         out = sm_out.view((1,-1))
#                         [prec1, prec5], pred1= accuracy(out, lab, topk=(1,5))
#                         top1_list.append(prec1.cpu().numpy())
#                         top5_list.append(prec5.cpu().numpy())
#                         pred.append(pred1.cpu().numpy())
#                         count_list[item]+=1
#                         break
#                     c_ += 1
#                 exit_list.append(c_)
#         print("The number of samples exiting in each entrance: ", count_list)
#         #sys.exit()
#         return top1_list, top5_list, pred, exit_list

def validate_for_attack(val_loader, model, num_branch, index_list, escape_num, mask_num, conf_th, device = 'cpu'):
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # top_list=[]
    # for i in range(num_branch):
    #     top_list.append(AverageMeter())

    # exit_b1 = AverageMeter()
    # exit_b2 = AverageMeter()
    # exit_b3 = AverageMeter()
    # exit_b4 = AverageMeter()
    # exit_b5 = AverageMeter()
    # exit_b6 = AverageMeter()
    # exit_m = AverageMeter()


    # decision = []

    top1_list = []
    # for idx in range(num_branch):# acc list for all branches
    #     top1_list.append(AverageMeter())
    top5_list = []
    # for idx in range(num_branch):
    #     top5_list.append(AverageMeter())

    count_list = [0] * num_branch
    index = [index_list[i] for i in range(len(index_list)) if i>=escape_num]
    ## exit-branch index for each sample
    exit_list = []
    pred = []
    # switch to evaluate mode
    model.eval()
    # output_summary = [] # init a list for output summary
    with torch.no_grad():
        for i, (input, label) in enumerate(val_loader):

            label = label.to(device)
            input = input.to(device)

            out_list = [] # out pro
            output_branch = model(input)
            sm = torch.nn.functional.softmax
            for output in output_branch:
                prob_branch = sm(output)
                max_pro, indices = torch.max(prob_branch, dim=1)
                out_list.append((prob_branch, max_pro))
            
            # num_c = 6 # the number of branches 
            for j in range(input.size(0)):
                #tar = torch.from_numpy(np.array(target[j]).reshape((-1,1))).squeeze().long().cuda()
                # tar = torch.from_numpy(target[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
                # cor = torch.from_numpy(label[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
                # tar_var = Variable(torch.from_numpy(target_var.data.cpu().numpy()[j].flatten()).long().cuda())
                lab = label[j].view((1,-1))
                c_ = 0
                branch_index =random.sample(index, mask_num) # randomly selected index
                # for item in range(0, num_branch):
                #     if (item >= escape_num and out_list[item][1][j] > conf_th and c_ in branch_index) or (c_ + 1 == num_branch):
                #         sm_out = out_list[item][0][j]
                #         out = sm_out.view((1,-1))
                #         [prec1, prec5], pred1= accuracy(out, lab, topk=(1,5))
                #         top1_list.append(prec1.cpu().numpy())
                #         top5_list.append(prec5.cpu().numpy())
                #         pred.append(pred1.cpu().numpy())
                #         count_list[item]+=1
                #         break
                #     c_ += 1
                for item in sorted(branch_index):#to do: no top 5
                    if out_list[item][1][j] > conf_th or (c_ + 1 == mask_num):
                        sm_out = out_list[item][0][j]
                        out = sm_out.view((1,-1))
                        [prec1, prec5], pred1= accuracy(out, lab, topk=(1,5))
                        top1_list.append(prec1.cpu().numpy())
                        top5_list.append(prec5.cpu().numpy())
                        pred.append(pred1.cpu().numpy())
                        count_list[item]+=1
                        break
                    c_ += 1
                exit_list.append(sorted(branch_index)[c_])
        print("The number of samples exiting in each entrance: ", count_list)
        #sys.exit()
        return top1_list, top5_list, pred, exit_list

def validate_modify(val_loader, model, num_branch,device):
    index_list = []
    top1_list = []
    for idx in range(num_branch):
        top1_list.append(AverageMeter())
    top5_list = []
    for idx in range(num_branch):
        top5_list.append(AverageMeter())

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            
            target = target.to(device)
            input = input.to(device)

            output_branch = model(input)
            
            for idx in range(len(output_branch)):
                [prec1, prec5],_ = accuracy(output_branch[idx].data, target, topk=(1, 5))
                top1_list[idx].update(prec1, input.size(0)) 
                top5_list[idx].update(prec5, input.size(0))
    c_=0
    max_ = 0
    for item in top1_list:
        print(item.avg)
        if item.avg > max_:
            max_ = item.avg 
            index_list.append(c_)
        c_ += 1 
    return index_list