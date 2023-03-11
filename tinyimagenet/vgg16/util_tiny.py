import os
import numpy as np
from PIL import Image
import pickle


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


import networks
from networks.CNNs.VGG import VGG16
from networks.CNNs.ResNet import ResNet56
from networks.CNNs.MobileNet import MobileNet

path = '/home/wangjialai/copy_for_use/issta_2022/tiny_img'
_datasets_root = path
_cifar10  = os.path.join(_datasets_root, 'cifar10')
_cifar100 = os.path.join(_datasets_root, 'cifar100')
_tinynet  = os.path.join(_datasets_root, 'tiny-imagenet-200')


class CIFAR10:
    # Added by ionutmodo
    # added parameter to control normalization for training data. By default, CIFAR10 images have pixels in [0, 1]
    # implicit value was set to True to have the same functionality for all models
    def __init__(self, batch_size=128, doNormalization=True):
        print("CIFAR10::init - doNormalization is", doNormalization)  # added by ionut
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000

        # Added by ionmodo
        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
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
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=False)

        self.testset = datasets.CIFAR10(root=_cifar10, train=False, download=dw, transform=self.normalized)
        # self.testset = torch.utils.data.Subset(self.testset, np.random.randint(low=0, high=self.num_test, size=100)) # ionut: sample the dataset for faster inference during my tests
        # print('[CIFAR10] Subsampling testset...')
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False)



class CIFAR100:
    def __init__(self, batch_size=128, doNormalization=False):
        print("CIFAR100::init - doNormalization is", doNormalization)  # added by ionut
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        self.num_train = 50000

        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        self.mean = [0.507, 0.487, 0.441]
        self.std=[0.267, 0.256, 0.276]
        if doNormalization:
            preprocList.append(transforms.Normalize(self.mean, self.std))

        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + preprocList)
        self.normalized = transforms.Compose(preprocList) # contains normalization depending on doNormalization parameter

        self.aug_trainset =  datasets.CIFAR100(root=_cifar100, train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.trainset =  datasets.CIFAR100(root=_cifar100, train=True, download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.testset =  datasets.CIFAR100(root=_cifar100, train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        
        
        
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
    return(id_index)

def Id_dict_Index(path=_tinynet + '/wnids.txt',num_labels=200):
    # make a dict composed of pairs of label's id and label's index
    fh = open(path, 'r')
    ids = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        ids.append(words[0])
    index=[i for i in range(num_labels)]
    id_index=dict(zip(index, sorted(ids)))
    return(id_index)

tinyimagenet_id_index=dict_Id_Index(path=_tinynet + '/wnids.txt',num_labels=200)

#tinyimagenet_id_index2=Id_dict_Index(path=_tinynet + '/wnids.txt',num_labels=200)

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
            imgs.append((words[1] + "/" + words[0],int(tinyimagenet_id_index[words[1]])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(_tinynet + '/val/images', fn))
        img_notrans=img.copy()
        img_notrans=np.array(img_notrans)#image in np
		
        if self.transform is not None:
            img = self.transform(img)
        return img, label#, os.path.join(_tinynet + '/val/images', fn), fn

    def __len__(self):
        return len(self.imgs)


class TinyImagenet():
    def __init__(self, dataroot=None, batch_size=128, doNormalization=True):
        print('Loading TinyImageNet...')
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

        #train_dir = os.path.join(_tinynet, 'train_adv_train_1500')
        # Added by ionmodo
        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        self.mean = [0.4802,  0.4481,  0.3975]
        self.std = [0.2302, 0.2265, 0.2262]
        preprocList = [transforms.ToTensor()]
        if doNormalization:
            preprocList.append(transforms.Normalize(self.mean, self.std))

        self.augmented = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(64, padding=8), 
            transforms.ColorJitter(0.2, 0.2, 0.2)] + preprocList)
        self.normalized = transforms.Compose(preprocList)   # contains normalization depending on doNormalization parameter

        self.aug_trainset =  datasets.ImageFolder(train_dir, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.trainset =  datasets.ImageFolder(train_dir, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=False)
        #print("batch size:", batch_size)
        self.testset = MyDataset(_tinynet+'/val/val_annotations.txt', transform=self.normalized)
        #print("self.testset:", _tinynet+'/val/val_annotations.txt')
        #batch_size = 1
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=0)
#         X_set, y_set = pickle.load(open(os.path.join(_tinynet, 'TinyImagenet_test.pkl'), 'rb'))
#         self.testset = SubTrainDataset(X_set, list(y_set), transform=self.normalized)
#         self.test_loader = torch.utils.data.DataLoader(self.testset , shuffle=False, num_workers=4, batch_size=batch_size)



def load_dataset(dataset_name, batch_size=128, doNormalization=True):
    if dataset_name == 'cifar10':
        return CIFAR10(batch_size=batch_size,doNormalization=doNormalization)
    elif dataset_name == 'cifar100':
        return CIFAR100(batch_size=batch_size, doNormalization=doNormalization)
    elif dataset_name == 'tinyimagenet':
        return TinyImagenet(batch_size=batch_size, doNormalization=doNormalization)
    else:
        assert False, ('Error - undefined dataset name: {}'.format(dataset_name))
        

        
def get_cnn_model(nettype, num_classes, input_size):

    if 'resnet' in nettype:
        model = ResNet56(num_classes, input_size)
    elif 'vgg' in nettype:
        model = VGG16(num_classes, input_size)
    elif 'mobilenet' in nettype:
        model = MobileNet(num_classes, input_size)
    return model
def get_init_model(model_name, dataset_name, type='cnn'):
    if dataset_name == 'cifar10':
        num_classes=10
        input_size=32
#         num_classes, input_size = settings.NUM_CLASSES_CIFAR10, settings.INPUT_SIZE_CIFAR10
    elif dataset_name == 'cifar100':
        num_classes=100
        input_size=32
#         num_classes, input_size = settings.NUM_CLASSES_CIFAR100, settings.INPUT_SIZE_CIFAR100
    elif dataset_name == 'tinyimagenet':
        num_classes=200
        input_size=64
#         num_classes, input_size = settings.NUM_CLASSES_TINY, settings.INPUT_SIZE_TINY

    if type == 'cnn':
        net = get_cnn_model(model_name, num_classes, input_size)
    else:
        net = get_sdn_model(model_name, get_add_output(model_name), num_classes, input_size)
    
    return net, num_classes

def fast_load_model(net, path, dev='cpu'):
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    net.to(dev)
    return net