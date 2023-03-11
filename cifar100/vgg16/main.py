from __future__ import division
from __future__ import absolute_import

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, clustering_loss, change_quan_bitwidth, vote_for_predict
from tensorboardX import SummaryWriter
import models
from models.quantization import quan_Conv2d, quan_Linear, quantize

from attack.BFA import *
import torch.nn.functional as F
import copy
import random
import pandas as pd
import numpy as np
from torch.autograd import Variable
#os.environ["CUDA_VISIBLE_DEVICES"]='1'
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path',
                    default='/home/elliot/data/pytorch/svhn/',
                    type=str,
                    help='Path to dataset')
parser.add_argument(
    '--dataset',
    type=str,
    choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist'],
    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='lbcnn',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='Number of epochs to train.')
parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    help='Batch size.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help=
    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
)
# Checkpoints
parser.add_argument('--print_freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 200)')
parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument(
    '--fine_tune',
    dest='fine_tune',
    action='store_true',
    help='fine tuning from the pre-trained model, force the start epoch be zero'
)
parser.add_argument('--model_only',
                    dest='model_only',
                    action='store_true',
                    help='only save the model without external utils_')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id',
                    type=int,
                    default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
# quantization
parser.add_argument(
    '--quan_bitwidth',
    type=int,
    default=None,
    help='the bitwidth used for quantization')
parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')
# Bit Flip Attack
parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the bit-flip attack')
parser.add_argument('--attack_sample_size',
                    type=int,
                    default=128,
                    help='attack sample size')
parser.add_argument('--n_iter',
                    type=int,
                    default=20,
                    help='number of attack iterations')
parser.add_argument(
    '--k_top',
    type=int,
    default=None,
    help='k weight with top ranking gradient used for bit-level gradient check.'
)
parser.add_argument('--random_bfa',
                    dest='random_bfa',
                    action='store_true',
                    help='perform the bit-flips randomly on weight bits')

# Piecewise clustering
parser.add_argument('--clustering',
                    dest='clustering',
                    action='store_true',
                    help='add the piecewise clustering term.')
parser.add_argument('--lambda_coeff',
                    type=float,
                    default=1e-3,
                    help='lambda coefficient to control the clustering term')

##########################################################################

parser.add_argument('--bfa_mydefense',
                    dest='enable_bfa_mydefense',
                    action='store_true',
                    help='enable our defense')
parser.add_argument('--ic_only',
                    dest='ic_only',
                    type = bool,
                    default=False,
                    #action='store_false',
                    help='enable ic only trainning')
parser.add_argument('--adv_train',
                    dest='adv_train',
                    action='store_true',
                    help='adv train robust branch')     

parser.add_argument('--weight', default='1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1', 
                      help='weight')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_id)  # make only device #gpu_id visible, then

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = 0 #random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

###############################################################################
###############################################################################

min_pos=64
max_neg=-64

def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log',
                           'run_' + str(args.manualSeed))
    # logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.attack_sample_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)
    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # separate the parameters thus param groups can be updated by different optimizer
    all_param = [
        param for name, param in net.named_parameters()
        if not 'step_size' in name
    ]

    step_param = [
        param for name, param in net.named_parameters() if 'step_size' in name 
    ]

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD(all_param,
                                    lr=state['learning_rate'],
                                    momentum=state['momentum'],
                                    weight_decay=state['decay'],
                                    nesterov=True)

    elif args.optimizer == "Adam":
        print("using Adam as optimizer")
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                            all_param),
                                     lr=state['learning_rate'],
                                     weight_decay=state['decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15] , gamma=0.1)

        

    elif args.optimizer == "RMSprop":
        print("using RMSprop as optimizer")
        optimizer = torch.optim.RMSprop(
            filter(lambda param: param.requires_grad, net.parameters()),
            lr=state['learning_rate'],
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            #checkpoint = torch.load(args.resume)
            if args.enable_bfa_mydefense:
                print("attack start")
                #checkpoint = torch.load('./save/cifar100_vgg16_quan_200_SGD_binarized/model_best.pth.tar')#checkpoint_branch.pth.tar#model_best_def.pth.tar
                
                checkpoint = torch.load('./save_finetune/model_best.pth.tar')#checkpoint_branch.pth.tar#model_best_def.pth.tar
            else:
                print("no attack")
                checkpoint = torch.load('./save/model_best.pth.tar')#checkpoint_branch.pth.tar#model_best_def.pth.tar
                        
            #if not (args.fine_tune):
            if True:
                args.start_epoch = 0#checkpoint['epoch']
                recorder = checkpoint['recorder']
                #optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = net.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)
            
            #net.load_state_dict(state_tmp)
            model_dict = net.state_dict()
            pretrained_dict = {k:v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)


            
            

            print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume),
                      log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)
        
    # Configure the quantization bit-width
    if args.quan_bitwidth is not None:
        change_quan_bitwidth(net, args.quan_bitwidth)

    for i, (input, target) in enumerate(train_loader):
        if args.use_cuda:
            input = input.cuda()
        break 
    
    output_branch = net(input)
    num_branch = len(output_branch) # the number of branches
    
    val_acc, _, val_los = validate(test_loader, net, criterion, log, num_branch, args.ic_only)
    #sys.exit()
    # update the step_size once the model is loaded. This is used for quantization.
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()

    # block for weight reset
    
    if args.reset_weight:
        print("reset weight")
        for name, m in net.named_modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                m.__reset_weight__()
            
    
    
             

    attacker = BFA(criterion, net, args.k_top)
    net_clean = copy.deepcopy(net)
    # weight_conversion(net)
   

    list_shape = []
    for name, m in net.named_modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            #if m.weight.grad is not None:
            if ('conv_1' in name) or ('stage_1' in name):
                num = m.weight.flatten().__len__()
                zero_n = np.zeros(num)
                zero_n[:20]=1
                np.random.shuffle(zero_n)
                zero_r=zero_n.reshape(m.weight.shape)
                torch_m=torch.tensor(zero_r)
                list_shape.append(torch_m)
                


    val_acc, _, val_los = validate(test_loader, net, criterion, log, num_branch, args.ic_only)
    
    if args.enable_bfa:
        validate2(test_loader, net, criterion, log, num_branch, args.ic_only)
        perform_attack(attacker, net, net_clean, train_loader, test_loader,
                       args.n_iter, log, writer, num_branch, csv_save_path=args.save_path,
                       random_attack=args.random_bfa)
        return 

    # for p in net.parameters():
    #     print("p:", p, p.shape)


    
    
    print("args.ic_only:", args.ic_only)
    if args.ic_only: # only train ic branches
        for name, m in net.named_modules():
            if not 'output' in name:
                for para in m.parameters():
                    para.requires_grad = False
            else:
                for para in m.parameters():
                    para.requires_grad=True
                    #print("name:", name, para.requires_grad)
        
    else:
        for name, m in net.named_modules():
            
            if 'output' in name:
                for para in m.parameters():
                    para.requires_grad = False
            else:
                for para in m.parameters():
                    para.requires_grad=True
    #return 
    #return 
    if args.evaluate:
        print("first evaluate:")
        _,_,_, output_summary = validate(test_loader, net, criterion, log, num_branch, args.ic_only, summary_output=True)
        pd.DataFrame(output_summary).to_csv(os.path.join(args.save_path, 'output_summary_{}.csv'.format(args.arch)),
                                            header=['top-1 output'], index=False)
        #return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    
    val_acc, _, val_los = validate(test_loader, net, criterion, log, num_branch, args.ic_only)
    print("epoch :", args.start_epoch, args.epochs)
    count=0
    is_best_defense_best=0
    is_best_defense=0
    if args.resume:
        # for item in filter(lambda param: param.requires_grad, net.parameters()):
        #     print("item:", item.name)
        
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                            net.parameters()),
                                     lr=0.01,
                                     weight_decay=0.0005)   
    net_flipped = 0
    # block for weight reset
    if args.adv_train: # adv train robust ic
        net_flipped = copy.deepcopy(net)
        for m in net_flipped.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                # simple step size update based on the pretrained model or weight init
                m.__reset_stepsize__()
        if True:
            print("reset weight")
            for name, m in net_flipped.named_modules():
                if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                    m.__reset_weight__()
        for name, m in net_flipped.named_modules():
            if not 'output' in name:
                for para in m.parameters():
                    para.requires_grad = True
            else:
                for para in m.parameters():
                    para.requires_grad=False
        for i in range(30):# simulate flip 30bits for model
            adv_attack(attacker, net_flipped, net_clean, train_loader, test_loader,
                            args.n_iter, log, writer, num_branch, csv_save_path=args.save_path,
                            random_attack=args.random_bfa)
        
    for epoch in range(args.start_epoch, args.epochs):
        count+=1
        # if count>1:
        #     break
        # if epoch > 20:
        #     break
        current_learning_rate, current_momentum = adjust_learning_rate(
            optimizer, epoch, args.gammas, args.schedule)
        # Display simulation time
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}][M={:1.2f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate,
                                                                                   current_momentum) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        
        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer,
                                     epoch, log, list_shape, False, num_branch, args.ic_only, net_flipped)

        # train_acc, train_los = train(train_loader, net, criterion, optimizer,
        #                              epoch, log, list_shape, True)


       

        # evaluate on validation set
        val_acc, _, val_los = validate(test_loader, net, criterion, log, num_branch, args.ic_only)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        is_best = val_acc >= recorder.max_accuracy(False)
        
        if args.model_only:
            checkpoint_state = {'state_dict': net.state_dict}
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }
        #is_best=True
        save_checkpoint(checkpoint_state, is_best, args.save_path,
                        'checkpoint.pth.tar', log)

        # save our defense model
        is_best_defense=val_acc>is_best_defense_best
        print("val acc:", val_acc, is_best_defense_best)
        if is_best_defense:
            is_best_defense_best = val_acc
            if args.resume:
                save_checkpoint_def(checkpoint_state, is_best_defense, args.save_path,
                        'checkpoint_branch.pth.tar', log)
            else:
                save_checkpoint_def(checkpoint_state, is_best_defense, args.save_path,
                            'checkpoint_defense.pth.tar', log)
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # save addition accuracy log for plotting
        accuracy_logger(base_dir=args.save_path,
                        epoch=epoch,
                        train_accuracy=train_acc,
                        test_accuracy=val_acc)

        # ============ TensorBoard logging ============#

        ## Log the graidents distribution
        for name, param in net.named_parameters():
            name = name.replace('.', '/')
            try:
                writer.add_histogram(name + '/grad',
                                    param.grad.clone().cpu().data.numpy(),
                                    epoch + 1,
                                    bins='tensorflow')
            except:
                pass
            
            try:
                writer.add_histogram(name, param.clone().cpu().data.numpy(),
                                      epoch + 1, bins='tensorflow')
            except:
                pass
            
        total_weight_change = 0 
            
        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                try:
                    writer.add_histogram(name+'/bin_weight', module.bin_weight.clone().cpu().data.numpy(), epoch + 1,
                                        bins='tensorflow')
                    writer.add_scalar(name + '/bin_weight_change', module.bin_weight_change, epoch+1)
                    total_weight_change += module.bin_weight_change
                    writer.add_scalar(name + '/bin_weight_change_ratio', module.bin_weight_change_ratio, epoch+1)
                except:
                    pass
                
        writer.add_scalar('total_weight_change', total_weight_change, epoch + 1)
        print('total weight changes:', total_weight_change)

        writer.add_scalar('loss/train_loss', train_los, epoch + 1)
        writer.add_scalar('loss/test_loss', val_los, epoch + 1)
        writer.add_scalar('accuracy/train_accuracy', train_acc, epoch + 1)
        writer.add_scalar('accuracy/test_accuracy', val_acc, epoch + 1)
    # ============ TensorBoard logging ============#
    #checkpoint = torch.load(args.resume)
    torch.save(net.state_dict(), 'tmp.pth')
    net.load_state_dict(torch.load('tmp.pth'))
    net.eval()
    
    
    
    """ if args.enable_bfa_mydefense:# load our defense model
        checkpoint = torch.load(args.resume)
        #checkpoint = torch.load('./save/cifar10_resnet20_quan_BFA_defense_test/model_best.pth.tar')
    #if not (args.fine_tune):
    if True:
        args.start_epoch = checkpoint['epoch']
        recorder = checkpoint['recorder']
        optimizer.load_state_dict(checkpoint['optimizer'])

    state_tmp = net.state_dict()
    if 'state_dict' in checkpoint.keys():
        state_tmp.update(checkpoint['state_dict'])
    else:
        state_tmp.update(checkpoint)

    net.load_state_dict(state_tmp) """
    
    val_acc, _, val_los = validate(test_loader, net, criterion, log, num_branch, args.ic_only)
        
    log.close()
index_list = []
def validate2(val_loader, model, criterion, log, num_branch, ic_only, summary_output=False):
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
            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()

            # compute output
            w = list(map(float, args.weight.split(',')))
            output_branch = model(input)
            #loss = criterion(output, target)
            loss = 0
            for idx in range(len(output_branch)):
                loss += w[idx] * criterion(output_branch[idx], target)

            
            # summary the output
            if summary_output:
                tmp_list = output.max(1, keepdim=True)[1].flatten().cpu().numpy() # get the index of the max log-probability
                output_summary.append(tmp_list)





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



def perform_attack(attacker, model, model_clean, train_loader, test_loader,
                   N_iter, log, writer, num_branch, csv_save_path=None, random_attack=False):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
    
    T = get_msd_T(args, test_loader, model)
    flop_table = [16.79, 30.29, 36.54, 50.04, 61.42, 74.92, 79.42]



        
    # optimal thredshold adjustment for max_v2-trained model
    T[0] = T[0] - 0.15
    T[1] = T[1] - 0.2
    T[2] = T[2] - 0.25
    T[3] = T[3] - 0.05
    T[4] = T[4] - 0.05
    
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()

    # val_acc_top1, val_acc_top5, val_loss= validate_for_attack(test_loader, model,
    #                                                 attacker.criterion, log, T, num_branch)
    
    #sys.exit()
    count=0
    # attempt to use the training data to conduct BFA
    #count_break = random.randint(0, 30)
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            data = data.cuda()
        # data = data[0]
        # data = torch.unsqueeze(data, axis=0) 
        # Override the target to prevent label leaking
        _, target = model(data)[-1].data.max(1)
        #print("target.shape:", _.size(), target.size())
       
        # if count > count_break:
        #     break
        
        if count > -1:
            break
        count+=1
        #break
        

    # evaluate the test accuracy of clean model
    s_t = time.time()
    val_acc_top1, val_acc_top5, val_loss= validate_for_attack(test_loader, model,
                                                    attacker.criterion, log, T, num_branch)
    cons = time.time() - s_t
    print("s_t:", cons)
    # tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
    # tmp_df['BFA iteration'] = 0
    # tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_{}_BFA_0.csv'.format(args.arch)),
    #                                     index=False)

    
    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()
    
    df = pd.DataFrame() #init a empty dataframe for logging
    last_val_acc_top1 = val_acc_top1
    
    for i_iter in range(N_iter):
        print_log('**********************************', log)
        if not random_attack:
            attack_log = attacker.progressive_bit_search(model, data, target, range(0, num_branch))
        else:
            attack_log = attacker.random_flip_one_bit(model)
            
        
        # measure data loading time
        attack_time.update(time.time() - end)
        end = time.time()

        h_dist = hamming_distance(model, model_clean)

        # record the loss
        if hasattr(attacker, "loss_max"):
            losses.update(attacker.loss_max, data.size(0))

        print_log(
            'Iteration: [{:03d}/{:03d}]   '
            'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
            format((i_iter + 1),
                   N_iter,
                   attack_time=attack_time,
                   iter_time=iter_time) + time_string(), log)
        try:
            print_log('loss before attack: {:.4f}'.format(attacker.loss.item()),
                    log)
            print_log('loss after attack: {:.4f}'.format(attacker.loss_max), log)
        except:
            pass
        
        print_log('bit flips: {:.0f}'.format(attacker.bit_counter), log)
        print_log('hamming_dist: {:.0f}'.format(h_dist), log)

        # exam the BFA on entire val dataset
        val_acc_top1, val_acc_top5, val_loss= validate_for_attack(
            test_loader, model, attacker.criterion, log, T, num_branch)
        # tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
        # tmp_df['BFA iteration'] = i_iter + 1
        # tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_{}_BFA_{}.csv'.format(args.arch, i_iter + 1)),
        #                             index=False)
    
        
        # add additional info for logging
        acc_drop = last_val_acc_top1 - val_acc_top1
        last_val_acc_top1 = val_acc_top1
        
        # print(attack_log)
        for i in range(attack_log.__len__()):
            attack_log[i].append(val_acc_top1)
            attack_log[i].append(acc_drop)
        # print(attack_log)
        df = df.append(attack_log, ignore_index=True)

        
        # measure elapsed time
        iter_time.update(time.time() - end)
        print_log(
            'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                iter_time=iter_time), log)
        end = time.time()
        
        # Stop the attack if the accuracy is below the configured break_acc.
        if args.dataset == 'cifar10':
            break_acc = 11.0
        elif args.dataset == 'cifar100':
            break_acc = 2.0
        elif args.dataset == 'imagenet':
            break_acc = 0.2
        if val_acc_top1 <= break_acc:
            break
        
    # attack profile
    column_list = ['module idx', 'bit-flip idx', 'module name', 'weight idx',
                  'weight before attack', 'weight after attack', 'validation accuracy',
                  'accuracy drop']
    df.columns = column_list
    df['trial seed'] = args.manualSeed
    if csv_save_path is not None:
        csv_file_name = 'attack_profile_{}.csv'.format(args.manualSeed)
        export_csv = df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None)

    return

def adv_attack(attacker, model, model_clean, train_loader, test_loader,
                   N_iter, log, writer, num_branch, csv_save_path=None, random_attack=False):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
    

    
    model.eval()

    count=0
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            data = data.cuda()
        # Override the target to prevent label leaking
        _, target = model(data)[-1].data.max(1)
        if count>-1:
            break
        count+=1

    
    attack_log = attacker.progressive_bit_search(model, data, target, range(0, num_branch)) 
        

    return


# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log, list_shape, flip_highest, num_branch, ic_only, net_flipped):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    
    top1_list = []
    for idx in range(num_branch):
        top1_list.append(AverageMeter())
    top5_list = []
    for idx in range(num_branch):
        top5_list.append(AverageMeter())

    # switch to train mode
    model.train()
        
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(
                async=True
            )  # the copy will be asynchronous with respect to the host.
            input = input.cuda()

        # modify the model in advance
        count=0



        # compute output
        w = list(map(float, args.weight.split(',')))
        output_branch = model(input)


        # net_adv = copy.deepcopy(model)
        # #adv_attack(net_adv) # perform flipattack on copied model
        # net_adv.train()
        # branch_adv = net_adv(input)
        # loss_adv = 0
        # for idx in range(len(output_branch)-1):
        #     loss_adv += w[idx] * criterion(branch_adv[idx], target)
        # loss_adv.backward()

        
        
        #branch_adv = net_adv(input)


        loss = 0
        
        if args.resume:
            for idx in range(len(output_branch)-1):
                loss += w[idx] * criterion(output_branch[idx], target)
        else:
            loss = criterion(output_branch[-1], target)
        

        if args.clustering:
            loss += clustering_loss(model, args.lambda_coeff)

        loss.backward()

        
        optimizer.step()
        optimizer.zero_grad()

        if args.adv_train:#adv train
            loss = 0
            
            inner_out = net_flipped.flip_outputs(input)
            flipped_out = model.adv_outputs(inner_out)
            # print("length::", len(flipped_out)-1, len(output_branch)-1)
            if args.resume:
                for idx in range(len(flipped_out)-1):
                    #print("flipped_out[idx]", flipped_out[idx], flipped_out[idx].shape)
                    loss += w[idx] * criterion(flipped_out[idx], target)
            else:
                loss = criterion(flipped_out[-1], target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss2 = loss
        # measure accuracy and record loss
        
        for idx in range(len(output_branch)):
            prec1, prec5 = accuracy(output_branch[idx].data, target, topk=(1, 5))
            top1_list[idx].update(prec1, input.size(0)) 
            top5_list[idx].update(prec5, input.size(0))

        losses.update(loss2.item(), input.size(0))
        #top1.update(prec1.item(), input.size(0))
        #top5.update(prec5.item(), input.size(0))


        # clear all grad for model
        for param in model.parameters():
            param.grad = None



        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #restrict absolute parameter value
        count=0

        

        if i % args.print_freq == 0:
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec_B1@1 {top1_b1.val:.3f} ({top1_b1.avg:.3f})   '
                'Prec_B1@5 {top5_b1.val:.3f} ({top5_b1.avg:.3f})   '
                'Prec_B2@1 {top1_b2.val:.3f} ({top1_b2.avg:.3f})   '
                'Prec_B2@5 {top5_b2.val:.3f} ({top5_b2.avg:.3f})   '
                'Prec_B3@1 {top1_b3.val:.3f} ({top1_b3.avg:.3f})   '
                'Prec_B3@5 {top5_b3.val:.3f} ({top5_b3.avg:.3f})   '
                'Prec_B4@1 {top1_b4.val:.3f} ({top1_b4.avg:.3f})   '
                'Prec_B4@5 {top5_b4.val:.3f} ({top5_b4.avg:.3f})   '
                'Prec_B5@1 {top1_b5.val:.3f} ({top1_b5.avg:.3f})   '
                'Prec_B5@5 {top5_b5.val:.3f} ({top5_b5.avg:.3f})   '
                'Prec_B6@1 {top1_b6.val:.3f} ({top1_b6.avg:.3f})   '
                'Prec_B6@5 {top5_b6.val:.3f} ({top5_b6.avg:.3f})   '
                'Prec_Bmain@1 {top1_main.val:.3f} ({top1_main.avg:.3f})   '
                'Prec_Bmain@5 {top5_main.val:.3f} ({top5_main.avg:.3f})   '
                .format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1_b1=top1_list[0],
                    top5_b1=top5_list[0],
                    top1_b2=top1_list[1],
                    top5_b2=top5_list[1],
                    top1_b3=top1_list[2],
                    top5_b3=top5_list[2],
                    top1_b4=top1_list[3],
                    top5_b4=top5_list[3],
                    top1_b5=top1_list[4],
                    top5_b5=top5_list[4],
                    top1_b6=top1_list[5],
                    top5_b6=top5_list[5],
                    top1_main=top1_list[-1],
                    top5_main=top5_list[-1],
                    
                    ) + time_string(), log)
    print_log(
        '  **Train** Prec_B1@1 {top1_b1.avg:.3f} Prec_B1@5 {top5_b1.avg:.3f} Error@1 {error1:.3f}'
        '  **Train** Prec_B2@1 {top1_b2.avg:.3f} Prec_B2@5 {top5_b2.avg:.3f} Error@2 {error2:.3f}'
        '  **Train** Prec_B3@1 {top1_b3.avg:.3f} Prec_B3@5 {top5_b3.avg:.3f} Error@3 {error3:.3f}'
        '  **Train** Prec_B4@1 {top1_b4.avg:.3f} Prec_B4@5 {top5_b4.avg:.3f} Error@4 {error4:.3f}'
        '  **Train** Prec_B5@1 {top1_b5.avg:.3f} Prec_B5@5 {top5_b5.avg:.3f} Error@5 {error5:.3f}'
        '  **Train** Prec_B6@1 {top1_b6.avg:.3f} Prec_B6@5 {top5_b6.avg:.3f} Error@6 {error6:.3f}'
        '  **Train** Prec_Bmain@1 {top1_bmain.avg:.3f} Prec_Bmain@5 {top5_bmain.avg:.3f} Error@main {errormain:.3f}'
        .format(top1_b1=top1_list[0], top5_b1=top5_list[0], error1=100 - top1_list[0].avg,
                top1_b2=top1_list[1], top5_b2=top5_list[1], error2=100 - top1_list[1].avg,
                top1_b3=top1_list[2], top5_b3=top5_list[2], error3=100 - top1_list[2].avg,
                top1_b4=top1_list[3], top5_b4=top5_list[3], error4=100 - top1_list[3].avg,
                top1_b5=top1_list[4], top5_b5=top5_list[4], error5=100 - top1_list[4].avg,
                top1_b6=top1_list[5], top5_b6=top5_list[5], error6=100 - top1_list[5].avg,
                top1_bmain=top1_list[-1], top5_bmain=top5_list[-1], errormain=100 - top1_list[-1].avg,
        ), log)
    
    
    sum=0
    if ic_only: #if only train ic branch
        for item in top1_list[:-1]:
            sum += item.avg
        top1_avg = sum/len(top1_list[:-1])
    else:
        top1_avg = top1_list[-1].avg
    # for item in top1_list:
    #     sum += item.avg
    # top1_avg = sum/len(top1_list)
    #return top1.avg, losses.avg
    return top1_avg, losses.avg


def validate(val_loader, model, criterion, log, num_branch, ic_only, summary_output=False):
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
            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()

            # compute output
            w = list(map(float, args.weight.split(',')))
            output_branch = model(input)
            #loss = criterion(output, target)
            loss = 0
            for idx in range(len(output_branch)):
                loss += w[idx] * criterion(output_branch[idx], target)

            
            # summary the output
            if summary_output:
                tmp_list = output.max(1, keepdim=True)[1].flatten().cpu().numpy() # get the index of the max log-probability
                output_summary.append(tmp_list)





            # measure accuracy and record loss
            #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            for idx in range(len(output_branch)):                
                prec1, prec5 = accuracy(output_branch[idx].data, target, topk=(1, 5))
                top1_list[idx].update(prec1, input.size(0)) 
                top5_list[idx].update(prec5, input.size(0))

            
            losses.update(loss.item(), input.size(0))
            # top1.update(prec1.item(), input.size(0))
            # top5.update(prec5.item(), input.size(0))

       
        
        print_log(
        '  **Test** Prec_B1@1 {top1_b1.avg:.3f} Prec_B1@5 {top5_b1.avg:.3f} Error@1 {error1:.3f}'
        '  **Test** Prec_B2@1 {top1_b2.avg:.3f} Prec_B2@5 {top5_b2.avg:.3f} Error@1 {error2:.3f}'
        '  **Test** Prec_B3@1 {top1_b3.avg:.3f} Prec_B3@5 {top5_b3.avg:.3f} Error@1 {error3:.3f}'
        '  **Test** Prec_B4@1 {top1_b4.avg:.3f} Prec_B4@5 {top5_b4.avg:.3f} Error@1 {error4:.3f}'
        '  **Test** Prec_B5@1 {top1_b5.avg:.3f} Prec_B5@5 {top5_b5.avg:.3f} Error@1 {error5:.3f}'
        '  **Test** Prec_B6@1 {top1_b6.avg:.3f} Prec_B6@5 {top5_b6.avg:.3f} Error@1 {error6:.3f}'
        '  **Test** Prec_Bmain@1 {top1_main.avg:.3f} Prec_Bmain@5 {top5_main.avg:.3f} Error@1 {errormain:.3f}'
        .format(top1_b1=top1_list[0], top5_b1=top5_list[0], error1=100 - top1_list[0].avg,
                top1_b2=top1_list[1], top5_b2=top5_list[1], error2=100 - top1_list[1].avg,
                top1_b3=top1_list[2], top5_b3=top5_list[2], error3=100 - top1_list[2].avg,
                top1_b4=top1_list[3], top5_b4=top5_list[3], error4=100 - top1_list[3].avg,
                top1_b5=top1_list[4], top5_b5=top5_list[4], error5=100 - top1_list[4].avg,
                top1_b6=top1_list[5], top5_b6=top5_list[5], error6=100 - top1_list[5].avg,
                top1_main=top1_list[-1], top5_main=top5_list[-1], errormain=100 - top1_list[-1].avg,
        ), log)
        
    if summary_output:
        output_summary = np.asarray(output_summary).flatten()
        return top1.avg, top5.avg, losses.avg, output_summary
    else:

        sum=0
        if ic_only: #if only train ic branch
            for item in top1_list[:-1]:
                sum += item.avg
            top1_avg = sum/len(top1_list[:-1])
            sum=0
            for item in top5_list[:-1]:
                sum += item.avg
            top5_avg = sum/len(top5_list[:-1])
        else:
            top1_avg = top1_list[-1].avg
            top5_avg = top5_list[-1].avg

        # sum=0
        # for item in top1_list:
        #     sum += item.avg
        # top1_avg = sum/len(top1_list)
        
        # sum=0
        # for item in top5_list:
        #     sum += item.avg
        # top5_avg = sum/len(top5_list)
        #print("top1_list[0].avg:", top1_list[0].avg)
        #return top1_list[0].avg, top5_avg, losses.avg
        return top1_avg, top5_avg, losses.avg
        
        
        #return top1.avg, top5.avg, losses.avg




def get_msd_T(args, test_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to evaluation mode
    model.eval()
    end = time.time()
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    b5 = []
    b6 = []
    b7 = []

    index = 0   
    for i, (input, target) in enumerate(test_loader):
        if args.use_cuda:
            #target = target.cuda(async=True)
            input = input.cuda()
        target = target.squeeze().long().cuda(async=True)
        target_var = Variable(target, volatile=True)
        input_var = Variable(input, volatile=True)


        # compute output
        output_branch = model(input)
        
        ## dtnamic inference
        sm = torch.nn.functional.softmax
        prob_branch1 = sm(output_branch[0])
        prob_branch2 = sm(output_branch[1])
        prob_branch3 = sm(output_branch[2])
        prob_branch4 = sm(output_branch[3])
        prob_branch5 = sm(output_branch[4])
        prob_branch6 = sm(output_branch[5])
        prob_main = sm(output_branch[6])

        measure_branch1 = torch.sum(torch.mul(-prob_branch1, torch.log(prob_branch1 + 1e-5)), dim=1)
        measure_branch2 = torch.sum(torch.mul(-prob_branch2, torch.log(prob_branch2 + 1e-5)), dim=1)
        measure_branch3 = torch.sum(torch.mul(-prob_branch3, torch.log(prob_branch3 + 1e-5)), dim=1)
        measure_branch4 = torch.sum(torch.mul(-prob_branch4, torch.log(prob_branch4 + 1e-5)), dim=1)
        measure_branch5 = torch.sum(torch.mul(-prob_branch5, torch.log(prob_branch5 + 1e-5)), dim=1)
        measure_branch6 = torch.sum(torch.mul(-prob_branch6, torch.log(prob_branch6 + 1e-5)), dim=1)
        measure_main = torch.sum(torch.mul(-prob_main, torch.log(prob_main + 1e-5)), dim=1)

        for j in range(0, input.size(0)):
             b1.append((index, measure_branch1.data.cpu().numpy()[j]))
             b2.append((index, measure_branch2.data.cpu().numpy()[j]))
             b3.append((index, measure_branch3.data.cpu().numpy()[j]))
             b4.append((index, measure_branch4.data.cpu().numpy()[j]))
             b5.append((index, measure_branch5.data.cpu().numpy()[j]))
             b6.append((index, measure_branch6.data.cpu().numpy()[j]))
             b7.append((index, measure_main.data.cpu().numpy()[j]))
             index += 1


    data_len = len(b1)
    remove_idx = []
    T = []

    b1 = sorted(b1, key=lambda tuple:tuple[1])
    T.append(b1[int(data_len * 1.0 / 7.0)][1])
    remove_idx.extend([x[0] for x in b1[0:int(data_len * 1.0 / 7.0)]])

    b2 = [x for x in b2 if x[0] not in remove_idx]
    b2 = sorted(b2, key=lambda tuple:tuple[1])
    T.append(b2[int(data_len * 1.0 / 7.0)][1])
    remove_idx.extend([x[0] for x in b2[0:int(data_len * 1.0 / 7.0)]])

    b3 = [x for x in b3 if x[0] not in remove_idx]
    b3 = sorted(b3, key=lambda tuple:tuple[1])
    T.append(b3[int(data_len * 1.0 / 7.0)][1])
    remove_idx.extend([x[0] for x in b3[0:int(data_len * 1.0 / 7.0)]])

    b4 = [x for x in b4 if x[0] not in remove_idx]
    b4 = sorted(b4, key=lambda tuple:tuple[1])
    T.append(b4[int(data_len * 1.0 / 7.0)][1])
    remove_idx.extend([x[0] for x in b4[0:int(data_len * 1.0 / 7.0)]])


    b5 = [x for x in b5 if x[0] not in remove_idx]
    b5 = sorted(b5, key=lambda tuple:tuple[1])
    T.append(b5[int(data_len * 1.0 / 7.0)][1])
    remove_idx.extend([x[0] for x in b5[0:int(data_len * 1.0 / 7.0)]])


    b6 = [x for x in b6 if x[0] not in remove_idx]
    b6 = sorted(b6, key=lambda tuple:tuple[1])
    T.append(b6[int(data_len * 1.0 / 7.0)][1])

    b7 = [x for x in b7 if x[0] not in remove_idx]
    b7 = sorted(b7, key=lambda tuple:tuple[1])
    T.append(b7[int(data_len * 1.0 / 7.0)][1])

    print("T:", T)
    return T




##### specify for model attacks. To validate the effectiveness of branch model
def validate_for_attack(val_loader, model, criterion, log, T, num_branch):
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
    s1 = time.time()
    print("inde:", index_list)
    inference_cost = 0# count inference cost
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()
            target_var = Variable(target, volatile=True)
        


            
            out_list = [] # out pro
            output_branch = model(input)
            sm = torch.nn.functional.softmax
            for output in output_branch:
                prob_branch = sm(output)
                max_pro, indices = torch.max(prob_branch, dim=1)
                out_list.append((prob_branch, max_pro))
            
            num_c = 3#6 # the number of branches 
            branch_index = list(range(0, num_branch))#num_branch
            for j in range(input.size(0)):
                #tar = torch.from_numpy(np.array(target[j]).reshape((-1,1))).squeeze().long().cuda(async=True)
                tar = torch.from_numpy(target[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda(async=True)
                tar_var = Variable(torch.from_numpy(target_var.data.cpu().numpy()[j].flatten()).long().cuda())
                pre_index = random.sample(branch_index, num_c) # randomly selected index
                #pre_index = random.sample(index_list[:], num_c) # randomly selected index
                c_ = 0
                for item in sorted(pre_index):#to do: no top 5
                    if out_list[item][1][j] > 0.8 or (c_ + 1 == num_c):
                        # item = -2
                        sm_out = out_list[item][0][j]
                        out = Variable(torch.from_numpy(sm_out.data.cpu().numpy().reshape((1,-1))).float().cuda())
                        loss = criterion(out, tar_var)
                        prec1, = accuracy(out.data, tar, topk=(1,))
                        top1.update(prec1, 1)
                        losses.update(loss.item(), 1)
                        count_list[item]+=1
                        inference_cost += item
                        break
                    c_ += 1
        
        print("top1.avg!:", top1.avg, top5.avg)
        print_log('top1.avg: {:.4f}'.format(top1.avg), log)
        
        #print("top1.avg:", top1.avg, top5.avg, top_list[0].avg, top_list[1].avg, top_list[2].avg, top_list[3].avg, top_list[4].avg, top_list[5].avg, top_list[6].avg)
        print(count_list, sum(count_list))
        t_ = sum(count_list)*(num_branch-1)
        print('inference cost:', inference_cost, inference_cost/t_)
        sys.exit()
        return top1.avg, top5.avg, losses.avg
        #return res[0], top5.avg, losses.avg







def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        print("current model is best:", bestname)
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)

def save_checkpoint_def(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best_def.pth.tar')
        print("best def name:", bestname)
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


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


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()
