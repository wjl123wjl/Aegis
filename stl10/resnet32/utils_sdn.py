


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import os

import matplotlib
from torch.utils.data import dataset
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

import torch.nn.utils.prune as prune

import networks
from networks.CNNs.VGG import VGG16
from networks.CNNs.ResNet import ResNet56
from networks.CNNs.MobileNet import MobileNet

from networks.SDNs.VGG_SDN import VGG16_SDN
from networks.SDNs.ResNet_SDN import ResNet56_SDN
from networks.SDNs.MobileNet_SDN import MobileNet_SDN
from models.quantization import quan_Conv2d, quan_Linear


#########################################################
###################   Training   ########################
###################   Function  #########################
#########################################################
def get_cnn_model(nettype, num_classes, input_size):

    if 'resnet' in nettype:
        model = ResNet56(num_classes, input_size)
    elif 'vgg' in nettype:
        model = VGG16(num_classes, input_size)
    elif 'mobilenet' in nettype:
        model = MobileNet(num_classes, input_size)
    return model

def get_sdn_model(nettype, add_output, num_classes, input_size):

    if 'resnet' in nettype:
        return ResNet56_SDN(add_output, num_classes, input_size)
    elif 'vgg' in nettype:
        return VGG16_SDN(add_output, num_classes, input_size)
    elif 'mobilenet' in nettype:
        return MobileNet_SDN(add_output, num_classes, input_size)


def load_cnn(sdn):
    if isinstance(sdn, VGG16_SDN):
        return VGG16
    elif isinstance(sdn, ResNet56_SDN):
        return ResNet56
    elif isinstance(sdn, MobileNet_SDN):
        return MobileNet

def load_sdn(cnn):
    if isinstance(cnn, VGG16):
        return VGG16_SDN
    elif isinstance(cnn, ResNet56):
        return ResNet56_SDN
    elif isinstance(cnn, MobileNet):
        return MobileNet_SDN

def get_add_output(network):
    if 'vgg16' in network:
        return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    if 'resnet56' in network:
        return [ \
        [1, 1, 1, 1, 1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1, 1, 1, 1, 1]]
    if 'mobilenet' in network:
        return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

def cnn_to_sdn(cnn_model, add_output, device):
    # todo
    print ('[cnn_to_sdn] convert a CNN to an SDN...')
    sdn_model = (load_sdn(cnn_model))(add_output, cnn_model.num_classes, cnn_model.input_size)
    sdn_model.init_conv = cnn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, cnn_layer in enumerate(cnn_model.layers):
        sdn_layer = sdn_model.layers[layer_id]
        sdn_layer.layers = cnn_layer.layers
        layers.append(sdn_layer)

    sdn_model.layers = layers
    sdn_model.end_layers = cnn_model.end_layers
    sdn_model.to(device)
    return sdn_model

def sdn_to_cnn(sdn_model, device):
    # todo
    print ('[sdn_to_cnn] convert an SDN to a CNN...')
    
    cnn_model = load_cnn(sdn_model)(sdn_model.num_classes, sdn_model.input_size)
    cnn_model.init_conv = sdn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, sdn_layer in enumerate(sdn_model.layers):
        cnn_layer = cnn_model.layers[layer_id]
        cnn_layer.layers = sdn_layer.layers
        layers.append(cnn_layer)

    cnn_model.layers = layers
    cnn_model.end_layers = sdn_model.end_layers
    cnn_model.to(device)
    return cnn_model



def freeze_except_outputs(model):

    for param in model.init_conv.parameters():
        param.requires_grad = False

    for layer in model.layers:
        for param in layer.layers.parameters():
            param.requires_grad = False

    for param in model.end_layers.parameters():
        param.requires_grad = False

def freeze_outputs(model):

    for layer in model.layers:
        for param in layer.output.parameters():
            param.requires_grad = False


# the internal classifier for all SDNs
class InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, branch_linearshape, alpha=0.5):
        super(InternalClassifier, self).__init__()
        #red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.branch_linearshape = branch_linearshape
        self.output_channels = output_channels
        self.flat = nn.Flatten()
        if red_kernel_size == -1:
            self.linear = quan_Linear(output_channels*input_size*input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)
            # self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            # self.alpha_mult = nn.quantized.FloatFunctional()
            
            self.linear = quan_Linear(output_channels*red_input_size*red_input_size, num_classes)
            self.forward = self.forward_w_pooling
            self.red_input_size = red_input_size
            if self.branch_linearshape != -1:
                self.quan_layer_branch = quan_Linear(self.branch_linearshape, num_classes)
                self.branch_channels = [16, 64, 32]
                self.branch_layer=nn.Sequential(
                    nn.MaxPool2d(2, stride=2),
                    quan_Conv2d(self.output_channels, 
                    self.branch_channels[1], kernel_size=5,
                    stride=1, padding=2),
                    nn.BatchNorm2d(self.branch_channels[1]),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                )
        
        

    def forward_w_pooling(self, x):
        # avgp = self.alpha_mult.mul(self.alpha, self.max_pool(x))
        # maxp = self.alpha_mult.mul(1 - self.alpha, self.avg_pool(x))
        # mixed = avgp + maxp
        # return self.linear(mixed.view(mixed.size(0), -1))\
        if self.branch_linearshape != -1:
        #self.branch_layer.cuda()
            out_ = self.branch_layer(x)
            #print("out_size:", out_.size())
            out_ = self.quan_layer_branch(out_)
        else:
            maxp = self.avg_pool(x)
            return self.linear(self.flat(maxp))
        return out_
        
    
    # def forward_w_pooling_2(self, x):
    #     # avgp = self.alpha_mult.mul(self.alpha, self.max_pool(x))
    #     # maxp = self.alpha_mult.mul(1 - self.alpha, self.avg_pool(x))
    #     # mixed = avgp + maxp
    #     # return self.linear(mixed.view(mixed.size(0), -1))\
        
    #     maxp = self.avg_pool(x)
    #     print("x.size:", x.size(), maxp.size())
    #     return self.linear(self.flat(maxp))



    def forward_wo_pooling(self, x):
        return self.linear(self.flat(x))

# the formula for feature reduction in the internal classifiers
def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1


def get_lr(optimizers):

    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']



#########################################################
####################     AUX    #########################
###################   Function  #########################
#########################################################

def fast_load_model(net, path, dev='cpu'):
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    net.to(dev)
    return net

def test_threshold(output, threshold, start_from_include=0):
    '''
    no None in output list. 
    '''
    output = output[start_from_include:]
    output_num = len(output)
    output = torch.stack(output, dim=1)
    batch_max_conf, batch_pred = torch.max(torch.softmax(output, dim=2), dim=2)
    # above are two matrix of shape batch_size * output_num
    batch_out = torch.where(batch_max_conf > threshold, 1, -1)
    batch_out[:, -1] = 0
    batch_out_idx = torch.argmax(batch_out, dim=1)
    output_bool = torch.eye(output_num).to(batch_out_idx.device).index_select(0, batch_out_idx).bool()
    return start_from_include + batch_out_idx, batch_pred[output_bool]

# def select_threshold(threshold_list, test_loader, model, device='cpu'):
#     '''
#     output the prediction results for each threshold in threshold_list.
#     '''
#     result_dict = {}
#     for threshold in threshold_list:
#         targets = []
#         preds = []
#         for x, y in test_loader:
#             x, y = x.to(device), y.to(device)
#             targets.append(y)
#             output = model(x)
#             batch_out_ids, batch_pred = test_threshold(output, threshold)
#             preds.append(batch_pred)
#         targets = torch.cat(targets)
#         preds = torch.cat(preds)
#         acc = targets.eq(preds.view_as(targets)).sum().item() / len(targets)
#         result_dict[threshold] = acc
#         print("{:.2f},\t{:.2f}".format(threshold, 100 * acc))
#     return result_dict

# def early_exit(output, threshold):
#     '''
#     no None in output, output list 
#     '''
#     output = torch.cat(output, dim=0)
#     max_conf, pred = torch.max(torch.softmax(output, dim=1), dim=1)
#     qualify_out_id = torch.where(max_conf > threshold, 1, -1)
#     qualify_out_id[-1] = 0
#     out_id = torch.argmax(qualify_out_id)
#     return out_id.item(), pred[out_id].item()



def count_conv2d(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops * m.groups

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])

def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])

def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])

def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_linear(m, x, y):

    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def profile_sdn(model, input_size, device):
    inp = (1, 3, input_size, input_size)
    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        else:
            #print("Not implemented for ", m)
            pass

    model.apply(add_hooks)

    x = torch.zeros(inp)
    x = x.to(device)
    model(x)

    output_total_ops = {}
    output_total_params = {}

    total_ops = 0
    total_params = 0

    cur_output_id = 0
    cur_output_layer_id = -10
    wait_for = -10

    vgg = False
    for layer_id, m in enumerate(model.modules()):
        if isinstance(m, InternalClassifier):
            cur_output_layer_id = layer_id
        elif isinstance(m, networks.SDNs.VGG_SDN.FcBlockWOutput) and m.output is not None:
            vgg = True
            cur_output_layer_id = layer_id

        if layer_id == cur_output_layer_id + 1:
            if vgg:
                wait_for = 4
            elif isinstance(m, nn.Linear):
                wait_for = 1
            else:
                wait_for = 2
                if hasattr(m, 'avg_pool'):
                    wait_for = 3

        if len(list(m.children())) > 0: continue

        total_ops += m.total_ops
        total_params += m.total_params

        if layer_id == cur_output_layer_id + wait_for:
            output_total_ops[cur_output_id] = total_ops.numpy()[0]/1e9
            output_total_params[cur_output_id] = total_params.numpy()[0]/1e6
            cur_output_id += 1

    output_total_ops[cur_output_id] = total_ops.numpy()[0]/1e9
    output_total_params[cur_output_id] = total_params.numpy()[0]/1e6

    return output_total_ops, output_total_params