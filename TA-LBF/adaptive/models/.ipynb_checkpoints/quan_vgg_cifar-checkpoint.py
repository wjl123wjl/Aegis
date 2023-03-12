'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
import utils_sdn
from .quantization import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.init_conv = nn.Sequential()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            # quan_Linear(512, 512),
            # nn.ReLU(True),
            quan_Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            quan_Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, nolast=False):
        outputs = []
        fwd = self.init_conv(x)
        for layer in self.features:
            fwd, ic_output = layer(fwd, nolast)
            outputs.append(ic_output)
        x = fwd 

        #x = self.features(x)
        x = x.view(x.size(0), -1)
        
        if nolast:
            for i in range(len(self.classifier)-1):
                x = self.classifier[i](x)
            outputs.append(x)
            return outputs
        else:
            x = self.classifier(x)
            outputs.append(x)
            return outputs
    
class VGG1(nn.Module):
    '''
    VGG model for proflip
    '''
    def __init__(self, features):
        super(VGG1, self).__init__()
        self.init_conv = nn.Sequential()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            # quan_Linear(512, 512),
            # nn.ReLU(True),
            quan_Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            quan_Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        outputs = []
        fwd = self.init_conv(x)
        for layer in self.features:
            fwd, ic_output = layer(fwd, nolast=True)
            outputs.append(ic_output)
        x = fwd 

        #x = self.features(x)
        x = x.view(x.size(0), -1)
        for i in range(len(self.classifier)-1):
            x = self.classifier[i](x)
        outputs.append(x)
        return x

class FcBlockWOutput(nn.Module):
    def __init__(self, fc_params, output_params, flatten=False):
        super(FcBlockWOutput, self).__init__()
        input_size = fc_params[0]
        output_size = fc_params[1]
 
        add_output = output_params[0]
        num_classes = output_params[1]
        self.output_id = output_params[2]
        self.depth = 1

        fc_layers = []

        if flatten:
            fc_layers.append(nn.Flatten())

        fc_layers.append(quan_Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

        if add_output:
            self.output = quan_Linear(output_size, num_classes)

        else:
            self.output = None


    def forward(self, x, nolast=False):
        fwd = self.layers(x)
        if self.output is None:
            return fwd, None
        if nolast:
            return fwd,fwd
        return fwd, self.output(fwd)



class ConvBlockWOutput(nn.Module):
    def __init__(self, conv_params, output_params, branch_linearshape):
        super(ConvBlockWOutput, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        max_pool_size = conv_params[2]
        

        add_output = output_params[0]
        num_classes = output_params[1]
        input_size = output_params[2]
        self.output_id = output_params[3]

        self.depth = 1


        conv_layers = []
        conv_layers.append(quan_Conv2d(
            in_channels=input_channels, 
            out_channels=output_channels, 
            kernel_size=3, padding=1, stride=1))
        conv_layers.append(nn.BatchNorm2d(output_channels))
        conv_layers.append(nn.ReLU(inplace=True))

        #conv2d = quan_Conv2d(in_channels, v, kernel_size=3, padding=1)
            
        if max_pool_size > 1:# max_pool_size is 2
            conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_size))


        self.layers = nn.Sequential(*conv_layers)
        self.output = None
        if add_output:
            self.output = utils_sdn.InternalClassifier(input_size, output_channels, num_classes, branch_linearshape)

    def forward(self, x, nolast=False):
        fwd = self.layers(x)
        if self.output is None:
            return fwd, None
        if nolast:
            return fwd,self.output(fwd,nolast)
        return fwd, self.output(fwd)





def make_layers(cfg, add_output, num_classes, conv_channels, max_pool_sizes, fc_layer_sizes, input_size=32, batch_norm=False):
    layers = []
    in_channels = 3
    cur_input_size = input_size
    output_id = 0
    branch_linearshape=[16384, 4096, 4096, 1024, 1024, 1024, 256, 256]
    length_ = len(conv_channels)-len(branch_linearshape)
    branch_linearshape.extend(length_ * [-1])
    for layer_id, v in enumerate(conv_channels):
        if max_pool_sizes[layer_id] == 2:
            cur_input_size = int(cur_input_size/2)
        
        conv_params =  (in_channels, v, max_pool_sizes[layer_id])
        output_params = (add_output[layer_id], num_classes, cur_input_size, output_id)
        layers.append(ConvBlockWOutput(conv_params, output_params, branch_linearshape[layer_id])) 
        in_channels = v
        output_id += add_output[layer_id]
    

    fc_input_size = cur_input_size*cur_input_size*conv_channels[-1]
    for layer_id, width in enumerate(fc_layer_sizes[:-1]):
        fc_params = (fc_input_size, width)
        flatten = False
        if layer_id == 0:
            flatten = True

        add_output = add_output[layer_id + len(conv_channels)]
        output_params = (add_output, num_classes, output_id)
        layers.append(FcBlockWOutput(fc_params, output_params, flatten=flatten))
        fc_input_size = width
        output_id += add_output



    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11_quan(num_classes=10):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn_quan(num_classes=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13(num_classes=10):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn(num_classes=10):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16_quan(num_classes=10):#use bacthnorm, the same as vgg16_bn
    """VGG 16-layer model (configuration "D")"""
    add_output = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    conv_channels =  [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512] # the first element is input dimension
    max_pool_sizes = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    fc_layer_sizes = [512, 512]
    return VGG(make_layers(cfg['D'], add_output, num_classes, conv_channels, max_pool_sizes, fc_layer_sizes) )

def vgg16_quan1(num_classes=10):#use bacthnorm, the same as vgg16_bn
    """VGG 16-layer model (configuration "D")"""
    add_output = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    conv_channels =  [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512] # the first element is input dimension
    max_pool_sizes = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    fc_layer_sizes = [512, 512]
    return VGG1(make_layers(cfg['D'], add_output, num_classes, conv_channels, max_pool_sizes, fc_layer_sizes) )



def vgg16_bn(num_classes=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    add_output = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    conv_channels =  [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512] # the first element is input dimension
    max_pool_sizes = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    fc_layer_sizes = [512, 512]
    return VGG(make_layers(cfg['D'], add_output, num_classes, conv_channels, max_pool_sizes, fc_layer_sizes) )


def vgg19(num_classes=10):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn(num_classes=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))