import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import utils_sdn

from .quantization import *

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
    def __init__(self, num_classes, inplanes, planes, linearshape, stride=1,  downsample=None):
        super(ResNetBasicblock, self).__init__()
        self.expansion = 1
        self.conv_a = quan_Conv2d(inplanes,
                                planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = quan_Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

        #if self.add_output:
        if not linearshape == -1:
          self.output = utils_sdn.InternalClassifier(32, self.expansion*planes, num_classes, linearshape) # branch_linearshape
        else:
          self.output = None

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)
        # t1=(residual + basicblock).size()[1]
        # t2=(residual + basicblock).size()[2]
        # t3=(residual + basicblock).size()[3]
        
        # print("size:", t1*t2*t3)
        return F.relu(residual + basicblock, inplace=True), self.output(residual + basicblock)


class CifarResNet(nn.Module):
    """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
    def __init__(self, block, depth, num_classes):
        """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
        super(CifarResNet, self).__init__()

        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth -
                2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print('CifarResNet : Depth : {} , Layers for each block : {}'.format(
            depth, layer_blocks))

        self.num_classes = num_classes

        self.conv_1_3x3 = quan_Conv2d(3,
                                    16,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.branch_linearshape = [16384, 16384, 16384, 16384, 16384, 4096, 4096, 4096, 4096, 4096, 1024, 1024, 1024, 1024, 1024]
        self.b_index = 0
        self.stage_1, self.group1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2, self.group2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3, self.group3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = quan_Linear(64 * block.expansion, num_classes)

        ## branch network 1
        branch_channels = [16, 64, 32]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.num_classes, self.inplanes, planes, self.branch_linearshape[self.b_index], stride, downsample))
        print("self.b_index:", self.b_index)
        self.b_index += 1
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
          print("self.b_index:", self.b_index)
          layers.append(block(self.num_classes, self.inplanes, planes, self.branch_linearshape[self.b_index]))
          self.b_index += 1
        return nn.Sequential(*layers), layers

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        output_branch = []
        for g in range(1, 4):
          layer_num = len(getattr(self, 'group{}'.format(g)))
          for i in range(layer_num):
            x, branch_out = getattr(self, 'group{}'.format(g))[i](x)
            output_branch.append(branch_out)


        # x = self.stage_1(x)
        # x = self.stage_2(x)
        # x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #return self.classifier(x)
        x = self.classifier(x)
        output_branch.append(x)
        return output_branch


def resnet20_quan(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    return model


def resnet32_quan(num_classes=100):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 32, num_classes)
    return model


def resnet44_quan(num_classes=10):
    """Constructs a ResNet-44 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 44, num_classes)
    return model


def resnet56_quan(num_classes=10):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 56, num_classes)
    return model


def resnet110_quan(num_classes=10):
    """Constructs a ResNet-110 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 110, num_classes)
    return model
