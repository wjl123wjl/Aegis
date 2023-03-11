# from .vavanilla_resnet_cifar import vanilla_resnet20
from .vanilla_models.vanilla_resnet_imagenet import resnet18
from .quan_resnet_imagenet import resnet18_quan, resnet34_quan
from .quan_alexnet_imagenet import alexnet_quan


############## ResNet for CIFAR-10 ###########
from .vanilla_models.vanilla_resnet_cifar import vanilla_resnet20
# from .quan_resnet_cifar import resnet20_quan
# from .bin_resnet_cifar import resnet20_bin
from .quan_resnet_cifar import resnet32_quan, resnet32_quan1
from .quan_resnet_tiny import resnet32_quan_ti, resnet32_quan_ti1
from .quan_resnet_stl10 import resnet32_quan_stl10, resnet32_quan_stl101


############## VGG for CIFAR #############

from .vanilla_models.vanilla_vgg_cifar import vgg11_bn, vgg11
from .quan_vgg_cifar import vgg11_bn_quan, vgg11_quan, vgg16_quan, vgg16_quan1
# from .bin_vgg_cifar import vgg11_bn_bin
from .quan_vgg_tiny import vgg16_quan_ti1, vgg16_quan_ti
from .quan_vgg_stl10 import vgg16_quan_stl101, vgg16_quan_stl10
############# Mobilenet for ImageNet #######
from .vanilla_models.vanilla_mobilenet_imagenet import mobilenet_v2

from .quan_mobilenet_imagenet import mobilenet_v2_quan