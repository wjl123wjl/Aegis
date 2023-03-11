import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class _bin_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mu):
        
        ctx.mu = mu 
        output = input.clone().zero_()
        output[input.ge(0)] = 1
        output[input.lt(0)] = -1

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.mu
        return grad_input, None
    
quantize = _bin_func.apply


class quan_Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(quan_Conv2d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)
        self.N_bits = 1
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place change MSB to negative


    def forward(self, input):
        # uncomment for profiling the bit-flips caused by training.
        # if self.training:
        #     try:
        #         with torch.no_grad():
        #             weight_change = (self.bin_weight - quantize(self.weight,1)).abs()
        #             self.bin_weight_change = weight_change.sum().item()
        #             self.bin_weight_change_ratio = self.bin_weight_change / self.weight.numel()
        #             # print(self.bin_weight_change, self.bin_weight_change_ratio)
        #     except:
        #         pass
        
        if self.inf_with_weight:
            return F.conv2d(input, self.weight * self.step_size, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)
        else:
            self.__reset_stepsize__()
            self.bin_weight = quantize(self.weight, self.step_size) * self.step_size      
            return F.conv2d(input, self.bin_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().mean()

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True

class quan_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)
        self.N_bits = 1
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place change MSB to negative


    def forward(self, input):
        # uncomment for profiling the bit-flips caused by training.
        # if self.training:
        #     try:
        #         with torch.no_grad():
        #             weight_change = (self.bin_weight - quantize(self.weight,1)).abs()
        #             self.bin_weight_change = weight_change.sum().item()
        #             self.bin_weight_change_ratio = self.bin_weight_change / self.weight.numel()
        #             # print(self.bin_weight_change, self.bin_weight_change_ratio)
        #     except:
        #         pass
        
        if self.inf_with_weight:
            return F.linear(input, self.weight * self.step_size, self.bias)
        else:
            self.__reset_stepsize__()
            self.bin_weight = quantize(self.weight, self.step_size) * self.step_size      
            return F.linear(input, self.bin_weight * self.step_size, self.bias)
        
    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().mean()

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True
