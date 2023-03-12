import torch
import torch.nn as nn
import math
import utils


class ConvBlockWOutput(nn.Module):
    def __init__(self, conv_params, output_params):
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
        conv_layers.append(nn.Conv2d(
            in_channels=input_channels, 
            out_channels=output_channels, 
            kernel_size=3, padding=1, stride=1))
        conv_layers.append(nn.BatchNorm2d(output_channels))
        conv_layers.append(nn.ReLU())

        if max_pool_size > 1:
            conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_size))

        self.layers = nn.Sequential(*conv_layers)
        self.output = None
        if add_output:
            self.output = utils.InternalClassifier(input_size, output_channels, num_classes)

    def forward(self, x):
        fwd = self.layers(x)
        if self.output is None:
            return fwd, None
        return fwd, self.output(fwd)


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

        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

        if add_output:
            self.output = nn.Linear(output_size, num_classes)

        else:
            self.output = None


    def forward(self, x):
        fwd = self.layers(x)
        if self.output is None:
            return fwd, None
        return fwd, self.output(fwd)



class VGG16_SDN(nn.Module):
    def __init__(self, add_output, num_classes, input_size=32):
        super(VGG16_SDN, self).__init__()

        # read necessary parameters
        self.input_size = input_size
        if input_size == 32:
            self.fc_layer_sizes = [512, 512]
        else:
            self.fc_layer_sizes = [2048, 1024]
        self.num_classes = num_classes
        self.conv_channels =  [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512] # the first element is input dimension

        # read or assign defaults to the rest
        self.max_pool_sizes = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
        self.add_output = add_output
        self.num_output = sum(self.add_output) + 1
        self.init_conv = nn.Sequential() # just for compatibility with other models
        self.layers = nn.ModuleList()
        self.init_depth = 0
        self.end_depth = 2

        # add conv layers
        input_channel = 3
        cur_input_size = self.input_size
        output_id = 0
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size / 2)
            conv_params =  (input_channel, channel, self.max_pool_sizes[layer_id])
            add_output = self.add_output[layer_id]
            output_params = (add_output, self.num_classes, cur_input_size, output_id)
            self.layers.append(ConvBlockWOutput(conv_params, output_params))
            input_channel = channel
            output_id += add_output

        fc_input_size = cur_input_size*cur_input_size*self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            add_output = self.add_output[layer_id + len(self.conv_channels)]
            output_params = (add_output, self.num_classes, output_id)
            self.layers.append(FcBlockWOutput(fc_params, output_params, flatten=flatten))
            fc_input_size = width
            output_id += add_output

        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.ReLU())
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        outputs = []
        fwd = self.init_conv(x)
        for layer in self.layers:
            fwd, ic_output = layer(fwd)
            outputs.append(ic_output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)
        return outputs

    def forward_with_internal_representation(self, x):
        outputs = []

        outints = []
        fwd = self.init_conv(x)
        for layer in self.layers:
            fwd, output = layer(fwd)
            outints.append(fwd.detach())
            outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)
        return outputs, outints

    def single_early_exit(self, x, confidence_threshold):
        '''
        x of shape 1 * C * H * W
        '''
        fwd = self.init_conv(x)
        for layer_id, layer in enumerate(self.layers, 1):
            fwd, ic_output = layer(fwd)
            if ic_output is not None:
                max_conf = torch.max(torch.softmax(ic_output, dim=1))
                if max_conf >= confidence_threshold:
                    return layer_id, ic_output
        fwd = self.end_layers(fwd)
        return len(self.layers) + 1, fwd