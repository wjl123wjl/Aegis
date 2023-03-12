import torch
import torch.nn as nn
import utils


class BasicBlockWOutput(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, 
                    add_output, num_classes, input_size, 
                    output_id, stride=1):
        super(BasicBlockWOutput, self).__init__()

        self.output_id = output_id

        layers = nn.ModuleList()
        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU())
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

        layers.append(shortcut)
        layers.append(nn.ReLU())
        self.layers = layers
        self.skip_add = nn.quantized.FloatFunctional()
        self.output = None
        if add_output:
            self.output = utils.InternalClassifier(input_size, self.expansion*channels, num_classes)


    def forward(self, x):
        fwd = self.layers[0](x) # conv layers
        fwd = self.skip_add.add(fwd, self.layers[1](x)) # shortcut
        out = self.layers[2](fwd)
        if self.output is None:
            return out, None
        return out, self.output(fwd)



class ResNet56_SDN(nn.Module):
    def __init__(self, add_output, num_classes, input_size=32):
        super(ResNet56_SDN, self).__init__()

        self.num_blocks = [9, 9, 9]
        self.num_classes = num_classes
        self.input_size = input_size
        self.add_out_nonflat = add_output
        self.add_output = [item for sublist in self.add_out_nonflat for item in sublist]

        self.in_channels = 16
        self.num_output = sum(self.add_output) + 1

        self.init_depth = 1
        self.end_depth = 1
        self.cur_output_id = 0
        self.block = BasicBlockWOutput

        init_conv = []
        if self.input_size ==  32: # cifar10
            self.cur_input_size = self.input_size
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        else: # tiny imagenet
            self.cur_input_size = int(self.input_size/2)
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU())
        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layer(self.in_channels, block_id=0, stride=1))
        self.cur_input_size = int(self.cur_input_size/2)
        self.layers.extend(self._make_layer(32, block_id=1, stride=2))
        self.cur_input_size = int(self.cur_input_size/2)
        self.layers.extend(self._make_layer(64, block_id=2, stride=2))

        end_layers = []
        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(nn.Flatten())
        end_layers.append(nn.Linear(64*self.block.expansion, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        self.initialize_weights()

    def _make_layer(self, channels, block_id, stride):
        num_blocks = self.num_blocks[block_id]
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for cur_block_id, stride in enumerate(strides):
            add_output = self.add_out_nonflat[block_id][cur_block_id]

            layers.append(
                self.block(
                    self.in_channels, channels, 
                    add_output, self.num_classes, 
                    int(self.cur_input_size), 
                    self.cur_output_id, stride
            ))
            self.in_channels = channels * self.block.expansion
            self.cur_output_id += add_output

        return layers

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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