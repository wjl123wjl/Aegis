import torch
import torch.nn as nn
import utils


class BlockWOutput(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_channels, out_channels, params, stride=1):
        super(BlockWOutput, self).__init__()

        add_output = params[0]
        num_classes = params[1]
        input_size = params[2]
        self.output_id = params[3]

        self.depth = 2

        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False))
        conv_layers.append(nn.BatchNorm2d(in_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU())

        self.layers = nn.Sequential(*conv_layers)
        self.output = None
        if add_output:
            self.output = utils.InternalClassifier(input_size, out_channels, num_classes)

            
            

    def forward(self, x):
        fwd = self.layers(x)
        if self.output is None:
            return fwd, None
        return fwd, self.output(fwd)


class MobileNet_SDN(nn.Module):
    # (128,2) means conv channels=128, conv stride=2, by default conv stride=1
    def __init__(self,  add_output, num_classes, input_size=32, cfg=[64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]):
        super(MobileNet_SDN, self).__init__()

        self.cfg = cfg
        self.num_classes = num_classes
        self.input_size = input_size
        self.add_output = add_output

        self.num_output = sum(self.add_output) + 1
        self.in_channels = 32
        self.cur_input_size = self.input_size

        self.init_depth = 1
        self.end_depth = 1
        self.cur_output_id = 0

        init_conv = []
        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))
        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.in_channels))

        end_layers = []
        if self.input_size == 32: # cifar10 and cifar100
            end_layers.append(nn.AvgPool2d(2))
        elif self.input_size == 64: # tiny imagenet
            end_layers.append(nn.AvgPool2d(4))

        end_layers.append(nn.Flatten())
        end_layers.append(nn.Linear(1024, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

    def _make_layers(self, in_channels):
        layers = []

        for block_id, x in enumerate(self.cfg):
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if stride == 2:
                self.cur_input_size = int(self.cur_input_size/2)

            add_output = self.add_output[block_id]
            params  = (add_output, self.num_classes, self.cur_input_size, self.cur_output_id)
            layers.append(BlockWOutput(in_channels, out_channels, params, stride))
            in_channels = out_channels
            self.cur_output_id += add_output
        return layers

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
