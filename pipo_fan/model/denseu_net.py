import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
        
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class up_in(nn.Sequential):
    def __init__(self, num_input_features1, num_input_features2, num_output_features):
        super(up_in, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.add_module('conv1_1', nn.Conv2d(num_input_features1, num_input_features2,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('conv3_3', nn.Conv2d(num_input_features2, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, x,y):
        y = self.up(y)
        x = self.conv1_1(x)
        z = self.conv3_3(x+y)
        z = self.norm(z)
        z = self.relu(z)
        return z

class upblock(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(upblock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.add_module('conv3_3', nn.Conv2d(num_input_features, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, x,y):
        y = self.up(y)
        z = self.conv3_3(x+y)
        z = self.norm(z)
        z = self.relu(z)
        return z

class up_out(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(up_out, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.add_module('conv3_3', nn.Conv2d(num_input_features, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.dropout = nn.Dropout2d(p=0.3)
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, y):
        y = self.up(y)
        y = self.conv3_3(y)
        y = self.dropout(y)
        y = self.norm(y)
        y = self.relu(y)
        return y


class DenseUNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, num_channels = 3, num_classes = 2):

        super(DenseUNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.up1 = up_in(48*44, 48*46, 48*16)
        self.up2 = upblock(48*16, 48*8)
        self.up3 = upblock(48*8, 96)
        self.up4 = upblock(96,96)
        self.up5 = up_out(96,64)
        self.outconv = outconv(64,num_classes)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features.conv0(x)
        x0 = self.features.norm0(features)
        x0 = self.features.relu0(x0)
        x1 = self.features.pool0(x0)
        x1 = self.features.denseblock1(x1)
        x2 = self.features.transition1(x1)
        x2 = self.features.denseblock2(x2)
        x3 = self.features.transition2(x2)
        x3 = self.features.denseblock3(x3)
        x4 = self.features.transition3(x3)
        x4 = self.features.denseblock4(x4)
        
        y4 = self.up1(x3, x4)
        y3 = self.up2(x2, y4)
        y2 = self.up3(x1, y3)
        
        y1 = self.up4(x0, y2)
        y0 = self.up5(y1)
        out = self.outconv(y0)
        # out = F.softmax(out, dim=1)
        
        return out


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

