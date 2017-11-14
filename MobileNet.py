from __future__ import print_function
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np


def get_mask(in_channels, channels):
    mask = np.zeros((in_channels, channels, 3, 3))
    for _ in range(in_channels):
        mask[_, _ % channels, :, :] = 1.
    return mask


class DiagonalwiseRefactorization(nn.Module):
    def __init__(self, in_channels, stride=1, groups=1):
        super(DiagonalwiseRefactorization, self).__init__()
        channels = in_channels / groups
        self.in_channels = in_channels
        self.groups = groups
        self.stride = stride
        self.mask = nn.Parameter(torch.Tensor(get_mask(in_channels, channels)), requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(in_channels, channels, 3, 3), requires_grad=True)
        torch.nn.init.xavier_uniform(self.weight.data)
        self.weight.data.mul_(self.mask.data)

    def forward(self, x):
        weight = torch.mul(self.weight, self.mask)
        x = torch.nn.functional.conv2d(x, weight, bias=None, stride=self.stride, padding=1, groups=self.groups)
        return x


def DepthwiseConv2d(in_channels, stride=1):
    # The original Channel-by-channel Depthwise Convolution
    # return nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False)

    # Standard Convolution
    # return nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, bias=False)

    # Diagonalwise Refactorization
    # groups = 16
    groups = max(in_channels / 32, 1)
    return DiagonalwiseRefactorization(in_channels, stride, groups)


def PointwiseConv2d(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)),
            ('bn_conv', nn.BatchNorm2d(32)),
            ('relu_conv', nn.ReLU(inplace=True)),
        ]))

        __mobilenet_channels = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        __mobilenet_strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

        in_c = 32
        for _, (out_c, stride) in enumerate(zip(__mobilenet_channels, __mobilenet_strides)):
            self.features.add_module('dw_conv_{}'.format(_), DepthwiseConv2d(in_c, stride=stride))
            self.features.add_module('dw_norm_{}'.format(_), nn.BatchNorm2d(in_c))
            self.features.add_module('dw_relu_{}'.format(_), nn.ReLU(inplace=True))
            self.features.add_module('pw_conv_{}'.format(_), PointwiseConv2d(in_c, out_c))
            self.features.add_module('pw_norm_{}'.format(_), nn.BatchNorm2d(out_c))
            self.features.add_module('pw_relu_{}'.format(_), nn.ReLU(inplace=True))
            in_c = out_c

        self.avgpool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
