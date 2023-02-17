import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import normalize as standardize
import numpy as np


class SWSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, dim=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, dim=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / (var * fan_in + self.eps).sqrt()  # ** 0.5
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, activation_fn=nn.ReLU(inplace=True)):
        super(BasicBlock, self).__init__()
        downscale_factor = out_planes // in_planes
        if in_planes != out_planes:
            self.transition = nn.Sequential(
                nn.AvgPool2d(downscale_factor),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=True)
            )
            self.forward = self.transition_forward
        else:
            self.forward = self.regular_forward
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.activation_fn = activation_fn
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=downscale_factor, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

    def regular_forward(self, x):
        skip = x.clone()

        x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)

        x += skip

        return x

    def transition_forward(self, x):
        x = self.bn1(x)
        x = self.activation_fn(x)
        skip = x.clone()
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)

        x += self.transition(skip)

        return x


class NFBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, activation_fn=nn.ReLU(inplace=True)):
        super(NFBasicBlock, self).__init__()
        downscale_factor = out_planes // in_planes
        if in_planes != out_planes:
            self.transition = nn.Sequential(
                nn.AvgPool2d(downscale_factor),
                SWSConv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=True)
            )
            self.forward = self.transition_forward
        else:
            self.forward = self.regular_forward
        self.conv1 = SWSConv2d(in_planes, out_planes, kernel_size=3, stride=downscale_factor, padding=1, bias=True)
        self.activation_fn = activation_fn
        self.conv2 = SWSConv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

    def regular_forward(self, x):
        skip = x.clone()

        x = self.activation_fn(x)
        x = self.conv1(x)

        x = self.activation_fn(x)
        x = self.conv2(x)

        x += skip

        return x

    def transition_forward(self, x):
        x = self.activation_fn(x)
        skip = x.clone()
        x = self.conv1(x)

        x = self.activation_fn(x)
        x = self.conv2(x)

        x += self.transition(skip)

        return x


class ResNetV2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_factor=1, activation_fn=nn.ReLU(inplace=True),
                 dropout=0.0, zero_init_residuals=False,
                 means=(0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0)):
        super(ResNetV2, self).__init__()

        num_channels = [16] + [2 ** (4 + i) * width_factor for i in range(len(num_blocks))]

        self.head = nn.Conv2d(3, num_channels[0], kernel_size=3, stride=2, padding=1, bias=False)

        groups = []
        for group_idx in range(1, len(num_blocks) + 1):
            blocks = [block(num_channels[group_idx - 1], num_channels[group_idx], activation_fn)]
            for _ in range(num_blocks[group_idx - 1] - 1):
                blocks.append(block(num_channels[group_idx], num_channels[group_idx], activation_fn))
            groups.append(nn.Sequential(*blocks))
        self.groups = nn.Sequential(*groups)

        tail = [
            nn.BatchNorm2d(num_channels[-1]),
            activation_fn
        ]
        if dropout > 0.0:
            tail.append(nn.Dropout(dropout))
        tail += [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], num_classes, bias=True)
        ]
        self.tail = nn.Sequential(*tail)

        negative_slope = 0
        match activation_fn:
            case nn.ReLU():
                act = 'relu'
            case nn.Tanh():
                act = 'tanh'
            case nn.LeakyReLU():
                act = 'leaky_relu'
                negative_slope = activation_fn.negative_slope
            case nn.SELU():
                act = 'selu'
            case nn.Sigmoid():
                act = 'sigmoid'
            case _:
                act = 'linear'
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, negative_slope, 'fan_in', act)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.kaiming_normal_(self.tail[-1].weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.tail[-1].bias, 0.0)

        self.means = means
        self.stds = stds

    def forward(self, x):
        x = standardize(x, self.means, self.stds)
        x = self.head(x)
        x = self.groups(x)
        x = self.tail(x)
        return x


def resnet28v2(num_classes, **kwargs):
    return ResNetV2(BasicBlock, [4, 4, 4], num_classes, 1, **kwargs)


def wide_resnet28v2x10(num_classes, **kwargs):
    return ResNetV2(BasicBlock, [4, 4, 4], num_classes, 10, **kwargs)
