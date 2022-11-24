import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize as standardize


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class DownSampleSkip(nn.Module):
    def __init__(self, in_planes, out_planes, stride, use_batchnorm):
        super(DownSampleSkip, self).__init__()
        self.pool = nn.AvgPool2d(1, stride=stride)
        self._pad_size = int(out_planes - in_planes)
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_planes)
            self.forward = self._forward_impl1
        else:
            self.forward = self._forward_impl2

    def _forward_impl1(self, x):
        x = self.pool(x)
        x = F.pad(x, (0, 0, 0, 0, 0, self._pad_size))
        x = self.bn(x)
        return x

    def _forward_impl2(self, x):
        x = self.pool(x)
        x = F.pad(x, (0, 0, 0, 0, 0, self._pad_size))
        return x


class Conv2dSkip(nn.Module):
    def __init__(self, in_planes, out_planes, stride, use_batchnorm):
        super(Conv2dSkip, self).__init__()
        self.conv = conv1x1(in_planes, out_planes, stride)
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_planes)
            self.forward = self._forward_impl1
        else:
            self.forward = self._forward_impl2

    def _forward_impl1(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def _forward_impl2(self, x):
        x = self.conv(x)
        return x


def _make_block_group(block, num_blocks, in_planes, out_planes, stride=1, skip_type='B', fixup=False):
    assert stride == 1 or stride == 2, 'Unsupported stride. Stride must be 1 or 2.'
    assert skip_type == 'A' or skip_type == 'B', 'Skip connection type only supports "A"(down sample + pad) and "B"(convolution).'

    downsample = None
    if stride != 1 or in_planes != out_planes:
        if skip_type == 'A':
            downsample = DownSampleSkip(in_planes, out_planes, stride, not fixup)
        else:  # skip_type == 'B'
            downsample = Conv2dSkip(in_planes, out_planes, stride, not fixup)

    blocks = []
    blocks.append(block(in_planes, out_planes, stride, downsample))
    for _ in range(1, num_blocks):
        blocks.append(block(out_planes, out_planes))

    return nn.Sequential(*blocks)


class _StandardHead(nn.Module):
    def __init__(self, in_planes, out_planes, activation_fn):
        super(_StandardHead, self).__init__()
        self.conv = conv3x3(in_planes, out_planes)
        self.bn = nn.BatchNorm2d(out_planes)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        return x


class _FixupHead(nn.Module):
    def __init__(self, in_planes, out_planes, activation_fn):
        super(_FixupHead, self).__init__()
        self.conv = conv3x3(in_planes, out_planes)
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.conv(x)
        x = self.activation_fn(x + self.bias)
        return x


class _StandardTail(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(_StandardTail, self).__init__()
        self.fc = nn.Linear(in_planes, out_planes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class _FixupTail(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(_FixupTail, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(in_planes, out_planes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x + self.bias)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FixupBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(out_planes, out_planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_factor=1, skip_type='B',
                 means=(0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0),
                 fixup=False, zero_init_residual=False):
        super(ResNet, self).__init__()
        num_channels = [16, 16 * width_factor, 32 * width_factor, 64 * width_factor]
        relu = nn.ReLU(inplace=True)
        if fixup:
            self.head = _FixupHead(3, num_channels[0], relu)
        else:
            self.head = _StandardHead(3, num_channels[0], relu)
        self.group1 = _make_block_group(block, num_blocks[0], num_channels[0], num_channels[1], 1, skip_type, fixup)
        self.group2 = _make_block_group(block, num_blocks[1], num_channels[1], num_channels[2], 2, skip_type, fixup)
        self.group3 = _make_block_group(block, num_blocks[2], num_channels[2], num_channels[3], 2, skip_type, fixup)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if fixup:
            self.tail = _FixupTail(num_channels[3], num_classes)
        else:
            self.tail = _StandardTail(num_channels[3], num_classes)

        if fixup:
            for m in self.modules():
                if isinstance(m, FixupBasicBlock):
                    nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(
                        2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * sum(num_blocks) ** (-0.5))
                    nn.init.constant_(m.conv2.weight, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

        self.means = means
        self.stds = stds

    def forward(self, x):
        x = standardize(x, self.means, self.stds)

        x = self.head(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)

        x = self.avgpool(x)
        x = self.tail(x)

        return x


def resnet110(num_classes, **kwargs):
    return ResNet(BasicBlock, [18, 18, 18], num_classes, width_factor=1, skip_type='A', **kwargs)


def resnet110f(num_classes, **kwargs):
    return ResNet(FixupBasicBlock, [18, 18, 18], num_classes, width_factor=1, skip_type='A', fixup=True, **kwargs)


def wide_resnet28x10(num_classes, **kwargs):
    return ResNet(BasicBlock, [4, 4, 4], num_classes, width_factor=10, skip_type='B', **kwargs)


def wide_resnet28fx10(num_classes, **kwargs):
    return ResNet(FixupBasicBlock, [4, 4, 4], num_classes, width_factor=10, skip_type='B', fixup=True, **kwargs)
