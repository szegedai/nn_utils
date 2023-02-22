import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import normalize as standardize
import numpy as np
import warnings


def get_post_activation_std(activation_fn):
    if isinstance(activation_fn, nn.ReLU):
        return np.sqrt((1 - 1 / np.pi) / 2)
    elif isinstance(activation_fn, nn.SiLU):
        return 0.5595
    elif isinstance(activation_fn, nn.Tanh):
        return 0.6278  # This is only an empirical approximation!
    return 1.0


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
                nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False)
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
    def __init__(self, in_planes, out_planes, activation_fn, alpha=1.0, beta=1.0):
        super(NFBasicBlock, self).__init__()
        self.alpha = alpha
        self.beta = beta
        downscale_factor = out_planes // in_planes
        if in_planes != out_planes:
            self.transition = nn.Sequential(
                nn.AvgPool2d(downscale_factor),
                SWSConv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=True)
            )
            self.forward = self.transition_forward
        else:
            self.forward = self.regular_forward
        self.activation_fn1 = activation_fn
        self.conv1 = SWSConv2d(in_planes, out_planes, kernel_size=3, stride=downscale_factor, padding=1, bias=True)
        self.activation_fn2 = activation_fn
        self.conv2 = SWSConv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

        # The expected standard deviation produced by the activation function assuming that the pre-activations have
        # an approximately N(0, 1) distribution. This ensures that the output of the activation function has about
        # unite variance.
        # (Should be calculated for different activation functions!)
        self.post_activation_std = get_post_activation_std(activation_fn)

        self.skipinit_gain = nn.Parameter(torch.zeros(()))

    def regular_forward(self, x):
        skip = x.clone()

        # Beta ensures that the input of the residual path has unit variance.
        # The reference implementation in the paper may be incorrect
        # because they divide by beta after the activation function
        # and not before, like the following:
        # x = self.activation_fn(x) / self.post_activation_std / self.beta
        x = self.activation_fn1(x / self.beta) / self.post_activation_std
        x = self.conv1(x)

        x = self.activation_fn2(x) / self.post_activation_std
        x = self.conv2(x)

        # "Alpha is a scalar hyperparameter which controls the rate of variance
        # growth between blocks." (Basically a magic number to make thing work. XD)
        # Skip init gain regulates how much the residual block's output should
        # be considered. It is initialised to zero to ensure initial identity flaw
        # through the network.
        x = x * self.alpha * self.skipinit_gain + skip

        return x

    def transition_forward(self, x):
        x = self.activation_fn1(x / self.beta) / self.post_activation_std
        skip = x.clone()
        x = self.conv1(x)

        x = self.activation_fn2(x) / self.post_activation_std
        x = self.conv2(x)

        x = x * self.alpha + self.transition(skip)

        return x


class ResNetV2(nn.Module):
    def __init__(self, num_classes, num_blocks, block=BasicBlock, width_factor=1, activation_fn=nn.ReLU(inplace=True),
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

        self.bn = nn.BatchNorm2d(num_channels[-1])
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(num_channels[-1], num_classes, bias=True)

        self.reset_parameters()

        self.means = means
        self.stds = stds

    def reset_parameters(self):
        negative_slope = 0
        match self.activation_fn:
            case nn.ReLU():
                act = 'relu'
            case nn.Tanh():
                act = 'tanh'
            case nn.LeakyReLU():
                act = 'leaky_relu'
                negative_slope = self.activation_fn.negative_slope
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
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = standardize(x, self.means, self.stds)
        x = self.head(x)
        x = self.groups(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        x = torch.mean(x, (2, 3))
        x = self.dropout(x)
        x = self.linear(x)
        return x


class NFResNetV2(nn.Module):
    def __init__(self, num_classes, num_blocks, block=NFBasicBlock, width_factor=1, activation_fn=nn.ReLU(inplace=True),
                 dropout=0.0, alpha=0.2,
                 means=(0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0)):
        super(NFResNetV2, self).__init__()

        self.post_activation_std = get_post_activation_std(activation_fn)

        num_channels = [16] + [2 ** (4 + i) * width_factor for i in range(len(num_blocks))]

        self.head = SWSConv2d(3, num_channels[0], kernel_size=3, stride=2, padding=1, bias=True)

        expected_var = 1.0  # This assumption requires input standardization
        groups = []
        for group_idx in range(1, len(num_blocks) + 1):
            blocks = [block(num_channels[group_idx - 1], num_channels[group_idx], activation_fn,
                            alpha=alpha, beta=np.sqrt(expected_var))]
            expected_var = 1.0
            for _ in range(num_blocks[group_idx - 1] - 1):
                blocks.append(block(num_channels[group_idx], num_channels[group_idx], activation_fn,
                                    alpha=alpha, beta=np.sqrt(expected_var)))
                expected_var += alpha ** 2
            groups.append(nn.Sequential(*blocks))
        self.groups = nn.Sequential(*groups)

        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(num_channels[-1], num_classes, bias=True)

        self.reset_parameters()

        self.means = means
        self.stds = stds

    def reset_parameters(self):
        negative_slope = 0
        match self.activation_fn:
            case nn.ReLU():
                act = 'relu'
            case nn.Tanh():
                act = 'tanh'
            case nn.LeakyReLU():
                act = 'leaky_relu'
                negative_slope = self.activation_fn.negative_slope
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
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = standardize(x, self.means, self.stds)
        x = self.head(x)
        x = self.groups(x)
        x = self.activation_fn(x) / self.post_activation_std
        x = torch.mean(x, (2, 3))
        x = self.dropout(x)
        x = self.linear(x)
        return x


def resnet28v2(num_classes, **kwargs):
    return ResNetV2(num_classes, [4, 4, 4], BasicBlock, 1, **kwargs)


def wide_resnet28v2x10(num_classes, **kwargs):
    return ResNetV2(num_classes, [4, 4, 4], BasicBlock, 10, **kwargs)


def nf_resnet28v2(num_classes, **kwargs):
    return NFResNetV2(num_classes, [4, 4, 4], NFBasicBlock, 1, **kwargs)


def nf_wide_resnet28v2x10(num_classes, **kwargs):
    return NFResNetV2(num_classes, [4, 4, 4], NFBasicBlock, 10, **kwargs)
