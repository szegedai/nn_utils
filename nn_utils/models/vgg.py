import torch.nn as nn
import torchvision
from torchvision.transforms.functional import normalize as standardize


class VGG(torchvision.models.VGG):
    def __init__(self, architecture_type, num_classes, use_batchnorm, activation_fn=nn.ReLU(inplace=True), dropout=0.0,
                 means=(0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0)):
        super(VGG, self).__init__(
            torchvision.models.vgg.make_layers(
                torchvision.models.vgg.cfgs[architecture_type],
                batch_norm=use_batchnorm
            ),
            num_classes
        )

        self.classifier[2].p = dropout
        self.classifier[5].p = dropout

        for i in range(len(self.features)):
            if isinstance(self.features[i], nn.ReLU):
                self.features[i] = activation_fn
        for i in range(len(self.classifier)):
            if isinstance(self.classifier[i], nn.ReLU):
                self.classifier[i] = activation_fn

        self.reset_parameters()

        self.means = means
        self.stds = stds

    def reset_parameters(self):
        activation_fn = self.classifier[1]
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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, negative_slope, 'fan_in', act)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.kaiming_normal_(self.classifier[-1].weight, negative_slope, 'fan_in', 'linear')
        nn.init.constant_(self.classifier[-1].bias, 0.0)

    def forward(self, x):
        x = standardize(x, self.means, self.stds)
        return super(VGG, self).forward(x)


def vgg11(num_classes=10, **kwargs):
    return VGG('A', num_classes, False, **kwargs)


def vgg11bn(num_classes=10, **kwargs):
    return VGG('A', num_classes, True, **kwargs)


def vgg13(num_classes=10, **kwargs):
    return VGG('B', num_classes, False, **kwargs)


def vgg13bn(num_classes=10, **kwargs):
    return VGG('B', num_classes, True, **kwargs)


def vgg16(num_classes=10, **kwargs):
    return VGG('D', num_classes, False, **kwargs)


def vgg16bn(num_classes=10, **kwargs):
    return VGG('D', num_classes, True, **kwargs)


def vgg19(num_classes=10, **kwargs):
    return VGG('E', num_classes, False, **kwargs)


def vgg19bn(num_classes=10, **kwargs):
    return VGG('E', num_classes, True, **kwargs)
