import torch.nn as nn
import torchvision
from torchvision.transforms.functional import normalize as standardize


class _VGG(torchvision.models.VGG):
    def __init__(self, *args, means=(0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0), **kwargs):
        super(_VGG, self).__init__(*args, **kwargs)
        self.means = means
        self.stds = stds

    def forward(self, x):
        x = standardize(x, self.means, self.stds)
        return super(_VGG, self).forward(x)


def _vgg(a_type, num_classes, use_dropout, use_batchnorm, means=(0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0)):
    model = _VGG(
        torchvision.models.vgg.make_layers(torchvision.models.vgg.cfgs[a_type], batch_norm=use_batchnorm),
        num_classes,
        means=means,
        stds=stds
    )
    if not use_dropout:
        #model.classifier[2] = nn.Identity()
        #model.classifier[5] = nn.Identity()
        model.classifier = nn.Sequential(*[model.classifier[i] for i in (0, 1, 3, 4, 6)])
    return model


def vgg16(num_classes=10, use_dropout=False, **kwargs):
    return _vgg('D', num_classes, use_dropout, False, **kwargs)


def vgg16bn(num_classes=10, use_dropout=False, **kwargs):
    return _vgg('D', num_classes, use_dropout, True, **kwargs)


def vgg19(num_classes=10, use_dropout=False, **kwargs):
    return _vgg('E', num_classes, use_dropout, False, **kwargs)


def vgg19bn(num_classes=10, use_dropout=False, **kwargs):
    return _vgg('E', num_classes, use_dropout, True, **kwargs)
