import torch
import torch.nn as nn
import os
from .resnet import *

__all__ = ['wide_resnet28x10', 'wide_resnet28fx10', 'wide_resnet28nfx10',
           'resnet110', 'resnet110f', 'resnet110nf', 'ResNet', 'BasicBlock', 'FixupBasicBlock', 'NFBasicBlock']

def save_checkpoint(model, optimizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, path)


def load_checkpoint(model, optimizer, path, target_device=None):
    checkpoint = torch.load(path, target_device)
    model.load_state_dict(checkpoint['model_state'])
    if target_device is not None:
        model.to(target_device)
    optimizer.load_state_dict(checkpoint['optimizer_state'])


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model_state': model.state_dict()}, path)


def load_model(model, path, target_device=None):
    model.load_state_dict(torch.load(path, target_device)['model_state'])
    if target_device is not None:
        model.to(target_device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_layers(model: nn.Module, layers=None):
    for layer_name, layer in model.named_modules():
        if layers is None or layer_name in layers:
            layer.eval()
            for p in layer.parameters():
                p.requires_grad = False


def unfreeze_layers(model: nn.Module, layers=None):
    for layer_name, layer in model.named_modules():
        if layers is None or layer_name in layers:
            for p in layer.parameters():
                p.requires_grad = True
