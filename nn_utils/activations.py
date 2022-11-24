import torch
import torch.nn as nn
import models.resnet

'''def split_dict(d: dict):
    return d.keys(), torch.stack(list(d.values()))


def create_dict(k, v: torch.Tensor):
    return dict(zip(k, torch.unbind(v)))'''


def get_relevant_modules(model: nn.Module):
    module_ids = []
    prev_id = None
    for m_id, m in model.named_modules():
        if isinstance(m, nn.ReLU) and m.inplace and prev_id is not None:
            module_ids.append(prev_id)
        if isinstance(m, models.resnet.BasicBlock) or isinstance(m, models.resnet.FixupBasicBlock):
            module_ids.append(m_id)
        prev_id = m_id
    return module_ids


class ActivationExtractor(nn.Module):
    def __init__(self, model: nn.Module, module_ids=None):
        super().__init__()
        self.model = model
        if module_ids is None:
            module_ids = get_relevant_modules(model)
        self.module_ids = module_ids
        self.activations = {m_id: torch.empty(0) for m_id in self.module_ids}

        self.hooks = []

        for m_id in self.module_ids:
            layer = dict([*self.model.named_modules()])[m_id]
            self.hooks.append(layer.register_forward_hook(self.get_activation_hook(m_id)))

    def get_activation_hook(self, m_id: str):
        def fn(_, __, output):
            self.activations[m_id] = output

        return fn

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def forward(self, x):
        self.model(x)
        return self.activations


def get_activation_sums(model, layers, dataloader, attack=None, use_sign=False):
    if use_sign:
        pre_sum_fn = torch.sign
    else:
        pre_sum_fn = lambda x: x
    with torch.no_grad():
        model_device = next(model.parameters()).device
        extractor = ActivationExtractor(model, layers)
        it = iter(dataloader)

        num_samples = 0
        activation_counts = {}
        batch = next(it)
        batch[0], batch[1] = batch[0].to(model_device), batch[1].to(model_device)
        num_samples += len(batch[0])
        if attack is not None:
            activations = extractor(attack(*batch))
        else:
            activations = extractor(batch[0])
        for k, v in activations.items():
            activation_counts[k] = torch.sum(torch.sign(v), dim=0)

        while (batch := next(it, None)) is not None:
            batch[0], batch[1] = batch[0].to(model_device), batch[1].to(model_device)
            num_samples += len(batch[0])
            if attack is not None:
                activations = extractor(attack(*batch))
            else:
                activations = extractor(batch[0])
            for k, v in activations.items():
                activation_counts[k] += torch.sum(pre_sum_fn(v), dim=0)
        extractor.remove_hooks()
    return activation_counts, num_samples


def get_max_activations(model: nn.Module, layers, dataloader, attack=None):
    with torch.no_grad():
        model_device = next(model.parameters()).device
        extractor = ActivationExtractor(model, layers)
        it = iter(dataloader)

        max_activations = {}
        batch = next(it)
        batch[0], batch[1] = batch[0].to(model_device), batch[1].to(model_device)
        if attack is not None:
            activations = extractor(attack(*batch))
        else:
            activations = extractor(batch[0])
        for k, v in activations.items():
            max_activations[k] = torch.max(v, dim=0)[0]

        while (batch := next(it, None)) is not None:
            batch[0], batch[1] = batch[0].to(model_device), batch[1].to(model_device)
            if attack is not None:
                activations = extractor(attack(*batch))
            else:
                activations = extractor(batch[0])
            for k, v in activations.items():
                max_activations[k] = torch.maximum(torch.max(v, dim=0)[0], max_activations[k])
        extractor.remove_hooks()
    return max_activations


def get_avg_activations(model, layers, dataloader, attack=None):
    activation_sums, num_samples = get_activation_sums(model, layers, dataloader, attack, use_sign=False)
    with torch.no_grad():
        for k, v in activation_sums.items():
            activation_sums[k] /= num_samples
    return activation_sums


def get_activity(model, layers, dataloader, attack=None):
    activation_counts, num_samples = get_activation_sums(model, layers, dataloader, attack, use_sign=True)
    with torch.no_grad():
        for k, v in activation_counts.items():
            activation_counts[k] /= num_samples  # turn _count into ratios
    return activation_counts


def get_inactivity(model, layers, dataloader, attack=None):
    activation_counts, num_samples = get_activation_sums(model, layers, dataloader, attack, use_sign=True)
    with torch.no_grad():
        for k, v in activation_counts.items():
            activation_counts[k] /= num_samples  # turn _count into ratios
            activation_counts[k] = 1 - activation_counts[k]
    return activation_counts


def get_inactivity_ratios(activations):  # works with activation dicts and activity dicts too
    ratios = {}
    for k, v in activations.items():
        ratios[k] = ((torch.numel(v) - torch.count_nonzero(v)) / torch.numel(v)).item()
    return ratios


def get_inactivity_ratio(activations):  # works with activation dicts and activity dicts too
    num_activations = 0.0
    num_inactives = 0.0
    for k, v in activations.items():
        num_activations += torch.numel(v)
        num_inactives += (torch.numel(v) - torch.count_nonzero(v)).item()
    return num_inactives / num_activations


def flatten_tdict(td):  # td = tensor dict
    device = next(iter(td.values())).device
    ret = torch.tensor([], device=device)
    for v in td.values():
        ret = torch.cat((ret, torch.flatten(v)), dim=0)
    return ret


def split_tdict(td):
    ret = []
    for i in range(next(iter(td.values())).shape[0]):
        tmp = {}
        for k, v in td.items():
            tmp[k] = v[i]
        ret.append(tmp)
    return ret


def filter_tdict(d, filters):
    ret = {}
    for k, v in d.items():
        for f in filters:
            if f in k:
                ret[k] = v
                break
    return ret


def transform_tdict(fn, d, *args, **kwargs):
    ret = {}
    for k, v in d.items():
        ret[k] = fn(v, *args, **kwargs)
    return ret


def remap_tdict(d, remap):
    return {new_key: d[old_key] for old_key, new_key in remap.items()}
