import torch
import torchvision
import numpy as np
import json
import os
import random
from torch.utils.data import DataLoader
from nn_utils.activations import ActivationExtractor
from autoattack import AutoAttack
from torch.nn.functional import cross_entropy, log_softmax, one_hot


def set_determinism(determinism=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    torch.backends.cudnn.deterministic = determinism


def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)


def load_data(path, target_device=None):
    data = torch.load(path, target_device)
    try:
        data.to(target_device)
    except:
        pass
    return data


def split_dataset(dataset, split=0.1, seed=42):
    second_partition_size = int(len(dataset) * split)
    return torch.utils.data.random_split(
        dataset,
        [len(dataset) - second_partition_size, second_partition_size],
        torch.Generator().manual_seed(seed)
    )


def entropy(t, dim=()):
    return -(t * torch.log(t)).sum(dim=dim)


def quantize(t, decimals, dtype=torch.uint8):
    return (t.detach() * 10 ** decimals).to(dtype=dtype)


def top1_accuracy(one_hot=False):
    if one_hot:
        return lambda y_hat, y: torch.argmax(y_hat.detach(), dim=1).cpu().numpy() == torch.argmax(y.detach(), dim=1).cpu().numpy()
    return lambda y_hat, y: torch.argmax(y_hat.detach(), dim=1).cpu().numpy() == y.detach().cpu().numpy()


def evaluate(model, ds_loader, loss_fn, acc_fn=top1_accuracy(False), attack=None, batch_num_limit=float('inf')):
    model_device = next(model.parameters()).device
    loss_sum = 0
    acc_sum = 0
    num_samples = 0
    for i, (x, y) in enumerate(ds_loader):
        if i >= batch_num_limit:
            break
        current_batch_size = x.shape[0]
        num_samples += current_batch_size
        x, y = x.to(model_device), y.to(model_device)

        if attack is not None:
            x = attack.perturb(x, y)
        with torch.no_grad():
            output = model(x)

            loss = loss_fn(output, y)

        loss_sum += loss.item()
        batch_acc = acc_fn(output, y)
        acc_sum += np.sum(batch_acc)
    return loss_sum / num_samples, acc_sum / num_samples


def evaluate_acc(model, ds_loader, acc_fn=top1_accuracy(False), attack=None, batch_num_limit=float('inf')):
    autoattack = isinstance(attack, AutoAttack)
    model_device = next(model.parameters()).device
    acc_sum = 0
    num_samples = 0
    for i, (x, y) in enumerate(ds_loader):
        if i >= batch_num_limit:
            break
        current_batch_size = x.shape[0]
        num_samples += current_batch_size
        x, y = x.to(model_device), y.to(model_device)

        if attack is not None:
            if autoattack:
                x = attack.run_standard_evaluation(x, y, bs=current_batch_size)
            else:
                x = attack.perturb(x, y)
        with torch.no_grad():
            output = model(x)

        batch_acc = acc_fn(output, y)
        acc_sum += np.sum(batch_acc)
    return acc_sum / num_samples


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction_fn=torch.mean, one_hot=False, **kwargs):
        super().__init__()
        if reduction_fn is not None:
            self.reduce = reduction_fn
        else:
            self.reduce = lambda x: x
        if one_hot:
            self.forward = self._forward_impl_1
        else:
            self.forward = self._forward_impl_2
        self._cross_entropy_kwargs = kwargs

    def _forward_impl_1(self, y_hat, y):
        return self.reduce(-torch.sum(y * log_softmax(y_hat), dim=1))

    def _forward_impl_2(self, y_hat, y):
        return cross_entropy(y_hat, y, **self._cross_entropy_kwargs)


class OneHotEncoder:
    def __init__(self, num_classes, dtype=torch.int64):
        self.num_classes = num_classes
        self.dtype = dtype

    def __call__(self, x):
        return one_hot(torch.tensor(x, dtype=torch.int64), self.num_classes).to(dtype=self.dtype)


class UniformRandomDataset(torch.utils.data.Dataset):
    def __init__(self, length, data_shape, bounds=(0.0, 1.0), add_labels=False):
        super(UniformRandomDataset, self).__init__()
        self.length = length
        self.data_shape = data_shape
        self.bounds = bounds
        self.add_labels = add_labels

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.add_labels:
            return torch.empty(self.data_shape).uniform_(*self.bounds), -1
        return torch.empty(self.data_shape).uniform_(*self.bounds)


class NormalRandomDataset(torch.utils.data.Dataset):
    def __init__(self, length, data_shape, mean=0.0, std=1.0, bounds=(0.0, 1.0), add_labels=False):
        super(NormalRandomDataset, self).__init__()
        self.length = length
        self.data_shape = data_shape
        self.mean = mean
        self.std = std
        self.bounds = bounds
        self.add_labels = add_labels

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.add_labels:
            return torch.empty(self.data_shape).normal_(mean=self.mean, std=self.std).clamp_(*self.bounds), -1
        return torch.empty(self.data_shape).normal_(mean=self.mean, std=self.std).clamp_(*self.bounds)


class ModelActivationDataset(torch.utils.data.IterableDataset):
    def __init__(self, model, dataloader):
        super().__init__()
        self._model = model
        self._model_device = next(model.parameters()).device
        self._extractor = ActivationExtractor(self._model, self._model.get_relevant_layers())
        self._dl = dataloader
        self._dl_iter = None
        self._len = len(dataloader)

    def __iter__(self):
        self._dl_iter = iter(self._dl)
        return self

    def __next__(self):
        next_batch = next(self._dl_iter)
        if not next_batch:
            raise StopIteration
        self._model(next_batch[0].to(self._model_device))
        return self._extractor.activations

    def __len__(self):
        return self._len


class TripletCIFAR10(torch.utils.data.IterableDataset):
    def __init__(self, root='./datasets', split='train', transforms=None, get_labels=False):
        if transforms is None:
            transforms = []
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), *transforms
        ])
        _cifar10 = None
        if split == 'train':
            _cifar10 = torchvision.datasets.CIFAR10(root, train=True, transform=transforms, download=True)
        elif split == 'test':
            _cifar10 = torchvision.datasets.CIFAR10(root, train=False, transform=transforms, download=True)
        else:
            raise Exception('Split must be "train" or "test"')
        self._len = len(_cifar10)
        self._data_by_labels = {}
        for x, y in _cifar10:
            if not (y in self._data_by_labels):
                self._data_by_labels[y] = []
            self._data_by_labels[y].append(x)
        self._labels = set(self._data_by_labels.keys())
        self._get_labels = get_labels

    def __iter__(self):
        self._iteration_counter = 0
        return self

    def __next__(self):
        if self._iteration_counter >= self._len:
            raise StopIteration
        self._iteration_counter += 1
        a_label = random.choice(tuple(self._labels))
        n_label = random.choice(tuple(self._labels - {a_label}))
        if self._get_labels:
            return random.choice(self._data_by_labels[a_label]), \
                   random.choice(self._data_by_labels[a_label]), \
                   random.choice(self._data_by_labels[n_label]), \
                   a_label, a_label, n_label
        return random.choice(self._data_by_labels[a_label]), \
               random.choice(self._data_by_labels[a_label]), \
               random.choice(self._data_by_labels[n_label])

    def __len__(self):
        return self._len


class LogManager:
    def __init__(self, to_file=None, from_file=None):
        self.file = to_file
        self.records = []
        if from_file is not None:
            with open(from_file, 'r') as fp:
                for record in fp.readlines():
                    self.records.append(json.loads(record))
            if self.file is None:
                self.file = from_file

    def write_record(self, record):
        self.records.append(record)
        if self.file is not None:
            os.makedirs(os.path.dirname(self.file), exist_ok=True)
            with open(self.file, 'a') as fp:
                fp.write(json.dumps(self.records[-1]) + '\n')

    def write_all(self):
        assert self.file is not None, 'A file must be provided in order to write!'
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file, 'w') as fp:
            for r in self.records:
                fp.write(json.dumps(r) + '\n')

    def get_transposed(self):
        assert len(self.records) > 0, 'There must be at least one record in order to transpose!'

        transposed_metrics = {}
        for key in self.records[0].keys():
            transposed_metrics[key] = []

        for rec in self.records:
            assert rec.keys() == transposed_metrics.keys(), 'All record keys must be the same in order to transpose!'

        for rec in self.records:
            for k, v in rec.items():
                transposed_metrics[k].append(v)
        return transposed_metrics

    def has_col(self, col_name):
        ret = True
        for rec in self.records:
            ret &= col_name in rec
        return ret


def l1_norm(x):
    return x.abs()


def l2_norm(x):
    return x.pow(2.0)


def l1m_norm(x):
    return (x + x.abs()) / 2


def l2m_norm(x):
    return ((x + x.abs()) / 2).pow(2.0)


class WeightRegularization:
    def __init__(self, model=None, norm=lambda x: x, lam=0.0, lambda_map=None):
        self.model = model
        self.lam = lam
        self.lambda_map = lambda_map
        self.norm = norm

    def __call__(self):
        if self.lambda_map is not None:
            s = 0.0
            for layer_id, lam in self.lambda_map.items():
                s += sum(self.norm(p).sum() for p in self.model.get_submodule(layer_id).parameters()) * lam
            return s
        return sum(self.norm(p).sum() for p in self.model.parameters()) * self.lam


class RollingStatistics:
    def __init__(self, shape=(1,), axis=-1):
        self.axis = axis
        shape = list(shape)
        del shape[axis]
        self.shape = shape

        self._count = 0
        self._sum = np.zeros(self.shape)
        self._sq_sum = np.zeros(self.shape)

    def update(self, new_vals):
        new_vals = np.array(new_vals)
        self._sum += np.sum(new_vals, axis=self.axis)
        self._sq_sum += np.sum(np.square(new_vals), axis=self.axis)
        self._count += new_vals.shape[self.axis]

    def reset(self):
        self._count = 0
        self._sum = np.zeros(self.shape)
        self._sq_sum = np.zeros(self.shape)

    @property
    def mean(self):
        return self._sum / self._count

    @property
    def std(self):
        return np.sqrt(self._sq_sum / self._count - np.square(self._sum / self._count))

    @property
    def var(self):
        return self._sq_sum / self._count - np.square(self._sum / self._count)

    @property
    def count(self):
        return self._count


def create_data_loaders(datasets, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    ds_loaders = []
    for ds in datasets:
        ds_loaders.append(DataLoader(ds,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory))
    return ds_loaders


def load_mnist(train_transforms=None, test_transforms=None, target_transform=None):
    if train_transforms is None:
        train_transforms = []
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), *train_transforms
    ])
    if test_transforms is None:
        test_transforms = []
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), *test_transforms
    ])
    train_dataset = torchvision.datasets.MNIST('./datasets',
                                               train=True,
                                               transform=train_transforms,
                                               target_transform=target_transform,
                                               download=True)
    test_dataset = torchvision.datasets.MNIST('./datasets', train=False, transform=test_transforms, download=True)

    return train_dataset, test_dataset


def load_fashion_mnist(train_transforms=None, test_transforms=None, target_transform=None):
    if train_transforms is None:
        train_transforms = []
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), *train_transforms
    ])
    if test_transforms is None:
        test_transforms = []
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), *test_transforms
    ])
    train_dataset = torchvision.datasets.FashionMNIST('./datasets',
                                                      train=True,
                                                      transform=train_transforms,
                                                      target_transform=target_transform,
                                                      download=True)
    test_dataset = torchvision.datasets.FashionMNIST('./datasets',
                                                     train=False,
                                                     transform=test_transforms,
                                                     target_transform=target_transform,
                                                     download=True)

    return train_dataset, test_dataset


def load_cifar10(train_transforms=None, test_transforms=None, target_transform=None):
    if train_transforms is None:
        train_transforms = []
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), *train_transforms
    ])
    if test_transforms is None:
        test_transforms = []
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), *test_transforms
    ])
    train_dataset = torchvision.datasets.CIFAR10('./datasets',
                                                 train=True,
                                                 transform=train_transforms,
                                                 target_transform=target_transform,
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10('./datasets',
                                                train=False,
                                                transform=test_transforms,
                                                target_transform=target_transform,
                                                download=True)

    return train_dataset, test_dataset


def load_svhn(train_transforms=None, test_transforms=None, target_transform=None):
    if train_transforms is None:
        train_transforms = []
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), *train_transforms
    ])
    if test_transforms is None:
        test_transforms = []
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), *test_transforms
    ])
    train_dataset = torchvision.datasets.SVHN('./datasets',
                                              split='train',
                                              transform=train_transforms,
                                              download=True)
    test_dataset = torchvision.datasets.SVHN('./datasets',
                                             split='test',
                                             transform=test_transforms,
                                             target_transform=target_transform,
                                             download=True)

    return train_dataset, test_dataset


def load_imagenet(train_transforms=None, test_transforms=None, target_transform=None):
    if train_transforms is None:
        train_transforms = []
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), *train_transforms
    ])
    if test_transforms is None:
        test_transforms = []
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), *test_transforms
    ])
    train_dataset = torchvision.datasets.ImageFolder('./datasets/imagenet/training_data',
                                                     transform=train_transforms,
                                                     target_transform=target_transform)
    test_dataset = torchvision.datasets.ImageFolder('./datasets/imagenet/validation_data',
                                                    transform=test_transforms,
                                                    target_transform=target_transform)

    return train_dataset, test_dataset


def setup_mnist(batch_size=50, *args, **kwargs):
    return create_data_loaders(load_mnist(*args, **kwargs), batch_size, True, 4)


def setup_fashion_mnist(batch_size=50, *args, **kwargs):
    return create_data_loaders(load_fashion_mnist(*args, **kwargs), batch_size, True, 4)


def setup_cifar10(batch_size=256, *args, **kwargs):
    return create_data_loaders(load_cifar10(*args, **kwargs), batch_size, True, 4)


def setup_svhn(batch_size=256, *args, **kwargs):
    return create_data_loaders(load_svhn(*args, **kwargs), batch_size, True, 4)


def setup_imagenet(batch_size=256, *args, **kwargs):
    return create_data_loaders(load_imagenet(*args, **kwargs), batch_size, True, 4)
