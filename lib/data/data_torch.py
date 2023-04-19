# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

__all__ = ['make_cifar10', 'make_imagenet64', 'DATASETS']

import os
import pathlib
from typing import Tuple

import torch
import torch.distributed
import torch.nn.functional
import torchvision.datasets
import torchvision.transforms.functional
from lib.util import FLAGS
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from . import data

ML_DATA = pathlib.Path(os.getenv('ML_DATA'))


class DatasetTorch(data.Dataset):
    def make_dataloaders(self, **kwargs) -> Tuple[DataLoader, DataLoader]:
        batch = FLAGS.batch
        if torch.distributed.is_initialized():
            assert batch % torch.distributed.get_world_size() == 0
            batch //= torch.distributed.get_world_size()
        return (DataLoader(self.make_train(), shuffle=True, drop_last=True, batch_size=batch,
                           num_workers=4, prefetch_factor=8, persistent_workers=True, **kwargs),
                DataLoader(self.make_fid(), shuffle=True, drop_last=True, batch_size=batch,
                           num_workers=4, prefetch_factor=8, persistent_workers=True, **kwargs))


def normalize(x: torch.Tensor) -> torch.Tensor:
    return 2 * x - 1


def make_cifar10() -> DatasetTorch:
    transforms = [
        torchvision.transforms.ToTensor(),
        normalize,
    ]
    transforms_fid = Compose(transforms)
    transforms_train = Compose(transforms + [torchvision.transforms.RandomHorizontalFlip()])
    fid = lambda: torchvision.datasets.CIFAR10(str(ML_DATA), train=True, transform=transforms_fid, download=True)
    train = lambda: torchvision.datasets.CIFAR10(str(ML_DATA), train=True, transform=transforms_train, download=True)
    return DatasetTorch(32, train, fid)

def make_imagenet64() -> DatasetTorch:
    transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(64),
        normalize,
    ]
    transforms_fid = Compose(transforms)
    transforms_train = Compose(transforms + [torchvision.transforms.RandomHorizontalFlip()])
    fid = lambda: torchvision.datasets.ImageFolder(str(ML_DATA / "imagenet" / "train"), transform=transforms_fid)
    train = lambda: torchvision.datasets.ImageFolder(str(ML_DATA / "imagenet" / "train"), transform=transforms_train)
    return DatasetTorch(64, train, fid)


DATASETS = {
    'cifar10': make_cifar10,
    'imagenet64': make_imagenet64,
}
