#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
"""
Single machine, multi GPU training support
"""

__all__ = ['WrapModel', 'auto_distribute', 'barrier', 'device', 'device_id', 'gather_tensor', 'is_master', 'main',
           'print', 'reduce_dict_mean', 'tqdm', 'tqdm_module', 'tqdm_with', 'trange', 'world_size', 'wrap']

import builtins
import contextlib
import functools
import os
import time
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.distributed
import tqdm as tqdm_module

from .util import FLAGS, setup


class WrapModel(torch.nn.Module):
    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.module = m

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def auto_distribute(f: Callable) -> Callable:
    """Automatically make a function distributed"""

    @functools.wraps(f)
    def wrapped(node_rank: Optional[int], world_size: Optional[int], flag_values: SimpleNamespace, *args):
        if node_rank is None:
            return f(*args)
        setup(quiet=True, flags_values=flag_values)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12359'

        rank = node_rank
        torch.distributed.init_process_group('nccl', rank=rank, world_size=torch.cuda.device_count())
        time.sleep(1)
        try:
            return f(*args)
        finally:
            torch.distributed.destroy_process_group()

    return wrapped


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def device() -> str:
    return f'cuda:{device_id()}'


def device_id() -> int:
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank() % 8


def gather_tensor(x: torch.Tensor) -> torch.Tensor:
    """Returns a concatenated tensor from all the devices."""
    if not torch.distributed.is_initialized():
        return x
    x_list = [torch.empty_like(x) for _ in range(world_size())]
    torch.distributed.all_gather(x_list, x, async_op=False)
    return torch.cat(x_list, dim=0)


def is_master() -> bool:
    return device_id() == 0


def main(main_fn: Callable) -> Callable:
    """Main function that automatically handle multiprocessing"""

    @functools.wraps(main_fn)
    def wrapped(*args):
        setup()
        if torch.cuda.device_count() == 1:
            return main_fn(None, None, FLAGS, *args)
        num_gpus = torch.cuda.device_count()
        torch.multiprocessing.spawn(main_fn, args=(num_gpus, FLAGS, *args), nprocs=num_gpus, join=True)

    return wrapped


def print(*args, **kwargs):
    if is_master():
        builtins.print(*args, **kwargs)


def reduce_dict_mean(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Mean reduce the tensor in a dict."""
    if not torch.distributed.is_initialized():
        return d
    d = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)).to(device_id()) for k, v in d.items()}
    e = {k: [torch.empty_like(v) for _ in range(world_size())] for k, v in d.items()}
    # Ideally we should be using all_reduce, but it mysteriously returns incorrect results for the loss
    [v.wait() for v in [torch.distributed.all_gather(e[k], d[k], async_op=True) for k in d]]
    return {k: sum(v) / len(v) for k, v in e.items()}


def tqdm(iterable: Iterable, **kwargs) -> Iterable:
    return tqdm_module.tqdm(iterable, **kwargs)


def tqdm_with(**kwargs) -> Iterable:
    class Noop:
        def update(self, *args, **kwargs):
            pass

    @contextlib.contextmanager
    def noop():
        yield Noop()

    return tqdm_module.tqdm(**kwargs)


def trange(*args, **kwargs):
    return tqdm_module.trange(*args, **kwargs)


def rank() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_rank()


def world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def wrap(m: torch.nn.Module):
    if not torch.distributed.is_initialized():
        return WrapModel(m.to(device()))
    return torch.nn.parallel.DistributedDataParallel(m.to(device_id()), device_ids=[device_id()])
