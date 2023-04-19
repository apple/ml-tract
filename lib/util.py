#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

__all__ = ['FLAGS', 'artifact_dir', 'command_line', 'convert_256_to_11', 'cpu_count',
           'downcast', 'ilog2', 'int_str', 'local_kwargs', 'power_of_2', 'repeater', 'report_module_weights', 'setup',
           'time_format', 'to_numpy', 'to_png', 'tqdm', 'tqdm_with', 'trange']

import contextlib
import dataclasses
import inspect
import io
import multiprocessing
import os
import pathlib
import random
import re
import sys
from types import SimpleNamespace
from typing import Callable, Iterable, Optional, Union

import absl.flags
import numpy as np
import torch
import torch.backends.cudnn
import tqdm as tqdm_module
from absl import flags
from PIL import Image

FLAGS = SimpleNamespace()

flags.DEFINE_string('logdir', 'e', help='Directory whwer to save logs.')

SYSTEM_FLAGS = {'?', 'alsologtostderr', 'help', 'helpfull', 'helpshort', 'helpxml', 'log_dir', 'logger_levels',
                'logtostderr', 'only_check_args', 'pdb', 'pdb_post_mortem', 'profile_file', 'run_with_pdb',
                'run_with_profiling', 'showprefixforinfo', 'stderrthreshold', 'use_cprofile_for_profiling', 'v',
                'verbosity'}

@dataclasses.dataclass
class MemInfo:
    total: int  # KB
    res: int  # KB
    shared: int  # KB

    @classmethod
    def query(cls):
        with open(f'/proc/{os.getpid()}/statm', 'r') as f:
            return cls(*[int(x) for x in f.read().split(' ')[:3]])

    def __str__(self):
        gb = 1 << 20
        return f'Total {self.total / gb:.4f} GB | Res {self.res / gb:.4f} GB | Shared {self.shared / gb:.4f} GB'


def artifact_dir(*args) -> pathlib.Path:
    path = pathlib.Path(FLAGS.logdir)
    return path.joinpath(*args)


def command_line() -> str:
    argv = sys.argv[:]
    rex = re.compile(r'([!|*$#?~&<>{}()\[\]\\ "\'])')
    cmd = ' '.join(rex.sub(r'\\\1', v) for v in argv)
    return cmd


def convert_256_to_11(x: torch.Tensor) -> torch.Tensor:
    """Lossless conversion of 0,255 interval to -1,1 interval."""
    return x / 128 - 255 / 256


def cpu_count() -> int:
    return multiprocessing.cpu_count()


def downcast(x: Union[np.ndarray, np.dtype]) -> Union[np.ndarray, np.dtype]:
    """Downcast numpy float64 to float32."""
    if isinstance(x, np.dtype):
        return np.float32 if x == np.float64 else x
    if x.dtype == np.float64:
        return x.astype('f')
    return x


def ilog2(x: int) -> int:
    y = x.bit_length() - 1
    assert 1 << y == x
    return y


def int_str(s: str) -> int:
    p = 1
    if s.endswith('K'):
        s, p = s[:-1], 1 << 10
    elif s.endswith('M'):
        s, p = s[:-1], 1 << 20
    elif s.endswith('G'):
        s, p = s[:-1], 1 << 30
    return int(float(eval(s)) * p)


def local_kwargs(kwargs: dict, f: Callable) -> dict:
    """Return the kwargs from dict that are inputs to function f."""
    s = inspect.signature(f)
    p = s.parameters
    if next(reversed(p.values())).kind == inspect.Parameter.VAR_KEYWORD:
        return kwargs
    if len(kwargs) < len(p):
        return {k: v for k, v in kwargs.items() if k in p}
    return {k: kwargs[k] for k in p.keys() if k in kwargs}


def power_of_2(x: int) -> int:
    """Return highest power of 2 <= x"""
    return 1 << (x.bit_length() - 1)


def repeater(it: Iterable):
    """Helper function to repeat an iterator in a memory efficient way."""
    while True:
        for x in it:
            yield x


def report_module_weights(m: torch.nn.Module):
    weights = [(k, tuple(v.shape)) for k, v in m.named_parameters()]
    weights.append((f'Total ({len(weights)})', (sum(np.prod(x[1]) for x in weights),)))
    width = max(len(x[0]) for x in weights)
    return '\n'.join(f'{k:<{width}} {np.prod(s):>10} {str(s):>16}' for k, s in weights)


def setup(seed: Optional[int] = None, quiet: bool = False, flags_values: Optional[SimpleNamespace] = None):
    if flags_values:
        for k, v in vars(flags_values).items():
            setattr(FLAGS, k, v)
    else:
        for k in absl.flags.FLAGS:
            if k not in SYSTEM_FLAGS:
                setattr(FLAGS, k, getattr(absl.flags.FLAGS, k))
    torch.backends.cudnn.benchmark = True
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    if not quiet:
        print(f'{" Flags ":-^79s}')
        for k in sorted(vars(FLAGS)):
            print(f'{k:32s}: {getattr(FLAGS, k)}')
        print(f'{" System ":-^79s}')
        for k, v in {'cpus(system)': multiprocessing.cpu_count(),
                     'cpus(fixed)': cpu_count(),
                     'multiprocessing.start_method': torch.multiprocessing.get_start_method()}.items():
            print(f'{k:32s}: {v}')


def time_format(t: float) -> str:
    t = int(t)
    hours = t // 3600
    mins = (t // 60) % 60
    secs = t % 60
    return f'{hours:02d}:{mins:02d}:{secs:02d}'


def to_numpy(x: Union[np.ndarray, torch.Tensor]):
    if not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()


def to_png(x: Union[np.ndarray, torch.Tensor]) -> bytes:
    """Converts numpy array in (C, H, W) or (Rows, Cols, C, H, W) format into PNG format."""
    assert x.ndim in (3, 5)
    if isinstance(x, torch.Tensor):
        x = to_numpy(x)
    if x.ndim == 5:  # Image grid
        x = np.transpose(x, (2, 0, 3, 1, 4))
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3] * x.shape[4]))  # (C, H, W)
    if x.dtype in (np.float64, np.float32, np.float16):
        x = np.transpose(np.round(127.5 * (x + 1)), (1, 2, 0)).clip(0, 255).astype('uint8')
    elif x.dtype != np.uint8:
        raise ValueError('Unsupported array type, expecting float or uint8', x.dtype)
    if x.shape[2] == 1:
        x = np.broadcast_to(x, x.shape[:2] + (3,))
    with io.BytesIO() as f:
        Image.fromarray(x).save(f, 'png')
        return f.getvalue()


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

