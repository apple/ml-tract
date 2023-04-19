#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

__all__ = ['AutoNorm', 'CondAffinePost', 'CondAffineScaleThenOffset', 'CondLinearlyCombine', 'EMA',
           'EmbeddingTriangle', 'Residual']

from typing import Callable, Optional, Tuple, Sequence

import torch
import torch.nn
import torch.nn.functional

from .functional import expand_to


class AutoNorm(torch.nn.Module):
    def __init__(self, n: int, momentum: float):
        super().__init__()
        self.avg = EMA((n,), momentum)
        self.var = EMA((n,), momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = [1] * (x.ndim - 2)
        if self.training:
            reduce = tuple(i for i in range(x.ndim) if i != 1)
            avg = self.avg(x.mean(reduce))
            var = self.var((x - x.mean(reduce, keepdims=True)).square().mean(reduce))
        else:
            avg, var = self.avg(), self.var()

        return (x - avg.view(1, -1, *pad)) * var.clamp(1e-6).rsqrt().view(1, -1, *pad)

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.training
        pad = [1] * (x.ndim - 2)
        avg, var = self.avg(), self.var()
        return avg.view(1, -1, *pad) + x * var.clamp(1e-6).sqrt().view(1, -1, *pad)


class CondAffinePost(torch.nn.Module):
    def __init__(self, ncond: int, nout: int, op: torch.nn.Module, scale: bool = True):
        super().__init__()
        self.op = op
        self.scale = scale
        self.m = torch.nn.Linear(ncond, nout + (nout if scale else 0))
        self.cond: Optional[torch.Tensor] = None

    def set_cond(self, x: Optional[torch.Tensor]):
        self.cond = x if x is None else self.m(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale:
            w, b = expand_to(self.cond, x).chunk(2, dim=1)
            return self.op(x) * w + b
        return self.op(x) + expand_to(self.cond, x)


class CondAffineScaleThenOffset(torch.nn.Module):
    def __init__(self, ncond: int, nin: int, nout: int, op: torch.nn.Module, scale: bool = True):
        super().__init__()
        self.op = op
        self.nin = nin
        self.scale = scale
        self.m = torch.nn.Linear(ncond, nout + (nin if scale else 0))
        self.cond: Optional[torch.Tensor] = None

    def set_cond(self, x: Optional[torch.Tensor]):
        self.cond = x if x is None else self.m(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cond = expand_to(self.cond, x)
        if self.scale:
            w, b = cond[:, :self.nin], cond[:, self.nin:]
            return self.op(x * w) + b
        return self.op(x) + cond


class CondLinearlyCombine(torch.nn.Module):
    def __init__(self, ncond: int, n: int):
        super().__init__()
        self.n = n
        self.mix = torch.nn.Linear(ncond, n)
        self.cond: Optional[torch.Tensor] = None

    def set_cond(self, x: Optional[torch.Tensor]):
        self.cond = x if x is None else self.mix(x)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        cond = expand_to(self.cond, x[0])
        return sum(x[i] * cond[:, i:i + 1] for i in range(self.n))


class EMA(torch.nn.Module):
    def __init__(self, shape: Tuple[int, ...], momentum: float):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('step', torch.zeros((), dtype=torch.long))
        self.register_buffer('ema', torch.zeros(shape))

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.training:
            self.step.add_(1)
            mu = 1 - (1 - self.momentum) / (1 - self.momentum ** self.step)
            self.ema.add_((1 - mu) * (x - self.ema))
        return self.ema


class EmbeddingTriangle(torch.nn.Module):
    def __init__(self, dim: int, delta: float):
        """dim number of dimensions for embedding, delta is minimum distance between two values."""
        super().__init__()
        logres = -torch.tensor(max(2 ** -31, 2 * delta)).log2()
        logfreqs = torch.nn.functional.pad(torch.linspace(0, logres, dim - 1), (1, 0), mode='constant', value=-1)
        self.register_buffer('freq', torch.pow(2, logfreqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = 2 * (x.view(-1, 1) * self.freq).fmod(1)
        return 2 * (y * (y < 1) + (2 - y) * (y >= 1)) - 1


class Residual(torch.nn.Module):
    def __init__(self, residual: Callable, skip: Optional[Callable] = None):
        super().__init__()
        self.residual = residual
        self.skip = torch.nn.Identity() if skip is None else skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.residual(x)
