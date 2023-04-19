#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

__all__ = ['expand_to', 'float_index', 'label_smoothing', 'set_bn_momentum', 'set_cond', 'set_dropout',
           'default', 'log']

from typing import Optional

import torch
import torch.nn.functional


def expand_to(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Expand x to the number of dimensions in y."""
    return x.view(x.shape + (1,) * (y.ndim - x.ndim))


def float_index(x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    a, b = x[i.long()], x[i.ceil().long()]
    return a + i.frac() * (b - a)


def label_smoothing(x: torch.Tensor, q: float) -> torch.Tensor:
    u = torch.zeros_like(x) + 1 / x.shape[-1]
    return x + q * (u - x)


def set_bn_momentum(m: torch.nn.Module, momentum: float):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        print('Set momentum for', m)
        m.momentum = momentum


def set_cond(cond: Optional[torch.Tensor]):
    def apply_op(m: torch.nn.Module):
        if hasattr(m, 'set_cond'):
            m.set_cond(cond)

    return apply_op


def set_dropout(m: torch.nn.Module, p: float):
    if isinstance(m, torch.nn.modules.dropout._DropoutNd):
        print(f'Set dropout to {p} for', m)
        m.p = p


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))
