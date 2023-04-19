#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import copy
import itertools

import torch
import torch.optim.swa_utils


class ModuleEMA(torch.nn.Module):  # Preferred to PyTorch's builtin because this is pickable
    def __init__(self, m: torch.nn.Module, momentum: float):
        super().__init__()
        self.module = copy.deepcopy(m)
        self.momentum = momentum
        self.register_buffer('step', torch.zeros((), dtype=torch.long))

    def update(self, source: torch.nn.Module):
        self.step.add_(1)
        decay = (1 - self.momentum) / (1 - self.momentum ** self.step)
        with torch.no_grad():
            for p_self, p_source in zip(self.module.parameters(), source.parameters()):
                p_self.add_(p_source - p_self, alpha=decay)
            for p_self, p_source in zip(self.module.buffers(), source.buffers()):
                if torch.is_floating_point(p_source):
                    assert torch.is_floating_point(p_self)
                    p_self.add_(p_source - p_self, alpha=decay)
                else:
                    assert not torch.is_floating_point(p_self)
                    p_self.add_(p_source - p_self)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)


class AveragedModel(torch.optim.swa_utils.AveragedModel):
    def update_parameters(self, model):
        self_param = itertools.chain(self.module.parameters(), self.module.buffers())
        model_param = itertools.chain(model.parameters(), model.buffers())
        for p_swa, p_model in zip(self_param, model_param):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device)))
        self.n_averaged += 1


def module_exponential_moving_average(model: torch.nn.Module,
                                      momentum: float) -> torch.optim.swa_utils.AveragedModel:
    """Create an AverageModel using Stochastic Weight Averaging.

    Args:
        model: the torch Module to average.
        momentum: the running average momentum coefficient. I found values in 0.9, 0.99, 0.999,
          0.9999, ... to give good results. The closer to 1 the better, but the longer one needs
          to train.
    Returns:
        torch.optim.swa_utils.AveragedModel module that replicates the model behavior with SWA
          weights.
    """

    def ema(target: torch.Tensor, source: torch.Tensor, count: int) -> torch.Tensor:
        mu = 1 - (1 - momentum) / (1 - momentum ** (1 + count))
        return mu * target + (1 - mu) * source

    return AveragedModel(model, avg_fn=ema)


class HalfLifeEMA(torch.nn.Module):
    def __init__(self, m: torch.nn.Module, half_life: int = 500000, batch_size: int = 512):
        """
        EMA Module based of half life (units of samples/images).

        Args:
            half_life   :   Half life of EMA in units of samples.
        """
        super().__init__()
        self.module = copy.deepcopy(m)
        self.half_life = half_life
        self.batch_size = batch_size

    def update(self, source: torch.nn.Module):
        ema_beta = 0.5 ** (self.batch_size / self.half_life)

        with torch.no_grad():
            for p_self, p_source in zip(self.module.parameters(), source.parameters()):
                p_self.copy_(p_source.lerp(p_self, ema_beta))
            for p_self, p_source in zip(self.module.buffers(), source.buffers()):
                p_self.copy_(p_source)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)


class CopyModule(torch.nn.Module):
    def __init__(self, m: torch.nn.Module):
        """Copy Module for self-conditioning."""
        super().__init__()
        self.module = copy.deepcopy(m)

    def update(self, source: torch.nn.Module):

        with torch.no_grad():
            for p_self, p_source in zip(self.module.parameters(), source.parameters()):
                p_self.copy_(p_source)
            for p_self, p_source in zip(self.module.buffers(), source.buffers()):
                p_self.copy_(p_source)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)