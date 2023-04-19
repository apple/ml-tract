#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import copy
import functools
import pickle
import sys
from typing import Dict, Optional

import torch
import torch.nn.functional
from absl import app, flags

import lib
from lib.distributed import device, device_id
from lib.util import FLAGS, int_str

# Imports within edm/ are often relative to edm/ so we do this.
sys.path.append('edm')
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc


class EluDDIM05TCMultiStepx0(lib.train.TrainModel):
    SIGMA_DATA = 0.5
    SIGMA_MIN: float = 0.002
    SIGMA_MAX: float = 80.
    RHO: float = 7.

    def __init__(self, res: int, timesteps: int, **params):
        super().__init__("EluUNet", res, timesteps, **params)
        self.use_imagenet = FLAGS.dataset == "imagenet64"
        self.num_classes = 1000 if self.use_imagenet else 10

        # Setup pretrained model
        lib.distributed.barrier()
        if FLAGS.dataset == "imagenet64":
            pretrained_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl"
        elif FLAGS.dataset == "cifar10":
            pretrained_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl"
        else:
            raise ValueError("Only cifar10 or imagenet64 is supported for now.")
        with dnnlib.util.open_url(pretrained_url) as f:
            pretrained = pickle.load(f)['ema']
        lib.distributed.barrier()

        network_kwargs = self.get_pretrained_cifar10_network_kwargs()
        if self.use_imagenet:
            network_kwargs = self.get_pretrained_imagenet_network_kwargs()
        label_dim = self.num_classes if self.use_imagenet else 0
        interface_kwargs = dict(img_resolution=res, img_channels=3, label_dim=label_dim)
        model = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
        model.train().requires_grad_(True)
        misc.copy_params_and_buffers(src_module=pretrained, dst_module=model, require_all=False)
        del pretrained      # save memory

        self.time_schedule = tuple(int(x) for x in self.params.time_schedule.split(','))
        steps_per_phase = int_str(FLAGS.train_len) / (FLAGS.batch * (len(self.time_schedule) - 1))
        ema = self.params.ema_residual ** (1 / steps_per_phase)
        model.apply(functools.partial(lib.nn.functional.set_bn_momentum, momentum=1 - ema))
        model.apply(functools.partial(lib.nn.functional.set_dropout, p=self.params.dropout))
        self.model = lib.distributed.wrap(model)
        self.model_eval = lib.optim.ModuleEMA(model, momentum=ema).eval().requires_grad_(False).to(device_id())
        lib.distributed.barrier()

        # Disable dropout noise for teacher
        model.apply(functools.partial(lib.nn.functional.set_dropout, p=0))
        self.self_teacher = lib.optim.ModuleEMA(model, momentum=self.params.sema).to(device_id())
        self.self_teacher.eval().requires_grad_(False)
        self.teacher = copy.deepcopy(model).to(device_id())
        self.teacher.eval().requires_grad_(False)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=0.0)

        # Setup noise schedule
        sigma = torch.linspace(self.SIGMA_MIN ** (1 / self.RHO),
                               self.SIGMA_MAX ** (1 / self.RHO), timesteps, dtype=torch.double).pow(self.RHO)
        sigma = torch.cat([torch.zeros_like(sigma[:1]), sigma])
        self.register_buffer('sigma', sigma.to(device()))
        self.timesteps = timesteps

    def get_pretrained_cifar10_network_kwargs(self):
        network_kwargs = dnnlib.EasyDict()
        network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
        network_kwargs.class_name = 'training.networks.EDMPrecond'
        network_kwargs.augment_dim = 0
        network_kwargs.update(dropout=0.0, use_fp16=False)
        return network_kwargs

    def get_pretrained_imagenet_network_kwargs(self):
        network_kwargs = dnnlib.EasyDict()
        network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])
        network_kwargs.class_name = 'training.networks.EDMPrecond'
        network_kwargs.update(dropout=0.0, use_fp16=False)
        return network_kwargs

    @classmethod
    def c_in(cls, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma ** 2 + cls.SIGMA_DATA ** 2) ** -0.5

    @classmethod
    def c_skip(cls, sigma: torch.Tensor) -> torch.Tensor:
        return (cls.SIGMA_DATA ** 2) / (sigma ** 2 + cls.SIGMA_DATA ** 2)

    @classmethod
    def c_out(cls, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * cls.SIGMA_DATA * (cls.SIGMA_DATA ** 2 + sigma ** 2) ** -0.5

    @staticmethod
    def c_noise(sigma: torch.Tensor) -> torch.Tensor:
        return 0.25 * sigma.clamp(1e-20).log()

    def forward(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        step = self.timesteps // self.time_schedule[1]
        shape = n, self.COLORS, self.params.res, self.params.res

        xt = self.sigma[-1] * torch.randn(shape, generator=generator, dtype=torch.double).to(device())
        class_labels = (torch.eye(self.num_classes, device=device())[torch.randint(self.num_classes, size=[n], device=device())]) if self.use_imagenet else None

        for t in reversed(range(0, self.timesteps, step)):
            ix = torch.Tensor([t + step]).long().to(device_id()), torch.Tensor([t]).long().to(device_id())
            g = tuple(self.sigma[i].view(-1, 1, 1, 1) for i in ix)
            x0 = self.model_eval(xt, g[0], class_labels).to(torch.float64)
            xt = self.post_xt_x0(xt, x0, g[0], g[1])

        return xt.clamp(-1, 1).float()

    def post_xt_x0(self, xt: torch.Tensor, out: torch.Tensor, sigma: torch.Tensor, sigma1: torch.Tensor) -> torch.Tensor:
        x0 = torch.clip(out, -1., 1.)
        eps = (xt - x0) / sigma
        return torch.nan_to_num(x0 + eps * sigma1)

    def train_op(self, info: lib.train.TrainInfo, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.num_classes == 1000:    # imagenet
            y = torch.nn.functional.one_hot(y, self.num_classes).to(y.device)
        else:
            y = None

        with torch.no_grad():

            step = self.timesteps // self.time_schedule[1]
            index = torch.randint(1, 1 + (self.timesteps // step), (x.shape[0],), device=device()) * step
            semi_index = torch.randint(step, index.shape, device=device())
            ix = index - semi_index, (index - semi_index - 1).clamp(1), index - step

            s = tuple(self.sigma[i].view(-1, 1, 1, 1) for i in ix)
            noise = torch.randn_like(x).to(device())

            # RK step from teacher
            xt = x.double() + noise * s[0]
            x0 = self.teacher(xt, s[0], y)
            eps = (xt - x0) / s[0]
            xt_ = xt + (s[1] - s[0]) * eps
            x0_ = self.teacher(xt_, s[1], y)
            eps = .5 * (eps + (xt_ - x0_) / s[1])
            xt_ = xt + (s[1] - s[0]) * eps      # RK target from teacher; no RK needed for sigma_min

            # self-teacher step
            xt2 = self.post_xt_x0(xt_, self.self_teacher(xt_, s[1], y), s[1], s[2])
            xt2 += ((semi_index + 1) == step).view(-1, 1, 1, 1) * (xt_ - xt2)   # Only propagate inside phase semi_range

            xt2 = ((xt2 * s[0] - xt * s[2]) / (s[0] - s[2]))

            # Boundary and terminal condition: last time step, no RK and self-teaching needed
            target_without_precon = torch.where((index - semi_index - 1).view(-1, 1, 1, 1) == 0, x0.double(), xt2.double())

            target = (target_without_precon - self.c_skip(s[0]) * xt) / self.c_out(s[0])

        self.opt.zero_grad(set_to_none=True)
        pred = self.model(xt.float(), s[0].float(), y).double()
        pred = (pred - self.c_skip(s[0]) * xt) / self.c_out(s[0])

        weight = (s[0] ** 2 + self.SIGMA_DATA ** 2) * (self.c_out(s[0]) ** 2) * (s[0] * self.SIGMA_DATA) ** -2
        loss = (torch.nn.functional.mse_loss(pred.float(), target.float(), reduction='none')).mean((1, 2, 3))
        loss = (weight.float() * loss).mean()

        loss.backward()

        # LR warmup and clip gradient like EDM paper
        if self.params.lr_warmup is not None:
            for g in self.opt.param_groups:
                g['lr'] = self.params.lr * min(info.samples / max(int_str(self.params.lr_warmup), 1e-8), 1)
        for param in self.model.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        self.opt.step()
        self.self_teacher.update(self.model)
        self.model_eval.update(self.model)
        return {'loss/global': loss}


def check_steps():
    timesteps = [int(x) for x in FLAGS.time_schedule.split(',')]
    assert len(timesteps) > 1
    assert timesteps[0] == FLAGS.timesteps
    for i in range(len(timesteps) - 1):
        assert timesteps[i + 1] < timesteps[i]


@lib.distributed.auto_distribute
def main(_):
    check_steps()
    data = lib.data.DATASETS[FLAGS.dataset]()
    lib.distributed.barrier()
    model = EluDDIM05TCMultiStepx0(data.res, FLAGS.timesteps, batch=FLAGS.batch, lr=FLAGS.lr,
                                   ema_residual=FLAGS.ema_residual, sema=FLAGS.sema, lr_warmup=FLAGS.lr_warmup,
                                   aug_prob=FLAGS.aug_prob, dropout=FLAGS.dropout, time_schedule=FLAGS.time_schedule)
    lib.distributed.barrier()
    logdir = lib.util.artifact_dir(FLAGS.dataset, model.logdir)
    train, fid = data.make_dataloaders()
    model.train_loop(train, fid, FLAGS.batch, FLAGS.train_len, FLAGS.report_len, logdir, fid_len=FLAGS.fid_len)


if __name__ == '__main__':
    flags.DEFINE_float('ema_residual', 1e-3, help='Residual for the Exponential Moving Average of model.')
    flags.DEFINE_float('sema', 0.5, help='Exponential Moving Average of self-teacher.')
    flags.DEFINE_float('lr', 1e-3, help='Learning rate.')
    flags.DEFINE_string('lr_warmup', None, help='Warmup for LR in num samples, e.g. 4M')
    flags.DEFINE_integer('fid_len', 50000, help='Number of samples for FID evaluation.')
    flags.DEFINE_integer('timesteps', 40, help='Sampling timesteps.')
    flags.DEFINE_string('time_schedule', None, required=True,
                        help='Comma separated distillation timesteps, for example: 36,1.')
    flags.DEFINE_string('dataset', 'cifar10', help='Training dataset. Either cifar10 or imagenet64')
    flags.DEFINE_string('report_len', '1M', help='Reporting interval in samples.')
    flags.DEFINE_string('train_len', '64M', help='Training duration in samples per distillation logstep.')
    flags.DEFINE_float('aug_prob', 0.0, help='Probability of applying data augmentation in training.')
    flags.DEFINE_float('dropout', 0.0, help='Dropout probability for training.')
    flags.FLAGS.set_default('report_img_len', '1M')
    flags.FLAGS.set_default('report_fid_len', '4M')
    app.run(lib.distributed.main(main))
