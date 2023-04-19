#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import copy
import functools
import math
import os
import pathlib
import shutil
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional
from absl import app, flags

import lib
from lib.distributed import device, device_id, print
from lib.util import FLAGS
from lib.zoo.unet import UNet


def get_model(name: str):
    if name == 'cifar10':
        net = UNet(in_channel=3,
                   channel=256,
                   emb_channel=1024,
                   channel_multiplier=[1, 1, 1],
                   n_res_blocks=3,
                   attn_rezs=[8, 16],
                   attn_heads=1,
                   head_dim=None,
                   use_affine_time=True,
                   dropout=0.2,
                   num_output=1,
                   resample=True,
                   num_classes=1)
    elif name == 'imagenet64':
        # imagenet model is class conditional
        net = UNet(in_channel=3,
                   channel=192,
                   emb_channel=768,
                   channel_multiplier=[1, 2, 3, 4],
                   n_res_blocks=3,
                   init_rez=64,
                   attn_rezs=[8, 16, 32],
                   attn_heads=None,
                   head_dim=64,
                   use_affine_time=True,
                   dropout=0.,
                   num_output=2,  # predict signal and noise
                   resample=True,
                   num_classes=1000)
    else:
        raise NotImplementedError(name)
    return net


class BinaryDistillGoogleModel(lib.train.TrainModel):
    R_NONE, R_STEP, R_PHASE = 'none', 'step', 'phase'
    R_ALL = R_NONE, R_STEP, R_PHASE

    def __init__(self, name: str, res: int, timesteps: int, **params):
        super().__init__("GoogleUNet", res, timesteps, **params)
        self.num_classes = 1
        self.shape = 3, res, res
        self.timesteps = timesteps
        model = get_model(name)
        if 'cifar' in name:
            self.ckpt_path = 'ckpts/cifar_original.pt'
            self.predict_both = False
        elif 'imagenet' in name:
            self.ckpt_path = 'ckpts/imagenet_original.pt'
            self.predict_both = False
        elif 'imagenet' in name:
            self.ckpt_path = 'ckpts/imagenet_original.pt'
            self.num_classes = 1000
            self.predict_both = True
            self.EVAL_COLUMNS = self.EVAL_ROWS = 8
        else:
            raise NotImplementedError(name)

        model.apply(functools.partial(lib.nn.functional.set_bn_momentum, momentum=1 - self.params.ema))
        model.apply(functools.partial(lib.nn.functional.set_dropout, p=0))
        self.model = lib.distributed.wrap(model)
        self.model_eval = lib.optim.ModuleEMA(model, momentum=self.params.ema).to(device_id())
        self.teacher = copy.deepcopy(model).to(device_id())
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        self.register_buffer('phase', torch.zeros((), dtype=torch.long))
        self.cur_step = self.timesteps // 2

    def initialize_weights_from_teacher(self, logdir: pathlib.Path, teacher_ckpt: Optional[str] = None):
        teacher_ckpt_path = logdir / 'ckpt/teacher.ckpt'
        if device_id() == 0:
            os.makedirs(logdir / 'ckpt', exist_ok=True)
            shutil.copy2(self.ckpt_path, teacher_ckpt_path)

        lib.distributed.barrier()
        self.model.module.load_state_dict(torch.load(teacher_ckpt_path))
        self.model_eval.module.load_state_dict(torch.load(teacher_ckpt_path))
        self.self_teacher.module.load_state_dict(torch.load(teacher_ckpt_path))
        self.teacher.load_state_dict(torch.load(teacher_ckpt_path))

    def randn(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            assert generator.device == torch.device('cpu')
        return torch.randn((n, *self.shape), device='cpu', generator=generator, dtype=torch.double).to(self.device)

    def call_model(self, model: Callable, xt: torch.Tensor, index: torch.Tensor,
                   y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y is None:
            return model(xt.float(), index.float()).double()
        else:
            return model(xt.float(), index.float(), y.long()).double()

    def forward(self, samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        step = self.timesteps // self.cur_step
        xt = self.randn(samples, generator).to(device_id())
        if self.num_classes > 1:
            y = torch.randint(0, self.num_classes, (samples,)).to(xt)
        else:
            y = None

        for t in reversed(range(0, self.timesteps, step)):
            ix = torch.Tensor([t + step]).long().to(device_id()), torch.Tensor([t]).long().to(device_id())
            logsnr = tuple(self.logsnr_schedule_cosine(i / self.timesteps).to(xt.double()) for i in ix)
            g = tuple(torch.sigmoid(l).view(-1, 1, 1, 1) for l in logsnr)  # Get gamma values
            x0 = self.call_model(self.model_eval, xt, logsnr[0].repeat(xt.shape[0]), y)
            xt = self.post_xt_x0(xt, x0, g[0], g[1], clip_x=True)
        return xt

    @staticmethod
    def logsnr_schedule_cosine(t, logsnr_min=torch.Tensor([-20.]), logsnr_max=torch.Tensor([20.])):
        b = torch.arctan(torch.exp(-0.5 * logsnr_max)).to(t)
        a = torch.arctan(torch.exp(-0.5 * logsnr_min)).to(t) - b
        return -2. * torch.log(torch.tan(a * t + b))

    @staticmethod
    def predict_eps_from_x(z, x, logsnr):
        """eps = (z - alpha*x)/sigma."""
        assert logsnr.ndim == x.ndim
        return torch.sqrt(1. + torch.exp(logsnr)) * (z - x * torch.rsqrt(1. + torch.exp(-logsnr)))

    def post_xt_x0(self, xt: torch.Tensor, out: torch.Tensor, g: torch.Tensor, g1: torch.Tensor, clip_x=False) -> torch.Tensor:
        if self.predict_both:
            assert out.shape[1] == 6
            model_x, model_eps = out[:, :3], out[:, 3:]
            # reconcile the two predictions
            model_x_eps = (xt - model_eps * (1 - g).sqrt()) * g.rsqrt()
            wx = 1 - g
            x0 = wx * model_x + (1. - wx) * model_x_eps
        else:
            x0 = out
        if clip_x:
            x0 = torch.clip(x0, -1., 1.)
        eps = (xt - x0 * g.sqrt()) * (1 - g).rsqrt()
        return torch.nan_to_num(x0 * g1.sqrt() + eps * (1 - g1).sqrt())

    def train_op(self, info: lib.train.TrainInfo, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.num_classes == 1:
            y = None
        else:
            y = y[:, 0]
        with torch.no_grad():
            phase = int(info.progress * (1 - 1e-9) * math.log(self.timesteps, 2))
            if phase != self.phase:
                print(f'Refreshing teacher {phase}')
                self.phase.add_(1)
                self.teacher.load_state_dict(self.model_eval.module.state_dict())
                if self.params.reset == self.R_PHASE:
                    self.model_eval.step.mul_(0)
                self.cur_step = self.cur_step // 2
                assert self.cur_step >= 1

            step = self.timesteps // self.cur_step
            index = torch.randint(1, 1 + (self.timesteps // step), (x.shape[0],), device=device()) * step
            ix = index, index - step // 2, index - step
            logsnr = tuple(self.logsnr_schedule_cosine(i.double() / self.timesteps).to(x.double()) for i in ix)
            g = tuple(torch.sigmoid(l).view(-1, 1, 1, 1) for l in logsnr)  # Get gamma values
            noise = torch.randn_like(x)
            xt0 = x.double() * g[0].sqrt() + noise * (1 - g[0]).sqrt()
            xt1 = self.post_xt_x0(xt0, self.call_model(self.teacher, xt0, logsnr[0], y), g[0], g[1])
            x_hat = self.call_model(self.teacher, xt1, logsnr[1], y)
            xt2 = self.post_xt_x0(xt1, x_hat, g[1], g[2])
            # Find target such that self.post_xt_x0(xt0, target, g[0], g[2]) == xt2
            target = ((xt0 * (1 - g[2]).sqrt() - xt2 * (1 - g[0]).sqrt()) /
                      ((g[0] * (1 - g[2])).sqrt() - (g[2] * (1 - g[0])).sqrt()))
            # use predicted x0 as target when t=0
            target += (index == step).view(-1, 1, 1, 1) * (x_hat[:, :3] - target)

        self.opt.zero_grad(set_to_none=True)
        pred = self.call_model(self.model, xt0, logsnr[0], y)
        if self.predict_both:
            assert pred.shape[1] == 6
            model_x, model_eps = pred[:, :3], pred[:, 3:]
            # reconcile the two predictions
            model_x_eps = (xt0 - model_eps * (1 - g[0]).sqrt()) * g[0].rsqrt()
            wx = 1 - g[0]
            pred_x = wx * model_x + (1. - wx) * model_x_eps
        else:
            pred_x = pred

        loss = ((g[0] / (1 - g[0])).clamp(1) * (pred_x - target.detach()).square()).mean(0).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.opt.step()
        self.model_eval.update(self.model)
        return {'loss/global': loss, 'stat/timestep': self.cur_step}


@ lib.distributed.auto_distribute
def main(_):
    data = lib.data.DATASETS[FLAGS.dataset]()
    model = BinaryDistillGoogleModel(FLAGS.dataset, data.res, FLAGS.timesteps, reset=FLAGS.reset,
                                     batch=FLAGS.batch, lr=FLAGS.lr, ema=FLAGS.ema)
    logdir = lib.util.artifact_dir(FLAGS.dataset, model.logdir)
    # resume from previous run (ckpt will be loaded in train.py)
    if FLAGS.restart_ckpt:
        lib.distributed.barrier()

    if FLAGS.eval:
        if not FLAGS.restart_ckpt:
            model.initialize_weights_from_teacher(logdir, FLAGS.teacher_ckpt)
        model.eval()
        with torch.no_grad():
            generator = torch.Generator(device='cpu')
            generator.manual_seed(123623113456)
            model(4, generator)
    else:
        train, fid = data.make_dataloaders()
        if not FLAGS.restart_ckpt:
            model.initialize_weights_from_teacher(logdir, FLAGS.teacher_ckpt)
        model.train_loop(train, fid, FLAGS.batch, FLAGS.train_len, FLAGS.report_len, logdir, fid_len=FLAGS.fid_len)


if __name__ == '__main__':
    flags.DEFINE_bool('eval', False, help='Whether to run model evaluation.')
    flags.DEFINE_enum('reset', BinaryDistillGoogleModel.R_NONE, BinaryDistillGoogleModel.R_ALL, help='EMA reset mode.')
    flags.DEFINE_float('ema', 0.9995, help='Exponential Moving Average of model.')
    flags.DEFINE_float('lr', 2e-4, help='Learning rate.')
    flags.DEFINE_integer('fid_len', 4096, help='Number of samples for FID evaluation.')
    flags.DEFINE_integer('timesteps', 1024, help='Sampling timesteps.')
    flags.DEFINE_string('dataset', 'cifar10', help='Training dataset.')
    flags.DEFINE_string('train_len', '64M', help='Training duration in samples per distillation logstep.')
    flags.DEFINE_string('report_len', '1M', help='Reporting interval in samples.')
    flags.FLAGS.set_default('report_img_len', '1M')
    flags.FLAGS.set_default('report_fid_len', '4M')
    flags.DEFINE_string('restart_ckpt', None,
                        help='Trainer checkpoint in the form <taskid>:<ckpt_path> with <ckpt_path> of the form "ckpt/*.pth" .')
    flags.DEFINE_string('teacher_ckpt', None,
                        help='Teacher checkpoint in the form <taskid>:<ckpt_path> with <ckpt_path> of the form "ckpt/model_*.ckpt".')
    app.run(lib.distributed.main(main))
