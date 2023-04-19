#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
"""Compute FID and approximation at 50,000 for zip file of samples."""
import pathlib
import time
from types import SimpleNamespace
from typing import Optional

import lib
import torch
from absl import app, flags
from lib.distributed import auto_distribute, device_id
from lib.eval.fid import FID
from lib.io import Summary, SummaryWriter, zip_batch_as_png
from lib.util import FLAGS
from lib.zoo.unet import UNet


def logsnr_schedule_cosine(t, logsnr_min=torch.Tensor([-20.]), logsnr_max=torch.Tensor([20.])):
    b = torch.arctan(torch.exp(-0.5 * logsnr_max))
    a = torch.arctan(torch.exp(-0.5 * logsnr_min)) - b
    return -2. * torch.log(torch.tan(a * t + b))


def predict_eps_from_x(z, x, logsnr):
    """eps = (z - alpha*x)/sigma."""
    assert logsnr.ndim == x.ndim
    return torch.sqrt(1. + torch.exp(logsnr)) * (z - x * torch.rsqrt(1. + torch.exp(-logsnr)))


def predict_x_from_eps(z, eps, logsnr):
    """x = (z - sigma*eps)/alpha."""
    assert logsnr.ndim == eps.ndim
    return torch.sqrt(1. + torch.exp(-logsnr)) * (z - eps * torch.rsqrt(1. + torch.exp(logsnr)))


class ModelFID(torch.nn.Module):
    COLORS = 3

    def __init__(self, name: str, res: int, timesteps: int, **params):
        super().__init__()
        self.name = name
        if name == 'cifar10':
            self.model = UNet(in_channel=3,
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
                              num_classes=1).to(device_id())
            self.shape = 3, 32, 32
            self.mean_type = 'x'
            self.ckpt_name = 'cifar_original.pt'
            self.num_classes = 1
        elif name == 'imagenet64':
            # imagenet model is class conditional
            self.model = UNet(in_channel=3,
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
                              num_classes=1000).to(device_id())
            self.shape = 3, 64, 64
            self.mean_type = 'both'
            self.ckpt_name = 'imagenet_original.pt'
            self.num_classes = 1000
        else:
            raise NotImplementedError(name)
        self.params = SimpleNamespace(res=res, timesteps=timesteps, **params)
        self.timesteps = timesteps
        self.logstep = 0
        self.clip_x = True

    @property
    def logdir(self) -> str:
        params = ','.join(f'{k}={v}' for k, v in sorted(vars(self.params).items()))
        return f'{self.__class__.__name__}({params})'

    def initialize_weights(self, logdir: pathlib.Path):
        self.model.load_state_dict(torch.load(self.params.ckpt))

    def run_model(self, z, logsnr, y=None):
        if self.mean_type == 'x':
            model_x = self.model(z.float(), logsnr.float(), y).double()
            logsnr = logsnr[:, None, None, None]
        elif self.mean_type == 'both':
            output = self.model(z.float(), logsnr.float(), y).double()
            model_x, model_eps = output[:, :3], output[:, 3:]
            # reconcile the two predictions
            logsnr = logsnr[:, None, None, None]
            model_x_eps = predict_x_from_eps(z=z, eps=model_eps, logsnr=logsnr)
            wx = torch.sigmoid(-logsnr)
            model_x = wx * model_x + (1. - wx) * model_x_eps
        else:
            raise NotImplementedError(self.mean_type)

        # clipping
        if self.clip_x:
            model_x = torch.clip(model_x, -1., 1.)

        model_eps = predict_eps_from_x(z=z, x=model_x, logsnr=logsnr)
        return {'model_x': model_x,
                'model_eps': model_eps}

    def ddim_step(self, t, z_t, y=None, step=1024):
        logsnr_t = logsnr_schedule_cosine((t+step) / self.timesteps).to(z_t)
        logsnr_s = logsnr_schedule_cosine(t / self.timesteps).to(z_t)
        model_out = self.run_model(z=z_t, logsnr=logsnr_t.repeat(
            z_t.shape[0]), y=y.to(z_t).long() if y is not None else None)
        x_pred_t = model_out['model_x']
        eps_pred_t = model_out['model_eps']
        stdv_s = torch.sqrt(torch.sigmoid(-logsnr_s))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        z_s_pred = alpha_s * x_pred_t + stdv_s * eps_pred_t
        return torch.where(torch.Tensor([t]).to(x_pred_t) == 0, x_pred_t, z_s_pred)

    def sample_loop(self, init_x, y=None, step=1024):
        # loop over t = num_steps-1, ..., 0
        image = init_x
        for t in reversed(range(self.timesteps // step)):
            image = self.ddim_step(t * step, image, y, step=step)
        return image

    def forward(self, samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            assert generator.device == torch.device('cpu')
            init_x = torch.randn((samples, *self.shape), device='cpu', generator=generator, dtype=torch.double).to(device_id())
        else:
            init_x = torch.randn((samples, *self.shape), dtype=torch.double).to(device_id())
        if self.name == 'imagenet64':
            y = torch.randint(0, self.num_classes, (samples,)).to(device_id())
        else:
            y = None
        return self.sample_loop(init_x, y, step=1 << self.logstep)


@auto_distribute
def main(_):
    data = lib.data.DATASETS[FLAGS.dataset]()
    model = ModelFID(FLAGS.dataset, data.res, FLAGS.timesteps,
                     batch=FLAGS.batch, fid_len=FLAGS.fid_len, ckpt=FLAGS.ckpt)
    logdir = lib.util.artifact_dir(FLAGS.dataset, model.logdir)


    model.initialize_weights(logdir)
    model.eval()

    if FLAGS.eval:
        model.eval()
        with torch.no_grad():
            generator = torch.Generator(device='cpu')
            generator.manual_seed(123623113456)
            x = model(4, generator)
        open('debug_fid_model.png', 'wb').write(lib.util.to_png(x.view(2, 2, *x.shape[1:])))
        import numpy as np
        np.save('debug_arr_fid_model.npy', x.detach().cpu().numpy())
        return

    def eval(logstep: int):
        model.logstep = logstep
        summary = Summary()
        t0 = time.time()
        with torch.no_grad():
            fid = FID(FLAGS.dataset, (model.COLORS, model.params.res, model.params.res))
            fake_activations, fake_samples = fid.generate_activations_and_samples(model, FLAGS.fid_len)
            timesteps = model.params.timesteps >> model.logstep
            zip_batch_as_png(fake_samples, logdir / f'samples_{FLAGS.fid_len}_timesteps_{timesteps}.zip')
            fidn, fid50 = fid.approximate_fid(fake_activations)
        summary.scalar('eval/logstep', logstep)
        summary.scalar('eval/timesteps', timesteps)
        summary.scalar(f'eval/fid({FLAGS.fid_len})', fidn)
        summary.scalar('eval/fid(50000)', fid50)
        summary.scalar('system/eval_time', time.time() - t0)
        data_logger.write(summary, logstep)
        if lib.distributed.is_master():
            print(f'Logstep {logstep} Timesteps {timesteps}')
            print(summary)

    with SummaryWriter.create(logdir) as data_logger:
        if FLAGS.denoise_steps:
            logstep = lib.util.ilog2(FLAGS.timesteps // FLAGS.denoise_steps)
            eval(logstep)
        else:
            for logstep in range(lib.util.ilog2(FLAGS.timesteps) + 1):
                eval(logstep)


if __name__ == '__main__':
    flags.DEFINE_bool('eval', False, help='Whether to run model evaluation.')
    flags.DEFINE_integer('fid_len', 4096, help='Number of samples for FID evaluation.')
    flags.DEFINE_integer('timesteps', 1024, help='Sampling timesteps.')
    flags.DEFINE_string('dataset', 'cifar10', help='Dataset.')
    flags.DEFINE_integer('denoise_steps', None, help='Denoising timesteps.')
    flags.DEFINE_string('ckpt', None, help='Path to the model checkpoint.')
    app.run(lib.distributed.main(main))
