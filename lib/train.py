#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

__all__ = ['TrainInfo', 'TrainModel', 'DistillModel']

import dataclasses
import json
import pathlib
import time
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional

import torch.distributed
import torch.nn.functional
from absl import flags

from lib.eval.fid import FID

from .distributed import (gather_tensor, is_master, print,
                          rank, trange, world_size)
from .io import Checkpoint, Summary, SummaryWriter, zip_batch_as_png
from .util import (FLAGS, command_line, int_str, repeater,
                   report_module_weights, time_format)

flags.DEFINE_integer('logstart', 1, help='Logstep at which to start.')
flags.DEFINE_string('report_fid_len', '16M', help='How often to compute the FID during evaluations.')
flags.DEFINE_string('report_img_len', '4M', help='How often to sample images during evaluations.')


@dataclasses.dataclass
class TrainInfo:
    samples: int
    progress: float


class TrainModel(torch.nn.Module):
    COLORS = 3
    EVAL_ROWS = 16
    EVAL_COLUMNS = 16
    model: torch.nn.Module
    model_eval: torch.nn.Module
    train_op: Callable[..., Dict[str, torch.Tensor]]

    def __init__(self, arch: str, res: int, timesteps: int, **params):
        super().__init__()
        self.params = SimpleNamespace(arch=arch, res=res, timesteps=timesteps, **params)
        self.register_buffer('logstep', torch.zeros((), dtype=torch.long))

    @property
    def device(self) -> str:
        for x in self.model.parameters():
            return x.device

    @property
    def logdir(self) -> str:
        params = '_'.join(f'{k}@{v}' for k, v in sorted(vars(self.params).items()) if k not in ('arch',))
        return f'{self.__class__.__name__}({self.params.arch})/{params}'

    def __str__(self) -> str:
        return '\n'.join((
            f'{" Model ":-^80}', str(self.model),
            f'{" Parameters ":-^80}', report_module_weights(self.model),
            f'{" Config ":-^80}',
            '\n'.join(f'{k:20s}: {v}' for k, v in vars(self.params).items())
        ))

    def save_meta(self, logdir: pathlib.Path, data_logger: Optional[SummaryWriter] = None):
        if not is_master():
            return
        if data_logger is not None:
            summary = Summary()
            summary.text('info', f'<pre>{self}</pre>')
            data_logger.write(summary, 0)
        (logdir / 'params.json').open('w').write(json.dumps(vars(self.params), indent=4))
        (logdir / 'model.txt').open('w').write(str(self.model.module))
        (logdir / 'cmd.txt').open('w').write(command_line())

    def evaluate(self, summary: Summary,
                 logdir: pathlib.Path,
                 ckpt: Optional[Checkpoint] = None,
                 data_fid: Optional[Iterable] = None,
                 fid_len: int = 0, sample_imgs: bool = True):
        assert (self.EVAL_ROWS * self.EVAL_COLUMNS) % world_size() == 0
        self.eval()
        with torch.no_grad():
            if sample_imgs:
                generator = torch.Generator(device='cpu')
                generator.manual_seed(123623113456 + rank())
                fixed = self((self.EVAL_ROWS * self.EVAL_COLUMNS) // world_size(), generator)
                rand = self((self.EVAL_ROWS * self.EVAL_COLUMNS) // world_size())
                fixed, rand = (gather_tensor(x) for x in (fixed, rand))
                summary.png('eval/fixed', fixed.view(self.EVAL_ROWS, self.EVAL_COLUMNS, *fixed.shape[1:]))
                summary.png('eval/random', rand.view(self.EVAL_ROWS, self.EVAL_COLUMNS, *rand.shape[1:]))
            if fid_len and data_fid:
                fid = FID(FLAGS.dataset, (self.COLORS, self.params.res, self.params.res))
                fake_activations, fake_samples = fid.generate_activations_and_samples(self, FLAGS.fid_len)
                timesteps = self.params.timesteps >> self.logstep.item()
                zip_batch_as_png(fake_samples, logdir / f'samples_{fid_len}_timesteps_{timesteps}.zip')
                fidn, fid50 = fid.approximate_fid(fake_activations)
                summary.scalar(f'eval/fid({fid_len})', fidn)
                summary.scalar('eval/fid(50000)', fid50)
                if ckpt:
                    ckpt.save_file(self.model_eval.module, f'model_{fid50:.5f}.ckpt')

    def train_loop(self,
                   data_train: Iterable,
                   data_fid: Optional[Iterable],
                   batch: int,
                   train_len: str,
                   report_len: str,
                   logdir: pathlib.Path,
                   *,
                   fid_len: int = 4096,
                   keep_ckpts: int = 2):
        print(self)
        print(f'logdir: {logdir}')
        train_len, report_len, report_fid_len, report_img_len = (int_str(x) for x in (
            train_len, report_len, FLAGS.report_fid_len, FLAGS.report_img_len))
        assert report_len % batch == 0
        assert train_len % report_len == 0
        assert report_fid_len % report_len == 0
        assert report_img_len % report_len == 0
        data_train = repeater(data_train)
        ckpt = Checkpoint(self, logdir, keep_ckpts)
        start = ckpt.restore()[0]
        if start:
            print(f'Resuming training at {start} ({start / (1 << 20):.2f}M samples)')

        with SummaryWriter.create(logdir) as data_logger:
            if start == 0:
                self.save_meta(logdir, data_logger)

            for i in range(start, train_len, report_len):
                self.train()
                summary = Summary()
                range_iter = trange(i, i + report_len, batch, leave=False, unit='samples',
                                    unit_scale=batch,
                                    desc=f'Training kimg {i >> 10}/{train_len >> 10}')
                t0 = time.time()
                for samples in range_iter:
                    self.train_step(summary, TrainInfo(samples, samples / train_len), next(data_train))

                samples += batch
                t1 = time.time()
                summary.scalar('sys/samples_per_sec_train', report_len / (t1 - t0))
                compute_fid = (samples % report_fid_len == 0) or (samples >= train_len)
                self.evaluate(summary, logdir, ckpt, data_fid, fid_len=fid_len if compute_fid else 0,
                              sample_imgs=samples % report_img_len == 0)
                t2 = time.time()
                summary.scalar('sys/eval_time', t2 - t1)
                data_logger.write(summary, samples)
                ckpt.save(samples)
                print(f'{samples / (1 << 20):.2f}M/{train_len / (1 << 20):.2f}M samples, '
                      f'time left {time_format((t2 - t0) * (train_len - samples) / report_len)}\n{summary}')
        ckpt.save_file(self.model_eval.module, 'model.ckpt')

    def train_step(self, summary: Summary, info: TrainInfo, batch: List[torch.Tensor]) -> None:
        device = self.device
        metrics = self.train_op(info, *[x.to(device, non_blocking=True) for x in batch])
        summary.from_metrics(metrics)
