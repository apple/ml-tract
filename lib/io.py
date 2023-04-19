#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

__all__ = ['Checkpoint', 'Summary', 'SummaryWriter', 'zip_batch_as_png']

import enum
import io
import os
import pathlib
import zipfile
from time import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import imageio
import matplotlib.figure
import numpy as np
import torch
import torch.nn
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.util.tensor_util import make_tensor_proto

from .distributed import is_master, print, reduce_dict_mean
from .util import to_numpy, to_png


class Checkpoint:
    DIR_NAME: str = 'ckpt'
    FILE_MATCH: str = '*.pth'
    FILE_FORMAT: str = '%012d.pth'

    def __init__(self,
                 model: torch.nn.Module,
                 logdir: pathlib.Path,
                 keep_ckpts: int = 0):
        self.model = model
        self.logdir = logdir / self.DIR_NAME
        self.keep_ckpts = keep_ckpts

    @staticmethod
    def checkpoint_idx(filename: str) -> int:
        return int(os.path.basename(filename).split('.')[0])

    def restore(self, idx: Optional[int] = None) -> Tuple[int, Optional[pathlib.Path]]:
        if idx is None:
            all_ckpts = self.logdir.glob(self.FILE_MATCH)
            try:
                idx = self.checkpoint_idx(max(str(x) for x in all_ckpts))
            except ValueError:
                return 0, None
        ckpt = self.logdir / (self.FILE_FORMAT % idx)
        print(f'Resuming from: {ckpt}')
        with ckpt.open('rb') as f:
            self.model.load_state_dict(torch.load(f, map_location='cpu'))
        return idx, ckpt

    def save(self, idx: int) -> None:
        if not is_master():  # only save master's state
            return
        self.logdir.mkdir(exist_ok=True, parents=True)
        ckpt = self.logdir / (self.FILE_FORMAT % idx)
        with ckpt.open('wb') as f:
            torch.save(self.model.state_dict(), f)
        old_ckpts = sorted(self.logdir.glob(self.FILE_MATCH), key=str)
        for ckpt in old_ckpts[:-self.keep_ckpts]:
            ckpt.unlink()

    def save_file(self, model: torch.nn.Module, filename: str) -> None:
        if not is_master():  # only save master's state
            return
        self.logdir.mkdir(exist_ok=True, parents=True)
        with (self.logdir / filename).open('wb') as f:
            torch.save(model.state_dict(), f)

class Summary(dict):
    """Helper to generate summary_pb2.Summary protobufs."""

    # Inspired from https://github.com/google/objax/blob/master/objax/jaxboard.py

    class ProtoMode(enum.Flag):
        """Enum describing what to export to a tensorboard proto."""

        IMAGES = enum.auto()
        VIDEOS = enum.auto()
        OTHERS = enum.auto()
        ALL = IMAGES | VIDEOS | OTHERS

    class Scalar:
        """Class for a Summary Scalar."""

        def __init__(self, reduce: Callable[[Sequence[float]], float] = np.mean):
            self.values = []
            self.reduce = reduce

        def __call__(self):
            return self.reduce(self.values)

    class Text:
        """Class for a Summary Text."""

        def __init__(self, text: str):
            self.text = text

    class Image:
        """Class for a Summary Image."""

        def __init__(self, shape: Tuple[int, int, int], image_bytes: bytes):
            self.shape = shape  # (C, H, W)
            self.image_bytes = image_bytes

    class Video:
        """Class for a Summary Video."""

        def __init__(self, shape: Tuple[int, int, int], image_bytes: bytes):
            self.shape = shape  # (C, H, W)
            self.image_bytes = image_bytes

    def from_metrics(self, metrics: Dict[str, torch.Tensor]):
        metrics = reduce_dict_mean(metrics)
        for k, v in metrics.items():
            v = to_numpy(v)
            if np.isnan(v):
                raise ValueError('NaN', k)
            self.scalar(k, float(v))

    def gif(self, tag: str, imgs: List[np.ndarray]):
        assert imgs
        try:
            height, width, _ = imgs[0].shape
            vid_save_path = '/tmp/video.gif'
            imageio.mimsave(vid_save_path, [np.array(img) for i, img in enumerate(imgs) if i % 2 == 0], fps=30)
            with open(vid_save_path, 'rb') as f:
                encoded_image_string = f.read()
            self[tag] = Summary.Video((3, height, width), encoded_image_string)
        except AttributeError:
            # the kitchen and hand manipulation envs do not support rendering.
            return

    def plot(self, tag: str, fig: matplotlib.figure.Figure):
        byte_data = io.BytesIO()
        fig.savefig(byte_data, format='png')
        img_w, img_h = fig.canvas.get_width_height()
        self[tag] = Summary.Image((4, img_h, img_w), byte_data.getvalue())

    def png(self, tag: str, img: Union[np.ndarray, torch.Tensor]):
        if img.ndim == 3:
            shape = (img.shape[2], *img.shape[:2])
        elif img.ndim == 5:
            shape = (img.shape[2], img.shape[0] * img.shape[3], img.shape[1] * img.shape[4])
        else:
            raise ValueError(f'Unsupported image shape {img.shape}')
        self[tag] = Summary.Image(shape, to_png(img))

    def scalar(self, tag: str, value: float, reduce: Callable[[Sequence[float]], float] = np.mean):
        if tag not in self:
            self[tag] = Summary.Scalar(reduce)
        self[tag].values.append(value)

    def text(self, tag: str, text: str):
        self[tag] = Summary.Text(text)

    def proto(self, mode: ProtoMode = ProtoMode.ALL):
        entries = []
        for tag, value in self.items():
            if isinstance(value, Summary.Scalar):
                if mode & self.ProtoMode.OTHERS:
                    entries.append(summary_pb2.Summary.Value(tag=tag, simple_value=value()))
            elif isinstance(value, Summary.Text):
                if mode & self.ProtoMode.OTHERS:
                    metadata = summary_pb2.SummaryMetadata(
                        plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name='text'))
                    entries.append(summary_pb2.Summary.Value(
                        tag=tag, metadata=metadata,
                        tensor=make_tensor_proto(values=value.text.encode('utf-8'), shape=(1,))))
            elif isinstance(value, (Summary.Image, Summary.Video)):
                if mode & (self.ProtoMode.IMAGES | self.ProtoMode.VIDEOS):
                    image_summary = summary_pb2.Summary.Image(
                        encoded_image_string=value.image_bytes,
                        colorspace=value.shape[0],  # RGBA
                        height=value.shape[1],
                        width=value.shape[2])
                    entries.append(summary_pb2.Summary.Value(tag=tag, image=image_summary))
            else:
                raise NotImplementedError(tag, value)
        return summary_pb2.Summary(value=entries)

    def to_dict(self) -> Dict[str, Any]:
        entries = {}
        for tag, value in self.items():
            if isinstance(value, Summary.Scalar):
                entries[tag] = float(value())
            elif isinstance(value, (Summary.Text, Summary.Image, Summary.Video)):
                pass
            else:
                raise NotImplementedError(tag, value)
        return entries

    def __str__(self) -> str:
        return '\n'.join(f'    {k:40s}: {v:.6f}' for k, v in self.to_dict().items())


class SummaryForgetter:
    """Used as placeholder for workers, it basically does nothing."""

    def __init__(self,
                 logdir: pathlib.Path,
                 queue_size: int = 5,
                 write_interval: int = 5):
        self.logdir = logdir

    def write(self, summary: Summary, step: int):
        pass

    def close(self):
        """Flushes the event file to disk and close the file."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Inspired from https://github.com/google/objax/blob/master/objax/jaxboard.py
class SummaryWriter:
    """Writes entries to logdir to be consumed by TensorBoard and Weight & Biases."""

    def __init__(self,
                 logdir: pathlib.Path,
                 queue_size: int = 5,
                 write_interval: int = 5):
        (logdir / 'tb').mkdir(exist_ok=True, parents=True)
        self.logdir = logdir
        self.writer = EventFileWriter(logdir / 'tb', queue_size, write_interval)
        self.writer_image = EventFileWriter(logdir / 'tb', queue_size, write_interval, filename_suffix='images')

    def write(self, summary: Summary, step: int):
        """Add on event to the event file."""
        self.writer.add_event(
            event_pb2.Event(step=step, summary=summary.proto(summary.ProtoMode.OTHERS),
                            wall_time=time()))
        self.writer_image.add_event(
            event_pb2.Event(step=step, summary=summary.proto(summary.ProtoMode.IMAGES),
                            wall_time=time()))

    def close(self):
        """Flushes the event file to disk and close the file."""
        self.writer.close()
        self.writer_image.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def create(cls, logdir: pathlib.Path,
               queue_size: int = 5,
               write_interval: int = 5) -> Union[SummaryForgetter, 'SummaryWriter']:
        if is_master():
            return cls(logdir, queue_size, write_interval)
        return SummaryForgetter(logdir, queue_size, write_interval)


def zip_batch_as_png(x: Union[np.ndarray, torch.Tensor], filename: pathlib.Path):
    if not is_master():
        return
    assert x.ndim == 4
    with zipfile.ZipFile(filename, 'w') as fzip:
        for i in range(x.shape[0]):
            with fzip.open(f'{i:06d}.png', 'w') as f:
                f.write(to_png(x[i]))
