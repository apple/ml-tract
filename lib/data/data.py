#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

__all__ = ['Dataset']

from typing import Callable

from absl import flags

flags.DEFINE_integer('batch', 256, help='Batch size.')


class Dataset:
    def __init__(self, res: int, make_train: Callable, make_fid: Callable):
        self.res = res
        self.make_train = make_train
        self.make_fid = make_fid
