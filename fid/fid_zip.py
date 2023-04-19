#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
"""Compute FID and approximation at 50,000 for zip file of samples."""
import time
import zipfile

import lib
import torch
import torchvision.transforms.functional
from absl import app, flags
from lib.distributed import auto_distribute, device_id, is_master, world_size
from lib.util import FLAGS
from PIL import Image


@auto_distribute
def main(argv):
    def zip_iterator(filename: str, batch: int):
        with zipfile.ZipFile(filename, 'r') as fzip:
            x = []
            fn_list = [fn for fn in fzip.namelist() if fn.endswith('.png')]
            assert len(fn_list) >= FLAGS.fid_len
            for fn in fn_list[device_id()::world_size()]:
                with fzip.open(fn, 'r') as f:
                    y = torchvision.transforms.functional.to_tensor(Image.open(f))
                x.append(2 * y - 1)
                if len(x) == batch:
                    yield torch.stack(x), None
                    x = []

    t0 = time.time()
    data = lib.data.DATASETS[FLAGS.dataset]()
    fake = (x for x in zip_iterator(argv[1], FLAGS.batch // world_size()))
    with torch.no_grad():
        fid = lib.eval.FID(FLAGS.dataset, (3, data.res, data.res))
        fake_activations = fid.data_activations(fake, FLAGS.fid_len)
        fid, fid50 = fid.approximate_fid(fake_activations)
    if is_master():
        print(f'dataset={FLAGS.dataset}')
        print(f'fid{FLAGS.fid_len}={fid}')
        print(f'fid(50000)={fid50}')
        print(f'time={time.time() - t0}')


if __name__ == '__main__':
    flags.DEFINE_integer('fid_len', 4096, help='Number of samples for FID evaluation.')
    flags.DEFINE_string('dataset', 'cifar10', help='Training dataset.')
    app.run(lib.distributed.main(main))
