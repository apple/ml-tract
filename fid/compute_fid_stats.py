#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import os
import pathlib

import lib
import numpy as np
import torch
from absl import app, flags
from lib.distributed import auto_distribute
from lib.util import FLAGS, artifact_dir

ML_DATA = pathlib.Path(os.getenv('ML_DATA'))

@auto_distribute
def main(argv):
    data = lib.data.DATASETS[FLAGS.dataset]()
    real = data.make_dataloaders()[1]
    num_samples = len(real) * FLAGS.batch
    with torch.no_grad():
        fid = lib.eval.FID(FLAGS.dataset, (3, data.res, data.res))
        real_activations = fid.data_activations(real, num_samples, cpu=True)
        m_real, s_real = fid.calculate_activation_statistics(real_activations)
    np.save(f'{ML_DATA}/{FLAGS.dataset}_activation_mean.npy', m_real.numpy())
    np.save(f'{ML_DATA}/{FLAGS.dataset}_activation_std.npy', s_real.numpy())


if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'cifar10', help='Training dataset.')
    app.run(lib.distributed.main(main))
