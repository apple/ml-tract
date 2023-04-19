#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import pathlib
from typing import Iterable, Tuple

import numpy as np
import scipy
import torch
import torch.nn.functional
from lib.distributed import (barrier, device_id, gather_tensor, is_master,
                             trange, world_size)
from lib.util import FLAGS, to_numpy

from .inception_net import InceptionV3

ML_DATA = pathlib.Path(os.getenv('ML_DATA'))


class FID:
    def __init__(self, dataset: str, shape: Tuple[int, int, int], dims: int = 2048):
        assert dataset in ('cifar10', 'imagenet64')
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.dims = dims
        self.shape = shape
        self.model = InceptionV3([block_idx]).eval().to(device_id())
        self.post = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        if pathlib.Path(f'{ML_DATA}/{dataset}_activation_mean.npy').exists():
            self.real_activations_mean = torch.from_numpy(np.load(f'{ML_DATA}/{dataset}_activation_mean.npy'))
            self.real_activations_std = torch.from_numpy(np.load(f'{ML_DATA}/{dataset}_activation_std.npy'))

    def generate_activations_and_samples(self, model: torch.nn.Module, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        barrier()
        samples = torch.empty((n, *self.shape))
        activations = torch.empty((n, self.dims), dtype=torch.double).to(device_id())
        k = world_size()
        assert FLAGS.batch % k == 0
        for i in trange(0, n, FLAGS.batch, desc='Generating FID samples'):
            p = min(n - i, FLAGS.batch)
            x = model(FLAGS.batch // k).float()
            # Discretize to {0,...,255} and project back to [-1,1]
            x = torch.round(127.5 * (x + 1)).clamp(0, 255) / 127.5 - 1
            y = self.post(self.model(x)[0])
            samples[i: i + p] = gather_tensor(x)[:p]
            activations[i: i + p] = gather_tensor(y)[:p]
        return activations, samples

    def data_activations(self, iterator: Iterable, n: int, cpu: bool = False) -> torch.Tensor:
        activations = torch.empty((n, self.dims), dtype=torch.double)
        if not cpu:
            activations = activations.to(device_id())
        k = world_size()
        it = iter(iterator)
        for i in trange(0, n, FLAGS.batch, desc='Calculating activations'):
            x = next(it)[0]
            p = min((n - i) // k, x.shape[0])
            y = self.post(self.model(x.to(device_id()))[0])
            activations[i: i + k * p] = gather_tensor(y[:p]).cpu() if cpu else gather_tensor(y[:p])
        return activations

    @staticmethod
    def calculate_activation_statistics(activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return activations.mean(0), torch.cov(activations.T)

    def calculate_fid(self, fake_activations: torch.Tensor) -> float:
        m_fake, s_fake = self.calculate_activation_statistics(fake_activations)
        m_real = self.real_activations_mean.to(m_fake)
        s_real = self.real_activations_std.to(s_fake)
        return self.calculate_frechet_distance(m_fake, s_fake, m_real, s_real)

    def approximate_fid(self, fake_activations: torch.Tensor, n: int = 50_000) -> Tuple[float, float]:
        k = fake_activations.shape[0]
        fid = self.calculate_fid(fake_activations)
        fid_half = []
        for it in range(5):
            sel_fake = np.random.choice(k, k // 2, replace=False)
            fid_half.append(self.calculate_fid(fake_activations[sel_fake]))
        fid_half = np.median(fid_half)
        return fid, fid + (fid_half - fid) * (k / n - 1)

    def calculate_frechet_distance(self, mu1: torch.Tensor, sigma1: torch.Tensor,
                                   mu2: torch.Tensor, sigma2: torch.Tensor, eps: float = 1e-6) -> float:
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        if not is_master():
            return 0
        mu1, mu2, sigma1, sigma2 = (to_numpy(x) for x in (mu1, mu2, sigma1, sigma2))
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
        diff = mu1 - mu2

        # Product might be almost singular
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
        if not np.isfinite(covmean).all():
            print(f'fid calculation produces singular product;  adding {eps} to diagonal of cov estimates')
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
