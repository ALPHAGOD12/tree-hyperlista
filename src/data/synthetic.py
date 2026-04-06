"""Synthetic block-sparse data generation with mismatch regimes."""

import numpy as np
import torch
from typing import Optional, Dict, Any

from src.utils.sensing import get_sensing_matrix, perturbed_sensing


class BlockSparseDataset:
    """Generate block-sparse recovery problems y = Ax + epsilon."""

    def __init__(self, n: int = 500, m: int = 250, group_size: int = 10,
                 num_active_groups: int = 5, snr_db: float = 30.0,
                 matrix_type: str = 'gaussian', amplitude_dist: str = 'gaussian',
                 amplitude_range: tuple = (0.5, 2.0), seed: int = 42,
                 matrix_seed: int = 42, **matrix_kwargs):
        self.n = n
        self.m = m
        self.group_size = group_size
        self.num_groups = n // group_size
        self.num_active_groups = num_active_groups
        self.snr_db = snr_db
        self.amplitude_dist = amplitude_dist
        self.amplitude_range = amplitude_range
        self.seed = seed
        self.matrix_type = matrix_type

        self.A = get_sensing_matrix(m, n, matrix_type=matrix_type,
                                    seed=matrix_seed, **matrix_kwargs)
        self.A_tensor = torch.from_numpy(self.A).float()
        self.A_pinv = np.linalg.pinv(self.A).astype(np.float32)
        self.A_pinv_tensor = torch.from_numpy(self.A_pinv).float()

    def generate(self, num_samples: int, seed: Optional[int] = None,
                 num_active_groups: Optional[int] = None,
                 group_size: Optional[int] = None,
                 snr_db: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Generate a batch of block-sparse problems.

        Optional overrides allow mismatch testing.
        """
        rng = np.random.RandomState(seed if seed is not None else self.seed)
        sg = num_active_groups if num_active_groups is not None else self.num_active_groups
        gs = group_size if group_size is not None else self.group_size
        snr = snr_db if snr_db is not None else self.snr_db
        num_groups = self.n // gs

        X = np.zeros((num_samples, self.n), dtype=np.float32)
        for i in range(num_samples):
            active = rng.choice(num_groups, size=min(sg, num_groups), replace=False)
            for g in active:
                start = g * gs
                end = min(start + gs, self.n)
                if self.amplitude_dist == 'gaussian':
                    vals = rng.randn(end - start).astype(np.float32)
                    signs = np.sign(vals)
                    mags = np.abs(vals)
                    lo, hi = self.amplitude_range
                    mags = mags * (hi - lo) + lo
                    X[i, start:end] = signs * mags
                elif self.amplitude_dist == 'uniform':
                    lo, hi = self.amplitude_range
                    X[i, start:end] = rng.uniform(lo, hi, size=end - start) * \
                                      rng.choice([-1, 1], size=end - start)
                else:
                    X[i, start:end] = rng.randn(end - start).astype(np.float32)

        Y_clean = X @ self.A.T  # (num_samples, m)

        sigma = np.sqrt(np.mean(Y_clean ** 2) * 10 ** (-snr / 10.0))
        noise = rng.randn(num_samples, self.m).astype(np.float32) * sigma
        Y = Y_clean + noise

        return {
            'x': torch.from_numpy(X),
            'y': torch.from_numpy(Y.astype(np.float32)),
            'noise_std': sigma,
        }

    def generate_with_perturbed_A(self, num_samples: int, delta: float,
                                  seed: int = 99) -> Dict[str, torch.Tensor]:
        """Generate data using a perturbed sensing matrix (operator mismatch)."""
        A_pert = perturbed_sensing(self.A, delta, seed=seed)
        rng = np.random.RandomState(seed)
        sg = self.num_active_groups
        gs = self.group_size
        num_groups = self.n // gs

        X = np.zeros((num_samples, self.n), dtype=np.float32)
        for i in range(num_samples):
            active = rng.choice(num_groups, size=sg, replace=False)
            for g in active:
                start = g * gs
                end = start + gs
                vals = rng.randn(gs).astype(np.float32)
                signs = np.sign(vals)
                mags = np.abs(vals) * 1.5 + 0.5
                X[i, start:end] = signs * mags

        Y_clean = X @ A_pert.T
        snr = self.snr_db
        sigma = np.sqrt(np.mean(Y_clean ** 2) * 10 ** (-snr / 10.0))
        noise = rng.randn(num_samples, self.m).astype(np.float32) * sigma
        Y = Y_clean + noise

        return {
            'x': torch.from_numpy(X),
            'y': torch.from_numpy(Y.astype(np.float32)),
            'noise_std': sigma,
            'A_perturbed': torch.from_numpy(A_pert.astype(np.float32)),
        }


def get_default_config() -> Dict[str, Any]:
    """Default experimental configuration."""
    return {
        'n': 500,
        'm': 250,
        'group_size': 10,
        'num_active_groups': 5,
        'snr_db': 30.0,
        'matrix_type': 'gaussian',
        'amplitude_dist': 'gaussian',
        'amplitude_range': (0.5, 2.0),
        'num_train': 10000,
        'num_val': 2000,
        'num_test': 5000,
        'num_layers': 16,
        'seed': 42,
    }


def get_mismatch_configs() -> Dict[str, list]:
    """Mismatch sweep configurations."""
    return {
        'snr_db': [15.0, 20.0, 25.0, 35.0, 40.0],
        'num_active_groups': [3, 4, 6, 7, 8],
        'group_size': [5, 8, 12, 15],
        'operator_delta': [0.05, 0.1, 0.15, 0.2, 0.3],
    }
