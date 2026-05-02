"""Elementwise ISTA baseline for tree-sparse recovery comparison."""

from typing import Optional

import torch
import numpy as np
from src.utils.proximal import soft_threshold


class ElementwiseISTA:
    """Standard ISTA with elementwise soft-thresholding (for comparison)."""

    def __init__(self, A: np.ndarray, lam: float = 0.1, max_iter: int = 1000):
        self.A = torch.from_numpy(A).float()
        self.lam = lam
        self.max_iter = max_iter
        eigvals = np.linalg.eigvalsh(A.T @ A)
        self.L = float(eigvals.max())
        self.step_size = 1.0 / self.L
        self.num_hyperparams = 0

    @torch.no_grad()
    def solve(self, y: torch.Tensor, return_trajectory: bool = False,
              num_iter: Optional[int] = None) -> torch.Tensor:
        device = y.device
        A = self.A.to(device)
        batch_size = y.shape[0]
        n = A.shape[1]
        x = torch.zeros(batch_size, n, device=device)
        trajectory = [x.clone()] if return_trajectory else None
        iters = num_iter if num_iter is not None else self.max_iter

        for k in range(iters):
            residual = y - x @ A.t()
            grad = -residual @ A
            u = x - self.step_size * grad
            x_new = soft_threshold(u, self.lam * self.step_size)
            if return_trajectory:
                trajectory.append(x_new.clone())
            x = x_new

        if return_trajectory:
            return trajectory
        return x

    def parameters(self):
        return iter([])
