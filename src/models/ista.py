"""ISTA and FISTA with group-LASSO proximal operators."""

from typing import Optional

import torch
import numpy as np
from src.utils.proximal import block_soft_threshold, soft_threshold


class GroupISTA:
    """Iterative Shrinkage-Thresholding for group-LASSO."""

    def __init__(self, A: np.ndarray, group_size: int, lam: float = 0.1,
                 max_iter: int = 1000, tol: float = 1e-7):
        self.A_np = A
        self.A = torch.from_numpy(A).float()
        self.AtA = self.A.t() @ self.A
        self.group_size = group_size
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
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
            x_new = block_soft_threshold(u, self.lam * self.step_size,
                                         self.group_size)

            if return_trajectory:
                trajectory.append(x_new.clone())

            if not return_trajectory and (x_new - x).norm() < self.tol * max(x.norm(), 1.0):
                x = x_new
                break
            x = x_new

        if return_trajectory:
            return trajectory
        return x

    def parameters(self):
        return iter([])


class GroupFISTA:
    """Fast ISTA (FISTA) with group-LASSO proximal operators."""

    def __init__(self, A: np.ndarray, group_size: int, lam: float = 0.1,
                 max_iter: int = 1000, tol: float = 1e-7):
        self.A_np = A
        self.A = torch.from_numpy(A).float()
        self.group_size = group_size
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
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
        x_prev = x.clone()
        t = 1.0

        trajectory = [x.clone()] if return_trajectory else None
        iters = num_iter if num_iter is not None else self.max_iter

        for k in range(iters):
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = (t - 1.0) / t_new

            z = x + momentum * (x - x_prev)
            residual = y - z @ A.t()
            grad = -residual @ A
            u = z - self.step_size * grad
            x_new = block_soft_threshold(u, self.lam * self.step_size,
                                         self.group_size)

            if return_trajectory:
                trajectory.append(x_new.clone())

            if not return_trajectory and (x_new - x).norm() < self.tol * max(x.norm(), 1.0):
                x_prev = x
                x = x_new
                break

            x_prev = x
            x = x_new
            t = t_new

        if return_trajectory:
            return trajectory
        return x

    def parameters(self):
        return iter([])


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
