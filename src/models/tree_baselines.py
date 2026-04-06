"""Baseline models for tree-sparse recovery: Tree-ISTA, Tree-FISTA, Tree-LISTA."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from src.utils.tree_proximal import (
    tree_scores_fast,
    topk_tree_projection,
    tree_soft_threshold,
    hybrid_tree_threshold,
)


class TreeISTA:
    """Tree-ISTA: classical ISTA with tree-aware proximal operator."""

    def __init__(self, A: np.ndarray, tree_info: dict, lam: float = 0.1,
                 max_iter: int = 16, rho: float = 0.5, target_K: int = 30):
        self.A = torch.from_numpy(A).float()
        self.tree_info = tree_info
        self.parent = tree_info['parent']
        self.depth_arr = tree_info['depth']
        self.n = A.shape[1]
        self.m = A.shape[0]
        self.lam = lam
        self.max_iter = max_iter
        self.rho = rho
        self.target_K = target_K
        self.num_layers = max_iter

        eigvals = np.linalg.eigvalsh(A.T @ A)
        self.step_size = 1.0 / max(eigvals.max(), 1e-8)

    def to(self, device):
        self.A = self.A.to(device)
        return self

    def eval(self):
        return self

    def solve(self, y: torch.Tensor, return_trajectory: bool = False,
              num_iter: int = None) -> torch.Tensor:
        K = num_iter if num_iter is not None else self.max_iter
        device = y.device
        self.A = self.A.to(device)
        batch_size = y.shape[0]

        x = torch.zeros(batch_size, self.n, device=device)
        trajectory = [x.clone()] if return_trajectory else None

        for k in range(K):
            residual = y - x @ self.A.t()
            grad = -(self.A.t() @ residual.t()).t()
            u = x - self.step_size * grad

            x = hybrid_tree_threshold(u, self.lam, self.target_K,
                                      self.parent, self.depth_arr, self.rho)

            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x


class TreeFISTA:
    """Tree-FISTA: accelerated FISTA with tree-aware proximal operator."""

    def __init__(self, A: np.ndarray, tree_info: dict, lam: float = 0.1,
                 max_iter: int = 16, rho: float = 0.5, target_K: int = 30):
        self.A = torch.from_numpy(A).float()
        self.tree_info = tree_info
        self.parent = tree_info['parent']
        self.depth_arr = tree_info['depth']
        self.n = A.shape[1]
        self.m = A.shape[0]
        self.lam = lam
        self.max_iter = max_iter
        self.rho = rho
        self.target_K = target_K
        self.num_layers = max_iter

        eigvals = np.linalg.eigvalsh(A.T @ A)
        self.step_size = 1.0 / max(eigvals.max(), 1e-8)

    def to(self, device):
        self.A = self.A.to(device)
        return self

    def eval(self):
        return self

    def solve(self, y: torch.Tensor, return_trajectory: bool = False,
              num_iter: int = None) -> torch.Tensor:
        K = num_iter if num_iter is not None else self.max_iter
        device = y.device
        self.A = self.A.to(device)
        batch_size = y.shape[0]

        x = torch.zeros(batch_size, self.n, device=device)
        x_prev = torch.zeros_like(x)
        t = 1.0
        trajectory = [x.clone()] if return_trajectory else None

        for k in range(K):
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = (t - 1.0) / t_new
            z = x + momentum * (x - x_prev)

            residual = y - z @ self.A.t()
            grad = -(self.A.t() @ residual.t()).t()
            u = z - self.step_size * grad

            x_new = hybrid_tree_threshold(u, self.lam, self.target_K,
                                          self.parent, self.depth_arr, self.rho)

            x_prev = x
            x = x_new
            t = t_new

            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x


class TreeLISTA(nn.Module):
    """Tree-LISTA: learned ISTA with tree-aware proximal operator.

    Learns per-layer weight matrices and thresholds, but uses tree
    structure for support selection instead of elementwise thresholding.
    """

    def __init__(self, A: np.ndarray, tree_info: dict, num_layers: int = 16,
                 rho: float = 0.5, target_K: int = 30):
        super().__init__()
        m, n = A.shape
        self.n = n
        self.m = m
        self.num_layers = num_layers
        self.rho = rho
        self.target_K = target_K
        self.parent = tree_info['parent']
        self.depth_arr = tree_info['depth']

        self.register_buffer('A', torch.from_numpy(A).float())

        self.W1 = nn.ParameterList([
            nn.Parameter(torch.randn(n, m) * 0.01) for _ in range(num_layers)
        ])
        self.W2 = nn.ParameterList([
            nn.Parameter(torch.eye(n) + torch.randn(n, n) * 0.01)
            for _ in range(num_layers)
        ])
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_layers)
        ])

    def forward(self, y: torch.Tensor, return_trajectory: bool = False,
                num_layers: int = None) -> torch.Tensor:
        K = num_layers if num_layers is not None else self.num_layers
        device = y.device
        batch_size = y.shape[0]

        x = torch.zeros(batch_size, self.n, device=device)
        trajectory = [x.clone()] if return_trajectory else None

        for k in range(K):
            u = (self.W1[k] @ y.t()).t() + (self.W2[k] @ x.t()).t()
            theta = torch.abs(self.thresholds[k]).item()

            scores = tree_scores_fast(u, self.parent, self.depth_arr, self.rho)
            _, support_mask = topk_tree_projection(u, scores, self.target_K, self.parent)
            x = tree_soft_threshold(u, theta, support_mask)

            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x
