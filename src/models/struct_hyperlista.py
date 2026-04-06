"""Struct-HyperLISTA: HyperLISTA-style unfolding for block-sparse recovery."""

import torch
import torch.nn as nn
import numpy as np
from src.utils.proximal import (
    block_soft_threshold,
    topk_group_support,
    hybrid_group_threshold,
)
from src.utils.sensing import compute_mutual_coherence, compute_block_coherence


class StructHyperLISTA(nn.Module):
    """
    Struct-HyperLISTA: ultra-lightweight unfolded solver for structured
    (block) sparse recovery. Only 3 hyperparameters (c1, c2, c3).

    Reparameterized for stability:
        c1 controls threshold via normalized residual ratio
        c2 controls momentum via sigmoid bounding
        c3 controls group support count as fraction of total groups

    Support mechanisms:
        'block_soft'  - Block soft-thresholding with adaptive threshold
        'topk_group'  - Top-k group support selection
        'hybrid'      - Top-k selection + block shrinkage
    """

    def __init__(self, A: np.ndarray, group_size: int, num_layers: int = 16,
                 c1: float = 1.0, c2: float = 0.0, c3: float = 3.0,
                 support_mode: str = 'hybrid'):
        super().__init__()
        m, n = A.shape
        self.n = n
        self.m = m
        self.group_size = group_size
        self.num_groups = n // group_size
        self.num_layers = num_layers
        self.support_mode = support_mode
        self.num_hyperparams = 3

        self.register_buffer('A', torch.from_numpy(A).float())

        W = self._compute_symmetric_W(A)
        self.register_buffer('W', torch.from_numpy(W).float())

        A_pinv = np.linalg.pinv(A).astype(np.float32)
        self.register_buffer('A_pinv', torch.from_numpy(A_pinv).float())

        self.mu = compute_mutual_coherence(A)
        self.mu_block = compute_block_coherence(A, group_size)

        self.c1 = nn.Parameter(torch.tensor(c1))
        self.c2 = nn.Parameter(torch.tensor(c2))
        self.c3 = nn.Parameter(torch.tensor(c3))

    def _compute_symmetric_W(self, A: np.ndarray) -> np.ndarray:
        m, n = A.shape
        D = A / np.linalg.norm(A, axis=0, keepdims=True)
        for _ in range(200):
            gram = D.T @ D
            grad = 2.0 * D @ (gram - np.eye(n))
            D = D - 0.002 * grad
            norms = np.linalg.norm(D, axis=0, keepdims=True)
            D = D / np.maximum(norms, 1e-12)
        G = np.linalg.lstsq(A, D, rcond=None)[0]
        W = A @ G
        return W.astype(np.float32)

    def _count_active_groups(self, x: torch.Tensor) -> torch.Tensor:
        bs, n = x.shape
        gs = self.group_size
        ng = self.num_groups
        x_g = x[:, :ng * gs].reshape(bs, ng, gs)
        group_norms = x_g.norm(dim=-1)
        return (group_norms > 1e-6).float().sum(dim=-1, keepdim=True)

    def _compute_adaptive_params(self, x: torch.Tensor, y: torch.Tensor, k: int):
        """
        Reparameterized adaptive parameters -- all bounded and scale-invariant.

        theta = c1 * (residual_ratio)  -- threshold proportional to normalized residual
        beta  = sigmoid(c2) * (group_fraction) -- bounded momentum in [0, 1)
        p_group = c3 * layer_progress -- support count grows with layers
        """
        c1 = torch.abs(self.c1)
        c2 = self.c2
        c3 = torch.abs(self.c3)

        residual = y - x @ self.A.t()
        Apinv_res = (self.A_pinv @ residual.t()).t()
        res_l1 = Apinv_res.abs().sum(dim=-1, keepdim=True).clamp(min=1e-12)

        Apinv_y = (self.A_pinv @ y.t()).t()
        y_l1 = Apinv_y.abs().sum(dim=-1, keepdim=True).clamp(min=1e-12)

        residual_ratio = (res_l1 / y_l1).clamp(min=0.0, max=1.0)
        theta = c1 * residual_ratio

        active_groups = self._count_active_groups(x)
        group_fraction = active_groups / float(self.num_groups)
        beta = torch.sigmoid(c2) * group_fraction

        layer_progress = float(k + 1) / float(self.num_layers)
        ratio = (y_l1 / res_l1).clamp(min=1.0)
        log_ratio = torch.log(ratio).clamp(min=0.0)
        signal_estimate = torch.sigmoid(log_ratio - 1.0)
        p_group = (c3 * float(self.num_groups) *
                   torch.maximum(signal_estimate,
                                 torch.full_like(signal_estimate, layer_progress))
                   ).clamp(min=1.0, max=float(self.num_groups))

        return theta, beta, p_group

    def _apply_support_mechanism(self, u: torch.Tensor, theta: float,
                                 k_groups: int) -> torch.Tensor:
        if self.support_mode == 'block_soft':
            return block_soft_threshold(u, theta, self.group_size)
        elif self.support_mode == 'topk_group':
            return topk_group_support(u, k_groups, self.group_size)
        elif self.support_mode == 'hybrid':
            return hybrid_group_threshold(u, theta, k_groups, self.group_size)
        else:
            raise ValueError(f"Unknown support mode: {self.support_mode}")

    def forward(self, y: torch.Tensor, return_trajectory: bool = False,
                num_layers: int = None) -> torch.Tensor:
        K = num_layers if num_layers is not None else self.num_layers
        device = y.device
        batch_size = y.shape[0]

        x = torch.zeros(batch_size, self.n, device=device)
        x_prev = torch.zeros_like(x)
        trajectory = [x.clone()] if return_trajectory else None

        for k in range(K):
            theta, beta, p_group = self._compute_adaptive_params(x, y, k)

            if k > 0:
                z = x + beta * (x - x_prev)
            else:
                z = x

            residual = y - z @ self.A.t()
            u = z + (self.W.t() @ residual.t()).t()

            theta_scalar = theta.mean().item()
            k_groups = max(1, int(p_group.mean().item()))

            x_new = self._apply_support_mechanism(u, theta_scalar, k_groups)

            x_prev = x
            x = x_new

            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x
