"""Tree-HyperLISTA: Ultra-lightweight deep unfolding for tree-sparse recovery.

Replaces block support selection with tree-consistent support projection
in a momentum-accelerated ISTA backbone. Only 3 tunable hyperparameters
(c1, c2, c3) controlling step size/threshold, momentum, and support size.
"""

import torch
import torch.nn as nn
import numpy as np
from src.utils.tree_proximal import (
    tree_scores_fast,
    topk_tree_projection,
    threshold_ancestor_closure,
    tree_soft_threshold,
)
from src.utils.sensing import compute_mutual_coherence


class TreeHyperLISTA(nn.Module):
    """
    Tree-HyperLISTA: ultra-lightweight unfolded solver for tree-sparse
    recovery. Only 3 hyperparameters (c1, c2, c3).

    Reparameterized for stability:
        c1 controls threshold via normalized residual ratio
        c2 controls momentum via sigmoid bounding
        c3 controls tree support size as fraction of n

    Support mechanisms:
        'tree_hard'      - Hard top-K tree projection (no shrinkage)
        'tree_threshold' - Threshold + ancestor closure + soft shrink
        'hybrid_tree'    - Top-K tree selection + soft shrinkage (recommended)
    """

    def __init__(self, A: np.ndarray, tree_info: dict, num_layers: int = 16,
                 c1: float = 1.0, c2: float = 0.0, c3: float = 3.0,
                 support_mode: str = 'hybrid_tree', rho: float = 0.5):
        super().__init__()
        m, n = A.shape
        self.n = n
        self.m = m
        self.num_layers = num_layers
        self.support_mode = support_mode
        self.rho = rho
        self.num_hyperparams = 3

        self.parent = tree_info['parent']
        self.depth_arr = tree_info['depth']
        self.tree_n = tree_info['n']

        self.register_buffer('A', torch.from_numpy(A).float())

        W = self._compute_symmetric_W(A)
        self.register_buffer('W', torch.from_numpy(W).float())

        A_pinv = np.linalg.pinv(A).astype(np.float32)
        self.register_buffer('A_pinv', torch.from_numpy(A_pinv).float())

        self.mu = compute_mutual_coherence(A)

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

    def _count_active_nodes(self, x: torch.Tensor) -> torch.Tensor:
        return (x.abs() > 1e-6).float().sum(dim=-1, keepdim=True)

    def _compute_adaptive_params(self, x: torch.Tensor, y: torch.Tensor, k: int):
        """
        Reparameterized adaptive parameters -- all bounded and scale-invariant.

        theta = c1 * residual_ratio        -- threshold proportional to normalized residual
        beta  = sigmoid(c2) * active_frac   -- bounded momentum in [0, 1)
        K_tree = c3 * n * max(signal_est, layer_progress) -- tree support size
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

        active_nodes = self._count_active_nodes(x)
        active_fraction = active_nodes / float(self.n)
        beta = torch.sigmoid(c2) * active_fraction

        layer_progress = float(k + 1) / float(self.num_layers)
        ratio = (y_l1 / res_l1).clamp(min=1.0)
        log_ratio = torch.log(ratio).clamp(min=0.0)
        signal_estimate = torch.sigmoid(log_ratio - 1.0)
        K_tree = (c3 * float(self.n) *
                  torch.maximum(signal_estimate,
                                torch.full_like(signal_estimate, layer_progress))
                  ).clamp(min=1.0, max=float(self.tree_n * 0.6))

        return theta, beta, K_tree

    def _apply_tree_operator(self, u: torch.Tensor, theta_scalar: float,
                             K_int: int) -> torch.Tensor:
        """Apply tree-consistent structured operator T_{theta, K}(u)."""
        if self.support_mode == 'tree_hard':
            scores = tree_scores_fast(u, self.parent, self.depth_arr, self.rho)
            projected, _ = topk_tree_projection(u, scores, K_int, self.parent)
            return projected

        elif self.support_mode == 'tree_threshold':
            scores = tree_scores_fast(u, self.parent, self.depth_arr, self.rho)
            _, support_mask = threshold_ancestor_closure(
                u, scores, theta_scalar, self.parent)
            return tree_soft_threshold(u, theta_scalar, support_mask)

        elif self.support_mode == 'hybrid_tree':
            scores = tree_scores_fast(u, self.parent, self.depth_arr, self.rho)
            _, support_mask = topk_tree_projection(u, scores, K_int, self.parent)
            return tree_soft_threshold(u, theta_scalar, support_mask)

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
            theta, beta, K_tree = self._compute_adaptive_params(x, y, k)

            if k > 0:
                z = x + beta * (x - x_prev)
            else:
                z = x

            residual = y - z @ self.A.t()
            u = z + (self.W.t() @ residual.t()).t()

            theta_scalar = theta.mean().item()
            K_int = max(1, int(K_tree.mean().item()))

            x_new = self._apply_tree_operator(u, theta_scalar, K_int)

            x_prev = x
            x = x_new

            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x
