"""HyperLISTA: Hyperparameter Tuning is All You Need (Chen et al., NeurIPS 2021)."""

import torch
import torch.nn as nn
import numpy as np
from src.utils.proximal import soft_threshold_with_support
from src.utils.sensing import compute_mutual_coherence


class HyperLISTA(nn.Module):
    """
    HyperLISTA with only 3 hyperparameters (c1, c2, c3).
    Reparameterized for stability: all adaptive quantities are bounded.
    """

    def __init__(self, A: np.ndarray, num_layers: int = 16,
                 c1: float = 1.0, c2: float = 0.0, c3: float = 3.0):
        super().__init__()
        m, n = A.shape
        self.n = n
        self.m = m
        self.num_layers = num_layers
        self.num_hyperparams = 3

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
        for iteration in range(200):
            gram = D.T @ D
            grad = 2.0 * D @ (gram - np.eye(n))
            D = D - 0.002 * grad
            norms = np.linalg.norm(D, axis=0, keepdims=True)
            D = D / np.maximum(norms, 1e-12)
        G = np.linalg.lstsq(A, D, rcond=None)[0]
        W = A @ G
        return W.astype(np.float32)

    def _compute_adaptive_params(self, x: torch.Tensor, x_prev: torch.Tensor,
                                 y: torch.Tensor, k: int):
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

        x_l0 = (x.abs() > 1e-6).float().sum(dim=-1, keepdim=True)
        sparsity_fraction = x_l0 / float(self.n)
        beta = torch.sigmoid(c2) * sparsity_fraction

        ratio = (y_l1 / res_l1).clamp(min=1.0)
        log_ratio = torch.log(ratio).clamp(min=0.0)
        p = (c3 * log_ratio).clamp(max=float(self.n))

        return theta, beta, p

    def forward(self, y: torch.Tensor, return_trajectory: bool = False,
                num_layers: int = None) -> torch.Tensor:
        K = num_layers if num_layers is not None else self.num_layers
        device = y.device
        batch_size = y.shape[0]

        x = torch.zeros(batch_size, self.n, device=device)
        x_prev = torch.zeros_like(x)
        trajectory = [x.clone()] if return_trajectory else None

        for k in range(K):
            theta, beta, p = self._compute_adaptive_params(x, x_prev, y, k)

            if k > 0:
                z = x + beta * (x - x_prev)
            else:
                z = x

            residual = y - z @ self.A.t()
            u = z + (self.W.t() @ residual.t()).t()

            p_int = int(p.mean().item())
            theta_scalar = theta.mean().item()
            x_new = soft_threshold_with_support(u, theta_scalar, p_int)

            x_prev = x
            x = x_new

            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x
