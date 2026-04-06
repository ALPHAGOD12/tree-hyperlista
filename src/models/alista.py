"""ALISTA: Analytic LISTA (Liu et al., ICLR 2019)."""

import torch
import torch.nn as nn
import numpy as np
from src.utils.proximal import soft_threshold_with_support
from src.utils.sensing import compute_analytic_W, compute_mutual_coherence


class ALISTA(nn.Module):
    """ALISTA with analytic weight matrix and learned step sizes / thresholds."""

    def __init__(self, A: np.ndarray, num_layers: int = 16,
                 support_p: int = 5, support_pmax: int = None):
        super().__init__()
        m, n = A.shape
        self.n = n
        self.m = m
        self.num_layers = num_layers
        self.support_p_inc = support_p
        self.support_pmax = support_pmax if support_pmax is not None else n

        self.register_buffer('A', torch.from_numpy(A).float())

        W = self._compute_analytic_W(A)
        self.register_buffer('W', torch.from_numpy(W).float())

        self.mu = compute_mutual_coherence(A)

        self.step_sizes = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)
        ])
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_layers)
        ])

    def _compute_analytic_W(self, A: np.ndarray) -> np.ndarray:
        """Solve min ||W^T A||_F^2 s.t. diag(W^T A) = 1."""
        m, n = A.shape
        AtA = A.T @ A
        D = AtA.copy()
        np.fill_diagonal(D, 0)
        W = np.linalg.lstsq(AtA, np.eye(n), rcond=None)[0]
        W = A @ W
        return W.astype(np.float32)

    def forward(self, y: torch.Tensor, return_trajectory: bool = False,
                num_layers: int = None) -> torch.Tensor:
        K = num_layers if num_layers is not None else self.num_layers
        device = y.device
        A = self.A
        W = self.W
        batch_size = y.shape[0]

        x = torch.zeros(batch_size, self.n, device=device)
        trajectory = [x.clone()] if return_trajectory else None

        for k in range(K):
            gamma = torch.abs(self.step_sizes[k])
            theta = torch.abs(self.thresholds[k])
            p_k = min(self.support_p_inc * (k + 1), self.support_pmax)

            residual = y - x @ A.t()
            grad_step = x + gamma * (W.t() @ residual.t()).t()

            x = soft_threshold_with_support(grad_step, theta.item(), int(p_k))

            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x

    @property
    def num_hyperparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
