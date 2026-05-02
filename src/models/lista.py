"""LISTA: Learned ISTA (Gregor & LeCun, 2010) — elementwise baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.proximal import soft_threshold


class LISTA(nn.Module):
    """Standard LISTA unfolded network with elementwise thresholding."""

    def __init__(self, A: np.ndarray, num_layers: int = 16):
        super().__init__()
        m, n = A.shape
        self.n = n
        self.m = m
        self.num_layers = num_layers

        A_t = torch.from_numpy(A).float()
        L = float(torch.linalg.eigvalsh(A_t.t() @ A_t).max())
        S = torch.eye(n) - (1.0 / L) * A_t.t() @ A_t
        W = (1.0 / L) * A_t.t()

        self.W1 = nn.ParameterList()
        self.W2 = nn.ParameterList()
        self.thresholds = nn.ParameterList()

        for k in range(num_layers):
            self.W1.append(nn.Parameter(S.clone()))
            self.W2.append(nn.Parameter(W.clone()))
            self.thresholds.append(nn.Parameter(torch.ones(1) * 0.01))

    def forward(self, y: torch.Tensor, return_trajectory: bool = False,
                num_layers: int = None) -> torch.Tensor:
        K = num_layers if num_layers is not None else self.num_layers
        x = torch.zeros(y.shape[0], self.n, device=y.device)
        trajectory = [x.clone()] if return_trajectory else None

        for k in range(K):
            x = soft_threshold(
                F.linear(x, self.W1[k]) + F.linear(y, self.W2[k]),
                torch.abs(self.thresholds[k])
            )
            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x

    @property
    def num_hyperparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
