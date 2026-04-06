"""LISTA: Learned ISTA (Gregor & LeCun, 2010)."""

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


class BlockLISTA(nn.Module):
    """Block-aware LISTA with group thresholding (Ada-BlockLISTA style)."""

    def __init__(self, A: np.ndarray, group_size: int, num_layers: int = 16):
        super().__init__()
        m, n = A.shape
        self.n = n
        self.m = m
        self.num_layers = num_layers
        self.group_size = group_size
        self.num_groups = n // group_size

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
            self.thresholds.append(nn.Parameter(torch.ones(self.num_groups) * 0.01))

    def _block_threshold(self, x: torch.Tensor, thresholds: torch.Tensor):
        bs, n = x.shape
        gs = self.group_size
        ng = self.num_groups
        x_g = x[:, :ng * gs].reshape(bs, ng, gs)
        norms = x_g.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        th = torch.abs(thresholds).unsqueeze(0).unsqueeze(-1)
        shrink = torch.clamp(1.0 - th / norms, min=0.0)
        out = (shrink * x_g).reshape(bs, ng * gs)
        if n > ng * gs:
            out = torch.cat([out, x[:, ng * gs:]], dim=-1)
        return out

    def forward(self, y: torch.Tensor, return_trajectory: bool = False,
                num_layers: int = None) -> torch.Tensor:
        K = num_layers if num_layers is not None else self.num_layers
        x = torch.zeros(y.shape[0], self.n, device=y.device)
        trajectory = [x.clone()] if return_trajectory else None

        for k in range(K):
            u = F.linear(x, self.W1[k]) + F.linear(y, self.W2[k])
            x = self._block_threshold(u, self.thresholds[k])
            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x

    @property
    def num_hyperparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
