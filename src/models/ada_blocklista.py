"""Ada-BlockLISTA: Adaptive Block LISTA (Fu et al., ICASSP 2021).

Implements both tied and untied variants of the block-aware deep unfolding
architecture with per-layer step sizes and block thresholds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaBlockLISTA(nn.Module):
    """
    Ada-BlockLISTA with tied weights (one shared B matrix) and per-layer
    step sizes / block thresholds.

    Per layer k:
        z^{k} = x^{k} - gamma^{k} * B^T (B x^{k} - y)
        x^{k+1} = BlockSoftThreshold(z^{k}, alpha^{k})

    Params: n*m (shared B) + 2*K (gamma, alpha per layer)
    """

    def __init__(self, A: np.ndarray, group_size: int, num_layers: int = 16,
                 tied: bool = True):
        super().__init__()
        m, n = A.shape
        self.n = n
        self.m = m
        self.num_layers = num_layers
        self.group_size = group_size
        self.num_groups = n // group_size
        self.tied = tied

        A_t = torch.from_numpy(A).float()
        L = float(torch.linalg.eigvalsh(A_t.t() @ A_t).max())

        if tied:
            self.B = nn.Parameter(A_t.clone())
        else:
            self.B_layers = nn.ParameterList([
                nn.Parameter(A_t.clone()) for _ in range(num_layers)
            ])

        self.step_sizes = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0 / L)) for _ in range(num_layers)
        ])
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.ones(self.num_groups) * 0.01) for _ in range(num_layers)
        ])

    def _get_B(self, k: int) -> torch.Tensor:
        if self.tied:
            return self.B
        return self.B_layers[k]

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
            B = self._get_B(k)
            gamma = torch.abs(self.step_sizes[k])

            residual = F.linear(x, B) - y
            grad = F.linear(residual, B.t())
            z = x - gamma * grad

            x = self._block_threshold(z, self.thresholds[k])

            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x

    @property
    def num_hyperparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
