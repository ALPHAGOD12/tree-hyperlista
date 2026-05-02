"""Backbone ablation variants for Tree-HyperLISTA (Exp 1).

Each variant strips one or more architectural ingredients so we can
attribute NMSE gains to specific components. All variants share the
same interface as ``TreeHyperLISTA``/``ALISTA``/``HyperLISTA`` so they
plug into the same training and evaluation pipeline.

Variants, in order of strictly added structure:
    1. TreeALISTA          -- analytic W, per-layer (theta_k, gamma_k),
                              elementwise soft-thresholding.
    2. TreeALISTASym       -- symmetric W (Section 3.1 of HyperLISTA),
                              otherwise same as TreeALISTA.
    3. TreeALISTAMM        -- analytic W + Polyak momentum, elementwise
                              soft-thresholding.
    4. TreeALISTAMMSym     -- symmetric W + Polyak momentum, elementwise.
    5. TreeHyperLISTAElem  -- 3 hyperparams + sym W + momentum, but with
                              elementwise support (no tree prior).

The full Tree-HyperLISTA model (sym W + momentum + 3 hyperparams + tree
proximal) lives in ``src/models/tree_hyperlista.py``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.utils.proximal import soft_threshold, soft_threshold_with_support
from src.utils.sensing import compute_mutual_coherence


def _compute_analytic_W(A: np.ndarray) -> np.ndarray:
    """ALISTA-style analytic W (not necessarily symmetric).

    Solves min_{W} || W^T A ||_F^2 s.t. diag(W^T A) = 1 in closed form:
    W = A (A^T A)^{-1}.
    """
    m, n = A.shape
    AtA = A.T @ A
    W_sol = np.linalg.lstsq(AtA, np.eye(n), rcond=None)[0]
    W = A @ W_sol
    return W.astype(np.float32)


def _compute_symmetric_W(A: np.ndarray, num_iter: int = 200,
                        lr: float = 0.002) -> np.ndarray:
    """Symmetric W used by HyperLISTA (Section 3.1)."""
    m, n = A.shape
    D = A / np.linalg.norm(A, axis=0, keepdims=True)
    for _ in range(num_iter):
        gram = D.T @ D
        grad = 2.0 * D @ (gram - np.eye(n))
        D = D - lr * grad
        norms = np.linalg.norm(D, axis=0, keepdims=True)
        D = D / np.maximum(norms, 1e-12)
    G = np.linalg.lstsq(A, D, rcond=None)[0]
    W = A @ G
    return W.astype(np.float32)


class _BaseTreeBackbone(nn.Module):
    """Shared structure for Exp 1 backbone variants.

    Subclasses control the three binary flags:
        ``_symmetric_W``   : use symmetric or analytic ALISTA W
        ``_use_momentum``  : apply Polyak momentum between layers
        ``_proximal_mode`` : 'elem_fixed' (per-layer learned theta) or
                             'elem_hyper' (3 hyperparams, HyperLISTA).
    """

    _symmetric_W: bool = False
    _use_momentum: bool = False
    _proximal_mode: str = 'elem_fixed'

    def __init__(self, A: np.ndarray, num_layers: int = 16,
                 c1: float = 1.0, c2: float = 0.0, c3: float = 3.0):
        super().__init__()
        m, n = A.shape
        self.n = n
        self.m = m
        self.num_layers = num_layers

        self.register_buffer('A', torch.from_numpy(A).float())
        if self._symmetric_W:
            W_np = _compute_symmetric_W(A)
        else:
            W_np = _compute_analytic_W(A)
        self.register_buffer('W', torch.from_numpy(W_np).float())

        A_pinv = np.linalg.pinv(A).astype(np.float32)
        self.register_buffer('A_pinv', torch.from_numpy(A_pinv).float())

        self.mu = compute_mutual_coherence(A)

        if self._proximal_mode == 'elem_fixed':
            self.step_sizes = nn.ParameterList([
                nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)
            ])
            self.thresholds = nn.ParameterList([
                nn.Parameter(torch.tensor(0.1)) for _ in range(num_layers)
            ])
            if self._use_momentum:
                self.momenta = nn.ParameterList([
                    nn.Parameter(torch.tensor(0.0)) for _ in range(num_layers)
                ])
            self.num_hyperparams = (
                2 * num_layers + (num_layers if self._use_momentum else 0)
            )
        elif self._proximal_mode == 'elem_hyper':
            self.c1 = nn.Parameter(torch.tensor(c1))
            self.c2 = nn.Parameter(torch.tensor(c2))
            self.c3 = nn.Parameter(torch.tensor(c3))
            self.num_hyperparams = 3
        else:
            raise ValueError(self._proximal_mode)

    def _hyper_adaptive(self, x: torch.Tensor, y: torch.Tensor):
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
                num_layers: int | None = None) -> torch.Tensor:
        K = num_layers if num_layers is not None else self.num_layers
        device = y.device
        batch_size = y.shape[0]

        x = torch.zeros(batch_size, self.n, device=device)
        x_prev = torch.zeros_like(x)
        trajectory = [x.clone()] if return_trajectory else None

        for k in range(K):
            k_eff = min(k, self.num_layers - 1)

            if self._proximal_mode == 'elem_fixed':
                gamma = torch.abs(self.step_sizes[k_eff])
                theta = torch.abs(self.thresholds[k_eff])
                beta = (torch.sigmoid(self.momenta[k_eff])
                        if self._use_momentum else None)
            else:
                theta_vec, beta_vec, p_vec = self._hyper_adaptive(x, y)
                gamma = torch.tensor(1.0, device=device)
                theta = theta_vec.mean()
                beta = beta_vec.mean()

            if self._use_momentum and k > 0:
                z = x + beta * (x - x_prev)
            else:
                z = x

            residual = y - z @ self.A.t()
            u = z + gamma * (self.W.t() @ residual.t()).t()

            if self._proximal_mode == 'elem_fixed':
                x_new = soft_threshold(u, theta)
            else:
                p_int = int(p_vec.mean().item())
                x_new = soft_threshold_with_support(u, theta.item(), p_int)

            x_prev = x
            x = x_new
            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x


class TreeALISTA(_BaseTreeBackbone):
    """Backbone (1): analytic ALISTA W, per-layer (theta_k, gamma_k), elementwise."""
    _symmetric_W = False
    _use_momentum = False
    _proximal_mode = 'elem_fixed'


class TreeALISTASym(_BaseTreeBackbone):
    """Backbone (2): symmetric W, per-layer (theta_k, gamma_k), elementwise."""
    _symmetric_W = True
    _use_momentum = False
    _proximal_mode = 'elem_fixed'


class TreeALISTAMM(_BaseTreeBackbone):
    """Backbone (3): analytic W + Polyak momentum, elementwise."""
    _symmetric_W = False
    _use_momentum = True
    _proximal_mode = 'elem_fixed'


class TreeALISTAMMSym(_BaseTreeBackbone):
    """Backbone (4): symmetric W + Polyak momentum, elementwise."""
    _symmetric_W = True
    _use_momentum = True
    _proximal_mode = 'elem_fixed'


class TreeHyperLISTAElem(_BaseTreeBackbone):
    """Backbone (5): 3 hyperparams + sym W + momentum, elementwise proximal."""
    _symmetric_W = True
    _use_momentum = True
    _proximal_mode = 'elem_hyper'
