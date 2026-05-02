"""Differentiable Tree-HyperLISTA: fully end-to-end trainable.

Unlike the original Tree-HyperLISTA which uses non-differentiable tree
projection (requiring Bayesian optimization), this version uses smooth
relaxations that allow gradient-based training via backpropagation.

Key differences from tree_hyperlista.py:
  - Soft top-K selection (sigmoid-based) instead of hard argmax
  - Differentiable ancestor closure (soft-OR propagation)
  - Temperature annealing: start smooth, sharpen during training
  - Can be trained with standard Adam/SGD, not just Bayesian opt
  - Enables self-supervised test-time adaptation (gradients flow!)
"""

import torch
import torch.nn as nn
import numpy as np
from src.utils.diff_tree_proximal import (
    soft_tree_scores,
    diff_tree_projection,
    diff_tree_soft_threshold,
)
from src.utils.sensing import compute_mutual_coherence


class DiffTreeHyperLISTA(nn.Module):
    """Differentiable Tree-HyperLISTA with temperature-controlled relaxation.

    3 learned hyperparameters (c1, c2, c3) + temperature schedule.
    Fully trainable via backpropagation.
    """

    def __init__(self, A: np.ndarray, tree_info: dict, num_layers: int = 16,
                 c1: float = 1.0, c2: float = 0.0, c3: float = 3.0,
                 support_mode: str = 'hybrid_tree', rho: float = 0.5,
                 temperature: float = 5.0):
        super().__init__()
        m, n = A.shape
        self.n = n
        self.m = m
        self.num_layers = num_layers
        self.support_mode = support_mode
        self.rho = rho
        self.temperature = temperature
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
        """Same reparameterized formulas as TreeHyperLISTA."""
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

    def _apply_diff_tree_operator(self, u: torch.Tensor, theta: torch.Tensor,
                                   K_int: int) -> torch.Tensor:
        """Apply DIFFERENTIABLE tree operator.

        theta is kept as a tensor (not .item()) to preserve gradients.
        K_int is detached (int) since topk index selection can't be differentiable,
        but the soft sigmoid mask around it IS differentiable.
        """
        scores = soft_tree_scores(u, self.parent, self.depth_arr, self.rho)

        if self.support_mode == 'tree_hard':
            projected, _ = diff_tree_projection(
                u, scores, K_int, self.parent, self.depth_arr, self.temperature)
            return projected

        elif self.support_mode == 'hybrid_tree':
            _, soft_mask = diff_tree_projection(
                u, scores, K_int, self.parent, self.depth_arr, self.temperature)
            # Keep theta as tensor for differentiable soft-thresholding
            shrunk = torch.sign(u) * torch.clamp(torch.abs(u) - theta, min=0.0)
            return shrunk * soft_mask

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

            # K_int must be int for topk, but theta stays as tensor for gradients
            K_int = max(1, int(K_tree.mean().item()))
            theta_mean = theta.mean()  # keep as tensor!

            x_new = self._apply_diff_tree_operator(u, theta_mean, K_int)

            x_prev = x
            x = x_new

            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x

    def set_temperature(self, temp: float):
        """Set softmax temperature for projection sharpness."""
        self.temperature = temp


class SelfSupervisedDiffTreeHyperLISTA(nn.Module):
    """Self-supervised adaptation using differentiable tree projection.

    Because gradients now flow through the tree projection,
    test-time adaptation actually works.
    """

    def __init__(self, A: np.ndarray, tree_info: dict, num_layers: int = 16,
                 c1_init: float = 1.0, c2_init: float = 0.0, c3_init: float = 3.0,
                 support_mode: str = 'hybrid_tree', rho: float = 0.5,
                 adapt_steps: int = 15, adapt_lr: float = 0.05,
                 temperature: float = 5.0, num_restarts: int = 3):
        super().__init__()
        self.A_np = A
        self.tree_info = tree_info
        self.num_layers = num_layers
        self.c1_init = c1_init
        self.c2_init = c2_init
        self.c3_init = c3_init
        self.support_mode = support_mode
        self.rho = rho
        self.adapt_steps = adapt_steps
        self.adapt_lr = adapt_lr
        self.temperature = temperature
        self.num_restarts = num_restarts
        self.num_hyperparams = 3

        self.register_buffer('A', torch.from_numpy(A).float())

    def _adapt(self, y, c1_start, c2_start, c3_start):
        """Adapt (c1,c2,c3) using measurement consistency with differentiable tree."""
        device = y.device

        model = DiffTreeHyperLISTA(
            self.A_np, self.tree_info,
            num_layers=self.num_layers,
            c1=c1_start, c2=c2_start, c3=c3_start,
            support_mode=self.support_mode,
            rho=self.rho,
            temperature=self.temperature,
        ).to(device)

        A = model.A
        optimizer = torch.optim.Adam([model.c1, model.c2, model.c3],
                                     lr=self.adapt_lr)

        best_loss = float('inf')
        best_c = None

        with torch.enable_grad():
            for step in range(self.adapt_steps):
                optimizer.zero_grad()
                x_hat = model(y)
                residual = y - x_hat @ A.t()
                loss = (residual ** 2).sum(dim=-1).mean()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_c = (model.c1.item(), model.c2.item(), model.c3.item())

                loss.backward()
                optimizer.step()

        if best_c is not None:
            model.c1.data.fill_(best_c[0])
            model.c2.data.fill_(best_c[1])
            model.c3.data.fill_(best_c[2])

        return model, best_loss

    def forward(self, y, return_trajectory=False, num_layers=None):
        device = y.device
        best_loss = float('inf')
        best_model = None

        inits = [(self.c1_init, self.c2_init, self.c3_init)]
        for r in range(self.num_restarts - 1):
            rng = torch.Generator()
            rng.manual_seed(42 + r)
            inits.append((
                0.1 + torch.rand(1, generator=rng).item() * 5.0,
                -4.0 + torch.rand(1, generator=rng).item() * 6.0,
                0.1 + torch.rand(1, generator=rng).item() * 5.0,
            ))

        for c1_s, c2_s, c3_s in inits:
            model, loss = self._adapt(y, c1_s, c2_s, c3_s)
            if loss < best_loss:
                best_loss = loss
                best_model = model

        best_model.eval()
        # Final pass at high temperature for sharp output
        best_model.set_temperature(self.temperature * 3)
        with torch.no_grad():
            return best_model(y, return_trajectory=return_trajectory,
                             num_layers=num_layers)

    def solve(self, y, return_trajectory=False, num_iter=None):
        return self.forward(y, return_trajectory=return_trajectory,
                           num_layers=num_iter)

    def set_pretrained_init(self, c1, c2, c3):
        self.c1_init = c1
        self.c2_init = c2
        self.c3_init = c3

    def eval(self):
        return self

    def to(self, device):
        self.A = self.A.to(device)
        return self

    def parameters(self):
        return iter([])
