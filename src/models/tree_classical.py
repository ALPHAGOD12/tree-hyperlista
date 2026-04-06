"""Classical model-based algorithms for tree-sparse recovery: Tree-IHT and Tree-CoSaMP."""

import torch
import numpy as np
from typing import Optional
from src.utils.tree_proximal import (
    tree_scores_fast,
    topk_tree_projection,
    tree_soft_threshold,
    hard_tree_projection,
)


class TreeIHT:
    """Model-Based Iterative Hard Thresholding for tree-sparse recovery.

    At each iteration:
        x^{k+1} = P_{T,K}( x^k - eta * A^T (A x^k - y) )
    where P_{T,K} is the tree-consistent top-K projection.

    Reference: Baraniuk et al., "Model-Based Compressive Sensing", IEEE TIT 2010.
    """

    def __init__(self, A: np.ndarray, tree_info: dict, target_K: int = 30,
                 max_iter: int = 16, rho: float = 0.5, step_size: float = None):
        self.A = torch.from_numpy(A).float()
        self.tree_info = tree_info
        self.parent = tree_info['parent']
        self.depth_arr = tree_info['depth']
        self.n = A.shape[1]
        self.m = A.shape[0]
        self.target_K = target_K
        self.max_iter = max_iter
        self.num_layers = max_iter
        self.rho = rho
        self.num_hyperparams = 0

        eigvals = np.linalg.eigvalsh(A.T @ A)
        self.step_size = step_size if step_size else 1.0 / max(eigvals.max(), 1e-8)

    def to(self, device):
        self.A = self.A.to(device)
        return self

    def eval(self):
        return self

    def solve(self, y: torch.Tensor, return_trajectory: bool = False,
              num_iter: Optional[int] = None) -> torch.Tensor:
        K = num_iter if num_iter is not None else self.max_iter
        device = y.device
        self.A = self.A.to(device)
        batch_size = y.shape[0]

        x = torch.zeros(batch_size, self.n, device=device)
        trajectory = [x.clone()] if return_trajectory else None

        for k in range(K):
            residual = y - x @ self.A.t()
            grad = (self.A.t() @ residual.t()).t()
            u = x + self.step_size * grad

            x = hard_tree_projection(u, self.target_K, self.parent,
                                     self.depth_arr, self.rho)

            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x

    def parameters(self):
        return iter([])


class TreeCoSaMP:
    """Model-Based CoSaMP for tree-sparse recovery.

    At each iteration:
        1. Signal proxy: r = A^T (y - Ax), score nodes by subtree magnitudes
        2. Identify: pick 2K nodes via tree-consistent selection
        3. Merge: T = support(x) union new nodes (with ancestor closure)
        4. Estimate: least-squares on merged support
        5. Prune: keep best K-tree-consistent nodes

    Reference: Baraniuk et al., "Model-Based Compressive Sensing", IEEE TIT 2010.
    """

    def __init__(self, A: np.ndarray, tree_info: dict, target_K: int = 30,
                 max_iter: int = 16, rho: float = 0.5):
        self.A_np = A
        self.A = torch.from_numpy(A).float()
        self.tree_info = tree_info
        self.parent = tree_info['parent']
        self.depth_arr = tree_info['depth']
        self.n = A.shape[1]
        self.m = A.shape[0]
        self.target_K = target_K
        self.max_iter = max_iter
        self.num_layers = max_iter
        self.rho = rho
        self.num_hyperparams = 0

    def to(self, device):
        self.A = self.A.to(device)
        return self

    def eval(self):
        return self

    def _least_squares_batch(self, y: torch.Tensor, support_masks: torch.Tensor) -> torch.Tensor:
        """Batch least-squares estimation restricted to each sample's support."""
        device = y.device
        batch_size = y.shape[0]
        x_out = torch.zeros(batch_size, self.n, device=device)

        for b in range(batch_size):
            supp_idx = torch.where(support_masks[b])[0]
            if len(supp_idx) == 0:
                continue
            A_sub = self.A[:, supp_idx]
            try:
                x_sub = torch.linalg.lstsq(A_sub, y[b:b+1].t()).solution.squeeze(-1)
            except Exception:
                x_sub = torch.linalg.pinv(A_sub) @ y[b]
            x_out[b, supp_idx] = x_sub

        return x_out

    def solve(self, y: torch.Tensor, return_trajectory: bool = False,
              num_iter: Optional[int] = None) -> torch.Tensor:
        K = num_iter if num_iter is not None else self.max_iter
        device = y.device
        self.A = self.A.to(device)
        batch_size = y.shape[0]

        x = torch.zeros(batch_size, self.n, device=device)
        trajectory = [x.clone()] if return_trajectory else None

        for k in range(K):
            residual = y - x @ self.A.t()
            proxy = (self.A.t() @ residual.t()).t()

            scores_proxy = tree_scores_fast(proxy, self.parent, self.depth_arr, self.rho)
            _, ident_mask = topk_tree_projection(proxy, scores_proxy,
                                                 min(2 * self.target_K, self.n),
                                                 self.parent)

            current_support = (x.abs() > 1e-8).float()
            merged_mask = ((ident_mask.float() + current_support) > 0)

            for b in range(batch_size):
                merged_nodes = torch.where(merged_mask[b])[0].cpu().numpy()
                for node in merged_nodes:
                    cur = int(self.parent[node])
                    while cur >= 0:
                        merged_mask[b, cur] = True
                        cur = int(self.parent[cur]) if self.parent[cur] >= 0 else -1

            x_ls = self._least_squares_batch(y, merged_mask)

            scores_prune = tree_scores_fast(x_ls, self.parent, self.depth_arr, self.rho)
            x, _ = topk_tree_projection(x_ls, scores_prune, self.target_K, self.parent)

            if return_trajectory:
                trajectory.append(x.clone())

        return trajectory if return_trajectory else x

    def parameters(self):
        return iter([])
