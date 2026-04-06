"""Tree-aware proximal operators for tree-sparse recovery."""

import torch
import numpy as np
from typing import Dict, List, Tuple


def tree_scores(u: torch.Tensor, children: List[List[int]],
                depth: np.ndarray, rho: float = 0.5) -> torch.Tensor:
    """Compute subtree-aware scores for each node.

    s_i = |u_i| + rho * sum_{j in child(i)} |u_j|
              + rho^2 * sum_{j in grandchild(i)} |u_j| + ...

    Efficient bottom-up aggregation in O(n).
    """
    n = u.shape[-1]
    abs_u = torch.abs(u)

    subtree_sum = abs_u.clone()
    max_d = int(depth.max())

    for d in range(max_d, 0, -1):
        nodes_at_d = np.where(depth == d)[0]
        for node in nodes_at_d:
            parent_idx = -1
            for p_idx in range(n):
                if node in children[p_idx]:
                    parent_idx = p_idx
                    break
            if parent_idx >= 0:
                subtree_sum[..., parent_idx] += rho * subtree_sum[..., node]

    return subtree_sum


def tree_scores_fast(u: torch.Tensor, parent: np.ndarray,
                     depth: np.ndarray, rho: float = 0.5) -> torch.Tensor:
    """Fast subtree-aware scoring using parent array (bottom-up).

    s_i = |u_i| + rho * sum_{children} s_j (recursive, computed bottom-up).
    """
    n = u.shape[-1]
    abs_u = torch.abs(u)
    subtree_sum = abs_u.clone()
    max_d = int(depth.max())

    order_by_depth = np.argsort(-depth)

    for node in order_by_depth:
        p = parent[node]
        if p >= 0:
            subtree_sum[..., p] += rho * subtree_sum[..., node]

    return subtree_sum


def topk_tree_projection(u: torch.Tensor, scores: torch.Tensor,
                         K: int, parent: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-K tree-consistent support selection.

    Greedily select nodes with highest scores while enforcing
    ancestor closure (if a node is selected, all its ancestors are too).

    Returns (projected signal, support mask).
    """
    device = u.device
    batch_size = u.shape[0] if u.dim() > 1 else 1
    n = u.shape[-1]

    if u.dim() == 1:
        u = u.unsqueeze(0)
        scores = scores.unsqueeze(0)

    result = torch.zeros_like(u)
    mask = torch.zeros_like(u, dtype=torch.bool)

    for b in range(batch_size):
        s = scores[b].detach().cpu().numpy()
        sorted_indices = np.argsort(-s)

        selected = set()
        for idx in sorted_indices:
            if len(selected) >= K:
                break
            ancestors = []
            current = int(idx)
            while current >= 0 and current not in selected:
                ancestors.append(current)
                current = int(parent[current]) if parent[current] >= 0 else -1

            if len(selected) + len(ancestors) <= K:
                selected.update(ancestors)

        sel_list = sorted(selected)
        if sel_list:
            sel_tensor = torch.tensor(sel_list, device=device, dtype=torch.long)
            result[b, sel_tensor] = u[b, sel_tensor]
            mask[b, sel_tensor] = True

    if batch_size == 1 and u.shape[0] == 1:
        return result, mask

    return result, mask


def threshold_ancestor_closure(u: torch.Tensor, scores: torch.Tensor,
                               tau: float, parent: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Threshold-based support selection with ancestor closure.

    1. Select candidates: S_tilde = {i : s_i >= tau}
    2. Enforce tree consistency: S = ancestor_closure(S_tilde)
    """
    device = u.device
    batch_size = u.shape[0] if u.dim() > 1 else 1
    n = u.shape[-1]

    if u.dim() == 1:
        u = u.unsqueeze(0)
        scores = scores.unsqueeze(0)

    result = torch.zeros_like(u)
    mask = torch.zeros_like(u, dtype=torch.bool)

    for b in range(batch_size):
        s = scores[b].detach().cpu().numpy()
        candidates = set(np.where(s >= tau)[0].tolist())

        closed = set()
        for node in candidates:
            current = node
            while current >= 0:
                if current in closed:
                    break
                closed.add(current)
                current = int(parent[current]) if parent[current] >= 0 else -1

        sel_list = sorted(closed)
        if sel_list:
            sel_tensor = torch.tensor(sel_list, device=device, dtype=torch.long)
            result[b, sel_tensor] = u[b, sel_tensor]
            mask[b, sel_tensor] = True

    return result, mask


def tree_soft_threshold(u: torch.Tensor, theta: float,
                        support_mask: torch.Tensor) -> torch.Tensor:
    """Apply soft-thresholding within the selected tree support, zero elsewhere.

    x_i = sign(u_i) * max(|u_i| - theta, 0)  if i in S
    x_i = 0                                    if i not in S
    """
    shrunk = torch.sign(u) * torch.clamp(torch.abs(u) - theta, min=0.0)
    return shrunk * support_mask.float()


def hybrid_tree_threshold(u: torch.Tensor, theta: float, K: int,
                          parent: np.ndarray, depth: np.ndarray,
                          rho: float = 0.5) -> torch.Tensor:
    """Hybrid tree operator: tree scoring -> top-K tree projection -> soft threshold.

    This is the main structured operator T_{theta, K}(u):
    1. Compute subtree-aware scores
    2. Select tree-consistent support of size K
    3. Apply soft-thresholding within the support
    4. Zero outside the support
    """
    scores = tree_scores_fast(u, parent, depth, rho)
    _, support_mask = topk_tree_projection(u, scores, K, parent)
    return tree_soft_threshold(u, theta, support_mask)


def hard_tree_projection(u: torch.Tensor, K: int,
                         parent: np.ndarray, depth: np.ndarray,
                         rho: float = 0.5) -> torch.Tensor:
    """Hard tree projection: keep top-K tree-consistent support, zero the rest."""
    scores = tree_scores_fast(u, parent, depth, rho)
    projected, _ = topk_tree_projection(u, scores, K, parent)
    return projected
