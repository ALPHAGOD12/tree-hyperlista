"""Differentiable tree proximal operators for end-to-end training.

All operations avoid in-place tensor modifications to support autograd.
Uses index_add / scatter patterns instead of direct indexing assignment.
"""

import torch
import numpy as np
from typing import Tuple


def _precompute_tree_order(parent: np.ndarray, depth: np.ndarray):
    """Precompute bottom-up traversal order and parent indices."""
    order = np.argsort(-depth)  # deepest first
    # Filter to nodes that have a parent
    valid = [(int(node), int(parent[node])) for node in order if parent[node] >= 0]
    return valid


def soft_tree_scores(u: torch.Tensor, parent: np.ndarray,
                     depth: np.ndarray, rho: float = 0.5) -> torch.Tensor:
    """Differentiable subtree-aware scoring. No in-place ops.

    s_i = |u_i| + rho * sum_{children} s_j
    Built layer-by-layer from leaves to root using functional ops.
    """
    abs_u = torch.abs(u)
    n = u.shape[-1]
    max_d = int(depth.max())

    # Process each depth level: accumulate child contributions to parents
    # Use a list of per-node values, rebuilt at each level
    node_scores = abs_u  # start with |u_i|

    for d in range(max_d, 0, -1):
        children_at_d = np.where(depth == d)[0]
        if len(children_at_d) == 0:
            continue

        # Gather child scores and their parent indices
        child_indices = torch.tensor(children_at_d, dtype=torch.long, device=u.device)
        parent_indices = torch.tensor(
            [parent[c] for c in children_at_d], dtype=torch.long, device=u.device)

        # Get child scores
        child_scores = node_scores[..., child_indices]  # (..., num_children)

        # Create contribution tensor: zeros everywhere, child contributions at parent positions
        contrib = torch.zeros_like(node_scores)
        # Use scatter_add for differentiable accumulation
        if u.dim() == 1:
            contrib.scatter_add_(0, parent_indices, rho * child_scores)
        else:
            # Batch case: expand parent_indices to match batch dims
            expanded_parents = parent_indices.unsqueeze(0).expand(u.shape[0], -1)
            contrib.scatter_add_(1, expanded_parents, rho * child_scores)

        node_scores = node_scores + contrib

    return node_scores


def soft_topk_mask(scores: torch.Tensor, K: int,
                   temperature: float = 1.0) -> torch.Tensor:
    """Differentiable top-K selection via sigmoid threshold."""
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)

    batch_size, n = scores.shape
    K_clamped = min(K, n)

    topk_vals, _ = torch.topk(scores, K_clamped, dim=-1)
    threshold = topk_vals[:, -1:]  # K-th largest value

    mask = torch.sigmoid(temperature * (scores - threshold))
    return mask


def soft_ancestor_closure(mask: torch.Tensor, parent: np.ndarray,
                          depth: np.ndarray) -> torch.Tensor:
    """Differentiable ancestor closure using scatter-based max propagation."""
    n = mask.shape[-1]
    max_d = int(depth.max())
    result = mask

    for d in range(max_d, 0, -1):
        children_at_d = np.where(depth == d)[0]
        if len(children_at_d) == 0:
            continue

        child_indices = torch.tensor(children_at_d, dtype=torch.long, device=mask.device)
        parent_indices = torch.tensor(
            [parent[c] for c in children_at_d], dtype=torch.long, device=mask.device)

        child_vals = result[..., child_indices]

        if mask.dim() == 1:
            parent_vals = result[parent_indices]
            new_parent_vals = torch.maximum(parent_vals, child_vals)
            # Build new result: replace parent positions
            updates = torch.zeros_like(result)
            updates.scatter_(0, parent_indices, new_parent_vals - parent_vals)
            result = result + updates
        else:
            expanded_parents = parent_indices.unsqueeze(0).expand(mask.shape[0], -1)
            parent_vals = torch.gather(result, 1, expanded_parents)
            new_parent_vals = torch.maximum(parent_vals, child_vals)
            diff = new_parent_vals - parent_vals
            updates = torch.zeros_like(result)
            updates.scatter_add_(1, expanded_parents, diff)
            result = result + updates

    return result


def diff_tree_projection(u: torch.Tensor, scores: torch.Tensor,
                         K: int, parent: np.ndarray, depth: np.ndarray,
                         temperature: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully differentiable tree-consistent projection."""
    soft_mask = soft_topk_mask(scores, K, temperature=temperature)
    soft_mask = soft_ancestor_closure(soft_mask, parent, depth)
    projected = u * soft_mask
    return projected, soft_mask


def diff_tree_soft_threshold(u: torch.Tensor, theta: float,
                             soft_mask: torch.Tensor) -> torch.Tensor:
    """Soft-threshold within differentiable support mask."""
    shrunk = torch.sign(u) * torch.clamp(torch.abs(u) - theta, min=0.0)
    return shrunk * soft_mask


def diff_hybrid_tree(u: torch.Tensor, theta: float, K: int,
                     parent: np.ndarray, depth: np.ndarray,
                     rho: float = 0.5,
                     temperature: float = 5.0) -> torch.Tensor:
    """Complete differentiable hybrid tree operator."""
    scores = soft_tree_scores(u, parent, depth, rho)
    _, soft_mask = diff_tree_projection(u, scores, K, parent, depth, temperature)
    return diff_tree_soft_threshold(u, theta, soft_mask)
