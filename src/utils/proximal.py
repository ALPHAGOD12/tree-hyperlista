"""Proximal operators for structured sparse recovery."""

import torch


def soft_threshold(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Elementwise soft-thresholding."""
    return torch.sign(x) * torch.clamp(torch.abs(x) - theta, min=0.0)


def soft_threshold_with_support(x: torch.Tensor, theta: float,
                                p: int) -> torch.Tensor:
    """
    ALISTA/HyperLISTA support-aware thresholding.
    Top-p entries by magnitude bypass thresholding.
    """
    n = x.shape[-1]
    p = min(p, n)

    if p > 0:
        abs_x = torch.abs(x)
        _, top_idx = torch.topk(abs_x, p, dim=-1)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(-1, top_idx, True)
        result = torch.where(mask, x, soft_threshold(x, theta))
    else:
        result = soft_threshold(x, theta)
    return result


def block_soft_threshold(x: torch.Tensor, theta: float,
                         group_size: int) -> torch.Tensor:
    """
    Block (group) soft-thresholding proximal operator.
    prox(u)_b = max(0, 1 - theta / ||u_b||) * u_b
    """
    batch_size, n = x.shape
    num_groups = n // group_size

    x_grouped = x[:, :num_groups * group_size].reshape(batch_size, num_groups, group_size)
    group_norms = x_grouped.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    shrink = torch.clamp(1.0 - theta / group_norms, min=0.0)
    result_grouped = shrink * x_grouped
    result = result_grouped.reshape(batch_size, num_groups * group_size)

    if n > num_groups * group_size:
        result = torch.cat([result, x[:, num_groups * group_size:]], dim=-1)
    return result


def topk_group_support(x: torch.Tensor, k_groups: int,
                       group_size: int) -> torch.Tensor:
    """
    Top-k group support selection.
    Keep only the top k_groups by l2 norm, zero the rest.
    """
    batch_size, n = x.shape
    num_groups = n // group_size
    k_groups = min(k_groups, num_groups)

    x_grouped = x[:, :num_groups * group_size].reshape(batch_size, num_groups, group_size)
    group_norms = x_grouped.norm(dim=-1)

    _, top_idx = torch.topk(group_norms, k_groups, dim=-1)
    mask = torch.zeros(batch_size, num_groups, device=x.device)
    mask.scatter_(1, top_idx, 1.0)
    mask = mask.unsqueeze(-1).expand_as(x_grouped)

    result_grouped = x_grouped * mask
    result = result_grouped.reshape(batch_size, num_groups * group_size)

    if n > num_groups * group_size:
        result = torch.cat([result, torch.zeros(batch_size, n - num_groups * group_size,
                                                 device=x.device)], dim=-1)
    return result


def hybrid_group_threshold(x: torch.Tensor, theta: float, k_groups: int,
                           group_size: int) -> torch.Tensor:
    """
    Hybrid operator: first select top-k groups, then apply block
    soft-thresholding within the selected set.
    """
    batch_size, n = x.shape
    num_groups = n // group_size
    k_groups = min(k_groups, num_groups)

    x_grouped = x[:, :num_groups * group_size].reshape(batch_size, num_groups, group_size)
    group_norms = x_grouped.norm(dim=-1)

    _, top_idx = torch.topk(group_norms, k_groups, dim=-1)
    select_mask = torch.zeros(batch_size, num_groups, device=x.device)
    select_mask.scatter_(1, top_idx, 1.0)

    group_norms_safe = group_norms.clamp(min=1e-12)
    shrink = torch.clamp(1.0 - theta / group_norms_safe, min=0.0)

    combined = select_mask * shrink
    combined = combined.unsqueeze(-1).expand_as(x_grouped)

    result_grouped = combined * x_grouped
    result = result_grouped.reshape(batch_size, num_groups * group_size)

    if n > num_groups * group_size:
        result = torch.cat([result, torch.zeros(batch_size, n - num_groups * group_size,
                                                 device=x.device)], dim=-1)
    return result
