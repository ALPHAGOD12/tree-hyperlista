"""Elementwise proximal operators used by HyperLISTA and LISTA baselines."""

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
