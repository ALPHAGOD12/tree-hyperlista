"""Evaluation metrics for sparse recovery."""

import torch
import numpy as np


def nmse(x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """Normalized mean squared error: ||x_hat - x_true||^2 / ||x_true||^2."""
    diff = (x_hat - x_true).pow(2).sum(dim=-1)
    norm = x_true.pow(2).sum(dim=-1).clamp(min=1e-12)
    return (diff / norm).mean()


def nmse_db(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    """NMSE in decibels."""
    val = nmse(x_hat, x_true)
    if val <= 0:
        return -100.0
    return 10.0 * torch.log10(val.clamp(min=1e-40)).item()


def node_support(x: torch.Tensor, tol: float = 1e-3) -> torch.Tensor:
    """Return binary mask of active nodes (batch x n).
    Tolerance relative to max magnitude for robustness.
    """
    abs_x = x.abs()
    max_val = abs_x.max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    return (abs_x / max_val > tol).float()


def node_precision_recall(x_hat: torch.Tensor, x_true: torch.Tensor) -> tuple:
    """Node-level support precision and recall for tree-sparse signals."""
    supp_hat = node_support(x_hat)
    supp_true = node_support(x_true)

    tp = (supp_hat * supp_true).sum(dim=-1)
    pred_pos = supp_hat.sum(dim=-1).clamp(min=1e-12)
    true_pos = supp_true.sum(dim=-1).clamp(min=1e-12)

    precision = (tp / pred_pos).mean().item()
    recall = (tp / true_pos).mean().item()
    return precision, recall


def count_parameters(model) -> int:
    """Count trainable parameters in a PyTorch module."""
    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0


def count_hyperparameters(model) -> int:
    """Count tunable hyperparameters for HyperLISTA-style models."""
    if hasattr(model, 'num_hyperparams'):
        return model.num_hyperparams
    return count_parameters(model)
