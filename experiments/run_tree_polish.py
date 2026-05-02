"""Polish for Exp 9 / 10 / 11.

This script fills the remaining holes:

    - Exp 9: add two support-mechanism variants to the three existing
      ones -- a differentiable ancestor-weighted mechanism and a
      stochastic top-K (Gumbel) mechanism -- and compare all five.
    - Exp 10: re-run the rho decay curve on a denser grid
      rho in {0.0, 0.1, ..., 0.9, 0.95, 0.99}.
    - Exp 11: tuning wall-time vs n. The scalability runner
      ``experiments/run_tree_scaled.py`` already measures evaluation
      time; here we just record (n, tune_time_mean) so the scaling plot
      can be produced.
"""
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.tree_synthetic import TreeSparseDataset, build_balanced_tree
from src.models.tree_hyperlista import TreeHyperLISTA
from src.train import tune_hyper_model
from src.utils.metrics import nmse_db


TREE_DEPTH = 7
BRANCHING = 2
M_RATIO = 0.5
TARGET_K = 30
K_LAYERS = 16
N_TRAIN, N_VAL, N_TEST = 3000, 600, 1000
N_SEEDS = 2
N_TRIALS = 40
RHO_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def _eval(model, te, device):
    model.eval().to(device)
    with torch.no_grad():
        xh = model(te['y'].to(device))
    return nmse_db(xh, te['x'].to(device))


# ---------------------------------------------------------------------------
# Exp 10: denser rho sweep


def dense_rho_sweep(device='cpu'):
    os.makedirs('results/tree_ablation', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)
    tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)

    records = {r: [] for r in RHO_GRID}
    for seed in range(N_SEEDS):
        print(f"\n[rho sweep, seed {seed + 1}]")
        ds = TreeSparseDataset(tree_depth=TREE_DEPTH, branching=BRANCHING,
                               m_ratio=M_RATIO, target_sparsity=TARGET_K,
                               snr_db=30.0, seed=seed * 1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(N_TRAIN, seed=seed * 1000 + 1)
        va = ds.generate(N_VAL, seed=seed * 1000 + 2)
        te = ds.generate(N_TEST, seed=seed * 1000 + 3)

        for r in RHO_GRID:
            print(f"    rho={r}")
            res = tune_hyper_model(
                TreeHyperLISTA,
                {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
                 'support_mode': 'hybrid_tree', 'rho': r},
                tr, va, n_trials=N_TRIALS, device=device)
            records[r].append(_eval(res['model'], te, device))

    final = {'rho': RHO_GRID,
             'nmse_db_mean': [float(np.mean(records[r])) for r in RHO_GRID],
             'nmse_db_std': [float(np.std(records[r])) for r in RHO_GRID]}

    with open('results/tree_ablation/rho_dense.json', 'w') as f:
        json.dump(final, f, indent=2)
    print("[saved] results/tree_ablation/rho_dense.json")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.errorbar(final['rho'], final['nmse_db_mean'],
                yerr=final['nmse_db_std'], marker='o', color='#17becf',
                linewidth=1.6, capsize=3)
    ax.set_xlabel(r'$\rho$ (descendant decay weight)')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Exp 10 (dense): Descendant Decay Sweep')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('paper/tree_figures/tree_fig_rho_dense.pdf')
    fig.savefig('paper/tree_figures/tree_fig_rho_dense.png', dpi=200)
    plt.close(fig)
    print("[saved] paper/tree_figures/tree_fig_rho_dense.{pdf,png}")
    return final


# ---------------------------------------------------------------------------
# Exp 9: extended support-mechanism comparison
#
# The five mechanisms are:
#   M1  tree_hard       -- hard top-K tree projection, no shrinkage
#   M2  tree_threshold  -- threshold + ancestor closure + soft shrinkage
#   M3  hybrid_tree     -- top-K tree selection + soft shrinkage
#   M4  diff_ancestor   -- differentiable ancestor-weighted (soft mask)
#   M5  gumbel_topk     -- stochastic top-K via Gumbel noise


def _diff_ancestor_mechanism(u: torch.Tensor, parent: np.ndarray,
                             depth: np.ndarray, rho: float,
                             theta: float) -> torch.Tensor:
    """Soft differentiable ancestor-weighted thresholding.

    Each node keeps a continuous weight
        w_i = sigmoid(alpha * (s_i / mean(s) - 1))
    where ``s_i`` is the subtree score. The output is the soft-
    thresholded ``u_i`` scaled by ``w_i``. Provides a differentiable
    alternative to the hard top-K projection.
    """
    from src.utils.tree_proximal import tree_scores_fast, tree_soft_threshold
    scores = tree_scores_fast(u, parent, depth, rho)
    mean_s = scores.mean(dim=-1, keepdim=True).clamp(min=1e-8)
    w = torch.sigmoid(4.0 * (scores / mean_s - 1.0))
    shrunk = torch.sign(u) * torch.clamp(torch.abs(u) - theta, min=0.0)
    return shrunk * w


def _gumbel_topk_mechanism(u: torch.Tensor, parent: np.ndarray,
                           depth: np.ndarray, rho: float, theta: float,
                           K: int, tau: float = 0.5) -> torch.Tensor:
    """Stochastic top-K tree mechanism using Gumbel noise.

    Adds Gumbel noise to the subtree scores, then takes the hard top-K
    with ancestor closure. During eval we turn off the noise.
    """
    from src.utils.tree_proximal import (
        tree_scores_fast, topk_tree_projection, tree_soft_threshold,
    )
    scores = tree_scores_fast(u, parent, depth, rho)
    if torch.is_grad_enabled():
        g = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
        scores = scores + tau * g
    _, mask = topk_tree_projection(u, scores, K, parent)
    return tree_soft_threshold(u, theta, mask)


class _ExtendedTreeHyperLISTA(TreeHyperLISTA):
    """TreeHyperLISTA with two extra support mechanisms."""

    def _apply_tree_operator(self, u, theta_scalar, K_int):
        if self.support_mode == 'diff_ancestor':
            return _diff_ancestor_mechanism(
                u, self.parent, self.depth_arr, self.rho, theta_scalar)
        if self.support_mode == 'gumbel_topk':
            return _gumbel_topk_mechanism(
                u, self.parent, self.depth_arr, self.rho,
                theta_scalar, K_int, tau=0.3)
        return super()._apply_tree_operator(u, theta_scalar, K_int)


def mechanism_comparison(device='cpu'):
    os.makedirs('results/tree_ablation', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)
    tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)
    modes = ['tree_hard', 'tree_threshold', 'hybrid_tree',
             'diff_ancestor', 'gumbel_topk']
    records = {m: [] for m in modes}

    for seed in range(N_SEEDS):
        print(f"\n[mechanism compare, seed {seed + 1}]")
        ds = TreeSparseDataset(tree_depth=TREE_DEPTH, branching=BRANCHING,
                               m_ratio=M_RATIO, target_sparsity=TARGET_K,
                               snr_db=30.0, seed=seed * 1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(N_TRAIN, seed=seed * 1000 + 1)
        va = ds.generate(N_VAL, seed=seed * 1000 + 2)
        te = ds.generate(N_TEST, seed=seed * 1000 + 3)

        for m in modes:
            print(f"    mode={m}")
            res = tune_hyper_model(
                _ExtendedTreeHyperLISTA,
                {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
                 'support_mode': m, 'rho': 0.5},
                tr, va, n_trials=N_TRIALS, device=device)
            records[m].append(_eval(res['model'], te, device))

    final = {}
    for m in modes:
        final[m] = {
            'nmse_db_mean': float(np.mean(records[m])),
            'nmse_db_std': float(np.std(records[m])),
        }
    with open('results/tree_ablation/mechanism_ext.json', 'w') as f:
        json.dump(final, f, indent=2)
    print("[saved] results/tree_ablation/mechanism_ext.json")

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    names = modes
    labels = ['Hard', 'Thresh+Closure', 'Hybrid', 'Diff Ancestor',
              'Gumbel Top-K']
    means = [final[m]['nmse_db_mean'] for m in names]
    stds = [final[m]['nmse_db_std'] for m in names]
    colors = ['#9467bd', '#8c564b', '#e377c2', '#2ca02c', '#ff7f0e']
    ax.bar(labels, means, yerr=stds, color=colors, capsize=5,
           edgecolor='black', linewidth=0.5)
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Exp 9 (ext): Five Support Mechanisms')
    ax.grid(True, axis='y', alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
    fig.tight_layout()
    fig.savefig('paper/tree_figures/tree_fig_mechanisms_ext.pdf')
    fig.savefig('paper/tree_figures/tree_fig_mechanisms_ext.png', dpi=200)
    plt.close(fig)
    print("[saved] paper/tree_figures/tree_fig_mechanisms_ext.{pdf,png}")
    return final


# ---------------------------------------------------------------------------
# Exp 11: tuning time vs n


def tuning_time_vs_n(device='cpu', n_trials=20):
    os.makedirs('results/tree_scaled', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)

    configs = [
        (5, 2, 63, 0.5, 15),
        (7, 2, 255, 0.5, 30),
        (9, 2, 1023, 0.4, 80),
    ]
    record = []
    for depth, br, n_expected, m_ratio, K in configs:
        tree_info = build_balanced_tree(depth, br)
        n = tree_info['n']
        ds = TreeSparseDataset(tree_depth=depth, branching=br,
                               m_ratio=m_ratio, target_sparsity=K,
                               snr_db=30.0, seed=0, matrix_seed=1)
        tr = ds.generate(1500, seed=1)
        va = ds.generate(300, seed=2)
        t0 = time.time()
        tune_hyper_model(
            TreeHyperLISTA,
            {'A': ds.A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'hybrid_tree', 'rho': 0.5},
            tr, va, n_trials=n_trials, device=device)
        dt = time.time() - t0
        print(f"  n={n}: tune time = {dt:.2f}s")
        record.append({'n': n, 'tune_time_s': dt})

    with open('results/tree_scaled/tuning_time.json', 'w') as f:
        json.dump(record, f, indent=2)
    print("[saved] results/tree_scaled/tuning_time.json")

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    ns = [r['n'] for r in record]
    ts = [r['tune_time_s'] for r in record]
    ax.plot(ns, ts, marker='o', color='#17becf', linewidth=1.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Problem size $n$')
    ax.set_ylabel('Tuning wall time (s)')
    ax.set_title('Exp 11: Tuning Cost vs $n$')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig('paper/tree_figures/tree_fig_tune_time.pdf')
    fig.savefig('paper/tree_figures/tree_fig_tune_time.png', dpi=200)
    plt.close(fig)
    print("[saved] paper/tree_figures/tree_fig_tune_time.{pdf,png}")
    return record


if __name__ == '__main__':
    from src.utils.sensing import pick_device; dev = pick_device()
    dense_rho_sweep(device=dev)
    mechanism_comparison(device=dev)
    tuning_time_vs_n(device=dev)
