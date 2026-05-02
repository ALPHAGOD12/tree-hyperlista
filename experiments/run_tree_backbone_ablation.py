"""Exp 1 -- Backbone ablation for Tree-HyperLISTA.

Six variants are compared on the core tree-sparse problem (binary tree
depth 7, m = n/2, K = 30, SNR = 30 dB):

    1. Tree-ALISTA           : analytic W,   no momentum, elementwise
    2. Tree-ALISTA-Sym       : symmetric W,  no momentum, elementwise
    3. Tree-ALISTA-MM        : analytic W,   momentum,    elementwise
    4. Tree-ALISTA-MM-Sym    : symmetric W,  momentum,    elementwise
    5. Tree-HyperLISTA-Elem  : sym W + momentum + 3 hparams, elementwise
    6. Tree-HyperLISTA       : ours -- sym W + momentum + 3 hparams + tree

Output
------
  results/tree_ablation/backbone_results.json   -- per-layer NMSE curves
  paper/tree_figures/tree_fig_backbone.pdf/png  -- NMSE-vs-layer plot
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.tree_synthetic import TreeSparseDataset, build_balanced_tree
from src.models.tree_ablation_variants import (
    TreeALISTA, TreeALISTASym, TreeALISTAMM, TreeALISTAMMSym,
    TreeHyperLISTAElem,
)
from src.models.tree_hyperlista import TreeHyperLISTA
from src.train import train_unfolded_model, tune_hyper_model
from src.utils.metrics import nmse_db


TREE_DEPTH = 7
BRANCHING = 2
M_RATIO = 0.5
TARGET_K = 30
K_LAYERS = 16
RHO = 0.5
N_TRAIN, N_VAL, N_TEST = 3000, 600, 1000
N_SEEDS = 3
N_TRIALS = 40
TRAIN_EPOCHS = 120


def _traj_nmse(model, test_data, device):
    x_test = test_data['x'].to(device)
    y_test = test_data['y'].to(device)
    model.eval().to(device)
    with torch.no_grad():
        traj = model(y_test, return_trajectory=True)
    return [nmse_db(xk, x_test) for xk in traj]


def _train_backprop_variant(cls, A, tr, va, device):
    model = cls(A, num_layers=K_LAYERS)
    train_unfolded_model(model, tr, va, num_epochs=TRAIN_EPOCHS,
                         lr=1e-3, batch_size=128, device=device,
                         progressive=True, grad_clip=5.0)
    return model


def _tune_hyper_variant(cls, model_kwargs, tr, va, device, n_trials=N_TRIALS):
    result = tune_hyper_model(cls, model_kwargs, tr, va,
                              n_trials=n_trials, device=device)
    return result['model']


def run_backbone_ablation(device='cpu'):
    os.makedirs('results/tree_ablation', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)

    tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)
    n = tree_info['n']
    m = int(n * M_RATIO)
    print(f"Backbone ablation: n={n}, m={m}, K={TARGET_K}, SNR=30dB")

    trajectories = defaultdict(list)

    for seed in range(N_SEEDS):
        print(f"\n[seed {seed + 1}/{N_SEEDS}]")
        ds = TreeSparseDataset(tree_depth=TREE_DEPTH, branching=BRANCHING,
                               m_ratio=M_RATIO, target_sparsity=TARGET_K,
                               snr_db=30.0, seed=seed * 1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(N_TRAIN, seed=seed * 1000 + 1)
        va = ds.generate(N_VAL, seed=seed * 1000 + 2)
        te = ds.generate(N_TEST, seed=seed * 1000 + 3)

        print("  (1) Tree-ALISTA")
        m1 = _train_backprop_variant(TreeALISTA, A, tr, va, device)
        trajectories['Tree-ALISTA'].append(_traj_nmse(m1, te, device))

        print("  (2) Tree-ALISTA-Sym")
        m2 = _train_backprop_variant(TreeALISTASym, A, tr, va, device)
        trajectories['Tree-ALISTA-Sym'].append(_traj_nmse(m2, te, device))

        print("  (3) Tree-ALISTA-MM")
        m3 = _train_backprop_variant(TreeALISTAMM, A, tr, va, device)
        trajectories['Tree-ALISTA-MM'].append(_traj_nmse(m3, te, device))

        print("  (4) Tree-ALISTA-MM-Sym")
        m4 = _train_backprop_variant(TreeALISTAMMSym, A, tr, va, device)
        trajectories['Tree-ALISTA-MM-Sym'].append(_traj_nmse(m4, te, device))

        print("  (5) Tree-HyperLISTA-Elem")
        m5 = _tune_hyper_variant(
            TreeHyperLISTAElem, {'A': A, 'num_layers': K_LAYERS},
            tr, va, device)
        trajectories['Tree-HyperLISTA-Elem'].append(_traj_nmse(m5, te, device))

        print("  (6) Tree-HyperLISTA (ours)")
        m6 = _tune_hyper_variant(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'hybrid_tree', 'rho': RHO},
            tr, va, device)
        trajectories['Tree-HyperLISTA'].append(_traj_nmse(m6, te, device))

    agg = {}
    max_len = K_LAYERS + 1
    for name, trajs in trajectories.items():
        padded = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
        arr = np.array(padded)
        agg[name] = {
            'trajectory_mean': arr.mean(0).tolist(),
            'trajectory_std': arr.std(0).tolist(),
            'final_nmse_mean': float(arr[:, -1].mean()),
            'final_nmse_std': float(arr[:, -1].std()),
        }

    out_json = 'results/tree_ablation/backbone_results.json'
    with open(out_json, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\n[saved] {out_json}")

    _plot_backbone(agg)
    return agg


def _plot_backbone(agg):
    colors = {
        'Tree-ALISTA': '#888888', 'Tree-ALISTA-Sym': '#4c72b0',
        'Tree-ALISTA-MM': '#55a868', 'Tree-ALISTA-MM-Sym': '#c44e52',
        'Tree-HyperLISTA-Elem': '#8172b2', 'Tree-HyperLISTA': '#e377c2',
    }
    markers = {
        'Tree-ALISTA': 'v', 'Tree-ALISTA-Sym': 's',
        'Tree-ALISTA-MM': 'D', 'Tree-ALISTA-MM-Sym': 'o',
        'Tree-HyperLISTA-Elem': 'p', 'Tree-HyperLISTA': '*',
    }
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    for name, res in agg.items():
        mean = res['trajectory_mean']
        std = res['trajectory_std']
        xs = list(range(len(mean)))
        c = colors.get(name, '#333333')
        mk = markers.get(name, 'o')
        ax.plot(xs, mean, color=c, marker=mk, markevery=2, markersize=6,
                label=name, linewidth=1.6)
        lo = [m_ - s for m_, s in zip(mean, std)]
        hi = [m_ + s for m_, s in zip(mean, std)]
        ax.fill_between(xs, lo, hi, alpha=0.12, color=c)
    ax.set_xlabel('Layer / Iteration')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Exp 1: Backbone Ablation on Tree-Sparse Recovery')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig('paper/tree_figures/tree_fig_backbone.pdf')
    fig.savefig('paper/tree_figures/tree_fig_backbone.png', dpi=200)
    plt.close(fig)
    print("[saved] paper/tree_figures/tree_fig_backbone.{pdf,png}")


if __name__ == '__main__':
    from src.utils.sensing import pick_device; dev = pick_device()
    run_backbone_ablation(device=dev)
