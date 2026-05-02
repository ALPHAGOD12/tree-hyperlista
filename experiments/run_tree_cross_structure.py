"""Exp 7 -- Cross-structure stress test for Tree-HyperLISTA.

A 2 x 2 grid of (data structure) x (model structure):

                          elementwise data     tree-sparse data
    HyperLISTA            (1)                  (2)
    Tree-HyperLISTA       (3)                  (4)

Expected takeaway: Tree-HyperLISTA matches HyperLISTA on elementwise
signals (row swap, column 1) and dominates on tree-sparse signals
(column 2). Confirms the tree inductive bias helps where it applies
and does not hurt where it does not.

The model architecture (weights, momentum, etc.) is identical; only the
training data distribution and model proximal differ.

Output
------
  results/tree_cross/results.json
  paper/tree_figures/tree_fig_cross.pdf/png
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.tree_synthetic import (
    TreeSparseDataset, build_balanced_tree, generate_tree_support,
)
from src.models.hyperlista import HyperLISTA
from src.models.tree_hyperlista import TreeHyperLISTA
from src.train import tune_hyper_model
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


def _gen_elementwise(A, n, num_samples, K, snr_db, seed,
                     amp_range=(0.5, 2.0)):
    rng = np.random.RandomState(seed)
    m = A.shape[0]
    X = np.zeros((num_samples, n), dtype=np.float32)
    for i in range(num_samples):
        idx = rng.choice(n, size=K, replace=False)
        vals = rng.randn(K).astype(np.float32)
        mags = np.abs(vals) * (amp_range[1] - amp_range[0]) + amp_range[0]
        X[i, idx] = np.sign(vals) * mags
    Y_clean = X @ A.T
    sigma = np.sqrt(np.mean(Y_clean ** 2) * 10 ** (-snr_db / 10.0))
    Y = Y_clean + rng.randn(num_samples, m).astype(np.float32) * sigma
    return {
        'x': torch.from_numpy(X), 'y': torch.from_numpy(Y.astype(np.float32)),
    }


def _gen_tree(ds, num_samples, seed):
    return ds.generate(num_samples, seed=seed)


def _eval(model, data, device):
    model.eval().to(device)
    with torch.no_grad():
        xh = model(data['y'].to(device))
    return nmse_db(xh, data['x'].to(device))


def run_cross(device='cpu'):
    os.makedirs('results/tree_cross', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)

    tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)
    n = tree_info['n']
    m = int(n * M_RATIO)
    print(f"Cross structure: n={n}, m={m}")

    records = {  # model -> data_type -> list of nmse
        'HyperLISTA': {'elementwise': [], 'tree': []},
        'Tree-HyperLISTA': {'elementwise': [], 'tree': []},
    }

    for seed in range(N_SEEDS):
        print(f"\n[seed {seed + 1}/{N_SEEDS}]")
        ds = TreeSparseDataset(tree_depth=TREE_DEPTH, branching=BRANCHING,
                               m_ratio=M_RATIO, target_sparsity=TARGET_K,
                               snr_db=30.0, seed=seed * 1000, matrix_seed=42)
        A = ds.A

        # --- Tree training regime ---
        tr_t = _gen_tree(ds, N_TRAIN, seed * 1000 + 1)
        va_t = _gen_tree(ds, N_VAL, seed * 1000 + 2)
        te_t = _gen_tree(ds, N_TEST, seed * 1000 + 3)

        # --- Elementwise training regime (same m, n, A) ---
        tr_e = _gen_elementwise(A, n, N_TRAIN, TARGET_K, 30.0,
                                seed * 1000 + 11)
        va_e = _gen_elementwise(A, n, N_VAL, TARGET_K, 30.0,
                                seed * 1000 + 12)
        te_e = _gen_elementwise(A, n, N_TEST, TARGET_K, 30.0,
                                seed * 1000 + 13)

        # Two model cells: HyperLISTA and Tree-HyperLISTA (hybrid tree).
        # Each is tuned against the *matching* training distribution, but
        # tested on BOTH test distributions.
        print("  HyperLISTA tuned on elementwise")
        hl_e = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K_LAYERS},
                                tr_e, va_e, n_trials=N_TRIALS,
                                device=device)['model']
        print("  HyperLISTA tuned on tree")
        hl_t = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K_LAYERS},
                                tr_t, va_t, n_trials=N_TRIALS,
                                device=device)['model']
        print("  Tree-HyperLISTA tuned on elementwise")
        th_e = tune_hyper_model(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'hybrid_tree', 'rho': RHO},
            tr_e, va_e, n_trials=N_TRIALS, device=device)['model']
        print("  Tree-HyperLISTA tuned on tree")
        th_t = tune_hyper_model(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'hybrid_tree', 'rho': RHO},
            tr_t, va_t, n_trials=N_TRIALS, device=device)['model']

        records['HyperLISTA']['elementwise'].append(_eval(hl_e, te_e, device))
        records['HyperLISTA']['tree'].append(_eval(hl_t, te_t, device))
        records['Tree-HyperLISTA']['elementwise'].append(_eval(th_e, te_e, device))
        records['Tree-HyperLISTA']['tree'].append(_eval(th_t, te_t, device))

    final = {}
    for model, by_data in records.items():
        final[model] = {}
        for dt, reps in by_data.items():
            final[model][dt] = {
                'nmse_db_mean': float(np.mean(reps)),
                'nmse_db_std': float(np.std(reps)),
            }

    out_json = 'results/tree_cross/results.json'
    with open(out_json, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n[saved] {out_json}")

    _plot_cross(final)
    return final


def _plot_cross(data):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.2))
    data_types = ['elementwise', 'tree']
    models = ['HyperLISTA', 'Tree-HyperLISTA']
    x = np.arange(len(data_types))
    width = 0.35
    colors = {'HyperLISTA': '#d62728', 'Tree-HyperLISTA': '#e377c2'}
    for i, mdl in enumerate(models):
        means = [data[mdl][dt]['nmse_db_mean'] for dt in data_types]
        stds = [data[mdl][dt]['nmse_db_std'] for dt in data_types]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
               color=colors[mdl], label=mdl, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(['Elementwise data', 'Tree-sparse data'])
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Exp 7: Cross-Structure Stress Test')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig('paper/tree_figures/tree_fig_cross.pdf')
    fig.savefig('paper/tree_figures/tree_fig_cross.png', dpi=200)
    plt.close(fig)
    print("[saved] paper/tree_figures/tree_fig_cross.{pdf,png}")


if __name__ == '__main__':
    from src.utils.sensing import pick_device; dev = pick_device()
    run_cross(device=dev)
