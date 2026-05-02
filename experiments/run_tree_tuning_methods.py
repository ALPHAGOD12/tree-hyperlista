"""Exp 2 -- Tuning-method comparison for Tree-HyperLISTA.

Three ways of fitting ``(c1, c2, c3)`` are benchmarked on the same
architecture and data:

    TH-BP : optimise (c1, c2, c3) directly by backprop on MSE.
    TH-GS : coarse + fine grid search over (c1, c2, c3).
    TH-BO : Bayesian optimisation via Optuna (the default tuner).

Metrics reported: final test NMSE (dB), tuning wall-clock time, number
of model evaluations.

Output
------
  results/tree_tuning/results.json
  paper/tree_figures/tree_fig_tuning.pdf/png
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
from src.train import (
    tune_hyper_model, tune_hyper_grid_search, tune_hyper_backprop,
)
from src.utils.metrics import nmse_db


TREE_DEPTH = 7
BRANCHING = 2
M_RATIO = 0.5
TARGET_K = 30
K_LAYERS = 16
RHO = 0.5
N_TRAIN, N_VAL, N_TEST = 3000, 600, 1000
N_SEEDS = 3
N_TRIALS_BO = 60


def _eval(model, te, device):
    model.eval().to(device)
    with torch.no_grad():
        xh = model(te['y'].to(device))
    return nmse_db(xh, te['x'].to(device))


def run_tuning_methods(device='cpu'):
    os.makedirs('results/tree_tuning', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)

    tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)

    records = {'TH-BP': {'nmse': [], 'time': [], 'evals': []},
               'TH-GS': {'nmse': [], 'time': [], 'evals': []},
               'TH-BO': {'nmse': [], 'time': [], 'evals': []}}

    for seed in range(N_SEEDS):
        print(f"\n[seed {seed + 1}/{N_SEEDS}]")
        ds = TreeSparseDataset(tree_depth=TREE_DEPTH, branching=BRANCHING,
                               m_ratio=M_RATIO, target_sparsity=TARGET_K,
                               snr_db=30.0, seed=seed * 1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(N_TRAIN, seed=seed * 1000 + 1)
        va = ds.generate(N_VAL, seed=seed * 1000 + 2)
        te = ds.generate(N_TEST, seed=seed * 1000 + 3)
        model_kwargs = {'A': A, 'tree_info': tree_info,
                        'num_layers': K_LAYERS,
                        'support_mode': 'hybrid_tree', 'rho': RHO}

        print("  TH-BP (backprop)")
        bp = tune_hyper_backprop(TreeHyperLISTA, model_kwargs, tr, va,
                                 num_epochs=80, lr=5e-3, device=device)
        records['TH-BP']['nmse'].append(_eval(bp['model'], te, device))
        records['TH-BP']['time'].append(bp['tune_time_s'])
        records['TH-BP']['evals'].append(bp['n_evals'])

        print("  TH-GS (grid search)")
        gs = tune_hyper_grid_search(TreeHyperLISTA, model_kwargs, tr, va,
                                    device=device)
        records['TH-GS']['nmse'].append(_eval(gs['model'], te, device))
        records['TH-GS']['time'].append(gs['tune_time_s'])
        records['TH-GS']['evals'].append(gs['n_evals'])

        print("  TH-BO (Bayesian / Optuna)")
        t0 = time.time()
        bo = tune_hyper_model(TreeHyperLISTA, model_kwargs, tr, va,
                              n_trials=N_TRIALS_BO, device=device)
        bo_time = time.time() - t0
        records['TH-BO']['nmse'].append(_eval(bo['model'], te, device))
        records['TH-BO']['time'].append(bo_time)
        records['TH-BO']['evals'].append(N_TRIALS_BO)

    final = {}
    for name, r in records.items():
        final[name] = {
            'nmse_db_mean': float(np.mean(r['nmse'])),
            'nmse_db_std': float(np.std(r['nmse'])),
            'tune_time_mean': float(np.mean(r['time'])),
            'tune_time_std': float(np.std(r['time'])),
            'n_evals_mean': float(np.mean(r['evals'])),
        }

    out_json = 'results/tree_tuning/results.json'
    with open(out_json, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n[saved] {out_json}")

    _plot_tuning(final)
    return final


def _plot_tuning(data):
    names = ['TH-BP', 'TH-GS', 'TH-BO']
    nmse = [data[n]['nmse_db_mean'] for n in names]
    nmse_err = [data[n]['nmse_db_std'] for n in names]
    times = [data[n]['tune_time_mean'] for n in names]
    t_err = [data[n]['tune_time_std'] for n in names]
    colors = ['#1f77b4', '#2ca02c', '#e377c2']

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(names))
    a1.bar(x, nmse, yerr=nmse_err, color=colors, capsize=3,
           edgecolor='black', linewidth=0.5)
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylabel('NMSE (dB)')
    a1.set_title('Final Test NMSE')
    a1.grid(True, axis='y', alpha=0.3)

    a2.bar(x, times, yerr=t_err, color=colors, capsize=3,
           edgecolor='black', linewidth=0.5)
    a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_ylabel('Tuning wall time (s)')
    a2.set_title('Tuning Compute')
    a2.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Exp 2: Tuning Method Comparison (BP vs GS vs BO)')
    fig.tight_layout()
    fig.savefig('paper/tree_figures/tree_fig_tuning.pdf')
    fig.savefig('paper/tree_figures/tree_fig_tuning.png', dpi=200)
    plt.close(fig)
    print("[saved] paper/tree_figures/tree_fig_tuning.{pdf,png}")


if __name__ == '__main__':
    from src.utils.sensing import pick_device; dev = pick_device()
    run_tuning_methods(device=dev)
