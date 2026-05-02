"""Exp 8 -- Sparsity budget mismatch for Tree-HyperLISTA.

Tree-HyperLISTA's layerwise budget ``K_k`` is driven by hyperparameter
``c3``, tuned on a training distribution with sparsity ``K_tuned``. At
test time we sweep the *true* signal sparsity ``K_true`` and report the
resulting NMSE, for ratios ``K_true / K_tuned`` in
``{0.5, 0.75, 1.0, 1.5, 2.0}``.

Also compares against TH-hard and TH-hybrid to see which proximal is
more robust to sparsity drift, and against Tree-CoSaMP which requires
an explicit ``K``.

Output
------
  results/tree_sparsity/results.json
  paper/tree_figures/tree_fig_sparsity.pdf/png
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

from src.data.tree_synthetic import TreeSparseDataset, build_balanced_tree
from src.models.hyperlista import HyperLISTA
from src.models.tree_hyperlista import TreeHyperLISTA
from src.models.tree_classical import TreeCoSaMP
from src.train import tune_hyper_model
from src.utils.metrics import nmse_db


TREE_DEPTH = 7
BRANCHING = 2
M_RATIO = 0.5
K_TUNED = 30
RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]
K_LAYERS = 16
RHO = 0.5
N_TRAIN, N_VAL, N_TEST = 3000, 600, 600
N_SEEDS = 3
N_TRIALS = 40


def _eval(model, y, x, device):
    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model = model.to(device)
    with torch.no_grad():
        xh = model.solve(y) if hasattr(model, 'solve') else model(y)
    return nmse_db(xh, x)


def run_sparsity(device='cpu'):
    os.makedirs('results/tree_sparsity', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)

    tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)
    print(f"Sparsity mismatch: K_tuned={K_TUNED}, ratios={RATIOS}")

    records = {}  # model -> ratio -> list
    names = ['HyperLISTA', 'TH-hard', 'TH-hybrid', 'Tree-CoSaMP']
    for nm in names:
        records[nm] = {r: [] for r in RATIOS}

    for seed in range(N_SEEDS):
        print(f"\n[seed {seed + 1}/{N_SEEDS}]")
        ds = TreeSparseDataset(tree_depth=TREE_DEPTH, branching=BRANCHING,
                               m_ratio=M_RATIO, target_sparsity=K_TUNED,
                               snr_db=30.0, seed=seed * 1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(N_TRAIN, seed=seed * 1000 + 1)
        va = ds.generate(N_VAL, seed=seed * 1000 + 2)

        print("  tuning HyperLISTA / TH-hard / TH-hybrid")
        hl = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K_LAYERS},
                              tr, va, n_trials=N_TRIALS,
                              device=device)['model']
        th_hard = tune_hyper_model(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'tree_hard', 'rho': RHO},
            tr, va, n_trials=N_TRIALS, device=device)['model']
        th_hybrid = tune_hyper_model(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'hybrid_tree', 'rho': RHO},
            tr, va, n_trials=N_TRIALS, device=device)['model']
        tcos = TreeCoSaMP(A, tree_info, target_K=K_TUNED,
                          max_iter=K_LAYERS, rho=RHO).to(device)

        for r in RATIOS:
            K_true = max(2, int(round(K_TUNED * r)))
            te = ds.generate(N_TEST, seed=seed * 1000 + 50,
                             target_sparsity=K_true)
            y = te['y'].to(device); x = te['x'].to(device)
            records['HyperLISTA'][r].append(_eval(hl, y, x, device))
            records['TH-hard'][r].append(_eval(th_hard, y, x, device))
            records['TH-hybrid'][r].append(_eval(th_hybrid, y, x, device))
            records['Tree-CoSaMP'][r].append(_eval(tcos, y, x, device))

    final = {'ratios': RATIOS}
    for name, by_r in records.items():
        final[name] = {
            'nmse_db_mean': [float(np.mean(by_r[r])) for r in RATIOS],
            'nmse_db_std': [float(np.std(by_r[r])) for r in RATIOS],
        }

    out_json = 'results/tree_sparsity/results.json'
    with open(out_json, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n[saved] {out_json}")

    _plot_sparsity(final)
    return final


def _plot_sparsity(data):
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
    colors = {'HyperLISTA': '#d62728', 'TH-hard': '#9467bd',
              'TH-hybrid': '#e377c2', 'Tree-CoSaMP': '#17becf'}
    markers = {'HyperLISTA': 'p', 'TH-hard': 'P',
               'TH-hybrid': '*', 'Tree-CoSaMP': 'h'}
    ratios = data['ratios']
    for name in colors:
        if name not in data:
            continue
        ax.errorbar(ratios, data[name]['nmse_db_mean'],
                    yerr=data[name]['nmse_db_std'],
                    color=colors[name], marker=markers[name], markersize=7,
                    linewidth=1.6, capsize=3, label=name)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel(r'$K_\mathrm{true} / K_\mathrm{tuned}$')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Exp 8: Sparsity Budget Mismatch')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig('paper/tree_figures/tree_fig_sparsity.pdf')
    fig.savefig('paper/tree_figures/tree_fig_sparsity.png', dpi=200)
    plt.close(fig)
    print("[saved] paper/tree_figures/tree_fig_sparsity.{pdf,png}")


if __name__ == '__main__':
    from src.utils.sensing import pick_device; dev = pick_device()
    run_sparsity(device=dev)
