"""Exp 12 -- Low-SNR noise robustness sweep for Tree-HyperLISTA.

Train each model at SNR = 30 dB, evaluate across
SNR in {0, 5, 10, 15, 20, 25, 30, 40} dB. Classical tree methods like
Tree-CoSaMP tend to degrade sharply at low SNR; we test whether learned
hyperparameters help Tree-HyperLISTA stay accurate.

Output
------
  results/tree_lowsnr/results.json
  paper/tree_figures/tree_fig_lowsnr.pdf/png
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
from src.models.tree_baselines import TreeFISTA
from src.models.tree_classical import TreeCoSaMP
from src.train import train_unfolded_model, tune_hyper_model
from src.utils.metrics import nmse_db


TREE_DEPTH = 7
BRANCHING = 2
M_RATIO = 0.5
TARGET_K = 30
K_LAYERS = 16
RHO = 0.5
N_TRAIN, N_VAL, N_TEST = 3000, 600, 600
N_SEEDS = 3
N_TRIALS = 40
SNR_GRID = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]


def _eval(model, y, x, device):
    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model = model.to(device)
    with torch.no_grad():
        xh = model.solve(y) if hasattr(model, 'solve') else model(y)
    return nmse_db(xh, x)


def run_lowsnr(device='cpu'):
    os.makedirs('results/tree_lowsnr', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)
    tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)

    records = {}  # model -> snr -> list
    names = ['HyperLISTA', 'Tree-FISTA', 'Tree-CoSaMP', 'TH-hybrid']
    for nm in names:
        records[nm] = {s: [] for s in SNR_GRID}

    for seed in range(N_SEEDS):
        print(f"\n[seed {seed + 1}/{N_SEEDS}]")
        ds = TreeSparseDataset(tree_depth=TREE_DEPTH, branching=BRANCHING,
                               m_ratio=M_RATIO, target_sparsity=TARGET_K,
                               snr_db=30.0, seed=seed * 1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(N_TRAIN, seed=seed * 1000 + 1)
        va = ds.generate(N_VAL, seed=seed * 1000 + 2)

        print("  tuning HyperLISTA")
        hl = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K_LAYERS},
                              tr, va, n_trials=N_TRIALS,
                              device=device)['model']
        print("  tuning TH-hybrid")
        th = tune_hyper_model(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'hybrid_tree', 'rho': RHO},
            tr, va, n_trials=N_TRIALS, device=device)['model']
        tf = TreeFISTA(A, tree_info, lam=0.05, max_iter=K_LAYERS,
                       rho=RHO, target_K=TARGET_K).to(device)
        tc = TreeCoSaMP(A, tree_info, target_K=TARGET_K,
                        max_iter=K_LAYERS, rho=RHO).to(device)

        for snr in SNR_GRID:
            te = ds.generate(N_TEST, seed=seed * 1000 + 100,
                             snr_db=snr)
            y = te['y'].to(device); x = te['x'].to(device)
            records['HyperLISTA'][snr].append(_eval(hl, y, x, device))
            records['Tree-FISTA'][snr].append(_eval(tf, y, x, device))
            records['Tree-CoSaMP'][snr].append(_eval(tc, y, x, device))
            records['TH-hybrid'][snr].append(_eval(th, y, x, device))

    final = {'snr_db': SNR_GRID}
    for name, by_s in records.items():
        final[name] = {
            'nmse_db_mean': [float(np.mean(by_s[s])) for s in SNR_GRID],
            'nmse_db_std': [float(np.std(by_s[s])) for s in SNR_GRID],
        }

    out_json = 'results/tree_lowsnr/results.json'
    with open(out_json, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n[saved] {out_json}")
    _plot_lowsnr(final)
    return final


def _plot_lowsnr(data):
    colors = {'HyperLISTA': '#d62728', 'Tree-FISTA': '#666666',
              'Tree-CoSaMP': '#17becf', 'TH-hybrid': '#e377c2'}
    markers = {'HyperLISTA': 'p', 'Tree-FISTA': '^',
               'Tree-CoSaMP': 'h', 'TH-hybrid': '*'}
    snr = data['snr_db']
    fig, ax = plt.subplots(1, 1, figsize=(6.3, 4.2))
    for name in colors:
        if name not in data:
            continue
        ax.errorbar(snr, data[name]['nmse_db_mean'],
                    yerr=data[name]['nmse_db_std'],
                    color=colors[name], marker=markers[name],
                    markersize=7, linewidth=1.6, capsize=3, label=name)
    ax.set_xlabel('Test SNR (dB)')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Exp 12: Low-SNR Robustness')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig('paper/tree_figures/tree_fig_lowsnr.pdf')
    fig.savefig('paper/tree_figures/tree_fig_lowsnr.png', dpi=200)
    plt.close(fig)
    print("[saved] paper/tree_figures/tree_fig_lowsnr.{pdf,png}")


if __name__ == '__main__':
    from src.utils.sensing import pick_device; dev = pick_device()
    run_lowsnr(device=dev)
