"""Exp 4 -- Layer extrapolation for Tree-HyperLISTA.

All models are trained/tuned with K = 16 layers. At test time, each model
is unrolled for K in {16, 24, 32, 48, 64} without any additional training.

  - LISTA: per-layer weights; extra layers reuse the last trained layer's
           weights (the "ALISTA-Extra" strategy from the HyperLISTA paper).
  - HyperLISTA / Tree-HyperLISTA / TreeHyperLISTA-Elem: the three learned
    hyperparameters are layer-invariant, so extrapolation is native.

Output
------
  results/tree_extrapolation/results.json
  paper/tree_figures/tree_fig_extrapolation.pdf/png
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
from src.models.lista import LISTA
from src.models.hyperlista import HyperLISTA
from src.models.tree_hyperlista import TreeHyperLISTA
from src.models.tree_ablation_variants import TreeHyperLISTAElem
from src.train import train_unfolded_model, tune_hyper_model
from src.utils.metrics import nmse_db


TREE_DEPTH = 7
BRANCHING = 2
M_RATIO = 0.5
TARGET_K = 30
K_TRAIN = 16
K_EVALS = [16, 24, 32, 48, 64]
RHO = 0.5
N_TRAIN, N_VAL, N_TEST = 3000, 600, 1000
N_SEEDS = 3
N_TRIALS = 40
TRAIN_EPOCHS = 120


def _extend_lista_weights(lista: LISTA, target_K: int) -> LISTA:
    """Create a new LISTA whose extra layers replicate the last trained layer."""
    from copy import deepcopy
    import torch.nn as nn
    m, n = lista.m, lista.n
    new = LISTA.__new__(LISTA)
    nn.Module.__init__(new)
    new.n, new.m = n, m
    new.num_layers = target_K
    new.W1 = nn.ParameterList()
    new.W2 = nn.ParameterList()
    new.thresholds = nn.ParameterList()
    last = lista.num_layers - 1
    for k in range(target_K):
        src = min(k, last)
        new.W1.append(nn.Parameter(lista.W1[src].detach().clone()))
        new.W2.append(nn.Parameter(lista.W2[src].detach().clone()))
        new.thresholds.append(nn.Parameter(lista.thresholds[src].detach().clone()))
    return new


def _final_nmse(model, y, x, K, device):
    model.eval().to(device)
    with torch.no_grad():
        out = model(y, num_layers=K)
    return nmse_db(out, x)


def run_extrapolation(device='cpu'):
    os.makedirs('results/tree_extrapolation', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)

    tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)
    n = tree_info['n']
    m = int(n * M_RATIO)
    print(f"Layer extrapolation: train K={K_TRAIN}, test K in {K_EVALS} "
          f"| n={n}, m={m}")

    results = defaultdict(lambda: defaultdict(list))  # [name][K] -> [nmse...]

    for seed in range(N_SEEDS):
        print(f"\n[seed {seed + 1}/{N_SEEDS}]")
        ds = TreeSparseDataset(tree_depth=TREE_DEPTH, branching=BRANCHING,
                               m_ratio=M_RATIO, target_sparsity=TARGET_K,
                               snr_db=30.0, seed=seed * 1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(N_TRAIN, seed=seed * 1000 + 1)
        va = ds.generate(N_VAL, seed=seed * 1000 + 2)
        te = ds.generate(N_TEST, seed=seed * 1000 + 3)
        y = te['y'].to(device)
        x = te['x'].to(device)

        print("  training LISTA @ K=16")
        lista = LISTA(A, num_layers=K_TRAIN)
        train_unfolded_model(lista, tr, va, num_epochs=TRAIN_EPOCHS,
                             lr=1e-3, batch_size=128, device=device)

        print("  tuning HyperLISTA / TH-Elem / TH-Tree")
        hl = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K_TRAIN},
                              tr, va, n_trials=N_TRIALS, device=device)['model']
        thE = tune_hyper_model(TreeHyperLISTAElem,
                               {'A': A, 'num_layers': K_TRAIN},
                               tr, va, n_trials=N_TRIALS, device=device)['model']
        thT = tune_hyper_model(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_TRAIN,
             'support_mode': 'hybrid_tree', 'rho': RHO},
            tr, va, n_trials=N_TRIALS, device=device)['model']

        for K in K_EVALS:
            lista_ext = _extend_lista_weights(lista, K).to(device)
            results['LISTA-Extra'][K].append(_final_nmse(lista_ext, y, x, K, device))
            results['HyperLISTA'][K].append(_final_nmse(hl, y, x, K, device))
            results['Tree-HyperLISTA-Elem'][K].append(_final_nmse(thE, y, x, K, device))
            results['Tree-HyperLISTA'][K].append(_final_nmse(thT, y, x, K, device))

    agg = {}
    for name, by_k in results.items():
        agg[name] = {
            'K': K_EVALS,
            'nmse_db_mean': [float(np.mean(by_k[K])) for K in K_EVALS],
            'nmse_db_std': [float(np.std(by_k[K])) for K in K_EVALS],
        }
    out_json = 'results/tree_extrapolation/results.json'
    with open(out_json, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\n[saved] {out_json}")

    _plot_extrapolation(agg)
    return agg


def _plot_extrapolation(agg):
    colors = {
        'LISTA-Extra': '#1f77b4', 'HyperLISTA': '#d62728',
        'Tree-HyperLISTA-Elem': '#8172b2', 'Tree-HyperLISTA': '#e377c2',
    }
    markers = {
        'LISTA-Extra': 's', 'HyperLISTA': 'p',
        'Tree-HyperLISTA-Elem': 'D', 'Tree-HyperLISTA': '*',
    }
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.4))
    for name, res in agg.items():
        ax.errorbar(res['K'], res['nmse_db_mean'], yerr=res['nmse_db_std'],
                    color=colors.get(name, '#333'),
                    marker=markers.get(name, 'o'), markersize=7,
                    linewidth=1.6, capsize=3, label=name)
    ax.axvline(K_TRAIN, color='gray', linestyle='--', alpha=0.7,
               label=f'Trained K={K_TRAIN}')
    ax.set_xlabel('Test-time Unrolling K')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Exp 4: Layer Extrapolation (trained at K=16)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig('paper/tree_figures/tree_fig_extrapolation.pdf')
    fig.savefig('paper/tree_figures/tree_fig_extrapolation.png', dpi=200)
    plt.close(fig)
    print("[saved] paper/tree_figures/tree_fig_extrapolation.{pdf,png}")


if __name__ == '__main__':
    from src.utils.sensing import pick_device; dev = pick_device()
    run_extrapolation(device=dev)
