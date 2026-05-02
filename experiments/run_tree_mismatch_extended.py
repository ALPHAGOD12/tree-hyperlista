"""Exp 3 -- Extended mismatch robustness sweep for Tree-HyperLISTA.

Train on one regime, evaluate under shifted distributions along four new
axes beyond the original SNR / sparsity / operator-delta:

    (a) magnitude_distribution:  train nonzeros ~ N(0,1), test with
        amplitude range scale sigma in {1.0, 1.5, 2.0, 3.0}.
    (b) sensing_matrix_type:     train on Gaussian A, test on correlated
        Toeplitz / partial orthogonal / ill-conditioned sensing matrices
        (same m, n, columns normalized).
    (c) tree_topology_mismatch:  train on a balanced binary tree, test
        on several alternative trees (ternary, shuffled parents, deeper
        tree truncated to the same n).
    (d) tree_consistency_violation: test supports have a fraction f of
        their active nodes flipped to random inactive nodes.

Baselines evaluated: LISTA, HyperLISTA, Tree-FISTA, Tree-CoSaMP,
TH-hybrid (tree).
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
    TreeSparseDataset, build_balanced_tree, build_shuffled_tree,
    generate_topology_mismatched, generate_consistency_violated,
)
from src.models.lista import LISTA
from src.models.hyperlista import HyperLISTA
from src.models.tree_hyperlista import TreeHyperLISTA
from src.models.tree_baselines import TreeFISTA
from src.models.tree_classical import TreeCoSaMP
from src.train import train_unfolded_model, tune_hyper_model
from src.utils.metrics import nmse_db
from src.utils.sensing import get_sensing_matrix


TREE_DEPTH = 7
BRANCHING = 2
M_RATIO = 0.5
TARGET_K = 30
K_LAYERS = 16
RHO = 0.5
N_TRAIN, N_VAL, N_TEST = 3000, 600, 600
N_SEEDS = 2
N_TRIALS = 40
TRAIN_EPOCHS = 120


def _eval(model, y, x, device):
    model.eval().to(device) if hasattr(model, 'eval') else model
    with torch.no_grad():
        if hasattr(model, 'solve'):
            xh = model.solve(y)
        else:
            xh = model(y)
    return nmse_db(xh, x)


def _sweep_magnitude(ds, models, device):
    scales = [1.0, 1.5, 2.0, 3.0]
    out = {'values': scales}
    for name in models:
        out[name] = {'nmse': []}
    for s in scales:
        test = ds.generate(N_TEST, seed=999,
                           # override amplitude range by mutating ds tmp
                           )
        # mutate amplitude for this sweep: regenerate manually
        ds2 = TreeSparseDataset(
            tree_depth=TREE_DEPTH, branching=BRANCHING,
            m_ratio=M_RATIO, target_sparsity=TARGET_K,
            snr_db=ds.snr_db, matrix_type=ds.matrix_type,
            amplitude_dist='gaussian',
            amplitude_range=(0.5 * s, 2.0 * s),
            seed=ds.seed, matrix_seed=42)
        # reuse the same A as ds
        ds2.A = ds.A
        ds2.A_tensor = ds.A_tensor
        ds2.A_pinv = ds.A_pinv
        ds2.A_pinv_tensor = ds.A_pinv_tensor
        test = ds2.generate(N_TEST, seed=999)
        y = test['y'].to(device); x = test['x'].to(device)
        for name, m in models.items():
            out[name]['nmse'].append(_eval(m, y, x, device))
    return out


def _sweep_sensing(ds, models, device, n_rows, n_cols):
    types = ['correlated', 'partial_orthogonal', 'ill_conditioned']
    out = {'values': types}
    for name in models:
        out[name] = {'nmse': []}
    for t in types:
        try:
            A_new = get_sensing_matrix(n_rows, n_cols, matrix_type=t,
                                       seed=123)
        except Exception as e:
            print(f"    {t}: {e}")
            for name in models:
                out[name]['nmse'].append(float('nan'))
            continue
        # generate tree-sparse signals, then y = A_new x
        rng = np.random.RandomState(999)
        from src.data.tree_synthetic import generate_tree_support
        tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)
        X = np.zeros((N_TEST, n_cols), dtype=np.float32)
        for i in range(N_TEST):
            support = generate_tree_support(tree_info, TARGET_K, rng)
            active = np.where(support)[0]
            vals = rng.randn(len(active)).astype(np.float32)
            X[i, active] = np.sign(vals) * (np.abs(vals) * 1.5 + 0.5)
        Y_clean = X @ A_new.T
        sigma = np.sqrt(np.mean(Y_clean ** 2) * 10 ** (-ds.snr_db / 10.0))
        Y = (Y_clean + rng.randn(N_TEST, n_rows).astype(np.float32) * sigma)
        Y = Y.astype(np.float32)
        y = torch.from_numpy(Y).float().to(device)
        x = torch.from_numpy(X.astype(np.float32)).float().to(device)
        for name, m in models.items():
            out[name]['nmse'].append(_eval(m, y, x, device))
    return out


def _sweep_topology(A, models, device):
    tree_train = build_balanced_tree(TREE_DEPTH, BRANCHING)
    variants = {
        'shuffled_parents': build_shuffled_tree(tree_train, seed=123),
        # Use a ternary tree with similar n: depth 5 -> n = (3**6 - 1)/2 = 364
        # but we must keep same n; truncate to n = tree_train['n']
    }
    # For same-n alternatives, we'll use two types:
    # 1. ternary depth 5 truncated to the same n
    try:
        tern = build_balanced_tree(5, 3)
        if tern['n'] >= tree_train['n']:
            # truncate: reuse first n nodes with their parents (valid BFS order)
            n_t = tree_train['n']
            parent = tern['parent'][:n_t].copy()
            parent[parent >= n_t] = 0
            children = [[] for _ in range(n_t)]
            depth_arr = np.zeros(n_t, dtype=np.int64)
            for i in range(1, n_t):
                children[parent[i]].append(i)
                depth_arr[i] = depth_arr[parent[i]] + 1
            variants['ternary_truncated'] = {
                'n': n_t, 'parent': parent, 'children': children,
                'depth': depth_arr, 'leaves': [i for i in range(n_t) if not children[i]],
                'descendants': [set() for _ in range(n_t)],
                'max_depth': int(depth_arr.max()), 'branching': 3,
            }
    except Exception as e:
        print(f"    ternary construction failed: {e}")

    out = {'values': list(variants.keys())}
    for name in models:
        out[name] = {'nmse': []}
    for vname, tree_test in variants.items():
        test = generate_topology_mismatched(
            tree_train, tree_test, A, N_TEST, TARGET_K, 30.0, seed=999)
        y = test['y'].to(device); x = test['x'].to(device)
        for name, m in models.items():
            out[name]['nmse'].append(_eval(m, y, x, device))
    return out


def _sweep_consistency(A, tree_info, models, device):
    fracs = [0.0, 0.1, 0.25, 0.5]
    out = {'values': fracs}
    for name in models:
        out[name] = {'nmse': []}
    for f in fracs:
        test = generate_consistency_violated(
            tree_info, A, N_TEST, TARGET_K, 30.0, f, seed=999)
        y = test['y'].to(device); x = test['x'].to(device)
        for name, m in models.items():
            out[name]['nmse'].append(_eval(m, y, x, device))
    return out


def run_extended_mismatch(device='cpu'):
    os.makedirs('results/tree_mismatch_ext', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)

    tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)
    n = tree_info['n']
    m = int(n * M_RATIO)
    print(f"Extended mismatch: n={n}, m={m}")

    all_axes = {'magnitude': {}, 'sensing': {}, 'topology': {}, 'consistency': {}}

    for seed in range(N_SEEDS):
        print(f"\n[seed {seed + 1}/{N_SEEDS}]")
        ds = TreeSparseDataset(tree_depth=TREE_DEPTH, branching=BRANCHING,
                               m_ratio=M_RATIO, target_sparsity=TARGET_K,
                               snr_db=30.0, seed=seed * 1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(N_TRAIN, seed=seed * 1000 + 1)
        va = ds.generate(N_VAL, seed=seed * 1000 + 2)

        print("  training LISTA")
        lista = LISTA(A, num_layers=K_LAYERS)
        train_unfolded_model(lista, tr, va, num_epochs=TRAIN_EPOCHS,
                             lr=1e-3, batch_size=128, device=device)
        print("  tuning HyperLISTA")
        hl = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K_LAYERS},
                              tr, va, n_trials=N_TRIALS, device=device)['model']
        print("  tuning Tree-HyperLISTA (hybrid)")
        th = tune_hyper_model(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'hybrid_tree', 'rho': RHO},
            tr, va, n_trials=N_TRIALS, device=device)['model']

        tf = TreeFISTA(A, tree_info, lam=0.05, max_iter=K_LAYERS,
                       rho=RHO, target_K=TARGET_K).to(device)
        tcos = TreeCoSaMP(A, tree_info, target_K=TARGET_K,
                          max_iter=K_LAYERS, rho=RHO).to(device)

        models = {
            'LISTA': lista, 'HyperLISTA': hl, 'Tree-FISTA': tf,
            'Tree-CoSaMP': tcos, 'TH-hybrid': th,
        }

        print("  axis: magnitude")
        mag = _sweep_magnitude(ds, models, device)
        print("  axis: sensing matrix")
        sens = _sweep_sensing(ds, models, device, m, n)
        print("  axis: tree topology")
        topo = _sweep_topology(A, models, device)
        print("  axis: tree consistency")
        cons = _sweep_consistency(A, tree_info, models, device)

        for axis, res in [('magnitude', mag), ('sensing', sens),
                          ('topology', topo), ('consistency', cons)]:
            if 'values' not in all_axes[axis]:
                all_axes[axis]['values'] = res['values']
            for name, d in res.items():
                if name == 'values':
                    continue
                all_axes[axis].setdefault(name, []).append(d['nmse'])

    final = {}
    for axis, data in all_axes.items():
        final[axis] = {'values': data.get('values', [])}
        for name, reps in data.items():
            if name == 'values':
                continue
            arr = np.array(reps, dtype=float)
            final[axis][name] = {
                'nmse_db_mean': np.nanmean(arr, axis=0).tolist(),
                'nmse_db_std': np.nanstd(arr, axis=0).tolist(),
            }

    out_json = 'results/tree_mismatch_ext/results.json'
    with open(out_json, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n[saved] {out_json}")

    _plot_mm_ext(final)
    return final


def _plot_mm_ext(data):
    axis_info = {
        'magnitude': ('Amplitude scale $\\sigma$', 'Magnitude mismatch'),
        'sensing': ('Sensing matrix type', 'Sensing matrix mismatch'),
        'topology': ('Test tree topology', 'Tree topology mismatch'),
        'consistency': ('Fraction violating parent-before-child',
                        'Tree consistency violation'),
    }
    colors = {
        'LISTA': '#1f77b4', 'HyperLISTA': '#d62728',
        'Tree-FISTA': '#666666', 'Tree-CoSaMP': '#17becf',
        'TH-hybrid': '#e377c2',
    }
    markers = {
        'LISTA': 's', 'HyperLISTA': 'p', 'Tree-FISTA': '^',
        'Tree-CoSaMP': 'h', 'TH-hybrid': '*',
    }
    for axis, (xlab, title) in axis_info.items():
        d = data.get(axis, {})
        vals = d.get('values', [])
        if not vals:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
        x = list(range(len(vals)))
        for name, series in d.items():
            if name == 'values':
                continue
            means = series['nmse_db_mean']
            stds = series['nmse_db_std']
            ax.errorbar(x, means, yerr=stds,
                        color=colors.get(name, '#333'),
                        marker=markers.get(name, 'o'), markersize=7,
                        linewidth=1.6, capsize=3, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in vals], rotation=20)
        ax.set_xlabel(xlab)
        ax.set_ylabel('NMSE (dB)')
        ax.set_title(f'Exp 3 -- {title}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(f'paper/tree_figures/tree_fig_mm_{axis}.pdf')
        fig.savefig(f'paper/tree_figures/tree_fig_mm_{axis}.png', dpi=200)
        plt.close(fig)
        print(f"[saved] paper/tree_figures/tree_fig_mm_{axis}.{{pdf,png}}")


if __name__ == '__main__':
    from src.utils.sensing import pick_device; dev = pick_device()
    run_extended_mismatch(device=dev)
