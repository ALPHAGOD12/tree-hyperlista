"""
Scalability experiments for Tree-HyperLISTA at larger tree sizes (n=1023, n=4095).
Compares Tree-HyperLISTA vs Tree-IHT, Tree-FISTA, Tree-CoSaMP, and LISTA.
Reports NMSE and wall-clock time.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
import time
from collections import defaultdict

from src.data.tree_synthetic import TreeSparseDataset, build_balanced_tree
from src.models.lista import LISTA
from src.models.hyperlista import HyperLISTA
from src.models.tree_hyperlista import TreeHyperLISTA
from src.models.tree_baselines import TreeFISTA
from src.models.tree_classical import TreeIHT, TreeCoSaMP
from src.train import train_unfolded_model, tune_hyper_model
from src.utils.metrics import nmse_db, node_precision_recall, count_parameters

K_LAYERS = 16
N_SEEDS = 3
N_TRIALS = 50
TRAIN_EPOCHS = 150
RHO = 0.5

SCALE_CONFIGS = [
    {'depth': 7, 'branching': 2, 'n': 255, 'm_ratio': 0.5, 'target_K': 30},
    {'depth': 9, 'branching': 2, 'n': 1023, 'm_ratio': 0.4, 'target_K': 80},
]


def timed_evaluate(model, y_test, x_test, device, num_runs=3):
    """Evaluate model with timing."""
    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model = model.to(device)

    y = y_test.to(device)
    x = x_test.to(device)

    with torch.no_grad():
        if hasattr(model, 'solve'):
            _ = model.solve(y[:10])
        else:
            _ = model(y[:10])

    times = []
    for _ in range(num_runs):
        t0 = time.time()
        with torch.no_grad():
            if hasattr(model, 'solve'):
                x_hat = model.solve(y)
            else:
                x_hat = model(y)
        times.append(time.time() - t0)

    val_nmse = nmse_db(x_hat, x)
    prec, rec = node_precision_recall(x_hat, x)
    avg_time = np.mean(times)

    return {
        'nmse_db': val_nmse,
        'precision': prec,
        'recall': rec,
        'time_sec': avg_time,
        'num_params': count_parameters(model) if hasattr(model, 'parameters') else 0,
    }


def run_scale_experiment(config, device):
    """Run experiment for a single scale configuration."""
    depth = config['depth']
    branching = config['branching']
    n = config['n']
    m_ratio = config['m_ratio']
    target_K = config['target_K']

    tree_info = build_balanced_tree(depth, branching)
    actual_n = tree_info['n']
    m = int(actual_n * m_ratio)
    print(f"\n  Scale: depth={depth}, n={actual_n}, m={m}, K={target_K}")

    results_all = defaultdict(lambda: defaultdict(list))

    for seed in range(N_SEEDS):
        print(f"\n    Seed {seed+1}/{N_SEEDS}")

        ds = TreeSparseDataset(tree_depth=depth, branching=branching,
                               m_ratio=m_ratio, target_sparsity=target_K,
                               snr_db=30.0, seed=seed*1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(4000, seed=seed*1000+1)
        va = ds.generate(800, seed=seed*1000+2)
        te = ds.generate(1000, seed=seed*1000+3)

        models = {}

        best_lam, best_nmse = 0.1, float('inf')
        for lam_try in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
            g = TreeFISTA(A, tree_info, lam=lam_try, max_iter=K_LAYERS,
                          rho=RHO, target_K=target_K)
            g = g.to(device)
            with torch.no_grad():
                xh = g.solve(va['y'][:200].to(device))
                v = nmse_db(xh, va['x'][:200].to(device))
            if v < best_nmse:
                best_nmse = v
                best_lam = lam_try

        print(f"      Tree-FISTA (lam={best_lam})")
        models['Tree-FISTA'] = TreeFISTA(A, tree_info, lam=best_lam,
                                          max_iter=K_LAYERS, rho=RHO, target_K=target_K)

        print("      Tree-IHT")
        models['Tree-IHT'] = TreeIHT(A, tree_info, target_K=target_K,
                                      max_iter=K_LAYERS, rho=RHO)

        print("      Tree-CoSaMP")
        models['Tree-CoSaMP'] = TreeCoSaMP(A, tree_info, target_K=target_K,
                                            max_iter=K_LAYERS, rho=RHO)

        if actual_n <= 1100:
            print("      LISTA")
            lista = LISTA(A, num_layers=K_LAYERS)
            train_unfolded_model(lista, tr, va, num_epochs=TRAIN_EPOCHS,
                                 lr=1e-3, batch_size=64, device=device)
            models['LISTA'] = lista

        print("      HyperLISTA")
        hl = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K_LAYERS},
                              tr, va, n_trials=N_TRIALS, device=device)
        models['HyperLISTA'] = hl['model']

        print("      Tree-HyperLISTA (hybrid)")
        th = tune_hyper_model(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'hybrid_tree', 'rho': RHO},
            tr, va, n_trials=N_TRIALS, device=device)
        models['TH-hybrid'] = th['model']

        print("      Tree-HyperLISTA (hard)")
        th_hard = tune_hyper_model(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'tree_hard', 'rho': RHO},
            tr, va, n_trials=N_TRIALS, device=device)
        models['TH-hard'] = th_hard['model']

        print("      Evaluating...")
        for name, mdl in models.items():
            res = timed_evaluate(mdl, te['y'], te['x'], device)
            for k, v in res.items():
                results_all[name][k].append(v)
            print(f"        {name}: {res['nmse_db']:.2f} dB, "
                  f"time={res['time_sec']:.3f}s, params={res['num_params']}")

    return {
        name: {
            k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for k, v in results_all[name].items()
        }
        for name in results_all
    }, actual_n


def run_all_scales(device='cpu'):
    os.makedirs('results/tree_scaled', exist_ok=True)

    all_results = {}

    for config in SCALE_CONFIGS:
        results, actual_n = run_scale_experiment(config, device)
        all_results[f'n={actual_n}'] = results

        print(f"\n  Summary for n={actual_n}:")
        print(f"  {'Model':<20} {'NMSE(dB)':<16} {'Time(s)':<12} {'#Params':<10}")
        print("  " + "=" * 58)
        for name in results:
            r = results[name]
            print(f"  {name:<20} "
                  f"{r['nmse_db']['mean']:>7.2f}+/-{r['nmse_db']['std']:.2f}  "
                  f"{r['time_sec']['mean']:>7.3f}  "
                  f"{r['num_params']['mean']:.0f}")

    with open('results/tree_scaled/scaled_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nScaled results saved to results/tree_scaled/scaled_results.json")
    return all_results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    run_all_scales(device=device)
