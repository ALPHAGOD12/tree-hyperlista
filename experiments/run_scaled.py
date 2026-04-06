"""
Scaled experiments at n=2048 for GPU execution (Colab T4 or similar).

Runs only the feasible models at large scale:
  Group-FISTA, ALISTA, Ada-BlockLISTA (tied), HyperLISTA, SH-topk, SH-hybrid
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
import time
from collections import defaultdict

from src.data.synthetic import BlockSparseDataset
from src.models.ista import GroupFISTA
from src.models.alista import ALISTA
from src.models.hyperlista import HyperLISTA
from src.models.struct_hyperlista import StructHyperLISTA
from src.models.ada_blocklista import AdaBlockLISTA
from src.train import train_unfolded_model, tune_hyper_model
from src.evaluate import evaluate_model, evaluate_trajectory
from src.utils.metrics import nmse_db, count_parameters

N, M = 2048, 1024
GS, SG = 16, 32
K = 16
N_TRAIN, N_VAL, N_TEST = 8000, 1000, 2000
N_SEEDS = 3
N_TRIALS = 60
TRAIN_EPOCHS = 80


def build_and_train_scaled(A, gs, K, tr, va, device):
    models = {}

    best_lam, best_nmse = 0.1, float('inf')
    for lam_try in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        g = GroupFISTA(A, gs, lam=lam_try, max_iter=K)
        with torch.no_grad():
            xh = g.solve(va['y'][:100].to(device))
            v = nmse_db(xh, va['x'][:100].to(device))
        if v < best_nmse:
            best_nmse = v
            best_lam = lam_try
    print(f"  Best lambda: {best_lam} (val={best_nmse:.2f} dB)")

    print("  [1] Group-FISTA")
    models['Group-FISTA'] = GroupFISTA(A, gs, lam=best_lam, max_iter=K)

    print("  [2] ALISTA")
    alista = ALISTA(A, num_layers=K)
    train_unfolded_model(alista, tr, va, num_epochs=TRAIN_EPOCHS,
                         lr=1e-3, batch_size=64, device=device)
    models['ALISTA'] = alista

    print("  [3] Ada-BlockLISTA (tied)")
    abl = AdaBlockLISTA(A, gs, num_layers=K, tied=True)
    train_unfolded_model(abl, tr, va, num_epochs=TRAIN_EPOCHS,
                         lr=5e-4, batch_size=64, device=device)
    models['Ada-BLISTA-T'] = abl

    print("  [4] HyperLISTA")
    hl = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K},
                          tr, va, n_trials=N_TRIALS, device=device)
    models['HyperLISTA'] = hl['model']

    for mode in ['topk_group', 'hybrid']:
        print(f"  [5] SH-{mode}")
        sh = tune_hyper_model(
            StructHyperLISTA,
            {'A': A, 'group_size': gs, 'num_layers': K, 'support_mode': mode},
            tr, va, n_trials=N_TRIALS, device=device)
        models[f'SH-{mode}'] = sh['model']

    return models


def run_scaled(device='cuda'):
    os.makedirs('results/scaled', exist_ok=True)
    core_all = defaultdict(lambda: defaultdict(list))

    for seed in range(N_SEEDS):
        t0 = time.time()
        print(f"\n{'='*60}\n  SCALED SEED {seed+1}/{N_SEEDS}  (n={N}, m={M})\n{'='*60}")

        ds = BlockSparseDataset(n=N, m=M, group_size=GS,
                                num_active_groups=SG, snr_db=30.0,
                                seed=seed*1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(N_TRAIN, seed=seed*1000+1)
        va = ds.generate(N_VAL, seed=seed*1000+2)
        te = ds.generate(N_TEST, seed=seed*1000+3)

        models = build_and_train_scaled(A, GS, K, tr, va, device)

        print("\n  Evaluating...")
        for name, mdl in models.items():
            traj = evaluate_trajectory(mdl, te, GS, device=device, num_layers=K)
            final = evaluate_model(mdl, te, GS, device=device)
            np_ = count_parameters(mdl)
            nh = getattr(mdl, 'num_hyperparams', np_)

            core_all[name]['traj_nmse'].append(traj['nmse_db'])
            core_all[name]['final_nmse'].append(final['nmse_db'])
            core_all[name]['prec'].append(final['precision'])
            core_all[name]['rec'].append(final['recall'])
            core_all[name]['nparams'] = np_
            core_all[name]['nhyper'] = nh

        elapsed = time.time() - t0
        print(f"  Seed {seed+1} done in {elapsed:.0f}s")

    def pad_and_agg(trajs):
        ml = max(len(t) for t in trajs)
        pad = [t+[t[-1]]*(ml-len(t)) for t in trajs]
        arr = np.array(pad)
        return arr.mean(0).tolist(), arr.std(0).tolist()

    core_agg = {}
    for name in core_all:
        d = core_all[name]
        tm, ts = pad_and_agg(d['traj_nmse'])
        core_agg[name] = {
            'trajectory_mean': tm, 'trajectory_std': ts,
            'nmse_db_mean': float(np.mean(d['final_nmse'])),
            'nmse_db_std': float(np.std(d['final_nmse'])),
            'precision_mean': float(np.mean(d['prec'])),
            'recall_mean': float(np.mean(d['rec'])),
            'num_params': d['nparams'], 'num_hyperparams': d['nhyper'],
        }

    with open('results/scaled/scaled_results.json', 'w') as f:
        json.dump(core_agg, f, indent=2)

    print(f"\n{'Model':<22} {'NMSE(dB)':<16} {'Prec':<10} {'Rec':<10} {'#P':<10}")
    print("=" * 68)
    for name in core_agg:
        r = core_agg[name]
        print(f"{name:<22} {r['nmse_db_mean']:>7.2f}+/-{r['nmse_db_std']:.2f}  "
              f"{r['precision_mean']:.3f}     {r['recall_mean']:.3f}     "
              f"{r['num_params']:<10}")

    return core_agg


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Problem: n={N}, m={M}, gs={GS}, sg={SG}")
    run_scaled(device=device)
