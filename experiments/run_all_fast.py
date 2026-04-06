"""
Fast experiment runner with reduced dimensions for CPU feasibility.
Produces all JSON results for figures and paper.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
from collections import defaultdict

from src.data.synthetic import BlockSparseDataset
from src.models.ista import GroupISTA, GroupFISTA
from src.models.lista import LISTA, BlockLISTA
from src.models.alista import ALISTA
from src.models.hyperlista import HyperLISTA
from src.models.struct_hyperlista import StructHyperLISTA
from src.train import train_unfolded_model, tune_hyper_model
from src.evaluate import evaluate_model, evaluate_trajectory, mismatch_sweep
from src.utils.metrics import nmse_db, count_parameters


# Reduced config for CPU
N, M, GS, SG = 200, 100, 10, 4
K = 12
N_TRAIN, N_VAL, N_TEST = 3000, 500, 1000
N_SEEDS = 3
N_TRIALS = 40
TRAIN_EPOCHS = 40


def build_and_train_models(A, gs, K, tr, va, device):
    models = {}

    print("  [1] Group-ISTA")
    models['Group-ISTA'] = GroupISTA(A, gs, lam=0.1, max_iter=K)
    print("  [2] Group-FISTA")
    models['Group-FISTA'] = GroupFISTA(A, gs, lam=0.1, max_iter=K)

    print("  [3] LISTA")
    lista = LISTA(A, num_layers=K)
    train_unfolded_model(lista, tr, va, num_epochs=TRAIN_EPOCHS,
                         lr=5e-4, batch_size=256, device=device)
    models['LISTA'] = lista

    print("  [4] BlockLISTA")
    blista = BlockLISTA(A, gs, num_layers=K)
    train_unfolded_model(blista, tr, va, num_epochs=TRAIN_EPOCHS,
                         lr=5e-4, batch_size=256, device=device)
    models['BlockLISTA'] = blista

    print("  [5] ALISTA")
    alista = ALISTA(A, num_layers=K)
    train_unfolded_model(alista, tr, va, num_epochs=TRAIN_EPOCHS,
                         lr=5e-4, batch_size=256, device=device)
    models['ALISTA'] = alista

    print("  [6] HyperLISTA")
    hl = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K},
                          tr, va, n_trials=N_TRIALS, device=device)
    models['HyperLISTA'] = hl['model']

    for mode in ['block_soft', 'topk_group', 'hybrid']:
        print(f"  [7] SH-{mode}")
        sh = tune_hyper_model(
            StructHyperLISTA,
            {'A': A, 'group_size': gs, 'num_layers': K, 'support_mode': mode},
            tr, va, n_trials=N_TRIALS, device=device)
        models[f'SH-{mode}'] = sh['model']

    return models


def run_all(device='cpu'):
    os.makedirs('results/core', exist_ok=True)
    os.makedirs('results/mismatch', exist_ok=True)
    os.makedirs('results/ablation', exist_ok=True)

    core_all = defaultdict(lambda: defaultdict(list))
    mm_all = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    abl_all = defaultdict(lambda: defaultdict(list))

    for seed in range(N_SEEDS):
        print(f"\n{'='*60}\n  SEED {seed+1}/{N_SEEDS}\n{'='*60}")

        ds = BlockSparseDataset(n=N, m=M, group_size=GS,
                                num_active_groups=SG, snr_db=30.0,
                                seed=seed*1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(N_TRAIN, seed=seed*1000+1)
        va = ds.generate(N_VAL, seed=seed*1000+2)
        te = ds.generate(N_TEST, seed=seed*1000+3)

        models = build_and_train_models(A, GS, K, tr, va, device)

        # CORE trajectories
        print("\n  [CORE] Trajectories...")
        for name, mdl in models.items():
            traj = evaluate_trajectory(mdl, te, GS, device=device, num_layers=K)
            final = evaluate_model(mdl, te, GS, device=device)
            np_ = count_parameters(mdl)
            nh = getattr(mdl, 'num_hyperparams', np_)

            core_all[name]['traj_nmse'].append(traj['nmse_db'])
            core_all[name]['traj_prec'].append(traj['precision'])
            core_all[name]['traj_rec'].append(traj['recall'])
            core_all[name]['final_nmse'].append(final['nmse_db'])
            core_all[name]['prec'].append(final['precision'])
            core_all[name]['rec'].append(final['recall'])
            core_all[name]['nparams'] = np_
            core_all[name]['nhyper'] = nh

        # MISMATCH
        mm_cfgs = {
            'snr_db': [15.0, 20.0, 25.0, 35.0, 40.0],
            'num_active_groups': [2, 3, 5, 6, 7],
            'group_size': [5, 8, 12, 15],
            'operator_delta': [0.05, 0.1, 0.15, 0.2, 0.3],
        }
        key_models = {k: v for k, v in models.items()
                      if k in ['Group-ISTA','Group-FISTA','BlockLISTA',
                               'ALISTA','HyperLISTA','SH-hybrid']}
        for sw, vals in mm_cfgs.items():
            print(f"  [MISMATCH] {sw}")
            for mn, mdl in key_models.items():
                r = mismatch_sweep(mdl, ds, sw, vals, GS,
                                   num_test=500, device=device)
                mm_all[sw][mn]['nmse'].append(r['nmse_db'])

        # ABLATION: sensitivity
        print("  [ABLATION] Sensitivity...")
        best_sh = models['SH-hybrid']
        bc1, bc2, bc3 = best_sh.c1.item(), best_sh.c2.item(), best_sh.c3.item()

        for pn, pr in [('c1', np.linspace(0.05,2.0,10)),
                       ('c2', np.linspace(0.0,1.5,10)),
                       ('c3', np.linspace(0.1,4.0,10))]:
            sens = []
            for v in pr:
                kw = {'c1': bc1, 'c2': bc2, 'c3': bc3}
                kw[pn] = float(v)
                m = StructHyperLISTA(A, GS, num_layers=K,
                                     support_mode='hybrid', **kw).to(device)
                m.eval()
                with torch.no_grad():
                    xh = m(te['y'].to(device))
                    sens.append(nmse_db(xh, te['x'].to(device)))
            abl_all[f'sens_{pn}']['values'] = pr.tolist()
            abl_all[f'sens_{pn}']['nmse'].append(sens)

        # Momentum ablation
        for c2v in [0.0, 0.1, 0.3, 0.5, 1.0]:
            m = StructHyperLISTA(A, GS, num_layers=K, c1=bc1,
                                 c2=c2v, c3=bc3, support_mode='hybrid').to(device)
            m.eval()
            with torch.no_grad():
                xh = m(te['y'].to(device))
                abl_all[f'mom_{c2v}']['nmse'].append(
                    nmse_db(xh, te['x'].to(device)))

    # ============= AGGREGATE =============
    print(f"\n{'='*60}\n  AGGREGATING\n{'='*60}")

    core_agg = {}
    for name in core_all:
        d = core_all[name]
        trajs = d['traj_nmse']
        ml = max(len(t) for t in trajs)
        pad = [t+[t[-1]]*(ml-len(t)) for t in trajs]
        arr = np.array(pad)

        pt = d['traj_prec']; rt = d['traj_rec']
        mlp = max(len(t) for t in pt)
        pp = [t+[t[-1]]*(mlp-len(t)) for t in pt]
        rp = [t+[t[-1]]*(mlp-len(t)) for t in rt]

        core_agg[name] = {
            'trajectory_mean': arr.mean(0).tolist(),
            'trajectory_std': arr.std(0).tolist(),
            'precision_trajectory': np.array(pp).mean(0).tolist(),
            'recall_trajectory': np.array(rp).mean(0).tolist(),
            'nmse_db_mean': float(np.mean(d['final_nmse'])),
            'nmse_db_std': float(np.std(d['final_nmse'])),
            'precision_mean': float(np.mean(d['prec'])),
            'recall_mean': float(np.mean(d['rec'])),
            'num_params': d['nparams'], 'num_hyperparams': d['nhyper'],
        }

    with open('results/core/core_results.json', 'w') as f:
        json.dump(core_agg, f, indent=2)

    mm_cfgs_out = {
        'snr_db': [15.0, 20.0, 25.0, 35.0, 40.0],
        'num_active_groups': [2, 3, 5, 6, 7],
        'group_size': [5, 8, 12, 15],
        'operator_delta': [0.05, 0.1, 0.15, 0.2, 0.3],
    }
    mm_agg = {}
    for sw in mm_all:
        mm_agg[sw] = {'values': mm_cfgs_out[sw]}
        for mn in mm_all[sw]:
            arr = np.array(mm_all[sw][mn]['nmse'])
            mm_agg[sw][mn] = {
                'nmse_db_mean': arr.mean(0).tolist(),
                'nmse_db_std': arr.std(0).tolist(),
            }
    with open('results/mismatch/mismatch_results.json', 'w') as f:
        json.dump(mm_agg, f, indent=2)

    abl_agg = {}
    for pn in ['c1','c2','c3']:
        k = f'sens_{pn}'
        arr = np.array(abl_all[k]['nmse'])
        abl_agg[f'sensitivity_{pn}'] = {
            'values': abl_all[k]['values'],
            'nmse_db_mean': arr.mean(0).tolist(),
            'nmse_db_std': arr.std(0).tolist(),
        }
    for mode in ['block_soft','topk_group','hybrid']:
        abl_agg[f'mechanism_{mode}'] = {
            'nmse_db_mean': core_agg.get(f'SH-{mode}',{}).get('nmse_db_mean',0),
            'nmse_db_std': core_agg.get(f'SH-{mode}',{}).get('nmse_db_std',0),
            'precision_mean': core_agg.get(f'SH-{mode}',{}).get('precision_mean',0),
            'recall_mean': core_agg.get(f'SH-{mode}',{}).get('recall_mean',0),
        }
    for c2v in [0.0, 0.1, 0.3, 0.5, 1.0]:
        vals = abl_all[f'mom_{c2v}']['nmse']
        abl_agg[f'momentum_c2={c2v}'] = {
            'nmse_db_mean': float(np.mean(vals)),
            'nmse_db_std': float(np.std(vals)),
        }
    with open('results/ablation/ablation_results.json', 'w') as f:
        json.dump(abl_agg, f, indent=2)

    # Print table
    print(f"\n{'Model':<22} {'NMSE(dB)':<16} {'Prec':<10} {'Rec':<10} {'#P':<10} {'#H':<6}")
    print("="*74)
    for name in core_agg:
        r = core_agg[name]
        print(f"{name:<22} {r['nmse_db_mean']:>7.2f}+/-{r['nmse_db_std']:.2f}  "
              f"{r['precision_mean']:.4f}    {r['recall_mean']:.4f}    "
              f"{r['num_params']:<10} {r['num_hyperparams']:<6}")

    return core_agg, mm_agg, abl_agg


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    run_all(device=device)
