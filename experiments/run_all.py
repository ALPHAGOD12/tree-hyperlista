"""
Unified experiment runner: core + mismatch + ablation experiments.
Produces all JSON results and figures needed for the paper.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
from collections import defaultdict

from src.data.synthetic import BlockSparseDataset, get_default_config, get_mismatch_configs
from src.models.ista import GroupISTA, GroupFISTA
from src.models.lista import LISTA, BlockLISTA
from src.models.alista import ALISTA
from src.models.hyperlista import HyperLISTA
from src.models.struct_hyperlista import StructHyperLISTA
from src.train import train_unfolded_model, tune_hyper_model
from src.evaluate import evaluate_model, evaluate_trajectory, mismatch_sweep
from src.utils.metrics import nmse_db, count_parameters, group_precision_recall


def run_all(device='cpu', num_seeds=3):
    os.makedirs('results/core', exist_ok=True)
    os.makedirs('results/mismatch', exist_ok=True)
    os.makedirs('results/ablation', exist_ok=True)

    cfg = get_default_config()
    N, M, GS = cfg['n'], cfg['m'], cfg['group_size']
    SG = cfg['num_active_groups']
    K = cfg['num_layers']

    core_all = defaultdict(lambda: defaultdict(list))
    mm_all = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    abl_all = defaultdict(lambda: defaultdict(list))

    for seed in range(num_seeds):
        print(f"\n{'='*70}")
        print(f"  SEED {seed+1}/{num_seeds}")
        print(f"{'='*70}")

        ds = BlockSparseDataset(n=N, m=M, group_size=GS,
                                num_active_groups=SG, snr_db=cfg['snr_db'],
                                seed=seed*1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(cfg['num_train'], seed=seed*1000+1)
        va = ds.generate(cfg['num_val'], seed=seed*1000+2)
        te = ds.generate(cfg['num_test'], seed=seed*1000+3)

        models = {}

        # Classical
        print("[1/8] Group-ISTA")
        models['Group-ISTA'] = GroupISTA(A, GS, lam=0.1, max_iter=K)
        print("[2/8] Group-FISTA")
        models['Group-FISTA'] = GroupFISTA(A, GS, lam=0.1, max_iter=K)

        # LISTA
        print("[3/8] LISTA (training...)")
        lista = LISTA(A, num_layers=K)
        train_unfolded_model(lista, tr, va, num_epochs=80, lr=5e-4, device=device)
        models['LISTA'] = lista

        # BlockLISTA
        print("[4/8] BlockLISTA (training...)")
        blista = BlockLISTA(A, GS, num_layers=K)
        train_unfolded_model(blista, tr, va, num_epochs=80, lr=5e-4, device=device)
        models['BlockLISTA'] = blista

        # ALISTA
        print("[5/8] ALISTA (training...)")
        alista = ALISTA(A, num_layers=K)
        train_unfolded_model(alista, tr, va, num_epochs=80, lr=5e-4, device=device)
        models['ALISTA'] = alista

        # HyperLISTA
        print("[6/8] HyperLISTA (tuning...)")
        hl = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K},
                              tr, va, n_trials=60, device=device)
        models['HyperLISTA'] = hl['model']

        # Struct-HyperLISTA variants
        for i, mode in enumerate(['block_soft', 'topk_group', 'hybrid']):
            print(f"[{7+i}/9] Struct-HyperLISTA-{mode} (tuning...)")
            sh = tune_hyper_model(
                StructHyperLISTA,
                {'A': A, 'group_size': GS, 'num_layers': K, 'support_mode': mode},
                tr, va, n_trials=60, device=device)
            models[f'SH-{mode}'] = sh['model']

        # ---- CORE: trajectories ----
        print("\n[CORE] Evaluating trajectories...")
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

        # ---- MISMATCH sweeps ----
        mm_cfgs = get_mismatch_configs()
        key_models = {k: v for k, v in models.items()
                      if k in ['Group-ISTA','Group-FISTA','BlockLISTA',
                               'ALISTA','HyperLISTA','SH-hybrid']}
        for sw_name, sw_vals in mm_cfgs.items():
            print(f"\n[MISMATCH] {sw_name}: {sw_vals}")
            for mname, mdl in key_models.items():
                r = mismatch_sweep(mdl, ds, sw_name, sw_vals, GS,
                                   num_test=1000, device=device)
                mm_all[sw_name][mname]['nmse'].append(r['nmse_db'])

        # ---- ABLATION: sensitivity ----
        print("\n[ABLATION] Hyperparameter sensitivity...")
        best_sh = models['SH-hybrid']
        best_c1 = best_sh.c1.item()
        best_c2 = best_sh.c2.item()
        best_c3 = best_sh.c3.item()

        for pname, prange in [('c1', np.linspace(0.05,2.0,12)),
                               ('c2', np.linspace(0.0,1.5,12)),
                               ('c3', np.linspace(0.1,4.0,12))]:
            sens = []
            for v in prange:
                kw = {'c1': best_c1, 'c2': best_c2, 'c3': best_c3}
                kw[pname] = float(v)
                m = StructHyperLISTA(A, GS, num_layers=K,
                                     support_mode='hybrid', **kw).to(device)
                m.eval()
                with torch.no_grad():
                    xh = m(te['y'].to(device))
                    sens.append(nmse_db(xh, te['x'].to(device)))
            abl_all[f'sens_{pname}']['values'] = prange.tolist()
            abl_all[f'sens_{pname}']['nmse'].append(sens)

        # Momentum ablation
        for c2v in [0.0, 0.1, 0.3, 0.5, 1.0]:
            m = StructHyperLISTA(A, GS, num_layers=K, c1=best_c1,
                                 c2=c2v, c3=best_c3, support_mode='hybrid').to(device)
            m.eval()
            with torch.no_grad():
                xh = m(te['y'].to(device))
                abl_all[f'mom_c2={c2v}']['nmse'].append(
                    nmse_db(xh, te['x'].to(device)))

    # =========== AGGREGATE ===========
    print("\n\n" + "="*70)
    print("  AGGREGATING RESULTS")
    print("="*70)

    # Core
    core_agg = {}
    for name in core_all:
        d = core_all[name]
        trajs = d['traj_nmse']
        maxlen = max(len(t) for t in trajs)
        padded = [t + [t[-1]]*(maxlen-len(t)) for t in trajs]
        arr = np.array(padded)

        prec_t = d['traj_prec']
        rec_t = d['traj_rec']
        maxlen_p = max(len(t) for t in prec_t)
        prec_pad = [t+[t[-1]]*(maxlen_p-len(t)) for t in prec_t]
        rec_pad = [t+[t[-1]]*(maxlen_p-len(t)) for t in rec_t]

        core_agg[name] = {
            'trajectory_mean': arr.mean(0).tolist(),
            'trajectory_std': arr.std(0).tolist(),
            'precision_trajectory': np.array(prec_pad).mean(0).tolist(),
            'recall_trajectory': np.array(rec_pad).mean(0).tolist(),
            'nmse_db_mean': float(np.mean(d['final_nmse'])),
            'nmse_db_std': float(np.std(d['final_nmse'])),
            'precision_mean': float(np.mean(d['prec'])),
            'recall_mean': float(np.mean(d['rec'])),
            'num_params': d['nparams'],
            'num_hyperparams': d['nhyper'],
        }

    with open('results/core/core_results.json', 'w') as f:
        json.dump(core_agg, f, indent=2)

    # Mismatch
    mm_agg = {}
    for sw in mm_all:
        mm_agg[sw] = {'values': get_mismatch_configs()[sw]}
        for mname in mm_all[sw]:
            arr = np.array(mm_all[sw][mname]['nmse'])
            mm_agg[sw][mname] = {
                'nmse_db_mean': arr.mean(0).tolist(),
                'nmse_db_std': arr.std(0).tolist(),
            }
    with open('results/mismatch/mismatch_results.json', 'w') as f:
        json.dump(mm_agg, f, indent=2)

    # Ablation
    abl_agg = {}
    for pname in ['c1','c2','c3']:
        k = f'sens_{pname}'
        arr = np.array(abl_all[k]['nmse'])
        abl_agg[k.replace('sens_','sensitivity_')] = {
            'values': abl_all[k]['values'],
            'nmse_db_mean': arr.mean(0).tolist(),
            'nmse_db_std': arr.std(0).tolist(),
        }

    for mode in ['block_soft','topk_group','hybrid']:
        k = f'mechanism_{mode}'
        abl_agg[k] = {
            'nmse_db_mean': core_agg.get(f'SH-{mode}',{}).get('nmse_db_mean',0),
            'nmse_db_std': core_agg.get(f'SH-{mode}',{}).get('nmse_db_std',0),
            'precision_mean': core_agg.get(f'SH-{mode}',{}).get('precision_mean',0),
            'recall_mean': core_agg.get(f'SH-{mode}',{}).get('recall_mean',0),
        }

    for c2v in [0.0, 0.1, 0.3, 0.5, 1.0]:
        k = f'mom_c2={c2v}'
        vals = abl_all[k]['nmse']
        abl_agg[f'momentum_c2={c2v}'] = {
            'nmse_db_mean': float(np.mean(vals)),
            'nmse_db_std': float(np.std(vals)),
        }

    with open('results/ablation/ablation_results.json', 'w') as f:
        json.dump(abl_agg, f, indent=2)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Model':<22} {'NMSE(dB)':<16} {'Prec':<10} {'Rec':<10} {'#Params':<12} {'#Hyper':<8}")
    print(f"{'='*90}")
    for name in core_agg:
        r = core_agg[name]
        print(f"{name:<22} {r['nmse_db_mean']:>7.2f}+/-{r['nmse_db_std']:.2f}  "
              f"{r['precision_mean']:.4f}    {r['recall_mean']:.4f}    "
              f"{r['num_params']:<12} {r['num_hyperparams']:<8}")

    return core_agg, mm_agg, abl_agg


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    run_all(device=device, num_seeds=3)
