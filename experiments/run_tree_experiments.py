"""
Tree-HyperLISTA experiment runner: core, mismatch, ablation across 5 seeds.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
from collections import defaultdict

from src.data.tree_synthetic import TreeSparseDataset, build_balanced_tree
from src.models.ista import ElementwiseISTA
from src.models.lista import LISTA
from src.models.hyperlista import HyperLISTA
from src.models.tree_hyperlista import TreeHyperLISTA
from src.models.tree_baselines import TreeISTA, TreeFISTA, TreeLISTA
from src.models.tree_classical import TreeIHT, TreeCoSaMP
from src.train import train_unfolded_model, tune_hyper_model
from src.utils.metrics import nmse_db, node_precision_recall, count_parameters

TREE_DEPTH = 7
BRANCHING = 2
N_TREE = (2 ** (TREE_DEPTH + 1)) - 1  # 255 for binary tree depth 7
M_RATIO = 0.5
TARGET_K = 30
K_LAYERS = 16
N_TRAIN, N_VAL, N_TEST = 3000, 600, 1000
N_SEEDS = 3
N_TRIALS = 50
TRAIN_EPOCHS = 150
RHO = 0.5


def evaluate_tree_model(model, test_data, device='cpu', num_layers=K_LAYERS):
    """Evaluate a single model on test data, return NMSE and node precision/recall."""
    x_test = test_data['x'].to(device)
    y_test = test_data['y'].to(device)

    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model = model.to(device)

    with torch.no_grad():
        if hasattr(model, 'solve'):
            x_hat = model.solve(y_test)
        else:
            x_hat = model(y_test)

    val_nmse = nmse_db(x_hat, x_test)
    prec, rec = node_precision_recall(x_hat, x_test)
    n_params = count_parameters(model) if hasattr(model, 'parameters') else 0

    return {
        'nmse_db': val_nmse,
        'precision': prec,
        'recall': rec,
        'num_params': n_params,
    }


def evaluate_tree_trajectory(model, test_data, device='cpu', num_layers=K_LAYERS):
    """Evaluate model at each layer/iteration for convergence curves."""
    x_test = test_data['x'].to(device)
    y_test = test_data['y'].to(device)

    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model = model.to(device)

    with torch.no_grad():
        if hasattr(model, 'solve'):
            trajectory = model.solve(y_test, return_trajectory=True, num_iter=num_layers)
        else:
            trajectory = model(y_test, return_trajectory=True, num_layers=num_layers)

    results = {'nmse_db': [], 'precision': [], 'recall': []}
    for x_k in trajectory:
        results['nmse_db'].append(nmse_db(x_k, x_test))
        p, r = node_precision_recall(x_k, x_test)
        results['precision'].append(p)
        results['recall'].append(r)

    return results


def tree_mismatch_sweep(model, dataset, sweep_param, sweep_values,
                        num_test=500, device='cpu'):
    """Evaluate model under distribution shift for tree-sparse signals."""
    results = {'values': sweep_values, 'nmse_db': [], 'precision': [], 'recall': []}

    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model = model.to(device)

    for val in sweep_values:
        if sweep_param == 'operator_delta':
            test_data = dataset.generate_with_perturbed_A(num_test, delta=val, seed=999)
        else:
            kwargs = {sweep_param: val}
            test_data = dataset.generate(num_test, seed=999, **kwargs)

        x_test = test_data['x'].to(device)
        y_test = test_data['y'].to(device)

        with torch.no_grad():
            if hasattr(model, 'solve'):
                x_hat = model.solve(y_test)
            else:
                x_hat = model(y_test)

        results['nmse_db'].append(nmse_db(x_hat, x_test))
        p, r = node_precision_recall(x_hat, x_test)
        results['precision'].append(p)
        results['recall'].append(r)

    return results


def build_and_train(A, tree_info, tr, va, device):
    """Build all models and train the learned ones."""
    models = {}

    # --- Lambda tuning for classical methods ---
    best_lam, best_nmse = 0.1, float('inf')
    for lam_try in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        g = TreeFISTA(A, tree_info, lam=lam_try, max_iter=K_LAYERS,
                      rho=RHO, target_K=TARGET_K)
        g = g.to(device)
        with torch.no_grad():
            xh = g.solve(va['y'][:200].to(device))
            v = nmse_db(xh, va['x'][:200].to(device))
        if v < best_nmse:
            best_nmse = v
            best_lam = lam_try
    print(f"  Best lambda for Tree-ISTA/FISTA: {best_lam} (val={best_nmse:.2f} dB)")

    print("  [1] Tree-ISTA")
    models['Tree-ISTA'] = TreeISTA(A, tree_info, lam=best_lam, max_iter=K_LAYERS,
                                   rho=RHO, target_K=TARGET_K)

    print("  [2] Tree-FISTA")
    models['Tree-FISTA'] = TreeFISTA(A, tree_info, lam=best_lam, max_iter=K_LAYERS,
                                     rho=RHO, target_K=TARGET_K)

    print("  [3] LISTA (elementwise)")
    lista = LISTA(A, num_layers=K_LAYERS)
    train_unfolded_model(lista, tr, va, num_epochs=TRAIN_EPOCHS,
                         lr=1e-3, batch_size=128, device=device)
    models['LISTA'] = lista

    print("  [4] Tree-LISTA")
    tree_lista = TreeLISTA(A, tree_info, num_layers=K_LAYERS,
                           rho=RHO, target_K=TARGET_K)
    train_unfolded_model(tree_lista, tr, va, num_epochs=TRAIN_EPOCHS,
                         lr=5e-4, batch_size=128, device=device,
                         progressive=False, grad_clip=1.0)
    models['Tree-LISTA'] = tree_lista

    print("  [5] Tree-IHT")
    models['Tree-IHT'] = TreeIHT(A, tree_info, target_K=TARGET_K,
                                  max_iter=K_LAYERS, rho=RHO)

    print("  [6] Tree-CoSaMP")
    models['Tree-CoSaMP'] = TreeCoSaMP(A, tree_info, target_K=TARGET_K,
                                        max_iter=K_LAYERS, rho=RHO)

    print("  [7] HyperLISTA (elementwise)")
    hl = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K_LAYERS},
                          tr, va, n_trials=N_TRIALS, device=device)
    models['HyperLISTA'] = hl['model']

    for mode in ['tree_hard', 'tree_threshold', 'hybrid_tree']:
        print(f"  [8] TH-{mode}")
        extra_kw = {}
        if mode == 'tree_threshold':
            extra_kw = {'n_trials': N_TRIALS + 40}
        th = tune_hyper_model(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': mode, 'rho': RHO},
            tr, va, n_trials=extra_kw.get('n_trials', N_TRIALS), device=device)
        models[f'TH-{mode}'] = th['model']

    return models


def run_all(device='cpu'):
    os.makedirs('results/tree_core', exist_ok=True)
    os.makedirs('results/tree_mismatch', exist_ok=True)
    os.makedirs('results/tree_ablation', exist_ok=True)

    tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)
    n = tree_info['n']
    m = int(n * M_RATIO)
    print(f"Tree: depth={TREE_DEPTH}, branching={BRANCHING}, n={n}, m={m}")

    core_all = defaultdict(lambda: defaultdict(list))
    mm_all = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    abl_all = defaultdict(lambda: defaultdict(list))

    for seed in range(N_SEEDS):
        print(f"\n{'='*60}\n  SEED {seed+1}/{N_SEEDS}\n{'='*60}")

        ds = TreeSparseDataset(tree_depth=TREE_DEPTH, branching=BRANCHING,
                               m_ratio=M_RATIO, target_sparsity=TARGET_K,
                               snr_db=30.0, seed=seed*1000, matrix_seed=42)
        A = ds.A
        tr = ds.generate(N_TRAIN, seed=seed*1000+1)
        va = ds.generate(N_VAL, seed=seed*1000+2)
        te = ds.generate(N_TEST, seed=seed*1000+3)

        models = build_and_train(A, tree_info, tr, va, device)

        print("\n  [CORE] Evaluating...")
        for name, mdl in models.items():
            traj = evaluate_tree_trajectory(mdl, te, device=device, num_layers=K_LAYERS)
            final = evaluate_tree_model(mdl, te, device=device)
            np_ = count_parameters(mdl) if hasattr(mdl, 'parameters') else 0
            nh = getattr(mdl, 'num_hyperparams', np_)

            core_all[name]['traj_nmse'].append(traj['nmse_db'])
            core_all[name]['traj_prec'].append(traj['precision'])
            core_all[name]['traj_rec'].append(traj['recall'])
            core_all[name]['final_nmse'].append(final['nmse_db'])
            core_all[name]['prec'].append(final['precision'])
            core_all[name]['rec'].append(final['recall'])
            core_all[name]['nparams'] = np_
            core_all[name]['nhyper'] = nh

        mm_cfgs = {
            'snr_db': [15.0, 20.0, 25.0, 35.0, 40.0],
            'target_sparsity': [15, 20, 30, 40, 50],
            'operator_delta': [0.05, 0.1, 0.15, 0.2, 0.3],
        }
        key_models = {k: v for k, v in models.items()
                      if k in ['Tree-ISTA', 'Tree-FISTA', 'LISTA',
                               'Tree-LISTA', 'HyperLISTA',
                               'Tree-IHT', 'Tree-CoSaMP',
                               'TH-tree_hard', 'TH-hybrid_tree']}
        for sw, vals in mm_cfgs.items():
            print(f"  [MM] {sw}")
            for mn, mdl in key_models.items():
                r = tree_mismatch_sweep(mdl, ds, sw, vals, num_test=500, device=device)
                mm_all[sw][mn]['nmse'].append(r['nmse_db'])

        print("  [ABL] Sensitivity...")
        best_th = models['TH-hybrid_tree']
        bc1 = best_th.c1.item()
        bc2 = best_th.c2.item()
        bc3 = best_th.c3.item()

        for pn, pr in [('c1', np.linspace(0.01, 8.0, 15)),
                       ('c2', np.linspace(-4.0, 3.0, 15)),
                       ('c3', np.linspace(0.5, 12.0, 15))]:
            sens = []
            for v in pr:
                kw = {'c1': bc1, 'c2': bc2, 'c3': bc3}
                kw[pn] = float(v)
                mdl_s = TreeHyperLISTA(A, tree_info, num_layers=K_LAYERS,
                                       support_mode='hybrid_tree', rho=RHO, **kw).to(device)
                mdl_s.eval()
                with torch.no_grad():
                    xh = mdl_s(te['y'].to(device))
                    val = nmse_db(xh, te['x'].to(device))
                sens.append(val if np.isfinite(val) else 10.0)
            abl_all[f'sens_{pn}']['values'] = pr.tolist()
            abl_all[f'sens_{pn}']['nmse'].append(sens)

        print("  [ABL] Rho sweep...")
        for rho_v in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            mdl_r = TreeHyperLISTA(A, tree_info, num_layers=K_LAYERS,
                                   c1=bc1, c2=bc2, c3=bc3,
                                   support_mode='hybrid_tree', rho=rho_v).to(device)
            mdl_r.eval()
            with torch.no_grad():
                xh = mdl_r(te['y'].to(device))
                val = nmse_db(xh, te['x'].to(device))
            abl_all[f'rho_{rho_v}']['nmse'].append(val if np.isfinite(val) else 10.0)

    # ============= AGGREGATE =============
    print(f"\n{'='*60}\n  AGGREGATING\n{'='*60}")

    def pad_and_agg(trajs):
        ml = max(len(t) for t in trajs)
        pad = [t+[t[-1]]*(ml-len(t)) for t in trajs]
        arr = np.array(pad)
        return arr.mean(0).tolist(), arr.std(0).tolist()

    core_agg = {}
    for name in core_all:
        d = core_all[name]
        tm, ts = pad_and_agg(d['traj_nmse'])
        pm, _ = pad_and_agg(d['traj_prec'])
        rm, _ = pad_and_agg(d['traj_rec'])
        core_agg[name] = {
            'trajectory_mean': tm, 'trajectory_std': ts,
            'precision_trajectory': pm, 'recall_trajectory': rm,
            'nmse_db_mean': float(np.mean(d['final_nmse'])),
            'nmse_db_std': float(np.std(d['final_nmse'])),
            'precision_mean': float(np.mean(d['prec'])),
            'precision_std': float(np.std(d['prec'])),
            'recall_mean': float(np.mean(d['rec'])),
            'recall_std': float(np.std(d['rec'])),
            'num_params': d['nparams'], 'num_hyperparams': d['nhyper'],
        }
    with open('results/tree_core/core_results.json', 'w') as f:
        json.dump(core_agg, f, indent=2)

    mm_agg = {}
    for sw in mm_all:
        mm_agg[sw] = {'values': mm_cfgs[sw]}
        for mn in mm_all[sw]:
            arr = np.array(mm_all[sw][mn]['nmse'])
            mm_agg[sw][mn] = {
                'nmse_db_mean': arr.mean(0).tolist(),
                'nmse_db_std': arr.std(0).tolist(),
            }
    with open('results/tree_mismatch/mismatch_results.json', 'w') as f:
        json.dump(mm_agg, f, indent=2)

    abl_agg = {}
    for pn in ['c1', 'c2', 'c3']:
        k = f'sens_{pn}'
        arr = np.array(abl_all[k]['nmse'])
        abl_agg[f'sensitivity_{pn}'] = {
            'values': abl_all[k]['values'],
            'nmse_db_mean': arr.mean(0).tolist(),
            'nmse_db_std': arr.std(0).tolist(),
        }
    for mode in ['tree_hard', 'tree_threshold', 'hybrid_tree']:
        abl_agg[f'mechanism_{mode}'] = {
            'nmse_db_mean': core_agg.get(f'TH-{mode}', {}).get('nmse_db_mean', 0),
            'nmse_db_std': core_agg.get(f'TH-{mode}', {}).get('nmse_db_std', 0),
            'precision_mean': core_agg.get(f'TH-{mode}', {}).get('precision_mean', 0),
            'recall_mean': core_agg.get(f'TH-{mode}', {}).get('recall_mean', 0),
        }
    for rho_v in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        vals = abl_all[f'rho_{rho_v}']['nmse']
        abl_agg[f'rho={rho_v}'] = {
            'nmse_db_mean': float(np.mean(vals)),
            'nmse_db_std': float(np.std(vals)),
        }
    with open('results/tree_ablation/ablation_results.json', 'w') as f:
        json.dump(abl_agg, f, indent=2)

    print(f"\n{'Model':<22} {'NMSE(dB)':<16} {'Prec':<10} {'Rec':<10} {'#P':<10} {'#H':<6}")
    print("=" * 74)
    for name in core_agg:
        r = core_agg[name]
        print(f"{name:<22} {r['nmse_db_mean']:>7.2f}+/-{r['nmse_db_std']:.2f}  "
              f"{r['precision_mean']:.3f}     {r['recall_mean']:.3f}     "
              f"{r['num_params']:<10} {r['num_hyperparams']:<6}")

    return core_agg, mm_agg, abl_agg


if __name__ == '__main__':
    from src.utils.sensing import pick_device; device = pick_device()
    print(f"Device: {device}")
    run_all(device=device)
