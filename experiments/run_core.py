"""Core synthetic experiments: convergence, support recovery, parameter counts."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
from tqdm import tqdm

from src.data.synthetic import BlockSparseDataset, get_default_config
from src.models.ista import GroupISTA, GroupFISTA
from src.models.lista import LISTA, BlockLISTA
from src.models.alista import ALISTA
from src.models.hyperlista import HyperLISTA
from src.models.struct_hyperlista import StructHyperLISTA
from src.train import train_unfolded_model, tune_hyper_model
from src.evaluate import evaluate_model, evaluate_trajectory
from src.utils.metrics import count_parameters


def run_core_experiments(device: str = 'cpu', num_seeds: int = 3,
                         output_dir: str = 'results/core'):
    os.makedirs(output_dir, exist_ok=True)
    cfg = get_default_config()
    all_results = {}

    for seed in range(num_seeds):
        print(f"\n{'='*60}")
        print(f"  Seed {seed + 1}/{num_seeds}")
        print(f"{'='*60}")

        dataset = BlockSparseDataset(
            n=cfg['n'], m=cfg['m'], group_size=cfg['group_size'],
            num_active_groups=cfg['num_active_groups'],
            snr_db=cfg['snr_db'], seed=seed * 1000, matrix_seed=42
        )
        A = dataset.A
        train_data = dataset.generate(cfg['num_train'], seed=seed * 1000 + 1)
        val_data = dataset.generate(cfg['num_val'], seed=seed * 1000 + 2)
        test_data = dataset.generate(cfg['num_test'], seed=seed * 1000 + 3)

        K = cfg['num_layers']
        gs = cfg['group_size']
        models = {}

        # --- Classical baselines ---
        print("Building Group-ISTA...")
        models['Group-ISTA'] = GroupISTA(A, gs, lam=0.1, max_iter=K)
        print("Building Group-FISTA...")
        models['Group-FISTA'] = GroupFISTA(A, gs, lam=0.1, max_iter=K)

        # --- LISTA ---
        print("Training LISTA...")
        lista = LISTA(A, num_layers=K)
        train_unfolded_model(lista, train_data, val_data,
                             num_epochs=80, lr=5e-4, device=device)
        models['LISTA'] = lista

        # --- BlockLISTA ---
        print("Training BlockLISTA...")
        blista = BlockLISTA(A, gs, num_layers=K)
        train_unfolded_model(blista, train_data, val_data,
                             num_epochs=80, lr=5e-4, device=device)
        models['BlockLISTA'] = blista

        # --- ALISTA ---
        print("Training ALISTA...")
        alista = ALISTA(A, num_layers=K)
        train_unfolded_model(alista, train_data, val_data,
                             num_epochs=80, lr=5e-4, device=device)
        models['ALISTA'] = alista

        # --- HyperLISTA ---
        print("Tuning HyperLISTA...")
        hl_result = tune_hyper_model(
            HyperLISTA, {'A': A, 'num_layers': K},
            train_data, val_data, n_trials=60, device=device
        )
        models['HyperLISTA'] = hl_result['model']

        # --- Struct-HyperLISTA (3 variants) ---
        for mode in ['block_soft', 'topk_group', 'hybrid']:
            print(f"Tuning Struct-HyperLISTA ({mode})...")
            sh_result = tune_hyper_model(
                StructHyperLISTA,
                {'A': A, 'group_size': gs, 'num_layers': K, 'support_mode': mode},
                train_data, val_data, n_trials=60, device=device
            )
            name = f'SH-{mode}'
            models[name] = sh_result['model']

        # --- Evaluate convergence trajectories ---
        print("\nEvaluating convergence trajectories...")
        seed_results = {}
        for name, model in models.items():
            print(f"  {name}...")
            traj = evaluate_trajectory(model, test_data, gs, device=device,
                                       num_layers=K)
            final = evaluate_model(model, test_data, gs, device=device)

            n_params = count_parameters(model) if hasattr(model, 'parameters') else 0
            n_hyper = getattr(model, 'num_hyperparams', n_params)

            seed_results[name] = {
                'trajectory': traj,
                'final_nmse_db': final['nmse_db'],
                'precision': final['precision'],
                'recall': final['recall'],
                'num_params': n_params,
                'num_hyperparams': n_hyper,
            }

        all_results[f'seed_{seed}'] = seed_results

    # --- Aggregate across seeds ---
    print("\nAggregating results...")
    aggregated = {}
    model_names = list(all_results['seed_0'].keys())

    for name in model_names:
        nmse_finals = []
        precs = []
        recs = []
        trajectories = []

        for seed in range(num_seeds):
            r = all_results[f'seed_{seed}'][name]
            nmse_finals.append(r['final_nmse_db'])
            precs.append(r['precision'])
            recs.append(r['recall'])
            trajectories.append(r['trajectory']['nmse_db'])

        max_len = max(len(t) for t in trajectories)
        padded = []
        for t in trajectories:
            padded.append(t + [t[-1]] * (max_len - len(t)))

        aggregated[name] = {
            'nmse_db_mean': float(np.mean(nmse_finals)),
            'nmse_db_std': float(np.std(nmse_finals)),
            'precision_mean': float(np.mean(precs)),
            'recall_mean': float(np.mean(recs)),
            'trajectory_mean': [float(np.mean([p[i] for p in padded]))
                                for i in range(max_len)],
            'trajectory_std': [float(np.std([p[i] for p in padded]))
                               for i in range(max_len)],
            'num_params': all_results['seed_0'][name]['num_params'],
            'num_hyperparams': all_results['seed_0'][name]['num_hyperparams'],
        }

    # Compute precision/recall trajectories (averaged)
    for name in model_names:
        prec_trajs = []
        rec_trajs = []
        for seed in range(num_seeds):
            prec_trajs.append(all_results[f'seed_{seed}'][name]['trajectory']['precision'])
            rec_trajs.append(all_results[f'seed_{seed}'][name]['trajectory']['recall'])

        max_len = max(len(t) for t in prec_trajs)
        prec_padded = [t + [t[-1]] * (max_len - len(t)) for t in prec_trajs]
        rec_padded = [t + [t[-1]] * (max_len - len(t)) for t in rec_trajs]

        aggregated[name]['precision_trajectory'] = [
            float(np.mean([p[i] for p in prec_padded])) for i in range(max_len)]
        aggregated[name]['recall_trajectory'] = [
            float(np.mean([p[i] for p in rec_padded])) for i in range(max_len)]

    with open(os.path.join(output_dir, 'core_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)

    # --- Print summary ---
    print(f"\n{'='*80}")
    print(f"{'Model':<25} {'NMSE(dB)':<15} {'Prec':<10} {'Rec':<10} {'#Params':<12} {'#Hyper':<10}")
    print(f"{'='*80}")
    for name in model_names:
        r = aggregated[name]
        print(f"{name:<25} {r['nmse_db_mean']:>8.2f}±{r['nmse_db_std']:.2f}  "
              f"{r['precision_mean']:.4f}    {r['recall_mean']:.4f}    "
              f"{r['num_params']:<12} {r['num_hyperparams']:<10}")

    return aggregated


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    run_core_experiments(device=device, num_seeds=3)
