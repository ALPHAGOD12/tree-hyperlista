"""Ablation studies: support mechanisms, momentum, hyperparameter sensitivity."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json

from src.data.synthetic import BlockSparseDataset, get_default_config
from src.models.struct_hyperlista import StructHyperLISTA
from src.train import tune_hyper_model
from src.evaluate import evaluate_model, evaluate_trajectory
from src.utils.metrics import nmse_db


def run_ablation_experiments(device: str = 'cpu', num_seeds: int = 3,
                              output_dir: str = 'results/ablation'):
    os.makedirs(output_dir, exist_ok=True)
    cfg = get_default_config()
    K = cfg['num_layers']
    gs = cfg['group_size']

    all_results = {}

    for seed in range(num_seeds):
        print(f"\n{'='*60}")
        print(f"  Seed {seed + 1}/{num_seeds}")
        print(f"{'='*60}")

        dataset = BlockSparseDataset(
            n=cfg['n'], m=cfg['m'], group_size=gs,
            num_active_groups=cfg['num_active_groups'],
            snr_db=cfg['snr_db'], seed=seed * 1000, matrix_seed=42
        )
        A = dataset.A
        train_data = dataset.generate(cfg['num_train'], seed=seed * 1000 + 1)
        val_data = dataset.generate(cfg['num_val'], seed=seed * 1000 + 2)
        test_data = dataset.generate(cfg['num_test'], seed=seed * 1000 + 3)

        seed_results = {}

        # --- Ablation 1: Support mechanism comparison ---
        print("\n--- Support Mechanism Comparison ---")
        for mode in ['block_soft', 'topk_group', 'hybrid']:
            print(f"  Tuning {mode}...")
            result = tune_hyper_model(
                StructHyperLISTA,
                {'A': A, 'group_size': gs, 'num_layers': K, 'support_mode': mode},
                train_data, val_data, n_trials=60, device=device
            )
            model = result['model']
            eval_r = evaluate_model(model, test_data, gs, device=device)
            traj = evaluate_trajectory(model, test_data, gs, device=device, num_layers=K)

            seed_results[f'mechanism_{mode}'] = {
                'nmse_db': eval_r['nmse_db'],
                'precision': eval_r['precision'],
                'recall': eval_r['recall'],
                'c1': result['c1'],
                'c2': result['c2'],
                'c3': result['c3'],
                'trajectory': traj['nmse_db'],
            }

        # --- Ablation 2: Momentum ablation ---
        print("\n--- Momentum Ablation ---")
        for c2_val in [0.0, 0.1, 0.3, 0.5, 1.0]:
            label = f'momentum_c2={c2_val}'
            print(f"  {label}...")
            model = StructHyperLISTA(
                A, gs, num_layers=K, c1=0.5, c2=c2_val, c3=0.5,
                support_mode='hybrid'
            ).to(device)
            model.eval()
            with torch.no_grad():
                x_hat = model(test_data['y'].to(device))
                val = nmse_db(x_hat, test_data['x'].to(device))
            seed_results[label] = {'nmse_db': val}

        # --- Ablation 3: Hyperparameter sensitivity ---
        print("\n--- Hyperparameter Sensitivity ---")
        best_result = tune_hyper_model(
            StructHyperLISTA,
            {'A': A, 'group_size': gs, 'num_layers': K, 'support_mode': 'hybrid'},
            train_data, val_data, n_trials=60, device=device
        )
        best_c1, best_c2, best_c3 = best_result['c1'], best_result['c2'], best_result['c3']

        for param_name, param_range in [
            ('c1', np.linspace(0.05, 2.0, 15)),
            ('c2', np.linspace(0.0, 1.5, 15)),
            ('c3', np.linspace(0.1, 4.0, 15)),
        ]:
            sensitivities = []
            for val in param_range:
                kwargs = {'c1': best_c1, 'c2': best_c2, 'c3': best_c3}
                kwargs[param_name] = float(val)
                model = StructHyperLISTA(
                    A, gs, num_layers=K, support_mode='hybrid', **kwargs
                ).to(device)
                model.eval()
                with torch.no_grad():
                    x_hat = model(test_data['y'].to(device))
                    n = nmse_db(x_hat, test_data['x'].to(device))
                sensitivities.append(float(n))

            seed_results[f'sensitivity_{param_name}'] = {
                'values': param_range.tolist(),
                'nmse_db': sensitivities,
                'best_value': float(kwargs[param_name]),
            }

        # --- Ablation 4: Number of layers ---
        print("\n--- Layer Count Ablation ---")
        layer_counts = [2, 4, 6, 8, 10, 12, 16, 20, 24]
        layer_results = []
        model = best_result['model'].to(device)
        model.eval()
        for nk in layer_counts:
            with torch.no_grad():
                x_hat = model(test_data['y'].to(device), num_layers=min(nk, K))
                n = nmse_db(x_hat, test_data['x'].to(device))
            layer_results.append(float(n))
        seed_results['layer_ablation'] = {
            'layers': layer_counts,
            'nmse_db': layer_results,
        }

        all_results[f'seed_{seed}'] = seed_results

    # Aggregate
    aggregated = {}

    # Mechanism comparison
    for mode in ['block_soft', 'topk_group', 'hybrid']:
        key = f'mechanism_{mode}'
        nmses = [all_results[f'seed_{s}'][key]['nmse_db'] for s in range(num_seeds)]
        precs = [all_results[f'seed_{s}'][key]['precision'] for s in range(num_seeds)]
        recs = [all_results[f'seed_{s}'][key]['recall'] for s in range(num_seeds)]
        aggregated[key] = {
            'nmse_db_mean': float(np.mean(nmses)),
            'nmse_db_std': float(np.std(nmses)),
            'precision_mean': float(np.mean(precs)),
            'recall_mean': float(np.mean(recs)),
            'trajectories': [all_results[f'seed_{s}'][key]['trajectory']
                             for s in range(num_seeds)],
        }

    # Momentum ablation
    for c2_val in [0.0, 0.1, 0.3, 0.5, 1.0]:
        key = f'momentum_c2={c2_val}'
        nmses = [all_results[f'seed_{s}'][key]['nmse_db'] for s in range(num_seeds)]
        aggregated[key] = {
            'nmse_db_mean': float(np.mean(nmses)),
            'nmse_db_std': float(np.std(nmses)),
        }

    # Sensitivity
    for param_name in ['c1', 'c2', 'c3']:
        key = f'sensitivity_{param_name}'
        all_sens = [all_results[f'seed_{s}'][key]['nmse_db'] for s in range(num_seeds)]
        arr = np.array(all_sens)
        aggregated[key] = {
            'values': all_results['seed_0'][key]['values'],
            'nmse_db_mean': arr.mean(axis=0).tolist(),
            'nmse_db_std': arr.std(axis=0).tolist(),
        }

    # Layer ablation
    all_layers = [all_results[f'seed_{s}']['layer_ablation']['nmse_db']
                  for s in range(num_seeds)]
    arr = np.array(all_layers)
    aggregated['layer_ablation'] = {
        'layers': all_results['seed_0']['layer_ablation']['layers'],
        'nmse_db_mean': arr.mean(axis=0).tolist(),
        'nmse_db_std': arr.std(axis=0).tolist(),
    }

    with open(os.path.join(output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print("Support Mechanism Comparison:")
    for mode in ['block_soft', 'topk_group', 'hybrid']:
        r = aggregated[f'mechanism_{mode}']
        print(f"  {mode:<15} NMSE: {r['nmse_db_mean']:.2f}±{r['nmse_db_std']:.2f}"
              f"  Prec: {r['precision_mean']:.4f}  Rec: {r['recall_mean']:.4f}")

    print("\nMomentum Ablation:")
    for c2_val in [0.0, 0.1, 0.3, 0.5, 1.0]:
        r = aggregated[f'momentum_c2={c2_val}']
        print(f"  c2={c2_val:<5} NMSE: {r['nmse_db_mean']:.2f}±{r['nmse_db_std']:.2f}")

    return aggregated


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    run_ablation_experiments(device=device, num_seeds=3)
