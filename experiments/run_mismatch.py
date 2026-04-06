"""Mismatch sweep experiments: noise, sparsity, block-size, operator perturbation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json

from src.data.synthetic import BlockSparseDataset, get_default_config, get_mismatch_configs
from src.models.ista import GroupISTA, GroupFISTA
from src.models.lista import BlockLISTA
from src.models.alista import ALISTA
from src.models.hyperlista import HyperLISTA
from src.models.struct_hyperlista import StructHyperLISTA
from src.train import train_unfolded_model, tune_hyper_model
from src.evaluate import mismatch_sweep


def run_mismatch_experiments(device: str = 'cpu', num_seeds: int = 3,
                              output_dir: str = 'results/mismatch'):
    os.makedirs(output_dir, exist_ok=True)
    cfg = get_default_config()
    mismatch_cfgs = get_mismatch_configs()
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

        models = {}

        print("Building Group-ISTA / Group-FISTA...")
        models['Group-ISTA'] = GroupISTA(A, gs, lam=0.1, max_iter=K)
        models['Group-FISTA'] = GroupFISTA(A, gs, lam=0.1, max_iter=K)

        print("Training BlockLISTA...")
        blista = BlockLISTA(A, gs, num_layers=K)
        train_unfolded_model(blista, train_data, val_data,
                             num_epochs=80, lr=5e-4, device=device)
        models['BlockLISTA'] = blista

        print("Training ALISTA...")
        alista = ALISTA(A, num_layers=K)
        train_unfolded_model(alista, train_data, val_data,
                             num_epochs=80, lr=5e-4, device=device)
        models['ALISTA'] = alista

        print("Tuning HyperLISTA...")
        hl_result = tune_hyper_model(
            HyperLISTA, {'A': A, 'num_layers': K},
            train_data, val_data, n_trials=60, device=device
        )
        models['HyperLISTA'] = hl_result['model']

        print("Tuning Struct-HyperLISTA (hybrid)...")
        sh_result = tune_hyper_model(
            StructHyperLISTA,
            {'A': A, 'group_size': gs, 'num_layers': K, 'support_mode': 'hybrid'},
            train_data, val_data, n_trials=60, device=device
        )
        models['SH-hybrid'] = sh_result['model']

        seed_results = {}
        for sweep_name, sweep_values in mismatch_cfgs.items():
            print(f"\n  Sweep: {sweep_name} = {sweep_values}")
            sweep_results = {}
            for model_name, model in models.items():
                print(f"    {model_name}...")
                r = mismatch_sweep(model, dataset, sweep_name, sweep_values,
                                   gs, num_test=1000, device=device)
                sweep_results[model_name] = r
            seed_results[sweep_name] = sweep_results

        all_results[f'seed_{seed}'] = seed_results

    # Aggregate across seeds
    print("\nAggregating mismatch results...")
    aggregated = {}
    model_names = list(all_results['seed_0'][list(mismatch_cfgs.keys())[0]].keys())

    for sweep_name in mismatch_cfgs:
        aggregated[sweep_name] = {'values': mismatch_cfgs[sweep_name]}
        for model_name in model_names:
            nmse_per_seed = []
            for seed in range(num_seeds):
                nmse_per_seed.append(
                    all_results[f'seed_{seed}'][sweep_name][model_name]['nmse_db']
                )

            nmse_arr = np.array(nmse_per_seed)
            aggregated[sweep_name][model_name] = {
                'nmse_db_mean': nmse_arr.mean(axis=0).tolist(),
                'nmse_db_std': nmse_arr.std(axis=0).tolist(),
            }

    with open(os.path.join(output_dir, 'mismatch_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    for sweep_name in mismatch_cfgs:
        print(f"\n--- Mismatch: {sweep_name} ---")
        vals = mismatch_cfgs[sweep_name]
        header = f"{'Model':<20}" + "".join(f"{v:>10}" for v in vals)
        print(header)
        for model_name in model_names:
            means = aggregated[sweep_name][model_name]['nmse_db_mean']
            row = f"{model_name:<20}" + "".join(f"{m:>10.2f}" for m in means)
            print(row)

    return aggregated


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    run_mismatch_experiments(device=device, num_seeds=3)
