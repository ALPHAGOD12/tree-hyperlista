"""Evaluation pipeline for sparse recovery experiments."""

import torch
import numpy as np
from typing import Dict, Any, List
from src.utils.metrics import nmse, nmse_db, group_precision_recall, count_parameters


def evaluate_model(model, test_data: Dict[str, torch.Tensor],
                   group_size: int, device: str = 'cpu',
                   num_iter: int = None) -> Dict[str, float]:
    """Evaluate a single model on test data."""
    x_test = test_data['x'].to(device)
    y_test = test_data['y'].to(device)

    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model = model.to(device)

    with torch.no_grad():
        if hasattr(model, 'solve'):
            x_hat = model.solve(y_test, num_iter=num_iter)
        else:
            x_hat = model(y_test)

    val_nmse = nmse_db(x_hat, x_test)
    prec, rec = group_precision_recall(x_hat, x_test, group_size)
    n_params = count_parameters(model) if hasattr(model, 'parameters') else 0

    return {
        'nmse_db': val_nmse,
        'precision': prec,
        'recall': rec,
        'num_params': n_params,
    }


def evaluate_trajectory(model, test_data: Dict[str, torch.Tensor],
                        group_size: int, device: str = 'cpu',
                        num_layers: int = 16) -> Dict[str, List[float]]:
    """Evaluate model at each layer/iteration for convergence curves."""
    x_test = test_data['x'].to(device)
    y_test = test_data['y'].to(device)

    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model = model.to(device)

    with torch.no_grad():
        if hasattr(model, 'solve'):
            trajectory = model.solve(y_test, return_trajectory=True,
                                     num_iter=num_layers)
        else:
            trajectory = model(y_test, return_trajectory=True,
                               num_layers=num_layers)

    results = {'nmse_db': [], 'precision': [], 'recall': []}
    for x_k in trajectory:
        results['nmse_db'].append(nmse_db(x_k, x_test))
        p, r = group_precision_recall(x_k, x_test, group_size)
        results['precision'].append(p)
        results['recall'].append(r)

    return results


def mismatch_sweep(model, dataset, sweep_param: str, sweep_values: list,
                   group_size: int, num_test: int = 1000,
                   device: str = 'cpu') -> Dict[str, list]:
    """Evaluate model under distribution shift."""
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

        gs = group_size if sweep_param != 'group_size' else val
        results['nmse_db'].append(nmse_db(x_hat, x_test))
        p, r = group_precision_recall(x_hat, x_test, gs)
        results['precision'].append(p)
        results['recall'].append(r)

    return results
