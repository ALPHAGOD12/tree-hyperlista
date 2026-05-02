"""Training pipeline for unfolded sparse recovery models."""

import time
import torch
import torch.nn as nn
import numpy as np
import optuna
from itertools import product
from typing import Dict, Any

from src.utils.metrics import nmse, nmse_db


def train_unfolded_model(model: nn.Module, train_data: Dict[str, torch.Tensor],
                         val_data: Dict[str, torch.Tensor],
                         num_epochs: int = 200, lr: float = 1e-3,
                         batch_size: int = 256, progressive: bool = True,
                         device: str = 'cpu',
                         grad_clip: float = 5.0) -> Dict[str, list]:
    """Train LISTA / ALISTA with backprop."""
    model = model.to(device)
    x_train, y_train = train_data['x'].to(device), train_data['y'].to(device)
    x_val, y_val = val_data['x'].to(device), val_data['y'].to(device)

    num_layers = model.num_layers
    history = {'train_loss': [], 'val_nmse_db': []}
    best_val = float('inf')
    best_state = None

    if progressive:
        stages = []
        layers_per_stage = max(1, num_layers // 4)
        for stage in range(4):
            K = min((stage + 1) * layers_per_stage, num_layers)
            stages.append((K, num_epochs // 4))
        if stages[-1][0] < num_layers:
            stages.append((num_layers, num_epochs // 4))
    else:
        stages = [(num_layers, num_epochs)]

    for stage_idx, (K, stage_epochs) in enumerate(stages):
        stage_lr = lr * (0.5 ** stage_idx)
        optimizer = torch.optim.Adam(model.parameters(), lr=stage_lr,
                                     weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_epochs, eta_min=stage_lr * 0.01)

        for epoch in range(stage_epochs):
            model.train()
            perm = torch.randperm(x_train.shape[0])
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, x_train.shape[0], batch_size):
                idx = perm[i:i + batch_size]
                y_batch, x_batch = y_train[idx], x_train[idx]

                x_hat = model(y_batch, num_layers=K)
                loss = nn.functional.mse_loss(x_hat, x_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()

            model.eval()
            with torch.no_grad():
                x_val_hat = model(y_val, num_layers=K)
                val_nmse = nmse_db(x_val_hat, x_val)

            history['train_loss'].append(epoch_loss / max(num_batches, 1))
            history['val_nmse_db'].append(val_nmse)

            if val_nmse < best_val:
                best_val = val_nmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 25 == 0:
                print(f"    stage {stage_idx+1} epoch {epoch+1}/{stage_epochs}  val={val_nmse:.2f}dB", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def tune_hyper_model(model_class, model_kwargs: dict,
                     train_data: Dict[str, torch.Tensor],
                     val_data: Dict[str, torch.Tensor],
                     n_trials: int = 100, device: str = 'cpu') -> Dict[str, Any]:
    """Bayesian optimization over (c1, c2, c3) for HyperLISTA / Struct-HyperLISTA."""
    x_val, y_val = val_data['x'].to(device), val_data['y'].to(device)
    x_train, y_train = train_data['x'].to(device), train_data['y'].to(device)

    best_result = {'nmse_db': float('inf'), 'c1': 0.5, 'c2': 0.3, 'c3': 0.5}

    def objective(trial):
        c1 = trial.suggest_float('c1', 0.01, 10.0, log=True)
        c2 = trial.suggest_float('c2', -8.0, 3.0)
        c3 = trial.suggest_float('c3', 0.05, 10.0, log=True)

        kwargs = {**model_kwargs, 'c1': c1, 'c2': c2, 'c3': c3}
        model = model_class(**kwargs).to(device)
        model.eval()

        with torch.no_grad():
            x_hat = model(y_val[:500])
            val_nmse = nmse_db(x_hat, x_val[:500])

        if not np.isfinite(val_nmse):
            return 100.0  # large penalty instead of 0.0
        return val_nmse

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_result.update({
        'c1': best['c1'], 'c2': best['c2'], 'c3': best['c3'],
        'nmse_db': study.best_value,
    })

    kwargs = {**model_kwargs, 'c1': best['c1'], 'c2': best['c2'], 'c3': best['c3']}
    model = model_class(**kwargs).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    for epoch in range(80):
        perm = torch.randperm(x_train.shape[0])
        for i in range(0, min(x_train.shape[0], 3000), 256):
            idx = perm[i:i + 256]
            x_hat = model(y_train[idx])
            loss = nn.functional.mse_loss(x_hat, x_train[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"    fine-tune epoch {epoch + 1}/80", flush=True)

    model.eval()
    with torch.no_grad():
        x_hat = model(y_val)
        best_result['nmse_db_finetuned'] = nmse_db(x_hat, x_val)

    best_result['model'] = model
    return best_result


def tune_hyper_grid_search(model_class, model_kwargs: dict,
                           train_data: Dict[str, torch.Tensor],
                           val_data: Dict[str, torch.Tensor],
                           coarse_grid: Dict[str, list] = None,
                           fine_halfwidth: float = 1.5,
                           fine_points: int = 5,
                           device: str = 'cpu') -> Dict[str, Any]:
    """Coarse + fine grid search over ``(c1, c2, c3)``.

    Mirrors the HyperLISTA paper's grid-search tuner. The coarse pass
    explores a broad log-linear grid; the fine pass zooms around the
    best coarse point by a multiplicative factor ``fine_halfwidth``.
    Returns the best ``(c1, c2, c3)`` and the tuning wall time.
    """
    x_val, y_val = val_data['x'].to(device), val_data['y'].to(device)

    if coarse_grid is None:
        coarse_grid = {
            'c1': [0.1, 0.5, 1.0, 2.0, 5.0],
            'c2': [-3.0, -1.0, 0.0, 1.0, 2.0],
            'c3': [0.2, 0.5, 1.0, 3.0, 6.0],
        }

    def _eval(c1, c2, c3):
        kw = {**model_kwargs, 'c1': c1, 'c2': c2, 'c3': c3}
        m = model_class(**kw).to(device).eval()
        with torch.no_grad():
            xh = m(y_val[:500])
            v = nmse_db(xh, x_val[:500])
        return float(v) if np.isfinite(v) else 100.0

    t0 = time.time()
    n_evals = 0
    best = {'c1': 1.0, 'c2': 0.0, 'c3': 1.0, 'nmse_db': float('inf')}

    for c1, c2, c3 in product(coarse_grid['c1'], coarse_grid['c2'],
                              coarse_grid['c3']):
        v = _eval(c1, c2, c3)
        n_evals += 1
        if v < best['nmse_db']:
            best = {'c1': c1, 'c2': c2, 'c3': c3, 'nmse_db': v}

    fw = fine_halfwidth
    fine_c1 = np.geomspace(best['c1'] / fw, best['c1'] * fw, fine_points)
    fine_c3 = np.geomspace(best['c3'] / fw, best['c3'] * fw, fine_points)
    fine_c2 = np.linspace(best['c2'] - 1.0, best['c2'] + 1.0, fine_points)
    for c1, c2, c3 in product(fine_c1, fine_c2, fine_c3):
        v = _eval(float(c1), float(c2), float(c3))
        n_evals += 1
        if v < best['nmse_db']:
            best = {'c1': float(c1), 'c2': float(c2), 'c3': float(c3),
                    'nmse_db': v}

    elapsed = time.time() - t0
    kw = {**model_kwargs, 'c1': best['c1'], 'c2': best['c2'],
          'c3': best['c3']}
    model = model_class(**kw).to(device)
    return {'c1': best['c1'], 'c2': best['c2'], 'c3': best['c3'],
            'nmse_db': best['nmse_db'], 'model': model,
            'tune_time_s': elapsed, 'n_evals': n_evals}


def tune_hyper_backprop(model_class, model_kwargs: dict,
                        train_data: Dict[str, torch.Tensor],
                        val_data: Dict[str, torch.Tensor],
                        num_epochs: int = 150, lr: float = 5e-3,
                        batch_size: int = 256, device: str = 'cpu',
                        init: Dict[str, float] = None) -> Dict[str, Any]:
    """Directly optimise ``(c1, c2, c3)`` by backprop on MSE.

    Useful only when the 3 hyperparameters are learnable
    ``nn.Parameter``s and the rest of the model (W, A) is fixed.
    """
    x_train, y_train = train_data['x'].to(device), train_data['y'].to(device)
    x_val, y_val = val_data['x'].to(device), val_data['y'].to(device)

    init = init or {'c1': 1.0, 'c2': 0.0, 'c3': 1.0}
    kw = {**model_kwargs, **init}
    model = model_class(**kw).to(device)

    if not all(hasattr(model, p) for p in ['c1', 'c2', 'c3']):
        raise ValueError(
            "tune_hyper_backprop requires model to expose c1, c2, c3")

    for nm, p in model.named_parameters():
        p.requires_grad_(nm in ('c1', 'c2', 'c3'))

    opt_params = [model.c1, model.c2, model.c3]
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.05)

    n_evals = 0
    t0 = time.time()
    best_val = float('inf')
    best_c = None
    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(x_train.shape[0])
        for i in range(0, min(x_train.shape[0], 2000), batch_size):
            idx = perm[i:i + batch_size]
            xh = model(y_train[idx])
            loss = nn.functional.mse_loss(xh, x_train[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 5.0)
            optimizer.step()
            n_evals += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            xh = model(y_val[:500])
            v = nmse_db(xh, x_val[:500])
        if v < best_val:
            best_val = v
            best_c = (model.c1.item(), model.c2.item(), model.c3.item())

    elapsed = time.time() - t0
    c1, c2, c3 = best_c
    kw = {**model_kwargs, 'c1': c1, 'c2': c2, 'c3': c3}
    final = model_class(**kw).to(device)
    return {'c1': c1, 'c2': c2, 'c3': c3,
            'nmse_db': best_val, 'model': final,
            'tune_time_s': elapsed, 'n_evals': n_evals}
