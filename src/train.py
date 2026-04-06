"""Training pipeline for unfolded sparse recovery models."""

import torch
import torch.nn as nn
import numpy as np
import optuna
from typing import Dict, Any

from src.utils.metrics import nmse, nmse_db


def train_unfolded_model(model: nn.Module, train_data: Dict[str, torch.Tensor],
                         val_data: Dict[str, torch.Tensor],
                         num_epochs: int = 200, lr: float = 1e-3,
                         batch_size: int = 256, progressive: bool = True,
                         device: str = 'cpu',
                         grad_clip: float = 5.0) -> Dict[str, list]:
    """Train LISTA / ALISTA / BlockLISTA with backprop."""
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

    model.eval()
    with torch.no_grad():
        x_hat = model(y_val)
        best_result['nmse_db_finetuned'] = nmse_db(x_hat, x_val)

    best_result['model'] = model
    return best_result
