"""
Differentiable Tree-HyperLISTA: Image CS experiments.

Tests:
  1. DiffTH-hybrid: end-to-end trainable via backprop (not Bayesian opt)
  2. DiffTH-SS: self-supervised with differentiable gradients
  3. DiffTH-Amort: pre-tuned + SS adaptation
  vs baselines: Tree-CoSaMP, TH-hybrid (original), HyperLISTA
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
import pywt

from src.data.image_cs import load_set11, image_cs_experiment
from src.data.wavelet_tree import build_simple_binary_wavelet_tree
from src.data.tree_synthetic import generate_tree_support
from src.models.diff_tree_hyperlista import (
    DiffTreeHyperLISTA, SelfSupervisedDiffTreeHyperLISTA
)
from src.models.tree_hyperlista import TreeHyperLISTA
from src.models.tree_classical import TreeCoSaMP
from src.models.tree_baselines import TreeFISTA
from src.models.hyperlista import HyperLISTA
from src.train import tune_hyper_model
from src.utils.metrics import nmse_db

PATCH_SIZE = 32
WAVELET = 'haar'
LEVEL = 2
K_LAYERS = 16
TARGET_K = 60
RHO = 0.5
CS_RATIOS = [0.25, 0.5]  # skip 0.1 for speed — SS struggles at low CS
N_TRIALS = 60


def generate_wavelet_tree_sparse_data(tree_info, A, num_samples, target_K,
                                       snr_db=40.0, seed=42):
    rng = np.random.RandomState(seed)
    n = tree_info['n']
    m = A.shape[0]
    X = np.zeros((num_samples, n), dtype=np.float32)
    for i in range(num_samples):
        support = generate_tree_support(tree_info, target_K, rng)
        active = np.where(support)[0]
        vals = rng.randn(len(active)).astype(np.float32)
        X[i, active] = np.sign(vals) * (np.abs(vals) * 1.5 + 0.5)
    Y_clean = X @ A.T
    sigma = np.sqrt(np.mean(Y_clean ** 2) * 10 ** (-snr_db / 10.0))
    Y = Y_clean + rng.randn(num_samples, m).astype(np.float32) * sigma
    return {
        'x': torch.from_numpy(X).float(),
        'y': torch.from_numpy(Y).float(),
    }


def train_diff_model(model, train_data, val_data, num_epochs=80,
                     lr=5e-4, batch_size=256, device='cpu'):
    """Train DiffTreeHyperLISTA with backprop (not Bayesian opt!)."""
    model = model.to(device)
    x_tr = train_data['x'].to(device)
    y_tr = train_data['y'].to(device)
    x_va = val_data['x'].to(device)
    y_va = val_data['y'].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        model.train()

        # Temperature annealing: start smooth, sharpen
        temp = 2.0 + (10.0 - 2.0) * min(epoch / max(num_epochs * 0.7, 1), 1.0)
        model.set_temperature(temp)

        perm = torch.randperm(x_tr.shape[0])
        for i in range(0, x_tr.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            x_hat = model(y_tr[idx])
            loss = torch.nn.functional.mse_loss(x_hat, x_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            model.set_temperature(10.0)
            x_val_hat = model(y_va[:500])
            val_nmse = nmse_db(x_val_hat, x_va[:500])
        if val_nmse < best_val:
            best_val = val_nmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1}/{num_epochs}: val_nmse={val_nmse:.2f} dB, "
                  f"temp={temp:.1f}, best={best_val:.2f} dB")

    if best_state:
        model.load_state_dict(best_state)
    model.set_temperature(10.0)
    return model


def run_experiment(device='cpu'):
    os.makedirs('results/diff_tree_cs', exist_ok=True)

    print("Loading test images...")
    images = load_set11('data/Set11')
    print(f"  Loaded {len(images)} images")

    test_patch = np.zeros((PATCH_SIZE, PATCH_SIZE))
    coeffs = pywt.wavedec2(test_patch, WAVELET, level=LEVEL)
    coeff_arr, _ = pywt.coeffs_to_array(coeffs)
    n = coeff_arr.size

    tree_info = build_simple_binary_wavelet_tree(n, level=LEVEL)
    print(f"  Tree: n={n}, depth={tree_info['max_depth']}")

    all_results = {}

    for cs_ratio in CS_RATIOS:
        m = int(n * cs_ratio)
        print(f"\n{'='*60}")
        print(f"  CS RATIO = {cs_ratio} (m={m}, n={n})")
        print(f"{'='*60}")

        rng = np.random.RandomState(42)
        Phi = rng.randn(m, n).astype(np.float32) / np.sqrt(m)
        norms = np.linalg.norm(Phi, axis=0, keepdims=True)
        Phi = Phi / np.maximum(norms, 1e-12)

        tr = generate_wavelet_tree_sparse_data(tree_info, Phi, 3000, TARGET_K, seed=100)
        va = generate_wavelet_tree_sparse_data(tree_info, Phi, 500, TARGET_K, seed=200)

        models = {}

        # --- Baselines ---
        print("  [1] Tree-CoSaMP")
        models['Tree-CoSaMP'] = TreeCoSaMP(
            Phi, tree_info, target_K=TARGET_K, max_iter=K_LAYERS, rho=RHO)

        print("  [2] Tree-FISTA")
        models['Tree-FISTA'] = TreeFISTA(
            Phi, tree_info, lam=0.05, max_iter=K_LAYERS, rho=RHO, target_K=TARGET_K)

        print("  [3] HyperLISTA")
        hl = tune_hyper_model(HyperLISTA, {'A': Phi, 'num_layers': K_LAYERS},
                              tr, va, n_trials=N_TRIALS, device=device)
        models['HyperLISTA'] = hl['model']

        # --- Original TH (Bayesian opt) ---
        print("  [4] TH-hybrid (Bayesian opt)")
        th = tune_hyper_model(
            TreeHyperLISTA,
            {'A': Phi, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'hybrid_tree', 'rho': RHO},
            tr, va, n_trials=N_TRIALS, device=device)
        models['TH-hybrid'] = th['model']
        pre_c1, pre_c2, pre_c3 = th['c1'], th['c2'], th['c3']

        # --- DiffTH: trained with backprop ---
        print("  [5] DiffTH-hybrid (backprop trained)")
        diff_model = DiffTreeHyperLISTA(
            Phi, tree_info, num_layers=K_LAYERS,
            support_mode='hybrid_tree', rho=RHO, temperature=5.0)
        diff_model = train_diff_model(diff_model, tr, va,
                                       num_epochs=80, lr=5e-4, device=device)
        models['DiffTH-backprop'] = diff_model

        # --- DiffTH-SS: self-supervised, NO training data ---
        print("  [6] DiffTH-SS (self-supervised, no training data)")
        ss_model = SelfSupervisedDiffTreeHyperLISTA(
            Phi, tree_info, num_layers=K_LAYERS,
            support_mode='hybrid_tree', rho=RHO,
            adapt_steps=15, adapt_lr=0.05,
            temperature=5.0, num_restarts=3)
        models['DiffTH-SS'] = ss_model

        # --- DiffTH-Amort: pre-tuned + SS adaptation ---
        print("  [7] DiffTH-Amort (pre-tuned + SS)")
        amort = SelfSupervisedDiffTreeHyperLISTA(
            Phi, tree_info, num_layers=K_LAYERS,
            support_mode='hybrid_tree', rho=RHO,
            adapt_steps=10, adapt_lr=0.03,
            temperature=5.0, num_restarts=1)
        amort.set_pretrained_init(pre_c1, pre_c2, pre_c3)
        models['DiffTH-Amort'] = amort

        # --- Run ---
        print("\n  Running image CS experiment...")
        summary = image_cs_experiment(
            images, models, cs_ratio=cs_ratio, patch_size=PATCH_SIZE,
            wavelet=WAVELET, level=LEVEL,
            snr_db=40.0, device=device, seed=42)

        all_results[f'ratio_{cs_ratio}'] = summary

        print(f"\n  Results at CS ratio {cs_ratio}:")
        print(f"  {'Model':<25} {'PSNR(dB)':<16} {'SSIM':<12}")
        print("  " + "=" * 53)
        for name in summary:
            r = summary[name]
            print(f"  {name:<25} {r['psnr_mean']:>6.2f}+/-{r['psnr_std']:.2f}"
                  f"  {r['ssim_mean']:.4f}+/-{r['ssim_std']:.4f}")

    with open('results/diff_tree_cs/diff_tree_cs_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to results/diff_tree_cs/diff_tree_cs_results.json")
    return all_results


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")
    run_experiment(device=device)
