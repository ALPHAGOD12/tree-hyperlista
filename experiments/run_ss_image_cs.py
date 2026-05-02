"""
Self-Supervised Tree-HyperLISTA: Image CS experiments.

Compares:
  1. TH-hybrid (pre-tuned on synthetic data, no adaptation)
  2. TH-SS (self-supervised, NO training data, adapts per-batch)
  3. TH-Amortized (pre-tuned init + per-batch adaptation)
  4. Tree-CoSaMP (classical, no learning)
  5. HyperLISTA (elementwise, 3 params)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
import pywt

from src.data.image_cs import (load_set11, image_cs_experiment)
from src.data.wavelet_tree import build_simple_binary_wavelet_tree
from src.data.tree_synthetic import generate_tree_support
from src.models.tree_hyperlista import TreeHyperLISTA
from src.models.tree_hyperlista_ss import (
    SelfSupervisedTreeHyperLISTA, AmortizedSSTreeHyperLISTA
)
from src.models.tree_baselines import TreeFISTA
from src.models.tree_classical import TreeCoSaMP
from src.models.hyperlista import HyperLISTA
from src.train import tune_hyper_model
from src.utils.metrics import nmse_db

PATCH_SIZE = 32
WAVELET = 'haar'
LEVEL = 2
K_LAYERS = 16
TARGET_K = 60
RHO = 0.5
CS_RATIOS = [0.1, 0.25, 0.5]
N_TRIALS = 60


def generate_wavelet_tree_sparse_data(tree_info, A, num_samples, target_K,
                                       snr_db=40.0, seed=42):
    """Generate tree-sparse training data aligned with wavelet tree."""
    rng = np.random.RandomState(seed)
    n = tree_info['n']
    m = A.shape[0]
    X = np.zeros((num_samples, n), dtype=np.float32)
    for i in range(num_samples):
        support = generate_tree_support(tree_info, target_K, rng)
        active_indices = np.where(support)[0]
        vals = rng.randn(len(active_indices)).astype(np.float32)
        signs = np.sign(vals)
        mags = np.abs(vals) * 1.5 + 0.5
        X[i, active_indices] = signs * mags
    Y_clean = X @ A.T
    sigma = np.sqrt(np.mean(Y_clean ** 2) * 10 ** (-snr_db / 10.0))
    noise = rng.randn(num_samples, m).astype(np.float32) * sigma
    Y = Y_clean + noise
    return {
        'x': torch.from_numpy(X).float(),
        'y': torch.from_numpy(Y.astype(np.float32)).float(),
        'noise_std': sigma,
    }


def run_ss_experiment(device='cpu'):
    os.makedirs('results/ss_image_cs', exist_ok=True)

    print("Loading test images...")
    images = load_set11('data/Set11')
    print(f"  Loaded {len(images)} images")

    test_patch = np.zeros((PATCH_SIZE, PATCH_SIZE))
    coeffs = pywt.wavedec2(test_patch, WAVELET, level=LEVEL)
    coeff_arr, _ = pywt.coeffs_to_array(coeffs)
    n = coeff_arr.size
    print(f"  n={n}")

    tree_info = build_simple_binary_wavelet_tree(n, level=LEVEL)
    print(f"  Tree: n={tree_info['n']}, depth={tree_info['max_depth']}")

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

        models = {}

        # --- 1. Tree-CoSaMP (classical baseline) ---
        print("  [1] Tree-CoSaMP")
        models['Tree-CoSaMP'] = TreeCoSaMP(
            Phi, tree_info, target_K=TARGET_K,
            max_iter=K_LAYERS, rho=RHO)

        # --- 2. Tree-FISTA (classical baseline) ---
        print("  [2] Tree-FISTA")
        models['Tree-FISTA'] = TreeFISTA(
            Phi, tree_info, lam=0.05, max_iter=K_LAYERS,
            rho=RHO, target_K=TARGET_K)

        # --- 3. HyperLISTA (elementwise, pre-tuned) ---
        print("  [3] HyperLISTA (pre-tuned on synthetic)")
        tr = generate_wavelet_tree_sparse_data(
            tree_info, Phi, 3000, TARGET_K, snr_db=40.0, seed=100)
        va = generate_wavelet_tree_sparse_data(
            tree_info, Phi, 500, TARGET_K, snr_db=40.0, seed=200)
        hl = tune_hyper_model(HyperLISTA, {'A': Phi, 'num_layers': K_LAYERS},
                              tr, va, n_trials=N_TRIALS, device=device)
        models['HyperLISTA'] = hl['model']

        # --- 4. TH-hybrid (pre-tuned, no adaptation) ---
        print("  [4] TH-hybrid (pre-tuned, no adaptation)")
        th = tune_hyper_model(
            TreeHyperLISTA,
            {'A': Phi, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'hybrid_tree', 'rho': RHO},
            tr, va, n_trials=N_TRIALS, device=device)
        models['TH-hybrid'] = th['model']
        pretrained_c1 = th['c1']
        pretrained_c2 = th['c2']
        pretrained_c3 = th['c3']

        # --- 5. TH-SS: Self-Supervised, NO training data ---
        # Uses default init, adapts purely from measurements
        print("  [5] TH-SS (self-supervised, no training data)")
        for n_steps in [10, 20]:
            for lr in [0.05]:
                name = f'TH-SS-{n_steps}steps'
                models[name] = SelfSupervisedTreeHyperLISTA(
                    Phi, tree_info, num_layers=K_LAYERS,
                    c1_init=1.0, c2_init=0.0, c3_init=3.0,
                    support_mode='hybrid_tree', rho=RHO,
                    adapt_steps=n_steps, adapt_lr=lr,
                    num_restarts=3)

        # --- 6. TH-Amortized: Pre-tuned init + per-batch adaptation ---
        print("  [6] TH-Amortized (pre-tuned + adaptation)")
        for n_steps in [5, 10]:
            name = f'TH-Amort-{n_steps}steps'
            amort = AmortizedSSTreeHyperLISTA(
                Phi, tree_info, num_layers=K_LAYERS,
                support_mode='hybrid_tree', rho=RHO,
                adapt_steps=n_steps, adapt_lr=0.03,
                num_restarts=1)
            amort.set_pretrained_init(pretrained_c1, pretrained_c2, pretrained_c3)
            models[name] = amort

        # --- Run image CS ---
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

    with open('results/ss_image_cs/ss_image_cs_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nAll self-supervised image CS results saved.")
    return all_results


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")
    run_ss_experiment(device=device)
