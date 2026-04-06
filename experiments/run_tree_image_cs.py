"""
Wavelet-domain image CS experiments with tree-sparse recovery models.

Trains Tree-HyperLISTA and baselines on synthetic tree-sparse data
matching the wavelet tree structure, then evaluates on real image patches.

Key fix: training data tree structure is aligned with the wavelet
coefficient tree so that the tree-consistent support selection
operates on a meaningful hierarchy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
import pywt

from src.data.image_cs import (load_set11, image_cs_experiment,
                                extract_patches, dwt2_to_vector)
from src.data.wavelet_tree import build_wavelet_tree, build_simple_binary_wavelet_tree
from src.data.tree_synthetic import generate_tree_support
from src.models.tree_hyperlista import TreeHyperLISTA
from src.models.tree_baselines import TreeISTA, TreeFISTA
from src.models.tree_classical import TreeIHT, TreeCoSaMP
from src.models.hyperlista import HyperLISTA
from src.models.alista import ALISTA
from src.models.struct_hyperlista import StructHyperLISTA
from src.models.ista import GroupFISTA
from src.train import train_unfolded_model, tune_hyper_model
from src.utils.metrics import nmse_db

PATCH_SIZE = 32
WAVELET = 'haar'
LEVEL = 2
K_LAYERS = 16
TARGET_K = 60
RHO = 0.5
GS = 16  # group size for block baselines
CS_RATIOS = [0.1, 0.25, 0.5]
TRAIN_EPOCHS = 150
N_TRIALS = 60


def generate_wavelet_tree_sparse_data(tree_info, A, num_samples, target_K,
                                       snr_db=40.0, seed=42):
    """Generate tree-sparse training data using the wavelet tree structure.

    This ensures the training data has tree-consistent supports that match
    the actual wavelet coefficient hierarchy.
    """
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
        'x': torch.from_numpy(X),
        'y': torch.from_numpy(Y),
        'noise_std': sigma,
    }


def estimate_wavelet_sparsity(images, patch_size, wavelet, level,
                               threshold_frac=0.05):
    """Estimate typical tree sparsity of wavelet coefficients from real images."""
    total_coeffs = 0
    total_nonzero = 0
    for img in images[:3]:
        patches, _ = extract_patches(img, patch_size)
        for p in patches[:20]:
            vec, _, _ = dwt2_to_vector(p, wavelet, level)
            abs_v = np.abs(vec)
            thresh = threshold_frac * abs_v.max()
            total_nonzero += np.sum(abs_v > thresh)
            total_coeffs += len(vec)
    avg_sparsity = total_nonzero / max(total_coeffs / len(images[:3]) / 20, 1)
    return int(avg_sparsity)


def build_tree_models_for_cs(A_np, tree_info, K_layers, target_K,
                              tr, va, device):
    """Build and train all models for image CS comparison."""
    models = {}
    n = A_np.shape[1]

    # --- Classical tree baselines ---
    best_lam, best_nmse = 0.1, float('inf')
    for lam_try in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        g = TreeFISTA(A_np, tree_info, lam=lam_try, max_iter=K_layers,
                      rho=RHO, target_K=target_K)
        g = g.to(device)
        with torch.no_grad():
            xh = g.solve(va['y'][:200].to(device))
            v = nmse_db(xh, va['x'][:200].to(device))
        if v < best_nmse:
            best_nmse = v
            best_lam = lam_try
    print(f"    Best lambda: {best_lam} (val={best_nmse:.2f} dB)")

    print("    Tree-FISTA")
    models['Tree-FISTA'] = TreeFISTA(A_np, tree_info, lam=best_lam,
                                      max_iter=K_layers, rho=RHO,
                                      target_K=target_K)

    print("    Tree-IHT")
    models['Tree-IHT'] = TreeIHT(A_np, tree_info, target_K=target_K,
                                  max_iter=K_layers, rho=RHO)

    print("    Tree-CoSaMP")
    models['Tree-CoSaMP'] = TreeCoSaMP(A_np, tree_info, target_K=target_K,
                                        max_iter=K_layers, rho=RHO)

    # --- Elementwise baselines ---
    print("    ALISTA")
    alista = ALISTA(A_np, num_layers=K_layers)
    train_unfolded_model(alista, tr, va, num_epochs=TRAIN_EPOCHS,
                         lr=5e-3, batch_size=128, device=device)
    models['ALISTA'] = alista

    print("    HyperLISTA (elementwise)")
    hl = tune_hyper_model(HyperLISTA, {'A': A_np, 'num_layers': K_layers},
                          tr, va, n_trials=N_TRIALS, device=device)
    models['HyperLISTA'] = hl['model']

    # --- Block baselines (Struct-HyperLISTA) ---
    print("    Group-FISTA (block)")
    best_lam_b, best_nmse_b = 0.1, float('inf')
    for lam_try in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        g = GroupFISTA(A_np, GS, lam=lam_try, max_iter=K_layers)
        with torch.no_grad():
            xh = g.solve(va['y'][:100])
            v = nmse_db(xh, va['x'][:100])
        if v < best_nmse_b:
            best_nmse_b = v
            best_lam_b = lam_try
    models['Group-FISTA'] = GroupFISTA(A_np, GS, lam=best_lam_b, max_iter=K_layers)

    print("    SH-topk (block)")
    sh_topk = tune_hyper_model(
        StructHyperLISTA,
        {'A': A_np, 'group_size': GS, 'num_layers': K_layers,
         'support_mode': 'topk_group'},
        tr, va, n_trials=N_TRIALS, device=device)
    models['SH-topk'] = sh_topk['model']

    # --- Tree-HyperLISTA (proposed) ---
    print("    TH-hybrid (tree, proposed)")
    th = tune_hyper_model(
        TreeHyperLISTA,
        {'A': A_np, 'tree_info': tree_info, 'num_layers': K_layers,
         'support_mode': 'hybrid_tree', 'rho': RHO},
        tr, va, n_trials=N_TRIALS, device=device)
    models['TH-hybrid'] = th['model']

    print("    TH-hard (tree)")
    th_hard = tune_hyper_model(
        TreeHyperLISTA,
        {'A': A_np, 'tree_info': tree_info, 'num_layers': K_layers,
         'support_mode': 'tree_hard', 'rho': RHO},
        tr, va, n_trials=N_TRIALS, device=device)
    models['TH-hard'] = th_hard['model']

    return models


def run_tree_image_cs(device='cpu'):
    os.makedirs('results/tree_image_cs', exist_ok=True)

    print("Loading test images...")
    images = load_set11('data/Set11')
    print(f"  Loaded {len(images)} images")

    test_patch = np.zeros((PATCH_SIZE, PATCH_SIZE))
    coeffs = pywt.wavedec2(test_patch, WAVELET, level=LEVEL)
    coeff_arr, _ = pywt.coeffs_to_array(coeffs)
    n = coeff_arr.size
    print(f"  Wavelet coefficients per patch: n={n}")

    est_K = estimate_wavelet_sparsity(images, PATCH_SIZE, WAVELET, LEVEL)
    target_K = max(est_K, TARGET_K)
    print(f"  Estimated wavelet sparsity: ~{est_K}, using target_K={target_K}")

    # Build wavelet tree matching the coefficient layout
    tree_info = build_simple_binary_wavelet_tree(n, level=LEVEL)
    print(f"  Wavelet tree: n={tree_info['n']}, max_depth={tree_info['max_depth']}, "
          f"leaves={len(tree_info['leaves'])}")

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

        # Generate tree-sparse training data using the wavelet tree
        print("  Generating tree-sparse training data (wavelet tree aligned)...")
        tr = generate_wavelet_tree_sparse_data(
            tree_info, Phi, num_samples=5000,
            target_K=target_K, snr_db=40.0, seed=100)
        va = generate_wavelet_tree_sparse_data(
            tree_info, Phi, num_samples=1000,
            target_K=target_K, snr_db=40.0, seed=200)

        models = build_tree_models_for_cs(Phi, tree_info, K_LAYERS,
                                           target_K, tr, va, device)

        print("\n  Running image CS experiment...")
        summary = image_cs_experiment(
            images, models, cs_ratio=cs_ratio, patch_size=PATCH_SIZE,
            wavelet=WAVELET, level=LEVEL, group_size=GS,
            snr_db=40.0, device=device, seed=42)

        all_results[f'ratio_{cs_ratio}'] = summary

        print(f"\n  Results at CS ratio {cs_ratio}:")
        print(f"  {'Model':<20} {'PSNR(dB)':<16} {'SSIM':<12}")
        print("  " + "=" * 48)
        for name in summary:
            r = summary[name]
            print(f"  {name:<20} {r['psnr_mean']:>6.2f}+/-{r['psnr_std']:.2f}"
                  f"  {r['ssim_mean']:.4f}+/-{r['ssim_std']:.4f}")

    with open('results/tree_image_cs/tree_image_cs_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nAll tree image CS results saved to "
          "results/tree_image_cs/tree_image_cs_results.json")
    return all_results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    run_tree_image_cs(device=device)
