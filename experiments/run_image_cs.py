"""
Real-data image CS experiments: wavelet-domain compressed sensing on Set11.

Trains models on synthetic block-sparse data matching wavelet statistics,
then evaluates on real image patches.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json

from src.data.image_cs import load_set11, image_cs_experiment
from src.data.synthetic import BlockSparseDataset
from src.models.ista import GroupFISTA
from src.models.alista import ALISTA
from src.models.hyperlista import HyperLISTA
from src.models.struct_hyperlista import StructHyperLISTA
from src.models.ada_blocklista import AdaBlockLISTA
from src.train import train_unfolded_model, tune_hyper_model
from src.utils.sensing import gaussian_sensing

PATCH_SIZE = 32
WAVELET = 'haar'
LEVEL = 2
K = 16
GS = 16
SG = 8
CS_RATIOS = [0.1, 0.25, 0.5]
TRAIN_EPOCHS = 150
N_TRIALS = 60


def build_models_for_cs(A_np, gs, K, tr, va, device):
    """Build and train models for a given sensing matrix."""
    models = {}

    best_lam, best_nmse = 0.1, float('inf')
    for lam_try in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        from src.utils.metrics import nmse_db
        g = GroupFISTA(A_np, gs, lam=lam_try, max_iter=K)
        with torch.no_grad():
            xh = g.solve(va['y'][:100])
            v = nmse_db(xh, va['x'][:100])
        if v < best_nmse:
            best_nmse = v
            best_lam = lam_try

    print(f"    Group-FISTA (lam={best_lam})")
    models['Group-FISTA'] = GroupFISTA(A_np, gs, lam=best_lam, max_iter=K)

    print("    ALISTA")
    alista = ALISTA(A_np, num_layers=K)
    train_unfolded_model(alista, tr, va, num_epochs=TRAIN_EPOCHS,
                         lr=5e-3, batch_size=128, device=device)
    models['ALISTA'] = alista

    print("    Ada-BlockLISTA (tied)")
    abl = AdaBlockLISTA(A_np, gs, num_layers=K, tied=True)
    train_unfolded_model(abl, tr, va, num_epochs=TRAIN_EPOCHS * 2,
                         lr=5e-4, batch_size=128, progressive=False, device=device)
    models['Ada-BLISTA-T'] = abl

    print("    HyperLISTA")
    hl = tune_hyper_model(HyperLISTA, {'A': A_np, 'num_layers': K},
                          tr, va, n_trials=N_TRIALS, device=device)
    models['HyperLISTA'] = hl['model']

    for mode in ['topk_group', 'hybrid']:
        print(f"    SH-{mode}")
        sh = tune_hyper_model(
            StructHyperLISTA,
            {'A': A_np, 'group_size': gs, 'num_layers': K, 'support_mode': mode},
            tr, va, n_trials=N_TRIALS, device=device)
        models[f'SH-{mode}'] = sh['model']

    return models


def run_image_cs(device='cpu'):
    os.makedirs('results/image_cs', exist_ok=True)

    print("Loading test images...")
    images = load_set11('data/Set11')
    print(f"  Loaded {len(images)} images")

    import pywt
    test_patch = np.zeros((PATCH_SIZE, PATCH_SIZE))
    coeffs = pywt.wavedec2(test_patch, WAVELET, level=LEVEL)
    coeff_arr, _ = pywt.coeffs_to_array(coeffs)
    n = coeff_arr.size
    print(f"  Wavelet coefficients per patch: n={n}")

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

        print("  Training on synthetic block-sparse data...")
        ds = BlockSparseDataset(n=n, m=m, group_size=GS,
                                num_active_groups=SG, snr_db=40.0,
                                seed=42, matrix_seed=42)
        A_np = Phi
        ds.A = A_np
        ds.A_tensor = torch.from_numpy(A_np).float()
        ds.A_pinv = np.linalg.pinv(A_np).astype(np.float32)
        ds.A_pinv_tensor = torch.from_numpy(ds.A_pinv).float()
        ds.n = n
        ds.m = m

        tr = ds.generate(5000, seed=100)
        va = ds.generate(1000, seed=200)

        models = build_models_for_cs(A_np, GS, K, tr, va, device)

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

    with open('results/image_cs/image_cs_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nAll image CS results saved to results/image_cs/image_cs_results.json")
    return all_results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    run_image_cs(device=device)
