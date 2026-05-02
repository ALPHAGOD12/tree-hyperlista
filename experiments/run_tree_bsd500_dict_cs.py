"""Exp 6 (optional) -- BSD500 patch-dictionary image CS.

Appendix B of the HyperLISTA paper builds a learned patch dictionary
``T`` from BSD500 and then performs CS in the dictionary domain:

    y = A x,  A = Phi * T,  f_hat = T * x_hat.

This experiment mirrors that setup using a tree-structured dictionary:
each atom of ``T`` corresponds to a node of a binary tree over the
patch wavelet coefficients, so that tree-sparse ``x`` naturally induces
patch reconstructions with coarse-to-fine support.

It is optional because it adds a dictionary-learning dependency. If
BSD500 is not locally available, the script falls back to random
natural-image patches (Set11) for ``T`` construction via KSVD-like
patch mining. The KSVD-like routine here is a pragmatic substitute --
run ``pip install scikit-learn`` to use sklearn's MiniBatch dict
learning for a better dictionary.

The script is structured as a function so it can be skipped if data is
missing, without breaking the rest of the suite.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from skimage.io import imread
    from skimage.color import rgb2gray
    from skimage.transform import resize
    _SKIMAGE_OK = True
except Exception:
    _SKIMAGE_OK = False

from src.data.tree_synthetic import build_balanced_tree
from src.models.tree_hyperlista import TreeHyperLISTA
from src.models.hyperlista import HyperLISTA
from src.train import tune_hyper_model
from src.utils.metrics import nmse_db


PATCH = 16
N_ATOMS = 255           # binary tree depth 7
CS_RATIOS = [0.10, 0.25, 0.50]
K_LAYERS = 16
N_TRAIN_PATCHES = 5000
N_TEST_PATCHES = 1000
N_TRIALS = 30
DICT_ITERS = 40


def _load_image_patches(image_dir: str, patch_size: int,
                        num_patches: int, rng: np.random.RandomState):
    """Sample ``num_patches`` grayscale patches from images in ``image_dir``.

    Falls back to ``data/Set11`` (bundled with most CS codebases) or
    synthetic smooth noise if no images are found.
    """
    files = []
    if os.path.isdir(image_dir):
        for ext in ['*.jpg', '*.png', '*.tif', '*.bmp']:
            files.extend(glob.glob(os.path.join(image_dir, ext)))
    if not _SKIMAGE_OK or not files:
        print(f"  [warn] no images in {image_dir!r}; using synthetic texture")
        X = rng.randn(num_patches, patch_size * patch_size).astype(np.float32)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    patches = []
    for path in files:
        try:
            img = imread(path)
            if img.ndim == 3:
                img = rgb2gray(img)
            img = img.astype(np.float32)
            if img.max() > 1.5:
                img = img / 255.0
            H, W = img.shape
            for _ in range(max(10, num_patches // max(len(files), 1))):
                if len(patches) >= num_patches:
                    break
                i0 = rng.randint(0, max(1, H - patch_size))
                j0 = rng.randint(0, max(1, W - patch_size))
                p = img[i0:i0 + patch_size, j0:j0 + patch_size]
                if p.shape == (patch_size, patch_size):
                    patches.append(p.flatten() - p.mean())
            if len(patches) >= num_patches:
                break
        except Exception as e:
            print(f"  [warn] skip {path}: {e}")
    X = np.stack(patches[:num_patches]).astype(np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def _ksvd_like_dict(patches: np.ndarray, n_atoms: int, n_iter: int = 30):
    """Tiny KSVD-style dictionary learning.

    Not production quality but sufficient to demonstrate the pipeline.
    Returns a dictionary ``T`` of shape ``(patch_dim, n_atoms)``.
    """
    d, n = patches.shape[1], patches.shape[0]
    rng = np.random.RandomState(0)
    T = rng.randn(d, n_atoms)
    T /= np.linalg.norm(T, axis=0, keepdims=True) + 1e-8

    for it in range(n_iter):
        # sparse coding: keep the 10 largest inner products per patch
        C = patches @ T
        thresh = np.sort(np.abs(C), axis=1)[:, -10:-9]
        C = np.where(np.abs(C) >= thresh, C, 0.0)
        # dictionary update: least squares with C fixed
        new_T, *_ = np.linalg.lstsq(C, patches, rcond=None)
        T = new_T.T
        T /= np.linalg.norm(T, axis=0, keepdims=True) + 1e-8
        if it % 10 == 0:
            err = np.linalg.norm(patches - C @ T.T) / np.sqrt(n)
            print(f"    KSVD iter {it}: avg err = {err:.4f}")
    return T  # (d, n_atoms)


def run_bsd500_dict_cs(image_dir='data/BSD500', device='cpu'):
    os.makedirs('results/tree_bsd500_dict_cs', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)
    rng = np.random.RandomState(7)

    print("Step 1 -- mining training patches")
    train_patches = _load_image_patches(image_dir, PATCH, N_TRAIN_PATCHES, rng)
    test_patches = _load_image_patches(image_dir, PATCH, N_TEST_PATCHES, rng)
    print(f"  train patches: {train_patches.shape}, "
          f"test: {test_patches.shape}")

    print("Step 2 -- learning a patch dictionary T (KSVD-like)")
    T = _ksvd_like_dict(train_patches, N_ATOMS, n_iter=DICT_ITERS)
    print(f"  T shape = {T.shape}")

    # Pick tree depth so the number of atoms matches 2^(d+1) - 1.
    depth = int(round(np.log2(N_ATOMS + 1))) - 1
    tree_info = build_balanced_tree(depth, 2)
    if tree_info['n'] != N_ATOMS:
        raise ValueError(
            f"N_ATOMS={N_ATOMS} is not of the form 2^(d+1)-1; "
            f"balanced binary tree gives n={tree_info['n']} (depth={depth}).")
    print(f"  tree depth={depth}, n={tree_info['n']}")

    results = {}
    for ratio in CS_RATIOS:
        print(f"\nStep 3 -- CS ratio = {ratio}")
        m = max(4, int(round(PATCH * PATCH * ratio)))
        Phi = rng.randn(m, PATCH * PATCH).astype(np.float32) / np.sqrt(m)
        Phi /= np.linalg.norm(Phi, axis=0, keepdims=True) + 1e-8
        A = (Phi @ T).astype(np.float32)
        A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-8

        def _encode(patches):
            G, *_ = np.linalg.lstsq(T, patches.T, rcond=None)
            X = G.T.astype(np.float32)
            Y = (X @ A.T).astype(np.float32)
            noise = rng.randn(*Y.shape).astype(np.float32) * \
                (np.linalg.norm(Y) / Y.size) * 0.05
            return X, Y + noise

        X_tr, Y_tr = _encode(train_patches)
        X_va, Y_va = _encode(train_patches[:800])
        X_te, Y_te = _encode(test_patches)
        tr = {'x': torch.from_numpy(X_tr), 'y': torch.from_numpy(Y_tr)}
        va = {'x': torch.from_numpy(X_va), 'y': torch.from_numpy(Y_va)}
        te = {'x': torch.from_numpy(X_te), 'y': torch.from_numpy(Y_te)}

        print("  tuning HyperLISTA and Tree-HyperLISTA")
        hl = tune_hyper_model(HyperLISTA, {'A': A, 'num_layers': K_LAYERS},
                              tr, va, n_trials=N_TRIALS,
                              device=device)['model']
        th = tune_hyper_model(
            TreeHyperLISTA,
            {'A': A, 'tree_info': tree_info, 'num_layers': K_LAYERS,
             'support_mode': 'hybrid_tree', 'rho': 0.5},
            tr, va, n_trials=N_TRIALS, device=device)['model']

        def _psnr(recon, true):
            mse = ((recon - true) ** 2).mean()
            return 10 * np.log10(1.0 / max(mse, 1e-12))

        with torch.no_grad():
            x_hat_hl = hl(te['y'].to(device)).cpu().numpy()
            x_hat_th = th(te['y'].to(device)).cpu().numpy()
        patch_hl = x_hat_hl @ T.T
        patch_th = x_hat_th @ T.T

        psnr_hl = np.mean([_psnr(patch_hl[i], test_patches[i])
                           for i in range(len(test_patches))])
        psnr_th = np.mean([_psnr(patch_th[i], test_patches[i])
                           for i in range(len(test_patches))])
        results[f"ratio_{ratio}"] = {
            'HyperLISTA': {'psnr_mean': float(psnr_hl)},
            'TH-hybrid':  {'psnr_mean': float(psnr_th)},
        }
        print(f"    PSNR  HyperLISTA={psnr_hl:.2f}   "
              f"TH-hybrid={psnr_th:.2f}")

    with open('results/tree_bsd500_dict_cs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n[saved] results/tree_bsd500_dict_cs/results.json")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ratios = [r for r in CS_RATIOS]
    hl_vals = [results[f"ratio_{r}"]['HyperLISTA']['psnr_mean'] for r in ratios]
    th_vals = [results[f"ratio_{r}"]['TH-hybrid']['psnr_mean'] for r in ratios]
    width = 0.35
    x = np.arange(len(ratios))
    ax.bar(x - width / 2, hl_vals, width, label='HyperLISTA',
           color='#d62728', edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, th_vals, width, label='TH-hybrid',
           color='#e377c2', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.2f}' for r in ratios])
    ax.set_xlabel('CS ratio')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Exp 6 (optional): BSD500 Dictionary CS')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('paper/tree_figures/tree_fig_bsd500_dict.pdf')
    fig.savefig('paper/tree_figures/tree_fig_bsd500_dict.png', dpi=200)
    plt.close(fig)
    print("[saved] paper/tree_figures/tree_fig_bsd500_dict.{pdf,png}")
    return results


if __name__ == '__main__':
    from src.utils.sensing import pick_device; dev = pick_device()
    run_bsd500_dict_cs(device=dev)
