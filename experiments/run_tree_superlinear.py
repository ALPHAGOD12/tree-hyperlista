"""Exp 5 -- Per-instance tuning and near-superlinear convergence.

Mirrors Appendix A of the HyperLISTA paper. Two phases:

    Phase 1 (per-instance Optuna): start from the dataset-tuned
        ``(c1, c2, c3)`` and run a small number of Optuna trials per
        test instance. Expectation: on a log-NMSE vs layer plot, the
        curve is concave (near-superlinear) for well-behaved instances.

    Phase 2 (CG switch): once the selected tree support stabilises for
        a few consecutive layers, restrict the remaining iterations to
        conjugate gradient on that support. This is the tree analogue
        of Algorithm 1, line 10 of the HyperLISTA paper.

We use a small problem (depth 6 binary tree, n=127, m=63, K=16,
SNR=40 dB) to keep per-instance search affordable.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import torch
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.tree_synthetic import TreeSparseDataset, build_balanced_tree
from src.models.tree_hyperlista import TreeHyperLISTA
from src.train import tune_hyper_model
from src.utils.metrics import nmse_db


TREE_DEPTH = 6
BRANCHING = 2
M_RATIO = 0.5
TARGET_K = 16
K_LAYERS = 24
RHO = 0.5
N_TRAIN, N_VAL = 2000, 400
N_TRIALS_GLOBAL = 50
N_TRIALS_LOCAL = 12
N_INSTANCES = 50


def cg_solve_on_support(A: torch.Tensor, y: torch.Tensor,
                        support_mask: torch.Tensor,
                        num_iter: int = 20, tol: float = 1e-10) -> torch.Tensor:
    """Least-squares CG restricted to a fixed support.

    For each sample b, solve min_x || A_S x - y ||_2^2 via CG, where S is
    the index set support_mask[b]. The full vector x_b is zero outside S.
    """
    batch, n = support_mask.shape
    device = y.device
    x = torch.zeros(batch, n, device=device)
    for b in range(batch):
        idx = support_mask[b].nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        A_s = A[:, idx]
        AtA = A_s.t() @ A_s
        rhs = A_s.t() @ y[b]
        xb = torch.zeros(idx.numel(), device=device)
        r = rhs - AtA @ xb
        p = r.clone()
        rs_old = torch.dot(r, r)
        for _ in range(num_iter):
            Ap = AtA @ p
            denom = torch.dot(p, Ap).clamp(min=1e-16)
            alpha = rs_old / denom
            xb = xb + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if rs_new < tol:
                break
            p = r + (rs_new / rs_old.clamp(min=1e-16)) * p
            rs_old = rs_new
        x[b, idx] = xb
    return x


def _per_instance_optuna(model_kwargs, c0, y_one, x_one, n_trials,
                         device, K_layers):
    c1_0, c2_0, c3_0 = c0

    def obj(trial):
        c1 = trial.suggest_float(
            'c1', max(0.01, c1_0 * 0.3), c1_0 * 3.0, log=True)
        c2 = trial.suggest_float('c2', c2_0 - 2.0, c2_0 + 2.0)
        c3 = trial.suggest_float(
            'c3', max(0.05, c3_0 * 0.3), c3_0 * 3.0, log=True)
        kw = {**model_kwargs, 'c1': c1, 'c2': c2, 'c3': c3,
              'num_layers': K_layers}
        m = TreeHyperLISTA(**kw).to(device).eval()
        with torch.no_grad():
            xh = m(y_one.unsqueeze(0))
        v = nmse_db(xh, x_one.unsqueeze(0))
        return v if np.isfinite(v) else 50.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _trajectory_nmse(model, y_one, x_one):
    model.eval()
    with torch.no_grad():
        traj = model(y_one.unsqueeze(0), return_trajectory=True)
    return [nmse_db(t, x_one.unsqueeze(0)) for t in traj]


def run_superlinear(device='cpu'):
    os.makedirs('results/tree_superlinear', exist_ok=True)
    os.makedirs('paper/tree_figures', exist_ok=True)

    tree_info = build_balanced_tree(TREE_DEPTH, BRANCHING)
    n = tree_info['n']
    m = int(n * M_RATIO)
    print(f"Superlinear: n={n}, m={m}, K_layers={K_LAYERS}, "
          f"instances={N_INSTANCES}")

    ds = TreeSparseDataset(tree_depth=TREE_DEPTH, branching=BRANCHING,
                           m_ratio=M_RATIO, target_sparsity=TARGET_K,
                           snr_db=40.0, seed=7, matrix_seed=13)
    tr = ds.generate(N_TRAIN, seed=1)
    va = ds.generate(N_VAL, seed=2)
    te = ds.generate(N_INSTANCES, seed=3)

    print("  [phase 0] dataset-wide tuning")
    model_kwargs = {'A': ds.A, 'tree_info': tree_info,
                    'num_layers': K_LAYERS,
                    'support_mode': 'hybrid_tree', 'rho': RHO}
    glob = tune_hyper_model(TreeHyperLISTA, model_kwargs, tr, va,
                            n_trials=N_TRIALS_GLOBAL, device=device)
    c0 = (glob['c1'], glob['c2'], glob['c3'])
    print(f"     global (c1,c2,c3) = ({c0[0]:.3f}, {c0[1]:.3f}, {c0[2]:.3f})")

    glob_model = glob['model']

    trajs_global, trajs_local, trajs_cg = [], [], []

    for idx in range(N_INSTANCES):
        y_one = te['y'][idx].to(device)
        x_one = te['x'][idx].to(device)

        tg = _trajectory_nmse(glob_model, y_one, x_one)
        trajs_global.append(tg)

        best = _per_instance_optuna(model_kwargs, c0, y_one, x_one,
                                    N_TRIALS_LOCAL, device, K_LAYERS)
        model_local = TreeHyperLISTA(
            **{**model_kwargs, **best}).to(device)
        tl = _trajectory_nmse(model_local, y_one, x_one)
        trajs_local.append(tl)

        # --- CG switch: run layer by layer, switch when support is stable
        model_local.eval()
        with torch.no_grad():
            traj = model_local(y_one.unsqueeze(0), return_trajectory=True)
        # detect stable support: identical support for ``stable_for`` layers
        stable_for = 3
        supports = [(t.abs() > 1e-6).float() for t in traj]
        switch_k = None
        for kk in range(stable_for, len(supports)):
            if all(torch.equal(supports[kk], supports[kk - j])
                   for j in range(1, stable_for + 1)):
                switch_k = kk
                break
        nmse_cg_traj = [nmse_db(t, x_one.unsqueeze(0)) for t in traj]
        if switch_k is not None:
            supp = supports[switch_k][0].bool()
            A_t = torch.from_numpy(ds.A).float().to(device)
            x_cg = cg_solve_on_support(
                A_t, y_one.unsqueeze(0), supp.unsqueeze(0), num_iter=20)
            # replace tail with CG-refined NMSE (constant after switch)
            v_cg = nmse_db(x_cg, x_one.unsqueeze(0))
            for kk in range(switch_k, len(nmse_cg_traj)):
                nmse_cg_traj[kk] = v_cg
        trajs_cg.append(nmse_cg_traj)

    tg = np.array(trajs_global)
    tl = np.array(trajs_local)
    tc = np.array(trajs_cg)

    out = {
        'K_layers': K_LAYERS,
        'n_instances': N_INSTANCES,
        'global': {
            'nmse_mean': tg.mean(0).tolist(),
            'nmse_std': tg.std(0).tolist(),
        },
        'per_instance': {
            'nmse_mean': tl.mean(0).tolist(),
            'nmse_std': tl.std(0).tolist(),
        },
        'cg_switch': {
            'nmse_mean': tc.mean(0).tolist(),
            'nmse_std': tc.std(0).tolist(),
        },
    }
    with open('results/tree_superlinear/results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("[saved] results/tree_superlinear/results.json")

    _plot_superlinear(out)
    return out


def _plot_superlinear(data):
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.4))
    for key, color, label in [('global', '#1f77b4', 'Global hparams'),
                              ('per_instance', '#e377c2', 'Per-instance'),
                              ('cg_switch', '#2ca02c', '+ CG switch')]:
        m_ = data[key]['nmse_mean']
        s_ = data[key]['nmse_std']
        x = list(range(len(m_)))
        ax.plot(x, m_, color=color, linewidth=1.8, label=label,
                marker='o', markersize=4, markevery=2)
        lo = [a - b for a, b in zip(m_, s_)]
        hi = [a + b for a, b in zip(m_, s_)]
        ax.fill_between(x, lo, hi, alpha=0.12, color=color)
    ax.set_xlabel('Layer')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Exp 5: Per-Instance Tuning and CG Switch')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig('paper/tree_figures/tree_fig_superlinear.pdf')
    fig.savefig('paper/tree_figures/tree_fig_superlinear.png', dpi=200)
    plt.close(fig)
    print("[saved] paper/tree_figures/tree_fig_superlinear.{pdf,png}")


if __name__ == '__main__':
    from src.utils.sensing import pick_device; dev = pick_device()
    run_superlinear(device=dev)
