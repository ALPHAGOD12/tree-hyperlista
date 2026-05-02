"""Generate all publication-quality figures for tree-sparse recovery paper."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

COLORS = {
    'Tree-ISTA': '#888888', 'Tree-FISTA': '#666666',
    'LISTA': '#1f77b4', 'Tree-LISTA': '#2ca02c',
    'HyperLISTA': '#d62728',
    'Tree-IHT': '#ff7f0e', 'Tree-CoSaMP': '#17becf',
    'TH-tree_hard': '#9467bd', 'TH-tree_threshold': '#8c564b',
    'TH-hybrid_tree': '#e377c2',
}
MARKERS = {
    'Tree-ISTA': 'v', 'Tree-FISTA': '^',
    'LISTA': 's', 'Tree-LISTA': 'D',
    'HyperLISTA': 'p',
    'Tree-IHT': 'H', 'Tree-CoSaMP': 'h',
    'TH-tree_hard': 'P', 'TH-tree_threshold': 'X', 'TH-hybrid_tree': '*',
}

def load_json(path):
    with open(path) as f:
        return json.load(f)

def fig1_convergence(data, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    for name in data:
        traj = data[name]['trajectory_mean']
        std = data[name].get('trajectory_std', [0]*len(traj))
        iters = list(range(len(traj)))
        c = COLORS.get(name, '#333333')
        m = MARKERS.get(name, 'o')
        ax.plot(iters, traj, color=c, marker=m, markevery=2, markersize=5,
                label=name, linewidth=1.5)
        lo = [t - s for t, s in zip(traj, std)]
        hi = [t + s for t, s in zip(traj, std)]
        ax.fill_between(iters, lo, hi, alpha=0.12, color=c)
    ax.set_xlabel('Layer / Iteration')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Convergence: NMSE vs Layer (Tree-Sparse)')
    ax.legend(ncol=2, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 'tree_fig1_convergence.pdf'))
    fig.savefig(os.path.join(out_dir, 'tree_fig1_convergence.png'))
    plt.close(fig)

def fig2_support(data, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in data:
        if 'precision_trajectory' not in data[name]:
            continue
        prec = data[name]['precision_trajectory']
        rec = data[name]['recall_trajectory']
        iters = list(range(len(prec)))
        c = COLORS.get(name, '#333333')
        m = MARKERS.get(name, 'o')
        ax1.plot(iters, prec, color=c, marker=m, markevery=2, markersize=5,
                 label=name, linewidth=1.5)
        ax2.plot(iters, rec, color=c, marker=m, markevery=2, markersize=5,
                 label=name, linewidth=1.5)
    ax1.set_xlabel('Layer / Iteration')
    ax1.set_ylabel('Node Precision')
    ax1.set_title('Support Precision (Tree)')
    ax1.legend(ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Layer / Iteration')
    ax2.set_ylabel('Node Recall')
    ax2.set_title('Support Recall (Tree)')
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tree_fig2_support.pdf'))
    fig.savefig(os.path.join(out_dir, 'tree_fig2_support.png'))
    plt.close(fig)

def fig_mismatch(mm_data, out_dir):
    sweep_titles = {
        'snr_db': ('Test SNR (dB)', 'Noise Robustness'),
        'target_sparsity': ('Tree Sparsity $K$', 'Sparsity Robustness'),
        'operator_delta': ('Perturbation $\\delta$', 'Operator Perturbation'),
    }
    fignum_map = {'snr_db': 3, 'target_sparsity': 4, 'operator_delta': 5}
    for sweep_name, (xlabel, title) in sweep_titles.items():
        if sweep_name not in mm_data:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        vals = mm_data[sweep_name]['values']
        for model_name in mm_data[sweep_name]:
            if model_name == 'values':
                continue
            means = mm_data[sweep_name][model_name]['nmse_db_mean']
            stds = mm_data[sweep_name][model_name].get('nmse_db_std', [0]*len(means))
            c = COLORS.get(model_name, '#333333')
            mk = MARKERS.get(model_name, 'o')
            ax.errorbar(vals, means, yerr=stds, color=c, marker=mk, markersize=5,
                        label=model_name, linewidth=1.5, capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('NMSE (dB)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fignum = fignum_map[sweep_name]
        fig.savefig(os.path.join(out_dir, f'tree_fig{fignum}_{sweep_name}.pdf'))
        fig.savefig(os.path.join(out_dir, f'tree_fig{fignum}_{sweep_name}.png'))
        plt.close(fig)

def fig_ablation(abl_data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, param in enumerate(['c1', 'c2', 'c3']):
        key = f'sensitivity_{param}'
        if key not in abl_data:
            continue
        vals = abl_data[key]['values']
        means = abl_data[key]['nmse_db_mean']
        stds = abl_data[key].get('nmse_db_std', [0]*len(means))
        axes[i].plot(vals, means, 'o-', color='#e377c2', linewidth=1.5)
        lo = [m_ - s for m_, s in zip(means, stds)]
        hi = [m_ + s for m_, s in zip(means, stds)]
        axes[i].fill_between(vals, lo, hi, alpha=0.15, color='#e377c2')
        axes[i].set_xlabel(f'${param}$')
        axes[i].set_ylabel('NMSE (dB)')
        axes[i].set_title(f'Sensitivity to ${param}$')
        axes[i].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tree_fig6_sensitivity.pdf'))
    fig.savefig(os.path.join(out_dir, 'tree_fig6_sensitivity.png'))
    plt.close(fig)

    modes = ['tree_hard', 'tree_threshold', 'hybrid_tree']
    labels = ['Hard Tree', 'Thresh+Closure', 'Hybrid Tree']
    nmses = [abl_data.get(f'mechanism_{m}', {}).get('nmse_db_mean', 0) for m in modes]
    stds = [abl_data.get(f'mechanism_{m}', {}).get('nmse_db_std', 0) for m in modes]
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.bar(labels, nmses, yerr=stds, color=['#9467bd', '#8c564b', '#e377c2'],
           capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Tree Support Mechanism Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(os.path.join(out_dir, 'tree_fig7_mechanisms.pdf'))
    fig.savefig(os.path.join(out_dir, 'tree_fig7_mechanisms.png'))
    plt.close(fig)

    rho_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    rho_nmses = [abl_data.get(f'rho={r}', {}).get('nmse_db_mean', 0) for r in rho_vals]
    rho_stds = [abl_data.get(f'rho={r}', {}).get('nmse_db_std', 0) for r in rho_vals]
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.errorbar(rho_vals, rho_nmses, yerr=rho_stds, marker='o', color='#17becf',
                linewidth=1.5, capsize=3)
    ax.set_xlabel('$\\rho$ (decay weight)')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Effect of Descendant Decay $\\rho$')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 'tree_fig8_rho.pdf'))
    fig.savefig(os.path.join(out_dir, 'tree_fig8_rho.png'))
    plt.close(fig)


def fig9_image_cs(ics_data, out_dir):
    """Bar chart of PSNR across CS ratios for tree image CS."""
    ratios = []
    for key in sorted(ics_data.keys()):
        ratios.append(key)

    if not ratios:
        return

    fig, axes = plt.subplots(1, len(ratios), figsize=(5 * len(ratios), 5))
    if len(ratios) == 1:
        axes = [axes]

    ics_colors = {
        'Tree-FISTA': '#666666', 'Tree-IHT': '#ff7f0e',
        'Tree-CoSaMP': '#17becf', 'ALISTA': '#1f77b4',
        'HyperLISTA': '#d62728',
        'TH-hard': '#9467bd', 'TH-hybrid': '#e377c2',
    }

    for idx, ratio_key in enumerate(ratios):
        ax = axes[idx]
        ratio_data = ics_data[ratio_key]
        names = list(ratio_data.keys())
        psnrs = [ratio_data[n]['psnr_mean'] for n in names]
        stds = [ratio_data[n].get('psnr_std', 0) for n in names]
        colors = [ics_colors.get(n, '#333333') for n in names]

        bars = ax.bar(range(len(names)), psnrs, yerr=stds, color=colors,
                      capsize=3, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('PSNR (dB)')
        ratio_val = ratio_key.replace('ratio_', '')
        ax.set_title(f'CS Ratio = {ratio_val}')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Wavelet-Domain Image CS: Tree vs Elementwise',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tree_fig9_image_cs.pdf'))
    fig.savefig(os.path.join(out_dir, 'tree_fig9_image_cs.png'))
    plt.close(fig)


def generate_all_figures(results_dir='results', out_dir='paper/tree_figures'):
    os.makedirs(out_dir, exist_ok=True)
    core_path = os.path.join(results_dir, 'tree_core', 'core_results.json')
    mm_path = os.path.join(results_dir, 'tree_mismatch', 'mismatch_results.json')
    abl_path = os.path.join(results_dir, 'tree_ablation', 'ablation_results.json')
    ics_path = os.path.join(results_dir, 'tree_image_cs', 'tree_image_cs_results.json')

    if os.path.exists(core_path):
        data = load_json(core_path)
        fig1_convergence(data, out_dir)
        fig2_support(data, out_dir)
        print("Generated Fig 1-2 (convergence, support)")
    if os.path.exists(mm_path):
        mm_data = load_json(mm_path)
        fig_mismatch(mm_data, out_dir)
        print("Generated Fig 3-5 (mismatch)")
    if os.path.exists(abl_path):
        abl_data = load_json(abl_path)
        fig_ablation(abl_data, out_dir)
        print("Generated Fig 6-8 (ablation)")
    if os.path.exists(ics_path):
        ics_data = load_json(ics_path)
        fig9_image_cs(ics_data, out_dir)
        print("Generated Fig 9 (image CS)")

    # ------------------------------------------------------------------
    # New figures from the Tree-HyperLISTA experiment suite.
    # Each experiment runner is responsible for writing its own
    # ``paper/tree_figures/tree_fig_*.{pdf,png}``; we re-use them via
    # a consolidated copy step so they appear alongside Fig 1-9.
    suite_figs = [
        'tree_fig_backbone',          # Exp 1
        'tree_fig_tuning',            # Exp 2
        'tree_fig_mm_magnitude',      # Exp 3
        'tree_fig_mm_sensing',
        'tree_fig_mm_topology',
        'tree_fig_mm_consistency',
        'tree_fig_extrapolation',     # Exp 4
        'tree_fig_superlinear',       # Exp 5
        'tree_fig_bsd500_dict',       # Exp 6 (optional)
        'tree_fig_cross',             # Exp 7
        'tree_fig_sparsity',          # Exp 8
        'tree_fig_mechanisms_ext',    # Exp 9 (extended)
        'tree_fig_rho_dense',         # Exp 10 (dense)
        'tree_fig_tune_time',         # Exp 11 (tuning time)
        'tree_fig_lowsnr',            # Exp 12
    ]
    missing = [f for f in suite_figs
               if not os.path.exists(os.path.join(out_dir, f + '.pdf'))]
    if missing:
        print(f"[note] {len(missing)} suite figures not yet generated "
              f"(run the corresponding experiment):\n  " + "\n  ".join(missing))
    else:
        print("All Tree-HyperLISTA suite figures are present.")


if __name__ == '__main__':
    generate_all_figures()
