"""Generate all publication-quality figures for the paper."""

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
    'Group-ISTA': '#888888', 'Group-FISTA': '#666666',
    'LISTA': '#1f77b4', 'BlockLISTA': '#2ca02c', 'ALISTA': '#ff7f0e',
    'Ada-BLISTA-T': '#17becf', 'Ada-BLISTA-U': '#bcbd22',
    'HyperLISTA': '#d62728',
    'SH-block_soft': '#9467bd', 'SH-topk_group': '#8c564b',
    'SH-hybrid': '#e377c2',
}
MARKERS = {
    'Group-ISTA': 'v', 'Group-FISTA': '^',
    'LISTA': 's', 'BlockLISTA': 'D', 'ALISTA': 'o',
    'Ada-BLISTA-T': 'h', 'Ada-BLISTA-U': 'H',
    'HyperLISTA': 'p',
    'SH-block_soft': 'P', 'SH-topk_group': 'X', 'SH-hybrid': '*',
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
        ax.plot(iters, traj, color=c, marker=m, markevery=2, markersize=5, label=name, linewidth=1.5)
        lo = [t - s for t, s in zip(traj, std)]
        hi = [t + s for t, s in zip(traj, std)]
        ax.fill_between(iters, lo, hi, alpha=0.12, color=c)
    ax.set_xlabel('Layer / Iteration')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Convergence: NMSE vs Layer')
    ax.legend(ncol=2, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 'fig1_convergence.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig1_convergence.png'))
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
        ax1.plot(iters, prec, color=c, marker=m, markevery=2, markersize=5, label=name, linewidth=1.5)
        ax2.plot(iters, rec, color=c, marker=m, markevery=2, markersize=5, label=name, linewidth=1.5)
    ax1.set_xlabel('Layer / Iteration'); ax1.set_ylabel('Group Precision')
    ax1.set_title('Group Support Precision'); ax1.legend(ncol=2, fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Layer / Iteration'); ax2.set_ylabel('Group Recall')
    ax2.set_title('Group Support Recall'); ax2.legend(ncol=2, fontsize=8); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2_support.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig2_support.png'))
    plt.close(fig)

def fig_mismatch(mm_data, out_dir):
    sweep_titles = {
        'snr_db': ('Test SNR (dB)', 'Noise Robustness'),
        'num_active_groups': ('Test Group Sparsity $s_g$', 'Sparsity Robustness'),
        'group_size': ('Test Block Size $d$', 'Block Size Robustness'),
        'operator_delta': ('Perturbation $\\delta$', 'Operator Perturbation'),
    }
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
        ax.set_xlabel(xlabel); ax.set_ylabel('NMSE (dB)')
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fignum = {'snr_db': 3, 'num_active_groups': 4, 'group_size': 5, 'operator_delta': 6}[sweep_name]
        fig.savefig(os.path.join(out_dir, f'fig{fignum}_{sweep_name}.pdf'))
        fig.savefig(os.path.join(out_dir, f'fig{fignum}_{sweep_name}.png'))
        plt.close(fig)

def fig_ablation(abl_data, out_dir):
    # Sensitivity plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, param in enumerate(['c1', 'c2', 'c3']):
        key = f'sensitivity_{param}'
        if key not in abl_data:
            continue
        vals = abl_data[key]['values']
        means = abl_data[key]['nmse_db_mean']
        stds = abl_data[key].get('nmse_db_std', [0]*len(means))
        axes[i].plot(vals, means, 'o-', color='#e377c2', linewidth=1.5)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        axes[i].fill_between(vals, lo, hi, alpha=0.15, color='#e377c2')
        axes[i].set_xlabel(f'${param}$'); axes[i].set_ylabel('NMSE (dB)')
        axes[i].set_title(f'Sensitivity to ${param}$'); axes[i].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig7_sensitivity.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig7_sensitivity.png'))
    plt.close(fig)

    # Mechanism comparison bar chart
    modes = ['block_soft', 'topk_group', 'hybrid']
    labels = ['Block Soft', 'Top-k Group', 'Hybrid']
    nmses = [abl_data.get(f'mechanism_{m}', {}).get('nmse_db_mean', 0) for m in modes]
    stds = [abl_data.get(f'mechanism_{m}', {}).get('nmse_db_std', 0) for m in modes]
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    bars = ax.bar(labels, nmses, yerr=stds, color=['#9467bd', '#8c564b', '#e377c2'],
                  capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('NMSE (dB)'); ax.set_title('Support Mechanism Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(os.path.join(out_dir, 'fig8_mechanisms.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig8_mechanisms.png'))
    plt.close(fig)

def generate_all_figures(results_dir: str = 'results', out_dir: str = 'paper/figures'):
    os.makedirs(out_dir, exist_ok=True)
    core_path = os.path.join(results_dir, 'core', 'core_results.json')
    mm_path = os.path.join(results_dir, 'mismatch', 'mismatch_results.json')
    abl_path = os.path.join(results_dir, 'ablation', 'ablation_results.json')

    if os.path.exists(core_path):
        data = load_json(core_path)
        fig1_convergence(data, out_dir)
        fig2_support(data, out_dir)
        print("Generated Fig 1-2 (convergence, support)")
    if os.path.exists(mm_path):
        mm_data = load_json(mm_path)
        fig_mismatch(mm_data, out_dir)
        print("Generated Fig 3-6 (mismatch)")
    if os.path.exists(abl_path):
        abl_data = load_json(abl_path)
        fig_ablation(abl_data, out_dir)
        print("Generated Fig 7-8 (ablation)")

if __name__ == '__main__':
    generate_all_figures()
