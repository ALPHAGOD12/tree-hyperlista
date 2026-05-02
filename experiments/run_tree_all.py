"""Top-level runner for the full Tree-HyperLISTA experiment suite.

Executes Exp 1 through Exp 12 in the order suggested in the plan. Each
call is wrapped in a try/except so a late failure does not lose earlier
results. Set ``--fast`` to use the smoke-test knobs (shrunken tree,
fewer seeds and trials) for a quick end-to-end check.
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch


def _patch_fast(module):
    """Shrink a heavy experiment module to smoke-test sizes."""
    for k, v in [
        ('TREE_DEPTH', 5), ('N_TRAIN', 400), ('N_VAL', 100),
        ('N_TEST', 100), ('N_SEEDS', 1), ('N_TRIALS', 8),
        ('N_TRIALS_BO', 8), ('TRAIN_EPOCHS', 10),
        ('K_LAYERS', 6), ('K_TRAIN', 6), ('TARGET_K', 8),
        ('K_TUNED', 8), ('RATIOS', [0.5, 1.0, 1.5]),
        ('SNR_GRID', [10.0, 20.0, 30.0]),
        ('K_EVALS', [6, 10, 16]), ('N_INSTANCES', 5),
        ('N_TRIALS_GLOBAL', 8), ('N_TRIALS_LOCAL', 4),
        ('RHO_GRID', [0.1, 0.5, 0.9]),
    ]:
        if hasattr(module, k):
            setattr(module, k, v)


def _run(step_name, fn, fast, module=None):
    print(f"\n{'=' * 70}\n  {step_name}\n{'=' * 70}")
    if fast and module is not None:
        _patch_fast(module)
    t0 = time.time()
    try:
        fn()
        print(f"  [done] {step_name}  ({time.time() - t0:.1f}s)")
    except Exception as e:
        traceback.print_exc()
        print(f"  [FAILED] {step_name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true',
                        help='Shrink all experiments for smoke testing.')
    parser.add_argument('--skip-optional', action='store_true',
                        help='Skip BSD500 dictionary CS.')
    parser.add_argument('--skip-exp1', action='store_true',
                        help='Skip Exp 1 (backbone ablation) — use when it already ran.')
    parser.add_argument('--skip-exp4', action='store_true',
                        help='Skip Exp 4 (layer extrapolation) — use when it already ran.')
    parser.add_argument('--skip-exp3', action='store_true',
                        help='Skip Exp 3 (extended mismatch) — use when it already ran.')
    parser.add_argument('--skip-exp7', action='store_true',
                        help='Skip Exp 7 (cross-structure) — use when it already ran.')
    parser.add_argument('--skip-exp8', action='store_true',
                        help='Skip Exp 8 (sparsity mismatch) — use when it already ran.')
    parser.add_argument('--skip-exp5', action='store_true',
                        help='Skip Exp 5 (superlinear/CG) — use when it already ran.')
    parser.add_argument('--skip-exp2', action='store_true',
                        help='Skip Exp 2 (tuning methods) — use when it already ran.')
    parser.add_argument('--skip-exp12', action='store_true',
                        help='Skip Exp 12 (low-SNR) — use when it already ran.')
    args = parser.parse_args()
    from src.utils.sensing import pick_device
    device = pick_device()
    print(f"Device: {device} | fast={args.fast}")

    from experiments import (
        run_tree_backbone_ablation as exp1,
        run_tree_layer_extrapolation as exp4,
        run_tree_mismatch_extended as exp3,
        run_tree_cross_structure as exp7,
        run_tree_sparsity_mismatch as exp8,
        run_tree_superlinear as exp5,
        run_tree_tuning_methods as exp2,
        run_tree_lowsnr as exp12,
        run_tree_polish as pol,
        run_tree_bsd500_dict_cs as exp6,
    )

    if not args.skip_exp1:
        _run('Exp 1 -- Backbone ablation',
             lambda: exp1.run_backbone_ablation(device=device),
             args.fast, exp1)
    else:
        print("\n[skipped] Exp 1 -- Backbone ablation  (--skip-exp1 flag set)")
    if not args.skip_exp4:
        _run('Exp 4 -- Layer extrapolation',
             lambda: exp4.run_extrapolation(device=device),
             args.fast, exp4)
    else:
        print("\n[skipped] Exp 4 -- Layer extrapolation  (--skip-exp4 flag set)")

    if not args.skip_exp3:
        _run('Exp 3 -- Extended mismatch',
             lambda: exp3.run_extended_mismatch(device=device),
             args.fast, exp3)
    else:
        print("\n[skipped] Exp 3 -- Extended mismatch  (--skip-exp3 flag set)")
    if not args.skip_exp7:
        _run('Exp 7 -- Cross-structure stress test',
             lambda: exp7.run_cross(device=device),
             args.fast, exp7)
    else:
        print("\n[skipped] Exp 7 -- Cross-structure stress test  (--skip-exp7 flag set)")

    if not args.skip_exp8:
        _run('Exp 8 -- Sparsity budget mismatch',
             lambda: exp8.run_sparsity(device=device),
             args.fast, exp8)
    else:
        print("\n[skipped] Exp 8 -- Sparsity budget mismatch  (--skip-exp8 flag set)")

    if not args.skip_exp5:
        _run('Exp 5 -- Superlinear / CG switch',
             lambda: exp5.run_superlinear(device=device),
             args.fast, exp5)
    else:
        print("\n[skipped] Exp 5 -- Superlinear / CG switch  (--skip-exp5 flag set)")

    if not args.skip_exp2:
        _run('Exp 2 -- Tuning methods (BP/GS/BO)',
             lambda: exp2.run_tuning_methods(device=device),
             args.fast, exp2)
    else:
        print("\n[skipped] Exp 2 -- Tuning methods (BP/GS/BO)  (--skip-exp2 flag set)")

    if not args.skip_exp12:
        _run('Exp 12 -- Low-SNR sweep',
             lambda: exp12.run_lowsnr(device=device),
             args.fast, exp12)
    else:
        print("\n[skipped] Exp 12 -- Low-SNR sweep  (--skip-exp12 flag set)")

    _run('Polish Exp 9/10/11',
         lambda: (pol.dense_rho_sweep(device=device),
                  pol.mechanism_comparison(device=device),
                  pol.tuning_time_vs_n(device=device)),
         args.fast, pol)
    if not args.skip_optional:
        _run('Exp 6 (optional) -- BSD500 dictionary CS',
             lambda: exp6.run_bsd500_dict_cs(device=device),
             args.fast, exp6)

    from experiments import plot_tree_results
    plot_tree_results.generate_all_figures()
    print("\nAll done.")


if __name__ == '__main__':
    main()
