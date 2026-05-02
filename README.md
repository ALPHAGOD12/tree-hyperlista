# Tree-HyperLISTA: Ultra-Lightweight Deep Unfolding for Tree-Sparse Recovery

This repository extends **HyperLISTA** (NeurIPS 2021) to **tree-structured sparsity** with the same minimal parameterization: only **3 learned hyperparameters** `(c1, c2, c3)`.

## Key Idea

> **Structural inductive bias in the proximal operator is far more powerful than heavy parameterization.**

By replacing elementwise soft-thresholding in the HyperLISTA backbone with a tree-aware proximal operator, we achieve state-of-the-art recovery while retaining only 3 adaptive hyperparameters.

```
Elementwise (HyperLISTA) → Tree (Tree-HyperLISTA)
     3 params                    3 params
```

## Results at a Glance

### Tree-HyperLISTA (Tree-Sparse Recovery)

| Method | NMSE (dB) | Recall | #Params |
|--------|-----------|--------|---------|
| Tree-FISTA | -8.31 | 0.820 | 0 |
| LISTA | -15.39 | 0.999 | 1,558,576 |
| HyperLISTA | -14.09 | 0.962 | 3 |
| Tree-CoSaMP | classical | - | 0 |
| **TH-hybrid (ours)** | **-29.40** | **1.000** | **3** |

Tree-HyperLISTA outperforms HyperLISTA by **15.3 dB** and LISTA by **14.0 dB** with **500,000x fewer parameters**.

## Architecture

Momentum-accelerated gradient step with an analytic weight matrix `W`:

```
For each layer k = 0, ..., K-1:
  1. Momentum:       z = x + β_k (x - x_prev)
  2. Gradient step:  u = z + W^T (y - A z)
  3. Proximal:       x_new = Π(u; θ_k, p_k)    ← tree-aware operator
```

The 3 hyperparameters `(c1, c2, c3)` control all per-layer quantities:
- `c1 → θ_k` (threshold): scales with normalized residual ratio
- `c2 → β_k` (momentum): bounded via sigmoid, scales with active fraction
- `c3 → p_k` (support size): grows with layers via signal estimate + ramp

### Tree-Aware Proximal Operator

| Model | Proximal Operator Π |
|-------|-------------------|
| HyperLISTA | Elementwise soft-thresholding |
| Tree-HyperLISTA | Subtree scoring → Top-K tree projection with ancestor closure → Soft-threshold |

## Repository Structure

```
src/
  models/
    ista.py                  - ElementwiseISTA (baseline)
    lista.py                 - LISTA (learned baseline)
    alista.py                - ALISTA (analytic weights, 32 params)
    hyperlista.py            - HyperLISTA (3 params, elementwise)
    tree_hyperlista.py       - Tree-HyperLISTA (3 params, tree-aware)
    tree_hyperlista_ss.py    - Self-supervised / amortized variants
    diff_tree_hyperlista.py  - Differentiable tree proximal variant
    tree_baselines.py        - Tree-ISTA, Tree-FISTA, Tree-LISTA
    tree_classical.py        - Tree-IHT, Tree-CoSaMP (Baraniuk et al.)
  data/
    tree_synthetic.py        - Tree-sparse data generation
    wavelet_tree.py          - Wavelet coefficient tree builder
    image_cs.py              - Image CS pipeline (patches → DWT → sensing → recovery)
  utils/
    proximal.py              - Elementwise proximal operators
    tree_proximal.py         - Tree proximal operators (scoring, projection, thresholding)
    diff_tree_proximal.py    - Differentiable tree proximal operators
    metrics.py               - NMSE, support precision/recall
    sensing.py               - Sensing matrix generation
  train.py                   - Training pipeline (backprop + Bayesian optimization)

experiments/
  run_tree_experiments.py    - Core tree-sparse experiments (+ mismatch + ablation)
  run_tree_image_cs.py       - Wavelet-domain image CS (tree models)
  run_tree_scaled.py         - Scaled tree experiments (n=1023, n=4095)
  run_diff_tree_cs.py        - Differentiable tree variant image CS
  run_ss_image_cs.py         - Self-supervised variant image CS
  plot_tree_results.py       - Generate figures for paper

paper/
  tree_main.tex              - Tree-HyperLISTA paper
  proposal.tex               - Project proposal (original HyperLISTA extension idea)
  references.bib             - Bibliography
  tree_figures/              - Paper figures

results/                     - Experiment output (JSON)
data/Set11/                  - Test images for image CS
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run All Tree-Sparse Experiments

```bash
python experiments/run_tree_experiments.py   # Core + mismatch + ablation
python experiments/run_tree_image_cs.py      # Wavelet-domain image CS
python experiments/run_tree_scaled.py        # Scalability study
python experiments/plot_tree_results.py      # Generate paper figures
```

### Compile Paper

```bash
cd paper && pdflatex tree_main.tex && bibtex tree_main && pdflatex tree_main.tex && pdflatex tree_main.tex
```

## Technical Details

### Tree-Aware Scoring (Bottom-Up Aggregation)

The key innovation is subtree-aware scoring:

```
s_i = |u_i| + ρ * Σ_{children j} s_j    (computed bottom-up in O(n))
```

where `ρ ∈ (0,1)` is a fixed decay weight. This propagates "how active is this subtree" from leaves to root.

### Tree-Consistent Support Selection

Greedy top-K with ancestor closure: sort nodes by score, greedily add the highest-scored node **plus all its ancestors** until budget K is reached. This guarantees tree consistency (parent always included if child is).

### Convergence Guarantee

Under tree-RIP with constant `δ_{3K} < 1/3`, Tree-HyperLISTA converges linearly:

```
‖x^(k+1) - x*‖₂ ≤ α ‖x^(k) - x*‖₂ + β ‖ε‖₂
```

where `α = 3δ_{3K}/(1 - δ_{3K}) < 1`.

## Dependencies

- Python >= 3.8
- PyTorch >= 2.0
- NumPy >= 1.24
- SciPy >= 1.10
- Matplotlib >= 3.7
- Optuna >= 3.3
- scikit-image >= 0.20
- PyWavelets >= 1.4

## Citation

If you use this code, please cite:

```bibtex
@article{verma2025treehyperlista,
  title={Tree-HyperLISTA: Ultra-Lightweight Deep Unfolding for Tree-Sparse Recovery},
  author={Verma, Shresth and Agrawal, Aditya},
  year={2025}
}
```

## Authors

**Shresth Verma** and **Aditya Agrawal**  
Department of Computer Science and Engineering, IIT Bombay
