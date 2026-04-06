# Tree-HyperLISTA & Struct-HyperLISTA: Ultra-Lightweight Deep Unfolding for Structured Sparse Recovery

This repository contains the code and paper for two complementary works on structured sparse recovery with minimal parameterization:

1. **Struct-HyperLISTA** — Block-sparse recovery with 3 hyperparameters
2. **Tree-HyperLISTA** — Tree-sparse recovery with 3 hyperparameters (extends Struct-HyperLISTA to hierarchical sparsity)

Both methods replace elementwise thresholding in the HyperLISTA backbone with structure-aware proximal operators while retaining only **3 learned hyperparameters** `(c1, c2, c3)`.

## Key Idea

> **Structural inductive bias in the proximal operator is far more powerful than heavy parameterization.**

Instead of learning millions of weights (LISTA: 1.5M params) or even dozens (ALISTA: 32 params), we show that the right proximal operator — one that respects the signal's geometric structure — combined with just 3 adaptive hyperparameters is sufficient for state-of-the-art recovery.

```
Elementwise (HyperLISTA) → Block (Struct-HyperLISTA) → Tree (Tree-HyperLISTA)
     3 params                    3 params                    3 params
```

## Results at a Glance

### Struct-HyperLISTA (Block-Sparse Recovery)

| Method | NMSE (dB) | #Params |
|--------|-----------|---------|
| Group-FISTA | -8.29 | 0 |
| LISTA | -7.78 | 1,500,016 |
| BlockLISTA | -16.34 | 1,500,400 |
| ALISTA | -6.55 | 32 |
| Ada-BLISTA-T | -14.50 | 31,666 |
| HyperLISTA | -4.23 | 3 |
| **SH-hybrid (ours)** | **-19.69** | **3** |

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

Both models share the same backbone — a momentum-accelerated gradient step with an analytic weight matrix `W`:

```
For each layer k = 0, ..., K-1:
  1. Momentum:       z = x + β_k (x - x_prev)
  2. Gradient step:  u = z + W^T (y - A z)
  3. Proximal:       x_new = Π(u; θ_k, p_k)    ← this is what changes
```

The 3 hyperparameters `(c1, c2, c3)` control all per-layer quantities via reparameterized adaptive formulas:
- `c1 → θ_k` (threshold): scales with normalized residual ratio
- `c2 → β_k` (momentum): bounded via sigmoid, scales with active fraction
- `c3 → p_k` (support size): grows with layers via signal estimate + ramp

### What differs: the proximal operator

| Model | Proximal Operator Π |
|-------|-------------------|
| HyperLISTA | Elementwise soft-thresholding |
| Struct-HyperLISTA | Block soft-threshold / Top-k group / Hybrid |
| Tree-HyperLISTA | Subtree scoring → Top-K tree projection with ancestor closure → Soft-threshold |

## Repository Structure

```
src/
  models/
    ista.py                  - Group-ISTA / Group-FISTA (classical baselines)
    lista.py                 - LISTA / BlockLISTA (learned baselines)
    alista.py                - ALISTA (analytic weights, 32 params)
    hyperlista.py            - HyperLISTA (3 params, elementwise)
    struct_hyperlista.py     - Struct-HyperLISTA (3 params, block-aware)
    ada_blocklista.py        - Ada-BlockLISTA (31K/500K params)
    tree_hyperlista.py       - Tree-HyperLISTA (3 params, tree-aware)
    tree_baselines.py        - Tree-ISTA, Tree-FISTA, Tree-LISTA
    tree_classical.py        - Tree-IHT, Tree-CoSaMP (Baraniuk et al.)
  data/
    synthetic.py             - Block-sparse data generation
    tree_synthetic.py        - Tree-sparse data generation
    wavelet_tree.py          - Wavelet coefficient tree builder
    image_cs.py              - Image CS pipeline (patches → DWT → sensing → recovery)
  utils/
    proximal.py              - Block proximal operators
    tree_proximal.py         - Tree proximal operators (scoring, projection, thresholding)
    metrics.py               - NMSE, support precision/recall
    sensing.py               - Sensing matrix generation
  train.py                   - Training pipeline (backprop + Bayesian optimization)
  evaluate.py                - Evaluation pipeline

experiments/
  run_core.py                - Core block-sparse experiments
  run_mismatch.py            - Mismatch robustness experiments
  run_ablation.py            - Ablation studies
  run_image_cs.py            - Wavelet-domain image CS (block models)
  run_tree_experiments.py    - Core tree-sparse experiments
  run_tree_image_cs.py       - Wavelet-domain image CS (tree models)
  run_tree_scaled.py         - Scaled tree experiments
  plot_results.py            - Generate figures for block-sparse paper
  plot_tree_results.py       - Generate figures for tree-sparse paper

paper/
  main.tex                   - Struct-HyperLISTA paper (NeurIPS format)
  tree_main.tex              - Tree-HyperLISTA paper
  proposal.tex               - Project proposal
  references.bib             - Bibliography
  figures/                   - Block-sparse paper figures
  tree_figures/              - Tree-sparse paper figures

results/                     - Experiment output (JSON)
data/Set11/                  - Test images for image CS
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run All Block-Sparse Experiments

```bash
python experiments/run_core.py           # Core results (Table 1)
python experiments/run_mismatch.py       # Mismatch robustness
python experiments/run_ablation.py       # Ablation studies
python experiments/run_image_cs.py       # Image CS experiments
python experiments/plot_results.py       # Generate paper figures
```

### Run All Tree-Sparse Experiments

```bash
python experiments/run_tree_experiments.py   # Core + mismatch + ablation
python experiments/run_tree_image_cs.py      # Wavelet-domain image CS
python experiments/plot_tree_results.py      # Generate paper figures
```

### Quick Run (Fast, Reduced Scale)

```bash
python experiments/run_all_fast.py       # All block experiments (reduced trials)
```

### Compile Papers

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd paper && pdflatex tree_main.tex && bibtex tree_main && pdflatex tree_main.tex && pdflatex tree_main.tex
```

## Technical Details

### Tree-Aware Scoring (Bottom-Up Aggregation)

The key innovation in Tree-HyperLISTA is the subtree-aware scoring:

```
s_i = |u_i| + ρ * Σ_{children j} s_j    (computed bottom-up in O(n))
```

where `ρ ∈ (0,1)` is a fixed decay weight. This propagates information about "how active is this subtree" from leaves to root.

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
@article{verma2025structhyperlista,
  title={How Little Learning Is Needed for Structured Sparse Recovery?},
  author={Verma, Shresth and Agrawal, Aditya},
  year={2025}
}

@article{verma2025treehyperlista,
  title={Tree-HyperLISTA: Ultra-Lightweight Deep Unfolding for Tree-Sparse Recovery},
  author={Verma, Shresth and Agrawal, Aditya},
  year={2025}
}
```

## Authors

**Shresth Verma** and **Aditya Agrawal**
Department of Computer Science and Engineering, IIT Bombay
