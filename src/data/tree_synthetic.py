"""Tree-sparse synthetic data generation with mismatch regimes."""

import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple

from src.utils.sensing import get_sensing_matrix, perturbed_sensing


def build_balanced_tree(depth: int, branching: int = 2) -> Dict[str, Any]:
    """Build a balanced D-ary tree with given depth and branching factor.

    Returns a dict with:
        n: total number of nodes
        parent: parent[i] = parent of node i (-1 for root)
        children: children[i] = list of children of node i
        depth_arr: depth_arr[i] = depth of node i (root=0)
        descendants: descendants[i] = set of all descendants of node i
        leaves: list of leaf node indices
    """
    n = (branching ** (depth + 1) - 1) // (branching - 1) if branching > 1 else depth + 1
    parent = np.full(n, -1, dtype=np.int64)
    children = [[] for _ in range(n)]
    depth_arr = np.zeros(n, dtype=np.int64)

    next_id = 1
    queue = [0]
    while queue:
        node = queue.pop(0)
        if depth_arr[node] < depth:
            for _ in range(branching):
                if next_id >= n:
                    break
                child = next_id
                next_id += 1
                parent[child] = node
                children[node].append(child)
                depth_arr[child] = depth_arr[node] + 1
                queue.append(child)

    descendants = [set() for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for c in children[i]:
            descendants[i].add(c)
            descendants[i].update(descendants[c])

    leaves = [i for i in range(n) if len(children[i]) == 0]

    return {
        'n': n,
        'parent': parent,
        'children': children,
        'depth': depth_arr,
        'descendants': descendants,
        'leaves': leaves,
        'max_depth': depth,
        'branching': branching,
    }


def get_ancestors(node: int, parent: np.ndarray) -> List[int]:
    """Return list of all ancestors of node (including node itself)."""
    ancestors = [node]
    current = node
    while parent[current] != -1:
        current = parent[current]
        ancestors.append(current)
    return ancestors


def generate_tree_support(tree_info: Dict, target_size: int,
                          rng: np.random.RandomState) -> np.ndarray:
    """Generate a tree-consistent support of approximately target_size.

    Strategy: pick random leaves, include their ancestor paths, until
    the support reaches the target size. This guarantees tree consistency
    (if a child is active, its parent is active).
    """
    n = tree_info['n']
    parent = tree_info['parent']
    leaves = tree_info['leaves']
    support = set()

    shuffled_leaves = rng.permutation(leaves)
    for leaf in shuffled_leaves:
        if len(support) >= target_size:
            break
        path = get_ancestors(leaf, parent)
        support.update(path)

    support_arr = np.zeros(n, dtype=bool)
    support_list = sorted(support)[:target_size]
    for idx in support_list:
        support_arr[idx] = True

    if not support_arr[0]:
        support_arr[0] = True

    for i in range(n):
        if support_arr[i] and parent[i] != -1:
            support_arr[parent[i]] = True

    return support_arr


class TreeSparseDataset:
    """Generate tree-sparse recovery problems y = Ax + epsilon."""

    def __init__(self, tree_depth: int = 7, branching: int = 2,
                 m_ratio: float = 0.5, target_sparsity: int = 30,
                 snr_db: float = 30.0, matrix_type: str = 'gaussian',
                 amplitude_dist: str = 'gaussian',
                 amplitude_range: tuple = (0.5, 2.0),
                 seed: int = 42, matrix_seed: int = 42, **matrix_kwargs):
        self.tree_info = build_balanced_tree(tree_depth, branching)
        self.n = self.tree_info['n']
        self.m = int(self.n * m_ratio)
        self.target_sparsity = target_sparsity
        self.snr_db = snr_db
        self.amplitude_dist = amplitude_dist
        self.amplitude_range = amplitude_range
        self.seed = seed
        self.matrix_type = matrix_type
        self.tree_depth = tree_depth
        self.branching = branching

        self.A = get_sensing_matrix(self.m, self.n, matrix_type=matrix_type,
                                    seed=matrix_seed, **matrix_kwargs)
        self.A_tensor = torch.from_numpy(self.A).float()
        self.A_pinv = np.linalg.pinv(self.A).astype(np.float32)
        self.A_pinv_tensor = torch.from_numpy(self.A_pinv).float()

    def generate(self, num_samples: int, seed: Optional[int] = None,
                 target_sparsity: Optional[int] = None,
                 snr_db: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Generate a batch of tree-sparse problems."""
        rng = np.random.RandomState(seed if seed is not None else self.seed)
        K = target_sparsity if target_sparsity is not None else self.target_sparsity
        snr = snr_db if snr_db is not None else self.snr_db

        X = np.zeros((num_samples, self.n), dtype=np.float32)
        supports = []

        for i in range(num_samples):
            support = generate_tree_support(self.tree_info, K, rng)
            supports.append(support)
            active_indices = np.where(support)[0]

            if self.amplitude_dist == 'gaussian':
                vals = rng.randn(len(active_indices)).astype(np.float32)
                signs = np.sign(vals)
                mags = np.abs(vals)
                lo, hi = self.amplitude_range
                mags = mags * (hi - lo) + lo
                X[i, active_indices] = signs * mags
            elif self.amplitude_dist == 'uniform':
                lo, hi = self.amplitude_range
                X[i, active_indices] = (rng.uniform(lo, hi, size=len(active_indices)) *
                                        rng.choice([-1, 1], size=len(active_indices)))
            else:
                X[i, active_indices] = rng.randn(len(active_indices)).astype(np.float32)

        Y_clean = X @ self.A.T
        sigma = np.sqrt(np.mean(Y_clean ** 2) * 10 ** (-snr / 10.0))
        noise = rng.randn(num_samples, self.m).astype(np.float32) * sigma
        Y = Y_clean + noise

        return {
            'x': torch.from_numpy(X),
            'y': torch.from_numpy(Y.astype(np.float32)),
            'noise_std': sigma,
        }

    def generate_with_perturbed_A(self, num_samples: int, delta: float,
                                  seed: int = 99) -> Dict[str, torch.Tensor]:
        """Generate data using a perturbed sensing matrix (operator mismatch)."""
        A_pert = perturbed_sensing(self.A, delta, seed=seed)
        rng = np.random.RandomState(seed)
        K = self.target_sparsity

        X = np.zeros((num_samples, self.n), dtype=np.float32)
        for i in range(num_samples):
            support = generate_tree_support(self.tree_info, K, rng)
            active_indices = np.where(support)[0]
            vals = rng.randn(len(active_indices)).astype(np.float32)
            signs = np.sign(vals)
            mags = np.abs(vals) * 1.5 + 0.5
            X[i, active_indices] = signs * mags

        Y_clean = X @ A_pert.T
        sigma = np.sqrt(np.mean(Y_clean ** 2) * 10 ** (-self.snr_db / 10.0))
        noise = rng.randn(num_samples, self.m).astype(np.float32) * sigma
        Y = Y_clean + noise

        return {
            'x': torch.from_numpy(X),
            'y': torch.from_numpy(Y.astype(np.float32)),
            'noise_std': sigma,
            'A_perturbed': torch.from_numpy(A_pert.astype(np.float32)),
        }


def get_tree_default_config() -> Dict[str, Any]:
    """Default experimental configuration for tree-sparse recovery."""
    return {
        'tree_depth': 7,
        'branching': 2,
        'm_ratio': 0.5,
        'target_sparsity': 30,
        'snr_db': 30.0,
        'matrix_type': 'gaussian',
        'amplitude_dist': 'gaussian',
        'amplitude_range': (0.5, 2.0),
        'num_train': 10000,
        'num_val': 2000,
        'num_test': 5000,
        'num_layers': 16,
        'seed': 42,
    }


def get_tree_mismatch_configs() -> Dict[str, list]:
    """Mismatch sweep configurations for tree-sparse recovery."""
    return {
        'snr_db': [15.0, 20.0, 25.0, 35.0, 40.0],
        'target_sparsity': [15, 20, 30, 40, 50],
        'operator_delta': [0.05, 0.1, 0.15, 0.2, 0.3],
    }
