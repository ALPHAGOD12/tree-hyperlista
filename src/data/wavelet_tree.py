"""
Wavelet tree structure builder for image compressed sensing.

Wavelet coefficients have a natural quad-tree parent-child relationship:
each coarse-scale coefficient has 4 children at the next finer scale
(in 2D DWT with standard decomposition).
"""

import numpy as np
import pywt
from typing import Dict, Any, Tuple


def build_wavelet_tree(patch_size: int = 32, wavelet: str = 'haar',
                       level: int = 2) -> Tuple[Dict[str, Any], Any, Tuple]:
    """Build a tree structure from the 2D DWT subband hierarchy.

    In a 2D multi-level wavelet decomposition, each coefficient at level l
    corresponds to a 2x2 block of coefficients at level l+1.

    Returns:
        tree_info: dict with parent, children, depth, n, etc.
        slices: pywt coefficient slices for reconstruction
        coeff_shape: shape of the flattened coefficient array
    """
    test_patch = np.zeros((patch_size, patch_size))
    coeffs = pywt.wavedec2(test_patch, wavelet, level=level)
    coeff_arr, slices = pywt.coeffs_to_array(coeffs)
    n = coeff_arr.size
    coeff_shape = coeff_arr.shape
    h, w = coeff_shape

    parent = np.full(n, -1, dtype=np.int64)
    depth_arr = np.zeros(n, dtype=np.int64)
    children_list = [[] for _ in range(n)]

    def rc_to_flat(r, c):
        return r * w + c

    coarse_h = patch_size // (2 ** level)
    coarse_w = patch_size // (2 ** level)

    for r in range(coarse_h):
        for c in range(coarse_w):
            flat_idx = rc_to_flat(r, c)
            depth_arr[flat_idx] = 0

    for lev in range(level, 0, -1):
        parent_h = patch_size // (2 ** lev)
        parent_w = patch_size // (2 ** lev)
        child_h = patch_size // (2 ** (lev - 1))
        child_w = patch_size // (2 ** (lev - 1))

        parent_depth = level - lev
        child_depth = parent_depth + 1

        for pr in range(parent_h):
            for pc in range(parent_w):
                p_flat = rc_to_flat(pr, pc)
                depth_arr[p_flat] = parent_depth

                for dr in range(2):
                    for dc in range(2):
                        cr = pr * 2 + dr
                        cc = pc * 2 + dc

                        if cr < child_h and cc < child_w:
                            if cr < parent_h and cc < parent_w:
                                continue
                            c_flat = rc_to_flat(cr, cc)
                            if c_flat < n:
                                parent[c_flat] = p_flat
                                children_list[p_flat].append(c_flat)
                                depth_arr[c_flat] = child_depth

    orphans = []
    for i in range(n):
        if parent[i] == -1 and depth_arr[i] == 0:
            continue
        if parent[i] == -1 and depth_arr[i] > 0:
            orphans.append(i)

    if orphans:
        for o in orphans:
            r = o // w
            c = o % w
            pr, pc = r // 2, c // 2
            p_flat = rc_to_flat(pr, pc)
            if p_flat < n and p_flat != o:
                parent[o] = p_flat
                children_list[p_flat].append(o)

    leaves = [i for i in range(n) if len(children_list[i]) == 0]

    descendants = [set() for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for c in children_list[i]:
            descendants[i].add(c)
            descendants[i].update(descendants[c])

    tree_info = {
        'n': n,
        'parent': parent,
        'children': children_list,
        'depth': depth_arr,
        'descendants': descendants,
        'leaves': leaves,
        'max_depth': level,
        'branching': 4,
    }

    return tree_info, slices, coeff_shape


def build_simple_binary_wavelet_tree(n: int, level: int = 2) -> Dict[str, Any]:
    """Build a simplified binary tree approximation for wavelet coefficients.

    For n coefficients with `level` decomposition levels, partition into
    groups by scale and create parent-child links between adjacent scales.
    This is simpler and works even when the actual wavelet layout is complex.
    """
    parent = np.full(n, -1, dtype=np.int64)
    depth_arr = np.zeros(n, dtype=np.int64)
    children_list = [[] for _ in range(n)]

    coarse_size = n // (4 ** level)
    if coarse_size < 1:
        coarse_size = 1

    boundaries = [0, coarse_size]
    current = coarse_size
    for lev in range(level):
        band_size = current * 3
        boundaries.append(boundaries[-1] + band_size)
        current *= 4
    boundaries[-1] = min(boundaries[-1], n)

    for i in range(boundaries[0], boundaries[1]):
        depth_arr[i] = 0

    for lev_idx in range(1, len(boundaries) - 1):
        parent_start = boundaries[lev_idx - 1]
        parent_end = boundaries[lev_idx]
        child_start = boundaries[lev_idx]
        child_end = boundaries[min(lev_idx + 1, len(boundaries) - 1)]

        n_parents = parent_end - parent_start
        n_children = child_end - child_start

        if n_parents == 0 or n_children == 0:
            continue

        children_per_parent = max(1, n_children // n_parents)

        for ci in range(n_children):
            child_idx = child_start + ci
            if child_idx >= n:
                break
            pi = min(ci // children_per_parent, n_parents - 1)
            parent_idx = parent_start + pi

            parent[child_idx] = parent_idx
            children_list[parent_idx].append(child_idx)
            depth_arr[child_idx] = lev_idx

    leaves = [i for i in range(n) if len(children_list[i]) == 0]

    descendants = [set() for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for c in children_list[i]:
            descendants[i].add(c)
            descendants[i].update(descendants[c])

    return {
        'n': n,
        'parent': parent,
        'children': children_list,
        'depth': depth_arr,
        'descendants': descendants,
        'leaves': leaves,
        'max_depth': max(depth_arr) if len(depth_arr) > 0 else 0,
        'branching': 4,
    }
