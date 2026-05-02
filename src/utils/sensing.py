"""Sensing matrix generation utilities."""

import numpy as np
import torch
from scipy.linalg import toeplitz, dft


def normalize_columns(A: np.ndarray) -> np.ndarray:
    """Normalize each column of A to unit l2 norm."""
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return A / norms


def gaussian_sensing(m: int, n: int, seed: int = 42) -> np.ndarray:
    """IID Gaussian sensing matrix with unit-norm columns."""
    rng = np.random.RandomState(seed)
    A = rng.randn(m, n) / np.sqrt(m)
    return normalize_columns(A)


def correlated_sensing(m: int, n: int, rho: float = 0.5,
                       seed: int = 42) -> np.ndarray:
    """Correlated (Toeplitz) sensing matrix."""
    rng = np.random.RandomState(seed)
    col = rho ** np.arange(n)
    C_sqrt = np.linalg.cholesky(toeplitz(col[:min(m, n)])).T
    G = rng.randn(m, min(m, n))
    A_raw = G @ C_sqrt
    if n > m:
        extra = rng.randn(m, n - m)
        A_raw = np.hstack([A_raw, extra])
    return normalize_columns(A_raw[:, :n])


def partial_orthogonal_sensing(m: int, n: int, seed: int = 42) -> np.ndarray:
    """Partial orthogonal sensing (random rows of DCT-like matrix)."""
    rng = np.random.RandomState(seed)
    F = dft(n, scale='sqrtn').real
    rows = rng.choice(n, size=m, replace=False)
    A = F[rows, :]
    return normalize_columns(A)


def ill_conditioned_sensing(m: int, n: int, condition_number: float = 100.0,
                            seed: int = 42) -> np.ndarray:
    """Ill-conditioned sensing matrix with specified condition number."""
    rng = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rng.randn(m, m))
    V, _ = np.linalg.qr(rng.randn(n, n))
    singular_vals = np.linspace(1.0, 1.0 / condition_number, min(m, n))
    S = np.zeros((m, n))
    np.fill_diagonal(S, singular_vals)
    A = U @ S @ V.T
    return normalize_columns(A)


def perturbed_sensing(A: np.ndarray, delta: float, seed: int = 99) -> np.ndarray:
    """Perturb sensing matrix: A_new = A + delta * E, E random."""
    rng = np.random.RandomState(seed)
    E = rng.randn(*A.shape) / np.sqrt(A.shape[0])
    A_new = A + delta * E
    return normalize_columns(A_new)


def get_sensing_matrix(m: int, n: int, matrix_type: str = 'gaussian',
                       seed: int = 42, **kwargs) -> np.ndarray:
    """Factory function for sensing matrices."""
    builders = {
        'gaussian': gaussian_sensing,
        'correlated': correlated_sensing,
        'partial_orthogonal': partial_orthogonal_sensing,
        'ill_conditioned': ill_conditioned_sensing,
    }
    if matrix_type not in builders:
        raise ValueError(f"Unknown matrix type: {matrix_type}")
    return builders[matrix_type](m, n, seed=seed, **kwargs)


def compute_analytic_W(A: np.ndarray, alpha: float = 1.0,
                       max_iter: int = 500) -> np.ndarray:
    """
    Compute analytic weight matrix W for ALISTA/HyperLISTA via symmetric
    Jacobian formulation: min_D ||D^T D - I||_F^2 + (1/alpha)||D - GA||_F^2
    subject to diag(D^T D) = 1.
    Returns W = (G^T G) A such that W^T A is approximately symmetric.
    """
    m, n = A.shape
    D = A.copy()
    D = D / np.linalg.norm(D, axis=0, keepdims=True)

    for _ in range(max_iter):
        gram = D.T @ D
        grad_gram = 2.0 * D @ (gram - np.eye(n))
        grad_penalty = (2.0 / alpha) * (D - D)
        grad = grad_gram + grad_penalty

        D = D - 0.001 * grad

        col_norms = np.sqrt(np.sum(D ** 2, axis=0, keepdims=True))
        D = D / np.maximum(col_norms, 1e-12)

    W = np.linalg.lstsq(A, D, rcond=None)[0]
    W = A @ W
    return W.astype(np.float32)


def compute_mutual_coherence(A: np.ndarray) -> float:
    """Mutual coherence: max_{i!=j} |<a_i, a_j>|."""
    A_norm = normalize_columns(A)
    G = A_norm.T @ A_norm
    np.fill_diagonal(G, 0)
    return np.max(np.abs(G))


def pick_device() -> str:
    """Return 'cuda' > 'mps' > 'cpu' based on availability.

    Apple Silicon (MPS) is preferred on macOS hosts without CUDA; falls
    back to CPU when neither is available.
    """
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None \
            and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def compute_symmetric_W_gpu(A_tensor: torch.Tensor, num_iter: int = 200,
                            lr: float = 0.002) -> torch.Tensor:
    """GPU-accelerated symmetric W computation for HyperLISTA."""
    device = A_tensor.device
    m, n = A_tensor.shape
    norms = A_tensor.norm(dim=0, keepdim=True).clamp(min=1e-12)
    D = A_tensor / norms

    I_n = torch.eye(n, device=device)
    for _ in range(num_iter):
        gram = D.t() @ D
        grad = 2.0 * D @ (gram - I_n)
        D = D - lr * grad
        norms = D.norm(dim=0, keepdim=True).clamp(min=1e-12)
        D = D / norms

    G = torch.linalg.lstsq(A_tensor, D).solution
    W = A_tensor @ G
    return W
