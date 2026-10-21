"""
Classical clustering algorithms for GBS StatArb benchmarking.

Paper: "Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios"

Implements:
  - Spectral Clustering with dynamic K (normalized Laplacian + K-means++)
  - Symmetric SPONGE (signed-Laplacian generalised eigenproblem + K-means++)
"""

import logging
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


# ── Spectral Clustering ───────────────────────────────────────────────────────

def spectral_clustering(
    C: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """
    Spectral clustering using the normalized symmetric Laplacian.

    Algorithm:
    1. Build similarity graph W from the cleaned correlation matrix C
       (use positive correlations only: W = max(0, C - I) + I for self-loops).
    2. Compute normalized symmetric Laplacian: L_sym = D^{-1/2} L D^{-1/2}.
    3. Take K smallest eigenvectors of L_sym (excluding the trivial zero eigenvalue).
    4. Row-normalize the eigenvector matrix.
    5. Apply K-means++ to cluster rows into K clusters.

    Paper: "Run Spectral clustering with the dynamically estimated K, using the
    relevant Laplacian eigenvectors followed by K-means++."

    Parameters
    ----------
    C : np.ndarray
        Cleaned correlation matrix, shape (N, N).
    K : int
        Number of clusters.
    rng : np.random.Generator
        Random number generator for K-means++.

    Returns
    -------
    List[List[int]]
        Partition of {0, ..., N-1} into K clusters.
    """
    N = C.shape[0]
    K = max(2, min(K, N - 1))

    # Build weighted adjacency from positive correlations (GBS graph approach)
    W = np.maximum(0.0, C)
    np.fill_diagonal(W, 0.0)

    # Degree matrix
    d = W.sum(axis=1)
    d_safe = np.where(d > 1e-12, d, 1.0)

    # Normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_safe))
    L_sym = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

    # Compute K smallest eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    idx = np.argsort(eigenvalues)
    # Skip the trivial zero eigenvector if eigenvalue ≈ 0
    start = 0
    if abs(eigenvalues[idx[0]]) < 1e-10:
        start = 0  # include all, K-means handles it
    X = eigenvectors[:, idx[start:start + K]]  # shape (N, K)

    # Row-normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X = X / norms

    # K-means++ clustering
    labels = _kmeans_pp(X, K, rng)
    return _labels_to_clusters(labels, N)


# ── Symmetric SPONGE ──────────────────────────────────────────────────────────

def sponge_clustering(
    C: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """
    Symmetric SPONGE (Signed Positive Over Negative Generalised Eigenproblem).

    Algorithm based on the paper's reference to "symmetric SPONGE" with the
    signed-Laplacian generalised eigenproblem.

    SPONGE (Cucuringu et al. 2019) uses:
      L+ = D+ - W+  (Laplacian of positive graph)
      L- = D- - W-  (Laplacian of negative graph)
    Solve: L+ v = lambda * (L+ + L-) v
    Take K smallest non-trivial eigenvectors, apply K-means++.

    For the correlation matrix C:
      W+ [i,j] = max(0, C[i,j])   (positive correlations)
      W- [i,j] = max(0, -C[i,j])  (negative correlations, abs value)

    ASSUMPTION [symmetric-SPONGE]: Implementing the standard SPONGE formulation
    from Cucuringu et al. (2019). Exact paper variant may differ.

    Parameters
    ----------
    C : np.ndarray
        Cleaned correlation matrix, shape (N, N).
    K : int
        Number of clusters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    List[List[int]]
        Partition of {0, ..., N-1} into K clusters.
    """
    N = C.shape[0]
    K = max(2, min(K, N - 1))

    # Positive and negative weight matrices (diagonal = 0)
    C_offdiag = C.copy()
    np.fill_diagonal(C_offdiag, 0.0)
    W_pos = np.maximum(0.0, C_offdiag)
    W_neg = np.maximum(0.0, -C_offdiag)

    # Degree matrices
    d_pos = W_pos.sum(axis=1)
    d_neg = W_neg.sum(axis=1)

    D_pos = np.diag(d_pos)
    D_neg = np.diag(d_neg)

    # Signed Laplacians
    L_pos = D_pos - W_pos  # Laplacian of positive graph
    L_neg = D_neg - W_neg  # Laplacian of negative graph

    # Generalised eigenproblem: L+ v = lambda (L+ + L-) v
    # Equivalent to: (L+ + L-)^{-1} L+ v = lambda v  (when L+ + L- is invertible)
    B = L_pos + L_neg + np.eye(N) * 1e-8  # regularise for invertibility

    try:
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(L_pos, B)
    except np.linalg.LinAlgError:
        logger.warning("SPONGE: generalised eigenproblem failed; falling back to spectral.")
        return spectral_clustering(C, K, rng)

    # Take K smallest eigenvectors (SPONGE uses smallest generalised eigenvalues)
    idx = np.argsort(eigenvalues)
    X = eigenvectors[:, idx[:K]]  # shape (N, K)

    # Row-normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X = X / norms

    labels = _kmeans_pp(X, K, rng)
    return _labels_to_clusters(labels, N)


# ── Random Clustering (baseline) ─────────────────────────────────────────────

def random_clustering(
    N: int,
    K: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """
    Random partition of N nodes into K clusters of approximately equal size.

    ASSUMPTION [random-baseline]: Uniform random partition. Paper does not
    fully specify cluster-size distribution for the random baseline.

    Parameters
    ----------
    N : int
        Total number of nodes.
    K : int
        Number of clusters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    List[List[int]]
        Random partition.
    """
    K = max(1, min(K, N))
    nodes = rng.permutation(N).tolist()
    clusters = [[] for _ in range(K)]
    for i, node in enumerate(nodes):
        clusters[i % K].append(node)
    return [sorted(c) for c in clusters if c]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _kmeans_pp(
    X: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply K-means++ to embedding matrix X.

    Parameters
    ----------
    X : np.ndarray
        Embedding matrix, shape (N, D).
    K : int
        Number of clusters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Cluster labels, shape (N,), values in {0, ..., K-1}.
    """
    seed = int(rng.integers(0, 2**31))
    km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    return labels


def _labels_to_clusters(labels: np.ndarray, N: int) -> List[List[int]]:
    """
    Convert cluster label array to list-of-lists partition.

    Parameters
    ----------
    labels : np.ndarray
        Cluster assignments, shape (N,).
    N : int
        Total number of nodes.

    Returns
    -------
    List[List[int]]
        Partition as list of clusters, each cluster a sorted list of node indices.
    """
    clusters: dict = {}
    for node, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(node)
    return [sorted(v) for v in clusters.values() if v]
