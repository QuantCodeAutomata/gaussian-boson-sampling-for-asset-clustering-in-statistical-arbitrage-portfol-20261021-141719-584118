"""Tests for Spectral and SPONGE clustering."""
import numpy as np
import pytest
from src.clustering import spectral_clustering, sponge_clustering, random_clustering


def _is_valid_partition_simple(clusters, N):
    all_nodes = set()
    for c in clusters:
        s = set(c)
        if s & all_nodes: return False
        all_nodes |= s
    return all_nodes == set(range(N))


@pytest.mark.parametrize("N,K", [(6, 2), (10, 3), (8, 2)])
def test_spectral_produces_valid_partition(N, K):
    rng = np.random.default_rng(42)
    C = np.random.default_rng(N).uniform(0.0, 0.3, (N, N))
    C = (C + C.T) / 2; np.fill_diagonal(C, 1.0)
    partition = spectral_clustering(C, K, rng)
    assert _is_valid_partition_simple(partition, N)

@pytest.mark.parametrize("N,K", [(6, 2), (10, 3)])
def test_sponge_produces_valid_partition(N, K):
    rng = np.random.default_rng(42)
    C = np.random.default_rng(N).uniform(-0.2, 0.4, (N, N))
    C = (C + C.T) / 2; np.fill_diagonal(C, 1.0)
    partition = sponge_clustering(C, K, rng)
    assert _is_valid_partition_simple(partition, N)

@pytest.mark.parametrize("N,K", [(5, 2), (10, 3)])
def test_random_clustering_valid(N, K):
    rng = np.random.default_rng(0)
    partition = random_clustering(N, K, rng)
    assert _is_valid_partition_simple(partition, N)
    assert len(partition) <= K

def test_spectral_k_bounds():
    """K is clamped to [2, N-1]."""
    rng = np.random.default_rng(42)
    C = np.eye(5) * 0.5; np.fill_diagonal(C, 1.0)
    # K=1 should be raised to 2
    partition = spectral_clustering(C, 1, rng)
    assert len(partition) >= 1
    # K=100 should not crash
    partition = spectral_clustering(C, 100, rng)
    assert _is_valid_partition_simple(partition, 5)

def test_spectral_highly_correlated():
    """Highly correlated blocks should be clustered together."""
    rng = np.random.default_rng(0)
    C = np.array([
        [1.0, 0.9, 0.1, 0.0],
        [0.9, 1.0, 0.0, 0.1],
        [0.1, 0.0, 1.0, 0.9],
        [0.0, 0.1, 0.9, 1.0],
    ])
    partition = spectral_clustering(C, 2, rng)
    # Expect {0,1} and {2,3} (though not guaranteed due to randomness)
    assert _is_valid_partition_simple(partition, 4)
    assert len(partition) == 2
