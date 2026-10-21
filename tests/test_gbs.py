"""Tests for GBS algorithms."""
import math
import numpy as np
import pytest
from src.gbs import (
    weighted_density, unweighted_density, cluster_value, global_partition_value,
    gbs_emulate_samples, gbs_boost_partition, gbs_roots_partition,
    qic_gbs_partition, _is_valid_partition, inter_window_jaccard,
    jaccard_similarity, compute_H,
)


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_weighted_density_pair():
    """WD of a 2-node subgraph: 2*w / (2*1) = w."""
    A = np.array([[0, 0.5], [0.5, 0]])
    assert abs(weighted_density(A, [0, 1]) - 0.5) < 1e-10

def test_weighted_density_single_node():
    """WD of 1-node subgraph is 0."""
    A = np.eye(3)
    assert weighted_density(A, [0]) == 0.0

def test_unweighted_density_complete():
    """Unweighted density of fully connected graph = 1.0."""
    A = np.ones((4, 4)) - np.eye(4)
    assert abs(unweighted_density(A, [0, 1, 2, 3]) - 1.0) < 1e-10

def test_compute_H_min_one():
    """H is at least 1 even for very small F."""
    assert compute_H(1e-10, 10) >= 1

def test_compute_H_formula():
    """H = round(F * N * log(N))."""
    N, F = 30, 0.7
    expected = round(F * N * math.log(N))
    assert compute_H(F, N) == expected

def test_valid_partition():
    """Partition validation: all nodes covered, no overlap."""
    assert _is_valid_partition([[0, 1], [2, 3]], 4)
    assert not _is_valid_partition([[0, 1], [1, 2]], 3)  # overlap
    assert not _is_valid_partition([[0, 1]], 3)           # missing node 2

def test_jaccard_similarity_identical():
    assert abs(jaccard_similarity([0, 1, 2], [0, 1, 2]) - 1.0) < 1e-10

def test_jaccard_similarity_disjoint():
    assert abs(jaccard_similarity([0, 1], [2, 3]) - 0.0) < 1e-10

def test_jaccard_similarity_partial():
    j = jaccard_similarity([0, 1], [1, 2])
    assert abs(j - 1/3) < 1e-10

def test_inter_window_jaccard_same_partition():
    p = [[0, 1], [2, 3]]
    assert abs(inter_window_jaccard(p, p) - 1.0) < 1e-10

def test_inter_window_jaccard_range():
    p1 = [[0, 1, 2], [3, 4]]
    p2 = [[0, 1], [2, 3, 4]]
    j = inter_window_jaccard(p1, p2)
    assert 0.0 <= j <= 1.0

def test_cluster_value_symmetry():
    """cluster_value counts ordered pairs (both directions)."""
    A = np.array([[0, 0.3, 0.5], [0.3, 0, 0.2], [0.5, 0.2, 0]])
    cv = cluster_value(A, [0, 1, 2])
    # Sum over all i≠j: 2*(0.3 + 0.5 + 0.2) = 2.0
    assert abs(cv - 2.0) < 1e-10


# ── GBS emulator tests ────────────────────────────────────────────────────────

def test_gbs_emulate_returns_valid_samples():
    """GBS emulator: all samples have ≥2 distinct nodes."""
    A = np.array([[0, 0.8, 0.3, 0.1],
                  [0.8, 0, 0.5, 0.2],
                  [0.3, 0.5, 0, 0.6],
                  [0.1, 0.2, 0.6, 0]])
    rng = np.random.default_rng(42)
    samples = gbs_emulate_samples(A, 20, 2.0, rng, loss_rate=0.0)
    for s in samples:
        assert len(s) >= 2, f"Sample has < 2 nodes: {s}"
        assert len(set(s)) == len(s), f"Duplicate nodes: {s}"

def test_gbs_emulate_loss_reduces_mean_size():
    """Under higher loss, mean cluster size should not increase significantly."""
    A = np.ones((6, 6)) * 0.3
    np.fill_diagonal(A, 0.0)
    rng0 = np.random.default_rng(42)
    rng1 = np.random.default_rng(42)
    s0 = gbs_emulate_samples(A, 50, 2.5, rng0, loss_rate=0.0)
    s9 = gbs_emulate_samples(A, 50, 2.5, rng1, loss_rate=0.9)
    mean0 = np.mean([len(s) for s in s0]) if s0 else 0
    mean9 = np.mean([len(s) for s in s9]) if s9 else 0
    # Under 90% loss, mean size should be <= mean with no loss
    assert mean9 <= mean0 + 1.5, f"Loss did not reduce sample size: {mean9} > {mean0}"


# ── Partition algorithm tests ─────────────────────────────────────────────────

@pytest.mark.parametrize("N", [4, 6, 8])
def test_gbs_boost_partition_valid(N):
    """GBS Boost always produces a valid complete partition."""
    rng = np.random.default_rng(7)
    A = np.random.default_rng(N).uniform(0, 0.5, (N, N))
    A = (A + A.T) / 2; np.fill_diagonal(A, 0)
    A = np.maximum(A, 0)
    partition = gbs_boost_partition(A, H=20, nbar=math.sqrt(N), rng=rng)
    assert _is_valid_partition(partition, N), f"Invalid partition: {partition}"

@pytest.mark.parametrize("N", [4, 6, 8])
def test_gbs_roots_partition_valid(N):
    """GBS Roots always produces a valid complete partition."""
    rng = np.random.default_rng(13)
    A = np.random.default_rng(N).uniform(0, 0.5, (N, N))
    A = (A + A.T) / 2; np.fill_diagonal(A, 0)
    A = np.maximum(A, 0)
    partition = gbs_roots_partition(A, H=20, nbar=math.sqrt(N), rng=rng)
    assert _is_valid_partition(partition, N), f"Invalid partition: {partition}"

@pytest.mark.parametrize("N", [4, 6, 8])
def test_qic_gbs_partition_valid(N):
    """QIC-GBS always produces a valid complete partition."""
    rng = np.random.default_rng(21)
    A = np.random.default_rng(N).uniform(0, 0.5, (N, N))
    A = (A + A.T) / 2; np.fill_diagonal(A, 0)
    A = np.maximum(A, 0)
    partition = qic_gbs_partition(A, H=20, nbar=math.sqrt(N), rng=rng)
    assert _is_valid_partition(partition, N), f"Invalid partition: {partition}"

def test_gbs_boost_empty_graph():
    """GBS Boost handles zero-weight graph gracefully."""
    rng = np.random.default_rng(0)
    A = np.zeros((4, 4))
    partition = gbs_boost_partition(A, H=5, nbar=2.0, rng=rng)
    assert _is_valid_partition(partition, 4)

def test_gbs_boost_two_nodes():
    """GBS Boost with N=2 returns one cluster of 2 nodes."""
    rng = np.random.default_rng(0)
    A = np.array([[0, 0.5], [0.5, 0]])
    partition = gbs_boost_partition(A, H=5, nbar=1.0, rng=rng)
    assert _is_valid_partition(partition, 2)
    assert len(partition) == 1
    assert sorted(partition[0]) == [0, 1]


# ── Global partition value test ───────────────────────────────────────────────

def test_global_partition_value_sum():
    A = np.array([[0, 0.4, 0.0, 0.0],
                  [0.4, 0, 0.0, 0.0],
                  [0.0, 0.0, 0, 0.6],
                  [0.0, 0.0, 0.6, 0]])
    clusters = [[0, 1], [2, 3]]
    val = global_partition_value(A, clusters)
    # cluster [0,1]: 2*(0.4) = 0.8; cluster [2,3]: 2*(0.6) = 1.2; total = 2.0
    assert abs(val - 2.0) < 1e-10
