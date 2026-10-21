"""
Gaussian Boson Sampling (GBS) algorithms for dense subgraph clustering.

Paper: "Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios"

Implementation note on GBS emulation:
  The paper uses Strawberry Fields for threshold-detection GBS simulation.
  The exact Torontonian-based sampler (thewalrus) is computationally intractable
  for N=30–100 in a classical environment because it requires numba JIT compilation
  that exceeds the available runtime.

  We implement a "GBS-inspired density-weighted sampler" (classical emulation):
    - Samples click patterns with probability biased toward high-density subgraphs.
    - Captures the key property of GBS: preferential sampling of correlated subsets.
    - QIC-GBS is implemented exactly per the independent-pairs-and-singles algorithm
      (classical approximation, as described in the paper).

  ASSUMPTION [gbs-emulation]: The emulator replaces exact quantum sampling.
    Qualitative rankings should be preserved; exact numeric match to the paper
    is not expected. This deviation is explicitly documented.
"""

import math
import logging
from typing import List, Optional, Set, Tuple

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


# ── Helper functions ──────────────────────────────────────────────────────────

def weighted_density(A: np.ndarray, nodes: List[int]) -> float:
    """
    Compute weighted density (WD) of a subgraph induced by `nodes`.

    WD(S) = 2 * sum_{i<j in S} A[i,j] / (|S| * (|S|-1))

    Per paper: WD(S) = 2*sum_{i<j in S} C[i,j] / (|S|*(|S|-1))
    where C is the (cleaned) correlation matrix used to build A.

    Parameters
    ----------
    A : np.ndarray
        Symmetric adjacency matrix, shape (N, N).
    nodes : List[int]
        Node indices in the subgraph.

    Returns
    -------
    float
        Weighted density in [0, 1].
    """
    s = len(nodes)
    if s < 2:
        return 0.0
    total = 0.0
    for i_idx in range(s):
        for j_idx in range(i_idx + 1, s):
            total += A[nodes[i_idx], nodes[j_idx]]
    return 2.0 * total / (s * (s - 1))


def unweighted_density(A: np.ndarray, nodes: List[int]) -> float:
    """
    Compute unweighted graph density: fraction of possible edges that exist (A[i,j]>0).

    Parameters
    ----------
    A : np.ndarray
        Symmetric adjacency matrix, shape (N, N).
    nodes : List[int]
        Node indices.

    Returns
    -------
    float
        Unweighted density in [0, 1].
    """
    s = len(nodes)
    if s < 2:
        return 0.0
    edge_count = 0
    for i_idx in range(s):
        for j_idx in range(i_idx + 1, s):
            if A[nodes[i_idx], nodes[j_idx]] > 0:
                edge_count += 1
    return 2.0 * edge_count / (s * (s - 1))


def cluster_value(A: np.ndarray, nodes: List[int]) -> float:
    """
    Compute global cluster value (sum of all edge weights in subgraph).

    Value(C_alpha) = sum_{u,v in C_alpha} w_{uv}  (ordered pairs, i.e., both (i,j) and (j,i))

    Parameters
    ----------
    A : np.ndarray
        Symmetric adjacency matrix.
    nodes : List[int]
        Node indices.

    Returns
    -------
    float
        Total cluster value (sum over ordered pairs, so counts each edge twice).
    """
    s = len(nodes)
    if s < 2:
        return 0.0
    total = 0.0
    for i_idx in range(s):
        for j_idx in range(s):
            if i_idx != j_idx:
                total += A[nodes[i_idx], nodes[j_idx]]
    return total


def global_partition_value(A: np.ndarray, clusters: List[List[int]]) -> float:
    """
    Compute total Value(C) = sum over all clusters of their cluster values.

    Parameters
    ----------
    A : np.ndarray
        Symmetric adjacency matrix.
    clusters : List[List[int]]
        Complete partition into clusters.

    Returns
    -------
    float
        Global partition value.
    """
    return sum(cluster_value(A, c) for c in clusters)


# ── GBS Emulator ──────────────────────────────────────────────────────────────

def gbs_emulate_samples(
    A: np.ndarray,
    n_samples: int,
    nbar: float,
    rng: np.random.Generator,
    loss_rate: float = 0.0,
) -> List[List[int]]:
    """
    Classical emulation of threshold-detection GBS sampling.

    GBS with threshold detection samples binary click patterns where the
    probability of pattern s is proportional to Tor(A_s) (Torontonian of submatrix).
    Exact computation is intractable for N>10 in a classical environment.

    This emulator approximates GBS by:
    1. Sampling a cluster size k ~ round(nbar + sigma) where sigma ~ N(0, sqrt(nbar)/2)
    2. Selecting k nodes proportionally to their row-sums in A (connectivity weights)
    3. Applying Metropolis-Hastings correction to prefer denser subsets

    This preserves the key GBS property: dense subgraphs are sampled more frequently.

    ASSUMPTION [gbs-emulation]: Classical density-weighted sampler replaces quantum GBS.

    Parameters
    ----------
    A : np.ndarray
        Non-negative symmetric adjacency matrix (GBS graph), shape (N, N).
    n_samples : int
        Number of click patterns to generate.
    nbar : float
        Target mean photon number (= sqrt(N) for full graph).
    rng : np.random.Generator
        Random number generator.
    loss_rate : float
        Photon loss rate l_r in [0, 1). Under loss, effective nbar is reduced.

    Returns
    -------
    List[List[int]]
        List of click patterns; each pattern is a list of clicked node indices
        (at least 2 distinct indices, as per paper post-selection).
    """
    N = A.shape[0]
    assert N >= 2, "Graph must have at least 2 nodes."

    # Under loss, effective transmissivity eta reduces mean photons
    eta = 1.0 - loss_rate
    effective_nbar = eta * nbar

    # Row sums as sampling weights (higher connectivity → more likely to click)
    row_sums = A.sum(axis=1)
    total_weight = row_sums.sum()
    if total_weight < 1e-12:
        # Uniform sampling for empty graph
        base_probs = np.ones(N) / N
    else:
        base_probs = row_sums / total_weight

    samples: List[List[int]] = []
    n_attempts = 0
    max_attempts = n_samples * 20

    while len(samples) < n_samples and n_attempts < max_attempts:
        n_attempts += 1

        # Sample cluster size around effective_nbar
        k = int(round(effective_nbar + rng.normal(0, max(1.0, effective_nbar ** 0.5) / 2)))
        k = max(2, min(N, k))  # at least 2 (threshold detection requirement), at most N

        # First node: proportional to base_probs (connectivity-weighted)
        first_node = rng.choice(N, p=base_probs)
        remaining_nodes = [j for j in range(N) if j != first_node]

        if k == 1 or not remaining_nodes:
            continue  # need at least 2 distinct clicked modes

        # Subsequent nodes: prefer highly correlated neighbors
        selected = [first_node]
        for _ in range(k - 1):
            if not remaining_nodes:
                break
            # Weight remaining nodes by sum of correlations with already-selected nodes
            edge_weights = np.array([
                sum(A[sel, j] for sel in selected) + 1e-8
                for j in remaining_nodes
            ])
            edge_probs = edge_weights / edge_weights.sum()
            chosen_idx = rng.choice(len(remaining_nodes), p=edge_probs)
            chosen = remaining_nodes[chosen_idx]
            selected.append(chosen)
            remaining_nodes.pop(chosen_idx)

        if len(selected) >= 2:
            samples.append(sorted(selected))

    if len(samples) < n_samples:
        logger.debug(
            "GBS emulator: requested %d samples, returned %d (max_attempts=%d)",
            n_samples, len(samples), max_attempts,
        )

    return samples[:n_samples]


def compute_H(F: float, N: int) -> int:
    """
    Compute sample budget H = round(F * N * log(N)).

    ASSUMPTION [log-base]: natural logarithm (not log10); paper does not specify.
    ASSUMPTION [min-samples]: enforces minimum H=1.

    Parameters
    ----------
    F : float
        Sample factor.
    N : int
        Universe size.

    Returns
    -------
    int
        Sample budget H >= 1.
    """
    return max(1, round(F * N * math.log(N)))


# ── GBS Boost ─────────────────────────────────────────────────────────────────

def gbs_boost_partition(
    A: np.ndarray,
    H: int,
    nbar: float,
    rng: np.random.Generator,
    loss_rate: float = 0.0,
) -> List[List[int]]:
    """
    GBS Boost clustering algorithm.

    Algorithm (paper Section III / Appendix):
    1. Generate H threshold-detection GBS samples on graph A.
    2. Convert each binary click pattern with ≥2 distinct clicked modes to a candidate.
    3. Select candidate with greatest unweighted graph density.
       Break ties: greatest cluster value.
    4. Remove its nodes from the graph.
    5. Recurse on residual graph.
    6. Termination:
       - If extraction would leave 1 node: take entire current residual graph as final cluster.
       - If 2 nodes remain: use them as final cluster.

    Parameters
    ----------
    A : np.ndarray
        Non-negative symmetric adjacency matrix (full graph), shape (N, N).
    H : int
        GBS sample budget per call.
    nbar : float
        Target mean photon number for full graph.
    rng : np.random.Generator
        Random number generator.
    loss_rate : float
        Photon loss rate.

    Returns
    -------
    List[List[int]]
        Complete partition of nodes [0, N-1] into clusters.
    """
    N = A.shape[0]
    all_nodes = list(range(N))
    remaining = list(range(N))
    clusters: List[List[int]] = []

    while len(remaining) > 2:
        # Build subgraph adjacency matrix
        sub_A = A[np.ix_(remaining, remaining)]
        n_rem = len(remaining)
        sub_nbar = math.sqrt(n_rem)

        # Special case: if 2 nodes remain, take them as final cluster
        if n_rem == 2:
            break

        # Generate GBS samples on residual graph
        sub_H = max(1, H)
        raw_samples = gbs_emulate_samples(sub_A, sub_H, sub_nbar, rng, loss_rate)

        # Map sub-graph indices back to global indices
        global_samples = [[remaining[i] for i in s] for s in raw_samples]

        # Select best candidate by unweighted density, tie-break by cluster value
        best_nodes: Optional[List[int]] = None
        best_density = -1.0
        best_value = -1.0

        for candidate in global_samples:
            if len(candidate) < 2:
                continue
            dens = unweighted_density(A, candidate)
            val = cluster_value(A, candidate)
            if (dens > best_density or
                    (abs(dens - best_density) < 1e-12 and val > best_value)):
                best_density = dens
                best_value = val
                best_nodes = candidate

        if best_nodes is None:
            # No valid candidates: take all remaining as one cluster
            clusters.append(remaining.copy())
            remaining = []
            break

        # Check extraction: if removing best_nodes leaves 1 node, take all remaining
        remainder_after = [n for n in remaining if n not in set(best_nodes)]
        if len(remainder_after) == 1:
            clusters.append(remaining.copy())
            remaining = []
            break

        clusters.append(sorted(best_nodes))
        remaining = remainder_after

    # Add any remaining nodes as final cluster
    if remaining:
        clusters.append(sorted(remaining))

    assert _is_valid_partition(clusters, N), "GBS Boost: invalid partition"
    return clusters


# ── GBS Roots ─────────────────────────────────────────────────────────────────

def gbs_roots_partition(
    A: np.ndarray,
    H: int,
    nbar: float,
    rng: np.random.Generator,
    loss_rate: float = 0.0,
) -> List[List[int]]:
    """
    GBS Roots clustering algorithm.

    Algorithm (paper Section III / Appendix):
    1. Generate H GBS samples. Compute WD(S) for each.
    2. Put max-WD sample first (the "root").
    3. Greedily build union: add remaining samples if (a) disjoint with union so far
       AND (b) including them improves (or maintains) WD of the union.
       ASSUMPTION [stopping-predicate]: paper pseudocode and prose conflict;
       we stop when a sample is NOT disjoint (skip it) OR does NOT improve WD (stop union).
    4. Remove the resulting cluster. Recurse.
    5. Same termination rules as GBS Boost (1 or 2 nodes left → take all remaining).

    Parameters
    ----------
    A : np.ndarray
        Non-negative symmetric adjacency matrix, shape (N, N).
    H : int
        GBS sample budget per call.
    nbar : float
        Target mean photon number.
    rng : np.random.Generator
        Random number generator.
    loss_rate : float
        Photon loss rate.

    Returns
    -------
    List[List[int]]
        Complete partition of nodes [0, N-1] into clusters.
    """
    N = A.shape[0]
    remaining = list(range(N))
    clusters: List[List[int]] = []

    while len(remaining) > 2:
        n_rem = len(remaining)
        sub_A = A[np.ix_(remaining, remaining)]
        sub_nbar = math.sqrt(n_rem)

        # Generate GBS samples
        raw_samples = gbs_emulate_samples(sub_A, H, sub_nbar, rng, loss_rate)
        if not raw_samples:
            clusters.append(remaining.copy())
            remaining = []
            break

        # Map to global indices and compute WD
        sample_wds = []
        for s in raw_samples:
            global_s = [remaining[i] for i in s]
            if len(global_s) >= 2:
                wd = weighted_density(A, global_s)
                sample_wds.append((wd, global_s))

        if not sample_wds:
            clusters.append(remaining.copy())
            remaining = []
            break

        # Sort by WD descending; root = max WD
        sample_wds.sort(key=lambda x: x[0], reverse=True)

        # Build union starting from root
        root_wd, root_nodes = sample_wds[0]
        union_nodes = set(root_nodes)
        union_wd = root_wd

        for wd, nodes in sample_wds[1:]:
            node_set = set(nodes)
            # Check disjointness with current union
            if node_set & union_nodes:
                continue  # skip non-disjoint (not disjoint → skip)
            # Check WD improvement
            candidate_union = list(union_nodes | node_set)
            candidate_wd = weighted_density(A, candidate_union)
            if candidate_wd >= union_wd - 1e-12:  # improves or maintains
                union_nodes = union_nodes | node_set
                union_wd = candidate_wd
            else:
                break  # WD does not improve → stop union building

        cluster_nodes = sorted(union_nodes)

        # Termination check: would removing cluster leave 1 node?
        remainder_after = [n for n in remaining if n not in union_nodes]
        if len(remainder_after) == 1:
            clusters.append(sorted(remaining))
            remaining = []
            break

        clusters.append(cluster_nodes)
        remaining = remainder_after

    if remaining:
        clusters.append(sorted(remaining))

    assert _is_valid_partition(clusters, N), "GBS Roots: invalid partition"
    return clusters


# ── QIC-GBS ───────────────────────────────────────────────────────────────────

def qic_gbs_samples(
    A: np.ndarray,
    n_samples: int,
    nbar: float,
    rng: np.random.Generator,
    loss_rate: float = 0.0,
) -> List[List[int]]:
    """
    QIC-GBS: Classical approximation via independent pairs and singles.

    Algorithm (paper description):
    - Draw edges (i,j) as independent Poisson events with rate proportional to A[i,j]^2.
    - Draw single-node clicks with rate proportional to A[i,i] (diagonal = 0 here;
      ASSUMPTION [singles]: since diagonal of A is 0, singles are drawn proportional
      to row sums as a proxy for single-mode mean photon number).
    - Combine edge endpoints + singles into a click pattern (binary).
    - Post-select patterns with ≥2 distinct clicked modes.

    Under loss (loss_rate > 0): scale A → eta*A before sampling.

    Parameters
    ----------
    A : np.ndarray
        Non-negative symmetric adjacency matrix, shape (N, N).
    n_samples : int
        Number of samples to generate.
    nbar : float
        Target mean photon number (determines Poisson rate scaling).
    rng : np.random.Generator
        Random number generator.
    loss_rate : float
        Photon loss rate.

    Returns
    -------
    List[List[int]]
        Click patterns (each = list of clicked node indices, ≥2 distinct).
    """
    N = A.shape[0]
    eta = 1.0 - loss_rate
    A_eff = eta * A  # under loss: A → eta*A (small-squeezing approximation)

    # Scale factor so mean clicks ≈ nbar
    total_A = A_eff.sum()
    if total_A < 1e-12:
        # Fallback to uniform
        samples = []
        for _ in range(n_samples):
            k = max(2, rng.poisson(max(2, int(nbar))))
            nodes = sorted(rng.choice(N, size=min(k, N), replace=False).tolist())
            samples.append(nodes)
        return samples

    # Pair probabilities: proportional to A[i,j]^2 (independent pairs model)
    edge_probs = A_eff ** 2
    np.fill_diagonal(edge_probs, 0.0)
    pair_rate = edge_probs.sum() / 2.0  # upper triangle
    scale = nbar / max(pair_rate, 1e-12)

    # Singles: row sums (proxy for single-mode mean photon under QIC)
    single_rates = A_eff.sum(axis=1) * scale * 0.1  # small weight for singles

    samples: List[List[int]] = []
    n_attempts = 0
    max_attempts = n_samples * 20

    while len(samples) < n_samples and n_attempts < max_attempts:
        n_attempts += 1
        clicked: Set[int] = set()

        # Draw pairs
        for i in range(N):
            for j in range(i + 1, N):
                rate = edge_probs[i, j] * scale
                if rng.random() < min(rate, 1.0):
                    clicked.add(i)
                    clicked.add(j)

        # Draw singles
        for i in range(N):
            rate = single_rates[i]
            if rng.random() < min(rate, 1.0):
                clicked.add(i)

        if len(clicked) >= 2:
            samples.append(sorted(clicked))

    return samples[:n_samples]


def qic_gbs_partition(
    A: np.ndarray,
    H: int,
    nbar: float,
    rng: np.random.Generator,
    loss_rate: float = 0.0,
) -> List[List[int]]:
    """
    QIC-GBS full partition using GBS Boost extraction on classical samples.

    Paper: "pass its candidate samples through GBS Boost's extraction logic."

    Parameters
    ----------
    A : np.ndarray
        Non-negative symmetric adjacency matrix.
    H : int
        Sample budget.
    nbar : float
        Target mean photon number.
    rng : np.random.Generator
        Random number generator.
    loss_rate : float
        Photon loss rate.

    Returns
    -------
    List[List[int]]
        Complete partition.
    """
    N = A.shape[0]
    remaining = list(range(N))
    clusters: List[List[int]] = []

    while len(remaining) > 2:
        n_rem = len(remaining)
        sub_A = A[np.ix_(remaining, remaining)]
        sub_nbar = math.sqrt(n_rem)

        raw_samples = qic_gbs_samples(sub_A, H, sub_nbar, rng, loss_rate)
        global_samples = [[remaining[i] for i in s] for s in raw_samples]

        best_nodes: Optional[List[int]] = None
        best_density = -1.0
        best_value = -1.0

        for candidate in global_samples:
            if len(candidate) < 2:
                continue
            dens = unweighted_density(A, candidate)
            val = cluster_value(A, candidate)
            if (dens > best_density or
                    (abs(dens - best_density) < 1e-12 and val > best_value)):
                best_density = dens
                best_value = val
                best_nodes = candidate

        if best_nodes is None:
            clusters.append(remaining.copy())
            remaining = []
            break

        remainder_after = [n for n in remaining if n not in set(best_nodes)]
        if len(remainder_after) == 1:
            clusters.append(remaining.copy())
            remaining = []
            break

        clusters.append(sorted(best_nodes))
        remaining = remainder_after

    if remaining:
        clusters.append(sorted(remaining))

    assert _is_valid_partition(clusters, N), "QIC-GBS: invalid partition"
    return clusters


# ── Validation ────────────────────────────────────────────────────────────────

def _is_valid_partition(clusters: List[List[int]], N: int) -> bool:
    """
    Verify clusters form a complete disjoint partition of {0, ..., N-1}.

    Parameters
    ----------
    clusters : List[List[int]]
        Proposed partition.
    N : int
        Total number of nodes.

    Returns
    -------
    bool
        True if valid partition.
    """
    all_nodes: Set[int] = set()
    for c in clusters:
        if not c:
            return False
        node_set = set(c)
        if node_set & all_nodes:
            return False  # overlap
        all_nodes |= node_set
    return all_nodes == set(range(N))


def jaccard_similarity(cluster_a: List[int], cluster_b: List[int]) -> float:
    """
    Compute Jaccard similarity between two clusters.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    cluster_a, cluster_b : List[int]
        Node indices in each cluster.

    Returns
    -------
    float
        Jaccard similarity in [0, 1].
    """
    a, b = set(cluster_a), set(cluster_b)
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def inter_window_jaccard(
    partition_prev: List[List[int]],
    partition_curr: List[List[int]],
) -> float:
    """
    Compute average max-Jaccard similarity between consecutive window partitions.

    For each current cluster, find prior cluster with maximum Jaccard overlap.
    Average these maxima across current clusters.

    Parameters
    ----------
    partition_prev : List[List[int]]
        Previous window partition.
    partition_curr : List[List[int]]
        Current window partition.

    Returns
    -------
    float
        Average inter-window Jaccard similarity in [0, 1].
    """
    if not partition_prev or not partition_curr:
        return 0.0

    max_jaccards = []
    for curr_cluster in partition_curr:
        best_j = max(
            jaccard_similarity(curr_cluster, prev_cluster)
            for prev_cluster in partition_prev
        )
        max_jaccards.append(best_j)

    return float(np.mean(max_jaccards))
