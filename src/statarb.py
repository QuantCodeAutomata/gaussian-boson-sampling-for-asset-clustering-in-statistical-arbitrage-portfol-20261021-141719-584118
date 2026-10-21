"""
Statistical Arbitrage strategy implementation.

Paper: "Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios"

Implements the market-neutral contrarian portfolio strategy:
  - Within each cluster: classify winners vs. losers from cumulative residual deviations.
  - Short winners, long losers (contrarian mean-reversion bet).
  - Equal weight within each leg; zero-cost (long = short capital).
  - Hold 3 days or stop-win at 2% cumulative portfolio return.
  - Advance all windows by 3 days.
  - Zero transaction costs and zero price impact.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Formation signal ──────────────────────────────────────────────────────────

def compute_winner_loser(
    residuals_formation: pd.DataFrame,
    cluster_nodes: List[int],
    threshold: float = 0.0,
) -> Tuple[List[int], List[int]]:
    """
    Classify stocks within a cluster as previous winners or previous losers.

    Algorithm (paper Appendix A):
    "For each constituent, sum its deviation from the cluster mean.
     With p=0, classify positive cumulative deviations as previous winners
     and negative deviations as previous losers."

    Parameters
    ----------
    residuals_formation : pd.DataFrame
        Residual returns over the formation window (5 days), shape (5, N).
    cluster_nodes : List[int]
        Node indices (columns) of the cluster.
    threshold : float
        p parameter (= 0 per paper).

    Returns
    -------
    Tuple[List[int], List[int]]
        (winners, losers) as lists of global node indices.
    """
    if len(cluster_nodes) < 2:
        return [], []

    # Cluster mean return per day
    cluster_ret = residuals_formation.iloc[:, cluster_nodes]
    cluster_mean = cluster_ret.mean(axis=1)

    # Cumulative deviation from cluster mean
    deviations = (cluster_ret.subtract(cluster_mean, axis=0)).sum(axis=0)

    winners = [node for node, dev in zip(cluster_nodes, deviations.values)
               if dev > threshold]
    losers = [node for node, dev in zip(cluster_nodes, deviations.values)
              if dev <= threshold]

    return winners, losers


def build_cluster_portfolio(
    winners: List[int],
    losers: List[int],
) -> Optional[Dict[int, float]]:
    """
    Build zero-cost contrarian position for a single cluster.

    Short winners (negative weights), long losers (positive weights).
    Equal weight within each leg. Total absolute long capital = total absolute short capital.

    Normalise: sum of |weights| = 2 (so long leg = 1, short leg = -1).

    Parameters
    ----------
    winners : List[int]
        Previous winner node indices (to short).
    losers : List[int]
        Previous loser node indices (to long).

    Returns
    -------
    Dict[int, float] or None
        Weight per node (positive = long, negative = short).
        Returns None if either leg is empty.
    """
    if not winners or not losers:
        return None  # skip clusters without both legs

    n_win = len(winners)
    n_los = len(losers)

    weights: Dict[int, float] = {}
    for node in losers:
        weights[node] = 1.0 / n_los     # long losers, equal weight
    for node in winners:
        weights[node] = -1.0 / n_win    # short winners, equal weight

    return weights


def aggregate_cluster_portfolios(
    cluster_weights: List[Dict[int, float]],
    n_stocks: int,
) -> np.ndarray:
    """
    Aggregate portfolio weights across all clusters.

    ASSUMPTION [cross-cluster-aggregation]: Equal-weight across clusters.
    Paper does not fully specify cross-cluster capital aggregation.
    We normalise the aggregate weight vector so abs sum = 2 (market-neutral).

    Parameters
    ----------
    cluster_weights : List[Dict[int, float]]
        Per-cluster weight dictionaries (None entries are skipped).
    n_stocks : int
        Total number of stocks N.

    Returns
    -------
    np.ndarray
        Aggregated weight vector, shape (N,). Sum of positive = 1, negative = -1.
    """
    valid_weights = [w for w in cluster_weights if w is not None]
    if not valid_weights:
        return np.zeros(n_stocks)

    n_valid = len(valid_weights)
    agg = np.zeros(n_stocks)

    for w_dict in valid_weights:
        for node, w in w_dict.items():
            agg[node] += w / n_valid  # equal weight across clusters

    # Renormalise so long legs sum to 1 and short legs sum to -1
    pos = agg[agg > 0].sum()
    neg = agg[agg < 0].sum()

    if pos > 1e-12:
        agg[agg > 0] /= pos
    if neg < -1e-12:
        agg[agg < 0] /= abs(neg)

    return agg


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(
    residuals: pd.DataFrame,
    total_returns: pd.DataFrame,
    market_returns: pd.Series,
    partition_fn,
    corr_window: int = 5,
    beta_window: int = 60,
    k_est_window: int = 20,
    holding_days: int = 3,
    stop_win: float = 0.02,
    winner_loser_threshold: float = 0.0,
    seed: int = 42,
) -> Dict:
    """
    Run the full rolling statistical arbitrage backtest for one year.

    Rolling scheme:
    - At each rebalance date T, use the last 5 days of residuals for correlation.
    - Build partition using partition_fn(A, rng).
    - Classify winners/losers, build portfolio.
    - Hold for 3 trading days or stop-win at 2%.
    - Advance all windows by 3 days.

    Parameters
    ----------
    residuals : pd.DataFrame
        Residual returns, shape (T, N). Index = trading dates.
    total_returns : pd.DataFrame
        Total stock returns, shape (T, N).
    market_returns : pd.Series
        Market factor returns.
    partition_fn : callable
        Function(A, rng) -> List[List[int]] giving a cluster partition.
    corr_window : int
        Formation window for residual correlation (5 days).
    beta_window : int
        Beta estimation window (60 days).
    k_est_window : int
        K estimation window (20 days).
    holding_days : int
        Holding period in trading days.
    stop_win : float
        Early stop-win threshold (cumulative portfolio return).
    winner_loser_threshold : float
        p parameter for winner/loser classification.
    seed : int
        Random seed.

    Returns
    -------
    Dict with keys:
        cumulative_pnl (np.ndarray): cumulative portfolio P&L over time.
        daily_returns (np.ndarray): daily portfolio returns.
        dates (List): trading dates of each period.
        partitions (List): partition used at each rebalance.
        jaccard_similarities (List[float]): inter-window Jaccard per period.
        wds (List[float]): mean weighted density per period.
        cluster_values (List[float]): partition value per period.
        annual_return (float): total return over the full year.
    """
    from src.data import (
        compute_residual_correlation, clean_correlation_matrix,
        build_gbs_graph, estimate_k_from_eigenspacing,
    )
    from src.gbs import inter_window_jaccard, global_partition_value, weighted_density

    rng = np.random.default_rng(seed)
    N = residuals.shape[1]

    # Find evaluation period start: need at least corr_window rows of data
    min_start = corr_window  # residuals need at least 5 days
    trading_dates = residuals.index.tolist()
    n_dates = len(trading_dates)

    all_daily_returns: List[float] = []
    all_dates: List = []
    all_partitions: List[List[List[int]]] = []
    jaccard_sims: List[float] = []
    wds: List[float] = []
    cluster_vals: List[float] = []

    prev_partition: Optional[List[List[int]]] = None
    t = min_start

    while t + holding_days <= n_dates:
        # ── Formation window ──────────────────────────────────────────────────
        form_end = t
        form_start = max(0, form_end - corr_window)
        res_window = residuals.iloc[form_start:form_end]

        if len(res_window) < 2:
            t += holding_days
            continue

        # Compute residual correlation matrix
        C_raw = compute_residual_correlation(res_window, window=len(res_window))
        C_clean = clean_correlation_matrix(C_raw)
        A = build_gbs_graph(C_clean)

        # Build partition
        try:
            partition = partition_fn(A, rng)
        except Exception as exc:
            logger.warning("Partition failed at t=%d: %s; using trivial partition.", t, exc)
            partition = [[i] for i in range(N)]

        # ── Cluster metrics ───────────────────────────────────────────────────
        # Mean weighted density
        wd_vals = []
        for cluster in partition:
            if len(cluster) >= 2:
                wd_vals.append(weighted_density(A, cluster))
        mean_wd = float(np.mean(wd_vals)) if wd_vals else 0.0
        wds.append(mean_wd)

        # Global partition value
        gp_val = global_partition_value(A, partition)
        cluster_vals.append(gp_val)

        # Jaccard similarity with previous partition
        if prev_partition is not None:
            jac = inter_window_jaccard(prev_partition, partition)
        else:
            jac = float("nan")
        jaccard_sims.append(jac)
        prev_partition = partition

        # ── Portfolio construction ────────────────────────────────────────────
        cluster_weights = []
        for cluster in partition:
            if len(cluster) < 2:
                cluster_weights.append(None)
                continue
            winners, losers = compute_winner_loser(
                res_window, cluster, winner_loser_threshold
            )
            w = build_cluster_portfolio(winners, losers)
            cluster_weights.append(w)

        portfolio_weights = aggregate_cluster_portfolios(cluster_weights, N)

        # ── Holding period ────────────────────────────────────────────────────
        period_returns: List[float] = []
        cumulative_pnl = 0.0

        for h in range(holding_days):
            if t + h >= n_dates:
                break
            day_rets = total_returns.iloc[t + h].values
            # Replace NaN with 0
            day_rets = np.nan_to_num(day_rets, nan=0.0)
            day_portfolio_ret = float(portfolio_weights @ day_rets)
            period_returns.append(day_portfolio_ret)
            cumulative_pnl += day_portfolio_ret

            # Stop-win check
            if cumulative_pnl >= stop_win:
                logger.debug("Stop-win triggered at h=%d, cumPnL=%.4f", h, cumulative_pnl)
                break

        all_daily_returns.extend(period_returns)
        all_dates.extend([trading_dates[t + h] for h in range(len(period_returns))])
        all_partitions.append(partition)

        # Advance by holding_days (3 trading days)
        t += holding_days

    daily_ret_arr = np.array(all_daily_returns)
    cumulative_pnl = np.cumprod(1 + daily_ret_arr) - 1 if len(daily_ret_arr) > 0 else np.array([0.0])
    annual_return = float(cumulative_pnl[-1]) if len(cumulative_pnl) > 0 else 0.0

    return {
        "cumulative_pnl": cumulative_pnl,
        "daily_returns": daily_ret_arr,
        "dates": all_dates,
        "partitions": all_partitions,
        "jaccard_similarities": jaccard_sims,
        "wds": wds,
        "cluster_values": cluster_vals,
        "annual_return": annual_return,
    }


def run_multiple_runs(
    residuals: pd.DataFrame,
    total_returns: pd.DataFrame,
    market_returns: pd.Series,
    partition_fn_factory,
    n_runs: int,
    base_seed: int = 0,
    **backtest_kwargs,
) -> List[Dict]:
    """
    Run backtest multiple times with different seeds.

    Parameters
    ----------
    residuals, total_returns, market_returns : DataFrames/Series
        Market data.
    partition_fn_factory : callable
        Function(seed) -> partition_fn. Creates a new partition function with given seed.
    n_runs : int
        Number of independent runs.
    base_seed : int
        Base seed; run i uses seed = base_seed + i.

    Returns
    -------
    List[Dict]
        List of backtest result dicts.
    """
    results = []
    for run in range(n_runs):
        seed = base_seed + run
        logger.info("Run %d/%d (seed=%d)", run + 1, n_runs, seed)
        partition_fn = partition_fn_factory(seed)
        result = run_backtest(
            residuals=residuals,
            total_returns=total_returns,
            market_returns=market_returns,
            partition_fn=partition_fn,
            seed=seed,
            **backtest_kwargs,
        )
        results.append(result)
    return results
