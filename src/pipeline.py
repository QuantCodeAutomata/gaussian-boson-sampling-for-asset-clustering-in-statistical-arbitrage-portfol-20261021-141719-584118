"""
Common experiment pipeline utilities shared across all experiments.
Paper: "Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios"
"""
import logging, math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.gbs import gbs_boost_partition, gbs_roots_partition, qic_gbs_partition, compute_H
from src.clustering import spectral_clustering, sponge_clustering, random_clustering
from src.data import (
    load_prices, load_fama_french_market, compute_total_returns,
    compute_rolling_beta, compute_residual_returns,
    compute_residual_correlation, clean_correlation_matrix,
    build_gbs_graph, estimate_k_from_eigenspacing, get_sp500_universe,
)
from src.statarb import run_backtest
from src.metrics import compute_metrics, summarise_runs, welch_t_test

logger = logging.getLogger(__name__)


def make_partition_fn(method: str, cfg: ExperimentConfig, seed: int, loss_rate: float = 0.0,
                      K_override: Optional[int] = None) -> Callable:
    """Create a partition function for a given method, seed, and optional fixed K."""
    rng = np.random.default_rng(seed)

    def _get_K(N):
        return K_override if K_override is not None else max(2, int(math.sqrt(N)))

    if method == "gbs_boost":
        def fn(A, r):
            N = A.shape[0]; H = compute_H(cfg.F_boost, N)
            return gbs_boost_partition(A, H, math.sqrt(N), r, loss_rate)
    elif method == "gbs_roots":
        def fn(A, r):
            N = A.shape[0]; H = compute_H(cfg.F_roots, N)
            return gbs_roots_partition(A, H, math.sqrt(N), r, loss_rate)
    elif method == "qic_gbs":
        def fn(A, r):
            N = A.shape[0]; H = compute_H(cfg.F_qic, N)
            return qic_gbs_partition(A, H, math.sqrt(N), r, loss_rate)
    elif method == "spectral":
        def fn(A, r):
            N = A.shape[0]; C = np.maximum(0, A) + np.eye(N)
            return spectral_clustering(C, _get_K(N), r)
    elif method == "sponge":
        def fn(A, r):
            N = A.shape[0]; C = A + np.eye(N)
            return sponge_clustering(C, _get_K(N), r)
    elif method == "random":
        def fn(A, r):
            N = A.shape[0]
            return random_clustering(N, _get_K(N), r)
    else:
        raise ValueError(f"Unknown method: {method}")
    return fn


def estimate_K_for_method(method: str, residuals_20d: pd.DataFrame,
                            cfg: ExperimentConfig, seed: int) -> int:
    """Estimate K via eigenvalue spacing permutation test for classical methods."""
    if method not in ("spectral", "sponge"):
        return 2
    try:
        return estimate_k_from_eigenspacing(
            residuals_20d.values, B=cfg.k_perm_replicates,
            alpha=cfg.k_alpha, rng=np.random.default_rng(seed + 9999))
    except Exception as e:
        logger.warning("K estimation failed: %s", e)
        return 2


def load_and_prepare_data(cfg: ExperimentConfig, universe_seed: int = 42
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """Load market data, compute betas and residuals, trim to eval year."""
    eval_year = cfg.eval_year
    load_start = f"{eval_year - 1}-06-01"
    load_end   = f"{eval_year}-12-31"

    tickers = get_sp500_universe(cfg.n_stocks, eval_year, seed=universe_seed)
    logger.info("Loading %d tickers from %s to %s", len(tickers), load_start, load_end)

    prices = load_prices(tickers, load_start, load_end)
    tickers = [t for t in tickers if t in prices.columns]

    mkt = load_fama_french_market(load_start, load_end)
    ret = compute_total_returns(prices)
    betas = compute_rolling_beta(ret, mkt, window=cfg.beta_window)
    res = compute_residual_returns(ret, mkt, betas)

    # Trim: keep lookback + eval year
    lookback = cfg.beta_window + cfg.corr_window + 10
    all_idx = res.index
    eval_mask = (all_idx >= f"{eval_year}-01-01") & (all_idx <= f"{eval_year}-12-31")
    eval_idx = all_idx[eval_mask]
    if len(eval_idx) == 0:
        raise ValueError(f"No data for year {eval_year}")
    first = all_idx.get_loc(eval_idx[0])
    start = max(0, first - lookback)

    res  = res.iloc[start:]
    ret  = ret.iloc[start:]

    # Drop mostly-NaN tickers
    good = res.notna().mean() > 0.8
    tickers_good = res.columns[good].tolist()
    res  = res[tickers_good].fillna(0.0)
    ret  = ret[tickers_good].fillna(0.0)

    actual_n = min(cfg.n_stocks, len(tickers_good))
    res  = res.iloc[:, :actual_n]
    ret  = ret.iloc[:, :actual_n]

    logger.info("Prepared: %d stocks, %d rows", res.shape[1], len(res))
    return res, ret, mkt, tickers_good[:actual_n]


def run_all_methods(cfg: ExperimentConfig, residuals: pd.DataFrame,
                    total_returns: pd.DataFrame, market_returns: pd.Series,
                    methods: List[str], n_runs_override: Optional[Dict[str, int]] = None,
                    loss_rate: float = 0.0) -> Dict[str, List[Dict]]:
    """Run all specified methods and collect per-run results."""
    GBS_METHODS = {"gbs_boost", "gbs_roots", "qic_gbs"}
    results: Dict[str, List[Dict]] = {}

    # 20-day window for K estimation
    k_res = residuals.iloc[:cfg.k_est_window] if len(residuals) >= cfg.k_est_window else residuals

    for method in methods:
        n_runs = (n_runs_override or {}).get(method,
                  cfg.n_runs_gbs if method in GBS_METHODS else cfg.n_runs_classical)
        logger.info("Method=%s  runs=%d  loss=%.2f", method, n_runs, loss_rate)

        runs = []
        for i in range(n_runs):
            seed = i * 1000 + abs(hash(method)) % 1000
            K = estimate_K_for_method(method, k_res, cfg, seed)
            pfn = make_partition_fn(method, cfg, seed, loss_rate, K_override=K if method in ("spectral","sponge") else None)
            try:
                r = run_backtest(residuals=residuals, total_returns=total_returns,
                                 market_returns=market_returns, partition_fn=pfn,
                                 corr_window=cfg.corr_window, beta_window=cfg.beta_window,
                                 k_est_window=cfg.k_est_window, holding_days=cfg.holding_days,
                                 stop_win=cfg.stop_win,
                                 winner_loser_threshold=cfg.winner_loser_threshold, seed=seed)
                r["metrics"] = compute_metrics(r["daily_returns"], r["wds"],
                                               r["cluster_values"], r["jaccard_similarities"],
                                               cfg.risk_free_rate, cfg.ann_factor)
                runs.append(r)
            except Exception as e:
                logger.error("Run %d method %s: %s", i, method, e)
        results[method] = runs

        if runs:
            arr = [r["metrics"].get("annual_return", 0) for r in runs]
            shr = [r["metrics"].get("sharpe_ratio", 0) for r in runs]
            logger.info("  -> return=%.3f±%.3f  sharpe=%.3f±%.3f",
                        np.mean(arr), np.std(arr), np.mean(shr), np.std(shr))
    return results


def compile_summary_table(all_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Compile mean±std summary across runs for all methods."""
    rows = []
    for method, runs in all_results.items():
        if not runs: continue
        sm = summarise_runs([r["metrics"] for r in runs if "metrics" in r])
        row = {"method": method}
        for k, v in sm.items():
            row[f"{k}_mean"] = v["mean"]
            row[f"{k}_std"]  = v["std"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("method") if rows else pd.DataFrame()
