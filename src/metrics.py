"""
Performance metrics for GBS Statistical Arbitrage experiments.

Paper: "Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios"

ASSUMPTION [risk-free-rate]: Annual risk-free rate = 0 (not specified in paper).
ASSUMPTION [annualisation]: 252 trading days per year (standard; paper unspecified).
ASSUMPTION [downside-deviation]: Target return = 0 for Sortino ratio.
ASSUMPTION [sharpe]: Using simple mean/std of daily returns × sqrt(252); annualised Sharpe.
"""

from typing import Dict, List, Optional

import numpy as np
import scipy.stats as stats


def annual_return(daily_returns: np.ndarray) -> float:
    """
    Compute total (compound) return over the evaluation period.

    R_total = prod(1 + r_t) - 1

    Parameters
    ----------
    daily_returns : np.ndarray
        Daily portfolio returns.

    Returns
    -------
    float
        Compound total return.
    """
    if len(daily_returns) == 0:
        return 0.0
    return float(np.prod(1 + daily_returns) - 1)


def sharpe_ratio(
    daily_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    ann_factor: int = 252,
) -> float:
    """
    Compute annualised Sharpe ratio.

    Sharpe = (mean(r) - rf/ann) / std(r) * sqrt(ann)

    ASSUMPTION [annualisation]: 252 trading days.
    ASSUMPTION [rf=0]: risk-free rate zero (paper unspecified).

    Parameters
    ----------
    daily_returns : np.ndarray
        Daily portfolio returns.
    risk_free_rate : float
        Annual risk-free rate.
    ann_factor : int
        Annualisation factor (252 trading days).

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / ann_factor
    excess = daily_returns - daily_rf
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(ann_factor))


def sortino_ratio(
    daily_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    ann_factor: int = 252,
) -> float:
    """
    Compute annualised Sortino ratio.

    Sortino = (mean(r) - rf/ann) / downside_std(r) * sqrt(ann)

    ASSUMPTION [downside]: target return = 0 (paper unspecified).

    Parameters
    ----------
    daily_returns : np.ndarray
        Daily portfolio returns.
    risk_free_rate : float
        Annual risk-free rate.
    ann_factor : int
        Annualisation factor.

    Returns
    -------
    float
        Annualised Sortino ratio.
    """
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / ann_factor
    excess = daily_returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < 1e-12:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(ann_factor))


def compute_metrics(
    daily_returns: np.ndarray,
    wds: Optional[List[float]] = None,
    cluster_values: Optional[List[float]] = None,
    jaccard_sims: Optional[List[float]] = None,
    risk_free_rate: float = 0.0,
    ann_factor: int = 252,
) -> Dict[str, float]:
    """
    Compute all reported performance metrics for one backtest run.

    Parameters
    ----------
    daily_returns : np.ndarray
        Daily portfolio returns.
    wds : List[float], optional
        Mean weighted density per rebalance period.
    cluster_values : List[float], optional
        Global partition value per rebalance period.
    jaccard_sims : List[float], optional
        Inter-window Jaccard similarity per transition.
    risk_free_rate : float
        Annual risk-free rate.
    ann_factor : int
        Trading days per year.

    Returns
    -------
    Dict[str, float]
        Dictionary of metric name → value.
    """
    metrics: Dict[str, float] = {
        "annual_return": annual_return(daily_returns),
        "sharpe_ratio": sharpe_ratio(daily_returns, risk_free_rate, ann_factor),
        "sortino_ratio": sortino_ratio(daily_returns, risk_free_rate, ann_factor),
    }

    if wds is not None and len(wds) > 0:
        metrics["mean_weighted_density"] = float(np.nanmean(wds))
        metrics["std_weighted_density"] = float(np.nanstd(wds))

    if cluster_values is not None and len(cluster_values) > 0:
        metrics["mean_cluster_value"] = float(np.nanmean(cluster_values))

    if jaccard_sims is not None and len(jaccard_sims) > 0:
        valid_jac = [j for j in jaccard_sims if not np.isnan(j)]
        metrics["mean_jaccard"] = float(np.mean(valid_jac)) if valid_jac else float("nan")

    return metrics


def summarise_runs(
    run_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean ± std across multiple runs for each metric.

    Parameters
    ----------
    run_metrics : List[Dict[str, float]]
        Per-run metric dictionaries.

    Returns
    -------
    Dict[str, Dict[str, float]]
        {metric_name: {"mean": ..., "std": ...}}
    """
    if not run_metrics:
        return {}

    all_keys = set().union(*[m.keys() for m in run_metrics])
    summary: Dict[str, Dict[str, float]] = {}

    for key in sorted(all_keys):
        values = [m[key] for m in run_metrics if key in m and not np.isnan(m.get(key, float("nan")))]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
                "n": len(values),
            }

    return summary


def welch_t_test(
    returns_a: List[float],
    returns_b: List[float],
) -> Dict[str, float]:
    """
    Conduct two-sided Welch's t-test on terminal returns.

    Paper: "Conduct two-sided Welch tests on terminal returns ... using a 5% significance level."
    Uses scipy.stats.ttest_ind(..., equal_var=False).

    Parameters
    ----------
    returns_a : List[float]
        Terminal returns from method A.
    returns_b : List[float]
        Terminal returns from method B.

    Returns
    -------
    Dict[str, float]
        {"t_stat": ..., "p_value": ..., "significant": bool_as_int}
    """
    t_stat, p_value = stats.ttest_ind(returns_a, returns_b, equal_var=False)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": int(p_value < 0.05),
    }
