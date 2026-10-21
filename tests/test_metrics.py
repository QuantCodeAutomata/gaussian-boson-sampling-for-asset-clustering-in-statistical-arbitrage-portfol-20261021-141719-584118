"""Tests for performance metrics."""
import numpy as np
import pytest
from src.metrics import (
    annual_return, sharpe_ratio, sortino_ratio, compute_metrics,
    summarise_runs, welch_t_test,
)


def test_annual_return_zero_returns():
    assert annual_return(np.zeros(10)) == 0.0

def test_annual_return_compound():
    """(1+r)^2 - 1 for constant daily return r."""
    r = 0.01
    ret = np.full(10, r)
    expected = (1 + r) ** 10 - 1
    assert abs(annual_return(ret) - expected) < 1e-10

def test_annual_return_empty():
    assert annual_return(np.array([])) == 0.0

def test_sharpe_ratio_constant_returns():
    """Zero std → zero Sharpe."""
    ret = np.full(10, 0.001)
    sr = sharpe_ratio(ret)
    assert sr == 0.0

def test_sharpe_ratio_positive():
    """Positive mean returns with noise → positive Sharpe."""
    rng = np.random.default_rng(0)
    ret = rng.normal(0.001, 0.005, 252)
    sr = sharpe_ratio(ret)
    assert sr > 0  # expect positive with positive mean

def test_sharpe_ratio_range():
    """Sharpe should be finite for normal returns."""
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0005, 0.01, 252)
    sr = sharpe_ratio(ret)
    assert np.isfinite(sr)

def test_sortino_no_downside():
    """All positive returns → inf Sortino (or large)."""
    ret = np.full(10, 0.001)
    so = sortino_ratio(ret)
    assert so == float("inf") or so > 0

def test_sortino_mixed():
    rng = np.random.default_rng(42)
    ret = rng.normal(0.001, 0.01, 100)
    so = sortino_ratio(ret)
    assert np.isfinite(so)

def test_compute_metrics_keys():
    ret = np.random.default_rng(0).normal(0.001, 0.005, 100)
    m = compute_metrics(ret, [0.3, 0.4], [1.0, 1.5], [0.5, 0.6])
    assert "annual_return" in m
    assert "sharpe_ratio" in m
    assert "sortino_ratio" in m
    assert "mean_weighted_density" in m
    assert "mean_jaccard" in m

def test_summarise_runs():
    runs = [{"annual_return": 0.2, "sharpe_ratio": 1.5},
            {"annual_return": 0.3, "sharpe_ratio": 2.0}]
    s = summarise_runs(runs)
    assert abs(s["annual_return"]["mean"] - 0.25) < 1e-10
    assert abs(s["sharpe_ratio"]["mean"] - 1.75) < 1e-10

def test_welch_t_test_identical():
    """Identical distributions → p≈1 (not significant)."""
    a = [0.1, 0.2, 0.15, 0.18, 0.12]
    result = welch_t_test(a, a)
    assert result["p_value"] >= 0.05  # not significant (same distribution trivially)
    assert "t_stat" in result

def test_welch_t_test_different():
    """Very different distributions → likely significant."""
    a = [0.1, 0.11, 0.09, 0.105, 0.095]
    b = [0.5, 0.48, 0.52, 0.50, 0.49]
    result = welch_t_test(a, b)
    assert result["p_value"] < 0.01  # should be highly significant
