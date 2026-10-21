"""Tests for data loading and preprocessing utilities (without live downloads)."""
import numpy as np
import pandas as pd
import pytest
from src.data import (
    compute_total_returns, compute_rolling_beta, compute_residual_returns,
    compute_residual_correlation, clean_correlation_matrix, build_gbs_graph,
    estimate_k_from_eigenspacing, get_sp500_universe,
)


def make_prices(n_stocks=5, n_days=100):
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, (n_days, n_stocks))
    prices = 100 * np.cumprod(1 + returns, axis=0)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=dates, columns=[f"S{i}" for i in range(n_stocks)])


def test_compute_total_returns_shape():
    prices = make_prices()
    ret = compute_total_returns(prices)
    assert ret.shape == prices.shape

def test_compute_total_returns_first_row_nan():
    prices = make_prices()
    ret = compute_total_returns(prices)
    assert ret.iloc[0].isna().all()

def test_compute_rolling_beta_shape():
    prices = make_prices()
    ret = compute_total_returns(prices).fillna(0)
    mkt = ret.mean(axis=1)
    betas = compute_rolling_beta(ret, mkt, window=20)
    assert betas.shape == ret.shape

def test_compute_residual_returns_shape():
    prices = make_prices()
    ret = compute_total_returns(prices).fillna(0)
    mkt = ret.mean(axis=1)
    betas = compute_rolling_beta(ret, mkt, window=20)
    res = compute_residual_returns(ret, mkt, betas)
    assert res.shape == ret.shape

def test_compute_residual_correlation_symmetric():
    data = np.random.default_rng(0).normal(0, 0.01, (10, 5))
    C = compute_residual_correlation(
        pd.DataFrame(data, columns=[f"S{i}" for i in range(5)]), window=10)
    assert C.shape == (5, 5)
    assert np.allclose(C, C.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(C), np.ones(5), atol=1e-6)

def test_build_gbs_graph_properties():
    C = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.4], [-0.3, 0.4, 1.0]])
    A = build_gbs_graph(C)
    assert A.shape == (3, 3)
    assert np.all(A >= 0), "A must be non-negative"
    assert np.allclose(A, A.T, atol=1e-10), "A must be symmetric"
    assert np.allclose(np.diag(A), 0.0), "Diagonal must be 0"
    # Negative correlations → 0 in A
    assert A[0, 2] == 0.0

def test_clean_correlation_matrix_preserves_symmetry():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (20, 8))
    C = np.corrcoef(data.T)
    Cc = clean_correlation_matrix(C)
    assert np.allclose(Cc, Cc.T, atol=1e-10)
    assert Cc.shape == C.shape

def test_clean_correlation_matrix_diagonal_one():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (20, 6))
    C = np.corrcoef(data.T)
    Cc = clean_correlation_matrix(C)
    np.testing.assert_allclose(np.diag(Cc), np.ones(6), atol=1e-6)

def test_estimate_k_from_eigenspacing_returns_int():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.01, (20, 8))
    K = estimate_k_from_eigenspacing(data, B=20, rng=rng)
    assert isinstance(K, int)
    assert K >= 2

def test_get_sp500_universe_size():
    tickers = get_sp500_universe(30, 2020, seed=42)
    assert len(tickers) == 30

def test_get_sp500_universe_no_duplicates():
    tickers = get_sp500_universe(50, 2020)
    assert len(tickers) == len(set(tickers))
