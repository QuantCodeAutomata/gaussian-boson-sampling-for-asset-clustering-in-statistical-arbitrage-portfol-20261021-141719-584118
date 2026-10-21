"""Tests for statistical arbitrage strategy."""
import numpy as np
import pandas as pd
import pytest
from src.statarb import (
    compute_winner_loser, build_cluster_portfolio, aggregate_cluster_portfolios,
)


def test_winner_loser_basic():
    """Stocks above cluster mean → winners, below → losers."""
    dates = pd.date_range("2020-01-02", periods=5, freq="B")
    # Stock 0 positive, stock 1 negative residuals (relative to mean)
    data = np.array([[0.01, -0.01], [0.02, -0.02], [0.01, -0.01],
                     [0.02, -0.02], [0.01, -0.01]])
    res = pd.DataFrame(data, index=dates, columns=["S0", "S1"])
    winners, losers = compute_winner_loser(res, [0, 1])
    assert 0 in winners and 1 in losers, f"Expected 0=winner, 1=loser, got {winners}, {losers}"

def test_winner_loser_single_stock():
    """Single stock: no winners and losers (both legs need stocks)."""
    dates = pd.date_range("2020-01-02", periods=5, freq="B")
    res = pd.DataFrame([[0.01], [0.02], [0.01], [0.02], [0.01]],
                       index=dates, columns=["S0"])
    winners, losers = compute_winner_loser(res, [0])
    assert winners == [] and losers == []

def test_build_cluster_portfolio_zero_cost():
    """Portfolio is market-neutral: sum of positive weights = sum of |negative weights| = 1."""
    w = build_cluster_portfolio([0], [1, 2])
    assert w is not None
    pos_sum = sum(v for v in w.values() if v > 0)
    neg_sum = sum(abs(v) for v in w.values() if v < 0)
    assert abs(pos_sum - 1.0) < 1e-10, f"Long sum not 1: {pos_sum}"
    assert abs(neg_sum - 1.0) < 1e-10, f"Short sum not 1: {neg_sum}"

def test_build_cluster_portfolio_equal_weights():
    """Equal weights within each leg."""
    w = build_cluster_portfolio([0, 1], [2, 3])  # 2 shorts, 2 longs
    assert abs(w[0] - w[1]) < 1e-10, "Shorts not equal"
    assert abs(w[2] - w[3]) < 1e-10, "Longs not equal"
    assert abs(w[2] - 0.5) < 1e-10

def test_build_cluster_portfolio_missing_leg():
    """Returns None if either leg is empty."""
    assert build_cluster_portfolio([], [0, 1]) is None
    assert build_cluster_portfolio([0, 1], []) is None

def test_aggregate_cluster_portfolios_market_neutral():
    """Aggregate portfolio: long weights sum ≈ 1, short weights sum ≈ -1."""
    w1 = {0: 0.5, 1: 0.5, 2: -1.0}
    w2 = {3: 1.0, 4: -0.5, 5: -0.5}
    agg = aggregate_cluster_portfolios([w1, w2], 6)
    long_sum  = float(np.sum(agg[agg > 0]))
    short_sum = float(np.sum(agg[agg < 0]))
    assert abs(long_sum  - 1.0) < 1e-6, f"Long sum: {long_sum}"
    assert abs(short_sum + 1.0) < 1e-6, f"Short sum: {short_sum}"

def test_aggregate_all_none():
    """All-None cluster weights → zero portfolio."""
    agg = aggregate_cluster_portfolios([None, None], 4)
    assert np.all(agg == 0)
