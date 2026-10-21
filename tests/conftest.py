"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def small_corr_matrix():
    """4x4 positive-definite correlation matrix."""
    rng = np.random.default_rng(42)
    A_raw = rng.uniform(0.1, 0.5, (4, 4))
    A_sym = (A_raw + A_raw.T) / 2
    np.fill_diagonal(A_sym, 1.0)
    return A_sym

@pytest.fixture
def medium_corr_matrix():
    """10x10 positive-definite correlation matrix."""
    rng = np.random.default_rng(42)
    A = rng.uniform(0.0, 0.4, (10, 10))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 1.0)
    return A

@pytest.fixture
def small_gbs_graph(small_corr_matrix):
    """4-node GBS adjacency matrix."""
    from src.gbs import weighted_density
    A = np.maximum(0.0, small_corr_matrix - np.eye(4))
    return A

@pytest.fixture
def sample_residuals():
    """Small residual-return DataFrame for testing."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-02", periods=80, freq="B")
    data = rng.normal(0, 0.01, (80, 8))
    return pd.DataFrame(data, index=dates, columns=[f"S{i}" for i in range(8)])

@pytest.fixture
def sample_total_returns(sample_residuals):
    """Total returns matching residuals (for simplicity, same values)."""
    return sample_residuals.copy()
