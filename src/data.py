"""
Data loading and preprocessing for GBS Statistical Arbitrage experiments.

Paper: "Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios"

Data source note:
  The paper uses CRSP via Wharton Research Data Services (proprietary).
  This module uses yfinance as a publicly available substitute and documents
  all differences explicitly.

ASSUMPTION [data-source]: yfinance dividend-adjusted close prices replace CRSP.
ASSUMPTION [universe]: S&P 500 constituents from Wikipedia snapshot; exact paper
  universe (stratified by previous-year top performers per GICS sector) is
  approximated using available sector information.
ASSUMPTION [fama-french]: Fama-French daily market factors from Ken French's
  data library via pandas_datareader.
"""

import os
import time
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


# ── S&P 500 constituents with approximate GICS sectors ──────────────────────

# Pre-defined stratified universe tickers per sector (2020 universe approximation)
# ASSUMPTION [universe-fixed]: Using a fixed representative set of S&P 500 stocks
# because exact paper constituents are unavailable without CRSP.
SECTOR_TICKERS: Dict[str, List[str]] = {
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OKE", "KMI",
               "HAL", "DVN", "BKR", "FANG", "PXD"],
    "Materials": ["LIN", "APD", "ECL", "SHW", "NEM", "FCX", "NUE", "VMC", "MLM", "PPG",
                  "ALB", "CE", "CF", "MOS", "IFF"],
    "Industrials": ["HON", "UPS", "CAT", "GE", "MMM", "BA", "RTX", "LMT", "NOC", "CSX",
                    "NSC", "FDX", "UNP", "DE", "EMR", "ROK", "ITW", "PH", "ETN", "CARR"],
    "ConsumerDiscretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",
                               "GM", "F", "EBAY", "APTV", "DHI", "LEN", "PHM"],
    "ConsumerStaples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MO", "PM", "GIS", "K",
                         "SYY", "HRL", "CAG", "CPB", "MKC"],
    "HealthCare": ["JNJ", "UNH", "PFE", "ABT", "TMO", "MRK", "ABBV", "BMY", "LLY",
                    "AMGN", "GILD", "CI", "CVS", "HUM", "BIIB"],
    "Financials": ["BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC",
                    "AXP", "BLK", "SPGI", "ICE", "CME", "TRV"],
    "InformationTechnology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ACN",
                               "AMD", "TXN", "INTC", "QCOM", "IBM", "NOW", "AMAT", "LRCX"],
    "CommunicationServices": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T",
                               "ATVI", "EA", "TTWO", "TMUS", "CHTR", "DISH", "NWS", "FOX"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "XEL", "EIX", "WEC", "ES",
                   "PEG", "ETR", "ED", "FE", "AES"],
    "RealEstate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "WELL", "DLR", "AVB", "EQR",
                    "SPG", "O", "VICI", "IRM", "WY", "UDR"],
}

# All tickers (130 candidates for subsetting)
ALL_TICKERS: List[str] = [t for tickers in SECTOR_TICKERS.values() for t in tickers]


def get_sp500_universe(
    n_stocks: int,
    eval_year: int,
    seed: int = 42,
) -> List[str]:
    """
    Build stratified N-stock universe from S&P 500, preserving GICS sector proportions.

    ASSUMPTION [stratification]: We allocate stocks proportionally across 11 sectors
    and select them deterministically (seed-based) from predefined lists.
    The paper builds the universe from the previous year's top performers; we
    use a fixed representative set as an approximation.

    Parameters
    ----------
    n_stocks : int
        Target universe size N.
    eval_year : int
        Evaluation year (used to note assumption).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[str]
        Sorted list of ticker symbols.
    """
    rng = np.random.default_rng(seed)
    n_sectors = len(SECTOR_TICKERS)
    base_per_sector = n_stocks // n_sectors
    remainder = n_stocks - base_per_sector * n_sectors

    selected: List[str] = []
    for i, (sector, tickers) in enumerate(SECTOR_TICKERS.items()):
        quota = base_per_sector + (1 if i < remainder else 0)
        quota = min(quota, len(tickers))
        chosen = rng.choice(tickers, size=quota, replace=False).tolist()
        selected.extend(chosen)

    logger.info(
        "Universe: %d stocks for year %d (stratified; CRSP not available)",
        len(selected), eval_year,
    )
    return sorted(selected)


def load_prices(
    tickers: List[str],
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Download dividend-adjusted daily close prices using yfinance.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols.
    start_date : str
        Start date in 'YYYY-MM-DD' format (inclusive).
    end_date : str
        End date in 'YYYY-MM-DD' format (inclusive).
    max_retries : int
        Number of download retries on failure.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (days, n_tickers), dtype float64, DatetimeIndex.
        Missing values are forward-filled then backward-filled.
    """
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,   # dividend-adjusted closes (approximates CRSP total-return)
                progress=False,
            )
            break
        except Exception as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"yfinance download failed after {max_retries} attempts: {exc}")
            time.sleep(2 ** attempt)

    # Extract 'Close' column — yfinance ≥0.2 returns MultiIndex (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        # MultiIndex: first level is Price, second is Ticker
        prices = raw["Close"].copy()
        # yfinance may name the column axis "Ticker" — rename to plain column names
        prices.columns.name = None
    else:
        # Single-ticker: flat columns
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    # Keep only requested tickers that were returned
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available]  # ensure consistent ordering
    prices = prices.sort_index()

    # Handle missing data: forward-fill then backward-fill
    n_missing_before = prices.isna().sum().sum()
    prices = prices.ffill().bfill()
    n_missing_after = prices.isna().sum().sum()

    # Drop tickers with >20% missing after fill
    completeness = 1 - prices.isna().mean()
    good_tickers = completeness[completeness >= 0.8].index.tolist()
    prices = prices[good_tickers]

    logger.info(
        "Prices loaded: %d days × %d tickers, %d missing filled, %d dropped",
        len(prices), len(good_tickers),
        n_missing_before - n_missing_after,
        len(tickers) - len(good_tickers),
    )
    return prices


def load_fama_french_market(start_date: str, end_date: str) -> pd.Series:
    """
    Load Fama-French daily excess market return (Mkt-RF) from Ken French's library.

    Returns
    -------
    pd.Series
        Daily excess market return (decimal, not percent), DatetimeIndex.
    """
    try:
        ff = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                            start=start_date, end=end_date)[0]
        mkt_rf = ff["Mkt-RF"] / 100.0  # convert from percent to decimal
        # PeriodIndex must be converted to Timestamp before pd.to_datetime (pandas>=2)
        if hasattr(mkt_rf.index, "to_timestamp"):
            mkt_rf.index = mkt_rf.index.to_timestamp()
        else:
            mkt_rf.index = pd.to_datetime(mkt_rf.index)
        return mkt_rf
    except Exception as exc:
        logger.warning("Fama-French data unavailable (%s); using SPY proxy.", exc)
        # Fall back to SPY as market proxy
        spy_raw = yf.download("SPY", start=start_date, end=end_date,
                              auto_adjust=True, progress=False)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy = spy_raw["Close"].iloc[:, 0]
        else:
            spy = spy_raw["Close"]
        spy_ret = spy.pct_change().dropna()
        spy_ret.index = pd.to_datetime(spy_ret.index)
        spy_ret.name = "mkt"
        return spy_ret


def compute_total_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily total returns: R[t,i] = P[t,i] / P[t-1,i] - 1.

    ASSUMPTION [dividends]: yfinance adjusted close already incorporates dividends,
    so total return ≈ pct_change() on adjusted close.

    Parameters
    ----------
    prices : pd.DataFrame
        Dividend-adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Daily total returns, same shape as prices (first row = NaN).
    """
    return prices.pct_change()


def compute_rolling_beta(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """
    Estimate rolling single-factor market beta over a trailing window.

    beta_i = Cov(R_i, R_mkt) / Var(R_mkt)

    Parameters
    ----------
    returns : pd.DataFrame
        Stock total returns, shape (T, N).
    market_returns : pd.Series
        Market factor returns, length T.
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.DataFrame
        Rolling beta estimates, same shape as returns.
    """
    aligned_mkt = market_returns.reindex(returns.index).ffill().bfill()
    # Use numpy array for fast in-place assignment (avoids pandas iloc shape issues)
    n = len(returns)
    n_stocks = returns.shape[1]
    betas_arr = np.full((n, n_stocks), np.nan)

    mkt_arr = aligned_mkt.values.astype(float)
    ret_arr = returns.values.astype(float)

    for t in range(window, n):
        mkt_w = mkt_arr[t - window:t]
        ret_w = ret_arr[t - window:t, :]
        mkt_var = np.var(mkt_w, ddof=1)
        if mkt_var < 1e-12:
            betas_arr[t] = 0.0
            continue
        mkt_demeaned = mkt_w - mkt_w.mean()
        ret_demeaned = ret_w - ret_w.mean(axis=0)
        cov_im = (ret_demeaned * mkt_demeaned[:, None]).mean(axis=0)
        betas_arr[t] = cov_im / mkt_var

    return pd.DataFrame(betas_arr, index=returns.index, columns=returns.columns)


def compute_residual_returns(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    betas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute market residual returns: R_res[t,i] = R[t,i] - beta_i * R_mkt[t].

    Parameters
    ----------
    returns : pd.DataFrame
        Stock total returns.
    market_returns : pd.Series
        Market factor returns.
    betas : pd.DataFrame
        Rolling beta estimates.

    Returns
    -------
    pd.DataFrame
        Residual returns, same shape as returns.
    """
    aligned_mkt = market_returns.reindex(returns.index).ffill().bfill()
    mkt_matrix = pd.DataFrame(
        np.outer(aligned_mkt.values, np.ones(returns.shape[1])),
        index=returns.index, columns=returns.columns,
    )
    residuals = returns - betas * mkt_matrix
    return residuals


def compute_residual_correlation(
    residuals: pd.DataFrame,
    window: int = 5,
) -> np.ndarray:
    """
    Compute N×N Pearson correlation matrix from trailing window of residual returns.

    Parameters
    ----------
    residuals : pd.DataFrame
        Residual returns.
    window : int
        Trailing window length.

    Returns
    -------
    np.ndarray
        Correlation matrix, shape (N, N). NaN-safe; NaN replaced by 0.
    """
    assert len(residuals) >= window, (
        f"Insufficient data: {len(residuals)} rows < window {window}"
    )
    recent = residuals.iloc[-window:].values  # shape (window, N)
    # Replace NaN with column mean
    col_means = np.nanmean(recent, axis=0)
    for j in range(recent.shape[1]):
        mask = np.isnan(recent[:, j])
        recent[mask, j] = col_means[j]

    corr = np.corrcoef(recent.T)  # shape (N, N)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def clean_correlation_matrix(
    C: np.ndarray,
    residuals_20d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply Random Matrix Theory (RMT) cleaning to the correlation matrix.

    Removes the largest market eigenmode and random-noise eigenmodes (Marchenko-Pastur
    boundary), retaining statistically significant principal components.

    Paper reference: Appendix A describes removing the market mode and MP noise band.

    Parameters
    ----------
    C : np.ndarray
        Raw correlation matrix, shape (N, N).
    residuals_20d : np.ndarray, optional
        Residual returns over 20-day window, shape (20, N). Not used in
        5-day cleaning (rank-deficient), but provided for reference.

    Returns
    -------
    np.ndarray
        Cleaned correlation matrix, shape (N, N).
    """
    N = C.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Marchenko-Pastur upper bound: lambda_max = (1 + sqrt(N/T))^2
    # For 5-day window with N stocks: T=5, very rank-deficient
    # For 20-day window: T=20
    T = residuals_20d.shape[0] if residuals_20d is not None else 5
    q = N / T  # ratio
    # For q > 1 (rank deficient), almost all eigenvalues are noise
    if q <= 1:
        lambda_mp_max = (1 + np.sqrt(q)) ** 2
    else:
        # Use all eigenvalues up to rank (T-1)
        lambda_mp_max = (1 + np.sqrt(1.0 / q)) ** 2 * q  # scaled

    # Remove largest eigenmode (market mode = top eigenvector)
    # Remove eigenvalues within MP noise band
    C_cleaned = np.zeros((N, N))
    for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        if i == 0:
            continue  # skip market mode
        if lam <= lambda_mp_max:
            continue  # skip noise eigenvalues
        C_cleaned += lam * np.outer(vec, vec)

    # Restore diagonal to 1 (correlation matrix convention)
    np.fill_diagonal(C_cleaned, 1.0)
    # Clip to [-1, 1]
    C_cleaned = np.clip(C_cleaned, -1.0, 1.0)

    # If cleaning removes everything (very small window), fall back to raw C
    if np.allclose(C_cleaned[~np.eye(N, dtype=bool)], 0.0):
        return C

    return C_cleaned


def build_gbs_graph(C: np.ndarray) -> np.ndarray:
    """
    Convert cleaned correlation matrix to GBS graph matrix: A = max(0, C - I).

    Zero the diagonal and discard all negative off-diagonal correlations.

    Parameters
    ----------
    C : np.ndarray
        Cleaned correlation matrix, shape (N, N).

    Returns
    -------
    np.ndarray
        Non-negative symmetric weighted adjacency matrix, shape (N, N).
    """
    A = np.maximum(0.0, C - np.eye(C.shape[0]))
    assert np.allclose(A, A.T, atol=1e-10), "A must be symmetric"
    assert np.all(A >= 0), "A must be non-negative"
    assert np.allclose(np.diag(A), 0.0), "A diagonal must be zero"
    return A


def estimate_k_from_eigenspacing(
    residuals_20d: np.ndarray,
    B: int = 100,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Estimate cluster count K from eigenvalue spacing permutation test.

    Algorithm (paper Appendix A):
    1. Compute correlation matrix from 20-day residual returns.
    2. Order eigenvalues descending: lambda_1 >= lambda_2 >= ...
    3. Compute spacings: s_k = lambda_k - lambda_{k+1}
    4. Generate B column-wise permutation replicates of standardized residuals.
    5. Use top spacings from permutations to calibrate alpha=0.05 threshold.
    6. Count significant spacings until first failure → K.

    ASSUMPTION [B]: Paper shows B=1000 as example; using B=100 by default for speed.

    Parameters
    ----------
    residuals_20d : np.ndarray
        Residual returns over 20 days, shape (20, N).
    B : int
        Number of permutation replicates.
    alpha : float
        Right-tail significance level.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    int
        Estimated cluster count K (>= 2).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T, N = residuals_20d.shape

    # Standardize residuals (column-wise)
    means = residuals_20d.mean(axis=0)
    stds = residuals_20d.std(axis=0)
    stds[stds < 1e-10] = 1.0
    standardized = (residuals_20d - means) / stds

    # Compute observed correlation matrix and eigenvalues
    corr = np.corrcoef(standardized.T)
    corr = np.nan_to_num(corr, nan=0.0)
    eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]

    # Compute observed spacings
    n_eig = min(N, T - 1)  # rank
    spacings = eigenvalues[:n_eig - 1] - eigenvalues[1:n_eig]

    # Generate null distribution of top spacings via permutation
    null_spacings = np.zeros((B, n_eig - 1))
    for b in range(B):
        perm_data = np.zeros_like(standardized)
        for j in range(N):
            perm_data[:, j] = rng.permutation(standardized[:, j])
        perm_corr = np.corrcoef(perm_data.T)
        perm_corr = np.nan_to_num(perm_corr, nan=0.0)
        perm_eig = np.sort(np.linalg.eigvalsh(perm_corr))[::-1]
        perm_sp = perm_eig[:n_eig - 1] - perm_eig[1:n_eig]
        null_spacings[b] = perm_sp

    # Threshold: (1-alpha) quantile of null spacings at each position
    thresholds = np.quantile(null_spacings, 1 - alpha, axis=0)

    # Count significant spacings until first failure
    K = 0
    for k in range(len(spacings)):
        if spacings[k] > thresholds[k]:
            K += 1
        else:
            break

    return max(2, K)


def load_vix(start_date: str, end_date: str) -> pd.Series:
    """
    Load daily CBOE VIX levels using yfinance.

    Parameters
    ----------
    start_date, end_date : str
        Date range.

    Returns
    -------
    pd.Series
        Daily VIX close levels.
    """
    vix = yf.download("^VIX", start=start_date, end=end_date,
                      auto_adjust=True, progress=False)["Close"]
    vix.index = pd.to_datetime(vix.index)
    return vix
