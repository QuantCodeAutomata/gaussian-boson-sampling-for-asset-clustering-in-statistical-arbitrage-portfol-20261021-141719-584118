"""
Configuration parameters for GBS Statistical Arbitrage experiments.

Paper: "Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios"
All parameters are directly from the paper methodology unless noted as assumptions.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Unified configuration for all GBS StatArb experiments."""

    # ── Universe ─────────────────────────────────────────────────────────────
    n_stocks: int = 100                   # stock universe size (N)
    eval_year: int = 2020                 # evaluation calendar year

    # GICS sectors (11 sectors) used for stratified sampling
    # SIC→sector mapping is approximated; exact mapping is in data module
    sectors: List[str] = field(default_factory=lambda: [
        "Energy", "Materials", "Industrials", "ConsumerDiscretionary",
        "ConsumerStaples", "HealthCare", "Financials", "InformationTechnology",
        "CommunicationServices", "Utilities", "RealEstate",
    ])

    # ── Rolling windows (in trading days) ────────────────────────────────────
    beta_window: int = 60                 # trailing days for beta estimation
    corr_window: int = 5                  # trailing days for residual correlation matrix
    k_est_window: int = 20               # trailing days for K estimation (classical methods)

    # ── GBS parameters ───────────────────────────────────────────────────────
    photon_loss: float = 0.0             # uniform photon loss rate l_r (0=lossless)
    # nbar = sqrt(dim(A)) is computed dynamically per subgraph; set globally here
    # as reference for the full N-dimensional graph
    nbar_factor: float = 1.0             # nbar = nbar_factor * sqrt(N)

    # Sample budgets H = F * N * log(N), log base = natural log (assumption; paper unspecified)
    F_boost: float = 0.7                 # sample factor for GBS Boost  (paper: H=0.7*N*log(N))
    F_roots: float = 0.6                 # sample factor for GBS Roots  (paper: H=0.6*N*log(N))
    F_qic: float = 1.0                   # sample factor for QIC-GBS    (paper: H=N*log(N))

    # ── Classical K estimation ────────────────────────────────────────────────
    k_perm_replicates: int = 100         # B permutation replicates for K estimation
    # Paper's appendix shows B=1000 as example; 1000 is slow → default 100 with note
    k_alpha: float = 0.05                # significance level for spacing threshold

    # ── Strategy parameters ───────────────────────────────────────────────────
    holding_days: int = 3                # holding period in trading days
    stop_win: float = 0.02               # stop-win threshold (2% cumulative return)
    winner_loser_threshold: float = 0.0  # p=0 (positive devn=winner, negative=loser)

    # ── Performance annualisation ─────────────────────────────────────────────
    # Assumption: 252 trading days per year, risk-free = 0 (not specified in paper)
    ann_factor: int = 252
    risk_free_rate: float = 0.0          # annual risk-free rate (paper unspecified → 0)

    # ── Run counts ────────────────────────────────────────────────────────────
    n_runs_gbs: int = 20                 # independent GBS runs (paper: ~20)
    n_runs_classical: int = 20           # classical runs (paper: ~100; reduced for speed)

    # ── Scaling experiment parameters (exp_4) ────────────────────────────────
    sample_factors: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])

    # ── Regime experiment (exp_3) ─────────────────────────────────────────────
    regime_years: List[int] = field(default_factory=lambda: [2008, 2017, 2020, 2022])

    # ── Universe scaling (exp_5) ──────────────────────────────────────────────
    n_stocks_fine: List[int] = field(default_factory=lambda: list(range(20, 31)))
    n_stocks_coarse: List[int] = field(default_factory=lambda: [12, 20, 30, 40, 50])


# ── Experiment-specific configurations (override defaults as needed) ──────────

def exp1_config() -> ExperimentConfig:
    """Experiment 1: Lossless 100-stock 2020 benchmark."""
    return ExperimentConfig(n_stocks=100, eval_year=2020, photon_loss=0.0)


def exp2_config() -> ExperimentConfig:
    """Experiment 2: 50-stock photon-loss stress test 2020."""
    return ExperimentConfig(n_stocks=50, eval_year=2020, photon_loss=0.0)


def exp3_config(year: int) -> ExperimentConfig:
    """Experiment 3: 50-stock multi-regime."""
    return ExperimentConfig(n_stocks=50, eval_year=year, photon_loss=0.0)


def exp4_config() -> ExperimentConfig:
    """Experiment 4: 30-stock sampling-budget scaling."""
    return ExperimentConfig(n_stocks=30, eval_year=2020, photon_loss=0.0)


def exp5_config(n_stocks: int) -> ExperimentConfig:
    """Experiment 5: Universe scaling."""
    return ExperimentConfig(n_stocks=n_stocks, eval_year=2020, photon_loss=0.0,
                            F_boost=5.0, F_roots=5.0, F_qic=5.0)
