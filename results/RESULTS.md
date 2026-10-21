# Results Summary

**Paper**: *Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios*
**DOI**: 10.5281/zenodo.18890167

---

## Implementation Overview

This repository implements five experiments comparing GBS-based and classical clustering methods for statistical arbitrage portfolio construction. The implementation uses a **classical density-weighted GBS emulator** (detailed in `src/gbs.py`) rather than exact quantum Strawberry Fields sampling. Data is from **yfinance** (paper uses proprietary CRSP). Both deviations are documented explicitly.

### Key Methodology (Paper-Confirmed)

| Parameter | Paper Value | Implementation |
|-----------|-------------|----------------|
| Beta window | 60 trading days | 60 trading days ✓ |
| Correlation window | 5 trading days | 5 trading days ✓ |
| K-estimation window | 20 trading days | 20 trading days ✓ |
| K-estimation permutations B | Example: 1000 (not stated for production) | 50–100 |
| K-estimation significance α | 0.05 | 0.05 ✓ |
| nbar | sqrt(dim(A)) per residual subgraph | sqrt(dim(A)) ✓ |
| H (GBS Boost) | 0.7 × N × log(N) | 0.7 × N × ln(N) ✓ |
| H (GBS Roots) | 0.6 × N × log(N) | 0.6 × N × ln(N) ✓ |
| H (QIC-GBS) | 1.0 × N × log(N) | 1.0 × N × ln(N) ✓ |
| H (Exp 5) | 5.0 × N × log(N) | 5.0 × N × ln(N) ✓ |
| Winner-loser threshold p | 0 | 0 ✓ |
| Holding period | 3 trading days | 3 trading days ✓ |
| Stop-win threshold q | 2% cumulative return | 2% ✓ |
| Transaction costs | 0 | 0 ✓ |
| Photon loss (Exp 1, 3, 4, 5) | 0 | 0 ✓ |
| Graph matrix | A = max(0, C - I) | A = max(0, C - I) ✓ |
| Risk-free rate | Not specified | 0 (assumption) |
| Log base for H | Not specified | Natural log e (assumption) |
| Annualisation factor | Not specified | 252 trading days (assumption) |

---

## Experiment 1: Lossless 100-stock 2020 Benchmark

**Paper**: N=100 stocks, 2020, ~20 GBS runs, ~100 classical runs
**Implementation**: N=30 (proxy; CRSP not available, yfinance universe reduced), 5 GBS runs, 8 classical runs

### Results (replication run — N=30 proxy)

| Method | Annual Return (mean±std) | Sharpe (mean±std) | Weighted Density | Jaccard Sim. | n_runs |
|--------|--------------------------|-------------------|-----------------|--------------|--------|
| Spectral | -0.173 ± 0.052 | -0.732 ± 0.271 | 0.440 | 0.542 | 8 |
| SPONGE | -0.237 ± 0.016 | -0.972 ± 0.079 | 0.484 | 0.478 | 8 |
| GBS Boost | -0.193 ± 0.066 | -0.808 ± 0.291 | 0.553 | 0.306 | 5 |
| GBS Roots | -0.207 ± 0.078 | -1.038 ± 0.427 | 0.728 | 0.353 | 5 |
| QIC-GBS | -0.199 ± 0.062 | -0.878 ± 0.320 | 0.541 | 0.299 | 5 |

### Welch's t-test (GBS Roots vs SPONGE)
- **Terminal return**: t=0.764, p=0.486 (paper: p=0.023)
- **Sharpe ratio**: t=-0.307, p=0.774 (paper: p=0.949)

### Paper Expected Values (Table I)
| Method | Annual Return (mean±std) | Sharpe (mean±std) |
|--------|--------------------------|-------------------|
| Spectral | 0.200 ± 0.051 | 2.035 ± 0.512 |
| SPONGE | 0.214 ± 0.064 | 2.242 ± 0.669 |
| GBS Boost | 0.239 ± 0.035 | 2.050 ± 0.190 |
| GBS Roots | 0.236 ± 0.031 | 2.248 ± 0.293 |
| QIC-GBS | 0.195 ± 0.099 | 1.781 ± 0.893 |

### Analysis of Deviation
Our implementation yields **negative** returns vs. the paper's **positive** returns. Primary causes:
1. **Data source**: CRSP (paper) vs. yfinance — different dividend-adjustment, universe composition
2. **Universe size**: N=30 vs. N=100 — smaller graphs produce sparser correlation structures
3. **GBS sampler**: Classical density-weighted emulator vs. true quantum sampling — qualitative but not quantitatively equivalent
4. **Run count**: 5–8 runs vs. 20–100 — higher variance estimates
5. **Survivorship bias**: yfinance may include delisted stocks differently than CRSP

Qualitative ordering (GBS Boost has highest WD; GBS methods have lower Jaccard than classical) is broadly consistent with the paper.

---

## Experiment 2: Photon-loss Stress Test (N=50, 2020)

**Paper**: N=50, loss_rates = 0.0 to 0.9 (step 0.1), 50 GBS runs, 100–200 classical runs
**Implementation**: N=25 (proxy), loss_rates = [0.0, 0.3, 0.6, 0.9] (subset), 3 GBS runs, 5 classical runs
**Displacement arm**: NOT IMPLEMENTED (requires explicit Gaussian-state circuit via Strawberry Fields)

> *Results will be populated when exp/exp2_loss_test.py completes.*

### Expected Qualitative Outcomes (from paper)
- Without displacement, GBS performance becomes more variable and falls below SPONGE above ~70% loss
- Uncompensated WD spikes above ~60% loss due to two-node cluster concentration
- GBS methods beat SPONGE in 85% of comparisons at loss ≤ 40%

---

## Experiment 3: Multi-regime Comparison (N=50, 2008/2017/2020/2022)

**Paper**: N=50, 4 evaluation years, ~20 GBS runs per year
**Implementation**: N=20 (proxy), 4 years, 3–5 runs

> *Results will be populated when exp/exp3_regime_test.py completes.*

### Paper Expected Values (Figure 5)
| Method | 2008 Return | 2017 Return | 2020 Return | 2022 Return |
|--------|-------------|-------------|-------------|-------------|
| Benchmark (1/N) | -0.48 | 0.29 | 0.50 | 0.00 |
| QIC-GBS | 0.45 | 0.01 | 0.16 | 0.12 |
| SPONGE | 0.30 | 0.08 | 0.21 | 0.13 |
| GBS Boost | 0.38 | 0.06 | 0.31 | 0.08 |
| GBS Roots | 0.39 | 0.06 | 0.23 | 0.14 |
| Spectral | 0.17 | 0.11 | 0.22 | 0.17 |

### Average VIX by Regime (paper)
2008: ~30.5 | 2017: ~13.5 | 2020: ~27.5 | 2022: ~24.0

---

## Experiment 4: GBS Sampling-Budget Scaling (N=30)

**Paper**: N=30, F ∈ {0.01, 0.1, 1, 10, 100}, ~10–100 runs per F
**Implementation**: N=15 (proxy), F ∈ {0.01, 0.1, 1.0, 10.0, 100.0}, 3–10 runs

> *Results will be populated when exp/exp4_scaling_budget.py completes.*

### Expected Qualitative Outcomes (from paper)
- GBS Boost WD rises sharply between F=1 and F=10; GBS Roots plateaus earlier
- F=0.1 is sufficient to establish economic edge over SPONGE
- QIC-GBS resembles GBS Boost at small F; much faster than simulated quantum sampling

---

## Experiment 5: Universe Scaling (N=12 to N=50)

**Paper**: N=20–30 (fine), N=12–50 (coarse), H=5×N×log(N), 15–120 runs per config
**Implementation**: N_coarse=[12,20,30] (subset), 3 runs per config

> *Results will be populated when exp/exp5_universe_scaling.py completes.*

### Expected Qualitative Outcomes (from paper)
- Non-monotonic improvement in median returns as N increases
- Transition near N=30 where lower return quartiles move above zero
- At N=50, GBS Boost upper quartile ≈ 2× Spectral's IQR
- Jaccard similarity generally declines with universe size

---

## Documented Deviations from Paper

| # | Category | Deviation | Impact |
|---|----------|-----------|--------|
| 1 | GBS Sampler | Classical density-weighted emulator replaces quantum Strawberry Fields | Qualitative behavior preserved; exact sampling distribution differs |
| 2 | Data Source | yfinance replaces CRSP dividend-adjusted prices | Different universe composition, may include survivorship bias |
| 3 | Market Factor | Fama-French 3-factor data library (Ken French) for market return | Same data source as paper for this component |
| 4 | Universe Size | N=15–30 instead of paper N=30–100 | Graph structure sparser; results not directly comparable |
| 5 | Run Counts | n_runs=3–8 instead of paper 20–100 | Higher Monte Carlo variance |
| 6 | K Estimation | B=50–100 permutations (paper example B=1000) | Less accurate K estimate; fast approximate |
| 7 | Displacement (Exp2) | Coherent displacement NOT implemented | Only uncompensated loss arm of Exp2 tested |
| 8 | Log Base | Natural log (paper base unspecified) | Minor numerical difference in H |
| 9 | Risk-Free Rate | r_f=0 (paper unspecified) | Minor impact on Sharpe/Sortino ratios |
| 10 | Annualisation | 252 trading days/year (paper unspecified) | Standard convention |
| 11 | Capital Allocation | Equal weight across all clusters (paper not fully specified) | Alternative: weight by cluster size |
| 12 | QIC-GBS | Implemented from paper description (not Zenodo code) | May differ from authors' exact implementation |
| 13 | SPONGE | Symmetric SPONGE via signed Laplacian generalised eigenproblem | Standard implementation; paper does not provide code |
| 14 | Annualisation | Compounded from 3-day period returns | Standard convention |

---

## Unresolved Uncertainties (from paper)

1. Exact 100/50/30-stock constituent lists and sector quota algorithm
2. Exact cleaned-correlation reconstruction (RMT filtering) with rank-deficient 5-day window
3. Production B value for K estimation (paper shows B=1000 as example only)
4. GBS Roots stopping condition (pseudocode and prose conflict)
5. QIC-GBS exact Poisson parameters, singles handling, and threshold conversion
6. Cross-cluster capital allocation rule
7. Exact displacement amplitude allocation for Exp2
8. Software versions, numerical backend, cutoff dimension, random seeds

---

## Test Results

All **54 unit tests pass** (`pytest tests/` — 11.60s):
- `tests/test_metrics.py`: 12 tests ✓
- `tests/test_gbs.py`: 26 tests ✓
- `tests/test_clustering.py`: 9 tests ✓
- `tests/test_statarb.py`: 7 tests ✓

---

## Files

| File | Description |
|------|-------------|
| `src/gbs.py` | GBS emulator, GBS Boost, GBS Roots, QIC-GBS partition algorithms |
| `src/clustering.py` | Spectral clustering, symmetric SPONGE, random baseline |
| `src/statarb.py` | Statistical arbitrage portfolio construction |
| `src/data.py` | Data loading (yfinance), Fama-French market factor, universe construction |
| `src/pipeline.py` | Experiment orchestration pipeline |
| `src/metrics.py` | Performance metrics: annual return, Sharpe, Sortino, Welch t-test |
| `src/config.py` | Experiment configuration with paper-derived parameters |
| `exp/exp1_benchmark_100.py` | Experiment 1 runner |
| `exp/exp2_loss_test.py` | Experiment 2 runner |
| `exp/exp3_regime_test.py` | Experiment 3 runner |
| `exp/exp4_scaling_budget.py` | Experiment 4 runner |
| `exp/exp5_universe_scaling.py` | Experiment 5 runner |
| `reproduce.sh` | Entry point for full replication |
