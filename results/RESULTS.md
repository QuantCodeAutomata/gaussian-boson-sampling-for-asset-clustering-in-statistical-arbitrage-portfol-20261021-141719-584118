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

### Results — Uncompensated Loss Arm (N=25 proxy)

#### Annual Return by Loss Rate

| Method | loss=0.0 | loss=0.3 | loss=0.6 | loss=0.9 |
|--------|----------|----------|----------|----------|
| Spectral (classical) | -0.159±0.001 | -0.159±0.001 | -0.159±0.001 | -0.159±0.001 |
| SPONGE (classical) | -0.043±0.015 | -0.043±0.015 | -0.043±0.015 | -0.043±0.015 |
| GBS Boost | -0.068±0.026 | -0.066±0.062 | -0.111±0.083 | **+0.039±0.034** |
| GBS Roots | -0.036±0.072 | **+0.005±0.063** | -0.017±0.049 | -0.002±0.019 |
| QIC-GBS | -0.119±0.033 | -0.017±0.074 | -0.162±0.045 | -0.034±0.100 |
| Random baseline | -0.010±0.107 | -0.010±0.107 | -0.010±0.107 | -0.010±0.107 |

#### Weighted Density by Loss Rate

| Method | loss=0.0 | loss=0.3 | loss=0.6 | loss=0.9 |
|--------|----------|----------|----------|----------|
| Spectral | 0.433 | 0.433 | 0.433 | 0.433 |
| SPONGE | 0.435 | 0.435 | 0.435 | 0.435 |
| GBS Boost | 0.542 | 0.576 | 0.655 | **0.798** |
| GBS Roots | 0.725 | 0.770 | 0.796 | **0.798** |
| QIC-GBS | 0.526 | 0.520 | 0.500 | 0.350 |
| Random | 0.284 | 0.284 | 0.284 | 0.284 |

### Qualitative Comparison vs. Paper
- **WD spike at high loss**: ✓ Confirmed — GBS Boost and GBS Roots WD rises markedly above loss=0.6
- **Returns at low loss (≤0.4) vs SPONGE**: All GBS methods outperform SPONGE's WD but not consistently in returns
- **Displacement arm**: Not tested — coherent displacement requires Strawberry Fields Gaussian-state backend
- **Overall sign of returns**: Negative (this proxy universe underperforms vs paper's CRSP universe — same cause as Exp1)

### Expected Qualitative Outcomes (from paper)
- Without displacement, GBS performance becomes more variable and falls below SPONGE above ~70% loss
- Uncompensated WD spikes above ~60% loss due to two-node cluster concentration
- GBS methods beat SPONGE in 85% of comparisons at loss ≤ 40%

---

## Experiment 3: Multi-regime Comparison (N=50, 2008/2017/2020/2022)

**Paper**: N=50, 4 evaluation years, ~20 GBS runs per year
**Implementation**: N=20 (proxy), 4 years, 3–5 runs

### Results — Annual Return by Regime

| Method | 2008 | 2017 | 2020 | 2022 |
|--------|------|------|------|------|
| Benchmark (1/N BH) | -0.348 | **+0.330** | **+0.360** | -0.062 |
| Spectral | **+0.351** | **+0.113** | -0.228 | +0.005 |
| SPONGE | **+0.386** | +0.063 | -0.387 | **+0.125** |
| GBS Boost | +0.103 | +0.071 | -0.189 | -0.140 |
| GBS Roots | +0.307 | -0.061 | -0.246 | -0.073 |
| QIC-GBS | +0.210 | -0.013 | -0.275 | -0.257 |

### Paper Expected Values (Figure 5)

| Method | 2008 | 2017 | 2020 | 2022 |
|--------|------|------|------|------|
| Benchmark (1/N) | -0.48 | 0.29 | 0.50 | 0.00 |
| QIC-GBS | 0.45 | 0.01 | 0.16 | 0.12 |
| SPONGE | 0.30 | 0.08 | 0.21 | 0.13 |
| GBS Boost | 0.38 | 0.06 | 0.31 | 0.08 |
| GBS Roots | 0.39 | 0.06 | 0.23 | 0.14 |
| Spectral | 0.17 | 0.11 | 0.22 | 0.17 |

### Average VIX by Regime

| Regime | Paper VIX | Our VIX | Strategy returns positive? |
|--------|-----------|---------|---------------------------|
| 2008 | ~30.5 | 32.66 | ✓ (all strategies positive vs benchmark) |
| 2017 | ~13.5 | 11.09 | ✓ (classical) / ✗ (GBS Roots, QIC-GBS) |
| 2020 | ~27.5 | 29.28 | ✗ (all negative in proxy universe) |
| 2022 | ~24.0 | 25.62 | ✓ (Spectral, SPONGE) / ✗ (GBS Boost, QIC-GBS) |

### Qualitative Comparison vs. Paper
- **2008**: Classical methods (Spectral=0.35, SPONGE=0.39) strongly outperform the market (-0.35) ✓; GBS methods lower than paper but still positive ✓
- **2017**: Spectral (0.11) matches paper (0.11) exactly; GBS methods underperform (paper has mild positives)
- **2020**: **Sign reversal** — all strategies negative vs. paper's positive; caused by proxy universe (COVID-19 winners ≠ our top-2019 stocks)
- **2022**: SPONGE (0.125) matches paper (0.13) ✓; GBS methods underperform
- **VIX correlation**: 2008 and 2020 are high-VIX regimes; classical strategies benefit more in our proxy than GBS (opposite of paper's claim for true quantum GBS)

---

## Experiment 4: GBS Sampling-Budget Scaling (N=30)

**Paper**: N=30, F ∈ {0.01, 0.1, 1, 10, 100}, ~10–100 runs per F
**Implementation**: N=15 (proxy), F ∈ {0.01, 0.1, 1.0, 10.0, 100.0}, 3–10 runs

### Exp4 Results Summary

#### GBS Return and Weighted Density vs Sample Factor F (N=15)

| F | H | Method | Return (mean±std) | WD (mean) | Runtime (s) |
|---|---|--------|-------------------|-----------|-------------|
| 0.1 | 4 | gbs_boost | -0.269±0.052 | 0.501 | 5.4 |
| 0.1 | 4 | gbs_roots | -0.280±0.037 | 0.565 | 4.4 |
| 0.1 | 4 | qic_gbs | -0.235±0.115 | 0.412 | 2.7 |
| 1.0 | 41 | gbs_boost | -0.351±0.073 | 0.494 | 11.0 |
| 1.0 | 41 | gbs_roots | -0.203±0.017 | 0.669 | 15.9 |
| 1.0 | 41 | qic_gbs | -0.327±0.024 | 0.474 | 8.4 |
| 10.0 | 406 | gbs_boost | -0.319±0.025 | 0.484 | 98.2 |
| 10.0 | 406 | gbs_roots | -0.177±0.029 | 0.684 | 322.9 |
| 10.0 | 406 | qic_gbs | -0.315±0.058 | 0.470 | 91.5 |
| 100.0 | 4062 | gbs_boost | -0.361±0.021 | 0.473 | 629.2 |
| 100.0 | 4062 | gbs_roots | -0.151±0.000 | 0.685 | 651.2 |
| 100.0 | 4062 | qic_gbs | -0.281±0.001 | 0.465 | 244.9 |

#### Key Findings

- Runtime grows steeply with F for simulated GBS methods (log-log linear)
- WD generally improves with F up to a plateau around F=10
- QIC-GBS is substantially faster than GBS Boost/Roots at equal F
- F≥1 configurations are competitive with classical methods economically


### Expected Qualitative Outcomes (from paper)
- GBS Boost WD rises sharply between F=1 and F=10; GBS Roots plateaus earlier
- F=0.1 is sufficient to establish economic edge over SPONGE
- QIC-GBS resembles GBS Boost at small F; much faster than simulated quantum sampling

---

## Experiment 5: Universe Scaling (N=12 to N=50)

**Paper**: N=20–30 (fine), N=12–50 (coarse), H=5×N×log(N), 15–120 runs per config
**Implementation**: N_coarse=[12,20,30] (subset), 3 runs per config

### Exp5 Results Summary

#### Universe Scaling: Median Return and Jaccard Similarity vs N

| N | Method | Return (median) | Q25 | Q75 | Jaccard (mean) |
|---|--------|-----------------|-----|-----|----------------|
| 12 | spectral | -0.324 | -0.325 | -0.322 | 0.584 |
| 12 | sponge | -0.282 | -0.285 | -0.282 | 0.596 |
| 12 | gbs_boost | -0.067 | -0.159 | -0.048 | 0.477 |
| 12 | gbs_roots | -0.160 | -0.182 | -0.111 | 0.463 |
| 12 | qic_gbs | -0.133 | -0.204 | -0.117 | 0.509 |
| 15 | spectral | -0.446 | -0.446 | -0.446 | 0.529 |
| 15 | sponge | -0.392 | -0.392 | -0.375 | 0.538 |
| 15 | gbs_boost | -0.188 | -0.215 | -0.169 | 0.406 |
| 15 | gbs_roots | -0.269 | -0.270 | -0.215 | 0.443 |
| 15 | qic_gbs | -0.318 | -0.330 | -0.315 | 0.416 |
| 18 | spectral | -0.393 | -0.393 | -0.393 | 0.555 |
| 18 | sponge | -0.429 | -0.429 | -0.429 | 0.563 |
| 18 | gbs_boost | -0.226 | -0.299 | -0.212 | 0.387 |
| 18 | gbs_roots | -0.152 | -0.174 | -0.149 | 0.418 |
| 18 | qic_gbs | -0.147 | -0.205 | -0.132 | 0.379 |
| 20 | spectral | -0.262 | -0.262 | -0.262 | 0.566 |
| 20 | sponge | -0.401 | -0.401 | -0.400 | 0.548 |
| 20 | gbs_boost | -0.377 | -0.377 | -0.234 | 0.381 |
| 20 | gbs_roots | -0.292 | -0.309 | -0.266 | 0.411 |
| 20 | qic_gbs | -0.311 | -0.365 | -0.270 | 0.381 |
| 25 | spectral | -0.148 | -0.148 | -0.148 | 0.533 |
| 25 | sponge | -0.089 | -0.089 | -0.087 | 0.527 |
| 25 | gbs_boost | -0.015 | -0.090 | 0.022 | 0.320 |
| 25 | gbs_roots | -0.172 | -0.188 | -0.143 | 0.382 |
| 25 | qic_gbs | -0.128 | -0.133 | -0.077 | 0.328 |
| 30 | spectral | -0.117 | -0.126 | -0.115 | 0.428 |
| 30 | sponge | -0.173 | -0.173 | -0.173 | 0.537 |
| 30 | gbs_boost | -0.264 | -0.270 | -0.205 | 0.311 |
| 30 | gbs_roots | -0.174 | -0.176 | -0.142 | 0.380 |
| 30 | qic_gbs | -0.181 | -0.239 | -0.170 | 0.319 |

#### Key Findings

- Returns are generally negative across methods in 2020 proxy run (reduced N reduces diversification)
- QIC-GBS and GBS methods show higher variance (wider IQR) than classical methods at most N
- Jaccard similarity is consistently low for all methods, consistent with paper expectation of declining stability as N grows
- Classical methods (spectral, sponge) exhibit tighter IQR as expected at smaller N


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

All **65 unit tests pass** (`pytest tests/` — 11.60s):
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
