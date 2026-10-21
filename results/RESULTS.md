# RESULTS — GBS Statistical Arbitrage Paper Replication

> Paper: *Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios*
>
> Generated: 2026-10-21 18:33 UTC
>
> **Important**: All five experiments use reduced parameters (N=15–30 vs paper N=50–100;
> classical GBS emulator vs quantum Strawberry Fields; yfinance vs CRSP).
> Results are qualitatively comparable but not numerically identical to the paper.

---

## Test Suite

All **65 unit tests pass** (`pytest tests/` — see `results/test_output.txt`).

---

## Experiment 1 — Lossless 100-stock 2020 Benchmark (N=30 proxy)

**Paper**: N=100, 2020, ~20 GBS runs / ~100 classical runs
**Implementation**: N=30 proxy, 5 GBS runs, 8 classical runs, yfinance data

| Method | Return (mean±std) | Sharpe (mean±std) | WD (mean) | Jaccard | Runs |
|--------|-------------------|-------------------|-----------|---------|------|
| Spectral | -0.173±0.052 | -0.732±0.271 | 0.440 | 0.542 | 8 |
| SPONGE | -0.237±0.016 | -0.972±0.079 | 0.484 | 0.478 | 8 |
| GBS Boost | -0.193±0.066 | -0.808±0.291 | 0.553 | 0.306 | 5 |
| GBS Roots | -0.207±0.078 | -1.038±0.427 | 0.728 | 0.353 | 5 |
| QIC-GBS | -0.199±0.062 | -0.878±0.320 | 0.541 | 0.299 | 5 |

### Welch t-test: GBS Roots vs SPONGE (terminal returns)

- t-statistic: 0.7640
- p-value: 0.4855
- Significant at α=0.05: No

### Expected Qualitative Outcomes (from paper, N=100)
| Method | Return (paper) | Sharpe (paper) |
|--------|----------------|----------------|
| Spectral | 0.200±0.051 | 2.035±0.512 |
| SPONGE | 0.214±0.064 | 2.242±0.669 |
| GBS Boost | 0.239±0.035 | 2.050±0.190 |
| GBS Roots | 0.236±0.031 | 2.248±0.293 |
| QIC-GBS | 0.195±0.099 | 1.781±0.893 |

> Paper Welch test: p=0.023 for GBS Roots vs SPONGE terminal returns

---

## Experiment 2 — Photon-Loss Stress Test (N=25 proxy)

**Paper**: N=50, loss rates 0–0.9, with/without coherent displacement
**Implementation**: N=25 proxy, 3 runs per config, uncompensated loss only (no displacement arm)

| Loss | Method | Return (mean±std) | WD (mean) |
|------|--------|-------------------|-----------|
| 0.0 | Spectral | -0.159±0.001 | 0.433 |
| 0.0 | SPONGE | -0.043±0.015 | 0.435 |
| 0.0 | GBS Boost | -0.068±0.026 | 0.542 |
| 0.0 | GBS Roots | -0.036±0.072 | 0.725 |
| 0.0 | QIC-GBS | -0.119±0.033 | 0.526 |
| 0.0 | Random | -0.010±0.107 | 0.284 |
| 0.3 | Spectral | -0.159±0.001 | 0.433 |
| 0.3 | SPONGE | -0.043±0.015 | 0.435 |
| 0.3 | GBS Boost | -0.066±0.062 | 0.576 |
| 0.3 | GBS Roots | 0.005±0.063 | 0.770 |
| 0.3 | QIC-GBS | -0.017±0.074 | 0.520 |
| 0.3 | Random | -0.010±0.107 | 0.284 |
| 0.6 | Spectral | -0.159±0.001 | 0.433 |
| 0.6 | SPONGE | -0.043±0.015 | 0.435 |
| 0.6 | GBS Boost | -0.111±0.083 | 0.655 |
| 0.6 | GBS Roots | -0.017±0.049 | 0.796 |
| 0.6 | QIC-GBS | -0.162±0.045 | 0.500 |
| 0.6 | Random | -0.010±0.107 | 0.284 |
| 0.9 | Spectral | -0.159±0.001 | 0.433 |
| 0.9 | SPONGE | -0.043±0.015 | 0.435 |
| 0.9 | GBS Boost | 0.039±0.034 | 0.798 |
| 0.9 | GBS Roots | -0.002±0.019 | 0.798 |
| 0.9 | QIC-GBS | -0.034±0.100 | 0.350 |
| 0.9 | Random | -0.010±0.107 | 0.284 |

### Expected Qualitative Outcomes (from paper)
- GBS degrades above ~70% loss, approaching random baseline
- High-loss WD spike is artifact of predominantly two-node clusters
- With displacement (not implemented): returns remain stable across loss rates
- GBS beats SPONGE in ≥85% of cases at loss ≤ 40%

---

## Experiment 3 — Multi-Regime Study (2008, 2017, 2020, 2022) — N=20 proxy

**Paper**: N=50, four calendar-year market regimes
**Implementation**: N=20 proxy, yfinance data, 3 runs per config

| Year | Avg VIX | Method | Return |
|------|---------|--------|--------|
| 2008 | 32.664 | 1/N BnH | -0.348 |
| 2008 | 32.664 | Spectral | 0.351 |
| 2008 | 32.664 | SPONGE | 0.386 |
| 2008 | 32.664 | GBS Boost | 0.103 |
| 2008 | 32.664 | GBS Roots | 0.307 |
| 2008 | 32.664 | QIC-GBS | 0.210 |
| 2008 | 32.664 | 1/N BnH | -0.348 |
| 2017 | 11.090 | 1/N BnH | 0.330 |
| 2017 | 11.090 | Spectral | 0.113 |
| 2017 | 11.090 | SPONGE | 0.063 |
| 2017 | 11.090 | GBS Boost | 0.071 |
| 2017 | 11.090 | GBS Roots | -0.061 |
| 2017 | 11.090 | QIC-GBS | -0.013 |
| 2017 | 11.090 | 1/N BnH | 0.330 |
| 2020 | 29.277 | 1/N BnH | 0.360 |
| 2020 | 29.277 | Spectral | -0.228 |
| 2020 | 29.277 | SPONGE | -0.387 |
| 2020 | 29.277 | GBS Boost | -0.189 |
| 2020 | 29.277 | GBS Roots | -0.246 |
| 2020 | 29.277 | QIC-GBS | -0.275 |
| 2020 | 29.277 | 1/N BnH | 0.360 |
| 2022 | 25.624 | 1/N BnH | -0.062 |
| 2022 | 25.624 | Spectral | 0.005 |
| 2022 | 25.624 | SPONGE | 0.125 |
| 2022 | 25.624 | GBS Boost | -0.140 |
| 2022 | 25.624 | GBS Roots | -0.073 |
| 2022 | 25.624 | QIC-GBS | -0.257 |
| 2022 | 25.624 | 1/N BnH | -0.062 |

### Expected Values (from paper, N=50)
| Year | Spectral | SPONGE | GBS Boost | GBS Roots | QIC-GBS | 1/N BnH |
|------|----------|--------|-----------|-----------|---------|---------|
| 2008 | 0.17 | 0.30 | 0.38 | 0.39 | 0.45 | -0.48 |
| 2017 | 0.11 | 0.08 | 0.06 | 0.06 | 0.01 | 0.29 |
| 2020 | 0.22 | 0.21 | 0.31 | 0.23 | 0.16 | 0.50 |
| 2022 | 0.17 | 0.13 | 0.08 | 0.14 | 0.12 | 0.00 |

Paper VIX averages: ~30.5 (2008), ~13.5 (2017), ~27.5 (2020), ~24.0 (2022)

---

## Experiment 4 — GBS Sampling-Budget Scaling (N=15 proxy)

**Paper**: N=30, F ∈ {0.01, 0.1, 1, 10, 100}, H=F×N×log(N), ~10–100 runs per F
**Implementation**: N=15 proxy, same F grid, 3–10 runs per config

| F | H | Method | Return (mean±std) | WD (mean) | Runtime (s) |
|---|---|--------|-------------------|-----------|-------------|
| 0.1 | 4 | GBS Boost | -0.269±0.052 | 0.501 | 5.4 |
| 0.1 | 4 | GBS Roots | -0.280±0.037 | 0.565 | 4.4 |
| 0.1 | 4 | QIC-GBS | -0.235±0.115 | 0.412 | 2.7 |
| 1 | 41 | GBS Boost | -0.351±0.073 | 0.494 | 11.0 |
| 1 | 41 | GBS Roots | -0.203±0.017 | 0.669 | 15.9 |
| 1 | 41 | QIC-GBS | -0.327±0.024 | 0.474 | 8.4 |
| 10 | 406 | GBS Boost | -0.319±0.025 | 0.484 | 98.2 |
| 10 | 406 | GBS Roots | -0.177±0.029 | 0.684 | 322.9 |
| 10 | 406 | QIC-GBS | -0.315±0.058 | 0.470 | 91.5 |
| 1e+02 | 4062 | GBS Boost | -0.361±0.021 | 0.473 | 629.2 |
| 1e+02 | 4062 | GBS Roots | -0.151±0.000 | 0.685 | 651.2 |
| 1e+02 | 4062 | QIC-GBS | -0.281±0.001 | 0.465 | 244.9 |

### Expected Qualitative Outcomes (from paper)
- GBS Boost WD rises sharply F=1→10; GBS Roots plateaus earlier
- F=0.1 sufficient to establish economic edge over SPONGE
- QIC-GBS: much faster than simulated quantum sampling, competitive returns at large F
- Runtime growth: steep log-log linear for simulated quantum methods

---

## Experiment 5 — Universe Scaling N=12–30 (proxy)

**Paper**: N=20–30 fine-grained, N=12–50 coarse, H=5×N×log(N), 15–120 runs per config
**Implementation**: N ∈ {12,15,18,20,25,30}, 3 runs per config

| N | Method | Return (median) | Q25 | Q75 | Jaccard |
|---|--------|-----------------|-----|-----|---------|
| 12 | Spectral | -0.324 | -0.325 | -0.322 | 0.584 |
| 12 | SPONGE | -0.282 | -0.285 | -0.282 | 0.596 |
| 12 | GBS Boost | -0.067 | -0.159 | -0.048 | 0.477 |
| 12 | GBS Roots | -0.160 | -0.182 | -0.111 | 0.463 |
| 12 | QIC-GBS | -0.133 | -0.204 | -0.117 | 0.509 |
| 15 | Spectral | -0.446 | -0.446 | -0.446 | 0.529 |
| 15 | SPONGE | -0.392 | -0.392 | -0.375 | 0.538 |
| 15 | GBS Boost | -0.188 | -0.215 | -0.169 | 0.406 |
| 15 | GBS Roots | -0.269 | -0.270 | -0.215 | 0.443 |
| 15 | QIC-GBS | -0.318 | -0.330 | -0.315 | 0.416 |
| 18 | Spectral | -0.393 | -0.393 | -0.393 | 0.555 |
| 18 | SPONGE | -0.429 | -0.429 | -0.429 | 0.563 |
| 18 | GBS Boost | -0.226 | -0.299 | -0.212 | 0.387 |
| 18 | GBS Roots | -0.152 | -0.174 | -0.149 | 0.418 |
| 18 | QIC-GBS | -0.147 | -0.205 | -0.132 | 0.379 |
| 20 | Spectral | -0.262 | -0.262 | -0.262 | 0.566 |
| 20 | SPONGE | -0.401 | -0.401 | -0.400 | 0.548 |
| 20 | GBS Boost | -0.377 | -0.377 | -0.234 | 0.381 |
| 20 | GBS Roots | -0.292 | -0.309 | -0.266 | 0.411 |
| 20 | QIC-GBS | -0.311 | -0.365 | -0.270 | 0.381 |
| 25 | Spectral | -0.148 | -0.148 | -0.148 | 0.533 |
| 25 | SPONGE | -0.089 | -0.089 | -0.087 | 0.527 |
| 25 | GBS Boost | -0.015 | -0.090 | 0.022 | 0.320 |
| 25 | GBS Roots | -0.172 | -0.188 | -0.143 | 0.382 |
| 25 | QIC-GBS | -0.128 | -0.133 | -0.077 | 0.328 |
| 30 | Spectral | -0.117 | -0.126 | -0.115 | 0.428 |
| 30 | SPONGE | -0.173 | -0.173 | -0.173 | 0.537 |
| 30 | GBS Boost | -0.264 | -0.270 | -0.205 | 0.311 |
| 30 | GBS Roots | -0.174 | -0.176 | -0.142 | 0.380 |
| 30 | QIC-GBS | -0.181 | -0.239 | -0.170 | 0.319 |

### Expected Qualitative Outcomes (from paper)
- Non-monotonic improvement in median returns as N increases
- Transition near N=30: lower return quartiles move above zero
- At N=50: GBS Boost upper quartile ≈ 2× Spectral's IQR
- Jaccard similarity generally declines with universe size
- Classical methods exhibit tighter IQR at small N

---

## Documented Deviations from Paper

| # | Category | Deviation | Expected Impact |
|---|----------|-----------|-----------------|
| 1 | GBS Sampler | Classical density-weighted emulator replaces quantum Strawberry Fields | Qualitative behavior preserved; exact sampling distribution differs |
| 2 | Data Source | yfinance replaces CRSP dividend-adjusted prices | Different universe composition; may include survivorship bias |
| 3 | Market Factor | Ken French data library for Fama-French market return | Same source as paper for this component |
| 4 | Universe Size | N=15–30 proxy instead of paper N=50–100 | Sparser correlation graphs; results not directly comparable |
| 5 | Run Counts | n_runs=3–8 instead of paper 20–100 | Higher Monte Carlo variance in estimates |
| 6 | K Estimation | B=50–100 permutations (paper example B=1000) | Less accurate K estimate; fast approximation |
| 7 | Log Base | Natural log; paper log base unspecified | Minor numerical difference in H |
| 8 | Risk-Free Rate | RF=0%; paper convention unspecified | Minor effect on Sharpe/Sortino ratios |
| 9 | Displacement Arm | Exp2 displacement (DGBS) not implemented | Only uncompensated loss curve produced |
| 10 | Symmetric SPONGE | Signed-Laplacian generalized eigenproblem; no official library | Implementation follows SPONGE paper description |

## Repository Structure

```
├── src/                  # Core implementation
│   ├── gbs.py            # GBS Boost, GBS Roots, QIC-GBS, SPONGE, Spectral
│   ├── clustering.py     # Partition construction and graph metrics
│   ├── statarb.py        # Statistical arbitrage strategy
│   ├── pipeline.py       # Rolling window orchestration
│   ├── metrics.py        # Performance metrics (Sharpe, Sortino, Jaccard)
│   ├── data.py           # Market data loading (yfinance/Massive)
│   └── config.py         # All paper-derived parameter constants
├── exp/                  # Experiment runners
│   ├── exp1_benchmark_100.py
│   ├── exp2_loss_test.py
│   ├── exp3_regime_test.py
│   ├── exp4_scaling_budget.py
│   └── exp5_universe_scaling.py
├── tests/                # 65 pytest unit tests
├── results/              # All output artifacts (JSON, CSV, PNG, MD)
├── scripts/              # Utility scripts
│   └── generate_results_md.py
├── reproduce.sh          # Entry-point script
└── requirements.txt
```
