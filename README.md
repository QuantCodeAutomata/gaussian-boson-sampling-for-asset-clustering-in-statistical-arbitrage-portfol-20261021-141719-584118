# GBS Statistical Arbitrage Portfolio Clustering

**Paper replication**: "Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios"

## Overview

This repository replicates five experiments from the paper comparing GBS-based and classical clustering methods for statistical arbitrage portfolio construction:

- **Exp 1**: Lossless 100-stock benchmark in 2020 (N=30 proxy due to compute constraints)
- **Exp 2**: Photon-loss stress test for 50-stock universe (N=25 proxy)
- **Exp 3**: Multi-regime comparison across 2008, 2017, 2020, 2022 (N=20 proxy)
- **Exp 4**: GBS sampling-budget scaling (N=15 proxy)
- **Exp 5**: Universe scaling from N=12 to N=50 (N=12–30 proxy)

## Key Implementation Notes

### GBS Emulation
The paper uses Strawberry Fields for exact quantum simulation. The `strawberryfields.apps` module has import issues in the current environment. We use a classical **density-weighted GBS emulator** that:
- Samples click patterns with probability biased toward dense subgraphs
- Preserves the qualitative behavior of GBS (preferential sampling of correlated subsets)
- Is documented as a deviation from the paper's exact methodology

### Data
- Paper uses CRSP via Wharton Research Data Services (proprietary)
- This implementation uses **yfinance** for S&P 500 stocks (publicly available)
- Fama-French market factor from Ken French's data library

### Reduced Parameters
Due to compute constraints:
- Stock universe N reduced from 50–100 to 15–30
- Run counts reduced from 20–100 to 3–8
- Results show qualitative trends, not exact paper values

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run all experiments
bash reproduce.sh

# Run individual experiments
python exp/exp1_benchmark_100.py
python exp/exp2_loss_test.py
python exp/exp3_regime_test.py
python exp/exp4_scaling_budget.py
python exp/exp5_universe_scaling.py

# Run tests
pytest tests/ -v
```

## Results

See `results/RESULTS.md` for a summary of outputs and comparison with paper values.

## Repository Structure

```
├── src/                    # Core algorithms
│   ├── config.py           # Configuration parameters
│   ├── data.py             # Data loading and preprocessing
│   ├── gbs.py              # GBS algorithms (Boost, Roots, QIC-GBS)
│   ├── clustering.py       # Spectral, SPONGE clustering
│   ├── statarb.py          # Statistical arbitrage strategy
│   ├── metrics.py          # Performance metrics
│   └── pipeline.py         # Shared experiment pipeline
├── exp/                    # Experiment scripts
├── tests/                  # Pytest test suite
├── results/                # Outputs (plots, tables, metrics)
│   └── RESULTS.md
├── data/                   # Data directory (empty — data downloaded at runtime)
├── reproduce.sh            # Entry point
└── README.md
```

## Deviations from Paper

| Deviation | Impact | Status |
|-----------|--------|--------|
| Classical GBS emulator instead of quantum SF | Qualitative behavior preserved | DOCUMENTED |
| yfinance instead of CRSP | Different universe, potential data differences | DOCUMENTED |
| N reduced (15–30 vs 50–100) | Results not directly comparable | DOCUMENTED |
| n_runs reduced (3–8 vs 20–100) | Higher variance in results | DOCUMENTED |
| Displacement arm in Exp2 not implemented | Exp2 only shows uncompensated loss | MISSING |
| B=50–100 permutations vs paper B=1000 | K estimation less accurate | DOCUMENTED |
| log base = natural log (paper unspecified) | Minor parameter difference | ASSUMED |
| Risk-free rate = 0 (paper unspecified) | Minor effect on Sharpe | ASSUMED |
