#!/usr/bin/env bash
# Entry point: run all GBS StatArb experiments
# Paper: "Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios"

set -o pipefail

echo "====================================================================="
echo "GBS Statistical Arbitrage Portfolio Clustering — Paper Replication"
echo "====================================================================="
echo ""

# Ensure results directory exists
mkdir -p results

# Set up Python path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ── Run tests first ──────────────────────────────────────────────────────────
echo "--- Running unit tests ---"
python -m pytest tests/ -v --tb=short -q 2>&1 | tee results/test_output.txt
TEST_STATUS=$?
echo "Test exit status: $TEST_STATUS"
echo ""

# ── Run all experiments ──────────────────────────────────────────────────────
echo "--- Experiment 1: Lossless 100-stock 2020 benchmark (N=30 proxy) ---"
python exp/exp1_benchmark_100.py 2>&1 | tee results/exp1_log.txt
echo ""

echo "--- Experiment 2: Photon-loss stress test (N=25 proxy) ---"
python exp/exp2_loss_test.py 2>&1 | tee results/exp2_log.txt
echo ""

echo "--- Experiment 3: Multi-regime 2008/2017/2020/2022 (N=20 proxy) ---"
python exp/exp3_regime_test.py 2>&1 | tee results/exp3_log.txt
echo ""

echo "--- Experiment 4: Sampling budget scaling (N=15 proxy) ---"
python exp/exp4_scaling_budget.py 2>&1 | tee results/exp4_log.txt
echo ""

echo "--- Experiment 5: Universe scaling N=12-30 ---"
python exp/exp5_universe_scaling.py 2>&1 | tee results/exp5_log.txt
echo ""

# ── Generate RESULTS.md ───────────────────────────────────────────────────────
python - << 'PYEOF'
import json, os, sys

lines = [
    "# Results Summary",
    "",
    "Paper: *Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios*",
    "",
    "**Note**: Results use reduced parameters (N=15–30 vs paper N=50–100; n_runs=3–8 vs paper 20–100).",
    "Classical GBS emulator replaces quantum Strawberry Fields sampling.",
    "Data source: yfinance (paper uses CRSP).",
    "",
]

for exp_id in range(1, 6):
    path = f"results/exp{exp_id}_results.json"
    lines.append(f"## Experiment {exp_id}")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        lines.append("```json")
        lines.append(json.dumps(data, indent=2, default=str)[:3000])
        lines.append("```")
    else:
        lines.append(f"*Results not found at {path}*")
    lines.append("")

lines += [
    "## Documented Deviations from Paper",
    "",
    "| # | Deviation | Impact |",
    "|---|-----------|--------|",
    "| 1 | Classical GBS density-weighted emulator (not exact quantum SF) | Qualitative behavior preserved |",
    "| 2 | yfinance instead of CRSP | Different universe composition |",
    "| 3 | N=15–30 instead of paper N=50–100 | Results not directly comparable |",
    "| 4 | n_runs=3–8 instead of paper 20–100 | Higher variance in estimates |",
    "| 5 | Displacement arm (Exp2) not implemented | Only uncompensated loss tested |",
    "| 6 | B=50–100 permutations for K estimation (paper B=1000) | Less accurate K |",
    "| 7 | Natural log (paper log base unspecified) | Minor numerical difference |",
    "| 8 | Risk-free rate = 0 (paper unspecified) | Minor Sharpe effect |",
    "| 9 | stop-win 2%, 3-day hold, equal weights within legs | Per paper |",
    "",
]

with open("results/RESULTS.md", "w") as f:
    f.write("\n".join(lines))
print("Generated results/RESULTS.md")
PYEOF

echo ""
echo "====================================================================="
echo "All experiments complete. See results/ directory for outputs."
echo "====================================================================="
