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
echo "--- Generating RESULTS.md ---"
python scripts/generate_results_md.py

echo ""
echo "====================================================================="
echo "All experiments complete. See results/ directory for outputs."
echo "====================================================================="
