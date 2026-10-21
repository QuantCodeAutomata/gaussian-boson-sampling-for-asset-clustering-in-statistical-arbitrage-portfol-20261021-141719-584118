"""
Experiment 1: Lossless 100-stock statistical-arbitrage benchmark, 2020.

Paper Section IV.A: Compare GBS Boost, GBS Roots, Spectral, symmetric SPONGE,
and QIC-GBS in a lossless N=100 stock universe using rolling residual-correlation
graphs to construct market-neutral contrarian portfolios.

Expected paper results (Table I):
  Spectral:  return=0.200±0.051, Sharpe=2.035±0.512
  SPONGE:    return=0.214±0.064, Sharpe=2.242±0.669
  GBS Boost: return=0.239±0.035, Sharpe=2.050±0.190
  GBS Roots: return=0.236±0.031, Sharpe=2.248±0.293
  QIC-GBS:   return=0.195±0.099, Sharpe=1.781±0.893

ASSUMPTION [reduced-params]: Running with N=30 stocks and n_runs=5 (GBS), 8 (classical)
  due to compute constraints. Paper uses N=100, ~20 GBS runs, ~100 classical runs.
ASSUMPTION [universe]: yfinance-based universe approximates CRSP data.
"""
import logging
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ExperimentConfig
from src.pipeline import load_and_prepare_data, run_all_methods, compile_summary_table
from src.metrics import welch_t_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    os.makedirs("results", exist_ok=True)

    # ASSUMPTION [reduced-params]: Use N=30 for practical compute time
    cfg = ExperimentConfig(
        n_stocks=30, eval_year=2020, photon_loss=0.0,
        n_runs_gbs=5, n_runs_classical=8,
        k_perm_replicates=50,
    )
    logger.info("Exp1: N=%d, year=%d, GBS runs=%d, classical runs=%d",
                cfg.n_stocks, cfg.eval_year, cfg.n_runs_gbs, cfg.n_runs_classical)

    residuals, total_returns, market_returns, tickers = load_and_prepare_data(cfg)

    methods = ["spectral", "sponge", "gbs_boost", "gbs_roots", "qic_gbs"]
    all_results = run_all_methods(cfg, residuals, total_returns, market_returns, methods)

    # ── Summary table ──────────────────────────────────────────────────────
    table = compile_summary_table(all_results)
    logger.info("\nSummary Table:\n%s", table.to_string())
    table.to_csv("results/exp1_summary_table.csv")

    # ── Welch t-test: GBS Roots vs SPONGE ─────────────────────────────────
    roots_returns = [r["metrics"]["annual_return"] for r in all_results.get("gbs_roots", []) if "metrics" in r]
    sponge_returns = [r["metrics"]["annual_return"] for r in all_results.get("sponge", []) if "metrics" in r]
    roots_sharpes  = [r["metrics"]["sharpe_ratio"]  for r in all_results.get("gbs_roots", []) if "metrics" in r]
    sponge_sharpes = [r["metrics"]["sharpe_ratio"]  for r in all_results.get("sponge", []) if "metrics" in r]

    welch_return = welch_t_test(roots_returns, sponge_returns) if (roots_returns and sponge_returns) else {}
    welch_sharpe = welch_t_test(roots_sharpes, sponge_sharpes) if (roots_sharpes and sponge_sharpes) else {}
    logger.info("Welch return test (GBS Roots vs SPONGE): %s", welch_return)
    logger.info("Welch Sharpe test (GBS Roots vs SPONGE): %s", welch_sharpe)

    # ── Figure: cumulative returns ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"spectral": "blue", "sponge": "orange", "gbs_boost": "green",
              "gbs_roots": "red", "qic_gbs": "purple"}
    for method, runs in all_results.items():
        if not runs: continue
        for run in runs:
            pnl = run["cumulative_pnl"]
            ax.plot(range(len(pnl)), pnl, color=colors.get(method, "grey"), alpha=0.3, linewidth=0.8)
        # Mean line
        min_len = min(len(r["cumulative_pnl"]) for r in runs)
        mean_pnl = np.mean([r["cumulative_pnl"][:min_len] for r in runs], axis=0)
        ax.plot(range(min_len), mean_pnl, color=colors.get(method, "grey"),
                linewidth=2, label=method)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Exp1: Cumulative Returns — Lossless 2020 Benchmark (N=30 proxy)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("results/exp1_cumulative_returns.png", dpi=150)
    plt.close()
    logger.info("Saved results/exp1_cumulative_returns.png")

    # ── Save JSON results ─────────────────────────────────────────────────
    summary_dict = {}
    for method, runs in all_results.items():
        if not runs: continue
        metrics_list = [r["metrics"] for r in runs if "metrics" in r]
        if not metrics_list: continue
        returns = [m["annual_return"] for m in metrics_list]
        sharpes = [m["sharpe_ratio"]  for m in metrics_list]
        wds     = [m.get("mean_weighted_density", 0) for m in metrics_list]
        jacs    = [m.get("mean_jaccard", 0) for m in metrics_list if not np.isnan(m.get("mean_jaccard", float("nan")))]
        summary_dict[method] = {
            "annual_return_mean": float(np.mean(returns)),
            "annual_return_std":  float(np.std(returns)),
            "sharpe_ratio_mean":  float(np.mean(sharpes)),
            "sharpe_ratio_std":   float(np.std(sharpes)),
            "weighted_density_mean": float(np.mean(wds)),
            "jaccard_mean":       float(np.mean(jacs)) if jacs else None,
            "n_runs": len(runs),
        }
    summary_dict["welch_return_roots_vs_sponge"] = welch_return
    summary_dict["welch_sharpe_roots_vs_sponge"] = welch_sharpe
    summary_dict["assumptions"] = [
        "N=30 used instead of paper N=100 (compute constraint)",
        "yfinance replaces CRSP",
        "Classical GBS emulator replaces quantum Strawberry Fields",
        "n_runs_gbs=5 instead of paper 20; n_runs_classical=8 instead of 100",
    ]

    with open("results/exp1_results.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
    logger.info("Saved results/exp1_results.json")
    logger.info("\nExp1 DONE. Summary:\n%s",
                json.dumps({k: {kk: f"{vv:.4f}" if isinstance(vv, float) else vv
                               for kk, vv in v.items()}
                           for k, v in summary_dict.items() if isinstance(v, dict)}, indent=2))
    return summary_dict


if __name__ == "__main__":
    main()
