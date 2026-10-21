"""
Experiment 3: Lossless 50-stock comparison across 2008, 2017, 2020, 2022.

Paper Section IV.C: Compare GBS and classical methods in crash, bull, dynamic,
and bear market regimes. Benchmark against 1/N equal-weighted portfolio.

Expected paper returns:
  2008: benchmark=-0.48, Spectral=0.17, SPONGE=0.30, GBS Boost=0.38, GBS Roots=0.39
  2017: benchmark=0.29,  Spectral=0.11, SPONGE=0.08, GBS Boost=0.06, GBS Roots=0.06
  2020: benchmark=0.50,  Spectral=0.22, SPONGE=0.21, GBS Boost=0.31, GBS Roots=0.23
  2022: benchmark=0.00,  Spectral=0.17, SPONGE=0.13, GBS Boost=0.08, GBS Roots=0.14

ASSUMPTION [reduced-params]: N=20, n_runs=3, years=[2020, 2022] first (2008/2017 optional).
"""
import logging, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ExperimentConfig
from src.pipeline import load_and_prepare_data, run_all_methods, compile_summary_table
from src.data import load_vix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def equal_weight_return(total_returns: pd.DataFrame, eval_year: int) -> float:
    """Compute 1/N buy-and-hold return over eval year."""
    mask = total_returns.index.year == eval_year
    yr = total_returns.loc[mask]
    weights = np.ones(yr.shape[1]) / yr.shape[1]
    # Compound daily
    port_ret = yr.fillna(0).values @ weights
    return float(np.prod(1 + port_ret) - 1)


def main():
    os.makedirs("results", exist_ok=True)

    # ASSUMPTION [reduced-params]: N=20, n_runs=3, all 4 years
    years = [2008, 2017, 2020, 2022]
    methods = ["spectral", "sponge", "gbs_boost", "gbs_roots", "qic_gbs"]

    regime_results = {}
    vix_avgs = {}

    for year in years:
        logger.info("=== Regime Year: %d ===", year)
        cfg = ExperimentConfig(
            n_stocks=20, eval_year=year, photon_loss=0.0,
            n_runs_gbs=3, n_runs_classical=5,
            k_perm_replicates=30,
        )
        try:
            residuals, total_returns, market_returns, tickers = load_and_prepare_data(cfg)
            res = run_all_methods(cfg, residuals, total_returns, market_returns, methods)
            bm_ret = equal_weight_return(total_returns, year)
            regime_results[year] = {"methods": res, "benchmark_return": bm_ret}
            logger.info("Year %d benchmark return: %.3f", year, bm_ret)

            # VIX average
            try:
                vix = load_vix(f"{year}-01-01", f"{year}-12-31")
                vix_avgs[year] = float(vix.mean())
                logger.info("Year %d avg VIX: %.1f", year, vix_avgs[year])
            except Exception as e:
                logger.warning("VIX unavailable for %d: %s", year, e)
                vix_avgs[year] = None
        except Exception as e:
            logger.error("Year %d failed: %s", year, e)
            regime_results[year] = {"error": str(e)}

    # ── Summary table ──────────────────────────────────────────────────────
    rows = []
    for year in years:
        yr_res = regime_results.get(year, {})
        if "methods" not in yr_res: continue
        bm = yr_res.get("benchmark_return", None)
        row = {"year": year, "benchmark": bm, "avg_vix": vix_avgs.get(year)}
        for method in methods:
            runs = yr_res["methods"].get(method, [])
            if runs:
                rets = [r["metrics"].get("annual_return", 0) for r in runs if "metrics" in r]
                row[method] = float(np.mean(rets)) if rets else None
        rows.append(row)

    df = pd.DataFrame(rows).set_index("year")
    logger.info("\nRegime Summary:\n%s", df.to_string())
    df.to_csv("results/exp3_regime_table.csv")

    # ── Plot: returns by regime ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.12
    plot_methods = ["benchmark"] + methods
    colors = {"benchmark": "black", "spectral": "blue", "sponge": "orange",
              "gbs_boost": "green", "gbs_roots": "red", "qic_gbs": "purple"}
    for i, method in enumerate(plot_methods):
        vals = [df.loc[y, method] if y in df.index and method in df.columns else None for y in years]
        vals_plot = [v if v is not None else 0 for v in vals]
        ax.bar(x + (i - 2.5) * width, vals_plot, width, label=method, color=colors.get(method, "grey"), alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Market Regime Year")
    ax.set_ylabel("Annual Total Return")
    ax.set_title("Exp3: Returns Across Market Regimes (N=20 proxy)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/exp3_regime_returns.png", dpi=150)
    plt.close()

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "regime_returns": {str(y): {m: df.loc[y, m] if (y in df.index and m in df.columns) else None
                                    for m in ["benchmark"] + methods}
                           for y in years if y in df.index},
        "vix_averages": {str(k): v for k, v in vix_avgs.items()},
        "assumptions": [
            "N=20 used instead of paper N=50 (compute constraint)",
            "n_runs=3 (GBS), 5 (classical) instead of paper values",
            "yfinance replaces CRSP",
        ],
        "paper_reference": {
            "2008": {"spectral": 0.17, "sponge": 0.30, "gbs_boost": 0.38,
                     "gbs_roots": 0.39, "benchmark": -0.48},
            "2017": {"spectral": 0.11, "sponge": 0.08, "gbs_boost": 0.06,
                     "gbs_roots": 0.06, "benchmark": 0.29},
            "2020": {"spectral": 0.22, "sponge": 0.21, "gbs_boost": 0.31,
                     "gbs_roots": 0.23, "benchmark": 0.50},
            "2022": {"spectral": 0.17, "sponge": 0.13, "gbs_boost": 0.08,
                     "gbs_roots": 0.14, "benchmark": 0.00},
        },
    }
    with open("results/exp3_results.json", "w") as f:
        import json
        json.dump(output, f, indent=2, default=lambda x: None if x != x else x)
    logger.info("Saved results/exp3_*.json and .png")
    return output


if __name__ == "__main__":
    main()
