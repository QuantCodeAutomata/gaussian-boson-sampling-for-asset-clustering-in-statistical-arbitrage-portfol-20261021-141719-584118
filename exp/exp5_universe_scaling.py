"""
Experiment 5: Universe scaling N=12 to 30, evaluating return IQR and Jaccard.

Paper Appendix J: Test how universe size affects return distributions and
cluster persistence.

ASSUMPTION [reduced-params]: N from 12 to 20 in steps, n_runs=3.
ASSUMPTION [nested]: Stocks added greedily maintaining sector proportionality.
"""
import logging, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ExperimentConfig, exp5_config
from src.pipeline import load_and_prepare_data, run_all_methods

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    os.makedirs("results", exist_ok=True)

    n_sizes = [12, 15, 18, 20, 25, 30]  # ASSUMPTION: reduced grid (paper: 12-50)
    methods = ["spectral", "sponge", "gbs_boost", "gbs_roots", "qic_gbs"]
    n_runs_gbs = 3
    n_runs_cls = 5

    scaling_data = {}

    for N in n_sizes:
        logger.info("=== N=%d ===", N)
        cfg = ExperimentConfig(
            n_stocks=N, eval_year=2020, photon_loss=0.0,
            n_runs_gbs=n_runs_gbs, n_runs_classical=n_runs_cls,
            k_perm_replicates=20,
            F_boost=5.0, F_roots=5.0, F_qic=5.0,  # H=5*N*log(N) per paper exp5
        )
        try:
            residuals, total_returns, market_returns, tickers = load_and_prepare_data(cfg)
            res = run_all_methods(cfg, residuals, total_returns, market_returns, methods)
            scaling_data[N] = {}
            for method in methods:
                runs = res.get(method, [])
                if not runs: continue
                rets  = [r["metrics"].get("annual_return", 0) for r in runs if "metrics" in r]
                jacs  = [r["metrics"].get("mean_jaccard", 0) for r in runs if "metrics" in r
                         and not np.isnan(r["metrics"].get("mean_jaccard", float("nan")))]
                scaling_data[N][method] = {
                    "return_median": float(np.median(rets)),
                    "return_q25":    float(np.percentile(rets, 25)),
                    "return_q75":    float(np.percentile(rets, 75)),
                    "jaccard_mean":  float(np.mean(jacs)) if jacs else None,
                }
        except Exception as e:
            logger.error("N=%d failed: %s", N, e)

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"spectral": "blue", "sponge": "orange", "gbs_boost": "green",
              "gbs_roots": "red", "qic_gbs": "purple"}

    for method in methods:
        xs = [N for N in n_sizes if N in scaling_data and method in scaling_data[N]]
        if not xs: continue
        medians = [scaling_data[N][method]["return_median"] for N in xs]
        q25     = [scaling_data[N][method]["return_q25"] for N in xs]
        q75     = [scaling_data[N][method]["return_q75"] for N in xs]
        jacs    = [scaling_data[N][method].get("jaccard_mean") for N in xs]
        c = colors[method]
        axes[0].plot(xs, medians, marker="o", color=c, label=method, linewidth=2)
        axes[0].fill_between(xs, q25, q75, color=c, alpha=0.15)
        valid_jac = [(x, j) for x, j in zip(xs, jacs) if j is not None]
        if valid_jac:
            axes[1].plot([v[0] for v in valid_jac], [v[1] for v in valid_jac],
                         marker="s", color=c, label=method, linewidth=2)

    axes[0].set_xlabel("Universe Size N"); axes[0].set_ylabel("Median Annual Return")
    axes[0].set_title("Exp5: Return vs Universe Size"); axes[0].legend(fontsize=8)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Universe Size N"); axes[1].set_ylabel("Mean Jaccard Similarity")
    axes[1].set_title("Exp5: Cluster Stability vs Universe Size"); axes[1].legend(fontsize=8)
    plt.suptitle("Exp5: Universe Scaling Analysis")
    plt.tight_layout()
    plt.savefig("results/exp5_scaling.png", dpi=150)
    plt.close()

    output = {
        "scaling_data": {str(N): {m: {k: (float(v) if isinstance(v, float) else v)
                                      for k, v in mv.items()}
                                  for m, mv in mv2.items()}
                         for N, mv2 in scaling_data.items()},
        "assumptions": [
            "N sizes [12,15,18,20,25,30] instead of paper 12-50 (compute constraint)",
            "Non-nested universe: each N independently sampled",
            "n_runs=3 (GBS), 5 (classical)",
            "F=5 (H=5*N*log(N)) per paper exp5 specification",
        ],
    }
    with open("results/exp5_results.json", "w") as f:
        json.dump(output, f, indent=2, default=lambda x: None if x != x else x)
    logger.info("Saved results/exp5_*.json and .png")
    return output


if __name__ == "__main__":
    main()
