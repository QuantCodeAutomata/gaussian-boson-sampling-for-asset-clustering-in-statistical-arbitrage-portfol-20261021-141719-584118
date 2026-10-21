"""
Experiment 2: Photon-loss stress test, N=50 stocks.

Paper Section IV.B: Vary photon loss l_r from 0 to 0.9, measure performance
of GBS methods vs classical benchmarks and random baseline.

ASSUMPTION [reduced-params]: N=25, loss_rates=[0, 0.3, 0.6, 0.9], n_runs=3.
ASSUMPTION [no-displacement]: Coherent displacement not implemented here;
  would require explicit Gaussian-state circuit manipulation via Strawberry Fields.
  The displacement arm is documented as MISSING and the deviation is recorded.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    os.makedirs("results", exist_ok=True)

    # ASSUMPTION [reduced-params]: N=25, loss_rates subset, n_runs=3
    cfg = ExperimentConfig(
        n_stocks=25, eval_year=2020, photon_loss=0.0,
        n_runs_gbs=3, n_runs_classical=5,
        k_perm_replicates=30,
    )
    # Loss rates to test (subset for speed; paper uses 0, 0.1, ..., 0.9)
    loss_rates = [0.0, 0.3, 0.6, 0.9]
    methods = ["spectral", "sponge", "gbs_boost", "gbs_roots", "qic_gbs", "random"]

    logger.info("Exp2: N=%d, loss_rates=%s", cfg.n_stocks, loss_rates)
    residuals, total_returns, market_returns, tickers = load_and_prepare_data(cfg)

    all_loss_results = {}
    for lr in loss_rates:
        logger.info("--- Loss rate %.1f ---", lr)
        cfg.photon_loss = lr
        res = run_all_methods(cfg, residuals, total_returns, market_returns,
                              methods, loss_rate=lr)
        all_loss_results[lr] = res

    # ── Aggregate metrics ─────────────────────────────────────────────────
    metrics_by_loss = {}
    for lr, res in all_loss_results.items():
        metrics_by_loss[lr] = {}
        for method, runs in res.items():
            if not runs: continue
            returns = [r["metrics"].get("annual_return", 0) for r in runs if "metrics" in r]
            wds     = [r["metrics"].get("mean_weighted_density", 0) for r in runs if "metrics" in r]
            metrics_by_loss[lr][method] = {
                "annual_return_mean": float(np.mean(returns)) if returns else None,
                "annual_return_std":  float(np.std(returns)) if returns else None,
                "weighted_density_mean": float(np.mean(wds)) if wds else None,
            }

    # ── Plot: return vs loss rate ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"spectral": "blue", "sponge": "orange", "gbs_boost": "green",
              "gbs_roots": "red", "qic_gbs": "purple", "random": "grey"}
    for method in methods:
        returns_by_lr = [metrics_by_loss.get(lr, {}).get(method, {}).get("annual_return_mean") for lr in loss_rates]
        stds_by_lr    = [metrics_by_loss.get(lr, {}).get(method, {}).get("annual_return_std", 0) for lr in loss_rates]
        valid = [(lr, r, s) for lr, r, s in zip(loss_rates, returns_by_lr, stds_by_lr) if r is not None]
        if not valid: continue
        xs, ys, ss = zip(*valid)
        axes[0].errorbar(xs, ys, yerr=ss, label=method, color=colors.get(method, "grey"),
                         marker="o", linewidth=2)

        wds_by_lr = [metrics_by_loss.get(lr, {}).get(method, {}).get("weighted_density_mean") for lr in loss_rates]
        valid_wd = [(lr, wd) for lr, wd in zip(loss_rates, wds_by_lr) if wd is not None]
        if valid_wd:
            xs2, ys2 = zip(*valid_wd)
            axes[1].plot(xs2, ys2, label=method, color=colors.get(method, "grey"), marker="s", linewidth=2)

    axes[0].set_xlabel("Photon Loss Rate"); axes[0].set_ylabel("Annual Return")
    axes[0].set_title("Exp2: Return vs Photon Loss (N=25)"); axes[0].legend()
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Photon Loss Rate"); axes[1].set_ylabel("Weighted Density")
    axes[1].set_title("Exp2: Weighted Density vs Photon Loss (N=25)"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("results/exp2_loss_vs_return.png", dpi=150)
    plt.close()

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "metrics_by_loss": {str(lr): {m: {k: (float(v) if isinstance(v, float) else v)
                                          for k, v in mv.items()}
                                       for m, mv in mv2.items()}
                            for lr, mv2 in metrics_by_loss.items()},
        "deviations": ["Displacement arm NOT implemented — requires Strawberry Fields Gaussian-state simulation"],
        "assumptions": [
            "N=25 used instead of paper N=50 (compute constraint)",
            "loss_rates=[0,0.3,0.6,0.9] subset instead of 0..0.9 in 0.1 steps",
            "n_runs=3 (GBS), 5 (classical) instead of paper 50/100-200",
        ],
    }
    with open("results/exp2_results.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved results/exp2_results.json and results/exp2_loss_vs_return.png")
    return output


if __name__ == "__main__":
    main()
