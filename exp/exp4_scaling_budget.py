"""
Experiment 4: GBS sampling-budget scaling study, N=30.

Paper Appendix J: Vary H = F*N*log(N) over F in {0.01, 0.1, 1, 10, 100}.
Measure weighted density, total return, and runtime vs sample factor F.

ASSUMPTION [reduced-params]: N=15, F=[0.1, 1, 10], n_runs=3 for compute constraint.
ASSUMPTION [runtime]: Wall-clock time measures classical GBS emulation, not quantum hardware.
"""
import logging, json, os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ExperimentConfig
from src.pipeline import load_and_prepare_data, run_all_methods, make_partition_fn
from src.gbs import compute_H, gbs_boost_partition, gbs_roots_partition, qic_gbs_partition
from src.clustering import spectral_clustering, sponge_clustering
from src.statarb import run_backtest
from src.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import math


def main():
    os.makedirs("results", exist_ok=True)

    # ASSUMPTION [reduced-params]: N=15, F-values subset
    sample_factors = [0.1, 1.0, 10.0, 100.0]
    gbs_methods = ["gbs_boost", "gbs_roots", "qic_gbs"]
    n_runs = 3

    cfg_base = ExperimentConfig(
        n_stocks=15, eval_year=2020, photon_loss=0.0,
        n_runs_gbs=n_runs, n_runs_classical=5,
        k_perm_replicates=20,
    )
    residuals, total_returns, market_returns, tickers = load_and_prepare_data(cfg_base)

    # Classical horizontal benchmarks (independent of F)
    from src.pipeline import estimate_K_for_method
    k_res = residuals.iloc[:cfg_base.k_est_window] if len(residuals) >= cfg_base.k_est_window else residuals

    classical_results = {}
    for method in ["spectral", "sponge"]:
        cfg_base.n_runs_classical = 5
        runs = []
        for i in range(5):
            seed = i * 1000 + abs(hash(method)) % 1000
            K = estimate_K_for_method(method, k_res, cfg_base, seed)
            pfn = make_partition_fn(method, cfg_base, seed, K_override=K)
            try:
                r = run_backtest(residuals=residuals, total_returns=total_returns,
                                 market_returns=market_returns, partition_fn=pfn,
                                 corr_window=cfg_base.corr_window, beta_window=cfg_base.beta_window,
                                 k_est_window=cfg_base.k_est_window, holding_days=cfg_base.holding_days,
                                 stop_win=cfg_base.stop_win, seed=seed)
                r["metrics"] = compute_metrics(r["daily_returns"], r["wds"],
                                               r["cluster_values"], r["jaccard_similarities"])
                runs.append(r)
            except Exception as e:
                logger.error("%s run %d: %s", method, i, e)
        classical_results[method] = runs

    # GBS methods with varying F
    scaling_results = {}
    for F in sample_factors:
        logger.info("=== F=%.2f ===", F)
        scaling_results[F] = {}
        cfg = ExperimentConfig(
            n_stocks=15, eval_year=2020, photon_loss=0.0,
            n_runs_gbs=n_runs, n_runs_classical=5,
            F_boost=F, F_roots=F, F_qic=F,
            k_perm_replicates=20,
        )
        for method in gbs_methods:
            runs = []
            t_start = time.perf_counter()
            for i in range(n_runs):
                seed = i * 1000 + abs(hash(method)) % 1000
                pfn = make_partition_fn(method, cfg, seed, loss_rate=0.0)
                try:
                    r = run_backtest(residuals=residuals, total_returns=total_returns,
                                     market_returns=market_returns, partition_fn=pfn,
                                     corr_window=cfg.corr_window, beta_window=cfg.beta_window,
                                     k_est_window=cfg.k_est_window, holding_days=cfg.holding_days,
                                     stop_win=cfg.stop_win, seed=seed)
                    r["metrics"] = compute_metrics(r["daily_returns"], r["wds"],
                                                   r["cluster_values"], r["jaccard_similarities"])
                    runs.append(r)
                except Exception as e:
                    logger.error("%s F=%.1f run %d: %s", method, F, i, e)
            elapsed = time.perf_counter() - t_start
            H_val = compute_H(F, 15)
            logger.info("  %s F=%.2f H=%d runtime=%.2fs", method, F, H_val, elapsed)
            scaling_results[F][method] = {"runs": runs, "runtime_s": elapsed, "H": H_val}

    # ── Aggregate ─────────────────────────────────────────────────────────
    metrics_by_F = {}
    for F in sample_factors:
        metrics_by_F[F] = {}
        for method in gbs_methods:
            sr = scaling_results[F].get(method, {})
            runs = sr.get("runs", [])
            if not runs: continue
            rets = [r["metrics"].get("annual_return", 0) for r in runs if "metrics" in r]
            wds  = [r["metrics"].get("mean_weighted_density", 0) for r in runs if "metrics" in r]
            metrics_by_F[F][method] = {
                "annual_return_mean": float(np.mean(rets)),
                "annual_return_std":  float(np.std(rets)),
                "weighted_density_mean": float(np.mean(wds)),
                "runtime_s": sr.get("runtime_s", 0),
                "H": sr.get("H", 0),
            }

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"gbs_boost": "green", "gbs_roots": "red", "qic_gbs": "purple"}

    for method in gbs_methods:
        xs = [F for F in sample_factors if method in metrics_by_F.get(F, {})]
        ys_ret = [metrics_by_F[F][method]["annual_return_mean"] for F in xs]
        ys_wd  = [metrics_by_F[F][method]["weighted_density_mean"] for F in xs]
        ys_rt  = [metrics_by_F[F][method]["runtime_s"] for F in xs]
        c = colors[method]
        if xs:
            axes[0].semilogx(xs, ys_ret, marker="o", color=c, label=method)
            axes[1].semilogx(xs, ys_wd, marker="s", color=c, label=method)
            axes[2].loglog(xs, ys_rt, marker="^", color=c, label=method)

    for method, runs in classical_results.items():
        if not runs: continue
        rets = [r["metrics"].get("annual_return", 0) for r in runs if "metrics" in r]
        axes[0].axhline(np.mean(rets), linestyle="--", label=method, alpha=0.7)

    axes[0].set_xlabel("Sample Factor F"); axes[0].set_ylabel("Annual Return")
    axes[0].set_title("Return vs F"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Sample Factor F"); axes[1].set_ylabel("Weighted Density")
    axes[1].set_title("WD vs F"); axes[1].legend(fontsize=8)
    axes[2].set_xlabel("Sample Factor F"); axes[2].set_ylabel("Runtime (s)")
    axes[2].set_title("Runtime vs F"); axes[2].legend(fontsize=8)
    plt.suptitle("Exp4: GBS Sampling Budget Scaling (N=15 proxy)")
    plt.tight_layout()
    plt.savefig("results/exp4_scaling.png", dpi=150)
    plt.close()

    output = {
        "metrics_by_F": {str(F): {m: {k: float(v) if isinstance(v, float) else v for k, v in mv.items()}
                                    for m, mv in mv2.items()}
                          for F, mv2 in metrics_by_F.items()},
        "assumptions": [
            "N=15 instead of paper N=30 (compute constraint)",
            "F subset [0.1, 1, 10, 100] instead of [0.01, 0.1, 1, 10, 100]",
            "n_runs=3 per configuration",
            "Runtime measures classical emulation, not quantum hardware",
        ],
    }
    with open("results/exp4_results.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved results/exp4_*.json and .png")
    return output


if __name__ == "__main__":
    main()
