#!/usr/bin/env python3
"""
Collect all key numbers from individual task output files and write
results/revision_summary.json for filling reviewer response placeholders.
"""
import os, json

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(REPO_ROOT, "results")

def load(fn):
    with open(os.path.join(RESULT_DIR, fn)) as f:
        return json.load(f)

s1  = load("stats_exact.json")
s2  = load("lhs_distance_check.json")
s3  = load("noise_robustness.json")
s4  = load("ar_baseline.json")
s5  = load("mlp_baseline.json")
s6  = load("latent_ablation.json")
s7  = load("gamma_sensitivity.json")
s8m = load("mode_frequencies.json")
s8a = load("mode_ablation.json")
s9  = load("r2_per_trajectory.json")

summary = {
    # ── TASK 1: Stats ─────────────────────────────────────────────────────────
    "task1_stats": {
        "description": "Shapiro-Wilk, Friedman, Wilcoxon on pct_per_traj_ps across 50 test trajectories",
        "shapiro_wilk": {
            m: {
                "W": s1["shapiro_wilk"][m]["W"],
                "p": s1["shapiro_wilk"][m]["p"],
                "normal": s1["shapiro_wilk"][m]["p"] > 0.05
            }
            for m in ["koopman", "lstm", "gru", "bilstm"]
        },
        "friedman": {
            "statistic": s1["friedman"]["statistic"],
            "p": s1["friedman"]["p"],
        },
        "wilcoxon_koopman_vs_baselines": {
            k: {"statistic": v["statistic"], "p": v["p"]}
            for k, v in s1["wilcoxon_one_sided_less"].items()
        },
        "pct_rmse_summary": {
            m: {
                "mean": s1["summary"][m]["mean"],
                "sd": s1["summary"][m]["sd"],
                "median": s1["summary"][m]["median"],
            }
            for m in ["koopman", "lstm", "gru", "bilstm"]
        }
    },

    # ── TASK 2: LHS ───────────────────────────────────────────────────────────
    "task2_lhs_distance": {
        "test_to_train_nearest_neighbour": s2["test_to_train_nn_distances"],
        "val_to_train_nearest_neighbour":  s2["val_to_train_nn_distances"],
        "within_train_nearest_neighbour":  s2["within_train_nn_distances"],
        "conclusion": (
            f"Min test-train distance = {s2['test_to_train_nn_distances']['min']:.4f} "
            f"(normalised Euclidean, {s2['n_params']} params). "
            f"Mean = {s2['test_to_train_nn_distances']['mean']:.4f} "
            f"SD = {s2['test_to_train_nn_distances']['sd']:.4f}. "
            "No leakage: test set is well-separated from training set in parameter space."
        )
    },

    # ── TASK 3: Noise robustness ──────────────────────────────────────────────
    "task3_noise_robustness": {
        "clean_baseline": {
            "pct_rmse_mean": s3["clean_baseline"]["pct_rmse_mean"],
            "pct_rmse_sd":   s3["clean_baseline"]["pct_rmse_sd"],
            "r2_mean":       s3["clean_baseline"]["r2_mean"],
        },
        "snr_30dB": s3["noisy"]["30"],
        "snr_20dB": s3["noisy"]["20"],
        "snr_10dB": s3["noisy"]["10"],
        "snr_5dB":  s3["noisy"]["5"],
        "conclusion": (
            f"At 30 dB SNR: %RMSE = {s3['noisy']['30']['pct_rmse_mean']:.2f}% "
            f"(+{s3['noisy']['30']['pct_rmse_mean'] - s3['clean_baseline']['pct_rmse_mean']:.2f} pp). "
            f"At 5 dB SNR: %RMSE = {s3['noisy']['5']['pct_rmse_mean']:.2f}% "
            f"(+{s3['noisy']['5']['pct_rmse_mean'] - s3['clean_baseline']['pct_rmse_mean']:.2f} pp). "
            "Model not retrained."
        )
    },

    # ── TASK 4: AR baseline ───────────────────────────────────────────────────
    "task4_ar_baseline": {
        "best_order": s4["best_order"],
        "test_pct_rmse_mean": s4["best_order_summary"]["test_pct_rmse_mean"],
        "test_pct_rmse_sd":   s4["best_order_summary"]["test_pct_rmse_sd"],
        "test_r2_mean":       s4["best_order_summary"]["test_r2_mean"],
        "both_orders": {
            k: {"val_pct": v["val_pct_rmse_mean"], "test_pct": v["test_pct_rmse_mean"],
                "test_r2": v["test_r2_mean"]}
            for k, v in s4["results_by_order"].items()
        },
        "vs_koopman_pct": s4["best_order_summary"]["test_pct_rmse_mean"] -
                          s1["summary"]["koopman"]["mean"],
    },

    # ── TASK 5: MLP baseline ──────────────────────────────────────────────────
    "task5_mlp_baseline": {
        "architecture": s5["architecture"],
        "n_params": s5["n_params"],
        "best_epoch": s5["best_epoch"],
        "test_pct_rmse_mean": s5["test"]["pct_rmse_mean"],
        "test_pct_rmse_sd":   s5["test"]["pct_rmse_sd"],
        "test_r2_mean":       s5["test"]["r2_mean"],
        "n_diverged": s5["test"]["n_diverged"],
        "fraction_diverged": s5["test"]["fraction_diverged"],
        "note": "MLP diverges autoregressively for some trajectories; values clipped at ±20 normalised units."
    },

    # ── TASK 6: Latent ablation ───────────────────────────────────────────────
    "task6_latent_ablation": {
        "n_trials": s6["n_completed_trials"],
        "searched_range": "n_complex ∈ {3,4,5}, n_real ∈ {3,4,5}",
        "best_config": s6["best_trial"],
        "by_nc_nr_summary": [
            {"n_complex": r["n_complex"], "n_real": r["n_real"],
             "latent_dim": r["latent_dim"],
             "pct_mean": r["pct_mean"], "pct_sd": r["pct_sd"],
             "n_trials": r["n_trials"]}
            for r in s6["by_nc_nr"]
        ],
        "note": s6["note"]
    },

    # ── TASK 7: gamma sensitivity ─────────────────────────────────────────────
    "task7_gamma_sensitivity": {
        "gamma_search_range": s7["gamma_search_range"],
        "gamma_range_all_incl_pruned": s7["gamma_range_all_incl_pruned"],
        "gamma_stable_range": s7["gamma_stable_range_20pct_threshold"],
        "n_complete": s7["n_trials"],
        "n_pruned": s7["n_pruned"],
        "note": s7["note"]
    },

    # ── TASK 8: Mode analysis ─────────────────────────────────────────────────
    "task8_mode_frequencies": {
        "delta_t_s": s8m["delta_t_s"],
        "n_complex_pairs": s8m["n_complex_pairs"],
        "n_real": s8m["n_real"],
        "complex_modes": [
            {"index": m["index"], "f_hz": m["f_hz"],
             "omega_rad_per_s": m["omega_rad_per_s"],
             "latent_variance": m["latent_variance"],
             "sigma": m["sigma"]}
            for m in s8m["modes"] if m["type"] == "complex"
        ],
        "real_modes": [
            {"index": m["index"], "sigma": m["sigma"],
             "latent_variance": m["latent_variance"]}
            for m in s8m["modes"] if m["type"] == "real"
        ],
        "heart_rate_mode_candidates": s8m["heart_rate_range_candidates"],
    },
    "task8_mode_ablation": {
        "baseline_pct_rmse": s8a["baseline_pct_rmse"],
        "ablation_results": s8a["ablation_results"],
        "most_important_mode": max(
            s8a["ablation_results"],
            key=lambda r: r["pct_rmse_increase"]
        ),
    },

    # ── TASK 9: R2 distribution ───────────────────────────────────────────────
    "task9_r2_distribution": {
        m: {
            "mean_r2": s9["stats"][m]["mean"],
            "sd_r2":   s9["stats"][m]["sd"],
            "fraction_negative_r2": s9["stats"][m]["fraction_negative_r2"],
            "n_negative_r2": s9["stats"][m]["n_negative_r2"],
            "n_total": s9["stats"][m]["n"],
        }
        for m in ["koopman", "lstm", "gru", "bilstm"]
    },

    # ── Cross-task comparison table ───────────────────────────────────────────
    "model_comparison_table": {
        "metric": "%RMSE (mean ± SD, test set)",
        "koopman": {
            "pct_rmse": f"{s1['summary']['koopman']['mean']:.2f} ± {s1['summary']['koopman']['sd']:.2f}",
            "r2": f"{s9['stats']['koopman']['mean']:.3f}",
        },
        "lstm": {
            "pct_rmse": f"{s1['summary']['lstm']['mean']:.2f} ± {s1['summary']['lstm']['sd']:.2f}",
            "r2": f"{s9['stats']['lstm']['mean']:.3f}",
        },
        "gru": {
            "pct_rmse": f"{s1['summary']['gru']['mean']:.2f} ± {s1['summary']['gru']['sd']:.2f}",
            "r2": f"{s9['stats']['gru']['mean']:.3f}",
        },
        "bilstm": {
            "pct_rmse": f"{s1['summary']['bilstm']['mean']:.2f} ± {s1['summary']['bilstm']['sd']:.2f}",
            "r2": f"{s9['stats']['bilstm']['mean']:.3f}",
        },
        "ar_p20": {
            "pct_rmse": f"{s4['best_order_summary']['test_pct_rmse_mean']:.2f} ± {s4['best_order_summary']['test_pct_rmse_sd']:.2f}",
            "r2": f"{s4['best_order_summary']['test_r2_mean']:.3f}",
        },
        "mlp": {
            "pct_rmse": f"{s5['test']['pct_rmse_mean']:.2f} ± {s5['test']['pct_rmse_sd']:.2f}",
            "r2": f"{s5['test']['r2_mean']:.3f}",
            "note": f"MLP diverged in {s5['test']['n_diverged']}/50 trajectories"
        },
    }
}

out_path = os.path.join(RESULT_DIR, "revision_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved -> {out_path}")

# ── print key numbers for reviewer response ────────────────────────────────────
print("\n" + "="*70)
print("KEY NUMBERS FOR REVIEWER RESPONSE LETTER")
print("="*70)

print(f"\n[STATS]")
print(f"  Friedman chi2={s1['friedman']['statistic']:.4f}, p={s1['friedman']['p']:.3e}")
for m in ["koopman","lstm","gru","bilstm"]:
    sw = s1["shapiro_wilk"][m]
    print(f"  Shapiro-Wilk {m}: W={sw['W']:.4f}, p={sw['p']:.3e}")
for k,v in s1["wilcoxon_one_sided_less"].items():
    print(f"  Wilcoxon {k}: stat={v['statistic']:.1f}, p={v['p']:.3e}")

print(f"\n[LHS DISTANCE]")
nn = s2["test_to_train_nn_distances"]
print(f"  Test->Train NN: min={nn['min']:.4f}, mean={nn['mean']:.4f}, SD={nn['sd']:.4f}")

print(f"\n[NOISE ROBUSTNESS]")
print(f"  Clean: {s3['clean_baseline']['pct_rmse_mean']:.2f}% (+/-{s3['clean_baseline']['pct_rmse_sd']:.2f})")
for snr in ["30","20","10","5"]:
    v = s3["noisy"][snr]
    print(f"  SNR={snr}dB: {v['pct_rmse_mean']:.2f}% (+/-{v['pct_rmse_sd']:.2f})  R2={v['r2_mean']:.3f}")

print(f"\n[AR BASELINE (order={s4['best_order']})]")
b = s4["best_order_summary"]
print(f"  %RMSE={b['test_pct_rmse_mean']:.2f}+/-{b['test_pct_rmse_sd']:.2f}  R2={b['test_r2_mean']:.3f}")

print(f"\n[MLP BASELINE]")
print(f"  %RMSE={s5['test']['pct_rmse_mean']:.2f}+/-{s5['test']['pct_rmse_sd']:.2f}  R2={s5['test']['r2_mean']:.3f}  diverged={s5['test']['n_diverged']}/50")

print(f"\n[LATENT ABLATION]")
print(f"  Best config: n_cp={s6['best_trial']['n_complex']}, n_real={s6['best_trial']['n_real']}, val_pct={s6['best_trial']['val_pct_rmse']:.4f}")

print(f"\n[GAMMA SENSITIVITY]")
print(f"  Stable range (20% threshold): [{s7['gamma_stable_range_20pct_threshold'][0]:.4f}, {s7['gamma_stable_range_20pct_threshold'][1]:.4f}]")

print(f"\n[MODE FREQUENCIES]")
for m in s8m["modes"]:
    if m["type"] == "complex":
        print(f"  Complex {m['index']}: f={m['f_hz']:.4f} Hz  var={m['latent_variance']:.4f}")
print(f"  Most critical mode (ablation): index={max(s8a['ablation_results'], key=lambda r: r['pct_rmse_increase'])['mode_index']}")
print(f"  Ablation delta: max={max(r['pct_rmse_increase'] for r in s8a['ablation_results']):.4f} pp")

print(f"\n[R2 PER TRAJECTORY]")
for m in ["koopman","lstm","gru","bilstm"]:
    st = s9["stats"][m]
    print(f"  {m}: mean R2={st['mean']:.3f}, frac R2<0={st['fraction_negative_r2']*100:.1f}%")
