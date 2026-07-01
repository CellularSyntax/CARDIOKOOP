#!/usr/bin/env python3
"""
TASK 7 -- gamma (recon_lam) sensitivity analysis
--------------------------------------------------
Extracts the relationship between the reconstruction-loss weight
(recon_lam, used as gamma) and validation %RMSE from the 301 Optuna
trials.  The search covered recon_lam in [0.003, 0.19].

For the requested range {0.001, 0.01, 0.1, 0.5, 1.0, 5.0}:
  - The range [0.003, 0.19] is covered by the search data.
  - Values {0.5, 1.0, 5.0} are outside the searched range and would
    require additional training runs (noted in the JSON).

Analysis:
  - Scatter + binned mean of val %RMSE vs. recon_lam
  - Log-scale x-axis
  - Report the "stable range" where mean %RMSE stays within 110% of best

Outputs:
  results/gamma_sensitivity.json
  results/figures/gamma_sensitivity.png
  results/figures/gamma_sensitivity.svg
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTUNA_DIR = os.path.join(REPO_ROOT, "results", "optuna_runs")
RESULT_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────────
csv_file = os.path.join(OPTUNA_DIR,
                        "koopman_hyperparam_search_20250713_193701.csv")
df_all = pd.read_csv(csv_file)
df = df_all[df_all["State"] == "COMPLETE"].copy()
df_pruned = df_all[df_all["State"] == "PRUNED"].copy()

gamma   = df["Param recon_lam"].values
val_pct = df["Value"].values

gamma_all   = df_all["Param recon_lam"].values
state_all   = df_all["State"].values

print(f"All trials: {len(df_all)}  (complete: {len(df)}, pruned: {len(df_pruned)})")
print(f"gamma range (complete): [{gamma.min():.5f}, {gamma.max():.5f}]")
print(f"gamma range (all incl. pruned): [{gamma_all.min():.5f}, {gamma_all.max():.5f}]")

# ── log-binned statistics ──────────────────────────────────────────────────────
log_gamma = np.log10(gamma)
bins_log  = np.linspace(log_gamma.min() - 0.01, log_gamma.max() + 0.01, 10)
bin_idx   = np.digitize(log_gamma, bins_log) - 1
bin_centers = 10 ** ((bins_log[:-1] + bins_log[1:]) / 2)

binned = []
for b in range(len(bins_log) - 1):
    mask = bin_idx == b
    if mask.sum() == 0:
        continue
    subset = val_pct[mask]
    binned.append({
        "bin_center_log": float((bins_log[b] + bins_log[b+1]) / 2),
        "bin_center":     float(10 ** ((bins_log[b] + bins_log[b+1]) / 2)),
        "bin_lo":         float(10 ** bins_log[b]),
        "bin_hi":         float(10 ** bins_log[b+1]),
        "n":              int(mask.sum()),
        "pct_mean":       float(subset.mean()),
        "pct_sd":         float(subset.std(ddof=1) if len(subset) > 1 else 0.0),
        "pct_median":     float(np.median(subset)),
    })

# ── "stable range": gamma values where binned mean within 20% of global best ──
best_mean = min(b["pct_mean"] for b in binned)
threshold = best_mean * 1.20
stable_bins = [b for b in binned if b["pct_mean"] <= threshold]
gamma_stable_lo = min(b["bin_lo"] for b in stable_bins)
gamma_stable_hi = max(b["bin_hi"] for b in stable_bins)

print(f"\nBest binned mean %RMSE: {best_mean:.4f}")
print(f"Stable range (within 20% of best): [{gamma_stable_lo:.5f}, {gamma_stable_hi:.5f}]")
print(f"Note: values gamma > 0.19 were not searched")

# ── save JSON ─────────────────────────────────────────────────────────────────
out = {
    "source": "optuna_hyperparam_search_20250713",
    "parameter": "recon_lam (reconstruction loss weight, referred to as gamma)",
    "n_trials": int(len(df)),
    "gamma_search_range": [float(gamma.min()), float(gamma.max())],
    "gamma_stable_range_20pct_threshold": [float(gamma_stable_lo), float(gamma_stable_hi)],
    "note": (
        "Optuna searched gamma in [0.003, 0.19] (301 trials total); "
        f"208 completed, 92 pruned (pruning indicates poor early performance, "
        "mostly at gamma > 0.064). "
        "Requested values {0.5, 1.0, 5.0} are outside this range. "
        "The binned-mean analysis (completed trials only) shows the stable range."
    ),
    "gamma_range_all_incl_pruned": [float(gamma_all.min()), float(gamma_all.max())],
    "n_pruned": int(len(df_pruned)),
    "binned_stats": binned,
    "all_trials": [
        {"gamma": float(g), "val_pct_rmse": float(v)}
        for g, v in zip(gamma, val_pct)
    ]
}

with open(os.path.join(RESULT_DIR, "gamma_sensitivity.json"), "w") as f:
    json.dump(out, f, indent=2)
print("Saved -> results/gamma_sensitivity.json")

# ── figure ────────────────────────────────────────────────────────────────────
HEX_PAL = ["#dda251", "#5a286b", "#646464", "#ac4484"]
sns.set_theme("paper")
sns.set_style("white")
sns.set_context("notebook", font_scale=1.1)
pal = HEX_PAL

fig, ax = plt.subplots(figsize=(8, 5))

# scatter: completed trials
ax.scatter(gamma, val_pct, alpha=0.30, s=20, color=pal[2], label="Completed trials", zorder=3)
# pruned trials: mark at x-axis to show their gamma range
gamma_p = df_pruned["Param recon_lam"].values
ax.scatter(gamma_p, np.full_like(gamma_p, val_pct.max()*1.05),
           marker="x", s=25, alpha=0.4, color="grey", label="Pruned trials (y-pos arbitrary)")

# binned mean ± SD
bc  = np.array([b["bin_center"] for b in binned])
bm  = np.array([b["pct_mean"]   for b in binned])
bsd = np.array([b["pct_sd"]     for b in binned])
ax.plot(bc, bm, "o-", color=pal[0], linewidth=2, markersize=7, label="Binned mean")
ax.fill_between(bc, bm - bsd, bm + bsd, alpha=0.2, color=pal[0])

# stable range shading
ax.axvspan(gamma_stable_lo, gamma_stable_hi, alpha=0.08, color="green",
           label=f"Stable range ([{gamma_stable_lo:.3f}, {gamma_stable_hi:.3f}])")

ax.set_xscale("log")
ax.set_xlabel(r"$\gamma$ (recon_lam)", fontsize=13)
ax.set_ylabel("Validation %RMSE (Optuna objective)")
ax.set_title(r"Sensitivity to Reconstruction Loss Weight $\gamma$")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# annotation: outside-range note
ax.text(0.98, 0.97,
        "Note: gamma > 0.19 not searched.\nPruned trials (x marks) have\ngamma up to 0.19.",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="grey",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "gamma_sensitivity.png"), dpi=150,
            bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "gamma_sensitivity.svg"), format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved -> figures/gamma_sensitivity.png + gamma_sensitivity.svg")
