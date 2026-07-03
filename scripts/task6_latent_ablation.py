#!/usr/bin/env python3
"""
TASK 6 -- Latent dimensionality ablation
-----------------------------------------
Stratifies the 301 Optuna hyperparameter search trials by
(n_complex_pairs, n_real) and plots mean %RMSE (± SD) per configuration.

The search covers n_complex ∈ {3,4,5} and n_real ∈ {3,4,5}.
Configurations outside this range are not covered by the search and would
require separate training; this is noted in the output.

Optuna objective is the validation %RMSE (stored in column 'Value').

Outputs:
  results/latent_ablation.json
  results/figures/latent_ablation.png
  results/figures/latent_ablation.svg
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTUNA_DIR  = os.path.join(REPO_ROOT, "results", "optuna_runs")
RESULT_DIR  = os.path.join(REPO_ROOT, "results")
FIG_DIR     = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── load optuna results ────────────────────────────────────────────────────────
csv_file = os.path.join(OPTUNA_DIR,
                        "koopman_hyperparam_search_20250713_193701.csv")
df = pd.read_csv(csv_file)
# keep only completed trials
df = df[df["State"] == "COMPLETE"].copy()
print(f"Completed trials: {len(df)}")

nc = df["Param num_complex_pairs"].astype(int)
nr = df["Param num_real"].astype(int)
val_pct = df["Value"].values   # validation %RMSE (objective)

# ── stratify by (n_complex, n_real) ──────────────────────────────────────────
configs = sorted(set(zip(nc, nr)))
print(f"Unique (n_complex, n_real) configs in search: {configs}")

rows = []
for (nc_val, nr_val) in configs:
    mask = (nc == nc_val) & (nr == nr_val)
    subset = val_pct[mask]
    latent_dim = 2 * nc_val + nr_val
    rows.append({
        "n_complex": int(nc_val),
        "n_real":    int(nr_val),
        "latent_dim": int(latent_dim),
        "n_trials":  int(mask.sum()),
        "pct_mean":  float(subset.mean()),
        "pct_sd":    float(subset.std(ddof=1) if len(subset) > 1 else 0.0),
        "pct_median":float(np.median(subset)),
        "pct_min":   float(subset.min()),
        "pct_max":   float(subset.max()),
    })

df_res = pd.DataFrame(rows).sort_values("latent_dim")
print(df_res.to_string())

# also: aggregate by n_complex only (n_real fixed at best = 4)
nr_fixed = 4
by_nc = []
for nc_val in sorted(nc.unique()):
    mask = (nc == nc_val) & (nr == nr_fixed)
    subset = val_pct[mask]
    if len(subset) == 0:
        continue
    by_nc.append({
        "n_complex": int(nc_val),
        "n_real_fixed": int(nr_fixed),
        "latent_dim": int(2 * nc_val + nr_fixed),
        "n_trials": int(len(subset)),
        "pct_mean": float(subset.mean()),
        "pct_sd":   float(subset.std(ddof=1) if len(subset) > 1 else 0.0),
    })
df_nc = pd.DataFrame(by_nc)
print("\nWith n_real fixed at 4:")
print(df_nc.to_string())

# ── save JSON ─────────────────────────────────────────────────────────────────
out = {
    "source": "optuna_hyperparam_search_20250713",
    "n_completed_trials": int(len(df)),
    "note": ("Optuna search covered n_complex ∈ {3,4,5}, n_real ∈ {3,4,5}. "
             "Values n_complex ∈ {1,2,6} were not explored and would require "
             "separate training runs."),
    "metric": "validation %RMSE (Optuna objective)",
    "by_nc_nr": rows,
    "by_n_complex_nr_fixed4": by_nc,
    "best_trial": {
        "n_complex": int(nc[df["Value"].idxmin()]),
        "n_real":    int(nr[df["Value"].idxmin()]),
        "val_pct_rmse": float(df["Value"].min()),
    }
}

with open(os.path.join(RESULT_DIR, "latent_ablation.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved -> results/latent_ablation.json")

# ── figure ────────────────────────────────────────────────────────────────────
HEX_PAL = ["#dda251", "#5a286b", "#646464", "#ac4484"]
sns.set_theme("paper")
sns.set_style("white")
sns.set_context("notebook", font_scale=1.1)
pal = HEX_PAL

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: heatmap of mean %RMSE by (n_complex, n_real)
pivot = df_res.pivot(index="n_real", columns="n_complex", values="pct_mean")
ax = axes[0]
im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd_r",
               vmin=df_res["pct_mean"].min()*0.95,
               vmax=df_res["pct_mean"].max()*1.05)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{c}" for c in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"{r}" for r in pivot.index])
ax.set_xlabel("# complex pairs")
ax.set_ylabel("# real modes")
ax.set_title("Mean val. %RMSE by latent config.")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            n = df_res.loc[(df_res.n_complex == pivot.columns[j]) &
                           (df_res.n_real == pivot.index[i]), "n_trials"].values
            n_str = f"{int(n[0])}" if len(n) > 0 else "?"
            ax.text(j, i, f"{val:.2f}\n(n={n_str})", ha="center", va="center",
                    fontsize=9, color="black")
plt.colorbar(im, ax=ax, label="Mean val. %RMSE")

# Panel B: marginal plot -- n_complex with n_real fixed at best (4)
ax2 = axes[1]
if len(df_nc) > 0:
    x  = [row["n_complex"] for row in by_nc]
    y  = [row["pct_mean"]  for row in by_nc]
    ye = [row["pct_sd"]    for row in by_nc]
    ax2.errorbar(x, y, yerr=ye, fmt="o-", color=pal[0], linewidth=1.8,
                 markersize=7, capsize=4, capthick=1.5,
                 label=f"n_real = {nr_fixed}")
    ax2.set_xlabel("# complex pairs ($n_{{cp}}$)")
    ax2.set_ylabel("Mean val. %RMSE (± SD)")
    ax2.set_title(f"Effect of # complex pairs\n(n_real = {nr_fixed} fixed)")
    ax2.set_xticks(x)
    ax2.legend(fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    # Mark best
    best_x = x[int(np.argmin(y))]
    best_y = y[int(np.argmin(y))]
    ax2.axvline(best_x, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax2.text(best_x + 0.05, best_y * 0.98,
             f"Best: n_cp={best_x}\n{best_y:.2f}%", fontsize=9)

plt.suptitle("Latent Dimensionality Ablation (from Optuna search, 301 trials)",
             fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "latent_ablation.png"), dpi=150,
            bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "latent_ablation.svg"), format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved -> figures/latent_ablation.png + latent_ablation.svg")
