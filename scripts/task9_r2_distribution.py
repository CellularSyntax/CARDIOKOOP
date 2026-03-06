#!/usr/bin/env python3
"""
TASK 9 -- Per-trajectory R2 distribution
------------------------------------------
For each model (Koopman, LSTM, GRU, BiLSTM), loads per-trajectory R2
from the postprocessing pickles and:
  - Reports fraction of trajectories with R2 < 0
  - Plots violin + box plots

Outputs:
  results/r2_per_trajectory.json
  results/figures/r2_distribution.png
"""
import os, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_ORDER  = ["koopman", "lstm", "gru", "bilstm"]
DISPLAY_NAMES = {"koopman": "Koopman", "lstm": "LSTM",
                 "gru": "GRU", "bilstm": "BiLSTM"}

# ── load R2 per trajectory from all pickles ───────────────────────────────────
r2_data = {}
for m in MODEL_ORDER:
    path = os.path.join(RESULT_DIR, m, f"{m}_postprocessing_results.pkl")
    with open(path, "rb") as f:
        res = pickle.load(f)
    r2_data[m] = res["r2_per_trajectory"].astype(float)

# ── compute statistics ─────────────────────────────────────────────────────────
stats = {}
for m, r2 in r2_data.items():
    neg_frac = float((r2 < 0).mean())
    stats[m] = {
        "n": int(len(r2)),
        "mean": float(r2.mean()),
        "sd":   float(r2.std(ddof=1)),
        "median": float(np.median(r2)),
        "q25":  float(np.percentile(r2, 25)),
        "q75":  float(np.percentile(r2, 75)),
        "min":  float(r2.min()),
        "max":  float(r2.max()),
        "fraction_negative_r2": neg_frac,
        "n_negative_r2": int((r2 < 0).sum()),
        "per_trajectory": r2.tolist(),
    }
    print(f"{m:8s}: mean={r2.mean():.3f}  sd={r2.std(ddof=1):.3f}  "
          f"R2<0: {(r2<0).sum()}/{len(r2)} ({neg_frac*100:.1f}%)")

# ── save JSON ─────────────────────────────────────────────────────────────────
out = {
    "metric": "R2 per test trajectory (global R2 averaged over all signals)",
    "models": MODEL_ORDER,
    "stats": stats,
}
with open(os.path.join(RESULT_DIR, "r2_per_trajectory.json"), "w") as f:
    json.dump(out, f, indent=2)
print("Saved -> results/r2_per_trajectory.json")

# ── figure ─────────────────────────────────────────────────────────────────────
pal = sns.color_palette("colorblind")
plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Panel A: violin plot ──────────────────────────────────────────────────────
ax = axes[0]
data_list  = [r2_data[m] for m in MODEL_ORDER]
labels     = [DISPLAY_NAMES[m] for m in MODEL_ORDER]
parts = ax.violinplot(data_list, positions=range(len(labels)),
                      showmeans=False, showmedians=True, widths=0.7)

for j, (pc, col) in enumerate(zip(parts["bodies"], pal)):
    pc.set_facecolor(col)
    pc.set_alpha(0.7)
parts["cmedians"].set_color("black")
parts["cmedians"].set_linewidth(2)
for key in ["cbars", "cmins", "cmaxes"]:
    parts[key].set_color("black")
    parts[key].set_linewidth(1)

# overlay individual points (jittered)
for j, (m, r2) in enumerate(r2_data.items()):
    jitter = np.random.default_rng(j).uniform(-0.08, 0.08, len(r2))
    ax.scatter(j + jitter, r2, s=15, alpha=0.5, color=pal[j], zorder=3)

ax.axhline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
           label="R² = 0")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel(r"$R^2$ per trajectory")
ax.set_title(r"$R^2$ Distribution per Trajectory")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# ── Panel B: fraction of negative R2 ─────────────────────────────────────────
ax2 = axes[1]
fracs = [stats[m]["fraction_negative_r2"] * 100 for m in MODEL_ORDER]
bars = ax2.bar(range(len(labels)), fracs,
               color=pal[:len(labels)], alpha=0.85, edgecolor="white")
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels)
ax2.set_ylabel("Trajectories with R² < 0 (%)")
ax2.set_title("Fraction of Negative-R² Trajectories")
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, max(fracs) * 1.25)

for bar, frac, m in zip(bars, fracs, MODEL_ORDER):
    n_neg = stats[m]["n_negative_r2"]
    n_tot = stats[m]["n"]
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f"{frac:.1f}%\n({n_neg}/{n_tot})",
             ha="center", va="bottom", fontsize=10)

plt.suptitle("Per-Trajectory R² Distributions (Test Set)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "r2_distribution.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print("Saved -> figures/r2_distribution.png")
