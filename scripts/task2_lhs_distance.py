#!/usr/bin/env python3
"""
TASK 2 -- LHS data leakage check
---------------------------------
Loads the 500-row parameter space CSV (matching the split used in
create_dataset.py: seed=42, 80/10/10 split).
Normalises parameter vectors, computes pairwise Euclidean distances
between test-set parameters and all training-set parameters, and reports
min / mean / SD plus a histogram.

Outputs:
  results/lhs_distance_check.json
  figures/lhs_distances.png
  figures/lhs_distances.svg
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(REPO_ROOT, "data")
RESULT_DIR  = os.path.join(REPO_ROOT, "results")
FIG_DIR     = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── load parameter space ──────────────────────────────────────────────────────
params_csv = os.path.join(DATA_DIR, "cardiovascular_parameter_space.csv")
df = pd.read_csv(params_csv)
print(f"Parameter space: {df.shape[0]} sims x {df.shape[1]} params")

# ── replicate the exact split from create_dataset.py ─────────────────────────
# seed=42, shuffle, then 80/10/10
n_total = len(df)
np.random.seed(42)
idx = np.random.permutation(n_total)

n_train = int(0.8 * n_total)   # 400
n_val   = int(0.1 * n_total)   # 50
# test   = remaining 50

idx_train = idx[:n_train]
idx_val   = idx[n_train:n_train + n_val]
idx_test  = idx[n_train + n_val:]
print(f"Split: {len(idx_train)} train | {len(idx_val)} val | {len(idx_test)} test")

params = df.values.astype(float)

# ── z-score normalisation using training set stats ────────────────────────────
train_params = params[idx_train]
mean_p = train_params.mean(axis=0)
std_p  = train_params.std(axis=0)
std_p[std_p == 0] = 1.0

params_norm = (params - mean_p) / std_p

train_norm = params_norm[idx_train]   # (400, 11)
val_norm   = params_norm[idx_val]     # (50, 11)
test_norm  = params_norm[idx_test]    # (50, 11)

# ── pairwise Euclidean distances: test vs. train ──────────────────────────────
# shape: (50_test, 400_train)
dists_test_train = cdist(test_norm, train_norm, metric="euclidean")
min_dist_per_test = dists_test_train.min(axis=1)   # nearest training neighbour

# also check val vs. train for completeness
dists_val_train  = cdist(val_norm, train_norm, metric="euclidean")
min_dist_per_val = dists_val_train.min(axis=1)

# also train vs. train (leave-one-out: exclude self)
dists_train_train = cdist(train_norm, train_norm, metric="euclidean")
np.fill_diagonal(dists_train_train, np.inf)
min_dist_within_train = dists_train_train.min(axis=1)

def dist_stats(d):
    return {
        "min":  float(d.min()),
        "mean": float(d.mean()),
        "sd":   float(d.std(ddof=1)),
        "max":  float(d.max()),
        "q25":  float(np.percentile(d, 25)),
        "q75":  float(np.percentile(d, 75)),
        "n":    int(len(d))
    }

print("\nNearest-neighbour distances in normalised parameter space:")
print(f"  Test->Train  min={min_dist_per_test.min():.4f}  mean={min_dist_per_test.mean():.4f}  sd={min_dist_per_test.std(ddof=1):.4f}")
print(f"  Val->Train   min={min_dist_per_val.min():.4f}   mean={min_dist_per_val.mean():.4f}   sd={min_dist_per_val.std(ddof=1):.4f}")
print(f"  Train->Train min={min_dist_within_train.min():.4f}  mean={min_dist_within_train.mean():.4f}  sd={min_dist_within_train.std(ddof=1):.4f}")

result = {
    "n_train": int(len(idx_train)),
    "n_val":   int(len(idx_val)),
    "n_test":  int(len(idx_test)),
    "n_params": int(df.shape[1]),
    "param_names": df.columns.tolist(),
    "normalisation": "z-score (mean/std from training set)",
    "distance_metric": "Euclidean (L2)",
    "test_to_train_nn_distances": dist_stats(min_dist_per_test),
    "val_to_train_nn_distances":  dist_stats(min_dist_per_val),
    "within_train_nn_distances":  dist_stats(min_dist_within_train),
    "all_pairwise_test_train": {
        "mean": float(dists_test_train.mean()),
        "sd":   float(dists_test_train.std()),
        "min":  float(dists_test_train.min()),
        "max":  float(dists_test_train.max())
    }
}

with open(os.path.join(RESULT_DIR, "lhs_distance_check.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved -> results/lhs_distance_check.json")

# ── figure ────────────────────────────────────────────────────────────────────
HEX_PAL = ["#dda251", "#5a286b", "#646464", "#ac4484"]
sns.set_theme("paper")
sns.set_style("white")
sns.set_context("notebook", font_scale=1.1)
pal = HEX_PAL

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

sets = [
    (min_dist_per_test,      pal[0], "Test vs. Train\n(nearest-neighbour)"),
    (min_dist_per_val,       pal[1], "Val vs. Train\n(nearest-neighbour)"),
    (min_dist_within_train,  pal[2], "Within-Train\n(leave-one-out)"),
]

for ax, (d, col, title) in zip(axes, sets):
    ax.hist(d, bins=20, color=col, alpha=0.8, edgecolor="white", linewidth=0.6)
    ax.axvline(d.mean(), color="k", linestyle="--", linewidth=1.2,
               label=f"mean={d.mean():.2f}")
    ax.set_title(title)
    ax.set_xlabel("Euclidean distance (normalised space)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=10)

fig.suptitle("LHS Parameter-Space Coverage: Pairwise Distance Distribution",
             fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lhs_distances.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "lhs_distances.svg"), format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved -> figures/lhs_distances.png + lhs_distances.svg")
