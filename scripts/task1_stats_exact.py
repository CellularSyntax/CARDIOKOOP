#!/usr/bin/env python3
"""
TASK 1 — Exact statistical tests
---------------------------------
Loads the four postprocessing pickles (Koopman, LSTM, GRU, BiLSTM),
extracts per-trajectory %RMSE, and computes:
  • Shapiro-Wilk normality test (W, p) per model
  • Friedman test across all four models
  • Pairwise Wilcoxon signed-rank tests: Koopman vs. each baseline
Saves results/stats_exact.json
"""
import os, sys, json, pickle
import numpy as np
from scipy.stats import shapiro, friedmanchisquare, wilcoxon

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(REPO_ROOT, "results")
OUT_FILE   = os.path.join(RESULT_DIR, "stats_exact.json")

# ── load all four pickles ──────────────────────────────────────────────────────
model_order = ["koopman", "lstm", "gru", "bilstm"]
pct_data = {}
for m in model_order:
    path = os.path.join(RESULT_DIR, m, f"{m}_postprocessing_results.pkl")
    with open(path, "rb") as f:
        res = pickle.load(f)
    pct_data[m] = res["pct_per_traj_ps"].astype(float)  # shape (50,)

print("Loaded %RMSE distributions:")
for m, v in pct_data.items():
    print(f"  {m}: mean={v.mean():.3f}  sd={v.std(ddof=1):.3f}  n={len(v)}")

# ── Shapiro-Wilk ───────────────────────────────────────────────────────────────
shapiro_results = {}
for m, v in pct_data.items():
    W, p = shapiro(v)
    shapiro_results[m] = {"W": float(W), "p": float(p)}
    print(f"Shapiro-Wilk {m}: W={W:.6f}, p={p:.6e}")

# ── Friedman ───────────────────────────────────────────────────────────────────
arrays = [pct_data[m] for m in model_order]
stat_f, p_f = friedmanchisquare(*arrays)
friedman_result = {"statistic": float(stat_f), "p": float(p_f)}
print(f"\nFriedman: chi2={stat_f:.4f}, p={p_f:.6e}")

# ── Pairwise Wilcoxon (Koopman vs others) ──────────────────────────────────────
wilcoxon_results = {}
for m in ["lstm", "gru", "bilstm"]:
    stat_w, p_w = wilcoxon(pct_data["koopman"], pct_data[m],
                           alternative="less")   # Koopman < baseline
    wilcoxon_results[f"koopman_vs_{m}"] = {"statistic": float(stat_w), "p": float(p_w)}
    print(f"Wilcoxon Koopman<{m}: stat={stat_w:.4f}, p={p_w:.6e}")

# ── also run pairwise with two-sided for completeness ─────────────────────────
wilcoxon_twosided = {}
baselines = ["lstm", "gru", "bilstm"]
for m in baselines:
    stat_w, p_w = wilcoxon(pct_data["koopman"], pct_data[m], alternative="two-sided")
    wilcoxon_twosided[f"koopman_vs_{m}"] = {"statistic": float(stat_w), "p": float(p_w)}

# ── summary stats ─────────────────────────────────────────────────────────────
summary = {}
for m, v in pct_data.items():
    summary[m] = {
        "mean": float(v.mean()),
        "sd":   float(v.std(ddof=1)),
        "median": float(np.median(v)),
        "q25":  float(np.percentile(v, 25)),
        "q75":  float(np.percentile(v, 75)),
        "n":    int(len(v))
    }

out = {
    "metric": "pct_per_traj_ps (%RMSE per trajectory, per-signal averaged)",
    "summary": summary,
    "shapiro_wilk": shapiro_results,
    "friedman": friedman_result,
    "wilcoxon_one_sided_less": wilcoxon_results,
    "wilcoxon_two_sided": wilcoxon_twosided
}

os.makedirs(RESULT_DIR, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved -> {OUT_FILE}")
