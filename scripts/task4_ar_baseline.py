#!/usr/bin/env python3
"""
TASK 4 -- Linear AR baseline
------------------------------
Fits one AR(p) model per signal on all training trajectories (flattened).
Tests orders p=10 and p=20; picks the better one on the validation set.
Evaluates on the test set using the same %RMSE and R2 metrics as Koopman.

Strategy:
  - Fit AR(p) per signal independently using numpy least-squares (Toeplitz system).
  - Autoregressive rollout from the first p steps.
  - Compare p=10 vs p=20 by validation %RMSE.

Outputs:
  results/ar_baseline.json
"""
import os, json, sys
import numpy as np

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(REPO_ROOT, "data")
RESULT_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

SIG_PREFIX = "csv_data_500_12sigs"
T          = 1500
SEQ_LEN    = T   # full trajectory
np.random.seed(42)

# ── load data (already normalised) ───────────────────────────────────────────
def load_seqs(split):
    """Returns (N_traj, T, D) normalised array."""
    path = os.path.join(DATA_DIR, f"{SIG_PREFIX}_{split}1_x.csv")
    x = np.loadtxt(path, delimiter=",")
    n = x.shape[0] // T
    return x.reshape(n, T, x.shape[1])

x_train = load_seqs("train")   # (400, 1500, 12)
x_val   = load_seqs("val")     # ( 50, 1500, 12)
x_test  = load_seqs("test")    # ( 50, 1500, 12)

sig_mean = np.load(os.path.join(DATA_DIR, "normalization_mean.npy"))
sig_std  = np.load(os.path.join(DATA_DIR, "normalization_std.npy"))

n_sig = x_train.shape[2]
print(f"Loaded: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

# ── fit AR(p) per signal using ordinary least squares ─────────────────────────
def fit_ar(seqs, p):
    """
    seqs: (N, T, D)
    Returns: coeffs (D, p) -- AR coefficients per signal
    """
    N, T, D = seqs.shape
    coeffs = np.zeros((D, p))
    for d in range(D):
        # Build Toeplitz-style X (predictors) and y (targets)
        # across all trajectories, all time steps t=p..T-1
        rows_X, rows_y = [], []
        for i in range(N):
            ts = seqs[i, :, d]           # (T,)
            for t in range(p, T):
                rows_X.append(ts[t-p:t][::-1])   # [x_{t-1}, ..., x_{t-p}]
                rows_y.append(ts[t])
        X = np.array(rows_X)   # (N*(T-p), p)
        y = np.array(rows_y)   # (N*(T-p),)
        # OLS
        c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        coeffs[d] = c
    return coeffs

# ── rollout from the first p steps ───────────────────────────────────────────
def ar_rollout(seqs_norm, coeffs, p):
    """
    seqs_norm: (N, T, D)  normalised inputs
    coeffs:    (D, p)
    Returns:   (N, T, D)  normalised predictions
               (first p steps are copied from input)
    """
    N, T, D = seqs_norm.shape
    preds = seqs_norm.copy()
    for i in range(N):
        for t in range(p, T):
            for d in range(D):
                preds[i, t, d] = (coeffs[d] * preds[i, t-p:t, d][::-1]).sum()
    return preds

# ── per-signal ptp %RMSE metric ───────────────────────────────────────────────
def pct_rmse_traj(true_n, pred_n):
    """true_n, pred_n: (N, T, D) normalised; denorm for metric."""
    true  = true_n * sig_std + sig_mean
    pred  = pred_n * sig_std + sig_mean
    vals  = []
    for i in range(true.shape[0]):
        rmse = np.sqrt(((true[i] - pred[i]) ** 2).mean(0))
        ptp  = np.ptp(true[i], axis=0) + 1e-8
        vals.append((100.0 * rmse / ptp).mean())
    return np.array(vals)

def r2_traj(true_n, pred_n):
    """Global R2 per trajectory."""
    true = true_n * sig_std + sig_mean
    pred = pred_n * sig_std + sig_mean
    vals = []
    for i in range(true.shape[0]):
        t = true[i].ravel(); p = pred[i].ravel()
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum() + 1e-8
        vals.append(1.0 - ss_res / ss_tot)
    return np.array(vals)

# ── fit and evaluate both orders ──────────────────────────────────────────────
best_p, best_coeffs, best_val_pct = None, None, np.inf

results_by_order = {}
for p in [10, 20]:
    print(f"\nFitting AR(p={p}) ...")
    coeffs = fit_ar(x_train, p)

    pred_val  = ar_rollout(x_val,  coeffs, p)
    pct_val   = pct_rmse_traj(x_val, pred_val)
    print(f"  Validation %RMSE: {pct_val.mean():.3f} +- {pct_val.std(ddof=1):.3f}")

    pred_test = ar_rollout(x_test, coeffs, p)
    pct_test  = pct_rmse_traj(x_test, pred_test)
    r2_test   = r2_traj(x_test, pred_test)
    print(f"  Test      %RMSE: {pct_test.mean():.3f} +- {pct_test.std(ddof=1):.3f}  R2: {r2_test.mean():.4f}")

    results_by_order[str(p)] = {
        "val_pct_rmse_mean": float(pct_val.mean()),
        "val_pct_rmse_sd":   float(pct_val.std(ddof=1)),
        "test_pct_rmse_mean": float(pct_test.mean()),
        "test_pct_rmse_sd":   float(pct_test.std(ddof=1)),
        "test_pct_rmse_ci95": float(1.96 * pct_test.std(ddof=1) / np.sqrt(len(pct_test))),
        "test_r2_mean":       float(r2_test.mean()),
        "test_r2_sd":         float(r2_test.std(ddof=1)),
        "per_trajectory_pct": pct_test.tolist(),
        "per_trajectory_r2":  r2_test.tolist(),
    }

    if pct_val.mean() < best_val_pct:
        best_val_pct  = pct_val.mean()
        best_p        = p
        best_coeffs   = coeffs

# ── save ─────────────────────────────────────────────────────────────────────
best_res = results_by_order[str(best_p)]
print(f"\nBest order on validation: p={best_p}  (val %RMSE={best_val_pct:.3f})")
print(f"Test: %RMSE={best_res['test_pct_rmse_mean']:.3f}+-{best_res['test_pct_rmse_sd']:.3f}  R2={best_res['test_r2_mean']:.4f}")

out = {
    "model": "Linear AR per signal (OLS)",
    "orders_tested": [10, 20],
    "best_order": best_p,
    "selection_criterion": "validation %RMSE",
    "results_by_order": results_by_order,
    "best_order_summary": best_res,
}

with open(os.path.join(RESULT_DIR, "ar_baseline.json"), "w") as f:
    json.dump(out, f, indent=2)
print("Saved -> results/ar_baseline.json")
