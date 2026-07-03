#!/usr/bin/env python3
"""
TASK 3 -- Noise robustness experiment
--------------------------------------
Loads the trained Koopman checkpoint and the *test* set.
Adds additive white Gaussian noise (AWGN) to each of the 12 state signals
at SNR levels 30, 20, 10, 5 dB.  Does NOT retrain.
Evaluates %RMSE and R2 at each SNR vs. clean baseline.

Outputs:
  results/noise_robustness.json
  results/figures/noise_robustness.png
  results/figures/noise_robustness.svg
"""
import os, glob, re, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score as sk_r2

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
DATA_DIR   = os.path.join(REPO_ROOT, "data")
CKPT_DIR   = os.path.join(RESULT_DIR, "koopman")
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load model ─────────────────────────────────────────────────────────────────
ckpts = glob.glob(os.path.join(CKPT_DIR, "*_model.ckpt"))
ckpt_path = max(ckpts, key=lambda f: int(re.search(r"_(\d{4})_", f).group(1)))
print(f"Loading checkpoint: {ckpt_path}")

ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
params = ckpt["params"]

# import model class
import sys
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from cardiokoop.postprocessing.postprocess_utils import load_weights_koopman

W, b, model = load_weights_koopman(
    ckpt_path,
    len(params["widths"]) - 1,
    len(params["widths_omega_real"]) - 1,
    params["num_real"],
    params["num_complex_pairs"],
)
model.to(device).eval()

# ── load test data ─────────────────────────────────────────────────────────────
data_name = params["data_name"]
X_norm = np.loadtxt(os.path.join(DATA_DIR, f"{data_name}_test1_x.csv"), delimiter=",")
U      = np.loadtxt(os.path.join(DATA_DIR, f"{data_name}_test1_u.csv"), delimiter=",")
sig_mean = np.load(os.path.join(DATA_DIR, "normalization_mean.npy"))
sig_std  = np.load(os.path.join(DATA_DIR, "normalization_std.npy"))

U = (U - U.mean(0)) / (U.std(0) + 1e-8)

T, n_sig = params["len_time"], X_norm.shape[1]
B = X_norm.shape[0] // T
X_rs  = X_norm.reshape(B, T, n_sig)   # normalised
U_rs  = U.reshape(B, T, -1)

# ── helper: autoregressive rollout ─────────────────────────────────────────────
def rollout(x0_t, u_seq_t):
    """
    x0_t: (B, D) tensor — initial state (normalised)
    u_seq_t: (T, B, ctrl) tensor
    Returns: numpy (B, T, D) in normalised space
    """
    with torch.no_grad():
        y = model.encoder(x0_t)
        outs = []
        for t in range(u_seq_t.size(0)):
            outs.append(model.decoder(y).cpu().numpy())
            y = model._advance(y, model.get_omegas(y), u_seq_t[t])
    return np.stack(outs, axis=1)   # (B, T, D)

def pct_rmse(true, pred):
    """Per-signal RMSE / PTP, then averaged -- matches postprocess_koopman metric."""
    rmse = np.sqrt(((true - pred) ** 2).mean(0))   # (D,)
    ptp  = np.ptp(true, axis=0) + 1e-8             # (D,)
    return float((100.0 * rmse / ptp).mean())

def r2_flat(true, pred):
    """Global R2 (flattened) -- matches postprocess_koopman global_r2_flat."""
    t = true.ravel(); p = pred.ravel()
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum() + 1e-8
    return float(1.0 - ss_res / ss_tot)

# ── clean baseline ─────────────────────────────────────────────────────────────
x0   = torch.tensor(X_rs[:, 0, :], dtype=torch.float32, device=device)
u_t  = torch.tensor(U_rs, dtype=torch.float32, device=device).permute(1, 0, 2)

pred_clean_norm = rollout(x0, u_t)                          # (B,T,D)
pred_clean      = pred_clean_norm * sig_std + sig_mean
X_true          = X_rs * sig_std + sig_mean

# per-trajectory metrics for clean
pct_clean_traj = np.array([pct_rmse(X_true[i], pred_clean[i]) for i in range(B)])
r2_clean_traj  = np.array([r2_flat(X_true[i], pred_clean[i]) for i in range(B)])
print(f"Clean: %RMSE={pct_clean_traj.mean():.3f}  R2={r2_clean_traj.mean():.4f}")

# ── noise experiment ──────────────────────────────────────────────────────────
SNR_DB = [30, 20, 10, 5]
rng    = np.random.default_rng(0)

noise_results = {}

for snr_db in SNR_DB:
    # signal power per channel across all trajectories (in normalised space, std ~1)
    sig_power = (X_rs.var(axis=(0, 1)))              # (D,)
    # noise power from SNR (linear)
    snr_lin   = 10 ** (snr_db / 10.0)
    noise_std = np.sqrt(sig_power / snr_lin)         # per-channel noise std

    # add noise to the initial condition only (first time step)
    noise_x0 = rng.normal(0, noise_std[None, :], size=(B, n_sig))
    x0_noisy = X_rs[:, 0, :] + noise_x0              # still normalised

    x0_t_noisy = torch.tensor(x0_noisy, dtype=torch.float32, device=device)
    pred_noisy_norm = rollout(x0_t_noisy, u_t)
    pred_noisy      = pred_noisy_norm * sig_std + sig_mean

    pct_traj = np.array([pct_rmse(X_true[i], pred_noisy[i]) for i in range(B)])
    r2_traj  = np.array([r2_flat(X_true[i], pred_noisy[i]) for i in range(B)])

    noise_results[str(snr_db)] = {
        "pct_rmse_mean":  float(pct_traj.mean()),
        "pct_rmse_sd":    float(pct_traj.std(ddof=1)),
        "pct_rmse_ci95":  float(1.96 * pct_traj.std(ddof=1) / np.sqrt(B)),
        "r2_mean":        float(r2_traj.mean()),
        "r2_sd":          float(r2_traj.std(ddof=1)),
        "per_trajectory_pct": pct_traj.tolist(),
        "per_trajectory_r2":  r2_traj.tolist(),
    }
    print(f"SNR={snr_db:2d} dB: %RMSE={pct_traj.mean():.3f}+-{pct_traj.std(ddof=1):.3f}  R2={r2_traj.mean():.4f}")

# ── save JSON ─────────────────────────────────────────────────────────────────
out = {
    "description": "AWGN added to initial state (t=0) of normalised signals; model NOT retrained",
    "snr_levels_db": SNR_DB,
    "clean_baseline": {
        "pct_rmse_mean": float(pct_clean_traj.mean()),
        "pct_rmse_sd":   float(pct_clean_traj.std(ddof=1)),
        "pct_rmse_ci95": float(1.96 * pct_clean_traj.std(ddof=1) / np.sqrt(B)),
        "r2_mean":       float(r2_clean_traj.mean()),
        "r2_sd":         float(r2_clean_traj.std(ddof=1)),
        "per_trajectory_pct": pct_clean_traj.tolist(),
        "per_trajectory_r2":  r2_clean_traj.tolist(),
    },
    "noisy": noise_results
}

with open(os.path.join(RESULT_DIR, "noise_robustness.json"), "w") as f:
    json.dump(out, f, indent=2)
print("Saved -> results/noise_robustness.json")

# ── figure ────────────────────────────────────────────────────────────────────
HEX_PAL = ["#dda251", "#5a286b", "#646464", "#ac4484"]
sns.set_theme("paper")
sns.set_style("white")
sns.set_context("notebook", font_scale=1.1)
pal = HEX_PAL

snr_vals = [float("inf")] + [float(s) for s in SNR_DB]   # inf = clean
labels   = ["Clean"] + [f"{s} dB" for s in SNR_DB]

pct_means = [pct_clean_traj.mean()] + [noise_results[str(s)]["pct_rmse_mean"] for s in SNR_DB]
pct_cis   = [1.96*pct_clean_traj.std(ddof=1)/np.sqrt(B)] + [noise_results[str(s)]["pct_rmse_ci95"] for s in SNR_DB]
r2_means  = [r2_clean_traj.mean()] + [noise_results[str(s)]["r2_mean"] for s in SNR_DB]
r2_cis    = [1.96*r2_clean_traj.std(ddof=1)/np.sqrt(B)] + [noise_results[str(s)]["r2_sd"]*1.96/np.sqrt(B) for s in SNR_DB]

x_pos = np.arange(len(labels))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# %RMSE
ax = axes[0]
ax.bar(x_pos, pct_means, yerr=pct_cis, color=pal[0], alpha=0.85,
       capsize=4, error_kw={"linewidth": 1.5}, edgecolor="white")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel("%RMSE")
ax.set_xlabel("Input noise level (SNR)")
ax.set_title("%RMSE vs. Input SNR")
ax.grid(axis="y", alpha=0.3)

# R²
ax = axes[1]
ax.bar(x_pos, r2_means, yerr=r2_cis, color=pal[1], alpha=0.85,
       capsize=4, error_kw={"linewidth": 1.5}, edgecolor="white")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel(r"$R^2$")
ax.set_xlabel("Input noise level (SNR)")
ax.set_title(r"$R^2$ vs. Input SNR")
ax.grid(axis="y", alpha=0.3)

plt.suptitle("Koopman Model: Noise Robustness (no retraining)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "noise_robustness.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "noise_robustness.svg"), format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved -> figures/noise_robustness.png + noise_robustness.svg")
