#!/usr/bin/env python3
"""
TASK 3b -- Noise robustness experiment for ALL baseline architectures
----------------------------------------------------------------------
Mirrors task3_noise_robustness.py but covers GRU, LSTM, BiLSTM, MLP, AR(20).

Adds AWGN to the initial state (t=0) of normalised signals at SNR levels
30, 20, 10, 5 dB.  Models are NOT retrained.

For RNN models (window=10): noise is applied to the full warm-up window
(steps 0..win-1) so that the perturbation is comparable to the Koopman case
where x0 is the sole initial condition.

For MLP: noise applied to x[0] (single-step initial condition).
For AR(20): noise applied to the full AR window (steps 0..19).

Outputs:
    results/noise_robustness_baselines.json
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR   = os.path.join(REPO_ROOT, "data")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── shared metric helpers ───────────────────────────────────────────────────
def pct_rmse_traj(true, pred):
    """Per-signal RMSE/ptp, averaged over signals — matches postprocess metric."""
    rmse = np.sqrt(((true - pred) ** 2).mean(0))   # (D,)
    ptp  = np.ptp(true, axis=0) + 1e-8             # (D,)
    return float((100.0 * rmse / ptp).mean())

def r2_flat(true, pred):
    t = true.ravel(); p = pred.ravel()
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum() + 1e-8
    return float(1.0 - ss_res / ss_tot)

def traj_stats(true_phys, pred_phys, B):
    pct_traj = np.array([pct_rmse_traj(true_phys[i], pred_phys[i]) for i in range(B)])
    r2_traj  = np.array([r2_flat(true_phys[i], pred_phys[i])       for i in range(B)])
    return pct_traj, r2_traj

def make_record(pct_traj, r2_traj):
    B = len(pct_traj)
    return {
        "pct_rmse_mean":      float(pct_traj.mean()),
        "pct_rmse_sd":        float(pct_traj.std(ddof=1)),
        "pct_rmse_ci95":      float(1.96 * pct_traj.std(ddof=1) / np.sqrt(B)),
        "r2_mean":            float(r2_traj.mean()),
        "r2_sd":              float(r2_traj.std(ddof=1)),
        "per_trajectory_pct": pct_traj.tolist(),
        "per_trajectory_r2":  r2_traj.tolist(),
    }

# ── load test data ─────────────────────────────────────────────────────────
DATA_NAME = "csv_data_500_12sigs"
T         = 1500
X_norm    = np.loadtxt(os.path.join(DATA_DIR, f"{DATA_NAME}_test1_x.csv"), delimiter=",")
sig_mean  = np.load(os.path.join(DATA_DIR, "normalization_mean.npy"))
sig_std   = np.load(os.path.join(DATA_DIR, "normalization_std.npy"))

B      = X_norm.shape[0] // T
n_sig  = X_norm.shape[1]
X_rs   = X_norm.reshape(B, T, n_sig)   # normalised (B, T, D)
X_true = X_rs * sig_std + sig_mean      # physical units

# noise levels
SNR_DB = [30, 20, 10, 5]
rng    = np.random.default_rng(42)

# signal power per channel (normalised space, ~1 but exact)
sig_power = X_rs.var(axis=(0, 1))   # (D,)

def noisy_x0(snr_db, win=1):
    """Return X_rs with noise added to the first `win` time steps."""
    snr_lin   = 10 ** (snr_db / 10.0)
    noise_std = np.sqrt(sig_power / snr_lin)          # (D,)
    X_noisy   = X_rs.copy()
    noise     = rng.normal(0, noise_std[None, None, :], size=(B, win, n_sig))
    X_noisy[:, :win, :] += noise
    return X_noisy

# ─────────────────────────────────────────────────────────────────────────────
# 1. GRU
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== GRU ===")
from cardiokoop.training.train_gru import SmallGRU, autoregressive_rollout as gru_rollout

_ckpt = torch.load(os.path.join(RESULT_DIR, "gru",
                   "csv_data_500_12sigs_20250625_170206_model.ckpt"),
                   map_location=device, weights_only=False)
_p    = _ckpt["params"]
WIN   = _p["window"]

gru_model = SmallGRU(n_sig, _p["hidden"], _p["dropout"], n_sig).to(device)
gru_model.load_state_dict(_ckpt["state_dict"])
gru_model.eval()

def gru_preds(X_batch):
    return np.stack([
        gru_rollout(gru_model, X_batch[i], WIN, device) * sig_std + sig_mean
        for i in range(B)
    ])

gru_results = {}
pct_c, r2_c = traj_stats(X_true, gru_preds(X_rs), B)
gru_results["clean_baseline"] = make_record(pct_c, r2_c)
print(f"  Clean: %RMSE={pct_c.mean():.3f}  R²={r2_c.mean():.4f}")

for snr in SNR_DB:
    Xn = noisy_x0(snr, win=WIN)
    pct_t, r2_t = traj_stats(X_true, gru_preds(Xn), B)
    gru_results[str(snr)] = make_record(pct_t, r2_t)
    print(f"  SNR={snr:2d} dB: %RMSE={pct_t.mean():.3f}  R²={r2_t.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== LSTM ===")
from cardiokoop.training.train_lstm import SmallLSTM, autoregressive_rollout as lstm_rollout

_ckpt = torch.load(os.path.join(RESULT_DIR, "lstm",
                   "csv_data_500_12sigs_20250625_184322_model.ckpt"),
                   map_location=device, weights_only=False)
_p    = _ckpt["params"]

lstm_model = SmallLSTM(n_sig, _p["hidden"], _p["dropout"], n_sig).to(device)
lstm_model.load_state_dict(_ckpt["state_dict"])
lstm_model.eval()

def lstm_preds(X_batch):
    return np.stack([
        lstm_rollout(lstm_model, X_batch[i], WIN, device) * sig_std + sig_mean
        for i in range(B)
    ])

lstm_results = {}
pct_c, r2_c = traj_stats(X_true, lstm_preds(X_rs), B)
lstm_results["clean_baseline"] = make_record(pct_c, r2_c)
print(f"  Clean: %RMSE={pct_c.mean():.3f}  R²={r2_c.mean():.4f}")

for snr in SNR_DB:
    Xn = noisy_x0(snr, win=WIN)
    pct_t, r2_t = traj_stats(X_true, lstm_preds(Xn), B)
    lstm_results[str(snr)] = make_record(pct_t, r2_t)
    print(f"  SNR={snr:2d} dB: %RMSE={pct_t.mean():.3f}  R²={r2_t.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BiLSTM
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== BiLSTM ===")
from cardiokoop.training.train_bilstm import SmallBiLSTM, autoregressive_rollout as bilstm_rollout

_ckpt = torch.load(os.path.join(RESULT_DIR, "bilstm",
                   "csv_data_500_12sigs_20250625_204256_model.ckpt"),
                   map_location=device, weights_only=False)
_p    = _ckpt["params"]

bilstm_model = SmallBiLSTM(n_sig, _p["hidden"], _p["dropout"], n_sig).to(device)
bilstm_model.load_state_dict(_ckpt["state_dict"])
bilstm_model.eval()

def bilstm_preds(X_batch):
    return np.stack([
        bilstm_rollout(bilstm_model, X_batch[i], WIN, device) * sig_std + sig_mean
        for i in range(B)
    ])

bilstm_results = {}
pct_c, r2_c = traj_stats(X_true, bilstm_preds(X_rs), B)
bilstm_results["clean_baseline"] = make_record(pct_c, r2_c)
print(f"  Clean: %RMSE={pct_c.mean():.3f}  R²={r2_c.mean():.4f}")

for snr in SNR_DB:
    Xn = noisy_x0(snr, win=WIN)
    pct_t, r2_t = traj_stats(X_true, bilstm_preds(Xn), B)
    bilstm_results[str(snr)] = make_record(pct_t, r2_t)
    print(f"  SNR={snr:2d} dB: %RMSE={pct_t.mean():.3f}  R²={r2_t.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MLP
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== MLP ===")
CLIP_VAL = 20.0

class _FlatMLP(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, dim),
        )
    def forward(self, x): return self.net(x)

_mlp_ckpt  = torch.load(
    os.path.join(REPO_ROOT, "checkpoints", "mlp_baseline.pt"),
    map_location=device, weights_only=False
)
mlp_model = _FlatMLP(dim=n_sig).to(device)
mlp_model.load_state_dict(_mlp_ckpt["state_dict"])
mlp_model.eval()

def _mlp_rollout(x_seq_n):
    T_, D = x_seq_n.shape
    out = np.empty_like(x_seq_n)
    out[0] = x_seq_n[0]
    prev = torch.tensor(x_seq_n[0], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        for t in range(1, T_):
            nxt = mlp_model(prev).clamp(-CLIP_VAL, CLIP_VAL)
            out[t] = nxt.cpu().numpy()[0]
            prev = nxt
    return out

def mlp_preds(X_batch):
    return np.stack([
        _mlp_rollout(X_batch[i]) * sig_std + sig_mean
        for i in range(B)
    ])

mlp_results = {}
pct_c, r2_c = traj_stats(X_true, mlp_preds(X_rs), B)
mlp_results["clean_baseline"] = make_record(pct_c, r2_c)
print(f"  Clean: %RMSE={pct_c.mean():.3f}  R²={r2_c.mean():.4f}")

for snr in SNR_DB:
    Xn = noisy_x0(snr, win=1)     # MLP: only x[0] is the initial condition
    pct_t, r2_t = traj_stats(X_true, mlp_preds(Xn), B)
    mlp_results[str(snr)] = make_record(pct_t, r2_t)
    print(f"  SNR={snr:2d} dB: %RMSE={pct_t.mean():.3f}  R²={r2_t.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. AR(20)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== AR(20) ===")
AR_P = 20

x_train_norm = np.loadtxt(
    os.path.join(DATA_DIR, f"{DATA_NAME}_train1_x.csv"), delimiter=","
)
B_train = x_train_norm.shape[0] // T
x_train_rs = x_train_norm.reshape(B_train, T, n_sig)

def _fit_ar(seqs, p):
    N, T_, D = seqs.shape
    coeffs = np.zeros((D, p))
    for d in range(D):
        rows_X, rows_y = [], []
        for i in range(N):
            ts = seqs[i, :, d]
            for t in range(p, T_):
                rows_X.append(ts[t-p:t][::-1])
                rows_y.append(ts[t])
        c, _, _, _ = np.linalg.lstsq(np.array(rows_X), np.array(rows_y), rcond=None)
        coeffs[d] = c
    return coeffs

def _ar_rollout(seqs, coeffs, p):
    N, T_, D = seqs.shape
    preds = seqs.copy()
    for i in range(N):
        for t in range(p, T_):
            preds[i, t, :] = (coeffs * preds[i, t-p:t, :][::-1, :].T).sum(axis=1)
    return preds

print("  Fitting AR(20) on training set...")
ar_coeffs = _fit_ar(x_train_rs, AR_P)
print("  Done.")

def ar_preds(X_batch):
    pred_norm = _ar_rollout(X_batch.copy(), ar_coeffs, AR_P)
    return pred_norm * sig_std + sig_mean

ar_results = {}
pct_c, r2_c = traj_stats(X_true, ar_preds(X_rs), B)
ar_results["clean_baseline"] = make_record(pct_c, r2_c)
print(f"  Clean: %RMSE={pct_c.mean():.3f}  R²={r2_c.mean():.4f}")

for snr in SNR_DB:
    Xn = noisy_x0(snr, win=AR_P)   # AR: all AR_P warm-up steps are initial conditions
    pct_t, r2_t = traj_stats(X_true, ar_preds(Xn), B)
    ar_results[str(snr)] = make_record(pct_t, r2_t)
    print(f"  SNR={snr:2d} dB: %RMSE={pct_t.mean():.3f}  R²={r2_t.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out = {
    "description": (
        "AWGN added to initial warm-up window of normalised signals; "
        "models NOT retrained. "
        "Window sizes: GRU/LSTM/BiLSTM=10, MLP=1, AR(20)=20. "
        "Matches Koopman noise_robustness.json SNR levels."
    ),
    "snr_levels_db": SNR_DB,
    "GRU":    {"clean_baseline": gru_results["clean_baseline"],
               "noisy":         {s: gru_results[str(s)] for s in SNR_DB}},
    "LSTM":   {"clean_baseline": lstm_results["clean_baseline"],
               "noisy":         {s: lstm_results[str(s)] for s in SNR_DB}},
    "BiLSTM": {"clean_baseline": bilstm_results["clean_baseline"],
               "noisy":         {s: bilstm_results[str(s)] for s in SNR_DB}},
    "MLP":    {"clean_baseline": mlp_results["clean_baseline"],
               "noisy":         {s: mlp_results[str(s)] for s in SNR_DB}},
    "AR(20)": {"clean_baseline": ar_results["clean_baseline"],
               "noisy":         {s: ar_results[str(s)] for s in SNR_DB}},
}

out_path = os.path.join(RESULT_DIR, "noise_robustness_baselines.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved -> {out_path}")
