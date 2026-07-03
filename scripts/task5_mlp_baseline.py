#!/usr/bin/env python3
"""
TASK 5 -- Flat MLP autoregressive baseline
-------------------------------------------
MLP receives a single initial state x(0) in R^12 and autoregressively
predicts 1499 further steps (one step at a time, feeding back its own output).

Architecture mirrors the Koopman encoder/decoder:
  12 -> 256 -> 512 -> 256 -> 12  (3 hidden layers, ReLU)

Training:
  - Same 80/10/10 split and z-score normalisation as Koopman
  - Adam lr=1e-3, MSE loss, early stopping (patience=10 validation epochs)
  - 80 max epochs, batch size 256
  - Sliding-window approach: predict x(t+1) from x(t)

Outputs:
  results/mlp_baseline.json
  checkpoints/mlp_baseline.pt
"""
import os, json, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(REPO_ROOT, "data")
RESULT_DIR  = os.path.join(REPO_ROOT, "results")
CKPT_DIR    = os.path.join(REPO_ROOT, "checkpoints")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIG_PREFIX = "csv_data_500_12sigs"
T          = 1500
torch.manual_seed(42)
np.random.seed(42)

# ── load data ─────────────────────────────────────────────────────────────────
def load_seqs(split):
    path = os.path.join(DATA_DIR, f"{SIG_PREFIX}_{split}1_x.csv")
    x = np.loadtxt(path, delimiter=",")
    n = x.shape[0] // T
    return x.reshape(n, T, x.shape[1])

x_train = load_seqs("train")   # (400, 1500, 12)
x_val   = load_seqs("val")     # ( 50, 1500, 12)
x_test  = load_seqs("test")    # ( 50, 1500, 12)
sig_mean = np.load(os.path.join(DATA_DIR, "normalization_mean.npy"))
sig_std  = np.load(os.path.join(DATA_DIR, "normalization_std.npy"))

n_sig = x_train.shape[2]       # 12

# ── build one-step datasets ──────────────────────────────────────────────────
def make_1step_dataset(seqs):
    """Each sample: (x_t) -> (x_{t+1}), across all trajectories."""
    N, T_, D = seqs.shape
    x = seqs[:, :-1, :].reshape(-1, D)
    y = seqs[:, 1:,  :].reshape(-1, D)
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))

Xtr, Ytr = make_1step_dataset(x_train)
Xva, Yva = make_1step_dataset(x_val)
print(f"Train samples: {len(Xtr)}, Val samples: {len(Xva)}")

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=256, shuffle=True,
                          num_workers=0)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=512,
                          num_workers=0)

# ── model ─────────────────────────────────────────────────────────────────────
class FlatMLP(nn.Module):
    """Single-step MLP: 12 -> 256 -> 512 -> 256 -> 12 (mirrors Koopman enc/dec)."""
    def __init__(self, dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, dim),
        )
    def forward(self, x):
        return self.net(x)

model = FlatMLP(n_sig).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"MLP params: {n_params:,}")

opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
lossF  = nn.MSELoss()

# ── training with early stopping ─────────────────────────────────────────────
EPOCHS   = 80
PATIENCE = 10
best_val  = np.inf
no_imp    = 0
ckpt_path = os.path.join(CKPT_DIR, "mlp_baseline.pt")

tr_losses, va_losses = [], []

for ep in range(1, EPOCHS + 1):
    model.train(); tr = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = lossF(model(xb), yb)
        loss.backward()
        opt.step()
        tr += loss.item() * xb.size(0)
    tr /= len(train_loader.dataset)

    model.eval(); va = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            va += lossF(model(xb), yb).item() * xb.size(0)
    va /= len(val_loader.dataset)
    tr_losses.append(tr); va_losses.append(va)
    print(f"[{ep:02d}/{EPOCHS}] train={tr:.6f}  val={va:.6f}")

    if va < best_val * (1 - 1e-4):
        best_val = va
        no_imp   = 0
        torch.save({"state_dict": model.state_dict(),
                    "n_params": n_params,
                    "n_sig": n_sig,
                    "epoch": ep}, ckpt_path)
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print(f"Early stopping at epoch {ep} (no improvement for {PATIENCE} epochs)")
            break

# reload best
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Best val loss: {best_val:.6f} at epoch {ckpt['epoch']}")

# ── autoregressive rollout ────────────────────────────────────────────────────
CLIP_VAL = 20.0   # clip to ±20 normalised units to prevent runaway

def ar_rollout_mlp(model, x_seq_n, device):
    """
    x_seq_n: (T, D) normalised.  Returns (T, D) normalised predictions.
    First step is copied; then autoregressively predicted.
    Values are clipped to ±CLIP_VAL to prevent complete divergence.
    """
    T_, D = x_seq_n.shape
    out = np.empty_like(x_seq_n)
    out[0] = x_seq_n[0]
    prev = torch.tensor(x_seq_n[0], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        for t in range(1, T_):
            nxt = model(prev).clamp(-CLIP_VAL, CLIP_VAL)
            out[t] = nxt.cpu().numpy()[0]
            prev = nxt
    return out

# ── metrics ──────────────────────────────────────────────────────────────────
def pct_rmse(true_n, pred_n):
    """Per-signal RMSE/PTP, averaged -- matches postprocess metric."""
    true = true_n * sig_std + sig_mean
    pred = pred_n * sig_std + sig_mean
    rmse = np.sqrt(((true - pred) ** 2).mean(0))
    ptp  = np.ptp(true, axis=0) + 1e-8
    return float((100.0 * rmse / ptp).mean())

def r2_score(true_n, pred_n):
    true = (true_n * sig_std + sig_mean).ravel()
    pred = (pred_n * sig_std + sig_mean).ravel()
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum() + 1e-8
    return float(1.0 - ss_res / ss_tot)

# evaluate on test set
pct_test, r2_test, diverged_test = [], [], []
for i in range(x_test.shape[0]):
    pred_n = ar_rollout_mlp(model, x_test[i], device)
    div = bool(np.isnan(pred_n).any() or (np.abs(pred_n) > CLIP_VAL * 0.95).mean() > 0.1)
    diverged_test.append(div)
    pct_test.append(pct_rmse(x_test[i], pred_n))
    r2_test.append(r2_score(x_test[i], pred_n))
pct_test = np.array(pct_test)
r2_test  = np.array(r2_test)

# evaluate on val set too
pct_val_eval, r2_val_eval = [], []
for i in range(x_val.shape[0]):
    pred_n = ar_rollout_mlp(model, x_val[i], device)
    pct_val_eval.append(pct_rmse(x_val[i], pred_n))
    r2_val_eval.append(r2_score(x_val[i], pred_n))
pct_val_eval = np.array(pct_val_eval)
r2_val_eval  = np.array(r2_val_eval)

B = len(pct_test)
n_diverged = sum(diverged_test)
print(f"\nTest: %RMSE={np.nanmean(pct_test):.3f}+-{np.nanstd(pct_test, ddof=1):.3f}  "
      f"R2={np.nanmean(r2_test):.4f}  diverged={n_diverged}/{B}")
print(f" Val: %RMSE={pct_val_eval.mean():.3f}+-{pct_val_eval.std(ddof=1):.3f}  "
      f"R2={r2_val_eval.mean():.4f}")

# ── save ──────────────────────────────────────────────────────────────────────
out = {
    "model": "FlatMLP autoregressive (12->256->512->256->12, ReLU)",
    "n_params": int(n_params),
    "architecture": "12->256->512->256->12",
    "activation": "ReLU",
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "batch_size": 256,
    "max_epochs": EPOCHS,
    "patience": PATIENCE,
    "best_epoch": int(ckpt["epoch"]),
    "train_loss_curve": tr_losses,
    "val_loss_curve": va_losses,
    "validation": {
        "pct_rmse_mean": float(pct_val_eval.mean()),
        "pct_rmse_sd":   float(pct_val_eval.std(ddof=1)),
        "r2_mean":       float(r2_val_eval.mean()),
    },
    "test": {
        "pct_rmse_mean":  float(np.nanmean(pct_test)),
        "pct_rmse_sd":    float(np.nanstd(pct_test, ddof=1)),
        "pct_rmse_ci95":  float(1.96 * np.nanstd(pct_test, ddof=1) / np.sqrt(B)),
        "r2_mean":        float(np.nanmean(r2_test)),
        "r2_sd":          float(np.nanstd(r2_test, ddof=1)),
        "n_diverged":     int(n_diverged),
        "fraction_diverged": float(n_diverged / B),
        "clip_value_used": float(CLIP_VAL),
        "per_trajectory_pct": [float(x) for x in pct_test],
        "per_trajectory_r2":  [float(x) for x in r2_test],
        "per_trajectory_diverged": diverged_test,
    }
}

with open(os.path.join(RESULT_DIR, "mlp_baseline.json"), "w") as f:
    json.dump(out, f, indent=2)
print("Saved -> results/mlp_baseline.json")
print(f"Saved -> checkpoints/mlp_baseline.pt")
