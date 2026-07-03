#!/usr/bin/env python3
"""
Measure wall-clock training time for MLP and AR(20) baselines.

Does NOT overwrite any existing results files.
Saves timing only to:  results/training_times.json

Usage:
    cd <repo_root>
    python scripts/measure_training_times.py
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(REPO_ROOT, "data")
RESULT_DIR = os.path.join(REPO_ROOT, "results")
OUT_FILE   = os.path.join(RESULT_DIR, "training_times.json")

SIG_PREFIX = "csv_data_500_12sigs"
T          = 1500
np.random.seed(42)
torch.manual_seed(42)

# ── load data ─────────────────────────────────────────────────────────────────
def load_seqs(split):
    path = os.path.join(DATA_DIR, f"{SIG_PREFIX}_{split}1_x.csv")
    x = np.loadtxt(path, delimiter=",")
    n = x.shape[0] // T
    return x.reshape(n, T, x.shape[1])

print("Loading data ...")
x_train = load_seqs("train")   # (400, 1500, 12)
x_val   = load_seqs("val")     # ( 50, 1500, 12)
n_sig   = x_train.shape[2]
print(f"  train={x_train.shape}  val={x_val.shape}")

timing = {}

# ══════════════════════════════════════════════════════════════════════════════
# AR(20) — fit OLS coefficients (same as task4_ar_baseline.py, timing only)
# ══════════════════════════════════════════════════════════════════════════════
def fit_ar(seqs, p):
    N, T_, D = seqs.shape
    coeffs = np.zeros((D, p))
    for d in range(D):
        rows_X, rows_y = [], []
        for i in range(N):
            ts = seqs[i, :, d]
            for t in range(p, T_):
                rows_X.append(ts[t-p:t][::-1])
                rows_y.append(ts[t])
        X = np.array(rows_X)
        y = np.array(rows_y)
        c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        coeffs[d] = c
    return coeffs

print("\n── AR(20) fitting ───────────────────────────────────────────────────────")
t0 = time.perf_counter()
_ = fit_ar(x_train, p=20)
ar_time_s = time.perf_counter() - t0
ar_time_min = ar_time_s / 60.0

print(f"  Done. Wall-clock time: {ar_time_s:.1f} s  ({ar_time_min:.2f} min)")
timing["AR(20)"] = {
    "training_time_s":   round(ar_time_s,   2),
    "training_time_min": round(ar_time_min, 2),
    "note": "OLS fit of AR(20) coefficients on 400 training trajectories",
}

# ══════════════════════════════════════════════════════════════════════════════
# MLP — full training loop with early stopping (same as task5_mlp_baseline.py)
# ══════════════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n── MLP training (device: {device}) ─────────────────────────────────────")

def make_1step_dataset(seqs):
    N, T_, D = seqs.shape
    x = seqs[:, :-1, :].reshape(-1, D)
    y = seqs[:, 1:,  :].reshape(-1, D)
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))

Xtr, Ytr = make_1step_dataset(x_train)
Xva, Yva = make_1step_dataset(x_val)
train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=256, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=512, num_workers=0)

class FlatMLP(nn.Module):
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

mlp   = FlatMLP(n_sig).to(device)
opt   = torch.optim.Adam(mlp.parameters(), lr=1e-3)
lossF = nn.MSELoss()

EPOCHS   = 80
PATIENCE = 10
best_val = np.inf
no_imp   = 0
best_epoch = 0

t0 = time.perf_counter()

for ep in range(1, EPOCHS + 1):
    mlp.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        lossF(mlp(xb), yb).backward()
        opt.step()

    mlp.eval(); va = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            va += lossF(mlp(xb), yb).item() * xb.size(0)
    va /= len(val_loader.dataset)
    print(f"  [{ep:02d}/{EPOCHS}] val={va:.6f}")

    if va < best_val * (1 - 1e-4):
        best_val = va; no_imp = 0; best_epoch = ep
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print(f"  Early stopping at epoch {ep}")
            break

mlp_time_s   = time.perf_counter() - t0
mlp_time_min = mlp_time_s / 60.0

print(f"  Done. Best epoch: {best_epoch}. Wall-clock time: {mlp_time_s:.1f} s  ({mlp_time_min:.2f} min)")
timing["MLP"] = {
    "training_time_s":   round(mlp_time_s,   2),
    "training_time_min": round(mlp_time_min, 2),
    "best_epoch":        best_epoch,
    "device":            str(device),
    "note": "Full MLP training loop with early stopping (patience=10, max 80 epochs)",
}

# ══════════════════════════════════════════════════════════════════════════════
# Save — new file only, existing results untouched
# ══════════════════════════════════════════════════════════════════════════════
with open(OUT_FILE, "w") as f:
    json.dump(timing, f, indent=2)

print(f"\n{'='*60}")
print(f"  AR(20) training time : {timing['AR(20)']['training_time_min']:.2f} min")
print(f"  MLP    training time : {timing['MLP']['training_time_min']:.2f} min")
print(f"  Saved -> {OUT_FILE}")
