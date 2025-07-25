#!/usr/bin/env python
"""
Small-GRU baseline – training + autoregressive validation
=========================================================

Identical structure & output to LSTM/Koopman/BiLSTM scripts:
• Z-score normalisation for x (+ optional u)  
• Norm stats saved in checkpoint  
• `--use-control` fully supported  
• Roll-out logic matches GRU postprocessing  
"""

# ───────────────────────── imports ────────────────────────────
import os, time, datetime, argparse
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# ──────────────── CLI ───────────────────────────────
cli = argparse.ArgumentParser()
cli.add_argument("--exp-folder", "-f", default="results", help="Save directory for checkpoints & CSVs")
cli.add_argument("--use-control", action="store_true", help="Concatenate control u(t) to input window")
cli.add_argument("--control-dim", type=int, default=1, help="Dimensionality of the control signal")
args = cli.parse_args()

USE_CTRL = args.use_control
CTRL_DIM = args.control_dim
OUT_DIR = args.exp_folder
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────── hyper-parameters ──────────────────────
DATA_NAME = "csv_data_500_12sigs"
WINDOW = 10
HIDDEN = 512
DROPOUT = 0.1
LR = 1e-3
EPOCHS = 30
BATCH = 256
SEQ_LEN = 1500

TRAIN_X = f"./data/{DATA_NAME}_train1_x.csv"
VAL_X = f"./data/{DATA_NAME}_val1_x.csv"
TRAIN_U = f"./data/{DATA_NAME}_train1_u.csv"
VAL_U = f"./data/{DATA_NAME}_val1_u.csv"

# ────────────── GRU model ────────────────────────
class SmallGRU(nn.Module):
    def __init__(self, in_dim, hidden, dropout, out_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        y, _ = self.gru(x)
        return self.fc(y[:, -1])  # predict next step

# ──────────── autoregressive roll-out ─────────────────
def autoregressive_rollout(model, x_seq_n, win, dev, u_seq=None, u_mean=None, u_std=None):
    T, D = x_seq_n.shape
    out_n = np.empty_like(x_seq_n)
    out_n[:win] = x_seq_n[:win]

    if u_seq is not None:
        u_n = (u_seq - u_mean) / u_std
        buf = np.hstack([x_seq_n[:win], u_n[:win]])
    else:
        buf = x_seq_n[:win]

    buf_t = torch.from_numpy(buf).unsqueeze(0).float().to(dev)

    model.eval()
    with torch.no_grad():
        for t in range(win, T):
            y = model(buf_t).cpu().numpy()[0]
            out_n[t] = y

            nxt = y
            if u_seq is not None:
                nxt = np.hstack([nxt, u_n[t]])
            nxt = torch.from_numpy(nxt).unsqueeze(0).float().to(dev)
            buf_t = torch.cat([buf_t[:, 1:], nxt.unsqueeze(1)], dim=1)

    return out_n

# ──────────── load & normalise data ────────────────
x_tr = np.loadtxt(TRAIN_X, delimiter=',')
x_va = np.loadtxt(VAL_X, delimiter=',')

x_mean, x_std = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True) + 1e-8
x_tr_n = (x_tr - x_mean) / x_std
x_va_n = (x_va - x_mean) / x_std

if USE_CTRL:
    u_tr = np.loadtxt(TRAIN_U, delimiter=',')
    u_va = np.loadtxt(VAL_U, delimiter=',')

    if u_tr.ndim == 1:
        u_tr = u_tr.reshape(-1, CTRL_DIM)
        u_va = u_va.reshape(-1, CTRL_DIM)

    u_mean, u_std = u_tr.mean(0, keepdims=True), u_tr.std(0, keepdims=True) + 1e-8
    u_tr_n = (u_tr - u_mean) / u_std
    u_va_n = (u_va - u_mean) / u_std
else:
    u_tr_n = u_va_n = None
    u_mean = u_std = None

# ──────────── sliding-window datasets ────────────────
def make_dataset(x_n, u_n):
    X, Y = [], []
    for i in range(len(x_n) - WINDOW):
        win_x = x_n[i:i+WINDOW]
        if USE_CTRL:
            win_x = np.hstack([win_x, u_n[i:i+WINDOW]])
        X.append(win_x)
        Y.append(x_n[i+WINDOW])
    return torch.from_numpy(np.stack(X)).float(), torch.from_numpy(np.stack(Y)).float()

# ──────────────────── main ────────────────────────────
if __name__ == "__main__":
    X_tr, Y_tr = make_dataset(x_tr_n, u_tr_n)
    X_va, Y_va = make_dataset(x_va_n, u_va_n)

    train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, Y_va), batch_size=BATCH)

    IN_DIM = X_tr.shape[2]
    OUT_DIM = Y_tr.shape[1]

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallGRU(IN_DIM, HIDDEN, DROPOUT, OUT_DIM).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossF = nn.MSELoss()

    tr_losses, va_losses = [], []

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}", leave=False):
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossF(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                va_loss += lossF(model(xb), yb).item() * xb.size(0)
        va_loss /= len(val_loader.dataset)

        tr_losses.append(tr_loss)
        va_losses.append(va_loss)
        print(f"[{ep:02d}] train={tr_loss:.5f}  val={va_loss:.5f}")

    # ───────────── roll-out on validation ──────────────
    x_va_den = x_va_n * x_std + x_mean
    if USE_CTRL:
        u_va_den = u_va_n * u_std + u_mean

    N_SEQ = x_va_n.shape[0] // SEQ_LEN
    per_mse = []

    for s in range(N_SEQ):
        xs_n = x_va_n[s * SEQ_LEN : (s + 1) * SEQ_LEN]
        if USE_CTRL:
            us = u_va_den[s * SEQ_LEN : (s + 1) * SEQ_LEN]
            preds_n = autoregressive_rollout(model, xs_n, WINDOW, dev, us, u_mean, u_std)
        else:
            preds_n = autoregressive_rollout(model, xs_n, WINDOW, dev)

        preds = preds_n * x_std + x_mean
        gt = x_va_den[s * SEQ_LEN : (s + 1) * SEQ_LEN]
        mse = np.mean((preds[WINDOW:] - gt[WINDOW:]) ** 2)
        per_mse.append(mse)

        t = np.arange(SEQ_LEN)
        plt.figure(figsize=(10, 4))
        for i in range(2):
            plt.plot(t, gt[:, i], label=f"True s{i}")
            plt.plot(t, preds[:, i], label=f"Pred s{i}")
        plt.title(f"Seq {s} – MSE={mse:.4f}")
        plt.legend()
        plt.show()

    print("Per-seq MSE:", [f"{m:.4f}" for m in per_mse])
    print(f"Mean MSE = {np.mean(per_mse):.4f}")

    # ───────────── save checkpoint + CSV ──────────────
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(OUT_DIR, f"{DATA_NAME}_{stamp}_model.ckpt")
    csv_path = os.path.join(OUT_DIR, f"{DATA_NAME}_{stamp}_error.csv")

    torch.save({
        'params': dict(
            data_name=DATA_NAME,
            window=WINDOW,
            hidden=HIDDEN,
            dropout=DROPOUT,
            lr=LR,
            epochs=EPOCHS,
            batch_size=BATCH,
            seq_len=SEQ_LEN,
            use_control=USE_CTRL,
            control_dim=CTRL_DIM,
            x_mean=x_mean,
            x_std=x_std,
            u_mean=u_mean,
            u_std=u_std
        ),
        'state_dict': model.state_dict()
    }, ckpt_path)

    np.savetxt(csv_path, np.stack([tr_losses, va_losses], 1), delimiter=',', header='train,val', comments='')

    print(f"\n✓ checkpoint  → {ckpt_path}")
    print(f"✓ train/val CSV → {csv_path}")
