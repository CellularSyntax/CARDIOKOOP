#!/usr/bin/env python
"""
Small-BiLSTM baseline – training + autoregressive validation
===========================================================

Differences vs. your original draft
-----------------------------------
✔ **z-score normalisation** for *both* x–signals **and** optional control u(t)  
✔ Normalisation statistics (`x_mean/std`, optional `u_mean/std`) saved inside
  the checkpoint for consistent post-processing  
✔ `--use-control` flag handled everywhere (training windows, roll-out, plots)  
✔ Clean “one-window-per-step” autoregressive roll-out identical to Koopman/GRU  
"""

# ───────────────────────── imports / CLI ────────────────────────────
import os, time, datetime, argparse
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# ---------- CLI -----------------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--exp-folder",  "-f", default="results",
                 help="Where to store checkpoints / CSVs")
cli.add_argument("--use-control", action="store_true",
                 help="Concatenate control u(t) to the input window")
cli.add_argument("--control-dim", type=int, default=1,
                 help="Dimensionality of u(t) columns")
args = cli.parse_args()
USE_CTRL   = args.use_control
CTRL_DIM   = args.control_dim
OUT_DIR    = args.exp_folder
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- hyper-params & file names --------------------------------
DATA_NAME  = "csv_data_500_12sigs"
WINDOW     = 10
HIDDEN     = 512
DROPOUT    = 0.1
LR         = 1e-3
EPOCHS     = 30
BATCH      = 256
SEQ_LEN    = 1500

TRAIN_X    = f"./data/{DATA_NAME}_train1_x.csv"
TRAIN_U    = f"./data/{DATA_NAME}_train1_u.csv"
VAL_X      = f"./data/{DATA_NAME}_val1_x.csv"
VAL_U      = f"./data/{DATA_NAME}_val1_u.csv"

# ──────────────────── model definitions ─────────────────────────────
class SmallBiLSTM(nn.Module):
    def __init__(self, in_dim:int, hidden:int, dropout:float, out_dim:int):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1,
                            dropout=dropout, batch_first=True,
                            bidirectional=True)
        self.fc   = nn.Linear(hidden*2, out_dim)

    def forward(self, x):
        y,_ = self.lstm(x)
        return self.fc(y[:, -1])           # next-step prediction


# ───────────────────── autoregressive roll-out ──────────────────────
def autoregressive_rollout(model:nn.Module,
            x_seq_n:np.ndarray,
            win:int,
            dev:torch.device,
            u_seq:np.ndarray|None=None,
            u_mean=None, u_std=None) -> np.ndarray:
    """
    x_seq_n must be **normalised** already. Returns *normalised* predictions.
    """
    T, D = x_seq_n.shape
    out_n = np.empty_like(x_seq_n)
    out_n[:win] = x_seq_n[:win]

    if u_seq is not None:
        u_n = (u_seq - u_mean)/u_std
        buf = np.hstack([x_seq_n[:win], u_n[:win]])          # (W, D+D_u)
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
            buf_t = torch.cat([buf_t[:,1:], nxt.unsqueeze(1)], 1)
    return out_n


# ─────────────────────── load & normalise data ──────────────────────
x_train = np.loadtxt(TRAIN_X, delimiter=',')
x_val   = np.loadtxt(VAL_X,   delimiter=',')

x_mean, x_std = x_train.mean(0, keepdims=True), x_train.std(0, keepdims=True)+1e-8
x_train_n = (x_train - x_mean)/x_std
x_val_n   = (x_val   - x_mean)/x_std

if USE_CTRL:
    u_train = np.loadtxt(TRAIN_U, delimiter=',')
    u_val   = np.loadtxt(VAL_U,   delimiter=',')
    if u_train.ndim == 1:
        u_train = u_train.reshape(-1, CTRL_DIM)
        u_val   = u_val.reshape(-1, CTRL_DIM)
    u_mean, u_std = u_train.mean(0, keepdims=True), u_train.std(0, keepdims=True)+1e-8
    u_train_n = (u_train - u_mean)/u_std
    u_val_n   = (u_val   - u_mean)/u_std
else:
    u_train_n = u_val_n = None
    u_mean = u_std = None

# ───────────────────── sliding-window dataset ───────────────────────
def make_dataset(x_n, u_n):
    X,Y = [],[]
    for i in range(len(x_n)-WINDOW):
        win_x = x_n[i:i+WINDOW]
        if USE_CTRL:
            win_x = np.hstack([win_x, u_n[i:i+WINDOW]])
        X.append(win_x)
        Y.append(x_n[i+WINDOW])
    return torch.from_numpy(np.stack(X)).float(), torch.from_numpy(np.stack(Y)).float()

if __name__ == "__main__":
    X_tr, Y_tr = make_dataset(x_train_n, u_train_n)
    X_va, Y_va = make_dataset(x_val_n,   u_val_n)

    train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, Y_va), batch_size=BATCH)

    IN_DIM  = X_tr.shape[2]
    OUT_DIM = Y_tr.shape[1]

    # ───────────────────────── model / optim ────────────────────────────
    dev   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallBiLSTM(IN_DIM, HIDDEN, DROPOUT, OUT_DIM).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    lossF = nn.MSELoss()

    # ───────────────────────── train loop ───────────────────────────────
    tr_losses, va_losses = [], []
    for ep in range(1, EPOCHS+1):
        model.train(); tr = 0.0
        for xb,yb in tqdm(train_loader, desc=f"Ep {ep}/{EPOCHS}", leave=False):
            xb,yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossF(model(xb), yb); loss.backward(); opt.step()
            tr += loss.item()*xb.size(0)
        tr /= len(train_loader.dataset)

        model.eval(); va = 0.0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(dev), yb.to(dev)
                va += lossF(model(xb), yb).item()*xb.size(0)
        va /= len(val_loader.dataset)

        tr_losses.append(tr); va_losses.append(va)
        print(f"[{ep:02d}] train={tr:.5f}  val={va:.5f}")

    # ─────────────────── per-sequence roll-outs (val) ───────────────────
    x_val_den = x_val_n*x_std + x_mean                       # GT in original scale
    if USE_CTRL:
        u_val_den = u_val_n*u_std + u_mean

    N_SEQ = x_val_n.shape[0]//SEQ_LEN
    per_mse = []

    for s in range(N_SEQ):
        xs_n = x_val_n[s*SEQ_LEN:(s+1)*SEQ_LEN]
        if USE_CTRL:
            us  = u_val_den[s*SEQ_LEN:(s+1)*SEQ_LEN]
            preds_n = autoregressive_rollout(model, xs_n, WINDOW, dev, us, u_mean, u_std)
        else:
            preds_n = autoregressive_rollout(model, xs_n, WINDOW, dev)

        preds = preds_n*x_std + x_mean
        gt    = x_val_den[s*SEQ_LEN:(s+1)*SEQ_LEN]
        mse   = np.mean((preds[WINDOW:] - gt[WINDOW:])**2)
        per_mse.append(mse)

        # quick plot (first 2 signals)
        t = np.arange(SEQ_LEN)
        plt.figure(figsize=(10,4))
        for i in range(2):
            plt.plot(t, gt[:,i],    label=f'True s{i}')
            plt.plot(t, preds[:,i], label=f'Pred s{i}')
        plt.title(f"Seq {s} – MSE={mse:.4f}"); plt.legend(); plt.show()

    # ───────────────────── save checkpoint + CSV ────────────────────────
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path  = os.path.join(OUT_DIR, f"{DATA_NAME}_{stamp}_model.ckpt")
    csv_path   = os.path.join(OUT_DIR, f"{DATA_NAME}_{stamp}_error.csv")

    torch.save({
        'params': dict(
            data_name = DATA_NAME,
            window    = WINDOW,
            hidden    = HIDDEN,
            dropout   = DROPOUT,
            lr        = LR,
            epochs    = EPOCHS,
            batch_size= BATCH,
            seq_len   = SEQ_LEN,
            use_control = USE_CTRL,
            control_dim = CTRL_DIM,
            x_mean    = x_mean,
            x_std     = x_std,
            u_mean    = u_mean,
            u_std     = u_std
        ),
        'state_dict': model.state_dict()
    }, ckpt_path)

    np.savetxt(csv_path,
            np.stack([tr_losses, va_losses],1),
            delimiter=',', header='train,val', comments='')

    print(f"\n✓ checkpoint → {ckpt_path}")
    print(f"✓ train/val CSV → {csv_path}")
