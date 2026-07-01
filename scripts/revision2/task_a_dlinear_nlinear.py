#!/usr/bin/env python3
"""
TASK A (editor mandatory #1 / R1.1) -- DLinear + NLinear direct baselines
=========================================================================
Adds the two canonical linear direct-forecasting baselines of
Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023
to the CARDIOKOOP baseline suite.

Both are **direct (non-autoregressive) multi-step forecasters**: a single linear
map produces the whole forecast horizon in one shot (no feedback of predictions).

Setup (documented so it is fair and canonical):
  * Look-back window  L = 10 steps -- identical to the warm-up window the existing
    LSTM/GRU/BiLSTM baselines observe, so the linear baselines get exactly the
    same head-start as the RNN baselines (and the metric is computed over the
    same full 1500-step trajectory).
  * Horizon           H = T - L = 1490 steps, predicted directly.
  * Control feature   s = V_usv,step (the pre-load step level; the control is a
    single step at t=500 to a per-trajectory level, so its whole information
    content is this scalar).  s is standardised with TRAIN statistics and given
    to BOTH models as an extra input feature -- the same information the Koopman
    control net receives.  This strengthens the baselines (and hence the
    comparison).
  * Per-channel ("individual") weights, one linear map per signal.

  NLinear : subtract the last look-back value, one Linear(L+1 -> H), add it back.
  DLinear : series-decompose the look-back into trend (moving average, kernel 5)
            + remainder, one Linear(L+1 -> H) each, sum.

Fit on train1, early-stopped on val1, evaluated on the seed-42 test1 split with
the shared metric functions (identical to Tables 3/4/5).  The frozen Koopman
model is NOT touched.

Outputs
  results/dlinear/dlinear_postprocessing_results.pkl   (baseline pkl format)
  results/nlinear/nlinear_postprocessing_results.pkl
  results/revision2/task_a_dlinear_nlinear.json        (headline summary)
  checkpoints/dlinear_baseline.pt / nlinear_baseline.pt
"""
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

DEVICE = C.get_device()
L_LOOKBACK = 10          # same warm-up window as the RNN baselines
KERNEL     = 5           # DLinear moving-average kernel
EPOCHS     = 30000       # trained to convergence (full-batch, tiny linear model)
PATIENCE   = 3000
LR         = 1e-2


# ───────────────────────── models ─────────────────────────
def moving_avg(x, k):
    """Series-decomposition trend via replicate-padded average pooling. x:(B,D,L)."""
    pad = k // 2
    xp = F.pad(x, (pad, pad), mode="replicate")
    return F.avg_pool1d(xp, kernel_size=k, stride=1)


class NLinear(nn.Module):
    """Per-channel NLinear with an extra control feature."""
    def __init__(self, n_ch, L, H):
        super().__init__()
        self.L, self.H = L, H
        # weight over [look-back (L), control (1)] -> horizon (H), per channel
        self.W = nn.Parameter(torch.zeros(n_ch, H, L + 1))
        self.b = nn.Parameter(torch.zeros(n_ch, H))
        nn.init.normal_(self.W, std=1.0 / (L + 1))

    def forward(self, lb, s):
        # lb: (B, D, L) look-back ; s: (B, 1) control scalar
        last = lb[:, :, -1:]                      # (B, D, 1)
        x = lb - last                             # subtract last look-back value
        s_feat = s.unsqueeze(1).expand(-1, lb.size(1), -1)   # (B, D, 1)
        feat = torch.cat([x, s_feat], dim=2)      # (B, D, L+1)
        dev = torch.einsum("bdk,dhk->bdh", feat, self.W) + self.b
        return dev + last                         # add the value back -> (B, D, H)


class DLinear(nn.Module):
    """Per-channel DLinear (trend + remainder) with an extra control feature."""
    def __init__(self, n_ch, L, H, kernel=KERNEL):
        super().__init__()
        self.L, self.H, self.kernel = L, H, kernel
        self.Wt = nn.Parameter(torch.zeros(n_ch, H, L + 1))
        self.Ws = nn.Parameter(torch.zeros(n_ch, H, L + 1))
        self.bt = nn.Parameter(torch.zeros(n_ch, H))
        self.bs = nn.Parameter(torch.zeros(n_ch, H))
        nn.init.normal_(self.Wt, std=1.0 / (L + 1))
        nn.init.normal_(self.Ws, std=1.0 / (L + 1))

    def forward(self, lb, s):
        trend = moving_avg(lb, self.kernel)       # (B, D, L)
        seas  = lb - trend
        s_feat = s.unsqueeze(1).expand(-1, lb.size(1), -1)   # (B, D, 1)
        ft = torch.cat([trend, s_feat], dim=2)
        fs = torch.cat([seas,  s_feat], dim=2)
        out = (torch.einsum("bdk,dhk->bdh", ft, self.Wt) + self.bt
               + torch.einsum("bdk,dhk->bdh", fs, self.Ws) + self.bs)
        return out                                # (B, D, H)


# ───────────────────────── data ─────────────────────────
def make_tensors(X_norm, s_std):
    """Return look-back (B,D,L), horizon target (B,D,H), control (B,1)."""
    B, T, D = X_norm.shape
    lb  = X_norm[:, :L_LOOKBACK, :].transpose(0, 2, 1)          # (B, D, L)
    tgt = X_norm[:, L_LOOKBACK:, :].transpose(0, 2, 1)          # (B, D, H)
    lb  = torch.tensor(lb,  dtype=torch.float32, device=DEVICE)
    tgt = torch.tensor(tgt, dtype=torch.float32, device=DEVICE)
    s   = torch.tensor(s_std[:, None], dtype=torch.float32, device=DEVICE)
    return lb, tgt, s


def train_model(model, tr, va, tag):
    lb_tr, tgt_tr, s_tr = tr
    lb_va, tgt_va, s_va = va
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=300)
    lossf = nn.MSELoss()
    best_va, best_state, no_imp = np.inf, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        loss = lossf(model(lb_tr, s_tr), tgt_tr)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_loss = lossf(model(lb_va, s_va), tgt_va).item()
        sched.step(va_loss)
        if va_loss < best_va * (1 - 1e-5):
            best_va, no_imp = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
        if ep % 500 == 0 or ep == 1:
            print(f"  [{tag}] ep {ep:5d}  train={loss.item():.6f}  val={va_loss:.6f}  best={best_va:.6f}")
    model.load_state_dict(best_state)
    print(f"  [{tag}] best val MSE={best_va:.6f} (epoch-stopped at {ep})")
    return model


@torch.no_grad()
def full_prediction(model, X_norm, s_std, sig_mean, sig_std):
    """Direct forecast -> full (B,T,D) physical-unit prediction (first L observed)."""
    B, T, D = X_norm.shape
    lb, _, s = make_tensors(X_norm, s_std)
    horizon = model(lb, s).cpu().numpy().transpose(0, 2, 1)     # (B, H, D)
    pred_norm = X_norm.copy()
    pred_norm[:, L_LOOKBACK:, :] = horizon
    return pred_norm * sig_std + sig_mean


def timed_inference(model, X_norm, s_std, reps=5):
    lb, _, s = make_tensors(X_norm, s_std)
    B = X_norm.shape[0]
    with torch.no_grad():
        _ = model(lb, s)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        with torch.no_grad():
            _ = model(lb, s)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    per_traj = (time.time() - t0) / reps / B
    return np.full(B, per_traj)


def main():
    C.set_all_seeds(C.SEED)
    sig_mean, sig_std = C.load_norm_stats()

    Xtr, _, Utr_phys, _ = C.load_split("train")
    Xva, _, Uva_phys, _ = C.load_split("val")
    Xte, _, Ute_phys, Xte_phys = C.load_split("test")

    # control step level s = V_usv,step (final level of the step); standardise on train
    s_tr_raw = Utr_phys[:, -1, 0]
    s_va_raw = Uva_phys[:, -1, 0]
    s_te_raw = Ute_phys[:, -1, 0]
    s_mean, s_sd = s_tr_raw.mean(), s_tr_raw.std() + 1e-8
    std_tr = (s_tr_raw - s_mean) / s_sd
    std_va = (s_va_raw - s_mean) / s_sd
    std_te = (s_te_raw - s_mean) / s_sd

    D = Xtr.shape[2]
    H = C.T_STEPS - L_LOOKBACK
    tr = make_tensors(Xtr, std_tr)
    va = make_tensors(Xva, std_va)

    summary = {"config": {
        "lookback_L": L_LOOKBACK, "horizon_H": H, "kernel": KERNEL,
        "control_feature": "V_usv,step (step level), train-standardised",
        "split_eval": "test1 (seed-42)", "epochs_max": EPOCHS,
        "patience": PATIENCE, "lr": LR, "seed": C.SEED,
        "note": ("Direct non-autoregressive multi-step forecasters (Zeng et al. "
                 "AAAI 2023). Look-back = 10 steps (same warm-up as RNN "
                 "baselines); control step level given as an extra feature.")
    }}

    for name, cls in [("DLinear", DLinear), ("NLinear", NLinear)]:
        print(f"\n=== {name} ===")
        C.set_all_seeds(C.SEED)   # identical init/opt seed for both
        model = cls(D, L_LOOKBACK, H).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        model = train_model(model, tr, va, name)

        pred_phys = full_prediction(model, Xte, std_te, sig_mean, sig_std)
        inf_t = timed_inference(model, Xte, std_te)

        res = C.compute_results_dict(Xte_phys, pred_phys, Ute_phys, inf_t, n_params)
        # pooled R^2 (matches how MLP / AR(20) baselines are reported in the notebook)
        res["global_r2_flat"] = C.r2_flat(Xte_phys.reshape(-1, D), pred_phys.reshape(-1, D))

        out_pkl = os.path.join(C.RESULT_DIR, name.lower(),
                               f"{name.lower()}_postprocessing_results.pkl")
        C.save_pickle(res, out_pkl)
        torch.save({"state_dict": model.state_dict(), "n_params": n_params,
                    "lookback": L_LOOKBACK, "horizon": H,
                    "s_mean": float(s_mean), "s_sd": float(s_sd)},
                   os.path.join(C.CKPT_DIR, f"{name.lower()}_baseline.pt"))

        pct = res["pct_per_traj_ps"]
        summary[name] = {
            "n_params": int(n_params),
            "pct_rmse_mean": float(pct.mean()),
            "pct_rmse_ci95": float(C.ci95(pct)),
            "pct_rmse_sd": float(pct.std(ddof=1)),
            "r2_pooled": float(res["global_r2_flat"]),
            "r2_mean_per_signal": float(np.mean(res["r2_mean"])),
            "inference_time_s_per_traj": float(inf_t.mean()),
            "per_signal_pct_mean": {C.SIG_NAMES[j]: float(res["pct_mean"][j]) for j in range(D)},
        }
        print(f"  {name} test1: %RMSE={pct.mean():.2f} +- {C.ci95(pct):.2f}  "
              f"R2(pooled)={res['global_r2_flat']:.3f}  params={n_params:,}  "
              f"-> {out_pkl}")

    out_json = os.path.join(C.REV2_DIR, "task_a_dlinear_nlinear.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
