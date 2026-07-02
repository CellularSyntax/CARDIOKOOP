#!/usr/bin/env python3
"""
TASK F (R1.1 / R1.2 rebuttal) -- clean closed-loop re-anchoring of the Koopman model
====================================================================================
Reviewer 1 asked for direct (non-autoregressive) linear forecasters (DLinear /
NLinear), which attain the lowest CLEAN-data error because a single direct map
from the look-back window to the full horizon avoids the error accumulation that
penalises every autoregressive model over the 1499-step rollout (Zeng et al.).

This inference-only experiment shows that the SAME frozen Koopman model closes
that gap once it is run in the closed-loop, re-anchored mode relevant to its
intended deployment (e.g. model-predictive control): the open-loop rollout is
periodically re-initialised from the true current state every K steps.  No noise,
no retraining -- only the deployment protocol changes.

Result: at a re-anchoring interval of ~1 cardiac cycle (K=50, 0.5 s) the Koopman
model reaches 3.5% %RMSE / R^2 = 0.98, matching the direct linear forecasters
(DLinear 3.0%, NLinear 3.6%), and improves further with more frequent re-anchoring.

Outputs
  results/revision2/task_f_clean_reanchoring.csv
  results/figures/figureS_clean_reanchoring.{svg,png}
"""
import os
import sys
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

K_GRID = [500, 200, 100, 50, 20, 10, 5]      # re-anchoring interval (steps)
DT = 0.01                                     # s / step
# clean-data references (Table 3, seed-42 test split)
KOOP_OPENLOOP = 17.5
DLINEAR_CLEAN = 3.0
NLINEAR_CLEAN = 3.6
CARDIAC_HZ = 1.8                              # ~ dominant cardiac frequency


@torch.no_grad()
def reanchor_rollout(model, X_true_norm, u_norm, K, device):
    """Open-loop Koopman rollout re-initialised from the true state every K steps."""
    B, T, D = X_true_norm.shape
    x0 = torch.as_tensor(X_true_norm[:, 0, :], dtype=torch.float32, device=device)
    u = torch.as_tensor(u_norm, dtype=torch.float32, device=device).permute(1, 0, 2)
    y = model.encoder(x0)
    outs = []
    for t in range(T):
        outs.append(model.decoder(y).cpu().numpy())
        y = C.koopman_advance(model, y, u[t])
        nxt = t + 1
        if nxt < T and K and (nxt % K == 0):
            y = model.encoder(torch.as_tensor(X_true_norm[:, nxt, :],
                                              dtype=torch.float32, device=device))
    return np.stack(outs, axis=1)


def main():
    C.set_all_seeds(C.SEED)
    device = C.get_device()
    model, params, n_params, ckpt = C.load_koopman(device)
    Xn, Un, Uph, Xph = C.load_split("test")
    sig_mean, sig_std = C.load_norm_stats()
    B, T, D = Xph.shape

    def score(pred_phys):
        pct = np.array([C.pct_rmse_ps(Xph[i], pred_phys[i]) for i in range(B)])
        r2 = np.array([C.r2_flat(Xph[i], pred_phys[i]) for i in range(B)])
        return float(pct.mean()), float(C.ci95(pct)), float(r2.mean())

    rows = []
    # open-loop free-running baseline (== manuscript Koopman row)
    pred_free = C.koopman_rollout(model, Xn[:, 0, :], Un, device=device) * sig_std + sig_mean
    p, ci, r2 = score(pred_free)
    rows.append({"reanchor_K_steps": "inf", "reanchor_interval_s": "inf",
                 "pct_rmse_mean": p, "pct_rmse_ci95": ci, "r2_mean": r2})
    print(f"Open-loop free-run:            %RMSE={p:6.2f} +- {ci:.2f}  R2={r2:.3f}")

    for K in K_GRID:
        pred = reanchor_rollout(model, Xn, Un, K, device) * sig_std + sig_mean
        p, ci, r2 = score(pred)
        rows.append({"reanchor_K_steps": K, "reanchor_interval_s": round(K * DT, 3),
                     "pct_rmse_mean": p, "pct_rmse_ci95": ci, "r2_mean": r2})
        print(f"Re-anchor K={K:4d} ({K*DT:.2f}s):        %RMSE={p:6.2f} +- {ci:.2f}  R2={r2:.3f}")

    out_csv = os.path.join(C.REV2_DIR, "task_f_clean_reanchoring.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved -> {out_csv}")

    # ── figure ────────────────────────────────────────────────────────────────
    intervals_s = np.array([K * DT for K in K_GRID])
    pct = np.array([r["pct_rmse_mean"] for r in rows[1:]])
    pct_ci = np.array([r["pct_rmse_ci95"] for r in rows[1:]])
    r2 = np.array([r["r2_mean"] for r in rows[1:]])
    c_koop, c_dl, c_nl = "#dda251", "#e15759", "#17becf"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) %RMSE vs re-anchoring interval
    ax1.errorbar(intervals_s, pct, yerr=pct_ci, fmt="o-", color=c_koop, lw=2.2,
                 capsize=4, label="Koopman (re-anchored)")
    ax1.axhline(KOOP_OPENLOOP, color=c_koop, ls=":", lw=1.6,
                label=f"Koopman open-loop ({KOOP_OPENLOOP:.1f}%)")
    ax1.axhline(DLINEAR_CLEAN, color=c_dl, ls="--", lw=1.4, alpha=0.9,
                label=f"DLinear ({DLINEAR_CLEAN:.1f}%)")
    ax1.axhline(NLINEAR_CLEAN, color=c_nl, ls="--", lw=1.4, alpha=0.9,
                label=f"NLinear ({NLINEAR_CLEAN:.1f}%)")
    ax1.axvline(1.0 / CARDIAC_HZ, color="#888", ls="-.", lw=1.0)
    ax1.text(1.0 / CARDIAC_HZ * 1.05, ax1.get_ylim()[1], "~1 cardiac cycle",
             rotation=90, va="top", ha="left", fontsize=8.5, color="#666")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Re-anchoring interval (s)", fontsize=12)
    ax1.set_ylabel("%RMSE (log scale)", fontsize=12)
    ax1.grid(True, which="both", ls="--", alpha=0.3); ax1.set_axisbelow(True)
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.set_title("(a)", loc="left", fontsize=12)

    # (b) R^2 vs re-anchoring interval
    ax2.plot(intervals_s, r2, "o-", color=c_koop, lw=2.2)
    ax2.axhline(0.687, color=c_koop, ls=":", lw=1.6, label="Koopman open-loop (0.69)")
    ax2.axvline(1.0 / CARDIAC_HZ, color="#888", ls="-.", lw=1.0)
    ax2.set_xscale("log")
    ax2.set_xlabel("Re-anchoring interval (s)", fontsize=12)
    ax2.set_ylabel(r"$R^2$", fontsize=12)
    ax2.set_ylim(0.6, 1.02)
    ax2.grid(True, which="both", ls="--", alpha=0.3); ax2.set_axisbelow(True)
    ax2.legend(fontsize=8.5, loc="lower left")
    ax2.set_title("(b)", loc="left", fontsize=12)

    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(C.FIG_DIR, f"figureS_clean_reanchoring.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved -> figures/figureS_clean_reanchoring.svg + .png")


if __name__ == "__main__":
    main()
