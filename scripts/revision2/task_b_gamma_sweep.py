#!/usr/bin/env python3
"""
TASK B (editor mandatory #4 / R1.2) -- control-gain gamma sweep (inference only)
================================================================================
Loads the *frozen* final Koopman model and overrides ONLY the control gain
gamma (KoopmanNetControl_v2.control_gain, which scales the control term
B_u = gamma * tanh(control_net(u_t))) over the grid

    gamma in {0.0, 0.05, 0.1, 0.2, 0.5, 1.0}

re-running full test-set inference at each value.  NOTHING is retrained.

  * gamma = 0.1 is the trained value -> reproduces the manuscript Koopman row.
  * gamma = 0.0 doubles as a **control-net ablation**: the control term is
    switched off entirely, so the model cannot track the pre-load step.

Outputs
  results/revision2/task_b_gamma_sweep.csv
  results/figures/figureS_gamma_control_sweep.{svg,png}
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

GAMMA_GRID = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


def main():
    C.set_all_seeds(C.SEED)
    device = C.get_device()
    model, params, n_params, ckpt = C.load_koopman(device)
    trained_gamma = float(model.control_gain)
    print(f"Frozen checkpoint: {ckpt}")
    print(f"Trained control_gain (gamma) = {trained_gamma}")

    Xn, Un, Uph, Xph = C.load_split("test")
    sig_mean, sig_std = C.load_norm_stats()
    B, T, D = Xph.shape

    rows = []
    for g in GAMMA_GRID:
        pred = C.koopman_rollout(model, Xn[:, 0, :], Un, control_gain=g,
                                 control_act=torch.tanh, device=device) * sig_std + sig_mean
        pct = np.array([C.pct_rmse_ps(Xph[i], pred[i]) for i in range(B)])
        r2_pool = C.r2_flat(Xph.reshape(-1, D), pred.reshape(-1, D))
        r2_ps = np.mean([
            [C.r2_flat(Xph[i, :, j:j+1], pred[i, :, j:j+1]) for j in range(D)]
            for i in range(B)
        ])
        tag = ""
        if abs(g - trained_gamma) < 1e-9:
            tag = "trained"
        elif g == 0.0:
            tag = "control ablation"
        rows.append({
            "gamma": g,
            "pct_rmse_mean": pct.mean(),
            "pct_rmse_ci95": C.ci95(pct),
            "pct_rmse_sd": pct.std(ddof=1),
            "r2_pooled": r2_pool,
            "r2_mean_per_signal_per_traj": r2_ps,
            "note": tag,
        })
        print(f"  gamma={g:<5}  %RMSE={pct.mean():6.2f} +- {C.ci95(pct):.2f}  "
              f"R2(pooled)={r2_pool:6.3f}  {tag}")

    out_csv = os.path.join(C.REV2_DIR, "task_b_gamma_sweep.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved -> {out_csv}")

    # ── figure ──────────────────────────────────────────────────────────────
    g_arr   = np.array([r["gamma"] for r in rows])
    pct_arr = np.array([r["pct_rmse_mean"] for r in rows])
    pct_ci  = np.array([r["pct_rmse_ci95"] for r in rows])
    r2_arr  = np.array([r["r2_pooled"] for r in rows])
    R2_FLOOR = -5.0   # clip catastrophic (diverged) R^2 for readability
    AXLAB, TICK = 14, 12
    c1, c2 = "#5a286b", "#dda251"
    c_tr, c_ab = "#2e7d32", "#ac4484"                 # trained / control-off accents
    gi_tr = int(np.argmin(np.abs(g_arr - trained_gamma)))
    gi_ab = int(np.argmin(np.abs(g_arr - 0.0)))

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 4.7))

    # Panel (a): %RMSE (log) with shaded CI silhouette (values span >15 decades)
    e = np.minimum(pct_ci, pct_arr * 0.99)
    ax1.fill_between(g_arr, pct_arr - e, pct_arr + e, color=c1, alpha=0.18)
    ax1.plot(g_arr, pct_arr, "o-", color=c1, lw=2.3, ms=6, zorder=4)
    ax1.set_yscale("log")
    ax1.axvline(trained_gamma, color="k", ls=":", lw=1.3, alpha=0.7)
    # flag the trained and control-off points with open rings
    ax1.scatter([g_arr[gi_tr]], [pct_arr[gi_tr]], s=150, facecolors="none",
                edgecolors=c_tr, linewidths=2.4, zorder=6)
    ax1.scatter([g_arr[gi_ab]], [pct_arr[gi_ab]], s=150, facecolors="none",
                edgecolors=c_ab, linewidths=2.4, zorder=6)
    # key values placed in the empty upper-left (no curve crosses there)
    notes_a = [(f"$\\gamma$=0.1 (trained, optimal): {pct_arr[gi_tr]:.1f}%", c_tr),
               (f"$\\gamma$=0 (control off): {pct_arr[gi_ab]:.0f}%", c_ab),
               (r"$\gamma\geq$0.5: diverged", "#555555")]
    # place the notes just to the RIGHT of the trained-gamma line (x in data, y in axes)
    from matplotlib.transforms import blended_transform_factory
    trans_a = blended_transform_factory(ax1.transData, ax1.transAxes)
    for i, (txt, col) in enumerate(notes_a):
        ax1.text(trained_gamma + 0.03, 0.97 - i * 0.08, txt, transform=trans_a,
                 va="top", ha="left", fontsize=TICK - 1, color=col)
    ax1.set_xlabel(r"Control gain $\gamma$", fontsize=AXLAB)
    ax1.set_ylabel("%RMSE (log scale)", fontsize=AXLAB)
    ax1.tick_params(labelsize=TICK)
    ax1.grid(axis="y", which="both", ls="--", alpha=0.3); ax1.set_axisbelow(True)
    ax1.set_title("(a)", fontsize=13, loc="left")

    # Panel (b): pooled R^2 (clipped at the floor for readability)
    r2_clip = np.clip(r2_arr, R2_FLOOR, 1.0)
    ax3.plot(g_arr, r2_clip, "s-", color=c2, lw=2.3, ms=6, zorder=4)
    ax3.axhline(0, color="#888888", ls=":", lw=0.9)
    ax3.axvline(trained_gamma, color="k", ls=":", lw=1.3, alpha=0.7)
    ax3.scatter([g_arr[gi_tr]], [r2_clip[gi_tr]], s=150, facecolors="none",
                edgecolors=c_tr, linewidths=2.4, zorder=6)
    ax3.scatter([g_arr[gi_ab]], [r2_clip[gi_ab]], s=150, facecolors="none",
                edgecolors=c_ab, linewidths=2.4, zorder=6)
    ax3.set_ylim(R2_FLOOR - 0.4, 1.15)
    ax3.set_xlabel(r"Control gain $\gamma$", fontsize=AXLAB)
    ax3.set_ylabel(r"$R^2$ (pooled, clipped at $-5$)", fontsize=AXLAB)
    ax3.tick_params(labelsize=TICK)
    ax3.grid(axis="y", ls="--", alpha=0.3); ax3.set_axisbelow(True)
    ax3.set_title("(b)", fontsize=13, loc="left")
    off = [g for g, v in zip(g_arr, r2_arr) if v < R2_FLOOR]
    if off:
        ax3.text(0.97, 0.55, "diverged (off-scale):\n" + ", ".join(f"$\\gamma$={g}" for g in off),
                 transform=ax3.transAxes, ha="right", va="center",
                 fontsize=TICK - 2, style="italic", color="#555555")

    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(C.FIG_DIR, f"figureS_gamma_control_sweep.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> figures/figureS_gamma_control_sweep.svg + .png")


if __name__ == "__main__":
    main()
