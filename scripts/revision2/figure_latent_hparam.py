#!/usr/bin/env python3
"""
Figure S4 -- latent-dimension / hyperparameter ablation:
  (a) validation %RMSE heatmap over (#complex pairs, #real modes),
  (b) marginal effect of #complex pairs (n_real = 4 fixed),
  (c) sensitivity of validation %RMSE to the control gain (Optuna trials).

Standalone, reproducible rebuild of the former notebook cell, restyled to match
the other revision-2 supplementary figures (no subplot titles, panel letters,
enlarged fonts, silhouette bands, no overlapping annotations).

Reads:  results/latent_ablation.json, results/optuna_runs/koopman_hyperparam_search_*.csv
Writes: results/figures/figureS4.{svg,png}
"""
import os
import sys
import json
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

PAL = ["#dda251", "#5a286b", "#646464", "#ac4484"]   # line / highlight / scatter / -
AXLAB, TICK = 14, 12


def main():
    with open(os.path.join(C.RESULT_DIR, "latent_ablation.json")) as f:
        la = json.load(f)

    opt_csv = sorted(glob.glob(os.path.join(C.RESULT_DIR, "optuna_runs",
                                            "koopman_hyperparam_search_*.csv")))[-1]
    dfo = pd.read_csv(opt_csv)
    dfc = dfo[dfo["State"] == "COMPLETE"]
    cg = dfc["Param control_gain"].to_numpy(dtype=float)
    cv = dfc["Value"].to_numpy(dtype=float)

    # bin control_gain into 8 bins -> mean +- SD
    bins = np.linspace(cg.min() - 1e-4, cg.max() + 1e-4, 9)
    idx = np.digitize(cg, bins) - 1
    bc, bm, bsd = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        sub = cv[m]
        bc.append((bins[b] + bins[b + 1]) / 2)
        bm.append(sub.mean())
        bsd.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)
    bc, bm, bsd = map(np.array, (bc, bm, bsd))

    # heatmap over (n_complex, n_real)
    rows = la["by_nc_nr"]
    ncs = sorted({r["n_complex"] for r in rows})
    nrs = sorted({r["n_real"] for r in rows})
    hm = np.full((len(nrs), len(ncs)), np.nan)
    hm_sd = np.full_like(hm, np.nan)
    hm_n = np.zeros_like(hm)
    for r in rows:
        i, j = nrs.index(r["n_real"]), ncs.index(r["n_complex"])
        hm[i, j], hm_sd[i, j], hm_n[i, j] = r["pct_mean"], r["pct_sd"], r["n_trials"]

    marg = la["by_n_complex_nr_fixed4"]
    mx = np.array([r["n_complex"] for r in marg])
    my = np.array([r["pct_mean"] for r in marg])
    me = np.array([r["pct_sd"] for r in marg])
    bi = int(np.argmin(my))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.subplots_adjust(wspace=0.30)

    # ── (a) heatmap ────────────────────────────────────────────────────────────
    cmap = plt.get_cmap("YlOrRd_r")
    norm = Normalize(vmin=np.nanmin(hm) * 0.97, vmax=np.nanmax(hm) * 1.03)
    im = ax1.imshow(hm, aspect="auto", cmap=cmap, norm=norm)
    ax1.set_xticks(range(len(ncs))); ax1.set_xticklabels([str(c) for c in ncs], fontsize=TICK)
    ax1.set_yticks(range(len(nrs))); ax1.set_yticklabels([str(r) for r in nrs], fontsize=TICK)
    ax1.set_xlabel("# complex pairs", fontsize=AXLAB)
    ax1.set_ylabel("# real modes", fontsize=AXLAB)
    ax1.tick_params(labelsize=TICK)
    for i in range(len(nrs)):
        for j in range(len(ncs)):
            if np.isnan(hm[i, j]):
                continue
            rgba = cmap(norm(hm[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax1.text(j, i, f"{hm[i, j]:.2f}±{hm_sd[i, j]:.2f}\n(n={int(hm_n[i, j])})",
                     ha="center", va="center", fontsize=TICK - 2,
                     color="white" if lum < 0.5 else "black")
    cb = fig.colorbar(im, ax=ax1, shrink=0.85, pad=0.02)
    cb.set_label("Mean validation %RMSE", fontsize=TICK); cb.ax.tick_params(labelsize=TICK - 1)
    ax1.set_xticks(np.arange(len(ncs)) - 0.5, minor=True)
    ax1.set_yticks(np.arange(len(nrs)) - 0.5, minor=True)
    ax1.grid(which="minor", color="white", linewidth=1.5)
    ax1.tick_params(which="minor", bottom=False, left=False)
    ax1.set_title("(a)", loc="left", fontsize=13)

    # ── (b) marginal effect of # complex pairs (n_real = 4) ────────────────────
    ax2.fill_between(mx, my - me, my + me, color=PAL[0], alpha=0.2)
    ax2.plot(mx, my, "o-", color=PAL[0], lw=2.6, ms=8)
    ax2.scatter([mx[bi]], [my[bi]], s=170, facecolors="none",
                edgecolors=PAL[1], linewidths=2.4, zorder=6)
    ax2.set_xticks(mx); ax2.tick_params(labelsize=TICK)
    ax2.set_ylim(top=(my + me).max() * 1.35)                 # headroom for the note
    ax2.set_xlabel("# complex pairs", fontsize=AXLAB)
    ax2.set_ylabel("Mean validation %RMSE", fontsize=AXLAB)
    ax2.yaxis.grid(True, ls="--", alpha=0.3); ax2.set_axisbelow(True)
    ax2.text(0.03, 0.97, f"n$_{{real}}$ = 4 (fixed)\nbest: {mx[bi]} pairs ({my[bi]:.2f}%)",
             transform=ax2.transAxes, va="top", ha="left", fontsize=TICK - 1,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9))
    ax2.set_title("(b)", loc="left", fontsize=13)

    # ── (c) sensitivity to control gain ────────────────────────────────────────
    ax3.scatter(cg, cv, alpha=0.30, s=20, color=PAL[2], zorder=2,
                label=f"Individual trials (n = {len(cg)})")
    ax3.fill_between(bc, bm - bsd, bm + bsd, color=PAL[0], alpha=0.2, zorder=3)
    ax3.plot(bc, bm, "o-", color=PAL[0], lw=2.6, ms=8, zorder=4, label="Binned mean ± SD")
    ax3.set_xlabel("Control gain", fontsize=AXLAB)
    ax3.set_ylabel("Validation %RMSE", fontsize=AXLAB)
    ax3.tick_params(labelsize=TICK)
    ax3.set_ylim(top=max(cv.max(), (bm + bsd).max()) * 1.12)  # headroom for the legend
    ax3.yaxis.grid(True, ls="--", alpha=0.3); ax3.set_axisbelow(True)
    ax3.legend(fontsize=TICK - 1, frameon=True, framealpha=0.92, loc="upper left")
    # (the flat binned mean shows the insensitivity; stated in the caption rather than
    #  overlaid on the dense scatter)
    ax3.set_title("(c)", loc="left", fontsize=13)

    fig.tight_layout()
    out = os.path.join(C.FIG_DIR, "figureS4")
    for ext in ("svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"best (n_real=4): {mx[bi]} complex pairs -> {my[bi]:.2f}% ; "
          f"control_gain range [{cg.min():.3f}, {cg.max():.3f}], {len(cg)} trials")
    print("Saved -> figures/figureS4.svg + .png")


if __name__ == "__main__":
    main()
