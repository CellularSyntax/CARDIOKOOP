#!/usr/bin/env python3
"""
Regenerate Figure-5 panels a and g–j (formerly f–i) INCLUDING the two new
DLinear / NLinear baselines, reusing the *exact* plotting code from the manuscript
notebook (cells 13, 17, 21) so the aspect ratios and styling are unchanged.

Outputs (results/figures/):
  figure5a_rev2.{svg,png}                 panel a  (P_lv traces, 8 models x 2 rows)
  figure5_panels_gj_rev2.{svg,png}        panels g,h,i,j (R2 violin | per-signal R2
                                          heatmap | %RMSE vs SNR | R2 vs SNR heatmap)

The author integrates these into the multipanel Figure 5 manually.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "notebooks"))
import _common as C
import build_tables_figures as B          # reuse pred assembly (no side effects on import)
import notebook_utils as nu               # reuse coloured_pred

FIGDIR = C.FIG_DIR
models_all = C.MODELS_ALL                 # 8 models
custom_palette = C.PALETTE_HEX
_pal = dict(zip(models_all, custom_palette))
ylab = C.YLAB
SIG_G = [2, 7, 8, 10, 11]
ylab_g = [ylab[i] for i in SIG_G]
R2_CLIP = -1.0   # violin (panel g) floor; MLP is the only model below and is μ-labelled
x_labels = ["Clean", "30 dB", "20 dB", "10 dB", "5 dB"]
x_pos = np.arange(len(x_labels))
lw_koop, lw_bl = 2.8, 1.7
SNR = [30, 20, 10, 5]


def assemble_predictions():
    Xn, Un, Uph, Xte = C.load_split("test")
    sig_mean, sig_std = C.load_norm_stats()
    pred = {}
    pred["Koopman"], _ = B.koopman_pred(Xn, Un, sig_mean, sig_std)
    for m in ["GRU", "LSTM", "BiLSTM", "DLinear", "NLinear"]:
        r = C.load_pickle(os.path.join(C.RESULT_DIR, m.lower(), f"{m.lower()}_postprocessing_results.pkl"))
        pred[m] = np.asarray(r["pred_per_trajectory"])
    pred["MLP"], _, _ = B.mlp_pred(Xn, sig_mean, sig_std)
    pred["AR(20)"], _, _ = B.ar_pred(Xn, sig_mean, sig_std)
    return Xte, pred


def build_curve_data():
    koop = json.load(open(os.path.join(C.RESULT_DIR, "noise_robustness.json")))
    bl = json.load(open(os.path.join(C.RESULT_DIR, "noise_robustness_baselines.json")))
    lin = json.load(open(os.path.join(C.REV2_DIR, "noise_robustness_linear.json")))
    B_ = 50

    def _from(d):
        pm = np.array([d["clean_baseline"]["pct_rmse_mean"]] + [d["noisy"][str(s)]["pct_rmse_mean"] for s in SNR])
        pc = np.array([d["clean_baseline"]["pct_rmse_ci95"]] + [d["noisy"][str(s)]["pct_rmse_ci95"] for s in SNR])
        rm = np.array([d["clean_baseline"]["r2_mean"]] + [d["noisy"][str(s)]["r2_mean"] for s in SNR])
        rc = np.array([d["clean_baseline"]["r2_sd"] * 1.96 / np.sqrt(B_)] +
                      [d["noisy"][str(s)]["r2_sd"] * 1.96 / np.sqrt(B_) for s in SNR])
        return pm, pc, rm, rc

    cd = {"Koopman": _from(koop)}
    for m in ["GRU", "LSTM", "BiLSTM", "MLP", "AR(20)"]:
        cd[m] = _from(bl[m])
    for m in ["DLinear", "NLinear"]:
        cd[m] = _from(lin[m])
    return cd


# ── R² helpers (verbatim from notebook cell 17) ──────────────────────────────
def _r2_per_traj(true_phys, pred_phys):
    N = true_phys.shape[0]
    t = true_phys.reshape(N, -1); p = pred_phys.reshape(N, -1)
    ss_res = ((t - p) ** 2).sum(1)
    ss_tot = ((t - t.mean(1, keepdims=True)) ** 2).sum(1) + 1e-8
    return 1.0 - ss_res / ss_tot


def _r2_per_sig(true_phys, pred_phys):
    err = true_phys - pred_phys
    ss_res = (err ** 2).sum(1)
    ss_tot = ((true_phys - true_phys.mean(1, keepdims=True)) ** 2).sum(1) + 1e-8
    return (1.0 - ss_res / ss_tot).mean(0)


def panel_a(true, pred, models, out_stub):
    rmse = np.sqrt(((true - pred["Koopman"]) ** 2).mean(axis=(1, 2)))
    idxs = np.argsort(rmse)[[0, -1]]
    plv = 7
    n_cols = len(models)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4.0, 4.5))
    axes = np.atleast_2d(axes)
    plt.subplots_adjust(left=0.05, right=0.94, hspace=0.35, wspace=0.22)
    for r, i in enumerate(idxs):
        y = true[i][:, plv]
        errs = [np.abs(pred[m][i][:, plv] - y) for m in models]
        vmax = np.max([e.max() for e in errs]) + 1e-12
        norm = colors.Normalize(vmin=0, vmax=vmax)
        for c, m in enumerate(models):
            ax = axes[r, c]
            ax.plot(y, "k", lw=1.6, label="Ground Truth")
            nu.coloured_pred(ax, y, pred[m][i][:, plv], plt.get_cmap("plasma"), norm)
            ax.axvline(500, color="k", ls="--", lw=2.6)
            if r == 0:
                ax.set_title(m, fontsize=16)
            if c == 0:
                ax.set_ylabel(f"{ylab[plv]} [mmHg]", fontsize=16)
            if r == 1:
                ax.set_xlabel("Time Step", fontsize=16)
            ax.tick_params(labelsize=13)
            ax.grid(False)
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        fig.colorbar(sm, ax=axes[r, :], orientation="vertical", fraction=0.012, pad=0.01
                     ).ax.set_ylabel("|err|", fontsize=15)
    axes[0, -1].legend(frameon=False, fontsize=11, loc="lower left", bbox_to_anchor=(1.01, 1.02))
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{out_stub}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> figures/{out_stub}.svg + .png ({len(models)} models)")


def panels_gj(true, pred, curve_data, models, out_stub):
    models_all = models          # roster for this figure; _pal (module-level) stays keyed to all 8
    r2_traj_all = {m: _r2_per_traj(true, pred[m]) for m in models_all}
    r2_sig_all = {m: _r2_per_sig(true, pred[m]) for m in models_all}
    _df_violin = pd.concat(
        [pd.DataFrame({"Model": m, "R2": np.clip(r2_traj_all[m], R2_CLIP, None)}) for m in models_all],
        ignore_index=True)

    fig = plt.figure(figsize=(24, 3))
    gs = GridSpec(1, 4, width_ratios=[3.5, 4.5, 5.0, 4.5], figure=fig, wspace=0.35)
    plt.rcParams["axes.grid"] = False

    # (g) violin R² per trajectory
    ax0 = fig.add_subplot(gs[0])
    sns.violinplot(data=_df_violin, x="Model", y="R2", order=models_all, hue="Model",
                   palette=_pal, legend=False, inner=None, cut=0, linewidth=0.8, alpha=0.55, ax=ax0)
    sns.stripplot(data=_df_violin, x="Model", y="R2", order=models_all, hue="Model",
                  palette=_pal, legend=False, size=2.8, alpha=0.55, jitter=True, ax=ax0)
    for xi, m in enumerate(models_all):
        mu = np.clip(r2_traj_all[m], R2_CLIP, None).mean()
        ax0.scatter(xi, mu, marker="D", s=38, color="white", zorder=6, edgecolors=_pal[m], linewidths=1.8)
    for xi, m in enumerate(models_all):
        mu = r2_traj_all[m].mean()
        if mu < R2_CLIP:
            ax0.text(xi, R2_CLIP + 0.10, f"μ={mu:.0f}", ha="center", va="bottom",
                     fontsize=8.5, color="black", style="italic")
    ax0.axhline(0, color="#888888", lw=0.9, ls=":", alpha=0.7)
    ax0.set_ylim(R2_CLIP, 1.05)
    ax0.set_xticks(range(len(models_all)))
    ax0.set_xticklabels(models_all, rotation=30, ha="right", fontsize=13)
    ax0.set_xlabel(""); ax0.set_ylabel(r"$R^2$ per trajectory", fontsize=15)
    ax0.tick_params(labelsize=14); ax0.yaxis.grid(True, linestyle="--", alpha=0.35); ax0.set_axisbelow(True)

    # (h) per-signal R² heatmap
    ax1 = fig.add_subplot(gs[1])
    _hm_g = np.array([r2_sig_all[m][SIG_G] for m in models_all])
    im_g = ax1.imshow(np.clip(_hm_g, -3.0, 1.0), aspect="auto", cmap="RdYlGn",
                      vmin=-3.0, vmax=1.0, interpolation="nearest")
    for ri, m in enumerate(models_all):
        for ci in range(len(SIG_G)):
            val = _hm_g[ri, ci]
            ax1.text(ci, ri, f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}",
                     ha="center", va="center", fontsize=9.5, color="black")
    ax1.set_xticks(np.arange(len(SIG_G))); ax1.set_xticklabels(ylab_g, fontsize=12)
    ax1.set_yticks(np.arange(len(models_all))); ax1.set_yticklabels(models_all, fontsize=12)
    ax1.set_xlabel(r"$R^2$ (mean across trajectories)", fontsize=13)
    ax1.set_xticks(np.arange(len(SIG_G)) - 0.5, minor=True)
    ax1.set_yticks(np.arange(len(models_all)) - 0.5, minor=True)
    ax1.grid(which="minor", color="white", linewidth=1.2); ax1.tick_params(which="minor", bottom=False, left=False)
    cb_g = fig.colorbar(im_g, ax=ax1, shrink=0.85, pad=0.02)
    cb_g.set_label(r"$R^2$ (clipped at −3)", fontsize=10); cb_g.ax.tick_params(labelsize=9)

    # (i) %RMSE vs SNR
    ax2 = fig.add_subplot(gs[2])
    for m in models_all:
        pm, pc, rm, rc = curve_data[m]
        is_k = (m == "Koopman")
        lw = 3.6 if is_k else lw_bl
        ms = 9 if is_k else 5
        z = 6 if is_k else 3
        ax2.fill_between(x_pos, np.clip(pm - pc, 1e-3, None), pm + pc, color=_pal[m], alpha=0.15, zorder=z - 1)
        ax2.plot(x_pos, pm, color=_pal[m], linewidth=lw, marker="o", markersize=ms, label=m, zorder=z)
    ax2.set_yscale("log"); ax2.set_xticks(x_pos); ax2.set_xticklabels(x_labels, fontsize=13)
    ax2.set_xlabel("Input Noise Level (SNR)", fontsize=14); ax2.set_ylabel("%RMSE (log scale)", fontsize=14)
    ax2.tick_params(labelsize=13); ax2.yaxis.grid(True, which="both", linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)
    ymax_i = max((curve_data[m][0] + curve_data[m][1]).max() for m in models_all)
    ax2.set_ylim(top=ymax_i * 15)   # headroom so the legend clears the curves
    ax2.legend(fontsize=9.5, frameon=True, loc="upper left", ncol=2)

    # (j) R² vs SNR heatmap
    ax3 = fig.add_subplot(gs[3])
    _hm = np.array([curve_data[m][2] for m in models_all])
    im = ax3.imshow(np.clip(_hm, -3.0, 1.0), aspect="auto", cmap="RdYlGn",
                    vmin=-3.0, vmax=1.0, interpolation="nearest")
    for ri, m in enumerate(models_all):
        for ci in range(len(x_labels)):
            val = _hm[ri, ci]
            ax3.text(ci, ri, f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}",
                     ha="center", va="center", fontsize=9.5, color="black")
    ax3.set_xticks(np.arange(len(x_labels))); ax3.set_xticklabels(x_labels, fontsize=12)
    ax3.set_yticks(np.arange(len(models_all))); ax3.set_yticklabels(models_all, fontsize=12)
    ax3.set_xlabel("Input Noise Level (SNR)", fontsize=14); ax3.set_ylabel(r"$R^2$", fontsize=14)
    ax3.set_xticks(np.arange(len(x_labels)) - 0.5, minor=True)
    ax3.set_yticks(np.arange(len(models_all)) - 0.5, minor=True)
    ax3.grid(which="minor", color="white", linewidth=1.2); ax3.tick_params(which="minor", bottom=False, left=False)
    cb = fig.colorbar(im, ax=ax3, shrink=0.85, pad=0.02)
    cb.set_label(r"$R^2$ (clipped at −3)", fontsize=10); cb.ax.tick_params(labelsize=9)

    plt.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{out_stub}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> figures/{out_stub}.svg + .png ({len(models)} models)")


def main():
    C.set_all_seeds(C.SEED)
    true, pred = assemble_predictions()
    curve_data = build_curve_data()
    # reduced 6-model roster -> main-text Figure 5 panels
    panel_a(true, pred, C.MODELS_MAIN, "figure5a_rev2")
    panels_gj(true, pred, curve_data, C.MODELS_MAIN, "figure5_panels_gj_rev2")
    # full 8-model roster -> supplementary Figure S9 panels
    panel_a(true, pred, C.MODELS_ALL, "figureS9a_full8")
    panels_gj(true, pred, curve_data, C.MODELS_ALL, "figureS9_panels_gj_full8")


if __name__ == "__main__":
    main()
