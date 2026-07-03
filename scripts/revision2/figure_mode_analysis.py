#!/usr/bin/env python3
"""
Figure S6 -- Koopman mode analysis (latent variance | complex-mode frequencies |
mode-ablation impact).  Standalone, reproducible rebuild of the former notebook
cell, restyled to match the other revision-2 supplementary figures (no subplot
titles, panel letters, enlarged fonts, no overlapping annotations, headroom for
in-plot labels/legends).

Reads:  results/mode_frequencies.json, results/mode_ablation.json
Writes: results/figures/figureSX_mode_analysis.{svg,png}
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

C_COMPLEX, C_DOM, C_REAL = "#dda251", "#ac4484", "#646464"   # complex / dominant / real
AXLAB, TICK = 14, 12


def main():
    with open(os.path.join(C.RESULT_DIR, "mode_frequencies.json")) as f:
        mf = json.load(f)
    with open(os.path.join(C.RESULT_DIR, "mode_ablation.json")) as f:
        ma = json.load(f)
    modes = mf["modes"]

    # ── ordered labels / variances / colours for all latent modes ──────────────
    labels, variances, colors = [], [], []
    for m in modes:
        if m["type"] == "complex":
            labels.append(f"CP-{m['index']} ({abs(m['f_hz']):.2f} Hz)")
            colors.append(C_DOM if m["index"] == 3 else C_COMPLEX)
        else:
            labels.append(f"R-{m['index']}")
            colors.append(C_REAL)
        variances.append(m["latent_variance"])
    variances = np.array(variances)
    x = np.arange(len(labels))
    dom_i = labels.index("CP-3 (1.77 Hz)")
    dom_pct = variances[dom_i] * 100.0   # latent_variance is already a fraction of the total

    cp = [m for m in modes if m["type"] == "complex"]
    cpx = np.arange(len(cp))

    abl = sorted(ma["ablation_results"], key=lambda r: r["mode_index"])
    abl_lbls = [f"CP-{r['mode_index']} ({abs(r['f_hz']):.2f} Hz)" for r in abl]
    abl_delta = np.array([r["pct_rmse_increase"] for r in abl])
    abl_cols = [C_DOM if r["mode_index"] == 3 else C_COMPLEX for r in abl]
    baseline = ma["baseline_pct_rmse"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.subplots_adjust(wspace=0.26)

    # ── (a) latent variance per mode ───────────────────────────────────────────
    ax1.bar(x, variances, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylim(top=variances.max() * 6)                    # headroom for label + legend
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=TICK - 1)
    ax1.set_ylabel("Latent variance (log scale)", fontsize=AXLAB)
    ax1.tick_params(axis="y", labelsize=TICK)
    ax1.yaxis.grid(True, which="both", linestyle="--", alpha=0.4); ax1.set_axisbelow(True)
    # clean value label above the dominant bar (no diagonal arrow)
    ax1.text(dom_i, variances[dom_i] * 1.4, f"{dom_pct:.1f}%",
             ha="center", va="bottom", fontsize=TICK, color=C_DOM, fontweight="bold")
    ax1.legend(handles=[Patch(facecolor=C_COMPLEX, label="Complex pair"),
                        Patch(facecolor=C_DOM, label="Dominant complex pair"),
                        Patch(facecolor=C_REAL, label="Real mode")],
               fontsize=TICK - 1, frameon=True, framealpha=0.9, loc="upper right")
    ax1.set_title("(a)", loc="left", fontsize=13)

    # ── (b) |frequency| of the complex modes ───────────────────────────────────
    fabs = np.array([abs(m["f_hz"]) for m in cp])
    ax2.bar(cpx, fabs, color=[C_DOM if m["index"] == 3 else C_COMPLEX for m in cp],
            edgecolor="white", linewidth=0.8, zorder=3)
    ax2.set_yscale("log")
    ax2.axhspan(0.833, 2.5, alpha=0.14, color="green", zorder=0)
    ax2.set_ylim(top=fabs.max() * 4)                         # headroom for the T= labels
    ax2.set_xticks(cpx)
    ax2.set_xticklabels([f"CP-{m['index']}" for m in cp], fontsize=TICK)
    ax2.set_ylabel("|Frequency| Hz (log scale)", fontsize=AXLAB)
    ax2.tick_params(axis="y", labelsize=TICK)
    ax2.yaxis.grid(True, which="both", linestyle="--", alpha=0.4); ax2.set_axisbelow(True)
    # heart-rate band label placed inside the band over the tiny CP-1/CP-2 bars
    ax2.text(1.5, 1.44, "Heart rate range\n(50–150 bpm)", ha="center", va="center",
             fontsize=TICK - 1, color="#2e7d32", style="italic")
    for xi, m in zip(cpx, cp):
        fa = abs(m["f_hz"])
        lbl = f"T={m['period_s']:.2f} s" if fa > 0.1 else f"T={m['period_s']:.0f} s"
        ax2.text(xi, fa * 1.5, lbl, ha="center", va="bottom", fontsize=TICK - 1)
    ax2.set_title("(b)", loc="left", fontsize=13)

    # ── (c) mode-ablation impact ───────────────────────────────────────────────
    ax3.bar(np.arange(len(abl)), abl_delta, color=abl_cols,
            edgecolor="white", linewidth=0.8, zorder=3)
    ax3.axhline(baseline, color="grey", linestyle="--", linewidth=1.6,
                label=f"Baseline %RMSE ({baseline:.1f}%)")
    ax3.set_ylim(0, baseline * 1.28)                          # headroom over baseline for labels
    ax3.set_xticks(np.arange(len(abl)))
    ax3.set_xticklabels(abl_lbls, rotation=20, ha="right", fontsize=TICK - 1)
    ax3.set_ylabel("Δ %RMSE (percentage points)", fontsize=AXLAB)
    ax3.tick_params(axis="y", labelsize=TICK)
    ax3.yaxis.grid(True, linestyle="--", alpha=0.4); ax3.set_axisbelow(True)
    ax3.legend(fontsize=TICK - 1, frameon=True, framealpha=0.9, loc="upper left")
    for xi, r in zip(np.arange(len(abl)), abl):
        rel = r["pct_rmse_increase"] / baseline * 100
        ax3.text(xi, r["pct_rmse_increase"] + baseline * 0.02, f"+{rel:.0f}%",
                 ha="center", va="bottom", fontsize=TICK)
    ax3.set_title("(c)", loc="left", fontsize=13)

    fig.tight_layout()
    out = os.path.join(C.FIG_DIR, "figureSX_mode_analysis")
    for ext in ("svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"dominant CP-3: var={variances[dom_i]:.3f} -> {dom_pct:.1f}% of latent variance")
    print("Saved -> figures/figureSX_mode_analysis.svg + .png")


if __name__ == "__main__":
    main()
