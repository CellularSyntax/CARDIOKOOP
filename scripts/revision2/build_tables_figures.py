#!/usr/bin/env python3
"""
Regenerate Table 3 / Table 4 / Table 5 and the Figure-5 comparison panels with
the two new direct baselines (DLinear, NLinear) added, on the seed-42 TEST split
with ONE consistent set of metric functions for every model.

Why a consistent recompute?  The revision-1 notebook is internally inconsistent:
the Koopman row is evaluated on val1 while every baseline uses test1, and the R^2
column mixes two definitions (sklearn multi-output for the pickled models vs
pooled/flattened for MLP/AR).  To make DLinear/NLinear directly comparable, this
script recomputes EVERY model on test1 with test1 ground truth using the shared
metric functions (%RMSE = pct_per_traj_ps; R^2 = pooled global_r2_flat).  The
Koopman headline barely moves (val 17.30 -> test 17.54).  See REVISION2_RESULTS.md.

Nothing is retrained here except the analytic AR(20) OLS fit (as in the notebook).
The frozen Koopman checkpoint and the trained baseline checkpoints are only run
in inference.

Outputs (results/revision2/):
  table3_overall_comparison.csv / .tex
  table4_per_signal.csv / .tex
  table5_noise_robustness_full.csv / .tex
  figure5_rev2_comparison.svg / .png     (per-signal %RMSE, accuracy-vs-time,
                                          cumulative %RMSE, parameter count)
  figure6_rev2_noise.svg / .png          (AWGN robustness, all 8 architectures)
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C
from task_a_dlinear_nlinear import DLinear, NLinear, L_LOOKBACK, full_prediction, make_tensors

DEVICE = C.get_device()
SIM_TIME_S = 2.05   # reference full-order ODE simulation time per trajectory (manuscript)
TRAIN_TIMES_MIN = {  # wall-clock training times (min); RNN/Koopman/MLP/AR from manuscript
    "Koopman": "47", "GRU": "289", "LSTM": "312", "BiLSTM": "401",
    "MLP": "9", "AR(20)": "<1", "DLinear": "~2", "NLinear": "~2",
}
SNR_DB = [30, 20, 10, 5]


# ─────────────────── assemble test1 predictions for every model ───────────────
def koopman_pred(Xn, Un, sig_mean, sig_std):
    model, params, n_params, _ = C.load_koopman(DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    pred = C.koopman_rollout(model, Xn[:, 0, :], Un, device=DEVICE) * sig_std + sig_mean
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / Xn.shape[0]
    return pred, n_params, np.full(Xn.shape[0], dt)


def mlp_pred(Xn, sig_mean, sig_std):
    ck = torch.load(os.path.join(C.CKPT_DIR, "mlp_baseline.pt"), map_location=DEVICE, weights_only=False)
    class _MLP(nn.Module):
        def __init__(s, dim=12):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 512),
                                  nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, dim))
        def forward(s, x): return s.net(x)
    m = _MLP(Xn.shape[2]).to(DEVICE); m.load_state_dict(ck["state_dict"]); m.eval()
    CLIP = 20.0
    B, T, D = Xn.shape
    out = np.empty_like(Xn)
    t0 = time.time()
    for i in range(B):
        o = np.empty((T, D)); o[0] = Xn[i, 0]
        prev = torch.tensor(Xn[i, 0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            for t in range(1, T):
                prev = m(prev).clamp(-CLIP, CLIP)
                o[t] = prev.cpu().numpy()[0]
        out[i] = o
    dt = (time.time() - t0) / B
    return out * sig_std + sig_mean, int(ck["n_params"]), np.full(B, dt)


def ar_pred(Xn, sig_mean, sig_std, p=20):
    xtr = np.loadtxt(os.path.join(C.DATA_DIR, f"{C.DATA_NAME}_train1_x.csv"), delimiter=",")
    T = C.T_STEPS; ntr = xtr.shape[0] // T; D = xtr.shape[1]
    xtr = xtr.reshape(ntr, T, D)
    coeffs = np.zeros((D, p))
    for d in range(D):
        rX, ry = [], []
        for i in range(ntr):
            ts = xtr[i, :, d]
            for t in range(p, T):
                rX.append(ts[t-p:t][::-1]); ry.append(ts[t])
        coeffs[d], *_ = np.linalg.lstsq(np.array(rX), np.array(ry), rcond=None)
    B = Xn.shape[0]
    pred = Xn.copy()
    t0 = time.time()
    for i in range(B):
        for t in range(p, T):
            pred[i, t, :] = (coeffs * pred[i, t-p:t, :][::-1, :].T).sum(axis=1)
    dt = (time.time() - t0) / B
    return pred * sig_std + sig_mean, p * D, np.full(B, dt)


def load_linear(name):
    ck = torch.load(os.path.join(C.CKPT_DIR, f"{name.lower()}_baseline.pt"),
                    map_location=DEVICE, weights_only=False)
    cls = DLinear if name == "DLinear" else NLinear
    m = cls(12, ck["lookback"], ck["horizon"]).to(DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


# ───────────────────── noise (AWGN) robustness for the linear baselines ─────────
def linear_awgn_curve(name, Xn, Xte_phys, s_std, sig_mean, sig_std, sig_power, rng):
    """AWGN added to the L-step look-back window (== RNN convention), direct forecast."""
    m, ck = load_linear(name)
    B, T, D = Xn.shape
    def metrics(pred):
        pct = np.array([C.pct_rmse_ps(Xte_phys[i], pred[i]) for i in range(B)])
        r2  = np.array([C.r2_flat(Xte_phys[i], pred[i]) for i in range(B)])
        return {"pct_rmse_mean": float(pct.mean()), "pct_rmse_ci95": float(C.ci95(pct)),
                "r2_mean": float(r2.mean()), "r2_sd": float(r2.std(ddof=1))}
    rec = {"clean_baseline": metrics(full_prediction(m, Xn, s_std, sig_mean, sig_std)),
           "noisy": {}}
    for snr in SNR_DB:
        snr_lin = 10 ** (snr / 10.0)
        nstd = np.sqrt(sig_power / snr_lin)
        Xnn = Xn.copy()
        Xnn[:, :L_LOOKBACK, :] += rng.normal(0, nstd[None, None, :], size=(B, L_LOOKBACK, D))
        rec["noisy"][str(snr)] = metrics(full_prediction(m, Xnn, s_std, sig_mean, sig_std))
    return rec


def main():
    C.set_all_seeds(C.SEED)
    Xn, Un, Uph, Xte = C.load_split("test")
    sig_mean, sig_std = C.load_norm_stats()
    B, T, D = Xte.shape

    print("Assembling test1 predictions for all models...")
    preds, nparams, inftimes = {}, {}, {}
    preds["Koopman"], nparams["Koopman"], inftimes["Koopman"] = koopman_pred(Xn, Un, sig_mean, sig_std)
    for m in ["GRU", "LSTM", "BiLSTM", "DLinear", "NLinear"]:
        r = C.load_pickle(os.path.join(C.RESULT_DIR, m.lower(), f"{m.lower()}_postprocessing_results.pkl"))
        preds[m] = np.asarray(r["pred_per_trajectory"])
        nparams[m] = int(r["n_params"]); inftimes[m] = np.asarray(r["inference_times_s"])
    preds["MLP"], nparams["MLP"], inftimes["MLP"] = mlp_pred(Xn, sig_mean, sig_std)
    preds["AR(20)"], nparams["AR(20)"], inftimes["AR(20)"] = ar_pred(Xn, sig_mean, sig_std)

    models = C.MODELS_ALL
    results = {m: C.compute_results_dict(Xte, preds[m], Uph, inftimes[m], nparams[m]) for m in models}

    # ── Table 3: overall comparison ───────────────────────────────────────────
    rows3 = []
    for m in models:
        r = results[m]
        pct = r["pct_per_traj_ps"]; inf = r["inference_times_s"].mean()
        rows3.append({
            "Model": m,
            "R2": round(r["global_r2_flat"], 3),
            "%RMSE (mean +/- CI)": f"{pct.mean():.1f} +/- {C.ci95(pct):.1f}",
            "Inference Time (s)": f"{inf:.2e}",
            "Speedup vs sim": f"{SIM_TIME_S / inf:.3g}x",
            "Parameter Count": f"{nparams[m]:,}",
            "Training Time (min)": TRAIN_TIMES_MIN.get(m, "-"),
        })
    df3 = pd.DataFrame(rows3).set_index("Model")
    df3.to_csv(os.path.join(C.REV2_DIR, "table3_overall_comparison.csv"))
    with open(os.path.join(C.REV2_DIR, "table3_overall_comparison.tex"), "w") as f:
        f.write(df3.to_latex(escape=False, column_format="lrrrrrr",
                caption=("Overall forecasting comparison on the seed-42 test split "
                         "($B=50$, 1499-step horizon). %RMSE = mean per-signal "
                         "PTP-normalised RMSE averaged over signals; $R^2$ pooled. "
                         "DLinear/NLinear (Zeng et al. 2023) added."),
                label="tab:overall"))
    print("\n=== Table 3 (overall, test1, consistent metrics) ===")
    print(df3.to_string())

    # ── Table 4: per-signal %RMSE and R^2 ─────────────────────────────────────
    rows4 = []
    for j in range(D):
        row = {"Signal": C.SIG_NAMES[j]}
        for m in models:
            row[f"{m} %RMSE"] = round(float(results[m]["pct_mean"][j]), 2)
        rows4.append(row)
    df4 = pd.DataFrame(rows4).set_index("Signal")
    df4.to_csv(os.path.join(C.REV2_DIR, "table4_per_signal.csv"))
    with open(os.path.join(C.REV2_DIR, "table4_per_signal.tex"), "w") as f:
        f.write(df4.to_latex(escape=False, column_format="l" + "r" * len(models),
                caption=("Per-signal %RMSE on the seed-42 test split for all "
                         "architectures (incl. DLinear/NLinear)."),
                label="tab:per_signal"))
    print("\n=== Table 4 (per-signal %RMSE, test1) ===")
    print(df4.to_string())

    # ── Figure 5 (rev2): 4-panel comparison ───────────────────────────────────
    pal = dict(zip(models, C.PALETTE_HEX))
    fig = plt.figure(figsize=(22, 4.2))
    gs = GridSpec(1, 4, width_ratios=[6, 3, 4, 3], figure=fig)

    SIG_MAIN = [2, 7, 8, 10, 11]
    ax0 = fig.add_subplot(gs[0])
    w = 0.8 / len(models)
    xm = np.arange(len(SIG_MAIN))
    for i, m in enumerate(models):
        means = results[m]["pct_mean"][SIG_MAIN]
        ax0.bar(xm + i * w, means, w, label=m, color=pal[m])
    ax0.set_yscale("log"); ax0.set_ylim(bottom=0.3)
    ax0.set_xticks(xm + w * (len(models) - 1) / 2)
    ax0.set_xticklabels([C.YLAB[i] for i in SIG_MAIN], fontsize=12)
    ax0.set_ylabel("%RMSE (log)", fontsize=13); ax0.set_title("(b) Per-signal %RMSE", fontsize=12)
    ax0.legend(fontsize=8.5, ncol=4, loc="upper left", frameon=False)
    ax0.grid(axis="y", which="both", ls="--", alpha=0.3); ax0.set_axisbelow(True)

    ax1 = fig.add_subplot(gs[1])
    for m in models:
        inf = results[m]["inference_times_s"].mean()
        ax1.scatter(inf, results[m]["pct_per_traj_ps"].mean(),
                    s=110 if m == "Koopman" else 60, color=pal[m],
                    marker="*" if m == "Koopman" else "o", label=m)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Inference time (s, log)", fontsize=12)
    ax1.set_ylabel("Avg %RMSE (log)", fontsize=12)
    ax1.set_title("(c) Accuracy vs. speed", fontsize=12)
    ax1.grid(True, which="both", ls="--", alpha=0.3); ax1.set_axisbelow(True)

    ax2 = fig.add_subplot(gs[2])
    for m in models:
        cum = results[m]["cumulative_pct_per_traj"]
        ax2.plot(np.nanmean(cum, axis=0), color=pal[m], lw=2, label=m)
    ax2.set_yscale("log")
    ax2.set_xlabel("Time step", fontsize=12); ax2.set_ylabel("Cumulative %RMSE (log)", fontsize=12)
    ax2.set_title("(d) Cumulative %RMSE", fontsize=12)
    ax2.legend(fontsize=8.5, ncol=2); ax2.grid(True, which="both", ls="--", alpha=0.3)

    ax3 = fig.add_subplot(gs[3])
    npv = [nparams[m] for m in models]
    ax3.bar(range(len(models)), npv, color=[pal[m] for m in models])
    ax3.set_yscale("log"); ax3.set_ylabel("Parameters (log)", fontsize=12)
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=40, ha="right", fontsize=9)
    ax3.set_title("(e) Model size", fontsize=12)
    ax3.grid(axis="y", which="both", ls="--", alpha=0.3); ax3.set_axisbelow(True)

    fig.suptitle("Figure 5 (rev.2): forecasting comparison incl. DLinear/NLinear "
                 "(seed-42 test split)", fontsize=13, y=1.03)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(C.FIG_DIR, f"figure5_rev2_comparison.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved -> figures/figure5_rev2_comparison.svg + .png")

    # ── Table 5 + Figure 6: AWGN noise robustness, all 8 architectures ─────────
    with open(os.path.join(C.RESULT_DIR, "noise_robustness.json")) as f:
        nr_koop = json.load(f)
    with open(os.path.join(C.RESULT_DIR, "noise_robustness_baselines.json")) as f:
        nr_bl = json.load(f)
    sig_power = Xn.var(axis=(0, 1))
    rng = np.random.default_rng(C.SEED)
    _, _, Ute_phys, _ = C.load_split("test")
    xtr = np.loadtxt(os.path.join(C.DATA_DIR, f"{C.DATA_NAME}_train1_u.csv"), delimiter=",")
    s_mean = xtr.reshape(-1, T)[:, -1].mean(); s_sd = xtr.reshape(-1, T)[:, -1].std() + 1e-8
    s_te = (Ute_phys[:, -1, 0] - s_mean) / s_sd
    nr_lin = {m: linear_awgn_curve(m, Xn, Xte, s_te, sig_mean, sig_std, sig_power, rng)
              for m in ["DLinear", "NLinear"]}

    def curve(m):
        if m == "Koopman":
            d = nr_koop
            pm = [d["clean_baseline"]["pct_rmse_mean"]] + [d["noisy"][str(s)]["pct_rmse_mean"] for s in SNR_DB]
            rm = [d["clean_baseline"]["r2_mean"]] + [d["noisy"][str(s)]["r2_mean"] for s in SNR_DB]
        elif m in nr_lin:
            d = nr_lin[m]
            pm = [d["clean_baseline"]["pct_rmse_mean"]] + [d["noisy"][str(s)]["pct_rmse_mean"] for s in SNR_DB]
            rm = [d["clean_baseline"]["r2_mean"]] + [d["noisy"][str(s)]["r2_mean"] for s in SNR_DB]
        else:
            d = nr_bl[m]
            pm = [d["clean_baseline"]["pct_rmse_mean"]] + [d["noisy"][str(s)]["pct_rmse_mean"] for s in SNR_DB]
            rm = [d["clean_baseline"]["r2_mean"]] + [d["noisy"][str(s)]["r2_mean"] for s in SNR_DB]
        return np.array(pm), np.array(rm)

    labels = ["Clean"] + [f"{s} dB" for s in SNR_DB]
    rows5 = []
    for li, lab in enumerate(labels):
        row = {"SNR": lab}
        for m in models:
            pm, _ = curve(m)
            row[m] = round(float(pm[li]), 1)
        rows5.append(row)
    df5 = pd.DataFrame(rows5).set_index("SNR")
    df5.to_csv(os.path.join(C.REV2_DIR, "table5_noise_robustness_full.csv"))
    with open(os.path.join(C.REV2_DIR, "table5_noise_robustness_full.tex"), "w") as f:
        f.write(df5.to_latex(escape=False, column_format="l" + "r" * len(models),
                caption=("%RMSE under AWGN applied to the initial observation/warm-up "
                         "window, all architectures (incl. DLinear/NLinear), seed-42 "
                         "test split, no retraining."),
                label="tab:noise_full"))
    print("\n=== Table 5 (AWGN %RMSE, test1) ===")
    print(df5.to_string())

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for m in models:
        pm, rm = curve(m)
        lw = 2.8 if m == "Koopman" else 1.7
        axes[0].plot(x, pm, "o-", color=pal[m], lw=lw, label=m)
        axes[1].plot(x, np.clip(rm, -3, 1.05), "o-", color=pal[m], lw=lw, label=m)
    axes[0].set_yscale("log"); axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("%RMSE (log)"); axes[0].set_xlabel("Input noise (SNR)")
    axes[0].set_title("(a) %RMSE vs. SNR"); axes[0].legend(fontsize=9, ncol=2)
    axes[0].grid(axis="y", which="both", ls="--", alpha=0.3)
    axes[1].axhline(0, color="#888", ls=":", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel(r"$R^2$ (clipped)"); axes[1].set_xlabel("Input noise (SNR)")
    axes[1].set_title(r"(b) $R^2$ vs. SNR"); axes[1].legend(fontsize=9, ncol=2)
    axes[1].grid(axis="y", ls="--", alpha=0.3)
    fig.suptitle("Figure 6 (rev.2): AWGN noise robustness, all architectures", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(C.FIG_DIR, f"figure6_rev2_noise.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved -> figures/figure6_rev2_noise.svg + .png")

    # persist the noise-with-linear json for reproducibility
    with open(os.path.join(C.REV2_DIR, "noise_robustness_linear.json"), "w") as f:
        json.dump(nr_lin, f, indent=2)


if __name__ == "__main__":
    main()
