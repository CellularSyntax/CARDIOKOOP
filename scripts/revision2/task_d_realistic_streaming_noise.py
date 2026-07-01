#!/usr/bin/env python3
"""
TASK D (editor mandatory #3 / R1.3, R2.3) -- realistic + streaming noise
========================================================================
Extends the existing AWGN noise harness (task3_noise_robustness.py) with two
more realistic, reviewer-requested variants, evaluated on the frozen Koopman
model (inference only) over the same SNR sweep {30, 20, 10, 5} dB and the same
seed-42 test split / metric functions.

(a) SENSOR-REALISTIC initial-condition corruption:
    the initial observation is corrupted by additive white Gaussian noise PLUS a
    low-frequency baseline-wander component (a sub-cardiac ~0.05-0.3 Hz drift,
    the classic physiological sensor artifact).  At each SNR the total noise
    power matches the AWGN harness; baseline wander carries WANDER_FRAC of it and
    white noise the rest.  This is a strictly harder, colored-noise version of
    the t=0 corruption.

(b) RE-ANCHORING / STREAMING rollout:
    instead of injecting noise only at t=0, the forecast is periodically
    re-initialised from a FRESH noisy observation of the true state every K steps
    (K=100 ~ 2 cardiac beats at ~1.8 Hz, dt=0.01 s), so the (awgn+wander) noise
    process is sampled throughout the 1499-step horizon.  This mimics an online
    deployment where the model is re-anchored to incoming (noisy) measurements.

Both are compared against the clean baseline.  Nothing is retrained.

Outputs
  results/revision2/task_d_realistic_streaming_noise.json
  results/revision2/task_d_realistic_streaming_noise.csv
  results/figures/figureS_realistic_streaming_noise.{svg,png}
"""
import os
import sys
import csv
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

SNR_DB      = [30, 20, 10, 5]
WANDER_FRAC = 0.5          # share of noise power in the low-frequency baseline wander
WANDER_HZ   = (0.05, 0.30) # sub-cardiac baseline-wander band
K_REANCHOR  = 100          # streaming re-anchor interval (~2 beats)
DT          = 0.01
SEED        = 42


def noise_process(snr_db, sig_power, shape, rng):
    """(B,T,D) additive noise = white AWGN + low-frequency baseline wander."""
    B, T, D = shape
    snr_lin   = 10 ** (snr_db / 10.0)
    total_pow = sig_power / snr_lin                      # (D,) per-channel noise power
    awgn_std  = np.sqrt((1 - WANDER_FRAC) * total_pow)  # (D,)
    wander_pow = WANDER_FRAC * total_pow                 # (D,)

    awgn = rng.normal(0.0, 1.0, size=(B, T, D)) * awgn_std[None, None, :]

    t = np.arange(T) * DT
    f   = rng.uniform(WANDER_HZ[0], WANDER_HZ[1], size=(B, D))
    phi = rng.uniform(0, 2 * np.pi, size=(B, D))
    amp = np.sqrt(2.0 * wander_pow)[None, :]             # sin power = A^2/2 -> A=sqrt(2P)
    wander = amp[:, None, :] * np.sin(
        2 * np.pi * f[:, None, :] * t[None, :, None] + phi[:, None, :])
    return awgn + wander


@torch.no_grad()
def streaming_rollout(model, X_true_norm, u_norm, noise, K, device):
    """
    Koopman rollout that re-anchors from a fresh noisy observation of the true
    state every K steps.  X_true_norm:(B,T,D) noise:(B,T,D) -> pred (B,T,D) norm.
    """
    B, T, D = X_true_norm.shape
    x0 = torch.as_tensor(X_true_norm[:, 0, :] + noise[:, 0, :], dtype=torch.float32, device=device)
    u  = torch.as_tensor(u_norm, dtype=torch.float32, device=device).permute(1, 0, 2)
    y = model.encoder(x0)
    outs = []
    for t in range(T):
        outs.append(model.decoder(y).cpu().numpy())
        y = C.koopman_advance(model, y, u[t])
        nxt = t + 1
        if nxt < T and nxt % K == 0:
            x_obs = torch.as_tensor(X_true_norm[:, nxt, :] + noise[:, nxt, :],
                                    dtype=torch.float32, device=device)
            y = model.encoder(x_obs)
    return np.stack(outs, axis=1)


def metrics(Xph, pred, B, D):
    pct = np.array([C.pct_rmse_ps(Xph[i], pred[i]) for i in range(B)])
    r2  = np.array([C.r2_flat(Xph[i], pred[i]) for i in range(B)])
    return {
        "pct_rmse_mean": float(pct.mean()),
        "pct_rmse_ci95": float(C.ci95(pct)),
        "pct_rmse_sd":   float(pct.std(ddof=1)),
        "r2_mean":       float(r2.mean()),
        "r2_sd":         float(r2.std(ddof=1)),
    }


def main():
    C.set_all_seeds(SEED)
    device = C.get_device()
    model, params, n_params, ckpt = C.load_koopman(device)
    Xn, Un, Uph, Xph = C.load_split("test")
    sig_mean, sig_std = C.load_norm_stats()
    B, T, D = Xph.shape
    sig_power = Xn.var(axis=(0, 1))
    rng = np.random.default_rng(SEED)

    # clean baseline
    pred_clean = C.koopman_rollout(model, Xn[:, 0, :], Un, device=device) * sig_std + sig_mean
    clean = metrics(Xph, pred_clean, B, D)
    print(f"Clean: %RMSE={clean['pct_rmse_mean']:.2f}  R2={clean['r2_mean']:.3f}")

    realistic_ic, streaming = {}, {}
    for snr in SNR_DB:
        noise = noise_process(snr, sig_power, (B, T, D), rng)

        # (a) realistic corruption of the initial condition only
        x0_noisy = Xn[:, 0, :] + noise[:, 0, :]
        pred_a = C.koopman_rollout(model, x0_noisy, Un, device=device) * sig_std + sig_mean
        realistic_ic[str(snr)] = metrics(Xph, pred_a, B, D)

        # (b) streaming re-anchoring every K steps
        pred_b = streaming_rollout(model, Xn, Un, noise, K_REANCHOR, device) * sig_std + sig_mean
        streaming[str(snr)] = metrics(Xph, pred_b, B, D)

        print(f"SNR={snr:2d}dB | realistic-IC %RMSE={realistic_ic[str(snr)]['pct_rmse_mean']:6.2f} "
              f"R2={realistic_ic[str(snr)]['r2_mean']:.3f} | "
              f"streaming(K={K_REANCHOR}) %RMSE={streaming[str(snr)]['pct_rmse_mean']:6.2f} "
              f"R2={streaming[str(snr)]['r2_mean']:.3f}")

    out = {
        "description": ("Realistic (AWGN + baseline wander) IC corruption and "
                        "re-anchoring/streaming rollout on the frozen Koopman model; "
                        "no retraining."),
        "snr_levels_db": SNR_DB,
        "wander_fraction": WANDER_FRAC, "wander_band_hz": list(WANDER_HZ),
        "reanchor_K_steps": K_REANCHOR, "reanchor_interval_s": K_REANCHOR * DT,
        "seed": SEED,
        "clean_baseline": clean,
        "realistic_ic": realistic_ic,
        "streaming_reanchor": streaming,
    }
    with open(os.path.join(C.REV2_DIR, "task_d_realistic_streaming_noise.json"), "w") as f:
        json.dump(out, f, indent=2)

    # tabular CSV
    out_csv = os.path.join(C.REV2_DIR, "task_d_realistic_streaming_noise.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "snr_db", "pct_rmse_mean", "pct_rmse_ci95", "r2_mean"])
        w.writerow(["clean", "inf", clean["pct_rmse_mean"], clean["pct_rmse_ci95"], clean["r2_mean"]])
        for snr in SNR_DB:
            r = realistic_ic[str(snr)]
            w.writerow(["realistic_ic", snr, r["pct_rmse_mean"], r["pct_rmse_ci95"], r["r2_mean"]])
        for snr in SNR_DB:
            r = streaming[str(snr)]
            w.writerow([f"streaming_K{K_REANCHOR}", snr, r["pct_rmse_mean"], r["pct_rmse_ci95"], r["r2_mean"]])
    print(f"Saved -> {out_csv} (+ .json)")

    # ── figure ──────────────────────────────────────────────────────────────
    labels = ["Clean"] + [f"{s} dB" for s in SNR_DB]
    x = np.arange(len(labels))
    def curve(d, key):
        return np.array([clean[key]] + [d[str(s)][key] for s in SNR_DB])
    def ci(d):
        return np.array([clean["pct_rmse_ci95"]] + [d[str(s)]["pct_rmse_ci95"] for s in SNR_DB])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    pa, ps = "#5a286b", "#dda251"

    ax = axes[0]
    ax.errorbar(x, curve(realistic_ic, "pct_rmse_mean"), yerr=ci(realistic_ic),
                fmt="o-", color=pa, lw=2, capsize=4, label="Realistic IC (AWGN+wander)")
    ax.errorbar(x, curve(streaming, "pct_rmse_mean"), yerr=ci(streaming),
                fmt="s--", color=ps, lw=2, capsize=4, label=f"Streaming re-anchor (K={K_REANCHOR})")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("%RMSE"); ax.set_xlabel("Noise level (SNR)")
    ax.set_title("(a) %RMSE vs. noise"); ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.plot(x, curve(realistic_ic, "r2_mean"), "o-", color=pa, lw=2, label="Realistic IC")
    ax.plot(x, curve(streaming, "r2_mean"), "s--", color=ps, lw=2, label=f"Streaming (K={K_REANCHOR})")
    ax.axhline(0, color="#888", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$R^2$"); ax.set_xlabel("Noise level (SNR)")
    ax.set_title(r"(b) $R^2$ vs. noise"); ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Koopman robustness: realistic IC corruption vs. streaming re-anchoring "
                 "(no retraining)", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(C.FIG_DIR, f"figureS_realistic_streaming_noise.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved -> figures/figureS_realistic_streaming_noise.svg + .png")


if __name__ == "__main__":
    main()
