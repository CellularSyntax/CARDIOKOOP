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

    # clean baseline -- open-loop free-running (the clean reference for the single-shot
    # 'realistic IC' scenario)
    pred_clean = C.koopman_rollout(model, Xn[:, 0, :], Un, device=device) * sig_std + sig_mean
    clean = metrics(Xph, pred_clean, B, D)
    print(f"Clean (open-loop): %RMSE={clean['pct_rmse_mean']:.2f}  R2={clean['r2_mean']:.3f}")

    # clean RE-ANCHORED baseline (streaming with zero noise) -- the matching clean
    # reference for the streaming curve, so each scenario is compared against its own
    # no-noise operating point (avoids the spurious clean->30 dB 'dip')
    pred_stream_clean = (streaming_rollout(model, Xn, Un, np.zeros((B, T, D)), K_REANCHOR, device)
                         * sig_std + sig_mean)
    clean_stream = metrics(Xph, pred_stream_clean, B, D)
    print(f"Clean (re-anchored K={K_REANCHOR}): %RMSE={clean_stream['pct_rmse_mean']:.2f}  "
          f"R2={clean_stream['r2_mean']:.3f}")

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
        "clean_reanchored_baseline": clean_stream,
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
        w.writerow(["clean_reanchored", "inf", clean_stream["pct_rmse_mean"],
                    clean_stream["pct_rmse_ci95"], clean_stream["r2_mean"]])
        for snr in SNR_DB:
            r = realistic_ic[str(snr)]
            w.writerow(["realistic_ic", snr, r["pct_rmse_mean"], r["pct_rmse_ci95"], r["r2_mean"]])
        for snr in SNR_DB:
            r = streaming[str(snr)]
            w.writerow([f"streaming_K{K_REANCHOR}", snr, r["pct_rmse_mean"], r["pct_rmse_ci95"], r["r2_mean"]])
    print(f"Saved -> {out_csv} (+ .json)")

    # ── figure: noise illustration (a,b) + robustness curves (c,d) ───────────
    labels = ["Clean"] + [f"{s} dB" for s in SNR_DB]
    x = np.arange(len(labels))

    def curve(d, key, cpt):
        return np.array([cpt[key]] + [d[str(s)][key] for s in SNR_DB])

    def ci_pct(d, cpt):
        return np.array([cpt["pct_rmse_ci95"]] + [d[str(s)]["pct_rmse_ci95"] for s in SNR_DB])

    def ci_r2(d, cpt):
        sd = np.array([cpt["r2_sd"]] + [d[str(s)]["r2_sd"] for s in SNR_DB])
        return 1.96 * sd / np.sqrt(B)

    C_REAL, C_STREAM = "#5a286b", "#dda251"      # realistic-IC / streaming
    C_AWGN, C_WANDER = "#9a9a9a", "#2e7ec0"      # noise components
    AXLAB, TICK = 13, 11

    # representative noise sample (one channel, one SNR) for the illustration
    ch_ill, snr_ill, nwin = 7, 10, 300           # P_lv, 10 dB, first 3 s
    tt = np.arange(T) * DT
    tp = sig_power[ch_ill] / (10 ** (snr_ill / 10.0))
    awgn_std = np.sqrt((1 - WANDER_FRAC) * tp)
    wpow = WANDER_FRAC * tp
    rng_i = np.random.default_rng(SEED + 7)
    awgn_c = rng_i.normal(0, 1, T) * awgn_std
    f_i, phi_i = rng_i.uniform(*WANDER_HZ), rng_i.uniform(0, 2 * np.pi)
    wander_c = np.sqrt(2 * wpow) * np.sin(2 * np.pi * f_i * tt + phi_i)
    total_c = awgn_c + wander_c
    sl = slice(0, nwin)
    plv_clean = Xph[0, :, ch_ill]
    plv_noisy = plv_clean + total_c * sig_std[ch_ill]
    reanchor_t = np.arange(K_REANCHOR, nwin, K_REANCHOR) * DT

    fig, (axa, axb, axc, axd) = plt.subplots(
        1, 4, figsize=(20, 4.3), gridspec_kw={"width_ratios": [1.4, 1.4, 1.0, 1.0]})
    fig.subplots_adjust(wspace=0.32)

    # (a) noise-model composition
    axa.plot(tt[sl], awgn_c[sl], color=C_AWGN, lw=0.8, label="AWGN")
    axa.plot(tt[sl], wander_c[sl], color=C_WANDER, lw=2.2, label="Baseline wander")
    axa.plot(tt[sl], total_c[sl], color=C_REAL, lw=1.1, alpha=0.9, label="Total")
    axa.axhline(0, color="#bbb", lw=0.7)
    _amax = np.max(np.abs(np.concatenate([awgn_c[sl], wander_c[sl], total_c[sl]])))
    axa.set_ylim(-_amax * 1.15, _amax * 1.9)     # top headroom so the legend clears the traces
    axa.set_xlabel("Time (s)", fontsize=AXLAB); axa.set_ylabel("Noise (norm. units)", fontsize=AXLAB)
    axa.tick_params(labelsize=TICK); axa.grid(alpha=0.25); axa.set_axisbelow(True)
    axa.legend(fontsize=TICK - 1, loc="upper right", framealpha=0.9)
    axa.set_title("(a)", loc="left", fontsize=13)

    # (b) effect on a representative signal + re-anchoring
    axb.plot(tt[sl], plv_clean[sl], color="k", lw=1.7, label="Clean")
    axb.plot(tt[sl], plv_noisy[sl], color=C_REAL, lw=1.0, alpha=0.85, label="Noisy observation")
    for rt in reanchor_t:
        axb.axvline(rt, color=C_STREAM, ls="--", lw=1.1, alpha=0.75)
    axb.plot([], [], color=C_STREAM, ls="--", lw=1.1, label=f"Re-anchor (every {K_REANCHOR * DT:.0f} s)")
    _pall = np.concatenate([plv_clean[sl], plv_noisy[sl]])
    _prng = _pall.max() - _pall.min()
    axb.set_ylim(_pall.min() - 0.08 * _prng, _pall.max() + 0.55 * _prng)   # top headroom for legend
    axb.set_xlabel("Time (s)", fontsize=AXLAB); axb.set_ylabel(r"$P_{lv}$ (mmHg)", fontsize=AXLAB)
    axb.tick_params(labelsize=TICK); axb.grid(alpha=0.25); axb.set_axisbelow(True)
    axb.legend(fontsize=TICK - 1, loc="upper right", framealpha=0.9)
    axb.set_title("(b)", loc="left", fontsize=13)

    # (c) %RMSE vs SNR, shaded CI silhouettes, log-y
    _cmax = 0.0
    for d, cpt, c, lab, mk in [(realistic_ic, clean, C_REAL, "Realistic IC (AWGN+wander)", "o"),
                               (streaming, clean_stream, C_STREAM, f"Streaming (K={K_REANCHOR})", "s")]:
        m, e = curve(d, "pct_rmse_mean", cpt), ci_pct(d, cpt)
        axc.fill_between(x, np.clip(m - e, 1e-3, None), m + e, color=c, alpha=0.18)
        axc.plot(x, m, color=c, lw=2.3, marker=mk, ms=6, label=lab)
        _cmax = max(_cmax, np.nanmax(m + e))
    axc.set_yscale("log")
    axc.set_ylim(top=_cmax * 3.2)     # top headroom so the upper-left legend clears the curves
    axc.set_xticks(x); axc.set_xticklabels(labels, fontsize=TICK)
    axc.set_xlabel("Noise level (SNR)", fontsize=AXLAB); axc.set_ylabel("%RMSE (log)", fontsize=AXLAB)
    axc.tick_params(labelsize=TICK); axc.grid(axis="y", which="both", ls="--", alpha=0.3); axc.set_axisbelow(True)
    axc.legend(fontsize=TICK - 1.5, loc="upper left", framealpha=0.9)
    axc.text(0.97, 0.04, "IC diverges @ 5 dB", transform=axc.transAxes, ha="right", va="bottom",
             fontsize=9, style="italic", color=C_REAL)
    axc.set_title("(c)", loc="left", fontsize=13)

    # (d) R² vs SNR, shaded CI silhouettes
    for d, cpt, c, lab, mk in [(realistic_ic, clean, C_REAL, "Realistic IC", "o"),
                               (streaming, clean_stream, C_STREAM, f"Streaming (K={K_REANCHOR})", "s")]:
        m, e = curve(d, "r2_mean", cpt), ci_r2(d, cpt)
        axd.fill_between(x, m - e, m + e, color=c, alpha=0.18)
        axd.plot(x, m, color=c, lw=2.3, marker=mk, ms=6, label=lab)
    axd.axhline(0, color="#888", ls=":", lw=0.9)
    axd.set_ylim(-0.08, 1.03)
    axd.set_xticks(x); axd.set_xticklabels(labels, fontsize=TICK)
    axd.set_xlabel("Noise level (SNR)", fontsize=AXLAB); axd.set_ylabel(r"$R^2$", fontsize=AXLAB)
    axd.tick_params(labelsize=TICK); axd.grid(axis="y", ls="--", alpha=0.3); axd.set_axisbelow(True)
    axd.legend(fontsize=TICK - 1.5, loc="lower left", framealpha=0.9)
    axd.set_title("(d)", loc="left", fontsize=13)

    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(C.FIG_DIR, f"figureS_realistic_streaming_noise.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved -> figures/figureS_realistic_streaming_noise.svg + .png")


if __name__ == "__main__":
    main()
