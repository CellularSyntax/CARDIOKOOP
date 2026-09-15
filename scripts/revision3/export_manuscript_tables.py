#!/usr/bin/env python3
"""
Revision 3 (Array, minor revision) — export every number that appears in the
manuscript's Tables 3, 4 and 5 and in the statistics paragraph from ONE run
against the frozen final Koopman checkpoint and the committed baseline
predictions, on the seed-42 **test1** split (B = 50 trajectories, T = 1500
steps, 1499-step autoregressive horizon, 12 signals).

Why this script exists
----------------------
* The committed ``results/koopman/koopman_postprocessing_results.pkl`` is the
  *validation*-split (val1) evaluation from revision 1, whereas every baseline
  pickle is test1.  This script writes a test1 Koopman pickle
  (``results/revision3/koopman_test1_postprocessing_results.pkl``) with the
  same keys as the baseline pickles so that every manuscript number has a
  committed file of origin.
* The MLP and AR(20) rows of the revision-1 Table 4 were produced by older code;
  they are regenerated here with the revision-2 functions
  (``scripts/revision2/build_tables_figures.py``: ``mlp_pred`` with the committed
  ``checkpoints/mlp_baseline.pt`` and clip 20, ``ar_pred`` = OLS AR(20) fitted on
  train1).
* The statistics (Shapiro–Wilk, Friedman over the six dynamical models,
  paired two-sided Wilcoxon Koopman vs. each baseline, Bonferroni x5) are
  recomputed on the per-trajectory %RMSE of exactly these predictions.

Metric conventions (identical to ``scripts/revision2/_common.py``)
------------------------------------------------------------------
* %RMSE  : per signal, RMSE over the 1500 steps divided by the peak-to-peak
           range of the true signal, averaged over the 12 signals -> one value
           per trajectory (``pct_per_traj_ps``).  Table 3/5 report
           mean ± 95 % CI (1.96·SD/√50) over the 50 trajectories.
* R² pooled (per trajectory) : all 12 signals x 1500 steps of one trajectory
           concatenated (``r2_flat``); Table 3 / Table 5 / Fig. 5g report
           mean ± 95 % CI over the 50 trajectories.
* R² per signal : sklearn ``r2_score`` of one signal of one trajectory; Table 4
           reports mean ± 95 % CI over the 50 trajectories.
* R² signal-averaged (per trajectory) : mean of the 12 per-signal R² of one
           trajectory (``r2_per_trajectory`` in the pickles).  Used only for the
           "trajectories with R² < 0" statements of the per-signal analysis.
* Bland–Altman : bias = mean error, LoA = bias ± 1.96·SD(error), both averaged
           over trajectories.
* The statistical unit is the test trajectory (n = 50) everywhere.

Outputs (results/revision3/) — .md/.json/.tsv only (``*.csv`` is Git-LFS filtered)
----------------------------------------------------------------------------------
  koopman_test1_postprocessing_results.pkl
  table3_overall.md / .json / .tsv
  table4_per_signal_full.md / .json / .tsv
  table5_noise.md / .json / .tsv
  statistics.json
  r2_conventions.json
  run_info.json   (platform, torch version, dtype, checkpoint, timings)

Usage
-----
  python scripts/revision3/export_manuscript_tables.py            # float64 Koopman rollout (default)
  python scripts/revision3/export_manuscript_tables.py --dtype float32

Numerical note: the 1499-step rollout amplifies floating-point differences;
CPU-f32 / CPU-f64 / GPU runs of the same checkpoint differ by <= 0.1 pp %RMSE
(Koopman 17.46 / 17.50 / 17.54).  All statements in the manuscript are robust
to these differences (see REVISION3_NOTES.md).
"""
import os
import sys
import json
import time
import platform
import argparse

import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REV2 = os.path.join(os.path.dirname(HERE), "revision2")
sys.path.insert(0, REV2)

import _common as C                                   # noqa: E402
import build_tables_figures as BT                     # noqa: E402  (mlp_pred, ar_pred, SNR_DB, SIM_TIME_S, TRAIN_TIMES_MIN)

REV3_DIR = os.path.join(C.RESULT_DIR, "revision3")
os.makedirs(REV3_DIR, exist_ok=True)

MODELS      = C.MODELS_ALL                             # Koopman, GRU, LSTM, BiLSTM, MLP, AR(20), DLinear, NLinear
TABLE_ORDER = ["Koopman", "LSTM", "GRU", "BiLSTM", "MLP", "AR(20)", "DLinear", "NLinear"]   # manuscript row order
DYNAMICAL   = ["Koopman", "LSTM", "GRU", "BiLSTM", "MLP", "AR(20)"]                          # autoregressive models (stats)
BASELINES   = ["LSTM", "GRU", "BiLSTM", "MLP", "AR(20)"]
SNR_DB      = BT.SNR_DB                                # [30, 20, 10, 5]

# Published per-trajectory inference times [s] and speed-ups vs. the 2.05 s full-order
# simulation (manuscript Table 3, measured on the original workstation when the
# checkpoints were produced).  Copied verbatim from build_tables_figures.main(); they are
# re-used rather than re-measured for exactly the reason documented there.
PUB_INF     = {"Koopman": 0.005, "GRU": 0.640, "LSTM": 0.750, "BiLSTM": 1.600, "MLP": 0.169, "AR(20)": 0.008}
PUB_SPD     = {"Koopman": 379.2, "GRU": 2.9, "LSTM": 2.5, "BiLSTM": 1.2, "MLP": 12.1, "AR(20)": 257.2}
PUB_SPD_LIN = {"DLinear": 6028.0, "NLinear": 11388.0}


# ───────────────────────────── helpers ─────────────────────────────
def koopman_rollout_dtype(model, x0_norm, u_seq_norm, dtype):
    """Batched Koopman rollout in the requested floating-point precision (CPU or GPU)."""
    device = next(model.parameters()).device
    model = model.to(dtype)
    x0 = torch.as_tensor(x0_norm, dtype=dtype, device=device)
    u = torch.as_tensor(u_seq_norm, dtype=dtype, device=device).permute(1, 0, 2)
    outs = []
    with torch.no_grad():
        y = model.encoder(x0)
        for t in range(u.size(0)):
            outs.append(model.decoder(y).cpu().numpy())
            y = C.koopman_advance(model, y, u[t])
    return np.stack(outs, axis=1).astype(np.float64)


def per_traj_pooled_r2(true, pred):
    return np.array([C.r2_flat(true[i], pred[i]) for i in range(true.shape[0])])


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def tsv_table(header, rows):
    return "\n".join(["\t".join(header)] + ["\t".join(str(c) for c in r) for r in rows]) + "\n"


def write_table(stem, header, rows_md, rows_json, caption):
    with open(os.path.join(REV3_DIR, stem + ".md"), "w") as f:
        f.write(f"**{caption}**\n\n" + md_table(header, rows_md))
    with open(os.path.join(REV3_DIR, stem + ".tsv"), "w") as f:
        f.write(tsv_table(header, rows_md))
    with open(os.path.join(REV3_DIR, stem + ".json"), "w") as f:
        json.dump({"caption": caption, "rows": rows_json}, f, indent=2)


def bonf(p, k=5):
    return float(min(1.0, p * k))


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                    help="precision of the Koopman rollout (default float64)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.threads:
        torch.set_num_threads(args.threads)

    C.set_all_seeds(C.SEED)
    device = C.get_device()
    t_start = time.time()

    Xn, Un, Uph, Xte = C.load_split("test")
    sig_mean, sig_std = C.load_norm_stats()
    B, T, D = Xte.shape
    assert (B, T, D) == (50, C.T_STEPS, 12), (B, T, D)

    # ── (i) Koopman: frozen checkpoint, test1 rollout ─────────────────────
    model, params, n_koop, ckpt_path = C.load_koopman(device)
    assert n_koop == 17_239_600, n_koop
    t0 = time.time()
    pred_koop_norm = koopman_rollout_dtype(model, Xn[:, 0, :], Un, dtype)
    t_koop = time.time() - t0
    pred_koop = pred_koop_norm * sig_std + sig_mean
    # published per-trajectory inference times (Table 3, measured on the original
    # workstation) are re-used exactly as in scripts/revision2/build_tables_figures.py
    inf_koop = np.full(B, PUB_INF["Koopman"])
    koop_res = C.compute_results_dict(Xte, pred_koop, Uph, inf_koop, n_koop)
    koop_res["bias_ci"] = 1.96 * koop_res["bias_per_signal"].std(0, ddof=1) / np.sqrt(B)
    try:                                                   # optional (fastdtw not required)
        from fastdtw import fastdtw
        koop_res["dtw_per_trajectory"] = np.array(
            [fastdtw(Xte[i], pred_koop[i])[0] for i in range(B)])
    except Exception:
        pass
    koop_res["split"] = "test1"
    koop_res["checkpoint"] = os.path.relpath(ckpt_path, C.REPO_ROOT)
    koop_res["rollout_dtype"] = args.dtype
    koop_res["device"] = str(device)
    C.save_pickle(koop_res, os.path.join(REV3_DIR, "koopman_test1_postprocessing_results.pkl"))
    print(f"Koopman test1 rollout ({args.dtype}, {device}): %RMSE = {koop_res['pct_per_traj_ps'].mean():.3f}  "
          f"[{t_koop:.1f} s]")

    # ── (ii) baselines: committed pickles + regenerated MLP / AR(20) ──────
    preds, nparams, inftimes, speedup = {"Koopman": pred_koop}, {"Koopman": n_koop}, {}, {}
    for m in ["GRU", "LSTM", "BiLSTM", "DLinear", "NLinear"]:
        r = C.load_pickle(os.path.join(C.RESULT_DIR, m.lower(), f"{m.lower()}_postprocessing_results.pkl"))
        assert np.allclose(np.asarray(r["true_per_trajectory"]), Xte, atol=1e-6), f"{m} pickle is not test1"
        preds[m] = np.asarray(r["pred_per_trajectory"], dtype=np.float64)
        nparams[m] = int(r["n_params"])
    t0 = time.time()
    preds["MLP"], nparams["MLP"], _ = BT.mlp_pred(Xn, sig_mean, sig_std)
    t_mlp = time.time() - t0
    t0 = time.time()
    preds["AR(20)"], nparams["AR(20)"], _ = BT.ar_pred(Xn, sig_mean, sig_std)
    t_ar = time.time() - t0
    for m in ["Koopman", "GRU", "LSTM", "BiLSTM", "MLP", "AR(20)"]:
        inftimes[m] = PUB_INF[m]; speedup[m] = PUB_SPD[m]
    for m in ["DLinear", "NLinear"]:
        speedup[m] = PUB_SPD_LIN[m]; inftimes[m] = BT.SIM_TIME_S / PUB_SPD_LIN[m]

    res = {m: C.compute_results_dict(Xte, preds[m], Uph, np.full(B, inftimes[m]), nparams[m]) for m in MODELS}
    pooled = {m: per_traj_pooled_r2(Xte, preds[m]) for m in MODELS}          # (50,) pooled R² per trajectory
    pct = {m: res[m]["pct_per_traj_ps"] for m in MODELS}                     # (50,) %RMSE per trajectory
    sigavg = {m: res[m]["r2_per_trajectory"] for m in MODELS}                # (50,) signal-averaged R²

    # ── (iii-a) Table 3 ───────────────────────────────────────────────────
    hdr3 = ["Model", "R² (pooled, mean ± 95% CI)", "%RMSE (mean ± 95% CI)", "Inference time (s)",
            "Speed-up vs. simulation", "Parameters", "Training time (min)"]
    rows_md, rows_js = [], []
    for m in TABLE_ORDER:
        inf = inftimes[m]
        rows_md.append([m, f"{pooled[m].mean():.2f} ± {C.ci95(pooled[m]):.2f}",
                        f"{pct[m].mean():.1f} ± {C.ci95(pct[m]):.1f}",
                        ("<0.001" if inf < 1e-3 else f"{inf:.3g}"),
                        (f"{speedup[m]:.1f}×" if speedup[m] < 1000 else f"{speedup[m]:,.0f}×"),
                        f"{nparams[m]:,}", BT.TRAIN_TIMES_MIN.get(m, "-")])
        rows_js.append(dict(model=m, r2_pooled_mean=float(pooled[m].mean()), r2_pooled_ci95=C.ci95(pooled[m]),
                            r2_pooled_global=float(res[m]["global_r2_flat"]),
                            pct_rmse_mean=float(pct[m].mean()), pct_rmse_ci95=C.ci95(pct[m]),
                            pct_rmse_sd=float(pct[m].std(ddof=1)),
                            inference_time_s=float(inf), speedup_vs_simulation=float(speedup[m]),
                            n_params=int(nparams[m]), training_time_min=BT.TRAIN_TIMES_MIN.get(m, "-")))
    write_table("table3_overall", hdr3, rows_md, rows_js,
                "Table 3 — overall comparison on the seed-42 test split (n = 50 trajectories, 1499-step horizon). "
                "R² = pooled per-trajectory R² (all signals and steps of a trajectory concatenated), mean ± 95% CI "
                "across trajectories; %RMSE = per-signal PTP-normalised RMSE averaged over signals, mean ± 95% CI. "
                "Inference times / speed-ups are the published workstation measurements (see build_tables_figures.py).")
    print("\n" + md_table(hdr3, rows_md))

    # ── (iii-b) Table 4 (full, 8 models x 12 signals) ─────────────────────
    hdr4 = ["Model", "Signal", "RMSE ± 95% CI", "%RMSE ± 95% CI", "R² ± 95% CI", "Bias [LoA]"]
    rows_md, rows_js = [], []
    for m in TABLE_ORDER:
        r = res[m]
        for j, s in enumerate(C.SIG_NAMES):
            u = C.UNITS[j]
            rows_md.append([m, s, f"{r['rmse_mean'][j]:.1f} ± {r['rmse_ci'][j]:.1f} {u}",
                            f"{r['pct_mean'][j]:.1f}% ± {r['pct_ci'][j]:.1f}%",
                            f"{r['r2_mean'][j]:.2f} ± {r['r2_ci'][j]:.2f}",
                            f"{r['bias_mean'][j]:.1f} [{r['loa_lower'][j]:.1f}, {r['loa_upper'][j]:.1f}] {u}"])
            rows_js.append(dict(model=m, signal=s, unit=u,
                                rmse=float(r["rmse_mean"][j]), rmse_ci95=float(r["rmse_ci"][j]),
                                pct_rmse=float(r["pct_mean"][j]), pct_rmse_ci95=float(r["pct_ci"][j]),
                                r2=float(r["r2_mean"][j]), r2_ci95=float(r["r2_ci"][j]),
                                bias=float(r["bias_mean"][j]), loa_lower=float(r["loa_lower"][j]),
                                loa_upper=float(r["loa_upper"][j])))
    write_table("table4_per_signal_full", hdr4, rows_md, rows_js,
                "Table 4 (full) — per-signal RMSE, %RMSE, R² (mean ± 95% CI across the 50 test trajectories) and "
                "Bland–Altman bias [95% limits of agreement] for all eight models on the seed-42 test split.")

    # ── (iii-c) Table 5: clean row = Table 3 predictions, noisy rows from JSONs ──
    with open(os.path.join(C.RESULT_DIR, "noise_robustness.json")) as f:
        nr_koop = json.load(f)
    with open(os.path.join(C.RESULT_DIR, "noise_robustness_baselines.json")) as f:
        nr_bl = json.load(f)
    with open(os.path.join(C.REV2_DIR, "noise_robustness_linear.json")) as f:
        nr_lin = json.load(f)

    def noisy(m, snr):
        d = nr_koop if m == "Koopman" else (nr_lin[m] if m in nr_lin else nr_bl[m])
        e = d["noisy"][str(snr)]
        return e["pct_rmse_mean"], e.get("pct_rmse_ci95", float("nan")), e["r2_mean"], e.get("r2_sd", float("nan"))

    hdr5 = ["Condition"] + [f"{m} %RMSE" for m in TABLE_ORDER] + [f"{m} R²" for m in TABLE_ORDER]
    rows_md, rows_js = [], []
    row = ["Clean"] + [f"{pct[m].mean():.1f}" for m in TABLE_ORDER] + [f"{pooled[m].mean():.2f}" for m in TABLE_ORDER]
    rows_md.append(row)
    rows_js.append(dict(condition="clean", source="table3 predictions",
                        **{m: dict(pct_rmse_mean=float(pct[m].mean()), pct_rmse_ci95=C.ci95(pct[m]),
                                   r2_mean=float(pooled[m].mean()), r2_sd=float(pooled[m].std(ddof=1)))
                           for m in TABLE_ORDER}))
    for snr in SNR_DB:
        vals = {m: noisy(m, snr) for m in TABLE_ORDER}
        rows_md.append([f"{snr} dB"] + [f"{vals[m][0]:.1f}" for m in TABLE_ORDER] + [f"{vals[m][2]:.2f}" for m in TABLE_ORDER])
        rows_js.append(dict(condition=f"{snr} dB", source="noise_robustness*.json",
                            **{m: dict(pct_rmse_mean=float(vals[m][0]), pct_rmse_ci95=float(vals[m][1]),
                                       r2_mean=float(vals[m][2]), r2_sd=float(vals[m][3])) for m in TABLE_ORDER}))
    write_table("table5_noise", hdr5, rows_md, rows_js,
                "Table 5 — %RMSE and pooled R² under AWGN on the initial observation / warm-up window (SNR 30–5 dB), "
                "seed-42 test split, no retraining. Clean-condition values are identical to Table 3 (same predictions); "
                "noisy conditions from results/noise_robustness.json, noise_robustness_baselines.json and "
                "revision2/noise_robustness_linear.json.")

    # ── (iv) statistics on per-trajectory %RMSE (n = 50) ───────────────────
    st = {"metric": "pct_per_traj_ps (%RMSE per trajectory, averaged over signals)", "n_trajectories": int(B),
          "unit_of_analysis": "test trajectory", "models": DYNAMICAL,
          "summary": {m: dict(mean=float(pct[m].mean()), sd=float(pct[m].std(ddof=1)), ci95=C.ci95(pct[m]),
                              median=float(np.median(pct[m])), q25=float(np.percentile(pct[m], 25)),
                              q75=float(np.percentile(pct[m], 75))) for m in DYNAMICAL}}
    st["shapiro_wilk"] = {}
    for m in DYNAMICAL:
        W, p = stats.shapiro(pct[m])
        st["shapiro_wilk"][m] = dict(W=float(W), p=float(p), normal_at_0_05=bool(p >= 0.05))
    fr = stats.friedmanchisquare(*[pct[m] for m in DYNAMICAL])
    st["friedman"] = dict(models=DYNAMICAL, k=len(DYNAMICAL), n=int(B), statistic=float(fr.statistic),
                          df=len(DYNAMICAL) - 1, p=float(fr.pvalue))
    st["wilcoxon_koopman_vs_baselines"] = {}
    for m in BASELINES:
        diff = pct[m] - pct["Koopman"]
        w = stats.wilcoxon(pct["Koopman"], pct[m], alternative="two-sided", method="exact", zero_method="wilcox")
        st["wilcoxon_koopman_vs_baselines"][m] = dict(
            statistic_W=float(w.statistic), p_two_sided=float(w.pvalue), p_bonferroni_x5=bonf(w.pvalue, 5),
            n_trajectories_worse_than_koopman=int((diff > 0).sum()),
            n_trajectories_better_than_koopman=int((diff < 0).sum()), n_ties=int((diff == 0).sum()),
            median_paired_difference_pp=float(np.median(diff)), mean_paired_difference_pp=float(diff.mean()),
            method="exact two-sided paired Wilcoxon signed-rank (scipy), zeros discarded")
    st["bonferroni"] = dict(n_comparisons=len(BASELINES), alpha_family=0.05, alpha_per_test=0.05 / len(BASELINES))
    st["note"] = ("Shapiro–Wilk p < 0.05 for all models motivates the non-parametric tests. Friedman omnibus across the six "
                  "autoregressive/dynamical models, followed by paired two-sided Wilcoxon signed-rank tests of Koopman vs. "
                  "each of the five baselines with Bonferroni correction (x5). DLinear/NLinear are direct (non-autoregressive) "
                  "forecasters and are not part of the omnibus test.")
    with open(os.path.join(REV3_DIR, "statistics.json"), "w") as f:
        json.dump(st, f, indent=2)
    print(f"Friedman chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.2e}")
    for m in BASELINES:
        w = st["wilcoxon_koopman_vs_baselines"][m]
        print(f"  Wilcoxon Koopman vs {m}: W = {w['statistic_W']:.0f}, p = {w['p_two_sided']:.2e} "
              f"(Bonferroni: {w['p_bonferroni_x5']:.2e}), worse = {w['n_trajectories_worse_than_koopman']}/50")

    # ── (v) R² conventions ────────────────────────────────────────────────
    conv = {"definitions": {
        "pooled_per_trajectory": "1 - SS_res/SS_tot over all 12 signals x 1500 steps of one trajectory concatenated (r2_flat); "
                                 "Table 3, Table 5 and Fig. 5g report mean ± 95% CI across the 50 trajectories.",
        "pooled_global": "r2_flat over all 50 trajectories concatenated (global_r2_flat in the pickles).",
        "per_signal": "sklearn r2_score of one signal of one trajectory; Table 4 reports mean ± 95% CI across trajectories.",
        "signal_averaged_per_trajectory": "mean of the 12 per-signal R² of one trajectory (r2_per_trajectory in the pickles); "
                                          "used for the per-signal analysis 'trajectories with R² < 0' statements.",
        "ci95": "1.96 * SD / sqrt(50) across trajectories"}, "models": {}}
    for m in TABLE_ORDER:
        r2ps = res[m]["r2_per_signal"]
        conv["models"][m] = dict(
            pooled=dict(mean=float(pooled[m].mean()), sd=float(pooled[m].std(ddof=1)), ci95=C.ci95(pooled[m]),
                        median=float(np.median(pooled[m])), min=float(pooled[m].min()), max=float(pooled[m].max()),
                        n_negative=int((pooled[m] < 0).sum()), global_pooled=float(res[m]["global_r2_flat"])),
            signal_averaged=dict(mean=float(sigavg[m].mean()), sd=float(sigavg[m].std(ddof=1)), ci95=C.ci95(sigavg[m]),
                                 median=float(np.median(sigavg[m])), n_negative=int((sigavg[m] < 0).sum())),
            per_signal_mean_r2={s: float(r2ps[:, j].mean()) for j, s in enumerate(C.SIG_NAMES)},
            per_signal_n_negative_trajectories={s: int((r2ps[:, j] < 0).sum()) for j, s in enumerate(C.SIG_NAMES)},
            mean_of_per_signal_r2=float(r2ps.mean()))
    with open(os.path.join(REV3_DIR, "r2_conventions.json"), "w") as f:
        json.dump(conv, f, indent=2)

    # ── run info ──────────────────────────────────────────────────────────
    info = dict(script=os.path.relpath(__file__, C.REPO_ROOT), split="test1", seed=C.SEED,
                checkpoint=os.path.relpath(ckpt_path, C.REPO_ROOT), n_params_koopman=int(n_koop),
                rollout_dtype=args.dtype, device=str(device), torch=torch.__version__, numpy=np.__version__,
                python=platform.python_version(), platform=platform.platform(),
                cpu=platform.processor() or platform.machine(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                wallclock_s=dict(koopman_rollout=t_koop, mlp_rollout=t_mlp, ar_fit_and_rollout=t_ar,
                                 total=time.time() - t_start),
                koopman_pct_rmse_mean=float(pct["Koopman"].mean()), koopman_r2_pooled_mean=float(pooled["Koopman"].mean()))
    with open(os.path.join(REV3_DIR, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nAll outputs written to {REV3_DIR}  (total {info['wallclock_s']['total']:.0f} s)")


if __name__ == "__main__":
    main()
