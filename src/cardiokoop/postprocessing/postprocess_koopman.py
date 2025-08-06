#!/usr/bin/env python
"""
Fast post-processing for a saved **Koopman** experiment
"""

# ───────────────────────── Imports ─────────────────────────
import os, sys, glob, re, time, pickle, argparse
import numpy as np, torch
from fastdtw import fastdtw

from cardiokoop.postprocessing.postprocess_utils import load_weights_koopman

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────── flag parsing helper ──────────────────
def _parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Koopman post-processing")
    p.add_argument("--exp-folder",  "-f", default="exp1_best_working_very_well",
                   help="Folder with *_model.ckpt / *_error.csv")
    p.add_argument("--model-path",  default=None,
                   help="Explicit checkpoint (else: newest in folder)")
    p.add_argument("--error-csv",   default=None,
                   help="Explicit *_error.csv (else: newest in folder)")
    p.add_argument("--data-folder", "-d", default="data",
                   help="Root that contains normalised csv data")
    return p.parse_args(argv)

# ─────────────────── miscellaneous helpers ────────────────
def _newest(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files match {pattern!r}")
    return max(files, key=lambda f: int(re.search(r"_(\d+)_", f).group(1)))

def pct_rmse(seq, pred):
    return 100 * np.sqrt(((seq - pred) ** 2).mean()) / (np.ptp(seq) + 1e-8)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-8
    return 1.0 - ss_res / ss_tot

# ───────────────────────── main() ─────────────────────────
def main(argv=None):
    args = _parse(argv)

    exp_folder = args.exp_folder
    data_root  = args.data_folder

    # locate checkpoint / csv
    errfile   = args.error_csv  or _newest(f"{exp_folder}/*_error.csv")
    ckpt_file = args.model_path or _newest(f"{exp_folder}/*_model.ckpt")

    # load model
    ckpt   = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    params = ckpt["params"]; params["model_path"] = ckpt_file

    W, b, model = load_weights_koopman(
        ckpt_file,
        len(params["widths"]) - 1,
        len(params["widths_omega_real"]) - 1,
        params["num_real"],
        params["num_complex_pairs"],
    )
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    # validation data
    data_name = params["data_name"]         # e.g. csv_data_500_12sigs
    X = np.loadtxt(f"{data_root}/{data_name}_val1_x.csv", delimiter=",")
    U = np.loadtxt(f"{data_root}/{data_name}_val1_u.csv", delimiter=",")
    U_non_norm = U.copy()
    U = (U - U.mean(0)) / U.std(0)

    sig_std  = np.load(f"{data_root}/normalization_std.npy")
    sig_mean = np.load(f"{data_root}/normalization_mean.npy")

    T  = params["len_time"]
    B  = X.shape[0] // T
    X_rs = X.reshape(B, T, -1)
    U_rs = U.reshape(B, T, -1)

    # vectorised rollout
    def rollout_batch(enc, dec, get_omegas, advance, x0, u):
        y = enc(x0); outs = []
        for t in range(u.size(0)):
            outs.append(dec(y))
            y = advance(y, get_omegas(y), u[t])
        return torch.stack(outs, 0)   # (T,B,D)

    x0   = torch.tensor(X_rs[:, 0, :], dtype=torch.float32, device=device)
    uSeq = torch.tensor(U_rs, dtype=torch.float32, device=device).permute(1, 0, 2)

    with torch.no_grad():
        pred_norm = rollout_batch(model.encoder, model.decoder,
                                  model.get_omegas, model._advance,
                                  x0, uSeq).cpu().numpy()

    preds = pred_norm.transpose(1, 0, 2) * sig_std + sig_mean
    X_rs_denorm = X_rs * sig_std + sig_mean

    # ── metrics ───────────────────────────────────────────────
    mse_sig = []; rmse_sig = []; pct_sig = []; r2_sig = []
    bias_sig = []; sigma_sig = []; dtw_traj = []; pct_flat = []
    pct_ps   = []; mse_traj  = []; rmse_traj = []; r2_traj = []
    cum_pct  = []; t_inf = []

    for i in range(B):
        seq, pred = X_rs_denorm[i], preds[i]
        tic = time.time()

        err  = pred - seq
        mse  = (err ** 2).mean(0)
        rmse = np.sqrt(mse)
        pct = 100 * rmse / (np.ptp(seq, axis=0) + 1e-8)

        r2   = np.array([r2_score(seq[:, j], pred[:, j])
                         for j in range(seq.shape[1])])

        mse_sig.append(mse);    rmse_sig.append(rmse);  pct_sig.append(pct)
        r2_sig.append(r2);      bias_sig.append(err.mean(0))
        sigma_sig.append(err.std(0, ddof=1))

        mse_traj.append(mse.mean()); rmse_traj.append(rmse.mean())
        r2_traj.append(r2.mean());   pct_flat.append(pct_rmse(seq, pred))
        pct_ps.append(pct.mean())

        dtw_traj.append(np.mean(
            [fastdtw(seq[:, j], pred[:, j])[0] for j in range(seq.shape[1])]))
        cum_pct.append(np.cumsum(
            100 * np.sqrt(((seq - pred)**2).mean(1)) / (np.ptp(seq) + 1e-8)))
        t_inf.append(time.time() - tic)

    # global flattened RMSE
    seq_all  = X_rs_denorm.reshape(-1, X_rs_denorm.shape[2])
    pred_all = preds.reshape(-1, preds.shape[2])
    pct_global_flat = pct_rmse(seq_all, pred_all)
    global_r2_flat  = r2_score(seq_all, pred_all)

    # ── aggregate & save ─────────────────────────────────────
    rmse_arr = np.vstack(rmse_sig); pct_arr = np.vstack(pct_sig)
    r2_arr   = np.vstack(r2_sig);   bias_arr = np.vstack(bias_sig)
    sigma_arr= np.vstack(sigma_sig)
    Bfloat   = float(B)

    results = dict(
        mse_per_signal = rmse_arr**2,
        rmse_per_signal = rmse_arr,
        pct_per_signal = pct_arr,
        r2_per_signal  = r2_arr,
        bias_per_signal = bias_arr,
        sigma_per_signal= sigma_arr,

        mse_per_trajectory  = np.array(mse_traj),
        rmse_per_trajectory = np.array(rmse_traj),
        r2_per_trajectory   = np.array(r2_traj),
        pct_per_traj_flat   = np.array(pct_flat),
        pct_per_traj_ps     = np.array(pct_ps),
        dtw_per_trajectory  = np.array(dtw_traj),
        inference_times_s   = np.array(t_inf),

        rmse_mean = rmse_arr.mean(0),
        rmse_ci   = 1.96 * rmse_arr.std(0, ddof=1) / np.sqrt(Bfloat),
        pct_mean  = pct_arr.mean(0),
        pct_ci    = 1.96 * pct_arr.std(0, ddof=1) / np.sqrt(Bfloat),
        r2_mean   = r2_arr.mean(0),
        r2_ci     = 1.96 * r2_arr.std(0, ddof=1) / np.sqrt(Bfloat),
        bias_mean = bias_arr.mean(0),
        bias_ci   = 1.96 * bias_arr.std(0, ddof=1) / np.sqrt(Bfloat),
        loa_lower = bias_arr.mean(0) - 1.96 * sigma_arr.mean(0),
        loa_upper = bias_arr.mean(0) + 1.96 * sigma_arr.mean(0),

        cumulative_pct_per_traj = np.array(cum_pct),
        global_pct_rmse_flat    = pct_global_flat,
        global_r2_flat          = global_r2_flat,
        true_per_trajectory     = X_rs_denorm,
        u_per_trajectory        = np.array(U_non_norm),
        pred_per_trajectory     = preds,
        n_params                = n_params
    )

    # absolute, guaranteed path + auto-mkdir
    out_file = os.path.abspath(os.path.join(exp_folder,
                                            "koopman_postprocessing_results.pkl"))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, "wb") as f:
        pickle.dump(results, f)

    print(f"✅ Post-processing complete → {out_file}")

# ───────────────────────── CLI entry ─────────────────────────
if __name__ == "__main__":
    main()
