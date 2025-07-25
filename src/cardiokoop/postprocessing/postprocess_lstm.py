#!/usr/bin/env python
"""
Fast post-processing for the **Small-LSTM** baseline
===================================================

Outputs exactly the same metrics / pickle fields as the Koopman,
GRU and BiLSTM post-processors.
"""
# ───────────────────────── imports / CLI ─────────────────────────
import argparse, glob, re, time, pickle, os
import numpy as np, torch
from fastdtw import fastdtw
from sklearn.metrics import r2_score
from cardiokoop.training.train_lstm import SmallLSTM, autoregressive_rollout

def parse_args():
    p = argparse.ArgumentParser(description="Post-process Small-LSTM run")
    p.add_argument("--exp-folder",  default="../results")
    p.add_argument("--model-path")
    p.add_argument("--error-csv")
    p.add_argument("--data-folder", default="data")
    p.add_argument("--use-control", action="store_true",
                   help="Include control signal u as additional model input")
    p.add_argument("--control-dim", type=int, default=1,
                   help="Dimensionality of the control signal")
    return p.parse_args()

args         = parse_args()
exp_folder   = args.exp_folder
use_control  = args.use_control
control_dim  = args.control_dim
data_root    = args.data_folder

# ───────────────── locate newest files if not provided ─────────────
def newest(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No file matches {pattern}")
    return max(files, key=lambda f: int(re.search(r'_(\d+)_', f).group(1)))

errfile   = args.error_csv  or newest(f"{exp_folder}/*_error.csv")
ckpt_path = args.model_path or newest(f"{exp_folder}/*_model.ckpt")
ckpt   = torch.load(ckpt_path, map_location='cpu', weights_only=False)
params = ckpt['params']

# ─────────────────────────── data ──────────────────────────
data_name = params['data_name']
X_norm = np.loadtxt(f"{data_root}/{data_name}_test1_x.csv", delimiter=',')
U = np.loadtxt(f"{data_root}/{data_name}_test1_u.csv", delimiter=',')
U_non_norm = U.copy()
U = (U - U.mean(0)) / (U.std(0) + 1e-8)

mean = np.load(f"{data_root}/normalization_mean.npy")
std  = np.load(f"{data_root}/normalization_std.npy")

T, n_signals = params['seq_len'], X_norm.shape[1]
n_seqs       = X_norm.shape[0] // T
X_norm       = X_norm.reshape(n_seqs, T, n_signals)
U            = U.reshape(n_seqs, T, control_dim)
U_non_norm   = U_non_norm.reshape(n_seqs, T, control_dim)
X_true       = X_norm * std + mean                                   # de-norm

# ─────────────────────────── model ─────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model input_dim includes optional control dimension
input_dim = n_signals + (control_dim if use_control else 0)
model  = SmallLSTM(input_dim, params['hidden'], params['dropout'], n_signals).to(device)
# load checkpoint and eval
model.load_state_dict(ckpt['state_dict']); model.eval()
# total number of model parameters
n_params = sum(p.numel() for p in model.parameters())

# ───────────── helpers ─────────────
def pct_rmse(seq, pred):
    return 100*np.sqrt(((seq-pred)**2).mean())/(seq.ptp()+1e-8)

def mean_CI(arr, axis=0):
    arr = np.asarray(arr)
    m   = np.nanmean(arr, axis=axis)
    sd  = np.nanstd(arr,  axis=axis)
    n   = arr.shape[axis]
    return m, 1.96*sd/np.sqrt(n)

# ───────────────── rollout & metrics ──────────────────
def pct_rmse(seq, pred):
    return 100*np.sqrt(((seq - pred) ** 2).mean()) / (seq.ptp() + 1e-8)


t_sec = []           # inference time [s]
preds = []
u_trajs = []  # control signal trajectories (if used)

mse_sig = []; rmse_sig = []; pct_sig = []; r2_sig = []
bias_sig = []; sigma_sig = []; dtw_sig = []

mse_traj = []; rmse_traj = []; r2_traj = []
pct_flat = []; pct_ps = [];    dtw_traj = []
cum_pct  = []                  # cumulative %RMSE curves (Koopman style)

for i, (x_n, true) in enumerate(zip(X_norm, X_true)):
    tic = time.time()
    if use_control:
        u_n_norm = U[i]
        u_trajs.append(U_non_norm[i])  # store non-normalized control traj
        pred     = autoregressive_rollout(
                       model, x_n, params['window'], device, u_n_norm
                   ) * std + mean
    else:
        pred     = autoregressive_rollout(
                       model, x_n, params['window'], device
                   ) * std + mean
    t_sec.append(time.time() - tic)
    preds.append(pred)

    err     = pred - true
    err2    = err ** 2

    # ───────── per-signal metrics ─────────
    mse_k   = err2.mean(0)
    rmse_k  = np.sqrt(mse_k)
    pct_k   = 100 * rmse_k / (true.ptp(0) + 1e-8)

    r2_k    = [r2_score(true[:, j], pred[:, j]) for j in range(n_signals)]
    bias_k  = err.mean(0)
    sigma_k = err.std(0, ddof=1)

    # ───────── DTW (per signal) ─────────
    dtw_vals = [fastdtw(true[:, j], pred[:, j])[0] for j in range(n_signals)]

    # ─────── store per-signal arrays ──────
    mse_sig.append(mse_k);      rmse_sig.append(rmse_k)
    pct_sig.append(pct_k);      r2_sig.append(r2_k)
    bias_sig.append(bias_k);    sigma_sig.append(sigma_k)
    dtw_sig.append(dtw_vals)

    # ─────── store per-trajectory scalars ──────
    mse_traj.append(mse_k.mean())
    rmse_traj.append(rmse_k.mean())
    r2_traj.append(np.mean(r2_k))

    pct_flat.append(pct_rmse(true, pred))
    pct_ps  .append(pct_k.mean())
    dtw_traj.append(np.mean(dtw_vals))

    cum_pct.append(np.cumsum(
        100 * np.sqrt(err2.mean(1)) / (true.ptp() + 1e-8) ))

if __name__ == "__main__":
    # ───────── global flattened %RMSE (historic “old” style) ─────────
    all_true = X_true.reshape(-1, n_signals)
    all_pred = np.vstack(preds)
    pct_global_flat = pct_rmse(all_true, all_pred)
    global_r2_flat  = r2_score(all_true, all_pred)

    # ───────── table-ready aggregated stats (mean ± 95 % CI) ─────────
    rmse_arr  = np.vstack(rmse_sig)
    pct_arr   = np.vstack(pct_sig)
    r2_arr    = np.vstack(r2_sig)
    bias_arr  = np.vstack(bias_sig)
    sigma_arr = np.vstack(sigma_sig)

    B         = len(rmse_arr)                      # number of trajectories
    def ci(a): return 1.96 * a.std(0, ddof=1) / np.sqrt(B)

    rmse_ci   = ci(rmse_arr)
    pct_ci    = ci(pct_arr)
    r2_ci     = ci(r2_arr)
    bias_ci   = ci(bias_arr)
    loa_lower = bias_arr.mean(0) - 1.96 * sigma_arr.mean(0)
    loa_upper = bias_arr.mean(0) + 1.96 * sigma_arr.mean(0)


    # ───────── save results (identical key set) ─────────
    results = dict(
        # per-signal arrays (B × D)
        mse_per_signal       = np.array(mse_sig),
        rmse_per_signal      = np.array(rmse_sig),
        pct_per_signal       = np.array(pct_sig),
        r2_per_signal        = np.array(r2_sig),
        bias_per_signal      = np.array(bias_sig),
        sigma_per_signal     = np.array(sigma_sig),
        dtw_per_signal       = np.array(dtw_sig),

        # per-trajectory vectors (len = B)
        mse_per_trajectory   = np.array(mse_traj),
        rmse_per_trajectory  = np.array(rmse_traj),
        r2_per_trajectory    = np.array(r2_traj),
        pct_per_traj_flat    = np.array(pct_flat),
        pct_per_traj_ps      = np.array(pct_ps),
        dtw_per_trajectory   = np.array(dtw_traj),
        inference_times_s    = np.array(t_sec),

        # table-ready aggregates
        rmse_mean            = rmse_arr.mean(0),   rmse_ci = rmse_ci,
        pct_mean             = pct_arr.mean(0),    pct_ci  = pct_ci,
        r2_mean              = r2_arr.mean(0),     r2_ci   = r2_ci,
        bias_mean            = bias_arr.mean(0),   bias_ci = bias_ci,
        loa_lower            = loa_lower,          loa_upper = loa_upper,

        # misc / raw
        cumulative_pct_per_traj = np.array(cum_pct),
        global_pct_rmse_flat    = pct_global_flat,
        true_per_trajectory     = X_true,
        pred_per_trajectory     = np.stack(preds),
        u_per_trajectory        = np.array(u_trajs),
        n_params                = n_params,
        global_r2_flat       = global_r2_flat    
    )

    out_path = os.path.join(exp_folder, "lstm_postprocessing_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

print(f"Post-processing complete. Results saved to {out_path}")