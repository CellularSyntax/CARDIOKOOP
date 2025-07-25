#!/usr/bin/env python
"""
Fast post-processing for the **Small-GRU** baseline
==================================================

Outputs exactly the same metric set (and `results.pkl` structure) as the
Koopman, LSTM and BiLSTM post-processors.

Three %RMSE definitions
-----------------------
  • pct_traj_flat   – flattened RMSE per sequence (historic “old” style)  
  • pct_traj_ps     – per-signal %RMSE averaged inside each sequence  
  • pct_global_flat – one %RMSE for the whole validation set
"""
# ───────────────────────── imports / CLI ──────────────────────────
import argparse, glob, re, time, pickle, os
import numpy as np, torch
from fastdtw import fastdtw
from sklearn.metrics import r2_score
from cardiokoop.training.train_gru import SmallGRU, autoregressive_rollout

def parse_args():
    p = argparse.ArgumentParser(description="Post-process Small-GRU run")
    p.add_argument("--exp-folder",   default="../results", help="results dir")
    p.add_argument("--model-path",   help="explicit model.ckpt")
    p.add_argument("--error-csv",    help="explicit *_error.csv")
    p.add_argument("--data-folder",  default="data",       help="normalised csv root")
    p.add_argument("--use-control",  action="store_true",
                   help="Include control signal u as additional model input")
    p.add_argument("--control-dim", type=int, default=1,
                   help="Dimensionality of the control signal")
    return p.parse_args()

args         = parse_args()
exp_folder   = args.exp_folder
use_control  = args.use_control
control_dim  = args.control_dim
# ───────────────── locate newest ckpt / csv if not given ───────────
def newest(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files match {pattern!r}")
    return max(files, key=lambda f: int(re.search(r'_(\d+)_', f).group(1)))

errfile   = args.error_csv  or newest(f"{exp_folder}/*_error.csv")
ckpt_path = args.model_path or newest(f"{exp_folder}/*_model.ckpt")

ckpt   = torch.load(ckpt_path, map_location='cpu', weights_only=False)
params = ckpt['params']

# ─────────────────────────── data ────────────────────────────
data_root = args.data_folder
data_name = params['data_name']
X_norm = np.loadtxt(f"{data_root}/{data_name}_test1_x.csv", delimiter=',')
U = np.loadtxt(f"{data_root}/{data_name}_test1_u.csv", delimiter=',')
U_non_norm = U.copy()
U = (U - U.mean(0)) / (U.std(0) + 1e-8)

mean = np.load(f"{data_root}/normalization_mean.npy")
std  = np.load(f"{data_root}/normalization_std.npy")

window, T = params['window'], params['seq_len']
n_signals = X_norm.shape[1]
n_seqs    = X_norm.shape[0] // T
X_norm    = X_norm.reshape(n_seqs, T, n_signals)
U         = U.reshape(n_seqs, T, control_dim)
U_non_norm= U_non_norm.reshape(n_seqs, T, control_dim)
X_true    = X_norm * std + mean                                # de-norm

# ─────────────────────────── model ───────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = n_signals + (control_dim if use_control else 0)
model     = SmallGRU(input_dim, params['hidden'], params['dropout'], n_signals).to(device)
# load parameters and eval
model.load_state_dict(ckpt['state_dict']); model.eval()
# compute total number of trainable parameters
n_params = sum(p.numel() for p in model.parameters())

# ───────────── rollout & all metrics (per sequence) ──────────
def pct_rmse(seq, pred):
    return 100*np.sqrt(((seq-pred)**2).mean())/(seq.ptp()+1e-8)

t_infer=[]
preds   = []
u_trajs = []  # store non-normalized control trajectories

mse_sig=[];  rmse_sig=[]; pct_sig=[];  r2_sig=[]
bias_sig=[]; sigma_sig=[]; dtw_sig=[]
mse_traj=[]; rmse_traj=[]; r2_traj=[]
pct_flat=[]; pct_ps=[];    dtw_traj=[]
cum_pct=[]   # cumulative %RMSE curves (Koopman style)

for i, (x_n, x_org) in enumerate(zip(X_norm, X_true)):
    tic = time.time()
    if use_control:
        u_n_norm     = U[i]
        u_trajs.append(U_non_norm[i])  # store non-normalized control trajectory
        pred = autoregressive_rollout(
            model, x_n, window, device, u_n_norm
        ) * std + mean
    else:
        pred = autoregressive_rollout(model, x_n, window, device) * std + mean
    t_infer.append(time.time() - tic)
    preds.append(pred)

    diff   = pred - x_org
    diff2  = diff**2

    mse_k  = diff2.mean(0)
    rmse_k = np.sqrt(mse_k)
    den_k  = x_org.ptp(0) + 1e-8
    pct_k  = 100*rmse_k/den_k

    r2_k   = [r2_score(x_org[:,j], pred[:,j]) for j in range(n_signals)]
    bias_k = diff.mean(0)
    sigma_k= diff.std(0, ddof=1)

    loa_low_k  = bias_k - 1.96*sigma_k
    loa_high_k = bias_k + 1.96*sigma_k

    pct_f = pct_rmse(x_org, pred)
    dtw_vals = [fastdtw(x_org[:,j], pred[:,j])[0] for j in range(n_signals)]

    # store per-signal arrays
    mse_sig  .append(mse_k);   rmse_sig .append(rmse_k);  pct_sig .append(pct_k)
    r2_sig   .append(r2_k);    bias_sig .append(bias_k);  sigma_sig.append(sigma_k)
    dtw_sig  .append(dtw_vals)

    # store per-trajectory scalars
    mse_traj .append(mse_k.mean());  rmse_traj.append(rmse_k.mean()); r2_traj.append(np.mean(r2_k))
    pct_flat .append(pct_f);         pct_ps   .append(pct_k.mean());  dtw_traj.append(np.mean(dtw_vals))

    # Koopman-style cumulative %RMSE curve
    rmse_t  = np.sqrt(diff2.mean(1))
    cum_pct.append(np.cumsum(100*rmse_t/(x_org.ptp()+1e-8)))

if __name__ == "__main__":
    # ─────────────── global flattened %RMSE (old style) ─────────
    all_true = X_true.reshape(-1, n_signals)
    all_pred = np.vstack(preds)
    pct_global_flat = pct_rmse(all_true, all_pred)
    global_r2_flat  = r2_score(all_true, all_pred)


    # -------------- aggregated stats (mean ± 95 % CI) ---------
    def mean_CI(arr):
        arr = np.asarray(arr)
        mean = arr.mean(0)
        ci   = 1.96*arr.std(0, ddof=1)/np.sqrt(arr.shape[0])
        return mean, ci

    rmse_arr   = np.vstack(rmse_sig)
    pct_arr    = np.vstack(pct_sig)
    r2_arr     = np.vstack(r2_sig)
    bias_arr   = np.vstack(bias_sig)
    sigma_arr  = np.vstack(sigma_sig)

    rmse_mean,  rmse_ci  = mean_CI(rmse_arr)
    pct_mean,   pct_ci   = mean_CI(pct_arr)
    r2_mean,    r2_ci    = mean_CI(r2_arr)
    bias_mean,  bias_ci  = mean_CI(bias_arr)
    loa_lower   = bias_mean - 1.96*sigma_arr.mean(0)
    loa_upper   = bias_mean + 1.96*sigma_arr.mean(0)

    # ───────────────────────── save pickle ──────────────────────
    results = dict(
        # per-signal arrays (B × D)
        mse_per_signal       = rmse_arr**2,
        rmse_per_signal      = rmse_arr,
        pct_per_signal       = pct_arr,
        r2_per_signal        = r2_arr,
        bias_per_signal      = bias_arr,
        sigma_per_signal     = sigma_arr,

        # per-trajectory vectors
        mse_per_trajectory   = np.array(mse_traj),
        rmse_per_trajectory  = np.array(rmse_traj),
        r2_per_trajectory    = np.array(r2_traj),
        pct_per_traj_flat    = np.array(pct_flat),
        pct_per_traj_ps      = np.array(pct_ps),
        dtw_per_trajectory   = np.array(dtw_traj),
        inference_times_s    = np.array(t_infer),

        # aggregated (table-ready)
        rmse_mean            = rmse_mean,   rmse_ci = rmse_ci,
        pct_mean             = pct_mean,    pct_ci  = pct_ci,
        r2_mean              = r2_mean,     r2_ci   = r2_ci,
        bias_mean            = bias_mean,   bias_ci = bias_ci,
        loa_lower            = loa_lower,   loa_upper = loa_upper,

        # misc
        cumulative_pct_per_traj = np.array(cum_pct),
        global_pct_rmse_flat    = pct_global_flat,
        true_per_trajectory     = X_true,
        pred_per_trajectory     = np.stack(preds),
        u_per_trajectory        = np.array(u_trajs),
        n_params                = n_params,
        global_r2_flat         = global_r2_flat
    )

    os.makedirs(exp_folder, exist_ok=True)
    out_path = os.path.join(exp_folder, "gru_postprocessing_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

print(f"Post-processing complete. Results saved to {out_path}")