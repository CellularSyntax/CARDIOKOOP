#!/usr/bin/env python3
"""
Shared utilities for the *revision2* experiments (journal Array, minor revision 2).

This module centralises everything the revision-2 scripts need so that every new
number is produced with the **exact same metric conventions** the existing
Tables 3/4/5 use, on the **same seed-42 test split**, against the **frozen final
Koopman checkpoint** (no retraining of the main model anywhere).

Key conventions (verified against the committed post-processing pickles):

* Split:  the manuscript baselines (LSTM/GRU/BiLSTM/MLP/AR20) and the noise
  harness all evaluate on ``*_test1_*`` (the seed-42 held-out test split, B=50
  trajectories, T=1500 steps => 1499-step forecast horizon).  The Koopman table
  row was generated on ``val1`` in revision-1; the two splits are within
  ~0.24 pp %RMSE of each other (val 17.30 / test 17.54).  All *new* results here
  use ``test1`` so they are directly comparable to the baseline columns.
* Metric functions are copied verbatim from
  ``src/cardiokoop/postprocessing/postprocess_lstm.py`` /
  ``postprocess_koopman.py`` and the figure notebook, so the produced result
  dicts are byte-for-byte comparable to the existing baseline pickles.

Nothing in this module trains or modifies the frozen Koopman model.
"""
import os
import glob
import pickle

import numpy as np
import torch
from sklearn.metrics import r2_score as _sk_r2

# ───────────────────────── paths ─────────────────────────
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR     = os.path.join(REPO_ROOT, "src")
DATA_DIR    = os.path.join(REPO_ROOT, "data")
RESULT_DIR  = os.path.join(REPO_ROOT, "results")
REV2_DIR    = os.path.join(RESULT_DIR, "revision2")
FIG_DIR     = os.path.join(RESULT_DIR, "figures")
KOOP_DIR    = os.path.join(RESULT_DIR, "koopman")
CKPT_DIR    = os.path.join(REPO_ROOT, "checkpoints")

os.makedirs(REV2_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DATA_NAME = "csv_data_500_12sigs"
T_STEPS   = 1500          # steps per trajectory (=> 1499-step horizon)
SEED      = 42

# 12 hemodynamic signals (order matches the dataset columns / manuscript figures)
YLAB  = ['$P_{ra}$', '$V_{ra}$', '$P_{rv}$', '$V_{rv}$', '$P_{vp}$', '$P_{la}$',
         '$V_{la}$', '$P_{lv}$', '$V_{lv}$', '$Q_{i,lv}$', '$Q_{o,lv}$', '$P_{ao}$']
UNITS = ['mmHg', 'mL', 'mmHg', 'mL', 'mmHg', 'mmHg', 'mL', 'mmHg', 'mL',
         'mL/s', 'mL/s', 'mmHg']
SIG_NAMES = ['P_ra', 'V_ra', 'P_rv', 'V_rv', 'P_vp', 'P_la', 'V_la', 'P_lv',
             'V_lv', 'Q_i_lv', 'Q_o_lv', 'P_ao']

# canonical model / palette order used across the manuscript figures + the two
# new direct-linear baselines appended at the end (Task A)
MODELS_ALL   = ['Koopman', 'GRU', 'LSTM', 'BiLSTM', 'MLP', 'AR(20)', 'DLinear', 'NLinear']
PALETTE_HEX  = ["#dda251", "#5a286b", "#646464", "#ac4484", "#2e7ec0", "#4caf50",
                "#e15759", "#17becf"]
# Reduced roster for the (otherwise crowded) main-text Figure 5: the method, one
# RNN representative (GRU -- the strongest of GRU/LSTM/BiLSTM), BOTH reviewer-
# requested linear baselines (DLinear/NLinear), the naive MLP anchor, and the
# classical AR(20) anchor.  The full 8-model comparison lives in the supplement
# (Figure S9).  Colours are looked up from PALETTE_HEX by name, so a model keeps
# the same colour whether it appears in the 6-model or 8-model figure.
MODELS_MAIN  = ['Koopman', 'GRU', 'DLinear', 'NLinear', 'MLP', 'AR(20)']

if SRC_DIR not in os.sys.path:
    os.sys.path.insert(0, SRC_DIR)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ───────────────────── normalisation stats ───────────────────
def load_norm_stats():
    mean = np.load(os.path.join(DATA_DIR, "normalization_mean.npy"))
    std  = np.load(os.path.join(DATA_DIR, "normalization_std.npy"))
    return mean, std


def load_split(split="test"):
    """
    Load a dataset split.

    Returns
    -------
    X_norm : (B, T, D) z-scored states
    U_norm : (B, T, 1) z-scored control (as used by the Koopman control net)
    U_phys : (B, T, 1) raw control (V_usv,step, physical units)
    X_phys : (B, T, D) de-normalised states (physical units)
    """
    x = np.loadtxt(os.path.join(DATA_DIR, f"{DATA_NAME}_{split}1_x.csv"), delimiter=",")
    u = np.loadtxt(os.path.join(DATA_DIR, f"{DATA_NAME}_{split}1_u.csv"), delimiter=",")
    mean, std = load_norm_stats()
    D = x.shape[1]
    B = x.shape[0] // T_STEPS
    X_norm = x.reshape(B, T_STEPS, D)
    U_phys = u.reshape(B, T_STEPS, 1)
    # control z-scored exactly as in the post-processing / noise harnesses
    U_norm = ((u - u.mean(0)) / (u.std(0) + 1e-8)).reshape(B, T_STEPS, 1)
    X_phys = X_norm * std + mean
    return X_norm, U_norm, U_phys, X_phys


# ───────────────────── metric functions (verbatim) ───────────────────
def pct_rmse_ps(true, pred):
    """Per-signal RMSE / PTP, averaged over signals.  (== Table-3/4 %RMSE)"""
    rmse = np.sqrt(((true - pred) ** 2).mean(0))          # (D,)
    ptp  = np.ptp(true, axis=0) + 1e-8                     # (D,)
    return float((100.0 * rmse / ptp).mean())


def pct_rmse_flat(true, pred):
    """Global (flattened) %RMSE for one trajectory, ptp over the whole trajectory."""
    return float(100 * np.sqrt(((true - pred) ** 2).mean()) / (np.ptp(true) + 1e-8))


def r2_flat(true, pred):
    """Global flattened R^2 (all signals & steps pooled)."""
    t = true.ravel(); p = pred.ravel()
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum() + 1e-8
    return float(1.0 - ss_res / ss_tot)


def ci95(x):
    x = np.asarray(x)
    return float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x)))


def compute_results_dict(true_phys, pred_phys, u_phys, inference_times_s, n_params):
    """
    Build a post-processing results dict with EXACTLY the same keys / metric
    definitions as ``postprocess_lstm.py`` (and the Koopman post-processor), so
    the resulting pickle drops straight into the Table 3/4/5 + Figure 5 pipeline
    alongside the existing baseline pickles.

    Parameters
    ----------
    true_phys, pred_phys : (B, T, D) arrays in physical units
    u_phys               : (B, T, control_dim) raw control trajectories
    inference_times_s    : (B,) per-trajectory inference time [s]
    n_params             : int
    """
    B, T, D = true_phys.shape

    mse_sig = []; rmse_sig = []; pct_sig = []; r2_sig = []
    bias_sig = []; sigma_sig = []
    mse_traj = []; rmse_traj = []; r2_traj = []
    pct_flat = []; pct_ps = []; cum_pct = []

    for i in range(B):
        true = true_phys[i]; pred = pred_phys[i]
        err  = pred - true
        err2 = err ** 2

        mse_k  = err2.mean(0)
        rmse_k = np.sqrt(mse_k)
        pct_k  = 100 * rmse_k / (np.ptp(true, 0) + 1e-8)
        r2_k   = np.array([_sk_r2(true[:, j], pred[:, j]) for j in range(D)])
        bias_k = err.mean(0)
        sigma_k = err.std(0, ddof=1)

        mse_sig.append(mse_k);    rmse_sig.append(rmse_k)
        pct_sig.append(pct_k);    r2_sig.append(r2_k)
        bias_sig.append(bias_k);  sigma_sig.append(sigma_k)

        mse_traj.append(mse_k.mean()); rmse_traj.append(rmse_k.mean())
        r2_traj.append(r2_k.mean())
        pct_flat.append(pct_rmse_flat(true, pred))
        pct_ps.append(pct_k.mean())
        cum_pct.append(np.cumsum(100 * np.sqrt(err2.mean(1)) / (np.ptp(true) + 1e-8)))

    rmse_arr = np.vstack(rmse_sig); pct_arr = np.vstack(pct_sig)
    r2_arr   = np.vstack(r2_sig);   bias_arr = np.vstack(bias_sig)
    sigma_arr = np.vstack(sigma_sig)

    def _ci(a):
        return 1.96 * a.std(0, ddof=1) / np.sqrt(B)

    all_true = true_phys.reshape(-1, D)
    all_pred = pred_phys.reshape(-1, D)

    return dict(
        mse_per_signal      = rmse_arr ** 2,
        rmse_per_signal     = rmse_arr,
        pct_per_signal      = pct_arr,
        r2_per_signal       = r2_arr,
        bias_per_signal     = bias_arr,
        sigma_per_signal    = sigma_arr,

        mse_per_trajectory  = np.array(mse_traj),
        rmse_per_trajectory = np.array(rmse_traj),
        r2_per_trajectory   = np.array(r2_traj),
        pct_per_traj_flat   = np.array(pct_flat),
        pct_per_traj_ps     = np.array(pct_ps),
        inference_times_s   = np.asarray(inference_times_s),

        rmse_mean = rmse_arr.mean(0), rmse_ci = _ci(rmse_arr),
        pct_mean  = pct_arr.mean(0),  pct_ci  = _ci(pct_arr),
        r2_mean   = r2_arr.mean(0),   r2_ci   = _ci(r2_arr),
        bias_mean = bias_arr.mean(0),
        loa_lower = bias_arr.mean(0) - 1.96 * sigma_arr.mean(0),
        loa_upper = bias_arr.mean(0) + 1.96 * sigma_arr.mean(0),

        cumulative_pct_per_traj = np.array(cum_pct),
        global_pct_rmse_flat    = pct_rmse_ps(all_true, all_pred),
        global_r2_flat          = r2_flat(all_true, all_pred),
        true_per_trajectory     = true_phys,
        pred_per_trajectory     = pred_phys,
        u_per_trajectory        = np.asarray(u_phys),
        n_params                = int(n_params),
    )


# ───────────────────── frozen Koopman model ───────────────────
def find_koopman_ckpt():
    ckpts = glob.glob(os.path.join(KOOP_DIR, "*_model.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No Koopman checkpoint under {KOOP_DIR}")
    import re
    return max(ckpts, key=lambda f: int(re.search(r"_(\d+)_", f).group(1)))


def load_koopman(device=None):
    """Load the frozen final Koopman model + params (no retraining, ever)."""
    from cardiokoop.postprocessing.postprocess_utils import load_weights_koopman
    device = device or get_device()
    ckpt_path = find_koopman_ckpt()
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    W, b, model = load_weights_koopman(
        ckpt_path,
        len(params["widths"]) - 1,
        len(params["widths_omega_real"]) - 1,
        params["num_real"],
        params["num_complex_pairs"],
    )
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, params, n_params, ckpt_path


def koopman_advance(model, y, u_t, control_gain=None, control_act=torch.tanh):
    """
    One latent step of KoopmanNetControl_v2 with an *overridable* control gain
    (Task B) and control-net output activation (Task C).  Numerically identical
    to ``model._advance`` when ``control_gain=model.control_gain`` and
    ``control_act=torch.tanh`` (verified in task_b).
    """
    num_cp = model.num_complex_pairs
    dt     = model.delta_t
    g      = model.control_gain if control_gain is None else control_gain
    omegas = model.get_omegas(y)

    batch = y.size(0)
    y_complex = y[:, :2 * num_cp].view(batch, num_cp, 2)
    oc = torch.stack(omegas[:num_cp], dim=1)
    scale = torch.exp(torch.clamp(oc[:, :, 1], -5.0, 5.0) * dt)
    angle = oc[:, :, 0] * dt
    cos_a = scale * torch.cos(angle)
    sin_a = scale * torch.sin(angle)
    rot = torch.stack([
        torch.stack([cos_a, sin_a], dim=-1),
        torch.stack([-sin_a, cos_a], dim=-1),
    ], dim=-2)
    y_c = torch.matmul(y_complex.unsqueeze(-2), rot).squeeze(-2).reshape(batch, 2 * num_cp)

    y_real = y[:, 2 * num_cp:]
    orl = torch.stack(omegas[num_cp:], dim=1).squeeze(-1)
    y_r = y_real * torch.exp(torch.clamp(orl, -5.0, 5.0) * dt)

    B_u = g * control_act(model.control_net(u_t))
    return torch.cat([y_c, y_r], dim=1) + B_u


@torch.no_grad()
def koopman_rollout(model, x0_norm, u_seq_norm, control_gain=None,
                    control_act=torch.tanh, device=None):
    """
    Batched Koopman rollout in normalised space.

    x0_norm    : (B, D) initial state
    u_seq_norm : (B, T, 1) control sequence
    returns    : (B, T, D) normalised predictions
    """
    device = device or get_device()
    x0 = torch.as_tensor(x0_norm, dtype=torch.float32, device=device)
    u  = torch.as_tensor(u_seq_norm, dtype=torch.float32, device=device).permute(1, 0, 2)
    y = model.encoder(x0)
    outs = []
    for t in range(u.size(0)):
        outs.append(model.decoder(y).cpu().numpy())
        y = koopman_advance(model, y, u[t], control_gain=control_gain,
                            control_act=control_act)
    return np.stack(outs, axis=1)


# ───────────────────── io helpers ───────────────────
def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
