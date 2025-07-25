"""
notebook_utils.py

A collection of helper functions and constants for time‐series analysis,
Koopman network modeling, and visualization in Jupyter notebooks.  

Includes:
  - Core utilities for file handling and confidence‐interval computation.
  - Data & array operations (e.g., Hankel embedding, moving average).
  - Spectral methods: FFT‐based mode estimation and Dynamic Mode Decomposition.
  - PCA on Hankel‐embedded data to assess dimensionality.
  - Functions to load and preprocess model checkpoints, weights, and test data.
  - Activation functions (ReLU, sigmoid) and network forward passes.
  - Plotting routines for signals, control inputs, loss curves, and DMD spectra.
  - Label mappings for LaTeX‐formatted plot annotations.
"""

# Core utilities
import glob
import os
import pickle
import re
import sys

# Data & arrays
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Scientific computing
from scipy.linalg import eig, svd
from scipy.signal import find_peaks

# Machine learning
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

# Plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# Koopman networks
from cardiokoop.network.networkarch import KoopmanNet, KoopmanNetControl_v2

LABEL_MAP = {
    "Past_lv": r"$P^{*}_{\mathrm{lv}}$ [mmHg]",
    "Vast_lv": r"$V^{*}_{\mathrm{lv}}$ [mL]",
    "LVM": r"$LVM$ [g]",
    "Vm": r"$V_{\mathrm{m}}$ [mL]",
    "c3": r"$c_{3}$ [a.u.]",
    "c2": r"$c_{2}$ [a.u.]",
    "c1": r"$c_{1}$ [a.u.]",
    "c0": r"$c_{0}$ [mmHg]",
    "alpha": r"$\alpha$ [mmHg]",
    "beta": r"$\beta$ [1/mL]",
    "Vusv_step": r"$V_{\mathrm{usv,step}}$ [mL]"
}


def ci95(x):
    """Calculate the 95% confidence interval for the mean of x."""
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))

def coloured_pred(ax, y_t, y_p, cmap, norm):
    """Plot y_p coloured by |y_p - y_t|, per‐segment."""
    err  = np.abs(y_p - y_t)
    x    = np.arange(len(y_t))
    pts  = np.column_stack([x, y_p])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.6)
    lc.set_array(err[:-1])
    ax.add_collection(lc)
    ax.set_ylim(
        y_p.min() - 0.1 * np.abs(y_p).max(),
        y_p.max() + 0.1 * np.abs(y_p).max()
    )

def estimate_modes_fft(x, delta_t, threshold=1e-2, min_separation=1):
    """
    Estimate dominant frequencies from multivariate time series via FFT.

    Parameters
    ----------
    x : (T, n) array
        Input data (T samples, n variables).
    delta_t : float
        Time step between samples.
    threshold : float, optional
        Fraction of max power for peak detection.
    min_separation : int, optional
        Minimum bins between peaks.

    Returns
    -------
    freqs : ndarray
        FFT frequency bins.
    avg_psd : ndarray
        Mean power spectral density.
    peaks : ndarray
        Indices of detected peaks.
    peak_freqs : ndarray
        Frequencies at the peaks.
    """
    T, n = x.shape
    freqs = np.fft.rfftfreq(T, d=delta_t)
    psd = np.abs(np.fft.rfft(x, axis=0))**2
    avg_psd = psd.mean(axis=1)
    peaks, _ = find_peaks(
        avg_psd,
        height=threshold * np.max(avg_psd),
        distance=min_separation
    )
    return freqs, avg_psd, peaks, freqs[peaks]

def estimate_modes_pca_hankel(x, window=50, var_explained=0.95):
    """
    Compute cumulative variance from PCA on Hankel-embedded data and
    estimate the number of dominant modes.

    Parameters
    ----------
    x : array-like, shape (T, n)
        Time series data (T samples, n variables).
    window : int, optional
        Hankel window size (default=50).
    var_explained : float, optional
        Target cumulative variance ratio (default=0.95).

    Returns
    -------
    cumvar : ndarray
        Cumulative explained variance ratios from PCA.
    n_modes : int
        Estimated number of modes (half the components needed to reach var_explained).
    """
    T, n = x.shape
    H = np.concatenate(
        [sliding_window_view(x[:, i], window, axis=0).squeeze().reshape(-1, window)
         for i in range(n)],
        axis=0
    )
    pca = PCA(n_components=min(H.shape))
    pca.fit(H)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = np.searchsorted(cumvar, var_explained) + 1
    return cumvar, k // 2

def average_pca_hankel_curves(trajectories, window=100, var_explained=0.95):
    """
    Compute the mean cumulative variance curve and average PCA mode count 
    from Hankel-embedded trajectories.

    Parameters
    ----------
    trajectories : list of (T, n) arrays
        Time series trajectories to analyze.
    window : int, optional
        Hankel window size (default=100).
    var_explained : float, optional
        Target cumulative variance ratio per trajectory (default=0.95).

    Returns
    -------
    mean_cumvar : ndarray
        Mean cumulative explained variance across all trajectories.
    mean_p : int
        Average number of PCA modes needed to reach var_explained (full Hankel rank).
    """
    cumvars = []
    p_components = []
    for traj in trajectories:
        cumvar, p_pca = estimate_modes_pca_hankel(traj, window=window, var_explained=var_explained)
        cumvars.append(cumvar)
        p_components.append(2 * p_pca)  # full rank in Hankel

    maxlen = max(len(c) for c in cumvars)
    cumvars_padded = [np.pad(c, (0, maxlen - len(c)), constant_values=c[-1]) for c in cumvars]
    mean_cumvar = np.mean(np.stack(cumvars_padded), axis=0)
    mean_p = int(np.round(np.mean(p_components)))
    return mean_cumvar, mean_p
    return mean_cumvar, mean_p

def estimate_modes_dmd(x, delta_t, r=None, unit_circle_tol=1e-2):
    """
    Estimate DMD eigenvalues, growth/decay rates, and mode count.

    Parameters
    ----------
    x : (T, n) array
        Time series data (T samples, n variables).
    delta_t : float
        Time step between samples.
    r : int, optional
        Truncation rank for SVD (default: full rank).
    unit_circle_tol : float, optional
        Tolerance for selecting unit‐circle modes (default=1e-2).

    Returns
    -------
    lam : ndarray
        DMD eigenvalues.
    mu : ndarray
        Continuous-time eigenvalues (log(lambda)/delta_t).
    n_modes : int
        Number of oscillatory modes (pairs of unit-circle eigenvalues).
    """
    X, Y = x[:-1].T, x[1:].T
    U, S, Vh = svd(X, full_matrices=False)
    if r:
        U, S, Vh = U[:, :r], S[:r], Vh[:r]
    A_tilde = U.T @ Y @ Vh.T @ np.diag(1.0 / S)
    lam, _ = eig(A_tilde)
    mu = np.log(lam) / delta_t
    mask = (np.abs(np.abs(lam) - 1.0) < unit_circle_tol) & (np.abs(np.angle(lam)) > 1e-6)
    return lam, mu, np.sum(mask) // 2

def colored_pred(ax, y_t, y_p, cmap, norm):
    """Plot prediction colored by |error| at each time-step."""
    err  = np.abs(y_p - y_t)
    x    = np.arange(len(y_t))
    pts  = np.column_stack([x, y_p])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.6, linestyle='--')
    lc.set_array(err[:-1])
    ax.add_collection(lc)
    ax.set_ylim(
        y_p.min() - 0.1 * np.abs(y_p).max(),
        y_p.max() + 0.1 * np.abs(y_p).max()
    )

def plot_signal_panel(ax, gt, pred, cmap, ylabel=None, show_legend=False):
    """
    Plot ground-truth vs. prediction with error-based coloring and colorbar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    gt : array-like
        Ground-truth signal.
    pred : array-like
        Predicted signal.
    cmap : Colormap
        Colormap for encoding absolute error.
    ylabel : str, optional
        Y-axis label.
    show_legend : bool, optional
        Whether to display the legend.
    """
    ax.plot(gt, color='k', lw=1.6, label='GroundTruth')
    err = np.abs(pred - gt)
    norm = colors.Normalize(vmin=0, vmax=err.max() + 1e-12)
    colored_pred(ax, gt, pred, cmap, norm)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis='both', labelsize=22)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.08)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, cax=cax, orientation='vertical')
    cax.tick_params(labelsize=22)
    cax.set_ylabel('|err|', rotation=90, labelpad=15, fontsize=24)

    if show_legend:
        ax.legend(loc='upper left', fontsize=18, frameon=False, bbox_to_anchor=(0, 1.05))

def plot_control_panel(ax, u):
    """
    Plot a control input time series.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    u : array-like
        Control input values over time.
    """
    ax.plot(u, color='k', lw=1.2)
    ax.set_ylabel('u(t) [mL]', fontsize=24)
    ax.set_xlabel('Time Step', fontsize=24)
    ax.tick_params(axis='both', labelsize=22)

def latex_mode_label(mode):
    """
    Convert a mode identifier string into a LaTeX-formatted label.

    Parameters
    ----------
    mode : str
        Mode name, e.g., 'CP1_a', 'Real2', etc.

    Returns
    -------
    str
        LaTeX-formatted mode label, or the original string if unrecognized.
    """
    if "CP" in mode:
        num, part = mode.replace("CP", "").split("_")
        return rf"$\mathrm{{CP_{{{num}}}^{{{part}}}}}$"
    elif "Real" in mode:
        index = mode.replace("Real", "")
        return rf"$\mathrm{{Real_{{{index}}}}}$"
    return mode

def get_latest_errors(path, suffix_regex=r"_(\d+)_error\.csv"):
    """
    Load the most recent error file matching a glob pattern.

    Parameters
    ----------
    path : str or Path
        Glob pattern for error CSV files.
    suffix_regex : str, optional
        Regex to extract the numeric suffix used for sorting (default=r"_(\\d+)_error\\.csv").

    Returns
    -------
    ndarray
        Contents of the latest error CSV as a NumPy array.
    """
    files = glob.glob(str(path))
    files.sort(key=lambda x: int(re.search(suffix_regex, x).group(1)), reverse=True)
    return np.loadtxt(files[0], delimiter=',')

def load_latest_ckpt(path, suffix_regex=r"_(\d+)_model\.ckpt"):
    """
    Load the most recent model checkpoint matching a glob pattern and reconstruct the Koopman network.

    Parameters
    ----------
    path : str or Path
        Glob pattern for checkpoint files.
    suffix_regex : str, optional
        Regex to extract the numeric suffix for sorting (default=r"_(\\d+)_model\\.ckpt").

    Returns
    -------
    params : dict
        Checkpoint parameters dictionary.
    W : list
        List of Koopman network weight matrices.
    b : list
        List of Koopman network bias vectors.
    model : torch.nn.Module
        Reconstructed KoopmanNet model instance.
    """
    files = glob.glob(str(path))
    files.sort(key=lambda x: int(re.search(suffix_regex, x).group(1)), reverse=True)
    model_ckpt = files[0]
    ckpt = torch.load(model_ckpt, map_location='cpu', weights_only=False)
    params = ckpt['params']
    W, b, model = load_weights_koopman(
        model_ckpt,
        len(params['widths']) - 1,
        len(params['widths_omega_real']) - 1,
        params['num_real'],
        params['num_complex_pairs']
    )
    return params, W, b, model

def load_test_data(path, params):
    """
    Load and preprocess test data for Koopman modeling.

    Parameters
    ----------
    path : str or Path
        Base path prefix for test data CSV files (without suffix).
    params : dict
        Parameter dictionary containing 'len_time' and settings for data stacking.

    Returns
    -------
    n_traj : int
        Number of trajectories in the test dataset.
    Xk : ndarray
        First trajectory’s state data after stacking and squeezing.
    """
    X = np.loadtxt(f"{path}_test1_x.csv", delimiter=',')
    U = np.loadtxt(f"{path}_test1_u.csv", delimiter=',')
    U = (U - U.mean(0)) / U.std(0)
    max_shifts = num_shifts_in_stack(params)
    X_stacked, U_stacked = stack_data_with_control(X, U, max_shifts, params['len_time'])
    n_traj = X_stacked.shape[1] // (params['len_time'] - max_shifts)
    # Encode first trajectory
    Xk = np.squeeze(X_stacked[0])
    return n_traj, Xk

def omega_net_apply_one(ycoords, W, b, name, num_weights, act_type):
    """
    Apply the omega network to a single set of coordinates.

    Parameters
    ----------
    ycoords : ndarray, shape (T, d)
        Coordinates; if d==2, squared-norm is used as input.
    W : list
        Network weight matrices.
    b : list
        Network bias vectors.
    name : str
        Layer/name prefix for parameter lookup.
    num_weights : int
        Number of weight layers to apply.
    act_type : str
        Activation function type.

    Returns
    -------
    ndarray
        Network output for each time step.
    """
    if ycoords.shape[1] == 2:
        input = np.sum(np.square(ycoords), axis=1)
    else:
        input = ycoords

    # ensure column vector
    if input.ndim == 1:
        input = input[:, np.newaxis]

    return encoder_apply(input, W, b, name, num_weights, act_type)

def load_postprocessing_results(path):
    """
    Load post-processing results from a pickle file.

    Parameters
    ----------
    path : str or Path
        File path to the pickle file containing results.

    Returns
    -------
    object
        Unpickled results object.
    """
    with open(path, "rb") as f:
        results = pickle.load(f)
    return results

def reshape_latent_trajs(yk, n_traj, params):
    """
    Reshape flat latent trajectories into per-trajectory time series.

    Parameters
    ----------
    yk : array-like
        Flattened latent values of shape (n_traj * (len_time - num_shifts_middle) * num_evals).
    n_traj : int
        Number of separate trajectories.
    params : dict
        Dictionary containing:
        - 'len_time': total time steps per trajectory before shifting
        - 'num_shifts_middle': number of shifts applied in Hankel embedding
        - 'num_evals': number of latent dimensions (eigenvalues)

    Returns
    -------
    list of ndarray
        List of length `n_traj`, each an array of shape
        (`len_time - num_shifts_middle`, `num_evals`) for one trajectory.
    """
    len_time, shifts = params["len_time"], params["num_shifts_middle"]
    latent_seq_len = len_time - shifts
    yk_all = np.asarray(yk).reshape(n_traj, latent_seq_len, params['num_evals'])
    return [yk_all[i] for i in range(n_traj)]

from itertools import cycle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_errors_hankel_dmd(errors, data_sources, params, results_base_path, fig_name="errors_hankel_dmd"):
    """
    Plot and save three-panel figure showing:
      1. Log10 training and validation loss over epochs.
      2. PCA–Hankel cumulative variance curves with chosen component counts.
      3. DMD eigenvalue spectrum relative to the unit circle.
    """
    # Custom palette
    HEX_PALETTE = ["#dda251", "#5a286b", "#646464", "#ac4484"]

    # Color mapping for data_sources (PCA/DMD panels)
    labels_list = list(data_sources.keys())
    color_cycle = cycle(HEX_PALETTE)
    colors_ = {label: next(color_cycle) for label in labels_list}

    loss_colors = HEX_PALETTE + HEX_PALETTE  # ensure at least 2
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

    # ---------------------- (1) Loss plot ------------------------------
    errors = errors.copy()
    errors[:, :2] = np.log10(errors[:, :2])
    x = np.arange(errors.shape[0])
    for j, label in enumerate(["Training", "Validation"]):
        axes[0].plot(
            x, errors[:, j],
            color=loss_colors[j],
            alpha=0.8,
            label=label,
            linestyle='None',
            marker='o',
            markersize=5
        )
    axes[0].set(xlabel="Epoch", ylabel=r"$\log_{10}(\mathrm{Loss})$")
    axes[0].tick_params(labelsize=14)
    axes[0].legend(fontsize=14)
    axes[0].grid(False)

    # ------------------- (2) PCA–Hankel plot --------------------------
    for label, traj in data_sources.items():
        cumvar, p_pca = average_pca_hankel_curves(traj, window=100, var_explained=0.95)
        axes[1].plot(
            range(1, len(cumvar) + 1),
            cumvar,
            label=f"{label} ($PC$ = {p_pca})",
            color=colors_[label]
        )
        axes[1].axvline(p_pca, color=colors_[label], linestyle='--', lw=1)
    axes[1].axhline(0.95, color="#bbbbbb", linestyle="--", lw=1)
    axes[1].set(xlabel="Component", ylabel="Cumulative variance")
    axes[1].tick_params(labelsize=14)
    axes[1].legend(fontsize=14)
    axes[1].grid(False)

    # ---------------- (3) DMD spectrum plot ---------------------------
    theta = np.linspace(0, 2 * np.pi, 300)
    axes[2].set_facecolor('whitesmoke')
    circle = Circle((0, 0), 1, color='#dda251', alpha=0.08, zorder=0)  # light shade from custom palette
    axes[2].add_patch(circle)
    axes[2].plot(np.cos(theta), np.sin(theta), color="#646464", linestyle='--', lw=1, label="Unit circle")

    all_lams = []
    for label, traj in data_sources.items():
        x_concat = np.concatenate(traj, axis=0)
        lam, mu, p_dmd = estimate_modes_dmd(
            x_concat,
            params['delta_t'],
            r=20,
            unit_circle_tol=0.05
        )
        all_lams.append(lam)
        axes[2].scatter(
            lam.real, lam.imag,
            color=colors_[label],
            label=label,
            marker='x' if label == "Prediction" else 'o',
            alpha=0.95, s=25
        )

    lam_all = np.concatenate(all_lams) if len(all_lams) else np.array([0 + 0j])
    xr = max(1.2, 1.1 * np.max(np.abs(lam_all.real)))  # at least 1.2
    yr = max(1.2, 1.1 * np.max(np.abs(lam_all.imag)))
    axes[2].set_xlim(-xr, xr)
    axes[2].set_ylim(-yr, yr)

    # Labels inside/outside the unit circle
    axes[2].text(
        0, 0, "stable\noscillations",
        ha='center', va='center',
        color="#646464", fontsize=16, alpha=0.9, zorder=-1
    )
    axes[2].text(
        xr * 1.2, -yr * 0.75, "unstable",
        ha='right', va='top',
        color="#ac4484", fontsize=16, alpha=0.95
    )

    axes[2].set(xlabel=r"Re$(\lambda)$", ylabel=r"Im$(\lambda)$")
    axes[2].tick_params(labelsize=14)
    axes[2].axis("equal")
    axes[2].legend(fontsize=14, loc='upper left')
    axes[2].grid(False)

    # ------------------------------------------------------------------
    plt.tight_layout()
    (results_base_path / "figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(results_base_path / "figures" / f"{fig_name}.svg", format='svg')
    plt.show()


def omega_net_apply(ycoords, W, b, num_real, num_complex_pairs, _num_weights, act_type='relu'):
    """
    Apply complex-pair and real omega networks to input coordinates.

    Parameters
    ----------
    ycoords : ndarray, shape (T, d)
        Input coordinates (real and complex-pair concatenated).
    W : dict
        Weight matrices keyed by layer names.
    b : dict
        Bias vectors keyed by layer names.
    num_real : int
        Number of real-valued omega networks.
    num_complex_pairs : int
        Number of complex-pair omega networks.
    _num_weights : int
        (Unused) placeholder for total weight count.
    act_type : str, optional
        Activation function name (default='relu').

    Returns
    -------
    list of ndarray
        Omega-network outputs for each mode.
    """
    omegas = []
    # complex-pair networks
    for j in range(num_complex_pairs):
        prefix = f'OC{j+1}_'
        num_w = sum(1 for k in W if k.startswith(f'W{prefix}'))
        ind = 2 * j
        omegas.append(
            omega_net_apply_one(ycoords[:, ind:ind + 2], W, b, prefix, num_w, act_type)
        )
    # real networks
    for j in range(num_real):
        prefix = f'OR{j+1}_'
        num_w = sum(1 for k in W if k.startswith(f'W{prefix}'))
        ind = 2 * num_complex_pairs + j
        inp = ycoords[:, ind] if ycoords.ndim > 1 else ycoords[:, np.newaxis]
        omegas.append(
            omega_net_apply_one(inp, W, b, prefix, num_w, act_type)
        )
    return omegas


def varying_multiply(y, omegas, delta_t, num_real, num_complex_pairs):
    """
    Multiply latent coords by time-varying complex and real eigenvalues.

    Parameters
    ----------
    y : ndarray, shape (T, k)
        Latent coordinates (real and complex parts concatenated).
    omegas : list of ndarray
        Outputs from omega_net_apply for each mode.
    delta_t : float
        Time step size.
    num_real : int
        Number of real modes.
    num_complex_pairs : int
        Number of complex-conjugate mode pairs.

    Returns
    -------
    ndarray
        Updated latent coordinates after applying exp(ω Δt) multipliers.
    """
    k = y.shape[1]

    complex_list = []
    for j in range(num_complex_pairs):
        ind = 2 * j
        ystack = np.stack([y[:, ind:ind+2], y[:, ind:ind+2]], axis=2)
        L_stack = FormComplexConjugateBlock(omegas[j], delta_t)
        elmtwise_prod = ystack * L_stack
        complex_list.append(np.sum(elmtwise_prod, axis=1))
    if complex_list:
        complex_part = np.concatenate(complex_list, axis=1)

    real_list = []
    for j in range(num_real):
        ind = 2 * num_complex_pairs + j
        temp_y = y[:, ind]
        if temp_y.ndim == 1:
            temp_y = temp_y[:, np.newaxis]
        evals = np.exp(omegas[num_complex_pairs + j] * delta_t)
        real_list.append(temp_y * evals)
    if real_list:
        real_part = np.concatenate(real_list, axis=1)

    if complex_list and real_list:
        return np.concatenate([complex_part, real_part], axis=1)
    elif complex_list:
        return complex_part
    else:
        return real_part

def num_shifts_in_stack(params):
    """
    Determine the maximum shift count for data stacking.

    Parameters
    ----------
    params : dict
        Must include 'num_shifts', 'shifts', 'num_shifts_middle', 'shifts_middle'.

    Returns
    -------
    int
        Maximum number of shifts to use when stacking data.
    """
    max_shifts_to_stack = 1
    if params['num_shifts']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts']))
    if params['num_shifts_middle']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts_middle']))
    return max_shifts_to_stack

def stack_data_with_control(x_data, u_data, num_shifts, len_time):
    """
    Create time-shifted stacks of state and control trajectories.

    Parameters
    ----------
    x_data : ndarray, shape (num_traj*len_time, n_states)
        Flat state data for all trajectories.
    u_data : ndarray, shape (num_traj*len_time, n_controls)
        Flat control input data for all trajectories.
    num_shifts : int
        Number of past shifts to include (window size minus one).
    len_time : int
        Number of time steps per trajectory.

    Returns
    -------
    Xs : ndarray, shape (num_shifts+1, num_traj*(len_time-num_shifts), n_states)
        Stacked state arrays for each shift.
    Us : ndarray, shape (num_shifts, num_traj*(len_time-num_shifts), n_controls)
        Stacked control arrays for each shift (excluding the zero shift).
    """
    num_traj = x_data.shape[0] // len_time
    X = x_data.reshape(num_traj, len_time, -1)
    U = u_data.reshape(num_traj, len_time, -1)
    newlen = len_time - num_shifts

    Xs = np.stack([X[:, j : j + newlen, :] for j in range(num_shifts + 1)], axis=0)
    Us = np.stack([U[:, j : j + newlen, :] for j in range(num_shifts)], axis=0)

    Xs = Xs.reshape(num_shifts + 1, num_traj * newlen, X.shape[2])
    Us = Us.reshape(num_shifts,     num_traj * newlen, U.shape[2])
    return Xs, Us

def relu(x):
    """
    Apply the rectified linear unit activation element-wise.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    array-like
        Same shape as `x`, with negative entries set to zero.
    """
    return np.maximum(0, x)

def sigmoid(x):
    """
    Compute the sigmoid activation element-wise.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    array-like
        Same shape as `x`, with values in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))

def encoder_apply(x, weights, biases, name, num_weights, act_type='relu'):
    """
    Apply a feedforward encoder network to inputs.

    Parameters
    ----------
    x : array-like, shape (T, d)
        Input data.
    weights : dict
        Weight matrices keyed by 'W{name}{layer}'.
    biases : dict
        Bias vectors keyed by 'b{name}{layer}'.
    name : str
        Prefix for layer keys in weights/biases.
    num_weights : int
        Total number of layers.
    act_type : {'relu', 'sigmoid'}, optional
        Activation function for hidden layers.

    Returns
    -------
    ndarray
        Network output after the final linear layer.
    """
    prev_layer = x.copy()

    for i in range(num_weights - 1):
        h1 = (prev_layer @ weights[f'W{name}{i+1}'] +
              biases[f'b{name}{i+1}'])
        h1 = sigmoid(h1) if act_type == 'sigmoid' else relu(h1)
        prev_layer = h1.copy()

    final = (prev_layer @ weights[f'W{name}{num_weights}'] +
             biases[f'b{name}{num_weights}'])
    return final

def moving_average(x, window=10):
    """
    Compute the simple moving average of a 1D sequence.

    Parameters
    ----------
    x : array-like
        Input sequence.
    window : int, optional
        Size of the moving window (default=10).

    Returns
    -------
    ndarray
        Moving-averaged values, length = len(x) - window + 1.
    """
    return np.convolve(x, np.ones(window) / window, mode='valid')

def load_weights(fname, numWeights, type='E'):
    """
    Load network weights and biases from CSV files based on checkpoint filename.

    Parameters
    ----------
    fname : str
        Checkpoint file path (with .ckpt or .pkl extension or _model suffix).
    numWeights : int
        Number of layers to load.
    type : str, optional
        Layer type prefix for files (default='E').

    Returns
    -------
    W : dict
        Weight matrices keyed by 'W{type}{layer}'.
    b : dict
        Bias vectors keyed by 'b{type}{layer}'.
    """
    W = {}
    b = {}
    if fname.endswith("model.ckpt"):
        root = fname[:-len("model.ckpt")]
    elif fname.endswith("model.pkl"):
        root = fname[:-len("model.pkl")]
    else:
        root, _ = os.path.splitext(fname)
        if root.endswith("_model"):
            root = root[:-len("_model")]
    lastSize = None
    for j in range(numWeights):
        path_W = f"{root}W{type}{j+1}.csv"
        path_b = f"{root}b{type}{j+1}.csv"
        W1 = np.matrix(np.genfromtxt(path_W, delimiter=','))
        b1 = np.matrix(np.genfromtxt(path_b, delimiter=','))
        if lastSize is not None and W1.shape[0] != lastSize:
            if W1.shape[0] == 1 and W1.shape[1] == lastSize:
                W1 = W1.T
            else:
                print(f"error: sizes {lastSize} and {W1.shape}")
        lastSize = W1.shape[1]
        W[f"W{type}{j+1}"] = W1
        b[f"b{type}{j+1}"] = b1
    return W, b

def load_weights_koopman(fname, numWeights, numWeightsOmega, num_real, num_complex_pairs):
    """
    Load Koopman network weights/biases (and model) from checkpoint or CSVs.

    Parameters
    ----------
    fname : str
        Checkpoint (.ckpt) or file prefix for CSV weights.
    numWeights : int
        Number of encoder/decoder layers.
    numWeightsOmega : int
        Number of layers per omega network.
    num_real : int
        Number of real-valued omega networks.
    num_complex_pairs : int
        Number of complex-pair omega networks.

    Returns
    -------
    W : dict
        Weight matrices.
    b : dict
        Bias vectors.
    model : torch.nn.Module or None
        Loaded KoopmanNet model (None if loading from CSV).
    """
    # checkpoint-based loading
    if fname.endswith('.ckpt'):
        ckpt = torch.load(fname, map_location='cpu', weights_only=False)
        params = ckpt['params']
        model_cls = KoopmanNetControl_v2 if params.get('use_control', False) else KoopmanNet
        model = model_cls(params)
        sd = ckpt.get('state_dict', {})
        key_w0 = 'control_net.net.0.weight'
        if key_w0 in sd:
            w0 = sd[key_w0]
            in_ckpt, out_ckpt = w0.shape[1], w0.shape[0]
            lin0 = model.control_net.net[0]
            if getattr(lin0, 'in_features', None) != in_ckpt or getattr(lin0, 'out_features', None) != out_ckpt:
                model.control_net.net[0] = nn.Linear(in_ckpt, out_ckpt)
        model.load_state_dict(sd)
        model.cpu()

        W, b = {}, {}
        # encoder
        enc_layers = [m for m in model.encoder.net if isinstance(m, nn.Linear)]
        for j, layer in enumerate(enc_layers):
            W[f'WE{j+1}'] = np.matrix(layer.weight.detach().numpy().T)
            b[f'bE{j+1}'] = np.matrix(layer.bias.detach().numpy())
        # decoder
        dec_layers = [m for m in model.decoder.net if isinstance(m, nn.Linear)]
        for j, layer in enumerate(dec_layers):
            W[f'WD{j+1}'] = np.matrix(layer.weight.detach().numpy().T)
            b[f'bD{j+1}'] = np.matrix(layer.bias.detach().numpy())
        # omega nets
        for idx, om in enumerate(model.omega_nets):
            layers = [m for m in om.net if isinstance(m, nn.Linear)]
            prefix = (f'OC{idx+1}_' if idx < num_complex_pairs else f'OR{idx+1-num_complex_pairs}_')
            for j, layer in enumerate(layers):
                W[f'W{prefix}{j+1}'] = np.matrix(layer.weight.detach().numpy().T)
                b[f'b{prefix}{j+1}'] = np.matrix(layer.bias.detach().numpy())
        # control net
        if hasattr(model, 'control_net'):
            ctrl_layers = [m for m in model.control_net.net if isinstance(m, nn.Linear)]
            for j, layer in enumerate(ctrl_layers):
                W[f'WC{j+1}'] = np.matrix(layer.weight.detach().numpy().T)
                b[f'bC{j+1}'] = np.matrix(layer.bias.detach().numpy())
        return W, b, model

    # CSV-based loading
    d = int((numWeights - 1) / 2)
    weights, biases = load_weights(fname, d, 'E')
    Wd, Bd = load_weights(fname, d, 'D')
    weights.update(Wd)
    biases.update(Bd)
    for j in range(num_complex_pairs):
        Wj, Bj = load_weights(fname, numWeightsOmega, f'OC{j+1}_')
        weights.update(Wj)
        biases.update(Bj)
    for j in range(num_real):
        Wj, Bj = load_weights(fname, numWeightsOmega, f'OR{j+1}_')
        weights.update(Wj)
        biases.update(Bj)
    return weights, biases, None
