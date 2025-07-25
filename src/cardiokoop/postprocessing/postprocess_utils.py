"""
Shared utilities for notebook-based postprocessing and plotting.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from cardiokoop.network.networkarch import (
    KoopmanNet,
    KoopmanNetControl_v2
)

def omega_net_apply_one(ycoords, W, b, name, num_weights, act_type):
    if ycoords.shape[1] == 2:
        input = np.sum(np.square(ycoords), axis=1)
    else:
        input = ycoords

    # want input to be [?, 1]
    if len(input.shape) == 1:
        input = input[:, np.newaxis]

    return encoder_apply(input, W, b, name, num_weights, act_type)


def omega_net_apply(ycoords, W, b, num_real, num_complex_pairs, _num_weights, act_type='relu'):
    """
    Apply each omega network (complex-pair and real) to y-coordinates.
    Automatically detects layer counts per network from W keys.
    """
    omegas = []
    # complex-pair omega networks
    for j in range(num_complex_pairs):
        prefix = f'OC{j+1}_'
        # count layers for this omega net
        num_w = sum(1 for k in W if k.startswith(f'W{prefix}'))
        ind = 2 * j
        omegas.append(
            omega_net_apply_one(ycoords[:, ind:ind + 2], W, b, prefix, num_w, act_type)
        )
    # real omega networks
    for j in range(num_real):
        prefix = f'OR{j+1}_'
        num_w = sum(1 for k in W if k.startswith(f'W{prefix}'))
        ind = 2 * num_complex_pairs + j
        omegas.append(
            omega_net_apply_one(
                ycoords[:, ind] if ycoords.ndim > 1 else ycoords[:, np.newaxis],
                W, b, prefix, num_w, act_type
            )
        )
    return omegas


def varying_multiply(y, omegas, delta_t, num_real, num_complex_pairs):
    k = y.shape[1]

    complex_list = []

    for j in np.arange(num_complex_pairs):
        ind = 2 * j
        ystack = np.stack([np.asarray(y[:, ind:ind + 2]), np.asarray(y[:, ind:ind + 2])], axis=2)
        L_stack = FormComplexConjugateBlock(omegas[j], delta_t)
        elmtwise_prod = np.multiply(ystack, L_stack)
        complex_list.append(np.sum(elmtwise_prod, axis=1))

    if len(complex_list):
        complex_part = np.concatenate(complex_list, axis=1)

    real_list = []
    for j in np.arange(num_real):
        ind = 2 * num_complex_pairs + j
        temp_y = y[:, ind]
        if len(temp_y.shape) == 1:
            temp_y = temp_y[:, np.newaxis]
        temp_omegas = omegas[num_complex_pairs + j]
        evals = np.exp(temp_omegas * delta_t)
        item_real_list = np.multiply(temp_y, evals)
        real_list.append(item_real_list)

    if len(real_list):
        real_part = np.concatenate(real_list, axis=1)

    if len(complex_list) and len(real_list):
        return np.concatenate([complex_part, real_part], axis=1)

    elif len(complex_list):
        return complex_part

    else:
        return real_part

def num_shifts_in_stack(params):
    max_shifts_to_stack = 1
    if params['num_shifts']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts']))
    if params['num_shifts_middle']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts_middle']))

    return max_shifts_to_stack


def stack_data_with_control(x_data, u_data, num_shifts, len_time):
    """
    Stack state and control data for Koopman training using vectorized slicing.

    Returns:
      Xs: ndarray (num_shifts+1, num_traj*(len_time-num_shifts), state_dim)
      Us: ndarray (num_shifts,   num_traj*(len_time-num_shifts), control_dim)
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
    return np.maximum(0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def encoder_apply(x, weights, biases, name, num_weights, act_type='relu'):
    prev_layer = x.copy()

    for i in np.arange(num_weights - 1):
        h1 = np.dot(prev_layer, weights['W%s%d' % (name, i + 1)]) + biases['b%s%d' % (name, i + 1)]

        if act_type == 'sigmoid':
            h1 = sigmoid(h1)
        elif act_type == 'relu':
            h1 = relu(h1)

        prev_layer = h1.copy()
    final = np.dot(prev_layer, weights['W%s%d' % (name, num_weights)]) + biases['b%s%d' % (name, num_weights)]

    return final

def moving_average(x, window=10):
    """Simple moving average."""
    return np.convolve(x, np.ones(window)/window, mode='valid')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(x, window=10):
    """Simple moving average."""
    return np.convolve(x, np.ones(window)/window, mode='valid')

def PlotLosses(errors, logInd=(0, 1), alpha=0.3, smooth_window=10, ylabel=r"Loss"):
    """Plot a single loss (train & val) with smoothing, matching publication style."""
    print("Train = blue, Val = orange (raw = faint, smoothed = bold)")
    
    errors = errors.copy()
    for j in logInd:
        errors[:, j] = np.log10(errors[:, j])

    pal = sns.color_palette("colorblind")
    x = np.arange(errors.shape[0])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Raw curves
    ax.plot(x, errors[:, logInd[0]], color=pal[0], alpha=alpha, label="Training (raw)")
    ax.plot(x, errors[:, logInd[1]], color=pal[1], alpha=alpha, label="Validation (raw)")

    # Smoothed curves
    x_smooth = np.arange(smooth_window//2, len(errors) - smooth_window//2 + 1)
    y_train_smooth = moving_average(errors[:, logInd[0]], smooth_window)
    y_val_smooth = moving_average(errors[:, logInd[1]], smooth_window)
    ax.plot(x_smooth, y_train_smooth, color=pal[0], lw=2, label="Training (smoothed)")
    ax.plot(x_smooth, y_val_smooth, color=pal[1], lw=2, label="Validation (smoothed)")

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(False)
    ax.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.show()


def load_weights(fname, numWeights, type='E'):
    """Load weight and bias CSVs based on checkpoint prefix."""
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
    """Load Koopman network weights and biases from checkpoint or CSVs."""
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
        # encoder weights
        enc_layers = [m for m in model.encoder.net if isinstance(m, nn.Linear)]
        for j, layer in enumerate(enc_layers):
            W[f'WE{j+1}'] = np.matrix(layer.weight.detach().numpy().T)
            b[f'bE{j+1}'] = np.matrix(layer.bias.detach().numpy())
        # decoder weights
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
        # control net (optional)
        if hasattr(model, 'control_net'):
            ctrl_layers = [m for m in model.control_net.net if isinstance(m, nn.Linear)]
            for j, layer in enumerate(ctrl_layers):
                W[f'WC{j+1}'] = np.matrix(layer.weight.detach().numpy().T)
                b[f'bC{j+1}'] = np.matrix(layer.bias.detach().numpy())
        return W, b, model
    # CSV-based loading
    d = int((numWeights - 1) / 2)
    weights, biases = load_weights(fname, d, 'E')
    W, B = load_weights(fname, d, 'D')
    weights.update(W)
    biases.update(B)
    for j in range(num_complex_pairs):
        Wj, Bj = load_weights(fname, numWeightsOmega, f'OC{j+1}_')
        weights.update(Wj)
        biases.update(Bj)
    for j in range(num_real):
        Wj, Bj = load_weights(fname, numWeightsOmega, f'OR{j+1}_')
        weights.update(Wj)
        biases.update(Bj)
    return weights, biases, None