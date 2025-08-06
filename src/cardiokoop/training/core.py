import os
import time

import numpy as np
import torch
from tqdm import tqdm
import optuna

from cardiokoop.utils import (
    stack_data,
    check_progress,
    save_params,
    set_defaults,
    num_shifts_in_stack,
    stack_data_with_control,
)
from cardiokoop.network import (
    KoopmanNet,
    KoopmanNetControl_v2,
    KoopmanNetControl_RNN,
)

data_cache = {}
control_cache = {}

def get_csv_cached(path, cache):
    """Load a CSV once and keep it in memory."""
    if path not in cache:
        cache[path] = np.loadtxt(path, delimiter=',')
    return cache[path]
# ------------------------------------------------------------------


def define_loss(model, x, y_preds, g_list, params, u=None):
    eps = 1e-5
    device = x.device

    # === loss1: prediction at t ===
    if params.get('relative_loss', 0):
        loss1_den = x[:, 0, :].pow(2).mean() + eps
    else:
        loss1_den = 1.0
    loss1 = params['recon_lam'] * (y_preds[0] - x[:, 0, :]).pow(2).mean() / loss1_den

    # === loss2: predictions at shifted steps ===
    loss2 = torch.tensor(0.0, device=device)
    if params.get('num_shifts', 0) > 0:
        for j, shift in enumerate(params['shifts']):
            if params.get('relative_loss', 0):
                loss2_den = x[:, shift, :].pow(2).mean() + eps
            else:
                loss2_den = 1.0
            loss2 += params['recon_lam'] * (y_preds[j + 1] - x[:, shift, :]).pow(2).mean() / loss2_den
        loss2 /= params['num_shifts']

    # === loss3: mid-shift latent autoregression ===
    loss3 = torch.tensor(0.0, device=device)
    if params.get('num_shifts_middle', 0) > 0:
        next_step = g_list[0]
        count = 0
        for j in range(max(params['shifts_middle'])):
            if u is not None:
                u_t = u[:, j, :]
                omegas = model.get_omegas(next_step)
                next_step = model._advance(next_step, omegas, u_t)
            else:
                next_step = model._advance(next_step, model.get_omegas(next_step))

            if (j + 1) in params['shifts_middle']:
                if params.get('relative_loss', 0):
                    loss3_den = g_list[count + 1].pow(2).mean() + eps
                else:
                    loss3_den = 1.0
                loss3 += params['mid_shift_lam'] * (next_step - g_list[count + 1]).pow(2).mean() / loss3_den
                count += 1
        loss3 /= params['num_shifts_middle']

    # === loss_linf: max error at t and t+1 ===
    if params.get('relative_loss', 0):
        den1 = x[:, 0, :].abs().max() + eps
        den2 = x[:, 1, :].abs().max() + eps
    else:
        den1 = den2 = 1.0

    linf1 = (y_preds[0] - x[:, 0, :]).abs().max() / den1
    linf2 = (y_preds[1] - x[:, 1, :]).abs().max() / den2
    loss_linf = params.get('Linf_lam', 0.0) * (linf1 + linf2)

    total_loss = loss1 + loss2 + loss3 + loss_linf
    return loss1, loss2, loss3, loss_linf, total_loss

def define_regularization(model, params):
    loss_l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    if params.get('L1_lam', 0.0) > 0:
        for name, p in model.named_parameters():
            if 'bias' not in name:
                loss_l1 = loss_l1 + p.abs().sum() * params['L1_lam']
    loss_l2 = torch.tensor(0.0, device=next(model.parameters()).device)
    if params.get('L2_lam', 0.0) > 0:
        for name, p in model.named_parameters():
            if 'bias' not in name:
                loss_l2 = loss_l2 + p.pow(2).sum() * params['L2_lam']
    return loss_l1, loss_l2

def try_net(data_val, params, trial=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params.get("use_control", False):
        model = KoopmanNetControl_v2(params).to(device)
    elif params.get("use_control_rnn", False):
        model = KoopmanNetControl_RNN(params).to(device)
    else:
        model = KoopmanNet(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    max_shifts = num_shifts_in_stack(params)
    
    if params.get("use_control", False):
        control_val = np.loadtxt(f"./data/{params['data_name']}_val1_u.csv", delimiter=',')
        data_val_tensor, control_val_tensor = stack_data_with_control(data_val, control_val, max_shifts, params['len_time'])
        u_val = torch.from_numpy(control_val_tensor).float().to(device).permute(1, 0, 2)
    elif params.get("use_control_rnn", False):
        control_val = np.loadtxt(f"./data/{params['data_name']}_val1_u.csv", delimiter=',')
        data_val_tensor, control_val_tensor = stack_data_with_control(data_val, control_val, max_shifts, params['len_time'])
        u_val = torch.from_numpy(control_val_tensor).float().to(device).permute(1, 0, 2)
    else:
        data_val_tensor = stack_data(data_val, max_shifts, params['len_time'])

    x_val = torch.from_numpy(data_val_tensor).float().to(device).permute(1, 0, 2)

    best_error = float('inf')
    start = time.time()
    csv_path = params['model_path'].replace('model', 'error').replace('ckpt', 'csv')
    train_val_error = []
    # initialize global progress bar
    first_file_path = f"./data/{params['data_name']}_train1_x.csv"
    data_train0 = np.loadtxt(first_file_path, delimiter=',')
    data_train0_tensor = stack_data(data_train0, max_shifts, params['len_time'])
    num_samples0 = data_train0_tensor.shape[1]
    num_batches0 = max(1, int(num_samples0 // params['batch_size']))
    total_steps = params['data_train_len'] * params['num_passes_per_file'] * params['num_steps_per_file_pass'] * num_batches0
    pbar = tqdm(total=total_steps, desc='Training', unit='step')

    finished = False
    patience      = params.get("patience", 20)   # ← how many bad steps we allow
    no_improve_cnt = 0                           # ← running counter

    for f in range(params['data_train_len'] * params['num_passes_per_file']):
        file_num = (f % params['data_train_len']) + 1        # 1-based index

        # --------- fetch (or load+cache) the state data ----------
        x_path = f"./data/{params['data_name']}_train{file_num}_x.csv"
        data_train = get_csv_cached(x_path, data_cache)

        # --------- fetch (or load+cache) the control input data, only if needed ----------
        if params.get("use_control", False) or params.get("use_control_rnn", False):
            u_path = f"./data/{params['data_name']}_train{file_num}_u.csv"
            control_train = get_csv_cached(u_path, control_cache)
            data_train_tensor, control_train_tensor = stack_data_with_control(
                data_train, control_train, max_shifts, params['len_time']
            )
        else:
            data_train_tensor = stack_data(data_train, max_shifts, params['len_time'])

        num_samples = data_train_tensor.shape[1]

        # shuffle consistently
        indices = np.random.permutation(num_samples)
        data_train_tensor = data_train_tensor[:, indices, :]
        if params.get("use_control", False) or params.get("use_control_rnn", False):
            control_train_tensor = control_train_tensor[:, indices, :]


        num_batches = max(1, int(num_samples // params['batch_size']))
        for step in range(int(params['num_steps_per_file_pass']) * num_batches):
            if params['batch_size'] < num_samples:
                offset = (step * params['batch_size']) % (num_samples - params['batch_size'])
            else:
                offset = 0
            batch_np = data_train_tensor[:, offset:offset + params['batch_size'], :]
            x_batch = torch.from_numpy(batch_np).float().to(device).permute(1, 0, 2)

            optimizer.zero_grad()
            if params.get("use_control", False):
                u_batch = torch.from_numpy(control_train_tensor[:, offset:offset + params['batch_size'], :]).float().to(device).permute(1, 0, 2)
                y_preds, g_list = model(x_batch, u_batch)
            elif params.get("use_control_rnn", False):
                u_batch = torch.from_numpy(control_train_tensor[:, offset:offset + params['batch_size'], :]).float().to(device).permute(1, 0, 2)
                y_preds, g_list = model(x_batch, u_batch)
            else:
                y_preds, g_list = model(x_batch)

            if params.get("use_control", False):
                loss1, loss2, loss3, loss_linf, loss = define_loss(model, x_batch, y_preds, g_list, params, u=u_batch)
            elif params.get("use_control_rnn", False):
                loss1, loss2, loss3, loss_linf, loss = define_loss(model, x_batch, y_preds, g_list, params, u=u_batch)
            else:
                loss1, loss2, loss3, loss_linf, loss = define_loss(model, x_batch, y_preds, g_list, params)

            loss_l1, loss_l2 = define_regularization(model, params)
            total_loss = loss + loss_l1 + loss_l2
            total_loss.backward()
            optimizer.step()
            pbar.update(1)

            if step >= params['num_steps_per_file_pass'] * num_batches:
                break

            if step % params.get('print_every', 20) == 0:
                with torch.no_grad():
                    # Training loss (without regularization, already computed above)
                    train_err = loss.item()

                    # Compute validation loss and regularization separately
                    if params.get("use_control", False):
                        y_val_preds, g_val_list = model(x_val, u_val)
                        val_loss1, val_loss2, val_loss3, val_loss_linf, val_loss = define_loss(model, x_val, y_val_preds, g_val_list, params, u=u_val)
                    elif params.get("use_control_rnn", False):
                        y_val_preds, g_val_list = model(x_val, u_val)
                        val_loss1, val_loss2, val_loss3, val_loss_linf, val_loss = define_loss(model, x_val, y_val_preds, g_val_list, params, u=u_val)
                    else:
                        y_val_preds, g_val_list = model(x_val)
                        val_loss1, val_loss2, val_loss3, val_loss_linf, val_loss = define_loss(model, x_val, y_val_preds, g_val_list, params)

                    val_l1, val_l2 = define_regularization(model, params)
                    val_err = (val_loss + val_l1 + val_l2).item()

                elapsed = time.time() - start
                tqdm.write(
                    f"file {file_num}, step {step}, train_err={train_err:.6f}, val_err={val_err:.6f}, "
                    f"best_err={best_error:.6f}, elapsed={elapsed:.1f}s"
                )

                if val_err < best_error * (1 - 1e-3):
                    best_error = val_err
                    torch.save({'params': params, 'state_dict': model.state_dict()}, params['model_path'])
                    tqdm.write(f"✓ Saved improved model to {params['model_path']} (val_err={val_err:.6f})")
                    no_improve_cnt = 0
                else:
                    no_improve_cnt += 1
                    tqdm.write(f"✗ No improvement (no_improve_cnt={no_improve_cnt})")

                # --- right after the if/else block, still inside the print-every section ---
                if no_improve_cnt >= patience:
                    tqdm.write(f"Early stopping: no improvement for {patience} evals.")
                    finished = True
                    save_params(params)
                    break

                # periodic save regardless of improvement
                save_every = params.get('save_every', 0)
                if save_every and step % save_every == 0:
                    torch.save({'params': params, 'state_dict': model.state_dict()}, params['model_path'])

                # Calculate scale and %RMSE over entire validation set
                val_scale = x_val.std(dim=(0, 1)).mean().item() + 1e-8
                val_pct_rmse = 100.0 * val_loss.item()**0.5 / val_scale

                row = [[
                    train_err, val_err,
                    (loss + loss_l1 + loss_l2).item(),
                    (val_loss + val_l1 + val_l2).item(),
                    loss1.item(), val_loss1.item(),
                    loss2.item(), val_loss2.item(),
                    loss3.item(), val_loss3.item(),
                    loss_linf.item(), val_loss_linf.item(),
                    loss_l1.item(), val_l1.item(),
                    loss_l2.item(), val_l2.item(),
                    val_scale, val_pct_rmse
                ]]
                with open(csv_path, 'a') as csv_f:
                    np.savetxt(csv_f, row, delimiter=',')
               
        if step % params.get('print_every', 20) != 0:       # last batch didn’t log
            with torch.no_grad():
                if params.get("use_control", False):
                    y_val_preds, g_val_list = model(x_val, u_val)
                elif params.get("use_control_rnn", False):
                    y_val_preds, g_val_list = model(x_val, u_val)
                else:
                    y_val_preds, g_val_list = model(x_val)

                val_loss1, val_loss2, val_loss3, val_loss_linf, val_loss = define_loss(
                    model, x_val, y_val_preds, g_val_list, params,
                    u=u_val if params.get("use_control") or params.get("use_control_rnn") else None
                )
                val_l1, val_l2 = define_regularization(model, params)
                val_scale   = x_val.std(dim=(0, 1)).mean().item() + 1e-8
                val_pct_rmse = 100.0 * val_loss.item() ** 0.5 / val_scale

        if trial is not None and (f + 1) <= 4:
            step_idx = f + 1                  # f is the outer file-pass loop (1-based)
            trial.report(val_pct_rmse, step=step_idx)

            if trial.should_prune():
                tqdm.write("Optuna pruned this trial at pass "
                        f"{step_idx} (val %RMSE={val_pct_rmse:.2f})")
                raise optuna.TrialPruned()
        if finished:
            break

    params['time_exp'] = time.time() - start
    # load full checkpoint (params and state_dict) under PyTorch >=2.6
    ckpt = torch.load(params['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    #np.savetxt(csv_path, np.array(train_val_error), delimiter=',')
    save_params(params)
    pbar.close()

def main_exp(params, trial=None):
    set_defaults(params)
    os.makedirs(params['folder_name'], exist_ok=True)
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    data_val = np.loadtxt(f"./data/{params['data_name']}_val1_x.csv", delimiter=',')
    control_val = np.loadtxt(f"./data/{params['data_name']}_val1_u.csv", delimiter=',')
    try_net(data_val, params, trial=trial)
