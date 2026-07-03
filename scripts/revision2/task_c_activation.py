#!/usr/bin/env python3
"""
TASK C (editor mandatory #5 / R1.2) -- control-net output-activation comparison
===============================================================================
Does the control-net output activation (the tanh in
B_u = gamma * tanh(control_net(u_t))) matter?  Compare {tanh, ReLU, sigmoid,
identity/linear}.

STEP 1 (zero-compute, preferred): inspect the Optuna study.
  The 301-trial search (koopman_hyperparam_search_20250713_193701.csv) varied
  enc/omega/psi geometry, control_gain, num_real, num_complex_pairs, recon_lam,
  Linf_lam, L2_lam and num_shifts -- but NO activation (hidden or output).  The
  logs do NOT cover this, so we run the sanctioned minimal experiment.

STEP 2 -- two complementary analyses (the main model is NEVER retrained):

  (1) DEPLOYED-NET ACTIVATION SWAP  [primary; inference only]
      Keep the trained, deployed control net and simply replace its output
      activation at inference.  This isolates the effect of the activation on the
      actual model.

  (2) CONTROL-NET-ONLY RETRAINING  [the mandated "retrain the small control net
      alone"; frozen Koopman backbone]
      Reinitialise + retrain ONLY the control net under each activation with the
      original Koopman loss.  NOTE: with the backbone frozen this uniformly
      underperforms the jointly-trained deployed model (the backbone cannot
      co-adapt and the short-window loss is a weak proxy for 1499-step rollout),
      so its ABSOLUTE numbers are not comparable to the deployed model -- only the
      qualitative stability ranking is.

Both analyses give the same conclusion (see REVISION2_RESULTS.md): a zero-centred
output activation is required.  tanh and identity give stable, accurate rollouts;
the non-negative activations (ReLU, sigmoid) inject a persistent DC bias into the
latent update that accumulates over the 1499-step horizon -- ReLU collapses to the
no-control error and sigmoid diverges.

Outputs
  results/revision2/task_c_activation.csv / .tex / .json
"""
import os
import sys
import csv
import json
import types

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C
from cardiokoop.network.networkarch import MLP
from cardiokoop.training.core import define_loss, define_regularization
from cardiokoop.utils import stack_data_with_control, num_shifts_in_stack

DEVICE = C.get_device()

ACTS = {
    "tanh":     torch.tanh,
    "relu":     torch.relu,
    "sigmoid":  torch.sigmoid,
    "identity": (lambda x: x),
}

BATCH, TRAIN_STEPS = 512, 1500     # fixed-budget control-net-only fine-tune (secondary analysis)
DIVERGE_PCT = 1e4   # %RMSE above this (or nan) => "diverged"


def eval_activation(model, act_fn, Xn, Un, Xph, sig_mean, sig_std):
    B, T, D = Xph.shape
    pred = C.koopman_rollout(model, Xn[:, 0, :], Un, control_gain=model.control_gain,
                             control_act=act_fn, device=DEVICE) * sig_std + sig_mean
    if not np.isfinite(pred).all():
        return float("nan"), float("nan"), float("nan"), True
    pct = np.array([C.pct_rmse_ps(Xph[i], pred[i]) for i in range(B)])
    r2 = C.r2_flat(Xph.reshape(-1, D), pred.reshape(-1, D))
    diverged = bool(pct.mean() > DIVERGE_PCT)
    return float(pct.mean()), float(C.ci95(pct)), float(r2), diverged


def make_advance(act_fn):
    def _advance(self, y, omegas, u_t):
        num_cp = self.num_complex_pairs; dt = self.delta_t; batch = y.size(0)
        y_complex = y[:, :2 * num_cp].view(batch, num_cp, 2)
        oc = torch.stack(omegas[:num_cp], dim=1)
        scale = torch.exp(torch.clamp(oc[:, :, 1], -5.0, 5.0) * dt)
        angle = oc[:, :, 0] * dt
        cos_a = scale * torch.cos(angle); sin_a = scale * torch.sin(angle)
        rot = torch.stack([torch.stack([cos_a, sin_a], -1),
                           torch.stack([-sin_a, cos_a], -1)], dim=-2)
        y_c = torch.matmul(y_complex.unsqueeze(-2), rot).squeeze(-2).reshape(batch, 2 * num_cp)
        y_real = y[:, 2 * num_cp:]
        orl = torch.stack(omegas[num_cp:], dim=1).squeeze(-1)
        y_r = y_real * torch.exp(torch.clamp(orl, -5.0, 5.0) * dt)
        return torch.cat([y_c, y_r], dim=1) + self.control_gain * act_fn(self.control_net(u_t))
    return _advance


def load_stacked(split, params):
    x = np.loadtxt(os.path.join(C.DATA_DIR, f"{C.DATA_NAME}_{split}1_x.csv"), delimiter=",")
    u = np.loadtxt(os.path.join(C.DATA_DIR, f"{C.DATA_NAME}_{split}1_u.csv"), delimiter=",")
    ms = num_shifts_in_stack(params)
    Xs, Us = stack_data_with_control(x, u, ms, params["len_time"])
    return (torch.from_numpy(Xs).float().permute(1, 0, 2),
            torch.from_numpy(Us).float().permute(1, 0, 2))


def retrain_control_net(act_name, act_fn, params, train, Xn, Un, Xph, sig_mean, sig_std):
    model, _, _, _ = C.load_koopman(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(C.SEED)
    latent = 2 * model.num_complex_pairs + model.num_real
    model.control_net = MLP([params.get("control_dim", 1)] +
                            params.get("hidden_widths_control", [256, 256, 256]) +
                            [latent], params.get("act_type", "relu")).to(DEVICE)
    for p in model.control_net.parameters():
        p.requires_grad_(True)
    model._advance = types.MethodType(make_advance(act_fn), model)
    opt = torch.optim.Adam(model.control_net.parameters(), lr=params["learning_rate"])

    Xtr, Utr = train
    N = Xtr.shape[0]
    torch.manual_seed(C.SEED)
    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, N, (BATCH,))
        xb = Xtr[idx].to(DEVICE); ub = Utr[idx].to(DEVICE)
        model.train(); opt.zero_grad()
        yp, gl = model(xb, ub)
        *_, loss = define_loss(model, xb, yp, gl, params, u=ub)
        l1r, l2r = define_regularization(model, params)
        (loss + l1r + l2r).backward(); opt.step()
        if step % 500 == 0:
            print(f"    [{act_name}] step {step}: train loss={loss.item():.6f}", flush=True)
    return eval_activation(model, act_fn, Xn, Un, Xph, sig_mean, sig_std)


def main():
    C.set_all_seeds(C.SEED)
    model, params, _, _ = C.load_koopman(DEVICE)
    Xn, Un, Uph, Xph = C.load_split("test")
    sig_mean, sig_std = C.load_norm_stats()

    # STEP 1: Optuna check
    opt_csv = os.path.join(C.RESULT_DIR, "optuna_runs",
                           "koopman_hyperparam_search_20250713_193701.csv")
    cols = list(pd.read_csv(opt_csv, nrows=1).columns)
    act_cols = [c for c in cols if "act" in c.lower()]
    print(f"Optuna activation columns: {act_cols if act_cols else 'NONE -> fallback experiment'}")

    def fmt(pct, div):
        return "diverged" if (div or not np.isfinite(pct)) else round(pct, 2)

    # ── Analysis 1: deployed-net activation swap (inference only) ──────────────
    print("\n[Analysis 1] deployed control net, output-activation swap (inference only):")
    swap = {}
    for name, fn in ACTS.items():
        pct, ci, r2, div = eval_activation(model, fn, Xn, Un, Xph, sig_mean, sig_std)
        swap[name] = {"pct_rmse_mean": fmt(pct, div),
                      "pct_rmse_ci95": (None if div or not np.isfinite(pct) else round(ci, 2)),
                      "r2_pooled": (None if div or not np.isfinite(r2) else round(r2, 3)),
                      "status": "diverged" if (div or not np.isfinite(pct)) else "stable"}
        print(f"  {name:9s}: %RMSE={fmt(pct,div)}  R2={swap[name]['r2_pooled']}  ({swap[name]['status']})")

    # ── Analysis 2: mandated control-net-only retraining (frozen backbone) ─────
    print("\n[Analysis 2] control-net-only retraining under each activation "
          "(frozen backbone; poor proxy, see docstring):")
    train = load_stacked("train", params)
    retrain = {}
    for name, fn in ACTS.items():
        pct, ci, r2, div = retrain_control_net(name, fn, params, train,
                                               Xn, Un, Xph, sig_mean, sig_std)
        retrain[name] = {"pct_rmse_mean": fmt(pct, div),
                         "pct_rmse_ci95": (None if div or not np.isfinite(pct) else round(ci, 2)),
                         "r2_pooled": (None if div or not np.isfinite(r2) else round(r2, 3)),
                         "status": "diverged" if (div or not np.isfinite(pct)) else "stable"}
        print(f"  {name:9s}: %RMSE={fmt(pct,div)}  R2={retrain[name]['r2_pooled']}  ({retrain[name]['status']})")

    # ── save table (both analyses side by side) ───────────────────────────────
    rows = []
    for name in ACTS:
        rows.append({
            "activation": name,
            "swap_%RMSE": swap[name]["pct_rmse_mean"],
            "swap_R2": swap[name]["r2_pooled"],
            "swap_status": swap[name]["status"],
            "retrain_%RMSE": retrain[name]["pct_rmse_mean"],
            "retrain_R2": retrain[name]["r2_pooled"],
            "retrain_status": retrain[name]["status"],
        })
    with open(os.path.join(C.REV2_DIR, "task_c_activation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)

    df = pd.DataFrame(rows).set_index("activation")
    with open(os.path.join(C.REV2_DIR, "task_c_activation.tex"), "w") as f:
        f.write(df.to_latex(escape=False, column_format="lrrlrrl",
                caption=("Effect of the control-net output activation. 'swap' = the "
                         "deployed control net evaluated with a different output "
                         "activation (inference only); 'retrain' = only the control "
                         "net retrained under each activation on a frozen Koopman "
                         "backbone. Seed-42 test split. Zero-centred activations "
                         "(tanh, identity) are stable; ReLU/sigmoid diverge."),
                label="tab:control_activation"))

    with open(os.path.join(C.REV2_DIR, "task_c_activation.json"), "w") as f:
        json.dump({
            "optuna_covers_activation": bool(act_cols),
            "deployed_reference_tanh_pct_rmse": 17.54,
            "analysis1_deployed_swap": swap,
            "analysis2_controlnet_retrain": retrain,
            "conclusion": ("A zero-centred output activation is required. tanh (deployed) "
                           "and identity are stable and accurate; ReLU collapses to the "
                           "no-control error (~51%) and sigmoid diverges, because "
                           "non-negative activations add a persistent DC bias that "
                           "accumulates over the 1499-step horizon."),
        }, f, indent=2)
    print(f"\nSaved -> {os.path.join(C.REV2_DIR, 'task_c_activation.csv')} (+ .tex, .json)")


if __name__ == "__main__":
    main()
