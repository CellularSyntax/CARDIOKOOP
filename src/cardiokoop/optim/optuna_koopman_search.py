#!/usr/bin/env python3
"""
Random-search hyper-parameter optimization for the Koopman network (Optuna).

Run locally:
    python -m cardiokoop.optim.optuna_koopman_search --local --trials 30
Run via CLI wrapper:
    python -m cardiokoop.cli optuna_koopman_search --local --trials 30
Submit to SLURM (no --local):
    python -m cardiokoop.cli optuna_koopman_search --trials 100
"""

# TODO: Update all comments and docstrings mentioning "random search" to "TPE search" or "Bayesian optimization",
# since the code uses Optuna's TPESampler (Tree-structured Parzen Estimator) instead of pure random search.

import os, sys, copy, argparse, subprocess
import optuna
import numpy as np
from optuna.samplers import RandomSampler, TPESampler
import datetime
from optuna.pruners import HyperbandPruner

from cardiokoop.training import main_exp
from cardiokoop.training.train_koopman import params as base_params_module


# ───────────────────────────────── objective ──────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    """
    One Optuna trial → returns %RMSE (lower is better).

    Search space is centred on the current best run:
        * encoder funnel = [64, 32] (enc_factor pinned at 0.5)
        * Ω-net ≈ 3×1024
        * Ψ-net ≈ 3×256
        * k-dim = 4 real + 4 complex pairs
    """
    # deep-copy the default hyper-param template
    params = copy.deepcopy(base_params_module)

    # ── 1. Encoder / decoder funnel ────────────────────────────────
    enc_layers = trial.suggest_categorical("enc_layers", [2, 3])          # champion = 2
    enc_start  = trial.suggest_categorical("enc_start",  [64, 128, 256])  # 64 reproduces best run
    enc_factor = 0.5                                                      # fixed
    params["encoder_funnel_widths"] = [
        int(enc_start * (enc_factor ** i)) for i in range(enc_layers)
    ]
    params["decoder_widths"] = list(params["encoder_funnel_widths"])[::-1]

    # ── 2. Ω (omega) network ───────────────────────────────────────
    omega_layers = trial.suggest_categorical("omega_layers", [2, 3, 4])
    omega_w      = trial.suggest_categorical("omega_width",  [512, 1024])
    params["hidden_widths_omega"] = [omega_w] * omega_layers

    # ── 3. Ψ (control) network ─────────────────────────────────────
    psi_layers = trial.suggest_categorical("psi_layers", [2, 3, 4])
    psi_w      = trial.suggest_categorical("psi_width",   [128, 256, 512])
    params["hidden_widths_control"] = [psi_w] * psi_layers
    params["control_gain"] = trial.suggest_float("control_gain", 0.05, 0.20, log=True)

    # ── 4. Koopman eigen-structure ─────────────────────────────────
    num_real = trial.suggest_categorical("num_real",          [3, 4, 5])
    num_cmp  = trial.suggest_categorical("num_complex_pairs", [3, 4, 5])
    k, n = num_real + 2 * num_cmp, params["input_dim"]
    params.update({
        "num_real":          num_real,
        "num_complex_pairs": num_cmp,
        "num_evals":         k,
    })
    params["widths"] = [n] + params["encoder_funnel_widths"] + [k, k] \
                       + params["decoder_widths"] + [n]

    # ── 5. Loss weights ────────────────────────────────────────────
    params["recon_lam"] = trial.suggest_float("recon_lam", 3e-3, 2e-1, log=True)
    params["Linf_lam"]  = trial.suggest_float("Linf_lam",  1e-7, 1e-5, log=True)
    params["L2_lam"]    = trial.suggest_float("L2_lam",    1e-15, 1e-8, log=True)

    # ── 6. Forecast horizon ───────────────────────────────────────
    num_shifts = trial.suggest_categorical("num_shifts", list(range(8, 16)))
    params["num_shifts"] = params["num_shifts_middle"] = num_shifts

    # ── 7. Training dynamics (cheap wins) ─────────────────────────
    #params["batch_size"]    = trial.suggest_categorical("batch_size", [128, 256, 512])

    # ── 8. Use only the first 100 ICs for the optimization  ────────
    #params["num_initial_conditions"] = 100
    #max_shifts   = max(params["num_shifts"], params["num_shifts_middle"])
    #num_examples = params["num_initial_conditions"] * (params["len_time"] - max_shifts)
    #steps_to_see_all = num_examples / params["batch_size"]
    #params["num_steps_per_file_pass"] = int(steps_to_see_all + 1) * params["num_steps_per_batch"]
    # ------------------------------------------------------------------
    
    # ── 9. Per-trial output folder ─────────────────────────────────
    trial_root = os.path.join("./results/optuna_runs", f"opt_trial_{trial.number}")
    os.makedirs(trial_root, exist_ok=True)

    # ── 10. Single training run ─────────────────────────────────────
    fold_params = copy.deepcopy(params)
    fold_params["cv_fold"]    = 0          # required by training code
    fold_params["folder_name"] = trial_root
    os.makedirs(fold_params["folder_name"], exist_ok=True)
    main_exp(fold_params, trial=trial)  # pass the trial for pruning

    # ── 11. Evaluate & report to Optuna ────────────────────────────
    csv_path  = fold_params["model_path"].replace("model.ckpt", "error.csv")
    errors    = np.loadtxt(csv_path, delimiter=",")

    VAL_PCT_COL = 17            # 0-based → 18th column in the file

    best_idx        = int(np.argmin(errors[:, VAL_PCT_COL]))
    val_pct_rmse    = float(errors[best_idx, VAL_PCT_COL])

    print(f"Trial {trial.number:03d}  %RMSE = {val_pct_rmse:.2f}%")
    return val_pct_rmse          # this is what Optuna will optimise

# ──────────────────────────────────── main ────────────────────────────────────
def main(args=None) -> None:
    """
    Entry-point called either via:
        python -m cardiokoop.optim.optuna_koopman_search [flags]
    or via the cardiokoop CLI wrapper that passes `args` in as a list.
    """
    if args is None:                 # called directly with  -m  -> use CLI flags
        args = sys.argv[1:]

    p = argparse.ArgumentParser()
    p.add_argument("--local",  action="store_true",
                   help="Run locally (disable SLURM hand-off)")
    p.add_argument("--trials", type=int, default=50,
                   help="Number of random-search trials")
    cfg = p.parse_args(args)

    # ---- optional SLURM hand-off ------------------------------------
    if not cfg.local and "SLURM_JOB_ID" not in os.environ:
        print("Submitting self to SLURM …")
        subprocess.check_call(["sbatch"] + sys.argv)
        return

    # ---- run Optuna random-search -----------------------------------
    #sampler = RandomSampler(seed=42)
    sampler = TPESampler(multivariate=True, seed=42, n_startup_trials=20)
    storage = "sqlite:///koopman_optuna.db"        # or a shared RDB URL
    pruner  = HyperbandPruner(
        min_resource      = 1,   # first rung
        max_resource      = 4,   # last rung  (4× the min)
        reduction_factor  = 3    # how aggressively to drop trials
    )

    study = optuna.create_study(
        study_name       = f"koopman_hyperparam_search_{datetime.datetime.now():%Y%m%d_%H%M%S}",
        direction        = "minimize",
        sampler          = sampler,      # your TPE sampler
        pruner           = pruner,       #  ←  new
        storage          = storage,
        load_if_exists   = True,
    )

    #study   = optuna.create_study(
    #    study_name       =f"koopman_random_search_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    #    #study_name       =f"koopman_random_search",
    #    direction        ="minimize",
    #    sampler          =sampler,
    #    storage          =storage,
    #    load_if_exists   =True,
    #)
    study.optimize(objective, n_trials=cfg.trials)

    print("Finished.  Best-trial params:\n", study.best_trial.params) 


if __name__ == "__main__":           # allows `python optuna_koopman_search.py`
    print("CLI entry-point for Optuna Koopman search")
    main()
