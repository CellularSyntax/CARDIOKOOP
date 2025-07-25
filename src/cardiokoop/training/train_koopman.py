"""
Defines the default `params` dict for Koopman training
AND allows running a single training from the command line:

    python -m cardiokoop.training.train_koopman --exp-folder my_run
"""

import copy
import argparse
from cardiokoop.training import main_exp

params = {}

params["use_control"] = True
params["use_control_rnn"] = False
params["control_dim"] = 1
params["control_gain"] = 0.1
params["hidden_widths_control"] = [256, 256, 256]
params["encoder_funnel_widths"] = [64, 32]
params["decoder_widths"] = [64, 32]  # only necessary when using control_rnn
params["trainable_control_gain"] = True  # False → fixed, True → learnable

# settings related to dataset
params["data_name"] = "csv_data_500_12sigs"
params["data_train_len"] = 1
params["len_time"] = 1500
n = 12  # dimension of system (and input layer)
params["input_dim"] = n
num_initial_conditions = 400  # per training file
params["delta_t"] = 0.01

# settings related to saving results
params["folder_name"] = "exp1_best"

# settings related to network architecture
params["num_real"] = 4
params["num_complex_pairs"] = 4
params["num_evals"] = params["num_real"] + 2 * params["num_complex_pairs"]
k = params["num_evals"]  # dimension of y-coordinates
w = 1024
params["widths"] = [
    n,
    w // 2,
    w // 4,
    k,
    k,
    w // 4,
    w // 2,
    n,
]  # not used if using control_rnn
wo = 1024
params["hidden_widths_omega"] = [wo, wo, wo]

params["num_shifts"] = 10
params["num_shifts_middle"] = 10
max_shifts = max(params["num_shifts"], params["num_shifts_middle"])
num_examples = num_initial_conditions * (params["len_time"] - max_shifts)
print(f"Number of examples: {num_examples}")
params["shifts"] = list(range(1, max_shifts + 1))
params["recon_lam"] = 0.1
params["Linf_lam"] = 10 ** (-6)
params["L1_lam"] = 0.0
params["L2_lam"] = 10 ** (-15)
params["auto_first"] = 0
params["relative_loss"] = 0

# settings related to the training
params["num_passes_per_file"] = 15 * 6 * 10
params["num_steps_per_batch"] = 2
params["learning_rate"] = 10 ** (-3)
params["batch_size"] = 256
steps_to_see_all = num_examples / params["batch_size"]
params["num_steps_per_file_pass"] = (int(steps_to_see_all) + 1) * params[
    "num_steps_per_batch"
]

# settings for status printing and periodic saving
params["print_every"] = 20  # how many steps between status prints
params["save_every"] = 100  # how many steps between periodic saves (0 to disable)

# settings related to the timing
params["max_time"] = 4 * 60 * 60  # 4 hours
params["min_5min"] = 1.5
params["min_20min"] = 0.5
params["min_40min"] = 0.5
params["min_1hr"] = 0.3
params["min_2hr"] = 0.3
params["min_3hr"] = 0.2
params["min_halfway"] = 0.3

# ────────────────────────────────── CLI hook ─────────────────────────────────
def _cli():
    ap = argparse.ArgumentParser(description="Run a single Koopman training run")
    ap.add_argument("--exp-folder", "-f", help="Override params['folder_name']")
    ns = ap.parse_args()

    run_params = copy.deepcopy(params)
    if ns.exp_folder:
        run_params["folder_name"] = ns.exp_folder
    else:
        run_params["folder_name"] = "exp_default"

    main_exp(run_params)


if __name__ == "__main__":
    _cli()
