#!/usr/bin/env python3
import argparse, importlib, runpy, sys, random

def main() -> None:
    ascii_banner()
    parser = argparse.ArgumentParser(prog="cardiokoop")
    sp = parser.add_subparsers(dest="command", required=True)

    # ─────────────── data generation ────────────────
    from argparse import ArgumentDefaultsHelpFormatter
    p = sp.add_parser(
        "create_dataset",
        help="Generate training/validation CSVs",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "-i", "--input-dir",
        default="./raw_data/csv_sims",
        help="Directory containing raw sim_*.csv files"
    )
    p.add_argument(
        "-o", "--output-base",
        default="./data/csv_data_500_12sigs",
        help="Base folder for output CSVs"
    )
    p.add_argument(
        "-p", "--file-prefix",
        default="csv_data_500_12sigs",
        help="Prefix for *_x.csv and *_u.csv files"
    )
    p.add_argument(
        "-d", "--downsample-factor",
        type=int, default=1,
        help="Keep every Nth sample"
    )
    p.add_argument(
        "--t-start",
        type=int, default=2500,
        help="Start index in each raw time series"
    )
    p.add_argument(
        "--t-end",
        type=int, default=4000,
        help="End index in each raw time series"
    )
    p.add_argument(
        "-s", "--split-ratios",
        nargs=3, type=float, default=(0.8, 0.1, 0.1),
        metavar=("TRAIN","VAL","TEST"),
        help="Fractions for train/val/test (must sum to ≈1)"
    )
    p.add_argument(
        "-m", "--max-concat-per-file",
        type=int, default=500,
        help="Max windows per output CSV"
    )
    p.set_defaults(module="cardiokoop.data.create_dataset")

    # ─────────────── training commands ──────────────
    for mod in [
        "cardiokoop.training.train_koopman",
        "cardiokoop.training.train_lstm",
        "cardiokoop.training.train_bilstm",
        "cardiokoop.training.train_gru",
    ]:
        name = mod.split(".")[-1]
        p = sp.add_parser(name, help=f"Run {name.replace('_',' ')}")
        p.set_defaults(module=mod)
        p.add_argument("--exp-folder", "-f", default=None,
                       help="Override results folder")

    # ─────────────── Optuna search ──────────────────
    sp.add_parser("optuna_koopman_search",
                  help="Random-search hyper-params for Koopman"
                  ).set_defaults(module="cardiokoop.optim.optuna_koopman_search")

    # ─────────────── post-processing ────────────────
    p = sp.add_parser("postprocess_koopman",
                      help="Compute %%RMSE, R², LoA etc. for a trained Koopman model")
    p.set_defaults(module="cardiokoop.postprocessing.postprocess_koopman")
    p.add_argument("--exp-folder", "-f", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--error-csv",  default=None)
    p.add_argument("--data-folder", "-d", default="data")

    p = sp.add_parser("postprocess_lstm",
                      help="Compute %%RMSE, R², LoA etc. for a trained LSTM model")
    p.set_defaults(module="cardiokoop.postprocessing.postprocess_lstm")
    p.add_argument("--exp-folder", "-f", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--error-csv",  default=None)
    p.add_argument("--data-folder", "-d", default="data")

    p = sp.add_parser("postprocess_gru",
                      help="Compute %%RMSE, R², LoA etc. for a trained GRU model")
    p.set_defaults(module="cardiokoop.postprocessing.postprocess_gru")
    p.add_argument("--exp-folder", "-f", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--error-csv",  default=None)
    p.add_argument("--data-folder", "-d", default="data")

    p = sp.add_parser("postprocess_bilstm",
                      help="Compute %%RMSE, R², LoA etc. for a trained BiLSTM model")
    p.set_defaults(module="cardiokoop.postprocessing.postprocess_bilstm")
    p.add_argument("--exp-folder", "-f", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--error-csv",  default=None)
    p.add_argument("--data-folder", "-d", default="data")

    # ─────────────── parse & dispatch ───────────────
    args, _ = parser.parse_known_args()
    sub_argv = sys.argv[2:]
    mod = importlib.import_module(args.module)
    if hasattr(mod, "main"):
        mod.main(sub_argv)
    else:
        sys.argv = [args.command] + sub_argv
        runpy.run_module(args.module, run_name="__main__")

def ascii_banner() -> None:
    RESET = "\033[0m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Define a palette of ANSI colors to randomly choose from (excluding blue)
    COLOR_PALETTE = [
        "\033[31m",  # Red
        "\033[91m",  # Bright Red
        "\033[33m",  # Yellow (orange-ish)
        "\033[93m",  # Bright Yellow
    ]

    raw_logo = r"""
                (    (     (        )      )    )      )   (     
   (     (      )\ ) )\ )  )\ )  ( /(   ( /( ( /(   ( /(   )\ )  
   )\    )\    (()/((()/( (()/(  )\())  )\()))\())  )\()) (()/(  
 (((_)((((_)(   /(_))/(_)) /(_))((_)\ |((_)\((_)\  ((_)\   /(_)) 
 )\___ )\ _ )\ (_)) (_))_ (_))    ((_)|_ ((_) ((_)   ((_) (_))   
((/ __|(_)_\(_)| _ \ |   \|_ _|  / _ \| |/ / / _ \  / _ \ | _ \  
 | (__  / _ \  |   / | |) || |  | (_) | ' < | (_) || (_) ||  _/  
  \___|/_/ \_\ |_|_\ |___/|___|  \___/ _|\_\ \___/  \___/ |_|    
"""

    colored_logo = ''.join([
        f"{GRAY}@{RESET}" if c == '@' else f"{random.choice(COLOR_PALETTE)}{c}{RESET}"
        for c in raw_logo
    ])

    print(colored_logo)

    print("""
The copyrights of this software are owned by Medical University of Vienna. Please refer to the LICENSE and
README.md files for licensing instructions.The source code can be found at the following GitHub repository:
https://github.com/CellularSyntax/CARDIOKOOP
""")
    print("---")

if __name__ == "__main__":
    main()
