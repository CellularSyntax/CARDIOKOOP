#!/usr/bin/env python3
"""
create_dataset.py

Slice raw simulation CSVs into fixed‑length windows, normalize, split into
train/val/test, and save as CSV datasets for upstream modeling.

Usage:
  # with defaults
  python create_dataset.py

  # override parameters
  python create_dataset.py \
    --input-dir ./raw_data \
    --t-start 1000 --t-end 2000 \
    --split-ratios 0.7 0.2 0.1 \
    --max-concat-per-file 300
"""
import argparse
import os
import glob
import sys

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create downsampled, windowed dataset from raw CSV sims",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir", "-i", default="./raw_data/csv_sims",
        help="Directory containing raw sim_*.csv files"
    )
    parser.add_argument(
        "--output-base", "-o", default="./data/csv_data_500_12sigs",
        help="Base folder for output CSVs"
    )
    parser.add_argument(
        "--file-prefix", "-p", default="csv_data_500_12sigs",
        help="Prefix for *_x.csv and *_u.csv files"
    )
    parser.add_argument(
        "--downsample-factor", "-d", type=int, default=1,
        help="Keep every Nth sample"
    )
    parser.add_argument(
        "--t-start", type=int, default=2500,
        help="Start index in each raw time series"
    )
    parser.add_argument(
        "--t-end", type=int, default=4000,
        help="End index in each raw time series"
    )
    parser.add_argument(
        "--split-ratios", "-s", nargs=3, type=float,
        default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"),
        help="Fractions for train/val/test (must sum to ≈1)"
    )
    parser.add_argument(
        "--max-concat-per-file", "-m", type=int, default=500,
        help="Max windows per output CSV"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    input_dir           = args.input_dir
    output_base         = args.output_base
    file_prefix         = args.file_prefix
    DOWNSAMPLE_FACTOR   = args.downsample_factor
    t_start, t_end      = args.t_start, args.t_end
    split_ratios        = tuple(args.split_ratios)
    max_concat_per_file = args.max_concat_per_file

    window_size = (t_end - t_start) // DOWNSAMPLE_FACTOR
    stride = window_size

    # Phase 0: Prepare output directory
    os.makedirs(output_base, exist_ok=True)

    # Phase 1: Load & slice raw CSVs
    print(f"Loading and slicing CSVs from {input_dir}…")
    csv_files = sorted(glob.glob(os.path.join(input_dir, "sim_*.csv")))
    all_sigs = []
    for csv_path in tqdm(csv_files, desc="Loading", unit="file"):
        try:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)[:, 1:]
            if data.shape[0] < t_end:
                continue
            sig_win  = data[t_start:t_end, :-1][::DOWNSAMPLE_FACTOR]
            ctrl_win = data[t_start:t_end,  -1][::DOWNSAMPLE_FACTOR]
            all_sigs.append((sig_win, ctrl_win))
        except Exception as e:
            print(f"Failed {csv_path}: {e}")
    print(f"Loaded {len(all_sigs)} valid simulations.")

    # Phase 2: Shuffle & split
    np.random.seed(42)
    np.random.shuffle(all_sigs)
    n_total = len(all_sigs)
    n_train = int(split_ratios[0] * n_total)
    n_val   = int(split_ratios[1] * n_total)

    splits = {
        "train": all_sigs[:n_train],
        "test":  all_sigs[n_train + n_val:]
    }
    if split_ratios[1] > 0:
        splits["val"] = all_sigs[n_train:n_train + n_val]

    # Phase 3: Compute normalization stats on training set
    print("Computing normalization stats from training data…")
    print(f"Training set size: {len(splits['train'])} sequences")
    train_sigs = np.vstack([sig for sig, _ in splits["train"]])
    mean = train_sigs.mean(axis=0)
    std  = train_sigs.std(axis=0)
    std[std == 0] = 1.0

    for folder in (output_base, os.path.dirname(output_base)):
        np.save(os.path.join(folder, "normalization_mean.npy"), mean)
        np.save(os.path.join(folder, "normalization_std.npy"), std)

    # Phase 4: Sliding windows & export CSVs
    print("Generating windows and writing CSV files…")
    for split, sims in splits.items():
        print(f"→ {split}: {len(sims)} sims")
        rows_sig, rows_ctrl = [], []
        file_idx = 1

        for sig, ctrl in tqdm(sims, desc=f"{split} sims", unit="sim"):
            n_windows = (sig.shape[0] - window_size) // stride + 1
            for i in range(n_windows):
                start = i * stride
                end   = start + window_size
                win_sig  = (sig[start:end] - mean) / std
                win_ctrl = ctrl[start:end]

                rows_sig.extend(win_sig.tolist())
                rows_ctrl.extend(win_ctrl.reshape(-1, 1).tolist())

                if len(rows_sig) >= window_size * max_concat_per_file:
                    suffix = f"{split}{file_idx}"
                    out_x = os.path.join(
                        output_base,
                        f"{file_prefix}_{suffix}_x.csv"
                    )
                    out_u = os.path.join(
                        output_base,
                        f"{file_prefix}_{suffix}_u.csv"
                    )
                    np.savetxt(out_x, np.array(rows_sig), delimiter=",")
                    np.savetxt(out_u, np.array(rows_ctrl), delimiter=",")
                    rows_sig, rows_ctrl = [], []
                    file_idx += 1

        # Write any leftovers
        if rows_sig:
            suffix = f"{split}{file_idx}"
            np.savetxt(
                os.path.join(
                    output_base,
                    f"{file_prefix}_{suffix}_x.csv"
                ),
                np.array(rows_sig),
                delimiter=","
            )
            np.savetxt(
                os.path.join(
                    output_base,
                    f"{file_prefix}_{suffix}_u.csv"
                ),
                np.array(rows_ctrl),
                delimiter=","
            )
            print(f"Saved {suffix}: {len(rows_sig) // window_size} windows.")
    print("CSV export complete.")

    # Phase 5: Quick plot of all trajectories
    print("\nPlotting all trajectories per signal channel including control signal...")
    x_files = sorted(glob.glob(os.path.join(
        output_base, f"{file_prefix}_train1_x.csv"
    )))
    u_files = sorted(glob.glob(os.path.join(
        output_base, f"{file_prefix}_train1_u.csv"
    )))

    if x_files and u_files:
        x_data = np.loadtxt(x_files[0], delimiter=",")
        u_data = np.loadtxt(u_files[0], delimiter=",")

        n_signals    = x_data.shape[1]
        seq_len      = window_size
        n_sequences  = (x_data.shape[0] // seq_len)

        x_resh = x_data[:n_sequences*seq_len].reshape(n_sequences, seq_len, n_signals)
        u_resh = u_data[:n_sequences*seq_len].reshape(n_sequences, seq_len)

        total_plots = n_signals + 1
        nrows       = (total_plots + 1) // 2
        fig, axs   = plt.subplots(nrows=nrows, ncols=2,
                                  figsize=(12, 1.5*nrows))
        fig.suptitle("All Signal Trajectories (Overlayed)", fontsize=14)
        axs = axs.flatten()

        for ch in range(n_signals):
            for seq in x_resh:
                axs[ch].plot(seq[:, ch], alpha=0.2, linewidth=0.8)
            axs[ch].set_title(f"Signal {ch+1}")
            axs[ch].grid(True)

        ax_u = axs[n_signals]
        for u_seq in u_resh:
            ax_u.plot(u_seq, alpha=0.2, linewidth=0.8, color='black')
        ax_u.set_title("Control Signal (u)")
        ax_u.grid(True)

        for extra in range(total_plots, len(axs)):
            axs[extra].axis('off')

        plt.tight_layout(rect=[0,0,1,0.96])
        plt.show()
    else:
        print("No training file found for plotting.")


if __name__ == "__main__":
    main(sys.argv[1:])
