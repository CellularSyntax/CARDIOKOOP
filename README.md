<p align="center">
  <img src="assets/logo.png" alt="CardioKoop" width="1000"/>
</p>

# **CARDIOKOOP**
*Control-aware deep Koopman surrogate for real-time hemodynamic forecasting.*

[![Reproduce manuscript tables (Docker)](https://github.com/CellularSyntax/CARDIOKOOP/actions/workflows/reproduce.yml/badge.svg?branch=main)](https://github.com/CellularSyntax/CARDIOKOOP/actions/workflows/reproduce.yml)
[![Software DOI](https://img.shields.io/badge/Zenodo%20software-10.5281%2Fzenodo.22776287-blue)](https://doi.org/10.5281/zenodo.22776287)
[![Dataset DOI](https://img.shields.io/badge/Zenodo%20dataset-10.5281%2Fzenodo.21163127-blue)](https://doi.org/10.5281/zenodo.21163127)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Container](https://img.shields.io/badge/ghcr.io-cellularsyntax%2Fcardiokoop%3Av1.2.0-2496ed?logo=docker&logoColor=white)](https://github.com/CellularSyntax/CARDIOKOOP/pkgs/container/cardiokoop)
[![Release](https://img.shields.io/badge/release-v1.2.0-green)](https://github.com/CellularSyntax/CARDIOKOOP/releases/tag/v1.2.0)

CARDIOKOOP learns a **control-aware deep Koopman operator model** of a validated lumped-parameter cardiovascular model.
An encoder maps 12 hemodynamic signals (right/left atrial and ventricular pressures and volumes, pulmonary venous pressure,
left-ventricular in-/outflow, aortic pressure) to a latent space in which the dynamics are linear; a control network injects the
preload-step input (a step change of the unstressed venous volume at t = 500), and a decoder returns the physical signals. Starting
from one observed state, the surrogate rolls the full 1500-step trajectory out autoregressively in about 5 ms — roughly 380× faster
than the full-order simulation. The repository contains the package, the model checkpoints used for all reported results, the
dataset splits, the 500 raw simulations, seven baselines (LSTM, GRU, BiLSTM, MLP, AR(20), DLinear, NLinear), every script that
produced a number or a figure in the article, and the resulting tables and figures.

## Citation

If you use this code, data or pre-trained models, please cite the article **and** the software archive
(machine-readable metadata in [`CITATION.cff`](CITATION.cff)):

> **Haberbusch M., Brandt L.B., Aprile M., Lung D., Kuijper A., Moscato F.**
> *Real-Time Hemodynamic Prediction via Control-Aware Koopman Operator Models.* **Array** (Elsevier), 2026.

| Archive | DOI |
|---|---|
| Software, checkpoints, data and results — v1.2.0 (self-contained, all data files resolved, CI-verified container) | [10.5281/zenodo.22776287](https://doi.org/10.5281/zenodo.22776287) |
| Software — all versions (concept DOI) | [10.5281/zenodo.22776286](https://doi.org/10.5281/zenodo.22776286) |
| Dataset — seed-42 train/validation/test splits | [10.5281/zenodo.21163127](https://doi.org/10.5281/zenodo.21163127) |

## Quick start — reproduce all reported results

**Docker (one command).** The image contains the pinned environment, the package, the checkpoints and the committed results; it
downloads the seed-42 splits from the Zenodo dataset record (MD5-verified), regenerates Tables 3–5 and the statistics from the
checkpoints, and compares every number with the committed `results/tables/` at manuscript rounding (≈ 2–3 min on a laptop CPU):

```bash
docker pull ghcr.io/cellularsyntax/cardiokoop:v1.2.0
docker run --rm -v "$PWD/out:/workspace/out" ghcr.io/cellularsyntax/cardiokoop:v1.2.0
# or build it yourself (≈ 5 min): docker build -t cardiokoop . && docker run --rm -v "$PWD/out:/workspace/out" cardiokoop
```

`out/` then contains `table3_overall.*`, `table4_per_signal_full.*`, `table5_noise.*`, `statistics.json`, `r2_conventions.json`,
`koopman_test1_postprocessing_results.pkl`, `run_info.json` and the verification report `compare_report.md`; the container exits
non-zero if any number differs beyond the last printed digit. Options: `-e REPRO_THREADS=4`, `-e REPRO_DTYPE=float32`,
`-e REPRO_STRICT=1` (fail also on ±1 in the last digit), `-e CARDIOKOOP_DATA_DIR=/data -v <local splits>:/data`.

**Local installation** (Python 3.9–3.13; a CPU-only installation is sufficient for all tables, a GPU is needed only for training):

```bash
git clone https://github.com/CellularSyntax/CARDIOKOOP.git && cd CARDIOKOOP
git lfs pull                                    # resolves the *.csv data files (or download them from the Zenodo dataset record)
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0                        # CPU; CUDA: pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r environment/requirements-pinned.txt
pip install -e .
python scripts/tables/export_manuscript_tables.py     # Tables 3-5, statistics, R² conventions -> results/tables/ (~1 min on CPU)
python scripts/experiments/run_all.py                 # baselines, robustness and ablation experiments, Figure 5, Figs. S4, S6-S10
```

## Repository layout

```
CARDIOKOOP/
├── src/cardiokoop/            Python package (CLI `cardiokoop`)
│   ├── data/create_dataset.py     raw simulations -> normalised train/val/test splits
│   ├── network/networkarch.py     KoopmanNetControl_v2 (encoder, auxiliary omega nets, control net, decoder)
│   ├── training/                  train_koopman / train_lstm / train_gru / train_bilstm
│   ├── optim/                     Optuna hyper-parameter search for the Koopman network
│   └── postprocessing/            metric computation and result pickles for every architecture
├── data/                      seed-42 splits: csv_data_500_12sigs_{train1,val1,test1}_{x,u}.csv, normalisation statistics,
│                              cardiovascular_parameter_space.csv  (Git-LFS)
├── raw_data/csv_sims/         the 500 full-order lumped-parameter simulations (signals + parameters; Git-LFS)
├── checkpoints/               mlp_baseline.pt, dlinear_baseline.pt, nlinear_baseline.pt
├── results/
│   ├── koopman/ lstm/ gru/ bilstm/   model checkpoints + post-processing pickles (predictions and metrics)
│   ├── dlinear/ nlinear/ mlp/        baseline predictions (post-processing pickles)
│   ├── tables/                manuscript Tables 3-5, statistics, R² conventions, run metadata  (scripts/tables/)
│   ├── experiments/           experiment outputs: tables (CSV/TeX) and JSON                    (scripts/experiments/)
│   ├── *.json                 noise robustness, baselines, ablations, statistics               (scripts/experiments/task*_*.py)
│   ├── figures/               all manuscript and supplementary figure panels (SVG/PNG)
│   └── optuna_runs/           Optuna search history (301 trials)
├── scripts/
│   ├── experiments/           baseline, ablation and robustness experiments; table/figure builders (`run_all.py`)
│   ├── tables/                `export_manuscript_tables.py` (Tables 3-5 + statistics), `compare_results.py` (verification)
│   └── reproduce.sh           Docker/CI entry point: fetch splits from Zenodo -> export -> compare
├── notebooks/                 generate_figures_and_tables_revised_rev2.ipynb (Figures 2-4, S1-S3, S5) + notebook_utils.py
├── docs/REPRODUCIBILITY.md    per-table source map, metric conventions, checkpoint identity, CI pass criteria
├── environment/               requirements-pinned.txt, README.md (hardware and software versions)
├── Dockerfile  .github/workflows/reproduce.yml   container + CI reproduction check
├── CITATION.cff  LICENSE (MIT)  pyproject.toml
```

**Model checkpoints used for all reported results** (nothing is retrained by the reproduction scripts except the analytic OLS fit
of the AR(20) baseline; the Koopman checkpoint stores its hyper-parameters in `ckpt["params"]`, `scripts/check_ckpt_params.py`
prints the parameter counts):

| Model | File | Parameters |
|---|---|---|
| **Koopman** | `results/koopman/csv_data_500_12sigs_2025_06_18_18_50_13_876051_model.ckpt` | **17,239,600** |
| LSTM | `results/lstm/csv_data_500_12sigs_20250625_184322_model.ckpt` | 1,083,404 |
| GRU | `results/gru/csv_data_500_12sigs_20250625_170206_model.ckpt` | 814,092 |
| BiLSTM | `results/bilstm/csv_data_500_12sigs_20250625_204256_model.ckpt` | 2,166,796 |
| MLP (autoregressive) | `checkpoints/mlp_baseline.pt` (+ stored test predictions `results/mlp/mlp_postprocessing_results.pkl`) | 269,324 |
| DLinear / NLinear | `checkpoints/dlinear_baseline.pt`, `checkpoints/nlinear_baseline.pt` | 429,120 / 214,560 |
| AR(20) | analytic OLS fit on the training split (no file) | 240 |

## Experiments and figures

All experiments evaluate the seed-42 **test1** split (50 trajectories × 1500 steps × 12 signals, 1499-step horizon) against the
checkpoints above. `python scripts/experiments/run_all.py` runs tasks A–F and the figure builders in order
(`run_all.py a build panels` runs a subset); the `task1_*.py` … `task9_*.py` scripts are standalone.

| Experiment / figure | Script (`scripts/experiments/`) | Output |
|---|---|---|
| Tables 3, 4, 5 and the statistics paragraph | `../tables/export_manuscript_tables.py` | `results/tables/` |
| DLinear / NLinear direct linear baselines | `task_a_dlinear_nlinear.py` | `checkpoints/{dlinear,nlinear}_baseline.pt`, `results/{dlinear,nlinear}/`, `results/experiments/task_a_dlinear_nlinear.json` |
| MLP and AR(20) baselines | `task5_mlp_baseline.py`, `task4_ar_baseline.py` | `checkpoints/mlp_baseline.pt`, `results/mlp_baseline.json`, `results/ar_baseline.json` |
| Noise robustness, AWGN on the initial window (Table 5, Fig. 5 i–j) | `task3_noise_robustness.py`, `task3b_noise_robustness_baselines.py` | `results/noise_robustness.json`, `results/noise_robustness_baselines.json` |
| Realistic (AWGN + baseline wander) and streaming re-anchoring noise (Fig. S8, Table S6) | `task_d_realistic_streaming_noise.py` | `results/experiments/task_d_realistic_streaming_noise.{csv,json}`, `results/figures/figureS_realistic_streaming_noise.svg` |
| Control-gain γ sweep incl. control ablation (Fig. S7) | `task_b_gamma_sweep.py` | `results/experiments/task_b_gamma_sweep.csv`, `results/figures/figureS_gamma_control_sweep.svg` |
| Control-net output-activation comparison (Table S4) | `task_c_activation.py` | `results/experiments/task_c_activation.{csv,json,tex}` |
| Koopman latent mode analysis and mode ablation (Fig. S6, mode-ablation table) | `task8_mode_analysis.py`, `task_e_mode_ablation_export.py`, `figure_mode_analysis.py` | `results/mode_frequencies.json`, `results/mode_ablation.json`, `results/experiments/table_mode_ablation.{csv,tex}`, `results/figures/figureS6_mode_analysis.svg` |
| Closed-loop re-anchoring (Fig. S10) | `task_f_clean_reanchoring.py` | `results/experiments/task_f_clean_reanchoring.csv`, `results/figures/figureS_clean_reanchoring.svg` |
| Latent-dimension / hyper-parameter ablation (Fig. S4) | `task6_latent_ablation.py`, `figure_latent_hparam.py` | `results/latent_ablation.json`, `results/figures/figureS4.svg` |
| Reconstruction-weight sensitivity | `task7_gamma_sensitivity.py` | `results/gamma_sensitivity.json` |
| LHS design distances / leakage check (Table S3) | `task2_lhs_distance.py` | `results/lhs_distance_check.json`, `results/table_S3_lhs_distances.{csv,tex}` |
| Statistics and R² distributions (per-trajectory) | `task1_stats_exact.py`, `task9_r2_distribution.py` | `results/stats_exact.json`, `results/r2_per_trajectory.json` |
| Figure 5 (b–f) and eight-model comparison tables; Fig. S9 | `build_tables_figures.py` | `results/experiments/table{3,4,5}_*.{csv,tex}`, `results/figures/figure5_rev2_comparison.svg`, `figure6_rev2_noise.svg`, `figureS9_*.svg` |
| Figure 5 (a, g–j) | `regenerate_panels_a_gj.py` | `results/figures/figure5a_rev2.svg`, `figure5_panels_gj_rev2.svg` |
| Figures 2–4, S1–S3, S5 | `notebooks/generate_figures_and_tables_revised_rev2.ipynb` | `results/figures/figure{2,3a-c,3d,4}.svg`, `supplementary_figure{1,2,3}.svg`, `figureS_all_signals.svg` |
| Training times (Table 3) | `../measure_training_times.py` | `results/training_times.json` |

A complete table/figure/number → script → result-file map is in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Training from scratch (CLI)

```bash
cardiokoop create_dataset --help                 # raw_data/csv_sims -> data/ splits (+ normalisation statistics)
cardiokoop optuna_koopman_search --exp-folder results/optuna_koopman --local     # hyper-parameter search (Optuna)
cardiokoop train_koopman --exp-folder results/koopman_exp1
cardiokoop train_lstm --use-control --exp-folder results/lstm_ctrl              # also train_gru / train_bilstm
cardiokoop postprocess_koopman --exp-folder results/koopman                     # also postprocess_lstm / _gru / _bilstm
```

Each `train_*` command accepts `-f, --exp-folder`; `postprocess_*` accept `--model-path`, `--error-csv` and `--data-folder`.
Training the Koopman model took 47 min on the GPU listed below (RNN baselines 289–401 min). The seed-sensitivity re-run of the best
Optuna trial is in `results/seed_sensitivity_runs/`.

## Data

The dataset consists of 500 simulations of the lumped-parameter cardiovascular model (Latin-hypercube sampled parameters,
`raw_data/csv_sims/sim_*.csv` + `sim_*_params.json`), split with seed 42 into training / validation / test sets of trajectories
(`data/csv_data_500_12sigs_{train1,val1,test1}_{x,u}.csv`; 12 signals × 1500 steps, control input `u`) with the normalisation
statistics `data/normalization_{mean,std}.npy`. The splits are also published as the Zenodo dataset record
[10.5281/zenodo.21163127](https://doi.org/10.5281/zenodo.21163127). All `*.csv` files are stored with **Git LFS**: run `git lfs pull`
after cloning, or download the splits from Zenodo; the Zenodo software archive
[10.5281/zenodo.22776287](https://doi.org/10.5281/zenodo.22776287) contains all files resolved (≈ 1.2 GB) with a `MANIFEST.sha256`.

## Reproducibility notes

* The unit of analysis is the test trajectory (n = 50); 95 % CIs are 1.96·SD/√50 across trajectories, hypothesis tests
  (Shapiro–Wilk, Friedman across the six dynamical models, Bonferroni-corrected two-sided Wilcoxon) are paired across trajectories.
* The 1499-step autoregressive rollout amplifies floating-point differences between platforms; CPU float32 / CPU float64 / GPU runs
  of the Koopman checkpoint give 17.46 / 17.50 / 17.54 %RMSE — within 0.1 pp, all rounding to 17.5 %, with no rounded value or
  statistical conclusion changed. The committed `results/tables/` were generated on CPU in float64 (`run_info.json`).
* The MLP baseline is evaluated from its stored test predictions (`results/mlp/mlp_postprocessing_results.pkl`) because its diverging
  float32 rollout is platform-sensitive; every run re-rolls the checkpoint and reports the deviation in `mlp_recompute_check.json`
  (`--recompute-mlp` re-rolls instead of loading).
* The CI workflow builds the Docker image, regenerates the tables on an independent x86-64 runner and fails on any difference beyond
  the last printed digit; the verified image is pushed to `ghcr.io/cellularsyntax/cardiokoop`.
* Details — R² conventions (pooled / per-signal / signal-averaged), per-table source map, checkpoint identity, CI pass criteria and
  the image digest: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Environment

Training, the Optuna search and the timing measurements were run on an AMD Ryzen 7 3700X (64 GB RAM) with an NVIDIA GeForce
RTX 3070 Ti (8 GB) using **PyTorch 2.6** and **Optuna 4.7.0**; `environment/requirements-pinned.txt` pins every dependency of
`pyproject.toml` to a release compatible with PyTorch 2.6, and the `Dockerfile` (python:3.11-slim, CPU build of torch 2.6.0) is the
pinned environment of the CI check. See [`environment/README.md`](environment/README.md).

## License and contact

MIT License — see [`LICENSE`](LICENSE). Copyright 2025 The Neurocardiac Lab, Center for Medical Physics and Biomedical Engineering,
Medical University of Vienna. Questions and issues: [GitHub issues](https://github.com/CellularSyntax/CARDIOKOOP/issues) or the
corresponding author of the article.
