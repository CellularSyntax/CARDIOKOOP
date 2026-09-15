<p align="center">
  <img src="assets/logo.png" alt="CardioKoop" width="1000"/>
</p>

# **CARDIOKOOP**
*Control-aware Koopman deep learning framework for real-time hemodynamic forecasting and cardiovascular digital twin applications.*

[![Software DOI](https://img.shields.io/badge/Zenodo%20software-10.5281%2Fzenodo.22771300-blue)](https://doi.org/10.5281/zenodo.22771300)
[![Dataset DOI](https://img.shields.io/badge/Zenodo%20dataset-10.5281%2Fzenodo.21163127-blue)](https://doi.org/10.5281/zenodo.21163127)
[![Release](https://img.shields.io/badge/release-v1.1.0-green)](https://github.com/CellularSyntax/CARDIOKOOP/releases/tag/v1.1.0)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c)](environment/requirements-pinned.txt)
[![Reproduce manuscript tables (Docker)](https://github.com/CellularSyntax/CARDIOKOOP/actions/workflows/reproduce.yml/badge.svg?branch=main)](https://github.com/CellularSyntax/CARDIOKOOP/actions/workflows/reproduce.yml)
[![Container](https://img.shields.io/badge/ghcr.io-cellularsyntax%2Fcardiokoop-2496ed?logo=docker&logoColor=white)](https://github.com/CellularSyntax/CARDIOKOOP/pkgs/container/cardiokoop)

CARDIOKOOP learns **Koopman eigenfunctions** from multivariate cardiovascular simulations of a validated lumped-parameter model and
uses them as a **control-aware, real-time surrogate** for pressures, volumes and flows. The repository contains the package, the
frozen model checkpoints, the dataset splits, the raw simulations, every script that produced a number or a figure in the manuscript,
and the resulting tables.

---

## Citation

If you use this code, data or pre-trained models, please cite the article **and** the software archive
(machine-readable metadata in [`CITATION.cff`](CITATION.cff)):

> **Haberbusch M., Brandt L.B., Aprile M., Lung D., Kuijper A., Moscato F.**
> *Real-Time Hemodynamic Prediction via Control-Aware Koopman Operator Models.* **Array** (Elsevier), 2026, in press.

| Archive | DOI |
|---|---|
| Software, checkpoints, data and results — **v1.1.0** (this release; self-contained, all data files resolved) | [10.5281/zenodo.22771300](https://doi.org/10.5281/zenodo.22771300) |
| Software — all versions (concept DOI) | [10.5281/zenodo.21993894](https://doi.org/10.5281/zenodo.21993894) |
| Dataset — seed-42 train/validation/test splits | [10.5281/zenodo.21163127](https://doi.org/10.5281/zenodo.21163127) |

> **Note on Zenodo versions.** Record 10.5281/zenodo.22771013 is a GitHub-integration snapshot of tag v1.1.0 that contains Git-LFS pointer files only; the self-contained v1.1.0 archive with all data files resolved is 10.5281/zenodo.22771300. The concept DOI 10.5281/zenodo.21993894 always resolves to the latest version.


---

## Table of contents

- [Repository layout](#repository-layout)
- [Installation and environment](#installation-and-environment)
- [Reproduce with Docker](#reproduce-with-docker)
- [Frozen checkpoints](#frozen-checkpoints)
- [Reproducing the manuscript tables and figures](#reproducing-the-manuscript-tables-and-figures)
- [Numerical reproducibility](#numerical-reproducibility)
- [Git-LFS data files and the Zenodo archive](#git-lfs-data-files-and-the-zenodo-archive)
- [Training and post-processing from scratch (CLI)](#training-and-post-processing-from-scratch-cli)
- [Revision notes](#revision-notes)

## Repository layout

```
CARDIOKOOP/
├── src/cardiokoop/            Python package (CLI `cardiokoop`)
│   ├── data/create_dataset.py     raw simulations -> normalised train/val/test splits
│   ├── network/networkarch.py     KoopmanNetControl_v2 (encoder, auxiliary omega nets, control net, decoder)
│   ├── training/                  train_koopman / train_lstm / train_gru / train_bilstm
│   ├── optim/                     Optuna hyper-parameter search for the Koopman network
│   └── postprocessing/            metric computation and result pickles for every architecture
├── data/                      seed-42 splits (Git-LFS): csv_data_500_12sigs_{train1,val1,test1}_{x,u}.csv,
│                              normalization_{mean,std}.npy, cardiovascular_parameter_space.csv
├── raw_data/csv_sims/         the 500 full-order lumped-parameter simulations (signals + parameters; Git-LFS)
├── checkpoints/               mlp_baseline.pt, dlinear_baseline.pt, nlinear_baseline.pt
├── results/
│   ├── koopman/               FROZEN Koopman checkpoint + val1 post-processing pickle (revision 1)
│   ├── lstm/ gru/ bilstm/     RNN baseline checkpoints + test1 post-processing pickles
│   ├── dlinear/ nlinear/      direct linear baselines, test1 post-processing pickles
│   ├── mlp/                   frozen MLP baseline test1 predictions (v1.2.0; see REVISION3_NOTES.md section 8)
│   ├── *.json                 revision-1 result files (noise robustness, AR/MLP baselines, ablations, ...)
│   ├── revision2/             revision-2 tables (CSV/TeX) and JSONs   (scripts/revision2/)
│   ├── revision3/             revision-3 manuscript tables 3-5, statistics, R2 conventions, test1 Koopman pickle
│   ├── figures/               all manuscript / supplementary figure panels (SVG/PNG)
│   └── optuna_runs/           Optuna search history (301 trials)
├── scripts/                   revision-1 experiments (task1_*.py ... task9_*.py, export_tables.py, ...)
│   ├── revision2/             revision-2 experiments + table/figure builders (`run_all.py`)
│   ├── revision3/             `export_manuscript_tables.py` (single-run export of Tables 3-5 + statistics),
│   │                          `compare_results.py` (committed-vs-fresh check at manuscript rounding)
│   └── reproduce.sh           Docker/CI entry point: fetch splits from Zenodo -> export -> compare
├── notebooks/                 figure/table notebooks (see "Reproducing ..." for which one is current)
├── environment/               requirements-pinned.txt, README.md (hardware, CUDA, Python)
├── Dockerfile  .github/workflows/reproduce.yml   container + CI reproduction check (see "Reproduce with Docker")
├── CITATION.cff  REVISION2_RESULTS.md  REVISION3_NOTES.md  LICENSE (MIT)  pyproject.toml
```

## Installation and environment

Software stated in the manuscript: **PyTorch 2.6**, **Optuna 4.7.0**, CUDA GPU (NVIDIA RTX 3070 Ti, 8 GB) on an
AMD Ryzen 7 3700X workstation with 64 GB RAM. The Python interpreter version of the original workstation is to be confirmed by
the authors (PyTorch 2.6 supports Python 3.9-3.13); see [`environment/README.md`](environment/README.md).

```bash
git clone https://github.com/CellularSyntax/CARDIOKOOP.git
cd CARDIOKOOP
git lfs pull                       # resolves the 523 *.csv data/result files (not needed for the Zenodo archive)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # or `pip install torch==2.6.0` (CPU)
pip install -r environment/requirements-pinned.txt
pip install -e .
cardiokoop --help
```

`environment/requirements-pinned.txt` pins every dependency of `pyproject.toml` to a concrete release compatible with PyTorch 2.6.
A CPU-only installation is sufficient to regenerate all tables (about one minute, see below); a GPU is only needed to retrain models
or to rerun the Optuna search.

## Reproduce with Docker

The repository ships a `Dockerfile` (`python:3.11-slim`, CPU build of torch 2.6.0, every pin of
`environment/requirements-pinned.txt`, the package, the frozen checkpoints and the committed results). Its default command,
[`scripts/reproduce.sh`](scripts/reproduce.sh), (a) downloads the seed-42 `test1`/`train1` splits and the normalisation
statistics from the Zenodo dataset record [10.5281/zenodo.21163127](https://doi.org/10.5281/zenodo.21163127) when `data/*.csv`
are Git-LFS pointer files (MD5-verified against the record; `git lfs pull` is used as fallback), (b) runs
`scripts/revision3/export_manuscript_tables.py --out-dir /workspace/out` and (c) runs
`scripts/revision3/compare_results.py`, which compares every number of Tables 3, 4, 5, `statistics.json` and
`r2_conventions.json` with the committed `results/revision3/` **at manuscript rounding** (%RMSE/RMSE/CIs to 1 decimal, R² to 2
decimals, Friedman χ² to 2 decimals, p-values within a factor of 2, trajectory counts exact) and exits non-zero on any difference
beyond the last printed digit.

```bash
# build and run locally (≈ 5 min build, a few minutes run on a laptop CPU)
docker build -t cardiokoop .
docker run --rm -v "$PWD/out:/workspace/out" cardiokoop
# or use the CI-verified image
docker pull ghcr.io/cellularsyntax/cardiokoop:latest
docker run --rm -v "$PWD/out:/workspace/out" ghcr.io/cellularsyntax/cardiokoop:latest
```

`out/` then contains the regenerated `table3_overall.*`, `table4_per_signal_full.*`, `table5_noise.*`, `statistics.json`,
`r2_conventions.json`, `koopman_test1_postprocessing_results.pkl`, `run_info.json` and `compare_report.md` (the full
committed-vs-reproduced diff table). Useful options: `-e REPRO_THREADS=4` (torch CPU threads), `-e REPRO_DTYPE=float32`,
`-e REPRO_STRICT=1` (also fail on a ±1 change in the last printed digit), `-e CARDIOKOOP_DATA_DIR=/data -v <local splits>:/data`
(reuse already downloaded splits).

**What the CI check proves.** The GitHub Actions workflow [`reproduce.yml`](.github/workflows/reproduce.yml) (badge above) builds
this image from the current commit, runs the container with the repository mounted read-only, uploads `out/` as the workflow
artifact `reproduction-tables`, and — only if the comparison passes — pushes the image to
`ghcr.io/cellularsyntax/cardiokoop` (tags `sha-<short>`, `latest` on `main`, and the git tag on `v*` releases; the image digest
is written to `out/image_digest.txt` and to the job summary). A green run therefore certifies that the frozen checkpoints, the
committed baseline predictions, the Zenodo splits and the code in the repository regenerate every number of the manuscript's
Tables 3–5 and statistics paragraph, at the precision at which they are printed, in a pinned environment on an independent
x86-64 machine. Off-by-one differences in the last printed digit that arise from the 1499-step float rollout on a different CPU
are tolerated by the check but listed explicitly in `compare_report.md` and in [`REVISION3_NOTES.md`](REVISION3_NOTES.md),
section "Container verification".

## Frozen checkpoints

All manuscript numbers come from these committed weights; nothing is retrained by the reproduction scripts (except the analytic
OLS fit of the AR(20) baseline and the two small DLinear/NLinear models, which `scripts/revision2/task_a_dlinear_nlinear.py`
trains to convergence in about two minutes and stores in `checkpoints/`).

| Model | File | Parameters |
|---|---|---|
| **Koopman (final model)** | `results/koopman/csv_data_500_12sigs_2025_06_18_18_50_13_876051_model.ckpt` | **17,239,600** |
| LSTM | `results/lstm/csv_data_500_12sigs_20250625_184322_model.ckpt` | 1,083,404 |
| GRU | `results/gru/csv_data_500_12sigs_20250625_170206_model.ckpt` | 814,092 |
| BiLSTM | `results/bilstm/csv_data_500_12sigs_20250625_204256_model.ckpt` | 2,166,796 |
| MLP (autoregressive, clip 20) | `checkpoints/mlp_baseline.pt` | 269,324 |
| DLinear / NLinear | `checkpoints/dlinear_baseline.pt`, `checkpoints/nlinear_baseline.pt` | 429,120 / 214,560 |
| AR(20) | analytic OLS fit on train1 (no file; 240 coefficients) | 240 |

The Koopman checkpoint stores the hyper-parameters (`ckpt["params"]`) together with the weights; `scripts/check_ckpt_params.py`
prints the parameter counts. The seed-sensitivity re-run of the best Optuna trial is in `results/seed_sensitivity_runs/`.

## Reproducing the manuscript tables and figures

Evaluation split for **every** reported number: the seed-42 **test1** split (50 trajectories x 1500 steps x 12 signals, 1499-step
autoregressive horizon). The statistical unit is the trajectory (n = 50); 95 % CIs are 1.96 * SD / sqrt(50). A complete
table/figure/number -> script -> result-file map is in [`REVISION3_NOTES.md`](REVISION3_NOTES.md).

### Revision 3 — Tables 3, 4, 5 and the statistics (one command)

```bash
python scripts/revision3/export_manuscript_tables.py          # float64 Koopman rollout on CPU or GPU, ~1 min on CPU
```

Writes `results/revision3/`: `table3_overall.{md,json,tsv}`, `table4_per_signal_full.{md,json,tsv}` (8 models x 12 signals:
RMSE, %RMSE, R2 with 95 % CI, Bland-Altman bias and limits of agreement), `table5_noise.{md,json,tsv}`, `statistics.json`
(Shapiro-Wilk, Friedman over the six dynamical models, paired two-sided Wilcoxon Koopman vs. each baseline with Bonferroni x5),
`r2_conventions.json` (pooled / per-signal / signal-averaged R2 and negative-R2 counts per model),
`koopman_test1_postprocessing_results.pkl` (the Koopman **test1** evaluation, same keys as the baseline pickles), `run_info.json`
and `mlp_recompute_check.json`. Add `--out-dir <dir>` to write elsewhere (used by the Docker check) and `--recompute-mlp` to re-roll
the MLP baseline instead of loading the frozen `results/mlp/mlp_postprocessing_results.pkl`.

### Revision 2 — DLinear/NLinear, noise, gamma sweep, activation, mode ablation, Figures 5, S4, S6-S10

```bash
python scripts/revision2/run_all.py                           # all steps; or e.g. `run_all.py a build panels`
```

Steps (in order): `task_a_dlinear_nlinear.py` (trains + evaluates DLinear/NLinear), `task_b_gamma_sweep.py` (Fig. S7),
`task_c_activation.py` (Table S4), `task_d_realistic_streaming_noise.py` (Fig. S8, Table S6), `task_e_mode_ablation_export.py`
(mode-ablation table), `task_f_clean_reanchoring.py` (Fig. S10), `build_tables_figures.py` (revision-2 Tables 3/4/5 CSV/TeX,
Figure 5 b-f and the noise panels i-j, Fig. S9), `regenerate_panels_a_gj.py` (Figure 5 a, g-j), `figure_mode_analysis.py` (Fig. S6),
`figure_latent_hparam.py` (Fig. S4). Outputs: `results/revision2/`, `results/figures/*_rev2*`, `results/figures/figureS*`.
Narrative and per-item mapping: [`REVISION2_RESULTS.md`](REVISION2_RESULTS.md).

### Revision 1 — baselines, noise harness, ablations, statistics (`scripts/task*.py`)

| Script | Produces | Used for |
|---|---|---|
| `task1_stats_exact.py` | `results/stats_exact.json` | superseded by `results/revision3/statistics.json` (six models) |
| `task2_lhs_distance.py` | `results/lhs_distance_check.json`, `results/table_S3_lhs_distances.{csv,tex}` | Table S3 (LHS design distances) |
| `task3_noise_robustness.py` | `results/noise_robustness.json` | Table 5 noisy rows (Koopman), Figure 5 i-j |
| `task3b_noise_robustness_baselines.py` | `results/noise_robustness_baselines.json` | Table 5 noisy rows (GRU/LSTM/BiLSTM/MLP/AR) |
| `task4_ar_baseline.py` | `results/ar_baseline.json` | AR(20) order selection |
| `task5_mlp_baseline.py` | `results/mlp_baseline.json`, `checkpoints/mlp_baseline.pt` | MLP baseline |
| `task6_latent_ablation.py` | `results/latent_ablation.json` | Fig. S4 (latent-space / hyper-parameter sensitivity) |
| `task7_gamma_sensitivity.py` | `results/gamma_sensitivity.json` | gamma (reconstruction weight) sensitivity |
| `task8_mode_analysis.py` | `results/mode_frequencies.json`, `results/mode_ablation.json` | Fig. S6 (Koopman latent mode analysis), mode-ablation table |
| `task9_r2_distribution.py` | `results/r2_per_trajectory.json` | superseded by `results/revision3/r2_conventions.json` |
| `measure_training_times.py` | `results/training_times.json` | Table 3 training times (MLP, AR) |
| `export_tables.py` | LaTeX tables from the post-processing pickles | revision-1 tables |

### Notebooks

* `notebooks/generate_figures_and_tables_revised_rev2.ipynb` — **current**: Figures 2-4 and the supplementary figures that are
  not produced by the scripts above (uses `notebooks/notebook_utils.py`).
* `notebooks/generate_figures_and_tables_revised.ipynb` — revision 1, **superseded** (kept for provenance).
* `notebooks/generate_figures_and_tables.ipynb` — original submission, **superseded** (kept for provenance; its Koopman
  row is the val1 evaluation, see REVISION3_NOTES.md).

## Numerical reproducibility

The Koopman surrogate is rolled out autoregressively for 1499 steps, which amplifies floating-point differences between
platforms. Re-running the frozen checkpoint gives a mean %RMSE of **17.46 (CPU, float32)**, **17.50 (CPU, float64)** and
**17.54 (GPU, authors' workstation)** — differences of <= 0.1 percentage points, all rounding to 17.5 %; the pooled R2 is 0.69 in
every case, the per-signal Koopman values of Table 4 agree to <= 0.1 pp, and every statistical conclusion is unchanged
(Wilcoxon W = 0 for GRU/LSTM/BiLSTM/AR(20) on all platforms). The committed `results/revision3/` files were generated on CPU in
float64 (`run_info.json` records the platform); the RNN, DLinear and NLinear predictions are read from the committed pickles and are
therefore bit-identical everywhere. Use `--dtype float64` (default) for the most platform-independent result.
The autoregressive **MLP baseline** is the exception: its float32 rollout (clip ±20) diverges and amplifies CPU-architecture
differences beyond manuscript rounding (x86-64: %RMSE 132.8 vs. 132.2 on the committed platform), so since v1.2.0 its test1
predictions are frozen in `results/mlp/mlp_postprocessing_results.pkl` (bit-identical to the committed run) and loaded by
`export_manuscript_tables.py` by default; `--recompute-mlp` re-rolls the checkpoint, and every run writes
`mlp_recompute_check.json` with the deviation of a fresh float32/float64 rollout (see `REVISION3_NOTES.md`, section 8).

## Git-LFS data files and the Zenodo archive

All 523 `*.csv` files (`data/` splits 251 MB, `raw_data/csv_sims/` 662 MB, `results/**/*.csv`) are stored with **Git LFS**
(`.gitattributes`: `*.csv filter=lfs`). After cloning, run `git lfs pull`; without it these files are small pointer files starting
with `version https://git-lfs.github.com/spec/v1`. GitHub's "Download ZIP" and the v1.0.0 Zenodo snapshot made by the GitHub
integration contain those pointers only. The **v1.1.0 Zenodo archive (10.5281/zenodo.22771300)** is self-contained: every pointer is
replaced by the resolved file (about 0.9 GB uncompressed) and a `MANIFEST.sha256` lists every file. Revision-2/3 result files are
written as `.json`/`.md`/`.tsv`/`.tex` so that they are never LFS-filtered.

## Training and post-processing from scratch (CLI)

```bash
cardiokoop create_dataset --help                 # raw_data/csv_sims -> data/ splits (+ normalisation statistics)
cardiokoop train_koopman --exp-folder results/koopman_exp1
cardiokoop train_lstm --use-control --exp-folder results/lstm_ctrl      # also train_gru / train_bilstm
cardiokoop optuna_koopman_search --exp-folder results/optuna_koopman --local
cardiokoop postprocess_koopman --exp-folder results/koopman             # also postprocess_lstm / _gru / _bilstm
```

Each `train_*` command accepts `-f, --exp-folder`; `postprocess_*` accept `--model-path`, `--error-csv` and `--data-folder`.
Training the final Koopman model took 47 min on the workstation GPU (RNN baselines 289-401 min).

## Revision notes

* [`REVISION2_RESULTS.md`](REVISION2_RESULTS.md) — revision-2 experiments (DLinear/NLinear, realistic and streaming noise,
  gamma sweep, activation comparison, mode ablation) and their reviewer-item mapping.
* [`REVISION3_NOTES.md`](REVISION3_NOTES.md) — revision-3 map of every manuscript table/figure/number to its script and result
  file, the R2 conventions, the val-vs-test pickle issue and its resolution, the recomputed statistics and the CPU-vs-GPU note.

## License

MIT License — see [`LICENSE`](LICENSE). Copyright 2025 The Neurocardiac Lab, Center for Medical Physics and Biomedical
Engineering, Medical University of Vienna.
