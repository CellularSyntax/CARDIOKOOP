<p align="center">
  <img src="assets/logo.png" alt="CardioKoop" width="1000"/>
</p>

# **CARDIOKOOP**  
*Control-aware Koopman deep learning framework for real-time hemodynamic forecasting and cardiovascular digital twin applications.*

CardioKoop provides a high-performance pipeline to learn **Koopman eigenfunctions** from multivariate cardiovascular simulations, enabling **real-time surrogate modeling** of pressures, volumes, and flow signals. The framework includes tools for dataset generation (via a validated lumped-parameter model), neural Koopman operator training, benchmarking against RNN baselines (LSTM, GRU, BiLSTM), and visualization scripts for reproducible analysis.

---

## **Citation**
If you use this code, data, or pre-trained models in your research, please cite:  

**Haberbusch M., Brandt L.B., Aprile M., Lung D., Kuijper A., Moscato F.** Real-Time Hemodynamic Prediction via Control-Aware Koopman Operator Models.  *Journal Name Goes Here*, 2025. [DOI pending]

---


## Table of Contents

- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Training Models](#training-models)
- [Hyperparameter Search](#hyperparameter-search)
- [Post-processing Results](#post-processing-results)
- [Visualization Notebooks](#visualization-notebooks)

## Installation

```bash
git clone https://github.com/CellularSyntax/CardioKoop.git
cd CardioKoop
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
# Notebook dependencies
pip install fastdtw pandas seaborn notebook
```

## Data Preprocessing

Slice raw simulation CSVs into normalized train/validation/test windows:

1. Configure parameters in `src/cardiokoop/data/create_dataset.py`:
   - `input_dir`: path to raw CSV files (default: `./raw_data/res_130625_500/csv_sims`)
   - `output_base`: output folder for generated CSVs (default: `./data/csv_data_500_12sigs`)
   - `t_start`, `t_end`, `DOWNSAMPLE_FACTOR`, etc.
2. Run the CLI command:

```bash
cardiokoop create_dataset
```

Generated files in `data/`:
- `csv_data_500_12sigs_train*_x.csv` / `*_u.csv`
- `csv_data_500_12sigs_val*_x.csv` / `*_u.csv`
- `normalization_mean.npy`, `normalization_std.npy`

## Training Models

Use the `cardiokoop` CLI to train various models:

| Command           | Description                       |
|-------------------|-----------------------------------|
| `train_koopman`   | Train a Koopman network           |
| `train_lstm`      | Train an LSTM baseline            |
| `train_bilstm`    | Train a BiLSTM baseline           |
| `train_gru`       | Train a GRU baseline              |

Each command accepts `-f, --exp-folder` to set the output directory. Example:

```bash
cardiokoop train_koopman --exp-folder results/koopman_exp1
cardiokoop train_lstm --use-control --exp-folder results/lstm_ctrl
```

## Hyperparameter Search

Run an Optuna search for Koopman network hyperparameters:

```bash
cardiokoop optuna_koopman_search --exp-folder results/optuna_koopman
```

## Post-processing Results

Compute metrics, generate plots, and save a results pickle for further analysis:

| Command               | Description                          |
|-----------------------|--------------------------------------|
| `postprocess_koopman` | Post-process Koopman experiment      |
| `postprocess_lstm`    | Post-process LSTM experiment         |
| `postprocess_bilstm`  | Post-process BiLSTM experiment       |
| `postprocess_gru`     | Post-process GRU experiment          |

Example:

```bash
cardiokoop postprocess_koopman --exp-folder results/koopman_exp1
```

## Visualization Notebooks

Interactive notebooks for analyzing and comparing results:

- `notebooks/evaluate_koopman_model.ipynb` — evaluate and plot a single Koopman model
- `notebooks/compare_model_performance.ipynb` — summary tables and figures across models

Launch with Jupyter:

```bash
jupyter notebook notebooks/evaluate_koopman_model.ipynb
jupyter notebook notebooks/compare_model_performance.ipynb
# or use Jupyter Lab:
jupyter lab notebooks/compare_model_performance.ipynb
```
