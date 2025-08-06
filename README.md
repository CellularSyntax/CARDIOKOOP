<p align="center">
  <img src="assets/logo.png" alt="CardioKoop" width="1000"/>
</p>

# **CARDIOKOOP**  
*Control-aware Koopman deep learning framework for real-time hemodynamic forecasting and cardiovascular digital twin applications.*

CARDIOKOOP provides a high-performance pipeline to learn **Koopman eigenfunctions** from multivariate cardiovascular simulations, enabling **real-time surrogate modeling** of pressures, volumes, and flow signals. The framework includes tools for dataset generation (via a validated lumped-parameter model), neural Koopman operator training, benchmarking against RNN baselines (LSTM, GRU, BiLSTM), and visualization scripts for reproducible analysis.

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
git clone https://github.com/CellularSyntax/CARDIOKOOP.git
cd CARDIOKOOP
git lfs pull # make sure you have git-lfs installed so you can load the data files from the repository
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Check if the CARDIOKOOP launches without problems: 

```bash
cardiokoop --help
```

You should see: 

<img width="780" height="334" alt="image" src="https://github.com/user-attachments/assets/05e253b8-3ee4-46e7-9168-e486a130cd9c" />

## Data Preprocessing

Slice raw simulation CSVs into normalized train/validation/test windows:

1. Run the CLI command:

```bash
cardiokoop create_dataset
```

Generated files in `data/`:
- `csv_data_500_12sigs_train*_x.csv` / `*_u.csv`
- `csv_data_500_12sigs_val*_x.csv` / `*_u.csv`
- `normalization_mean.npy`, `normalization_std.npy`

2. Run the CLI command to get slicing options: 

```bash
cardiokoop create_dataset --help
```


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

Run an local Optuna search for Koopman network hyperparameters:

```bash
cardiokoop optuna_koopman_search --exp-folder results/optuna_koopman --local
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
cardiokoop postprocess_koopman --exp-folder results/koopman
```

## Visualization Notebooks

Interactive notebook to reproduce all results presented in the manuscript:

- `notebooks/generate_figures_and_tables.ipynb`

Launch with Jupyter:

```bash
jupyter notebook notebooks/generate_figures_and_tables.ipynb
# or use Jupyter Lab:
jupyter lab notebooks/generate_figures_and_tables.ipynb
```
