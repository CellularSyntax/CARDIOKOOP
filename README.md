<p align="center">
  <img src="assets/logo_v1.png" alt="CardioKoop logo" width="200"/>
</p>

# CardioKoop

> Neural networks to learn Koopman eigenfunctions

This repository builds on the code for the paper ["Deep learning for universal linear embeddings of nonlinear dynamics"](https://www.nature.com/articles/s41467-018-07210-0) by Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton. Our work extends these methods to **PLACEHOLDER FOR YOUR PROJECT**.

---

## Installation

Clone this repository and install the package in editable mode:

```bash
# Clone the GitHub repository
git clone https://github.com/your-org/CardioKoop.git
cd CardioKoop

# Install the package
pip install -e .
```

To install development dependencies (testing, linting, formatting):

```bash
pip install -e .[dev]
```

---

## Training the networks

Train different network architectures using the unified CLI:

```bash
# Train Koopman network
cardiokoop train_koopman [--exp-folder RESULTS_DIR]

# Train LSTM baseline
cardiokoop train_lstm [--exp-folder RESULTS_DIR]

# Train BiLSTM baseline
cardiokoop train_bilstm [--exp-folder RESULTS_DIR]

# Train GRU baseline
cardiokoop train_gru [--exp-folder RESULTS_DIR]
```

Alternatively, without installing the CLI entry point:

```bash
PYTHONPATH=src python -m cardiokoop.training.train_koopman
```

---

## Hyperparameter optimization

Run an Optuna hyperparameter search for the Koopman network:

```bash
cardiokoop optuna_koopman_search [--trials N] [--local]
```

---

## Post-processing

Compute metrics and generate plots after training with the post-processing scripts:

```bash
cardiokoop postprocess_koopman               --exp-folder RESULTS_DIR
cardiokoop postprocess_koopman_reinit_pred   --exp-folder RESULTS_DIR --reinject-every N
cardiokoop postprocess_koopman_reinit_gt     --exp-folder RESULTS_DIR --reinject-every N
cardiokoop postprocess_lstm                  --exp-folder RESULTS_DIR
cardiokoop postprocess_bilstm                --exp-folder RESULTS_DIR
cardiokoop postprocess_gru                   --exp-folder RESULTS_DIR
```

---

## Reproducing manuscript results

To reproduce all figures, tables, and extract summary metrics for the manuscript, execute and export the notebooks, then generate LaTeX tables:

```bash
# 1. Create a clean Python environment (e.g. virtualenv or conda) and install dependencies
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn fastdtw jupyter nbconvert

# 2. Execute the analysis notebooks
PYTHONPATH=src jupyter nbconvert --to notebook --execute notebooks/compare_model_performance.ipynb --output executed_compare.ipynb
PYTHONPATH=src jupyter nbconvert --to notebook --execute notebooks/evaluate_koopman_model.ipynb --output executed_eval.ipynb

# 3. Generate LaTeX tables from postprocessing results
python scripts/export_tables.py
```

The generated tables will be saved under `tables/` as `table_summary.tex` (aggregate metrics) and `table_signal.tex` (per-signal metrics),
ready to \input into the manuscript.

---

## Citation

If you use this code, please cite our paper:

> **[PLACEHOLDER]** Your Name et al. *Title of Your Manuscript*, Journal (Year).

---

## Acknowledgements

This implementation builds on the original CardioKoop code by Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton (2018).
# cardiokoop