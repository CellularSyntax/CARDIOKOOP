# Computational environment

## Training hardware (all training runs, Optuna search, timing measurements)

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 3700X (8 cores / 16 threads) |
| RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 3070 Ti, 8 GB VRAM |
| PyTorch | 2.6 |
| Optuna | 4.7.0 |
| Other packages | versions as used: see [`requirements-pinned.txt`](requirements-pinned.txt) |

Inference times and speed-ups reported in Table 3 were measured on this machine (GPU) when the
checkpoints were produced; `scripts/experiments/build_tables_figures.py` and
`scripts/tables/export_manuscript_tables.py` re-use those published values rather than
re-measuring them (rationale documented in the scripts).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # or plain `pip install torch==2.6.0` for CPU
pip install -r environment/requirements-pinned.txt
pip install -e .
git lfs pull      # only when cloning from GitHub; the Zenodo archive already contains the resolved data files
```

## Table-export run (CPU)

`results/tables/` was produced by `scripts/tables/export_manuscript_tables.py` on a CPU-only
machine (macOS 15.6, Apple Silicon, Python 3.11.15, torch 2.12.1, numpy 2.4.6, scipy 1.17.1,
scikit-learn 1.9.0; Koopman rollout in float64). Wall-clock: ~40 s for the 50-trajectory Koopman
rollout, ~1 min in total. `results/tables/run_info.json` records the exact platform of the
committed run. Regenerating the folder on another machine changes at most the last digit of a few
Koopman values (see the "Reproducibility notes" section of the main README and
`docs/REPRODUCIBILITY.md`).

## Container

`Dockerfile` (python:3.11-slim, CPU wheel of torch 2.6.0, every pin of `requirements-pinned.txt`)
is the pinned environment used by the continuous-integration reproduction check
(`.github/workflows/reproduce.yml`); the verified image is published as
`ghcr.io/cellularsyntax/cardiokoop:v1.2.0`.
