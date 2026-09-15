# Computational environment

## Original workstation (all training runs, Optuna search, timing measurements, revision-1/2 results)

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 3700X (8 cores / 16 threads) |
| RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 3070 Ti, 8 GB VRAM |
| Operating system | **TO BE CONFIRMED BY THE AUTHORS** (Linux/Windows distribution and version) |
| Python | **TO BE CONFIRMED BY THE AUTHORS** (`python --version`; PyTorch 2.6 supports Python 3.9–3.13) |
| CUDA toolkit / driver | **TO BE CONFIRMED BY THE AUTHORS** (`nvidia-smi`; torch 2.6.0 wheels ship CUDA 11.8 / 12.4 / 12.6 runtimes) |
| PyTorch | 2.6 (as stated in the manuscript) |
| Optuna | 4.7.0 (as stated in the manuscript) |
| Other packages | see [`requirements-pinned.txt`](requirements-pinned.txt) |

Inference times and speed-ups reported in Table 3 were measured on this machine (GPU) when the
checkpoints were produced; `scripts/revision2/build_tables_figures.py` and
`scripts/revision3/export_manuscript_tables.py` re-use those published values rather than
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

## Revision-3 verification run (CPU)

`results/revision3/` was produced by `scripts/revision3/export_manuscript_tables.py` on a CPU-only
machine (macOS 15.6, Apple Silicon, Python 3.11.15, torch 2.12.1, numpy 2.4.6, scipy 1.17.1,
scikit-learn 1.9.0; Koopman rollout in float64). Wall-clock: ~40 s for the 50-trajectory Koopman
rollout, ~1 min in total. `results/revision3/run_info.json` records the exact platform of the
committed run. The authors can regenerate the folder on the original workstation with the same
command; only the last digit of a few Koopman values is expected to change (see the
"Numerical reproducibility" section of the main README).
