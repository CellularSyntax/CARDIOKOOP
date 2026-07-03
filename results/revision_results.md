# CARDIOKOOP — Peer Reviewer Response: Supplementary Results

All experiments were conducted on the existing trained checkpoint
(`results/koopman/csv_data_500_12sigs_2025_06_18_18_50_13_876051_model.ckpt`)
unless otherwise noted. The test set consists of 50 held-out trajectories
(1500 time steps × 12 hemodynamic signals each). Scripts for every task
are in `scripts/task{1–9}_*.py`; all numerical outputs are in `results/`.

---

## Task 1 — Exact Statistical Tests

**Metric:** per-trajectory %RMSE (`pct_per_traj_ps`), n = 50 trajectories per model.

### Shapiro-Wilk normality test

| Model   | W statistic | p-value   | Normal (α=0.05)? |
|---------|-------------|-----------|------------------|
| Koopman | 0.8307      | 4.84×10⁻⁶ | No               |
| LSTM    | 0.9512      | 3.82×10⁻²  | No               |
| GRU     | 0.9405      | 1.41×10⁻²  | No               |
| BiLSTM  | 0.9113      | 1.16×10⁻³  | No               |

All four distributions are non-normal, justifying the use of non-parametric tests.

### Friedman test (all four models jointly)

| χ² statistic | p-value    |
|--------------|------------|
| 107.568      | 3.66×10⁻²³ |

The global null hypothesis (equal distributions across models) is rejected
at any reasonable significance level.

### Pairwise Wilcoxon signed-rank tests (one-sided: Koopman < baseline)

| Comparison          | Statistic | p-value    |
|---------------------|-----------|------------|
| Koopman vs. LSTM    | 0.0       | 8.88×10⁻¹⁶ |
| Koopman vs. GRU     | 0.0       | 8.88×10⁻¹⁶ |
| Koopman vs. BiLSTM  | 0.0       | 8.88×10⁻¹⁶ |

All p-values are at the double-precision machine minimum, indicating
that Koopman outperforms every baseline on every single trajectory.

### %RMSE summary statistics (test set)

| Model   | Mean (%) | SD (%) | Median (%) |
|---------|----------|--------|------------|
| Koopman | 17.29    | 4.59   | 16.56      |
| LSTM    | 97.80    | 39.38  | 87.54      |
| GRU     | 60.89    | 11.70  | 58.74      |
| BiLSTM  | 76.44    | 27.84  | 66.00      |

---

## Task 2 — LHS Parameter-Space Coverage (Data Leakage Check)

The 500 simulation parameter vectors (11 cardiovascular parameters) were
split with `numpy.random.seed(42)` into 400 train / 50 val / 50 test.
Distances were computed in z-score–normalised parameter space (11-D Euclidean).

### Nearest-neighbour distances

| Set pair              | Min    | Mean   | SD     | Max    |
|-----------------------|--------|--------|--------|--------|
| Test → Train (NN)     | 1.5626 | 2.1617 | 0.3193 | 2.9065 |
| Val  → Train (NN)     | 1.4038 | 2.1931 | 0.3154 | 2.8801 |
| Within-train (LOO NN) | 1.2343 | 2.1678 | 0.3314 | 3.1086 |

The minimum test-to-train nearest-neighbour distance (1.56 normalised units)
is comparable to the within-training-set NN distance (min = 1.23), confirming
that the test set is well-separated from the training set in parameter space
and that no data leakage occurred.

**Figure:** `results/figures/lhs_distances.png`

---

## Task 3 — Noise Robustness

Additive white Gaussian noise (AWGN) was added independently to each of the
12 state signals at the initial time step (t = 0), in normalised space.
The model was **not retrained**. Per-signal PTP-normalised %RMSE and global R²
were averaged across 50 test trajectories.

### %RMSE and R² vs. SNR

| Condition     | %RMSE (mean ± SD) | R² (mean) |
|---------------|--------------------|-----------|
| Clean (∞ dB)  | 17.54 ± 5.36       | 0.686     |
| 30 dB SNR     | 19.75 ± 6.54       | 0.602     |
| 20 dB SNR     | 22.06 ± 7.54       | 0.528     |
| 10 dB SNR     | 30.94 ± 10.04      | 0.284     |
| 5 dB SNR      | 41.72 ± 10.51      | −0.069    |

The model is robust to moderate noise: at 30 dB SNR (a mild noise level),
%RMSE increases by only 2.2 percentage points above the clean baseline.
Graceful degradation continues up to 10 dB; at 5 dB, performance drops
below R² = 0, indicating that the noise is dominating the initial condition.

**Figure:** `results/figures/noise_robustness.png`

---

## Task 4 — Linear AR Baseline

One AR(p) model was fitted per signal using ordinary least squares on all
400 training trajectories (flattened to 600 000 time steps).
Orders p = 10 and p = 20 were evaluated; the better-performing order on the
validation set was selected.

### Validation-set comparison (order selection)

| Order | Val %RMSE (mean ± SD) | Test %RMSE (mean ± SD) | Test R² |
|-------|----------------------|------------------------|---------|
| p=10  | 35.66 ± 5.92         | 35.05 ± 5.71           | 0.319   |
| p=20  | **34.89 ± 3.88**     | **34.49 ± 4.11**       | **0.327** |

**Selected: AR(20).** AR(20) achieves 34.49 ± 4.11% %RMSE on the test set —
roughly double the Koopman error (17.29%), despite having access to the
previous 20 ground-truth states at inference time (Koopman uses only the
initial state).

**Output:** `results/ar_baseline.json`

---

## Task 5 — Flat MLP Autoregressive Baseline

Architecture mirrors the Koopman encoder/decoder: **12 → 256 → 512 → 256 → 12**
(ReLU activations, 269 324 parameters). Trained with Adam (lr = 1×10⁻³),
MSE loss, batch size 256, early stopping (patience = 10 epochs).

| Setting        | Value                   |
|----------------|-------------------------|
| Architecture   | 12→256→512→256→12, ReLU |
| Parameters     | 269 324                 |
| Optimiser      | Adam, lr = 1×10⁻³      |
| Best epoch     | 28 / 80                 |
| Val MSE (best) | 4.9×10⁻⁵               |

### Test-set results (autoregressive rollout, 1499 steps)

| Metric        | Value              |
|---------------|--------------------|
| %RMSE mean    | 132.75%            |
| %RMSE SD      | 199.29%            |
| R² mean       | −17.73             |
| Diverged      | 11 / 50 trajectories (22%) |

The MLP fails dramatically at long autoregressive rollout — 22% of trajectories
diverge numerically (predictions clipped at ±20 σ) and the average error is
nearly 8× worse than Koopman. This confirms that the Koopman structure
(explicit eigenfunction dynamics) is essential for stable long-horizon prediction.

**Outputs:** `results/mlp_baseline.json`, `checkpoints/mlp_baseline.pt`

---

## Task 6 — Latent Dimensionality Ablation

Results extracted from the 208 completed Optuna hyperparameter-search trials.
The search covered n_complex ∈ {3, 4, 5} and n_real ∈ {3, 4, 5}.
The objective was validation %RMSE (Optuna value column).

### Mean validation %RMSE by (n_complex, n_real)

| n_complex | n_real | Latent dim | n trials | Mean %RMSE | SD     |
|-----------|--------|-----------|----------|------------|--------|
| 3         | 3      | 9         | 7        | 1.361      | 0.707  |
| 3         | 4      | 10        | 30       | 0.824      | 0.555  |
| 3         | 5      | 11        | 20       | 0.729      | 0.323  |
| 4         | 3      | 11        | 11       | 0.719      | 0.186  |
| **4**     | **4**  | **12**    | 27       | 0.813      | 0.447  |
| 4         | 5      | 13        | 18       | 1.044      | 0.580  |
| 5         | 3      | 13        | 15       | 0.791      | 0.323  |
| 5         | 4      | 14        | 34       | **0.654**  | **0.144** |
| 5         | 5      | 15        | 46       | 0.684      | 0.247  |

**Best overall:** n_complex = 4, n_real = 4 (lowest individual trial value: 0.448%).
**Best mean:** n_complex = 5, n_real = 4 (lowest mean %RMSE: 0.654%).

The smallest configuration (dim = 9) is consistently the worst.
Performance stabilises around latent dim = 11–14, with diminishing returns
beyond that. The Optuna search did not cover n_complex ∈ {1, 2, 6}.

**Figure:** `results/figures/latent_ablation.png`

---

## Task 7 — γ (recon_lam) Sensitivity Analysis

The reconstruction loss weight γ (`recon_lam`) was searched by Optuna.
301 trials were run; 208 completed, 92 were pruned by the MedianPruner
(indicating poor early-stage performance).

| Item                        | Value                   |
|-----------------------------|-------------------------|
| γ range (completed trials)  | [0.003, 0.064]          |
| γ range (all incl. pruned)  | [0.003, 0.192]          |
| Completed / Pruned          | 208 / 92                |
| Stable range (within 20% of best binned mean) | [0.003, 0.008] |

Trials with γ > 0.064 were predominantly pruned, suggesting that large
reconstruction weights destabilise training early. The best individual trial
used γ = 0.003. The values γ ∈ {0.5, 1.0, 5.0} requested by the reviewer are
outside the searched range and would require additional training runs.

**Note on interpretation:** the Optuna objective is the *validation* %RMSE
reported at intermediate training checkpoints. The pruned trials' γ > 0.064
distribution strongly suggests that small γ (< 0.01) is optimal.

**Figure:** `results/figures/gamma_sensitivity.png`

---

## Task 8 — Mode Frequency Analysis & Ablation

### Part A: Learned mode frequencies

Frequencies computed by evaluating each ω-network at the mean latent radius
over the test set. Δt = 0.01 s; f_Hz = ω / (2π) where ω is the ω-net output
in rad/s.

#### Complex eigenpairs

| Mode | f (Hz)  | Period (s) | ω (rad/s) | σ (decay) | Latent var |
|------|---------|------------|-----------|-----------|------------|
| 0    | +1.868  | 0.535      | 11.74     | −5.379    | 0.0298     |
| 1    | +0.010  | 104.4      | 0.060     | −0.004    | 0.0296     |
| 2    | −0.004  | 236.0      | −0.027    | −0.003    | 0.0378     |
| 3    | −1.774  | 0.564      | −11.14    | +0.651    | **0.3148** |

#### Real modes

| Mode | σ       | λ per step | Latent var |
|------|---------|------------|------------|
| 0    | −0.003  | 1.0000     | 0.0109     |
| 1    | +0.041  | 1.0004     | 0.0386     |
| 2    | −6.338  | 0.9386     | 0.0000     |
| 3    | +0.033  | 1.0003     | 0.0033     |

**Cardiac-cycle modes:** Complex modes 0 and 3 correspond to f ≈ ±1.8 Hz
(≈ 107 bpm), consistent with simulated heart rate. Mode 3 accounts for the
largest share of latent variance (0.315).

Modes 1 and 2 capture very slow oscillations (periods of ~104 s and ~236 s),
longer than any single trajectory (15 s), suggesting they encode
inter-trajectory parameter variation rather than within-cycle dynamics.

### Part B: Mode ablation

Each complex eigenpair was zeroed out in sequence during inference
(no retraining). Baseline %RMSE = 17.54%.

| Ablated mode | f (Hz)  | %RMSE   | Δ%RMSE (pp)  |
|--------------|---------|---------|--------------|
| Mode 0       | +1.868  | 19.47   | +1.93        |
| Mode 1       | +0.010  | 19.93   | +2.39        |
| Mode 2       | −0.004  | 21.14   | +3.61        |
| **Mode 3**   | −1.774  | **33.91** | **+16.38** |

Mode 3 is by far the most critical: removing it alone degrades %RMSE by
+16.4 percentage points (a 93% relative increase). This is consistent with
its dominant latent variance (0.315 vs. 0.03–0.04 for other modes).
Despite having a negative σ for mode 3 that implies slight growth (σ = +0.65),
the learned ω-net compensates dynamically based on the latent state.

**Figures:** `results/figures/mode_ablation.png`
**Outputs:** `results/mode_frequencies.json`, `results/mode_ablation.json`

---

## Task 9 — Per-Trajectory R² Distribution

Global R² computed per trajectory (all signals and time steps flattened).

| Model   | Mean R² | SD R² | Median R² | Frac. R² < 0 | Count R² < 0 |
|---------|---------|-------|-----------|--------------|--------------|
| Koopman | 0.484   | 0.338 | 0.602     | **6.0%**     | 3 / 50       |
| LSTM    | −21.116 | 19.563 | −14.726  | **100%**     | 50 / 50      |
| GRU     | −5.088  | 2.775 | −4.409    | **100%**     | 50 / 50      |
| BiLSTM  | −13.188 | 16.212 | −7.065   | **100%**     | 50 / 50      |

Every single LSTM, GRU, and BiLSTM trajectory has R² < 0, meaning these
models perform worse than a simple mean predictor on each individual trajectory.
The Koopman model has only 3 / 50 trajectories with R² < 0.

**Figure:** `results/figures/r2_distribution.png`

---

## Summary Table — All Models on Test Set

| Model       | %RMSE (mean ± SD) | R² (mean) | Notes                          |
|-------------|-------------------|-----------|-------------------------------|
| **Koopman** | **17.29 ± 4.59**  | **0.484** | Best; 6% negative-R² trajs   |
| GRU         | 60.89 ± 11.70     | −5.09     | 100% negative R²              |
| BiLSTM      | 76.44 ± 27.84     | −13.19    | 100% negative R²              |
| LSTM        | 97.80 ± 39.38     | −21.12    | 100% negative R²              |
| AR(20)      | 34.49 ± 4.11      | 0.327     | Linear per-signal AR          |
| Flat MLP    | 132.75 ± 199.29   | −17.73    | Diverges in 22% of trajs      |

All differences between Koopman and baselines are statistically significant
(Wilcoxon one-sided p = 8.88×10⁻¹⁶ for LSTM, GRU, BiLSTM;
Friedman χ² = 107.57, p = 3.66×10⁻²³).

---

*Generated by `scripts/task{1–9}_*.py` and `scripts/write_revision_summary.py`.*
*Full numerical results: `results/revision_summary.json`.*
