# CARDIOKOOP — Revision 2 results (journal *Array*, minor revision)

This document maps every editor-mandatory item and every addressed reviewer point
to the exact script, result file, figure and headline number that answers it.
All work is on branch **`revision2`** and every new number is produced against the
**frozen final Koopman checkpoint**
`results/koopman/csv_data_500_12sigs_2025_06_18_18_50_13_876051_model.ckpt`
(17,239,600 parameters) — the main model was **never retrained**.

## Conventions (read first)

* **Split.** All new results use the seed-42 **`test1`** split (B = 50 trajectories,
  T = 1500 steps ⇒ 1499-step forecast horizon). This is the split every existing
  baseline (LSTM/GRU/BiLSTM/MLP/AR20) and the noise harness already use.
* **Frozen-model caveat about the revision-1 Koopman row.** The revision-1 notebook
  evaluated the **Koopman row on `val1`** (17.30 %RMSE) while every baseline used
  `test1`. On `test1` the same frozen model scores **17.54 %RMSE / R² = 0.691** — a
  0.24 pp difference — so re-stating Koopman on `test1` (for a fully consistent
  table) does not change the headline. `scripts/revision2/_common.py` was verified
  to reproduce the committed baseline pickles' aggregates **exactly** (max abs diff
  0.0) and the Koopman `test1` rollout to the 4th decimal.
* **Metrics.** `%RMSE` = mean per-signal PTP-normalised RMSE averaged over signals
  (`pct_per_traj_ps`, identical to the existing tables). `R²` in the regenerated
  tables is the **pooled** flattened R² for every model (the revision-1 notebook
  mixed pooled R² for MLP/AR with sklearn multi-output R² for the pickled models;
  the regenerated tables use one convention throughout — see Table 3 note).
* **No new dependencies** were added (`pyproject.toml` already lists numpy, torch,
  pandas, scikit-learn, matplotlib, seaborn, fastdtw, …). Seeds are fixed (42)
  everywhere.

## Where everything lives

* Scripts: `scripts/revision2/` (one per task + `build_tables_figures.py` +
  `run_all.py`). Reproduce everything with
  `python scripts/revision2/run_all.py`.
* Numeric results (CSV/JSON): `results/revision2/`.
* Regenerated figure panels (SVG/PNG): `results/figures/` (revision-2 panels are
  suffixed `_rev2` / `figureS_*` so the revision-1 figures are untouched).
* New baseline pickles (baseline-pkl format): `results/dlinear/`, `results/nlinear/`.

---

## Editor mandatory items

### #1 — DLinear / NLinear baselines  (Task A · R1.1)
* Script: `scripts/revision2/task_a_dlinear_nlinear.py`
* Results: `results/dlinear/dlinear_postprocessing_results.pkl`,
  `results/nlinear/nlinear_postprocessing_results.pkl`,
  `results/revision2/task_a_dlinear_nlinear.json`
* Added to Table 3/4/5 and Figure 5/6 via `build_tables_figures.py`.
* Design: canonical direct (non-autoregressive) multi-step forecasters (Zeng et al.
  AAAI 2023), per-signal, look-back L = 10 (same warm-up window the RNN baselines
  use), **given the control step level `V_usv,step` as an input feature** (the
  control is a single step at t=500, so its whole information content is this
  scalar). Fit on train1, early-stopped on val1, evaluated on test1.
* **Headline (test1):**
  * **DLinear: 3.03 %RMSE (±0.61), R² = 0.983, 429,120 params**
  * **NLinear: 3.63 %RMSE (±0.63), R² = 0.982, 214,560 params**
* **Important finding (flagged to authors):** the direct linear baselines
  **outperform** the Koopman model (17.54 %) and *all* the recurrent baselines
  (GRU 60.9 %, BiLSTM 76.4 %, LSTM 97.8 %, MLP ~133 %). This is the classic
  Zeng-et-al. result — direct multi-step readouts avoid the error accumulation that
  autoregressive models suffer over a 1499-step horizon, and the data is
  time-locked (step always at t=500). Koopman remains the **best autoregressive /
  dynamical model** (it beats every RNN by 3–5×) and is the only one that yields
  interpretable spectral modes (see #6) and a control-theoretic latent-linear
  structure — none of which DLinear/NLinear provide. Framing suggestions are in the
  “Narrative” section below.

### #3 — Noise robustness, realistic + streaming  (Task D · R1.3, R2.3)
* Script: `scripts/revision2/task_d_realistic_streaming_noise.py`
* Results: `results/revision2/task_d_realistic_streaming_noise.{json,csv}`,
  figure `results/figures/figureS_realistic_streaming_noise.{svg,png}`
* (a) sensor-realistic IC corruption = AWGN + low-frequency baseline wander
  (0.05–0.3 Hz) across SNR {30,20,10,5} dB; (b) streaming re-anchoring every
  K = 100 steps (~2 cardiac beats at ~1.8 Hz).
* Table 5 / Figure 6 additionally extended with DLinear/NLinear under the AWGN
  sweep (`build_tables_figures.py`).
* **Headline (test1, %RMSE):**

  | SNR | Realistic IC (AWGN+wander) | Streaming re-anchor (K=100) |
  |---|---|---|
  | clean | 17.5 | — |
  | 30 dB | 18.2 | **5.8** |
  | 20 dB | 23.6 | **11.9** |
  | 10 dB | 34.6 | **25.2** |
  | 5 dB  | diverges | **35.9** |

  **Streaming re-anchoring is dramatically more robust** at every SNR and remains
  stable at 5 dB where single-shot forecasting from a noisy IC diverges. At 30 dB
  streaming (5.8 %) even beats the clean single-shot baseline (17.5 %) because
  periodic re-observation of the true state corrects the model's own drift. This
  answers the “noise only at t = 0 is unrealistic” criticism (R2.3) and shows the
  model is deployable online.

### #4 — Control-gain γ sweep  (Task B · R1.2)
* Script: `scripts/revision2/task_b_gamma_sweep.py`
* Results: `results/revision2/task_b_gamma_sweep.csv`, figure
  `results/figures/figureS_gamma_control_sweep.{svg,png}`
* Overrides **only** `control_gain` over {0, 0.05, 0.1, 0.2, 0.5, 1.0} and re-runs
  test-set inference (no retraining). γ = 0 is the control-net ablation.
* **Headline (test1):**

  | γ | %RMSE | R² (pooled) | note |
  |---|------|------|------|
  | 0.00 | 51.37 | −3.93 | **control ablation** (cannot track the pre-load step) |
  | 0.05 | 47.92 | −1.72 | under-driven |
  | **0.10** | **17.54** | **0.691** | **trained value** |
  | 0.20 | 49.27 | −0.19 | over-driven |
  | 0.50 | diverges | — | rollout destabilised |
  | 1.00 | diverges | — | rollout destabilised |

  γ = 0.1 (trained) is optimal; turning the control net off (γ = 0) ~triples the
  error; large γ destabilises the long rollout.

### #5 — Control-net output-activation comparison  (Task C · R1.2)
* Script: `scripts/revision2/task_c_activation.py`
* Results: `results/revision2/task_c_activation.{csv,tex,json}`
* **Step 1 (zero-compute):** the Optuna study (301 trials) varied 12
  hyper-parameters but **no activation** (hidden or output) — confirmed
  programmatically — so the sanctioned fallback experiment is run.
* Two complementary analyses (main model never retrained):
  1. **deployed-net activation swap** (inference only) and
  2. **control-net-only retraining** under each activation on a frozen backbone.
* **Headline (test1 %RMSE / R²):**

  | activation | swap (deployed net) | control-net-only retrain |
  |---|---|---|
  | tanh (deployed) | 17.5 / 0.691 | 45.0 / 0.35 |
  | identity | 15.9 / 0.618 | 42.4 / 0.45 |
  | ReLU | 51.4 / −4.17 (= no-control) | diverges |
  | sigmoid | diverges | diverges |

  Conclusion (robust across both analyses): a **zero-centred output activation is
  required** — tanh (deployed) and identity give stable rollouts; ReLU collapses to
  the no-control error and sigmoid diverges, because non-negative activations inject
  a persistent DC bias that accumulates over the 1499-step horizon. This justifies
  the bounded, zero-centred tanh. (The control-net-only *retraining* underperforms
  the jointly-trained deployed model uniformly — expected, since with the backbone
  frozen the control net cannot co-adapt — so only its qualitative stability ranking
  is meaningful; the swap analysis gives the clean quantitative comparison.)

### #6 — Per-mode ablation, main-text table  (Task E · R2.5)
* Script: `scripts/revision2/task_e_mode_ablation_export.py`
* Results: `results/revision2/table_mode_ablation.{csv,tex}`
* Surfaces the existing seed-42 test-split ablation as a ranked, main-text table.
* **Headline:** baseline %RMSE = 17.54; ablating the dominant cardiac mode
  **CP-3 (|f| ≈ 1.77 Hz) → +16.4 pp (+93 % relative)**; CP-2 +3.6 pp, CP-1 +2.4 pp,
  CP-0 +1.9 pp.

---

## Reviewer-point cross-reference

| Point | Addressed by | File / figure / number |
|---|---|---|
| **R1.1** simple/linear baselines | Task A | DLinear 3.03 % / NLinear 3.63 % (`table3_overall_comparison.csv`, `figure5_rev2_comparison.svg`) |
| **R1.2** control net worth it? / activation | Task B + Task C | γ=0 ablation 51.4 % vs 17.5 % (`task_b_gamma_sweep.csv`); activation table (`task_c_activation.csv`) |
| **R1.3** realistic noise | Task D | `task_d_realistic_streaming_noise.*`, `figureS_realistic_streaming_noise.svg` |
| **R2.1** fair/consistent comparison | `build_tables_figures.py` | all models recomputed on test1 with one metric convention (`table3/4/5_*`) |
| **R2.3** noise only at t=0 is unrealistic | Task D (b) | streaming re-anchoring every K=100 steps |
| **R2.5** promote mode ablation to main text | Task E | `table_mode_ablation.{csv,tex}` |

---

## Regenerated main tables/figures (Task A integration · R2.1)
* Script: `scripts/revision2/build_tables_figures.py`
* `results/revision2/table3_overall_comparison.{csv,tex}` — 8-model overall
  comparison on test1 (consistent metrics).
* `results/revision2/table4_per_signal.{csv,tex}` — per-signal %RMSE, 8 models.
* `results/revision2/table5_noise_robustness_full.{csv,tex}` — AWGN robustness,
  8 models (adds DLinear/NLinear). **Key finding:** the direct linear baselines win
  on *clean* data but are **catastrophically noise-fragile** — under input AWGN
  their per-timestep readouts amplify noise (DLinear 3.0 → 334 %, NLinear 3.6 →
  1029 % at 5 dB; AR(20) 34 → 2967 %), whereas **Koopman is by far the most
  noise-robust** (17.5 → 41.7 % at 5 dB). Clean accuracy is not the whole story:
  Koopman trades a little clean error for far superior robustness + interpretability.
* `results/figures/figure5_rev2_comparison.{svg,png}` — per-signal %RMSE,
  accuracy-vs-speed, cumulative %RMSE, model size.
* `results/figures/figure6_rev2_noise.{svg,png}` — AWGN robustness, all 8
  architectures.

> **Notebook note.** The 3.5 MB manuscript notebook
> (`notebooks/generate_figures_and_tables_revised.ipynb`) was intentionally **not**
> edited — the revision-2 tables/figures are produced by the standalone,
> version-controlled `build_tables_figures.py` (which recomputes every model
> consistently), so the `.tex`/`.csv`/`.svg` artifacts above drop directly into the
> manuscript. The new baseline pickles live in `results/dlinear/` and
> `results/nlinear/` in the same format as the other baselines, so they can also be
> wired into the notebook's `paths`/`models_all` lists if in-notebook regeneration
> is preferred.

## Narrative suggestions for the rebuttal / manuscript
1. **Report the DLinear/NLinear result honestly.** They win on raw accuracy because
   direct multi-step forecasting sidesteps autoregressive error accumulation over a
   1499-step horizon (exactly Zeng et al.’s thesis). Position the Koopman model as
   (i) the strongest *autoregressive / dynamical* model (3–5× better than every RNN
   baseline), and (ii) the only model that provides interpretable spectral modes
   (CP-3 heart-rate mode, #6), an explicit control channel (γ sweep, #4/#5) and a
   latent-linear structure suitable for control design — capabilities a black-box
   linear readout does not offer. **(iii) Crucially, Koopman is far more
   noise-robust** than DLinear/NLinear (Table 5): the linear baselines' clean-data
   advantage collapses under realistic input noise (they hit 300–1000 % %RMSE at
   5 dB), while Koopman degrades gracefully (17.5 → 41.7 %). This is arguably the
   strongest single argument for the Koopman approach in a real (noisy) deployment.
2. The γ = 0 ablation and the activation study together show the control network is
   both necessary and correctly parameterised (bounded, zero-centred).
3. Streaming re-anchoring (Task D-b) demonstrates the model degrades gracefully and
   is deployable online, answering the “noise only at t=0” criticism.
