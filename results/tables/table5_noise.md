**Table 5 — %RMSE and pooled R² under AWGN on the initial observation / warm-up window (SNR 30–5 dB), seed-42 test split, no retraining. Clean-condition values are identical to Table 3 (same predictions); noisy conditions from results/noise_robustness.json, noise_robustness_baselines.json and experiments/noise_robustness_linear.json.**

| Condition | Koopman %RMSE | LSTM %RMSE | GRU %RMSE | BiLSTM %RMSE | MLP %RMSE | AR(20) %RMSE | DLinear %RMSE | NLinear %RMSE | Koopman R² | LSTM R² | GRU R² | BiLSTM R² | MLP R² | AR(20) R² | DLinear R² | NLinear R² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Clean | 17.5 | 97.8 | 60.9 | 76.4 | 132.2 | 34.5 | 3.0 | 3.6 | 0.69 | -0.59 | -0.31 | -0.30 | -17.63 | 0.33 | 0.98 | 0.98 |
| 30 dB | 19.8 | 106.2 | 61.8 | 76.6 | 102.5 | 204.2 | 19.5 | 61.4 | 0.60 | -0.64 | -0.32 | -0.22 | -12.73 | 0.17 | 0.46 | -35.01 |
| 20 dB | 22.1 | 100.4 | 60.4 | 79.8 | 131.2 | 492.0 | 64.1 | 195.2 | 0.53 | -0.64 | -0.27 | -0.29 | -16.99 | -0.97 | -3.09 | -411.95 |
| 10 dB | 30.9 | 111.6 | 61.1 | 82.9 | 220.2 | 1931.2 | 191.7 | 561.4 | 0.28 | -0.71 | -0.39 | -0.38 | -37.14 | -16.06 | -45.17 | -4006.05 |
| 5 dB | 41.7 | 105.2 | 62.2 | 84.4 | 264.7 | 2966.8 | 333.7 | 1029.2 | -0.07 | -0.64 | -0.42 | -0.37 | -42.47 | -40.25 | -134.77 | -12933.73 |
