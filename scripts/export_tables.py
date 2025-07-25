#!/usr/bin/env python3
"""
Generate LaTeX tables for trajectory- and signal-level metrics
from the postprocessing results pickles.
"""
import os
import glob
import pickle

import numpy as np
import pandas as pd


def ci95(x: np.ndarray) -> float:
    """Compute 95%% confidence interval half-width."""
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))


def load_results():
    """Load all postprocessing result pickles from results/<model>/ directories."""
    results = {}
    for path in glob.glob('results/*/*postprocessing_results.pkl'):
        model = os.path.basename(os.path.dirname(path))
        results[model] = pickle.load(open(path, 'rb'))
    return results


def build_trajectory_metrics(results: dict) -> pd.DataFrame:
    """Construct DataFrame of aggregate trajectory metrics for each model."""
    rows = []
    for model, res in results.items():
        mse = res['mse_per_trajectory']
        rmse = res['rmse_per_trajectory']
        pct_flat = res['pct_per_traj_flat']
        pct_ps = res['pct_per_traj_ps']
        dtw = res['dtw_per_trajectory']
        r2 = res['r2_per_trajectory']
        inf = res['inference_times_s'] * 1000.0  # ms
        rows.append({
            'Model': model,
            'MSE (mean±CI)': f"{mse.mean():.3f}±{ci95(mse):.3f}",
            'RMSE (mean±CI)': f"{rmse.mean():.3f}±{ci95(rmse):.3f}",
            '%RMSE_flat (mean±CI)': f"{pct_flat.mean():.2f}±{ci95(pct_flat):.2f}",
            '%RMSE_ps (mean±CI)': f"{pct_ps.mean():.2f}±{ci95(pct_ps):.2f}",
            'DTW (mean±CI)': f"{dtw.mean():.3f}±{ci95(dtw):.3f}",
            'R2 (mean±CI)': f"{r2.mean():.3f}±{ci95(r2):.3f}",
            'Inf. time (ms)': f"{inf.mean():.3f}±{ci95(inf):.3f}",
            'Params': res.get('n_params', None)
        })
    df = pd.DataFrame(rows).set_index('Model')
    return df


def build_signal_metrics(results: dict) -> pd.DataFrame:
    """Construct DataFrame of per-signal metrics for each model and channel index."""
    rows = []
    for model, res in results.items():
        D = res['rmse_per_signal'].shape[1]
        for j in range(D):
            rows.append({
                'Model': model,
                'Signal': j,
                'R2 (mean±CI)': f"{res['r2_mean'][j]:.3f}±{res['r2_ci'][j]:.3f}",
                'RMSE (mean±CI)': f"{res['rmse_mean'][j]:.3f}±{res['rmse_ci'][j]:.3f}",
                '%RMSE (mean±CI)': f"{res['pct_mean'][j]:.2f}±{res['pct_ci'][j]:.2f}",
                'Bias (mean±CI)': f"{res['bias_mean'][j]:.3f}±{res['bias_ci'][j]:.3f}",
                'LoA': f"[{res['loa_lower'][j]:.3f},{res['loa_upper'][j]:.3f}]"
            })
    df = pd.DataFrame(rows)
    return df


def main():
    results = load_results()
    if not results:
        print("No result pickles found under results/*/*postprocessing_results.pkl")
        return

    df_traj = build_trajectory_metrics(results)
    df_signal = build_signal_metrics(results)
    os.makedirs('tables', exist_ok=True)
    df_traj.to_latex('tables/table_summary.tex', float_format=lambda x: str(x))
    df_signal.to_latex('tables/table_signal.tex', index=False, float_format=lambda x: str(x))
    print("Generated LaTeX tables in tables/table_summary.tex and tables/table_signal.tex")


if __name__ == '__main__':
    main()