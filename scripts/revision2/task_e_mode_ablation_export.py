#!/usr/bin/env python3
"""
TASK E (editor mandatory #6 / R2.5) -- main-text-ready mode-ablation export
===========================================================================
The per-mode ablation (each complex eigenpair zeroed in the latent advance,
Delta %RMSE) already exists (results/mode_ablation.json + mode_frequencies.json,
produced by scripts/task8_mode_analysis.py on the seed-42 test split).  This
script simply SURFACES it as a clean CSV + LaTeX table so it can be promoted from
the supplement into the main text.  Zero compute -- no model is run.

Uses the manuscript's 0-based mode labelling (CP-0..CP-3), so the dominant
cardiac mode is CP-3 (|f| ~ 1.77 Hz, Delta %RMSE ~ +16.4 pp).

Outputs
  results/revision2/table_mode_ablation.csv
  results/revision2/table_mode_ablation.tex
"""
import os
import sys
import json

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C


def main():
    with open(os.path.join(C.RESULT_DIR, "mode_ablation.json")) as f:
        abl = json.load(f)
    with open(os.path.join(C.RESULT_DIR, "mode_frequencies.json")) as f:
        freq = json.load(f)

    baseline = abl["baseline_pct_rmse"]
    cp_modes = {m["index"]: m for m in freq["modes"] if m["type"] == "complex"}

    rows = []
    for r in abl["ablation_results"]:
        idx = r["mode_index"]
        m = cp_modes.get(idx, {})
        rows.append({
            "Mode": f"CP-{idx}",
            "|f| (Hz)": round(abs(r["f_hz"]), 3),
            "Period (s)": round(m.get("period_s", float("nan")), 3),
            "Latent variance": round(m.get("latent_variance", float("nan")), 4),
            "%RMSE (ablated)": round(r["pct_rmse"], 2),
            "Delta %RMSE (pp)": round(r["pct_rmse_increase"], 2),
            "Delta %RMSE (rel. %)": round(100 * r["pct_rmse_increase"] / baseline, 1),
        })

    df = pd.DataFrame(rows).sort_values("Delta %RMSE (pp)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)

    out_csv = os.path.join(C.REV2_DIR, "table_mode_ablation.csv")
    df.to_csv(out_csv, index=False)

    tex = df.to_latex(
        index=False, escape=False,
        caption=(f"Per-mode ablation of the learned Koopman operator on the seed-42 "
                 f"test split. Each complex eigenpair is zeroed in the latent advance "
                 f"and the full 1499-step forecast is re-run (no retraining). "
                 f"Baseline (no ablation) \\%RMSE = {baseline:.2f}. The dominant "
                 f"cardiac mode CP-3 ($|f|\\approx1.77$ Hz) accounts for "
                 f"$+{df['Delta %RMSE (pp)'].max():.1f}$ pp."),
        label="tab:mode_ablation",
        column_format="llrrrrrr")
    with open(os.path.join(C.REV2_DIR, "table_mode_ablation.tex"), "w") as f:
        f.write(tex)

    print(f"Baseline %RMSE = {baseline:.2f} (test1)")
    print(df.to_string(index=False))
    print(f"\nSaved -> {out_csv} (+ .tex)")


if __name__ == "__main__":
    main()
