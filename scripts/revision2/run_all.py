#!/usr/bin/env python3
"""
Top-level runner for ALL revision-2 experiments (journal Array, minor revision 2).

Reproduces every new result against the frozen final Koopman checkpoint (no
retraining of the main model).  Run with the cardiokoop environment:

    python scripts/revision2/run_all.py            # everything
    python scripts/revision2/run_all.py a b e      # only tasks A, B, E

Order matters: Task A produces the DLinear/NLinear checkpoints + pickles that the
table/figure builder consumes.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STEPS = [
    ("a",      "task_a_dlinear_nlinear.py",           "Task A: DLinear + NLinear baselines (training)"),
    ("b",      "task_b_gamma_sweep.py",               "Task B: control-gain gamma sweep (Fig S7)"),
    ("c",      "task_c_activation.py",                "Task C: control-net activation comparison (Table S4)"),
    ("d",      "task_d_realistic_streaming_noise.py", "Task D: realistic + streaming noise (Fig S8)"),
    ("e",      "task_e_mode_ablation_export.py",      "Task E: mode-ablation export (Table)"),
    ("f",      "task_f_clean_reanchoring.py",         "Task F: clean re-anchoring sweep (Fig S10)"),
    ("build",  "build_tables_figures.py",             "Tables 3/4/5 + Figures 5/6 + Fig S9 comparison"),
    ("panels", "regenerate_panels_a_gj.py",           "Figure 5 panels a + g-j (+ Fig S9 panels)"),
    ("mode",   "figure_mode_analysis.py",             "Figure S6: Koopman mode analysis"),
    ("latent", "figure_latent_hparam.py",             "Figure S4: latent / hyperparameter ablation"),
]


def main(argv):
    wanted = [a.lower() for a in argv] or [s[0] for s in STEPS]
    todo = [s for s in STEPS if s[0] in wanted]
    if not todo:
        print(f"Nothing to run for {wanted}. Valid: {[s[0] for s in STEPS]}")
        return 1
    failures = []
    for key, script, desc in todo:
        print("\n" + "=" * 78 + f"\n>>> {desc}\n" + "=" * 78, flush=True)
        rc = subprocess.run([PY, os.path.join(HERE, script)]).returncode
        if rc != 0:
            failures.append(script)
            print(f"!!! {script} exited with code {rc}", flush=True)
    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("All requested revision-2 steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
