#!/usr/bin/env python3
"""
Compare a fresh run of ``scripts/revision3/export_manuscript_tables.py`` against the
committed ``results/revision3/`` at **manuscript rounding**.

The 1499-step autoregressive rollout amplifies floating-point differences between
platforms (CPU/GPU, x86/ARM, BLAS builds), so the reproduction check does not ask for
bit-identical numbers.  It asks that every number *as it is printed in the manuscript*
is unchanged:

    Table 3   %RMSE and its 95 % CI ............ 1 decimal
              pooled R² and its 95 % CI ........ 2 decimals
              parameters / inference time / speed-up ... exact (constants)
    Table 4   RMSE, %RMSE, CIs, bias, LoA ...... 1 decimal;  R² and CI ... 2 decimals
    Table 5   %RMSE 1 decimal, R² 2 decimals (clean row recomputed, noisy rows constants)
    statistics.json
              Friedman χ² ...................... 2 decimals
              p-values (Shapiro, Friedman, Wilcoxon, Bonferroni) ... within a factor of 2
              Shapiro W ........................ 3 decimals
              Wilcoxon W (rank sum) ............ integer (±1)
              trajectory counts (worse/better/ties) ... exact
              paired-difference / summary stats  1 decimal
    r2_conventions.json
              means / SDs / CIs / medians / min / max ... 2 decimals
              global pooled R² ................. 3 decimals (as quoted in REVISION3_NOTES.md)
              "trajectories with R² < 0" counts  exact

A metric PASSES if committed and reproduced values print identically at that precision.
A difference of exactly one unit in the last printed digit ("off-by-one rounding",
e.g. 17.5 vs 17.4) is tolerated unless ``--strict`` is given, but every such case is
listed explicitly — the manuscript tables depend on these digits.  Any larger difference
(or a p-value ratio > 2, or a count mismatch) is a VIOLATION and the script exits 1.

Usage
-----
  python scripts/revision3/compare_results.py --committed results/revision3 --fresh out/
  python scripts/revision3/compare_results.py ... --report out/compare_report.md --all
  python scripts/revision3/compare_results.py ... --tolerance-json my_tolerances.json

``--tolerance-json`` maps fnmatch patterns on the metric path (e.g.
``"table4/*/r2_ci95"``) to a rule: ``{"decimals": 2}``, ``{"factor": 2}``,
``{"abs": 0.05}`` or ``{"exact": true}``; the last matching pattern wins.
"""
import os
import sys
import json
import math
import fnmatch
import argparse

FILES = ["table3_overall.json", "table4_per_signal_full.json", "table5_noise.json",
         "statistics.json", "r2_conventions.json"]

ONE = {"decimals": 1}
TWO = {"decimals": 2}
THREE = {"decimals": 3}
EXACT = {"exact": True}
PFACT = {"factor": 2.0}
INT1 = {"decimals": 0}


# ───────────────────────── metric extraction ─────────────────────────
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def metrics_table3(d):
    out = {}
    for row in d["rows"]:
        m = row["model"]
        out[f"table3/{m}/pct_rmse_mean"] = (row["pct_rmse_mean"], ONE)
        out[f"table3/{m}/pct_rmse_ci95"] = (row["pct_rmse_ci95"], ONE)
        out[f"table3/{m}/r2_pooled_mean"] = (row["r2_pooled_mean"], TWO)
        out[f"table3/{m}/r2_pooled_ci95"] = (row["r2_pooled_ci95"], TWO)
        out[f"table3/{m}/r2_pooled_global"] = (row["r2_pooled_global"], THREE)
        out[f"table3/{m}/pct_rmse_sd"] = (row["pct_rmse_sd"], ONE)
        out[f"table3/{m}/n_params"] = (row["n_params"], EXACT)
        out[f"table3/{m}/inference_time_s"] = (row["inference_time_s"], EXACT)
        out[f"table3/{m}/speedup_vs_simulation"] = (row["speedup_vs_simulation"], EXACT)
    return out


def metrics_table4(d):
    out = {}
    for row in d["rows"]:
        k = f"table4/{row['model']}/{row['signal']}"
        for key, rule in [("rmse", ONE), ("rmse_ci95", ONE), ("pct_rmse", ONE), ("pct_rmse_ci95", ONE),
                          ("r2", TWO), ("r2_ci95", TWO), ("bias", ONE), ("loa_lower", ONE), ("loa_upper", ONE)]:
            out[f"{k}/{key}"] = (row[key], rule)
    return out


def metrics_table5(d):
    out = {}
    for row in d["rows"]:
        cond = row["condition"].replace(" ", "")
        for m, v in row.items():
            if not isinstance(v, dict):
                continue
            out[f"table5/{cond}/{m}/pct_rmse_mean"] = (v["pct_rmse_mean"], ONE)
            out[f"table5/{cond}/{m}/r2_mean"] = (v["r2_mean"], TWO)
            if _num(v.get("pct_rmse_ci95")) and not math.isnan(v["pct_rmse_ci95"]):
                out[f"table5/{cond}/{m}/pct_rmse_ci95"] = (v["pct_rmse_ci95"], ONE)
    return out


def metrics_statistics(d):
    out = {}
    out["statistics/n_trajectories"] = (d["n_trajectories"], EXACT)
    for m, s in d["summary"].items():
        for key in ["mean", "sd", "ci95", "median", "q25", "q75"]:
            out[f"statistics/summary/{m}/{key}"] = (s[key], ONE)
    for m, s in d["shapiro_wilk"].items():
        out[f"statistics/shapiro/{m}/W"] = (s["W"], THREE)
        out[f"statistics/shapiro/{m}/p"] = (s["p"], PFACT)
        out[f"statistics/shapiro/{m}/normal_at_0_05"] = (int(s["normal_at_0_05"]), EXACT)
    fr = d["friedman"]
    out["statistics/friedman/statistic"] = (fr["statistic"], TWO)
    out["statistics/friedman/p"] = (fr["p"], PFACT)
    out["statistics/friedman/k"] = (fr["k"], EXACT)
    out["statistics/friedman/n"] = (fr["n"], EXACT)
    for m, w in d["wilcoxon_koopman_vs_baselines"].items():
        k = f"statistics/wilcoxon/{m}"
        out[f"{k}/statistic_W"] = (w["statistic_W"], INT1)
        out[f"{k}/p_two_sided"] = (w["p_two_sided"], PFACT)
        out[f"{k}/p_bonferroni_x5"] = (w["p_bonferroni_x5"], PFACT)
        for key in ["n_trajectories_worse_than_koopman", "n_trajectories_better_than_koopman", "n_ties"]:
            out[f"{k}/{key}"] = (w[key], EXACT)
        out[f"{k}/median_paired_difference_pp"] = (w["median_paired_difference_pp"], ONE)
        out[f"{k}/mean_paired_difference_pp"] = (w["mean_paired_difference_pp"], ONE)
    return out


def metrics_r2conv(d):
    out = {}
    for m, s in d["models"].items():
        for key, v in s["pooled"].items():
            rule = EXACT if key == "n_negative" else (THREE if key == "global_pooled" else TWO)
            out[f"r2conv/{m}/pooled/{key}"] = (v, rule)
        for key, v in s["signal_averaged"].items():
            out[f"r2conv/{m}/signal_averaged/{key}"] = (v, EXACT if key == "n_negative" else TWO)
        for sig, v in s["per_signal_mean_r2"].items():
            out[f"r2conv/{m}/per_signal_mean_r2/{sig}"] = (v, TWO)
        for sig, v in s["per_signal_n_negative_trajectories"].items():
            out[f"r2conv/{m}/per_signal_n_negative/{sig}"] = (v, EXACT)
        out[f"r2conv/{m}/mean_of_per_signal_r2"] = (s["mean_of_per_signal_r2"], TWO)
    return out


EXTRACTORS = {"table3_overall.json": metrics_table3, "table4_per_signal_full.json": metrics_table4,
              "table5_noise.json": metrics_table5, "statistics.json": metrics_statistics,
              "r2_conventions.json": metrics_r2conv}


# ───────────────────────── comparison ─────────────────────────
def fmt(x, rule):
    if x is None:
        return "missing"
    if "decimals" in rule:
        return f"{x:.{rule['decimals']}f}"
    if "factor" in rule:
        return f"{x:.3g}"
    return repr(x) if isinstance(x, float) else str(x)


def compare(c, r, rule):
    """Return (status, detail) with status in {'ok', 'off_by_one', 'violation', 'missing'}."""
    if c is None or r is None:
        return "missing", ""
    if isinstance(c, float) and isinstance(r, float) and math.isnan(c) and math.isnan(r):
        return "ok", ""
    if rule.get("exact"):
        return ("ok", "") if c == r else ("violation", f"exact mismatch (diff {r - c:+.6g})")
    if "abs" in rule:
        return ("ok", "") if abs(r - c) <= rule["abs"] else ("violation", f"|diff| {abs(r - c):.6g} > {rule['abs']}")
    if "factor" in rule:
        f = rule["factor"]
        if c == r:
            return "ok", ""
        if c <= 0 or r <= 0:
            return "violation", "non-positive p-value"
        ratio = max(c, r) / min(c, r)
        return ("ok", f"ratio {ratio:.3g}") if ratio <= f else ("violation", f"ratio {ratio:.3g} > {f}")
    n = rule["decimals"]
    sc, sr = f"{c:.{n}f}", f"{r:.{n}f}"
    if sc == sr:
        return "ok", ""
    unit = 10.0 ** (-n)
    dprint = abs(float(sr) - float(sc))
    if dprint <= unit * (1 + 1e-9):
        return "off_by_one", f"printed {sc} -> {sr} (raw diff {r - c:+.3e})"
    return "violation", f"printed {sc} -> {sr} ({dprint / unit:.0f} units of last digit; raw diff {r - c:+.3e})"


def load_metrics(directory):
    all_m = {}
    for fn in FILES:
        p = os.path.join(directory, fn)
        if not os.path.exists(p):
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(2)
        with open(p) as f:
            all_m.update(EXTRACTORS[fn](json.load(f)))
    return all_m


def apply_tolerances(metrics, tol):
    if not tol:
        return metrics
    out = {}
    for k, (v, rule) in metrics.items():
        for pat, new_rule in tol.items():
            if fnmatch.fnmatch(k, pat):
                rule = new_rule
        out[k] = (v, rule)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--committed", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "..", "..", "results", "revision3"))
    ap.add_argument("--fresh", required=True, help="directory with the freshly generated JSON files")
    ap.add_argument("--tolerance-json", default=None, help="optional per-metric rule overrides (see module doc)")
    ap.add_argument("--report", default=None, help="write the full diff table (markdown) to this file")
    ap.add_argument("--all", action="store_true", help="print every metric, not only differences")
    ap.add_argument("--strict", action="store_true", help="treat off-by-one rounding differences as failures")
    args = ap.parse_args()

    tol = None
    if args.tolerance_json:
        with open(args.tolerance_json) as f:
            tol = json.load(f)
    cm = apply_tolerances(load_metrics(args.committed), tol)
    rm = apply_tolerances(load_metrics(args.fresh), tol)

    rows, counts = [], {"ok": 0, "off_by_one": 0, "violation": 0, "missing": 0}
    for k in list(cm) + [k for k in rm if k not in cm]:
        c, rule = cm.get(k, (None, {}))
        r, rule_r = rm.get(k, (None, {}))
        rule = rule or rule_r
        status, detail = compare(c, r, rule)
        counts[status] += 1
        rows.append((k, fmt(c, rule), fmt(r, rule), status, detail))

    hdr = ("metric", "committed", "reproduced", "status", "detail")
    width = max(len(r[0]) for r in rows)
    shown = [r for r in rows if args.all or r[3] != "ok"]
    print(f"{hdr[0]:<{width}}  {hdr[1]:>12}  {hdr[2]:>12}  {hdr[3]:<11} {hdr[4]}")
    for k, c, r, s, d in shown:
        print(f"{k:<{width}}  {c:>12}  {r:>12}  {s:<11} {d}")
    if not shown:
        print("(all metrics identical at manuscript rounding)")
    n = len(rows)
    print(f"\n{n} metrics compared: {counts['ok']} identical at printed precision, "
          f"{counts['off_by_one']} off-by-one in the last printed digit, "
          f"{counts['violation']} violations, {counts['missing']} missing")

    if args.report:
        with open(args.report, "w") as f:
            f.write("# Reproduction check — committed `results/revision3` vs. fresh run\n\n")
            f.write(f"* committed: `{os.path.abspath(args.committed)}`\n* reproduced: `{os.path.abspath(args.fresh)}`\n")
            f.write(f"* {n} metrics compared: **{counts['ok']} identical** at printed precision, "
                    f"**{counts['off_by_one']} off-by-one** in the last printed digit, "
                    f"**{counts['violation']} violations**, {counts['missing']} missing\n\n")
            for title, sel in [("Violations", ["violation", "missing"]), ("Off-by-one rounding differences", ["off_by_one"])]:
                sub = [r for r in rows if r[3] in sel]
                f.write(f"## {title} ({len(sub)})\n\n")
                if sub:
                    f.write("| metric | committed | reproduced | status | detail |\n|---|---|---|---|---|\n")
                    for k, c, r, s, d in sub:
                        f.write(f"| `{k}` | {c} | {r} | {s} | {d} |\n")
                else:
                    f.write("none\n")
                f.write("\n")
            f.write(f"## All metrics ({n})\n\n<details><summary>expand</summary>\n\n")
            f.write("| metric | committed | reproduced | status |\n|---|---|---|---|\n")
            for k, c, r, s, d in rows:
                f.write(f"| `{k}` | {c} | {r} | {s} |\n")
            f.write("\n</details>\n")

    fail = counts["violation"] + counts["missing"] + (counts["off_by_one"] if args.strict else 0)
    if fail:
        print(f"\nFAIL: {fail} metric(s) differ from the committed results beyond manuscript rounding"
              + (" (strict mode: off-by-one counted)" if args.strict else ""))
        sys.exit(1)
    print("\nPASS: every manuscript number is reproduced at its printed precision"
          + (f" ({counts['off_by_one']} off-by-one rounding case(s) listed above)" if counts["off_by_one"] else ""))


if __name__ == "__main__":
    main()
