#!/usr/bin/env python3
"""
TASK 8 -- Mode frequency analysis & ablation
---------------------------------------------
Part A: Extract angular frequencies omega from the trained Koopman model.
  - For each complex eigenpair, evaluate omega_net at the mean latent radius
    across the test set.
  - Convert to Hz: f_Hz = omega / (2*pi * delta_t)
  - Report all mode frequencies and variance contributions.

Part B: Mode ablation.
  - Sequentially zero out each complex eigenpair (set corresponding
    block in the latent advance step to zero) and re-run inference.
  - Report %RMSE increase per ablated mode.

Outputs:
  results/mode_frequencies.json
  results/mode_ablation.json
  results/figures/mode_ablation.png
  results/figures/mode_ablation.svg
"""
import os, glob, re, json, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
DATA_DIR   = os.path.join(REPO_ROOT, "data")
CKPT_DIR   = os.path.join(RESULT_DIR, "koopman")
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load model ─────────────────────────────────────────────────────────────────
ckpts = glob.glob(os.path.join(CKPT_DIR, "*_model.ckpt"))
ckpt_path = max(ckpts, key=lambda f: int(re.search(r"_(\d{4})_", f).group(1)))
print(f"Checkpoint: {ckpt_path}")

from cardiokoop.postprocessing.postprocess_utils import load_weights_koopman
ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
params = ckpt["params"]
W, b, model = load_weights_koopman(
    ckpt_path,
    len(params["widths"]) - 1,
    len(params["widths_omega_real"]) - 1,
    params["num_real"],
    params["num_complex_pairs"],
)
model.to(device).eval()

delta_t = params["delta_t"]   # 0.01 s
n_cp    = params["num_complex_pairs"]   # 4
n_real  = params["num_real"]            # 4

# ── load test data ─────────────────────────────────────────────────────────────
data_name = params["data_name"]
X_norm = np.loadtxt(os.path.join(DATA_DIR, f"{data_name}_test1_x.csv"), delimiter=",")
U      = np.loadtxt(os.path.join(DATA_DIR, f"{data_name}_test1_u.csv"), delimiter=",")
sig_mean = np.load(os.path.join(DATA_DIR, "normalization_mean.npy"))
sig_std  = np.load(os.path.join(DATA_DIR, "normalization_std.npy"))

U = (U - U.mean(0)) / (U.std(0) + 1e-8)
T  = params["len_time"]
B  = X_norm.shape[0] // T
X_rs = X_norm.reshape(B, T, -1)
U_rs = U.reshape(B, T, -1)

# ── get all latent trajectories ───────────────────────────────────────────────
x0   = torch.tensor(X_rs[:, 0, :], dtype=torch.float32, device=device)
u_t  = torch.tensor(U_rs, dtype=torch.float32, device=device).permute(1, 0, 2)

# Standard full rollout
def rollout_full(model, x0, u_t):
    with torch.no_grad():
        y = model.encoder(x0)
        outs = []
        for t in range(u_t.size(0)):
            outs.append(model.decoder(y).cpu().numpy())
            y = model._advance(y, model.get_omegas(y), u_t[t])
    return np.stack(outs, axis=1)   # (B, T, D)

pred_clean = rollout_full(model, x0, u_t)
X_true     = X_rs * sig_std + sig_mean
pred_clean_denorm = pred_clean * sig_std + sig_mean

# ── encode the full test set to get latent coordinates ────────────────────────
with torch.no_grad():
    x_flat = torch.tensor(X_norm, dtype=torch.float32, device=device)
    y_flat  = model.encoder(x_flat).cpu().numpy()   # (B*T, latent_dim)

print(f"Latent shape: {y_flat.shape}")

# ── PART A: extract mode frequencies ──────────────────────────────────────────
# For each complex pair j:
#   radius = sqrt(y_{2j}^2 + y_{2j+1}^2)  (mean over all samples)
#   omega_j, sigma_j = omega_net_j(mean_radius)
#   f_Hz = omega_j / (2*pi*delta_t)
#
# For each real mode j:
#   |lambda_j| = exp(sigma_j * delta_t)  -- growth/decay rate

mode_info = []
with torch.no_grad():
    for j in range(n_cp):
        pair = y_flat[:, 2*j:2*j+2]
        radius = np.sqrt((pair ** 2).sum(1))
        # variance contribution: std of the pair across time
        var_j = float(pair.var(0).sum())

        # evaluate omega_net at mean and median radius
        r_mean   = float(radius.mean())
        r_tensor = torch.tensor([[r_mean]], dtype=torch.float32, device=device)
        omega_out = model.omega_nets[j](r_tensor).cpu().numpy()[0]
        # omega_net output[0] = angular freq in rad/s (multiplied by delta_t in _advance)
        # angle_per_step = omega * delta_t   [rad/step]
        # f_Hz = omega / (2*pi)
        omega = omega_out[0]    # rad/s
        sigma = omega_out[1]    # growth/decay exponent [1/s]
        f_hz  = omega / (2 * np.pi)   # Hz

        mode_info.append({
            "type": "complex",
            "index": j,
            "omega_rad_per_s": float(omega),
            "omega_rad_per_step": float(omega * delta_t),
            "sigma": float(sigma),
            "f_hz": float(f_hz),
            "period_s": float(1.0 / abs(f_hz)) if abs(f_hz) > 1e-6 else float("inf"),
            "mean_radius": r_mean,
            "latent_variance": var_j,
        })
        print(f"  Complex {j}: omega={omega:.4f} rad/step, f={f_hz:.4f} Hz, "
              f"sigma={sigma:.4f}, var={var_j:.4f}")

    for j in range(n_real):
        idx_net = n_cp + j
        val = y_flat[:, 2*n_cp + j]
        var_j = float(val.var())
        v_mean = float(abs(val).mean())
        v_tensor = torch.tensor([[v_mean]], dtype=torch.float32, device=device)
        sigma_out = model.omega_nets[idx_net](v_tensor).cpu().numpy()[0, 0]
        decay_rate = float(sigma_out)

        mode_info.append({
            "type": "real",
            "index": j,
            "sigma": decay_rate,
            "decay_factor_per_step": float(np.exp(sigma_out * delta_t)),
            "mean_abs_value": v_mean,
            "latent_variance": var_j,
        })
        print(f"  Real {j}: sigma={decay_rate:.4f}, "
              f"lambda/step={float(np.exp(sigma_out*delta_t)):.4f}, var={var_j:.4f}")

# heart-rate identification: ~1.0-1.3 Hz
hr_candidates = [m for m in mode_info
                 if m["type"] == "complex" and 0.5 <= abs(m["f_hz"]) <= 3.0]
print("\nHeart-rate-range modes (~1.0-1.3 Hz):", hr_candidates)

# ── PART B: mode ablation ─────────────────────────────────────────────────────
def pct_rmse_ps(true, pred):
    """Per-signal PTP-normalised %RMSE averaged."""
    rmse = np.sqrt(((true - pred) ** 2).mean(0))
    ptp  = np.ptp(true, axis=0) + 1e-8
    return float((100.0 * rmse / ptp).mean())

# baseline on test set
baseline_pct = float(np.mean([
    pct_rmse_ps(X_true[i], pred_clean_denorm[i]) for i in range(B)
]))
print(f"\nBaseline (no ablation) %RMSE: {baseline_pct:.4f}")

ablation_results = []
for j in range(n_cp):
    # patch model: override _advance to zero out pair j
    orig_advance = model._advance.__func__

    def make_advance_ablated(j_ablate):
        def _advance_ablated(self, y, omegas, u_t):
            # call standard advance
            result = orig_advance(self, y, omegas, u_t)
            # zero out complex pair j_ablate
            result_np = result.clone()
            result_np[:, 2*j_ablate:2*j_ablate+2] = 0.0
            return result_np
        return _advance_ablated

    import types
    model._advance = types.MethodType(make_advance_ablated(j), model)

    pred_abl = rollout_full(model, x0, u_t)
    pred_abl_denorm = pred_abl * sig_std + sig_mean
    abl_pct = float(np.mean([
        pct_rmse_ps(X_true[i], pred_abl_denorm[i]) for i in range(B)
    ]))
    delta = abl_pct - baseline_pct
    print(f"Ablate complex mode {j}: %RMSE={abl_pct:.4f}  delta={delta:+.4f}")
    ablation_results.append({
        "mode_type": "complex",
        "mode_index": j,
        "f_hz": mode_info[j]["f_hz"],
        "pct_rmse": abl_pct,
        "pct_rmse_increase": delta,
    })

    # restore
    model._advance = types.MethodType(orig_advance, model)

# ── save JSONs ─────────────────────────────────────────────────────────────────
with open(os.path.join(RESULT_DIR, "mode_frequencies.json"), "w") as f:
    json.dump({
        "delta_t_s": float(delta_t),
        "n_complex_pairs": int(n_cp),
        "n_real": int(n_real),
        "modes": mode_info,
        "heart_rate_range_candidates": hr_candidates,
    }, f, indent=2)
print("Saved -> results/mode_frequencies.json")

with open(os.path.join(RESULT_DIR, "mode_ablation.json"), "w") as f:
    json.dump({
        "baseline_pct_rmse": baseline_pct,
        "ablation_results": ablation_results,
    }, f, indent=2)
print("Saved -> results/mode_ablation.json")

# ── figure ────────────────────────────────────────────────────────────────────
HEX_PAL = ["#dda251", "#5a286b", "#646464", "#ac4484"]
sns.set_theme("paper")
sns.set_style("white")
sns.set_context("notebook", font_scale=1.1)
pal = HEX_PAL

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: mode frequencies
ax = axes[0]
complex_modes = [m for m in mode_info if m["type"] == "complex"]
f_hz_vals = [abs(m["f_hz"]) for m in complex_modes]
var_vals   = [m["latent_variance"] for m in complex_modes]
# normalise variances for bubble size
max_var = max(var_vals) + 1e-10
sizes   = [200 * v / max_var + 50 for v in var_vals]

sc = ax.scatter(range(n_cp), f_hz_vals, s=sizes, c=var_vals,
                cmap="YlOrRd", edgecolors="black", linewidth=0.7, zorder=3)
plt.colorbar(sc, ax=ax, label="Latent variance")
ax.axhspan(0.8, 2.0, alpha=0.1, color="green", label="HR range (0.8–2.0 Hz)")
ax.set_xticks(range(n_cp))
ax.set_xticklabels([f"Mode {j}" for j in range(n_cp)])
ax.set_ylabel("|Frequency| (Hz)")
ax.set_title("Learned Koopman Mode Frequencies")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Panel B: ablation
ax2 = axes[1]
x_pos = range(len(ablation_results))
deltas = [r["pct_rmse_increase"] for r in ablation_results]
colors = [pal[0] if d > 0 else pal[1] for d in deltas]
bars = ax2.bar(x_pos, deltas, color=colors, alpha=0.85, edgecolor="white")
ax2.axhline(0, color="k", linewidth=0.8)
ax2.set_xticks(list(x_pos))
ax2.set_xticklabels([f"Mode {r['mode_index']}\n({abs(r['f_hz']):.2f} Hz)"
                     for r in ablation_results], fontsize=9)
ax2.set_ylabel("%RMSE increase (pp)")
ax2.set_title("Mode Ablation: %RMSE Change")
ax2.grid(axis="y", alpha=0.3)

# add value labels
for bar, d in zip(bars, deltas):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{d:+.2f}", ha="center", va="bottom", fontsize=9)

plt.suptitle(f"Koopman Mode Analysis (baseline %RMSE = {baseline_pct:.2f}%)",
             fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "mode_ablation.png"), dpi=150,
            bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "mode_ablation.svg"), format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved -> figures/mode_ablation.png + mode_ablation.svg")
