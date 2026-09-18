#!/usr/bin/env python3
"""
compare_motion_models.py
=========================
Head-to-head comparison of four motion-model approaches on the same
held-out test set:

  1. Constant velocity  — zero-parameter kinematic extrapolation
                          v = (pos[-1] - pos[-2]) / dt
                          The floor every learned model should beat.
  2. GMM (MDN)          — K-component Gaussian mixture, single forward pass,
                          no iterative sampling.
  3. CFM                — OT-CFM flow matching, anchored base point,
                          N_FE Euler ODE steps.
  4. Diffusion          — DDPM denoising, N(0,I) prior, T reverse steps.

All three learned models share the identical ContextEncoder architecture
and were trained on identical (X, y) data, so differences come from the
generative mechanism / output-distribution choice, not encoder capacity.

Reports:
  1. Accuracy   — velocity MAE / RMSE vs. ground truth on the test set
                  (best-of-N and mean-of-N over stochastic samples for the
                  three learned models; constant-velocity is deterministic
                  so best-of-N == mean-of-N == single-shot MAE for it).
  2. Speed      — wall-clock per propose() call, batched like the particle
                  filter does (N_particles per call).
  3. Diversity  — spread of samples drawn from the same context, a proxy
                  for how well each model captures multi-modal futures
                  at junctions/intersections.
  4. Plots      — accuracy/speed bar charts and a scatter of sampled
                  velocities vs. ground truth for a handful of contexts.

Usage (after installing the project):
    python -m mml_research.motion.compare_motion_models

Requires these checkpoints to already exist:
    ../../models/cfm_motion_model.pth
    ../../models/diffusion_motion_model.pth
    ../../models/gmm_motion_model.pth
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


from mml_core.motion.cfm_motion_model import CFMMotionModel
from mml_core.motion.diffusion_motion_model import DiffusionMotionModel
from mml_core.motion.gmm_motion_model import GMMMotionModel
from mml_core.motion.nf_motion_model import NFMotionModel
from mml_core.motion.utils import load_training_csv, chronological_split

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

N_SAMPLES_PER_CONTEXT = 8   # stochastic samples per context for accuracy/diversity
N_SPEED_PARTICLES = 500     # matches typical particle-filter batch size
N_SPEED_REPEATS = 20        # timing repeats for a stable estimate

COLORS = {
    "ConstVel": "gray",
    "GMM": "seagreen",
    "NF": "crimson",
    "CFM": "steelblue",
    "Diffusion": "darkorange",
}


# ── Constant-velocity baseline ────────────────────────────────────────────────

class ConstantVelocityModel:
    """
    Zero-parameter kinematic extrapolation baseline:
        v_pred = (pos[-1] - pos[-2]) / dt

    Deterministic — every call to propose() returns the same answer for the
    same context, so "N samples" are identical draws. Included to establish
    the floor that CFM/Diffusion/GMM must beat to justify their inference cost.
    """

    def __init__(self, block_size: int, input_dim: int = 2, dt: float = 0.033):
        self.block_size = block_size
        self.input_dim = input_dim
        self.dt = dt
        self.N_FE = 0   # zero network evaluations — for the speed report

    def propose(self, c_flat: np.ndarray) -> np.ndarray:
        c_flat = np.asarray(c_flat, dtype=np.float32)
        hist = c_flat.reshape(-1, self.block_size, self.input_dim)
        last_pos = hist[:, -1, :2]
        prev_pos = hist[:, -2, :2]
        v = (last_pos - prev_pos) / self.dt

        MAX_SPEED = 1.5
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * np.minimum(1.0, MAX_SPEED / (speed + 1e-9))
        return v.astype(np.float32)

    def propose_batched(self, c_flat: np.ndarray, n_samples_per_context: int = 1) -> np.ndarray:
        v = self.propose(c_flat)
        return np.repeat(v[:, None, :], n_samples_per_context, axis=1)

    def loss_history(self):
        return []


def load_models(models_dir: str):
    cfm_path = os.path.normpath(os.path.join(models_dir, "cfm_motion_model.pth"))
    diff_path = os.path.normpath(os.path.join(models_dir, "diffusion_motion_model.pth"))
    gmm_path = os.path.normpath(os.path.join(models_dir, "gmm_motion_model.pth"))
    nf_path = os.path.normpath(os.path.join(models_dir, "nf_motion_model.pth"))

    for name, p in [("CFM", cfm_path), ("Diffusion", diff_path),
                    ("GMM", gmm_path), ("NF", nf_path)]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} checkpoint not found: {p}")

    cfm = CFMMotionModel()
    cfm.load(cfm_path)

    diffusion = DiffusionMotionModel()
    diffusion.load(diff_path)

    gmm = GMMMotionModel()
    gmm.load(gmm_path)

    nf = NFMotionModel()
    nf.load(nf_path)

    const_vel = ConstantVelocityModel(block_size=cfm.block_size, dt=cfm.dt)

    return {
        "ConstVel": const_vel,
        "GMM": gmm,
        "NF": nf,
        "CFM": cfm,
        "Diffusion": diffusion,
    }


def accuracy_report(models: dict, X_test: np.ndarray, y_test: np.ndarray):
    """
    Draw N_SAMPLES_PER_CONTEXT proposals per test context from each model,
    then report best-of-N / mean-of-N MAE/RMSE and sample spread.
    """
    results = {}

    for name, model in models.items():
        t0 = time.time()
        samples = model.propose_batched(X_test, n_samples_per_context=N_SAMPLES_PER_CONTEXT)
        elapsed = time.time() - t0
        # samples: (N_contexts, N_SAMPLES_PER_CONTEXT, 2)

        err = samples - y_test[:, None, :]
        dist = np.linalg.norm(err, axis=-1)          # (N, S)

        best_of_n = dist.min(axis=1)
        mean_of_n = dist.mean(axis=1)
        spread = samples.std(axis=1).mean()

        results[name] = {
            "best_of_n_mae": float(best_of_n.mean()),
            "best_of_n_rmse": float(np.sqrt((best_of_n ** 2).mean())),
            "mean_of_n_mae": float(mean_of_n.mean()),
            "mean_of_n_rmse": float(np.sqrt((mean_of_n ** 2).mean())),
            "sample_spread_std": float(spread),
            "sampling_time_s": elapsed,
            "samples": samples,
        }

    return results


def speed_report(models: dict, X_test: np.ndarray):
    """
    Time propose() on a fixed-size batch, matching particle-filter usage.
    """
    idx = np.random.choice(len(X_test), size=min(N_SPEED_PARTICLES, len(X_test)),
                            replace=len(X_test) < N_SPEED_PARTICLES)
    c_batch = X_test[idx]

    timings = {}
    for name, model in models.items():
        model.propose(c_batch)   # warmup
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(N_SPEED_REPEATS):
            model.propose(c_batch)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) / N_SPEED_REPEATS

        timings[name] = elapsed

    return timings, len(c_batch)


def plot_comparison(models: dict, acc_results: dict, timings: dict, n_particles: int,
                     X_test: np.ndarray, y_test: np.ndarray, save_path: str):
    names = list(models.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ── (0,0) Accuracy: best-of-N MAE ─────────────────────────────────────────
    ax = axes[0, 0]
    metrics = ["best_of_n_mae", "mean_of_n_mae"]
    labels = [f"best-of-{N_SAMPLES_PER_CONTEXT}\nMAE", f"mean-of-{N_SAMPLES_PER_CONTEXT}\nMAE"]
    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [acc_results[name][m] for m in metrics]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width,
               label=name, color=COLORS.get(name))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("m/s")
    ax.set_title("Velocity accuracy")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # ── (0,1) Sample diversity ────────────────────────────────────────────────
    ax = axes[0, 1]
    spreads = [acc_results[n]["sample_spread_std"] for n in names]
    ax.bar(names, spreads, color=[COLORS.get(n) for n in names])
    for i, v in enumerate(spreads):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylabel("sample spread (std, m/s)")
    ax.set_title(f"Sample diversity across {N_SAMPLES_PER_CONTEXT} draws\n(proxy for multi-modality capture)")
    ax.grid(alpha=0.3, axis="y")

    # ── (1,0) Speed ───────────────────────────────────────────────────────────
    ax = axes[1, 0]
    vals_ms = [timings[n] * 1000 for n in names]
    ax.bar(names, vals_ms, color=[COLORS.get(n) for n in names])
    for i, v in enumerate(vals_ms):
        ax.text(i, v, f"{v:.3f} ms", ha="center", va="bottom")
    ax.set_ylabel("ms per propose() call")
    ax.set_title(f"Inference speed  (batch={n_particles} particles, {device})")
    ax.grid(alpha=0.3, axis="y")

    # ── (1,1) Scatter: sampled velocities vs ground truth for a few contexts ──
    ax = axes[1, 1]
    n_show = min(4, len(X_test))
    show_idx = np.linspace(0, len(X_test) - 1, n_show).astype(int)
    markers = {"ConstVel": "x", "GMM": "s", "NF": "d", "CFM": "o", "Diffusion": "^"}
    cmap = plt.get_cmap("tab10")
    for i, ci in enumerate(show_idx):
        color = cmap(i)
        for name in names:
            s = acc_results[name]["samples"][ci]   # (S, 2)
            ax.scatter(s[:, 0], s[:, 1], marker=markers.get(name, "o"), s=30,
                       color=color, alpha=0.6,
                       label=f"{name}" if i == 0 else None)
        gt = y_test[ci]
        ax.scatter(gt[0], gt[1], marker="*", s=200, color=color, edgecolors="black", zorder=5)
    ax.set_xlabel("vx (m/s)")
    ax.set_ylabel("vy (m/s)")
    title_markers = "  ".join(f"{markers[n]}={n}" for n in names)
    ax.set_title(f"Sampled velocities per context\n({title_markers}, ★=ground truth)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    print(f"\nComparison plot saved -> {save_path}")


def main():
    this_dir = str(motion_resource_dir() / "mml_guidance/mml_network")
    data_dir = os.path.join(this_dir, "..", "sim_data", "training_data")
    models_dir = os.path.join(this_dir, "..", "..", "models")

    csv_path = os.path.normpath(os.path.join(
        data_dir, "converted_noisy_5_v2_velocities_training_data.csv"
    ))

    print("=" * 70)
    print("Motion model comparison: ConstVel vs GMM vs CFM vs Diffusion")
    print("=" * 70)
    print(f"Device: {device}")

    X_all, y_all = load_training_csv(csv_path, input_dim=2)
    _, X_test, _, y_test = chronological_split(X_all, y_all, train_frac=0.7)
    print(f"Test set: {X_test.shape}")

    models = load_models(models_dir)

    print()
    for name, model in models.items():
        if hasattr(model, "network"):
            n_params = sum(p.numel() for p in model.network.parameters())
            extra = ""
            if hasattr(model, "N_FE"):
                extra = f"  (N_FE={model.N_FE} Euler steps)"
            elif hasattr(model, "T"):
                extra = f"  (T={model.T} denoising steps)"
            elif hasattr(model, "n_flows"):
                extra = f"  (n_flows={model.n_flows} RealNVP blocks, single forward pass)"
            elif hasattr(model, "K"):
                extra = f"  (K={model.K} mixture components, single forward pass)"
            print(f"{name:10s} params: {n_params:,}{extra}")
        else:
            print(f"{name:10s} params: 0  (zero-parameter kinematic baseline)")

    print("\n--- Accuracy & diversity (test set) ---")
    acc_results = accuracy_report(models, X_test, y_test)
    for name in models:
        r = acc_results[name]
        print(f"\n{name}:")
        print(f"  best-of-{N_SAMPLES_PER_CONTEXT} MAE  : {r['best_of_n_mae']:.4f} m/s")
        print(f"  best-of-{N_SAMPLES_PER_CONTEXT} RMSE : {r['best_of_n_rmse']:.4f} m/s")
        print(f"  mean-of-{N_SAMPLES_PER_CONTEXT} MAE  : {r['mean_of_n_mae']:.4f} m/s")
        print(f"  mean-of-{N_SAMPLES_PER_CONTEXT} RMSE : {r['mean_of_n_rmse']:.4f} m/s")
        print(f"  sample spread (std)   : {r['sample_spread_std']:.4f} m/s")

    print("\n--- Speed (particle-filter-like batch) ---")
    timings, n_particles = speed_report(models, X_test)
    for name, t in timings.items():
        print(f"  {name:10s}: {t * 1000:.3f} ms / propose()  (batch={n_particles})")

    fastest_learned = min(
        (t for n, t in timings.items() if n != "ConstVel"), default=None
    )
    if fastest_learned:
        print("\n  Relative to ConstVel baseline:")
        for name, t in timings.items():
            if name == "ConstVel":
                continue
            print(f"    {name:10s}: {t / timings['ConstVel']:.1f}x slower than ConstVel")

    plot_path = os.path.normpath(os.path.join(models_dir, "motion_model_comparison.png"))
    plot_comparison(models, acc_results, timings, n_particles, X_test, y_test, plot_path)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
