#!/usr/bin/env python3
"""
nfe_sweep.py
============
Compute-budget sweep: accuracy vs. number of network function evaluations (NFE).

This is the experiment that tests the actual claim behind flow matching:
OT-CFM learns (near-)straight probability paths, so its ODE should be
integrable in very few Euler steps, whereas a DDPM's reverse process needs
many denoising steps to stay accurate.

Nothing here is retrained. We take the already-trained checkpoints and vary
ONLY the inference-time compute budget:

  * CFM       : N_FE Euler steps            (N_FE swept)
  * Diffusion : DDIM-strided ancestral sampling over the SAME trained T=20
                cosine schedule, using S of the 20 timesteps (S swept).
                Re-deriving a fresh T=S schedule would be a strawman — the
                network was trained against the T=20 noise levels, so the
                fair low-step sampler is a strided one.
  * NF        : single inverse pass, by construction (one reference point).
                Note its "1 pass" invokes 2 RealNVP blocks x 4 s/t nets = 8
                subnetwork evaluations.

Usage (after installing the project):
    python -m mml_research.motion.nfe_sweep

Output:
    ../../models/nfe_sweep.png
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from mml_core.motion.cfm_motion_model import load_cfm_model
from mml_core.motion.diffusion_motion_model import load_diffusion_model
from mml_core.motion.nf_motion_model import load_nf_model
from mml_core.motion.utils import load_training_csv, chronological_split

device = "cuda" if torch.cuda.is_available() else "cpu"

N_SAMPLES = 8            # stochastic draws per context (best-of-N accuracy)
N_SPEED_PARTICLES = 500  # latency batch, matches the particle filter
MAX_SPEED = 1.5          # m/s, same clip as every motion model

CFM_STEPS = [1, 2, 3, 5, 10, 20]
DIFF_STEPS = [1, 2, 4, 5, 10, 20]


# ──────────────────────────────────────────────────────────────────────────
# DDIM-strided sampler for the trained DDPM
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def diffusion_propose_strided(model, c_flat, n_steps, eta=1.0):
    """
    Ancestral sampling using only `n_steps` of the trained T-step schedule.

    eta=1.0 recovers DDPM-like stochasticity (needed: we are measuring
    multimodal coverage, so a deterministic eta=0 sampler would collapse
    every draw to the same point).
    """
    c_t = torch.tensor(np.asarray(c_flat, dtype=np.float32), device=device)
    N = c_t.shape[0]
    model.network.eval()
    emb = model.network.encode_context(c_t)

    # Evenly spaced subsequence of the trained timesteps, high -> low.
    ts = np.linspace(0, model.T - 1, n_steps).round().astype(int)
    ts = np.unique(ts)[::-1]

    ab = model.alphas_cumprod
    y = torch.randn((N, 2), device=device)

    for i, t_idx in enumerate(ts):
        t = torch.full((N,), int(t_idx), dtype=torch.long, device=device)
        eps = model.network.predict_noise(y, t, emb)

        ab_t = ab[int(t_idx)]
        prev_idx = int(ts[i + 1]) if i + 1 < len(ts) else -1
        ab_prev = ab[prev_idx] if prev_idx >= 0 else torch.tensor(1.0, device=device)

        x0 = (y - torch.sqrt(1.0 - ab_t) * eps) / torch.sqrt(ab_t)

        sigma = eta * torch.sqrt(
            (1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev)
        ) if prev_idx >= 0 else torch.tensor(0.0, device=device)

        dir_term = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0)) * eps
        y = torch.sqrt(ab_prev) * x0 + dir_term
        if prev_idx >= 0:
            y = y + sigma * torch.randn_like(y)

    v = y.cpu().numpy().astype(np.float32)
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.minimum(1.0, MAX_SPEED / (speed + 1e-9))


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def accuracy(sample_fn, X_test, y_test):
    """best-of-N MAE, mean-of-N MAE, and sample spread over N_SAMPLES draws."""
    draws = np.stack([sample_fn(X_test) for _ in range(N_SAMPLES)], axis=1)
    dist = np.linalg.norm(draws - y_test[:, None, :], axis=2)   # (Nctx, N)
    spread = np.linalg.norm(draws - draws.mean(axis=1, keepdims=True), axis=2).mean()
    mean_pred = draws.mean(axis=1)
    return {
        "best_of_n_mae": float(dist.min(axis=1).mean()),
        "mean_of_n_mae": float(np.linalg.norm(mean_pred - y_test, axis=1).mean()),
        "spread": float(spread),
    }


def latency_ms(sample_fn, X_test, repeats=20):
    idx = np.random.choice(len(X_test), size=N_SPEED_PARTICLES,
                           replace=len(X_test) < N_SPEED_PARTICLES)
    batch = X_test[idx]
    sample_fn(batch)  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        sample_fn(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0


def main():
    this_dir = str(motion_resource_dir() / "mml_guidance/mml_network")
    models_dir = os.path.normpath(os.path.join(this_dir, "..", "..", "models"))
    csv_path = os.path.normpath(os.path.join(
        this_dir, "..", "sim_data", "training_data",
        "converted_noisy_5_v2_velocities_training_data.csv"))

    X_all, y_all = load_training_csv(csv_path, input_dim=2)
    _, X_test, _, y_test = chronological_split(X_all, y_all, train_frac=0.7)
    print(f"Test contexts: {X_test.shape[0]}\n")

    # The loaders return PF-facing shims; unwrap to reach N_FE / T / the schedule.
    cfm = load_cfm_model(os.path.join(models_dir, "cfm_motion_model.pth"))
    cfm = getattr(cfm, "_cfm", cfm)
    diff = load_diffusion_model(os.path.join(models_dir, "diffusion_motion_model.pth"))._diffusion
    nf = load_nf_model(os.path.join(models_dir, "nf_motion_model.pth"))._nf

    rows = {"CFM": [], "Diffusion": []}

    print("=" * 74)
    print(f"{'Model':<12}{'NFE':>5}{'best-of-8 MAE':>16}{'mean-of-8 MAE':>16}"
          f"{'spread':>9}{'ms/500p':>10}")
    print("=" * 74)

    for n in CFM_STEPS:
        cfm.N_FE = n
        fn = lambda X: cfm.propose(X)
        a = accuracy(fn, X_test, y_test)
        ms = latency_ms(fn, X_test)
        rows["CFM"].append((n, a, ms))
        print(f"{'CFM':<12}{n:>5}{a['best_of_n_mae']:>16.4f}"
              f"{a['mean_of_n_mae']:>16.4f}{a['spread']:>9.3f}{ms:>10.2f}")

    print("-" * 74)

    for n in DIFF_STEPS:
        fn = lambda X, n=n: diffusion_propose_strided(diff, X, n)
        a = accuracy(fn, X_test, y_test)
        ms = latency_ms(fn, X_test)
        rows["Diffusion"].append((n, a, ms))
        print(f"{'Diffusion':<12}{n:>5}{a['best_of_n_mae']:>16.4f}"
              f"{a['mean_of_n_mae']:>16.4f}{a['spread']:>9.3f}{ms:>10.2f}")

    print("-" * 74)

    nf_fn = lambda X: nf.propose(X)
    nf_acc = accuracy(nf_fn, X_test, y_test)
    nf_ms = latency_ms(nf_fn, X_test)
    print(f"{'NF':<12}{1:>5}{nf_acc['best_of_n_mae']:>16.4f}"
          f"{nf_acc['mean_of_n_mae']:>16.4f}{nf_acc['spread']:>9.3f}{nf_ms:>10.2f}")
    print("=" * 74)

    # ── Plot: accuracy vs NFE, and the accuracy/latency Pareto front ──────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    for name, color, marker in [("CFM", "steelblue", "o"),
                                ("Diffusion", "darkorange", "^")]:
        xs = [r[0] for r in rows[name]]
        ys = [r[1]["best_of_n_mae"] for r in rows[name]]
        ax.plot(xs, ys, marker=marker, color=color, lw=2, label=name)
    ax.axhline(nf_acc["best_of_n_mae"], color="crimson", ls="--", lw=1.5,
               label="NF (1 pass)")
    ax.set_xscale("log")
    ax.set_xticks(CFM_STEPS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Network function evaluations (NFE)")
    ax.set_ylabel("best-of-8 MAE  (m/s)")
    ax.set_title("Accuracy vs. inference compute\n(lower is better)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    for name, color, marker in [("CFM", "steelblue", "o"),
                                ("Diffusion", "darkorange", "^")]:
        xs = [r[0] for r in rows[name]]
        ys = [r[1]["spread"] for r in rows[name]]
        ax.plot(xs, ys, marker=marker, color=color, lw=2, label=name)
    ax.axhline(nf_acc["spread"], color="crimson", ls="--", lw=1.5, label="NF")
    ax.set_xscale("log")
    ax.set_xticks(CFM_STEPS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Network function evaluations (NFE)")
    ax.set_ylabel("sample spread")
    ax.set_title("Sample diversity vs. compute\n(multimodality proxy)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2]
    for name, color, marker in [("CFM", "steelblue", "o"),
                                ("Diffusion", "darkorange", "^")]:
        xs = [r[2] for r in rows[name]]
        ys = [r[1]["best_of_n_mae"] for r in rows[name]]
        ax.plot(xs, ys, marker=marker, color=color, lw=2, label=name, alpha=0.85)
        for r in rows[name]:
            ax.annotate(f"{r[0]}", (r[2], r[1]["best_of_n_mae"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.scatter([nf_ms], [nf_acc["best_of_n_mae"]], color="crimson", marker="d",
               s=70, zorder=5, label="NF")
    ax.set_xlabel("latency (ms / 500 particles)")
    ax.set_ylabel("best-of-8 MAE  (m/s)")
    ax.set_title("Pareto front: accuracy vs. latency\n(labels = NFE)")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out = os.path.join(models_dir, "nfe_sweep.png")
    plt.savefig(out, dpi=130)
    print(f"\nPlot → {out}")


if __name__ == "__main__":
    main()
