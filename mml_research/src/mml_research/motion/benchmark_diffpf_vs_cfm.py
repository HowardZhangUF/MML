#!/usr/bin/env python3
"""
benchmark_diffpf_vs_cfm.py
===========================
Filter-level tracking benchmark: classical particle filter + CFM motion
model (our method) vs. the adapted DiffPF generative posterior sampler,
on IDENTICAL simulated occluded trajectories (same seeds).

This is the level at which DiffPF can be compared at all — it replaces the
whole PF update (predict + weight + resample) with conditional generative
sampling, so the one-step velocity comparison in compare_motion_models.py
does not apply to it.

Trackers
--------
PF+CFM:
    Bootstrap PF mirroring ParticleFilter.predict_cfm():
      predict  — CFM proposes a velocity per particle from its history
                 window; positions integrate v * dt + process noise.
      update   — when visible: Gaussian likelihood weights + systematic
                 resampling. When occluded: predict only.
      estimate — weighted particle mean.

DiffPF (adapted):
    Constant-velocity propagation + conditional DDPM samples a fresh
    equally-weighted particle set per step, conditioned on the measurement
    (or a null token during occlusion) and the propagated particles.

Metrics (per run, aggregated over runs)
---------------------------------------
  rmse_all       — position RMSE over all tracked steps
  rmse_visible   — RMSE on steps with a measurement
  rmse_occluded  — RMSE on occluded steps (the interesting regime)
  recovery_time  — mean steps after an occlusion ends until error < 0.3 m
  ms_per_step    — wall-clock filter cost

Usage (after installing the project):
    python -m mml_research.motion.benchmark_diffpf_vs_cfm --n_runs 5 --n_steps 2000
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import os
import sys
import time
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt


from mml_core.motion.cfm_motion_model import CFMMotionModel
from mml_core.motion.nf_motion_model import NFMotionModel
from mml_core.tracking.diffpf_tracker import DiffPFTracker
from mml_research.motion.train_diffpf import simulate_full_trajectory
from mml_research.motion.generate_synthetic_data import DT, NOISE_STD

RECOVERY_THRESHOLD = 0.3   # metres — matches the occlusion-recovery experiment


# ── Bootstrap PF with a pluggable motion model (mirrors ParticleFilter) ──────

class BootstrapPFTracker:
    """
    Classical bootstrap PF; the motion model only needs .propose(), .dt and
    .block_size — works for CFMMotionModel, NFMotionModel, etc.
    """

    def __init__(self, motion_model, n_particles: int = 500,
                 process_noise: float = 0.01, meas_sigma: float = 0.05,
                 occlusion_sigma_gain: float = 0.0, sigma_max: float = 2.5,
                 base_sigma: float = None):
        self.motion_model = motion_model
        self.N = n_particles
        self.N_th = motion_model.block_size
        self.process_noise = process_noise
        self.meas_sigma = meas_sigma
        self.particles = None    # (N_th, N, 2)
        self.weights = None      # (N,)
        # Occlusion-adaptive proposal spread: anchor_sigma is an
        # inference-time input to the CFM sampler, so the filter can widen
        # the proposal as an occlusion drags on (no retraining needed).
        # sigma_t = min(sigma_max, base * (1 + gain * steps_since_meas)).
        self.occlusion_sigma_gain = occlusion_sigma_gain
        self.sigma_max = sigma_max
        self.base_sigma = (base_sigma if base_sigma is not None
                           else getattr(motion_model, "anchor_sigma", None))
        self.steps_since_meas = 0

    def reset(self, init_measurement: np.ndarray, init_spread: float = 0.05):
        init = np.asarray(init_measurement, dtype=np.float32)
        base = init[None, :] + init_spread * np.random.randn(self.N, 2).astype(np.float32)
        self.particles = np.repeat(base[None, :, :], self.N_th, axis=0)
        self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)
        self.steps_since_meas = 0
        # Trackers share the motion-model object; restore the checkpoint
        # sigma in case an adaptive tracker mutated it in a previous run.
        if self.base_sigma is not None:
            self.motion_model.anchor_sigma = self.base_sigma

    def _predict(self):
        c_flat = self.particles.transpose(1, 0, 2).reshape(self.N, self.N_th * 2)
        v = self.motion_model.propose(c_flat)              # (N, 2)
        new_pos = (
            self.particles[-1]
            + v * self.motion_model.dt
            + self.process_noise * np.random.randn(self.N, 2).astype(np.float32)
        )
        self.particles[:-1] = self.particles[1:]
        self.particles[-1] = new_pos

    def _update(self, measurement: np.ndarray):
        d2 = np.sum((self.particles[-1] - measurement[None, :]) ** 2, axis=1)
        # Shift by the minimum squared distance before exponentiating so the
        # best particle's likelihood is exactly 1 — after a long occlusion
        # every particle can be metres away and the unshifted exp underflows
        # all weights to zero (NaN on normalization).
        lik = np.exp(-0.5 * (d2 - d2.min()) / self.meas_sigma ** 2) + 1e-300
        self.weights *= lik
        w_sum = self.weights.sum()
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)
        else:
            self.weights /= w_sum

        # Systematic resampling on effective-sample-size collapse.
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.N / 2:
            positions = (np.arange(self.N) + np.random.uniform()) / self.N
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0
            idx = np.searchsorted(cumsum, positions)
            self.particles = self.particles[:, idx, :]
            self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)

    def step(self, measurement):
        if self.occlusion_sigma_gain > 0.0 and self.base_sigma is not None:
            self.motion_model.anchor_sigma = min(
                self.sigma_max,
                self.base_sigma * (1.0 + self.occlusion_sigma_gain
                                   * self.steps_since_meas))
        self._predict()
        if measurement is not None:
            self._update(np.asarray(measurement, dtype=np.float32))
            self.steps_since_meas = 0
        else:
            self.steps_since_meas += 1
        return np.average(self.particles[-1], axis=0, weights=self.weights)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_tracker(tracker, positions, visible, rng):
    """Run one tracker over a trajectory; returns (errors, vis_mask, ms_per_step)."""
    first_vis = int(np.argmax(visible))
    tracker.reset(positions[first_vis] + NOISE_STD * rng.standard_normal(2).astype(np.float32))

    errors, vis_mask = [], []
    t0 = time.time()
    for t in range(first_vis + 1, len(positions)):
        if visible[t]:
            meas = positions[t] + NOISE_STD * rng.standard_normal(2).astype(np.float32)
        else:
            meas = None
        est = tracker.step(meas)
        errors.append(float(np.linalg.norm(est - positions[t])))
        vis_mask.append(bool(visible[t]))
    elapsed = time.time() - t0

    return np.array(errors), np.array(vis_mask), 1000.0 * elapsed / len(errors)


def recovery_times(errors: np.ndarray, vis_mask: np.ndarray):
    """Steps after each occlusion→visible transition until error < threshold."""
    times = []
    t = 1
    while t < len(vis_mask):
        if vis_mask[t] and not vis_mask[t - 1]:      # occlusion just ended
            k = t
            while k < len(errors) and errors[k] >= RECOVERY_THRESHOLD:
                k += 1
            if k < len(errors):
                times.append(k - t)
        t += 1
    return times


def _mean_std(xs):
    """Mean and (population) std of a list, tolerating empties."""
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.std(xs))


def summarize(name, all_errors, all_vis, all_ms, all_recovery):
    e = np.concatenate(all_errors)
    v = np.concatenate(all_vis)
    rec = [r for run in all_recovery for r in run]

    # Per-run metrics so every aggregate number carries a run-to-run std.
    per_run_all = [float(np.sqrt(np.mean(er ** 2))) for er in all_errors]
    per_run_vis = [
        float(np.sqrt(np.mean(er[vm] ** 2)))
        for er, vm in zip(all_errors, all_vis) if vm.any()
    ]
    per_run_occ = [
        float(np.sqrt(np.mean(er[~vm] ** 2)))
        for er, vm in zip(all_errors, all_vis) if (~vm).any()
    ]
    # One recovery-time number per run (mean of that run's events).
    per_run_rec = [float(np.mean(r)) if len(r) else None for r in all_recovery]

    all_m, all_s = _mean_std(per_run_all)
    vis_m, vis_s = _mean_std(per_run_vis)
    occ_m, occ_s = _mean_std(per_run_occ)
    rec_m, rec_s = _mean_std(per_run_rec)
    ms_m, ms_s = _mean_std(list(all_ms))

    stats = {
        # Pooled point estimates (over all steps of all runs).
        "rmse_all": float(np.sqrt(np.mean(e ** 2))),
        "rmse_visible": float(np.sqrt(np.mean(e[v] ** 2))),
        "rmse_occluded": float(np.sqrt(np.mean(e[~v] ** 2))) if (~v).any() else float("nan"),
        # Mean +/- std across the per-run metric (the error bars the paper needs).
        "rmse_all_mean": all_m,       "rmse_all_std": all_s,
        "rmse_visible_mean": vis_m,   "rmse_visible_std": vis_s,
        "rmse_occluded_mean": occ_m,  "rmse_occluded_std": occ_s,
        "recovery_mean": rec_m,       "recovery_std": rec_s,
        "ms_per_step_mean": ms_m,     "ms_per_step_std": ms_s,
        "n_runs": len(all_errors),
        "n_recovery_events": len(rec),
        # Per-run arrays for the CSV / paired significance tests.
        "_per_run_all": per_run_all,
        "_per_run_vis": per_run_vis,
        "_per_run_occ": per_run_occ,
        "_per_run_rec": per_run_rec,
        "_per_run_ms": list(all_ms),
    }
    print(f"\n{name}:   (mean +/- run-to-run std over {stats['n_runs']} runs)")
    print(f"  RMSE (all steps)     : {all_m:.4f} +/- {all_s:.4f} m")
    print(f"  RMSE (visible)       : {vis_m:.4f} +/- {vis_s:.4f} m")
    print(f"  RMSE (occluded)      : {occ_m:.4f} +/- {occ_s:.4f} m")
    print(f"  recovery time        : {rec_m:.2f} +/- {rec_s:.2f} steps "
          f"(to err < {RECOVERY_THRESHOLD} m, n={len(rec)} events)")
    print(f"  filter cost          : {ms_m:.2f} +/- {ms_s:.2f} ms/step")
    return stats


def main():
    parser = argparse.ArgumentParser(description="PF+CFM vs adapted DiffPF, filter level")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--pf_particles", type=int, default=500)
    parser.add_argument("--cfm_ckpt", type=str, default="cfm_motion_model.pth",
                        help="CFM checkpoint filename in models/ (e.g. "
                             "cfm_motion_model_anchor1.0.pth for the "
                             "wide-anchor ablation)")
    parser.add_argument("--adaptive_gain", type=float, default=0.05,
                        help="Per-occluded-step growth rate of the CFM "
                             "anchor sigma for the PF+CFM-AS tracker")
    parser.add_argument("--sigma_max", type=float, default=2.5,
                        help="Cap on the adaptive anchor sigma")
    parser.add_argument("--models-dir", default=None, help="Override the models dir")
    args = parser.parse_args()

    this_dir = str(motion_resource_dir() / "mml_guidance/mml_network")
    models_dir = args.models_dir or os.path.normpath(os.path.join(this_dir, '..', '..', 'models'))
    cfm_path = os.path.join(models_dir, args.cfm_ckpt)
    nf_path = os.path.join(models_dir, "nf_motion_model.pth")
    diffpf_path = os.path.join(models_dir, "diffpf_tracker.pth")

    print("=" * 70)
    print("Filter-level benchmark: PF+CFM vs PF+NF vs adapted DiffPF")
    print("=" * 70)
    print(f"Runs: {args.n_runs} x {args.n_steps} steps  (identical trajectories per run)")

    cfm = CFMMotionModel()
    cfm.load(cfm_path)

    nf = NFMotionModel()
    nf.load(nf_path)

    diffpf = DiffPFTracker()
    diffpf.load(diffpf_path)

    cfm_sigma0 = cfm.anchor_sigma

    tracker_names = ["PF+CFM", "PF+CFM-AS", "PF+NF", "DiffPF"]
    results = {name: ([], [], [], []) for name in tracker_names}
    example = None   # (positions, visible, errors per tracker) from the first run

    for run in range(args.n_runs):
        rng = np.random.default_rng(1000 + run)
        start_node = run % 8
        positions, visible = simulate_full_trajectory(start_node, args.n_steps, rng)

        for name, tracker in [
            ("PF+CFM", BootstrapPFTracker(cfm, n_particles=args.pf_particles,
                                          base_sigma=cfm_sigma0)),
            ("PF+CFM-AS", BootstrapPFTracker(
                cfm, n_particles=args.pf_particles,
                occlusion_sigma_gain=args.adaptive_gain,
                sigma_max=args.sigma_max, base_sigma=cfm_sigma0)),
            ("PF+NF", BootstrapPFTracker(nf, n_particles=args.pf_particles)),
            ("DiffPF", diffpf),
        ]:
            # identical measurement noise per tracker via a re-seeded rng
            meas_rng = np.random.default_rng(5000 + run)
            # Seed the global RNGs the filters draw from internally (particle
            # init, process noise, resampling offsets, torch sampling noise)
            # so every tracker sees identical noise streams and reruns are
            # reproducible.
            np.random.seed(9000 + run)
            torch.manual_seed(9000 + run)
            errors, vis_mask, ms = run_tracker(tracker, positions, visible, meas_rng)
            rec = recovery_times(errors, vis_mask)

            results[name][0].append(errors)
            results[name][1].append(vis_mask)
            results[name][2].append(ms)
            results[name][3].append(rec)

            if run == 0:
                if example is None:
                    example = {"positions": positions, "visible": visible, "errors": {}}
                example["errors"][name] = errors

        print(f"  run {run + 1}/{args.n_runs} done "
              f"({100.0 * (1 - visible.mean()):.1f}% occluded)")

    print("\n--- Aggregate results ---")
    stats = {}
    for name, (all_e, all_v, all_ms, all_rec) in results.items():
        stats[name] = summarize(name, all_e, all_v, all_ms, all_rec)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    tracker_colors = [("PF+CFM", "steelblue"), ("PF+CFM-AS", "seagreen"),
                      ("PF+NF", "crimson"), ("DiffPF", "purple")]

    ax = axes[0]
    vis = example["visible"][1 + int(np.argmax(example["visible"])):]
    n_plot = min(1500, len(example["errors"]["PF+CFM"]))
    for name, color in tracker_colors:
        ax.plot(example["errors"][name][:n_plot], lw=1.0, label=name, color=color)
    occluded = ~vis[:n_plot]
    ax.fill_between(np.arange(n_plot), 0, ax.get_ylim()[1], where=occluded,
                     alpha=0.15, color="red", label="occluded")
    ax.axhline(RECOVERY_THRESHOLD, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("position error (m)")
    ax.set_title("Tracking error over one run (red = occlusion)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    metrics = ["rmse_all_mean", "rmse_visible_mean", "rmse_occluded_mean"]
    errs = ["rmse_all_std", "rmse_visible_std", "rmse_occluded_std"]
    labels = ["RMSE all", "RMSE visible", "RMSE occluded"]
    x = np.arange(len(metrics))
    width = 0.8 / len(tracker_colors)
    for i, (name, color) in enumerate(tracker_colors):
        vals = [stats[name][m] for m in metrics]
        stds = [stats[name][s] for s in errs]
        ax.bar(x + (i - (len(tracker_colors) - 1) / 2) * width, vals, width,
               yerr=stds, capsize=3, label=name, color=color,
               error_kw=dict(lw=0.8, ecolor="black"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("m  (mean +/- run-to-run std)")
    cost_str = ", ".join(
        f"{name} {stats[name]['ms_per_step_mean']:.1f} ms/step"
        for name, _ in tracker_colors
    )
    ax.set_title(f"Aggregate over {args.n_runs} runs   |   {cost_str}")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(models_dir, "diffpf_vs_cfm_benchmark.png")
    plt.savefig(plot_path, dpi=130)
    print(f"\nBenchmark plot saved -> {plot_path}")

    # ── Per-run CSV (one row per run per tracker) for auditing + paired tests ──
    import csv
    csv_path = os.path.join(models_dir, "diffpf_vs_cfm_per_run.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tracker", "run", "rmse_all", "rmse_visible",
                    "rmse_occluded", "recovery_steps", "ms_per_step"])
        for name in tracker_names:
            s = stats[name]
            for r in range(s["n_runs"]):
                w.writerow([
                    name, r,
                    f"{s['_per_run_all'][r]:.6f}" if r < len(s['_per_run_all']) else "",
                    f"{s['_per_run_vis'][r]:.6f}" if r < len(s['_per_run_vis']) else "",
                    f"{s['_per_run_occ'][r]:.6f}" if r < len(s['_per_run_occ']) else "",
                    "" if s['_per_run_rec'][r] is None else f"{s['_per_run_rec'][r]:.6f}",
                    f"{s['_per_run_ms'][r]:.6f}",
                ])
    print(f"Per-run CSV saved   -> {csv_path}")

    # ── Compact mean +/- std table ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"SUMMARY  ({args.n_runs} runs x {args.n_steps} steps, mean +/- run-to-run std)")
    print("=" * 78)
    print(f"{'tracker':<11}{'RMSE all':>15}{'RMSE vis':>15}"
          f"{'RMSE occ':>15}{'recovery':>13}")
    for name in tracker_names:
        s = stats[name]
        print(f"{name:<11}"
              f"{s['rmse_all_mean']:>8.3f}+-{s['rmse_all_std']:<5.3f}"
              f"{s['rmse_visible_mean']:>8.3f}+-{s['rmse_visible_std']:<5.3f}"
              f"{s['rmse_occluded_mean']:>8.3f}+-{s['rmse_occluded_std']:<5.3f}"
              f"{s['recovery_mean']:>7.2f}+-{s['recovery_std']:<5.2f}")
    print("=" * 78)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
