#!/usr/bin/env python3
"""
sweep_process_noise.py
======================
Choose each particle filter's process noise the same way for every tracker,
on validation runs that the benchmark never uses.

For every PF tracker and every value in the grid, run the exit-prediction
benchmark protocol on runs first_run .. first_run+n_runs-1 (default 100..119;
the benchmark uses 0..99) and record RMSE and exit-prediction scores.
Selection rule, fixed before looking at results: lowest mean RMSE (all steps).

Outputs (in --out-dir):
    noise_sweep.csv          one row per (tracker, noise)
    noise_sweep_runs.csv     one row per (tracker, noise, run)
    chosen_noise.txt         the --process_noise arguments for the benchmark
    noise_sensitivity.png    RMSE / exit success vs process noise per tracker

Usage (after installing the project):
    python -m mml_research.motion.sweep_process_noise --models-dir <dir> --out-dir <dir> --workers 16
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import copy

import numpy as np
import matplotlib.pyplot as plt

from mml_research.motion.benchmark_exit_prediction import (
    PF_TRACKERS, make_parser, parse_noise, run_group, worker_pool, write_csv,
)

GRID = [0.005, 0.01, 0.02, 0.032, 0.045, 0.07]


def main():
    parser = make_parser()
    parser.set_defaults(first_run=100, n_runs=20, trackers=PF_TRACKERS)
    parser.add_argument("--grid", type=float, nargs="+", default=GRID)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = list(range(args.first_run, args.first_run + args.n_runs))
    # One job per (noise value, run); every job runs all selected PF trackers.
    jobs = []
    for q in args.grid:
        job_args = copy.copy(args)
        job_args.process_noise = parse_noise([f"all={q}"])
        jobs += [(job_args, [run]) for run in runs]

    print(f"Noise sweep: {args.trackers} x grid {args.grid} on runs "
          f"{runs[0]}..{runs[-1]} ({len(jobs)} jobs, {args.workers} workers)")
    run_rows, event_rows = [], []
    with worker_pool(args.workers) as pool:
        for r, e in pool.map(run_group, *zip(*jobs)):
            run_rows += r
            event_rows += e

    rows = []
    for name in args.trackers:
        for q in args.grid:
            rr = [r for r in run_rows if r["tracker"] == name and r["process_noise"] == q]
            ev = [e for e in event_rows if e["tracker"] == name and e["process_noise"] == q]
            rmse = np.array([r["rmse_all"] for r in rr])
            rows.append({
                "tracker": name, "process_noise": q, "runs": len(rr),
                "rmse_all": float(rmse.mean()), "rmse_all_std": float(rmse.std()),
                "rmse_visible": float(np.mean([r["rmse_visible"] for r in rr])),
                "rmse_occluded": float(np.nanmean([r["rmse_occluded"] for r in rr])),
                "runs_lost": int((rmse > 0.5).sum()),
                "exit_events": len(ev),
                "exit_success_rate": float(np.mean([e["success"] for e in ev])),
                "exit_majority_rate": float(np.mean([e["majority"] for e in ev])),
                "exit_p_correct": float(np.mean([e["p_correct"] for e in ev])),
            })
    write_csv(os.path.join(args.out_dir, "noise_sweep.csv"), rows)
    write_csv(os.path.join(args.out_dir, "noise_sweep_runs.csv"), run_rows)

    chosen = {}
    print(f"\n{'Tracker':<11}{'noise':>7}{'RMSE all':>10}{'RMSE vis':>10}{'RMSE occ':>10}"
          f"{'lost':>6}{'top-1':>8}{'>50%':>8}{'P(corr)':>9}")
    for name in args.trackers:
        tr = [r for r in rows if r["tracker"] == name]
        best = min(tr, key=lambda r: r["rmse_all"])
        chosen[name] = best["process_noise"]
        for r in tr:
            mark = "  <- chosen" if r is best else ""
            print(f"{name:<11}{r['process_noise']:>7g}{r['rmse_all']:>10.3f}{r['rmse_visible']:>10.3f}"
                  f"{r['rmse_occluded']:>10.3f}{r['runs_lost']:>6d}{100 * r['exit_success_rate']:>7.1f}%"
                  f"{100 * r['exit_majority_rate']:>7.1f}%{r['exit_p_correct']:>9.3f}{mark}")
        print()

    arg_line = " ".join(f"{k}={v:g}" for k, v in chosen.items())
    with open(os.path.join(args.out_dir, "chosen_noise.txt"), "w") as f:
        f.write(arg_line + "\n")
    print(f"Chosen (lowest RMSE all): --process_noise {arg_line}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("rmse_all", "RMSE all (m)", True), ("exit_majority_rate", "Exit success (>50% mass)", False),
              ("exit_p_correct", "P(correct exit)", False)]
    for ax, (key, label, log) in zip(axes, panels):
        for name in args.trackers:
            tr = sorted((r for r in rows if r["tracker"] == name), key=lambda r: r["process_noise"])
            ax.plot([r["process_noise"] for r in tr], [r[key] for r in tr], marker="o", label=name)
            ax.scatter([chosen[name]], [next(r[key] for r in tr if r["process_noise"] == chosen[name])],
                       s=120, facecolors="none", edgecolors="black", zorder=3)
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("process noise (m per step)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Process-noise sensitivity on validation runs {runs[0]}..{runs[-1]} "
                 "(circle = chosen by lowest RMSE all)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "noise_sensitivity.png"), dpi=150)
    print(f"Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
