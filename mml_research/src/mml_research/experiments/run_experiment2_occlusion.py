#!/usr/bin/env python3
"""Experiment 2: passive tracking under scheduled measurement blackouts."""

from __future__ import annotations

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


from mml_research.experiments.experiment_suite_common import (
    ALL_METHODS,
    canonical_methods,
    ensure_output_dir,
    load_motion_adapters,
    paired_difference_summary,
    parse_int_list,
    run_controlled_occlusion_grid,
    save_protocol,
    summarize_occlusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate common bootstrap PFs through controlled occlusion."
    )
    parser.add_argument("--road-types", nargs="+", default=["straight", "two_way", "three_way"])
    parser.add_argument("--gap-lengths", default="0,10,20,40,60")
    parser.add_argument("--episodes-per-cell", type=int, default=40)
    parser.add_argument("--particles", type=int, default=250)
    parser.add_argument("--process-std", type=float, default=0.01)
    parser.add_argument("--measurement-std", type=float, default=0.05)
    parser.add_argument("--hit-radius", type=float, default=0.30)
    parser.add_argument("--recovery-threshold", type=float, default=0.30)
    parser.add_argument("--recovery-streak", type=int, default=3)
    parser.add_argument("--recovery-deadline", type=int, default=20)
    parser.add_argument("--adaptive-gain", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS))
    parser.add_argument(
        "--output-dir", default="experiment_results/experiment2_occlusion"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = canonical_methods(args.methods)
    gaps = parse_int_list(args.gap_lengths)
    output = ensure_output_dir(args.output_dir)
    adapters = load_motion_adapters(methods)
    raw, manifest, episodes = run_controlled_occlusion_grid(
        adapters,
        road_types=args.road_types,
        gap_lengths=gaps,
        episodes_per_cell=args.episodes_per_cell,
        seed=args.seed,
        n_particles=args.particles,
        process_std=args.process_std,
        measurement_std=args.measurement_std,
        hit_radius=args.hit_radius,
        recovery_threshold=args.recovery_threshold,
        recovery_streak=args.recovery_streak,
        recovery_deadline=args.recovery_deadline,
        adaptive_gain=args.adaptive_gain,
        sigma_max=args.sigma_max,
    )
    raw.to_csv(output / "per_event_metrics.csv", index=False)
    manifest.to_csv(output / "episode_manifest.csv", index=False)
    trajectory_rows = []
    for item in episodes:
        for step in range(len(item.truth)):
            trajectory_rows.append(
                {
                    "road_type": item.road_type,
                    "gap_length": item.gap_length,
                    "episode": item.episode,
                    "seed": item.seed,
                    "step": step,
                    "truth_x": item.truth[step, 0],
                    "truth_y": item.truth[step, 1],
                    "sensor_x": item.sensor_positions[step, 0],
                    "sensor_y": item.sensor_positions[step, 1],
                    "measurement_available": int(item.measurement_mask[step]),
                    "measurement_x": (
                        item.measurements[step, 0]
                        if item.measurement_mask[step]
                        else float("nan")
                    ),
                    "measurement_y": (
                        item.measurements[step, 1]
                        if item.measurement_mask[step]
                        else float("nan")
                    ),
                }
            )
    pd.DataFrame(trajectory_rows).to_csv(
        output / "prerecorded_trajectories.csv", index=False
    )
    summary = summarize_occlusion(raw)
    summary.to_csv(output / "metric_table.csv", index=False)
    summary.to_latex(
        output / "metric_table.tex", index=False, float_format="%.4f", na_rep="--"
    )
    paired_difference_summary(
        raw,
        ["road_type", "gap_length", "episode"],
        [
            "particle_hit",
            "recovery_success",
            "failure_aware_latency",
            "mort_success_steps",
            "occlusion_rmse",
            "rmse_all",
        ],
        reference_method="PF+CFM-adaptive",
    ).to_csv(output / "paired_tests_vs_cfm_adaptive.csv", index=False)

    pooled = (
        raw.groupby(["method", "gap_length"], as_index=False)
        .agg(
            trr=("recovery_success", "mean"),
            failure_aware_latency=("failure_aware_latency", "mean"),
            mort=("mort_success_steps", "mean"),
            occlusion_rmse=("occlusion_rmse", "mean"),
            particle_hit_rate=("particle_hit", "mean"),
        )
    )
    pooled.to_csv(output / "pooled_curves.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method in methods:
        subset = pooled[pooled["method"] == method].sort_values("gap_length")
        recovery = subset[subset["gap_length"] > 0]
        axes[0].plot(recovery["gap_length"], recovery["trr"], marker="o", label=method)
        axes[1].plot(
            recovery["gap_length"],
            recovery["failure_aware_latency"],
            marker="o",
            label=method,
        )
    axes[0].set(xlabel="Occlusion length (updates)", ylabel="TRR", ylim=(-0.02, 1.02))
    axes[1].set(
        xlabel="Occlusion length (updates)",
        ylabel="Failure-aware latency score (updates)",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    figure.savefig(output / "recovery_curves.png", dpi=200)
    plt.close(figure)

    n_scheduled = int(manifest["scheduled_recovery_event"].sum())
    save_protocol(
        output / "PROTOCOL.md",
        "Experiment 2 protocol",
        [
            "Targets and sensor paths are generated once per episode and reused by every method.",
            "Occlusion is a scheduled measurement blackout; target dynamics continue normally.",
            f"Road types: {', '.join(args.road_types)}.",
            f"Gap lengths: {gaps}; zero is a control and is excluded from TRR.",
            f"Episodes per road/gap cell: {args.episodes_per_cell}.",
            f"Scheduled nonzero-gap recovery events: {n_scheduled} per method.",
            f"Particle count: {args.particles}; hard systematic resampling is shared.",
            f"Particle hit radius: {args.hit_radius:.3f} m at the last hidden update.",
            f"Recovery: error below {args.recovery_threshold:.3f} m for {args.recovery_streak} consecutive updates, with first-streak onset no later than {args.recovery_deadline} updates after reappearance.",
            f"Failures receive a capped benchmark latency of {args.recovery_deadline + 1}; this is not an observed recovery time.",
            "MORT is the successful-recovery mean only and must be reported beside TRR.",
            f"Adaptive sigma: sigma=min({args.sigma_max:.3f}, 1+{args.adaptive_gain:.3f}*measurement_age).",
        ],
    )
    print(f"Experiment 2 outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
