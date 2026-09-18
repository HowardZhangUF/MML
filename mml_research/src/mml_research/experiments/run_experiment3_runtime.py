#!/usr/bin/env python3
"""Experiment 3: motion quality and recovery versus inference cost."""

from __future__ import annotations

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from mml_research.experiments.experiment_suite_common import (
    ControlledBootstrapPF,
    SCENARIOS,
    ensure_output_dir,
    evaluate_distribution_context,
    load_motion_adapter,
    make_road_context,
    parse_int_list,
    run_controlled_occlusion_grid,
    save_protocol,
    seed_everything,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the quality-runtime study.")
    parser.add_argument("--cfm-nfe", default="1,2,5,10,20,50")
    parser.add_argument("--diffusion-steps", default="1,2,5,10,20")
    parser.add_argument("--contexts-per-scenario", type=int, default=10)
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--energy-samples", type=int, default=150)
    parser.add_argument("--episodes-per-cell", type=int, default=10)
    parser.add_argument("--gap-lengths", default="20,40,60")
    parser.add_argument("--particles", type=int, default=250)
    parser.add_argument("--timing-repeats", type=int, default=50)
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3701)
    parser.add_argument(
        "--output-dir", default="experiment_results/experiment3_runtime"
    )
    return parser.parse_args()


def build_configurations(cfm_nfe: list[int], diffusion_steps: list[int]):
    configurations = [
        ("PF+CV", "baseline", 1, load_motion_adapter("PF+CV")),
        ("PF+DMMN", "baseline", 1, load_motion_adapter("PF+DMMN")),
        ("PF+NF", "baseline", 1, load_motion_adapter("PF+NF")),
    ]
    for budget in cfm_nfe:
        configurations.append(
            (
                "PF+CFM-fixed",
                "NFE",
                budget,
                load_motion_adapter("PF+CFM-fixed", cfm_nfe=budget),
            )
        )
    configurations.append(
        (
            "PF+CFM-adaptive",
            "NFE",
            2,
            load_motion_adapter("PF+CFM-adaptive", cfm_nfe=2),
        )
    )
    for budget in diffusion_steps:
        configurations.append(
            (
                "PF+Diffusion",
                "DDIM steps",
                budget,
                load_motion_adapter("PF+Diffusion", diffusion_steps=budget),
            )
        )
    return configurations


def time_adapter(adapter, particles: int, warmup: int, repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    history = rng.normal(0.0, 0.25, size=(particles, adapter.block_size, 2)).astype(
        np.float32
    )
    contexts = history.reshape(particles, -1)
    seed_everything(seed)
    for _ in range(warmup):
        adapter.propose(contexts)
    synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        adapter.propose(contexts)
    synchronize()
    inference_ms = 1000.0 * (time.perf_counter() - start) / repeats

    pf = ControlledBootstrapPF(
        adapter,
        np.zeros(2, dtype=np.float32),
        particles,
        np.random.default_rng(seed + 1),
        process_std=0.01,
        measurement_std=0.05,
        adaptive_gain=0.05,
        sigma_max=2.5,
    )
    measurement = np.array([0.1, -0.1], dtype=np.float32)
    for _ in range(warmup):
        pf.step(measurement, measurement_age=0)
    synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        pf.step(measurement, measurement_age=0)
    synchronize()
    pf_ms = 1000.0 * (time.perf_counter() - start) / repeats
    return inference_ms, pf_ms


def distribution_quality(adapter, args, config_index: int) -> float:
    values = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for context_id in range(args.contexts_per_scenario):
            context = make_road_context(
                scenario,
                context_id,
                args.seed + 10000 * scenario_index,
            )
            row, _ = evaluate_distribution_context(
                adapter,
                context,
                args.samples,
                args.horizon,
                args.seed + 100000 * scenario_index + context_id,
                road_width=0.18,
                energy_samples=args.energy_samples,
            )
            values.append(row["energy_distance"])
    return float(np.mean(values))


def recovery_quality(adapter, args, config_index: int) -> float:
    raw, _, _ = run_controlled_occlusion_grid(
        {adapter.name: adapter},
        road_types=["straight", "two_way", "three_way"],
        gap_lengths=parse_int_list(args.gap_lengths),
        episodes_per_cell=args.episodes_per_cell,
        seed=args.seed + 500000,
        n_particles=args.particles,
        process_std=0.01,
        measurement_std=0.05,
        hit_radius=0.30,
        recovery_threshold=0.30,
        recovery_streak=3,
        recovery_deadline=20,
        adaptive_gain=0.05,
        sigma_max=2.5,
    )
    return float(raw["recovery_success"].mean())


def main() -> None:
    args = parse_args()
    cfm_nfe = parse_int_list(args.cfm_nfe)
    diffusion_steps = parse_int_list(args.diffusion_steps)
    output = ensure_output_dir(args.output_dir)
    rows = []
    configurations = build_configurations(cfm_nfe, diffusion_steps)

    for index, (method, budget_name, budget, adapter) in enumerate(configurations):
        print(
            f"Runtime configuration {index + 1}/{len(configurations)}: "
            f"{method}, {budget_name}={budget}"
        )
        inference_ms, pf_ms = time_adapter(
            adapter,
            args.particles,
            args.timing_warmup,
            args.timing_repeats,
            args.seed + index * 100,
        )
        energy = distribution_quality(adapter, args, index)
        trr = recovery_quality(adapter, args, index)
        rows.append(
            {
                "method": method,
                "budget_name": budget_name,
                "budget": budget,
                "motion_inference_ms": inference_ms,
                "complete_pf_update_ms": pf_ms,
                "tracking_hz": 1000.0 / pf_ms,
                "energy_distance": energy,
                "trr": trr,
                "particles": args.particles,
                "prediction_horizon": args.horizon,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(output / "quality_runtime_table.csv", index=False)
    table.to_latex(
        output / "quality_runtime_table.tex",
        index=False,
        float_format="%.4f",
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method, group in table.groupby("method", sort=False):
        ordered = group.sort_values("motion_inference_ms")
        axes[0].plot(
            ordered["motion_inference_ms"],
            ordered["energy_distance"],
            marker="o",
            label=method,
        )
        axes[1].plot(
            ordered["motion_inference_ms"], ordered["trr"], marker="o", label=method
        )
    axes[0].set(xlabel="Motion inference (ms)", ylabel="Energy Distance")
    axes[1].set(xlabel="Motion inference (ms)", ylabel="TRR", ylim=(-0.02, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    figure.savefig(output / "quality_runtime_tradeoff.png", dpi=200)
    plt.close(figure)

    save_protocol(
        output / "PROTOCOL.md",
        "Experiment 3 protocol",
        [
            f"CFM NFE sweep: {cfm_nfe}.",
            f"Diffusion DDIM step sweep: {diffusion_steps}.",
            f"Particle count: {args.particles}; horizon: {args.horizon}.",
            f"Timing uses {args.timing_warmup} warm-up and {args.timing_repeats} measured synchronized calls.",
            "Motion inference and complete PF update are timed separately.",
            "Diffusion uses a valid reduced-step DDIM sampler; the checkpoint noise schedule is not overwritten.",
            "Energy Distance is computed using the Experiment 1 protocol.",
            "TRR is computed using scheduled nonzero measurement blackouts and the Experiment 2 recovery definition.",
        ],
    )
    print(f"Experiment 3 outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
