#!/usr/bin/env python3
"""Experiment 1: standalone conditional motion-distribution quality."""

from __future__ import annotations

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from mml_research.experiments.experiment_suite_common import (
    EXPERIMENT1_METHODS,
    SCENARIOS,
    canonical_methods,
    confidence_summary,
    ensure_output_dir,
    evaluate_distribution_context,
    load_motion_adapters,
    make_road_context,
    paired_difference_summary,
    save_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate motion models outside the particle filter."
    )
    parser.add_argument("--contexts-per-scenario", type=int, default=120)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--energy-samples", type=int, default=200)
    parser.add_argument("--road-width", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--methods", nargs="+", default=list(EXPERIMENT1_METHODS)
    )
    parser.add_argument(
        "--output-dir", default="experiment_results/experiment1_distribution"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = canonical_methods(args.methods)
    if "PF+CFM-adaptive" in methods:
        raise ValueError(
            "Experiment 1 excludes adaptive sigma because measurement age is undefined."
        )
    output = ensure_output_dir(args.output_dir)
    adapters = load_motion_adapters(methods)
    rows = []
    scenarios = list(SCENARIOS)
    total = len(scenarios) * args.contexts_per_scenario

    for scenario_index, scenario in enumerate(scenarios):
        for context_id in range(args.contexts_per_scenario):
            sequence = scenario_index * args.contexts_per_scenario + context_id + 1
            if sequence == 1 or sequence % max(1, total // 20) == 0:
                print(f"Distribution context {sequence}/{total}")
            context = make_road_context(
                scenario,
                context_id,
                args.seed + 10000 * scenario_index,
            )
            for adapter in adapters.values():
                row, _ = evaluate_distribution_context(
                    adapter,
                    context,
                    args.samples,
                    args.horizon,
                    args.seed + sequence * 101,
                    args.road_width,
                    args.energy_samples,
                )
                rows.append(row)

    raw = pd.DataFrame(rows)
    raw.to_csv(output / "per_context_metrics.csv", index=False)
    summary = confidence_summary(
        raw,
        ["method", "scenario"],
        ["branch_jsd_bits", "energy_distance", "offroad_rate"],
    )
    summary.to_csv(output / "metric_table.csv", index=False)
    summary.to_latex(
        output / "metric_table.tex",
        index=False,
        float_format="%.4f",
        na_rep="--",
    )
    paired_difference_summary(
        raw,
        ["scenario", "context_id"],
        ["branch_jsd_bits", "energy_distance", "offroad_rate"],
        reference_method="PF+CFM-fixed",
    ).to_csv(output / "paired_tests_vs_cfm_fixed.csv", index=False)

    scenario_labels = {
        "straight": "Straight",
        "two_way_balanced": "Two-way 0.5/0.5",
        "two_way_unbalanced": "Two-way 0.7/0.3",
        "three_way": "Three-way 0.5/0.3/0.2",
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, scenario in zip(axes.ravel(), scenarios):
        subset = raw[raw["scenario"] == scenario]
        probabilities = SCENARIOS[scenario][1]
        branch_columns = [f"predicted_branch_{i}" for i in range(len(probabilities))]
        x = np.arange(len(probabilities))
        width = 0.8 / (len(methods) + 1)
        axis.bar(
            x - 0.4 + width / 2,
            probabilities,
            width,
            label="Simulator truth",
            color="black",
        )
        for method_index, method in enumerate(methods):
            means = subset[subset["method"] == method][branch_columns].mean().to_numpy()
            axis.bar(
                x - 0.4 + width * (method_index + 1.5),
                means,
                width,
                label=method,
            )
        axis.set_title(scenario_labels[scenario])
        axis.set_xticks(x, [f"Branch {i + 1}" for i in x])
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Probability mass")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.savefig(output / "branch_probability_plots.png", dpi=200)
    plt.close(figure)

    save_protocol(
        output / "PROTOCOL.md",
        "Experiment 1 protocol",
        [
            "Motion models are evaluated outside the particle filter.",
            f"Scenarios: {', '.join(scenarios)}.",
            f"Contexts per scenario: {args.contexts_per_scenario}.",
            f"Predicted and simulator futures per context: {args.samples} each.",
            f"Prediction horizon: {args.horizon} updates.",
            "Branch JSD includes off-road as an additional category, so invalid mass is not hidden.",
            f"Energy Distance uses full flattened trajectories and at most {args.energy_samples} samples per set.",
            f"Off-road means the 90th-percentile distance to the nearest branch centerline exceeds {args.road_width:.3f} m.",
            "Lower is better for all three metrics.",
        ],
    )
    print(f"Experiment 1 outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
