#!/usr/bin/env python3
"""Analyze the existing Experiment 1-3 CSVs without rerunning models."""

from __future__ import annotations

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = project_root()
DEFAULT_INPUT = ROOT / "experiment_results/paper_suite"
DEFAULT_OUTPUT = DEFAULT_INPUT / "analysis_existing"

METHOD_ORDER = [
    "PF+CV",
    "PF+DMMN",
    "PF+Diffusion",
    "PF+NF",
    "PF+CFM-fixed",
    "PF+CFM-adaptive",
]
COLORS = {
    "PF+CV": "#2f4858",
    "PF+DMMN": "#7a6c5d",
    "PF+Diffusion": "#2a9d8f",
    "PF+NF": "#457b9d",
    "PF+CFM-fixed": "#e76f51",
    "PF+CFM-adaptive": "#c1121f",
}
E1_DOMAIN = {
    "straight": "OOD-location: straight",
    "two_way_balanced": "OOD-location: balanced two-way",
    "two_way_unbalanced": "OOD-location+prior: 0.7/0.3",
    "three_way": "OOD-topology: three-way",
}
E2_DOMAIN = {
    "straight": "Procedural OOD: straight",
    "two_way": "Procedural OOD: two-way",
    "three_way": "Procedural OOD: three-way",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate statistical and paper-facing analysis from Experiments 1-3."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=91021)
    return parser.parse_args()


def bootstrap_mean_ci(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    chunk = 1000
    means = []
    for start in range(0, repetitions, chunk):
        size = min(chunk, repetitions - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means.append(values[indices].mean(axis=1))
    samples = np.concatenate(means)
    return tuple(np.quantile(samples, [0.025, 0.975]).astype(float))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return pd.Series(adjusted, index=p_values.index)
    ordered = valid[np.argsort(values[valid])]
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def paired_effect(
    candidate: np.ndarray,
    reference: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    keep = np.isfinite(candidate) & np.isfinite(reference)
    differences = candidate[keep] - reference[keep]
    if len(differences) == 0:
        return {
            "n_pairs": 0,
            "mean_difference": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "wilcoxon_p": np.nan,
        }
    low, high = bootstrap_mean_ci(differences, repetitions, rng)
    if np.allclose(differences, 0.0):
        p_value = 1.0
    else:
        try:
            p_value = float(
                wilcoxon(
                    differences,
                    zero_method="pratt",
                    alternative="two-sided",
                    method="auto",
                ).pvalue
            )
        except ValueError:
            p_value = np.nan
    return {
        "n_pairs": len(differences),
        "mean_difference": float(differences.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "wilcoxon_p": p_value,
    }


def mcnemar_effect(candidate: np.ndarray, reference: np.ndarray) -> dict:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    keep = np.isfinite(candidate) & np.isfinite(reference)
    candidate = candidate[keep].astype(int)
    reference = reference[keep].astype(int)
    candidate_only = int(np.sum((candidate == 1) & (reference == 0)))
    reference_only = int(np.sum((candidate == 0) & (reference == 1)))
    discordant = candidate_only + reference_only
    p_value = (
        float(binomtest(candidate_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    differences = candidate - reference
    rng = np.random.default_rng(87031 + len(candidate))
    low, high = bootstrap_mean_ci(differences, 10000, rng)
    return {
        "n_pairs": len(candidate),
        "candidate_success_rate": float(candidate.mean()) if len(candidate) else np.nan,
        "reference_success_rate": float(reference.mean()) if len(reference) else np.nan,
        "mean_difference": float(differences.mean()) if len(differences) else np.nan,
        "ci95_low": low,
        "ci95_high": high,
        "candidate_only_successes": candidate_only,
        "reference_only_successes": reference_only,
        "mcnemar_exact_p": p_value,
    }


def group_mean_summary(
    frame: pd.DataFrame,
    groups: list[str],
    metrics: list[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for keys, subset in frame.groupby(groups, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["n_rows"] = len(subset)
        for metric in metrics:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy()
            low, high = bootstrap_mean_ci(values, repetitions, rng)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def save_latex(frame: pd.DataFrame, path: Path) -> None:
    frame.to_latex(path, index=False, float_format="%.4f", na_rep="--")


def audit_data(e1: pd.DataFrame, e2: pd.DataFrame, e3: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add(name: str, observed, expected, passed: bool, note: str = "") -> None:
        checks.append(
            {
                "check": name,
                "observed": observed,
                "expected": expected,
                "passed": bool(passed),
                "note": note,
            }
        )

    add("Experiment 1 rows", len(e1), 2400, len(e1) == 2400)
    e1_counts = e1.groupby(["method", "scenario"]).size()
    add(
        "Experiment 1 rows per method/scenario",
        f"{e1_counts.min()}..{e1_counts.max()}",
        120,
        bool((e1_counts == 120).all()),
    )
    add("Experiment 1 duplicate rows", int(e1.duplicated().sum()), 0, not e1.duplicated().any())
    add("Experiment 2 rows", len(e2), 3600, len(e2) == 3600)
    e2_counts = e2.groupby(["method", "road_type", "gap_length"]).size()
    add(
        "Experiment 2 rows per method/road/gap",
        f"{e2_counts.min()}..{e2_counts.max()}",
        40,
        bool((e2_counts == 40).all()),
    )
    methods_per_episode = e2.groupby(["road_type", "gap_length", "episode"])["method"].nunique()
    add(
        "Experiment 2 methods per paired episode",
        f"{methods_per_episode.min()}..{methods_per_episode.max()}",
        6,
        bool((methods_per_episode == 6).all()),
    )
    seeds_per_episode = e2.groupby(["road_type", "gap_length", "episode"])["seed"].nunique()
    add(
        "Experiment 2 shared seed per episode",
        int(seeds_per_episode.max()),
        1,
        bool((seeds_per_episode == 1).all()),
    )
    zero = e2[e2["gap_length"] == 0]
    add(
        "Zero-gap controls excluded from recovery",
        int(zero["recovery_success"].notna().sum()),
        0,
        not zero["recovery_success"].notna().any(),
    )
    nonzero = e2[e2["gap_length"] > 0]
    add(
        "Nonzero-gap recovery denominator",
        int(nonzero["recovery_success"].notna().sum()),
        2880,
        int(nonzero["recovery_success"].notna().sum()) == 2880,
        "480 scheduled events per method.",
    )
    add("Experiment 3 configurations", len(e3), 15, len(e3) == 15)
    add("Experiment 3 duplicate rows", int(e3.duplicated().sum()), 0, not e3.duplicated().any())
    return pd.DataFrame(checks)


def analyze_experiment1(
    e1: pd.DataFrame,
    output: Path,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    e1 = e1.copy()
    e1["evaluation_domain"] = e1["scenario"].map(E1_DOMAIN)
    metrics = ["branch_jsd_bits", "energy_distance", "offroad_rate"]
    stratified = group_mean_summary(
        e1,
        ["method", "scenario", "evaluation_domain"],
        metrics,
        repetitions,
        seed,
    )
    pooled = group_mean_summary(e1, ["method"], metrics, repetitions, seed + 1)
    stratified.to_csv(output / "experiment1_stratified_summary.csv", index=False)
    pooled.to_csv(output / "experiment1_pooled_ood_summary.csv", index=False)
    save_latex(pooled, output / "experiment1_pooled_ood_summary.tex")

    comparisons = []
    rng = np.random.default_rng(seed + 2)
    reference = "PF+CFM-fixed"
    candidates = [method for method in e1["method"].unique() if method != reference]
    scopes = [("pooled_ood", e1)] + [
        (scenario, e1[e1["scenario"] == scenario]) for scenario in e1["scenario"].unique()
    ]
    for scope, scoped in scopes:
        for metric in metrics:
            pivot = scoped.pivot(
                index=["scenario", "context_id"], columns="method", values=metric
            )
            for candidate in candidates:
                effect = paired_effect(
                    pivot[candidate].to_numpy(),
                    pivot[reference].to_numpy(),
                    repetitions,
                    rng,
                )
                comparisons.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "candidate": candidate,
                        "reference": reference,
                        "direction": "candidate_minus_reference; lower is better",
                        **effect,
                    }
                )
    paired = pd.DataFrame(comparisons)
    paired["holm_p"] = holm_adjust(paired["wilcoxon_p"])
    paired.to_csv(output / "experiment1_paired_vs_cfm_fixed.csv", index=False)

    scenarios = list(E1_DOMAIN)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    labels = ["Branch JSD (bits)", "Energy Distance", "Off-road rate"]
    for axis, metric, label in zip(axes, metrics, labels):
        x = np.arange(len(scenarios), dtype=float)
        width = 0.15
        for method_index, method in enumerate(METHOD_ORDER[:-1]):
            subset = stratified[stratified["method"] == method].set_index("scenario")
            values = np.array([subset.loc[item, f"{metric}_mean"] for item in scenarios])
            low = np.array([subset.loc[item, f"{metric}_ci95_low"] for item in scenarios])
            high = np.array([subset.loc[item, f"{metric}_ci95_high"] for item in scenarios])
            axis.bar(
                x + (method_index - 2) * width,
                values,
                width,
                yerr=np.vstack([values - low, high - values]),
                capsize=2,
                label=method,
                color=COLORS[method],
            )
        axis.set_xticks(x, ["Straight", "2-way\nbalanced", "2-way\n0.7/0.3", "3-way"])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=5)
    figure.savefig(output / "experiment1_ood_metrics.png", dpi=220)
    plt.close(figure)
    return pooled, paired


def summarize_experiment2_groups(
    frame: pd.DataFrame,
    groups: list[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    continuous = [
        "failure_aware_latency",
        "mort_success_steps",
        "occlusion_rmse",
        "rmse_all",
        "runtime_ms_per_update",
    ]
    for keys, subset in frame.groupby(groups, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["n_episodes"] = len(subset)
        for metric in ["particle_hit", "recovery_success"]:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna().astype(int)
            successes = int(values.sum()) if len(values) else 0
            low, high = wilson_interval(successes, len(values))
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_n"] = len(values)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        for metric in continuous:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy()
            low, high = bootstrap_mean_ci(values, repetitions, rng)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_n"] = len(values)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_experiment2(
    e2: pd.DataFrame,
    output: Path,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    e2 = e2.copy()
    e2["evaluation_domain"] = e2["road_type"].map(E2_DOMAIN)
    stratified = summarize_experiment2_groups(
        e2,
        ["method", "road_type", "evaluation_domain", "gap_length"],
        repetitions,
        seed,
    )
    pooled_gap = summarize_experiment2_groups(
        e2,
        ["method", "gap_length"],
        repetitions,
        seed + 1,
    )
    stratified.to_csv(output / "experiment2_stratified_summary.csv", index=False)
    pooled_gap.to_csv(output / "experiment2_pooled_by_gap.csv", index=False)

    compact_columns = [
        "method",
        "gap_length",
        "particle_hit_mean",
        "recovery_success_mean",
        "failure_aware_latency_mean",
        "mort_success_steps_mean",
        "occlusion_rmse_mean",
        "rmse_all_mean",
    ]
    save_latex(
        pooled_gap[compact_columns], output / "experiment2_pooled_by_gap.tex"
    )

    reference = "PF+CFM-adaptive"
    candidates = [method for method in e2["method"].unique() if method != reference]
    nonzero = e2[e2["gap_length"] > 0]
    scopes = [("pooled_nonzero", nonzero)] + [
        (
            f"{road}_gap{gap}",
            nonzero[(nonzero["road_type"] == road) & (nonzero["gap_length"] == gap)],
        )
        for road in nonzero["road_type"].unique()
        for gap in sorted(nonzero["gap_length"].unique())
    ]

    binary_rows = []
    continuous_rows = []
    rng = np.random.default_rng(seed + 2)
    pair_index = ["road_type", "gap_length", "episode", "seed"]
    for scope, scoped in scopes:
        for metric in ["particle_hit", "recovery_success"]:
            pivot = scoped.pivot(index=pair_index, columns="method", values=metric)
            for candidate in candidates:
                binary_rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "candidate": candidate,
                        "reference": reference,
                        "direction": "candidate_minus_reference; higher is better",
                        **mcnemar_effect(
                            pivot[candidate].to_numpy(), pivot[reference].to_numpy()
                        ),
                    }
                )
        for metric in [
            "failure_aware_latency",
            "occlusion_rmse",
            "rmse_all",
        ]:
            pivot = scoped.pivot(index=pair_index, columns="method", values=metric)
            for candidate in candidates:
                continuous_rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "candidate": candidate,
                        "reference": reference,
                        "direction": "candidate_minus_reference; lower is better",
                        **paired_effect(
                            pivot[candidate].to_numpy(),
                            pivot[reference].to_numpy(),
                            repetitions,
                            rng,
                        ),
                    }
                )

    binary = pd.DataFrame(binary_rows)
    binary["holm_p"] = holm_adjust(binary["mcnemar_exact_p"])
    binary.to_csv(output / "experiment2_binary_paired_vs_cfm_adaptive.csv", index=False)
    continuous = pd.DataFrame(continuous_rows)
    continuous["holm_p"] = holm_adjust(continuous["wilcoxon_p"])
    continuous.to_csv(
        output / "experiment2_continuous_paired_vs_cfm_adaptive.csv", index=False
    )

    adaptive_rows = []
    for gap in sorted(nonzero["gap_length"].unique()):
        scoped = nonzero[nonzero["gap_length"] == gap]
        for metric in [
            "particle_hit",
            "recovery_success",
            "failure_aware_latency",
            "occlusion_rmse",
            "rmse_all",
        ]:
            pivot = scoped.pivot(index=pair_index, columns="method", values=metric)
            if metric in {"particle_hit", "recovery_success"}:
                effect = mcnemar_effect(
                    pivot["PF+CFM-adaptive"].to_numpy(),
                    pivot["PF+CFM-fixed"].to_numpy(),
                )
                p_value = effect.pop("mcnemar_exact_p")
                effect["test_p"] = p_value
            else:
                effect = paired_effect(
                    pivot["PF+CFM-adaptive"].to_numpy(),
                    pivot["PF+CFM-fixed"].to_numpy(),
                    repetitions,
                    rng,
                )
                p_value = effect.pop("wilcoxon_p")
                effect["test_p"] = p_value
            adaptive_rows.append(
                {
                    "gap_length": gap,
                    "metric": metric,
                    "candidate": "PF+CFM-adaptive",
                    "reference": "PF+CFM-fixed",
                    "direction": (
                        "adaptive_minus_fixed; higher is better"
                        if metric in {"particle_hit", "recovery_success"}
                        else "adaptive_minus_fixed; lower is better"
                    ),
                    **effect,
                }
            )
    adaptive = pd.DataFrame(adaptive_rows)
    adaptive["holm_p"] = holm_adjust(adaptive["test_p"])
    adaptive.to_csv(output / "experiment2_adaptive_sigma_ablation.csv", index=False)

    methods = [method for method in METHOD_ORDER if method in e2["method"].unique()]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    plot_specs = [
        ("recovery_success", "TRR", True),
        ("failure_aware_latency", "Failure-aware latency", False),
        ("occlusion_rmse", "Occlusion RMSE (m)", False),
    ]
    for axis, (metric, label, binary_metric) in zip(axes, plot_specs):
        for method in methods:
            subset = pooled_gap[
                (pooled_gap["method"] == method) & (pooled_gap["gap_length"] > 0)
            ].sort_values("gap_length")
            values = subset[f"{metric}_mean"].to_numpy()
            low = subset[f"{metric}_ci95_low"].to_numpy()
            high = subset[f"{metric}_ci95_high"].to_numpy()
            axis.errorbar(
                subset["gap_length"],
                values,
                yerr=np.vstack([values - low, high - values]),
                marker="o",
                linewidth=1.6,
                capsize=2,
                label=method,
                color=COLORS[method],
            )
        axis.set_xlabel("Occlusion length (updates)")
        axis.set_ylabel(label)
        if binary_metric:
            axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=3)
    figure.savefig(output / "experiment2_recovery_analysis.png", dpi=220)
    plt.close(figure)
    return pooled_gap, adaptive


def pareto_mask(x: np.ndarray, y: np.ndarray, y_higher_is_better: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.ones(len(x), dtype=bool)
    for index in range(len(x)):
        if y_higher_is_better:
            dominates = (x <= x[index]) & (y >= y[index]) & (
                (x < x[index]) | (y > y[index])
            )
        else:
            dominates = (x <= x[index]) & (y <= y[index]) & (
                (x < x[index]) | (y < y[index])
            )
        if np.any(dominates):
            keep[index] = False
    return keep


def analyze_experiment3(e3: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    e3 = e3.copy()
    e3["configuration"] = e3.apply(
        lambda row: f"{row['method']} ({row['budget_name']}={int(row['budget'])})",
        axis=1,
    )
    e3["pareto_energy_vs_inference"] = pareto_mask(
        e3["motion_inference_ms"], e3["energy_distance"], False
    )
    e3["pareto_trr_vs_inference"] = pareto_mask(
        e3["motion_inference_ms"], e3["trr"], True
    )
    e3["pareto_energy_learned_only"] = False
    e3["pareto_trr_learned_only"] = False
    learned_index = e3.index[e3["method"] != "PF+CV"]
    e3.loc[learned_index, "pareto_energy_learned_only"] = pareto_mask(
        e3.loc[learned_index, "motion_inference_ms"],
        e3.loc[learned_index, "energy_distance"],
        False,
    )
    e3.loc[learned_index, "pareto_trr_learned_only"] = pareto_mask(
        e3.loc[learned_index, "motion_inference_ms"],
        e3.loc[learned_index, "trr"],
        True,
    )
    e3.to_csv(output / "experiment3_pareto_table.csv", index=False)
    pareto = e3[
        e3["pareto_energy_vs_inference"]
        | e3["pareto_trr_vs_inference"]
        | e3["pareto_energy_learned_only"]
        | e3["pareto_trr_learned_only"]
    ].copy()
    pareto.to_csv(output / "experiment3_nondominated_configurations.csv", index=False)

    cfm = e3[e3["method"] == "PF+CFM-fixed"].copy()
    diffusion = e3[e3["method"] == "PF+Diffusion"].copy()
    matched_rows = []
    for _, cfm_row in cfm.iterrows():
        nearest_index = (
            diffusion["motion_inference_ms"] - cfm_row["motion_inference_ms"]
        ).abs().idxmin()
        diffusion_row = diffusion.loc[nearest_index]
        matched_rows.append(
            {
                "cfm_nfe": int(cfm_row["budget"]),
                "cfm_ms": cfm_row["motion_inference_ms"],
                "diffusion_ddim_steps": int(diffusion_row["budget"]),
                "diffusion_ms": diffusion_row["motion_inference_ms"],
                "runtime_difference_ms": (
                    cfm_row["motion_inference_ms"] - diffusion_row["motion_inference_ms"]
                ),
                "cfm_energy_distance": cfm_row["energy_distance"],
                "diffusion_energy_distance": diffusion_row["energy_distance"],
                "energy_difference_cfm_minus_diffusion": (
                    cfm_row["energy_distance"] - diffusion_row["energy_distance"]
                ),
                "cfm_trr": cfm_row["trr"],
                "diffusion_trr": diffusion_row["trr"],
                "trr_difference_cfm_minus_diffusion": (
                    cfm_row["trr"] - diffusion_row["trr"]
                ),
            }
        )
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(output / "experiment3_compute_matched_cfm_diffusion.csv", index=False)
    save_latex(matched, output / "experiment3_compute_matched_cfm_diffusion.tex")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for method, subset in e3.groupby("method", sort=False):
        subset = subset.sort_values("motion_inference_ms")
        axes[0].plot(
            subset["motion_inference_ms"],
            subset["energy_distance"],
            marker="o",
            label=method,
            color=COLORS.get(method, "#555555"),
        )
        axes[1].plot(
            subset["motion_inference_ms"],
            subset["trr"],
            marker="o",
            label=method,
            color=COLORS.get(method, "#555555"),
        )
    energy_front = e3[e3["pareto_energy_vs_inference"]]
    trr_front = e3[e3["pareto_trr_vs_inference"]]
    learned_energy_front = e3[e3["pareto_energy_learned_only"]]
    learned_trr_front = e3[e3["pareto_trr_learned_only"]]
    axes[0].scatter(
        energy_front["motion_inference_ms"],
        energy_front["energy_distance"],
        s=110,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label="Pareto",
    )
    axes[1].scatter(
        trr_front["motion_inference_ms"],
        trr_front["trr"],
        s=110,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label="Pareto",
    )
    axes[0].scatter(
        learned_energy_front["motion_inference_ms"],
        learned_energy_front["energy_distance"],
        s=70,
        marker="s",
        facecolors="none",
        edgecolors="#555555",
        linewidths=1.0,
        label="Learned-only Pareto",
    )
    axes[1].scatter(
        learned_trr_front["motion_inference_ms"],
        learned_trr_front["trr"],
        s=70,
        marker="s",
        facecolors="none",
        edgecolors="#555555",
        linewidths=1.0,
        label="Learned-only Pareto",
    )
    axes[0].set(xlabel="Motion inference (ms)", ylabel="Energy Distance")
    axes[1].set(xlabel="Motion inference (ms)", ylabel="TRR", ylim=(-0.03, 1.03))
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=4)
    figure.savefig(output / "experiment3_pareto_analysis.png", dpi=220)
    plt.close(figure)
    return e3, matched


def write_audit_report(audit: pd.DataFrame, output: Path) -> None:
    lines = ["# Existing-data audit", ""]
    for _, row in audit.iterrows():
        status = "PASS" if row["passed"] else "FAIL"
        note = f" {row['note']}" if row["note"] else ""
        lines.append(
            f"- **{status}** `{row['check']}`: observed {row['observed']}; "
            f"expected {row['expected']}.{note}"
        )
    (output / "DATA_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def write_evaluation_report(
    output: Path,
    e1_pooled: pd.DataFrame,
    e2_pooled: pd.DataFrame,
    adaptive: pd.DataFrame,
    e3: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    learned = e1_pooled[e1_pooled["method"] != "PF+CV"]
    min_offroad = float(learned["offroad_rate_mean"].min())
    max_offroad = float(learned["offroad_rate_mean"].max())
    e1_best_jsd = e1_pooled.loc[e1_pooled["branch_jsd_bits_mean"].idxmin(), "method"]
    e1_best_energy = e1_pooled.loc[e1_pooled["energy_distance_mean"].idxmin(), "method"]
    nonzero = e2_pooled[e2_pooled["gap_length"] > 0]
    trr_by_method = nonzero.groupby("method")["recovery_success_mean"].mean()
    e2_best = str(trr_by_method.idxmax())
    adaptive_trr = adaptive[adaptive["metric"] == "recovery_success"]
    adaptive_rmse = adaptive[adaptive["metric"] == "occlusion_rmse"]
    cfm = e3[e3["method"] == "PF+CFM-fixed"]
    diffusion = e3[e3["method"] == "PF+Diffusion"]
    best_cfm_trr = cfm.loc[cfm["trr"].idxmax()]
    best_diff_trr = diffusion.loc[diffusion["trr"].idxmax()]
    global_energy_front = ", ".join(
        e3.loc[e3["pareto_energy_vs_inference"], "configuration"]
    )
    global_trr_front = ", ".join(
        e3.loc[e3["pareto_trr_vs_inference"], "configuration"]
    )
    learned_energy_front = ", ".join(
        e3.loc[e3["pareto_energy_learned_only"], "configuration"]
    )
    learned_trr_front = ", ".join(
        e3.loc[e3["pareto_trr_learned_only"], "configuration"]
    )
    audit_pass = bool(audit["passed"].all())

    lines = [
        "# Evaluation of existing Experiments 1-3",
        "",
        "## Scope",
        "",
        "This report analyzes existing CSV rows only. No simulator, tracker, or model was rerun.",
        "The current Experiment 1 and 2 scenarios are procedural evaluations outside the fixed training graph.",
        "They are therefore labeled OOD. Because no original-map ID partition is present, this report cannot calculate an ID-to-OOD generalization gap.",
        "",
        "## Integrity",
        "",
        f"All automated integrity checks passed: **{audit_pass}**.",
        "Contexts and episodes, rather than samples, particles, or time steps, are the statistical units.",
        "",
        "## Experiment 1: motion distribution",
        "",
        f"The lowest pooled JSD is produced by **{e1_best_jsd}**, and the lowest pooled Energy Distance by **{e1_best_energy}**.",
        f"Across learned methods, pooled off-road rates range from **{min_offroad:.3f}** to **{max_offroad:.3f}**.",
        "These off-road rates are a validity warning. They may represent genuine OOD failure, but an original-map diagnostic is required before attributing them to model architecture.",
        "Do not use the pooled OOD ranking as evidence of in-distribution motion-model quality.",
        "",
        "## Experiment 2: controlled occlusion",
        "",
        f"The highest macro-average TRR over nonzero gaps is produced by **{e2_best}**.",
        f"Adaptive-minus-fixed CFM TRR effects across gaps range from **{adaptive_trr['mean_difference'].min():.3f}** to **{adaptive_trr['mean_difference'].max():.3f}**.",
        f"Adaptive-minus-fixed occlusion-RMSE effects range from **{adaptive_rmse['mean_difference'].min():.3f}** to **{adaptive_rmse['mean_difference'].max():.3f} m**; negative favors adaptive CFM.",
        "TRR and particle-hit comparisons use exact paired McNemar tests. Continuous metrics use paired bootstrap intervals and Wilcoxon tests.",
        "MORT remains success-only and must always be displayed beside TRR or failure-aware latency.",
        "",
        "## Experiment 3: quality-runtime tradeoff",
        "",
        f"The strongest CFM TRR in the sweep is **{best_cfm_trr['trr']:.3f}** at NFE={int(best_cfm_trr['budget'])} and {best_cfm_trr['motion_inference_ms']:.2f} ms.",
        f"The strongest diffusion TRR is **{best_diff_trr['trr']:.3f}** at {int(best_diff_trr['budget'])} DDIM steps and {best_diff_trr['motion_inference_ms']:.2f} ms.",
        f"The global Energy-Distance frontier is **{global_energy_front}**; the global TRR frontier is **{global_trr_front}**.",
        f"After excluding CV, the learned-model Energy-Distance frontier is **{learned_energy_front}**, and the learned-model TRR frontier is **{learned_trr_front}**.",
        "Quality values inherit the same procedural/OOD limitation as Experiments 1 and 2. Runtime values remain useful as descriptive measurements.",
        "The current timing CSV contains one aggregate timing per configuration, so it does not support timing confidence intervals.",
        "",
        "## What can be claimed now",
        "",
        "- Relative performance on the existing procedural/OOD contexts and scheduled occlusions.",
        "- Paired fixed-versus-adaptive CFM effects on those episodes.",
        "- Descriptive quality-runtime and Pareto comparisons on the existing benchmark.",
        "",
        "## What cannot be claimed yet",
        "",
        "- An ID-to-OOD generalization gap, because no comparable original-map ID rows exist.",
        "- Effects of different training-data distributions, because the CSVs do not contain checkpoint or training-distribution identifiers.",
        "- That architecture alone caused off-road failure, before the original-map diagnostic passes.",
        "",
        "## Minimum next evaluation",
        "",
        "Create one cached original-map ID set, run the same frozen checkpoints on it, and append rows with `test_distribution`, `checkpoint_id`, and `train_distribution` columns.",
        "Existing procedural/OOD rows should remain unchanged and become the OOD side of the final comparison.",
    ]
    (output / "EVALUATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    e1_path = args.input_root / "experiment1_distribution/per_context_metrics.csv"
    e2_path = args.input_root / "experiment2_occlusion/per_event_metrics.csv"
    e3_path = args.input_root / "experiment3_runtime/quality_runtime_table.csv"
    for path in [e1_path, e2_path, e3_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input CSV not found: {path}")

    e1 = pd.read_csv(e1_path)
    e2 = pd.read_csv(e2_path)
    e3 = pd.read_csv(e3_path)
    audit = audit_data(e1, e2, e3)
    audit.to_csv(output / "data_audit.csv", index=False)
    write_audit_report(audit, output)
    if not audit["passed"].all():
        raise RuntimeError(f"Data-integrity audit failed; see {output / 'DATA_AUDIT.md'}")

    e1_pooled, _ = analyze_experiment1(
        e1, output, args.bootstrap_repetitions, args.seed
    )
    e2_pooled, adaptive = analyze_experiment2(
        e2, output, args.bootstrap_repetitions, args.seed + 1000
    )
    e3_analyzed, _ = analyze_experiment3(e3, output)
    write_evaluation_report(
        output, e1_pooled, e2_pooled, adaptive, e3_analyzed, audit
    )
    print(f"Existing-data analysis written to: {output}")


if __name__ == "__main__":
    main()
