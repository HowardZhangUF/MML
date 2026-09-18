#!/usr/bin/env python3
"""Experiment 4: full active multi-target simulation factorial benchmark."""

from __future__ import annotations

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from mml_research.experiments.experiment_suite_common import (
    ALL_METHODS,
    canonical_methods,
    confidence_summary,
    ensure_output_dir,
    paired_difference_summary,
    save_protocol,
)


ROOT = project_root()
FULL_METHOD_MAP = {
    "PF+CV": "MultiPFVel",
    "PF+DMMN": "MultiPFTra",
    "PF+Diffusion": "PF+Diffusion",
    "PF+NF": "PF+NF",
    "PF+CFM-fixed": "PF+CFM",
    "PF+CFM-adaptive": "PF+CFM-AS",
}
GUIDANCE_MAP = {
    "predefined": "Predefined",
    "information": "MultiPFInfo",
}
COMPACT_METHODS = ("PF+CV", "PF+CFM-fixed", "PF+CFM-adaptive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full info_mml_main.py active-tracking factorial."
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--base-seed", type=int, default=4701)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(ALL_METHODS),
        help=(
            "Default is all six core methods (12 cells across two guidance modes). "
            "Pass PF+CV PF+CFM-fixed PF+CFM-adaptive for the compact six-cell table."
        ),
    )
    parser.add_argument(
        "--guidance", nargs="+", choices=list(GUIDANCE_MAP), default=list(GUIDANCE_MAP)
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--output-dir", default="experiment_results/experiment4_active"
    )
    return parser.parse_args()


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def run_one(
    args: argparse.Namespace,
    output: Path,
    method: str,
    guidance: str,
    run: int,
) -> dict:
    internal_method = FULL_METHOD_MAP[method]
    internal_guidance = GUIDANCE_MAP[guidance]
    seed = args.base_seed + run
    command = [
        args.python,
        "-m",
        "scripts.info_mml_main",
        "--prediction_method",
        internal_method,
        "--guidance_mode",
        internal_guidance,
        "--run_number",
        str(run),
        "--seed",
        str(seed),
        "--num_steps",
        str(args.steps),
        "--output_dir",
        str(output),
        "--initial_target_mode",
        "visible",
        "--benchmark_metrics",
        "--headless",
        "--no_fan",
    ]
    env = os.environ.copy()
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_dir = ensure_output_dir(output / "logs")
    log_path = log_dir / f"{method}_{guidance}_run{run:03d}.log"
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Full simulation failed for {method}/{guidance}/run {run}. "
            f"See {log_path}."
        )

    benchmark_path = output / f"{internal_method}_run{run:03d}_benchmark.csv"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing benchmark output: {benchmark_path}")
    benchmark = pd.read_csv(benchmark_path).iloc[0]
    pattern = (
        f"{internal_method}_{internal_guidance}_run{run:03d}_seed{seed}_"
        "tracking_metrics_*.csv"
    )
    metric_paths = sorted(output.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not metric_paths:
        raise FileNotFoundError(f"No per-step metric file matched {pattern!r}.")
    metrics = pd.read_csv(metric_paths[-1])
    cardinality = pd.to_numeric(metrics["cardinality_error"], errors="coerce")
    return {
        "method": method,
        "guidance": guidance,
        "run": run,
        "seed": seed,
        "ospa": finite_mean(metrics["ospa"]),
        "cardinality_error": finite_mean(cardinality),
        "track_loss_rate": float((cardinality.fillna(1.0) > 0).mean()),
        "localization_rmse": finite_mean(metrics["rmse"]),
        "tracking_hz": 1000.0 / float(benchmark["ms_per_step"]),
        "filter_ms_per_update": float(benchmark["ms_per_step"]),
        "matched_target_fraction": float(benchmark["matched_target_fraction"]),
        "rmse_all_matched": float(benchmark["rmse_all"]),
        "metric_file": str(metric_paths[-1]),
        "log_file": str(log_path),
    }


def main() -> None:
    args = parse_args()
    methods = canonical_methods(args.methods)
    output = ensure_output_dir(args.output_dir)
    per_run_path = output / "per_run_metrics.csv"
    rows = []
    if args.skip_existing and per_run_path.exists():
        rows = pd.read_csv(per_run_path).to_dict("records")
    complete = {
        (str(row["method"]), str(row["guidance"]), int(row["run"])) for row in rows
    }
    total = len(methods) * len(args.guidance) * args.runs
    count = 0
    for method in methods:
        for guidance in args.guidance:
            for run in range(args.runs):
                count += 1
                key = (method, guidance, run)
                if key in complete:
                    print(f"Skipping existing {count}/{total}: {key}")
                    continue
                print(f"Full simulation {count}/{total}: {method}, {guidance}, run={run}")
                rows.append(run_one(args, output, method, guidance, run))
                pd.DataFrame(rows).to_csv(per_run_path, index=False)

    per_run = pd.DataFrame(rows)
    summary = confidence_summary(
        per_run,
        ["method", "guidance"],
        [
            "ospa",
            "cardinality_error",
            "track_loss_rate",
            "localization_rmse",
            "tracking_hz",
            "filter_ms_per_update",
            "matched_target_fraction",
            "rmse_all_matched",
        ],
    )
    summary.to_csv(output / "full_system_table.csv", index=False)
    summary.to_latex(
        output / "full_system_table.tex",
        index=False,
        float_format="%.4f",
        na_rep="--",
    )
    paired_rows = []
    for guidance in args.guidance:
        subset = per_run[per_run["guidance"] == guidance]
        paired = paired_difference_summary(
            subset,
            ["run", "seed"],
            [
                "ospa",
                "cardinality_error",
                "track_loss_rate",
                "localization_rmse",
                "tracking_hz",
            ],
            reference_method="PF+CFM-adaptive",
        )
        if len(paired):
            paired.insert(0, "guidance", guidance)
            paired_rows.append(paired)
    if paired_rows:
        pd.concat(paired_rows, ignore_index=True).to_csv(
            output / "paired_tests_vs_cfm_adaptive.csv", index=False
        )
    save_protocol(
        output / "PROTOCOL.md",
        "Experiment 4 protocol",
        [
            "This experiment calls scripts/info_mml_main.py for the full simulation stack.",
            f"Methods: {', '.join(methods)}.",
            f"Guidance conditions: {', '.join(args.guidance)}.",
            f"Runs per configuration: {args.runs}; simulation steps: {args.steps}.",
            f"Paired run seeds are base_seed+run with base_seed={args.base_seed}.",
            "Predefined guidance follows a fixed lawnmower path independent of detections.",
            "Information guidance is MultiPFInfo and may change visibility by design.",
            "Every trial starts with every target visible and outside occlusions.",
            "Primary metric: mean per-update OSPA. Supporting metrics: cardinality error, track-loss rate, localization RMSE, and tracking Hz.",
            "Track-loss rate is the fraction of tracking updates with nonzero cardinality error.",
        ],
    )
    print(f"Experiment 4 outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
