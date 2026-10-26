#!/usr/bin/env python3
"""
audit.py — freeze the producing experiment (guide 9 steps 1-3).

Writes <out-dir>/baseline_manifest.json: the command and settings that
produced the preliminary exit table (from its log), the hashes of every
baseline checkpoint and result CSV, the metric definitions, units, and the
episode splits with a disjointness check on episode ids and simulator seeds.

    python -m mml_research.motion.cfm_variants.audit
"""

from __future__ import annotations

import argparse
import glob
import os

from mml_research.motion.benchmark_exit_prediction import EXIT_HORIZON, LABELS
from mml_research.motion.cfm_variants.common import (
    CONFIG_PATH, DEFAULT_MODELS_DIR, DEFAULT_OUT_DIR, FILTER_SEED, MEAS_SEED, ROOT, SPLITS, TRAJ_SEED,
    git_commit, load_config, sha256, write_json,
)

METRICS = {
    "process_noise": "std (m) of isotropic Gaussian position jitter added once per predict step; Q = q^2 I",
    "exit top-1 (success)": "label with the largest weighted particle mass, EXIT_HORIZON steps after the target "
                            "re-appears, predicted without measurements from the last occluded step, equals the "
                            "side the target exits; labels N/E/S/W/in, 'in' never counts as an exit; ties -> "
                            "first label in N,E,S,W,in order (np.argmax)",
    "exit >50% (majority)": "more than half the weighted mass lies beyond the actual exit side",
    "P(correct)": "weighted mass beyond the actual exit side (timed-region probability, not a route choice)",
    "RMSE": "per-episode RMSE of the weighted particle mean after the step (post-resampling), averaged over "
            "episodes; +- in the old table = population std across episodes",
    "scoring time": "fork at exit_step-1 (last occluded step), 1+EXIT_HORIZON predict-only steps; the returning "
                    "measurement is never used",
    "lane-cue reference": "events whose exit matches the approach lane (north lane -> N, south lane -> S)",
}


def main():
    p = argparse.ArgumentParser(description="Audit the baseline exit benchmark")
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--models-dir", dest="models_dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    cfg = load_config(args.config)

    log = os.path.join(ROOT, "logs", "benchmark_all13_tuned.log")
    head = open(log).read().splitlines()[:2] if os.path.exists(log) else []
    results = {}
    for d in ("results_all13", os.path.join("hpg", "results_all13")):
        for f in sorted(glob.glob(os.path.join(ROOT, d, "*.csv"))):
            results[os.path.relpath(f, ROOT)] = sha256(f)
    ckpts = {os.path.basename(f): sha256(f) for f in sorted(glob.glob(os.path.join(args.models_dir, "*.pth")))}

    seeds = {name: {"trajectory": {TRAJ_SEED + r for r in runs}, "measurement": {MEAS_SEED + r for r in runs},
                    "filter": {FILTER_SEED + r for r in runs}} for name, runs in SPLITS.items()}
    names = list(SPLITS)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ep = set(SPLITS[a]) & set(SPLITS[b])
            sd = set().union(*seeds[a].values()) & set().union(*seeds[b].values())
            overlaps[f"{a}/{b}"] = {"episodes": len(ep), "seeds": len(sd)}
    disjoint = all(v["episodes"] == 0 and v["seeds"] == 0 for v in overlaps.values())

    write_json(os.path.join(args.out_dir, "baseline_manifest.json"), {
        "commit": git_commit(),
        "producing_runner": "python -m mml_research.motion.benchmark_exit_prediction",
        "producing_log_head": head,
        "producing_settings": {"runs": "0..99", "n_steps": 2000, "pf_particles": 500, "exit_horizon": EXIT_HORIZON,
                               "labels": LABELS, "meas_sigma_filter": 0.05, "sensor_noise_std": 0.02,
                               "resampling": "systematic when ESS < N/2", "estimate": "weighted mean"},
        "baseline_results_sha256": results, "checkpoints_sha256": ckpts, "metric_definitions": METRICS,
        "config": cfg, "splits": {k: [v[0], v[-1], len(v)] for k, v in SPLITS.items()},
        "split_overlaps": overlaps, "splits_disjoint": disjoint,
        "note": "dev = the preliminary table's episodes; test = new episodes 1000..1099 never used before",
    })
    print(f"splits disjoint (episodes and seeds): {disjoint}; {len(ckpts)} checkpoints, {len(results)} result files")
    if not disjoint:
        raise SystemExit("split overlap detected")


if __name__ == "__main__":
    main()
