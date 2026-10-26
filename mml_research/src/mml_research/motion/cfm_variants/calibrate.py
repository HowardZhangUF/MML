#!/usr/bin/env python3
"""
calibrate.py — M2 noise calibration on validation episodes, then lock.

Frozen CFM (weights, source, solver); only the added position noise q
(Q = q^2 I) varies over config m2.grid. Preregistered rule (config m2):
lowest mean validation energy score among grid values whose mean RMSE (all
steps) is within rmse_tolerance of the grid's best RMSE; ties -> smaller q.

Also writes locked.json: the selected q, the selection table, and the hashes
of every checkpoint and of config.yaml. run.py --split test refuses to run if
any of them changed afterwards.

    python -m mml_research.motion.cfm_variants.calibrate --workers 16
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# One BLAS/OpenMP thread per worker process (the pools supply the parallelism).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import numpy as np

from mml_research.motion.cfm_variants.common import (
    CONFIG_PATH, DEFAULT_MODELS_DIR, DEFAULT_OUT_DIR, SPLITS, git_commit, load_config, sha256, write_json,
)
from mml_research.motion.cfm_variants.methods import ALL_METHODS, checkpoint_files
from mml_research.motion.cfm_variants.run import run_jobs, write_csv


def select(rows, tol):
    best_rmse = min(r["rmse_all"] for r in rows)
    ok = [r for r in rows if r["rmse_all"] <= (1 + tol) * best_rmse]
    return min(ok, key=lambda r: (r["es_mean"], r["process_noise"]))


def main():
    p = argparse.ArgumentParser(description="M2: choose the CFM position noise on validation episodes")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--n_runs", type=int, default=None)
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--models-dir", dest="models_dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    cfg = load_config(args.config)
    grid, tol = cfg["m2"]["grid"], cfg["m2"]["rmse_tolerance"]
    runs = SPLITS[cfg["m2"]["selection_split"]][:args.n_runs]
    out = os.path.join(args.out_dir, "m2_calibration")
    os.makedirs(out, exist_ok=True)

    jobs = [("PF+CFM", 0, float(q), run) for q in grid for run in runs]
    print(f"M2 sweep: q in {grid} on episodes {runs[0]}..{runs[-1]} ({len(jobs)} jobs)", flush=True)
    t0 = time.time()
    run_rows, event_rows, _, _ = run_jobs(jobs, args.workers, args.config, args.models_dir, args.out_dir,
                                          args.n_steps, log=lambda m: print(m, flush=True))
    rows = []
    for q in grid:
        rr = [r for r in run_rows if r["process_noise"] == q]
        ev = [e for e in event_rows if e["process_noise"] == q]
        rows.append({
            "process_noise": q, "runs": len(rr),
            **{k: float(np.nanmean([r[k] for r in rr])) for k in
               ("rmse_all", "rmse_visible", "rmse_occluded", "es_mean", "es_occluded", "ess_pre_mean",
                "off_road_mass", "ms_per_step")},
            "lost": int(sum(r["lost"] for r in rr)),
            "exit_success_rate": float(np.mean([e["success"] for e in ev])),
            "exit_p_correct": float(np.mean([e["p_correct"] for e in ev])),
            "brier": float(np.mean([e["brier"] for e in ev])),
        })
    chosen = select(rows, tol)
    write_csv(os.path.join(out, "m2_sweep.csv"), rows)
    write_csv(os.path.join(out, "m2_sweep_runs.csv"), run_rows)
    print(f"\n{'q':>7}{'RMSE all':>10}{'RMSE occ':>10}{'ES':>9}{'top-1':>8}{'P(corr)':>9}{'Brier':>8}")
    for r in rows:
        mark = "  <- selected" if r is chosen else ""
        print(f"{r['process_noise']:>7g}{r['rmse_all']:>10.4f}{r['rmse_occluded']:>10.4f}{r['es_mean']:>9.4f}"
              f"{100 * r['exit_success_rate']:>7.1f}%{r['exit_p_correct']:>9.3f}{r['brier']:>8.3f}{mark}")

    seeds = cfg["training"]["seeds"]
    files = checkpoint_files(ALL_METHODS, seeds, cfg, args.models_dir, args.out_dir)
    write_json(os.path.join(args.out_dir, "locked.json"), {
        "m2_selected_q": chosen["process_noise"],
        "m2_rule": f"min mean val energy score s.t. RMSE_all <= (1+{tol}) * min grid RMSE; ties -> smaller q",
        "m2_table": rows, "validation_runs": runs,
        "checkpoints": {k: sha256(v) for k, v in files.items()}, "checkpoint_files": files,
        "config_sha256": sha256(args.config), "commit": git_commit(),
        "locked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"selected q = {chosen['process_noise']:g}; locked.json written ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
