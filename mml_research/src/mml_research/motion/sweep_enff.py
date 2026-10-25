#!/usr/bin/env python3
"""
sweep_enff.py
=============
Choose the Ensemble Flow Filter's hyperparameters on validation runs, by the
same rule as every other tracker's process noise (lowest mean RMSE over all
steps, runs 100..119 by default; the benchmark uses 0..99).

EnFF (baselines/mml_baselines/enff) has three knobs here: the process noise
q of its constant-velocity forecast, the Local-guidance scale lambda, and the
F2P path std sigma_min (the two the DAFM authors tune per dataset,
sweeps/topk_hyperparameters.csv). All three are searched jointly.

Outputs (in --out-dir):
    enff_sweep.csv        one row per (q, lambda, sigma_min)
    enff_sweep_runs.csv   one row per (q, lambda, sigma_min, run)
    chosen_enff.txt       the benchmark arguments for the chosen setting

Usage (after installing the project, baselines/ on PYTHONPATH):
    python -m mml_research.motion.sweep_enff --models-dir <dir> --out-dir <dir> --workers 24
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import copy
import itertools

import numpy as np

from mml_research.motion.benchmark_exit_prediction import make_parser, parse_noise, run_group, worker_pool, write_csv
from mml_research.motion.sweep_process_noise import GRID

LAMBDAS = [6.25e-5, 1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3]
SIGMA_MINS = [1e-3, 1e-2]


def main():
    parser = make_parser()
    parser.set_defaults(first_run=100, n_runs=20, trackers=["EnFF"])
    parser.add_argument("--grid", type=float, nargs="+", default=GRID)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--sigma_mins", type=float, nargs="+", default=SIGMA_MINS)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = list(range(args.first_run, args.first_run + args.n_runs))
    settings = list(itertools.product(args.grid, args.lambdas, args.sigma_mins))
    jobs, keys = [], []
    for q, lam, smin in settings:
        a = copy.copy(args)
        a.process_noise = parse_noise([f"EnFF={q}"])
        a.enff_lambda, a.enff_sigma_min = lam, smin
        for run in runs:
            jobs.append((a, [run]))
            keys.append((q, lam, smin))
    print(f"EnFF sweep: {len(settings)} settings x runs {runs[0]}..{runs[-1]} "
          f"({len(jobs)} jobs, {args.workers} workers)")

    run_rows, event_rows = [], []
    with worker_pool(args.workers) as pool:
        for (q, lam, smin), (r, e) in zip(keys, pool.map(run_group, *zip(*jobs))):
            for row in r:
                row.update(enff_lambda=lam, enff_sigma_min=smin)
            for row in e:
                row.update(enff_lambda=lam, enff_sigma_min=smin)
            run_rows += r
            event_rows += e

    rows = []
    for q, lam, smin in settings:
        sel = lambda x: (x["process_noise"] == q and x["enff_lambda"] == lam and x["enff_sigma_min"] == smin)
        rr = [r for r in run_rows if sel(r)]
        ev = [e for e in event_rows if sel(e)]
        rmse = np.array([r["rmse_all"] for r in rr])
        rows.append({
            "process_noise": q, "enff_lambda": lam, "enff_sigma_min": smin, "runs": len(rr),
            "rmse_all": float(rmse.mean()), "rmse_all_std": float(rmse.std()),
            "rmse_visible": float(np.mean([r["rmse_visible"] for r in rr])),
            "rmse_occluded": float(np.nanmean([r["rmse_occluded"] for r in rr])),
            "runs_lost": int((rmse > 0.5).sum()),
            "exit_events": len(ev),
            "exit_success_rate": float(np.mean([e["success"] for e in ev])),
            "exit_majority_rate": float(np.mean([e["majority"] for e in ev])),
            "exit_p_correct": float(np.mean([e["p_correct"] for e in ev])),
        })
    write_csv(os.path.join(args.out_dir, "enff_sweep.csv"), rows)
    write_csv(os.path.join(args.out_dir, "enff_sweep_runs.csv"), run_rows)

    print(f"\n{'q':>7}{'lambda':>9}{'sig_min':>9}{'RMSE all':>10}{'RMSE vis':>10}{'RMSE occ':>10}"
          f"{'lost':>6}{'top-1':>8}{'>50%':>8}{'P(corr)':>9}")
    best = min(rows, key=lambda r: r["rmse_all"])
    for r in rows:
        mark = "  <- chosen" if r is best else ""
        print(f"{r['process_noise']:>7g}{r['enff_lambda']:>9g}{r['enff_sigma_min']:>9g}{r['rmse_all']:>10.4f}"
              f"{r['rmse_visible']:>10.4f}{r['rmse_occluded']:>10.4f}{r['runs_lost']:>6d}"
              f"{100 * r['exit_success_rate']:>7.1f}%{100 * r['exit_majority_rate']:>7.1f}%"
              f"{r['exit_p_correct']:>9.3f}{mark}")
    line = (f"--process_noise EnFF={best['process_noise']:g} --enff_lambda {best['enff_lambda']:g} "
            f"--enff_sigma_min {best['enff_sigma_min']:g}")
    with open(os.path.join(args.out_dir, "chosen_enff.txt"), "w") as f:
        f.write(line + "\n")
    print(f"\nChosen (lowest RMSE all): {line}")
    print(f"Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
