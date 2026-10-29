#!/usr/bin/env python3
"""
run.py — run methods on a split of episodes; one job per (method, seed, episode).

    python -m mml_research.motion.cfm_variants.run --split test --workers 16
    python -m mml_research.motion.cfm_variants.run --split dev --methods PF+CFM PF+DMMN PF+NF

Filters run on CPU (one torch thread per worker) as in benchmark_exit_prediction.
`--split test` refuses to run without locked.json (written by calibrate.py) and
checks that every checkpoint still has the hash recorded there.

Outputs in <out-dir>/<split>/:
    episodes.csv   one row per (method, seed, episode)
    events.csv     one row per (method, seed, exit event)
    steps_<method>.npz   per-step error / visibility / step time, ESS before
                   resampling and resample flags, concatenated over episodes
    clouds.npz     fork clouds of the first events (seed 0)
    manifest.json  commit, command, config, checkpoint hashes, hardware, seeds
"""

from __future__ import annotations

import os

# Filters run on CPU; set before torch is imported (also in spawned workers).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# One BLAS/OpenMP thread per worker process (the pools supply the parallelism).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from mml_research.motion.cfm_variants.common import (
    CONFIG_PATH, DEFAULT_MODELS_DIR, DEFAULT_OUT_DIR, FILTER_SEED, MEAS_SEED, SPLITS, TRAJ_SEED,
    git_commit, load_config, read_json, sha256, write_json,
)
from mml_research.motion.cfm_variants.evaluation import run_episode
from mml_research.motion.cfm_variants.methods import (
    ALL_METHODS, METHODS, SEEDED, ModelCache, checkpoint_files, resolve_q,
)

_CACHE = None
# Relative cost, so the slow jobs start first.
_COST = {"m3": 3.0, "m1": 1.3, "m4": 1.6, "gauss": 0.8, "cfm": 1.0, "cfm_as": 1.0, "nf": 1.0, "dmmn": 0.6,
         "tcfm": 36.0, "trajflow": 8.0, "moflow": 3.5, "enff": 0.4, "cv": 0.1, "diffpf": 0.2}


def _job(cfg_path, models_dir, out_dir, name, seed, q, run, n_steps, keep_clouds):
    global _CACHE
    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    if _CACHE is None:
        _CACHE = ModelCache(cfg, models_dir, out_dir)
    factory = _CACHE.tracker_factory(name, seed, q)
    run_row, events, steps, clouds = run_episode(factory, run, n_steps, cfg["evaluation"], keep_clouds)
    tag = {"method": name, "seed": seed if name in SEEDED else "", "process_noise": q}
    return {**tag, **run_row}, [{**tag, **e} for e in events], (name, seed, run, steps), clouds


def run_jobs(jobs, workers, cfg_path, models_dir, out_dir, n_steps, keep_clouds=0, log=print):
    """jobs: list of (name, seed, q, run). Returns (run_rows, event_rows, steps, clouds)."""
    jobs = sorted(jobs, key=lambda j: -_COST[METHODS[j[0]][0]])
    run_rows, event_rows, steps, clouds = [], [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        futs = [pool.submit(_job, cfg_path, models_dir, out_dir, name, seed, q, run, n_steps,
                            keep_clouds if seed in (0, "") else 0)
                for name, seed, q, run in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r, e, s, c = f.result()
            run_rows.append(r)
            event_rows += e
            steps.append(s)
            clouds += [{"method": r["method"], **x} for x in c]
            if i % max(1, len(jobs) // 50) == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)} jobs done ({time.time() - t0:.0f}s)")
    order = {n: k for k, n in enumerate(ALL_METHODS)}
    key = lambda r: (order[r["method"]], str(r["seed"]), r["process_noise"], r["run"])
    run_rows.sort(key=key)
    event_rows.sort(key=lambda e: (*key(e), e["enter_step"]))
    return run_rows, event_rows, steps, clouds


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def merge_rows(path, new_rows, methods):
    """Existing CSV rows of other methods + new_rows, in ALL_METHODS order (stable within a method)."""
    with open(path, newline="") as f:
        old = [r for r in csv.DictReader(f) if r["method"] not in methods]
    order = {n: k for k, n in enumerate(ALL_METHODS)}
    rows = old + new_rows
    rows.sort(key=lambda r: order[r["method"]])
    fields = list(dict.fromkeys(k for r in rows for k in r))
    return [{k: r.get(k, "") for k in fields} for r in rows]


def write_steps(out, steps):
    by_method = {}
    for name, seed, run, s in steps:
        by_method.setdefault(name, []).append((seed, run, s))
    for name, items in by_method.items():
        items.sort(key=lambda x: (x[0], x[1]))
        arrs = {}
        for key in ("errors", "vis", "step_ms", "ess_pre", "resampled"):
            arrs[key] = np.concatenate([s[key] for _, _, s in items])
            arrs[f"{key}_len"] = np.array([len(s[key]) for _, _, s in items])
        arrs["seed"] = np.array([sd for sd, _, _ in items])
        arrs["run"] = np.array([r for _, r, _ in items])
        safe = name.replace("+", "p").replace("/", "_")
        np.savez_compressed(os.path.join(out, f"steps_{safe}.npz"), **arrs)


def environment():
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count(), "torch_threads_per_worker": 1,
            "device": "cpu", "precision": "float32 networks, float64 weights",
            "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "host": platform.node()}


def main():
    p = argparse.ArgumentParser(description="Run CFM-variant benchmark methods on a split")
    p.add_argument("--split", choices=list(SPLITS), default="test")
    p.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    p.add_argument("--n_runs", type=int, default=None, help="Use only the first n episodes of the split")
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--models-dir", dest="models_dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--tag", default="", help="Suffix for the output folder")
    p.add_argument("--merge", action="store_true",
                   help="Add these methods to the folder's existing results (their old rows are replaced)")
    args = p.parse_args()
    cfg = load_config(args.config)
    seeds = cfg["training"]["seeds"]
    runs = SPLITS[args.split][:args.n_runs]
    out = os.path.join(args.out_dir, args.split + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(out, exist_ok=True)

    files = checkpoint_files(args.methods, seeds, cfg, args.models_dir, args.out_dir)
    hashes = {k: sha256(v) for k, v in files.items()}
    if args.split == "test":
        lock_path = os.path.join(args.out_dir, "locked.json")
        if not os.path.exists(lock_path):
            sys.exit("locked.json missing: run calibrate.py (validation) before the test split")
        locked = read_json(lock_path)
        add_path = os.path.join(args.out_dir, "locked_addendum.json")
        if any(METHODS[m][1] == "addendum" for m in args.methods) and not os.path.exists(add_path):
            sys.exit("locked_addendum.json missing: run lock_addendum.py before testing the open-source methods")
        if os.path.exists(add_path):
            locked["checkpoints"] = {**read_json(add_path)["checkpoints"], **locked["checkpoints"]}
        changed = [k for k, h in hashes.items() if k in locked["checkpoints"] and locked["checkpoints"][k] != h]
        missing = [k for k in hashes if k not in locked["checkpoints"]]
        if changed or missing:
            sys.exit(f"checkpoints differ from locked.json: changed={changed} missing={missing}")
        if sha256(args.config) != locked["config_sha256"]:
            sys.exit("config.yaml changed since locking")

    jobs = []
    for name in args.methods:
        q = resolve_q(name, cfg, args.out_dir)
        for seed in (seeds if name in SEEDED else [""]):
            jobs += [(name, seed if seed != "" else 0, q, run) for run in runs]
    print(f"{args.split}: {len(args.methods)} methods, episodes {runs[0]}..{runs[-1]} ({len(runs)}), "
          f"{len(jobs)} jobs, {args.workers} workers -> {out}", flush=True)
    t0 = time.time()
    run_rows, event_rows, steps, clouds = run_jobs(
        jobs, args.workers, args.config, args.models_dir, args.out_dir, args.n_steps,
        cfg["evaluation"]["clouds_saved_per_method"], log=lambda m: print(m, flush=True))
    for rows in (run_rows, event_rows):
        for r in rows:
            r["split"] = args.split
    manifest = "manifest.json"
    if args.merge and os.path.exists(os.path.join(out, "episodes.csv")):
        run_rows = merge_rows(os.path.join(out, "episodes.csv"), run_rows, args.methods)
        event_rows = merge_rows(os.path.join(out, "events.csv"), event_rows, args.methods)
        old = os.path.join(out, "clouds.npz")
        if os.path.exists(old):
            kept = [c for c in np.load(old, allow_pickle=True)["clouds"] if c["method"] not in args.methods]
            clouds = kept + clouds
        manifest = f"manifest_merge_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_csv(os.path.join(out, "episodes.csv"), run_rows)
    write_csv(os.path.join(out, "events.csv"), event_rows)
    write_steps(out, steps)
    if clouds:
        np.savez_compressed(os.path.join(out, "clouds.npz"), clouds=np.array(clouds, dtype=object))
    write_json(os.path.join(out, manifest), {
        "commit": git_commit(), "command": " ".join(sys.argv), "config": cfg,
        "config_sha256": sha256(args.config), "checkpoints": files, "checkpoint_sha256": hashes,
        "split": args.split, "runs": runs, "n_steps": args.n_steps,
        "seed_streams": {"trajectory": f"{TRAJ_SEED}+run", "measurement": f"{MEAS_SEED}+run",
                         "filter_global_np_and_torch": f"{FILTER_SEED}+run"},
        "process_noise": {n: resolve_q(n, cfg, args.out_dir) for n in args.methods},
        "environment": environment(), "wall_seconds": time.time() - t0,
    })
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
