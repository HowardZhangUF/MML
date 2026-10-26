#!/usr/bin/env python3
"""
prepare.py — causal PF rollout dataset on the train episodes (guide 4.3, 9 step 4).

For each train episode one bootstrap PF (tracker/noise cycling through
config prepare.rollout_trackers) runs the benchmark loop. Before every
`every`-th step k it logs, for `per_step` particles drawn by weight:

    hist  (H, 2)  the particle's positions x_{k-H}..x_{k-1}
    mask  (H,)    measurement used at those steps (cloud-level)
    d, U          time since last measurement, cloud covariance summary at k-1
    v             true physical velocity (x_true_k - x_true_{k-1}) / dt
    vis           whether z_k exists (M3 trains on these contexts)
    run, k

The particle draw uses its own RNG, so logging does not change the filter.
Contexts and targets stay paired; nothing is shuffled here.

    python -m mml_research.motion.cfm_variants.prepare --workers 16
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# One BLAS/OpenMP thread per worker process (the pools supply the parallelism).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from mml_research.motion.cfm_variants.common import (
    CONFIG_PATH, DEFAULT_MODELS_DIR, DEFAULT_OUT_DIR, FILTER_SEED, MEAS_SEED, SPLITS, TRAJ_SEED,
    git_commit, load_config, sha256, write_json,
)
from mml_research.motion.cfm_variants.methods import ModelCache
from mml_research.motion.cfm_variants.tracker import MotionSampler, VariantPFTracker
from mml_research.motion.generate_synthetic_data import DT, NODE_POS, NOISE_STD
from mml_research.motion.train_diffpf import simulate_full_trajectory


def rollout(cfg_path, models_dir, run, which, q, n_steps):
    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    pc = cfg["prepare"]
    cache = ModelCache(cfg, models_dir, "")
    model = cache.cfm() if which == "cfm" else cache.model("PF+DMMN", 0)
    positions, visible = simulate_full_trajectory(run % len(NODE_POS), n_steps,
                                                  np.random.default_rng(TRAJ_SEED + run))
    meas_rng = np.random.default_rng(MEAS_SEED + run)
    np.random.seed(FILTER_SEED + run)
    torch.manual_seed(FILTER_SEED + run)
    pick_rng = np.random.default_rng(77_000 + run)
    f = cfg["filter"]
    tr = VariantPFTracker(MotionSampler(model, q), f["particles"], f["measurement_sigma"], f["init_spread"])
    first = int(np.argmax(visible))
    tr.reset(positions[first] + NOISE_STD * meas_rng.standard_normal(2).astype(np.float32))
    out = {k: [] for k in ("hist", "mask", "d", "U", "v", "vis", "run", "k")}
    P = pc["per_step"]
    for t in range(first + 1, len(positions)):
        meas = positions[t] + NOISE_STD * meas_rng.standard_normal(2).astype(np.float32) if visible[t] else None
        if t % pc["every"] == 0:
            ctx = tr.context()
            pick = pick_rng.choice(tr.N, P, replace=True, p=tr.weights)
            out["hist"].append(tr.particles[:, pick, :].transpose(1, 0, 2))
            out["mask"].append(np.repeat(ctx.mask[None], P, 0))
            out["d"].append(np.full(P, ctx.d, np.float32))
            out["U"].append(np.repeat(ctx.U[None], P, 0))
            out["v"].append(np.repeat(((positions[t] - positions[t - 1]) / DT)[None], P, 0))
            out["vis"].append(np.full(P, visible[t]))
            out["run"].append(np.full(P, run))
            out["k"].append(np.full(P, t))
        tr.step(meas)
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description="Collect causal PF rollout contexts on train episodes")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--n_runs", type=int, default=None)
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--models-dir", dest="models_dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    cfg = load_config(args.config)
    combos = cfg["prepare"]["rollout_trackers"]
    runs = SPLITS["train"][:args.n_runs]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Rollouts on train episodes {runs[0]}..{runs[-1]} with {combos}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(rollout, args.config, args.models_dir, run, *combos[i % len(combos)], args.n_steps)
                for i, run in enumerate(runs)]
        parts = [f.result() for f in futs]
    data = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
    path = os.path.join(args.out_dir, "rollouts.npz")
    np.savez_compressed(path, **data)
    occ = 1 - data["vis"].mean()
    print(f"{len(data['v'])} contexts ({100 * occ:.1f}% before an occluded step) in {time.time() - t0:.0f}s -> {path}")
    write_json(os.path.join(args.out_dir, "rollouts_manifest.json"), {
        "commit": git_commit(), "runs": runs, "rollout_trackers": combos, "n_steps": args.n_steps,
        "n_contexts": int(len(data["v"])), "occluded_fraction": float(occ), "sha256": sha256(path),
        "fields": {"hist": "(H,2) particle positions x_{k-H}..x_{k-1}", "mask": "(H,) measurement used",
                   "d": "seconds since last measurement", "U": "cloud covariance summary at k-1",
                   "v": "true velocity (x_k - x_{k-1})/dt", "vis": "z_k exists"},
    })


if __name__ == "__main__":
    main()
