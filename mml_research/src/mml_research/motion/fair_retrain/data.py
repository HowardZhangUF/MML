#!/usr/bin/env python3
"""
data.py — the two shared training sets (config recipes R0 / R1), one file each.

Every model of a recipe trains on the same rows. Fields (N rows):
    hist  (N, 5, 2)   history positions x_{k-5}..x_{k-1} (absolute, m)
    mask  (N, 5)      1 = measurement used at that history step (R0: all ones)
    d     (N,)        seconds since the last measurement (R0: 0)
    U     (N, 3)      cloud covariance summary at k-1 (R0: zeros)
    V     (N, 12, 2)  future per-interval velocities; V[:, 0] is the one-step target
    run   (N,)        episode index (240..247 = holdout)

R0  measurement histories. Each train episode's measurement stream is drawn
    exactly as evaluation draws it (MEAS_SEED + run, a draw only at the reset
    step and at visible steps). Windows of 5 consecutive measurements whose
    time span is <= MAX_GAP, as generate_synthetic_data.build_dataset; V are
    the velocities between the next 12 measurements, (z_j - z_{j-1}) / (t_j -
    t_{j-1}), the CSV label convention. Noisy, and they skip occlusions.
R1  cfm_variants rollouts.npz rows, unchanged (M1's training data); V are the
    true velocities (x_{k+j} - x_{k+j-1}) / dt, j = 0..11, so V[:, 0] equals
    the rollout target v. Rows within 12 steps of the episode end are dropped.

    python -m mml_research.motion.fair_retrain.data --out-dir artifacts/motion/fair_v1
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from mml_research.motion.cfm_variants.common import (
    MEAS_SEED, TRAIN_HOLDOUT_RUNS, TRAJ_SEED, git_commit, sha256, write_json,
)
from mml_research.motion.fair_retrain.common import DEFAULT_OUT_DIR, VARIANTS_DIR, load_config
from mml_research.motion.generate_synthetic_data import DT, MAX_GAP, NODE_POS, NOISE_STD
from mml_research.motion.train_diffpf import simulate_full_trajectory

H = 5


def episode(run: int, n_steps: int = 2000):
    return simulate_full_trajectory(run % len(NODE_POS), n_steps, np.random.default_rng(TRAJ_SEED + run))


def build_r0(runs, horizon: int):
    out = {k: [] for k in ("hist", "V", "run")}
    for run in runs:
        pos, vis = episode(run)
        rng = np.random.default_rng(MEAS_SEED + run)
        first = int(np.argmax(vis))
        steps = [first] + [t for t in range(first + 1, len(pos)) if vis[t]]
        z = np.stack([pos[t] + NOISE_STD * rng.standard_normal(2).astype(np.float32) for t in steps])
        t = np.asarray(steps) * DT
        for i in range(len(z) - H - horizon + 1):
            if t[i + H - 1] - t[i] > MAX_GAP:
                continue
            j = np.arange(i + H, i + H + horizon)
            out["hist"].append(z[i:i + H])
            out["V"].append((z[j] - z[j - 1]) / (t[j] - t[j - 1])[:, None])
            out["run"].append(run)
    d = {k: np.asarray(v) for k, v in out.items()}
    n = len(d["run"])
    d.update(mask=np.ones((n, H), np.float32), d=np.zeros(n, np.float32), U=np.zeros((n, 3), np.float32))
    return d


def build_r1(rollouts_path: str, horizon: int):
    r = dict(np.load(rollouts_path))
    V = np.full((len(r["v"]), horizon, 2), np.nan, np.float32)
    for run in np.unique(r["run"]):
        pos, _ = episode(int(run))
        vel = np.vstack([np.full((1, 2), np.nan), np.diff(pos, axis=0) / DT]).astype(np.float32)
        rows = np.flatnonzero(r["run"] == run)
        k = r["k"][rows]
        ok = k + horizon - 1 < len(pos)
        idx = k[ok, None] + np.arange(horizon)
        V[rows[ok]] = vel[idx]
    keep = np.isfinite(V).all((1, 2))
    err = np.abs(V[keep, 0] - r["v"][keep]).max()
    assert err < 1e-4, f"R1 future step 1 differs from the rollout target by {err}"
    return {"hist": r["hist"][keep], "mask": r["mask"][keep], "d": r["d"][keep], "U": r["U"][keep],
            "V": V[keep], "run": r["run"][keep]}, int((~keep).sum())


def load_recipe(out_dir: str, recipe: str) -> dict:
    d = dict(np.load(os.path.join(out_dir, f"data_{recipe}.npz")))
    d["holdout"] = np.isin(d["run"], list(TRAIN_HOLDOUT_RUNS))
    return d


def main():
    p = argparse.ArgumentParser(description="Build the shared R0 / R1 training sets")
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--rollouts", default=os.path.join(VARIANTS_DIR, "rollouts.npz"))
    args = p.parse_args()
    cfg = load_config()
    os.makedirs(args.out_dir, exist_ok=True)
    runs = list(range(*cfg["train_runs"]))
    r0 = build_r0(runs, cfg["horizon"])
    r1, dropped = build_r1(args.rollouts, cfg["horizon"])
    meta = {"commit": git_commit(), "runs": runs, "holdout_runs": sorted(TRAIN_HOLDOUT_RUNS),
            "horizon": cfg["horizon"], "rollouts": os.path.abspath(args.rollouts),
            "rollouts_sha256": sha256(args.rollouts), "r1_rows_dropped_near_episode_end": dropped}
    for name, d in (("R0", r0), ("R1", r1)):
        d = {k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in d.items()}
        path = os.path.join(args.out_dir, f"data_{name}.npz")
        np.savez_compressed(path, **d)
        ho = np.isin(d["run"], list(TRAIN_HOLDOUT_RUNS))
        v = d["V"][:, 0]
        meta[name] = {"rows": int(len(v)), "train_rows": int((~ho).sum()), "holdout_rows": int(ho.sum()),
                      "target_speed_mean": float(np.linalg.norm(v, axis=1).mean()),
                      "target_speed_std": float(np.linalg.norm(v, axis=1).std()), "sha256": sha256(path)}
        print(f"{name}: {meta[name]}")
    write_json(os.path.join(args.out_dir, "data_manifest.json"), meta)


if __name__ == "__main__":
    main()
