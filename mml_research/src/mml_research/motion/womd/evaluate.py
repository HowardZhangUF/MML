#!/usr/bin/env python3
"""
evaluate.py — WOMD tracking track: every method in the particle filter on whole
9.1 s agent tracks with simulated measurements and occlusions.

Episode = one track of tracks.npz (91 frames, all valid). Seeded per (part,
track): measurement noise N(0, meas_noise^2 I) on every visible frame;
visibility = the first `obs` frames, then alternating visible / occluded runs
(config tracking). The filter runs in model units (positions / scale, every
particle's context in its own canonical frame via Canonical); errors are
reported in metres.

    calibrate  every method x q grid on the calibration tracks (holdout training
               shards), seed-0 checkpoints; fair_retrain's M2 rule picks q;
               writes locked.json
    test       the test tracks (validation split), all seeds, locked q

    python -m mml_research.motion.womd.evaluate calibrate --workers 96
    python -m mml_research.motion.womd.evaluate test --workers 96
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from mml_research.motion.cfm_variants.calibrate import select
from mml_research.motion.cfm_variants.common import git_commit, read_json, sha256, write_json
from mml_research.motion.cfm_variants.evaluation import energy_score
from mml_research.motion.cfm_variants.run import environment, write_csv
from mml_research.motion.ethucy.evaluate import _COST, chunks, methods, model_key, visibility
from mml_research.motion.fair_retrain.common import ckpt
from mml_research.motion.fair_retrain.evaluate import load_model
from mml_research.motion.womd.common import CONFIG_PATH, DEFAULT_OUT_DIR, RECIPE, TYPES, Canonical, load_config


def load_tracks(out_dir, part):
    t = np.load(os.path.join(out_dir, "tracks.npz"))
    return t[f"{part}_xy"], t[f"{part}_type"]


def episode(factory, track, part, idx, cfg):
    tc, sc = cfg["tracking"], cfg["scale"]
    seed = zlib.crc32(f"womd/{part}/{idx}".encode())
    rng = np.random.default_rng(seed)
    vis = visibility(len(track), cfg["obs"], rng, tc["visible_run"], tc["occluded_run"])
    z = track + tc["meas_noise"] * rng.standard_normal(track.shape).astype(np.float32)
    np.random.seed(seed % 2 ** 31)
    torch.manual_seed(seed)
    tr = factory()
    tr.reset(z[0] / sc)
    T = len(track) - 1
    err, ms, es = np.empty(T), np.empty(T), []
    for i, t in enumerate(range(1, len(track))):
        t0 = time.perf_counter()
        est = tr.step(z[t] / sc if vis[t] else None)
        ms[i] = 1000 * (time.perf_counter() - t0)
        err[i] = np.linalg.norm(est * sc - track[t])
        if i % tc["energy_every"] == 0:
            pts, w = tr.cloud()
            es.append(energy_score(pts * sc, w, track[t]))
    v = vis[1:]
    return {"part": part, "track": idx, "occluded_frac": float(1 - v.mean()),
            "rmse_all": float(np.sqrt(np.mean(err ** 2))), "rmse_visible": float(np.sqrt(np.mean(err[v] ** 2))),
            "rmse_occluded": float(np.sqrt(np.mean(err[~v] ** 2))) if (~v).any() else np.nan,
            "es_mean": float(np.mean(es)), "ms_per_step": float(ms.mean())}


def build_factory(cfg, out_dir, name, seed, q):
    from mml_research.motion.cfm_variants.tracker import MotionSampler, VariantPFTracker
    tc, sc = cfg["tracking"], cfg["scale"]
    if name == "EnFF":
        from mml_baselines.enff.tracker import EnFFTracker
        e = cfg["enff"]
        return lambda: EnFFTracker(n_particles=tc["particles"], process_noise=q, meas_sigma=tc["filter_sigma"] / sc,
                                   sigma_min=e["sigma_min"], guidance_lambda=e["lambda"],
                                   sampling_time_step_count=e["steps"], dt=cfg["frame_dt"], history=cfg["obs"])
    m = Canonical(load_model(cfg, out_dir, RECIPE, model_key(name), seed), cfg["canon_min_disp"] / sc)
    return lambda: VariantPFTracker(MotionSampler(m, q), n_particles=tc["particles"],
                                    meas_sigma=tc["filter_sigma"] / sc, init_spread=tc["init_spread"] / sc)


def _job(cfg_path, out_dir, part, name, seed, q_m, idxs):
    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    tracks, types = load_tracks(out_dir, part)
    factory = build_factory(cfg, out_dir, name, seed, q_m / cfg["scale"])
    tag = {"method": name, "seed": "" if name in cfg["training_free"] else seed, "process_noise_m": q_m}
    return [{**tag, "type": TYPES[int(types[i]) - 1], **episode(factory, tracks[i], part, i, cfg)} for i in idxs]


def run_jobs(jobs, workers, cfg_path, out_dir):
    """jobs: (part, method, seed, q in metres, track indices)."""
    jobs = sorted(jobs, key=lambda j: -_COST[model_key(j[1])])
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(workers) as pool:
        futs = [pool.submit(_job, cfg_path, out_dir, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if i % max(1, len(jobs) // 50) == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} jobs done ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["method"], str(r["seed"]), r["process_noise_m"], r["track"]))
    return rows


def calibrate(args, cfg):
    tc = cfg["tracking"]
    n = len(load_tracks(args.out_dir, "cal")[0])
    jobs = [("cal", m, 0, float(q), c) for m in methods(cfg) for q in tc["grid_m"] for c in chunks(n)]
    print(f"calibration: {n} tracks, {len(jobs)} jobs", flush=True)
    rows = run_jobs(jobs, args.workers, args.config, args.out_dir)
    os.makedirs(os.path.join(args.out_dir, "tracking"), exist_ok=True)
    write_csv(os.path.join(args.out_dir, "tracking", "cal_tracks.csv"), rows)
    table, chosen = [], {}
    for m in methods(cfg):
        cand = []
        for q in tc["grid_m"]:
            rr = [r for r in rows if r["method"] == m and r["process_noise_m"] == q]
            cand.append({"method": m, "process_noise": q, "tracks": len(rr),
                         **{k: float(np.nanmean([r[k] for r in rr])) for k in
                            ("rmse_all", "rmse_occluded", "es_mean", "ms_per_step")}})
        best = select(cand, tc["rmse_tolerance"])
        chosen[m] = best["process_noise"]
        table += [{**c, "selected": int(c is best)} for c in cand]
        print(f"  {m:11s} q = {best['process_noise']:<5g} m  cal RMSE {best['rmse_all']:.3f} m, "
              f"occluded {best['rmse_occluded']:.3f} m", flush=True)
    write_csv(os.path.join(args.out_dir, "tracking", "calibration.csv"), table)
    files = {f"{RECIPE}_{m}_s{s}": ckpt(args.out_dir, RECIPE, m, s) for m in cfg["models"]
             for s in cfg["training"]["seeds"]}
    write_json(os.path.join(args.out_dir, "locked.json"), {
        "process_noise_m": chosen, "scale": cfg["scale"],
        "rule": "fair_retrain M2 rule on the calibration tracks (holdout training shards), seed-0 checkpoints",
        "checkpoints": {k: sha256(v) for k, v in files.items()}, "config_sha256": sha256(args.config),
        "tracks_sha256": sha256(os.path.join(args.out_dir, "tracks.npz")),
        "commit": git_commit(), "locked_at": time.strftime("%F %T")})
    print("locked.json written")


def test(args, cfg):
    lock = read_json(os.path.join(args.out_dir, "locked.json"))
    changed = [k for k, h in lock["checkpoints"].items()
               if sha256(os.path.join(args.out_dir, "models", k + ".pth")) != h]
    if changed or sha256(args.config) != lock["config_sha256"] \
            or sha256(os.path.join(args.out_dir, "tracks.npz")) != lock["tracks_sha256"]:
        sys.exit(f"changed since locking: checkpoints {changed}, config or tracks")
    n = len(load_tracks(args.out_dir, "test")[0])
    jobs = []
    for m in methods(cfg):
        q = float(lock["process_noise_m"][m])
        for seed in ([0] if m in cfg["training_free"] else cfg["training"]["seeds"]):
            jobs += [("test", m, seed, q, c) for c in chunks(n)]
    print(f"test: {n} tracks, {len(jobs)} jobs", flush=True)
    t0 = time.time()
    rows = run_jobs(jobs, args.workers, args.config, args.out_dir)
    write_csv(os.path.join(args.out_dir, "tracking", "test_tracks.csv"), rows)
    write_json(os.path.join(args.out_dir, "tracking", "manifest.json"), {
        "commit": git_commit(), "command": " ".join(sys.argv), "config": cfg, "lock": lock,
        "environment": environment(), "wall_seconds": time.time() - t0})
    print(f"done in {time.time() - t0:.0f}s")


def main():
    p = argparse.ArgumentParser(description="WOMD tracking track")
    p.add_argument("stage", choices=["calibrate", "test"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    cfg = load_config(args.config)
    (calibrate if args.stage == "calibrate" else test)(args, cfg)


if __name__ == "__main__":
    main()
