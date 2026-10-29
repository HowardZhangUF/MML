#!/usr/bin/env python3
"""
evaluate.py — ETH-UCY tracking track: every method in the particle filter on
real pedestrian tracks with simulated measurements and occlusions.

Episode = one track (>= min_track_len contiguous frames). Seeded per
(split, part, track): measurement noise N(0, meas_noise^2 I) on every visible
frame; visibility = the first `obs` frames, then alternating visible /
occluded runs (config tracking). The filter runs in model units (translated
contexts via Translated, positions / scale); errors are reported in metres.

    calibrate  each split x method x q grid on the first n val tracks, seed-0
               checkpoints; fair_retrain's M2 rule picks q; writes locked.json
    test       all test tracks, all seeds, locked q

    python -m mml_research.motion.ethucy.evaluate calibrate --workers 96
    python -m mml_research.motion.ethucy.evaluate test --workers 96
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
from mml_research.motion.ethucy.common import CONFIG_PATH, DEFAULT_OUT_DIR, Translated, load_config
from mml_research.motion.ethucy.data import load_split
from mml_research.motion.fair_retrain.common import MODEL_LABEL, ckpt
from mml_research.motion.fair_retrain.evaluate import load_model

LABEL_MODEL = {v: k for k, v in MODEL_LABEL.items()}
_COST = {"tcfm": 36.0, "trajflow": 8.0, "moflow": 3.5, "m4": 1.6, "cfm": 1.0, "nf": 1.0, "dmmn_gauss": 0.8,
         "dmmn": 0.6, "EnFF": 0.4, "PF+CV": 0.1}
CHUNK = 25          # tracks per job


def methods(cfg):
    return [MODEL_LABEL[m] for m in cfg["models"]] + list(cfg["training_free"])


def model_key(name):
    return LABEL_MODEL.get(name, name)


def visibility(T, obs, rng, vis_run, occ_run):
    v = np.ones(T, bool)
    t, visible = obs, False
    while t < T:
        n = rng.integers(occ_run[0], occ_run[1] + 1) if not visible else rng.integers(vis_run[0], vis_run[1] + 1)
        v[t:t + n] = visible
        t += n
        visible = not visible
    return v


def episode(factory, track, split, part, idx, cfg):
    tc, sc = cfg["tracking"], cfg["scale"]
    seed = zlib.crc32(f"{split}/{part}/{idx}".encode())
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
    return {"split": split, "part": part, "track": idx, "frames": len(track),
            "occluded_frac": float(1 - v.mean()),
            "rmse_all": float(np.sqrt(np.mean(err ** 2))), "rmse_visible": float(np.sqrt(np.mean(err[v] ** 2))),
            "rmse_occluded": float(np.sqrt(np.mean(err[~v] ** 2))) if (~v).any() else np.nan,
            "es_mean": float(np.mean(es)), "ms_per_step": float(ms.mean())}


def build_factory(cfg, out_dir, split, name, seed, q):
    from mml_research.motion.cfm_variants.tracker import MotionSampler, VariantPFTracker
    tc, sc = cfg["tracking"], cfg["scale"]
    if name == "EnFF":
        from mml_baselines.enff.tracker import EnFFTracker
        e = cfg["enff"]
        return lambda: EnFFTracker(n_particles=tc["particles"], process_noise=q, meas_sigma=tc["filter_sigma"] / sc,
                                   sigma_min=e["sigma_min"], guidance_lambda=e["lambda"],
                                   sampling_time_step_count=e["steps"], dt=cfg["frame_dt"], history=cfg["obs"])
    m = Translated(load_model(cfg, out_dir, split, model_key(name), seed))

    def make():
        t = VariantPFTracker(MotionSampler(m, q), n_particles=tc["particles"], meas_sigma=tc["filter_sigma"] / sc,
                             init_spread=tc["init_spread"] / sc)
        return t
    return make


def _job(cfg_path, out_dir, split, part, name, seed, q, idxs):
    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    tracks = load_split(out_dir, split)[f"tracks_{part}"]
    factory = build_factory(cfg, out_dir, split, name, seed, q)
    tag = {"method": name, "seed": "" if name in cfg["training_free"] else seed, "process_noise": q}
    return [{**tag, **episode(factory, tracks[i], split, part, i, cfg)} for i in idxs]


def run_jobs(jobs, workers, cfg_path, out_dir):
    """jobs: (split, part, method, seed, q, track indices)."""
    jobs = sorted(jobs, key=lambda j: -_COST[model_key(j[2])])
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(workers) as pool:
        futs = [pool.submit(_job, cfg_path, out_dir, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if i % max(1, len(jobs) // 50) == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} jobs done ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["split"], r["method"], str(r["seed"]), r["process_noise"], r["track"]))
    return rows


def chunks(n):
    return [list(range(i, min(n, i + CHUNK))) for i in range(0, n, CHUNK)]


def calibrate(args, cfg):
    tc = cfg["tracking"]
    jobs = []
    for split in cfg["splits"]:
        n = min(len(load_split(args.out_dir, split)["tracks_val"]), tc["calibration_tracks"])
        jobs += [(split, "val", m, 0, float(q), c) for m in methods(cfg) for q in tc["grid"] for c in chunks(n)]
    print(f"calibration: {len(jobs)} jobs", flush=True)
    rows = run_jobs(jobs, args.workers, args.config, args.out_dir)
    os.makedirs(os.path.join(args.out_dir, "tracking"), exist_ok=True)
    write_csv(os.path.join(args.out_dir, "tracking", "val_tracks.csv"), rows)
    table, chosen = [], {}
    for split in cfg["splits"]:
        for m in methods(cfg):
            cand = []
            for q in tc["grid"]:
                rr = [r for r in rows if r["split"] == split and r["method"] == m and r["process_noise"] == q]
                cand.append({"split": split, "method": m, "process_noise": q, "tracks": len(rr),
                             **{k: float(np.nanmean([r[k] for r in rr])) for k in
                                ("rmse_all", "rmse_occluded", "es_mean", "ms_per_step")}})
            best = select(cand, tc["rmse_tolerance"])
            chosen[f"{split}/{m}"] = best["process_noise"]
            table += [{**c, "selected": int(c is best)} for c in cand]
            print(f"  {split:6s} {m:11s} q = {best['process_noise']:<5g} (model units)  val RMSE "
                  f"{best['rmse_all']:.3f} m, occluded {best['rmse_occluded']:.3f} m", flush=True)
    write_csv(os.path.join(args.out_dir, "tracking", "calibration.csv"), table)
    files = {f"{s}_{m}_s{seed}": ckpt(args.out_dir, s, m, seed) for s in cfg["splits"] for m in cfg["models"]
             for seed in cfg["training"]["seeds"]}
    write_json(os.path.join(args.out_dir, "locked.json"), {
        "process_noise_model_units": chosen, "scale": cfg["scale"],
        "rule": "fair_retrain M2 rule on the first calibration_tracks val tracks, seed-0 checkpoints",
        "checkpoints": {k: sha256(v) for k, v in files.items()}, "config_sha256": sha256(args.config),
        "commit": git_commit(), "locked_at": time.strftime("%F %T")})
    print("locked.json written")


def test(args, cfg):
    lock = read_json(os.path.join(args.out_dir, "locked.json"))
    changed = [k for k, h in lock["checkpoints"].items()
               if sha256(os.path.join(args.out_dir, "models", k + ".pth")) != h]
    if changed or sha256(args.config) != lock["config_sha256"]:
        sys.exit(f"changed since locking: {changed}, config {sha256(args.config) != lock['config_sha256']}")
    jobs = []
    for split in cfg["splits"]:
        n = len(load_split(args.out_dir, split)["tracks_test"])
        for m in methods(cfg):
            q = float(lock["process_noise_model_units"][f"{split}/{m}"])
            for seed in ([0] if m in cfg["training_free"] else cfg["training"]["seeds"]):
                jobs += [(split, "test", m, seed, q, c) for c in chunks(n)]
    print(f"test: {len(jobs)} jobs", flush=True)
    t0 = time.time()
    rows = run_jobs(jobs, args.workers, args.config, args.out_dir)
    write_csv(os.path.join(args.out_dir, "tracking", "test_tracks.csv"), rows)
    write_json(os.path.join(args.out_dir, "tracking", "manifest.json"), {
        "commit": git_commit(), "command": " ".join(sys.argv), "config": cfg, "lock": lock,
        "environment": environment(), "wall_seconds": time.time() - t0})
    print(f"done in {time.time() - t0:.0f}s")


def main():
    p = argparse.ArgumentParser(description="ETH-UCY tracking track")
    p.add_argument("stage", choices=["calibrate", "test"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    cfg = load_config(args.config)
    (calibrate if args.stage == "calibrate" else test)(args, cfg)


if __name__ == "__main__":
    main()
