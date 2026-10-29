#!/usr/bin/env python3
"""
data.py — ETH-UCY (SocialGAN leave-one-scene-out splits, the files MoFlow uses).

Raw layout: <raw>/datasets/<split>/{train,val,test}/*.txt, rows "frame ped x y"
(metres, a frame every 0.4 s; frame ids step by 10). For each pedestrian the
rows are cut into contiguous segments (consecutive frames).

    windows  every 20-frame slice of a segment: 8 observed + 12 future frames
             (the standard protocol; equal to SocialGAN's per-pedestrian samples)
    tracks   whole segments of >= min_track_len frames, for the tracking track

Model units: every model sees positions translated so the last observed point
is the origin and divided by `scale` (config). Translation removes the
absolute scene coordinates, which differ between the leave-one-out scenes;
the scale keeps pedestrian speeds below the 1.5 unit/s clip every mml_core
motion model applies (1.5 units/s = 3 m/s at scale 2).

    python -m mml_research.motion.ethucy.data --raw <dir with datasets/> --out-dir artifacts/motion/ethucy_v1
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from mml_research.motion.cfm_variants.common import sha256, write_json
from mml_research.motion.ethucy.common import DEFAULT_OUT_DIR, load_config


def read_segments(path: str):
    """One file -> list of (T, 2) float32 arrays, contiguous frames of one pedestrian."""
    a = np.loadtxt(path, dtype=np.float64, ndmin=2)
    frames = np.unique(a[:, 0])
    step = np.min(np.diff(frames)) if len(frames) > 1 else 10.0
    segs = []
    for ped in np.unique(a[:, 1]):
        r = a[a[:, 1] == ped]
        r = r[np.argsort(r[:, 0])]
        cut = np.flatnonzero(np.diff(r[:, 0]) != step) + 1
        segs += [s[:, 2:4].astype(np.float32) for s in np.split(r, cut) if len(s) > 1]
    return segs


def windows(segs, obs: int, horizon: int):
    n = obs + horizon
    w = [s[i:i + n] for s in segs for i in range(len(s) - n + 1)]
    return np.stack(w) if w else np.zeros((0, n, 2), np.float32)


def to_model_units(w: np.ndarray, obs: int, dt: float, scale: float):
    """(N, obs+H, 2) metres -> hist (N, obs, 2) translated+scaled, V (N, H, 2) model-unit velocities."""
    origin = w[:, obs - 1:obs]
    hist = (w[:, :obs] - origin) / scale
    V = np.diff(w[:, obs - 1:], axis=1) / dt / scale
    return hist.astype(np.float32), V.astype(np.float32)


def load_split(out_dir: str, split: str) -> dict:
    return dict(np.load(os.path.join(out_dir, f"data_{split}.npz"), allow_pickle=True))


def recipe(cfg: dict, data: dict) -> dict:
    """Training dict in the fair_retrain layout: train windows + val windows (holdout)."""
    obs, H, dt, sc = cfg["obs"], cfg["horizon"], cfg["frame_dt"], cfg["scale"]
    parts = [to_model_units(data[f"win_{p}"], obs, dt, sc) for p in ("train", "val")]
    hist = np.concatenate([p[0] for p in parts])
    V = np.concatenate([p[1] for p in parts])
    n = len(hist)
    holdout = np.zeros(n, bool)
    holdout[len(parts[0][0]):] = True
    return {"hist": hist, "V": V, "holdout": holdout, "mask": np.ones((n, obs), np.float32),
            "d": np.zeros(n, np.float32), "U": np.zeros((n, 3), np.float32), "run": np.zeros(n, int)}


def main():
    p = argparse.ArgumentParser(description="Parse ETH-UCY into windows and tracks")
    p.add_argument("--raw", required=True, help="folder containing datasets/<split>/...")
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    cfg = load_config()
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {"raw": os.path.abspath(args.raw), "files": {}}
    for split in cfg["splits"]:
        out = {}
        for part in ("train", "val", "test"):
            files = sorted(glob.glob(os.path.join(args.raw, "datasets", split, part, "*.txt")))
            assert files, f"no files for {split}/{part}"
            segs = [s for f in files for s in read_segments(f)]
            meta["files"].update({os.path.relpath(f, args.raw): sha256(f) for f in files})
            out[f"win_{part}"] = windows(segs, cfg["obs"], cfg["horizon"])
            tracks = [s for s in segs if len(s) >= cfg["tracking"]["min_track_len"]]
            out[f"tracks_{part}"] = np.array(tracks, dtype=object)
        np.savez_compressed(os.path.join(args.out_dir, f"data_{split}.npz"), **out)
        meta[split] = {k: len(v) for k, v in out.items()}
        speed = np.linalg.norm(np.diff(out["win_train"], axis=1), axis=-1) / cfg["frame_dt"]
        meta[split]["train_speed_mean_mps"] = float(speed.mean())
        meta[split]["train_speed_p99_mps"] = float(np.percentile(speed, 99))
        print(split, meta[split], flush=True)
    write_json(os.path.join(args.out_dir, "data_manifest.json"), meta)


if __name__ == "__main__":
    main()
