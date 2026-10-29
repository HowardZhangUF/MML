#!/usr/bin/env python3
"""
data.py — WOMD v1.3.0 Scenario shards -> per-agent arrays (the tracks_to_predict agents).

    data_train.npz   the downloaded training shards (config data.train_list)
    data_val.npz     the official validation split (our test set)
    tracks.npz       tracking-track tracks: cal_* (holdout training shards), test_* (validation)

Fields per agent (N rows, 91 frames at 10 Hz, current frame 10):
    xy (N, 91, 2) positions minus the current position (m, world orientation)
    valid (N, 91), heading (N, 91) rad, vel0 (N, 2) current velocity (m/s),
    type (1 vehicle, 2 pedestrian, 3 cyclist), difficulty, track_id, scenario, shard

    python -m mml_research.motion.womd.data --out-dir artifacts/motion/womd_v1 --workers 16
"""

from __future__ import annotations

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mml_research.motion.cfm_variants.common import write_json
from mml_research.motion.womd.common import CONFIG_PATH, DEFAULT_OUT_DIR, TYPES, canonical, load_config

FIELDS = ("xy", "valid", "heading", "vel0", "type", "difficulty", "track_id", "scenario", "shard")


def parse_shard(path: str, shard: int) -> dict:
    from mml_research.motion.womd.scenario_io import scenarios
    out = {k: [] for k in FIELDS}
    for s in scenarios(path):
        cur = s.current_time_index
        for req in s.tracks_to_predict:
            tr = s.tracks[req.track_index]
            st = tr.states
            if not st[cur].valid or tr.object_type not in (1, 2, 3):
                continue
            xy = np.array([(a.center_x, a.center_y) for a in st], np.float64)
            out["xy"].append((xy - xy[cur]).astype(np.float32))
            out["valid"].append(np.array([a.valid for a in st], bool))
            out["heading"].append(np.array([a.heading for a in st], np.float32))
            out["vel0"].append(np.array([st[cur].velocity_x, st[cur].velocity_y], np.float32))
            out["type"].append(tr.object_type)
            out["difficulty"].append(req.difficulty)
            out["track_id"].append(tr.id)
            out["scenario"].append(s.scenario_id)
            out["shard"].append(shard)
    return {k: np.array(v) for k, v in out.items()}


def build(paths, workers):
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(parse_shard, paths, range(len(paths))))
    d = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
    d["xy"][~d["valid"]] = 0.0
    return d


def load(out_dir: str, part: str) -> dict:
    return dict(np.load(os.path.join(out_dir, f"data_{part}.npz")))


def impute_history(xy: np.ndarray, valid: np.ndarray, obs: int) -> np.ndarray:
    """Fill invalid history frames: linear interpolation, constant velocity before the first valid frame."""
    h = xy[:, :obs].copy()
    t = np.arange(obs)
    for i in np.flatnonzero(~valid[:, :obs].all(1)):
        ok = np.flatnonzero(valid[i, :obs])          # the current frame is always valid
        if len(ok) == 1:
            h[i] = h[i, ok[0]]
            continue
        for j in range(2):
            h[i, :, j] = np.interp(t, ok, xy[i, ok, j])
        a, b = ok[0], ok[1]
        lead = t < a
        h[i, lead] = xy[i, a] + (t[lead, None] - a) * (xy[i, b] - xy[i, a]) / (b - a)
    return h


def to_model_units(xy: np.ndarray, cfg: dict, hist: np.ndarray | None = None):
    """
    Metres -> canonical-frame model units: hist (N, obs, 2); V (N, H, 2) future per-frame
    velocities (zeros where frames are invalid); M (N, 2, 2) the rotation (p_canonical = p @ M).
    """
    obs, sc = cfg["obs"], cfg["scale"]
    hc, M = canonical((xy[:, :obs] if hist is None else hist) / sc, cfg["canon_min_disp"] / sc)
    V = np.diff(xy[:, obs - 1:], axis=1) / cfg["frame_dt"] / sc
    return hc.astype(np.float32), np.einsum("ntj,njk->ntk", V, M).astype(np.float32), M


def holdout_mask(cfg: dict, train: dict) -> np.ndarray:
    return train["shard"] >= train["shard"].max() + 1 - cfg["data"]["holdout_shards"]


def recipe(cfg: dict, train: dict) -> dict:
    """Training dict in the fair_retrain layout: fully valid agents, holdout = the last shards."""
    keep = train["valid"].all(1)
    hist, V, _ = to_model_units(train["xy"][keep], cfg)
    n = len(hist)
    return {"hist": hist, "V": V, "holdout": holdout_mask(cfg, train)[keep],
            "mask": np.ones((n, cfg["obs"]), np.float32), "d": np.zeros(n, np.float32),
            "U": np.zeros((n, 3), np.float32), "run": np.zeros(n, int)}


def select_tracks(cfg: dict, d: dict, pool: np.ndarray, per_type: int) -> np.ndarray:
    """Seeded per-type draw of fully valid agents among `pool` -> sorted row indices."""
    rng = np.random.default_rng(cfg["tracking"]["selection_seed"])
    full = d["valid"].all(1) & pool
    idx = [rng.permutation(np.flatnonzero(full & (d["type"] == t)))[:per_type] for t in (1, 2, 3)]
    return np.sort(np.concatenate(idx))


def main():
    p = argparse.ArgumentParser(description="Parse WOMD Scenario shards")
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-shards", dest="max_shards", type=int, default=None, help="smoke tests only")
    p.add_argument("--root", default=None, help="Override config data.root (smoke tests only)")
    p.add_argument("--config", default=CONFIG_PATH)
    args = p.parse_args()
    cfg = load_config(args.config)
    root = args.root or cfg["data"]["root"]
    with open(os.path.join(root, cfg["data"]["train_list"])) as f:
        train_paths = [os.path.join(root, "training", os.path.basename(l.strip())) for l in f if l.strip()]
    val_paths = sorted(glob.glob(os.path.join(root, "validation", "validation.tfrecord-*")))
    os.makedirs(args.out_dir, exist_ok=True)
    meta, tracks = {"root": root}, {}
    for part, paths in (("train", train_paths), ("val", val_paths)):
        paths = paths[:args.max_shards]
        d = build(paths, args.workers)
        np.savez_compressed(os.path.join(args.out_dir, f"data_{part}.npz"), **d)
        full = d["valid"].all(1)
        m = {"shards": [os.path.basename(x) for x in paths], "agents": int(len(full)),
             "fully_valid": int(full.sum()), "history_imputed": int((~d["valid"][:, :cfg["obs"]].all(1)).sum()),
             "per_type": {n: int((d["type"] == t).sum()) for t, n in zip((1, 2, 3), TYPES)}}
        if part == "train":
            r = recipe(cfg, d)
            speed = np.linalg.norm(r["V"], axis=-1) * cfg["scale"]
            m.update(rows=int(len(r["hist"])), holdout_rows=int(r["holdout"].sum()),
                     speed_p99_mps=float(np.percentile(speed, 99)), speed_p999_mps=float(np.percentile(speed, 99.9)),
                     clip_mps=1.5 * cfg["scale"], frac_above_clip=float((speed > 1.5 * cfg["scale"]).mean()))
        meta[part] = m
        print(part, {k: v for k, v in m.items() if k != "shards"}, flush=True)
        tc = cfg["tracking"]
        rows = (select_tracks(cfg, d, holdout_mask(cfg, d), tc["calibration_per_type"]) if part == "train"
                else select_tracks(cfg, d, np.ones(len(full), bool), tc["test_per_type"]))
        for k in ("xy", "type", "scenario", "track_id"):
            tracks[f"{'cal' if part == 'train' else 'test'}_{k}"] = d[k][rows]
    np.savez_compressed(os.path.join(args.out_dir, "tracks.npz"), **tracks)
    write_json(os.path.join(args.out_dir, "data_manifest.json"), meta)


if __name__ == "__main__":
    main()
