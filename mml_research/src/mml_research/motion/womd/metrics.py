"""
metrics.py — WOMD motion-prediction metrics (minADE, minFDE, miss rate) with
the official definitions of the challenge's motion_metrics config:

    predictions and ground truth sampled at 2 Hz (every 5th future frame),
    measurement steps 3 / 5 / 8 s; at step s, over valid ground-truth samples
        minADE = min_k mean_{t <= s} |p_k(t) - g(t)|
        minFDE = min_k |p_k(s) - g(s)|                     (needs g(s) valid)
        miss   = no k with |lat| <= a_lat c and |lon| <= a_lon c at s, where
                 (lon, lat) is p_k(s) - g(s) in the ground truth's heading frame
                 at s and c is the current-speed scale (0.5 below 1.4 m/s,
                 1.0 above 11 m/s, linear between)
    benchmark number = mean over the three steps, then over object types.
"""

from __future__ import annotations

import numpy as np

STEPS_PER_SAMPLE = 5          # 10 Hz frames per 2 Hz sample


def speed_scale(speed: np.ndarray, c: dict) -> np.ndarray:
    f = (speed - c["speed_lower"]) / (c["speed_upper"] - c["speed_lower"])
    return c["lower"] + (c["upper"] - c["lower"]) * np.clip(f, 0.0, 1.0)


def per_agent(pred, gt, valid, heading, vel0, cfg) -> dict:
    """
    pred (B, K, H, 2), gt (B, H, 2), valid / heading (B, H): future frames 1..H after the current
    one, metres, same frame (heading in that frame). -> {metric_<s>s: (B,)} with NaN where undefined.
    """
    fc = cfg["forecast"]
    idx = np.arange(STEPS_PER_SAMPLE - 1, pred.shape[2], STEPS_PER_SAMPLE)     # 0.5 s .. 8 s
    p, g, v, h = pred[:, :, idx], gt[:, idx], valid[:, idx], heading[:, idx]
    err = np.linalg.norm(p - g[:, None], axis=-1)                               # (B, K, S)
    c = speed_scale(np.linalg.norm(vel0, axis=-1), fc["speed_scale"])
    out = {}
    for sec, (a_lat, a_lon) in zip(fc["measurement_seconds"], fc["miss_thresholds"]):
        s = int(round(sec * 2)) - 1                                             # 2 Hz sample index
        w = v[:, :s + 1].astype(float)
        n = w.sum(1)
        ade = (err[:, :, :s + 1] * w[:, None]).sum(-1) / np.maximum(n, 1)[:, None]
        out[f"minADE_{sec}s"] = np.where(n > 0, ade.min(1), np.nan)
        ok = v[:, s]
        out[f"minFDE_{sec}s"] = np.where(ok, err[:, :, s].min(1), np.nan)
        cs, sn = np.cos(h[:, s]), np.sin(h[:, s])
        dx, dy = p[:, :, s, 0] - g[:, None, s, 0], p[:, :, s, 1] - g[:, None, s, 1]
        lon, lat = dx * cs[:, None] + dy * sn[:, None], -dx * sn[:, None] + dy * cs[:, None]
        hit = (np.abs(lat) <= a_lat * c[:, None]) & (np.abs(lon) <= a_lon * c[:, None])
        out[f"MR_{sec}s"] = np.where(ok, (~hit.any(1)).astype(float), np.nan)
    return out


def aggregate(rows: dict, types: np.ndarray, cfg: dict) -> dict:
    """Per-agent arrays -> {metric_<type>, metric (mean over steps, then types)}."""
    out = {}
    for m in ("minADE", "minFDE", "MR"):
        by_type = []
        for t, name in ((1, "vehicle"), (2, "pedestrian"), (3, "cyclist")):
            sel = types == t
            v = float(np.nanmean([np.nanmean(rows[f"{m}_{s}s"][sel]) for s in cfg["forecast"]["measurement_seconds"]]))
            out[f"{m}_{name}"] = v
            by_type.append(v)
        out[m] = float(np.mean(by_type))
    return out
