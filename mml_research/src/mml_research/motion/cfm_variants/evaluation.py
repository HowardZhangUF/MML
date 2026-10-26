"""
evaluation.py — one episode of one method, with the old exit-benchmark
metrics unchanged and the guide's additional diagnostics under new names.

Unchanged (benchmark_exit_prediction): trajectory/measurement/filter seeds,
event definition, the fork at the last occluded step (so scoring happens
before the first returning measurement), EXIT_HORIZON, labels N/E/S/W/in,
success (top-1), majority (>50%), p_correct, RMSE of the post-step weighted
mean, per-episode RMSE averaged over episodes.

Added (guide chapter 10):
    brier         multiclass Brier over the 5 timed-region labels incl. "in"
    exit_nll      -log max(p_correct, floor); zero_mass flags p_correct == 0
    es_exit       energy score of the fork cloud vs the true position at the
                  scoring time (exit step + EXIT_HORIZON)
    occ_rmse      RMSE over the event's occluded steps; occ_len its length
    recovery      first step after the occlusion ends from which the error
                  stays < threshold for `consecutive` steps; censored if not
                  reached within `horizon` steps
    es_mean       energy score of the weighted cloud every k steps (post-step)
    ess_pre_mean  mean ESS/N before resampling over measurement updates
    off_road_mass weighted particle mass > 0.25 m from every road segment
    ms_p95, ms_visible, ms_occluded   step-time percentiles / split
The energy score, off-road mass and forks are computed outside the timed
region and consume no random numbers.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from mml_research.motion.benchmark_exit_prediction import (
    EXIT_HORIZON, LABELS, SIDES, find_events, lane_rule_side, predict_ahead, score_event,
)
from mml_research.motion.cfm_variants.common import FILTER_SEED, MEAS_SEED, TRAJ_SEED
from mml_research.motion.generate_synthetic_data import NODE_POS, NOISE_STD, TRANS_MATRIX
from mml_research.motion.train_diffpf import simulate_full_trajectory

_EDGES = np.array([(NODE_POS[i], NODE_POS[j]) for i, j in zip(*np.nonzero(TRANS_MATRIX))], np.float32)
OFF_ROAD = 0.25


def energy_score(pts: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
    """sum_i w_i |x_i - y| - 1/2 sum_ij w_i w_j |x_i - x_j|  (lower is better)."""
    pts = pts.astype(np.float64)
    term1 = float(w @ np.linalg.norm(pts - y, axis=1))
    diff = pts[:, None, :] - pts[None, :, :]
    term2 = float(w @ np.sqrt((diff ** 2).sum(-1)) @ w)
    return term1 - 0.5 * term2


def off_road_mass(pts: np.ndarray, w: np.ndarray) -> float:
    a, b = _EDGES[:, 0], _EDGES[:, 1]
    ab = b - a
    t = np.clip(((pts[:, None, :] - a) * ab).sum(-1) / (ab ** 2).sum(-1), 0.0, 1.0)
    d = np.linalg.norm(pts[:, None, :] - (a + t[..., None] * ab), axis=-1).min(1)
    return float(w[d > OFF_ROAD].sum())


def recovery_steps(err: np.ndarray, start: int, thr: float, consecutive: int, horizon: int):
    """Steps from index `start` until err stays < thr for `consecutive` steps; None if censored."""
    below = err < thr
    for k in range(start, min(len(err), start + horizon) - consecutive + 1):
        if below[k:k + consecutive].all():
            return k - start
    return None


def run_episode(factory, run: int, n_steps: int, ev_cfg: dict, keep_clouds: int = 0):
    positions, visible = simulate_full_trajectory(run % len(NODE_POS), n_steps,
                                                  np.random.default_rng(TRAJ_SEED + run))
    first_vis = int(np.argmax(visible))
    events = find_events(positions, visible, first_vis + 1)

    meas_rng = np.random.default_rng(MEAS_SEED + run)
    np.random.seed(FILTER_SEED + run)
    torch.manual_seed(FILTER_SEED + run)
    tracker = factory()
    tracker.reset(positions[first_vis] + NOISE_STD * meas_rng.standard_normal(2).astype(np.float32))

    fork_at = {exit_t - 1: k for k, (_, exit_t, _, _) in enumerate(events)}
    clouds = {}
    T = len(positions) - first_vis - 1
    errors = np.empty(T, np.float64)
    vis = np.empty(T, bool)
    step_ms = np.empty(T, np.float32)
    es, es_vis, offroad = [], [], []
    k_es = ev_cfg["energy_score_every"]
    for i, t in enumerate(range(first_vis + 1, len(positions))):
        meas = positions[t] + NOISE_STD * meas_rng.standard_normal(2).astype(np.float32) if visible[t] else None
        t0 = time.perf_counter()
        est = tracker.step(meas)
        step_ms[i] = 1000.0 * (time.perf_counter() - t0)
        errors[i] = np.linalg.norm(est - positions[t])
        vis[i] = visible[t]
        if i % k_es == 0:
            pts, w = tracker.cloud()
            es.append(energy_score(pts, w, positions[t]))
            es_vis.append(bool(visible[t]))
            offroad.append(off_road_mass(pts, w))
        if t in fork_at:
            clouds[fork_at[t]] = predict_ahead(tracker, 1 + EXIT_HORIZON)

    off = first_vis + 1
    floor = ev_cfg["exit_nll_floor"]
    thr, cons, hor = ev_cfg["recovery_threshold"], ev_cfg["recovery_consecutive"], ev_cfg["recovery_horizon"]
    event_rows, saved = [], []
    for k, (start, exit_t, box, exit_side) in enumerate(events):
        pred, p_correct, mass = score_event(clouds[k], box, exit_side)
        onehot = np.eye(len(LABELS))[exit_side]
        t_score = exit_t + EXIT_HORIZON
        rec = recovery_steps(errors, exit_t - off, thr, cons, hor)
        occ_err = errors[start - off:exit_t - off]
        event_rows.append({
            "run": run, "enter_step": start, "exit_step": exit_t, "box": box,
            "lane_side": SIDES[lane_rule_side(positions, start, box)],
            "actual_side": SIDES[exit_side], "predicted_side": LABELS[pred],
            "success": int(pred == exit_side), "majority": int(p_correct > 0.5),
            "p_correct": p_correct,
            **{f"p_{s}": float(m) for s, m in zip(LABELS, mass)},
            "brier": float(((mass - onehot) ** 2).sum()),
            "exit_nll": float(-np.log(max(p_correct, floor))),
            "zero_mass": int(p_correct == 0.0),
            "es_exit": energy_score(*clouds[k], positions[t_score]) if t_score < len(positions) else np.nan,
            "occ_len": exit_t - start,
            "occ_rmse": float(np.sqrt(np.mean(occ_err ** 2))),
            "recovered": int(rec is not None), "recovery_steps": np.nan if rec is None else rec,
        })
        if len(saved) < keep_clouds:
            saved.append({"run": run, "event": k, "box": box, "actual_side": exit_side,
                          "pts": clouds[k][0].astype(np.float32), "w": clouds[k][1].astype(np.float32)})

    # Recovery after every occlusion of the episode (not only junction events).
    ends = np.flatnonzero(vis[1:] & ~vis[:-1]) + 1
    recs = [recovery_steps(errors, int(e), thr, cons, hor) for e in ends]
    rec_ok = [r for r in recs if r is not None]
    es = np.array(es)
    es_vis = np.array(es_vis, bool)
    ess = np.array(getattr(tracker, "ess_pre", []), float)
    res = np.array(getattr(tracker, "resampled", []), bool)
    run_row = {
        "run": run,
        "rmse_all": float(np.sqrt(np.mean(errors ** 2))),
        "rmse_visible": float(np.sqrt(np.mean(errors[vis] ** 2))),
        "rmse_occluded": float(np.sqrt(np.mean(errors[~vis] ** 2))) if (~vis).any() else np.nan,
        "exit_events": len(events),
        "exit_success": sum(e["success"] for e in event_rows),
        "exit_majority": sum(e["majority"] for e in event_rows),
        "exit_p_correct_sum": float(sum(e["p_correct"] for e in event_rows)),
        "ms_per_step": float(step_ms.mean()), "ms_p95": float(np.percentile(step_ms, 95)),
        "ms_visible": float(step_ms[vis].mean()),
        "ms_occluded": float(step_ms[~vis].mean()) if (~vis).any() else np.nan,
        "es_mean": float(es.mean()), "es_visible": float(es[es_vis].mean()),
        "es_occluded": float(es[~es_vis].mean()) if (~es_vis).any() else np.nan,
        "off_road_mass": float(np.mean(offroad)),
        "ess_pre_mean": float(ess.mean()) if len(ess) else np.nan,
        "resample_rate": float(res.mean()) if len(res) else np.nan,
        "numerical_failures": int(getattr(tracker, "numerical_failures", 0)),
        "n_occlusions": len(ends), "n_recovered": len(rec_ok),
        "recovery_mean_steps": float(np.mean(rec_ok)) if rec_ok else np.nan,
        "lost": int(np.sqrt(np.mean(errors ** 2)) > 0.5),
    }
    steps = {"errors": errors, "vis": vis, "step_ms": step_ms, "ess_pre": ess.astype(np.float32),
             "resampled": res}
    return run_row, event_rows, steps, saved
