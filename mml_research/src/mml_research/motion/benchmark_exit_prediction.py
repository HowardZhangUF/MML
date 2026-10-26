#!/usr/bin/env python3
"""
benchmark_exit_prediction.py
============================
Filter-level benchmark of every tracker on identical trajectories, scoring
whether each tracker's prediction leaves an occlusion box on the same side as
the target.

Trackers (same filter settings as benchmark_diffpf_vs_cfm.py and
benchmark_cv_dmmn.py):
    PF+CFM       bootstrap PF + CFM proposal, fixed anchor sigma
    PF+CFM-AS    same checkpoint, anchor sigma grows with occlusion length
    PF+CFM-LM    PF+CFM with the learned CFM measurement model as likelihood
    PF+CFM-P     PF+CFM with the learned measurement model and the
                 observation-conditioned CFM proposal, importance-weighted
                 p(z|x) f(v|C) / q(v|C,z) (mml_research.motion.proposal_pf)
    PF+CFM-Pnaive  same proposal with likelihood-only weights (uncorrected;
                 diagnostic for the importance correction, not a method)
    PF+NF        bootstrap PF + normalizing-flow proposal
    DiffPF       adapted DiffPF generative posterior sampler
    PF+CV        bootstrap PF + constant-velocity proposal
    PF+DMMN      bootstrap PF + ScratchTransformer velocity proposal
Open-source flow-matching baselines (ports in <repo>/baselines/mml_baselines,
put that folder on PYTHONPATH):
    PF+TrajFlow  bootstrap PF + TrajFlow forecaster (mode drawn by its score)
    PF+MoFlow    bootstrap PF + MoFlow one-step IMLE forecaster
    PF+TCFM      bootstrap PF + TCFM conditional-flow-matching forecaster
    EnFF         Ensemble Flow Filter (DAFM): CV forecast + flow-matching analysis

Exit-prediction metric
----------------------
Event: the target enters an occlusion box that contains a branching node
(a node with more than one successor), i.e. a box with more than one
possible exit, and later re-appears.

Prediction: the tracker runs predict-only while the target is hidden. At the
last occluded step it is forked and keeps predicting without measurements for
EXIT_HORIZON more steps, i.e. until the target is SPEED * EXIT_HORIZON past
the box edge. Each particle of that cloud is then labelled with the box side
it lies beyond (N / E / S / W), or "in" if it is still inside the box, and
particles keep their weights. Mass still inside never counts as an exit, so a
cloud that stalls or merely sits in the approach lane cannot score. DiffPF
draws a fresh particle set every step, so particles cannot be followed to a
first exit; scoring where the cloud is works for every tracker. The fork
restores the tracker and the global RNG state, so the main run (and its
RMSE) is identical to a run without forks.

    success      the label holding the most particle mass is the side the
                 target actually exits through
    majority     more than half the particle mass is beyond the actual exit
                 side (a near-uniform cloud can win the argmax by a few
                 percent; it cannot pass this)
    P(correct)   particle mass beyond the target's actual exit side

Process noise
-------------
Each bootstrap-PF tracker takes its own per-step position jitter
(--process_noise NAME=STD, default 0.01 m). Pick the values with
sweep_process_noise.py on validation runs (--first_run 100), never on the
benchmark runs. DiffPF has its own built-in propagation noise.

The lane rule baseline (always predict the turn signalled by the approach
lane) gives the best accuracy any tracker can reach on these trajectories.

Usage (after installing the project):
    python -m mml_research.motion.benchmark_exit_prediction \
        --models-dir <dir> --n_runs 100 --n_steps 2000 --workers 12
"""

from __future__ import annotations

from mml_core.config import motion_resource_dir

import os

# Filters run on CPU, one worker process per group of runs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import csv
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from mml_core.motion.cfm_motion_model import CFMMotionModel
from mml_core.motion.cfm_proposal_model import CFMProposalModel
from mml_core.motion.nf_motion_model import NFMotionModel
from mml_core.sensing.cfm_measurement_model import CFMMeasurementModel
from mml_core.tracking.diffpf_tracker import DiffPFTracker
from mml_baselines.enff.tracker import EnFFTracker
from mml_baselines.moflow.motion_model import MoFlowMotionModel
from mml_baselines.tcfm.motion_model import TCFMMotionModel
from mml_baselines.trajflow.motion_model import TrajFlowMotionModel
from mml_research.motion.benchmark_diffpf_vs_cfm import BootstrapPFTracker
from mml_research.motion.proposal_pf import ProposalPFTracker
from mml_research.motion.benchmark_cv_dmmn import (
    ConstantVelocityMotionModel,
    OfflineDMMNMotionModel,
)
from mml_research.motion.generate_synthetic_data import (
    DT, NOISE_STD, NODE_POS, TRANS_MATRIX, OCC_CENTERS, OCC_WIDTHS,
)
from mml_research.motion.train_diffpf import simulate_full_trajectory


TRACKERS = ["PF+CFM", "PF+CFM-AS", "PF+CFM-LM", "PF+CFM-P", "PF+CFM-Pnaive", "PF+NF", "DiffPF",
            "PF+CV", "PF+DMMN",
            "PF+TrajFlow", "PF+MoFlow", "PF+TCFM", "EnFF"]
# Trackers with a per-step process-noise parameter (every one but DiffPF).
PF_TRACKERS = [t for t in TRACKERS if t != "DiffPF"]
DEFAULT_PROCESS_NOISE = 0.01
SIDES = ["N", "E", "S", "W"]
LABELS = SIDES + ["in"]   # "in": particle still inside the box
EXIT_HORIZON = 10         # predict-only steps past the target's re-appearance


# ── Geometry ─────────────────────────────────────────────────────────────────

def box_of(p) -> int | None:
    for i, (c, w) in enumerate(zip(OCC_CENTERS, OCC_WIDTHS)):
        if abs(p[0] - c[0]) < w / 2 and abs(p[1] - c[1]) < w / 2:
            return i
    return None


def sides_of(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Index into SIDES for each point: the box side it lies beyond / towards."""
    d = np.atleast_2d(points) - center
    horizontal = np.abs(d[:, 0]) >= np.abs(d[:, 1])
    return np.where(horizontal,
                    np.where(d[:, 0] > 0, 1, 3),     # E / W
                    np.where(d[:, 1] > 0, 0, 2))     # N / S


def exit_labels(points: np.ndarray, box: int) -> np.ndarray:
    """Index into LABELS: the side a point lies beyond, or 4 if inside the box."""
    c, w = OCC_CENTERS[box], OCC_WIDTHS[box]
    inside = np.all(np.abs(np.atleast_2d(points) - c) < w / 2, axis=1)
    return np.where(inside, 4, sides_of(points, c))


def decision_boxes() -> set:
    """Boxes containing a node with more than one possible successor."""
    branching = [i for i in range(len(NODE_POS)) if np.count_nonzero(TRANS_MATRIX[i]) > 1]
    return {box_of(NODE_POS[i]) for i in branching} - {None}


def find_events(positions: np.ndarray, visible: np.ndarray, first_t: int) -> list:
    """Occlusions of decision boxes: (enter step, exit step, box, actual exit side)."""
    boxes = decision_boxes()
    events = []
    t = max(first_t, 1)
    while t < len(visible):
        if visible[t - 1] and not visible[t]:
            start, box = t, box_of(positions[t])
            while t < len(visible) and not visible[t]:
                t += 1
            if t < len(visible) and box in boxes:
                exit_side = int(sides_of(positions[t], OCC_CENTERS[box])[0])
                events.append((start, t, box, exit_side))
        t += 1
    return events


def lane_rule_side(positions: np.ndarray, start: int, box: int) -> int:
    """The exit signalled by the approach lane: north lane → N, south lane → S."""
    return 0 if positions[start - 1][1] > OCC_CENTERS[box][1] else 2


# ── Tracker execution ────────────────────────────────────────────────────────

def particle_cloud(tracker):
    """(positions, weights) of the tracker's current particle set."""
    if hasattr(tracker, "cloud"):
        return tracker.cloud()
    if isinstance(tracker, BootstrapPFTracker):
        return tracker.particles[-1].copy(), tracker.weights.copy()
    pts = tracker.particles.copy()
    return pts, np.full(len(pts), 1.0 / len(pts))


def predict_ahead(tracker, n_steps):
    """Particle cloud after n_steps of predict-only; tracker and RNGs are left untouched."""
    if hasattr(tracker, "get_state"):
        saved = tracker.get_state()
    elif isinstance(tracker, BootstrapPFTracker):
        saved = (tracker.particles.copy(), tracker.weights.copy(), tracker.steps_since_meas,
                 tracker.motion_model.anchor_sigma if tracker.base_sigma is not None else None)
    else:
        saved = (tracker.particles.copy(), tracker._prev_mean.copy())
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()

    for _ in range(n_steps):
        tracker.step(None)
    cloud = particle_cloud(tracker)

    if hasattr(tracker, "set_state"):
        tracker.set_state(saved)
    elif isinstance(tracker, BootstrapPFTracker):
        tracker.particles, tracker.weights, tracker.steps_since_meas, sigma = saved
        if sigma is not None:
            tracker.motion_model.anchor_sigma = sigma
    else:
        tracker.particles, tracker._prev_mean = saved
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    return cloud


def run_tracker_with_exits(tracker, positions, visible, rng, events):
    """
    Same loop as benchmark_diffpf_vs_cfm.run_tracker, plus, for every event,
    the predicted particle cloud EXIT_HORIZON steps after the target re-appears
    (forked at the last occluded step, so it never sees the exit measurement).
    """
    first_vis = int(np.argmax(visible))
    tracker.reset(positions[first_vis] + NOISE_STD * rng.standard_normal(2).astype(np.float32))

    fork_at = {exit_t - 1: k for k, (_, exit_t, _, _) in enumerate(events)}
    clouds = {}
    errors, vis_mask = [], []
    elapsed = 0.0
    for t in range(first_vis + 1, len(positions)):
        if visible[t]:
            meas = positions[t] + NOISE_STD * rng.standard_normal(2).astype(np.float32)
        else:
            meas = None
        t0 = time.perf_counter()
        est = tracker.step(meas)
        elapsed += time.perf_counter() - t0
        errors.append(float(np.linalg.norm(est - positions[t])))
        vis_mask.append(bool(visible[t]))
        if t in fork_at:
            clouds[fork_at[t]] = predict_ahead(tracker, 1 + EXIT_HORIZON)
    return np.array(errors), np.array(vis_mask), 1000.0 * elapsed / len(errors), clouds


def score_event(cloud, box, exit_side):
    pts, w = cloud
    mass = np.bincount(exit_labels(pts, box), weights=w, minlength=len(LABELS))
    mass = mass / mass.sum()
    pred = int(np.argmax(mass))
    return pred, float(mass[exit_side]), mass


def build_trackers(args):
    """Tracker factories for args.trackers; only the requested checkpoints are loaded."""
    want = set(args.trackers)
    path = lambda name: os.path.join(args.models_dir, name)
    n = args.pf_particles
    q = args.process_noise
    pf = lambda model, name, **kw: (lambda: BootstrapPFTracker(model, n_particles=n,
                                                               process_noise=q[name], **kw))
    factories = {}
    cfm_variants = {"PF+CFM", "PF+CFM-AS", "PF+CFM-LM", "PF+CFM-P", "PF+CFM-Pnaive"}
    if want & cfm_variants:
        cfm = CFMMotionModel()
        cfm.load(path(args.cfm_ckpt))
        cfm_sigma0 = cfm.anchor_sigma
        factories["PF+CFM"] = pf(cfm, "PF+CFM", base_sigma=cfm_sigma0)
        factories["PF+CFM-AS"] = pf(cfm, "PF+CFM-AS", occlusion_sigma_gain=args.adaptive_gain,
                                    sigma_max=args.sigma_max, base_sigma=cfm_sigma0)
    if want & (cfm_variants - {"PF+CFM", "PF+CFM-AS"}):
        meas = CFMMeasurementModel()
        meas.load(path(args.meas_ckpt))
        prop = None
        if want & {"PF+CFM-P", "PF+CFM-Pnaive"}:
            prop = CFMProposalModel()
            prop.load(path(args.proposal_ckpt))
        for name, proposal, weighting in (("PF+CFM-LM", None, "correct"),
                                          ("PF+CFM-P", prop, "correct"),
                                          ("PF+CFM-Pnaive", prop, "naive")):
            factories[name] = (lambda proposal=proposal, weighting=weighting, name=name:
                               ProposalPFTracker(cfm, meas, proposal, weighting=weighting,
                                                 n_particles=n, process_noise=q[name],
                                                 base_sigma=cfm_sigma0))
    if "PF+NF" in want:
        nf = NFMotionModel()
        nf.load(path("nf_motion_model.pth"))
        factories["PF+NF"] = pf(nf, "PF+NF")
    if "DiffPF" in want:
        diffpf = DiffPFTracker()
        diffpf.load(path("diffpf_tracker.pth"))
        factories["DiffPF"] = lambda: diffpf
    if "PF+CV" in want:
        factories["PF+CV"] = pf(ConstantVelocityMotionModel(block_size=5, dt=DT), "PF+CV")
    if "PF+DMMN" in want:
        factories["PF+DMMN"] = pf(OfflineDMMNMotionModel(path(args.dmmn_ckpt), dt=DT), "PF+DMMN")
    for name, cls, ckpt in (("PF+TrajFlow", TrajFlowMotionModel, args.trajflow_ckpt),
                            ("PF+MoFlow", MoFlowMotionModel, args.moflow_ckpt),
                            ("PF+TCFM", TCFMMotionModel, args.tcfm_ckpt)):
        if name in want:
            factories[name] = pf(cls(path(ckpt)), name)
    if "EnFF" in want:
        factories["EnFF"] = lambda: EnFFTracker(
            n_particles=n, process_noise=q["EnFF"], sigma_min=args.enff_sigma_min,
            guidance_lambda=args.enff_lambda, sampling_time_step_count=args.enff_steps, dt=DT)
    return factories


def run_group(args, runs):
    """Worker: every tracker on every run in `runs`. Seeds depend only on the run index."""
    torch.set_num_threads(1)
    factories = build_trackers(args)
    run_rows, event_rows = [], []

    for run in runs:
        rng = np.random.default_rng(1000 + run)
        positions, visible = simulate_full_trajectory(run % len(NODE_POS), args.n_steps, rng)
        first_vis = int(np.argmax(visible))
        events = find_events(positions, visible, first_vis + 1)

        for name in args.trackers:
            meas_rng = np.random.default_rng(5000 + run)
            np.random.seed(9000 + run)
            torch.manual_seed(9000 + run)
            errors, vis, ms, clouds = run_tracker_with_exits(
                factories[name](), positions, visible, meas_rng, events)

            noise = args.process_noise[name] if name in PF_TRACKERS else ""
            n_ok, n_maj, p_sum = 0, 0, 0.0
            for k, (start, exit_t, box, exit_side) in enumerate(events):
                pred, p_correct, mass = score_event(clouds[k], box, exit_side)
                n_ok += int(pred == exit_side)
                n_maj += int(p_correct > 0.5)
                p_sum += p_correct
                event_rows.append({
                    "tracker": name, "run": run, "process_noise": noise,
                    "enter_step": start, "exit_step": exit_t,
                    "box": box, "lane_side": SIDES[lane_rule_side(positions, start, box)],
                    "actual_side": SIDES[exit_side], "predicted_side": LABELS[pred],
                    "success": int(pred == exit_side), "majority": int(p_correct > 0.5),
                    "p_correct": round(p_correct, 6),
                    **{f"p_{s}": round(float(m), 6) for s, m in zip(LABELS, mass)},
                })

            run_rows.append({
                "tracker": name, "run": run,
                "process_noise": noise,
                "rmse_all": float(np.sqrt(np.mean(errors ** 2))),
                "rmse_visible": float(np.sqrt(np.mean(errors[vis] ** 2))),
                "rmse_occluded": float(np.sqrt(np.mean(errors[~vis] ** 2))) if (~vis).any() else float("nan"),
                "exit_events": len(events), "exit_success": n_ok, "exit_majority": n_maj,
                "exit_p_correct_sum": p_sum, "ms_per_step": ms,
            })
        print(f"  run {run} done ({len(events)} exit events)", flush=True)
    return run_rows, event_rows


# ── Reporting ────────────────────────────────────────────────────────────────

def summarize(run_rows, event_rows, trackers=TRACKERS):
    summary = {}
    for name in trackers:
        rows = [r for r in run_rows if r["tracker"] == name]
        ev = [e for e in event_rows if e["tracker"] == name]
        s = {"tracker": name, "process_noise": rows[0]["process_noise"] if rows else ""}
        for key in ["rmse_all", "rmse_visible", "rmse_occluded", "ms_per_step"]:
            vals = np.array([r[key] for r in rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            s[f"{key}_mean"], s[f"{key}_std"] = float(vals.mean()), float(vals.std())
        n = len(ev)
        k = sum(e["success"] for e in ev)
        s["exit_events"], s["exit_success"] = n, k
        s["exit_success_rate"] = k / n if n else float("nan")
        s["exit_majority"] = sum(e["majority"] for e in ev)
        s["exit_majority_rate"] = s["exit_majority"] / n if n else float("nan")
        s["exit_p_correct"] = float(np.mean([e["p_correct"] for e in ev])) if n else float("nan")
        summary[name] = s

    # Lane rule: the same events, scored by the approach-lane cue alone.
    ev = [e for e in event_rows if e["tracker"] == trackers[0]]
    lane_ok = sum(e["lane_side"] == e["actual_side"] for e in ev)
    return summary, {"events": len(ev), "success": lane_ok}


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_latex(path, summary, lane, args):
    best_rate = max(s["exit_success_rate"] for s in summary.values())
    best_maj = max(s["exit_majority_rate"] for s in summary.values())
    best_p = max(s["exit_p_correct"] for s in summary.values())
    labels = {name: name for name in TRACKERS}
    labels.update({
        "PF+CFM": "PF+CFM (fixed $\\sigma=1.0$)",
        "PF+CFM-AS": "PF+CFM-AS (adaptive $\\sigma$)",
        "PF+CFM-LM": "PF+CFM-LM (learned likelihood)",
        "PF+CFM-P": "PF+CFM-P (learned proposal + likelihood)",
        "PF+CFM-Pnaive": "PF+CFM-P, uncorrected weights",
    })

    def bold(text, on):
        return f"\\mathbf{{{text}}}" if on else text

    lines = [
        "% Generated by mml_research.motion.benchmark_exit_prediction.",
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{Benchmark results over {args.n_runs} seeded runs of {args.n_steps} steps; "
        f"all {len(args.trackers)} trackers share one protocol (filter level, identical trajectories and noise). "
        "Per-step process noise of each particle filter, chosen by lowest RMSE on separate "
        "validation runs: " + ", ".join(
            f"{name} {args.process_noise[name]:g}\\,m" for name in args.trackers if name in PF_TRACKERS
        ) + ". "
        "RMSE values are mean $\\pm$ run-to-run standard deviation (lower is better). "
        "Exit success: fraction of junction occlusions where, predicting without measurements "
        f"until {EXIT_HORIZON} steps after the target re-appears, most particle mass has left the "
        "box through the side the target exits through (top-1: the largest share; $>$50\\%: a majority); "
        "$P$(correct): mean particle mass beyond that side "
        f"(higher is better). The approach-lane cue predicts the exit in "
        f"{lane['success']}/{lane['events']} ({100 * lane['success'] / lane['events']:.1f}\\%) events, "
        "the best achievable success rate; chance is 50\\%.}",
        "\\label{tab:filter_benchmark}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccc}",
        "\\hline",
        "\\textbf{Tracker} & \\textbf{RMSE all (m)} & \\textbf{RMSE visible (m)} "
        "& \\textbf{RMSE occluded (m)} & \\textbf{Exit success (top-1)} "
        "& \\textbf{Exit success ($>$50\\%)} & \\textbf{$P$(correct exit)} "
        "& \\textbf{Runtime (ms/step)} \\\\",
        "\\hline",
    ]
    for name in args.trackers:
        s = summary[name]
        maj = bold(f"{100 * s['exit_majority_rate']:.1f}\\%", s["exit_majority_rate"] == best_maj)
        rate = bold(f"{100 * s['exit_success_rate']:.1f}\\%", s["exit_success_rate"] == best_rate)
        p_correct = bold(f"{s['exit_p_correct']:.3f}", s["exit_p_correct"] == best_p)
        lines.append(
            f"{labels[name]}\n"
            f"& ${s['rmse_all_mean']:.3f} \\pm {s['rmse_all_std']:.3f}$ "
            f"& ${s['rmse_visible_mean']:.3f} \\pm {s['rmse_visible_std']:.3f}$ "
            f"& ${s['rmse_occluded_mean']:.3f} \\pm {s['rmse_occluded_std']:.3f}$ "
            f"& ${rate}$ ({s['exit_success']}/{s['exit_events']}) "
            f"& ${maj}$ ({s['exit_majority']}/{s['exit_events']}) "
            f"& ${p_correct}$ "
            f"& ${s['ms_per_step_mean']:.1f}$ \\\\"
        )
    lines += ["\\hline", "\\end{tabular}%", "}", "\\end{table*}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def parse_noise(items) -> dict:
    """['PF+DMMN=0.045', 'all=0.02'] -> {tracker: std} for every PF tracker."""
    noise = {name: DEFAULT_PROCESS_NOISE for name in PF_TRACKERS}
    for item in items or []:
        name, value = item.split("=")
        for target in (PF_TRACKERS if name == "all" else [name]):
            if target not in noise:
                raise ValueError(f"Unknown PF tracker {target!r}; choose from {PF_TRACKERS}")
            noise[target] = float(value)
    return noise


def make_parser():
    parser = argparse.ArgumentParser(description="Exit-direction prediction benchmark, filter level")
    parser.add_argument("--n_runs", type=int, default=100)
    parser.add_argument("--first_run", type=int, default=0,
                        help="Run indices first_run .. first_run+n_runs-1 (seeds derive from the "
                             "index); use 100+ for validation runs")
    parser.add_argument("--trackers", nargs="+", default=TRACKERS, choices=TRACKERS)
    parser.add_argument("--process_noise", nargs="*", default=[], metavar="NAME=STD",
                        help="Per-step position jitter (m) per PF tracker, e.g. PF+DMMN=0.045; "
                             f"NAME 'all' sets every PF tracker. Default {DEFAULT_PROCESS_NOISE}")
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--pf_particles", type=int, default=500)
    parser.add_argument("--adaptive_gain", type=float, default=0.05)
    parser.add_argument("--sigma_max", type=float, default=2.5)
    parser.add_argument("--cfm_ckpt", default="cfm_motion_model_anchor1.0.pth")
    parser.add_argument("--meas_ckpt", default="cfm_measurement_model.pth")
    parser.add_argument("--proposal_ckpt", default="cfm_proposal_model.pth")
    parser.add_argument("--dmmn_ckpt", default="dmmn_scratch_transformer.pth")
    parser.add_argument("--trajflow_ckpt", default="trajflow_motion_model.pth")
    parser.add_argument("--moflow_ckpt", default="moflow_motion_model.pth")
    parser.add_argument("--tcfm_ckpt", default="tcfm_motion_model.pth")
    parser.add_argument("--enff_lambda", type=float, default=2.5e-4,
                        help="EnFF Local-guidance scale (choose with sweep_enff.py on validation runs)")
    parser.add_argument("--enff_sigma_min", type=float, default=1e-2,
                        help="EnFF F2P probability-path std (default: selected by sweep_enff.py)")
    parser.add_argument("--enff_steps", type=int, default=5,
                        help="EnFF sampling_time_step_count (Euler grid points on [0, 1])")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--models-dir", dest="models_dir",
                        default=str(motion_resource_dir() / "exit_benchmark" / "models"))
    parser.add_argument("--out-dir", dest="out_dir",
                        default=str(motion_resource_dir() / "exit_benchmark" / "results"))
    return parser


def worker_pool(n_workers):
    """Process pool for run_group. Forked workers cannot use CUDA, so spawn them when a GPU is visible
    (e.g. inside a SLURM GPU job, which sets CUDA_VISIBLE_DEVICES)."""
    ctx = multiprocessing.get_context("spawn" if os.environ.get("CUDA_VISIBLE_DEVICES") else None)
    return ProcessPoolExecutor(n_workers, mp_context=ctx)


def run_parallel(args):
    """Run args.trackers on runs first_run .. first_run+n_runs-1; returns sorted (run_rows, event_rows)."""
    runs = list(range(args.first_run, args.first_run + args.n_runs))
    groups = [g for g in (runs[i::args.workers] for i in range(args.workers)) if g]
    run_rows, event_rows = [], []
    if len(groups) == 1:
        run_rows, event_rows = run_group(args, groups[0])
    else:
        with worker_pool(len(groups)) as pool:
            for r, e in pool.map(run_group, [args] * len(groups), groups):
                run_rows += r
                event_rows += e
    run_rows.sort(key=lambda r: (TRACKERS.index(r["tracker"]), r["run"]))
    event_rows.sort(key=lambda e: (TRACKERS.index(e["tracker"]), e["run"], e["enter_step"]))
    return run_rows, event_rows


def main():
    args = make_parser().parse_args()
    args.process_noise = parse_noise(args.process_noise)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Exit-prediction benchmark: runs {args.first_run}..{args.first_run + args.n_runs - 1} "
          f"x {args.n_steps} steps, {args.workers} workers, models from {args.models_dir}")
    print("Process noise: " + ", ".join(f"{k} {v:g}" for k, v in args.process_noise.items()))
    run_rows, event_rows = run_parallel(args)

    summary, lane = summarize(run_rows, event_rows, args.trackers)
    write_csv(os.path.join(args.out_dir, "exit_benchmark_per_run.csv"), run_rows)
    write_csv(os.path.join(args.out_dir, "exit_benchmark_events.csv"), event_rows)
    write_csv(os.path.join(args.out_dir, "exit_benchmark_summary.csv"), list(summary.values()))
    write_latex(os.path.join(args.out_dir, "exit_benchmark_table.tex"), summary, lane, args)

    print(f"\n{'Tracker':<13}{'noise':>7}{'RMSE all':>16}{'RMSE vis':>16}{'RMSE occ':>16}"
          f"{'exit top-1':>20}{'exit >50%':>11}{'P(correct)':>12}{'ms/step':>9}")
    for name in args.trackers:
        s = summary[name]
        print(f"{name:<13}{str(s['process_noise']):>7}"
              f"{s['rmse_all_mean']:>9.3f}+-{s['rmse_all_std']:<5.3f}"
              f"{s['rmse_visible_mean']:>9.3f}+-{s['rmse_visible_std']:<5.3f}"
              f"{s['rmse_occluded_mean']:>9.3f}+-{s['rmse_occluded_std']:<5.3f}"
              f"{100 * s['exit_success_rate']:>9.1f}% ({s['exit_success']:>3}/{s['exit_events']})"
              f"{100 * s['exit_majority_rate']:>10.1f}%"
              f"{s['exit_p_correct']:>12.3f}{s['ms_per_step_mean']:>9.2f}")
    print(f"\nLane-cue ceiling: {lane['success']}/{lane['events']} "
          f"({100 * lane['success'] / lane['events']:.1f}%), chance 50%")
    print(f"Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
