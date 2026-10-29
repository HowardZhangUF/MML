#!/usr/bin/env python3
"""
evaluate.py — process-noise calibration (validation), lock, and test of every
fair_v1 method, with the cfm_variants filter and episode evaluation unchanged.

    calibrate  every method x config calibration.grid on validation episodes
               100..119 (seed-0 checkpoints); the M2 rule picks each method's q;
               writes val/ CSVs and locked.json (q, checkpoint and config hashes)
    test       episodes 1000..1099, all seeds, the locked q; refuses to run if a
               checkpoint or the config changed after locking

Method names: "<recipe>/<model>" (e.g. R1/TrajFlow) for the retrained models,
plus the training-free PF+CV and EnFF.

    python -m mml_research.motion.fair_retrain.evaluate calibrate --workers 96
    python -m mml_research.motion.fair_retrain.evaluate test --workers 96
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from mml_research.motion.cfm_variants.calibrate import select
from mml_research.motion.cfm_variants.common import git_commit, read_json, sha256, write_json
from mml_research.motion.cfm_variants.evaluation import run_episode
from mml_research.motion.cfm_variants.run import environment, write_csv, write_steps
from mml_research.motion.fair_retrain.common import (
    CONFIG_PATH, DEFAULT_OUT_DIR, MODEL_LABEL, all_methods, ckpt, load_config,
)
from mml_research.motion.generate_synthetic_data import DT

LABEL_MODEL = {v: k for k, v in MODEL_LABEL.items()}
# Relative cost per episode, so the slow jobs start first.
_COST = {"tcfm": 36.0, "trajflow": 8.0, "moflow": 3.5, "m4": 1.6, "m1": 1.3, "cfm": 1.0, "nf": 1.0,
         "dmmn_gauss": 0.8, "dmmn": 0.6, "enff": 0.4, "cv": 0.1}


def parse(name: str):
    """'R1/TrajFlow' -> ('R1', 'trajflow'); training-free -> (None, name)."""
    if "/" in name:
        recipe, label = name.split("/", 1)
        return recipe, LABEL_MODEL[label]
    return None, name


def kind(name: str) -> str:
    recipe, model = parse(name)
    if recipe is None:
        return "cv" if model == "PF+CV" else "enff"
    return "m1" if model.startswith("m1_") else model


def checkpoint_files(names, seeds, out_dir):
    files = {}
    for name in names:
        recipe, model = parse(name)
        if recipe is None:
            continue
        for s in seeds:
            files[f"{recipe}_{model}_s{s}"] = ckpt(out_dir, recipe, model, s)
            if model in ("m4", "dmmn_gauss"):
                files[f"{recipe}_dmmn_s{s}"] = ckpt(out_dir, recipe, "dmmn", s)
    return files


def load_model(cfg, out_dir, recipe, model, seed):
    """A trained (or training-free PF+CV) one-step motion model; dt / history from cfg (default: simulator)."""
    dt = cfg.get("dt", DT)
    if model == "PF+CV":
        from mml_research.motion.benchmark_cv_dmmn import ConstantVelocityMotionModel
        return ConstantVelocityMotionModel(block_size=cfg.get("obs", 5), dt=dt)
    path = ckpt(out_dir, recipe, model, seed)
    if model == "cfm":
        from mml_core.motion.cfm_motion_model import CFMMotionModel
        m = CFMMotionModel()
        m.load(path)
        return m
    if model == "nf":
        from mml_core.motion.nf_motion_model import NFMotionModel
        m = NFMotionModel()
        m.load(path)
        return m
    if model == "dmmn":
        from mml_research.motion.benchmark_cv_dmmn import OfflineDMMNMotionModel
        return OfflineDMMNMotionModel(path, dt=dt)
    if model in ("m4", "dmmn_gauss"):
        from mml_research.motion.cfm_variants.residual_model import (
            DMMNMean, GaussianResidualMotionModel, ResidualCFMMotionModel,
        )
        mean = DMMNMean(ckpt(out_dir, recipe, "dmmn", seed), dt)
        return (ResidualCFMMotionModel if model == "m4" else GaussianResidualMotionModel).load(path, mean)
    if model.startswith("m1_"):
        from mml_research.motion.cfm_variants.context_model import ContextCFMMotionModel
        return ContextCFMMotionModel.load(path)
    from mml_baselines.moflow.motion_model import MoFlowMotionModel
    from mml_baselines.tcfm.motion_model import TCFMMotionModel
    from mml_baselines.trajflow.motion_model import TrajFlowMotionModel
    return {"trajflow": TrajFlowMotionModel, "moflow": MoFlowMotionModel, "tcfm": TCFMMotionModel}[model](path)


def build_factory(cfg, out_dir, name, seed, q):
    from mml_research.motion.cfm_variants.tracker import MotionSampler, VariantPFTracker
    f = cfg["variants"]["filter"]
    recipe, model = parse(name)
    if model == "EnFF":
        from mml_baselines.enff.tracker import EnFFTracker
        e = cfg["enff"]
        return lambda: EnFFTracker(n_particles=f["particles"], process_noise=q, meas_sigma=f["measurement_sigma"],
                                   sigma_min=e["sigma_min"], guidance_lambda=e["lambda"],
                                   sampling_time_step_count=e["steps"], dt=DT)
    m = load_model(cfg, out_dir, recipe, model, seed)
    return lambda: VariantPFTracker(MotionSampler(m, q), n_particles=f["particles"],
                                    meas_sigma=f["measurement_sigma"], init_spread=f["init_spread"])


_FACTORIES = {}


def _job(cfg_path, out_dir, name, seed, q, run, n_steps, keep_clouds):
    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    key = (name, seed, q)
    if key not in _FACTORIES:          # one model load per (method, seed, q) and worker
        _FACTORIES.clear()
        _FACTORIES[key] = build_factory(cfg, out_dir, name, seed, q)
    run_row, events, steps, clouds = run_episode(_FACTORIES[key], run, n_steps, cfg["variants"]["evaluation"],
                                                 keep_clouds)
    tag = {"method": name, "seed": "" if parse(name)[0] is None else seed, "process_noise": q}
    return {**tag, **run_row}, [{**tag, **e} for e in events], (name, seed, run, steps), clouds


def run_jobs(jobs, workers, cfg_path, out_dir, n_steps, order, keep_clouds=0):
    """jobs: (name, seed, q, run), sorted so expensive and same-model jobs cluster."""
    jobs = sorted(jobs, key=lambda j: (-_COST[kind(j[0])], j[0], j[1], j[2], j[3]))
    rows, events, steps, clouds = [], [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        futs = [pool.submit(_job, cfg_path, out_dir, n, s, q, r, n_steps, keep_clouds if s == 0 else 0)
                for n, s, q, r in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r, e, s, c = f.result()
            rows.append(r)
            events += e
            steps.append(s)
            clouds += [{"method": r["method"], **x} for x in c]
            if i % max(1, len(jobs) // 100) == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} jobs done ({time.time() - t0:.0f}s)", flush=True)
    rank = {n: k for k, n in enumerate(order)}
    key = lambda r: (rank[r["method"]], str(r["seed"]), r["process_noise"], r["run"])
    rows.sort(key=key)
    events.sort(key=lambda e: (*key(e), e["enter_step"]))
    return rows, events, steps, clouds


def calibrate(args, cfg, methods):
    grid, tol = cfg["calibration"]["grid"], cfg["calibration"]["rmse_tolerance"]
    runs = list(range(*cfg["calibration"]["runs"]))[:args.n_runs]
    out = os.path.join(args.out_dir, "val")
    os.makedirs(out, exist_ok=True)
    jobs = [(m, 0, float(q), r) for m in methods for q in grid for r in runs]
    print(f"calibration: {len(methods)} methods x q {grid} x episodes {runs[0]}..{runs[-1]} = {len(jobs)} jobs",
          flush=True)
    rows, events, _, _ = run_jobs(jobs, args.workers, args.config, args.out_dir, args.n_steps, methods)
    write_csv(os.path.join(out, "episodes.csv"), rows)
    write_csv(os.path.join(out, "events.csv"), events)
    table, chosen = [], {}
    for m in methods:
        cand = []
        for q in grid:
            rr = [r for r in rows if r["method"] == m and r["process_noise"] == q]
            ev = [e for e in events if e["method"] == m and e["process_noise"] == q]
            cand.append({"method": m, "process_noise": q, "runs": len(rr),
                         **{k: float(np.nanmean([r[k] for r in rr])) for k in
                            ("rmse_all", "rmse_occluded", "es_mean", "ms_per_step")},
                         "exit_success_rate": float(np.mean([e["success"] for e in ev])),
                         "p_correct": float(np.mean([e["p_correct"] for e in ev]))})
        best = select(cand, tol)
        chosen[m] = best["process_noise"]
        for c in cand:
            c["selected"] = int(c is best)
        table += cand
        print(f"  {m:16s} q = {chosen[m]:<6g} (val RMSE {best['rmse_all']:.4f}, ES {best['es_mean']:.4f}, "
              f"exit {100 * best['exit_success_rate']:.1f}%)", flush=True)
    write_csv(os.path.join(out, "calibration.csv"), table)
    seeds = cfg["training"]["seeds"]
    files = checkpoint_files(methods, seeds, args.out_dir)
    write_json(os.path.join(args.out_dir, "locked.json"), {
        "process_noise": chosen,
        "rule": f"min mean val energy score s.t. RMSE_all <= (1+{tol}) * min grid RMSE; ties -> smaller q",
        "validation_runs": runs, "grid": grid,
        "checkpoints": {k: sha256(v) for k, v in files.items()}, "checkpoint_files": files,
        "config_sha256": sha256(args.config), "commit": git_commit(), "locked_at": time.strftime("%F %T"),
    })
    print("locked.json written")


def test(args, cfg, methods):
    lock = read_json(os.path.join(args.out_dir, "locked.json"))
    seeds = cfg["training"]["seeds"]
    files = checkpoint_files(methods, seeds, args.out_dir)
    changed = [k for k, v in files.items() if lock["checkpoints"].get(k) != sha256(v)]
    if changed or sha256(args.config) != lock["config_sha256"]:
        sys.exit(f"changed since locking: checkpoints {changed}, config "
                 f"{sha256(args.config) != lock['config_sha256']}")
    runs = list(range(*cfg["test_runs"]))[:args.n_runs]
    out = os.path.join(args.out_dir, "test")
    os.makedirs(out, exist_ok=True)
    jobs = [(m, s, float(lock["process_noise"][m]), r) for m in methods
            for s in (seeds if parse(m)[0] else [0]) for r in runs]
    print(f"test: {len(methods)} methods, episodes {runs[0]}..{runs[-1]}, {len(jobs)} jobs", flush=True)
    t0 = time.time()
    rows, events, steps, clouds = run_jobs(jobs, args.workers, args.config, args.out_dir, args.n_steps, methods,
                                           cfg["variants"]["evaluation"]["clouds_saved_per_method"])
    for rr in (rows, events):
        for r in rr:
            r["split"] = "test"
    write_csv(os.path.join(out, "episodes.csv"), rows)
    write_csv(os.path.join(out, "events.csv"), events)
    write_steps(out, steps)
    np.savez_compressed(os.path.join(out, "clouds.npz"), clouds=np.array(clouds, dtype=object))
    write_json(os.path.join(out, "manifest.json"), {
        "commit": git_commit(), "command": " ".join(sys.argv), "config": cfg, "config_sha256": sha256(args.config),
        "checkpoint_sha256": {k: sha256(v) for k, v in files.items()}, "runs": runs, "n_steps": args.n_steps,
        "process_noise": lock["process_noise"], "environment": environment(), "wall_seconds": time.time() - t0,
    })
    print(f"done in {time.time() - t0:.0f}s -> {out}")


def main():
    p = argparse.ArgumentParser(description="Calibrate / test the fair_v1 methods")
    p.add_argument("stage", choices=["calibrate", "test"])
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--n_runs", type=int, default=None, help="First n episodes only (smoke tests)")
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    cfg = load_config(args.config)
    methods = args.methods or all_methods(cfg)
    (calibrate if args.stage == "calibrate" else test)(args, cfg, methods)


if __name__ == "__main__":
    main()
