#!/usr/bin/env python3
"""
train.py — fit the learned variants on rollouts.npz (train episodes only).

    m1_control / m1_mask / m1_time / m1_full   ContextCFMMotionModel
    m4                                         ResidualCFMMotionModel (DMMN frozen)
    dmmn_gauss                                 GaussianResidualMotionModel (closed form, one fit)
    m3 / m3_joint                              LatentGaussianProposal on xi / on (xi, eta/q) (CFM frozen)

Every model uses the same declared budget (config training.steps optimizer
steps of batch training.batch); episodes 240..247 are held out and only
reported. One checkpoint per (method, seed) in <out-dir>/models/.

    python -m mml_research.motion.cfm_variants.train --methods all
"""

from __future__ import annotations

import os

# benchmark_cv_dmmn (imported for the DMMN mean) hides the GPU with setdefault;
# claim it first so training runs on CUDA when one exists.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import time

import numpy as np
import torch

from mml_research.motion.cfm_variants.common import (
    CONFIG_PATH, DEFAULT_MODELS_DIR, DEFAULT_OUT_DIR, TRAIN_HOLDOUT_RUNS, ckpt_path, git_commit,
    load_config, read_json, sha256, variant_models_dir, write_json,
)

TRAINABLE = ["m1_control", "m1_mask", "m1_time", "m1_full", "m4", "dmmn_gauss", "m3", "m3_joint"]


def main():
    p = argparse.ArgumentParser(description="Train the CFM variants")
    p.add_argument("--methods", nargs="+", default=["all"], choices=TRAINABLE + ["all"])
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--steps", type=int, default=None, help="Override the optimizer-step budget (smoke tests)")
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--models-dir", dest="models_dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    cfg = load_config(args.config)
    tc = cfg["training"]
    methods = TRAINABLE if "all" in args.methods else args.methods
    seeds = args.seeds if args.seeds is not None else tc["seeds"]
    n_steps = args.steps or tc["steps"]
    os.makedirs(variant_models_dir(args.out_dir), exist_ok=True)
    dt = cfg["filter"]["dt_seconds"]

    rpath = os.path.join(args.out_dir, "rollouts.npz")
    data = dict(np.load(rpath))
    holdout = np.isin(data["run"], list(TRAIN_HOLDOUT_RUNS))
    c_flat = data["hist"].reshape(len(data["hist"]), -1).astype(np.float32)
    print(f"rollouts: {len(c_flat)} contexts, {holdout.sum()} held out; device "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    mu = None
    if {"m4", "dmmn_gauss"} & set(methods):
        from mml_research.motion.cfm_variants.residual_model import DMMNMean
        dmmn = DMMNMean(os.path.join(args.models_dir, cfg["dmmn_checkpoint"]), dt)
        t0 = time.time()
        mu = dmmn(c_flat)
        e = data["v"] - mu
        print(f"DMMN mean on rollouts in {time.time() - t0:.0f}s; residual mean {e[~holdout].mean(0)}, "
              f"std {e[~holdout].std(0)}; holdout std {e[holdout].std(0)}", flush=True)

    cfm = None
    if {"m3", "m3_joint"} & set(methods):
        from mml_core.motion.cfm_motion_model import CFMMotionModel
        cfm = CFMMotionModel()
        cfm.load(os.path.join(args.models_dir, cfg["cfm"]["checkpoint"]))
        for prm in cfm.network.parameters():
            prm.requires_grad_(False)

    log = {}
    lp = os.path.join(args.out_dir, "training_log.json")
    if os.path.exists(lp):
        log = read_json(lp)
    for method in methods:
        for seed in ([0] if method == "dmmn_gauss" else seeds):
            path = ckpt_path(args.out_dir, method, seed)
            print(f"\n== {method} seed {seed} -> {path}", flush=True)
            t0 = time.time()
            if method.startswith("m1_"):
                from mml_research.motion.cfm_variants.context_model import ContextCFMMotionModel
                m = ContextCFMMotionModel(cfg["m1"]["variants"][method], 5, dt, cfg["cfm"]["source_std"],
                                          cfg["cfm"]["ode_steps"], tc["arch"])
                losses = m.fit(data, holdout, n_steps, tc["batch"], tc["lr"], seed)
            elif method == "m4":
                from mml_research.motion.cfm_variants.residual_model import ResidualCFMMotionModel
                m = ResidualCFMMotionModel(dmmn, dt, cfg["cfm"]["ode_steps"], cfg["m4"]["residual_scale_floor"],
                                           tc["arch"])
                losses = m.fit(c_flat, mu, data["v"], holdout, n_steps, tc["batch"], tc["lr"], seed)
            elif method == "dmmn_gauss":
                from mml_research.motion.cfm_variants.residual_model import GaussianResidualMotionModel
                m = GaussianResidualMotionModel(dmmn, dt=dt).fit(mu, data["v"], holdout)
                losses = ([], [])
                print(f"    b={m.b}, cov={m.cov.tolist()}")
            else:
                from mml_research.motion.cfm_variants.observation_proposal import LatentGaussianProposal
                vis = data["vis"].astype(bool)
                m = LatentGaussianProposal(cfm.n_embed, cfg["m3"]["hidden"], cfg["m3"]["scale_floor"],
                                           cfg["filter"]["measurement_sigma"], 4 if method == "m3_joint" else 2)
                losses = m.fit(cfm, c_flat[vis], holdout[vis], cfg["m3"]["train_process_noise"],
                               n_steps, tc["batch"], tc["lr"], seed)
            m.save(path)
            log[f"{method}_s{seed}"] = {"seconds": time.time() - t0, "train": losses[0], "holdout": losses[1],
                                        "sha256": sha256(path), "steps": n_steps}
            write_json(lp, {**log, "_meta": {"commit": git_commit(), "rollouts_sha256": sha256(rpath),
                                             "budget": {"steps": n_steps, "batch": tc["batch"], "lr": tc["lr"]}}})
    print("training done")


if __name__ == "__main__":
    main()
