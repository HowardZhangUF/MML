#!/usr/bin/env python3
"""
train.py — train every model on the WOMD training rows (fair_retrain's shared
loop and per-model losses), then score the forecasting track on the official
validation split (config forecast; metrics.py).

    python -m mml_research.motion.womd.train                       # train (resumes) + forecast
    python -m mml_research.motion.womd.train --skip-train          # forecast only
    python -m mml_research.motion.womd.train --models cfm --seeds 0 --steps 100 --max-agents 2000   # smoke
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import time

import numpy as np
import torch

# Initialise CUDA while the GPU is visible: cfm_variants.run and fair_retrain.evaluate (CPU
# evaluators) set CUDA_VISIBLE_DEVICES="" on import, which would leave fair_retrain.train's DEV on cpu.
# device_count() re-reads that variable until it is called once after init (torch.load of a cuda
# checkpoint then fails with "device_count() is 0"), so cache it here too.
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.device_count()

from mml_research.motion.cfm_variants.common import git_commit, read_json, sha256, write_json
from mml_research.motion.cfm_variants.run import write_csv
from mml_research.motion.ethucy.train import FORECASTERS, forecast
from mml_research.motion.fair_retrain.common import ckpt
from mml_research.motion.fair_retrain.evaluate import load_model
from mml_research.motion.fair_retrain.train import ORDER, train_one
from mml_research.motion.womd.common import CONFIG_PATH, DEFAULT_OUT_DIR, RECIPE, Canonical, canonical, load_config
from mml_research.motion.womd.data import impute_history, load, recipe
from mml_research.motion.womd.metrics import aggregate, per_agent

CHUNK = 8192            # agents per forecasting chunk


def forecast_world(m, model, hist, cfg):
    """hist (B, obs, 2) model units, current point at the origin, world orientation -> (B, K, H, 2) metres."""
    sc = cfg["scale"]
    hc, M = canonical(hist, cfg["canon_min_disp"] / sc)
    m = m if model in FORECASTERS else Canonical(m, cfg["canon_min_disp"] / sc)
    pc = forecast(m, model, hc, cfg["forecast"]["k"], cfg["horizon"], cfg["frame_dt"],
                  chunk=1024 if model in FORECASTERS else 4096)
    return np.einsum("bthk,bjk->bthj", pc, M) * sc


def score(cfg, out_dir, models, seeds, max_agents, log):
    val = load(out_dir, "val")
    n = len(val["type"]) if max_agents is None else min(max_agents, len(val["type"]))
    obs = cfg["obs"]
    hist = impute_history(val["xy"][:n], val["valid"][:n], obs) / cfg["scale"]
    gt, valid, heading = val["xy"][:n, obs:], val["valid"][:n, obs:], val["heading"][:n, obs:]
    os.makedirs(os.path.join(out_dir, "forecast"), exist_ok=True)
    rows = []
    for model in models + ["PF+CV"]:
        for seed in (seeds if model != "PF+CV" else [0]):
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = load_model(cfg, out_dir, RECIPE, model, seed)
            if model in FORECASTERS and torch.cuda.is_available():
                m.device = torch.device("cuda")
                m.model.to(m.device)
            t0 = time.time()
            parts = []
            for i in range(0, n, CHUNK):
                pred = forecast_world(m, model, hist[i:i + CHUNK], cfg)
                parts.append(per_agent(pred, gt[i:i + CHUNK], valid[i:i + CHUNK], heading[i:i + CHUNK],
                                       val["vel0"][i:i + CHUNK], cfg))
            pa = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            tag = f"{model}_s{seed}" if model != "PF+CV" else model
            np.savez_compressed(os.path.join(out_dir, "forecast", f"{tag}.npz"), **pa)
            r = aggregate(pa, val["type"][:n], cfg)
            rows.append({"method": model, "seed": seed if model != "PF+CV" else "", **r, "agents": n,
                         "seconds": time.time() - t0})
            log(f"  forecast {model:10s} s{seed}: minADE {r['minADE']:.3f}  minFDE {r['minFDE']:.3f}  "
                f"MR {r['MR']:.3f}  ({time.time() - t0:.0f}s)")
    return rows


def main():
    p = argparse.ArgumentParser(description="Train and forecast-score every model on WOMD")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--steps", type=int, default=None, help="Override the step budget (smoke tests only)")
    p.add_argument("--max-agents", dest="max_agents", type=int, default=None, help="smoke tests only")
    p.add_argument("--skip-train", dest="skip_train", action="store_true")
    p.add_argument("--skip-forecast", dest="skip_forecast", action="store_true")
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--config", default=CONFIG_PATH)
    args = p.parse_args()
    cfg = load_config(args.config)
    models = [m for m in ORDER if m in cfg["models"] and (args.models is None or m in args.models)]
    seeds = args.seeds if args.seeds is not None else cfg["training"]["seeds"]
    steps = args.steps or cfg["training"]["steps"]
    log = lambda m: print(m, flush=True)
    os.makedirs(os.path.join(args.out_dir, "models"), exist_ok=True)
    lp = os.path.join(args.out_dir, "training_log.json")
    tlog = read_json(lp) if os.path.exists(lp) else {}
    if not args.skip_train:
        d = recipe(cfg, load(args.out_dir, "train"))
        log(f"### {(~d['holdout']).sum()} train / {d['holdout'].sum()} holdout rows; models {models}; "
            f"budget {steps} x {cfg['training']['batch']}")
        for model in models:
            for seed in seeds:
                key = f"{RECIPE}_{model}_s{seed}"
                path = ckpt(args.out_dir, RECIPE, model, seed)
                if key in tlog and tlog[key]["steps"] == steps and os.path.exists(path) \
                        and sha256(path) == tlog[key]["sha256"]:
                    log(f"== {model} seed {seed}: trained, skipping")
                    continue
                log(f"\n== {model} seed {seed}")
                t0 = time.time()
                path, hist = train_one(RECIPE, model, seed, d, cfg, args.out_dir, steps, log)
                tlog[key] = {"seconds": time.time() - t0, "steps": steps, "history": hist, "sha256": sha256(path)}
                write_json(lp, {**tlog, "_meta": {"commit": git_commit(), "config_sha256": sha256(args.config)}})
                log(f"   saved {path} ({time.time() - t0:.0f}s)")
    if not args.skip_forecast:
        rows = score(cfg, args.out_dir, models, seeds, args.max_agents, log)
        write_csv(os.path.join(args.out_dir, "forecast.csv"), rows)
    log("done")


if __name__ == "__main__":
    main()
