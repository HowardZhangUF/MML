#!/usr/bin/env python3
"""
train.py — train every model on every ETH-UCY split (fair_retrain's shared loop
and per-model losses), then score the forecasting track on the split's test windows.

Forecasting track (standard protocol): 8 observed frames, 12 future frames,
best-of-K (K = 20) minADE / minFDE in metres.
    one-step models (CFM, NF, DMMN, DMMN+Gauss, M4, PF+CV): K independent
        autoregressive rollouts, each step v ~ p(v | last 8 positions), x += dt v
        (DMMN and CV are deterministic: their K rollouts coincide)
    forecasters (TrajFlow, MoFlow, TCFM): their own K-trajectory output
        (TrajFlow: the 20 NMS modes; MoFlow: the 20-member IMLE set; TCFM: 20 draws)

    python -m mml_research.motion.ethucy.train --splits eth hotel univ zara1 zara2   # resumes
    python -m mml_research.motion.ethucy.train --splits eth --models cfm --seeds 0 --steps 100   # smoke
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

from mml_research.motion.cfm_variants.common import read_json, sha256, write_json, git_commit
from mml_research.motion.cfm_variants.run import write_csv
from mml_research.motion.ethucy.common import CONFIG_PATH, DEFAULT_OUT_DIR, Translated, load_config
from mml_research.motion.ethucy.data import load_split, recipe, to_model_units
from mml_research.motion.fair_retrain.common import ckpt
from mml_research.motion.fair_retrain.evaluate import load_model
from mml_research.motion.fair_retrain.train import ORDER, train_one

FORECASTERS = ("trajflow", "moflow", "tcfm")


@torch.no_grad()
def forecast(m, model: str, hist: np.ndarray, K: int, H: int, dt: float, chunk: int = 512) -> np.ndarray:
    """hist (B, obs, 2) model units, last point at the origin -> (B, K, H, 2) future positions."""
    out = []
    for i in range(0, len(hist), chunk):
        h = hist[i:i + chunk]
        B, obs = h.shape[:2]
        if model in FORECASTERS:
            from mml_baselines.common.data import history_features
            feats = m.h_norm.norm(history_features(h.reshape(B, -1), m.block_size, m.dt))
            x = torch.from_numpy(feats.astype(np.float32)).to(m.device)
            trajs, probs = [], []
            while sum(t.shape[1] for t in trajs) < K:
                t, p = m.forecast(x)
                trajs.append(t)
                probs.append(p)
            t, p = torch.cat(trajs, 1), torch.cat(probs, 1)
            top = p.argsort(dim=1, descending=True)[:, :K]          # TrajFlow: highest-scored modes
            out.append(t[torch.arange(B, device=t.device)[:, None], top].cpu().numpy())
        else:
            c = np.repeat(h, K, axis=0)
            steps = []
            for _ in range(H):
                nxt = c[:, -1] + dt * m.propose(c.reshape(len(c), -1))
                steps.append(nxt)
                c = np.concatenate([c[:, 1:], nxt[:, None]], axis=1)
            out.append(np.stack(steps, 1).reshape(B, K, H, 2))
    return np.concatenate(out)


def forecast_metrics(pred: np.ndarray, fut: np.ndarray) -> dict:
    """pred (B, K, H, 2), fut (B, H, 2), same units."""
    err = np.linalg.norm(pred - fut[:, None], axis=-1)              # (B, K, H)
    return {"minADE": float(err.mean(-1).min(-1).mean()), "minFDE": float(err[..., -1].min(-1).mean()),
            "meanADE": float(err.mean(-1).mean()), "windows": int(len(fut))}


def score_split(cfg, out_dir, split, models, seeds, log):
    data = load_split(out_dir, split)
    obs, H, dt, sc, K = cfg["obs"], cfg["horizon"], cfg["frame_dt"], cfg["scale"], cfg["forecast"]["k"]
    w = data["win_test"][:cfg["forecast"]["max_windows"]]
    hist, V = to_model_units(w, obs, dt, sc)
    fut = (w[:, obs:] - w[:, obs - 1:obs]) / sc
    rows = []
    for model in models + ["PF+CV"]:
        for seed in (seeds if model != "PF+CV" else [0]):
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = load_model(cfg, out_dir, split, model, seed)
            if model not in FORECASTERS:
                m = Translated(m)
            elif torch.cuda.is_available():
                m.device = torch.device("cuda")
                m.model.to(m.device)
            t0 = time.time()
            r = forecast_metrics(forecast(m, model, hist, K, H, dt) * sc, fut * sc)
            rows.append({"split": split, "method": model, "seed": seed if model != "PF+CV" else "", **r,
                         "seconds": time.time() - t0})
            log(f"  forecast {split:6s} {model:10s} s{seed}: minADE {r['minADE']:.3f}  minFDE {r['minFDE']:.3f}")
    return rows


def main():
    p = argparse.ArgumentParser(description="Train and forecast-score every model on ETH-UCY")
    p.add_argument("--splits", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--steps", type=int, default=None, help="Override the step budget (smoke tests only)")
    p.add_argument("--skip-train", dest="skip_train", action="store_true")
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--config", default=CONFIG_PATH)
    args = p.parse_args()
    cfg = load_config(args.config)
    splits = args.splits or cfg["splits"]
    models = [m for m in ORDER if m in cfg["models"] and (args.models is None or m in args.models)]
    seeds = args.seeds if args.seeds is not None else cfg["training"]["seeds"]
    steps = args.steps or cfg["training"]["steps"]
    log = lambda m: print(m, flush=True)
    os.makedirs(os.path.join(args.out_dir, "models"), exist_ok=True)
    lp = os.path.join(args.out_dir, "training_log.json")
    tlog = read_json(lp) if os.path.exists(lp) else {}
    fc_path = os.path.join(args.out_dir, "forecast.csv")
    fc_rows = []
    if os.path.exists(fc_path):         # keep the rows of splits not rerun now
        import csv
        with open(fc_path, newline="") as f:
            fc_rows = [r for r in csv.DictReader(f) if r["split"] not in splits]
    for split in splits:
        if not args.skip_train:
            d = recipe(cfg, load_split(args.out_dir, split))
            log(f"\n### {split}: {(~d['holdout']).sum()} train / {d['holdout'].sum()} val windows; models {models}")
            for model in models:
                for seed in seeds:
                    key, path = f"{split}_{model}_s{seed}", ckpt(args.out_dir, split, model, seed)
                    if key in tlog and tlog[key]["steps"] == steps and os.path.exists(path) \
                            and sha256(path) == tlog[key]["sha256"]:
                        log(f"== {split} {model} seed {seed}: trained, skipping")
                        continue
                    log(f"\n== {split} {model} seed {seed}")
                    t0 = time.time()
                    path, hist = train_one(split, model, seed, d, cfg, args.out_dir, steps, log)
                    tlog[f"{split}_{model}_s{seed}"] = {"seconds": time.time() - t0, "steps": steps,
                                                        "history": hist, "sha256": sha256(path)}
                    write_json(lp, {**tlog, "_meta": {"commit": git_commit(), "config_sha256": sha256(args.config)}})
        fc_rows += score_split(cfg, args.out_dir, split, models, seeds, log)
        write_csv(fc_path, fc_rows)
    log("done")


if __name__ == "__main__":
    main()
