#!/usr/bin/env python3
"""
Train the TrajFlow port on the benchmark's training stream.

    python -m mml_baselines.trajflow.train --out <models dir>

Writes <out>/trajflow_motion_model.pth and <out>/trajflow_train.log.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from mml_baselines.common.scaffold import (
    count_parameters, forecast_metrics, load_split, pick_device, save_checkpoint, train_loop,
)
from mml_baselines.trajflow.motion_model import build_trajflow


def kmeans(points: np.ndarray, k: int, iters: int = 100, seed: int = 0) -> np.ndarray:
    """Intention points: k-means of training goals (MTR's cluster_64_center_dict, here K centres)."""
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), k, replace=False)]
    for _ in range(iters):
        label = ((points[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.array([points[label == j].mean(0) if np.any(label == j) else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return centers.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--num_query", type=int, default=16)
    p.add_argument("--num_motion_modes", type=int, default=6)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--nms_dist", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "trajflow_train.log"), "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    torch.manual_seed(0)
    device = pick_device(args.device)
    split, h_norm, q_norm = load_split(args.horizon)
    intention_points = kmeans(split["train"]["q"][:, -1].numpy(), args.num_query)
    config = {"horizon": args.horizon, "d_model": args.d_model, "heads": args.heads, "depth": args.depth,
              "dropout": args.dropout, "num_motion_modes": args.num_motion_modes, "nms_dist": args.nms_dist,
              "num_inter_layers": 2, "sampling_steps": 1, "num_query": args.num_query}
    model = build_trajflow(config, intention_points, q_norm)
    log(f"TrajFlow port: {count_parameters(model):,} params, train {len(split['train']['q'])} "
        f"test {len(split['test']['q'])}, config {config}")

    def eval_fn(m):
        r = forecast_metrics(lambda h: m.sample(h), split["test"], device)
        return f"test minADE {r['minADE']:.4f} minFDE {r['minFDE']:.4f} vel_rmse {r['vel_rmse']:.3f}"

    # TrajFlow OPT: AdamW, lr 1e-4, weight decay 0.01, EMA 0.999, grad clip 1000.
    ema_model, _ = train_loop(
        model, lambda m, b: m.loss(b["hist"], b["q"], b["v"]), split["train"],
        steps=args.steps, batch=args.batch, lr=args.lr, weight_decay=0.01, ema_beta=0.999,
        grad_clip=1000.0, device=device, log=log, eval_fn=eval_fn)

    final = forecast_metrics(lambda h: ema_model.sample(h), split["test"], device)
    log(f"final (EMA) test: {final}")
    save_checkpoint(os.path.join(args.out, "trajflow_motion_model.pth"), ema_model.cpu(), config,
                    h_norm, q_norm, extra={"intention_points": intention_points, "test_metrics": final})
    log("saved trajflow_motion_model.pth")


if __name__ == "__main__":
    main()
