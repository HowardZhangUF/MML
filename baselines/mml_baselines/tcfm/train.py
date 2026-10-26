#!/usr/bin/env python3
"""
Train the TCFM port on the benchmark's training stream.

    python -m mml_baselines.tcfm.train --out <models dir>

Writes <out>/tcfm_motion_model.pth and <out>/tcfm_train.log.
"""

from __future__ import annotations

import argparse
import os

import torch

from mml_baselines.common.scaffold import (
    count_parameters, forecast_metrics, load_split, pick_device, save_checkpoint, train_loop,
)
from mml_baselines.tcfm.motion_model import build_tcfm


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=12, help="must be divisible by 4 (two down-samplings)")
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--lstm_out_dim", type=int, default=32)
    p.add_argument("--n_timesteps", type=int, default=10, help="Euler steps at sampling time")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "tcfm_train.log"), "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    torch.manual_seed(0)
    device = pick_device(args.device)
    split, h_norm, q_norm = load_split(args.horizon)
    lo, hi = torch.as_tensor(q_norm.lo), torch.as_tensor(q_norm.hi)
    for s in split.values():
        s["qn"] = (s["q"] - lo) / (hi - lo) * 2 - 1
    config = {"horizon": args.horizon, "dim": args.dim, "dim_mults": args.dim_mults,
              "lstm_out_dim": args.lstm_out_dim, "n_timesteps": args.n_timesteps}
    model = build_tcfm(config, q_norm)
    log(f"TCFM port: {count_parameters(model):,} params, train {len(split['train']['q'])} "
        f"test {len(split['test']['q'])}, config {config}")

    def eval_fn(m):
        r = forecast_metrics(lambda h: m.sample(h), split["test"], device)
        return f"test ADE {r['minADE']:.4f} FDE {r['minFDE']:.4f} vel_rmse {r['vel_rmse']:.3f}"

    # TCFM Trainer: Adam lr 2e-4, batch 32 x 2 accumulation (= 64), no grad clip. Its EMA is a
    # no-op (Trainer sets self.ema_model = self.model), so the raw weights are used: ema_beta 0.
    model, _ = train_loop(model, lambda m, b: m.loss(b["hist"], b["qn"]),
                          {"hist": split["train"]["hist"], "qn": split["train"]["qn"]},
                          steps=args.steps, batch=args.batch, lr=args.lr, weight_decay=0.0, ema_beta=0.0,
                          grad_clip=None, device=device, log=log, eval_fn=eval_fn)
    final = forecast_metrics(lambda h: model.sample(h), split["test"], device)
    log(f"final test: {final}")
    save_checkpoint(os.path.join(args.out, "tcfm_motion_model.pth"), model.cpu(), config, h_norm, q_norm,
                    extra={"test_metrics": final})
    log("saved tcfm_motion_model.pth")


if __name__ == "__main__":
    main()
