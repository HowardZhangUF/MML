#!/usr/bin/env python3
"""
Train the MoFlow port: flow-matching teacher, teacher samples on the
training set, then the one-step IMLE student used by the particle filter.

    python -m mml_baselines.moflow.train --out <models dir>

Writes <out>/moflow_teacher.pth, <out>/moflow_motion_model.pth (student) and
<out>/moflow_train.log.
"""

from __future__ import annotations

import argparse
import os

import torch

from mml_baselines.common.scaffold import (
    count_parameters, forecast_metrics, load_split, pick_device, save_checkpoint, train_loop,
)
from mml_baselines.moflow.motion_model import build_student, build_teacher


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--num_query", type=int, default=20)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--social_hidden", type=int, default=64)
    p.add_argument("--enc_layers", type=int, default=2)
    p.add_argument("--dec_blocks", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--teacher_steps", type=int, default=30000)
    p.add_argument("--student_steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "moflow_train.log"), "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    torch.manual_seed(0)
    device = pick_device(args.device)
    split, h_norm, q_norm = load_split(args.horizon)
    config = {k: getattr(args, k) for k in ("horizon", "num_query", "d_model", "social_hidden",
                                            "enc_layers", "dec_blocks", "heads", "dropout")}
    config.update(sampling_steps=10, num_to_gen=30)
    lo, hi = torch.as_tensor(q_norm.lo), torch.as_tensor(q_norm.hi)
    for s in split.values():
        s["qn"] = (s["q"] - lo) / (hi - lo) * 2 - 1

    # MoFlow OPTIMIZATION: AdamW lr 1e-4, wd 0.01, cosine with 5% warmup, grad clip 1.0, EMA 0.995.
    opt = dict(batch=args.batch, lr=args.lr, weight_decay=0.01, ema_beta=0.995, grad_clip=1.0,
               device=device, log=log, cosine=True)

    # ── stage 1: flow-matching teacher ──
    teacher = build_teacher(config, q_norm)
    log(f"MoFlow teacher: {count_parameters(teacher):,} params, config {config}")

    def eval_teacher(m):
        r = forecast_metrics(lambda h: m.sample(h), split["test"], device)
        return f"test minADE {r['minADE']:.4f} minFDE {r['minFDE']:.4f} vel_rmse {r['vel_rmse']:.3f}"

    teacher, _ = train_loop(teacher, lambda m, b: m.loss(b["hist"], b["qn"]),
                            {"hist": split["train"]["hist"], "qn": split["train"]["qn"]},
                            steps=args.teacher_steps, warmup=max(1, args.teacher_steps // 20),
                            eval_fn=eval_teacher, **opt)
    t_metrics = forecast_metrics(lambda h: teacher.sample(h), split["test"], device)
    log(f"teacher final (EMA) test: {t_metrics}")
    save_checkpoint(os.path.join(args.out, "moflow_teacher.pth"), teacher.cpu(), config, h_norm, q_norm,
                    extra={"test_metrics": t_metrics})

    # ── teacher samples on the training set (the IMLE distillation targets) ──
    teacher.to(device)
    torch.manual_seed(1)
    with torch.no_grad():
        targets = torch.cat([teacher.sample(h.to(device))[0].cpu()
                             for h in split["train"]["hist"].split(1024)])
    log(f"teacher samples for distillation: {tuple(targets.shape)}")

    # ── stage 2: one-step IMLE student ──
    student = build_student(config, q_norm)
    log(f"MoFlow IMLE student: {count_parameters(student):,} params")

    def eval_student(m):
        r = forecast_metrics(lambda h: m.sample(h), split["test"], device)
        return f"test minADE {r['minADE']:.4f} minFDE {r['minFDE']:.4f} vel_rmse {r['vel_rmse']:.3f}"

    student, _ = train_loop(student, lambda m, b: m.loss(b["hist"], b["teacher"]),
                            {"hist": split["train"]["hist"], "teacher": targets},
                            steps=args.student_steps, warmup=max(1, args.student_steps // 20),
                            eval_fn=eval_student, **opt)
    s_metrics = forecast_metrics(lambda h: student.sample(h), split["test"], device)
    log(f"student final (EMA) test: {s_metrics}")
    save_checkpoint(os.path.join(args.out, "moflow_motion_model.pth"), student.cpu(), config, h_norm,
                    q_norm, extra={"test_metrics": s_metrics, "teacher_test_metrics": t_metrics})
    log("saved moflow_teacher.pth, moflow_motion_model.pth")


if __name__ == "__main__":
    main()
