#!/usr/bin/env python3
"""
train.py — retrain every learned model on a shared recipe (config.yaml).

One loop for all models (`fit`): the recipe's training rows, `steps`
optimizer steps of `batch` rows drawn with the seed's generator, the
method's native optimizer settings (config optim), and checkpoint selection
by the lowest loss of the model's own objective on a fixed holdout subset
(episodes 240..247, fixed noise draws) every `eval_every` steps. A model
contributes only its per-batch loss and its save format, so each checkpoint
loads with the class the benchmarks already use.

    cfm / nf / dmmn         mml_core CFM / NF, ScratchTransformer (DMMN)
    trajflow / moflow / tcfm   the open-source ports (MoFlow: teacher, then the
                            one-step IMLE student on teacher samples; each stage
                            gets the full step budget)
    m1_*                    cfm_variants ContextCFMMotionModel (R1 only)
    dmmn_gauss / m4         Gaussian / CFM residual on this recipe's DMMN, same seed

    python -m mml_research.motion.fair_retrain.train --recipes R0 R1
    python -m mml_research.motion.fair_retrain.train --recipes R0 --models cfm --seeds 0 --steps 200   # smoke
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import copy
import time

import numpy as np
import torch

from mml_research.motion.cfm_variants.common import read_json, sha256, write_json, git_commit
from mml_research.motion.fair_retrain.common import CONFIG_PATH, DEFAULT_OUT_DIR, ckpt, load_config
from mml_research.motion.fair_retrain.data import load_recipe
from mml_research.motion.generate_synthetic_data import DT

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ORDER = ["dmmn", "dmmn_gauss", "m4", "cfm", "nf", "m1_control", "m1_mask", "m1_time", "m1_full",
         "trajflow", "tcfm", "moflow"]     # DMMN first: the residual models need it


def T(a):
    return torch.as_tensor(np.asarray(a, np.float32), device=DEV)


# ── the shared loop ──────────────────────────────────────────────────────────

def fit(module, loss_fn, data, holdout, *, steps, batch, seed, eval_every, ho_rows, optim, log):
    """
    data / holdout: dicts of equal-length tensors on DEV. loss_fn(module, batch_dict) -> scalar.
    Returns (best module (EMA weights if the method uses EMA), history list).
    """
    o = optim
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    module.to(DEV).train()
    params = [p for p in module.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW(params, lr=o["lr"], weight_decay=o["wd"]) if o["opt"] == "adamw"
           else torch.optim.Adam(params, lr=o["lr"]))
    warm = int(o["warmup"] * steps) if o["warmup"] < 1 else int(o["warmup"])

    def lr_at(step):
        f = min(1.0, (step + 1) / warm) if warm else 1.0
        if o["schedule"] == "cosine":       # to 1% of lr, as the CFM / NF / M1 trainers
            f *= 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * step / steps))
        return o["lr"] * f

    ema = None
    if o["ema"]:
        ema = copy.deepcopy(module).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
    n = len(next(iter(data.values())))
    ho_gen = np.random.default_rng(1234)
    n_ho = len(next(iter(holdout.values())))
    chunk = batch       # eval-mode transformer fast paths reject much larger launches (MoFlow)
    ho_idx = torch.as_tensor(ho_gen.choice(n_ho, min(n_ho, ho_rows), replace=False), device=DEV)

    @torch.no_grad()
    def holdout_loss(m):
        was = m.training
        m.eval()
        with torch.random.fork_rng(devices=[DEV] if DEV.type == "cuda" else []):
            torch.manual_seed(4321)        # same noise draws at every evaluation
            vals = [loss_fn(m, {k: v[ho_idx[i:i + chunk]] for k, v in holdout.items()}).item()
                    for i in range(0, len(ho_idx), chunk)]
        m.train(was)
        return float(np.mean(vals))

    best, best_state, hist, run, t0 = np.inf, None, [], [], time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=DEV, generator=gen)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = loss_fn(module, {k: v[idx] for k, v in data.items()})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if o["clip"]:
            torch.nn.utils.clip_grad_norm_(params, o["clip"])
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for e, p in zip(ema.parameters(), module.parameters()):
                    e.lerp_(p.detach(), 1.0 - o["ema"])
                for e, b in zip(ema.buffers(), module.buffers()):
                    e.copy_(b)
        run.append(loss.item())
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            cur = ema if ema is not None else module
            h = holdout_loss(cur)
            if h < best:
                best, best_state = h, copy.deepcopy(cur.state_dict())
            hist.append({"step": step + 1, "train": float(np.mean(run)), "holdout": h})
            log(f"    step {step + 1:6d}  train {np.mean(run):9.5f}  holdout {h:9.5f}  best {best:9.5f}  "
                f"{time.time() - t0:5.0f}s")
            run = []
    final = ema if ema is not None else module
    final.load_state_dict(best_state)
    final.eval()
    return final, hist


# ── per-model pieces ─────────────────────────────────────────────────────────

def split(d, keys):
    tr, ho = ~d["holdout"], d["holdout"]
    return ({k: T(v[tr]) for k, v in keys.items()}, {k: T(v[ho]) for k, v in keys.items()})


def c_flat(d):
    return d["hist"].reshape(len(d["hist"]), -1)


def train_cfm(d, cfg, seed, run_fit, path):
    from mml_core.motion.cfm_motion_model import CFMMotionModel
    a = cfg["arch"]["cfm"]
    m = CFMMotionModel(dt=cfg.get("dt", DT), **a)
    sigma = m.anchor_sigma

    def loss(net, b):
        c, y = b["c"], b["v"]
        x0 = m._finite_difference_velocity(c) + sigma * torch.randn_like(y)
        tau = torch.rand(len(y), 1, device=y.device)
        return ((net((1 - tau) * x0 + tau * y, tau, c) - (y - x0)) ** 2).mean()

    tr, ho = split(d, {"c": c_flat(d), "v": d["V"][:, 0]})
    m.network, hist = run_fit(m.network, loss, tr, ho, "cfm")
    m.save(path)
    return hist


def train_nf(d, cfg, seed, run_fit, path):
    from mml_core.motion.nf_motion_model import NFMotionModel
    m = NFMotionModel(dt=cfg.get("dt", DT), **cfg["arch"]["nf"])
    tr, ho = split(d, {"c": c_flat(d), "v": d["V"][:, 0]})
    m.network, hist = run_fit(m.network, lambda net, b: -net(b["v"], b["c"]).mean(), tr, ho, "nf")
    m.save(path)
    return hist


def train_dmmn(d, cfg, seed, run_fit, path):
    from mml_core.motion.scratch_transformer import ScratchTransformer
    net = ScratchTransformer(**cfg["arch"]["dmmn"])
    tr, ho = split(d, {"c": c_flat(d), "v": d["V"][:, 0]})
    net, hist = run_fit(net, lambda m, b: ((m(b["c"]) - b["v"]) ** 2).mean(), tr, ho, "dmmn")
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, path)   # OfflineDMMNMotionModel format
    return hist


def _cfm_net_loss(net, b, src_std):
    """cfm_variants.cfm_nets.train_cfm's objective: y0 = src + std * eps, straight path to y1."""
    y1 = b["y"]
    y0 = b["src"] + src_std * torch.randn_like(y1)
    tau = torch.rand(len(y1), 1, device=y1.device)
    emb = net.encode(b["s"], b["g"] if b["g"].shape[1] else None)
    return ((net.predict_velocity((1 - tau) * y0 + tau * y1, tau, emb) - (y1 - y0)) ** 2).mean()


def train_m1(d, cfg, seed, run_fit, path, model):
    from mml_research.motion.cfm_variants.context_model import ContextCFMMotionModel
    from mml_research.motion.cfm_variants.features import Normalizer, global_features
    v = cfg["variants"]
    m = ContextCFMMotionModel(v["m1"]["variants"][model], d["hist"].shape[1], cfg.get("dt", DT),
                              v["cfm"]["source_std"], v["cfm"]["ode_steps"],
                              v["training"]["arch"])
    glob = global_features(m.features, d["d"], d["U"])
    if m.n_global:
        m.norm = Normalizer().fit(glob[~d["holdout"]])
    hist = d["hist"].astype(np.float32)
    tr, ho = split(d, {"s": m.step_inputs(hist, d["mask"].astype(np.float32)), "g": m.norm(glob),
                       "src": m.anchor(hist), "y": d["V"][:, 0]})
    m.net, h = run_fit(m.net, lambda net, b: _cfm_net_loss(net, b, m.anchor_sigma), tr, ho, "m1")
    m.info = {"seed": seed, "recipe_rows": int(len(hist)), "fair_retrain": True}
    m.save(path)
    return h


def train_residual(d, cfg, seed, run_fit, path, model, dmmn_path):
    from mml_research.motion.cfm_variants.residual_model import (
        DMMNMean, GaussianResidualMotionModel, ResidualCFMMotionModel,
    )
    from mml_research.motion.cfm_variants.features import Normalizer
    v = cfg["variants"]
    dt = cfg.get("dt", DT)
    dmmn = DMMNMean(dmmn_path, dt)
    c = c_flat(d).astype(np.float32)
    mu = dmmn(c)
    y = d["V"][:, 0]
    ho = d["holdout"]
    if model == "dmmn_gauss":
        GaussianResidualMotionModel(dmmn, dt=dt).fit(mu, y, ho).save(path)
        return []
    m = ResidualCFMMotionModel(dmmn, dt, v["cfm"]["ode_steps"], v["m4"]["residual_scale_floor"],
                               v["training"]["arch"])
    e = y - mu       # ResidualCFMMotionModel.fit's standardization
    m.S = np.maximum(e[~ho].std(0), m.scale_floor).astype(np.float32)
    m.b = e[~ho].mean(0).astype(np.float32)
    m.mu_norm = Normalizer().fit(mu[~ho])
    tr, hd = split(d, {"s": c, "g": m.mu_norm(mu), "src": np.zeros_like(e), "y": e / m.S})
    m.net, h = run_fit(m.net, lambda net, b: _cfm_net_loss(net, b, 1.0), tr, hd, "m4")
    m.info = {"seed": seed, "S": m.S.tolist(), "residual_mean": m.b.tolist(), "fair_retrain": True}
    m.save(path)
    return h


def _port_data(d, horizon, dt=DT):
    from mml_baselines.common.data import MinMax, history_features
    feats = history_features(c_flat(d), d["hist"].shape[1], dt)
    tr = ~d["holdout"]
    h_norm = MinMax().fit(feats[tr])
    Q = (dt * np.cumsum(d["V"][:, :horizon], axis=1)).astype(np.float32)
    q_norm = MinMax().fit(Q[tr])
    return h_norm.norm(feats).astype(np.float32), Q, q_norm.norm(Q).astype(np.float32), h_norm, q_norm


def train_port(d, cfg, seed, run_fit, path, model, log, student_steps):
    from mml_baselines.common.scaffold import save_checkpoint
    H, dt = cfg["horizon"], cfg.get("dt", DT)
    hist, Q, Qn, h_norm, q_norm = _port_data(d, H, dt)
    config = {"horizon": H, "block_size": int(d["hist"].shape[1]), "dt": dt, **cfg["arch"][model]}
    V = d["V"][:, :H]
    if model == "trajflow":
        from mml_baselines.trajflow.motion_model import build_trajflow
        from mml_baselines.trajflow.train import kmeans
        ip = kmeans(Q[~d["holdout"], -1], config["num_query"], seed=seed)
        net = build_trajflow(config, ip, q_norm)
        tr, ho = split(d, {"hist": hist, "q": Q, "v": V})
        net, h = run_fit(net, lambda m, b: m.loss(b["hist"], b["q"], b["v"]), tr, ho, "trajflow")
        save_checkpoint(path, net.cpu(), config, h_norm, q_norm, extra={"intention_points": ip, "seed": seed})
        return h
    if model == "tcfm":
        from mml_baselines.tcfm.motion_model import build_tcfm
        net = build_tcfm(config, q_norm)
        tr, ho = split(d, {"hist": hist, "qn": Qn})
        net, h = run_fit(net, lambda m, b: m.loss(b["hist"], b["qn"]), tr, ho, "tcfm")
        save_checkpoint(path, net.cpu(), config, h_norm, q_norm, extra={"seed": seed})
        return h
    from mml_baselines.moflow.motion_model import build_student, build_teacher
    teacher = build_teacher(config, q_norm)
    tr, ho = split(d, {"hist": hist, "qn": Qn})
    log("    MoFlow stage 1: flow-matching teacher")
    teacher, h1 = run_fit(teacher, lambda m, b: m.loss(b["hist"], b["qn"]), tr, ho, "moflow")
    save_checkpoint(path.replace(".pth", "_teacher.pth"), teacher, config, h_norm, q_norm)
    teacher.to(DEV)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        targets = torch.cat([teacher.sample(T(hist[i:i + 2048]))[0].cpu() for i in range(0, len(hist), 2048)])
    log(f"    MoFlow stage 2: IMLE student on teacher samples {tuple(targets.shape)}")
    student = build_student(config, q_norm)
    tr2, ho2 = split(d, {"hist": hist, "teacher": targets.numpy()})
    student, h2 = run_fit(student, lambda m, b: m.loss(b["hist"], b["teacher"]), tr2, ho2, "moflow",
                          steps=student_steps)
    save_checkpoint(path, student.cpu(), config, h_norm, q_norm, extra={"seed": seed})
    return {"teacher": h1, "student": h2}


def train_one(recipe, model, seed, d, cfg, out_dir, steps, log):
    tc = cfg["training"]
    path = ckpt(out_dir, recipe, model, seed)

    def run_fit(module, loss_fn, tr, ho, optim_key, steps=steps):
        return fit(module, loss_fn, tr, ho, steps=steps, batch=tc["batch"], seed=seed,
                   eval_every=min(tc["eval_every"], steps), ho_rows=tc["holdout_rows"],
                   optim=cfg["optim"][optim_key], log=log)

    if model == "cfm":
        return path, train_cfm(d, cfg, seed, run_fit, path)
    if model == "nf":
        return path, train_nf(d, cfg, seed, run_fit, path)
    if model == "dmmn":
        return path, train_dmmn(d, cfg, seed, run_fit, path)
    if model.startswith("m1_"):
        return path, train_m1(d, cfg, seed, run_fit, path, model)
    if model in ("m4", "dmmn_gauss"):
        return path, train_residual(d, cfg, seed, run_fit, path, model, ckpt(out_dir, recipe, "dmmn", seed))
    student_steps = min(steps, tc["moflow_student_steps"])
    return path, train_port(d, cfg, seed, run_fit, path, model, log, student_steps)


def main():
    p = argparse.ArgumentParser(description="Fair retraining of every learned motion model")
    p.add_argument("--recipes", nargs="+", default=["R0", "R1"])
    p.add_argument("--models", nargs="+", default=None, help="Subset of config models (default: all)")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--steps", type=int, default=None, help="Override the step budget (smoke tests only)")
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--config", default=CONFIG_PATH)
    args = p.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg["training"]["seeds"]
    steps = args.steps or cfg["training"]["steps"]
    os.makedirs(os.path.join(args.out_dir, "models"), exist_ok=True)
    lp = os.path.join(args.out_dir, "training_log.json")
    log_all = read_json(lp) if os.path.exists(lp) else {}
    print(f"device {DEV}, budget {steps} steps x {cfg['training']['batch']}", flush=True)
    for recipe in args.recipes:
        d = load_recipe(args.out_dir, recipe)
        models = [m for m in ORDER if m in cfg["models"][recipe] and (args.models is None or m in args.models)]
        print(f"\n### {recipe}: {len(d['run'])} rows ({d['holdout'].sum()} holdout); models {models}", flush=True)
        for model in models:
            for seed in seeds:
                print(f"\n== {recipe} {model} seed {seed}", flush=True)
                t0 = time.time()
                path, hist = train_one(recipe, model, seed, d, cfg, args.out_dir, steps,
                                       lambda m: print(m, flush=True))
                log_all[f"{recipe}_{model}_s{seed}"] = {"seconds": time.time() - t0, "steps": steps,
                                                        "history": hist, "sha256": sha256(path)}
                write_json(lp, {**log_all, "_meta": {"commit": git_commit(), "config_sha256": sha256(args.config),
                                                     "batch": cfg["training"]["batch"]}})
                print(f"   saved {path} ({time.time() - t0:.0f}s)", flush=True)
    print("training done")


if __name__ == "__main__":
    main()
