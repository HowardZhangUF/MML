#!/usr/bin/env python3
"""
verify.py — correctness checks before interpreting results (guide 12.1, 6.6, 9 steps 5-6).

Hard checks (must pass):
    regression      VariantPFTracker == BootstrapPFTracker for CFM / DMMN / NF
    generator       G(C, xi) == CFMMotionModel.propose for the same draw
    m3_identity     q = N(0, I): zero log correction, positions == prior path
    m3_no_obs       without z the M3 sampler is the prior sampler
    m3_analytic     latent IS recovers analytic posteriors (x = xi, x = xi + eta)
    m4_reconstruct  mu + S r* == v*
    m2_covariance   Cov(x | C) == dt^2 Cov(v | C) + q^2 I through the sampler
    m1_causal       the context passed to the sampler at k never depends on z_k
Diagnostics (reported):
    solver          endpoint change for N_FE 10 / 20 / 40 with identical inputs
    m3_ess          one-step ESS/N, bootstrap vs learned proposal, held-out contexts
    m4_residuals    train vs held-out residual mean / std (overfit check)
    one_step        sample bias / RMSE / spread of each motion model on held-out
                    PF-rollout contexts, visible vs occluded

    python -m mml_research.motion.cfm_variants.verify
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import time

import numpy as np
import torch

from mml_research.motion.benchmark_diffpf_vs_cfm import BootstrapPFTracker
from mml_research.motion.cfm_variants.common import (
    CONFIG_PATH, DEFAULT_MODELS_DIR, DEFAULT_OUT_DIR, TRAIN_HOLDOUT_RUNS, ckpt_path, load_config, write_json,
)
from mml_research.motion.cfm_variants.features import global_features
from mml_research.motion.cfm_variants.methods import ModelCache
from mml_research.motion.cfm_variants.observation_proposal import (
    IdentityProposal, cfm_generate_from_base, mvn_tril_log_prob, std_normal_log_prob,
)
from mml_research.motion.cfm_variants.tracker import (
    LatentProposalSampler, MotionSampler, VariantPFTracker, flat_context,
)
from mml_research.motion.generate_synthetic_data import NODE_POS, NOISE_STD
from mml_research.motion.train_diffpf import simulate_full_trajectory


def _trajectory(run=0, n=2000):
    pos, vis = simulate_full_trajectory(run % len(NODE_POS), n, np.random.default_rng(1000 + run))
    return pos, vis


def _run(tracker, pos, vis, run, n_steps):
    meas_rng = np.random.default_rng(5000 + run)
    np.random.seed(9000 + run)
    torch.manual_seed(9000 + run)
    first = int(np.argmax(vis))
    tracker.reset(pos[first] + NOISE_STD * meas_rng.standard_normal(2).astype(np.float32))
    est = []
    for t in range(first + 1, min(len(pos), first + 1 + n_steps)):
        z = pos[t] + NOISE_STD * meas_rng.standard_normal(2).astype(np.float32) if vis[t] else None
        est.append(tracker.step(z))
    return np.array(est)


def check_regression(cache, cfg, n_steps):
    pos, vis = _trajectory(0)
    q0 = cfg["filter"]["baseline_process_noise"]
    out = {}
    for name, model, q in (("PF+CFM", cache.cfm(), q0), ("PF+DMMN", cache.model("PF+DMMN", 0), q0),
                           ("PF+NF", cache.model("PF+NF", 0), cfg["filter"]["nf_process_noise"])):
        old = _run(BootstrapPFTracker(model, n_particles=500, process_noise=q), pos, vis, 0, n_steps)
        new = _run(VariantPFTracker(MotionSampler(model, q), 500), pos, vis, 0, n_steps)
        out[name] = float(np.abs(old - new).max())
    occluded = int((~vis[int(np.argmax(vis)) + 1:][:n_steps]).sum())
    return {"max_abs_estimate_diff": out, "steps": n_steps, "occluded_steps_covered": occluded,
            "pass": all(v == 0.0 for v in out.values())}


@torch.no_grad()
def _contexts(n=500, seed=0):
    pos, vis = _trajectory(3)
    rng = np.random.default_rng(seed)
    idx = rng.integers(5, len(pos), n)
    hist = np.stack([pos[i - 5:i] for i in idx]) + 0.03 * rng.standard_normal((n, 5, 2)).astype(np.float32)
    return hist.astype(np.float32)


@torch.no_grad()
def check_generator(cache):
    cfm = cache.cfm()
    hist = _contexts()
    c = torch.tensor(hist.reshape(len(hist), -1))
    torch.manual_seed(1)
    v_ref = cfm.propose(c.numpy())
    torch.manual_seed(1)
    emb = cfm.network.encode_context(c)
    anchor = cfm._finite_difference_velocity(c)
    v = cfm_generate_from_base(cfm, emb, anchor, torch.randn_like(anchor))
    d = float(np.abs(v - v_ref).max())
    return {"max_abs_diff": d, "pass": d == 0.0}


@torch.no_grad()
def check_m3_identity(cache, q):
    cfm = cache.cfm()
    hist = _contexts().transpose(1, 0, 2).copy()           # (H, N, 2)
    ctx = None
    z = hist[-1, 0] + np.float32(0.02)
    np.random.seed(5); torch.manual_seed(5)
    p_prior, _ = MotionSampler(cfm, q).predict(hist, ctx, z)
    np.random.seed(5); torch.manual_seed(5)
    p_id, lc = LatentProposalSampler(cfm, IdentityProposal(), q).predict(hist, ctx, z)
    np.random.seed(5); torch.manual_seed(5)
    p_none, lc_none = LatentProposalSampler(cfm, IdentityProposal(), q).predict(hist, ctx, None)
    ident = {"max_abs_log_correction": float(np.abs(lc).max()),
             "max_abs_position_diff": float(np.abs(p_id - p_prior).max())}
    ident["pass"] = ident["max_abs_log_correction"] == 0.0 and ident["max_abs_position_diff"] == 0.0
    no_obs = {"max_abs_position_diff": float(np.abs(p_none - p_prior).max()), "log_correction": lc_none}
    no_obs["pass"] = no_obs["max_abs_position_diff"] == 0.0 and lc_none is None
    # Tracker level: identity proposal vs PF+CFM over a run (log-space vs linear weights only).
    pos, vis = _trajectory(0)
    a = _run(VariantPFTracker(MotionSampler(cfm, q), 500), pos, vis, 0, 300)
    b = _run(VariantPFTracker(LatentProposalSampler(cfm, IdentityProposal(), q), 500), pos, vis, 0, 300)
    ident["tracker_max_abs_estimate_diff_300_steps"] = float(np.abs(a - b).max())
    # M3-joint with q = N(0, I_4): the weight correction must vanish (eta draws come from torch instead).
    joint = IdentityProposal()
    joint.dim = 4
    def dist4(cfm_, emb, anchor, x_prev, z_):
        mean = torch.zeros(len(anchor), 4)
        chol = torch.eye(4).expand(len(anchor), 4, 4)
        return mean, chol, lambda u: mvn_tril_log_prob(u, mean, chol)
    joint.distribution = dist4
    _, lc4 = LatentProposalSampler(cfm, joint, q).predict(hist, ctx, z)
    ident["joint_max_abs_log_correction"] = float(np.abs(lc4).max())
    ident["pass"] = ident["pass"] and ident["joint_max_abs_log_correction"] < 1e-5
    return ident, no_obs


def check_m3_analytic(n=400_000, seed=0):
    """Latent IS with the code path's densities on x = xi (+ eta), z = x + nu, independent coordinates."""
    g = torch.Generator().manual_seed(seed)
    results, ok = [], True
    for eta_var, R, z in ((0.0, 0.4, 1.2), (0.25, 0.4, 1.2), (0.25, 0.1, -0.7)):
        prior_var = 1.0 + eta_var
        m_true, v_true = prior_var / (prior_var + R) * z, prior_var * R / (prior_var + R)
        # Nondegenerate proposals: prior, shifted/narrowed towards z, wide and offset the wrong way.
        for b, s in ((0.0, 1.0), (z / 3, 0.8), (-0.4 * np.sign(z), 1.5), (0.6 * z, 0.5)):
            mean = torch.full((n, 2), b, dtype=torch.float64)
            chol = torch.zeros(n, 2, 2, dtype=torch.float64)
            chol[:, 0, 0] = chol[:, 1, 1] = s
            xi = mean + s * torch.randn(n, 2, generator=g, dtype=torch.float64)
            eta = np.sqrt(eta_var) * torch.randn(n, 2, generator=g, dtype=torch.float64)
            x = xi + eta
            logw = (-0.5 * ((z - x) ** 2).sum(-1) / R) + std_normal_log_prob(xi) - mvn_tril_log_prob(xi, mean, chol)
            w = torch.softmax(logw, 0)
            m_est = float((w[:, None] * x).sum(0).mean())
            v_est = float((w[:, None] * (x - m_est) ** 2).sum(0).mean())
            err = max(abs(m_est - m_true), abs(v_est - v_true))
            ess = float(1.0 / (w ** 2).sum()) / n
            ok &= err < 0.01
            results.append({"eta_var": eta_var, "R": R, "z": z, "proposal": [b, s], "mean": m_est, "var": v_est,
                            "true_mean": m_true, "true_var": v_true, "abs_err": err, "ess_frac": ess})
        if eta_var == 0.0:
            continue
        # Joint proposal on u = (xi, eta / sd) per coordinate, prior N(0, I_4) (the M3-joint weight).
        sd = np.sqrt(eta_var)
        for b, s in ((0.0, 1.0), (z / 4, 0.7)):
            mean = torch.full((n, 4), b, dtype=torch.float64)
            chol = s * torch.eye(4, dtype=torch.float64).expand(n, 4, 4)
            u = mean + s * torch.randn(n, 4, generator=g, dtype=torch.float64)
            x = u[:, :2] + sd * u[:, 2:]
            logw = (-0.5 * ((z - x) ** 2).sum(-1) / R) + std_normal_log_prob(u) - mvn_tril_log_prob(u, mean, chol)
            w = torch.softmax(logw, 0)
            m_est = float((w[:, None] * x).sum(0).mean())
            v_est = float((w[:, None] * (x - m_est) ** 2).sum(0).mean())
            err = max(abs(m_est - m_true), abs(v_est - v_true))
            ok &= err < 0.01
            results.append({"eta_var": eta_var, "R": R, "z": z, "proposal": ["joint", b, s], "mean": m_est,
                            "var": v_est, "true_mean": m_true, "true_var": v_true, "abs_err": err,
                            "ess_frac": float(1.0 / (w ** 2).sum()) / n})
    return {"cases": results, "tolerance": 0.01, "pass": bool(ok)}


@torch.no_grad()
def check_m2_covariance(cache, q, K=20000):
    cfm = cache.cfm()
    h = _contexts(1)[0]
    hist = np.repeat(h[:, None, :], K, axis=1).astype(np.float32)
    torch.manual_seed(0); np.random.seed(0)
    v = cfm.propose(flat_context(hist))
    torch.manual_seed(1); np.random.seed(1)
    x, _ = MotionSampler(cfm, q).predict(hist, None, None)
    emp = np.cov((x - hist[-1]).T)
    pred = cfm.dt ** 2 * np.cov(v.T) + q ** 2 * np.eye(2)
    rel = float(np.abs(emp - pred).max() / np.abs(pred).max())
    return {"q": q, "empirical": emp.tolist(), "dt2_cov_v_plus_Q": pred.tolist(), "max_rel_err": rel,
            "pass": rel < 0.05}


class _Spy:
    def __init__(self, inner):
        self.inner, self.block_size, self.dt, self.seen = inner, inner.block_size, inner.dt, []

    def predict(self, particles, ctx, z):
        self.seen.append(ctx)
        return self.inner.predict(particles, ctx, z)


def check_m1_causal(cache, q):
    cfm = cache.cfm()
    pos, vis = _trajectory(0)
    spy = _Spy(MotionSampler(cfm, q))
    tr = VariantPFTracker(spy, 500)
    np.random.seed(0); torch.manual_seed(0)
    first = int(np.argmax(vis))
    tr.reset(pos[first])
    ok = True
    # Same state, step with and without z_k: the context handed to the sampler must match.
    for t in range(first + 1, first + 400):
        state = tr.get_state()
        rs, ts = np.random.get_state(), torch.get_rng_state()
        tr.step(pos[t]); c_with = spy.seen[-1]
        tr.set_state(state); np.random.set_state(rs); torch.set_rng_state(ts)
        tr.step(None); c_without = spy.seen[-1]
        ok &= (np.array_equal(c_with.mask, c_without.mask) and c_with.d == c_without.d
               and np.array_equal(c_with.U, c_without.U))
        tr.set_state(state); np.random.set_state(rs); torch.set_rng_state(ts)
        tr.step(pos[t] if vis[t] else None)
    # d counts occluded steps; mask[-1] says whether z_{k-1} was used.
    ages = [c.d for c in spy.seen[2::3]]
    return {"pass": bool(ok), "max_d_seen": float(max(ages)),
            "note": "ctx built before the update; resampling reindexes whole histories, mask/d/U are cloud-level"}


@torch.no_grad()
def check_solver(cache, n=2000):
    cfm = cache.cfm()
    hist = _contexts(n, seed=3)
    c = torch.tensor(hist.reshape(n, -1))
    emb = cfm.network.encode_context(c)
    anchor = cfm._finite_difference_velocity(c)
    xi = torch.randn(n, 2, generator=torch.Generator().manual_seed(0))
    old = cfm.N_FE
    out, ref = {}, None
    try:
        for K in (40, 20, 10):
            cfm.N_FE = K
            t0 = time.perf_counter()
            v = cfm_generate_from_base(cfm, emb[:500], anchor[:500], xi[:500])
            ms = 1000 * (time.perf_counter() - t0)
            v = cfm_generate_from_base(cfm, emb, anchor, xi)
            ref = v if ref is None else ref
            out[K] = {"mean_abs_dv_vs_40": float(np.abs(v - ref).sum(1).mean()),
                      "p95_abs_dv_vs_40": float(np.percentile(np.abs(v - ref).sum(1), 95)),
                      "ms_per_500": ms}
    finally:
        cfm.N_FE = old
    return out


@torch.no_grad()
def check_m3_ess(cache, cfg, seeds, holdout_ctx, n_ctx=200, K=500):
    cfm = cache.cfm()
    q = cfg["m3"]["train_process_noise"]
    R = cfg["filter"]["measurement_sigma"]
    out = {}
    rng = np.random.default_rng(0)
    idx = rng.choice(len(holdout_ctx), n_ctx, replace=False)
    for (ck, name), seed in [(cn, sd) for cn in (("m3", "M3"), ("m3_joint", "M3-joint")) for sd in seeds]:
        path = ckpt_path(cache.out_dir, ck, seed)
        if not os.path.exists(path):
            continue
        prop = cache.model(name, seed)
        eb, ep, bad = [], [], 0
        for i in idx:
            c = torch.tensor(np.repeat(holdout_ctx[i:i + 1], K, 0))
            emb = cfm.network.encode_context(c)
            anchor = cfm._finite_difference_velocity(c)
            x_prev = c[:, -2:]
            v_true = cfm_generate_from_base(cfm, emb[:1], anchor[:1], torch.randn(1, 2))
            z = x_prev[0].numpy() + cfm.dt * v_true[0] + q * np.random.randn(2) + R * np.random.randn(2)
            z_t = torch.tensor(z, dtype=torch.float32).expand_as(x_prev)
            v_b = cfm_generate_from_base(cfm, emb, anchor, torch.randn_like(anchor))
            x_b = x_prev.numpy() + cfm.dt * v_b + q * np.random.randn(K, 2)
            lw = -0.5 * ((x_b - z) ** 2).sum(1) / R ** 2
            w = np.exp(lw - lw.max()); w /= w.sum()
            eb.append(1 / (w ** 2).sum() / K)
            mean, chol, logq = prop.distribution(cfm, emb, anchor, x_prev, z_t)
            u = mean + torch.einsum("bij,bj->bi", chol, torch.randn_like(mean))
            lc = (std_normal_log_prob(u) - logq(u)).double().numpy()
            v_p = cfm_generate_from_base(cfm, emb, anchor, u[:, :2])
            eta = q * (u[:, 2:].numpy() if prop.dim == 4 else np.random.randn(K, 2))
            x_p = x_prev.numpy() + cfm.dt * v_p + eta
            lw = -0.5 * ((x_p - z) ** 2).sum(1) / R ** 2 + lc
            bad += int((~np.isfinite(lw)).sum())
            w = np.exp(lw - lw.max()); w /= w.sum()
            ep.append(1 / (w ** 2).sum() / K)
        out[f"{name}_seed{seed}"] = {"ess_bootstrap_mean": float(np.mean(eb)), "ess_proposal_mean": float(np.mean(ep)),
                              "ess_bootstrap_median": float(np.median(eb)),
                              "ess_proposal_median": float(np.median(ep)), "nonfinite_logw": bad}
    return out


def check_m4(cache, cfg, data, holdout, seeds):
    from mml_research.motion.cfm_variants.residual_model import ResidualCFMMotionModel
    out = {}
    c_flat = data["hist"].reshape(len(data["hist"]), -1)
    sub = np.random.default_rng(0).choice(len(c_flat), min(50000, len(c_flat)), replace=False)
    mu = cache.dmmn_mean()(c_flat[sub])
    v, ho = data["v"][sub], holdout[sub]
    for seed in seeds:
        path = ckpt_path(cache.out_dir, "m4", seed)
        if not os.path.exists(path):
            continue
        m = cache.model("M4", seed)
        r = (v - mu) / m.S
        err = float(np.abs(mu + m.S * r - v).max())
        out[f"seed{seed}"] = {"S": m.S.tolist(), "reconstruction_max_abs_err": err, "pass": err < 1e-5}
    e = v - mu
    out["residuals"] = {"train_mean": e[~ho].mean(0).tolist(), "train_std": e[~ho].std(0).tolist(),
                        "holdout_mean": e[ho].mean(0).tolist(), "holdout_std": e[ho].std(0).tolist()}
    return out


def one_step(cache, cfg, data, holdout, seeds, n=4000):
    """Bias / RMSE / spread of one-step velocity samples vs the true velocity on held-out rollout contexts."""
    rng = np.random.default_rng(1)
    ho_idx = np.flatnonzero(holdout)
    res = {}
    torch.manual_seed(0)
    for occluded in (False, True):
        pool = ho_idx[data["vis"][ho_idx] != occluded]
        if not len(pool):
            continue
        idx = rng.choice(pool, min(n, len(pool)), replace=False)
        hist, v = data["hist"][idx], data["v"][idx]
        c = hist.reshape(len(idx), -1)
        models = {"PF+CFM": cache.cfm(), "PF+NF": cache.model("PF+NF", 0), "PF+DMMN": cache.model("PF+DMMN", 0)}
        if os.path.exists(ckpt_path(cache.out_dir, "dmmn_gauss", 0)):
            models["DMMN+Gauss"] = cache.model("DMMN+Gauss", 0)
        for name in ("M4", "M1-control", "M1-mask", "M1-time", "M1-full"):
            from mml_research.motion.cfm_variants.methods import METHODS
            if os.path.exists(ckpt_path(cache.out_dir, METHODS[name][2], seeds[0])):
                models[name] = cache.model(name, seeds[0])
        for name, m in models.items():
            draws = []
            for _ in range(8):
                if getattr(m, "uses_context", False):
                    draws.append(m.sample(hist, data["mask"][idx].astype(np.float32), data["d"][idx], data["U"][idx]))
                else:
                    draws.append(m.propose(c))
            draws = np.stack(draws)                      # (8, n, 2)
            resid = draws - v[None]
            res.setdefault(name, {})["occluded" if occluded else "visible"] = {
                "bias": resid.mean((0, 1)).tolist(),
                "rmse_sample": float(np.sqrt((resid ** 2).sum(-1).mean())),
                "rmse_sample_mean": float(np.sqrt(((draws.mean(0) - v) ** 2).sum(-1).mean())),
                "spread": float(np.sqrt(draws.var(0).sum(-1).mean())),
            }
    return res


def main():
    p = argparse.ArgumentParser(description="Correctness checks for the CFM variants")
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--models-dir", dest="models_dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--regression_steps", type=int, default=600)
    args = p.parse_args()
    torch.set_num_threads(4)
    cfg = load_config(args.config)
    cache = ModelCache(cfg, args.models_dir, args.out_dir)
    q0 = cfg["filter"]["baseline_process_noise"]
    seeds = cfg["training"]["seeds"]
    report = {}

    def step(name, fn):
        t0 = time.time()
        report[name] = fn()
        ok = report[name].get("pass") if isinstance(report[name], dict) else None
        tag = "" if ok is None else ("PASS" if ok else "FAIL")
        print(f"[{tag or 'info':>4}] {name} ({time.time() - t0:.0f}s): "
              f"{ {k: v for k, v in report[name].items() if k != 'cases'} }", flush=True)

    step("regression", lambda: check_regression(cache, cfg, args.regression_steps))
    step("generator", lambda: check_generator(cache))
    ident, no_obs = check_m3_identity(cache, q0)
    report["m3_identity"], report["m3_no_obs"] = ident, no_obs
    print(f"[{'PASS' if ident['pass'] else 'FAIL'}] m3_identity: {ident}")
    print(f"[{'PASS' if no_obs['pass'] else 'FAIL'}] m3_no_obs: {no_obs}")
    step("m3_analytic", check_m3_analytic)
    for c in report["m3_analytic"]["cases"]:
        print(f"         eta_var={c['eta_var']} R={c['R']} z={c['z']} q={c['proposal']}: "
              f"mean {c['mean']:.4f}/{c['true_mean']:.4f} var {c['var']:.4f}/{c['true_var']:.4f} ESS {c['ess_frac']:.2f}")
    step("m2_covariance", lambda: check_m2_covariance(cache, q0))
    step("m1_causal", lambda: check_m1_causal(cache, q0))
    step("solver", lambda: check_solver(cache))

    rpath = os.path.join(args.out_dir, "rollouts.npz")
    if os.path.exists(rpath):
        data = dict(np.load(rpath))
        holdout = np.isin(data["run"], list(TRAIN_HOLDOUT_RUNS))
        if not holdout.any():                       # reduced (smoke-test) rollouts
            holdout = data["run"] == data["run"].max()
        ho_vis = np.flatnonzero(holdout & data["vis"].astype(bool))
        ctx = data["hist"][ho_vis].reshape(len(ho_vis), -1).astype(np.float32)
        step("m3_ess", lambda: check_m3_ess(cache, cfg, seeds, ctx))
        step("m4", lambda: check_m4(cache, cfg, data, holdout, seeds))
        step("one_step", lambda: one_step(cache, cfg, data, holdout, seeds))
        report["m4"]["pass"] = all(v.get("pass", True) for v in report["m4"].values() if isinstance(v, dict))

    hard = ["regression", "generator", "m3_identity", "m3_no_obs", "m3_analytic", "m2_covariance", "m1_causal"]
    if "m4" in report:
        hard.append("m4")
    report["all_hard_checks_pass"] = all(report[k]["pass"] for k in hard)
    write_json(os.path.join(args.out_dir, "verify.json"), report)
    print(f"\nall hard checks pass: {report['all_hard_checks_pass']}  -> {os.path.join(args.out_dir, 'verify.json')}")


if __name__ == "__main__":
    main()
