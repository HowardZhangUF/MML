#!/usr/bin/env python3
"""
train_cfm_proposal.py
=====================
Train the two learned components that turn PF+CFM from a bootstrap filter
into a proposal filter (mml_research.motion.proposal_pf):

  1. CFMMeasurementModel p_psi(z | x) on (true position, measurement) pairs
     from the simulated sensor — road positions plus positions drawn
     uniformly over the arena, so particles off the road stay in-distribution.
  2. CFMProposalModel q_phi(v | C, z) on joint samples of the state-space
     model (frozen CFM motion model + process noise + learned measurement
     model). Its histories C are the motion model's training windows plus
     particle histories logged from PF runs at visible steps — the inputs the
     proposal sees inside the filter, including histories right after an
     occlusion.

Every simulated trajectory here uses run indices >= --first_run (default
200): the benchmark uses runs 0..99 and process-noise selection 100..119.

Usage (after installing the project):
    python -m mml_research.motion.train_cfm_proposal --models-dir <dir> --data-dir <dir>
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from mml_core.config import motion_resource_dir
from mml_core.motion.cfm_motion_model import CFMMotionModel
from mml_core.motion.cfm_proposal_model import CFMProposalModel
from mml_core.motion.utils import load_training_csv
from mml_core.sensing.cfm_measurement_model import CFMMeasurementModel
from mml_research.motion.benchmark_diffpf_vs_cfm import BootstrapPFTracker
from mml_research.motion.generate_synthetic_data import DT, NODE_POS, NOISE_STD
from mml_research.motion.train_diffpf import simulate_full_trajectory

device = "cuda" if torch.cuda.is_available() else "cpu"
CSV_NAME = "converted_noisy_5_v2_velocities_training_data.csv"


def measurement_pairs(first_run: int, n_runs: int, n_steps: int, n_uniform: int, bounds, seed: int):
    """(x, z): road positions of simulated runs plus uniform arena positions, z = x + sensor noise."""
    rng = np.random.default_rng(seed)
    xs = [simulate_full_trajectory(run % len(NODE_POS), n_steps, np.random.default_rng(1000 + run))[0]
          for run in range(first_run, first_run + n_runs)]
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    xs.append(rng.uniform(lo, hi, size=(n_uniform, 2)).astype(np.float32))
    x = np.concatenate(xs).astype(np.float32)
    z = x + NOISE_STD * rng.standard_normal(x.shape).astype(np.float32)
    return x, z


def pf_histories(cfm, runs, n_steps, process_noises, per_step, every, tracker_factory=None):
    """Particle histories (N, block_size * 2) at visible steps of PF runs, before each update."""
    out = []
    for k, run in enumerate(runs):
        q = process_noises[k % len(process_noises)]
        rng = np.random.default_rng(1000 + run)
        positions, visible = simulate_full_trajectory(run % len(NODE_POS), n_steps, rng)
        meas_rng = np.random.default_rng(5000 + run)
        np.random.seed(9000 + run)
        torch.manual_seed(9000 + run)
        tracker = (tracker_factory(q) if tracker_factory else
                   BootstrapPFTracker(cfm, n_particles=500, process_noise=q))
        first = int(np.argmax(visible))
        tracker.reset(positions[first] + NOISE_STD * meas_rng.standard_normal(2).astype(np.float32))
        for t in range(first + 1, len(positions)):
            meas = (positions[t] + NOISE_STD * meas_rng.standard_normal(2).astype(np.float32)
                    if visible[t] else None)
            if meas is not None and t % every == 0:
                pick = np.random.choice(tracker.N, per_step, replace=False, p=tracker.weights)
                out.append(tracker.particles[:, pick, :].transpose(1, 0, 2).reshape(per_step, -1).copy())
            tracker.step(meas)
        print(f"  history run {run} (q={q}) done, {sum(len(o) for o in out)} histories", flush=True)
    return np.concatenate(out)


@torch.no_grad()
def proposal_diagnostics(cfm, meas, prop, contexts, n_ctx=200, K=500, q=0.01, seed=0):
    """
    Single-step importance-sampling efficiency on held-out model draws:
    ESS/K of K particles sharing a context, for the bootstrap proposal
    (weights p(z|x)) and the learned proposal (weights p f / q).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(len(contexts), (n_ctx,), generator=g)
    ess_boot, ess_prop = [], []
    for i in idx.tolist():
        c = torch.tensor(contexts[i:i + 1], device=device).repeat(K, 1)
        emb = cfm.network.encode_context(c)
        anchor = cfm._finite_difference_velocity(c)
        x_prev = c[:, -2:]
        # one "true" draw from the model gives the measurement
        v_true, _ = cfm.sample_with_log_prob(c[:1], emb[:1])
        z = meas.sample(x_prev[:1] + DT * v_true + q * torch.randn_like(v_true))[0]

        v_b, _ = cfm.sample_with_log_prob(c, emb)
        x_b = x_prev + DT * v_b + q * torch.randn_like(v_b)
        lw = meas.log_prob(z, x_b)
        w = torch.softmax(lw, 0)
        ess_boot.append(float(1.0 / (w ** 2).sum()) / K)

        eta = q * torch.randn_like(x_prev)
        m = (z - x_prev - eta) / DT
        v_p, log_q = prop.sample_with_log_prob(prop.condition(emb, anchor, m, x_prev), m)
        log_f, _ = cfm.log_prob(c, v_p, emb)
        lw = meas.log_prob(z, x_prev + DT * v_p + eta) + log_f - log_q
        w = torch.softmax(lw, 0)
        ess_prop.append(float(1.0 / (w ** 2).sum()) / K)
    return np.array(ess_boot), np.array(ess_prop)


@torch.no_grad()
def measurement_diagnostics(meas, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.uniform([-2, -1.5], [2.6, 1.5], size=(n, 2)), dtype=torch.float32, device=device)
    z_s = meas.sample(x)
    learned_std = (z_s - x).std(0).cpu().numpy()
    z = x + NOISE_STD * torch.randn_like(x)
    lp = meas.log_prob(z, x)
    true_lp = (-0.5 * ((z - x) ** 2).sum(-1) / NOISE_STD ** 2
               - 2 * np.log(NOISE_STD) - np.log(2 * np.pi))
    # log-density along a ray out to 25 sensor sigmas: must keep decreasing
    d = torch.linspace(0, 25 * NOISE_STD, 200, device=device)
    ray = torch.stack([d, 0.6 * d], -1)
    x0 = torch.tensor([[0.5, 0.2]], device=device).repeat(len(d), 1)
    lp_ray = meas.log_prob(x0 + ray, x0).cpu().numpy()
    return {
        "learned_residual_std": learned_std.tolist(),
        "mean_log_prob_learned": float(lp.mean()),
        "mean_log_prob_true_gaussian": float(true_lp.mean()),
        "ray_monotone": bool(np.all(np.diff(lp_ray) < 0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the CFM measurement model and proposal")
    root = motion_resource_dir() / "exit_benchmark"
    parser.add_argument("--models-dir", default=str(root / "models"))
    parser.add_argument("--data-dir", default=str(root / "data"))
    parser.add_argument("--cfm_ckpt", default="cfm_motion_model_anchor1.0.pth")
    parser.add_argument("--first_run", type=int, default=200)
    parser.add_argument("--meas_runs", type=int, default=16)
    parser.add_argument("--hist_runs", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--hist_noise", type=float, nargs="+", default=[0.01, 0.045],
                        help="Process noise of the PF runs whose histories are logged")
    parser.add_argument("--meas_epochs", type=int, default=300)
    parser.add_argument("--prop_epochs", type=int, default=150)
    parser.add_argument("--prop_hidden", type=int, default=128)
    parser.add_argument("--source_sigma", type=float, default=1.0)
    parser.add_argument("--train_noise", type=float, default=0.01,
                        help="Process noise of the model the proposal is trained on")
    parser.add_argument("--histories", default=None,
                        help="Reuse a saved .npy of PF histories instead of collecting them")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    os.makedirs(args.models_dir, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    cfm = CFMMotionModel()
    cfm.load(os.path.join(args.models_dir, args.cfm_ckpt))
    for p in cfm.network.parameters():
        p.requires_grad_(False)

    # 1. measurement model
    meas = CFMMeasurementModel()
    x, z = measurement_pairs(args.first_run, args.meas_runs, args.n_steps,
                             n_uniform=args.meas_runs * args.n_steps, bounds=meas.bounds, seed=1)
    print(f"Measurement pairs: {len(x)}")
    t0 = time.time()
    meas_losses = meas.train(x, z, epochs=args.meas_epochs)
    print(f"  trained in {time.time() - t0:.0f}s")
    meas.save(os.path.join(args.models_dir, f"cfm_measurement_model{suffix}.pth"))
    md = measurement_diagnostics(meas)
    print(f"  measurement diagnostics: {md}  (sensor std {NOISE_STD})")

    # 2. proposal contexts: motion-model training windows + PF particle histories
    X_csv, _ = load_training_csv(os.path.join(args.data_dir, CSV_NAME))
    if args.histories:
        H = np.load(args.histories)
    else:
        runs = list(range(args.first_run, args.first_run + args.hist_runs))
        H = pf_histories(cfm, runs, args.n_steps, args.hist_noise, per_step=32, every=2)
        np.save(os.path.join(args.data_dir, f"pf_histories{suffix}.npy"), H)
    contexts = np.concatenate([X_csv, H]).astype(np.float32)
    rng = np.random.default_rng(0)
    rng.shuffle(contexts)
    n_val = 2000
    val, train = contexts[:n_val], contexts[n_val:]
    print(f"Proposal contexts: {len(X_csv)} CSV windows + {len(H)} PF histories")

    prop = CFMProposalModel(n_embed=cfm.n_embed, hidden=args.prop_hidden,
                            source_sigma=args.source_sigma, dt=cfm.dt)
    t0 = time.time()
    prop_losses = prop.train(cfm, meas, train, epochs=args.prop_epochs, process_noise=args.train_noise)
    print(f"  trained in {time.time() - t0:.0f}s")
    prop.save(os.path.join(args.models_dir, f"cfm_proposal_model{suffix}.pth"))

    ess_b, ess_p = proposal_diagnostics(cfm, meas, prop, val, q=args.train_noise)
    print(f"  one-step ESS/N on {len(ess_b)} held-out model draws: "
          f"bootstrap {ess_b.mean():.3f} (median {np.median(ess_b):.3f}), "
          f"proposal {ess_p.mean():.3f} (median {np.median(ess_p):.3f})")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].plot(meas_losses); axes[0].set_title("measurement CFM loss"); axes[0].set_yscale("log")
    axes[1].plot(prop_losses); axes[1].set_title("proposal CFM loss"); axes[1].set_yscale("log")
    axes[2].hist([ess_b, ess_p], bins=20, label=["bootstrap", "learned proposal"])
    axes[2].set_title("one-step ESS / N (held-out)"); axes[2].legend()
    for ax in axes[:2]:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.models_dir, f"cfm_proposal_training{suffix}.png"), dpi=120)


if __name__ == "__main__":
    main()
