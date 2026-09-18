#!/usr/bin/env python3
"""
train_diffpf.py
================
Train the adapted DiffPF posterior sampler on the same 8-node Markov-graph
simulation used for the CFM / Diffusion / GMM motion models.

Unlike the motion-model training CSVs (visible steps only), DiffPF needs
FULL trajectories including occluded steps, because its central mechanism
is posterior sampling conditioned on (measurement OR null token, predicted
particles) — the occluded steps are exactly where the null-token branch is
trained.

Teacher forcing (mirrors the DiffPF training recipe, without the released
code's cross-batch particles.pt hack):
  The "predicted particles" conditioning at step t is approximated by a
  constant-velocity extrapolation from the last VISIBLE ground-truth
  position, with noise that grows with the occlusion gap length — i.e. what
  a reasonable filter prior would look like at that step.

Usage (after installing the project):
    python -m mml_research.motion.train_diffpf --epochs 800 --T 10 --n_cond_particles 20

Output:
    ../../models/diffpf_tracker.pth
    ../../models/diffpf_training_loss.png
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


from mml_core.tracking.diffpf_tracker import DiffPFTracker
from mml_research.motion.generate_synthetic_data import (
    NODE_POS, TRANS_MATRIX, DT, SPEED, NOISE_STD, in_occlusion,
)


def simulate_full_trajectory(start_node: int, n_steps: int, rng: np.random.Generator):
    """
    Walk the Markov graph, returning ALL ground-truth positions plus a
    visibility flag per step (True = outside occlusion zones).
    """
    pos = NODE_POS[start_node].copy()
    node = start_node

    positions = np.zeros((n_steps, 2), dtype=np.float32)
    visible = np.zeros(n_steps, dtype=bool)

    for i in range(n_steps):
        probs = TRANS_MATRIX[node]
        nxt = rng.choice(8, p=probs) if probs.sum() > 0 else node
        d = NODE_POS[nxt] - pos
        dist = np.linalg.norm(d)
        if dist < SPEED:
            node = nxt
            pos = NODE_POS[node].copy()
        else:
            pos += d / dist * SPEED

        positions[i] = pos
        visible[i] = not in_occlusion(pos)

    return positions, visible


def build_teacher_forced_dataset(
    tracker: DiffPFTracker,
    n_steps_per_node: int = 2000,
    drift_rate: float = 0.01,
    base_spread: float = 0.03,
    seed: int = 42,
):
    """
    Build (meas_cond, particle_cond, targets) training arrays in normalized
    coordinates, covering both visible and occluded steps.

    drift_rate:
        Extra particle spread per occluded step — simulates prior
        uncertainty growth while the target is hidden.
    base_spread:
        Particle spread around the anchor when the target is visible.
    """
    rng = np.random.default_rng(seed)
    K = tracker.n_cond_particles

    meas_list, part_list, target_list = [], [], []
    n_occluded = 0

    for start_node in range(8):
        positions, visible = simulate_full_trajectory(start_node, n_steps_per_node, rng)

        last_vis_pos = None    # last visible ground-truth position
        last_vis_vel = np.zeros(2, dtype=np.float32)
        prev_vis_pos = None
        gap = 0                # steps since last visible measurement

        for t in range(1, len(positions)):
            gt = positions[t]

            if visible[t - 1]:
                if prev_vis_pos is not None:
                    last_vis_vel = (positions[t - 1] - prev_vis_pos) / DT
                prev_vis_pos = positions[t - 1].copy()
                last_vis_pos = positions[t - 1].copy()
                gap = 0
            else:
                gap += 1

            if last_vis_pos is None:
                continue   # no anchor yet — filter would not be initialized

            # Filter-prior approximation: CV extrapolation from the last
            # visible position, spread growing with the occlusion gap.
            anchor = last_vis_pos + last_vis_vel * DT * (gap + 1)
            spread = base_spread + drift_rate * gap
            particles = anchor[None, :] + spread * rng.standard_normal((K, 2)).astype(np.float32)

            if visible[t]:
                noisy = gt + rng.standard_normal(2).astype(np.float32) * NOISE_STD
                m_norm = tracker._normalize(noisy)
                meas_vec = np.array([m_norm[0], m_norm[1], 1.0], dtype=np.float32)
            else:
                meas_vec = np.zeros(3, dtype=np.float32)
                n_occluded += 1

            meas_list.append(meas_vec)
            part_list.append(tracker._normalize(particles).reshape(-1))
            target_list.append(tracker._normalize(gt))

    meas_cond = np.stack(meas_list)
    particle_cond = np.stack(part_list)
    targets = np.stack(target_list)

    print(f"[build_teacher_forced_dataset] {len(targets)} examples "
          f"({n_occluded} occluded = {100.0 * n_occluded / len(targets):.1f}%)")
    return meas_cond, particle_cond, targets


def main():
    parser = argparse.ArgumentParser(description="Train the adapted DiffPF tracker")
    parser.add_argument("--epochs",           type=int,   default=800)
    parser.add_argument("--batch",            type=int,   default=512)
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--T",                type=int,   default=10,
                        help="DDPM steps (10 matches the DiffPF nav01 config)")
    parser.add_argument("--n_particles",      type=int,   default=100,
                        help="Posterior particle set size at inference")
    parser.add_argument("--n_cond_particles", type=int,   default=20,
                        help="Particles in the conditioning vector (20 matches DiffPF)")
    parser.add_argument("--hidden_dim",       type=int,   default=128)
    parser.add_argument("--steps_per_node",   type=int,   default=2000)
    parser.add_argument("--models-dir", default=None, help="Override the models dir")
    args = parser.parse_args()

    this_dir = str(motion_resource_dir() / "mml_guidance/mml_network")
    models_dir = args.models_dir or os.path.join(this_dir, '..', '..', 'models')
    save_path = os.path.normpath(os.path.join(models_dir, "diffpf_tracker.pth"))
    plot_path = os.path.normpath(os.path.join(models_dir, "diffpf_training_loss.png"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("=" * 60)
    print("Training adapted DiffPF tracker")
    print("=" * 60)
    print(f"T (DDPM steps)   : {args.T}")
    print(f"n_particles      : {args.n_particles}")
    print(f"n_cond_particles : {args.n_cond_particles}")
    print(f"hidden_dim       : {args.hidden_dim}")
    print(f"epochs           : {args.epochs}  batch={args.batch}  lr={args.lr}")

    tracker = DiffPFTracker(
        n_particles=args.n_particles,
        n_cond_particles=args.n_cond_particles,
        hidden_dim=args.hidden_dim,
        T=args.T,
        dt=DT,
    )
    n_params = sum(p.numel() for p in tracker.network.parameters())
    print(f"Parameters: {n_params:,}")

    print("\nBuilding teacher-forced dataset (full trajectories incl. occlusion)...")
    meas_cond, particle_cond, targets = build_teacher_forced_dataset(
        tracker, n_steps_per_node=args.steps_per_node,
    )

    print(f"\nStarting training for {args.epochs} epochs...")
    losses = tracker.train(
        meas_cond, particle_cond, targets,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        print_every=max(1, args.epochs // 10),
    )

    tracker.save(save_path)

    plt.figure(figsize=(9, 3))
    plt.plot(losses, lw=1.5, color="purple")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Noise-prediction MSE loss")
    plt.title(f"DiffPF Training Loss  (T={args.T}, "
              f"n_cond={args.n_cond_particles}, hidden={args.hidden_dim})\n"
              f"final={losses[-1]:.5f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"Loss curve → {plot_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print(f'  model_path = r"{os.path.abspath(save_path)}"')
    print("=" * 60)


if __name__ == "__main__":
    main()
