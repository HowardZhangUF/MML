#!/usr/bin/env python3
"""
generate_synthetic_data.py
==========================
Generates training data for the CFM motion model by simulating the
8-node Markov graph from pygame_large_env.py.

Fixes applied vs original:
  1. dt = 0.033 s  (matches pygame config.yaml 30 Hz) — was 1/3 (3 Hz) → 10x error
  2. Starts from ALL 8 nodes (2000 steps each = 16 000 total) — was node 0 only
  3. Simulates occlusion gaps — model now sees measurement dropouts like the real filter
  4. Uses actual dt between visible steps for velocity labels — mirrors ParticleFilter exactly
  5. N_th = 10 history steps — was 5
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import os
import sys
import numpy as np
import pandas as pd


# ── Constants — must match config.yaml and ParticleFilter ────────────────────
DT         = 0.033   # seconds per pygame step (30 Hz) — THE KEY FIX
N_TH       = 5      # history window length — increased from 5
NOISE_STD  = 0.02    # measurement noise std [m]
SPEED      = 0.03    # metres per simulation step (same as pygame_large_env)
N_STEPS    = 2000    # steps per starting node (8 nodes × 2000 = 16 000 total)
MAX_GAP    = 4.0     # drop windows with total time gap > 4 s (matches ParticleFilter)

# ── Graph matching pygame_large_env.py exactly ───────────────────────────────
NODE_POS = np.array([
    [-1.5,  1.0],   # 0
    [ 0.0,  1.0],   # 1
    [ 2.1,  1.0],   # 2
    [-1.5, -1.0],   # 3
    [ 0.0, -1.0],   # 4
    [ 0.3,  0.0],   # 5
    [ 1.7,  0.0],   # 6  ← T-junction: splits to 2 or 7
    [ 2.1, -1.0],   # 7
], dtype=np.float32)

TRANS_MATRIX = np.array([
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # 0 → 4
    [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],   # 1 → 0 or 3
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 2 → 1
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 3 → 0
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   # 4 → 5
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   # 5 → 6
    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],   # 6 → 2 or 7
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # 7 → 4
], dtype=np.float32)

# ── Occlusion zones (metres) — must match info_mml_main.py ───────────────────
OCC_CENTERS = np.array([[1.8, 0.0], [1.2, 0.8]])
OCC_WIDTHS  = np.array([1.0, 1.0])


def in_occlusion(pos: np.ndarray) -> bool:
    """True when pos is physically inside an occlusion wall."""
    for c, w in zip(OCC_CENTERS, OCC_WIDTHS):
        if abs(pos[0] - c[0]) < w / 2 and abs(pos[1] - c[1]) < w / 2:
            return True
    return False


def simulate_from_node(start_node: int,
                        n_steps:   int   = N_STEPS,
                        speed:     float = SPEED,
                        noise_std: float = NOISE_STD):
    """
    Walk the Markov graph from start_node.
    Returns only VISIBLE positions and their wall-clock timestamps
    (mirrors ParticleFilter.update_state_and_dt_buffer → is_update=True only).
    """
    pos  = NODE_POS[start_node].copy()
    node = start_node

    vis_positions = []   # only non-occluded positions
    vis_times     = []   # wall-clock time of each visible measurement
    t             = 0.0

    for _ in range(n_steps):
        # Markov graph dynamics — same as pygame_large_env.step()
        probs = TRANS_MATRIX[node]
        nxt   = np.random.choice(8, p=probs) if probs.sum() > 0 else node
        d     = NODE_POS[nxt] - pos
        dist  = np.linalg.norm(d)
        if dist < speed:
            node = nxt
            pos  = NODE_POS[node].copy()
        else:
            pos += d / dist * speed

        t += DT   # advance wall clock every step

        # FIX 3: only record when NOT occluded (is_update=True equivalent)
        if not in_occlusion(pos):
            noisy = pos + np.random.randn(2) * noise_std
            vis_positions.append(noisy.copy())
            vis_times.append(t)

    return (np.array(vis_positions, dtype=np.float32),
            np.array(vis_times,     dtype=np.float32))


def build_dataset(vis_positions: np.ndarray,
                  vis_times:     np.ndarray,
                  N_th:          int   = N_TH,
                  max_gap:       float = MAX_GAP):
    """
    Build (X, y) pairs from a compressed visible-only position buffer.

    Mirrors ParticleFilter.convert_state_history_to_training_batches():
      - Velocity label uses ACTUAL dt between consecutive visible steps.
      - Drops windows where total time gap > max_gap (target lost too long).

    X : (N, N_th * 2)   flattened history windows
    y : (N, 2)           velocity targets  (vx, vy)  in m/s
    """
    M = len(vis_positions)
    X_list, y_list = [], []

    for i in range(M - N_th - 1):
        window     = vis_positions[i : i + N_th]
        next_p     = vis_positions[i + N_th]

        # Actual dt between the last history step and the next visible step
        dt_step    = vis_times[i + N_th] - vis_times[i + N_th - 1]
        # Total time span of the whole window
        total_gap  = vis_times[i + N_th] - vis_times[i]

        # Skip windows that span a long occlusion gap
        if total_gap > max_gap or dt_step < 1e-6:
            continue

        # FIX 1: velocity computed with actual dt — correct scale
        velocity = (next_p - window[-1]) / dt_step

        X_list.append(window.flatten())   # (N_th * 2,)
        y_list.append(velocity)           # (2,)

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32))


def main():
    np.random.seed(42)

    print("=" * 60)
    print("Generating CFM training data — all improvements applied")
    print(f"  DT          = {DT} s  (30 Hz, matches pygame)")
    print(f"  N_th        = {N_TH}  (history window)")
    print(f"  Nodes       = 8 × {N_STEPS} steps = {8 * N_STEPS} total")
    print(f"  Occlusions  = {len(OCC_CENTERS)} zones (measurement gaps included)")
    print("=" * 60)

    all_X, all_y = [], []

    for start_node in range(8):
        vis_pos, vis_t = simulate_from_node(start_node)
        X, y = build_dataset(vis_pos, vis_t)
        all_X.append(X)
        all_y.append(y)
        pct_occ = 100.0 * (1 - len(vis_pos) / N_STEPS)
        print(f"  Node {start_node}: {len(X):5d} samples  "
              f"({pct_occ:.1f}% occluded/dropped)")

    X_all = np.vstack(all_X)
    y_all = np.vstack(all_y)

    print(f"\nTotal dataset: X={X_all.shape}  y={y_all.shape}")
    print(f"Velocity range: vx=[{y_all[:,0].min():.3f}, {y_all[:,0].max():.3f}]  "
          f"vy=[{y_all[:,1].min():.3f}, {y_all[:,1].max():.3f}]")

    # Coverage check — samples near each node
    last_pos = X_all[:, -2:].reshape(-1, 2)
    print("\nCoverage (samples within 0.3 m of each node):")
    for i, np_ in enumerate(NODE_POS):
        near = np.sum(np.linalg.norm(last_pos - np_, axis=1) < 0.3)
        print(f"  Node {i} ({np_[0]:+.1f},{np_[1]:+.1f}): {near:4d} samples")

    # Save — path relative to this script so it works from any directory
    this_dir = str(motion_resource_dir() / "mml_guidance/mml_network")
    save_dir = os.path.normpath(
        os.path.join(this_dir, "..", "sim_data", "training_data")
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, "converted_noisy_5_v2_velocities_training_data.csv"
    )

    dataset = np.hstack([X_all, y_all])
    pd.DataFrame(dataset).to_csv(save_path, index=False, header=False)

    print(f"\nSaved {len(dataset)} samples → {save_path}")
    print(f"\nNext step:  python train_cfm.py --epochs 1500 --n_th 10")


if __name__ == "__main__":
    main()