#!/usr/bin/env python3
"""
compare_trackers_visual.py
==========================
Live PyGame viewer that runs ALL trackers on ONE shared occluded trajectory
with IDENTICAL measurements, so a fair three-way (four-way) comparison can be
watched and audited by eye.

Why this instead of scripts/info_mml_main.py?
---------------------------------------------
info_mml_main.py runs the full guidance stack — PPO policy, GMM search,
multi-target GNN data association, FOV geometry. Only CFM is wired into that
ParticleFilter, and running metrics through it would fold all those subsystems
into the comparison. This viewer strips them away: every tracker sees the same
ground-truth trajectory, the same occlusion zones, and the same noisy
measurement each frame. The only thing that differs is the generative
mechanism + PF update — exactly what benchmark_diffpf_vs_cfm.py measures
headlessly, now rendered so you can verify the fairness yourself.

Trackers (matches benchmark_diffpf_vs_cfm.py)
---------------------------------------------
  PF+CFM      steel blue   bootstrap PF, CFM motion model, fixed anchor sigma
  PF+CFM-AS   sea green    same, occlusion-adaptive anchor sigma (OURS, best)
  PF+NF       crimson      bootstrap PF, NF-DPF conditional-flow motion model
  DiffPF      purple       conditional-diffusion posterior sampler (no weights)

Controls
--------
  SPACE  pause / resume
  p      toggle particle clouds
  UP/DN  slower / faster
  ESC/Q  quit (prints final RMSE table)

Usage (after installing the project, needs a display):
    conda activate tracking_ppo
    python -m mml_research.motion.compare_trackers_visual \
        --cfm_ckpt cfm_motion_model_anchor1.0.pth --seed 3 --fps 60
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import os
import sys
import argparse

import numpy as np
import torch


import pygame

from mml_core.motion.cfm_motion_model import CFMMotionModel
from mml_core.motion.nf_motion_model import NFMotionModel
from mml_core.tracking.diffpf_tracker import DiffPFTracker
from mml_research.motion.benchmark_diffpf_vs_cfm import BootstrapPFTracker
from mml_research.motion.train_diffpf import simulate_full_trajectory
from mml_research.motion.generate_synthetic_data import NOISE_STD, OCC_CENTERS, OCC_WIDTHS, NODE_POS

# ── World → screen mapping ──────────────────────────────────────────────────
SIM_W, SIM_H = 980, 760
HUD_W = 360
WIN_W, WIN_H = SIM_W + HUD_W, SIM_H
MARGIN = 70
X_MIN, X_MAX = -2.2, 2.7
Y_MIN, Y_MAX = -1.6, 1.6

BG = (18, 20, 26)
HUD_BG = (28, 30, 38)
GRID = (40, 44, 54)
WHITE = (235, 238, 245)
DIM = (120, 126, 140)
OCC_COL = (200, 70, 70)

TRACKERS = [
    ("PF+CFM",    (70, 130, 200)),
    ("PF+CFM-AS", (60, 190, 120)),
    ("PF+NF",     (220, 70, 90)),
    ("DiffPF",    (170, 100, 210)),
]


def w2s(p):
    sx = MARGIN + (p[0] - X_MIN) / (X_MAX - X_MIN) * (SIM_W - 2 * MARGIN)
    sy = MARGIN + (1.0 - (p[1] - Y_MIN) / (Y_MAX - Y_MIN)) * (SIM_H - 2 * MARGIN)
    return int(sx), int(sy)


def cloud_of(tracker):
    """Return the tracker's current particle positions as (M, 2) world coords."""
    if isinstance(tracker, DiffPFTracker):
        return tracker.particles if tracker.particles is not None else None
    # BootstrapPFTracker: particles are (N_th, N, 2); use the latest slice.
    if getattr(tracker, "particles", None) is None:
        return None
    return tracker.particles[-1]


def estimate_of(tracker):
    c = cloud_of(tracker)
    if c is None:
        return None
    if isinstance(tracker, DiffPFTracker):
        return c.mean(axis=0)
    return np.average(c, axis=0, weights=tracker.weights)


def main():
    ap = argparse.ArgumentParser(description="Live 4-way tracker comparison viewer")
    ap.add_argument("--cfm_ckpt", type=str, default="cfm_motion_model_anchor1.0.pth")
    ap.add_argument("--n_steps", type=int, default=4000)
    ap.add_argument("--start_node", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--pf_particles", type=int, default=500)
    ap.add_argument("--adaptive_gain", type=float, default=0.05)
    ap.add_argument("--sigma_max", type=float, default=2.5)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    this_dir = str(motion_resource_dir() / "mml_guidance/mml_network")
    models_dir = os.path.normpath(os.path.join(this_dir, "..", "..", "models"))

    # Two independent CFM instances so the adaptive tracker's sigma mutation
    # cannot corrupt the fixed-sigma tracker (they interleave every frame).
    cfm_fixed = CFMMotionModel();  cfm_fixed.load(os.path.join(models_dir, args.cfm_ckpt))
    cfm_adapt = CFMMotionModel();  cfm_adapt.load(os.path.join(models_dir, args.cfm_ckpt))
    nf = NFMotionModel();          nf.load(os.path.join(models_dir, "nf_motion_model.pth"))
    diffpf = DiffPFTracker();      diffpf.load(os.path.join(models_dir, "diffpf_tracker.pth"))

    sigma0 = cfm_fixed.anchor_sigma
    trackers = {
        "PF+CFM":    BootstrapPFTracker(cfm_fixed, n_particles=args.pf_particles,
                                        base_sigma=sigma0),
        "PF+CFM-AS": BootstrapPFTracker(cfm_adapt, n_particles=args.pf_particles,
                                        occlusion_sigma_gain=args.adaptive_gain,
                                        sigma_max=args.sigma_max, base_sigma=sigma0),
        "PF+NF":     BootstrapPFTracker(nf, n_particles=args.pf_particles),
        "DiffPF":    diffpf,
    }

    # Shared trajectory + identical measurement stream for every tracker.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    traj_rng = np.random.default_rng(1000 + args.seed)
    meas_rng = np.random.default_rng(5000 + args.seed)
    positions, visible = simulate_full_trajectory(args.start_node, args.n_steps, traj_rng)

    first_vis = int(np.argmax(visible))
    init = positions[first_vis] + NOISE_STD * meas_rng.standard_normal(2).astype(np.float32)
    for tr in trackers.values():
        tr.reset(init)

    # Running error accumulators.
    sq = {n: 0.0 for n, _ in TRACKERS}
    cnt = {n: 0 for n, _ in TRACKERS}
    sq_occ = {n: 0.0 for n, _ in TRACKERS}
    cnt_occ = {n: 0 for n, _ in TRACKERS}

    # ── PyGame setup ────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("CFM vs NF-DPF vs DiffPF — same trajectory, same measurements")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 15)
    font_b = pygame.font.SysFont("monospace", 17, bold=True)
    big = pygame.font.SysFont("monospace", 20, bold=True)

    # Precompute occlusion rectangles in screen space.
    occ_rects = []
    for c, w in zip(OCC_CENTERS, OCC_WIDTHS):
        x0, y0 = w2s([c[0] - w / 2, c[1] + w / 2])   # top-left (y flipped)
        x1, y1 = w2s([c[0] + w / 2, c[1] - w / 2])   # bottom-right
        occ_rects.append(pygame.Rect(x0, y0, x1 - x0, y1 - y0))

    node_scr = [w2s(NODE_POS[i]) for i in range(len(NODE_POS))]

    show_clouds = True
    paused = False
    fps = args.fps
    t = first_vis + 1
    gt_trail = []

    running = True
    while running and t < len(positions):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_p:
                    show_clouds = not show_clouds
                elif ev.key == pygame.K_UP:
                    fps = min(240, fps + 15)
                elif ev.key == pygame.K_DOWN:
                    fps = max(5, fps - 15)

        if not paused:
            vis = bool(visible[t])
            if vis:
                meas = positions[t] + NOISE_STD * meas_rng.standard_normal(2).astype(np.float32)
            else:
                meas = None

            for name, tr in trackers.items():
                tr.step(meas)
                est = estimate_of(tr)
                if est is not None:
                    e2 = float(np.sum((est - positions[t]) ** 2))
                    sq[name] += e2;  cnt[name] += 1
                    if not vis:
                        sq_occ[name] += e2;  cnt_occ[name] += 1

            gt_trail.append(tuple(positions[t]))
            if len(gt_trail) > 60:
                gt_trail.pop(0)
            t += 1

        # ── Draw ────────────────────────────────────────────────────────────
        screen.fill(BG)
        pygame.draw.rect(screen, HUD_BG, pygame.Rect(SIM_W, 0, HUD_W, WIN_H))

        # Road graph edges + nodes.
        for i in range(len(NODE_POS)):
            for j in range(len(NODE_POS)):
                if i < j:
                    pygame.draw.line(screen, GRID, node_scr[i], node_scr[j], 1)
        for ns in node_scr:
            pygame.draw.circle(screen, (60, 66, 80), ns, 4)

        # Occlusion zones (translucent).
        occ_surf = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
        for r in occ_rects:
            pygame.draw.rect(occ_surf, (*OCC_COL, 55), r)
            pygame.draw.rect(occ_surf, (*OCC_COL, 160), r, 2)
        screen.blit(occ_surf, (0, 0))

        # Particle clouds.
        if show_clouds:
            cloud_surf = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
            for name, col in TRACKERS:
                c = cloud_of(trackers[name])
                if c is None:
                    continue
                step = max(1, len(c) // 120)     # thin to ~120 dots for speed
                for p in c[::step]:
                    sx, sy = w2s(p)
                    if 0 <= sx < SIM_W and 0 <= sy < SIM_H:
                        cloud_surf.set_at((sx, sy), (*col, 90))
            screen.blit(cloud_surf, (0, 0))

        # Ground-truth trail + current target.
        if len(gt_trail) > 1:
            pygame.draw.lines(screen, DIM, False, [w2s(p) for p in gt_trail], 1)
        cur = positions[min(t, len(positions) - 1)]
        cur_vis = bool(visible[min(t, len(visible) - 1)])
        gsx, gsy = w2s(cur)
        if cur_vis:
            pygame.draw.circle(screen, WHITE, (gsx, gsy), 8)
            pygame.draw.circle(screen, (0, 0, 0), (gsx, gsy), 8, 2)
        else:
            pygame.draw.circle(screen, WHITE, (gsx, gsy), 8, 2)   # hollow = occluded

        # Tracker estimates.
        for name, col in TRACKERS:
            est = estimate_of(trackers[name])
            if est is not None:
                ex, ey = w2s(est)
                pygame.draw.circle(screen, col, (ex, ey), 6)
                pygame.draw.circle(screen, (10, 10, 10), (ex, ey), 6, 1)

        # ── HUD ─────────────────────────────────────────────────────────────
        x0 = SIM_W + 18
        y = 20
        screen.blit(big.render("LIVE COMPARISON", True, WHITE), (x0, y)); y += 30
        state_txt = "OCCLUDED" if not cur_vis else "visible"
        state_col = OCC_COL if not cur_vis else (90, 200, 120)
        screen.blit(font_b.render(f"step {t}/{len(positions)}", True, WHITE), (x0, y)); y += 22
        screen.blit(font_b.render(f"target: {state_txt}", True, state_col), (x0, y)); y += 30

        screen.blit(font.render("RMSE (all / occluded), m", True, DIM), (x0, y)); y += 22
        for name, col in TRACKERS:
            rmse = np.sqrt(sq[name] / cnt[name]) if cnt[name] else 0.0
            rmse_o = np.sqrt(sq_occ[name] / cnt_occ[name]) if cnt_occ[name] else 0.0
            pygame.draw.circle(screen, col, (x0 + 7, y + 8), 7)
            label = f"{name:<10} {rmse:5.3f} / {rmse_o:5.3f}"
            screen.blit(font_b.render(label, True, col), (x0 + 22, y)); y += 26

        y += 14
        tag = trackers["PF+CFM-AS"].motion_model.anchor_sigma
        for line in [
            "white dot  = ground truth",
            "hollow dot = target occluded",
            "red zones  = occlusions",
            f"CFM-AS sigma now: {tag:.2f}",
            "",
            "SPACE pause   p clouds",
            f"UP/DN speed   fps={fps}",
            "ESC   quit",
        ]:
            screen.blit(font.render(line, True, DIM), (x0, y)); y += 20

        pygame.display.flip()
        clock.tick(fps if not paused else 30)

    pygame.quit()

    # ── Final table (mirrors benchmark_diffpf_vs_cfm.py) ────────────────────
    print("\n" + "=" * 56)
    print(f"Final RMSE over {t - first_vis - 1} rendered steps "
          f"(seed={args.seed}, start_node={args.start_node})")
    print("=" * 56)
    print(f"{'tracker':<12}{'RMSE all':>12}{'RMSE occ':>12}{'n_occ':>8}")
    for name, _ in TRACKERS:
        rmse = np.sqrt(sq[name] / cnt[name]) if cnt[name] else float("nan")
        rmse_o = np.sqrt(sq_occ[name] / cnt_occ[name]) if cnt_occ[name] else float("nan")
        print(f"{name:<12}{rmse:>12.4f}{rmse_o:>12.4f}{cnt_occ[name]:>8}")
    print("=" * 56)


if __name__ == "__main__":
    main()
