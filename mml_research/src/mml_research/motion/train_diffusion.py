#!/usr/bin/env python3
"""
train_diffusion.py
===================
Train the DDPM diffusion motion model on the same converted training CSV
used by train_cfm.py — a direct comparison point for CFMMotionModel.

Place at:
    mml_research/motion/train_diffusion.py

Typical usage (after installing the project):
    # First generate data (shared with CFM):
    python generate_synthetic_data.py

    # Train with settings matching the CFM run:
    python -m mml_research.motion.train_diffusion --epochs 1500 --n_th 10 --n_embed 64 --n_head 8 --n_layer 2 --noise_hidden 128 --T 20

    # Quick test run:
    python -m mml_research.motion.train_diffusion --epochs 200 --n_th 10

Output:
    ../../models/diffusion_motion_model.pth     (model weights)
    ../../models/diffusion_training_loss.png    (loss curve)
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ── Import from the same directory ───────────────────────────────────────────

from mml_core.motion.diffusion_motion_model import DiffusionMotionModel
from mml_core.motion.utils import load_training_csv, chronological_split

import torch
import torch.nn as nn


def main():
    # ── Parse arguments ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train the diffusion motion model")
    parser.add_argument("--epochs",         type=int,   default=1500,
                        help="Training epochs (1500 recommended)")
    parser.add_argument("--batch",          type=int,   default=512,
                        help="Mini-batch size (512 recommended)")
    parser.add_argument("--lr",             type=float, default=5e-4,
                        help="Initial Adam learning rate (cosine decay applied)")
    parser.add_argument("--n_th",           type=int,   default=10,
                        help="History window length N_th (must match ParticleFilter.N_th)")
    parser.add_argument("--T",              type=int,   default=20,
                        help="Diffusion steps (kept small for fair speed comparison with CFM's N_FE)")
    parser.add_argument("--n_embed",        type=int,   default=64,
                        help="Transformer embedding dim (must be divisible by n_head)")
    parser.add_argument("--n_head",         type=int,   default=8,
                        help="Attention heads (must divide n_embed)")
    parser.add_argument("--n_layer",        type=int,   default=2,
                        help="Number of transformer blocks")
    parser.add_argument("--noise_hidden",   type=int,   default=128,
                        help="Noise-prediction MLP hidden size")
    parser.add_argument("--time_embed_dim", type=int,   default=32,
                        help="Sinusoidal timestep embedding dim")
    parser.add_argument("--beta_schedule",  type=str,   default="cosine",
                        choices=["cosine", "linear"],
                        help="Diffusion beta schedule")
    parser.add_argument("--data-dir", default=None, help="Override the data dir")
    parser.add_argument("--models-dir", default=None, help="Override the models dir")
    args = parser.parse_args()

    # ── Validate n_embed divisible by n_head ──────────────────────────────────
    if args.n_embed % args.n_head != 0:
        raise ValueError(
            f"n_embed ({args.n_embed}) must be divisible by n_head ({args.n_head}). "
            f"Try n_head={args.n_embed // (args.n_embed // args.n_head)}"
        )

    # ── Resolve paths relative to this script ─────────────────────────────────
    this_dir = str(motion_resource_dir() / "mml_guidance/mml_network")
    data_dir = args.data_dir or os.path.join(this_dir, '..', 'sim_data', 'training_data')
    models_dir = args.models_dir or os.path.join(this_dir, '..', '..', 'models')

    csv_path  = os.path.normpath(os.path.join(
        data_dir, "converted_noisy_5_v2_velocities_training_data.csv"
    ))
    save_path = os.path.normpath(os.path.join(models_dir, "diffusion_motion_model.pth"))
    plot_path = os.path.normpath(os.path.join(models_dir, "diffusion_training_loss.png"))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("=" * 60)
    print("Training Diffusion Motion Model")
    print("=" * 60)
    print(f"Data           : {csv_path}")
    print(f"Save           : {save_path}")
    print(f"N_th           : {args.n_th}  (history window)")
    print(f"n_embed        : {args.n_embed}")
    print(f"n_head         : {args.n_head}")
    print(f"n_layer        : {args.n_layer}")
    print(f"noise_hidden   : {args.noise_hidden}")
    print(f"T              : {args.T}  (diffusion steps)")
    print(f"beta_schedule  : {args.beta_schedule}")
    print(f"epochs         : {args.epochs}  batch={args.batch}  lr={args.lr}")

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training CSV not found at:\n  {csv_path}\n"
            "Run generate_synthetic_data.py first to create it."
        )

    X_all, y_all = load_training_csv(csv_path, input_dim=2)

    # Verify N_th matches the data shape
    expected_cols = args.n_th * 2   # N_th positions × 2 coords
    if X_all.shape[1] != expected_cols:
        raise ValueError(
            f"Data has {X_all.shape[1]} feature columns "
            f"but --n_th {args.n_th} expects {expected_cols} columns.\n"
            f"Either re-run generate_synthetic_data.py with N_TH={args.n_th} "
            f"or set --n_th to {X_all.shape[1] // 2}."
        )

    # Chronological split — later data → train, earlier → test (matches train_cfm.py)
    X_train, X_test, y_train, y_test = chronological_split(
        X_all, y_all, train_frac=0.7
    )
    print(f"\nTrain: {X_train.shape}   Test: {X_test.shape}")
    print(f"Velocity magnitude: "
          f"vx=[{y_train[:,0].min():.3f},{y_train[:,0].max():.3f}]  "
          f"vy=[{y_train[:,1].min():.3f},{y_train[:,1].max():.3f}]")

    # ── Initialise model ──────────────────────────────────────────────────────
    print("\nInitialising DiffusionMotionModel...")
    diffusion = DiffusionMotionModel(
        input_dim      = 2,
        block_size     = args.n_th,
        n_embed        = args.n_embed,
        n_head         = args.n_head,
        n_layer        = args.n_layer,
        noise_hidden   = args.noise_hidden,
        time_embed_dim = args.time_embed_dim,
        T              = args.T,
        beta_schedule  = args.beta_schedule,
    )
    n_params = sum(p.numel() for p in diffusion.network.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs...")
    losses = diffusion.train(
        X_train, y_train,
        epochs      = args.epochs,
        batch_size  = args.batch,
        lr          = args.lr,
        print_every = max(1, args.epochs // 10),
    )

    # ── Evaluate on test set (noise-prediction MSE, consistent with training loss) ──
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion.network.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).to(dev)
        yt = torch.tensor(y_test, dtype=torch.float32).to(dev)
        B = yt.shape[0]

        t = torch.randint(0, diffusion.T, (B,), device=dev)
        eps = torch.randn_like(yt)
        sqrt_ab = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_1mab = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        y_t_noised = sqrt_ab * yt + sqrt_1mab * eps

        eps_pred = diffusion.network(y_t_noised, t, Xt)
        test_loss = nn.MSELoss()(eps_pred, eps).item()

    final_train_loss = losses[-1]
    print(f"\nFinal train loss (noise MSE): {final_train_loss:.6f}")
    print(f"Test  loss (noise MSE)      : {test_loss:.6f}")

    # ── Save model (includes all hyperparams in checkpoint) ───────────────────
    diffusion.save(save_path)
    print(f"\nModel saved → {save_path}")

    # ── Loss curve ────────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 3))
    plt.plot(losses, lw=1.5, color="darkorange")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Noise-prediction MSE loss")
    plt.title(
        f"Diffusion Training Loss  "
        f"(N_th={args.n_th}, n_embed={args.n_embed}, n_layer={args.n_layer}, "
        f"T={args.T}, schedule={args.beta_schedule})\n"
        f"final={final_train_loss:.5f}  test={test_loss:.5f}"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"Loss curve → {plot_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print(f'  model_path        = r"{os.path.abspath(save_path)}"')
    print(f'  prediction_method = "Diffusion"')
    print(f"  N_th in ParticleFilter.__init__() must be {args.n_th}")
    print("=" * 60)


if __name__ == "__main__":
    main()
