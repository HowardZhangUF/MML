#!/usr/bin/env python3
"""
train_nf.py
============
Train the conditional normalizing-flow (NF-DPF-style) motion model on the
same converted training CSV used by train_cfm.py / train_diffusion.py /
train_gmm.py — the fourth learned comparison point.

Usage (after installing the project):
    python -m mml_research.motion.train_nf --epochs 1500 --n_th 5 --n_embed 32 --n_head 8 --n_layer 2 --n_flows 2 --flow_hidden 32

Output:
    ../../models/nf_motion_model.pth
    ../../models/nf_training_loss.png
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


from mml_core.motion.nf_motion_model import NFMotionModel
from mml_core.motion.utils import load_training_csv, chronological_split

import torch


def main():
    parser = argparse.ArgumentParser(description="Train the NF (conditional RealNVP) motion model")
    parser.add_argument("--epochs",      type=int,   default=1500)
    parser.add_argument("--batch",       type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--n_th",        type=int,   default=10,
                        help="History window length N_th (must match ParticleFilter.N_th)")
    parser.add_argument("--n_embed",     type=int,   default=64,
                        help="Transformer embedding dim (must be divisible by n_head)")
    parser.add_argument("--n_head",      type=int,   default=8)
    parser.add_argument("--n_layer",     type=int,   default=2)
    parser.add_argument("--n_flows",     type=int,   default=2,
                        help="Stacked conditional RealNVP blocks (2 matches NF-DPF n_sequence)")
    parser.add_argument("--flow_hidden", type=int,   default=32,
                        help="Hidden width of coupling s/t networks")
    parser.add_argument("--data-dir", default=None, help="Override the data dir")
    parser.add_argument("--models-dir", default=None, help="Override the models dir")
    args = parser.parse_args()

    if args.n_embed % args.n_head != 0:
        raise ValueError(
            f"n_embed ({args.n_embed}) must be divisible by n_head ({args.n_head})."
        )

    this_dir = str(motion_resource_dir() / "mml_guidance/mml_network")
    data_dir = args.data_dir or os.path.join(this_dir, '..', 'sim_data', 'training_data')
    models_dir = args.models_dir or os.path.join(this_dir, '..', '..', 'models')

    csv_path  = os.path.normpath(os.path.join(
        data_dir, "converted_noisy_5_v2_velocities_training_data.csv"
    ))
    save_path = os.path.normpath(os.path.join(models_dir, "nf_motion_model.pth"))
    plot_path = os.path.normpath(os.path.join(models_dir, "nf_training_loss.png"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("=" * 60)
    print("Training NF (conditional RealNVP) Motion Model")
    print("=" * 60)
    print(f"Data        : {csv_path}")
    print(f"Save        : {save_path}")
    print(f"N_th        : {args.n_th}  (history window)")
    print(f"n_embed     : {args.n_embed}")
    print(f"n_head      : {args.n_head}")
    print(f"n_layer     : {args.n_layer}")
    print(f"n_flows     : {args.n_flows}  (RealNVP blocks)")
    print(f"flow_hidden : {args.flow_hidden}")
    print(f"epochs      : {args.epochs}  batch={args.batch}  lr={args.lr}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training CSV not found at:\n  {csv_path}\n"
            "Run generate_synthetic_data.py first to create it."
        )

    X_all, y_all = load_training_csv(csv_path, input_dim=2)

    expected_cols = args.n_th * 2
    if X_all.shape[1] != expected_cols:
        raise ValueError(
            f"Data has {X_all.shape[1]} feature columns "
            f"but --n_th {args.n_th} expects {expected_cols} columns."
        )

    X_train, X_test, y_train, y_test = chronological_split(
        X_all, y_all, train_frac=0.7
    )
    print(f"\nTrain: {X_train.shape}   Test: {X_test.shape}")

    print("\nInitialising NFMotionModel...")
    nf = NFMotionModel(
        input_dim   = 2,
        block_size  = args.n_th,
        n_embed     = args.n_embed,
        n_head      = args.n_head,
        n_layer     = args.n_layer,
        n_flows     = args.n_flows,
        flow_hidden = args.flow_hidden,
    )
    n_params = sum(p.numel() for p in nf.network.parameters())
    print(f"Parameters: {n_params:,}")

    print(f"\nStarting training for {args.epochs} epochs...")
    losses = nf.train(
        X_train, y_train,
        epochs      = args.epochs,
        batch_size  = args.batch,
        lr          = args.lr,
        print_every = max(1, args.epochs // 10),
    )

    # Test NLL — exact, the flow's signature capability
    nf.network.eval()
    with torch.no_grad():
        test_nll = float(-np.mean(nf.log_prob(X_test, y_test)))

    final_train_loss = losses[-1]
    print(f"\nFinal train loss (NLL): {final_train_loss:.6f}")
    print(f"Test  loss (NLL)      : {test_nll:.6f}")

    nf.save(save_path)

    plt.figure(figsize=(9, 3))
    plt.plot(losses, lw=1.5, color="crimson")
    plt.xlabel("Epoch")
    plt.ylabel("Exact NLL")
    plt.title(
        f"NF Training Loss  (N_th={args.n_th}, n_embed={args.n_embed}, "
        f"n_flows={args.n_flows}, flow_hidden={args.flow_hidden})\n"
        f"final={final_train_loss:.5f}  test={test_nll:.5f}"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"Loss curve → {plot_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print(f'  model_path        = r"{os.path.abspath(save_path)}"')
    print(f'  prediction_method = "NF"')
    print(f"  N_th in ParticleFilter.__init__() must be {args.n_th}")
    print("=" * 60)


if __name__ == "__main__":
    main()
