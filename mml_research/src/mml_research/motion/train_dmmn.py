#!/usr/bin/env python3
"""
train_dmmn.py
=============
Train the DMMN baseline (ScratchTransformer velocity regressor) on the same
converted training CSV as train_cfm.py / train_nf.py, with the architecture
of best_noisy_5_v2_velocities_occlusions_ScratchTransformer.pth (the shape
OfflineDMMNMotionModel in benchmark_cv_dmmn.py expects).

Usage (after installing the project):
    python -m mml_research.motion.train_dmmn --data-dir <dir> --models-dir <dir>

Output:
    <models-dir>/dmmn_scratch_transformer.pth   (plain state_dict)
"""
from mml_core.config import motion_resource_dir

import os
import argparse

import numpy as np
import torch
import torch.nn as nn

from mml_core.motion.scratch_transformer import ScratchTransformer
from mml_core.motion.utils import load_training_csv, chronological_split


def main():
    parser = argparse.ArgumentParser(description="Train the DMMN ScratchTransformer baseline")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default=None, help="Override the data dir")
    parser.add_argument("--models-dir", default=None, help="Override the models dir")
    args = parser.parse_args()

    this_dir = str(motion_resource_dir() / "mml_guidance/mml_network")
    data_dir = args.data_dir or os.path.join(this_dir, '..', 'sim_data', 'training_data')
    models_dir = args.models_dir or os.path.join(this_dir, '..', '..', 'models')
    csv_path = os.path.join(data_dir, "converted_noisy_5_v2_velocities_training_data.csv")
    save_path = os.path.join(models_dir, "dmmn_scratch_transformer.pth")
    os.makedirs(models_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    X, y = load_training_csv(csv_path, input_dim=2)
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    Xtr = torch.tensor(X_train, device=device)
    ytr = torch.tensor(y_train, device=device)
    Xte = torch.tensor(X_test, device=device)
    yte = torch.tensor(y_test, device=device)

    # Architecture of the original DMMN checkpoint.
    model = ScratchTransformer(input_dim=2, block_size=5, n_embed=20,
                               n_head=8, n_layer=2, hidden_dim=20).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(Xte), yte).item()
        if test_loss < best:
            best = test_loss
            torch.save(model.state_dict(), save_path)
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch + 1:4d}  train={loss.item():.4f}  test={test_loss:.4f}  best={best:.4f}")

    print(f"Best test MSE {best:.4f} → {save_path}")


if __name__ == "__main__":
    main()
