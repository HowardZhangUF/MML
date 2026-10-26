"""
cfm_nets.py — conditional CFM network and trainer shared by M1 and M4.

ConditionalCFMNet is CFMVelocityNetwork (same ContextEncoder and
VelocityHead) with two optional extensions:
    * extra per-step channels (step_dim > 2, e.g. the history mask)
    * global features, projected and added to the context embedding

With step_dim = 2 and no global features it is the baseline architecture.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from mml_core.motion.velocity_network import ContextEncoder, VelocityHead

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SPEED = 1.5


class ConditionalCFMNet(nn.Module):
    def __init__(self, step_dim: int = 2, block_size: int = 5, n_global: int = 0,
                 n_embed: int = 32, n_head: int = 8, n_layer: int = 2, vel_hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.step_dim, self.block_size, self.n_global = step_dim, block_size, n_global
        self.encoder = ContextEncoder(input_dim=step_dim, block_size=block_size, n_embed=n_embed,
                                      n_head=n_head, n_layer=n_layer, dropout=dropout)
        self.global_proj = nn.Linear(n_global, n_embed) if n_global else None
        self.vel_head = VelocityHead(n_embed=n_embed, hidden_dim=vel_hidden)

    def encode(self, steps_flat: torch.Tensor, glob: torch.Tensor | None) -> torch.Tensor:
        emb = self.encoder(steps_flat)
        if self.global_proj is not None:
            emb = emb + self.global_proj(glob)
        return emb

    def predict_velocity(self, x_tau, tau, emb):
        return self.vel_head(x_tau, tau, emb)


@torch.no_grad()
def euler_sample(net: ConditionalCFMNet, emb: torch.Tensor, x0: torch.Tensor, n_fe: int) -> torch.Tensor:
    """Fixed-step Euler integration of the learned field from tau=0 to 1 (CFMMotionModel.propose)."""
    x = x0
    delta = 1.0 / n_fe
    for s in range(n_fe):
        tau = torch.full((x.shape[0], 1), s * delta, dtype=torch.float32, device=x.device)
        x = x + delta * net.predict_velocity(x, tau, emb)
    return x


def clip_speed(v: np.ndarray, max_speed: float = MAX_SPEED) -> np.ndarray:
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    return (v * np.minimum(1.0, max_speed / (speed + 1e-9))).astype(np.float32)


def train_cfm(net: ConditionalCFMNet, steps_flat: np.ndarray, glob: np.ndarray,
              src_mean: np.ndarray, src_std: float, target: np.ndarray, holdout: np.ndarray,
              n_steps: int, batch: int, lr: float, seed: int, log_every: int = 1000):
    """
    Straight-path CFM on independent (source, target) pairs:
        y0 = src_mean + src_std * eps,  y1 = target,  y_tau = (1-tau) y0 + tau y1,
        loss = || u(y_tau, tau, C) - (y1 - y0) ||^2   (mean over batch and coordinates)
    `holdout` marks rows excluded from training; their loss is reported only.
    Returns (train-loss history, holdout-loss history).
    """
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=device)
    S, G, M, Y = T(steps_flat), T(glob), T(src_mean), T(target)
    tr = np.flatnonzero(~holdout)
    ho = np.flatnonzero(holdout)
    ho_idx = torch.tensor(gen.choice(ho, min(len(ho), 20000), replace=False), device=device) if len(ho) else None

    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.01)
    tr_t = torch.tensor(tr, device=device)

    def loss_on(idx):
        y1 = Y[idx]
        y0 = M[idx] + src_std * torch.randn_like(y1)
        tau = torch.rand(len(idx), 1, device=device)
        emb = net.encode(S[idx], G[idx] if G.shape[1] else None)
        u = net.predict_velocity((1 - tau) * y0 + tau * y1, tau, emb)
        return ((u - (y1 - y0)) ** 2).mean()

    train_hist, ho_hist = [], []
    run, t0 = [], time.time()
    for step in range(1, n_steps + 1):
        idx = tr_t[torch.randint(len(tr_t), (batch,), device=device)]
        loss = loss_on(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % log_every == 0 or step == n_steps:
            net.eval()
            with torch.no_grad():
                ho_loss = float(np.mean([loss_on(ho_idx).item() for _ in range(4)])) if ho_idx is not None else math.nan
            net.train()
            train_hist.append(float(np.mean(run)))
            ho_hist.append(ho_loss)
            print(f"    step {step:6d}/{n_steps}  train={train_hist[-1]:.4f}  holdout={ho_loss:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            run = []
    net.eval()
    return train_hist, ho_hist
