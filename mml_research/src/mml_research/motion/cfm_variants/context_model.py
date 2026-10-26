"""
context_model.py — M1 Context-CFM (guide chapter 4).

Same source and interpolation as the baseline CFM:
    y0 = a(C) + sigma0 * eps,   a(C) = (x_{k-1} - x_{k-2}) / dt
Only the conditioning changes, to C~ = (C, M, d, U) for the enabled features:
    features = []                      M1-control (rollout data, positions only)
               [mask]                  M1-mask
               [mask, time]            M1-time
               [mask, time, cov]       M1-full
Target: the true physical velocity (x_true_k - x_true_{k-1}) / dt paired with
the PF rollout context at k-1 (prepare.py).

The transition is ensemble-adaptive when U is used (it depends on the whole
cloud); it is still the declared transition, sampled as the proposal, so the
filter weights by the likelihood only.
"""

from __future__ import annotations

import numpy as np
import torch

from mml_research.motion.cfm_variants.cfm_nets import (
    ConditionalCFMNet, clip_speed, device, euler_sample, train_cfm,
)
from mml_research.motion.cfm_variants.features import Normalizer, global_features

MASK_SEMANTICS = "1 = a measurement was available and used at this history step; entries are particle states"


class ContextCFMMotionModel:
    uses_context = True

    def __init__(self, features=(), block_size: int = 5, dt: float = 0.033, anchor_sigma: float = 1.0,
                 N_FE: int = 10, arch: dict | None = None):
        self.features = list(features)
        self.block_size, self.dt = int(block_size), float(dt)
        self.anchor_sigma, self.N_FE = float(anchor_sigma), int(N_FE)
        self.arch = dict(arch or {})
        self.step_dim = 3 if "mask" in self.features else 2
        self.n_global = (1 if "time" in self.features else 0) + (3 if "cov" in self.features else 0)
        self.net = ConditionalCFMNet(self.step_dim, self.block_size, self.n_global, **self.arch).to(device)
        self.norm = Normalizer(np.zeros(self.n_global), np.ones(self.n_global))
        self.info = {}

    # ---------------------------------------------------------------- inputs

    def step_inputs(self, hist: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """hist (B, H, 2), mask (B, H) -> (B, H*step_dim)."""
        if self.step_dim == 2:
            return hist.reshape(len(hist), -1)
        return np.concatenate([hist, mask[:, :, None]], axis=2).reshape(len(hist), -1)

    def anchor(self, hist: np.ndarray) -> np.ndarray:
        return (hist[:, -1] - hist[:, -2]) / self.dt

    # ----------------------------------------------------------------- train

    def fit(self, data: dict, holdout: np.ndarray, n_steps: int, batch: int, lr: float, seed: int):
        hist = data["hist"].astype(np.float32)
        glob_raw = global_features(self.features, data["d"], data["U"])
        self.norm = Normalizer().fit(glob_raw[~holdout]) if self.n_global else self.norm
        hist_tr = self.step_inputs(hist, data["mask"].astype(np.float32))
        losses = train_cfm(self.net, hist_tr, self.norm(glob_raw), self.anchor(hist), self.anchor_sigma,
                           data["v"], holdout, n_steps, batch, lr, seed)
        self.info = {"seed": seed, "n_train": int((~holdout).sum()), "n_holdout": int(holdout.sum()),
                     "train_loss": losses[0], "holdout_loss": losses[1]}
        return losses

    # ------------------------------------------------------------- inference

    @torch.no_grad()
    def sample(self, hist: np.ndarray, mask: np.ndarray, d, U) -> np.ndarray:
        """Velocities for B contexts; mask (B, H), d (B,), U (B, 3)."""
        g = self.norm(global_features(self.features, d, U))
        S = torch.tensor(self.step_inputs(hist, mask), dtype=torch.float32, device=device)
        G = torch.tensor(g, dtype=torch.float32, device=device) if self.n_global else None
        emb = self.net.encode(S, G)
        a = torch.tensor(self.anchor(hist), dtype=torch.float32, device=device)
        v = euler_sample(self.net, emb, a + self.anchor_sigma * torch.randn_like(a), self.N_FE)
        return clip_speed(v.cpu().numpy())

    def propose_with_context(self, c_flat: np.ndarray, ctx) -> np.ndarray:
        N = len(c_flat)
        hist = c_flat.reshape(N, self.block_size, 2)
        return self.sample(hist, np.broadcast_to(ctx.mask, (N, self.block_size)),
                           np.full(N, ctx.d, np.float32), np.broadcast_to(ctx.U, (N, 3)))

    # ------------------------------------------------------------- save/load

    def save(self, path: str):
        torch.save({"state_dict": self.net.state_dict(),
                    "config": {"kind": "m1_context", "features": self.features, "block_size": self.block_size,
                               "dt": self.dt, "anchor_sigma": self.anchor_sigma, "N_FE": self.N_FE,
                               "arch": self.arch, "feature_order": ["x", "y"] + (["mask"] if self.step_dim == 3 else [])
                               + (["d_seconds"] if "time" in self.features else [])
                               + (["logP11", "logP22", "corr12"] if "cov" in self.features else []),
                               "mask_semantics": MASK_SEMANTICS, "normalizer": self.norm.to_dict(),
                               "target": "true physical velocity (m/s)"},
                    "info": self.info}, path)

    @classmethod
    def load(cls, path: str):
        ck = torch.load(path, map_location=device, weights_only=False)
        c = ck["config"]
        m = cls(c["features"], c["block_size"], c["dt"], c["anchor_sigma"], c["N_FE"], c["arch"])
        m.net.load_state_dict(ck["state_dict"])
        m.net.eval()
        m.norm = Normalizer.from_dict(c["normalizer"])
        m.info = ck.get("info", {})
        return m
