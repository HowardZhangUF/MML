"""
residual_model.py — M4 DMMN-CFM and its Gaussian control (guide chapter 7).

    mu(C)  frozen DMMN velocity (raw network output, before its speed clip)
    e*     = v* - mu(C)                       residual of the true velocity
    S      = diag(s_x, s_y), s = max(std(e* on train), floor)
    r*     = S^-1 e*

M4 (ResidualCFMMotionModel): CFM from r0 ~ N(0, I) to r*, conditioned on the
    history C and on mu(C) (a function of C, so no new information);
    v = mu(C) + S r, then the common speed clip.
Control (GaussianResidualMotionModel): v = mu(C) + b + L eps with b, L L^T the
    training mean and covariance of e* (no zero-mean assumption).
Both are sampled as the declared transition (bootstrap weights).
"""

from __future__ import annotations

import numpy as np
import torch

from mml_research.motion.benchmark_cv_dmmn import OfflineDMMNMotionModel
from mml_research.motion.cfm_variants.cfm_nets import (
    ConditionalCFMNet, clip_speed, device, euler_sample, train_cfm,
)
from mml_research.motion.cfm_variants.features import Normalizer


class DMMNMean:
    """Raw DMMN mean velocity mu(C) for flat contexts (N, 10)."""

    def __init__(self, checkpoint: str, dt: float):
        self.dmmn = OfflineDMMNMotionModel(checkpoint, dt=dt)
        self.block_size, self.dt = 5, dt

    def __call__(self, c_flat: np.ndarray, chunk: int = 65536) -> np.ndarray:
        c_flat = np.asarray(c_flat, np.float32)
        return np.concatenate([self.dmmn._forward_torch(c_flat[i:i + chunk])
                               for i in range(0, len(c_flat), chunk)])


class ResidualCFMMotionModel:
    def __init__(self, dmmn: DMMNMean, dt: float = 0.033, N_FE: int = 10, scale_floor: float = 0.05,
                 arch: dict | None = None):
        self.dmmn = dmmn
        self.block_size, self.dt, self.N_FE = 5, float(dt), int(N_FE)
        self.scale_floor = float(scale_floor)
        self.arch = dict(arch or {})
        self.net = ConditionalCFMNet(2, self.block_size, 2, **self.arch).to(device)
        self.S = np.ones(2, np.float32)
        self.b = np.zeros(2, np.float32)        # training residual mean (diagnostic only)
        self.mu_norm = Normalizer(np.zeros(2), np.ones(2))
        self.info = {}

    def fit(self, c_flat, mu, v, holdout, n_steps, batch, lr, seed):
        e = v - mu
        self.S = np.maximum(e[~holdout].std(0), self.scale_floor).astype(np.float32)
        self.b = e[~holdout].mean(0).astype(np.float32)
        self.mu_norm = Normalizer().fit(mu[~holdout])
        r = e / self.S
        losses = train_cfm(self.net, c_flat, self.mu_norm(mu), np.zeros_like(r), 1.0, r, holdout,
                           n_steps, batch, lr, seed)
        self.info = {"seed": seed, "S": self.S.tolist(), "residual_mean": self.b.tolist(),
                     "train_loss": losses[0], "holdout_loss": losses[1]}
        return losses

    @torch.no_grad()
    def sample_residual(self, c_flat: np.ndarray, mu: np.ndarray) -> np.ndarray:
        C = torch.tensor(c_flat, dtype=torch.float32, device=device)
        G = torch.tensor(self.mu_norm(mu), dtype=torch.float32, device=device)
        emb = self.net.encode(C, G)
        return euler_sample(self.net, emb, torch.randn(len(C), 2, device=device), self.N_FE).cpu().numpy()

    def propose(self, c_flat: np.ndarray) -> np.ndarray:
        c_flat = np.asarray(c_flat, np.float32)
        mu = self.dmmn(c_flat)
        return clip_speed(mu + self.S * self.sample_residual(c_flat, mu))

    def save(self, path):
        torch.save({"state_dict": self.net.state_dict(),
                    "config": {"kind": "m4_residual_cfm", "dt": self.dt, "N_FE": self.N_FE,
                               "scale_floor": self.scale_floor, "arch": self.arch, "S": self.S.tolist(),
                               "residual_mean": self.b.tolist(), "mu_normalizer": self.mu_norm.to_dict(),
                               "dmmn_checkpoint": self.dmmn.dmmn.checkpoint_path},
                    "info": self.info}, path)

    @classmethod
    def load(cls, path, dmmn: DMMNMean):
        ck = torch.load(path, map_location=device, weights_only=False)
        c = ck["config"]
        m = cls(dmmn, c["dt"], c["N_FE"], c["scale_floor"], c["arch"])
        m.net.load_state_dict(ck["state_dict"])
        m.net.eval()
        m.S = np.asarray(c["S"], np.float32)
        m.b = np.asarray(c["residual_mean"], np.float32)
        m.mu_norm = Normalizer.from_dict(c["mu_normalizer"])
        m.info = ck.get("info", {})
        return m


class GaussianResidualMotionModel:
    def __init__(self, dmmn: DMMNMean, b=None, cov=None, dt: float = 0.033):
        self.dmmn = dmmn
        self.block_size, self.dt = 5, float(dt)
        self.b = np.zeros(2, np.float32) if b is None else np.asarray(b, np.float32)
        self.cov = np.eye(2, dtype=np.float32) if cov is None else np.asarray(cov, np.float32)
        self.L = np.linalg.cholesky(self.cov).astype(np.float32)

    def fit(self, mu, v, holdout):
        e = (v - mu)[~holdout]
        self.b = e.mean(0).astype(np.float32)
        self.cov = np.cov(e.T).astype(np.float32)
        self.L = np.linalg.cholesky(self.cov).astype(np.float32)
        return self

    def propose(self, c_flat: np.ndarray) -> np.ndarray:
        c_flat = np.asarray(c_flat, np.float32)
        mu = self.dmmn(c_flat)
        eps = torch.randn(len(c_flat), 2).numpy()
        return clip_speed(mu + self.b + eps @ self.L.T)

    def save(self, path):
        torch.save({"config": {"kind": "dmmn_gaussian_residual", "b": self.b.tolist(),
                               "cov": self.cov.tolist(), "dt": self.dt}}, path)

    @classmethod
    def load(cls, path, dmmn: DMMNMean):
        c = torch.load(path, map_location="cpu", weights_only=False)["config"]
        return cls(dmmn, c["b"], c["cov"], c["dt"])
