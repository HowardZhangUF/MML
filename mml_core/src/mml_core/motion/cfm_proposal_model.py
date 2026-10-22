"""
cfm_proposal_model.py
=====================
CFMProposalModel — observation-conditioned flow-matching proposal
q_phi(v | C, z) for the particle filter, paired with a CFMMotionModel
f_theta(v | C) and a CFMMeasurementModel p_psi(z | x).

State-space model (one filter step, particle history C, last position x_prev):

    v   ~ f_theta(v | C)                  CFM motion model (velocity)
    eta ~ N(0, q^2 I)                     process noise
    x   = x_prev + dt * v + eta
    z   ~ p_psi(z | x)                    learned measurement model

The proposal draws eta from its prior and v from q_phi given the
measurement-implied velocity

    m = (z - x_prev - eta) / dt,

so the importance weight on the extended (v, eta) space is

    w  ∝  w_prev * p_psi(z | x) * f_theta(v | C) / q_phi(v | C, m)

(eta's prior cancels; no convolution density is needed).

What the proposal learns
------------------------
f_theta depends on C only through its context embedding e(C) and anchor
a(C), so the exact posterior p(v | C, m) is a function of
(e(C), a(C), x_prev, m). q_phi is a CFM conditioned on exactly those
features (e(C) from the frozen motion encoder) and is trained on joint
samples (C, v, m) drawn from the model above, i.e. it is an amortised
approximation of the optimal proposal. Its source is N(m, sigma_p^2 I).

Its log-density is the exact density of its N_FE-step Euler sampler
(flow_density.py), so the weights stay correct even where q_phi is a poor
approximation; quality only affects efficiency (ESS).
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from mml_core.motion.flow_density import MLPField, euler_forward, gaussian_log_prob


device = "cuda" if torch.cuda.is_available() else "cpu"


class CFMProposalModel:
    def __init__(self, n_embed: int = 32, hidden: int = 128, N_FE: int = 10,
                 source_sigma: float = 1.0, dt: float = 0.033):
        self.n_embed = int(n_embed)
        self.hidden = int(hidden)
        self.N_FE = int(N_FE)
        self.source_sigma = float(source_sigma)
        self.dt = float(dt)
        self._build()
        self._losses: List[float] = []

    def _build(self):
        H = self.hidden
        cond_dim = self.n_embed + 6      # e(C), a(C), m, x_prev
        self.net = nn.Sequential(
            nn.Linear(cond_dim + 3, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
            nn.Linear(H, 2),
        ).to(device)

    @staticmethod
    def condition(emb, anchor, m, x_prev):
        return torch.cat([emb, anchor, m, x_prev], dim=-1)

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def sample_with_log_prob(self, cond: torch.Tensor, m: torch.Tensor):
        """Velocities v ~ q_phi and their exact log-density. cond from condition()."""
        x0 = m + self.source_sigma * torch.randn_like(m)
        v, logdet = euler_forward(MLPField(self.net, cond), x0, self.N_FE)
        return v, gaussian_log_prob(x0, m, self.source_sigma) - logdet

    # ------------------------------------------------------------------ train

    def train(self, motion, meas, contexts: np.ndarray, epochs: int = 200,
              batch_size: int = 4096, lr: float = 1e-3, process_noise: float = 0.01,
              print_every: int = 20) -> List[float]:
        """
        Fit q_phi to the model posterior p(v | C, z).

        motion:   CFMMotionModel (frozen) — samples v and supplies e(C), a(C)
        meas:     CFMMeasurementModel (frozen) — samples z given x
        contexts: (N, block_size * 2) particle histories to train on
                  (fresh v, eta, z are drawn for every batch)
        """
        motion.network.eval()
        c_all = torch.tensor(np.asarray(contexts, np.float32), device=device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
        n = len(c_all)
        self._losses = []
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n, device=device)
            losses = []
            for i in range(0, n - batch_size + 1, batch_size):
                c = c_all[perm[i:i + batch_size]]
                with torch.no_grad():
                    emb = motion.network.encode_context(c)
                    anchor = motion._finite_difference_velocity(c)
                    v, _ = motion.sample_with_log_prob(c, emb)
                    x_prev = c[:, -2:]
                    eta = process_noise * torch.randn_like(v)
                    z = meas.sample(x_prev + self.dt * v + eta)
                    m = (z - x_prev - eta) / self.dt
                    cond = self.condition(emb, anchor, m, x_prev)
                x0 = m + self.source_sigma * torch.randn_like(m)
                tau = torch.rand(len(c), 1, device=device)
                x_tau = (1.0 - tau) * x0 + tau * v
                pred = self.net(torch.cat([cond, x_tau, tau], dim=-1))
                loss = ((pred - (v - x0)) ** 2).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            sched.step()
            self._losses.append(float(np.mean(losses)))
            if epoch % print_every == 0 or epoch == 1:
                print(f"  [proposal] epoch {epoch:4d}/{epochs}  loss={self._losses[-1]:.5f}")
        self.net.eval()
        return self._losses

    # -------------------------------------------------------------- save/load

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(),
                    "config": {"n_embed": self.n_embed, "hidden": self.hidden, "N_FE": self.N_FE,
                               "source_sigma": self.source_sigma, "dt": self.dt}}, path)
        print(f"[CFMProposalModel] saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.n_embed, self.hidden, self.N_FE = int(cfg["n_embed"]), int(cfg["hidden"]), int(cfg["N_FE"])
        self.source_sigma, self.dt = float(cfg["source_sigma"]), float(cfg["dt"])
        self._build()
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        print(f"[CFMProposalModel] loaded ← {path}  (hidden={self.hidden}, N_FE={self.N_FE}, "
              f"source_sigma={self.source_sigma})")
