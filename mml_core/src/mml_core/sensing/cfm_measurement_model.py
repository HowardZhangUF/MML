"""
cfm_measurement_model.py
========================
CFMMeasurementModel — learned measurement density p_psi(z | x) for the
particle filter, built from conditional flow matching.

The model describes the residual r = (z - x) / scale given the position x.
Unlike the motion CFM it flows in the data -> noise direction:

    tau = 0: the (normalised) residual r
    tau = 1: standard normal eps
    training:  y_tau = cos(a) r + sin(a) eps,   a = pi tau / 2,
               u_target = (pi / 2) (cos(a) eps - sin(a) r)

The variance-preserving (trigonometric) interpolant is used instead of the
straight line: for a near-Gaussian residual its optimal field is close to
zero, so the 10-step Euler map is nearly exact. With the straight line and
independent coupling the path variance dips to 1/2 at tau = 1/2 and the
Euler error inflates the learned noise level (0.023 vs 0.020 m here).

so the Euler map T: r -> eps is evaluated forward and the filter gets the
exact density of that map with one pass per particle, no inversion:

    log p(z | x) = log N(T(r); 0, I) + log |det dT/dr| - 2 log(scale).

Sampling (used only offline, to train the proposal) inverts T by Newton
(flow_density.euler_inverse).

`scale` is the empirical residual standard deviation of the training pairs;
it is data normalisation, the noise level itself is learned.
"""

from __future__ import annotations

import math
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from mml_core.motion.flow_density import MLPField, euler_forward, euler_inverse, gaussian_log_prob


device = "cuda" if torch.cuda.is_available() else "cpu"


class CFMMeasurementModel:
    """
    Parameters
    ----------
    hidden:
        Width of the vector-field MLP.
    N_FE:
        Euler steps of the map T.
    bounds:
        ((xmin, xmax), (ymin, ymax)) used to normalise the conditioning position.
    """

    def __init__(self, hidden: int = 64, N_FE: int = 10,
                 bounds=((-2.5, 3.0), (-2.0, 2.0))):
        self.hidden = int(hidden)
        self.N_FE = int(N_FE)
        self.bounds = tuple(tuple(float(v) for v in b) for b in bounds)
        self.scale = 1.0
        self._build()
        self._losses: List[float] = []

    def _build(self):
        H = self.hidden
        # input: [x_norm (2), r_tau (2), tau (1)] — the MLPField layout
        self.net = nn.Sequential(
            nn.Linear(5, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
            nn.Linear(H, 2),
        ).to(device)

    def _norm_pos(self, x: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor([b[0] for b in self.bounds], dtype=x.dtype, device=x.device)
        hi = torch.tensor([b[1] for b in self.bounds], dtype=x.dtype, device=x.device)
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def field(self, x: torch.Tensor) -> MLPField:
        return MLPField(self.net, self._norm_pos(x))

    # ------------------------------------------------------------------ train

    def train(self, x: np.ndarray, z: np.ndarray, epochs: int = 300,
              batch_size: int = 4096, lr: float = 1e-3, print_every: int = 50) -> List[float]:
        """x: (N, 2) true positions; z: (N, 2) measurements of them."""
        x_t = torch.tensor(np.asarray(x, np.float32), device=device)
        z_t = torch.tensor(np.asarray(z, np.float32), device=device)
        self.scale = float((z_t - x_t).std())
        r_all = (z_t - x_t) / self.scale
        xn_all = self._norm_pos(x_t)

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
        n = len(x_t)
        self._losses = []
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n, device=device)
            losses = []
            for i in range(0, n - batch_size + 1, batch_size):
                idx = perm[i:i + batch_size]
                r, xn = r_all[idx], xn_all[idx]
                eps = torch.randn_like(r)
                tau = torch.rand(len(r), 1, device=device)
                a = 0.5 * math.pi * tau
                y = torch.cos(a) * r + torch.sin(a) * eps
                target = 0.5 * math.pi * (torch.cos(a) * eps - torch.sin(a) * r)
                pred = self.net(torch.cat([xn, y, tau], dim=-1))
                loss = ((pred - target) ** 2).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            sched.step()
            self._losses.append(float(np.mean(losses)))
            if epoch % print_every == 0 or epoch == 1:
                print(f"  [meas] epoch {epoch:4d}/{epochs}  loss={self._losses[-1]:.5f}")
        self.net.eval()
        return self._losses

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def log_prob(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """log p(z | x) for (B, 2) positions x and one (2,) or (B, 2) measurement z."""
        r = (z - x) / self.scale
        eps, logdet = euler_forward(self.field(x), r, self.N_FE)
        return gaussian_log_prob(eps, 0.0, 1.0) + logdet - 2.0 * math.log(self.scale)

    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """One measurement per position (Newton inversion of T)."""
        r, _, _ = euler_inverse(self.field(x), torch.randn_like(x), self.N_FE)
        return x + self.scale * r

    # -------------------------------------------------------------- save/load

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(),
                    "config": {"hidden": self.hidden, "N_FE": self.N_FE,
                               "bounds": self.bounds, "scale": self.scale}}, path)
        print(f"[CFMMeasurementModel] saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.hidden, self.N_FE = int(cfg["hidden"]), int(cfg["N_FE"])
        self.bounds, self.scale = tuple(map(tuple, cfg["bounds"])), float(cfg["scale"])
        self._build()
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        print(f"[CFMMeasurementModel] loaded ← {path}  (hidden={self.hidden}, "
              f"N_FE={self.N_FE}, scale={self.scale:.4f})")
