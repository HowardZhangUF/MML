"""
observation_proposal.py — M3-latent-Gaussian (guide chapter 6).

The frozen CFM is a deterministic map of its standard-normal input xi:

    G(C, xi) = clip( Euler_{N_FE}( a(C) + sigma0 * xi ) ),   clip = MAX_SPEED rescale
    F_C(xi, eta) = x_{k-1} + dt * G(C, xi) + eta,             eta ~ N(0, q^2 I)

With z_k the filter draws xi ~ q(xi | C, z) = N(b, L L^T) instead of N(0, I)
and keeps eta's prior, so the augmented-space weight is

    w_k  ∝  w_{k-1} * l(z_k | x_k) * phi(xi) / q(xi | C, z_k)

(eta's density cancels; no Jacobian or inverse of G is needed). The clip is
part of G on both the prior and the proposal path, so it does not affect
correctness. Without z_k the prior sampler runs unchanged.

Proposal network: MLP on [e(C), a(C), G(C,0), (z - x_{k-1} - dt G(C,0)) / sigma_R]
-> b (2), lower-triangular L with softplus(.) + floor diagonal. Trained by
maximum likelihood on (xi, z) drawn from the frozen model with the filter's
observation model z = x + sigma_R * eps, i.e. q approximates p(xi | C, z).

M3-joint (dim = 4, an extension beyond the guide's first implementation):
with q = 0.045 m the position noise eta dominates the CFM's one-step spread
(dt * std(v) ~ 0.02 m), so z_k mostly informs eta. M3-joint proposes
u = (xi, eta / q) ~ N(b, L L^T) and weights by phi_4(u) / q(u | C, z_k), the
guide's "include eta's prior/proposal ratio" case.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mml_research.motion.cfm_variants.cfm_nets import MAX_SPEED, device

LOG_2PI = math.log(2.0 * math.pi)


def std_normal_log_prob(xi: torch.Tensor) -> torch.Tensor:
    return -0.5 * (xi ** 2).sum(-1) - 0.5 * xi.shape[-1] * LOG_2PI


def mvn_tril_log_prob(x: torch.Tensor, mean: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    """log N(x; mean, L L^T) for batched lower-triangular L (D = 2 written out, so N(0, I) gives exactly 0)."""
    r = x - mean
    D = x.shape[-1]
    if D == 2:
        y0 = r[:, 0] / chol[:, 0, 0]
        y1 = (r[:, 1] - chol[:, 1, 0] * y0) / chol[:, 1, 1]
        return -0.5 * (y0 ** 2 + y1 ** 2) - torch.log(chol[:, 0, 0] * chol[:, 1, 1]) - LOG_2PI
    y = torch.linalg.solve_triangular(chol, r.unsqueeze(-1), upper=False).squeeze(-1)
    logdet = torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return -0.5 * (y ** 2).sum(-1) - logdet - 0.5 * D * LOG_2PI


def _euler(cfm, emb, x):
    delta = 1.0 / cfm.N_FE
    for s in range(cfm.N_FE):
        tau = torch.full((x.shape[0], 1), s * delta, dtype=torch.float32, device=x.device)
        x = x + delta * cfm.network.predict_velocity(x, tau, emb)
    return x


def _clip_torch(v):
    speed = v.norm(dim=1, keepdim=True)
    return v * torch.clamp(MAX_SPEED / (speed + 1e-9), max=1.0)


@torch.no_grad()
def cfm_generate_from_base(cfm, emb, anchor, xi) -> np.ndarray:
    """G(C, xi) as numpy; identical arithmetic to CFMMotionModel.propose given xi = its randn draw."""
    x = _euler(cfm, emb, anchor + cfm.anchor_sigma * xi)
    v = x.cpu().numpy().astype(np.float32)
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.minimum(1.0, MAX_SPEED / (speed + 1e-9))


class LatentGaussianProposal:
    """
    dim = 2  M3 (guide 6.2): proposes xi only; eta keeps its prior.
    dim = 4  M3-joint (extension): proposes u = (xi, eta / q) jointly, prior N(0, I_4);
             the weight is phi_4(u) / q(u | C, z), i.e. phi(xi) r_Q(eta) / q(xi, eta) in
             the standardized coordinates (the eta Jacobian 1/q^2 cancels in the ratio).
    """

    def __init__(self, n_embed: int = 32, hidden: int = 128, scale_floor: float = 0.05,
                 meas_sigma: float = 0.05, dim: int = 2):
        self.n_embed, self.hidden, self.dim = int(n_embed), int(hidden), int(dim)
        self.scale_floor, self.meas_sigma = float(scale_floor), float(meas_sigma)
        D, H = self.dim, self.hidden
        self._tril = torch.tril_indices(D, D, offset=-1)
        self.net = nn.Sequential(
            nn.Linear(self.n_embed + 6, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
            nn.Linear(H, 2 * D + self._tril.shape[1]),
        ).to(device)
        # Start at the prior N(0, I): zero mean, softplus^-1(1 - floor) diagonal, zero off-diagonal.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.zero_()
            last.bias[D:2 * D] = math.log(math.expm1(1.0 - self.scale_floor))
        self.info = {}

    def features(self, cfm, emb, anchor, x_prev, z):
        ref = _clip_torch(_euler(cfm, emb, anchor))          # G(C, 0)
        innov = (z - x_prev - cfm.dt * ref) / self.meas_sigma
        return torch.cat([emb, anchor, ref, innov], dim=-1)

    def head(self, feats):
        D = self.dim
        out = self.net(feats)
        mean = out[:, :D]
        diag = F.softplus(out[:, D:2 * D]) + self.scale_floor
        chol = torch.diag_embed(diag)
        chol[:, self._tril[0], self._tril[1]] = out[:, 2 * D:]
        return mean, chol

    @torch.no_grad()
    def distribution(self, cfm, emb, anchor, x_prev, z):
        """(mean, chol, log_prob_fn) of q(u | C, z) for each particle."""
        mean, chol = self.head(self.features(cfm, emb, anchor, x_prev, z))
        return mean, chol, lambda u: mvn_tril_log_prob(u, mean, chol)

    def fit(self, cfm, contexts: np.ndarray, holdout: np.ndarray, process_noise: float,
            n_steps: int, batch: int, lr: float, seed: int, log_every: int = 1000):
        torch.manual_seed(seed)
        cfm.network.eval()
        C = torch.tensor(np.asarray(contexts, np.float32), device=device)
        tr = torch.tensor(np.flatnonzero(~holdout), device=device)
        ho = torch.tensor(np.flatnonzero(holdout)[:20000], device=device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.01)

        def nll(idx):
            c = C[idx]
            with torch.no_grad():
                emb = cfm.network.encode_context(c)
                anchor = cfm._finite_difference_velocity(c)
                x_prev = c[:, -2:]
                u = torch.randn(len(c), self.dim, device=device)
                v = _clip_torch(_euler(cfm, emb, anchor + cfm.anchor_sigma * u[:, :2]))
                eta = process_noise * (u[:, 2:] if self.dim == 4 else torch.randn_like(v))
                z = x_prev + cfm.dt * v + eta + self.meas_sigma * torch.randn_like(v)
                feats = self.features(cfm, emb, anchor, x_prev, z)
            mean, chol = self.head(feats)
            return -mvn_tril_log_prob(u, mean, chol).mean()

        prior_nll = 0.5 * self.dim * (1 + LOG_2PI)
        self.net.train()
        tr_hist, ho_hist, run, t0 = [], [], [], time.time()
        for step in range(1, n_steps + 1):
            loss = nll(tr[torch.randint(len(tr), (batch,), device=device)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run.append(loss.item())
            if step % log_every == 0 or step == n_steps:
                with torch.no_grad():
                    ho_loss = float(np.mean([nll(ho).item() for _ in range(4)])) if len(ho) else math.nan
                tr_hist.append(float(np.mean(run)))
                ho_hist.append(ho_loss)
                print(f"    step {step:6d}/{n_steps}  nll={tr_hist[-1]:.4f}  holdout={ho_loss:.4f}  "
                      f"(prior {prior_nll:.3f}, {time.time() - t0:.0f}s)", flush=True)
                run = []
        self.net.eval()
        self.info = {"seed": seed, "process_noise": process_noise, "prior_nll": prior_nll,
                     "train_nll": tr_hist, "holdout_nll": ho_hist}
        return tr_hist, ho_hist

    def save(self, path):
        torch.save({"state_dict": self.net.state_dict(),
                    "config": {"kind": "m3_latent_gaussian", "dim": self.dim, "n_embed": self.n_embed,
                               "hidden": self.hidden, "scale_floor": self.scale_floor,
                               "meas_sigma": self.meas_sigma,
                               "latent": "xi" if self.dim == 2 else "(xi, eta / q)",
                               "features": "[e(C), a(C), G(C,0), (z - x_prev - dt G(C,0)) / meas_sigma]"},
                    "info": self.info}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location=device, weights_only=False)
        c = ck["config"]
        m = cls(c["n_embed"], c["hidden"], c["scale_floor"], c["meas_sigma"], c.get("dim", 2))
        m.net.load_state_dict(ck["state_dict"])
        m.net.eval()
        m.info = ck.get("info", {})
        return m


class IdentityProposal:
    """q = N(0, I): the bootstrap-equivalence check of guide 6.6."""

    dim = 2

    @torch.no_grad()
    def distribution(self, cfm, emb, anchor, x_prev, z):
        mean = torch.zeros_like(anchor)
        chol = torch.eye(2, device=anchor.device).expand(len(anchor), 2, 2)
        return mean, chol, lambda xi: mvn_tril_log_prob(xi, mean, chol)
