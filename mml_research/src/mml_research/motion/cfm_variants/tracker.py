"""
tracker.py — the particle filter shared by every variant (guide 2.3, 3.3).

VariantPFTracker is BootstrapPFTracker with a pluggable sampler:

    positions, log_correction = sampler.predict(particles, ctx, z)

ctx carries only information through step k-1 (guide 2.4): the history mask,
time since the last measurement and the weighted-cloud covariance summary,
all computed after the update at k-1. The sampler never adds the measurement
likelihood; the tracker does, once.

    log_correction is None  the sampler drew from the declared transition;
                            the update is BootstrapPFTracker's, operation for
                            operation, so PF+CFM / PF+DMMN / PF+NF reproduce
                            the old benchmark exactly (verify.py checks it)
    log_correction given    log-space update  w *= l(z|x) exp(log_correction)

ESS before resampling, the resample flag and numerical failures are logged
for every measurement update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mml_research.motion.cfm_variants.features import cloud_summary


@dataclass
class Context:
    """Past information available before processing z_k (identical for every particle)."""
    mask: np.ndarray      # (H,) 1 = a measurement was available and used at that history step
    d: float              # seconds since the last used measurement
    U: np.ndarray         # (3,) log-variances and correlation of the weighted cloud at k-1
    steps: int            # steps since the last used measurement


class VariantPFTracker:
    def __init__(self, sampler, n_particles: int = 500, meas_sigma: float = 0.05,
                 init_spread: float = 0.05):
        self.sampler = sampler
        self.N = n_particles
        self.H = sampler.block_size
        self.dt = sampler.dt
        self.meas_sigma = meas_sigma
        self.init_spread = init_spread
        self.particles = None
        self.weights = None

    # ------------------------------------------------------------------ state

    def reset(self, init_measurement: np.ndarray):
        init = np.asarray(init_measurement, dtype=np.float32)
        base = init[None, :] + self.init_spread * np.random.randn(self.N, 2).astype(np.float32)
        self.particles = np.repeat(base[None, :, :], self.H, axis=0)
        self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)
        # The initial history replicates the first measurement, so every slot counts as observed.
        self.mask = np.ones(self.H, dtype=np.float32)
        self.steps_since_meas = 0
        self.U = cloud_summary(self.particles[-1], self.weights)
        self.ess_pre, self.resampled = [], []
        self.numerical_failures = 0
        if hasattr(self.sampler, "reset"):
            self.sampler.reset()

    def context(self) -> Context:
        return Context(self.mask.copy(), self.steps_since_meas * self.dt, self.U.copy(),
                       self.steps_since_meas)

    def get_state(self):
        return (self.particles.copy(), self.weights.copy(), self.mask.copy(), self.steps_since_meas,
                self.U.copy(), len(self.ess_pre), self.numerical_failures)

    def set_state(self, s):
        (self.particles, self.weights, self.mask, self.steps_since_meas, self.U, n_log,
         self.numerical_failures) = s
        del self.ess_pre[n_log:], self.resampled[n_log:]

    def cloud(self):
        return self.particles[-1].copy(), self.weights.copy()

    # ----------------------------------------------------------------- update

    def _resample_if_needed(self):
        n_eff = 1.0 / np.sum(self.weights ** 2)
        self.ess_pre.append(n_eff / self.N)
        do = n_eff < self.N / 2
        self.resampled.append(bool(do))
        if do:
            positions = (np.arange(self.N) + np.random.uniform()) / self.N
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0
            idx = np.searchsorted(cumsum, positions)
            # Whole histories follow their ancestor; mask / d / U are cloud-level.
            self.particles = self.particles[:, idx, :]
            self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)

    def _update(self, z: np.ndarray, log_correction):
        d2 = np.sum((self.particles[-1] - z[None, :]) ** 2, axis=1)
        if log_correction is None:
            # BootstrapPFTracker._update, unchanged.
            lik = np.exp(-0.5 * (d2 - d2.min()) / self.meas_sigma ** 2) + 1e-300
            self.weights *= lik
            w_sum = self.weights.sum()
            if not np.isfinite(w_sum) or w_sum <= 0.0:
                self.numerical_failures += 1
                self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)
            else:
                self.weights /= w_sum
        else:
            logw = (np.log(np.maximum(self.weights, 1e-300))
                    + np.asarray(log_correction, dtype=np.float64)
                    - 0.5 * d2.astype(np.float64) / self.meas_sigma ** 2)
            bad = ~np.isfinite(logw)
            if bad.any():
                self.numerical_failures += 1
                logw[bad] = -np.inf
            if not np.isfinite(logw).any():
                self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)
            else:
                w = np.exp(logw - logw.max())
                self.weights = w / w.sum()
        self._resample_if_needed()

    def step(self, measurement):
        ctx = self.context()
        z = None if measurement is None else np.asarray(measurement, dtype=np.float32)
        new_pos, log_correction = self.sampler.predict(self.particles, ctx, z)
        self.particles[:-1] = self.particles[1:]
        self.particles[-1] = new_pos
        self.mask[:-1] = self.mask[1:]
        if z is not None:
            self._update(z, log_correction)
            self.steps_since_meas = 0
            self.mask[-1] = 1.0
        else:
            if log_correction is not None:
                raise RuntimeError("a sampler returned a weight correction without a measurement")
            self.steps_since_meas += 1
            self.mask[-1] = 0.0
        self.U = cloud_summary(self.particles[-1], self.weights)
        return np.average(self.particles[-1], axis=0, weights=self.weights)


# ── Samplers ────────────────────────────────────────────────────────────────

def flat_context(particles: np.ndarray) -> np.ndarray:
    """(H, N, 2) particle histories -> (N, H*2), BootstrapPFTracker's layout."""
    H, N, _ = particles.shape
    return particles.transpose(1, 0, 2).reshape(N, H * 2)


class MotionSampler:
    """
    Bootstrap proposal = the declared transition
        v ~ p(v | C[, ctx]),  x_k = x_{k-1} + dt v + q eps.
    Models with `uses_context = True` receive ctx (M1); others only C.
    Operation order matches BootstrapPFTracker._predict.
    """

    def __init__(self, model, process_noise: float):
        self.model = model
        self.q = float(process_noise)
        self.block_size = model.block_size
        self.dt = model.dt

    def velocities(self, particles, ctx):
        c_flat = flat_context(particles)
        if getattr(self.model, "uses_context", False):
            return self.model.propose_with_context(c_flat, ctx)
        return self.model.propose(c_flat)

    def predict(self, particles, ctx, z):
        v = self.velocities(particles, ctx)
        N = particles.shape[1]
        new_pos = (particles[-1] + v * self.model.dt
                   + self.q * np.random.randn(N, 2).astype(np.float32))
        return new_pos, None


class AdaptiveSigmaSampler(MotionSampler):
    """PF+CFM-AS: CFM source std grows with occlusion length (BootstrapPFTracker's rule)."""

    def __init__(self, cfm, process_noise: float, gain: float, sigma_max: float):
        super().__init__(cfm, process_noise)
        self.base_sigma = cfm.anchor_sigma
        self.gain, self.sigma_max = gain, sigma_max

    def velocities(self, particles, ctx):
        self.model.anchor_sigma = min(self.sigma_max, self.base_sigma * (1.0 + self.gain * ctx.steps))
        try:
            return super().velocities(particles, ctx)
        finally:
            self.model.anchor_sigma = self.base_sigma


class LatentProposalSampler(MotionSampler):
    """
    M3: guide 6.2-6.5. Without a measurement this is exactly the prior sampler
    of the frozen CFM (same generator, same random draws). With z_k the CFM's
    standard-normal input is drawn from q(xi | C, z) and the sampler returns
    log phi(xi) - log q(xi | C, z); the position noise keeps its prior.
    M3-joint (proposal.dim == 4) also proposes eta / q and returns
    log phi_4(xi, eta/q) - log q(xi, eta/q | C, z).
    """

    def __init__(self, cfm, proposal, process_noise: float):
        super().__init__(cfm, process_noise)
        self.proposal = proposal

    @torch.no_grad()
    def predict(self, particles, ctx, z):
        if z is None:
            return super().predict(particles, ctx, z)
        from mml_research.motion.cfm_variants.observation_proposal import (
            cfm_generate_from_base, std_normal_log_prob,
        )
        cfm = self.model
        dev = next(cfm.network.parameters()).device
        c = torch.tensor(flat_context(particles), dtype=torch.float32, device=dev)
        emb = cfm.network.encode_context(c)
        anchor = cfm._finite_difference_velocity(c)
        x_prev = c[:, -2:]
        z_t = torch.tensor(z, dtype=torch.float32, device=dev).expand_as(x_prev)
        mean, chol, log_q_fn = self.proposal.distribution(cfm, emb, anchor, x_prev, z_t)
        u = mean + torch.einsum("bij,bj->bi", chol, torch.randn_like(mean))
        log_corr = std_normal_log_prob(u) - log_q_fn(u)
        v = cfm_generate_from_base(cfm, emb, anchor, u[:, :2])
        N = particles.shape[1]
        if self.proposal.dim == 4:      # M3-joint: eta = q * u[2:] was proposed too
            eta = self.q * u[:, 2:].cpu().numpy().astype(np.float32)
        else:                           # M3: eta keeps its prior
            eta = self.q * np.random.randn(N, 2).astype(np.float32)
        new_pos = particles[-1] + v * cfm.dt + eta
        return new_pos, log_corr.double().cpu().numpy()
