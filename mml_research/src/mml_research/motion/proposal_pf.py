"""
proposal_pf.py
==============
ProposalPFTracker — PF+CFM with a learned measurement model and an
observation-conditioned CFM proposal (see mml_core.motion.cfm_proposal_model
for the model and the weight derivation).

Steps without a measurement are exactly BootstrapPFTracker's (CFM motion
model + process noise), so occluded prediction — and the exit-prediction
forks of benchmark_exit_prediction — use the unchanged motion model. Only the
measurement update differs:

    proposal=None    bootstrap proposal, learned likelihood:
                     v ~ f_theta (clipped sampler), w *= p_psi(z | x)
    proposal given   v ~ q_phi(v | C, z), eta ~ N(0, q^2):
                     w *= p_psi(z | x) f_theta(v | C) / q_phi(v | C, z)
    weighting="naive"  proposal sampling with likelihood-only weights —
                       the uncorrected variant, kept as a diagnostic only.

f_theta here is the unclipped Euler sampler, whose density is exact; the
MAX_SPEED clip of CFMMotionModel.propose() has no density, so it is only used
on steps without a measurement.
"""

from __future__ import annotations

import numpy as np
import torch

from mml_research.motion.benchmark_diffpf_vs_cfm import BootstrapPFTracker


class ProposalPFTracker(BootstrapPFTracker):
    def __init__(self, motion_model, meas_model, proposal=None, weighting: str = "correct",
                 **kwargs):
        super().__init__(motion_model, **kwargs)
        if weighting not in ("correct", "naive"):
            raise ValueError(f"weighting must be 'correct' or 'naive', got {weighting!r}")
        self.meas_model = meas_model
        self.proposal = proposal
        self.weighting = weighting
        self.ess_history = []          # ESS / N after each measurement update
        self.max_newton_residual = 0.0

    def _device(self):
        return next(self.motion_model.network.parameters()).device

    def _resample_from_log(self, logw: np.ndarray):
        logw = np.where(np.isfinite(logw), logw, -np.inf)
        if not np.isfinite(logw).any():
            self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)
        else:
            w = np.exp(logw - logw.max())
            self.weights = w / w.sum()
        n_eff = 1.0 / np.sum(self.weights ** 2)
        self.ess_history.append(n_eff / self.N)
        if n_eff < self.N / 2:
            positions = (np.arange(self.N) + np.random.uniform()) / self.N
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0
            idx = np.searchsorted(cumsum, positions)
            self.particles = self.particles[:, idx, :]
            self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)

    @torch.no_grad()
    def _measurement_step(self, z: np.ndarray):
        dev = self._device()
        z_t = torch.tensor(z, dtype=torch.float32, device=dev)
        log_prev = np.log(np.maximum(self.weights, 1e-300))

        if self.proposal is None:
            self._predict()
            x = torch.tensor(self.particles[-1], device=dev)
            log_inc = self.meas_model.log_prob(z_t, x).cpu().numpy().astype(np.float64)
            self._resample_from_log(log_prev + log_inc)
            return

        mm = self.motion_model
        c = torch.tensor(self.particles.transpose(1, 0, 2).reshape(self.N, self.N_th * 2),
                         device=dev)
        emb = mm.network.encode_context(c)
        anchor = mm._finite_difference_velocity(c)
        x_prev = c[:, -2:]
        eta = self.process_noise * torch.randn_like(x_prev)
        m = (z_t - x_prev - eta) / mm.dt
        v, log_q = self.proposal.sample_with_log_prob(
            self.proposal.condition(emb, anchor, m, x_prev), m)
        x = x_prev + mm.dt * v + eta
        log_lik = self.meas_model.log_prob(z_t, x)
        if self.weighting == "correct":
            log_f, resid = mm.log_prob(c, v, emb)
            self.max_newton_residual = max(self.max_newton_residual, resid)
            log_inc = log_lik + log_f - log_q
        else:
            log_inc = log_lik

        self.particles[:-1] = self.particles[1:]
        self.particles[-1] = x.cpu().numpy()
        self._resample_from_log(log_prev + log_inc.cpu().numpy().astype(np.float64))

    def step(self, measurement):
        if measurement is None:
            return super().step(None)
        self._measurement_step(np.asarray(measurement, dtype=np.float32))
        self.steps_since_meas = 0
        return np.average(self.particles[-1], axis=0, weights=self.weights)
