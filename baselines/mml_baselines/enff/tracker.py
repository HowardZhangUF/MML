"""
Ensemble Flow Filter (EnFF; "Flow Matching for Efficient and Scalable Data
Assimilation", arXiv:2508.13313) as a target tracker.

Upstream: github.com/Utah-Math-Data-Science/Data-Assimilation-Flow-Matching
@ ad51a8b. See README.md in this folder for provenance and deviations.

EnFF is training-free. Each analysis step builds the marginal velocity
field of a flow-matching probability path whose endpoint is the empirical
forecast ensemble (src/dafm/filters.py EnsembleFlowFilter.velocity), adds a
guidance term pulling the predicted endpoint toward high observation
likelihood (src/dafm/flow_matching_guidance.py Local), and Euler-integrates
the ensemble along it. Kept from upstream:
    * the FilteringToPredictive path (EnFF-F2P, the paper's main variant):
      mean_t = (1-t) x_prev + t x_pred, std = sigma_min, so x_0 = previous
      analysis ensemble + sigma_min noise (src/dafm/prob_path.py);
    * marginal velocity = softmax-weighted conditional velocities
      dt_std/std (x - mean) + dt_mean over the ensemble members;
    * Local guidance: x1_hat = x_t + (1 - t) v, guidance = -lambda * grad E(x1_hat),
      E = -log N(y; x1_hat, obs_std^2) (unnormalized);
    * Euler sampler over linspace(0, 1, sampling_time_step_count);
    * equal-weight ensemble, no resampling; inflation scale 1.
The forecast (upstream: the known dynamical system) is the benchmark's
constant-velocity model plus Gaussian process noise, as in PF+CV.
"""

from __future__ import annotations

import numpy as np

MAX_SPEED = 1.5   # m/s, the PF+CV speed clip


class EnFFTracker:
    def __init__(self, n_particles: int = 500, process_noise: float = 0.01, meas_sigma: float = 0.05,
                 sigma_min: float = 1e-3, guidance_lambda: float = 1e-3, sampling_time_step_count: int = 5,
                 dt: float = 0.033, history: int = 5):
        self.N = n_particles
        self.process_noise = process_noise
        self.meas_sigma = meas_sigma
        self.sigma_min = sigma_min
        self.lam = guidance_lambda
        self.path_time = np.linspace(0.0, 1.0, sampling_time_step_count)
        self.dt = dt
        self.N_th = history
        self.particles = None          # (N_th, N, 2) history of analysis ensembles

    # -- benchmark interface ---------------------------------------------------

    def reset(self, init_measurement, init_spread: float = 0.05):
        init = np.asarray(init_measurement, dtype=np.float32)
        base = init[None] + init_spread * np.random.randn(self.N, 2).astype(np.float32)
        self.particles = np.repeat(base[None], self.N_th, axis=0)

    def cloud(self):
        return self.particles[-1].copy(), np.full(self.N, 1.0 / self.N)

    def get_state(self):
        return self.particles.copy()

    def set_state(self, state):
        self.particles = state

    def step(self, measurement):
        prev = self.particles[-1]
        pred = self._forecast()
        post = pred if measurement is None else self._assimilate(
            prev.astype(np.float64), pred.astype(np.float64), np.asarray(measurement, dtype=np.float64))
        self.particles[:-1] = self.particles[1:]
        self.particles[-1] = post
        return post.mean(axis=0)

    # -- filter ------------------------------------------------------------------

    def _forecast(self):
        v = (self.particles[-1] - self.particles[-2]) / self.dt
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * np.minimum(1.0, MAX_SPEED / (speed + 1e-9))
        return (self.particles[-1] + v * self.dt
                + self.process_noise * np.random.randn(self.N, 2)).astype(np.float32)

    def _assimilate(self, prev, pred, y):
        """EnsembleFlowFilter.assimilate with the F2P path and Local guidance."""
        s = self.sigma_min
        x = prev + s * np.random.randn(*prev.shape)               # F2P sample_noise at t = 0
        dt_means = pred - prev                                    # F2P dt_mean; dt_std = 0
        for t_now, t_next in zip(self.path_time[:-1], self.path_time[1:]):
            means = (1 - t_now) * prev + t_now * pred
            # |x_j - mean_i|^2 for every (x member j, path member i)
            d2 = (x ** 2).sum(1)[:, None] + (means ** 2).sum(1)[None, :] - 2.0 * x @ means.T
            logw = -0.5 * np.maximum(d2, 0.0) / s ** 2
            w = np.exp(logw - logw.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            velocity = w @ dt_means                               # sum_i w_i (0 * (x - mean_i) + dt_mean_i)
            x1_hat = x + (1 - t_now) * velocity
            grad_energy = (x1_hat - y) / self.meas_sigma ** 2
            x = x + (t_next - t_now) * (velocity - self.lam * grad_energy)
        return x.astype(np.float32)
