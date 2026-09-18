"""
particle_filter.py
==================
CFMParticleFilter — lightweight bootstrap particle filter used in the
Jupyter notebook tutorial and for standalone testing.

This is NOT the production ParticleFilter.py (mml_guidance/ParticleFilter.py).
It is kept in mml_network/ purely for notebook use and unit tests.
For the pygame / ROS integration see:
    mml_guidance/mml_guidance/ParticleFilter.py   (modified with elif "CFM": block)

Place at:
    mml_guidance/mml_guidance/mml_network/particle_filter.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from mml_core.motion.cfm_motion_model import CFMMotionModel
from mml_core.motion.utils import HistoryBuffer


# ── ParticleCloud data class ──────────────────────────────────────────────────

@dataclass
class ParticleCloud:
    """
    Particle state at one time step.

    positions  : (N, 2)   current (x, y) per particle
    log_weights: (N,)     unnormalised log importance weights
    """
    positions:   np.ndarray   # (N, 2)
    log_weights: np.ndarray   # (N,)

    @property
    def N(self) -> int:
        return self.positions.shape[0]

    @property
    def weights(self) -> np.ndarray:
        """Numerically stable normalised weights."""
        lw = self.log_weights - self.log_weights.max()
        w  = np.exp(lw)
        return w / (w.sum() + 1e-300)

    @property
    def ess(self) -> float:
        """Effective Sample Size  =  1 / sum(w_i^2)."""
        w = self.weights
        return float(1.0 / ((w ** 2).sum() + 1e-300))

    @property
    def mean(self) -> np.ndarray:
        """Weighted mean position. Shape: (2,)"""
        return (self.weights[:, None] * self.positions).sum(axis=0)

    @property
    def covariance(self) -> np.ndarray:
        """Weighted 2×2 position covariance."""
        w    = self.weights
        mu   = self.mean
        diff = self.positions - mu
        return (w[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(0)


# ── CFMParticleFilter ─────────────────────────────────────────────────────────

class CFMParticleFilter:
    """
    Bootstrap particle filter using CFMMotionModel as the motion model.

    Algorithm per step
    ------------------
    1. Push measurement into shared HistoryBuffer.
    2. Predict: build context window → CFM propose() → (N, 2) next positions.
    3. Update : Gaussian observation likelihood → log-weight update.
    4. Resample if ESS < threshold.

    Parameters
    ----------
    motion_model  : trained CFMMotionModel instance
    n_particles   : number of particles N
    sigma_obs     : measurement noise std [m]  (sets observation likelihood width)
    ess_threshold : resample when ESS < this  (default N/2)
    history_len   : N_th — must match block_size used to train the CFM
    input_dim     : 2 for [x,y], 3 for [x,y,dt]
    """

    def __init__(self,
                 motion_model:  CFMMotionModel,
                 n_particles:   int   = 200,
                 sigma_obs:     float = 0.10,
                 ess_threshold: Optional[float] = None,
                 history_len:   int   = 5,
                 input_dim:     int   = 4):

        self.model      = motion_model
        self.N          = n_particles
        self.sigma_obs  = sigma_obs
        self.ess_thresh = ess_threshold if ess_threshold else n_particles / 2
        self.history_len = history_len
        self.input_dim   = input_dim

        # Shared measurement history — same window for all particles.
        # Simplification: in the production PF each particle has its own
        # history slice from the full particles array.
        self.history = HistoryBuffer(capacity=history_len, dim=input_dim)

        self.cloud: Optional[ParticleCloud] = None
        self._estimates: List[np.ndarray]   = []
        self._ess_log:   List[float]         = []

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialise(self, init_pos: np.ndarray, spread: float = 0.3) -> None:
        """
        Scatter N particles in a Gaussian cloud around init_pos.

        Parameters
        ----------
        init_pos : (2,)  [x, y] initial estimate
        spread   : std of the initial scatter  [m]
        """
        positions  = init_pos + np.random.randn(self.N, 2) * spread
        self.cloud = ParticleCloud(
            positions   = positions.astype(np.float32),
            log_weights = np.zeros(self.N, dtype=np.float32),
        )
        self._estimates = []
        self._ess_log   = []
        self.history.reset()
        print(f"[CFMParticleFilter] Initialised {self.N} particles at {init_pos}")

    # ── Main update ───────────────────────────────────────────────────────────

    def step(self,
             measurement: Optional[np.ndarray],
             timestamp:   Optional[float] = None) -> np.ndarray:
        """
        Process one time step.

        Parameters
        ----------
        measurement : (2,) noisy [x, y], or None if target is occluded
        timestamp   : wall-clock time [s]; only used when input_dim=3

        Returns
        -------
        estimate : (2,)  weighted mean position
        """
        assert self.cloud is not None, "Call initialise() first."

        # ── 1. Update history (only when visible) ────────────────────────────
        if measurement is not None:
            if self.input_dim == 3 and timestamp is not None:
                obs = np.array([measurement[0], measurement[1], timestamp],
                               dtype=np.float32)
            else:
                obs = np.asarray(measurement[:2], dtype=np.float32)
            self.history.push(obs)

        # ── 2. Predict — CFM ODE propagation ─────────────────────────────────
        if self.history.is_ready():
            # Same context for all particles (shared history simplification)
            c_flat  = self.history.flat()                        # (N_th * 2,)
            c_batch = np.tile(c_flat, (self.N, 1))              # (N, N_th*2)
            # Each particle gets a DIFFERENT x0 inside propose() → multi-modal
            velocities               = self.motion_model_propose(c_batch)  # (N, 2)
            # Integrate one physical step:  x_{k+1} = x_k + v * dt
            # dt implicit in the trained velocity scale
            self.cloud.positions    += velocities
        else:
            # Not enough history yet — small random spread to keep particles alive
            self.cloud.positions    += np.random.randn(self.N, 2) * 0.02

        # ── 3. Weight update — Gaussian observation likelihood ────────────────
        if measurement is not None:
            z    = np.asarray(measurement[:2], dtype=np.float32)
            diff = z - self.cloud.positions            # (N, 2)
            maha = (diff ** 2).sum(axis=1)             # (N,)
            self.cloud.log_weights += -0.5 / (self.sigma_obs ** 2) * maha
            # Stabilise: subtract max before exp (log-sum-exp trick)
            self.cloud.log_weights -= self.cloud.log_weights.max()

        # ── 4. ESS + systematic resampling ────────────────────────────────────
        ess = self.cloud.ess
        self._ess_log.append(ess)
        if ess < self.ess_thresh:
            self.cloud = _systematic_resample(self.cloud)

        estimate = self.cloud.mean
        self._estimates.append(estimate.copy())
        return estimate

    def motion_model_propose(self, c_batch: np.ndarray) -> np.ndarray:
        """Delegate to CFMMotionModel.propose()."""
        return self.model.propose(c_batch)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def estimates(self) -> np.ndarray:
        """All position estimates so far. Shape: (T, 2)"""
        return np.array(self._estimates)

    @property
    def ess_log(self) -> np.ndarray:
        """ESS at each time step. Shape: (T,)"""
        return np.array(self._ess_log)


# ── Systematic resampling ─────────────────────────────────────────────────────

def _systematic_resample(cloud: ParticleCloud) -> ParticleCloud:
    """
    O(N) systematic resampling.
    Draws N particle indices from the weighted distribution,
    then resets all log-weights to 0 (uniform).
    """
    N      = cloud.N
    w      = cloud.weights                     # normalised
    cumsum = np.cumsum(w)
    u0     = np.random.uniform(0.0, 1.0 / N)
    us     = u0 + np.arange(N) / N            # N evenly-spaced points in [0,1)
    idxs   = np.clip(np.searchsorted(cumsum, us), 0, N - 1)

    return ParticleCloud(
        positions   = cloud.positions[idxs].copy(),
        log_weights = np.zeros(N, dtype=np.float32),
    )