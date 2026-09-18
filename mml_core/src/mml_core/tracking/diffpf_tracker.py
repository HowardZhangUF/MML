"""
diffpf_tracker.py
==================

Adaptation of DiffPF (Differentiable Particle Filtering with Generative
Sampling via Conditional Diffusion Models) to the 2D occluded-target
tracking task, as a filter-level SOTA baseline against the classical
particle filter + CFM motion model.

What is kept from DiffPF (the paper's mechanism)
------------------------------------------------
* A conditional DDPM samples a fresh, equally-weighted particle set from
  the posterior at every step — replacing the classical likelihood-weight
  + resample update entirely.
* Conditioning = (encoded current observation, encoded predicted particle
  set from the previous step).
* The prediction step is deliberately simple (their code: odometry + noise;
  here: constant-velocity propagation + noise) — all learned capacity sits
  in the posterior sampler.
* num_diffusion_iters = 10, matching their released nav01 config.

What is adapted (and must be reported as such in any paper)
-----------------------------------------------------------
* Their observation is a 24x24 RGB image encoded by a CNN. Our observation
  is a noisy 2D position when the target is visible and NOTHING when
  occluded, so the vision encoder is replaced by a small MLP on
  (x, y, visible_flag), with (0, 0, 0) as the null token during occlusion.
* Their prediction step uses the robot's own odometry (ground-truth
  displacement between frames). Our target is uncontrolled, so the
  analogous "dumb prediction" is constant-velocity propagation from the
  filter's own past estimates.
* Their shipped config (config/nav01_step1.yaml -> training_setting "step1")
  runs training/train_step1.py, which zeroes the particle conditioning
  vector (predict_particles.zero_(), lines 415/518); only the step-2 stage
  (training/train_step2.py) keeps it. Both stages pass particles across
  batches through a file on disk (particles.pt). We implement the recursive
  step-2 mechanism with particle conditioning enabled and the filter state
  held in memory. See BASELINE_PROVENANCE.md for the full mapping.

Place at:
    mml_guidance/mml_guidance/mml_network/diffpf_tracker.py
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mml_core.motion.diffusion_motion_model import _cosine_beta_schedule
from mml_core.motion.velocity_network import _SinusoidalTimestepEmbedding


device = "cuda" if torch.cuda.is_available() else "cpu"


class DiffPFNetwork(nn.Module):
    """
    Conditional noise-prediction network for the DiffPF posterior sampler.

    obs_cond = fuse( meas_encoder(x, y, visible), particle_encoder(flat particles) )
    eps_pred = noise_head( y_t, t_emb, obs_cond )

    All positions are fed in NORMALIZED coordinates (see DiffPFTracker).
    """

    def __init__(self, n_cond_particles: int = 20, enc_dim: int = 32,
                 cond_dim: int = 64, hidden_dim: int = 128,
                 time_embed_dim: int = 32):
        super().__init__()
        self.n_cond_particles = int(n_cond_particles)

        self.meas_encoder = nn.Sequential(
            nn.Linear(3, enc_dim), nn.ReLU(),
            nn.Linear(enc_dim, enc_dim),
        )
        self.particle_encoder = nn.Sequential(
            nn.Linear(self.n_cond_particles * 2, enc_dim), nn.ReLU(),
            nn.Linear(enc_dim, enc_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * enc_dim, cond_dim), nn.ReLU(),
        )

        self.time_embed = _SinusoidalTimestepEmbedding(time_embed_dim)
        self.noise_head = nn.Sequential(
            nn.Linear(2 + time_embed_dim + cond_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode_condition(self, meas: torch.Tensor,
                          particles_flat: torch.Tensor) -> torch.Tensor:
        """
        meas:            (B, 3)   normalized (x, y, visible_flag); (0,0,0) if occluded
        particles_flat:  (B, n_cond_particles * 2)   normalized predicted particles
        returns:         (B, cond_dim)
        """
        m = self.meas_encoder(meas)
        p = self.particle_encoder(particles_flat)
        return self.fusion(torch.cat([m, p], dim=-1))

    def predict_noise(self, y_t: torch.Tensor, t: torch.Tensor,
                       cond: torch.Tensor) -> torch.Tensor:
        """y_t: (B,2), t: (B,) long, cond: (B, cond_dim) -> (B, 2)"""
        t_emb = self.time_embed(t)
        return self.noise_head(torch.cat([y_t, t_emb, cond], dim=-1))


class DiffPFTracker:
    """
    DiffPF-style tracker for the 2D occluded-target task.

    Filter loop per time step:
      1. Constant-velocity propagate the particle set (+ noise).
      2. Encode conditioning from (measurement or null token, a fixed-size
         subsample of the propagated particles).
      3. Sample a fresh particle set from the conditional DDPM — this IS the
         posterior update; no explicit weighting/resampling happens.
      4. Estimate = mean of the new particle set.

    Coordinates are normalized to [-1, 1] using fixed environment bounds so
    the DDPM operates on a well-scaled target (mirrors DiffPF's min/max
    normalization of states).

    Parameters
    ----------
    n_particles:
        Size of the sampled posterior particle set per step.
    n_cond_particles:
        How many propagated particles are flattened into the conditioning
        vector (20 matches the DiffPF nav01 config).
    T:
        DDPM steps (10 matches their num_diffusion_iters).
    bounds:
        ((x_min, x_max), (y_min, y_max)) environment bounds for
        normalization.
    dt:
        Simulation time step.
    """

    def __init__(
        self,
        n_particles: int = 100,
        n_cond_particles: int = 20,
        enc_dim: int = 32,
        cond_dim: int = 64,
        hidden_dim: int = 128,
        time_embed_dim: int = 32,
        T: int = 10,
        beta_schedule: str = "cosine",
        bounds=((-2.0, 2.6), (-1.5, 1.5)),
        dt: float = 0.033,
        propagation_noise: float = 0.02,
    ):
        self.n_particles = int(n_particles)
        self.n_cond_particles = int(n_cond_particles)
        self.enc_dim = int(enc_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.T = int(T)
        self.beta_schedule = str(beta_schedule)
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
        self.dt = float(dt)
        self.propagation_noise = float(propagation_noise)

        self.network = DiffPFNetwork(
            n_cond_particles=self.n_cond_particles,
            enc_dim=self.enc_dim,
            cond_dim=self.cond_dim,
            hidden_dim=self.hidden_dim,
            time_embed_dim=self.time_embed_dim,
        ).to(device)

        self._register_schedule()
        self._losses: List[float] = []

        # Filter state
        self.particles: Optional[np.ndarray] = None   # (M, 2) world coords
        self._prev_mean: Optional[np.ndarray] = None  # (2,) for CV velocity

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self, p: np.ndarray) -> np.ndarray:
        (x0, x1), (y0, y1) = self.bounds
        out = np.empty_like(p, dtype=np.float32)
        out[..., 0] = 2.0 * (p[..., 0] - x0) / (x1 - x0) - 1.0
        out[..., 1] = 2.0 * (p[..., 1] - y0) / (y1 - y0) - 1.0
        return out

    def _unnormalize(self, p: np.ndarray) -> np.ndarray:
        (x0, x1), (y0, y1) = self.bounds
        out = np.empty_like(p, dtype=np.float32)
        out[..., 0] = (p[..., 0] + 1.0) / 2.0 * (x1 - x0) + x0
        out[..., 1] = (p[..., 1] + 1.0) / 2.0 * (y1 - y0) + y0
        return out

    # ------------------------------------------------------------------
    # DDPM schedule (same construction as DiffusionMotionModel)
    # ------------------------------------------------------------------

    def _register_schedule(self) -> None:
        if self.beta_schedule == "cosine":
            betas = _cosine_beta_schedule(self.T)
        elif self.beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.T)
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        self.posterior_variance = posterior_variance.to(device)

    # ------------------------------------------------------------------
    # Conditioning helpers (normalized coordinates)
    # ------------------------------------------------------------------

    def _build_condition(self, meas_norm: np.ndarray | None,
                          particles_norm: np.ndarray) -> torch.Tensor:
        """
        meas_norm:      (2,) normalized measurement or None (occluded)
        particles_norm: (M, 2) normalized propagated particles
        returns:        (1, cond_dim) condition embedding
        """
        if meas_norm is None:
            meas_vec = np.zeros(3, dtype=np.float32)
        else:
            meas_vec = np.array([meas_norm[0], meas_norm[1], 1.0], dtype=np.float32)

        # Fixed-size subsample of the particle set for the conditioning MLP
        # (flattened raw particles, faithful to the DiffPF conditioning input).
        M = particles_norm.shape[0]
        if M >= self.n_cond_particles:
            idx = np.linspace(0, M - 1, self.n_cond_particles).astype(int)
        else:
            idx = np.random.choice(M, self.n_cond_particles, replace=True)
        p_flat = particles_norm[idx].reshape(-1)

        meas_t = torch.tensor(meas_vec, dtype=torch.float32, device=device).unsqueeze(0)
        part_t = torch.tensor(p_flat, dtype=torch.float32, device=device).unsqueeze(0)
        return self.network.encode_condition(meas_t, part_t)

    @torch.no_grad()
    def _sample_posterior(self, cond: torch.Tensor, n: int) -> np.ndarray:
        """
        Ancestral DDPM sampling of n particles given one condition embedding.
        Returns (n, 2) normalized positions.
        """
        cond_rep = cond.expand(n, -1)
        y = torch.randn((n, 2), device=device)

        for t_idx in reversed(range(self.T)):
            t = torch.full((n,), t_idx, dtype=torch.long, device=device)
            eps_pred = self.network.predict_noise(y, t, cond_rep)

            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alphas_cumprod[t_idx]
            beta_t = self.betas[t_idx]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                y - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
            )
            if t_idx > 0:
                sigma_t = torch.sqrt(self.posterior_variance[t_idx])
                y = mean + sigma_t * torch.randn_like(y)
            else:
                y = mean

        return y.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Online filtering interface
    # ------------------------------------------------------------------

    def reset(self, init_measurement: np.ndarray, init_spread: float = 0.05) -> None:
        """Initialize the particle set around the first visible measurement."""
        init_measurement = np.asarray(init_measurement, dtype=np.float32)
        self.particles = (
            init_measurement[None, :]
            + init_spread * np.random.randn(self.n_particles, 2).astype(np.float32)
        )
        self._prev_mean = self.particles.mean(axis=0)

    @torch.no_grad()
    def step(self, measurement: np.ndarray | None) -> np.ndarray:
        """
        One filter step.

        measurement:
            Noisy (x, y) position when the target is visible, or None
            during occlusion.

        Returns
        -------
        estimate: (2,) posterior mean position (world coordinates).
        """
        if self.particles is None:
            raise RuntimeError("Call reset() with the first measurement before step().")

        self.network.eval()

        # 1. Constant-velocity propagation + noise (DiffPF's "dumb" predict).
        mean_now = self.particles.mean(axis=0)
        v_est = (mean_now - self._prev_mean) / self.dt
        pred = (
            self.particles
            + v_est[None, :] * self.dt
            + self.propagation_noise * np.random.randn(*self.particles.shape).astype(np.float32)
        )
        self._prev_mean = mean_now

        # 2. Conditioning from (measurement | null, predicted particles).
        pred_norm = self._normalize(pred)
        meas_norm = None if measurement is None else self._normalize(
            np.asarray(measurement, dtype=np.float32)
        )
        cond = self._build_condition(meas_norm, pred_norm)

        # 3. Generative posterior update: sample a fresh particle set.
        # Samples are clipped to the (known) environment bounds so one bad
        # draw during a long occlusion cannot push the conditioning outside
        # the normalized range the network was trained on.
        new_norm = np.clip(self._sample_posterior(cond, self.n_particles), -1.0, 1.0)
        self.particles = self._unnormalize(new_norm)

        # 4. Point estimate.
        return self.particles.mean(axis=0)

    # ------------------------------------------------------------------
    # Training (teacher-forced, mirrors the DiffPF training recipe)
    # ------------------------------------------------------------------

    def train(
        self,
        meas_cond: np.ndarray,
        particle_cond: np.ndarray,
        targets: np.ndarray,
        epochs: int = 800,
        batch_size: int = 512,
        lr: float = 5e-4,
        print_every: int = 100,
    ) -> List[float]:
        """
        Train the conditional DDPM on teacher-forced examples.

        Parameters (all in NORMALIZED coordinates)
        ----------
        meas_cond:     (N, 3)  (x, y, visible_flag); (0,0,0) rows for occluded steps
        particle_cond: (N, n_cond_particles * 2)  flattened predicted particles
        targets:       (N, 2)  ground-truth position at the same step
        """
        meas_t = torch.tensor(np.asarray(meas_cond, dtype=np.float32))
        part_t = torch.tensor(np.asarray(particle_cond, dtype=np.float32))
        y_t = torch.tensor(np.asarray(targets, dtype=np.float32))

        dataset = TensorDataset(meas_t, part_t, y_t)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        criterion = nn.MSELoss()

        self.network.train()
        self._losses = []

        for epoch in range(1, epochs + 1):
            epoch_losses = []

            for m_b, p_b, y_b in dataloader:
                m_b = m_b.to(device)
                p_b = p_b.to(device)
                y_b = y_b.to(device)
                B = y_b.shape[0]

                cond = self.network.encode_condition(m_b, p_b)

                t = torch.randint(0, self.T, (B,), device=device)
                eps = torch.randn_like(y_b)
                sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
                sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
                y_noised = sqrt_ab * y_b + sqrt_1mab * eps

                eps_pred = self.network.predict_noise(y_noised, t, cond)
                loss = criterion(eps_pred, eps)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            mean_loss = float(np.mean(epoch_losses))
            self._losses.append(mean_loss)

            if epoch % print_every == 0 or epoch == 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:5d}/{epochs}  "
                      f"loss={mean_loss:.6f}  lr={current_lr:.2e}")

        self.network.eval()
        return self._losses

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "config": {
                    "n_particles": self.n_particles,
                    "n_cond_particles": self.n_cond_particles,
                    "enc_dim": self.enc_dim,
                    "cond_dim": self.cond_dim,
                    "hidden_dim": self.hidden_dim,
                    "time_embed_dim": self.time_embed_dim,
                    "T": self.T,
                    "beta_schedule": self.beta_schedule,
                    "bounds": self.bounds,
                    "dt": self.dt,
                    "propagation_noise": self.propagation_noise,
                },
            },
            path,
        )
        print(f"[DiffPFTracker] saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})

        self.n_particles = int(cfg.get("n_particles", self.n_particles))
        self.n_cond_particles = int(cfg.get("n_cond_particles", self.n_cond_particles))
        self.enc_dim = int(cfg.get("enc_dim", self.enc_dim))
        self.cond_dim = int(cfg.get("cond_dim", self.cond_dim))
        self.hidden_dim = int(cfg.get("hidden_dim", self.hidden_dim))
        self.time_embed_dim = int(cfg.get("time_embed_dim", self.time_embed_dim))
        self.T = int(cfg.get("T", self.T))
        self.beta_schedule = str(cfg.get("beta_schedule", self.beta_schedule))
        self.bounds = tuple(tuple(b) for b in cfg.get("bounds", self.bounds))
        self.dt = float(cfg.get("dt", self.dt))
        self.propagation_noise = float(cfg.get("propagation_noise", self.propagation_noise))

        self.network = DiffPFNetwork(
            n_cond_particles=self.n_cond_particles,
            enc_dim=self.enc_dim,
            cond_dim=self.cond_dim,
            hidden_dim=self.hidden_dim,
            time_embed_dim=self.time_embed_dim,
        ).to(device)
        self.network.load_state_dict(ckpt["state_dict"])
        self.network.eval()

        self._register_schedule()

        print(f"[DiffPFTracker] loaded ← {path}  "
              f"(n_particles={self.n_particles}, "
              f"n_cond_particles={self.n_cond_particles}, T={self.T}, "
              f"bounds={self.bounds}, dt={self.dt})")

    def loss_history(self) -> List[float]:
        return self._losses


def load_diffpf_tracker(model_path: str, **kwargs) -> DiffPFTracker:
    """Load a saved DiffPF tracker; checkpoint config wins over kwargs."""
    tracker = DiffPFTracker(**kwargs)
    tracker.load(model_path)
    return tracker
