"""
diffusion_motion_model.py
==========================

DiffusionMotionModel — DDPM-style denoising diffusion trained as a direct
comparison to CFMMotionModel (cfm_motion_model.py).

DiffusionShim         — drop-in wrapper with the same .predict() signature as
                        CFMShim / ScratchTransformer, so ParticleFilter.py can
                        call:

                            self.motion_model.predict(X)

                        without changing predict_mml().

Main ideas
----------
This model targets exactly the same task as CFMMotionModel — generate a
velocity sample (vx, vy) conditioned on a flattened position history — but
uses the standard DDPM forward/reverse process instead of OT-CFM flow
matching. It reuses the same ContextEncoder architecture (see
velocity_network.py) so a CFM-vs-diffusion comparison isolates the
generative mechanism, not encoder capacity.

Forward (noising) process:
    y_t = sqrt(alpha_bar_t) * y_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0,I)

Training objective (predict the noise):
    L = || eps_theta(y_t, t, c_k) - eps ||^2

Reverse (sampling) process — ancestral DDPM sampling from t=T-1 down to 0:
    y_{t-1} = 1/sqrt(alpha_t) * ( y_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_theta )
              + sigma_t * z,   z ~ N(0,I)  (z=0 at t=0)

Unlike CFM's anchored base point, the diffusion prior here is standard
N(0, I) — the model must learn the full distribution from pure noise,
which is the more faithful "vanilla diffusion" comparison point.

Place at:
    mml_guidance/mml_guidance/mml_network/diffusion_motion_model.py
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mml_core.motion.velocity_network import DiffusionVelocityNetwork


device = "cuda" if torch.cuda.is_available() else "cpu"


def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule (Nichol & Dhariwal, 2021) — better-behaved than
    linear for small T, which matters here since T is kept small (20-50)
    to keep inference cost comparable to CFM's few-step ODE integration.
    """
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64) / T
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-4, 0.999).float()


class DiffusionMotionModel:
    """
    Context-conditioned DDPM motion model.

    Training target:
        y = (vx, vy)

    DDPM training:
        t       ~ Uniform{0, ..., T-1}
        eps     ~ N(0, I)
        y_t     = sqrt(alpha_bar_t) * y + sqrt(1 - alpha_bar_t) * eps
        eps_target = eps

    Inference:
        y_T ~ N(0, I), then ancestral sampling down to y_0.

    Parameters
    ----------
    input_dim, block_size, n_embed, n_head, n_layer, dropout:
        Same meaning as CFMMotionModel — shared ContextEncoder config.
    noise_hidden:
        Hidden size of the noise-prediction MLP head (analogous to
        vel_hidden in CFMMotionModel).
    time_embed_dim:
        Dimension of the sinusoidal timestep embedding.
    T:
        Number of diffusion steps. Kept small (default 20) so that
        inference cost is in the same ballpark as CFM's N_FE Euler steps —
        a fair speed comparison rather than a 1000-step reference DDPM.
    beta_schedule:
        "cosine" (recommended for small T) or "linear".
    dt:
        Simulation time step, same role as in CFMMotionModel.
    """

    def __init__(
        self,
        input_dim: int = 2,
        block_size: int = 10,
        n_embed: int = 64,
        n_head: int = 8,
        n_layer: int = 2,
        noise_hidden: int = 128,
        time_embed_dim: int = 32,
        T: int = 20,
        dropout: float = 0.1,
        beta_schedule: str = "cosine",
        dt: float = 0.033,
    ):
        self.input_dim = int(input_dim)
        self.block_size = int(block_size)
        self.n_embed = int(n_embed)
        self.n_head = int(n_head)
        self.n_layer = int(n_layer)
        self.noise_hidden = int(noise_hidden)
        self.time_embed_dim = int(time_embed_dim)
        self.T = int(T)
        self.dropout = float(dropout)
        self.beta_schedule = str(beta_schedule)
        self.dt = float(dt)

        self.network = DiffusionVelocityNetwork(
            input_dim=self.input_dim,
            block_size=self.block_size,
            n_embed=self.n_embed,
            n_head=self.n_head,
            n_layer=self.n_layer,
            noise_hidden=self.noise_hidden,
            time_embed_dim=self.time_embed_dim,
            dropout=self.dropout,
        ).to(device)

        self._register_schedule()

        self._losses: List[float] = []

    # ----------------------------------------------------------------------
    # Diffusion schedule
    # ----------------------------------------------------------------------

    def _register_schedule(self) -> None:
        if self.beta_schedule == "cosine":
            betas = _cosine_beta_schedule(self.T)
        elif self.beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.T)
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1), alphas_cumprod[:-1]], dim=0
        )

        # Posterior variance for ancestral sampling:
        #   sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        )

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - alphas_cumprod
        ).to(device)
        self.posterior_variance = posterior_variance.to(device)

    # ----------------------------------------------------------------------
    # Internal helpers (shared with CFMMotionModel — same context contract)
    # ----------------------------------------------------------------------

    def _expected_context_dim(self) -> int:
        return self.block_size * self.input_dim

    def _validate_context_tensor(self, c_flat: torch.Tensor) -> None:
        if c_flat.ndim != 2:
            raise ValueError(
                f"Diffusion context must be 2D, got shape {tuple(c_flat.shape)}."
            )
        expected = self._expected_context_dim()
        got = c_flat.shape[1]
        if got != expected:
            raise ValueError(
                "Diffusion context shape mismatch: "
                f"got {tuple(c_flat.shape)}, but expected "
                f"(batch, {expected}) = "
                f"(batch, block_size={self.block_size} * input_dim={self.input_dim})."
            )

    # ----------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1500,
        batch_size: int = 512,
        lr: float = 5e-4,
        print_every: int = 100,
    ) -> List[float]:
        """
        Train the diffusion noise-prediction network.

        Parameters mirror CFMMotionModel.train() exactly so the two models
        can be trained on identical data with identical schedules.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        expected_dim = self._expected_context_dim()
        if X.ndim != 2 or X.shape[1] != expected_dim:
            raise ValueError(
                f"Training X has shape {X.shape}, but diffusion model expects "
                f"(N, {expected_dim}) for block_size={self.block_size}, "
                f"input_dim={self.input_dim}."
            )
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                f"Training y has shape {y.shape}, but expected (N, 2)."
            )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.01,
        )
        criterion = nn.MSELoss()

        self.network.train()
        self._losses = []

        for epoch in range(1, epochs + 1):
            epoch_losses = []

            for c_batch, y_batch in dataloader:
                c_batch = c_batch.to(device)
                y_batch = y_batch.to(device)
                B = y_batch.shape[0]

                # Sample a random diffusion timestep per example.
                t = torch.randint(0, self.T, (B,), device=device)

                # Forward noising process.
                eps = torch.randn_like(y_batch)
                sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
                sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
                y_t_noised = sqrt_ab * y_batch + sqrt_1mab * eps

                eps_pred = self.network(y_t_noised, t, c_batch)
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
                print(
                    f"  Epoch {epoch:5d}/{epochs}  "
                    f"loss={mean_loss:.6f}  lr={current_lr:.2e}"
                )

        self.network.eval()
        return self._losses

    # ----------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def propose(self, c_flat: np.ndarray) -> np.ndarray:
        """
        Generate one velocity proposal per particle via ancestral DDPM
        sampling. Same call signature as CFMMotionModel.propose().

        Parameters
        ----------
        c_flat:
            Context batch with shape (N_particles, block_size * input_dim)

        Returns
        -------
        proposals:
            Velocity predictions with shape (N_particles, 2)
        """
        c_flat = np.asarray(c_flat, dtype=np.float32)

        if c_flat.ndim != 2:
            raise ValueError(
                f"Diffusion propose() expected 2D context array, got {c_flat.shape}."
            )

        expected_dim = self._expected_context_dim()
        if c_flat.shape[1] != expected_dim:
            raise ValueError(
                f"Diffusion propose() context shape mismatch: got {c_flat.shape}, "
                f"but expected (N_particles, {expected_dim}) because "
                f"block_size={self.block_size}, input_dim={self.input_dim}."
            )

        self.network.eval()

        c_t = torch.tensor(c_flat, dtype=torch.float32, device=device)
        N = c_t.shape[0]

        # Encode context once, reuse for all T reverse-diffusion steps.
        emb = self.network.encode_context(c_t)

        y = torch.randn((N, 2), device=device)

        for t_idx in reversed(range(self.T)):
            t = torch.full((N,), t_idx, dtype=torch.long, device=device)
            eps_pred = self.network.predict_noise(y, t, emb)

            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alphas_cumprod[t_idx]
            beta_t = self.betas[t_idx]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                y - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
            )

            if t_idx > 0:
                noise = torch.randn_like(y)
                sigma_t = torch.sqrt(self.posterior_variance[t_idx])
                y = mean + sigma_t * noise
            else:
                y = mean

        v = y.cpu().numpy().astype(np.float32)

        # Clip to physically plausible speed — same convention as CFM.
        MAX_SPEED = 1.5  # m/s
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * np.minimum(1.0, MAX_SPEED / (speed + 1e-9))

        return v

    def propose_batched(
        self,
        c_flat: np.ndarray,
        n_samples_per_context: int = 1,
    ) -> np.ndarray:
        """
        Generate multiple proposals per context (multi-modality visualization).

        Returns
        -------
        proposals: shape (N_contexts, n_samples_per_context, 2)
        """
        samples = []
        for _ in range(n_samples_per_context):
            samples.append(self.propose(c_flat))
        return np.stack(samples, axis=1)

    # ----------------------------------------------------------------------
    # Save / load
    # ----------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "block_size": self.block_size,
                    "n_embed": self.n_embed,
                    "n_head": self.n_head,
                    "n_layer": self.n_layer,
                    "noise_hidden": self.noise_hidden,
                    "time_embed_dim": self.time_embed_dim,
                    "T": self.T,
                    "dropout": self.dropout,
                    "beta_schedule": self.beta_schedule,
                    "dt": self.dt,
                },
            },
            path,
        )

        print(f"[DiffusionMotionModel] saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})

        loaded_input_dim = int(cfg.get("input_dim", self.input_dim))
        loaded_block_size = int(cfg.get("block_size", self.block_size))
        loaded_n_embed = int(cfg.get("n_embed", self.n_embed))
        loaded_n_head = int(cfg.get("n_head", self.n_head))
        loaded_n_layer = int(cfg.get("n_layer", self.n_layer))
        loaded_noise_hidden = int(cfg.get("noise_hidden", self.noise_hidden))
        loaded_time_embed_dim = int(cfg.get("time_embed_dim", self.time_embed_dim))
        loaded_T = int(cfg.get("T", self.T))
        loaded_dropout = float(cfg.get("dropout", self.dropout))
        loaded_beta_schedule = str(cfg.get("beta_schedule", self.beta_schedule))
        loaded_dt = float(cfg.get("dt", self.dt))

        needs_rebuild = (
            loaded_input_dim != self.input_dim
            or loaded_block_size != self.block_size
            or loaded_n_embed != self.n_embed
            or loaded_n_head != self.n_head
            or loaded_n_layer != self.n_layer
            or loaded_noise_hidden != self.noise_hidden
            or loaded_time_embed_dim != self.time_embed_dim
            or loaded_dropout != self.dropout
        )

        self.input_dim = loaded_input_dim
        self.block_size = loaded_block_size
        self.n_embed = loaded_n_embed
        self.n_head = loaded_n_head
        self.n_layer = loaded_n_layer
        self.noise_hidden = loaded_noise_hidden
        self.time_embed_dim = loaded_time_embed_dim
        self.T = loaded_T
        self.dropout = loaded_dropout
        self.beta_schedule = loaded_beta_schedule
        self.dt = loaded_dt

        if needs_rebuild:
            self.network = DiffusionVelocityNetwork(
                input_dim=self.input_dim,
                block_size=self.block_size,
                n_embed=self.n_embed,
                n_head=self.n_head,
                n_layer=self.n_layer,
                noise_hidden=self.noise_hidden,
                time_embed_dim=self.time_embed_dim,
                dropout=self.dropout,
            ).to(device)

        self._register_schedule()

        self.network.load_state_dict(ckpt["state_dict"])
        self.network.eval()

        print(
            f"[DiffusionMotionModel] loaded ← {path}  "
            f"(input_dim={self.input_dim}, block_size={self.block_size}, "
            f"n_embed={self.n_embed}, n_head={self.n_head}, "
            f"n_layer={self.n_layer}, noise_hidden={self.noise_hidden}, "
            f"T={self.T}, beta_schedule={self.beta_schedule}, dt={self.dt})"
        )

        if "dt" not in cfg:
            print(
                f"[DiffusionMotionModel] WARNING: checkpoint has no 'dt' field. "
                f"Assuming training dt={self.dt:.4f}. "
                f"If your sim runs at a different dt, particles may diverge."
            )

    def loss_history(self) -> List[float]:
        return self._losses


class DiffusionShim:
    """
    Drop-in wrapper so ParticleFilter.py can call:

        self.motion_model.predict(X)

    exactly like ScratchTransformer / CFMShim.
    """

    def __init__(self, diffusion: DiffusionMotionModel):
        self._diffusion = diffusion
        self.dt = diffusion.dt

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._diffusion.propose(X.astype(np.float32))

    def propose(self, X: np.ndarray) -> np.ndarray:
        return self._diffusion.propose(X.astype(np.float32))

    def parameters(self):
        return self._diffusion.network.parameters()

    def state_dict(self):
        return self._diffusion.network.state_dict()

    def load_state_dict(self, state_dict):
        return self._diffusion.network.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        if mode:
            self._diffusion.network.train()
        else:
            self._diffusion.network.eval()
        return self

    def eval(self):
        self._diffusion.network.eval()
        return self

    def online_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 5,
        lr: float = 5e-4,
    ):
        """Online retraining hook for ParticleFilter.optimize_learned_model()."""
        batch_size = min(64, len(X))
        self._diffusion.train(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            print_every=epochs + 1,
        )


def load_diffusion_model(
    model_path: str,
    input_dim: int = 2,
    block_size: int = 10,
    n_embed: int = 64,
    n_head: int = 8,
    n_layer: int = 2,
    noise_hidden: int = 128,
    time_embed_dim: int = 32,
    T: int = 20,
    dropout: float = 0.1,
    beta_schedule: str = "cosine",
    dt: float = 0.033,
) -> DiffusionShim:
    """
    Load a saved diffusion model and return a DiffusionShim.

    The function arguments are fallbacks. If the checkpoint contains a
    config, load() will rebuild the model using the saved values.

    Usage in ParticleFilter.py:

        from mml_core.motion.diffusion_motion_model import load_diffusion_model

        self.motion_model = load_diffusion_model(
            model_path=model_path,
            input_dim=2,
            block_size=self.N_th,
            dt=0.033,
        )
    """
    diffusion = DiffusionMotionModel(
        input_dim=input_dim,
        block_size=block_size,
        n_embed=n_embed,
        n_head=n_head,
        n_layer=n_layer,
        noise_hidden=noise_hidden,
        time_embed_dim=time_embed_dim,
        T=T,
        dropout=dropout,
        beta_schedule=beta_schedule,
        dt=dt,
    )

    diffusion.load(model_path)
    return DiffusionShim(diffusion)
