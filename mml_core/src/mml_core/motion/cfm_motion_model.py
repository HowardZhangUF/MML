"""
cfm_motion_model.py
===================

CFMMotionModel  — OT-CFM training + anchored Euler ODE inference.
CFMShim         — drop-in wrapper with the same .predict() signature as
                  ScratchTransformer, so ParticleFilter.py can call:

                      self.motion_model.predict(X)

                  without changing predict_mml().

Main ideas
----------
1. The model predicts target velocity (vx, vy), not next position directly.

2. The context is a flattened trajectory history:

       [x0, y0, x1, y1, ..., x_{N_th-1}, y_{N_th-1}]

   with shape:

       (batch, block_size * input_dim)

3. The CFM base point is anchored near the finite-difference velocity:

       v_anchor = (x_k - x_{k-1}) / dt

   instead of pure N(0, I). This keeps generated velocity proposals close
   to physically plausible motion.

4. The checkpoint stores all important hyperparameters, including dt, so
   inference reconstructs the same architecture used during training.

Place at:
    mml_guidance/mml_guidance/mml_network/cfm_motion_model.py
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mml_core.motion.velocity_network import CFMVelocityNetwork


device = "cuda" if torch.cuda.is_available() else "cpu"


class CFMMotionModel:
    """
    Context-conditioned OT-CFM motion model with anchored base distribution.

    Training target:
        y = (vx, vy)

    CFM training:
        x0      = v_anchor + anchor_sigma * N(0, I)
        tau     ~ Uniform(0, 1)
        x_tau   = (1 - tau) * x0 + tau * y
        u_target = y - x0

    Inference:
        Start from anchored velocity x0, then integrate the learned vector
        field from tau=0 to tau=1 using Euler steps.

    Parameters
    ----------
    input_dim:
        Number of features per history step. Usually 2 for [x, y].
    block_size:
        History length N_th. Must match ParticleFilter.N_th.
    n_embed:
        Transformer embedding dimension.
    n_head:
        Number of attention heads.
    n_layer:
        Number of Transformer blocks.
    vel_hidden:
        Hidden size for the CFM velocity MLP head.
    N_FE:
        Number of Euler flow-integration steps at inference.
    dropout:
        Dropout inside the context encoder.
    anchor_sigma:
        Standard deviation of the noise around the finite-difference
        velocity anchor.
    dt:
        Simulation time step used to convert displacement into velocity.
        For your pygame sim, this should usually be 0.033.
    """

    def __init__(
        self,
        input_dim: int = 2,
        block_size: int = 10,
        n_embed: int = 64,
        n_head: int = 8,
        n_layer: int = 2,
        vel_hidden: int = 128,
        N_FE: int = 10,
        dropout: float = 0.1,
        anchor_sigma: float = 0.3,
        dt: float = 0.033,
    ):
        self.input_dim = int(input_dim)
        self.block_size = int(block_size)
        self.n_embed = int(n_embed)
        self.n_head = int(n_head)
        self.n_layer = int(n_layer)
        self.vel_hidden = int(vel_hidden)
        self.N_FE = int(N_FE)
        self.dropout = float(dropout)
        self.anchor_sigma = float(anchor_sigma)
        self.dt = float(dt)

        self.network = CFMVelocityNetwork(
            input_dim=self.input_dim,
            block_size=self.block_size,
            n_embed=self.n_embed,
            n_head=self.n_head,
            n_layer=self.n_layer,
            vel_hidden=self.vel_hidden,
            dropout=self.dropout,
        ).to(device)

        self._losses: List[float] = []

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _expected_context_dim(self) -> int:
        return self.block_size * self.input_dim

    def _validate_context_tensor(self, c_flat: torch.Tensor) -> None:
        """
        Validate context shape before feeding it to the network.

        Expected:
            c_flat.shape == (B, block_size * input_dim)
        """
        if c_flat.ndim != 2:
            raise ValueError(
                f"CFM context must be 2D, got shape {tuple(c_flat.shape)}."
            )

        expected = self._expected_context_dim()
        got = c_flat.shape[1]

        if got != expected:
            raise ValueError(
                "CFM context shape mismatch: "
                f"got {tuple(c_flat.shape)}, but expected "
                f"(batch, {expected}) = "
                f"(batch, block_size={self.block_size} * input_dim={self.input_dim})."
            )

    def _finite_difference_velocity(self, c_flat: torch.Tensor) -> torch.Tensor:
        """
        Compute the anchor velocity from the last two history positions.

        For input_dim == 2:
            context step = [x, y]
            v_anchor = (p_k - p_{k-1}) / self.dt

        For input_dim >= 3:
            context step = [x, y, t] or [x, y, dt-like feature]
            If the third channel looks usable as time, use the local time
            difference. Otherwise fall back to self.dt.

        Returns
        -------
        v_anchor:
            Tensor with shape (B, 2)
        """
        self._validate_context_tensor(c_flat)

        B = c_flat.shape[0]
        hist = c_flat.view(B, self.block_size, self.input_dim)

        last_pos = hist[:, -1, :2]
        prev_pos = hist[:, -2, :2]

        if self.input_dim >= 3:
            local_dt = hist[:, -1, 2:3] - hist[:, -2, 2:3]
            local_dt = torch.abs(local_dt)

            # If local_dt is zero or invalid, fall back to self.dt.
            fallback_dt = torch.full_like(local_dt, self.dt)
            local_dt = torch.where(local_dt > 1e-6, local_dt, fallback_dt)
        else:
            local_dt = torch.full(
                (B, 1),
                self.dt,
                dtype=c_flat.dtype,
                device=c_flat.device,
            )

        v_anchor = (last_pos - prev_pos) / local_dt
        return v_anchor

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
        Train the CFM velocity network.

        Parameters
        ----------
        X:
            Context windows with shape:
                (N, block_size * input_dim)

        y:
            Target velocities with shape:
                (N, 2)

        epochs:
            Number of training epochs.

        batch_size:
            Mini-batch size.

        lr:
            Adam learning rate.

        print_every:
            Print training loss every this many epochs.

        Returns
        -------
        losses:
            List of mean training loss per epoch.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        expected_dim = self._expected_context_dim()
        if X.ndim != 2 or X.shape[1] != expected_dim:
            raise ValueError(
                f"Training X has shape {X.shape}, but CFM expects "
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

                # Anchored base distribution:
                # x0 is centered near finite-difference velocity.
                v_anchor = self._finite_difference_velocity(c_batch)
                x0 = v_anchor + self.anchor_sigma * torch.randn_like(y_batch)

                # OT-CFM interpolation.
                tau = torch.rand(c_batch.shape[0], 1, device=device)
                x_tau = (1.0 - tau) * x0 + tau * y_batch
                u_target = y_batch - x0

                u_pred = self.network(x_tau, tau, c_batch)
                loss = criterion(u_pred, u_target)

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
        Generate one velocity proposal per particle.

        Parameters
        ----------
        c_flat:
            Context batch with shape:
                (N_particles, block_size * input_dim)

        Returns
        -------
        proposals:
            Velocity predictions with shape:
                (N_particles, 2)
        """
        c_flat = np.asarray(c_flat, dtype=np.float32)

        if c_flat.ndim != 2:
            raise ValueError(
                f"CFM propose() expected 2D context array, got {c_flat.shape}."
            )

        expected_dim = self._expected_context_dim()
        if c_flat.shape[1] != expected_dim:
            raise ValueError(
                f"CFM propose() context shape mismatch: got {c_flat.shape}, "
                f"but expected (N_particles, {expected_dim}) because "
                f"block_size={self.block_size}, input_dim={self.input_dim}."
            )

        self.network.eval()

        c_t = torch.tensor(c_flat, dtype=torch.float32, device=device)

        # Encode context once.
        emb = self.network.encode_context(c_t)

        # Anchored initial velocity.
        v_anchor = self._finite_difference_velocity(c_t)
        x = v_anchor + self.anchor_sigma * torch.randn_like(v_anchor)

        delta = 1.0 / self.N_FE

        for s in range(self.N_FE):
            tau = torch.full(
                (c_t.shape[0], 1),
                s * delta,
                dtype=torch.float32,
                device=device,
            )
            v = self.network.predict_velocity(x, tau, emb)
            x = x + delta * v

        v = x.cpu().numpy().astype(np.float32)

        # Clip to physically plausible speed so any caller — particle filter,
        # trajectory fan, notebooks — is protected from runaway ODE outputs.
        MAX_SPEED = 1.5  # m/s
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * np.minimum(1.0, MAX_SPEED / (speed + 1e-9))

        return v

    # ----------------------------------------------------------------------
    # Exact density of the (unclipped) Euler sampler
    # ----------------------------------------------------------------------

    def velocity_field(self, emb: torch.Tensor):
        """The learned vector field u(x, tau) for fixed context embeddings."""
        from mml_core.motion.flow_density import MLPField

        # VelocityHead concatenates [emb, x_tau, tau], the MLPField layout.
        return MLPField(self.network.vel_head.net, emb)

    def sample_with_log_prob(self, c_t: torch.Tensor, emb: torch.Tensor | None = None):
        """
        Velocity samples of the N_FE-step Euler sampler WITHOUT the MAX_SPEED
        clip, with their exact log-density (see flow_density.py).
        Returns (v, log_prob), both torch tensors.
        """
        from mml_core.motion.flow_density import euler_forward, gaussian_log_prob

        emb = self.network.encode_context(c_t) if emb is None else emb
        anchor = self._finite_difference_velocity(c_t)
        x0 = anchor + self.anchor_sigma * torch.randn_like(anchor)
        v, logdet = euler_forward(self.velocity_field(emb), x0, self.N_FE)
        return v, gaussian_log_prob(x0, anchor, self.anchor_sigma) - logdet

    def log_prob(self, c_t: torch.Tensor, v: torch.Tensor, emb: torch.Tensor | None = None):
        """
        Exact log-density of velocities v under the unclipped Euler sampler,
        by Newton inversion of every step. Returns (log_prob, max residual).
        """
        from mml_core.motion.flow_density import euler_inverse, gaussian_log_prob

        emb = self.network.encode_context(c_t) if emb is None else emb
        anchor = self._finite_difference_velocity(c_t)
        x0, logdet, resid = euler_inverse(self.velocity_field(emb), v, self.N_FE)
        return gaussian_log_prob(x0, anchor, self.anchor_sigma) - logdet, resid

    @torch.no_grad()
    def propose_sequence(
        self,
        c_flat: np.ndarray,
        n_steps: int,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Roll the motion model forward n_steps physical time steps,
        encoding the context **once** and reusing the embedding for all steps.

        This is 5-7x faster than calling propose() n_steps times separately,
        because the Transformer encode_context pass (the bottleneck) runs once
        instead of n_steps times.

        Parameters
        ----------
        c_flat:
            Starting context, shape (N, block_size * input_dim).
        n_steps:
            Number of physical steps to roll forward.
        dt:
            Physical timestep for position integration. Defaults to self.dt.

        Returns
        -------
        positions:
            Shape (N, n_steps, 2) — predicted positions at each step.
        """
        if dt is None:
            dt = self.dt

        c_flat = np.asarray(c_flat, dtype=np.float32)
        N = c_flat.shape[0]

        c_t = torch.tensor(c_flat, dtype=torch.float32, device=device)
        hist = c_t.view(N, self.block_size, self.input_dim)  # (N, N_th, 2)

        # Encode context ONCE from the initial history and reuse for all steps.
        # The embedding changes slowly over a short horizon — re-encoding every
        # step gives no quality gain for visualization but costs 15x more compute.
        c_in = hist.reshape(N, self.block_size * self.input_dim)
        emb = self.network.encode_context(c_in)
        v_anchor = self._finite_difference_velocity(c_in)

        positions = np.zeros((N, n_steps, 2), dtype=np.float32)
        delta = 1.0 / self.N_FE

        for step in range(n_steps):
            x = v_anchor + self.anchor_sigma * torch.randn_like(v_anchor)

            for s in range(self.N_FE):
                tau = torch.full((N, 1), s * delta, dtype=torch.float32, device=device)
                x = x + delta * self.network.predict_velocity(x, tau, emb)

            v = x.cpu().numpy()
            speed = np.linalg.norm(v, axis=1, keepdims=True)
            v = v * np.minimum(1.0, 1.5 / (speed + 1e-9))

            last_pos = hist[:, -1, :2].cpu().numpy()
            new_pos = last_pos + v * dt
            positions[:, step, :] = new_pos

            # Update position history and refresh the anchor velocity.
            new_pos_t = torch.tensor(new_pos, dtype=torch.float32, device=device)
            hist = torch.cat([hist[:, 1:, :], new_pos_t.unsqueeze(1)], dim=1)
            v_anchor_np = (new_pos - hist[:, -2, :2].cpu().numpy()) / dt
            v_anchor = torch.tensor(v_anchor_np, dtype=torch.float32, device=device)

        return positions

    def propose_batched(
        self,
        c_flat: np.ndarray,
        n_samples_per_context: int = 1,
    ) -> np.ndarray:
        """
        Generate multiple proposals per context.

        Useful for visualizing multi-modality.

        Parameters
        ----------
        c_flat:
            Shape:
                (N_contexts, block_size * input_dim)

        n_samples_per_context:
            Number of samples generated from each context.

        Returns
        -------
        proposals:
            Shape:
                (N_contexts, n_samples_per_context, 2)
        """
        samples = []
        for _ in range(n_samples_per_context):
            samples.append(self.propose(c_flat))
        return np.stack(samples, axis=1)

    # ----------------------------------------------------------------------
    # Save / load
    # ----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save weights and all hyperparameters needed to reconstruct the model.
        """
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
                    "vel_hidden": self.vel_hidden,
                    "N_FE": self.N_FE,
                    "dropout": self.dropout,
                    "anchor_sigma": self.anchor_sigma,
                    "dt": self.dt,
                },
            },
            path,
        )

        print(f"[CFMMotionModel] saved → {path}")

    def load(self, path: str) -> None:
        """
        Load weights and reconstruct the architecture from checkpoint config.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})

        loaded_input_dim = int(cfg.get("input_dim", self.input_dim))
        loaded_block_size = int(cfg.get("block_size", self.block_size))
        loaded_n_embed = int(cfg.get("n_embed", self.n_embed))
        loaded_n_head = int(cfg.get("n_head", self.n_head))
        loaded_n_layer = int(cfg.get("n_layer", self.n_layer))
        loaded_vel_hidden = int(cfg.get("vel_hidden", self.vel_hidden))
        loaded_N_FE = int(cfg.get("N_FE", self.N_FE))
        loaded_dropout = float(cfg.get("dropout", self.dropout))
        loaded_anchor_sigma = float(cfg.get("anchor_sigma", self.anchor_sigma))
        loaded_dt = float(cfg.get("dt", self.dt))

        needs_rebuild = (
            loaded_input_dim != self.input_dim
            or loaded_block_size != self.block_size
            or loaded_n_embed != self.n_embed
            or loaded_n_head != self.n_head
            or loaded_n_layer != self.n_layer
            or loaded_vel_hidden != self.vel_hidden
            or loaded_dropout != self.dropout
        )

        self.input_dim = loaded_input_dim
        self.block_size = loaded_block_size
        self.n_embed = loaded_n_embed
        self.n_head = loaded_n_head
        self.n_layer = loaded_n_layer
        self.vel_hidden = loaded_vel_hidden
        self.N_FE = loaded_N_FE
        self.dropout = loaded_dropout
        self.anchor_sigma = loaded_anchor_sigma
        self.dt = loaded_dt

        if needs_rebuild:
            self.network = CFMVelocityNetwork(
                input_dim=self.input_dim,
                block_size=self.block_size,
                n_embed=self.n_embed,
                n_head=self.n_head,
                n_layer=self.n_layer,
                vel_hidden=self.vel_hidden,
                dropout=self.dropout,
            ).to(device)

        self.network.load_state_dict(ckpt["state_dict"])
        self.network.eval()

        print(
            f"[CFMMotionModel] loaded ← {path}  "
            f"(input_dim={self.input_dim}, block_size={self.block_size}, "
            f"n_embed={self.n_embed}, n_head={self.n_head}, "
            f"n_layer={self.n_layer}, vel_hidden={self.vel_hidden}, "
            f"N_FE={self.N_FE}, anchor_sigma={self.anchor_sigma}, dt={self.dt})"
        )

        if "dt" not in cfg:
            print(
                f"[CFMMotionModel] WARNING: checkpoint has no 'dt' field. "
                f"Assuming training dt={self.dt:.4f}. "
                f"If your sim runs at a different dt, particles may diverge."
            )

    def loss_history(self) -> List[float]:
        return self._losses


class CFMShim:
    """
    Drop-in wrapper so ParticleFilter.py can call:

        self.motion_model.predict(X)

    exactly like ScratchTransformer.
    """

    def __init__(self, cfm: CFMMotionModel):
        self._cfm = cfm
        self.dt = cfm.dt

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Input:
            X: (N_particles, N_th * input_dim)

        Output:
            velocity predictions: (N_particles, 2)
        """
        return self._cfm.propose(X.astype(np.float32))

    def propose(self, X: np.ndarray) -> np.ndarray:
        """
        Optional convenience method.
        """
        return self._cfm.propose(X.astype(np.float32))

    def parameters(self):
        return self._cfm.network.parameters()

    def state_dict(self):
        return self._cfm.network.state_dict()

    def load_state_dict(self, state_dict):
        return self._cfm.network.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        if mode:
            self._cfm.network.train()
        else:
            self._cfm.network.eval()
        return self

    def eval(self):
        self._cfm.network.eval()
        return self

    def online_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 5,
        lr: float = 5e-4,
    ):
        """
        Online retraining hook for ParticleFilter.optimize_learned_model().
        """
        batch_size = min(64, len(X))
        self._cfm.train(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            print_every=epochs + 1,
        )


def load_cfm_model(
    model_path: str,
    input_dim: int = 2,
    block_size: int = 10,
    n_embed: int = 64,
    n_head: int = 8,
    n_layer: int = 2,
    vel_hidden: int = 128,
    N_FE: int = 10,
    dropout: float = 0.1,
    anchor_sigma: float = 0.3,
    dt: float = 0.033,
) -> CFMShim:
    """
    Load a saved CFM model and return a CFMShim.

    The function arguments are fallbacks. If the checkpoint contains a config,
    load() will rebuild the model using the saved values automatically.

    Usage in ParticleFilter.py:

        from mml_core.motion.cfm_motion_model import load_cfm_model

        self.motion_model = load_cfm_model(
            model_path=model_path,
            input_dim=2,
            block_size=self.N_th,
            dt=0.033,
        )
    """
    cfm = CFMMotionModel(
        input_dim=input_dim,
        block_size=block_size,
        n_embed=n_embed,
        n_head=n_head,
        n_layer=n_layer,
        vel_hidden=vel_hidden,
        N_FE=N_FE,
        dropout=dropout,
        anchor_sigma=anchor_sigma,
        dt=dt,
    )

    cfm.load(model_path)
    return CFMShim(cfm)