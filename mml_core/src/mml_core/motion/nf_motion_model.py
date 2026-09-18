"""
nf_motion_model.py
===================

NFMotionModel — conditional normalizing-flow motion model, adapted from
NF-DPF (Normalizing Flow-based Differentiable Particle Filters,
arXiv 2403.01499) as a fourth learned comparison point alongside
CFMMotionModel (OT-CFM), DiffusionMotionModel (DDPM), and GMMMotionModel.

NFShim — drop-in wrapper with the same .predict() signature as the other
shims, so ParticleFilter.py can call self.motion_model.predict(X).

What is kept from NF-DPF
------------------------
* The conditional RealNVP architecture (RealNVP_cond in their nf/flows.py):
  affine coupling layers whose scale/translation nets are small Tanh MLPs
  conditioned on an encoded context, stacked n_flows deep (their
  n_sequence = 2), trained by exact maximum likelihood.
* The flow's role: a learned dynamic/proposal model inside a particle
  filter, sampling the next state (here: velocity) with an exactly
  computable density.

What is adapted (report in paper)
-------------------------------------
* Their conditioning is an encoded image observation; ours is the encoded
  position-history window (the same ContextEncoder all our motion models
  share, so capacity is matched across CFM / Diffusion / GMM / NF).
* The full NF-DPF also learns a measurement model and trains end-to-end
  through differentiable resampling. Our task has a known Gaussian
  measurement likelihood over direct position measurements, so only the
  flow-based dynamic model — the transferable component — is ported.

Training target:
    y = (vx, vy)
    L = -log p(y | c)   (exact, via change of variables)

Inference:
    y = f^{-1}(z; c),  z ~ N(0, I)  — single pass, no iteration
    (same latency class as GMM; no ODE steps, no denoising loop).

Place at:
    mml_guidance/mml_guidance/mml_network/nf_motion_model.py
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mml_core.motion.velocity_network import NFVelocityNetwork


device = "cuda" if torch.cuda.is_available() else "cpu"


class NFMotionModel:
    """
    Context-conditioned conditional-RealNVP motion model.

    Parameters
    ----------
    input_dim, block_size, n_embed, n_head, n_layer, dropout:
        Shared ContextEncoder config — same as the other motion models.
    n_flows:
        Number of stacked conditional RealNVP blocks (2 matches the
        NF-DPF n_sequence default).
    flow_hidden:
        Hidden width of the coupling s/t networks.
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
        n_flows: int = 2,
        flow_hidden: int = 32,
        dropout: float = 0.1,
        dt: float = 0.033,
    ):
        self.input_dim = int(input_dim)
        self.block_size = int(block_size)
        self.n_embed = int(n_embed)
        self.n_head = int(n_head)
        self.n_layer = int(n_layer)
        self.n_flows = int(n_flows)
        self.flow_hidden = int(flow_hidden)
        self.dropout = float(dropout)
        self.dt = float(dt)

        self.network = NFVelocityNetwork(
            input_dim=self.input_dim,
            block_size=self.block_size,
            n_embed=self.n_embed,
            n_head=self.n_head,
            n_layer=self.n_layer,
            n_flows=self.n_flows,
            flow_hidden=self.flow_hidden,
            dropout=self.dropout,
        ).to(device)

        self._losses: List[float] = []

    # ------------------------------------------------------------------
    # Shared context contract
    # ------------------------------------------------------------------

    def _expected_context_dim(self) -> int:
        return self.block_size * self.input_dim

    def _validate_context(self, c_flat: np.ndarray) -> None:
        if c_flat.ndim != 2:
            raise ValueError(
                f"NF context must be 2D, got shape {tuple(c_flat.shape)}."
            )
        expected = self._expected_context_dim()
        if c_flat.shape[1] != expected:
            raise ValueError(
                "NF context shape mismatch: "
                f"got {tuple(c_flat.shape)}, but expected (batch, {expected}) = "
                f"(batch, block_size={self.block_size} * input_dim={self.input_dim})."
            )

    # ------------------------------------------------------------------
    # Training — exact maximum likelihood
    # ------------------------------------------------------------------

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
        Train by minimizing exact NLL. Parameters mirror the other motion
        models so all four train on identical data with identical schedules.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._validate_context(X)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"Training y has shape {y.shape}, expected (N, 2).")

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        self.network.train()
        self._losses = []

        for epoch in range(1, epochs + 1):
            epoch_losses = []

            for c_batch, y_batch in dataloader:
                c_batch = c_batch.to(device)
                y_batch = y_batch.to(device)

                log_p = self.network(y_batch, c_batch)
                loss = -log_p.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Coupling-layer exponentials can spike gradients early on.
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
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
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def propose(self, c_flat: np.ndarray) -> np.ndarray:
        """
        One velocity sample per particle: encode context, one inverse flow
        pass from z ~ N(0, I). Same signature as the other motion models.
        """
        c_flat = np.asarray(c_flat, dtype=np.float32)
        self._validate_context(c_flat)

        self.network.eval()
        c_t = torch.tensor(c_flat, dtype=torch.float32, device=device)
        emb = self.network.encode_context(c_t)
        v = self.network.sample(emb).cpu().numpy().astype(np.float32)

        MAX_SPEED = 1.5  # m/s — same convention as the other motion models
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * np.minimum(1.0, MAX_SPEED / (speed + 1e-9))
        return v

    @torch.no_grad()
    def log_prob(self, c_flat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Exact conditional log-density log p(y | c) — the flow's signature
        capability (CFM/DDPM cannot provide this cheaply). Returns (N,).
        """
        c_flat = np.asarray(c_flat, dtype=np.float32)
        self._validate_context(c_flat)
        c_t = torch.tensor(c_flat, dtype=torch.float32, device=device)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32), device=device)
        emb = self.network.encode_context(c_t)
        return self.network.log_prob(y_t, emb).cpu().numpy()

    def propose_batched(
        self,
        c_flat: np.ndarray,
        n_samples_per_context: int = 1,
    ) -> np.ndarray:
        """Multiple proposals per context: (N_contexts, n_samples, 2)."""
        samples = []
        for _ in range(n_samples_per_context):
            samples.append(self.propose(c_flat))
        return np.stack(samples, axis=1)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

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
                    "n_flows": self.n_flows,
                    "flow_hidden": self.flow_hidden,
                    "dropout": self.dropout,
                    "dt": self.dt,
                },
            },
            path,
        )
        print(f"[NFMotionModel] saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})

        loaded = {k: cfg.get(k) for k in (
            "input_dim", "block_size", "n_embed", "n_head", "n_layer",
            "n_flows", "flow_hidden", "dropout", "dt",
        )}

        needs_rebuild = any(
            loaded[k] is not None and loaded[k] != getattr(self, k)
            for k in ("input_dim", "block_size", "n_embed", "n_head",
                      "n_layer", "n_flows", "flow_hidden", "dropout")
        )

        for k, v in loaded.items():
            if v is not None:
                setattr(self, k, type(getattr(self, k))(v))

        if needs_rebuild:
            self.network = NFVelocityNetwork(
                input_dim=self.input_dim,
                block_size=self.block_size,
                n_embed=self.n_embed,
                n_head=self.n_head,
                n_layer=self.n_layer,
                n_flows=self.n_flows,
                flow_hidden=self.flow_hidden,
                dropout=self.dropout,
            ).to(device)

        self.network.load_state_dict(ckpt["state_dict"])
        self.network.eval()

        print(
            f"[NFMotionModel] loaded ← {path}  "
            f"(input_dim={self.input_dim}, block_size={self.block_size}, "
            f"n_embed={self.n_embed}, n_head={self.n_head}, "
            f"n_layer={self.n_layer}, n_flows={self.n_flows}, "
            f"flow_hidden={self.flow_hidden}, dt={self.dt})"
        )

        if "dt" not in cfg:
            print(
                f"[NFMotionModel] WARNING: checkpoint has no 'dt' field. "
                f"Assuming training dt={self.dt:.4f}."
            )

    def loss_history(self) -> List[float]:
        return self._losses


class NFShim:
    """
    Drop-in wrapper so ParticleFilter.py can call:

        self.motion_model.predict(X)

    exactly like ScratchTransformer / CFMShim / DiffusionShim / GMMShim.
    """

    def __init__(self, nf: NFMotionModel):
        self._nf = nf
        self.dt = nf.dt

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._nf.propose(X.astype(np.float32))

    def propose(self, X: np.ndarray) -> np.ndarray:
        return self._nf.propose(X.astype(np.float32))

    def parameters(self):
        return self._nf.network.parameters()

    def state_dict(self):
        return self._nf.network.state_dict()

    def load_state_dict(self, state_dict):
        return self._nf.network.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        if mode:
            self._nf.network.train()
        else:
            self._nf.network.eval()
        return self

    def eval(self):
        self._nf.network.eval()
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
        self._nf.train(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            print_every=epochs + 1,
        )


def load_nf_model(
    model_path: str,
    input_dim: int = 2,
    block_size: int = 10,
    n_embed: int = 64,
    n_head: int = 8,
    n_layer: int = 2,
    n_flows: int = 2,
    flow_hidden: int = 32,
    dropout: float = 0.1,
    dt: float = 0.033,
) -> NFShim:
    """
    Load a saved NF motion model and return an NFShim. Checkpoint config
    wins over the fallback arguments, like the other loaders.
    """
    nf = NFMotionModel(
        input_dim=input_dim,
        block_size=block_size,
        n_embed=n_embed,
        n_head=n_head,
        n_layer=n_layer,
        n_flows=n_flows,
        flow_hidden=flow_hidden,
        dropout=dropout,
        dt=dt,
    )
    nf.load(model_path)
    return NFShim(nf)
