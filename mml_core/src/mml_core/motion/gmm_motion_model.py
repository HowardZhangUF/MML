"""
gmm_motion_model.py
====================

GMMMotionModel — context-conditioned Gaussian Mixture (Mixture Density
Network) motion model, trained as a third comparison point alongside
CFMMotionModel (OT-CFM) and DiffusionMotionModel (DDPM).

GMMShim         — drop-in wrapper with the same .predict() signature as
                  CFMShim / DiffusionShim, so ParticleFilter.py can call:

                      self.motion_model.predict(X)

                  without changing predict_mml().

Main ideas
----------
Same task as CFM/Diffusion — generate a velocity sample (vx, vy) conditioned
on a flattened position history — but the output distribution is a
K-component Gaussian mixture predicted in ONE forward pass, rather than an
iterative ODE integration (CFM) or reverse denoising loop (Diffusion). This
makes it the natural "cheap multimodal" baseline: it can represent multiple
plausible futures at a junction (one mode per branch) without paying for
multiple network evaluations at inference time.

Training objective — maximize the GMM log-likelihood of the true velocity:

    L = -log( sum_k pi_k * N(y | mu_k, diag(sigma_k^2)) )

Inference — sample a component index ~ Categorical(pi), then sample
y ~ N(mu_k, diag(sigma_k^2)) for the chosen component. A single forward pass
through the encoder + head produces the full mixture; sampling from it is
free (no repeated network calls).

Place at:
    mml_guidance/mml_guidance/mml_network/gmm_motion_model.py
"""

from __future__ import annotations

import os
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mml_core.motion.velocity_network import GMMVelocityNetwork


device = "cuda" if torch.cuda.is_available() else "cpu"

_LOG_2PI = math.log(2.0 * math.pi)


def gmm_nll_loss(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
                  y: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of y under a diagonal-covariance Gaussian mixture.

    pi:    (B, K)      mixture weights, sums to 1 over K
    mu:    (B, K, 2)   component means
    sigma: (B, K, 2)   component std devs (diagonal covariance)
    y:     (B, 2)      target velocity

    Returns scalar mean NLL over the batch.
    """
    y = y.unsqueeze(1)                                    # (B, 1, 2)
    var = sigma ** 2                                       # (B, K, 2)

    # Per-dimension log N(y | mu, sigma^2), summed over the 2 dims
    # (diagonal covariance => log-density factorizes across dimensions).
    log_prob_per_dim = -0.5 * (
        _LOG_2PI + 2.0 * torch.log(sigma) + ((y - mu) ** 2) / var
    )                                                       # (B, K, 2)
    log_prob_k = log_prob_per_dim.sum(dim=-1)               # (B, K)

    log_pi = torch.log(pi + 1e-12)                          # (B, K)
    log_mix = torch.logsumexp(log_pi + log_prob_k, dim=-1)   # (B,)

    return -log_mix.mean()


class GMMMotionModel:
    """
    Context-conditioned Gaussian Mixture (MDN) motion model.

    Training target:
        y = (vx, vy)

    Training loss:
        Negative log-likelihood of y under the predicted K-component mixture.

    Inference:
        Single forward pass through encoder + head produces (pi, mu, sigma).
        Sample a component ~ Categorical(pi), then sample from that
        component's Gaussian. No iterative refinement needed.

    Parameters
    ----------
    input_dim, block_size, n_embed, n_head, n_layer, dropout:
        Same meaning as CFMMotionModel / DiffusionMotionModel — shared
        ContextEncoder config, for an apples-to-apples comparison.
    gmm_hidden:
        Hidden size of the GMM head MLP trunk.
    K:
        Number of mixture components.
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
        gmm_hidden: int = 128,
        K: int = 4,
        dropout: float = 0.1,
        dt: float = 0.033,
    ):
        self.input_dim = int(input_dim)
        self.block_size = int(block_size)
        self.n_embed = int(n_embed)
        self.n_head = int(n_head)
        self.n_layer = int(n_layer)
        self.gmm_hidden = int(gmm_hidden)
        self.K = int(K)
        self.dropout = float(dropout)
        self.dt = float(dt)

        self.network = GMMVelocityNetwork(
            input_dim=self.input_dim,
            block_size=self.block_size,
            n_embed=self.n_embed,
            n_head=self.n_head,
            n_layer=self.n_layer,
            gmm_hidden=self.gmm_hidden,
            K=self.K,
            dropout=self.dropout,
        ).to(device)

        self._losses: List[float] = []

    # ----------------------------------------------------------------------
    # Internal helpers (shared contract with CFM/Diffusion motion models)
    # ----------------------------------------------------------------------

    def _expected_context_dim(self) -> int:
        return self.block_size * self.input_dim

    def _validate_context_tensor(self, c_flat: torch.Tensor) -> None:
        if c_flat.ndim != 2:
            raise ValueError(
                f"GMM context must be 2D, got shape {tuple(c_flat.shape)}."
            )
        expected = self._expected_context_dim()
        got = c_flat.shape[1]
        if got != expected:
            raise ValueError(
                "GMM context shape mismatch: "
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
        Train the GMM head with negative log-likelihood.

        Parameters mirror CFMMotionModel.train() / DiffusionMotionModel.train()
        so all three models can be trained on identical data.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        expected_dim = self._expected_context_dim()
        if X.ndim != 2 or X.shape[1] != expected_dim:
            raise ValueError(
                f"Training X has shape {X.shape}, but GMM model expects "
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

        self.network.train()
        self._losses = []

        for epoch in range(1, epochs + 1):
            epoch_losses = []

            for c_batch, y_batch in dataloader:
                c_batch = c_batch.to(device)
                y_batch = y_batch.to(device)

                pi, mu, sigma = self.network(c_batch)
                loss = gmm_nll_loss(pi, mu, sigma, y_batch)

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
        Generate one velocity proposal per particle: single forward pass to
        get the mixture, then sample a component and draw from its Gaussian.
        Same call signature as CFMMotionModel.propose() / DiffusionMotionModel.propose().

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
                f"GMM propose() expected 2D context array, got {c_flat.shape}."
            )

        expected_dim = self._expected_context_dim()
        if c_flat.shape[1] != expected_dim:
            raise ValueError(
                f"GMM propose() context shape mismatch: got {c_flat.shape}, "
                f"but expected (N_particles, {expected_dim}) because "
                f"block_size={self.block_size}, input_dim={self.input_dim}."
            )

        self.network.eval()

        c_t = torch.tensor(c_flat, dtype=torch.float32, device=device)
        N = c_t.shape[0]

        # Single forward pass — no iterative refinement needed.
        pi, mu, sigma = self.network(c_t)   # (N,K), (N,K,2), (N,K,2)

        # Sample a component index per row, then sample from that Gaussian.
        component = torch.multinomial(pi, num_samples=1).squeeze(-1)   # (N,)
        idx = torch.arange(N, device=device)
        mu_sel = mu[idx, component]        # (N, 2)
        sigma_sel = sigma[idx, component]  # (N, 2)

        eps = torch.randn_like(mu_sel)
        y = mu_sel + sigma_sel * eps

        v = y.cpu().numpy().astype(np.float32)

        # Clip to physically plausible speed — same convention as CFM/Diffusion.
        MAX_SPEED = 1.5  # m/s
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * np.minimum(1.0, MAX_SPEED / (speed + 1e-9))

        return v

    @torch.no_grad()
    def propose_mode(self, c_flat: np.ndarray) -> np.ndarray:
        """
        Return the mean of the highest-weight mixture component per context
        (deterministic MAP-ish estimate, useful for accuracy comparisons
        against a single-shot regressor).
        """
        c_flat = np.asarray(c_flat, dtype=np.float32)
        c_t = torch.tensor(c_flat, dtype=torch.float32, device=device)
        N = c_t.shape[0]

        pi, mu, _ = self.network(c_t)
        best_k = torch.argmax(pi, dim=-1)   # (N,)
        idx = torch.arange(N, device=device)
        v = mu[idx, best_k].cpu().numpy().astype(np.float32)

        MAX_SPEED = 1.5
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
                    "gmm_hidden": self.gmm_hidden,
                    "K": self.K,
                    "dropout": self.dropout,
                    "dt": self.dt,
                },
            },
            path,
        )

        print(f"[GMMMotionModel] saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})

        loaded_input_dim = int(cfg.get("input_dim", self.input_dim))
        loaded_block_size = int(cfg.get("block_size", self.block_size))
        loaded_n_embed = int(cfg.get("n_embed", self.n_embed))
        loaded_n_head = int(cfg.get("n_head", self.n_head))
        loaded_n_layer = int(cfg.get("n_layer", self.n_layer))
        loaded_gmm_hidden = int(cfg.get("gmm_hidden", self.gmm_hidden))
        loaded_K = int(cfg.get("K", self.K))
        loaded_dropout = float(cfg.get("dropout", self.dropout))
        loaded_dt = float(cfg.get("dt", self.dt))

        needs_rebuild = (
            loaded_input_dim != self.input_dim
            or loaded_block_size != self.block_size
            or loaded_n_embed != self.n_embed
            or loaded_n_head != self.n_head
            or loaded_n_layer != self.n_layer
            or loaded_gmm_hidden != self.gmm_hidden
            or loaded_K != self.K
            or loaded_dropout != self.dropout
        )

        self.input_dim = loaded_input_dim
        self.block_size = loaded_block_size
        self.n_embed = loaded_n_embed
        self.n_head = loaded_n_head
        self.n_layer = loaded_n_layer
        self.gmm_hidden = loaded_gmm_hidden
        self.K = loaded_K
        self.dropout = loaded_dropout
        self.dt = loaded_dt

        if needs_rebuild:
            self.network = GMMVelocityNetwork(
                input_dim=self.input_dim,
                block_size=self.block_size,
                n_embed=self.n_embed,
                n_head=self.n_head,
                n_layer=self.n_layer,
                gmm_hidden=self.gmm_hidden,
                K=self.K,
                dropout=self.dropout,
            ).to(device)

        self.network.load_state_dict(ckpt["state_dict"])
        self.network.eval()

        print(
            f"[GMMMotionModel] loaded ← {path}  "
            f"(input_dim={self.input_dim}, block_size={self.block_size}, "
            f"n_embed={self.n_embed}, n_head={self.n_head}, "
            f"n_layer={self.n_layer}, gmm_hidden={self.gmm_hidden}, "
            f"K={self.K}, dt={self.dt})"
        )

        if "dt" not in cfg:
            print(
                f"[GMMMotionModel] WARNING: checkpoint has no 'dt' field. "
                f"Assuming training dt={self.dt:.4f}. "
                f"If your sim runs at a different dt, particles may diverge."
            )

    def loss_history(self) -> List[float]:
        return self._losses


class GMMShim:
    """
    Drop-in wrapper so ParticleFilter.py can call:

        self.motion_model.predict(X)

    exactly like ScratchTransformer / CFMShim / DiffusionShim.
    """

    def __init__(self, gmm: GMMMotionModel):
        self._gmm = gmm
        self.dt = gmm.dt

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._gmm.propose(X.astype(np.float32))

    def propose(self, X: np.ndarray) -> np.ndarray:
        return self._gmm.propose(X.astype(np.float32))

    def parameters(self):
        return self._gmm.network.parameters()

    def state_dict(self):
        return self._gmm.network.state_dict()

    def load_state_dict(self, state_dict):
        return self._gmm.network.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        if mode:
            self._gmm.network.train()
        else:
            self._gmm.network.eval()
        return self

    def eval(self):
        self._gmm.network.eval()
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
        self._gmm.train(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            print_every=epochs + 1,
        )


def load_gmm_model(
    model_path: str,
    input_dim: int = 2,
    block_size: int = 10,
    n_embed: int = 64,
    n_head: int = 8,
    n_layer: int = 2,
    gmm_hidden: int = 128,
    K: int = 4,
    dropout: float = 0.1,
    dt: float = 0.033,
) -> GMMShim:
    """
    Load a saved GMM model and return a GMMShim.

    The function arguments are fallbacks. If the checkpoint contains a
    config, load() will rebuild the model using the saved values.

    Usage in ParticleFilter.py:

        from mml_core.motion.gmm_motion_model import load_gmm_model

        self.motion_model = load_gmm_model(
            model_path=model_path,
            input_dim=2,
            block_size=self.N_th,
            dt=0.033,
        )
    """
    gmm = GMMMotionModel(
        input_dim=input_dim,
        block_size=block_size,
        n_embed=n_embed,
        n_head=n_head,
        n_layer=n_layer,
        gmm_hidden=gmm_hidden,
        K=K,
        dropout=dropout,
        dt=dt,
    )

    gmm.load(model_path)
    return GMMShim(gmm)
