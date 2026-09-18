"""
velocity_network.py
===================
Defines the context-conditioned velocity network  v_theta(x_tau, tau, context).

Architecture
------------
               context c_k                    ODE state
               (N_th x input_dim)             (x_tau, tau)
                      |                            |
            ContextEncoder                         |
          (ScratchTransformer                      |
             backbone, reused)                     |
                      |                            |
               e  [B, n_embed]             [B, 3]  (x, y, tau)
                      |___________________________|
                                  |
                           VelocityHead
                        Linear(n_embed+3, h)
                               ReLU
                        Linear(h, h)
                               ReLU
                        Linear(h, 2)
                                  |
                         v  [B, 2]   (velocity direction)

The network is trained with the CFM (OT-CFM) loss:
    L = || v_theta(x_tau, tau, c_k) - (x_{k+1} - x_0) ||^2
where x_tau = (1-tau)*x_0 + tau*x_{k+1}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Transformer building blocks (same as ScratchTransformer) ─────────────────

class _Head(nn.Module):
    """Single causal self-attention head."""
    def __init__(self, head_size, input_dim, block_size, dropout=0.1):
        super().__init__()
        self.key   = nn.Linear(input_dim, head_size, bias=False)
        self.query = nn.Linear(input_dim, head_size, bias=False)
        self.value = nn.Linear(input_dim, head_size, bias=False)
        self.register_buffer("tril",
                             torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)
        return wei @ v


class _MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, input_dim, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            _Head(head_size, input_dim, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj    = nn.Linear(head_size * num_heads, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class _FeedForward(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10 * input_dim),
            nn.ReLU(),
            nn.Linear(10 * input_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _Block(nn.Module):
    def __init__(self, input_dim, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = input_dim // n_head
        self.sa   = _MultiHeadAttention(n_head, head_size, input_dim,
                                        dropout, block_size)
        self.ffwd = _FeedForward(input_dim, dropout)
        self.ln1  = nn.LayerNorm(input_dim)
        self.ln2  = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ── Context Encoder ──────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    """
    Encodes the measurement history window c_k into a fixed-size embedding.

    Input:  (B, N_th * input_dim)  flat history vector
    Output: (B, n_embed)           context embedding
    """
    def __init__(self, input_dim: int = 2, block_size: int = 5,
                 n_embed: int = 40, n_head: int = 6,
                 n_layer: int = 1, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.input_dim  = input_dim
        self.n_embed    = n_embed

        self.embed     = nn.Linear(input_dim, n_embed)
        self.pos_embed = nn.Embedding(block_size, n_embed)
        self.blocks    = nn.Sequential(*[
            _Block(n_embed, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed)

        # weight init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat: (B, T * C)  where T = block_size, C = input_dim
        returns: (B, n_embed)
        """
        B = x_flat.shape[0]
        x = x_flat.view(B, self.block_size, self.input_dim)  # (B, T, C)
        x = self.embed(x)                                      # (B, T, n_embed)
        pos = self.pos_embed(
            torch.arange(self.block_size, device=x.device))   # (T, n_embed)
        x = x + pos                                            # (B, T, n_embed)
        x = self.blocks(x)                                     # (B, T, n_embed)
        x = self.ln_f(x)                                       # (B, T, n_embed)
        return x[:, -1, :]                                     # (B, n_embed) last token


# ── Velocity Head ─────────────────────────────────────────────────────────────

class VelocityHead(nn.Module):
    """
    Maps (x_tau, tau, context_embedding) -> velocity direction.

    Input:  x_tau [B,2], tau [B,1], emb [B, n_embed]
            concatenated -> [B, n_embed + 3]
    Output: [B, 2]  velocity direction
    """
    def __init__(self, n_embed: int = 40, hidden_dim: int = 64):
        super().__init__()
        in_dim = n_embed + 2 + 1   # embedding + x_tau(2) + tau(1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_tau: torch.Tensor, tau: torch.Tensor,
                emb: torch.Tensor) -> torch.Tensor:
        """
        x_tau: (B, 2)
        tau:   (B, 1)  or scalar broadcast
        emb:   (B, n_embed)
        """
        if tau.dim() == 0:
            tau = tau.unsqueeze(0).expand(x_tau.shape[0], 1)
        elif tau.dim() == 1:
            tau = tau.unsqueeze(-1)
        inp = torch.cat([emb, x_tau, tau], dim=-1)  # (B, n_embed+3)
        return self.net(inp)


# ── Combined network ──────────────────────────────────────────────────────────

class CFMVelocityNetwork(nn.Module):
    """
    Full v_theta(x_tau, tau, c_k) network.

    At training time:
        forward(x_tau, tau, c_flat) -> (B, 2)

    At inference (particle filter propose step):
        encode_context(c_flat)           -> emb  (B, n_embed)
        predict_velocity(x_tau, tau, emb) -> vel  (B, 2)

    Splitting encode/predict lets you encode the context ONCE per
    time step and reuse it for all N_FE Euler integration steps,
    rather than re-running the transformer N_FE times per particle.
    """
    def __init__(self, input_dim: int = 2, block_size: int = 5,
                 n_embed: int = 40, n_head: int = 6,
                 n_layer: int = 1, vel_hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = ContextEncoder(
            input_dim=input_dim, block_size=block_size,
            n_embed=n_embed, n_head=n_head,
            n_layer=n_layer, dropout=dropout,
        )
        self.vel_head = VelocityHead(n_embed=n_embed, hidden_dim=vel_hidden)

    def encode_context(self, c_flat: torch.Tensor) -> torch.Tensor:
        """c_flat: (B, T*C) -> emb: (B, n_embed)"""
        return self.encoder(c_flat)

    def predict_velocity(self, x_tau: torch.Tensor, tau,
                         emb: torch.Tensor) -> torch.Tensor:
        """x_tau: (B,2), tau: scalar or (B,1), emb: (B, n_embed) -> (B, 2)"""
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=torch.float32,
                               device=x_tau.device)
        return self.vel_head(x_tau, tau, emb)

    def forward(self, x_tau: torch.Tensor, tau,
                c_flat: torch.Tensor) -> torch.Tensor:
        """Full forward pass — used during training."""
        emb = self.encode_context(c_flat)
        return self.predict_velocity(x_tau, tau, emb)


# ── Diffusion (DDPM) noise-prediction head ───────────────────────────────────

class _SinusoidalTimestepEmbedding(nn.Module):
    """
    Standard transformer/DDPM sinusoidal embedding for a discrete timestep t.

    Input:  t (B,) integer or float timestep indices in [0, T-1]
    Output: (B, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class NoisePredictionHead(nn.Module):
    """
    Maps (y_t, t, context_embedding) -> predicted noise epsilon.

    Same shape contract as VelocityHead, but conditions on a discrete
    diffusion timestep (via sinusoidal embedding) instead of continuous tau.

    Input:  y_t [B,2], t [B] (long, in [0, T-1]), emb [B, n_embed]
    Output: [B, 2]  predicted epsilon
    """
    def __init__(self, n_embed: int = 40, hidden_dim: int = 64,
                 time_embed_dim: int = 32):
        super().__init__()
        self.time_embed = _SinusoidalTimestepEmbedding(time_embed_dim)
        in_dim = n_embed + 2 + time_embed_dim   # context + y_t(2) + time_embed
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, y_t: torch.Tensor, t: torch.Tensor,
                emb: torch.Tensor) -> torch.Tensor:
        """
        y_t: (B, 2)
        t:   (B,)  long timestep indices
        emb: (B, n_embed)
        """
        t_emb = self.time_embed(t)                      # (B, time_embed_dim)
        inp = torch.cat([emb, y_t, t_emb], dim=-1)       # (B, n_embed+2+time_embed_dim)
        return self.net(inp)


class DiffusionVelocityNetwork(nn.Module):
    """
    Full epsilon_theta(y_t, t, c_k) network for DDPM-style training.

    Reuses the exact same ContextEncoder as CFMVelocityNetwork so the
    comparison between CFM and diffusion isolates the generative mechanism
    (flow matching vs. denoising diffusion), not the context representation.

    At training time:
        forward(y_t, t, c_flat) -> predicted epsilon (B, 2)

    At inference (reverse diffusion loop):
        encode_context(c_flat)      -> emb (B, n_embed)   [run ONCE]
        predict_noise(y_t, t, emb)  -> eps (B, 2)          [run T times]
    """
    def __init__(self, input_dim: int = 2, block_size: int = 5,
                 n_embed: int = 40, n_head: int = 6,
                 n_layer: int = 1, noise_hidden: int = 64,
                 time_embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = ContextEncoder(
            input_dim=input_dim, block_size=block_size,
            n_embed=n_embed, n_head=n_head,
            n_layer=n_layer, dropout=dropout,
        )
        self.noise_head = NoisePredictionHead(
            n_embed=n_embed, hidden_dim=noise_hidden,
            time_embed_dim=time_embed_dim,
        )

    def encode_context(self, c_flat: torch.Tensor) -> torch.Tensor:
        """c_flat: (B, T*C) -> emb: (B, n_embed)"""
        return self.encoder(c_flat)

    def predict_noise(self, y_t: torch.Tensor, t: torch.Tensor,
                       emb: torch.Tensor) -> torch.Tensor:
        """y_t: (B,2), t: (B,) long, emb: (B, n_embed) -> (B, 2)"""
        return self.noise_head(y_t, t, emb)

    def forward(self, y_t: torch.Tensor, t: torch.Tensor,
                c_flat: torch.Tensor) -> torch.Tensor:
        """Full forward pass — used during training."""
        emb = self.encode_context(c_flat)
        return self.predict_noise(y_t, t, emb)


# ── Gaussian Mixture (MDN) head ──────────────────────────────────────────────

class GMMHead(nn.Module):
    """
    Maps a context embedding to the parameters of a K-component 2D Gaussian
    mixture over velocity (vx, vy) — a single forward pass, no iterative
    sampling loop (unlike the CFM ODE or diffusion's reverse process).

    Diagonal covariance per component (independent vx/vy variances) — this
    keeps the parameter count comparable to the other heads and avoids the
    extra machinery needed to keep a full 2x2 covariance positive-definite.

    Input:  emb [B, n_embed]
    Output: pi [B, K], mu [B, K, 2], sigma [B, K, 2]  (sigma > 0 via softplus)
    """
    def __init__(self, n_embed: int = 40, hidden_dim: int = 64, K: int = 4,
                 min_sigma: float = 1e-3):
        super().__init__()
        self.K = int(K)
        self.min_sigma = float(min_sigma)
        self.trunk = nn.Sequential(
            nn.Linear(n_embed, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.pi_head = nn.Linear(hidden_dim, self.K)
        self.mu_head = nn.Linear(hidden_dim, self.K * 2)
        self.sigma_head = nn.Linear(hidden_dim, self.K * 2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, emb: torch.Tensor):
        """
        emb: (B, n_embed)
        returns: pi (B, K), mu (B, K, 2), sigma (B, K, 2)
        """
        B = emb.shape[0]
        h = self.trunk(emb)
        pi = F.softmax(self.pi_head(h), dim=-1)                       # (B, K)
        mu = self.mu_head(h).view(B, self.K, 2)                        # (B, K, 2)
        sigma = F.softplus(self.sigma_head(h)).view(B, self.K, 2)      # (B, K, 2)
        sigma = sigma + self.min_sigma
        return pi, mu, sigma


class GMMVelocityNetwork(nn.Module):
    """
    Full context-conditioned Gaussian-mixture velocity network.

    Reuses the exact same ContextEncoder as CFMVelocityNetwork /
    DiffusionVelocityNetwork so the three generative approaches (flow
    matching, denoising diffusion, mixture density) are compared on equal
    footing — same context representation, different output distribution.

    At training time:
        forward(c_flat) -> (pi, mu, sigma)   used in a GMM negative
                                              log-likelihood loss against y

    At inference:
        encode_context(c_flat) -> emb   (B, n_embed)   [run ONCE]
        predict_mixture(emb)   -> (pi, mu, sigma)       [single forward pass,
                                                          no iteration needed]
    """
    def __init__(self, input_dim: int = 2, block_size: int = 5,
                 n_embed: int = 40, n_head: int = 6,
                 n_layer: int = 1, gmm_hidden: int = 64,
                 K: int = 4, dropout: float = 0.1):
        super().__init__()
        self.K = int(K)
        self.encoder = ContextEncoder(
            input_dim=input_dim, block_size=block_size,
            n_embed=n_embed, n_head=n_head,
            n_layer=n_layer, dropout=dropout,
        )
        self.gmm_head = GMMHead(n_embed=n_embed, hidden_dim=gmm_hidden, K=K)

    def encode_context(self, c_flat: torch.Tensor) -> torch.Tensor:
        """c_flat: (B, T*C) -> emb: (B, n_embed)"""
        return self.encoder(c_flat)

    def predict_mixture(self, emb: torch.Tensor):
        """emb: (B, n_embed) -> pi (B,K), mu (B,K,2), sigma (B,K,2)"""
        return self.gmm_head(emb)

    def forward(self, c_flat: torch.Tensor):
        """Full forward pass — used during training."""
        emb = self.encode_context(c_flat)
        return self.predict_mixture(emb)


# ── Conditional normalizing flow (NF-DPF-style RealNVP) head ─────────────────

class _FlowFCNN(nn.Module):
    """
    Scale/translation network for a coupling layer — mirrors FCNN in
    Normalizing-Flows-DPFs/nf/flows.py (two hidden Tanh layers).
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)


class _CondRealNVPBlock(nn.Module):
    """
    One conditional RealNVP block — a faithful port of RealNVP_cond from
    Normalizing-Flows-DPFs/nf/flows.py: two affine coupling sub-layers per
    block, each s/t network conditioned on the context embedding. For 2D
    velocity (dim=2) each coupling transforms one coordinate given the
    other. The scale outputs are clamped for numerical stability (their
    code relies on small-variance init alone).
    """
    S_CLAMP = 5.0

    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.t1 = _FlowFCNN(half + cond_dim, half, hidden_dim)
        self.s1 = _FlowFCNN(half + cond_dim, half, hidden_dim)
        self.t2 = _FlowFCNN(half + cond_dim, half, hidden_dim)
        self.s2 = _FlowFCNN(half + cond_dim, half, hidden_dim)

        # Small-variance init keeps the flow near identity at start, as in
        # RealNVP_cond.zero_initialization(). NF-DPF's build_conditional_nf()
        # calls it with var=0.01; we use 0.1 (the function's own default),
        # which trained stably here alongside the S_CLAMP below.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """x -> z (data to latent). Returns (z, log_det)."""
        lower, upper = x[:, : self.dim // 2], x[:, self.dim // 2:]
        s1 = torch.clamp(self.s1(torch.cat([lower, cond], dim=-1)),
                         -self.S_CLAMP, self.S_CLAMP)
        t1 = self.t1(torch.cat([lower, cond], dim=-1))
        upper = t1 + upper * torch.exp(s1)
        s2 = torch.clamp(self.s2(torch.cat([upper, cond], dim=-1)),
                         -self.S_CLAMP, self.S_CLAMP)
        t2 = self.t2(torch.cat([upper, cond], dim=-1))
        lower = t2 + lower * torch.exp(s2)
        z = torch.cat([lower, upper], dim=1)
        log_det = s1.sum(dim=1) + s2.sum(dim=1)
        return z, log_det

    def inverse(self, z: torch.Tensor, cond: torch.Tensor):
        """z -> x (latent to data). Returns (x, log_det of the inverse)."""
        lower, upper = z[:, : self.dim // 2], z[:, self.dim // 2:]
        s2 = torch.clamp(self.s2(torch.cat([upper, cond], dim=-1)),
                         -self.S_CLAMP, self.S_CLAMP)
        t2 = self.t2(torch.cat([upper, cond], dim=-1))
        lower = (lower - t2) * torch.exp(-s2)
        s1 = torch.clamp(self.s1(torch.cat([lower, cond], dim=-1)),
                         -self.S_CLAMP, self.S_CLAMP)
        t1 = self.t1(torch.cat([lower, cond], dim=-1))
        upper = (upper - t1) * torch.exp(-s1)
        x = torch.cat([lower, upper], dim=1)
        log_det = -s1.sum(dim=1) - s2.sum(dim=1)
        return x, log_det


class NFVelocityNetwork(nn.Module):
    """
    Conditional normalizing-flow velocity model (NF-DPF-style baseline).

    Reuses the exact same ContextEncoder as the CFM / Diffusion / GMM
    networks; the output distribution is a stack of conditional RealNVP
    blocks over (vx, vy), giving EXACT log-likelihoods (the flow's
    signature property, and what distinguishes it from CFM's ODE which
    only supports sampling cheaply).

    Training:  maximize log p(y | c) = log N(f(y; c)) + log|det J_f|
    Sampling:  y = f^{-1}(z; c),  z ~ N(0, I)  — one pass, no iteration.
    """
    def __init__(self, input_dim: int = 2, block_size: int = 5,
                 n_embed: int = 40, n_head: int = 6,
                 n_layer: int = 1, n_flows: int = 2,
                 flow_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_flows = int(n_flows)
        self.encoder = ContextEncoder(
            input_dim=input_dim, block_size=block_size,
            n_embed=n_embed, n_head=n_head,
            n_layer=n_layer, dropout=dropout,
        )
        self.flows = nn.ModuleList([
            _CondRealNVPBlock(dim=2, cond_dim=n_embed, hidden_dim=flow_hidden)
            for _ in range(self.n_flows)
        ])

    def encode_context(self, c_flat: torch.Tensor) -> torch.Tensor:
        return self.encoder(c_flat)

    def log_prob(self, y: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Exact conditional log-density log p(y | c). Returns (B,)."""
        z = y
        total_log_det = torch.zeros(y.shape[0], device=y.device)
        for flow in self.flows:
            z, log_det = flow.forward(z, emb)
            total_log_det = total_log_det + log_det
        log_base = -0.5 * (z ** 2).sum(dim=1) - math.log(2.0 * math.pi)
        return log_base + total_log_det

    def sample(self, emb: torch.Tensor) -> torch.Tensor:
        """Draw one velocity sample per context embedding. Returns (B, 2)."""
        z = torch.randn(emb.shape[0], 2, device=emb.device)
        for flow in reversed(self.flows):
            z, _ = flow.inverse(z, emb)
        return z

    def forward(self, y: torch.Tensor, c_flat: torch.Tensor) -> torch.Tensor:
        """Training pass: exact log-likelihood of y given context."""
        emb = self.encode_context(c_flat)
        return self.log_prob(y, emb)
