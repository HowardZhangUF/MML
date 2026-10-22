"""
TCFM (Ye et al., "Efficient Trajectory Forecasting and Generation with
Conditional Flow Matching", arXiv:2403.10809) ported to the single-target road
benchmark.

Upstream: github.com/CORE-Robotics-Lab/TCFM @ b14f04d. See README.md in this
folder for provenance and deviations.

Mechanism kept from upstream
    * diffuser/models/cfm.py CFM.loss: torchcfm ConditionalFlowMatcher with
      sigma = 0 (x0 ~ N(0, I), x_t = t x1 + (1 - t) x0, u_t = x1 - x0), MSE
      between the predicted and conditional velocity over the whole future
      trajectory.
    * CFM.p_sample_loop_cfm: Euler integration of the learned velocity field
      from t = 0 to 1 (torchdiffeq.odeint(method="euler")).
    * diffuser/models/temporal_film.py ConditionalUnet1D: 1-D temporal U-Net
      of conditional residual blocks (Conv1d-GroupNorm-Mish), FiLM-style
      conditioning on [LSTM(history) | sinusoidal time embedding], strided
      down / transposed up-sampling, skip connections.
    * diffuser/models/encoder.py EncoderRNN: one-layer LSTM over the
      history, last hidden state as the condition.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPosEmb(nn.Module):
    """diffuser/models/helpers.py, verbatim."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Mish(nn.Module):
    """x * tanh(softplus(x)): identical to nn.Mish, ~1.7x faster on CPU at filter batch sizes."""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish (helpers.Conv1dBlock)."""

    def __init__(self, c_in, c_out, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2),
                                   nn.GroupNorm(n_groups, c_out), Mish())

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """temporal_film.ConditionalResidualBlock1D (cond_predict_scale=False: additive FiLM bias)."""

    def __init__(self, c_in, c_out, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(c_in, c_out, kernel_size, n_groups),
                                     Conv1dBlock(c_out, c_out, kernel_size, n_groups)])
        self.cond_encoder = nn.Sequential(Mish(), nn.Linear(cond_dim, c_out))
        self.residual_conv = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x) + self.cond_encoder(cond)[..., None]
        return self.blocks[1](out) + self.residual_conv(x)


class EncoderRNN(nn.Module):
    """diffuser/models/encoder.py: LSTM, last hidden state."""

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return hn[-1]


class ConditionalUnet1D(nn.Module):
    """temporal_film.ConditionalUnet1D with the LSTM 'detections' encoder (encoder_type='lstm')."""

    def __init__(self, transition_dim=2, dim=16, dim_mults=(1, 2, 4), lstm_in_dim=6, lstm_out_dim=32,
                 kernel_size=3, n_groups=8):
        super().__init__()
        down_dims = [dim * m for m in dim_mults]
        all_dims = [transition_dim] + down_dims
        start_dim = down_dims[0]
        self.diffusion_step_encoder = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), Mish(),
                                                    nn.Linear(dim * 4, dim))
        cond_dim = dim + lstm_out_dim
        self.film_encoder = EncoderRNN(lstm_in_dim, lstm_out_dim)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid = all_dims[-1]
        self.mid_modules = nn.ModuleList([ConditionalResidualBlock1D(mid, mid, cond_dim, kernel_size, n_groups),
                                          ConditionalResidualBlock1D(mid, mid, cond_dim, kernel_size, n_groups)])
        self.down_modules = nn.ModuleList()
        for i, (c_in, c_out) in enumerate(in_out):
            last = i >= len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(c_in, c_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(c_out, c_out, cond_dim, kernel_size, n_groups),
                nn.Conv1d(c_out, c_out, 3, 2, 1) if not last else nn.Identity()]))
        self.up_modules = nn.ModuleList()
        for i, (c_in, c_out) in enumerate(reversed(in_out[1:])):
            last = i >= len(in_out) - 1
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(c_out * 2, c_in, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(c_in, c_in, cond_dim, kernel_size, n_groups),
                nn.ConvTranspose1d(c_in, c_in, 4, 2, 1) if not last else nn.Identity()]))
        self.final_conv = nn.Sequential(Conv1dBlock(start_dim, start_dim, kernel_size),
                                        nn.Conv1d(start_dim, transition_dim, 1))

    def encode(self, history):
        return self.film_encoder(history)

    def forward(self, t, x, hist_code):
        """t [B] in [0, 1], x [B, H, 2], hist_code [B, lstm_out_dim] -> velocity [B, H, 2]."""
        feat = torch.cat([hist_code, self.diffusion_step_encoder(t.expand(x.shape[0]))], dim=-1)
        x = x.transpose(1, 2)
        h = []
        for res1, res2, down in self.down_modules:
            x = res2(res1(x, feat), feat)
            h.append(x)
            x = down(x)
        for m in self.mid_modules:
            x = m(x, feat)
        for res1, res2, up in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = up(res2(res1(x, feat), feat))
        return self.final_conv(x).transpose(1, 2)


class TCFM(nn.Module):
    """CFM (diffuser/models/cfm.py): sigma-0 conditional flow matching + Euler sampler."""

    def __init__(self, net: ConditionalUnet1D, q_norm, horizon, n_timesteps=10):
        super().__init__()
        self.net, self.H, self.n_timesteps = net, horizon, n_timesteps
        self.register_buffer("q_lo", torch.as_tensor(q_norm.lo))
        self.register_buffer("q_hi", torch.as_tensor(q_norm.hi))

    def loss(self, hist, x1):
        """x1 [B, H, 2] normalized future."""
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = t[:, None, None] * x1 + (1 - t[:, None, None]) * x0
        vt = self.net(t, xt, self.net.encode(hist))
        return torch.mean((vt - (x1 - x0)) ** 2)

    @torch.no_grad()
    def sample(self, hist):
        """One trajectory per history, metric [B, 1, H, 2], with probability 1."""
        code = self.net.encode(hist)
        x = torch.randn(hist.shape[0], self.H, 2, device=hist.device)
        ts = torch.linspace(0, 1, self.n_timesteps + 1, device=hist.device)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            x = x + (t1 - t0) * self.net(t0[None], x, code)
        x = (x + 1) / 2 * (self.q_hi - self.q_lo) + self.q_lo
        return x[:, None], torch.ones(hist.shape[0], 1, device=hist.device)
