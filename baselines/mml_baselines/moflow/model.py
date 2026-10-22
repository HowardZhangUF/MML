"""
MoFlow (Fu et al., CVPR 2025) ported to the single-target road benchmark.

Upstream: github.com/DSL-Lab/MoFlow @ f1b89b0, ETH-UCY configuration
(cfg/eth_ucy/cor_fm.yml, cfg/eth_ucy/imle.yml, fm_eth.py / imle_eth.py
defaults). See README.md in this folder for provenance and deviations.

Two stages, as upstream:
    1. Teacher: correlated flow matching over a set of K=20 future
       trajectories (models/flow_matching.py + models/backbone_eth_ucy.py
       ETHMotionTransformer): 'pred_data' objective with the 'direct'
       wrapper, logit-normal t (mean -0.5, std 1.5), timestep-dependent
       embedding dropout ('emb', k=20, m=0.5), agent-level winner-take-all
       regression (L2 per frame, summed over time) + cross-entropy on the
       winning component, 10-step Euler sampler.
    2. Student (the "MoFlow" model, models/imle.py + ETHIMLETransformer):
       one-step generator mapping one noise vector to a set of K
       trajectories, trained by IMLE with the set-to-set Chamfer distance to
       the teacher's K samples, best of M=30 noise vectors per example.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPosEmb(nn.Module):
    """models/context_encoder/mtr_encoder.py SinusoidalPosEmb, verbatim."""

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim, self.theta = dim, theta

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(self.theta) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def build_mlps(c_in, channels):
    """models/utils/common_layers.build_mlps(ret_before_act=True, without_norm=True)."""
    layers = []
    for k, c in enumerate(channels):
        layers.append(nn.Linear(c_in, c))
        if k + 1 < len(channels):
            layers.append(nn.ReLU())
        c_in = c
    return nn.Sequential(*layers)


def tf_layer(d, heads, ff, dropout):
    return nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff, dropout=dropout,
                                      batch_first=True)


class ETHEncoder(nn.Module):
    """
    models/context_encoder/eth_encoder.py: SocialTransformer (flattened past
    -> linear -> transformer across agents, residual, out-MLP), agent
    positional encoding, transformer across agents. One agent here.
    """

    def __init__(self, past_frames, in_feat, d, social_hidden, n_layers, heads, dropout, agents=1):
        super().__init__()
        self.encode_past = nn.Linear(past_frames * in_feat, social_hidden, bias=False)
        self.social = nn.TransformerEncoder(tf_layer(social_hidden, 2, social_hidden, 0.1), num_layers=2)
        self.mlp_out = nn.Linear(social_hidden, d)
        self.pos_encoding = nn.Sequential(SinusoidalPosEmb(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.agent_query_embedding = nn.Embedding(agents, d)
        self.mlp_pe = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.transformer_encoder = nn.TransformerEncoder(tf_layer(d, heads, 4 * d, dropout), num_layers=n_layers)

    def forward(self, past):                           # [B, A, P, F]
        h = self.encode_past(past.flatten(2))          # [B, A, H]
        h = self.mlp_out(h + self.social(h))           # [B, A, D]
        a = torch.arange(h.shape[1], device=h.device)
        pe = self.mlp_pe(torch.cat([self.agent_query_embedding(a), self.pos_encoding(a.float())], dim=-1))
        return self.transformer_encoder(h + pe[None])


def chunked(layer, x, size=32768):
    """Apply a layer over the batch dim in chunks (CUDA grid limit for B*M*K length-1 sequences)."""
    return x if x.shape[0] == 0 else torch.cat([layer(c) for c in x.split(size)])


def modulate(x, shift, scale):
    """[B, K, A, D] modulated by [B, D] (mtr_decoder.modulate)."""
    return x * (1 + scale[:, None, None]) + shift[:, None, None]


class MTRDecoder(nn.Module):
    """
    models/motion_decoder/mtr_decoder.py: per block, adaLN time modulation,
    K-to-K then A-to-A self-attention. Upstream re-reads the block *input*
    (`rearrange(query_token, ...)`) inside the loop, which discards the
    modulation and every earlier block; this port chains the blocks
    (`cur_query`), the evident intent. See README.
    """

    def __init__(self, d, heads, n_blocks, dropout, use_adaln=True):
        super().__init__()
        self.attn_k = nn.ModuleList([tf_layer(d, heads, 4 * d, dropout) for _ in range(n_blocks)])
        self.attn_a = nn.ModuleList([tf_layer(d, heads, 4 * d, dropout) for _ in range(n_blocks)])
        self.use_adaln = use_adaln
        if use_adaln:
            self.t_adaLN = nn.ModuleList()
            for _ in range(n_blocks):
                m = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)
                self.t_adaLN.append(m)

    def forward(self, q, t_emb=None):                  # [B, K, A, D]
        B, K, A, D = q.shape
        for i in range(len(self.attn_k)):
            if self.use_adaln:
                shift, scale = self.t_adaLN[i](t_emb).chunk(2, dim=-1)
                q = modulate(q, shift, scale)
            q = chunked(self.attn_k[i], q.permute(0, 2, 1, 3).reshape(B * A, K, D))
            q = q.view(B, A, K, D).permute(0, 2, 1, 3).reshape(B * K, A, D)
            q = chunked(self.attn_a[i], q).view(B, K, A, D)
        return q


class MoFlowTeacherNet(nn.Module):
    """ETHMotionTransformer (backbone_eth_ucy.py), single agent."""

    def __init__(self, horizon, num_query=20, d=32, past_frames=5, in_feat=6, social_hidden=64,
                 enc_layers=2, dec_blocks=2, heads=4, dropout=0.1):
        super().__init__()
        out_dim = 2 * horizon
        self.context_encoder = ETHEncoder(past_frames, in_feat, d, social_hidden, enc_layers, heads, dropout)
        self.motion_query_embedding = nn.Embedding(num_query, d)
        self.agent_order_embedding = nn.Embedding(1, d)
        self.post_pe_cat_mlp = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, d))
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.noisy_y_mlp = nn.Sequential(nn.Linear(out_dim, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.noisy_y_attn_k = tf_layer(d, 4, 4 * d, dropout)
        self.noisy_y_attn_a = tf_layer(d, 4, 4 * d, dropout)
        self.init_emb_fusion_mlp = nn.Sequential(nn.Linear(3 * d, d), nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, d))
        self.motion_decoder = MTRDecoder(d, heads, dec_blocks, dropout)
        self.reg_head = build_mlps(d, [d, 2 * d, out_dim])            # REGRESSION_MLPS [128, 256, 24]
        self.cls_head = build_mlps(d, [d, d, 1])                      # CLASSIFICATION_MLPS [128, 128, 1]
        self.drop_logi_k, self.drop_logi_m = 20.0, 0.5

    def forward(self, y, t, past):
        """y [B, K, A, H*2] noisy, t [B] in [0, 1], past [B, A, P, F] -> (data [B, K, A, H*2], cls [B, K, A])."""
        B, K, A, _ = y.shape
        enc = self.context_encoder(past)[:, None].expand(-1, K, -1, -1)
        y_emb = self.noisy_y_mlp(y)
        t_emb = self.time_mlp(t * 1000.0)                            # flow-matching time upscaling
        k_pe = self.motion_query_embedding(torch.arange(K, device=y.device))[None, :, None]
        a_pe = self.agent_order_embedding(torch.arange(A, device=y.device))[None, None]
        y_emb = y_emb + k_pe + a_pe
        D = y_emb.shape[-1]
        y_emb = self.noisy_y_attn_k(y_emb.permute(0, 2, 1, 3).reshape(B * A, K, D))
        y_emb = y_emb.view(B, A, K, D).permute(0, 2, 1, 3).reshape(B * K, A, D)
        y_emb = self.noisy_y_attn_a(y_emb).view(B, K, A, D)
        if self.training:                                            # drop_method 'emb'
            p_m = torch.sigmoid(self.drop_logi_k * (t - self.drop_logi_m))[:, None, None, None]
            y_emb = y_emb.masked_fill(torch.rand_like(p_m) < p_m, 0.0)
        fusion = self.init_emb_fusion_mlp(torch.cat([enc, y_emb, t_emb[:, None, None].expand(-1, K, A, -1)], -1))
        token = self.motion_decoder(self.post_pe_cat_mlp(fusion + k_pe + a_pe), t_emb)
        return self.reg_head(token), self.cls_head(token).squeeze(-1)


class MoFlowTeacher(nn.Module):
    """FlowMatcher (models/flow_matching.py) around MoFlowTeacherNet."""

    def __init__(self, net, q_norm, horizon, num_query=20, sampling_steps=10,
                 logit_norm_mean=-0.5, logit_norm_std=1.5):
        super().__init__()
        self.net, self.H, self.K = net, horizon, num_query
        self.register_buffer("q_lo", torch.as_tensor(q_norm.lo))
        self.register_buffer("q_hi", torch.as_tensor(q_norm.hi))
        self.sampling_steps = sampling_steps
        self.mu, self.sd = logit_norm_mean, logit_norm_std

    def unnormalize(self, y):                                        # [..., H*2] -> [..., H, 2] metric
        y = y.unflatten(-1, (self.H, 2))
        return (y + 1) / 2 * (self.q_hi - self.q_lo) + self.q_lo

    def loss(self, hist, q_norm_traj):
        """p_losses with LOSS_NN_MODE 'agent', reduction 'sum', weights reg 1 / cls 1."""
        B = hist.shape[0]
        past = hist[:, None]                                         # one agent
        y1 = q_norm_traj.flatten(1)[:, None, None].expand(-1, self.K, 1, -1)
        t = torch.sigmoid(torch.randn(B, device=hist.device) * self.sd + self.mu)
        noise = torch.randn_like(y1)                                 # tied_noise False (fm_eth default)
        y_t = t[:, None, None, None] * y1 + (1 - t[:, None, None, None]) * noise
        data, cls = self.net(y_t, t, past)                           # 'direct' wrapper
        err = (self.unnormalize(data) - self.unnormalize(y1)).norm(dim=-1).sum(dim=-1)   # [B, K, A]
        sel = err.argmin(dim=1)                                      # [B, A]
        loss_reg = err.gather(1, sel[:, None]).squeeze(1).mean(-1)
        loss_cls = F.cross_entropy(cls.permute(0, 2, 1).reshape(-1, self.K), sel.reshape(-1))
        return loss_reg.mean() + loss_cls

    @torch.no_grad()
    def sample(self, hist):
        """10-step Euler sampler; returns (trajs [B, K, H, 2] metric, softmax(cls) [B, K])."""
        B = hist.shape[0]
        past = hist[:, None]
        y = torch.randn(B, self.K, 1, 2 * self.H, device=hist.device)
        dt = 1.0 / self.sampling_steps
        for i in range(self.sampling_steps):
            t = torch.full((B,), i * dt, device=hist.device)
            data, cls = self.net(y, t, past)
            y = y + (data - y) / (1 - i * dt) * dt
        return self.unnormalize(y[:, :, 0]), cls[:, :, 0].softmax(dim=-1)


class MoFlowIMLENet(nn.Module):
    """ETHIMLETransformer (backbone_eth_ucy.py), objective 'set', single agent."""

    def __init__(self, horizon, num_query=20, d=32, past_frames=5, in_feat=6, social_hidden=64,
                 enc_layers=2, dec_blocks=2, heads=4, dropout=0.1):
        super().__init__()
        self.K, self.d = num_query, d
        self.context_encoder = ETHEncoder(past_frames, in_feat, d, social_hidden, enc_layers, heads, dropout)
        self.motion_query_embedding = nn.Embedding(num_query, d)
        self.agent_order_embedding = nn.Embedding(1, d)
        self.noisy_vec_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.pe_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.init_emb_fusion_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, d))
        self.motion_decoder = MTRDecoder(d, heads, dec_blocks, dropout, use_adaln=False)
        self.reg_head = build_mlps(d, [d, 2 * d, 2 * horizon])

    def forward(self, past, M):
        """past [B, A, P, F] -> normalized trajectories [B, M, K, A, H*2]."""
        B, A = past.shape[:2]
        K, D = self.K, self.d
        enc = self.context_encoder(past)[:, None, None].expand(-1, M, K, -1, -1)
        noise_emb = self.noisy_vec_mlp(torch.randn(B, M, D, device=past.device))
        noise_emb = noise_emb[:, :, None, None].expand(-1, -1, K, A, -1)
        k_pe = self.motion_query_embedding(torch.arange(K, device=past.device))[None, None, :, None]
        a_pe = self.agent_order_embedding(torch.arange(A, device=past.device))[None, None, None]
        fusion = self.init_emb_fusion_mlp(torch.cat([enc, noise_emb], dim=-1))
        token = self.pe_mlp(fusion + k_pe + a_pe).reshape(B * M, K, A, D)
        return self.reg_head(self.motion_decoder(token)).view(B, M, K, A, -1)


class MoFlowIMLE(nn.Module):
    """IMLE (models/imle.py): chamfer weight 1, GT weight 0, reduction 'sum'."""

    def __init__(self, net, q_norm, horizon, num_to_gen=30):
        super().__init__()
        self.net, self.H, self.num_to_gen = net, horizon, num_to_gen
        self.register_buffer("q_lo", torch.as_tensor(q_norm.lo))
        self.register_buffer("q_hi", torch.as_tensor(q_norm.hi))

    def unnormalize(self, y):
        y = y.unflatten(-1, (self.H, 2))
        return (y + 1) / 2 * (self.q_hi - self.q_lo) + self.q_lo

    def loss(self, hist, teacher):
        """teacher [B, K, H, 2] metric teacher samples (y_t at latent_tau = 0)."""
        gen = self.unnormalize(self.net(hist[:, None], self.num_to_gen))[:, :, :, 0]   # [B, M, K, H, 2]
        pair = (gen[:, :, :, None] - teacher[:, None, None]).norm(dim=-1).sum(dim=-1)  # [B, M, K1, K2]
        chamfer = pair.min(dim=-1).values.mean(-1) + pair.min(dim=-2).values.mean(-1)  # [B, M]
        return chamfer.min(dim=-1).values.mean()

    @torch.no_grad()
    def sample(self, hist):
        """One noise vector -> K trajectories; the set carries no scores, so each is weighted 1/K."""
        trajs = self.unnormalize(self.net(hist[:, None], 1))[:, 0, :, 0]                # [B, K, H, 2]
        return trajs, torch.full(trajs.shape[:2], 1.0 / trajs.shape[1], device=hist.device)
