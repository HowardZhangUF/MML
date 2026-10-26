"""
TrajFlow (Yan et al., IROS 2025) ported to the single-target road benchmark.

Upstream: github.com/DSL-Lab/TrajFlow @ 7c63645. See README.md in this folder
for the file-by-file provenance and every deviation.

Mechanism kept from upstream
    * FlowMatcher: linear path x_t = t*x1 + (1-t)*x0, 'pred_data' objective,
      tied noise across the K queries, uniform t, 1 Euler sampling step
      (trajflow/denoising/flow_matching.py, cfgs SAMPLING_STEPS: 1).
    * Cross-time consistency (CTC) loss: round 1 at t=0, round 2 (prob 0.5)
      re-noises the detached round-1 prediction at a random t.
    * Denoising decoder: noisy-trajectory embedding + timestep embedding +
      center-agent feature -> fusion MLP -> 2-layer transformer over the K
      queries -> intention-point motion queries refined by stacked decoder
      layers with dynamic query centres; per-layer cls / reg / PL heads
      (trajflow/models/denoising_decoder/denoising_decoder.py).
    * Loss per layer: MTR GMM NLL on the positive mode, L1 velocity, BCE
      classification over EDA-distinct anchors, Plackett-Luce ranking
      (compute_loss.py, decoder_utils.get_center_gt_idx, motion_utils.*).
    * Inference: sigmoid scores, distance NMS down to NUM_MOTION_MODES.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


# ── upstream layer helpers (trajflow/models/layers/common_layers.py) ─────────

def build_mlps(c_in, mlp_channels, ret_before_act=False, without_norm=False, layer_norm=False):
    """build_mlps; BatchNorm1d -> LayerNorm when a norm is requested (see README)."""
    layers = []
    for k, c_out in enumerate(mlp_channels):
        if k + 1 == len(mlp_channels) and ret_before_act:
            layers.append(nn.Linear(c_in, c_out, bias=True))
        elif without_norm:
            layers.extend([nn.Linear(c_in, c_out, bias=True), nn.ReLU()])
        else:
            layers.extend([nn.Linear(c_in, c_out, bias=False), nn.LayerNorm(c_out), nn.ReLU()])
        c_in = c_out
    return nn.Sequential(*layers)


def gen_sineembed_for_position(pos_tensor, hidden_dim):
    """DAB-DETR sine embedding of 2-D points, verbatim. pos_tensor [B, N, 2]."""
    half = hidden_dim // 2
    dim_t = torch.arange(half, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half)
    x_embed = pos_tensor[:, :, 0] * 2 * math.pi
    y_embed = pos_tensor[:, :, 1] * 2 * math.pi
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    return torch.cat((pos_y, pos_x), dim=2)


class TimestepEmbedder(nn.Module):
    """DiT timestep embedder, as in TrajFlow (t in [0, 1], not rescaled)."""

    def __init__(self, hidden_size, frequency_embedding_size=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))
        self.freq = frequency_embedding_size

    def forward(self, t):
        half = self.freq // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


# ── upstream NMS / anchor helpers (trajflow/utils/motion_utils.py) ──────────

def batch_nms(pred_trajs, pred_scores, dist_thresh, num_ret_modes=6, return_mask=False):
    """motion_utils.batch_nms, verbatim (goal-distance NMS)."""
    batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape
    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_full = torch.arange(batch_size, device=pred_scores.device)[:, None].repeat(1, num_modes)
    sorted_pred_scores = pred_scores[bs_full, sorted_idxs]
    sorted_pred_trajs = pred_trajs[bs_full, sorted_idxs]
    sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]
    dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
    point_cover_mask = dist < dist_thresh

    point_val = sorted_pred_scores.clone()
    point_val_selected = torch.zeros_like(point_val)
    ret_mask_sorted = torch.ones_like(point_val).bool()
    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    bs_idxs = torch.arange(batch_size, device=pred_scores.device)
    if not return_mask:
        ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
        ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1)
        ret_idxs[:, k] = cur_idx
        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]
        filter_mask = new_cover_mask.clone()
        filter_mask[bs_idxs, cur_idx] = False
        filter_mask *= (point_val.max(dim=-1, keepdim=True).values > 0)
        ret_mask_sorted[filter_mask] = False
        point_val = point_val * (~new_cover_mask).float()
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected
        if return_mask and (point_val.max(dim=-1, keepdim=True).values <= 0).all():
            break
        if not return_mask:
            ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
            ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    ret_idxs = sorted_idxs[bs_idxs[:, None], ret_idxs]
    if return_mask:
        ret_mask = torch.zeros_like(ret_mask_sorted)
        ret_mask_sorted[torch.cumsum(ret_mask_sorted, dim=-1) > num_ret_modes] = False
        ret_mask[bs_idxs[:, None], sorted_idxs] = ret_mask_sorted
        return ret_mask, ret_idxs
    return ret_trajs, ret_scores, ret_idxs


def nll_loss_gmm_direct(pred_trajs, gt_trajs, nearest_mode_idxs,
                        log_std_range=(-5.3, 1.6), rho_limit=0.5):
    """
    MTR GMM loss (mtr_loss_utils.nll_loss_gmm_direct), positive mode given; no
    masking needed. Upstream clips std to [0.2 m, 150 m] for Waymo; our whole
    12-step future spans ~0.4 m, so the range is rescaled to [5 mm, 5 m].
    """
    nearest = pred_trajs[torch.arange(len(gt_trajs), device=gt_trajs.device), nearest_mode_idxs]
    dx, dy = (gt_trajs - nearest[..., 0:2]).unbind(-1)
    log_std1 = nearest[..., 2].clip(*log_std_range)
    log_std2 = nearest[..., 3].clip(*log_std_range)
    std1, std2 = log_std1.exp(), log_std2.exp()
    rho = nearest[..., 4].clip(-rho_limit, rho_limit)
    coef = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)
    expo = (0.5 / (1 - rho ** 2)) * (dx ** 2 / std1 ** 2 + dy ** 2 / std2 ** 2 - 2 * rho * dx * dy / (std1 * std2))
    return (coef + expo).sum(dim=-1)


def plackett_luce_loss(logits, preference_argsort):
    """
    compute_loss.plackett_luce_loss. Upstream's first_occurrence_mask_fast
    marks positions 0..u-1 (u = number of distinct entries in the row: it
    indexes with torch.unique's inverse indices); reproduced vectorized.
    """
    ordered = torch.gather(logits, 1, preference_argsort)
    log_probs = ordered - torch.logcumsumexp(ordered, dim=-1)
    srt = preference_argsort.sort(dim=-1).values
    n_unique = 1 + (srt[:, 1:] != srt[:, :-1]).sum(dim=-1, keepdim=True)
    mask = torch.arange(preference_argsort.shape[1], device=logits.device)[None] < n_unique
    return (-log_probs * mask.float()).mean(dim=-1)


# ── network ──────────────────────────────────────────────────────────────────

class PolylineEncoder(nn.Module):
    """
    MTR PointNetPolylineEncoder applied to the target's own history (the
    single agent polyline): pre-MLP per point, concat with the max-pooled
    feature, MLP, max-pool, out-MLP. Returns (pooled feature, point tokens).
    """

    def __init__(self, in_dim, hidden, n_layers=3, n_pre_layers=1):
        super().__init__()
        self.pre = build_mlps(in_dim, [hidden] * n_pre_layers)
        self.mlps = build_mlps(hidden * 2, [hidden] * (n_layers - n_pre_layers))
        self.out = build_mlps(hidden, [hidden, hidden], ret_before_act=True, without_norm=True)

    def forward(self, pts):                                   # [B, P, C]
        f = self.pre(pts)
        pooled = f.max(dim=1, keepdim=True).values.expand_as(f)
        f = self.mlps(torch.cat([f, pooled], dim=-1))
        return self.out(f.max(dim=1).values), f


class DMTDecoderLayer(nn.Module):
    """
    One query-refinement layer: self-attention across the K motion queries
    (position = intention query), then cross-attention from the queries
    (position = sine embedding of the dynamic query centre) to the history
    point tokens. Stands in for TrajFlow's dmt_decoder_layer obj/map
    attention; see README (no other agents, no map).
    """

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ca = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, q_pos, q_center_pos, kv, kv_pos):
        h = q + q_pos
        q = self.n1(q + self.sa(h, h, q)[0])
        q = self.n2(q + self.ca(q + q_center_pos, kv + kv_pos, kv)[0])
        return self.n3(q + self.ffn(q))


class TrajFlowNet(nn.Module):
    def __init__(self, horizon, intention_points, d_model=64, heads=4, depth=4, dropout=0.1,
                 hist_in=6, n_th=5):
        super().__init__()
        self.horizon, self.d = horizon, d_model
        self.num_query = len(intention_points)
        self.register_buffer("intention_points", torch.as_tensor(intention_points, dtype=torch.float32))

        # context encoder (MTR encoder reduced to the target's history polyline)
        self.encoder = PolylineEncoder(hist_in + 1, d_model)
        self.point_pos = nn.Parameter(torch.zeros(1, n_th, d_model))

        # denoising token / time embeddings and fusion (build_denoising_layers)
        self.x_embedder = build_mlps(2 * horizon, [d_model] * 2, ret_before_act=True, without_norm=True)
        self.t_embedder = TimestepEmbedder(d_model)
        self.feat_fusion_mlp = build_mlps(3 * d_model, [d_model] * 2, ret_before_act=True, layer_norm=True)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True)
        self.feat_tf_encoder = nn.TransformerEncoder(layer, num_layers=2)

        # motion queries from intention points (build_motion_query)
        self.intention_query_mlps = build_mlps(d_model, [d_model, d_model], ret_before_act=True)

        # decoder layers + per-layer heads (build_motion_head)
        self.layers = nn.ModuleList([DMTDecoderLayer(d_model, heads, dropout) for _ in range(depth)])
        fusion = build_mlps(2 * d_model, [d_model, d_model], ret_before_act=True)
        self.query_feature_fusion_layers = nn.ModuleList([copy.deepcopy(fusion) for _ in range(depth)])
        reg = build_mlps(d_model, [d_model, d_model, horizon * 7], ret_before_act=True)
        cls = build_mlps(d_model, [d_model, d_model, 1], ret_before_act=True)
        pl = build_mlps(d_model, [d_model, d_model, 1], ret_before_act=True, layer_norm=True)
        self.motion_reg_heads = nn.ModuleList([copy.deepcopy(reg) for _ in range(depth)])
        self.motion_cls_heads = nn.ModuleList([copy.deepcopy(cls) for _ in range(depth)])
        self.pl_readout_mlp = nn.ModuleList([copy.deepcopy(pl) for _ in range(depth)])

    def encode(self, hist):
        """hist [B, P, 6] normalized history features -> (center feature, point tokens)."""
        valid = torch.ones_like(hist[..., :1])
        center, tokens = self.encoder(torch.cat([hist, valid], dim=-1))
        return center, tokens

    def forward(self, x_t, t, center, tokens):
        """
        x_t [B, K, H, 2] noisy normalized trajectories, t [B].
        Returns pred_list: per layer (scores [B, K], trajs [B, K, H, 7] metric, pl_logits [B, K]).
        """
        B, K = x_t.shape[:2]
        t_emb = self.t_embedder(t)[:, None].expand(-1, K, -1)
        x_emb = self.x_embedder(x_t.flatten(2))
        c = center[:, None].expand(-1, K, -1)
        c = self.feat_tf_encoder(self.feat_fusion_mlp(torch.cat([x_emb, t_emb, c], dim=-1)))

        ip = self.intention_points[None].expand(B, -1, -1)
        intention_query = self.intention_query_mlps(gen_sineembed_for_position(ip, self.d))
        kv_pos = self.point_pos.expand(B, -1, -1)
        query_content = torch.zeros_like(intention_query)
        dynamic_center = ip
        pred_list = []
        for i, layer in enumerate(self.layers):
            center_pos = gen_sineembed_for_position(dynamic_center, self.d)
            agent_feat = layer(query_content, intention_query, center_pos, tokens, kv_pos)
            query_content = self.query_feature_fusion_layers[i](torch.cat([c, agent_feat], dim=-1))
            scores = self.motion_cls_heads[i](query_content).squeeze(-1)
            trajs = self.motion_reg_heads[i](query_content).view(B, K, self.horizon, 7)
            pl_logits = self.pl_readout_mlp[i](query_content).squeeze(-1)
            pred_list.append((scores, trajs, pl_logits))
            dynamic_center = trajs[:, :, -1, 0:2].detach()
        return pred_list


# ── flow matcher + losses ────────────────────────────────────────────────────

class TrajFlow(nn.Module):
    """FlowMatcher(DenoisingDecoder) with TrajFlow's training and sampling rules."""

    def __init__(self, net: TrajFlowNet, q_norm, num_motion_modes=6, nms_dist=0.05,
                 num_inter_layers=2, loss_weights=None, sampling_steps=1):
        super().__init__()
        self.net = net
        self.register_buffer("q_lo", torch.as_tensor(q_norm.lo))
        self.register_buffer("q_hi", torch.as_tensor(q_norm.hi))
        self.num_motion_modes = num_motion_modes
        self.nms_dist = nms_dist
        self.num_inter_layers = num_inter_layers
        self.w = {"cls": 1.0, "reg": 1.0, "vel": 0.5, "pl": 0.1, **(loss_weights or {})}
        self.sampling_steps = sampling_steps

    def normalize(self, q):
        return (q - self.q_lo) / (self.q_hi - self.q_lo) * 2 - 1

    # -- training ------------------------------------------------------------

    def _positive_idx(self, layer_idx, pred_list, scores, trajs, gt_q):
        """get_center_gt_idx (training branch) with EDA evolving + distinct anchors."""
        B = gt_q.shape[0]
        pos_layer = (layer_idx // self.num_inter_layers) * self.num_inter_layers - 1
        if pos_layer < 0:
            ip = self.net.intention_points[None].expand(B, -1, -1)
            anchor_trajs = ip.unsqueeze(-2)
            dist = (gt_q[:, None, -1] - ip).norm(dim=-1)
        else:
            anchor_trajs = pred_list[pos_layer][1].detach()
            dist = (gt_q[:, None] - anchor_trajs[..., 0:2]).norm(dim=-1).sum(dim=-1)
        select_mask, select_idx = batch_nms(anchor_trajs, scores.detach().sigmoid(), self.nms_dist,
                                            num_ret_modes=anchor_trajs.shape[1], return_mask=True)
        positive = dist.masked_fill(~select_mask, 1e10).argmin(dim=-1)
        return positive, select_mask, select_idx

    def decoder_loss(self, pred_list, gt_q, gt_v):
        total = 0.0
        B = gt_q.shape[0]
        for i, (scores, trajs, pl_logits) in enumerate(pred_list):
            positive, select_mask, select_idx = self._positive_idx(i, pred_list, scores, trajs, gt_q)
            loss_reg = nll_loss_gmm_direct(trajs[..., 0:5], gt_q, positive)
            pred_vel = trajs[torch.arange(B, device=gt_q.device), positive][..., 5:7]
            loss_vel = F.l1_loss(pred_vel, gt_v, reduction="none").sum(dim=(-1, -2))
            target = torch.zeros_like(scores)
            target[torch.arange(B, device=gt_q.device), positive] = 1.0
            loss_cls = (F.binary_cross_entropy_with_logits(scores, target, reduction="none")
                        * select_mask).sum(dim=-1)
            loss_pl = plackett_luce_loss(pl_logits, select_idx[:, :self.num_motion_modes])
            total = total + (self.w["reg"] * loss_reg + self.w["vel"] * loss_vel
                             + self.w["cls"] * loss_cls + self.w["pl"] * loss_pl)
        return (total / len(pred_list)).mean()

    def loss(self, hist, gt_q, gt_v):
        """CTC loss (_forward_ctc_loss): round 1 at t=0, round 2 with probability 0.5."""
        B, K = gt_q.shape[0], self.net.num_query
        center, tokens = self.net.encode(hist)
        noise = torch.randn(B, 1, *gt_q.shape[1:], device=gt_q.device).expand(-1, K, -1, -1)  # tied noise

        t = torch.zeros(B, device=gt_q.device)
        pred_list = self.net(noise, t, center, tokens)
        loss = self.decoder_loss(pred_list, gt_q, gt_v)
        if torch.rand(()) < 0.5:
            x1 = self.normalize(pred_list[-1][1][..., 0:2]).detach()
            t = torch.rand(B, device=gt_q.device)
            x_t = t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * noise
            loss = loss + self.decoder_loss(self.net(x_t, t, center, tokens), gt_q, gt_v)
        return loss

    # -- sampling ------------------------------------------------------------

    @torch.no_grad()
    def sample(self, hist):
        """
        Euler sampling with tied noise, then generate_final_prediction:
        returns NMS-selected trajectories [B, M, H, 2] (metric) and sigmoid scores [B, M].
        """
        B, K, H = hist.shape[0], self.net.num_query, self.net.horizon
        center, tokens = self.net.encode(hist)
        x = torch.randn(B, 1, H, 2, device=hist.device).expand(-1, K, -1, -1)
        dt = 1.0 / self.sampling_steps
        for i in range(self.sampling_steps):
            t = torch.full((B,), i * dt, device=hist.device)
            pred_list = self.net(x, t, center, tokens)
            data = self.normalize(pred_list[-1][1][..., 0:2])
            x = x + (data - x) / (1 - i * dt) * dt                  # pred_data -> velocity
        scores, trajs = pred_list[-1][0].sigmoid(), pred_list[-1][1]
        trajs, scores, _ = batch_nms(trajs, scores, self.nms_dist, num_ret_modes=self.num_motion_modes)
        return trajs[..., 0:2], scores
