# TrajFlow → `mml_baselines.trajflow`

**Paper.** Q. Yan, B. Zhang, Y. Zhang, D. Yang, J. White, D. Chen, J. Liu, L. Liu, B. Zhuang, S. Shi, et al.,
*TrajFlow: Multi-modal Motion Prediction via Flow Matching*, IROS 2025, [arXiv:2506.08541](https://arxiv.org/abs/2506.08541)
(citation as given in the upstream README).
**Code.** `github.com/DSL-Lab/TrajFlow`, commit `7c63645` (2025-12-24).

| File | Contents |
|---|---|
| `model.py` | network (`TrajFlowNet`), flow matcher and losses (`TrajFlow`), upstream helpers |
| `motion_model.py` | `TrajFlowMotionModel`: bootstrap-PF proposal (`propose`, `block_size`, `dt`) |
| `train.py` | `python -m mml_baselines.trajflow.train --out <models dir>` → `trajflow_motion_model.pth` |

## Upstream files consulted

| Upstream file | What we took |
|---|---|
| `trajflow/denoising/flow_matching.py` | `FlowMatcher`: linear path, `pred_data` objective, tied noise, uniform `t`, Euler sampler, `_forward_ctc_loss` (round 1 at `t=0`; round 2 with probability 0.5 re-noises the detached round-1 prediction with the same noise) |
| `runner/cfgs/waymo/trajflow+100_percent_data.yaml` | `SAMPLING_STEPS: 1`, `OBJECTIVE: pred_data`, `T_SCHEDULE: uniform`, `TIED_NOISE: True`, `CTC_LOSS: True`, `NUM_MOTION_MODES: 6`, loss weights cls 1 / reg 1 / vel 0.5 / pl 0.1, AdamW lr 1e-4 wd 0.01, EMA 0.999 |
| `trajflow/models/denoising_decoder/denoising_decoder.py` | denoiser: `x_embedder` (MLP on the flattened noisy trajectory), `TimestepEmbedder`, `feat_fusion_mlp`, 2-layer `feat_tf_encoder` across queries; decoder loop with intention queries, dynamic query centres, per-layer cls / reg (7 channels: GMM 5 + velocity 2) / PL heads; `get_decoder_loss` |
| `trajflow/models/denoising_decoder/decoder_utils.py` | `get_motion_query`, `get_center_gt_idx` (EDA anchors: intention points for the first `num_inter_layers=2` layers, then the predictions of layer `(l//2)*2-1`), `generate_final_prediction` (sigmoid scores + NMS) |
| `trajflow/utils/motion_utils.py` | `batch_nms` (verbatim), `select_distinct_anchors`, `get_evolving_anchors` |
| `trajflow/models/denoising_decoder/compute_loss.py`, `trajflow/utils/mtr_loss_utils.py` | `plackett_luce_loss`, `nll_loss_gmm_direct` (MTR GMM NLL) |
| `trajflow/models/layers/common_layers.py` | `build_mlps`, `gen_sineembed_for_position` (verbatim), `TimestepEmbedder` |
| `trajflow/models/context_encoder/mtr_encoder.py`, `layers/polyline_encoder.py` | PointNet polyline encoder for agent history |

## Ported faithfully

1. One-step flow matching with the **pred-data objective and tied noise** across all K queries; sampling is a
   single Euler step from `t=0`, so the network output is the forecast set.
2. The **cross-time-consistency (CTC) loss**.
3. The **query-based denoising decoder**: K learnable motion queries seeded by k-means intention points,
   refined layer by layer with dynamic query centres; heads per layer; all layer losses averaged.
4. The **MTR/EDA training targets**: positive mode by intention-point goal distance for the first two layers,
   then by distance to the evolving anchors, distinct-anchor NMS mask on the BCE classification, GMM NLL on the
   positive mode, L1 velocity loss, Plackett-Luce ranking loss.
5. **Inference output**: sigmoid scores and goal-distance NMS down to 6 modes.

## Adapted (disclose in the paper)

| Upstream | Port | Why |
|---|---|---|
| Waymo scene: MTR encoder over agents + map polylines, query-to-agent and query-to-map cross-attention with dynamic map collection | PointNet polyline encoder over the target's own 5-step history; decoder layers cross-attend to the history point tokens | Single target, no other agents, no map. The absolute position in the history features is the only map information, as for every other learned model in the benchmark |
| 80 future frames, 8 s | 12 future frames (`Q`, see `common/data.py`) | The occlusion box spans ~30 steps; 12 frames match MoFlow's ETH-UCY horizon and divide by 4 for TCFM's U-Net |
| 64 intention points from the MTR cluster file | 16 k-means centres of the training goals | Our goal distribution has a handful of modes (two lanes, straight or turn) |
| `D_QUERY` 512, `DEPTH` 6, 8 heads, ~tens of millions of params | `d_model` 32, depth 4, 4 heads: **160,600 params** | Capacity parity with the benchmark's learned models (57–68k) at affordable filter cost; depth 4 keeps both EDA stages (anchors from intention points, then from layer-2 predictions) |
| EDA distinct-anchor distance 2.5–3.5 m (scaled with trajectory length 10–50 m); GMM std clipped to [0.2 m, 150 m] | NMS distance 0.05 m; std clipped to [5 mm, 5 m] | Same ratios at our scale: a 12-step future spans ~0.36 m |
| `build_mlps` BatchNorm1d | LayerNorm | Batch statistics are ill-defined for the filter's per-step batches |
| Dense future prediction of other agents (+ its loss) | omitted | No other agents |
| Training: 30 epochs of Waymo | 30k steps × batch 128 (≈330 epochs of 11.6k samples) | Small dataset |

## How the particle filter uses it

`TrajFlowMotionModel.propose` turns each particle's 5-step history into TrajFlow's 6 NMS-selected trajectories
with sigmoid scores, draws one trajectory per particle with probability proportional to its score, and
returns that trajectory's first-step velocity `Q_1 / dt`. Process noise is added by the PF and chosen on
validation runs like every other tracker.
