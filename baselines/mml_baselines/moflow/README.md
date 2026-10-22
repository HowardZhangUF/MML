# MoFlow → `mml_baselines.moflow`

**Paper.** Y. Fu, Q. Yan, L. Wang, K. Li, R. Liao, *MoFlow: One-Step Flow Matching for Human Trajectory
Forecasting via Implicit Maximum Likelihood Estimation based Distillation*, CVPR 2025,
[arXiv:2503.09950](https://arxiv.org/abs/2503.09950).
**Code.** `github.com/DSL-Lab/MoFlow`, commit `f1b89b0` (2026-05-14). ETH-UCY configuration.

| File | Contents |
|---|---|
| `model.py` | teacher (`MoFlowTeacherNet` + `MoFlowTeacher` flow matcher), student (`MoFlowIMLENet` + `MoFlowIMLE`) |
| `motion_model.py` | `MoFlowMotionModel`: bootstrap-PF proposal from the **IMLE student** |
| `train.py` | `python -m mml_baselines.moflow.train --out <models dir>` → `moflow_teacher.pth`, `moflow_motion_model.pth` |

## Upstream files consulted

| Upstream file | What we took |
|---|---|
| `models/flow_matching.py` | `FlowMatcher`: `pred_data` objective, `direct` wrapper, logit-normal `t`, Euler sampler, `p_losses` with agent-level winner-take-all regression (L2 per frame, summed over time) and cross-entropy on the winning component |
| `fm_eth.py` (argparse defaults), `cfg/eth_ucy/cor_fm.yml` | `sampling_steps 10`, `t_schedule logit_normal` (mean −0.5, std 1.5), `fm_wrapper direct`, `drop_method emb` (logistic k=20, m=0.5), `tied_noise False`, `loss_nn_mode agent`, `loss_reg_reduction sum`, K=20, AdamW lr 1e-4 wd 0.01, cosine with warmup, grad clip 1.0 |
| `models/backbone_eth_ucy.py` | `ETHMotionTransformer` (teacher) and `ETHIMLETransformer` (student, `objective set`) |
| `models/context_encoder/eth_encoder.py` | `ETHEncoder` / `SocialTransformer`; past features `[absolute, relative, velocity]` |
| `models/motion_decoder/mtr_decoder.py` | `MTRDecoder`: adaLN time modulation (zero-init), K-to-K then A-to-A self-attention per block |
| `models/imle.py`, `imle_eth.py`, `cfg/eth_ucy/imle.yml` | IMLE set objective: Chamfer distance between each generated K-set and the teacher's K samples, min over `num_to_gen=30` noise vectors; `loss_reg_chamfer_weight 1`, `loss_reg_gt_weight 0`, `latent_tau 0` |
| `trainer/denoising_model_trainers.py`, `trainer/imle_trainers.py` | EMA 0.995; 5 % linear warmup then cosine |

## Ported faithfully

1. **Teacher**: correlated flow matching over a set of K=20 future trajectories, with the network seeing all K
   noisy trajectories at once (K-to-K attention), pred-data objective, logit-normal time, timestep-dependent
   embedding dropout, winner-take-all regression plus a classifier over the K components, 10 Euler steps.
2. **Teacher samples on the training set** as the distillation targets (`latent_tau = 0`: final samples).
3. **Student (the MoFlow model)**: one-step generator from a single noise vector to a set of K trajectories
   (no time input, no adaLN), trained by IMLE with the set-to-set Chamfer loss, best of M=30 noise draws.

## Adapted (disclose in the paper)

| Upstream | Port | Why |
|---|---|---|
| A = 2..N agents with social attention | A = 1 (the target) | Single target; the agent-axis layers run on a length-1 sequence and are kept for structure |
| 8 past / 12 future frames at 2.5 Hz | 5 past / 12 future frames at 30 Hz (`common/data.py`) | Benchmark history length `N_th = 5`; the same 12-frame horizon |
| Encoder input in original scale | min-max normalized `[absolute, relative, velocity]` features | Shared preprocessing across the three forecaster ports |
| Scalar min-max normalization of the future | per-axis min-max | Shared preprocessing |
| D 128, 4 encoder layers, 4 decoder blocks, social hidden 256 | D 32, 2 encoder layers, 2 decoder blocks, social hidden 64: **teacher 184,537 / student 148,760 params** | Capacity parity with the benchmark's learned models at affordable filter cost |
| `MTRDecoder.forward` re-reads its input `query_token` inside the block loop | blocks are chained (`cur_query`) | Upstream discards the adaLN modulation and the output of every block but the last; we use the evident intent (as we did for DiffPF's unused fusion MLP) |
| 150 epochs per stage | 30k steps × batch 128 per stage (≈330 epochs) | Small dataset |

## How the particle filter uses it

`MoFlowMotionModel.propose` runs the **one-step student** once per particle (one noise vector → 20
trajectories), picks one of the 20 uniformly (the IMLE set carries no scores), and returns its first-step
velocity `Q_1 / dt`. The teacher (`moflow_teacher.pth`) is only used for distillation.
