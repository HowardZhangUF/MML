# TCFM → `mml_baselines.tcfm`

**Paper.** S. Ye, M. C. Gombolay, *Efficient Trajectory Forecasting and Generation with Conditional Flow
Matching*, [arXiv:2403.10809](https://arxiv.org/abs/2403.10809).
**Code.** `github.com/CORE-Robotics-Lab/TCFM`, commit `b14f04d` (2024-11-05).

| File | Contents |
|---|---|
| `model.py` | `ConditionalUnet1D` (+ blocks, `EncoderRNN`), `TCFM` flow matcher / sampler |
| `motion_model.py` | `TCFMMotionModel`: bootstrap-PF proposal |
| `train.py` | `python -m mml_baselines.tcfm.train --out <models dir>` → `tcfm_motion_model.pth` |

## Upstream files consulted

| Upstream file | What we took |
|---|---|
| `diffuser/models/cfm.py` | `CFM.loss`: `torchcfm.ConditionalFlowMatcher(sigma=0.0)`, `x0 ~ N(0, I)`, MSE between `v_θ(t, x_t)` and `x1 − x0` over the whole trajectory; `p_sample_loop_cfm`: Euler `odeint` over `linspace(0, 1, n_timesteps + 1)` |
| `diffuser/models/temporal_film.py` | `ConditionalUnet1D` and `ConditionalResidualBlock1D` (FiLM bias, `cond_predict_scale=False`), condition = `[LSTM(detections) ‖ time embedding]` |
| `diffuser/models/helpers.py` | `SinusoidalPosEmb`, `Conv1dBlock` (Conv1d → GroupNorm(8) → Mish), stride-2 down / transposed-conv up-sampling |
| `diffuser/models/encoder.py` | `EncoderRNN`: one-layer LSTM, last hidden state |
| `config/cfm_aircraft.py`, `config/IROS24/*` | `ConditionalUnet1D` + `CFM`, `dim_mults (1, 4, 8)`, `n_diffusion_steps 100`, `history_length 10`, `LimitsNormalizer` |
| `diffuser/utils/training.py` | Adam lr 2e-4, batch 32 × 2 gradient accumulation. Its EMA is a no-op (`self.ema_model = self.model`), so the raw weights are what upstream evaluates; we do the same |

## Ported faithfully

1. **Conditional flow matching with σ = 0** on the whole future trajectory (straight paths from Gaussian
   noise), MSE on the conditional velocity.
2. **Temporal 1-D U-Net** of conditional residual blocks with FiLM conditioning on the LSTM history code and the
   flow time, skip connections, two down/up-sampling stages.
3. **Euler ODE sampling**, one independent sample per noise draw.

## Adapted (disclose in the paper)

| Upstream | Port | Why |
|---|---|---|
| History: 10 normalized absolute positions (lon/lat/alt) | 5 min-max-normalized `[absolute, relative, velocity]` features | Benchmark history `N_th = 5`; shared preprocessing across the forecaster ports |
| Future: 60 normalized absolute positions | 12 normalized relative positions `Q` | Shared horizon; relative futures are translation invariant |
| `dim 32`, `dim_mults (1, 4, 8)` (~millions of params) | `dim 8`, `dim_mults (1, 2, 4)`: **53,810 params** | Capacity parity with CFM (57,490 params); still three resolutions |
| `n_diffusion_steps 100` Euler steps | **10** Euler steps | Same function-evaluation budget as CFM (N_FE = 10) and DiffPF (T = 10); 100 steps × 500 particles per filter step is not affordable |
| `nn.Mish` | `x · tanh(softplus(x))` | Identical function, ~1.7× faster on CPU |
| 500k steps × batch 64 (effective) | 60k steps × batch 64 (≈330 epochs) | Small dataset |
| Constraint / RePaint / guided samplers in `cfm.py` | omitted | Not on the CFM forecasting path (`p_sample_loop` returns `p_sample_loop_cfm`) |

## How the particle filter uses it

`TCFMMotionModel.propose` draws one TCFM trajectory per particle (fresh noise, 10 Euler steps) and returns its
first-step velocity `Q_1 / dt`.
