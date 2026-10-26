# DAFM / Ensemble Flow Filter → `mml_baselines.enff`

**Paper.** *Flow Matching for Efficient and Scalable Data Assimilation*, [arXiv:2508.13313](https://arxiv.org/abs/2508.13313)
(code authors per the upstream LICENSE: T. Transue, B. Chen, S. Takao, B. Wang).
**Code.** `github.com/Utah-Math-Data-Science/Data-Assimilation-Flow-Matching`, commit `ad51a8b` (2026-06-17).

| File | Contents |
|---|---|
| `tracker.py` | `EnFFTracker`: CV forecast + EnFF analysis; benchmark interface (`reset`, `step`, `cloud`, `get_state`, `set_state`) |

Hyperparameters are chosen on validation runs by `python -m mml_research.motion.sweep_enff`.

## Upstream files consulted

| Upstream file | What we took |
|---|---|
| `src/dafm/filters.py` | `EnsembleFlowFilter.velocity` (softmax-weighted conditional velocities over ensemble members), `assimilate` (Euler over `prob_path.linspace_time`, guidance added to the velocity), `log_prob_to_weight_safe` |
| `src/dafm/prob_path.py` | `FilteringToPredictive` (F2P): `mean = t·x_pred + (1−t)·x_prev`, `dt_mean = x_pred − x_prev`, `std = σ_min`, `dt_std = 0`, `sample_noise = x_prev + σ_min·ε`; `linspace_time = linspace(0, 1, n)` |
| `src/dafm/flow_matching_guidance.py` | `Local` guidance: `x̂1 = x_t + (1 − t)·v`, `−λ ∇E(x̂1)` with constant schedule |
| `src/conf/filter.py`, `src/conf/prob_path.py` | Euler sampler default, inflation 1 |
| `sweeps/topk_hyperparameters.csv`, `notebooks/AblationSamplingTimeStepCount.ipynb` | EnFF-F2P is tuned over `σ_min` and `λ`; `sampling_time_step_count = 5` is among its best settings |
| `src/dafm/main.py` | the filtering loop: forecast with the dynamics, assimilate only when an observation exists |

## Ported faithfully

1. **Training-free analysis**: the velocity field is the closed-form marginal of the F2P conditional paths
   from the previous analysis ensemble to the forecast ensemble.
2. **Local energy guidance** with `E = −log N(y; x̂1, σ_obs²)` (unnormalized), Euler integration with
   `sampling_time_step_count = 5` (4 Euler steps).
3. **Equal-weight ensemble**, no resampling, no inflation.

## Adapted (disclose in the paper)

| Upstream | Port | Why |
|---|---|---|
| Forecast with the known dynamical system (Lorenz, KS, Navier–Stokes) | Constant-velocity forecast from each member's last two analysis states, speed clipped at 1.5 m/s, plus Gaussian process noise `q` | The target's dynamics are unknown; this is PF+CV's proposal, so EnFF vs PF+CV isolates the analysis step |
| `q = model_noise_std` 0 | `q` chosen on validation runs | Same selection rule as every other tracker |
| `σ_min`, `λ` tuned per dataset | chosen on validation runs 100–119 from `σ_min ∈ {1e-3, 1e-2}`, `λ ∈ {6.25e-5, …, 2e-3}` (factors of 2), jointly with `q` | Observation noise and state scale differ from the DAFM benchmarks |
| Observation every few model steps, always present | No observation while the target is occluded: forecast only | Occlusion |
| Observation noise = data noise | Likelihood σ_obs = 0.05 m | The value every other tracker's likelihood uses |
| Torch + `vmap` | NumPy, `‖x‖² + ‖m‖² − 2x·m` for the pairwise distances | Same arithmetic, faster on CPU |
