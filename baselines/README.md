# Open-source flow-matching baselines for the exit-prediction benchmark

Ports of four published methods, each in its own folder under `mml_baselines/`, kept separate from
`mml_python_sim` and `mml_core`. They only import the benchmark's data generator
(`mml_research.motion.generate_synthetic_data`). Each folder's `README.md` lists the upstream commit, the files
the port follows, and every deviation.

| Folder | Method | Upstream | Role in the benchmark | Params |
|---|---|---|---|---|
| `trajflow/` | TrajFlow (IROS 2025) | `DSL-Lab/TrajFlow` @ `7c63645` | PF+TrajFlow: motion model in the bootstrap PF | 160,600 |
| `moflow/` | MoFlow (CVPR 2025) | `DSL-Lab/MoFlow` @ `f1b89b0` | PF+MoFlow: one-step IMLE student in the bootstrap PF | 148,760 (teacher 184,537) |
| `tcfm/` | TCFM (arXiv 2403.10809) | `CORE-Robotics-Lab/TCFM` @ `b14f04d` | PF+TCFM: motion model in the bootstrap PF | 53,810 |
| `enff/` | Ensemble Flow Filter, DAFM (arXiv 2508.13313) | `Utah-Math-Data-Science/Data-Assimilation-Flow-Matching` @ `ad51a8b` | EnFF: its own tracker (CV forecast + EnFF analysis) | 0 (training-free) |
| `common/` | shared data and scaffolding | — | — | — |

## Common protocol

* **Training data.** `common/data.py` regenerates the exact measurement stream behind
  `converted_noisy_5_v2_velocities_training_data.csv` (same simulator and seed; inputs and first-step labels
  match the CSV to 0.0) and extends each sample with its next 12 visible measurements. Train/test is the CSV's
  chronological 70/30 split. The 100 samples per stream end without 12 future measurements are dropped
  (16,528 of 16,628).
* **History features.** The 5-step history as `[absolute, relative-to-last, finite-difference velocity]`,
  min-max normalized: MoFlow's ETH-UCY layout, used by all three forecasters.
* **Future.** `Q_k = dt · Σ_{j≤k} v_j`, the future relative to the last measurement (`v_j` per-interval
  velocities, the CSV's label convention). The particle filter uses `Q_1 / dt`, the quantity every other motion
  model proposes.
* **Filter use.** The forecasters plug into the same `BootstrapPFTracker` as PF+CFM/NF/CV/DMMN (500 particles,
  Gaussian likelihood σ = 0.05 m, systematic resampling at ESS < N/2). Each particle draws one forecast by the
  method's own mode probabilities and moves along its first step. Process noise is chosen on validation runs
  100–119 like every other tracker. EnFF's `q`, `λ`, `σ_min` are chosen jointly by the same rule.
* **Capacity.** Every port is shrunk to 54k–185k parameters, close to the benchmark's learned models
  (CFM 57k, NF 68k, DiffPF 37k), keeping each method's mechanism.

## Reproduce

```bash
cd <repo>
export PYTHONPATH=mml_core/src:mml_research/src:baselines   # or: pip install -e baselines
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=artifacts/motion/exit_benchmark

python -m mml_baselines.trajflow.train --out $M/models
python -m mml_baselines.moflow.train   --out $M/models
python -m mml_baselines.tcfm.train     --out $M/models

# validation-run selection (runs 100-119)
python -m mml_research.motion.sweep_process_noise --models-dir $M/models --out-dir $M/noise_sweep_new \
    --trackers PF+TrajFlow PF+MoFlow PF+TCFM --workers 24
python -m mml_research.motion.sweep_enff --models-dir $M/models --out-dir $M/enff_sweep --workers 24

# benchmark (runs 0-99) with every tracker, then the LaTeX tables
python -m mml_research.motion.benchmark_exit_prediction --models-dir $M/models --out-dir $M/results \
    --workers 24 --process_noise $(cat $M/noise_sweep/chosen_noise.txt) \
    --enff_lambda <chosen> --enff_sigma_min <chosen>
python -m mml_research.motion.make_benchmark_tables --root $M
```
