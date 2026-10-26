"""
cfm_variants
============
Four CFM modifications for the particle filter, benchmarked on the
exit-prediction protocol of benchmark_exit_prediction.py
(see CFM_PF_Modification_and_Benchmark_Guide.md):

    M1  Context-CFM       history mask, occlusion age and cloud covariance as
                          extra CFM conditioning, trained on causal PF rollouts
    M2  Calibrated-CFM    frozen CFM, added position noise Q chosen on
                          validation runs by a preregistered rule
    M3  Observation-CFM   Gaussian proposal on the CFM's standard-normal input
                          given z_k, weights corrected by phi(xi) / q(xi)
    M4  DMMN-CFM          frozen DMMN mean + CFM on its standardized residual
                          (control: DMMN + fitted Gaussian residual)

Pipeline (each step is `python -m mml_research.motion.cfm_variants.<step>`):
    audit -> prepare -> train -> verify -> calibrate -> run -> summarize
"""
