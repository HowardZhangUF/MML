| Method | q | RMSE all | RMSE vis | RMSE occ | Exit top-1 | >50% | P(correct) | Brier | ExitNLL | ES | Recovery | ESS/N | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PF+CFM | 0.045 | 0.074 ± 0.016 | 0.019 ± 0.000 | 0.182 ± 0.039 | 45.5% (319/701) | 45.5% | 0.422 | 0.677 | 1.04 | 0.0298 | 99.9% | 0.52 | 16.9 |
| PF+DMMN | 0.045 | 0.064 ± 0.032 | 0.019 ± 0.005 | 0.154 ± 0.079 | 91.0% (638/701) | 91.0% | 0.759 | 0.199 | 0.52 | 0.0275 | 99.9% | 0.54 | 10.1 |
| PF+NF | 0.032 | 0.062 ± 0.026 | 0.015 ± 0.001 | 0.153 ± 0.065 | 91.0% (638/701) | 90.7% | 0.661 | 0.329 | 0.68 | 0.0243 | 99.9% | 0.59 | 16.5 |

Lane-cue reference: 638/701 (91.0%).

Paired differences (method − control, mean [95% episode-bootstrap CI]):

| Method − control | ΔRMSE all | ΔRMSE occ | ΔES | Δexit top-1 | ΔP(correct) | ΔBrier | events differ |
|---|---|---|---|---|---|---|---|
