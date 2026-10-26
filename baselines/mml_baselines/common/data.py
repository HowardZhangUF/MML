"""
Trajectory-forecasting training data for the open-source baselines.

The learned motion models already in the benchmark (CFM, NF, DMMN, ...) are
trained on converted_noisy_5_v2_velocities_training_data.csv: a 5-step window
of noisy *visible* measurements -> the velocity to the next visible
measurement. TrajFlow, MoFlow and TCFM are trajectory forecasters and need a
multi-step future, so this module regenerates that exact measurement stream
(same simulator, same seed) and extends every CSV sample with the next
`horizon` visible measurements.

Kept identical to the CSV:
    * inputs X (5 noisy visible positions, flattened)
    * the first future step: Y[:, 0] is the CSV velocity label
    * the sample filter (window time span <= MAX_GAP)
    * the chronological 70/30 split of mml_core.motion.utils
Only change: samples whose next `horizon` visible measurements run past the
end of a node's stream are dropped (at most horizon-1 per start node).

Future representation
---------------------
v_k  = (p_k - p_{k-1}) / dt_k   per-interval velocity, CSV convention (m/s)
Q_k  = DT * sum_{j<=k} v_j      nominal future position relative to the last
                                history measurement (m). For gap-free windows
                                this is exactly p_k - p_0.
The forecasters model Q (a relative future trajectory, as in all three
upstream codebases); the particle filter uses v_1 = Q_1 / DT, the same
quantity every other motion model proposes.
"""

from __future__ import annotations

import math

import numpy as np

from mml_research.motion.generate_synthetic_data import (
    DT, MAX_GAP, N_TH, NODE_POS, simulate_from_node,
)

DEFAULT_HORIZON = 12
TRAIN_FRAC = 0.7


def build_trajectory_dataset(horizon: int = DEFAULT_HORIZON, n_th: int = N_TH, seed: int = 42):
    """
    Returns a dict of arrays, one row per CSV sample with a full future:
        X      (N, n_th*2)      history window, absolute positions
        V      (N, horizon, 2)  future per-interval velocities
        Q      (N, horizon, 2)  nominal future positions relative to X's last point
        row    (N,)             row index in the CSV file
        train  (N,)             bool, chronological split of load_training_csv
    """
    np.random.seed(seed)          # generate_synthetic_data.main() seeds the global RNG
    xs, vs, rows = [], [], []
    row = 0
    for start_node in range(len(NODE_POS)):
        pos, t = simulate_from_node(start_node)
        m = len(pos)
        for i in range(m - n_th - 1):
            dt_step = t[i + n_th] - t[i + n_th - 1]
            if t[i + n_th] - t[i] > MAX_GAP or dt_step < 1e-6:
                continue                      # same filter as build_dataset()
            this_row, row = row, row + 1
            last = i + n_th + horizon - 1     # index of the final future measurement
            if last >= m:
                continue
            idx = np.arange(i + n_th, last + 1)
            v = (pos[idx] - pos[idx - 1]) / (t[idx] - t[idx - 1])[:, None]
            xs.append(pos[i:i + n_th].reshape(-1))
            vs.append(v)
            rows.append(this_row)

    X = np.asarray(xs, dtype=np.float32)
    V = np.asarray(vs, dtype=np.float32)
    rows = np.asarray(rows)
    # load_training_csv() reads the header-less CSV with pandas' default header,
    # so file row 0 is dropped; chronological_split() then keeps the later 70%.
    n_loaded = row - 1
    split = math.floor(n_loaded * (1 - TRAIN_FRAC))
    loaded_idx = rows - 1
    keep = loaded_idx >= 0
    return {
        "X": X[keep], "V": V[keep], "Q": (DT * np.cumsum(V, axis=1))[keep].astype(np.float32),
        "row": rows[keep], "train": (loaded_idx >= split)[keep],
    }


def history_features(x_flat: np.ndarray, n_th: int = N_TH, dt: float = DT) -> np.ndarray:
    """
    (N, n_th*2) absolute positions -> (N, n_th, 6) per-step features
    [absolute x, y | relative to the last point | finite-difference velocity],
    the MoFlow ETH-UCY feature layout (abs, rel, vel), used by all three ports
    so the forecasters see the same history information.
    """
    p = np.asarray(x_flat, dtype=np.float32).reshape(-1, n_th, 2)
    rel = p - p[:, -1:, :]
    vel = np.zeros_like(p)
    vel[:, 1:] = (p[:, 1:] - p[:, :-1]) / dt
    return np.concatenate([p, rel, vel], axis=-1)


class MinMax:
    """Per-channel min-max scaling to [-1, 1] (TrajFlow 'linear', MoFlow 'min_max', TCFM LimitsNormalizer)."""

    def __init__(self, lo=None, hi=None):
        self.lo = None if lo is None else np.asarray(lo, dtype=np.float32)
        self.hi = None if hi is None else np.asarray(hi, dtype=np.float32)

    def fit(self, a: np.ndarray):
        flat = a.reshape(-1, a.shape[-1])
        self.lo, self.hi = flat.min(0).astype(np.float32), flat.max(0).astype(np.float32)
        return self

    def norm(self, a):
        return (a - self.lo) / (self.hi - self.lo) * 2 - 1

    def unnorm(self, a):
        return (a + 1) / 2 * (self.hi - self.lo) + self.lo

    def state(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}
