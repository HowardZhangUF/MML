"""
utils.py
========
Shared utilities for the mml_network CFM motion model stack.

Functions
---------
  load_training_csv        load a converted_*_training_data.csv file
  make_history_dataset     build (X, y) from a raw position array
  chronological_split      time-aware train/test split (matches train.py)
  HistoryBuffer            sliding-window buffer for online inference

Place at:
    mml_guidance/mml_guidance/mml_network/utils.py
"""

from __future__ import annotations

import math
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_training_csv(csv_path: str,
                      input_dim: int = 2
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a pre-converted training CSV produced by convert_data.py or
    generate_synthetic_data.py.

    CSV column layout (input_dim = 2):
        [x0, y0,  x1, y1,  ...,  x_{N_th}, y_{N_th},  vx, vy]
         ──────────────────────────────────── context ──  targets

    Parameters
    ----------
    csv_path  : path to the CSV file
    input_dim : 2 for (x,y), 3 for (x,y,dt) — must match how the CSV was made

    Returns
    -------
    X : (N, N_th * input_dim)   context windows
    y : (N, 2)                   velocity targets  (vx, vy)
    """
    df  = pd.read_csv(csv_path)
    arr = df.values.astype(np.float32)

    # Last input_dim columns = targets; everything before = context
    X = arr[:, :-input_dim]         # (N, N_th * input_dim)
    y = arr[:, -input_dim:]         # (N, input_dim)

    # If input_dim==3 the last col is a timestamp — drop it from targets
    if input_dim == 3:
        y = y[:, :2]                # keep only (vx, vy)

    print(f"[load_training_csv] X: {X.shape}  y: {y.shape}")
    return X, y


# ── Dataset construction from raw positions ───────────────────────────────────

def make_history_dataset(positions:   np.ndarray,
                          N_th:        int   = 5,
                          predict_vel: bool  = True,
                          dt:          float = 0.033
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) training pairs from a raw (T, 2) position array.

    Parameters
    ----------
    positions   : (T, 2)  noisy position measurements
    N_th        : history window length
    predict_vel : True  → y = finite-difference velocity  (vx, vy)
                  False → y = next position
    dt          : seconds per step (used only when predict_vel=True)

    Returns
    -------
    X : (N, N_th * 2)   flattened history windows
    y : (N, 2)           targets
    """
    T       = len(positions)
    X_list  = []
    y_list  = []

    for i in range(T - N_th - 1):
        window = positions[i : i + N_th]          # (N_th, 2)
        next_p = positions[i + N_th]              # (2,)

        if predict_vel:
            target = (next_p - window[-1]) / dt
        else:
            target = next_p

        X_list.append(window.flatten())           # (N_th*2,)
        y_list.append(target)                     # (2,)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"[make_history_dataset] X: {X.shape}  y: {y.shape}")
    return X, y


# ── Train / test split ────────────────────────────────────────────────────────

def chronological_split(X: np.ndarray,
                         y: np.ndarray,
                         train_frac: float = 0.7
                         ) -> Tuple[np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]:
    """
    Chronological (time-ordered) train/test split.

    Matches get_ordered_train_test() in train.py:
        LATER data  → training set   (the model sees recent behaviour)
        EARLIER data→ test set        (evaluate generalisation to older data)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    n     = len(X)
    split = math.floor(n * (1 - train_frac))      # index dividing old / new
    return X[split:], X[:split], y[split:], y[:split]


# ── Online history buffer ─────────────────────────────────────────────────────

class HistoryBuffer:
    """
    Fixed-capacity sliding window for online inference.

    Keeps the last `capacity` observations of dimension `dim`.
    Once full, the oldest entry is discarded on each push().

    Usage
    -----
    buf = HistoryBuffer(capacity=5, dim=2)
    buf.push(np.array([x, y]))        # call once per visible measurement
    if buf.is_ready():
        c_flat = buf.flat()           # (capacity * dim,)  → feed to CFM
    """

    def __init__(self, capacity: int = 5, dim: int = 2):
        self.capacity = capacity
        self.dim      = dim
        self._buf: List[np.ndarray] = []

    def reset(self):
        """Clear the buffer."""
        self._buf = []

    def push(self, obs: np.ndarray):
        """
        Append one observation of shape (dim,) and drop the oldest if full.
        """
        self._buf.append(np.asarray(obs, dtype=np.float32).flatten())
        if len(self._buf) > self.capacity:
            self._buf.pop(0)

    def is_ready(self) -> bool:
        """True once `capacity` observations have been collected."""
        return len(self._buf) == self.capacity

    def flat(self) -> np.ndarray:
        """
        Return the full buffer as a flat (capacity * dim,) array.
        Raises AssertionError if not yet full — check is_ready() first.
        """
        assert self.is_ready(), (
            f"HistoryBuffer not full: {len(self._buf)}/{self.capacity} steps."
        )
        return np.concatenate(self._buf)   # (capacity * dim,)

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return (f"HistoryBuffer(capacity={self.capacity}, dim={self.dim}, "
                f"filled={len(self._buf)})")