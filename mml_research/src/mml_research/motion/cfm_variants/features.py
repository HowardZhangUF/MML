"""
features.py — causal context features for M1 (guide 4.1).

    mask  (H,)  1 = a measurement was available and used at that history
                step. History entries are particle states either way; the
                mask says how they were formed, not that they equal z.
    d     ()    seconds since the last used measurement (0 right after one)
    U     (3,)  [log(P11+eps), log(P22+eps), P12 / sqrt((P11+eps)(P22+eps))]
                of the weighted particle cloud after the update at k-1

All three are cloud-level (shared by every particle) and never use z_k,
ground truth or the realised exit.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6
FEATURE_ORDER = ("mask", "time", "cov")     # per-step: mask; global: d, then U


def cloud_summary(pos: np.ndarray, w: np.ndarray) -> np.ndarray:
    mean = np.average(pos, axis=0, weights=w)
    d = pos - mean
    P = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
    p11, p22 = P[0, 0] + EPS, P[1, 1] + EPS
    return np.array([np.log(p11), np.log(p22), P[0, 1] / np.sqrt(p11 * p22)], dtype=np.float32)


def global_features(features, d, U) -> np.ndarray:
    """(B, n_global) raw global features in FEATURE_ORDER for the enabled feature set."""
    cols = []
    if "time" in features:
        cols.append(np.asarray(d, np.float32).reshape(-1, 1))
    if "cov" in features:
        cols.append(np.asarray(U, np.float32).reshape(-1, 3))
    if not cols:
        return np.zeros((len(np.atleast_1d(d)), 0), np.float32)
    return np.concatenate(cols, axis=1)


class Normalizer:
    """Per-column standardisation fitted on training data only."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, np.float32)
        self.std = None if std is None else np.asarray(std, np.float32)

    def fit(self, x: np.ndarray):
        self.mean = x.mean(0).astype(np.float32)
        self.std = np.maximum(x.std(0), 1e-3).astype(np.float32)
        return self

    def __call__(self, x):
        return (x - self.mean) / self.std if x.shape[-1] else x

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])
