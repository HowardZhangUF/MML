"""common.py — config, paths and the canonical-frame model wrapper of the WOMD benchmark."""

from __future__ import annotations

import os

import numpy as np

from mml_core.config import motion_resource_dir
from mml_research.motion.ethucy.common import load_config as load_ethucy_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_OUT_DIR = str(motion_resource_dir() / "womd_v1")
RECIPE = "womd"            # checkpoint prefix: models/womd_<model>_s<seed>.pth
TYPES = ["vehicle", "pedestrian", "cyclist"]


def load_config(path: str | None = None) -> dict:
    """WOMD config on top of the fair_retrain protocol (same base merge as ETH-UCY)."""
    return load_ethucy_config(path or CONFIG_PATH)


def frame(h: np.ndarray, min_disp: float) -> np.ndarray:
    """
    h (N, T, 2) translated so h[:, -1] = 0 -> (N, 2, 2) row-vector rotations M,
    p_canonical = p @ M, aligning h[:, -1] - h[:, 0] with +x (identity below min_disp).
    """
    d = h[:, -1] - h[:, 0]
    n = np.linalg.norm(d, axis=-1)
    ok = n >= min_disp
    u = d / np.maximum(n, 1e-12)[:, None]
    c, s = np.where(ok, u[:, 0], 1.0), np.where(ok, u[:, 1], 0.0)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], 1).astype(np.float32)


def canonical(h: np.ndarray, min_disp: float):
    """(N, T, 2) model units -> (translated + rotated history, M)."""
    h = h - h[:, -1:]
    M = frame(h, min_disp)
    return np.einsum("ntj,njk->ntk", h, M), M


class Canonical:
    """
    A motion model seen through the benchmark's canonical frame: each context
    is translated to its last point and rotated by its own first->last
    direction before propose(); the velocity is rotated back.
    """

    def __init__(self, model, min_disp: float):
        self.model, self.min_disp = model, min_disp
        self.block_size, self.dt = model.block_size, model.dt
        self.uses_context = False

    def propose(self, c_flat: np.ndarray) -> np.ndarray:
        c = np.asarray(c_flat, np.float32).reshape(len(c_flat), self.block_size, 2)
        hc, M = canonical(c, self.min_disp)
        v = self.model.propose(hc.reshape(len(c), -1))
        return np.einsum("nk,njk->nj", v, M).astype(np.float32)
