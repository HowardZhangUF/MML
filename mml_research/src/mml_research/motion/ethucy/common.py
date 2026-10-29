"""common.py — config, paths and the model-unit wrapper of the ETH-UCY benchmark."""

from __future__ import annotations

import os

import numpy as np
import yaml

from mml_core.config import motion_resource_dir
from mml_research.motion.fair_retrain.common import load_config as load_fair_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_OUT_DIR = str(motion_resource_dir() / "ethucy_v1")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path: str | None = None) -> dict:
    """ETH-UCY config on top of the fair_retrain protocol (optimizers, architectures, variants)."""
    path = path or CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = load_fair_config(os.path.normpath(os.path.join(os.path.dirname(path), cfg.pop("base"))))
    cfg = _merge(base, cfg)
    cfg["dt"] = cfg["frame_dt"]
    return cfg


class Translated:
    """
    A motion model seen through the benchmark's translation: each context is
    shifted so its last position is the origin before propose(); velocities
    are translation invariant, so nothing changes on the way out.
    """

    def __init__(self, model):
        self.model = model
        self.block_size, self.dt = model.block_size, model.dt
        self.uses_context = False

    def propose(self, c_flat: np.ndarray) -> np.ndarray:
        c = np.asarray(c_flat, np.float32)
        return self.model.propose(c - np.tile(c[:, -2:], self.block_size))
