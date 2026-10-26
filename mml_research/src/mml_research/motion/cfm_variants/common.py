"""
common.py — constants, episode splits, paths and small utilities shared by
every cfm_variants step.

Episode splits
--------------
An episode is one simulate_full_trajectory() run; every random stream of a
run derives from its index (trajectory 1000+run, measurements 5000+run,
filter 9000+run), as in benchmark_exit_prediction.py. Splits are disjoint
sets of run indices, chosen before any result was looked at:

    train  200..247   rollout data for M1 / M4 / M3 (240..247 held out for
                      the training-loss report only)
    val    100..139   M2 noise selection (the old noise sweep used 100..119)
    test   1000..1099 new locked test set, never used before this study
    dev    0..99      the episodes of the preliminary table; regression only
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

import yaml

from mml_core.config import motion_resource_dir

SPLITS = {
    "train": list(range(200, 248)),
    "val": list(range(100, 140)),
    "test": list(range(1000, 1100)),
    "dev": list(range(0, 100)),
}
TRAIN_HOLDOUT_RUNS = set(range(240, 248))

# Seed offsets of benchmark_exit_prediction.run_group.
TRAJ_SEED, MEAS_SEED, FILTER_SEED = 1000, 5000, 9000

ROOT = motion_resource_dir() / "exit_benchmark"
DEFAULT_MODELS_DIR = str(ROOT / "models")
DEFAULT_OUT_DIR = str(motion_resource_dir() / "cfm_variants_v1")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: str | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def variant_models_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "models")


def ckpt_path(out_dir: str, method: str, seed: int) -> str:
    return os.path.join(variant_models_dir(out_dir), f"{method}_s{seed}.pth")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)
