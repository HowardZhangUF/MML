"""common.py — paths, config and checkpoint names of the fair retraining study."""

from __future__ import annotations

import os

import yaml

from mml_core.config import motion_resource_dir
from mml_research.motion.cfm_variants.common import load_config as load_variants_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_OUT_DIR = str(motion_resource_dir() / "fair_v1")
VARIANTS_DIR = str(motion_resource_dir() / "cfm_variants_v1")

# Display names of the trained models.
MODEL_LABEL = {"cfm": "CFM", "nf": "NF", "dmmn": "DMMN", "dmmn_gauss": "DMMN+Gauss", "m4": "M4",
               "trajflow": "TrajFlow", "moflow": "MoFlow", "tcfm": "TCFM", "m1_control": "M1-control",
               "m1_mask": "M1-mask", "m1_time": "M1-time", "m1_full": "M1-full"}


def load_config(path: str | None = None) -> dict:
    path = path or CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["variants"] = load_variants_config(os.path.normpath(os.path.join(os.path.dirname(path),
                                                                         cfg["variants_config"])))
    return cfg


def ckpt(out_dir: str, recipe: str, model: str, seed: int) -> str:
    return os.path.join(out_dir, "models", f"{recipe}_{model}_s{seed}.pth")


def method_name(recipe: str, model: str) -> str:
    return f"{recipe}/{MODEL_LABEL[model]}"


def all_methods(cfg: dict) -> list[str]:
    """Evaluation order: learned (recipe-major) then training-free."""
    return [method_name(r, m) for r in ("R0", "R1") for m in cfg["models"][r]] + list(cfg["training_free"])
