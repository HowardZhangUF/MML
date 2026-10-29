"""
methods.py — method registry: benchmark name -> tracker factory.

Controlled comparisons hold Q at the baseline value (q = 0.045 m, the
validated noise of PF+CFM and PF+DMMN) unless Q is the intervention:

    PF+CFM         frozen CFM, q0           reference; = M2-original; M3's paired prior
    PF+CFM-AS      adaptive source, q0      CFM vs CFM-AS at matched Q (guide 5.4)
    PF+DMMN        frozen DMMN, q0          M4's reference
    PF+NF          frozen NF, tuned q       contextual baseline
    M2-zero        frozen CFM, q = 0
    M2-calibrated  frozen CFM, q from locked.json (calibrate.py, validation runs)
    DMMN+Gauss     DMMN + fitted Gaussian residual, q0
    M4             DMMN + residual CFM, q0                    (seeds)
    M3             CFM prior + latent Gaussian proposal on xi, q0   (seeds)
    M3-joint       extension: proposal on (xi, eta/q) jointly, q0  (seeds)
    M1-control / M1-mask / M1-time / M1-full, q0              (seeds)

Open-source / earlier baselines (frozen exit-benchmark checkpoints; q, and
EnFF's lambda / sigma_min, chosen on validation runs 100..119 by the exit
benchmark's sweep, recorded in locked_addendum.json by lock_addendum.py
before any of them ran on the test split):
    PF+CV, PF+TrajFlow, PF+MoFlow, PF+TCFM   bootstrap PF, tuned q
    EnFF                                     its own ensemble tracker
    DiffPF                                   its own generative tracker (no q)
"""

from __future__ import annotations

import os

import numpy as np

from mml_core.motion.cfm_motion_model import CFMMotionModel
from mml_core.motion.nf_motion_model import NFMotionModel
from mml_research.motion.benchmark_cv_dmmn import OfflineDMMNMotionModel
from mml_research.motion.cfm_variants.common import ckpt_path, read_json
from mml_research.motion.cfm_variants.tracker import (
    AdaptiveSigmaSampler, LatentProposalSampler, MotionSampler, VariantPFTracker,
)

# name: (kind, q key, checkpoint method name or None)
METHODS = {
    "PF+CFM": ("cfm", "baseline", None),
    "PF+CFM-AS": ("cfm_as", "baseline", None),
    "PF+DMMN": ("dmmn", "baseline", None),
    "PF+NF": ("nf", "nf", None),
    "M2-zero": ("cfm", 0.0, None),
    "M2-calibrated": ("cfm", "m2_selected", None),
    "DMMN+Gauss": ("gauss", "baseline", "dmmn_gauss"),
    "M4": ("m4", "baseline", "m4"),
    "M3": ("m3", "baseline", "m3"),
    "M3-joint": ("m3", "baseline", "m3_joint"),
    "M1-control": ("m1", "baseline", "m1_control"),
    "M1-mask": ("m1", "baseline", "m1_mask"),
    "M1-time": ("m1", "baseline", "m1_time"),
    "M1-full": ("m1", "baseline", "m1_full"),
    "PF+CV": ("cv", "addendum", None),
    "PF+TrajFlow": ("trajflow", "addendum", None),
    "PF+MoFlow": ("moflow", "addendum", None),
    "PF+TCFM": ("tcfm", "addendum", None),
    "EnFF": ("enff", "addendum", None),
    "DiffPF": ("diffpf", 0.0, None),
}
# Exit-benchmark checkpoints of the open-source ports / DiffPF.
PORT_CKPT = {"trajflow": "trajflow_motion_model.pth", "moflow": "moflow_motion_model.pth",
             "tcfm": "tcfm_motion_model.pth", "diffpf": "diffpf_tracker.pth"}
OPEN_SOURCE = ["PF+CV", "PF+TrajFlow", "PF+MoFlow", "PF+TCFM", "EnFF", "DiffPF"]
SEEDED = {name for name, (kind, _, ck) in METHODS.items() if ck and kind != "gauss"}
ALL_METHODS = list(METHODS)
# Paired comparisons reported by summarize.py: (method, its control).
COMPARISONS = [
    ("PF+CFM-AS", "PF+CFM"), ("M2-zero", "PF+CFM"), ("M2-calibrated", "PF+CFM"),
    ("M3", "PF+CFM"), ("M3-joint", "PF+CFM"), ("M3-joint", "M3"),
    ("DMMN+Gauss", "PF+DMMN"), ("M4", "PF+DMMN"), ("M4", "DMMN+Gauss"),
    ("M1-control", "PF+CFM"), ("M1-mask", "M1-control"), ("M1-time", "M1-mask"),
    ("M1-full", "M1-time"), ("M1-full", "M1-control"),
]


def resolve_q(name: str, cfg: dict, out_dir: str, q_override: float | None = None) -> float:
    if q_override is not None:
        return float(q_override)
    key = METHODS[name][1]
    if isinstance(key, float):
        return key
    if key == "baseline":
        return float(cfg["filter"]["baseline_process_noise"])
    if key == "nf":
        return float(cfg["filter"]["nf_process_noise"])
    if key == "m2_selected":
        return float(read_json(os.path.join(out_dir, "locked.json"))["m2_selected_q"])
    if key == "addendum":
        return float(read_json(os.path.join(out_dir, "locked_addendum.json"))["process_noise"][name])
    raise KeyError(key)


class _UniformCloud:
    """DiffPF exposes unweighted particles only; give it the cloud() the evaluation reads."""

    def __init__(self, tracker):
        self.tracker = tracker

    def reset(self, z):
        self.tracker.reset(z)

    def step(self, z):
        return self.tracker.step(z)

    def cloud(self):
        pts = self.tracker.particles.copy()
        return pts, np.full(len(pts), 1.0 / len(pts))

    def get_state(self):
        return self.tracker.particles.copy(), self.tracker._prev_mean.copy()

    def set_state(self, s):
        self.tracker.particles, self.tracker._prev_mean = s


class ModelCache:
    """Per-process model store, so each checkpoint loads once per worker."""

    def __init__(self, cfg: dict, models_dir: str, out_dir: str):
        self.cfg, self.models_dir, self.out_dir = cfg, models_dir, out_dir
        self._store = {}

    def _get(self, key, build):
        if key not in self._store:
            self._store[key] = build()
        return self._store[key]

    def cfm(self):
        def build():
            m = CFMMotionModel()
            m.load(os.path.join(self.models_dir, self.cfg["cfm"]["checkpoint"]))
            return m
        return self._get("cfm", build)

    def dmmn_mean(self):
        from mml_research.motion.cfm_variants.residual_model import DMMNMean
        return self._get("dmmn_mean", lambda: DMMNMean(
            os.path.join(self.models_dir, self.cfg["dmmn_checkpoint"]), self.cfg["filter"]["dt_seconds"]))

    def model(self, name: str, seed: int):
        kind, _, ck = METHODS[name]
        dt = self.cfg["filter"]["dt_seconds"]
        if kind in ("cfm", "cfm_as"):
            return self.cfm()
        if kind == "dmmn":
            return self._get("dmmn", lambda: OfflineDMMNMotionModel(
                os.path.join(self.models_dir, self.cfg["dmmn_checkpoint"]), dt=dt))
        if kind == "nf":
            def build():
                m = NFMotionModel()
                m.load(os.path.join(self.models_dir, self.cfg["nf_checkpoint"]))
                return m
            return self._get("nf", build)
        if kind == "gauss":
            from mml_research.motion.cfm_variants.residual_model import GaussianResidualMotionModel
            return self._get(("gauss",), lambda: GaussianResidualMotionModel.load(
                ckpt_path(self.out_dir, ck, 0), self.dmmn_mean()))
        if kind == "m4":
            from mml_research.motion.cfm_variants.residual_model import ResidualCFMMotionModel
            return self._get((ck, seed), lambda: ResidualCFMMotionModel.load(
                ckpt_path(self.out_dir, ck, seed), self.dmmn_mean()))
        if kind == "m3":
            from mml_research.motion.cfm_variants.observation_proposal import LatentGaussianProposal
            return self._get((ck, seed), lambda: LatentGaussianProposal.load(ckpt_path(self.out_dir, ck, seed)))
        if kind == "m1":
            from mml_research.motion.cfm_variants.context_model import ContextCFMMotionModel
            return self._get((ck, seed), lambda: ContextCFMMotionModel.load(ckpt_path(self.out_dir, ck, seed)))
        if kind == "cv":
            from mml_research.motion.benchmark_cv_dmmn import ConstantVelocityMotionModel
            return self._get("cv", lambda: ConstantVelocityMotionModel(block_size=5, dt=dt))
        if kind in ("trajflow", "moflow", "tcfm"):
            def build():
                from mml_baselines.moflow.motion_model import MoFlowMotionModel
                from mml_baselines.tcfm.motion_model import TCFMMotionModel
                from mml_baselines.trajflow.motion_model import TrajFlowMotionModel
                cls = {"trajflow": TrajFlowMotionModel, "moflow": MoFlowMotionModel, "tcfm": TCFMMotionModel}[kind]
                return cls(os.path.join(self.models_dir, PORT_CKPT[kind]))
            return self._get(kind, build)
        if kind in ("enff", "diffpf"):
            return None
        raise KeyError(kind)

    def tracker_factory(self, name: str, seed: int, q: float):
        cfg = self.cfg
        kind = METHODS[name][0]
        dt = cfg["filter"]["dt_seconds"]
        model = self.model(name, seed)
        if kind == "enff":
            from mml_baselines.enff.tracker import EnFFTracker
            e = read_json(os.path.join(self.out_dir, "locked_addendum.json"))["enff"]
            return lambda: EnFFTracker(n_particles=cfg["filter"]["particles"], process_noise=q,
                                       meas_sigma=cfg["filter"]["measurement_sigma"], sigma_min=e["sigma_min"],
                                       guidance_lambda=e["lambda"], sampling_time_step_count=e["steps"], dt=dt)
        if kind == "diffpf":
            from mml_core.tracking.diffpf_tracker import DiffPFTracker
            def build():
                t = DiffPFTracker()
                t.load(os.path.join(self.models_dir, PORT_CKPT["diffpf"]))
                return t
            tracker = self._get("diffpf", build)
            return lambda: _UniformCloud(tracker)
        if kind == "cfm_as":
            make = lambda: AdaptiveSigmaSampler(model, q, cfg["cfm"]["adaptive_gain"], cfg["cfm"]["sigma_max"])
        elif kind == "m3":
            make = lambda: LatentProposalSampler(self.cfm(), model, q)
        else:
            make = lambda: MotionSampler(model, q)
        f = cfg["filter"]
        return lambda: VariantPFTracker(make(), n_particles=f["particles"], meas_sigma=f["measurement_sigma"],
                                        init_spread=f["init_spread"])


def checkpoint_files(names, seeds, cfg, models_dir, out_dir):
    """Every checkpoint file the given methods load (for hashing/locking)."""
    files = {}
    for name in names:
        kind, _, ck = METHODS[name]
        if kind in ("cfm", "cfm_as", "m3"):
            files["cfm"] = os.path.join(models_dir, cfg["cfm"]["checkpoint"])
        if kind in ("dmmn", "gauss", "m4"):
            files["dmmn"] = os.path.join(models_dir, cfg["dmmn_checkpoint"])
        if kind == "nf":
            files["nf"] = os.path.join(models_dir, cfg["nf_checkpoint"])
        if kind in PORT_CKPT:
            files[kind] = os.path.join(models_dir, PORT_CKPT[kind])
        if ck:
            for s in ([0] if kind == "gauss" else seeds):
                files[f"{ck}_s{s}"] = ckpt_path(out_dir, ck, s)
    return files
