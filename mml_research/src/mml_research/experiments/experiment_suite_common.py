#!/usr/bin/env python3
"""Shared, reproducible machinery for the four paper experiments."""

from __future__ import annotations

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import math
import random
import sys
import time
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist


ROOT = project_root()

from mml_research.motion.benchmark_cv_dmmn import (  # noqa: E402
    ConstantVelocityMotionModel,
    OfflineDMMNMotionModel,
)
from mml_core.motion import load_cfm_model, load_diffusion_model, load_nf_model  # noqa: E402


DT = 0.033
ALL_METHODS = (
    "PF+CV",
    "PF+DMMN",
    "PF+Diffusion",
    "PF+NF",
    "PF+CFM-fixed",
    "PF+CFM-adaptive",
)
EXPERIMENT1_METHODS = ALL_METHODS[:-1]

METHOD_ALIASES = {
    "cv": "PF+CV",
    "pf+cv": "PF+CV",
    "dmmn": "PF+DMMN",
    "pf+dmmn": "PF+DMMN",
    "diffusion": "PF+Diffusion",
    "pf+diffusion": "PF+Diffusion",
    "nf": "PF+NF",
    "pf+nf": "PF+NF",
    "cfm": "PF+CFM-fixed",
    "cfm-fixed": "PF+CFM-fixed",
    "pf+cfm": "PF+CFM-fixed",
    "pf+cfm-fixed": "PF+CFM-fixed",
    "cfm-adaptive": "PF+CFM-adaptive",
    "pf+cfm-as": "PF+CFM-adaptive",
    "pf+cfm-adaptive": "PF+CFM-adaptive",
}

CHECKPOINTS = {
    "PF+DMMN": (
        motion_resource_dir()
        / "mml_guidance/mml_network/models"
        / "best_noisy_5_v2_velocities_occlusions_ScratchTransformer.pth"
    ),
    "PF+Diffusion": motion_resource_dir() / "models/diffusion_motion_model.pth",
    "PF+NF": motion_resource_dir() / "models/nf_motion_model.pth",
    "PF+CFM-fixed": motion_resource_dir() / "models/cfm_motion_model_anchor1.0.pth",
    "PF+CFM-adaptive": motion_resource_dir() / "models/cfm_motion_model_anchor1.0.pth",
}


def canonical_method(name: str) -> str:
    canonical = METHOD_ALIASES.get(name.strip().lower(), name.strip())
    if canonical not in ALL_METHODS:
        raise ValueError(f"Unknown method {name!r}; choose from {ALL_METHODS}.")
    return canonical


def canonical_methods(names: Sequence[str]) -> list[str]:
    methods = [canonical_method(name) for name in names]
    if len(methods) != len(set(methods)):
        raise ValueError("The method list contains aliases for the same method.")
    return methods


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class MotionAdapter:
    name: str
    model: object
    block_size: int
    dt: float = DT
    deterministic: bool = False
    diffusion_steps: int | None = None

    def set_measurement_age(
        self,
        age: int,
        adaptive_gain: float = 0.05,
        sigma_max: float = 2.5,
    ) -> None:
        if self.name != "PF+CFM-adaptive":
            return
        cfm = self.model._cfm
        cfm.anchor_sigma = min(sigma_max, 1.0 * (1.0 + adaptive_gain * age))

    def set_cfm_nfe(self, nfe: int) -> None:
        if not hasattr(self.model, "_cfm"):
            raise TypeError(f"{self.name} is not a CFM model.")
        self.model._cfm.N_FE = int(nfe)

    def propose(self, contexts: np.ndarray) -> np.ndarray:
        contexts = np.asarray(contexts, dtype=np.float32)
        if self.diffusion_steps is None:
            return np.asarray(self.model.propose(contexts), dtype=np.float32)
        return self._propose_diffusion_ddim(contexts, self.diffusion_steps)

    @torch.inference_mode()
    def _propose_diffusion_ddim(
        self,
        contexts: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """Reduced-step deterministic DDIM sampler for the runtime sweep."""
        diffusion = self.model._diffusion
        n_steps = max(1, min(int(n_steps), diffusion.T))
        device = next(diffusion.network.parameters()).device
        c_t = torch.as_tensor(contexts, dtype=torch.float32, device=device)
        embedding = diffusion.network.encode_context(c_t)
        sample = torch.randn((len(contexts), 2), device=device)
        timesteps = np.unique(
            np.rint(np.linspace(diffusion.T - 1, 0, n_steps)).astype(int)
        )[::-1]

        for index, timestep in enumerate(timesteps):
            t = torch.full(
                (len(contexts),), int(timestep), dtype=torch.long, device=device
            )
            eps = diffusion.network.predict_noise(sample, t, embedding)
            alpha_bar = diffusion.alphas_cumprod[int(timestep)]
            x0 = (
                sample - torch.sqrt(1.0 - alpha_bar) * eps
            ) / torch.sqrt(alpha_bar)
            next_timestep = (
                int(timesteps[index + 1]) if index + 1 < len(timesteps) else -1
            )
            if next_timestep < 0:
                sample = x0
            else:
                next_alpha_bar = diffusion.alphas_cumprod[next_timestep]
                sample = (
                    torch.sqrt(next_alpha_bar) * x0
                    + torch.sqrt(1.0 - next_alpha_bar) * eps
                )

        velocity = sample.cpu().numpy().astype(np.float32)
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        return velocity * np.minimum(1.0, 1.5 / (speed + 1e-9))


def load_motion_adapter(
    method: str,
    *,
    cfm_nfe: int | None = None,
    diffusion_steps: int | None = None,
) -> MotionAdapter:
    method = canonical_method(method)
    checkpoint = CHECKPOINTS.get(method)
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint for {method} not found: {checkpoint}")

    if method == "PF+CV":
        model = ConstantVelocityMotionModel(block_size=5, dt=DT)
        return MotionAdapter(method, model, model.block_size, model.dt, True)
    if method == "PF+DMMN":
        model = OfflineDMMNMotionModel(str(checkpoint), dt=DT)
        return MotionAdapter(method, model, model.block_size, model.dt, True)
    if method == "PF+Diffusion":
        model = load_diffusion_model(str(checkpoint), block_size=5, dt=DT)
        return MotionAdapter(
            method,
            model,
            model._diffusion.block_size,
            model.dt,
            False,
            diffusion_steps,
        )
    if method == "PF+NF":
        model = load_nf_model(str(checkpoint), block_size=5, dt=DT)
        return MotionAdapter(method, model, model._nf.block_size, model.dt)

    model = load_cfm_model(str(checkpoint), block_size=5, dt=DT)
    model._cfm.anchor_sigma = 1.0
    if cfm_nfe is not None:
        model._cfm.N_FE = int(cfm_nfe)
    return MotionAdapter(method, model, model._cfm.block_size, model.dt)


def load_motion_adapters(
    methods: Sequence[str],
    *,
    cfm_nfe: int | None = None,
    diffusion_steps: int | None = None,
) -> dict[str, MotionAdapter]:
    return {
        method: load_motion_adapter(
            method, cfm_nfe=cfm_nfe, diffusion_steps=diffusion_steps
        )
        for method in canonical_methods(methods)
    }


SCENARIOS = {
    "straight": (np.array([0.0]), np.array([1.0])),
    "two_way_balanced": (
        np.deg2rad(np.array([-45.0, 45.0])),
        np.array([0.5, 0.5]),
    ),
    "two_way_unbalanced": (
        np.deg2rad(np.array([-45.0, 45.0])),
        np.array([0.7, 0.3]),
    ),
    "three_way": (
        np.deg2rad(np.array([-55.0, 0.0, 55.0])),
        np.array([0.5, 0.3, 0.2]),
    ),
}


@dataclass(frozen=True)
class RoadContext:
    scenario: str
    context_id: int
    history: np.ndarray
    branch_angles: np.ndarray
    branch_probabilities: np.ndarray
    heading: float
    speed: float


def make_road_context(
    scenario: str,
    context_id: int,
    seed: int,
    history_length: int = 5,
) -> RoadContext:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, context_id]))
    relative_angles, probabilities = SCENARIOS[scenario]
    heading = float(rng.choice(np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])))
    speed = float(rng.uniform(0.65, 1.05))
    junction = rng.uniform(-0.35, 0.35, size=2)
    incoming = np.array([math.cos(heading), math.sin(heading)])
    offsets = np.arange(history_length - 1, -1, -1, dtype=float)
    history = junction - offsets[:, None] * speed * DT * incoming[None, :]
    return RoadContext(
        scenario=scenario,
        context_id=context_id,
        history=history.astype(np.float32),
        branch_angles=heading + relative_angles,
        branch_probabilities=probabilities.astype(float),
        heading=heading,
        speed=speed,
    )


def ideal_future(
    context: RoadContext,
    branch: int,
    horizon: int,
    turn_steps: int = 10,
) -> np.ndarray:
    position = context.history[-1].astype(float).copy()
    branch_heading = float(context.branch_angles[branch])
    points = []
    for step in range(horizon):
        fraction = min(1.0, (step + 1) / max(1, turn_steps))
        angle = context.heading + fraction * (branch_heading - context.heading)
        position = position + context.speed * DT * np.array(
            [math.cos(angle), math.sin(angle)]
        )
        points.append(position.copy())
    return np.asarray(points, dtype=np.float32)


def simulator_futures(
    context: RoadContext,
    n_samples: int,
    horizon: int,
    seed: int,
    process_std: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    branches = rng.choice(
        len(context.branch_probabilities),
        size=n_samples,
        p=context.branch_probabilities,
    )
    futures = np.empty((n_samples, horizon, 2), dtype=np.float32)
    prototypes = [
        ideal_future(context, branch, horizon)
        for branch in range(len(context.branch_probabilities))
    ]
    for index, branch in enumerate(branches):
        noise = rng.normal(0.0, process_std, size=(horizon, 2)).cumsum(axis=0)
        futures[index] = prototypes[int(branch)] + noise
    return futures, branches


def model_futures(
    adapter: MotionAdapter,
    history: np.ndarray,
    n_samples: int,
    horizon: int,
    seed: int,
) -> np.ndarray:
    seed_everything(seed)
    history = np.asarray(history, dtype=np.float32)[-adapter.block_size :]
    histories = np.repeat(history[None, :, :], n_samples, axis=0)
    futures = np.empty((n_samples, horizon, 2), dtype=np.float32)
    for step in range(horizon):
        velocity = adapter.propose(histories.reshape(n_samples, -1))
        next_position = histories[:, -1] + adapter.dt * velocity
        futures[:, step] = next_position
        histories = np.concatenate(
            [histories[:, 1:], next_position[:, None, :]], axis=1
        )
    return futures


def classify_futures(
    futures: np.ndarray,
    context: RoadContext,
    road_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    prototypes = np.stack(
        [
            ideal_future(context, branch, futures.shape[1])
            for branch in range(len(context.branch_probabilities))
        ]
    )
    distances = np.linalg.norm(
        futures[:, None, :, :] - prototypes[None, :, :, :], axis=-1
    )
    mean_distance = distances.mean(axis=-1)
    branch = np.argmin(mean_distance, axis=1)
    selected = distances[np.arange(len(futures)), branch]
    offroad = np.quantile(selected, 0.9, axis=1) > road_width
    return branch, offroad


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def energy_distance_samples(
    predicted: np.ndarray,
    reference: np.ndarray,
    max_samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    predicted = predicted.reshape(len(predicted), -1)
    reference = reference.reshape(len(reference), -1)
    if len(predicted) > max_samples:
        predicted = predicted[rng.choice(len(predicted), max_samples, replace=False)]
    if len(reference) > max_samples:
        reference = reference[rng.choice(len(reference), max_samples, replace=False)]
    return float(
        2.0 * cdist(predicted, reference).mean()
        - cdist(predicted, predicted).mean()
        - cdist(reference, reference).mean()
    )


def evaluate_distribution_context(
    adapter: MotionAdapter,
    context: RoadContext,
    n_samples: int,
    horizon: int,
    seed: int,
    road_width: float,
    energy_samples: int,
) -> tuple[dict, np.ndarray]:
    empirical, _ = simulator_futures(
        context, n_samples=n_samples, horizon=horizon, seed=seed + 11
    )
    predicted = model_futures(
        adapter, context.history, n_samples, horizon, seed=seed + 23
    )
    branches, offroad = classify_futures(predicted, context, road_width)
    predicted_mass = np.array(
        [np.mean((branches == branch) & ~offroad) for branch in range(len(context.branch_probabilities))]
        + [np.mean(offroad)],
        dtype=float,
    )
    target_mass = np.r_[context.branch_probabilities, 0.0]
    row = {
        "method": adapter.name,
        "scenario": context.scenario,
        "context_id": context.context_id,
        "branch_jsd_bits": jensen_shannon(target_mass, predicted_mass),
        "energy_distance": energy_distance_samples(
            predicted, empirical, energy_samples, seed + 31
        ),
        "offroad_rate": float(offroad.mean()),
    }
    for branch, probability in enumerate(predicted_mass[:-1]):
        row[f"predicted_branch_{branch}"] = float(probability)
        row[f"true_branch_{branch}"] = float(context.branch_probabilities[branch])
    return row, predicted_mass


def confidence_summary(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(list(group_columns), dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n"] = len(group)
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else (0.0 if len(values) == 1 else np.nan)
            )
            row[f"{metric}_ci95"] = (
                1.96 * row[f"{metric}_std"] / math.sqrt(len(values))
                if len(values)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference_summary(
    frame: pd.DataFrame,
    index_columns: Sequence[str],
    metric_columns: Sequence[str],
    reference_method: str,
) -> pd.DataFrame:
    """Paired candidate-minus-reference effects with normal 95% CIs."""
    rows = []
    methods = [method for method in frame["method"].unique() if method != reference_method]
    for metric in metric_columns:
        pivot = frame.pivot_table(
            index=list(index_columns),
            columns="method",
            values=metric,
            aggfunc="first",
        )
        if reference_method not in pivot:
            continue
        for method in methods:
            if method not in pivot:
                continue
            paired = pivot[[method, reference_method]].dropna()
            differences = (
                paired[method].to_numpy(dtype=float)
                - paired[reference_method].to_numpy(dtype=float)
            )
            if len(differences) == 0:
                continue
            std = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
            rows.append(
                {
                    "metric": metric,
                    "candidate_method": method,
                    "reference_method": reference_method,
                    "difference_direction": "candidate_minus_reference",
                    "n_pairs": len(differences),
                    "mean_paired_difference": float(np.mean(differences)),
                    "paired_difference_std": std,
                    "paired_difference_ci95": 1.96 * std / math.sqrt(len(differences)),
                }
            )
    return pd.DataFrame(rows)


class ControlledBootstrapPF:
    """One common hard-resampling bootstrap PF used by Experiment 2."""

    def __init__(
        self,
        adapter: MotionAdapter,
        initial_measurement: np.ndarray,
        n_particles: int,
        rng: np.random.Generator,
        process_std: float,
        measurement_std: float,
        adaptive_gain: float,
        sigma_max: float,
    ):
        self.adapter = adapter
        self.n_particles = int(n_particles)
        self.rng = rng
        self.process_std = float(process_std)
        self.measurement_std = float(measurement_std)
        self.adaptive_gain = float(adaptive_gain)
        self.sigma_max = float(sigma_max)
        initial = np.asarray(initial_measurement, dtype=np.float32)
        initial_particles = initial + rng.normal(
            0.0, measurement_std, size=(self.n_particles, 2)
        )
        self.histories = np.repeat(
            initial_particles[:, None, :], adapter.block_size, axis=1
        ).astype(np.float32)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    @property
    def particles(self) -> np.ndarray:
        return self.histories[:, -1]

    def _systematic_resample(self) -> None:
        positions = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, positions)
        self.histories = self.histories[indices]
        self.weights.fill(1.0 / self.n_particles)

    def step(self, measurement: np.ndarray | None, measurement_age: int) -> np.ndarray:
        self.adapter.set_measurement_age(
            measurement_age, self.adaptive_gain, self.sigma_max
        )
        velocity = self.adapter.propose(self.histories.reshape(self.n_particles, -1))
        next_position = self.particles + self.adapter.dt * velocity
        next_position += self.rng.normal(
            0.0, self.process_std, size=next_position.shape
        )
        self.histories = np.concatenate(
            [self.histories[:, 1:], next_position[:, None, :]], axis=1
        ).astype(np.float32)

        if measurement is not None:
            residual = self.particles - np.asarray(measurement)[None, :]
            log_likelihood = -0.5 * np.sum(residual * residual, axis=1) / (
                self.measurement_std**2
            )
            log_likelihood -= np.max(log_likelihood)
            self.weights *= np.exp(log_likelihood)
            total = self.weights.sum()
            if not np.isfinite(total) or total <= 1e-15:
                self.weights.fill(1.0 / self.n_particles)
            else:
                self.weights /= total

        estimate = np.average(self.particles, axis=0, weights=self.weights)
        ess = 1.0 / np.sum(self.weights**2)
        if ess < 0.5 * self.n_particles:
            self._systematic_resample()
        return estimate.astype(np.float32)


@dataclass(frozen=True)
class OcclusionEpisode:
    road_type: str
    gap_length: int
    episode: int
    seed: int
    branch: int
    truth: np.ndarray
    measurements: np.ndarray
    measurement_mask: np.ndarray
    sensor_positions: np.ndarray
    occlusion_start: int
    reappearance: int


def make_occlusion_episode(
    road_type: str,
    gap_length: int,
    episode: int,
    seed: int,
    measurement_std: float,
    deadline: int,
    streak: int,
    warmup: int = 12,
) -> OcclusionEpisode:
    scenario = {
        "straight": "straight",
        "two_way": "two_way_balanced",
        "three_way": "three_way",
    }[road_type]
    context_seed = int(np.random.SeedSequence([seed, episode, gap_length]).generate_state(1)[0])
    context = make_road_context(scenario, episode, context_seed, history_length=warmup)
    rng = np.random.default_rng(context_seed + 101)
    branch = int(
        rng.choice(len(context.branch_probabilities), p=context.branch_probabilities)
    )
    tail = deadline + streak + 4
    future = ideal_future(context, branch, gap_length + tail)
    truth = np.concatenate([context.history, future], axis=0)
    truth = truth + rng.normal(0.0, 0.0015, size=truth.shape).cumsum(axis=0)
    measurements = truth + rng.normal(0.0, measurement_std, size=truth.shape)
    occlusion_start = warmup
    reappearance = occlusion_start + gap_length
    mask = np.ones(len(truth), dtype=bool)
    mask[occlusion_start:reappearance] = False
    phase = np.linspace(0.0, np.pi, len(truth))
    sensor_positions = context.history[-1] + np.column_stack(
        [0.8 * np.cos(phase), -0.8 + 0.2 * np.sin(phase)]
    )
    return OcclusionEpisode(
        road_type=road_type,
        gap_length=gap_length,
        episode=episode,
        seed=context_seed,
        branch=branch,
        truth=truth.astype(np.float32),
        measurements=measurements.astype(np.float32),
        measurement_mask=mask,
        sensor_positions=sensor_positions.astype(np.float32),
        occlusion_start=occlusion_start,
        reappearance=reappearance,
    )


def recovery_from_errors(
    errors: np.ndarray,
    reappearance: int,
    threshold: float,
    streak: int,
    deadline: int,
) -> tuple[bool, float, float]:
    for latency in range(deadline + 1):
        start = reappearance + latency
        stop = start + streak
        if stop <= len(errors) and np.all(errors[start:stop] < threshold):
            return True, float(latency), float(latency)
    return False, float("nan"), float(deadline + 1)


def evaluate_occlusion_episode(
    adapter: MotionAdapter,
    episode: OcclusionEpisode,
    n_particles: int,
    process_std: float,
    measurement_std: float,
    hit_radius: float,
    recovery_threshold: float,
    recovery_streak: int,
    recovery_deadline: int,
    adaptive_gain: float,
    sigma_max: float,
) -> dict:
    seed_everything(episode.seed)
    rng = np.random.default_rng(
        np.random.SeedSequence([episode.seed, 0x5046])
    )
    pf = ControlledBootstrapPF(
        adapter,
        episode.measurements[0],
        n_particles,
        rng,
        process_std,
        measurement_std,
        adaptive_gain,
        sigma_max,
    )
    estimates = np.empty_like(episode.truth)
    estimates[0] = np.average(pf.particles, axis=0, weights=pf.weights)
    measurement_age = 0
    hit = np.nan
    start_time = time.perf_counter()
    for step in range(1, len(episode.truth)):
        measurement = (
            episode.measurements[step] if episode.measurement_mask[step] else None
        )
        if measurement is None:
            measurement_age += 1
        estimates[step] = pf.step(measurement, measurement_age)
        if step == episode.reappearance - 1 and episode.gap_length > 0:
            hit = float(
                np.min(np.linalg.norm(pf.particles - episode.truth[step], axis=1))
                <= hit_radius
            )
        if measurement is not None:
            measurement_age = 0
    synchronize()
    runtime_ms = 1000.0 * (time.perf_counter() - start_time) / (len(episode.truth) - 1)
    errors = np.linalg.norm(estimates - episode.truth, axis=1)

    if episode.gap_length == 0:
        success = np.nan
        mort = np.nan
        capped = np.nan
        occlusion_rmse = np.nan
    else:
        success_bool, mort, capped = recovery_from_errors(
            errors,
            episode.reappearance,
            recovery_threshold,
            recovery_streak,
            recovery_deadline,
        )
        success = float(success_bool)
        hidden_errors = errors[episode.occlusion_start : episode.reappearance]
        occlusion_rmse = float(np.sqrt(np.mean(hidden_errors**2)))

    return {
        "method": adapter.name,
        "road_type": episode.road_type,
        "gap_length": episode.gap_length,
        "episode": episode.episode,
        "seed": episode.seed,
        "realized_branch": episode.branch,
        "particle_hit": hit,
        "recovery_success": success,
        "mort_success_steps": mort,
        "failure_aware_latency": capped,
        "occlusion_rmse": occlusion_rmse,
        "rmse_all": float(np.sqrt(np.mean(errors**2))),
        "runtime_ms_per_update": runtime_ms,
    }


def run_controlled_occlusion_grid(
    adapters: dict[str, MotionAdapter],
    *,
    road_types: Sequence[str],
    gap_lengths: Sequence[int],
    episodes_per_cell: int,
    seed: int,
    n_particles: int,
    process_std: float,
    measurement_std: float,
    hit_radius: float,
    recovery_threshold: float,
    recovery_streak: int,
    recovery_deadline: int,
    adaptive_gain: float,
    sigma_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[OcclusionEpisode]]:
    from mml_research.experiments.experiment_runner import ControlledExperimentRunner, OcclusionExperimentConfig

    config = OcclusionExperimentConfig(
        road_types=road_types,
        gap_lengths=gap_lengths,
        episodes_per_cell=episodes_per_cell,
        seed=seed,
        n_particles=n_particles,
        process_std=process_std,
        measurement_std=measurement_std,
        hit_radius=hit_radius,
        recovery_threshold=recovery_threshold,
        recovery_streak=recovery_streak,
        recovery_deadline=recovery_deadline,
        adaptive_gain=adaptive_gain,
        sigma_max=sigma_max,
    )
    return ControlledExperimentRunner(
        adapters, config,
        episode_factory=make_occlusion_episode,
        episode_evaluator=evaluate_occlusion_episode,
    ).run()


def summarize_occlusion(frame: pd.DataFrame) -> pd.DataFrame:
    return confidence_summary(
        frame,
        ["method", "road_type", "gap_length"],
        [
            "particle_hit",
            "recovery_success",
            "failure_aware_latency",
            "mort_success_steps",
            "occlusion_rmse",
            "rmse_all",
            "runtime_ms_per_update",
        ],
    ).rename(
        columns={
            "particle_hit_mean": "particle_hit_rate",
            "recovery_success_mean": "trr",
            "mort_success_steps_mean": "mort_success_only_steps",
        }
    )


def parse_int_list(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated integer.")
    return values


def ensure_output_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_protocol(path: Path, title: str, lines: Iterable[str]) -> None:
    path.write_text("\n".join([f"# {title}", "", *lines, ""]), encoding="ascii")
