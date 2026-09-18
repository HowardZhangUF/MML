"""Controlled experiment scheduling; filter mechanics stay in their own module."""
from __future__ import annotations

from mml_core.config import project_root, motion_resource_dir, research_data_dir

from dataclasses import dataclass
from typing import Sequence
import pandas as pd


@dataclass(frozen=True)
class OcclusionExperimentConfig:
    """The parameters defining one ordered, paired experiment grid."""
    road_types: Sequence[str]
    gap_lengths: Sequence[int]
    episodes_per_cell: int
    seed: int
    n_particles: int
    process_std: float
    measurement_std: float
    hit_radius: float
    recovery_threshold: float
    recovery_streak: int
    recovery_deadline: int
    adaptive_gain: float
    sigma_max: float


class ControlledExperimentRunner:
    """Build paired episodes once, then evaluate adapters in their input order."""

    def __init__(self, adapters, config, *, episode_factory, episode_evaluator):
        self.adapters = adapters
        self.config = config
        self.episode_factory = episode_factory
        self.episode_evaluator = episode_evaluator

    def run(self):
        episodes = self.build_episodes()
        manifest = self.build_manifest(episodes)
        results = self.evaluate(episodes)
        return results, manifest, episodes

    def build_episodes(self):
        episodes = [
            self.episode_factory(
                road,
                gap,
                episode,
                self.config.seed,
                self.config.measurement_std,
                self.config.recovery_deadline,
                self.config.recovery_streak,
            )
            for road in self.config.road_types
            for gap in self.config.gap_lengths
            for episode in range(self.config.episodes_per_cell)
        ]
        return episodes

    @staticmethod
    def build_manifest(episodes):
        manifest = pd.DataFrame(
            [
                {
                    "road_type": item.road_type,
                    "gap_length": item.gap_length,
                    "episode": item.episode,
                    "seed": item.seed,
                    "realized_branch": item.branch,
                    "occlusion_start": item.occlusion_start,
                    "reappearance": item.reappearance,
                    "scheduled_recovery_event": int(item.gap_length > 0),
                    "sensor_path": "prerecorded_deterministic_arc",
                }
                for item in episodes
            ]
        )
        return manifest

    def evaluate(self, episodes):
        rows = []
        for index, episode in enumerate(episodes, start=1):
            if index == 1 or index % max(1, len(episodes) // 20) == 0:
                print(f"Controlled episode {index}/{len(episodes)}")
            for adapter in self.adapters.values():
                rows.append(
                    self.episode_evaluator(
                        adapter,
                        episode,
                        self.config.n_particles,
                        self.config.process_std,
                        self.config.measurement_std,
                        self.config.hit_radius,
                        self.config.recovery_threshold,
                        self.config.recovery_streak,
                        self.config.recovery_deadline,
                        self.config.adaptive_gain,
                        self.config.sigma_max,
                    )
                )
        return pd.DataFrame(rows)
