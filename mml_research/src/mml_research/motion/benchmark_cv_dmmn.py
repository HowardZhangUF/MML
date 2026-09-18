#!/usr/bin/env python3
"""
benchmark_cv_dmmn.py
====================

Append the two missing benchmark baselines to the existing filter-level
benchmark CSV without rerunning the completed PF+CFM / PF+NF / DiffPF rows.

Trackers
--------
PF+CV:
    Bootstrap particle filter with a clipped constant-velocity proposal.

PF+DMMN:
    Bootstrap particle filter using the offline ScratchTransformer velocity
    checkpoint:
        mml_network/models/best_noisy_5_v2_velocities_occlusions_ScratchTransformer.pth

Both trackers use the same BootstrapPFTracker, trajectory seeds, measurement
noise seeds, particle count, metrics, and per-run CSV schema as
benchmark_diffpf_vs_cfm.py.

Usage (from this directory or repo root):
    python -m mml_research.motion.benchmark_cv_dmmn --n_runs 100 --n_steps 2000
"""

from __future__ import annotations

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import argparse
import csv
import math
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np

# This workstation exposes an older CUDA GPU that the installed PyTorch build
# cannot run on. Keep this benchmark on CPU unless the caller explicitly set a
# device policy before launching the process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch


from mml_research.motion.benchmark_diffpf_vs_cfm import (  # noqa: E402
    BootstrapPFTracker,
    recovery_times,
    run_tracker,
    summarize,
)
from mml_research.motion.generate_synthetic_data import DT, NOISE_STD  # noqa: E402
from mml_research.motion.train_diffpf import simulate_full_trajectory  # noqa: E402


MAX_SPEED = 1.5
CSV_HEADER = [
    "tracker",
    "run",
    "rmse_all",
    "rmse_visible",
    "rmse_occluded",
    "recovery_steps",
    "ms_per_step",
]


class ConstantVelocityMotionModel:
    """Zero-parameter constant-velocity proposal for the bootstrap PF."""

    def __init__(self, block_size: int = 5, input_dim: int = 2, dt: float = DT):
        self.block_size = int(block_size)
        self.input_dim = int(input_dim)
        self.dt = float(dt)

    def propose(self, c_flat: np.ndarray) -> np.ndarray:
        c_flat = np.asarray(c_flat, dtype=np.float32)
        expected = self.block_size * self.input_dim
        if c_flat.ndim != 2 or c_flat.shape[1] != expected:
            raise ValueError(
                f"PF+CV context must have shape (N, {expected}), got {c_flat.shape}."
            )

        hist = c_flat.reshape(-1, self.block_size, self.input_dim)
        v = (hist[:, -1, :2] - hist[:, -2, :2]) / self.dt
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * np.minimum(1.0, MAX_SPEED / (speed + 1e-9))
        return v.astype(np.float32)


class OfflineDMMNMotionModel:
    """
    Offline DMMN baseline: the saved ScratchTransformer velocity model.

    This deliberately does not use the online MotionModel class. The network
    architecture matches ParticleFilter.py for prediction_method="Transformer".
    """

    def __init__(self, checkpoint_path: str, dt: float = DT):
        self.block_size = 5
        self.input_dim = 2
        self.dt = float(dt)
        self.checkpoint_path = os.path.abspath(checkpoint_path)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"DMMN checkpoint not found: {self.checkpoint_path}")

        state = torch.load(self.checkpoint_path, map_location="cpu")
        self.params = {
            key: value.detach().cpu().numpy().astype(np.float32)
            for key, value in state.items()
        }
        self.tparams = {
            key: value.detach().cpu().to(dtype=torch.float32)
            for key, value in state.items()
        }
        self.n_embed = 20
        self.n_head = 8
        self.head_size = self.n_embed // self.n_head
        self.n_layer = 2
        self._attention_weights = {}
        self._torch_attention_weights = {}
        for block_idx in range(self.n_layer):
            self._attention_weights[block_idx] = {
                kind: np.stack(
                    [
                        self.params[f"blocks.{block_idx}.sa.heads.{head_idx}.{kind}.weight"]
                        for head_idx in range(self.n_head)
                    ],
                    axis=0,
                )
                for kind in ("key", "query", "value")
            }
            self._torch_attention_weights[block_idx] = {
                kind: torch.stack(
                    [
                        self.tparams[f"blocks.{block_idx}.sa.heads.{head_idx}.{kind}.weight"]
                        for head_idx in range(self.n_head)
                    ],
                    dim=0,
                )
                for kind in ("key", "query", "value")
            }

    @staticmethod
    def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        y = x @ weight.T
        if bias is not None:
            y = y + bias
        return y

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * weight + bias

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=axis, keepdims=True)

    def _attention(self, x: np.ndarray, block_idx: int) -> np.ndarray:
        scale = self.head_size ** -0.5
        mask = self.params[f"blocks.{block_idx}.sa.heads.0.tril"][: x.shape[1], : x.shape[1]]
        weights_by_kind = self._attention_weights[block_idx]
        k = np.einsum("bte,hde->bhtd", x, weights_by_kind["key"], optimize=True)
        q = np.einsum("bte,hde->bhtd", x, weights_by_kind["query"], optimize=True)
        v = np.einsum("bte,hde->bhtd", x, weights_by_kind["value"], optimize=True)
        scores = np.einsum("bhtd,bhsd->bhts", q, k, optimize=True) * scale
        scores = np.where(mask[None, None, :, :] == 0, -np.inf, scores)
        att = self._softmax(scores, axis=-1)
        out = np.einsum("bhts,bhsd->bhtd", att, v, optimize=True)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self._linear(
            out,
            self.params[f"blocks.{block_idx}.sa.proj.weight"],
            self.params[f"blocks.{block_idx}.sa.proj.bias"],
        )

    def _feed_forward(self, x: np.ndarray, block_idx: int) -> np.ndarray:
        x = self._linear(
            x,
            self.params[f"blocks.{block_idx}.ffwd.net.0.weight"],
            self.params[f"blocks.{block_idx}.ffwd.net.0.bias"],
        )
        x = self._relu(x)
        return self._linear(
            x,
            self.params[f"blocks.{block_idx}.ffwd.net.2.weight"],
            self.params[f"blocks.{block_idx}.ffwd.net.2.bias"],
        )

    def _forward_numpy(self, c_flat: np.ndarray) -> np.ndarray:
        x = c_flat.reshape(-1, self.block_size, self.input_dim)
        x = self._linear(x, self.params["embed.weight"], self.params["embed.bias"])
        x = x + self.params["position_embedding_table.weight"][None, :, :]

        for block_idx in range(self.n_layer):
            ln1 = self._layer_norm(
                x,
                self.params[f"blocks.{block_idx}.ln1.weight"],
                self.params[f"blocks.{block_idx}.ln1.bias"],
            )
            x = x + self._attention(ln1, block_idx)
            ln2 = self._layer_norm(
                x,
                self.params[f"blocks.{block_idx}.ln2.weight"],
                self.params[f"blocks.{block_idx}.ln2.bias"],
            )
            x = x + self._feed_forward(ln2, block_idx)

        x = self._layer_norm(x, self.params["ln_f.weight"], self.params["ln_f.bias"])
        x = self._linear(x, self.params["lm_hidden.weight"], self.params["lm_hidden.bias"])
        x = self._relu(x)
        x = self._linear(x, self.params["lm_head.weight"], self.params["lm_head.bias"])
        return x[:, -1, :].astype(np.float32)

    @staticmethod
    def _linear_torch(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = x @ weight.T
        if bias is not None:
            y = y + bias
        return y

    @staticmethod
    def _layer_norm_torch(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + eps) * weight + bias

    def _attention_torch(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        scale = self.head_size ** -0.5
        mask = self.tparams[f"blocks.{block_idx}.sa.heads.0.tril"][: x.shape[1], : x.shape[1]]
        weights_by_kind = self._torch_attention_weights[block_idx]
        k = torch.einsum("bte,hde->bhtd", x, weights_by_kind["key"])
        q = torch.einsum("bte,hde->bhtd", x, weights_by_kind["query"])
        v = torch.einsum("bte,hde->bhtd", x, weights_by_kind["value"])
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
        scores = scores.masked_fill(mask[None, None, :, :] == 0, float("-inf"))
        att = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhts,bhsd->bhtd", att, v)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self._linear_torch(
            out,
            self.tparams[f"blocks.{block_idx}.sa.proj.weight"],
            self.tparams[f"blocks.{block_idx}.sa.proj.bias"],
        )

    def _feed_forward_torch(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        x = self._linear_torch(
            x,
            self.tparams[f"blocks.{block_idx}.ffwd.net.0.weight"],
            self.tparams[f"blocks.{block_idx}.ffwd.net.0.bias"],
        )
        x = torch.relu(x)
        return self._linear_torch(
            x,
            self.tparams[f"blocks.{block_idx}.ffwd.net.2.weight"],
            self.tparams[f"blocks.{block_idx}.ffwd.net.2.bias"],
        )

    @torch.inference_mode()
    def _forward_torch(self, c_flat: np.ndarray) -> np.ndarray:
        x_np = np.ascontiguousarray(
            c_flat.reshape(-1, self.block_size, self.input_dim),
            dtype=np.float32,
        )
        x = torch.from_numpy(x_np)
        x = self._linear_torch(x, self.tparams["embed.weight"], self.tparams["embed.bias"])
        x = x + self.tparams["position_embedding_table.weight"][None, :, :]

        for block_idx in range(self.n_layer):
            ln1 = self._layer_norm_torch(
                x,
                self.tparams[f"blocks.{block_idx}.ln1.weight"],
                self.tparams[f"blocks.{block_idx}.ln1.bias"],
            )
            x = x + self._attention_torch(ln1, block_idx)
            ln2 = self._layer_norm_torch(
                x,
                self.tparams[f"blocks.{block_idx}.ln2.weight"],
                self.tparams[f"blocks.{block_idx}.ln2.bias"],
            )
            x = x + self._feed_forward_torch(ln2, block_idx)

        x = self._layer_norm_torch(x, self.tparams["ln_f.weight"], self.tparams["ln_f.bias"])
        x = self._linear_torch(x, self.tparams["lm_hidden.weight"], self.tparams["lm_hidden.bias"])
        x = torch.relu(x)
        x = self._linear_torch(x, self.tparams["lm_head.weight"], self.tparams["lm_head.bias"])
        return x[:, -1, :].cpu().numpy().astype(np.float32)

    def propose(self, c_flat: np.ndarray) -> np.ndarray:
        c_flat = np.asarray(c_flat, dtype=np.float32)
        expected = self.block_size * self.input_dim
        if c_flat.ndim != 2 or c_flat.shape[1] != expected:
            raise ValueError(
                f"PF+DMMN context must have shape (N, {expected}), got {c_flat.shape}."
            )

        v = self._forward_torch(c_flat)
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * np.minimum(1.0, MAX_SPEED / (speed + 1e-9))
        return v


def _finite_or_blank(value) -> str:
    if value is None:
        return ""
    try:
        if not np.isfinite(value):
            return ""
    except TypeError:
        return ""
    return f"{float(value):.6f}"


def _align_per_run_arrays(
    stats: dict,
    all_errors: List[np.ndarray],
    all_vis: List[np.ndarray],
    all_ms: List[float],
    all_recovery: List[list],
) -> dict:
    """Keep CSV arrays aligned by run, using None for unavailable metrics."""
    stats["_per_run_all"] = [
        float(np.sqrt(np.mean(errors ** 2))) for errors in all_errors
    ]
    stats["_per_run_vis"] = [
        float(np.sqrt(np.mean(errors[vis_mask] ** 2))) if vis_mask.any() else None
        for errors, vis_mask in zip(all_errors, all_vis)
    ]
    stats["_per_run_occ"] = [
        float(np.sqrt(np.mean(errors[~vis_mask] ** 2))) if (~vis_mask).any() else None
        for errors, vis_mask in zip(all_errors, all_vis)
    ]
    stats["_per_run_rec"] = [
        float(np.mean(recovery)) if len(recovery) else None
        for recovery in all_recovery
    ]
    stats["_per_run_ms"] = [float(ms) for ms in all_ms]
    return stats


def _read_existing_rows(csv_path: str) -> List[dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and list(rows[0].keys()) != CSV_HEADER:
        raise ValueError(
            f"Unexpected CSV header in {csv_path}: {list(rows[0].keys())}; "
            f"expected {CSV_HEADER}."
        )
    return rows


def _write_rows(csv_path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def append_stats_to_csv(
    csv_path: str,
    stats: Dict[str, dict],
    tracker_names: List[str],
    replace_existing: bool = False,
) -> None:
    existing = _read_existing_rows(csv_path)
    counts = Counter(row["tracker"] for row in existing)
    present = [name for name in tracker_names if counts.get(name, 0) > 0]

    if present and not replace_existing:
        raise RuntimeError(
            "Refusing to append duplicate tracker rows for "
            f"{present}. Use --replace_existing to replace those rows."
        )

    if replace_existing:
        existing = [row for row in existing if row["tracker"] not in tracker_names]

    new_rows = []
    for name in tracker_names:
        s = stats[name]
        for run in range(s["n_runs"]):
            new_rows.append(
                {
                    "tracker": name,
                    "run": str(run),
                    "rmse_all": _finite_or_blank(s["_per_run_all"][run]),
                    "rmse_visible": _finite_or_blank(s["_per_run_vis"][run]),
                    "rmse_occluded": _finite_or_blank(s["_per_run_occ"][run]),
                    "recovery_steps": _finite_or_blank(s["_per_run_rec"][run]),
                    "ms_per_step": _finite_or_blank(s["_per_run_ms"][run]),
                }
            )

    _write_rows(csv_path, existing + new_rows)
    print(f"\nAppended {len(new_rows)} rows -> {csv_path}")


def _paired_vectors(rows: List[dict], left: str, right: str, metric: str):
    by_tracker = {name: {} for name in (left, right)}
    for row in rows:
        name = row["tracker"]
        if name not in by_tracker:
            continue
        value = row.get(metric, "")
        if value == "":
            continue
        try:
            by_tracker[name][int(row["run"])] = float(value)
        except ValueError:
            continue

    runs = sorted(set(by_tracker[left]) & set(by_tracker[right]))
    x = np.array([by_tracker[left][run] for run in runs], dtype=float)
    y = np.array([by_tracker[right][run] for run in runs], dtype=float)
    return runs, x, y


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction used by the regularized incomplete beta function."""
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = fpmin if abs(d) < fpmin else d
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b), evaluated without an optional SciPy dependency."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _paired_t_pvalue(diff: np.ndarray) -> float:
    """Two-sided paired Student-t p-value for a vector of paired differences."""
    if len(diff) < 2:
        return float("nan")
    std = float(np.std(diff, ddof=1))
    if std == 0.0:
        return 1.0 if float(np.mean(diff)) == 0.0 else 0.0
    t = abs(float(np.mean(diff)) / (std / math.sqrt(len(diff))))
    df = float(len(diff) - 1)
    return _regularized_beta(df / (df + t * t), df / 2.0, 0.5)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _wilcoxon_pvalue(diff: np.ndarray) -> float:
    """Two-sided signed-rank p-value using the tie-corrected normal approximation."""
    nonzero = diff[np.abs(diff) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    ranks = _average_ranks(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0.0].sum())
    mean = n * (n + 1.0) / 4.0
    _, counts = np.unique(np.abs(nonzero), return_counts=True)
    tie_term = float(np.sum(counts * (counts * counts - 1)))
    variance = (n * (n + 1.0) * (2.0 * n + 1.0) - tie_term / 2.0) / 24.0
    if variance <= 0.0:
        return 1.0 if w_plus == mean else 0.0
    z = max(0.0, (abs(w_plus - mean) - 0.5) / math.sqrt(variance))
    return math.erfc(z / math.sqrt(2.0))


def _paired_tests(cfm_as: np.ndarray, other: np.ndarray):
    diff = cfm_as - other
    out = {
        "mean_diff": float(np.mean(diff)),
        "cfm_as_better": int(np.sum(cfm_as < other)),
        "n": int(len(diff)),
        "paired_t_p": _paired_t_pvalue(diff),
        "wilcoxon_p": _wilcoxon_pvalue(diff),
    }
    return out


def print_paired_report(csv_path: str, tracker_names: List[str]) -> None:
    rows = _read_existing_rows(csv_path)
    print("\n--- Paired tests vs PF+CFM-AS (diff = CFM-AS - tracker; lower is better) ---")
    print(
        f"{'comparison':<16}{'metric':<16}{'mean diff':>12}"
        f"{'CFM-AS better':>16}{'paired-t p':>14}{'Wilcoxon p':>14}"
    )
    for tracker in tracker_names:
        for metric in ["rmse_all", "rmse_visible", "rmse_occluded", "recovery_steps"]:
            _, cfm_as, other = _paired_vectors(rows, "PF+CFM-AS", tracker, metric)
            if len(cfm_as) == 0:
                continue
            test = _paired_tests(cfm_as, other)
            print(
                f"{('vs ' + tracker):<16}{metric:<16}"
                f"{test['mean_diff']:>12.4f}"
                f"{test['cfm_as_better']:>9}/{test['n']:<6}"
                f"{test['paired_t_p']:>14.3g}"
                f"{test['wilcoxon_p']:>14.3g}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Append PF+CV and offline PF+DMMN benchmarks")
    parser.add_argument("--n_runs", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--pf_particles", type=int, default=500)
    parser.add_argument("--process_noise", type=float, default=0.01)
    parser.add_argument("--meas_sigma", type=float, default=0.05)
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=4,
        help="CPU threads for the fused Torch DMMN forward path.",
    )
    parser.add_argument("--replace_existing", action="store_true")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Per-run CSV to append. Defaults to ../../models/diffpf_vs_cfm_per_run.csv.",
    )
    parser.add_argument(
        "--dmmn_ckpt",
        type=str,
        default=None,
        help="Offline ScratchTransformer checkpoint path. Defaults to the repo checkpoint.",
    )
    parser.add_argument("--models-dir", default=None, help="Override the models dir")
    args = parser.parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    this_dir = str(motion_resource_dir() / "mml_guidance/mml_network")
    models_dir = args.models_dir or os.path.normpath(os.path.join(this_dir, '..', '..', 'models'))
    csv_path = args.csv_path or os.path.join(models_dir, "diffpf_vs_cfm_per_run.csv")
    dmmn_path = args.dmmn_ckpt or os.path.join(
        this_dir,
        "models",
        "best_noisy_5_v2_velocities_occlusions_ScratchTransformer.pth",
    )

    tracker_names = ["PF+CV", "PF+DMMN"]
    existing = _read_existing_rows(csv_path)
    counts = Counter(row["tracker"] for row in existing)
    present = [name for name in tracker_names if counts.get(name, 0) > 0]
    if present and not args.replace_existing:
        raise RuntimeError(
            f"{csv_path} already contains rows for {present}. "
            "Use --replace_existing to replace them."
        )

    print("=" * 70)
    print("Appending filter-level baselines: PF+CV and offline PF+DMMN")
    print("=" * 70)
    print(f"Runs       : {args.n_runs} x {args.n_steps} steps")
    print(f"Particles  : {args.pf_particles}")
    print(f"Torch CPU threads: {torch.get_num_threads()}")
    print(f"DMMN ckpt  : {os.path.abspath(dmmn_path)}")
    print(f"CSV target : {csv_path}")

    cv = ConstantVelocityMotionModel(block_size=5, dt=DT)
    dmmn = OfflineDMMNMotionModel(dmmn_path, dt=DT)

    results = {name: ([], [], [], []) for name in tracker_names}

    for run in range(args.n_runs):
        rng = np.random.default_rng(1000 + run)
        start_node = run % 8
        positions, visible = simulate_full_trajectory(start_node, args.n_steps, rng)

        for name, tracker in [
            (
                "PF+CV",
                BootstrapPFTracker(
                    cv,
                    n_particles=args.pf_particles,
                    process_noise=args.process_noise,
                    meas_sigma=args.meas_sigma,
                ),
            ),
            (
                "PF+DMMN",
                BootstrapPFTracker(
                    dmmn,
                    n_particles=args.pf_particles,
                    process_noise=args.process_noise,
                    meas_sigma=args.meas_sigma,
                ),
            ),
        ]:
            meas_rng = np.random.default_rng(5000 + run)
            np.random.seed(9000 + run)
            torch.manual_seed(9000 + run)
            errors, vis_mask, ms = run_tracker(tracker, positions, visible, meas_rng)
            rec = recovery_times(errors, vis_mask)

            results[name][0].append(errors)
            results[name][1].append(vis_mask)
            results[name][2].append(ms)
            results[name][3].append(rec)

        print(f"  run {run + 1}/{args.n_runs} done "
              f"({100.0 * (1 - visible.mean()):.1f}% occluded)")

    print("\n--- Aggregate results ---")
    stats = {}
    for name, (all_e, all_v, all_ms, all_rec) in results.items():
        stats[name] = _align_per_run_arrays(
            summarize(name, all_e, all_v, all_ms, all_rec),
            all_e,
            all_v,
            all_ms,
            all_rec,
        )

    append_stats_to_csv(
        csv_path,
        stats,
        tracker_names,
        replace_existing=args.replace_existing,
    )
    print_paired_report(csv_path, tracker_names)

    print("\n" + "=" * 78)
    print(f"SUMMARY  ({args.n_runs} runs x {args.n_steps} steps, mean +/- run-to-run std)")
    print("=" * 78)
    print(f"{'tracker':<11}{'RMSE all':>15}{'RMSE vis':>15}"
          f"{'RMSE occ':>15}{'recovery':>13}{'ms/step':>13}")
    for name in tracker_names:
        s = stats[name]
        print(
            f"{name:<11}"
            f"{s['rmse_all_mean']:>8.3f}+-{s['rmse_all_std']:<5.3f}"
            f"{s['rmse_visible_mean']:>8.3f}+-{s['rmse_visible_std']:<5.3f}"
            f"{s['rmse_occluded_mean']:>8.3f}+-{s['rmse_occluded_std']:<5.3f}"
            f"{s['recovery_mean']:>7.2f}+-{s['recovery_std']:<5.2f}"
            f"{s['ms_per_step_mean']:>7.2f}+-{s['ms_per_step_std']:<5.2f}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
