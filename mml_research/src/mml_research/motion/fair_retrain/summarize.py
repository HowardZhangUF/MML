#!/usr/bin/env python3
"""
summarize.py — fair_v1 test tables, paired differences and the overview figure.

Statistics are cfm_variants.summarize's (per-episode metrics averaged over the
three training seeds first; 95% whole-episode bootstrap CIs). Paired
comparisons answer the study's questions:
    R1/X − R0/X            the recipe effect for every architecture
    r/X − r/CFM            architecture vs CFM on the same recipe
    r/M4 − r/DMMN, r/DMMN+Gauss − r/DMMN
    M1 feature ladder      mask / time / cov against M1-control (R1)

    python -m mml_research.motion.fair_retrain.summarize
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mml_research.motion.cfm_variants.summarize import (
    GRAY, INK, INK2, SERIES, _style, fmt_paired, fmt_table, paired, summarize,
)
from mml_research.motion.fair_retrain.common import (
    CONFIG_PATH, DEFAULT_OUT_DIR, MODEL_LABEL, all_methods, load_config, method_name,
)

RECIPE_COLOR = {"R0": SERIES[0], "R1": SERIES[1]}
RECIPE_LABEL = {"R0": "R0: measurement histories, noisy labels (original recipe)",
                "R1": "R1: PF-rollout histories, true-velocity labels (M1 recipe)"}


def comparisons(cfg):
    out = []
    both = [m for m in cfg["models"]["R0"] if m in cfg["models"]["R1"]]
    out += [(method_name("R1", m), method_name("R0", m)) for m in both]
    for r in ("R0", "R1"):
        ms = cfg["models"][r]
        out += [(method_name(r, m), method_name(r, "cfm")) for m in ("nf", "dmmn", "trajflow", "moflow", "tcfm")
                if m in ms]
        out += [(method_name(r, m), method_name(r, "dmmn")) for m in ("dmmn_gauss", "m4") if m in ms]
    m1 = lambda m: method_name("R1", m)
    out += [(m1("m1_control"), m1("cfm")), (m1("m1_mask"), m1("m1_control")), (m1("m1_time"), m1("m1_mask")),
            (m1("m1_full"), m1("m1_time")), (m1("m1_full"), m1("m1_control"))]
    return out


def plot_overview(s, cfg, lane, path):
    """One row per model, R0 / R1 bars side by side; training-free methods in gray."""
    models = [m for m in MODEL_LABEL if any(m in cfg["models"][r] for r in ("R0", "R1"))]
    rows = [MODEL_LABEL[m] for m in models] + list(cfg["training_free"])
    s = s.set_index("method")
    panels = [
        ("rmse_all_mean", "RMSE all steps (m) ↓", ("rmse_all_ci_lo", "rmse_all_ci_hi"), "{:.3f}"),
        ("rmse_occluded_mean", "RMSE occluded (m) ↓", ("rmse_occluded_ci_lo", "rmse_occluded_ci_hi"), "{:.3f}"),
        ("exit_success_rate", "Exit success, top-1 ↑", None, "{:.0%}"),
        ("p_correct", "P(correct exit) ↑", ("p_correct_ci_lo", "p_correct_ci_hi"), "{:.2f}"),
        ("brier", "Exit Brier score ↓", None, "{:.2f}"),
        ("ms_per_step_mean", "Runtime (ms / step, log) ↓", None, "{:.0f}"),
    ]
    bars = []                      # (y, method, color)
    for i, label in enumerate(rows):
        y0 = len(rows) - 1 - i
        if label in cfg["training_free"]:
            bars.append((y0, label, GRAY))
            continue
        for j, r in enumerate(("R0", "R1")):
            name = f"{r}/{label}"
            if name in s.index:
                bars.append((y0 + (0.2 if j == 0 else -0.2), name, RECIPE_COLOR[r]))
    fig, axes = plt.subplots(2, 3, figsize=(16, 3 + 0.55 * len(rows)), sharey=True)
    for ax, (key, title, ci, fmt) in zip(axes.flat, panels):
        ys = np.array([b[0] for b in bars])
        vals = np.array([s.loc[b[1], key] for b in bars], float)
        ax.barh(ys, vals, height=0.38, color=[b[2] for b in bars], edgecolor="white", linewidth=1)
        ends = vals
        if ci:
            lo = np.array([s.loc[b[1], ci[0]] for b in bars], float)
            hi = np.array([s.loc[b[1], ci[1]] for b in bars], float)
            ax.errorbar(vals, ys, xerr=[vals - lo, hi - vals], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
            ends = np.fmax(vals, hi)
        for y, v, e in zip(ys, vals, ends):
            ax.text(e, y, " " + fmt.format(v), va="center", fontsize=6.5, color=INK2)
        if key == "exit_success_rate":
            ref = lane[0] / max(lane[1], 1)
            ax.axvline(ref, color=INK, lw=1, ls="--")
            ax.text(ref, len(rows) - 0.4, f"lane cue {ref:.0%}", fontsize=7, color=INK, ha="right")
            ends = np.append(ends, ref)
        if key == "ms_per_step_mean":
            ax.set_xscale("log")
            ax.set_xlim(np.nanmin(vals) / 2, np.nanmax(vals) * 4)
        else:
            ax.set_xlim(0, np.nanmax(ends) * 1.22)
        ax.set_title(title, fontsize=10)
        _style(ax)
        ax.grid(axis="y", visible=False)
    for ax in axes[:, 0]:
        ax.set_yticks(np.arange(len(rows))[::-1])
        ax.set_yticklabels(rows, fontsize=8, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RECIPE_COLOR[r]) for r in ("R0", "R1")]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=GRAY))
    fig.legend(handles, [RECIPE_LABEL["R0"], RECIPE_LABEL["R1"], "training-free (no training data)"],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    n_ep, n_ev = int(s["episodes"].iloc[0]), int(s["exit_events"].iloc[0])
    fig.suptitle(f"Fair retraining — every model on the same rows, budget, seeds and selection rules; locked test, "
                 f"{n_ep} episodes, {n_ev} exit events (mean of 3 seeds; whiskers: 95% episode-bootstrap CI)",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Summarize the fair_v1 test split")
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--config", default=CONFIG_PATH)
    args = p.parse_args()
    cfg = load_config(args.config)
    d = os.path.join(args.out_dir, "test")
    ep = pd.read_csv(os.path.join(d, "episodes.csv"))
    ev = pd.read_csv(os.path.join(d, "events.csv"))
    for df in (ep, ev):
        df["seed"] = df["seed"].fillna(-1).astype(int)
    rng = np.random.default_rng(0)
    n_boot = cfg["variants"]["evaluation"]["bootstrap_resamples"]
    order = all_methods(cfg)
    s, by_seed, t = summarize(ep, ev, n_boot, rng, order)
    pr = paired(t, ev.assign(seed=ev.seed.replace(-1, 0)), n_boot, rng, comparisons(cfg))
    e0 = ev[(ev.method == s.method.iloc[0]) & ev.seed.isin([-1, 0])]
    lane = (int((e0.lane_side == e0.actual_side).sum()), len(e0))
    s.to_csv(os.path.join(d, "summary.csv"), index=False)
    by_seed.to_csv(os.path.join(d, "summary_by_seed.csv"), index=False)
    pr.to_csv(os.path.join(d, "paired.csv"), index=False)
    md = (fmt_table(s, lane) + "\n\nPaired differences (method − control, mean [95% episode-bootstrap CI]):\n\n"
          + fmt_paired(pr))
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write(md + "\n")
    os.makedirs(os.path.join(d, "plots"), exist_ok=True)
    plot_overview(s, cfg, lane, os.path.join(d, "plots", "fair_overview.png"))
    print(md)
    print(f"\nwritten to {d}")


if __name__ == "__main__":
    main()
