#!/usr/bin/env python3
"""
plot_benchmark.py
=================
Figures for every exit-prediction benchmark table, all trackers, with the
same statistics as make_benchmark_tables.py (mean +- 95% CI: t-interval over
runs for RMSE / runtime, run-level bootstrap for exit metrics).

Writes into <root>/figures/ (PDF for the paper, PNG for viewing):
    benchmark_tuned       results/            per-tracker tuned process noise
    benchmark_shared      results_noise0.01/  every tracker at q = 0.01 m
    tradeoff              RMSE all vs P(correct exit), both benchmarks
    noise_sweep           validation sweep (runs 100-119): RMSE, exit >50%, P(correct)
    enff_sweep            EnFF lambda x sigma_min selection at its chosen q

Colour encodes the tracker group (CFM variants / earlier baselines /
open-source flow-matching ports); every tracker is named on an axis.

Usage:
    python -m mml_research.motion.plot_benchmark --root <exit_benchmark dir>
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import matplotlib.transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from mml_core.config import motion_resource_dir
from mml_research.motion.make_benchmark_tables import TRACKERS, benchmark_columns, pooled_ci, t_ci

# Reference palette, categorical slots 1-3 (validated all-pairs, light mode).
GROUP_COLOR = {"CFM": "#2a78d6", "baseline": "#eb6834", "open-source": "#1baf7a"}
GROUP_LABEL = {"CFM": "CFM variants (ours)", "baseline": "Earlier baselines",
               "open-source": "Open-source flow-matching ports"}
GROUP = {**{t: "CFM" for t in ("PF+CFM", "PF+CFM-AS", "PF+CFM-LM", "PF+CFM-P", "PF+CFM-Pnaive")},
         **{t: "baseline" for t in ("PF+NF", "DiffPF", "PF+CV", "PF+DMMN")},
         **{t: "open-source" for t in ("PF+TrajFlow", "PF+MoFlow", "PF+TCFM", "EnFF")}}
SHORT = {"PF+CFM": "PF+CFM (σ=1.0)", "PF+CFM-AS": "PF+CFM-AS", "PF+CFM-LM": "PF+CFM-LM",
         "PF+CFM-P": "PF+CFM-P", "PF+CFM-Pnaive": "PF+CFM-P (uncorr.)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])

METRICS = [  # key, title, percent, log-x, reference lines
    ("rmse_all", "RMSE all (m)", False, True, []),
    ("rmse_visible", "RMSE visible (m)", False, True, []),
    ("rmse_occluded", "RMSE occluded (m)", False, True, []),
    ("top1", "Exit top-1 (%)", True, False, [91.0, 50.0]),
    ("majority", "Exit >50% mass (%)", True, False, [91.0, 50.0]),
    ("p_correct", "P(correct exit)", False, False, []),
    ("ms_per_step", "Runtime (ms/step)", False, True, []),
]


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.size": 0, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 200,
    })


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


REF_STYLE = {91.0: ((0, (4, 2)), "lane-cue ceiling (91%)"), 50.0: ((0, (1, 1.5)), "chance (50%)")}


def legend_handles(refs=False):
    h = [Line2D([], [], marker="o", ls="", ms=6, color=c, label=GROUP_LABEL[g]) for g, c in GROUP_COLOR.items()]
    if refs:
        h += [Line2D([], [], color=INK2, lw=0.9, ls=ls, label=lab) for ls, lab in REF_STYLE.values()]
    return h


def plain_log_axis(ax, subs=(1, 2, 5)):
    """Log x axis with plain decimal labels (no colliding 10^x minor labels)."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=6.5)


def dot_panels(cols, trackers, title, noise=None):
    """One panel per metric; trackers on the shared y axis; dot = mean, bar = 95% CI."""
    n = len(trackers)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 0.28 * n + 1.6), sharey=True)
    y = np.arange(n)[::-1]
    for ax, (key, label, pct, logx, refs) in zip(axes, METRICS):
        for yi, t in zip(y, trackers):
            m, h = cols[t][key]
            if pct:
                m, h = 100 * m, 100 * h
            c = GROUP_COLOR[GROUP[t]]
            ax.errorbar(m, yi, xerr=h, fmt="o", ms=4.5, color=c, ecolor=c, elinewidth=1.4, capsize=0,
                        mec="white", mew=0.8, zorder=3)
        for v in refs:
            ax.axvline(v, color=INK2, lw=0.8, ls=REF_STYLE[v][0], zorder=1)
        if logx:
            plain_log_axis(ax, subs=(1,) if key == "ms_per_step" else (1, 2, 5))
        if pct:
            ax.set_xlim(-3, 100)
        ax.set_title(label, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        # faint separators between tracker groups
        for i in range(1, n):
            if GROUP[trackers[i]] != GROUP[trackers[i - 1]]:
                ax.axhline(y[i] + 0.5, color=GRID, lw=0.8)
    labels = [SHORT.get(t, t) + (f"  q={noise[t]:g}" if noise and noise.get(t) is not None else "")
              for t in trackers]
    axes[0].set_yticks(y, labels)
    axes[0].set_ylim(-0.7, n - 0.3)
    fig.suptitle(title, x=0.01, y=0.995, ha="left", color=INK, fontsize=10)
    fig.legend(handles=legend_handles(refs=True), loc="upper left", ncol=5, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.005, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def tradeoff(tuned, shared, trackers):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (cols, title) in zip(axes, [(tuned, "Tuned process noise"), (shared, "Shared q = 0.01 m")]):
        pts = []
        for t in trackers:
            if t not in cols:
                continue
            (x, xh), (yv, yh) = cols[t]["rmse_all"], cols[t]["p_correct"]
            c = GROUP_COLOR[GROUP[t]]
            ax.errorbar(x, yv, xerr=xh, yerr=yh, fmt="o", ms=6, color=c, ecolor=c, elinewidth=1,
                        mec="white", mew=1, zorder=3)
            pts.append([x, yv, [SHORT.get(t, t)]])
        merged = []                                  # points closer than ~1% RMSE and 0.02 P share a label
        for x, yv, names in pts:
            for m in merged:
                if abs(np.log10(x / m[0])) < 0.012 and abs(yv - m[1]) < 0.02:
                    m[2] += names
                    break
            else:
                merged.append([x, yv, names])
        plain_log_axis(ax)
        fig_ = ax.figure
        fig_.canvas.draw()
        rend = fig_.canvas.get_renderer()
        placed = []
        for x, yv, _ in merged:                       # keep labels off every marker
            px, py = ax.transData.transform((x, yv))
            placed.append(matplotlib.transforms.Bbox([[px - 6, py - 6], [px + 6, py + 6]]))
        cands = [(6, 3, "left"), (6, -10, "left"), (-6, 3, "right"), (-6, -10, "right"),
                 (6, 14, "left"), (6, -22, "left"), (-6, 14, "right"), (-6, -22, "right"),
                 (10, 28, "left"), (10, -36, "left"), (-10, 28, "right"), (-10, -36, "right")]
        for x, yv, names in sorted(merged, key=lambda m: -m[1]):
            label = "\n".join(names)
            for k, (dx, dy, ha) in enumerate(cands):
                txt = ax.annotate(label, (x, yv), xytext=(dx, dy), textcoords="offset points", ha=ha,
                                  fontsize=6.5, color=INK, linespacing=1.1,
                                  arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5) if k >= 4 else None)
                bb = txt.get_window_extent(rend).expanded(1.05, 1.35)
                if not any(bb.overlaps(o) for o in placed) or k == len(cands) - 1:
                    placed.append(bb)
                    break
                txt.remove()
        ax.set_xlabel("RMSE all (m, log, lower is better)")
        ax.set_title(title, loc="left", color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("P(correct exit) (higher is better)")
    fig.suptitle("Tracking accuracy vs exit prediction (mean ± 95% CI, runs 0-99)", x=0.01, y=0.995,
                 ha="left", color=INK, fontsize=10)
    fig.legend(handles=legend_handles(), loc="upper left", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.005, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def sweep_heatmaps(runs, chosen):
    grid = sorted(runs.process_noise.unique())
    trackers = [t for t in TRACKERS if t in set(runs.tracker)]
    stats = {"rmse": {}, "maj": {}, "p": {}}
    for t in trackers:
        for q in grid:
            r = runs[(runs.tracker == t) & np.isclose(runs.process_noise, q)]
            stats["rmse"][t, q] = t_ci(r.rmse_all)[0]
            stats["maj"][t, q] = pooled_ci(r.exit_majority, r.exit_events)[0] * 100
            stats["p"][t, q] = pooled_ci(r.exit_p_correct_sum, r.exit_events)[0]
    panels = [("rmse", "RMSE all (m, lower is better)", LogNorm(0.06, 1.3), "{:.3f}"),
              ("maj", "Exit >50% mass (%, higher is better)", Normalize(0, 100), "{:.0f}"),
              ("p", "P(correct exit) (higher is better)", Normalize(0, 1), "{:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.32 * len(trackers) + 1.8), sharey=True)
    for ax, (k, title, norm, fmt) in zip(axes, panels):
        mat = np.array([[stats[k][t, q] for q in grid] for t in trackers])
        cmap = SEQ.reversed() if k == "rmse" else SEQ        # darker = better in every panel
        ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        for i, t in enumerate(trackers):
            for j, q in enumerate(grid):
                v = mat[i, j]
                dark = (norm(v) < 0.45) if k == "rmse" else (norm(v) > 0.55)
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="white" if dark else INK)
                if np.isclose(chosen.get(t, np.nan), q):
                    ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False,
                                               ec="#eb6834", lw=1.8))
        ax.set_xticks(range(len(grid)), [f"{q:g}" for q in grid])
        ax.set_xlabel("process noise q (m per step)")
        ax.set_title(title, loc="left", color=INK)
        ax.spines[:].set_visible(False)
    axes[0].set_yticks(range(len(trackers)), [SHORT.get(t, t) for t in trackers])
    fig.suptitle("Process-noise sweep on validation runs 100-119 (orange box = q chosen by lowest RMSE all; "
                 "EnFF at its selected λ, σ_min)", x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def enff_heatmap(rows):
    best = rows.loc[rows.rmse_all.idxmin()]
    r = rows[np.isclose(rows.process_noise, best.process_noise)]
    lams, smins = sorted(r.enff_lambda.unique()), sorted(r.enff_sigma_min.unique())
    mat = np.array([[r[np.isclose(r.enff_lambda, l) & np.isclose(r.enff_sigma_min, s)].rmse_all.iloc[0]
                     for l in lams] for s in smins])
    norm = Normalize(mat.min(), mat.max())
    fig, ax = plt.subplots(figsize=(6.5, 1.9))
    ax.imshow(mat, cmap=SEQ.reversed(), norm=norm, aspect="auto")
    for i in range(len(smins)):
        for j in range(len(lams)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if norm(mat[i, j]) < 0.45 else INK)
            if np.isclose(mat[i, j], best.rmse_all):
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, ec="#eb6834", lw=1.8))
    ax.set_xticks(range(len(lams)), [f"{l:g}" for l in lams])
    ax.set_yticks(range(len(smins)), [f"σ_min={s:g}" for s in smins])
    ax.set_xlabel("guidance scale λ")
    ax.spines[:].set_visible(False)
    ax.set_title(f"EnFF validation RMSE all (m) at q = {best.process_noise:g} m (orange box = selected)",
                 loc="left", color=INK)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(motion_resource_dir() / "exit_benchmark"))
    args = p.parse_args()
    root = args.root
    out = os.path.join(root, "figures")
    os.makedirs(out, exist_ok=True)
    style()

    read = lambda d: (pd.read_csv(os.path.join(root, d, "exit_benchmark_per_run.csv")),
                      pd.read_csv(os.path.join(root, d, "exit_benchmark_events.csv")))
    tuned_runs, tuned_ev = read("results")
    shared_runs, shared_ev = read("results_noise0.01")
    trackers = [t for t in TRACKERS if t in set(tuned_runs.tracker)]
    shared_tr = [t for t in trackers if t in set(shared_runs.tracker)]
    tuned = benchmark_columns(tuned_runs, tuned_ev, trackers)
    shared = benchmark_columns(shared_runs, shared_ev, shared_tr)
    noise = {t: (None if tuned[t]["noise"] is None or pd.isna(tuned[t]["noise"]) else float(tuned[t]["noise"]))
             for t in trackers}

    print(f"Writing figures to {out}")
    save(dot_panels(tuned, trackers, "Exit-prediction benchmark, per-tracker tuned process noise "
                    "(100 runs × 2000 steps; mean ± 95% CI)", noise), out, "benchmark_tuned")
    save(dot_panels(shared, shared_tr, "Exit-prediction benchmark, shared process noise q = 0.01 m "
                    "(100 runs × 2000 steps; mean ± 95% CI)"), out, "benchmark_shared")
    save(tradeoff(tuned, shared, trackers), out, "tradeoff")

    sweep = pd.read_csv(os.path.join(root, "noise_sweep/noise_sweep_runs.csv"))
    chosen = dict(item.split("=") for item in open(os.path.join(root, "noise_sweep/chosen_noise.txt")).read().split())
    save(sweep_heatmaps(sweep, {k: float(v) for k, v in chosen.items()}), out, "noise_sweep")
    enff = os.path.join(root, "enff_sweep/enff_sweep.csv")
    if os.path.exists(enff):
        save(enff_heatmap(pd.read_csv(enff)), out, "enff_sweep")


if __name__ == "__main__":
    main()
