#!/usr/bin/env python3
"""
summarize.py — aggregate a split's episodes.csv / events.csv (guide 10, 11.3).

Statistics
----------
* Seeded methods: each episode's metric is first averaged over training
  seeds; seed-level aggregates are reported separately (summary_by_seed.csv).
* "±" in summary tables = standard deviation across episodes of the
  per-episode metric (the old table's convention); `*_ci` columns are 95%
  bootstrap intervals of the mean, resampling whole episodes.
* RMSE = mean over episodes of the per-episode RMSE (not a pooled RMSE).
* Exit counts are pooled over events (seeded: averaged over seeds).
* Paired comparisons (paired.csv): per-episode differences method - control,
  events aggregated within their episode first, 95% episode-bootstrap CI.
  `events_disagree` counts events whose top-1 success differs (seed 0), so
  equal totals are not mistaken for equal events.

    python -m mml_research.motion.cfm_variants.summarize --split test
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mml_research.motion.cfm_variants.common import DEFAULT_OUT_DIR, load_config, CONFIG_PATH
from mml_research.motion.cfm_variants.methods import ALL_METHODS, COMPARISONS

EP_METRICS = ["rmse_all", "rmse_visible", "rmse_occluded", "es_mean", "es_occluded", "off_road_mass",
              "ess_pre_mean", "resample_rate", "ms_per_step", "ms_p95", "ms_visible", "ms_occluded"]
EV_METRICS = ["success", "majority", "p_correct", "brier", "exit_nll", "zero_mass", "es_exit", "occ_rmse",
              "recovered"]
PAIRED = ["rmse_all", "rmse_occluded", "es_mean", "success", "p_correct", "brier", "exit_nll"]

# Reference palette (dataviz skill, light mode), fixed order; controls are drawn in gray.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY = "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"
PANELS = [  # (title, control, methods)
    ("M1 Context-CFM", "PF+CFM", ["M1-control", "M1-mask", "M1-time", "M1-full"]),
    ("M2 Calibrated-CFM / AS", "PF+CFM", ["M2-zero", "M2-calibrated", "PF+CFM-AS"]),
    ("M3 Observation-CFM", "PF+CFM", ["M3", "M3-joint"]),
    ("M4 DMMN-CFM", "PF+DMMN", ["DMMN+Gauss", "M4"]),
    ("Open-source ports", "PF+CFM", ["PF+TrajFlow", "PF+MoFlow", "PF+TCFM", "EnFF"]),
]


def boot_ci(x, n_boot, rng):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def episode_table(ep, ev):
    """One row per (method, seed, run): episode metrics + event metrics averaged within the episode."""
    evm = ev.groupby(["method", "seed", "run"])[EV_METRICS].mean().add_prefix("ev_").reset_index()
    t = ep.merge(evm, on=["method", "seed", "run"], how="left")
    return t


def summarize(ep, ev, n_boot, rng, order=None):
    t = episode_table(ep, ev)
    rows, seed_rows = [], []
    for m in [x for x in (order or ALL_METHODS) if x in set(ep.method)]:
        tm = t[t.method == m]
        per_ep = tm.groupby("run").mean(numeric_only=True)          # seed-averaged per episode
        em = ev[ev.method == m]
        n_seeds = tm.seed.nunique()
        r = {"method": m, "seeds": n_seeds, "process_noise": tm.process_noise.iloc[0], "episodes": len(per_ep)}
        for k in EP_METRICS:
            r[f"{k}_mean"], r[f"{k}_std"] = per_ep[k].mean(), per_ep[k].std(ddof=0)
        for k in ("rmse_all", "rmse_occluded"):
            r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] = boot_ci(per_ep[k], n_boot, rng)
        r["exit_events"] = len(em) / n_seeds
        r["exit_success"] = em.success.sum() / n_seeds
        r["exit_success_rate"] = em.success.mean()
        r["exit_majority_rate"] = em.majority.mean()
        for k in ("p_correct", "brier", "exit_nll", "zero_mass", "es_exit", "occ_rmse"):
            r[k] = em[k].mean()
        r["p_correct_ci_lo"], r["p_correct_ci_hi"] = boot_ci(per_ep["ev_p_correct"], n_boot, rng)
        r["recovery_rate"] = tm.n_recovered.sum() / max(tm.n_occlusions.sum(), 1)
        r["recovery_mean_steps_recovered_only"] = np.nanmean(tm.recovery_mean_steps)
        r["lost_episodes"] = tm.lost.sum() / n_seeds
        r["numerical_failures"] = int(tm.numerical_failures.sum())
        if n_seeds > 1:
            by_seed = tm.groupby("seed")
            for k in ("rmse_all", "rmse_occluded"):
                r[f"{k}_seed_std"] = by_seed[k].mean().std(ddof=0)
            r["exit_success_rate_seed_std"] = em.groupby("seed").success.mean().std(ddof=0)
            for s, g in by_seed:
                es = em[em.seed == s]
                seed_rows.append({"method": m, "seed": s, "rmse_all": g.rmse_all.mean(),
                                  "rmse_occluded": g.rmse_occluded.mean(), "es_mean": g.es_mean.mean(),
                                  "exit_success_rate": es.success.mean(), "p_correct": es.p_correct.mean(),
                                  "brier": es.brier.mean(), "ms_per_step": g.ms_per_step.mean()})
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(seed_rows), t


def paired(t, ev, n_boot, rng, comparisons=None):
    rows = []
    col = {k: (k if k in EP_METRICS else f"ev_{k}") for k in PAIRED}
    for m, c in (comparisons or COMPARISONS):
        if m not in set(t.method) or c not in set(t.method):
            continue
        a = t[t.method == m].groupby("run").mean(numeric_only=True)
        b = t[t.method == c].groupby("run").mean(numeric_only=True)
        runs = a.index.intersection(b.index)
        r = {"method": m, "control": c, "episodes": len(runs)}
        for k in PAIRED:
            d = (a.loc[runs, col[k]] - b.loc[runs, col[k]]).to_numpy()
            r[f"d_{k}"] = float(np.nanmean(d))
            r[f"d_{k}_ci_lo"], r[f"d_{k}_ci_hi"] = boot_ci(d, n_boot, rng)
        key = ["run", "enter_step"]
        ea = ev[(ev.method == m) & (ev.seed.astype(str).isin(["0", "0.0", "", "nan"]))].set_index(key).success
        eb = ev[(ev.method == c) & (ev.seed.astype(str).isin(["0", "0.0", "", "nan"]))].set_index(key).success
        common = ea.index.intersection(eb.index)
        r["events_disagree"] = int((ea.loc[common] != eb.loc[common]).sum())
        r["events_compared"] = len(common)
        rows.append(r)
    return pd.DataFrame(rows)


def fmt_table(s, lane):
    head = ("| Method | q | RMSE all | RMSE vis | RMSE occ | Exit top-1 | >50% | P(correct) | Brier | ExitNLL "
            "| ES | Recovery | ESS/N | ms/step |\n|" + "---|" * 14)
    lines = [head]
    for _, r in s.iterrows():
        lines.append(
            f"| {r.method}{' ×' + str(r.seeds) if r.seeds > 1 else ''} | {r.process_noise:g} "
            f"| {r.rmse_all_mean:.3f} ± {r.rmse_all_std:.3f} | {r.rmse_visible_mean:.3f} ± {r.rmse_visible_std:.3f} "
            f"| {r.rmse_occluded_mean:.3f} ± {r.rmse_occluded_std:.3f} "
            f"| {100 * r.exit_success_rate:.1f}% ({r.exit_success:.0f}/{r.exit_events:.0f}) "
            f"| {100 * r.exit_majority_rate:.1f}% | {r.p_correct:.3f} | {r.brier:.3f} | {r.exit_nll:.2f} "
            f"| {r.es_mean_mean:.4f} | {100 * r.recovery_rate:.1f}% | {r.ess_pre_mean_mean:.2f} | {r.ms_per_step_mean:.1f} |")
    lines.append(f"\nLane-cue reference: {lane[0]}/{lane[1]} ({100 * lane[0] / max(lane[1], 1):.1f}%).")
    return "\n".join(lines)


def fmt_paired(p):
    lines = ["| Method − control | ΔRMSE all | ΔRMSE occ | ΔES | Δexit top-1 | ΔP(correct) | ΔBrier | events differ |",
             "|" + "---|" * 8]
    ci = lambda r, k, f: f"{r[f'd_{k}']:{f}} [{r[f'd_{k}_ci_lo']:{f}}, {r[f'd_{k}_ci_hi']:{f}}]"
    for _, r in p.iterrows():
        lines.append(f"| {r.method} − {r.control} | {ci(r, 'rmse_all', '+.4f')} | {ci(r, 'rmse_occluded', '+.4f')} "
                     f"| {ci(r, 'es_mean', '+.4f')} | {ci(r, 'success', '+.3f')} | {ci(r, 'p_correct', '+.3f')} "
                     f"| {ci(r, 'brier', '+.3f')} | {r.events_disagree}/{r.events_compared} |")
    return "\n".join(lines)


def write_latex(path, s, lane, split):
    lines = ["% Generated by mml_research.motion.cfm_variants.summarize", "\\begin{table*}[t]", "\\centering",
             f"\\caption{{CFM variants on the {split} episodes. RMSE: mean $\\pm$ std over episodes of the "
             "per-episode RMSE (seeded methods averaged over 3 training seeds first). Exit columns as in the "
             f"exit benchmark; lane-cue reference {lane[0]}/{lane[1]}.}}",
             "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lcccccccc}", "\\hline",
             "Method & $q$ (m) & RMSE all & RMSE occ. & Exit top-1 & $P$(correct) & Brier & ES & ms/step \\\\",
             "\\hline"]
    for _, r in s.iterrows():
        lines.append(f"{r.method} & {r.process_noise:g} & ${r.rmse_all_mean:.3f} \\pm {r.rmse_all_std:.3f}$ "
                     f"& ${r.rmse_occluded_mean:.3f} \\pm {r.rmse_occluded_std:.3f}$ "
                     f"& {100 * r.exit_success_rate:.1f}\\% & {r.p_correct:.3f} & {r.brier:.3f} "
                     f"& {r.es_mean_mean:.4f} & {r.ms_per_step_mean:.1f} \\\\")
    lines += ["\\hline", "\\end{tabular}%", "}", "\\end{table*}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(alpha=0.25, lw=0.6)
    ax.title.set_color(INK)


def plot_occlusion(ev, out):
    bins = np.array([0, 20, 30, 40, 50, 70, 1000])
    centers = ["<20", "20-29", "30-39", "40-49", "50-69", "70+"]
    panels = [pn for pn in PANELS if set(pn[2]) & set(ev.method)]
    if not panels:          # e.g. the dev split: controls only
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6), sharey=True, squeeze=False)
    for ax, (title, ctrl, ms) in zip(axes[0], panels):
        for i, m in enumerate([ctrl] + ms):
            e = ev[ev.method == m]
            if e.empty:
                continue
            b = np.digitize(e.occ_len, bins) - 1
            y = [np.sqrt(np.mean(e.occ_rmse[b == k] ** 2)) if (b == k).any() else np.nan for k in range(len(centers))]
            kw = dict(color=GRAY, ls="--") if i == 0 else dict(color=SERIES[i - 1])
            ax.plot(centers, y, marker="o", ms=4, lw=2, label=m, **kw)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("occlusion length (steps)", fontsize=8, color=INK2)
        _style(ax)
        ax.legend(fontsize=7, frameon=False)
    axes[0, 0].set_ylabel("RMSE during the occlusion (m)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rmse_vs_occlusion.png"), dpi=150)
    plt.close(fig)


def plot_reliability(ev, out):
    """Predicted mass on each exit label vs how often the target actually exits there (all labels pooled)."""
    labels = ["N", "E", "S", "W"]
    edges = np.linspace(0, 1, 11)
    panels = [pn for pn in PANELS if set(pn[2]) & set(ev.method)]
    if not panels:          # e.g. the dev split: controls only
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.8), sharey=True, squeeze=False)
    for ax, (title, ctrl, ms) in zip(axes[0], panels):
        ax.plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls=":")
        for i, m in enumerate([ctrl] + ms):
            e = ev[ev.method == m]
            if e.empty:
                continue
            p = np.concatenate([e[f"p_{s}"].to_numpy() for s in labels])
            y = np.concatenate([(e.actual_side == s).to_numpy() for s in labels]).astype(float)
            b = np.clip(np.digitize(p, edges) - 1, 0, 9)
            xs = [p[b == k].mean() for k in range(10) if (b == k).sum() >= 10]
            ys = [y[b == k].mean() for k in range(10) if (b == k).sum() >= 10]
            kw = dict(color=GRAY, ls="--") if i == 0 else dict(color=SERIES[i - 1])
            ax.plot(xs, ys, marker="o", ms=4, lw=2, label=m, **kw)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("predicted exit mass", fontsize=8, color=INK2)
        _style(ax)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[0, 0].set_ylabel("observed exit frequency", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "exit_reliability.png"), dpi=150)
    plt.close(fig)


def plot_runtime(s, out):
    """Accuracy vs cost: two separate panels (no dual axis), points direct-labelled."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (key, label) in zip(axes, [("p_correct", "P(correct exit)  (higher is better)"),
                                       ("rmse_occluded_mean", "RMSE occluded (m)  (lower is better)")]):
        color = {m: c for _, c, ms in FAMILIES for m in ms}
        ax.scatter(s.ms_per_step_mean, s[key], s=40, color=[color.get(m, SERIES[0]) for m in s.method],
                   edgecolor="white", linewidth=1.5, zorder=3)
        for _, r in s.iterrows():
            ax.annotate(r.method, (r.ms_per_step_mean, r[key]), textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_xlabel("ms per filter step (log)", fontsize=8, color=INK2)
        ax.set_title(label, fontsize=10)
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "accuracy_vs_runtime.png"), dpi=150)
    plt.close(fig)


def regression(ep, ev, old_dir):
    """Dev split only: per-episode differences to the producing benchmark's per-run / event CSVs."""
    old_ep = pd.read_csv(os.path.join(old_dir, "exit_benchmark_per_run.csv"))
    old_ev = pd.read_csv(os.path.join(old_dir, "exit_benchmark_events.csv"))
    rows = []
    for m in sorted(set(ep.method) & set(old_ep.tracker)):
        a = ep[ep.method == m].set_index("run")
        b = old_ep[old_ep.tracker == m].set_index("run")
        runs = a.index.intersection(b.index)
        ea = ev[ev.method == m].set_index(["run", "enter_step"]).success
        eb = old_ev[(old_ev.tracker == m) & old_ev.run.isin(runs)].set_index(["run", "enter_step"]).success
        common = ea.index.intersection(eb.index)
        rows.append({"method": m, "episodes": len(runs),
                     "max_abs_d_rmse_all": float((a.loc[runs, "rmse_all"] - b.loc[runs, "rmse_all"]).abs().max()),
                     "max_abs_d_rmse_occluded": float((a.loc[runs, "rmse_occluded"]
                                                       - b.loc[runs, "rmse_occluded"]).abs().max()),
                     "old_exit_success": int(eb.sum()), "new_exit_success": int(ea.sum()),
                     "events_compared": len(common),
                     "events_disagree": int((ea.loc[common] != eb.loc[common]).sum())})
    return pd.DataFrame(rows)


FAMILIES = [  # (label, color, methods) — color by experiment, fixed order
    ("Baselines", GRAY, ["PF+CFM", "PF+CFM-AS", "PF+DMMN", "PF+NF"]),
    ("M1 Context-CFM", SERIES[0], ["M1-control", "M1-mask", "M1-time", "M1-full"]),
    ("M2 Calibrated-CFM", SERIES[1], ["M2-zero", "M2-calibrated"]),
    ("M3 Observation-CFM", SERIES[2], ["M3", "M3-joint"]),
    ("M4 DMMN-CFM", SERIES[3], ["DMMN+Gauss", "M4"]),
    ("Open-source ports", SERIES[4], ["PF+TrajFlow", "PF+MoFlow", "PF+TCFM", "EnFF"]),
    ("Classical / DiffPF", SERIES[5], ["PF+CV", "DiffPF"]),
]


def plot_overview(s, lane, out):
    """All methods x six metrics as horizontal bars, colored by experiment family."""
    order = [m for _, _, ms in FAMILIES for m in ms if m in set(s.method)]
    color = {m: c for _, c, ms in FAMILIES for m in ms}
    s = s.set_index("method").loc[order]
    y = np.arange(len(order))[::-1]
    panels = [
        ("rmse_all_mean", "RMSE all steps (m) ↓", ("rmse_all_ci_lo", "rmse_all_ci_hi"), "{:.3f}"),
        ("rmse_occluded_mean", "RMSE occluded (m) ↓", ("rmse_occluded_ci_lo", "rmse_occluded_ci_hi"), "{:.3f}"),
        ("exit_success_rate", "Exit success, top-1 ↑", None, "{:.1%}"),
        ("p_correct", "P(correct exit) ↑", ("p_correct_ci_lo", "p_correct_ci_hi"), "{:.3f}"),
        ("brier", "Exit Brier score ↓", None, "{:.3f}"),
        ("ms_per_step_mean", "Runtime (ms / step) ↓", None, "{:.1f}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 4 + 0.36 * len(order)), sharey=True)
    for ax, (key, title, ci, fmt) in zip(axes.flat, panels):
        vals = s[key].to_numpy()
        ax.barh(y, vals, height=0.7, color=[color[m] for m in order], edgecolor="white", linewidth=2)
        ends = vals
        if ci:
            lo, hi = s[ci[0]].to_numpy(), s[ci[1]].to_numpy()
            ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
            ends = np.maximum(vals, hi)
        for yi, v, e in zip(y, vals, ends):   # label past the whisker so they never overlap
            ax.text(e, yi, "  " + fmt.format(v), va="center", fontsize=7, color=INK2)
        if key == "exit_success_rate":
            ref = lane[0] / max(lane[1], 1)
            ax.axvline(ref, color=INK, lw=1, ls="--")
            ax.text(ref, y[0] + 0.9, f"lane cue {ref:.1%}", fontsize=7, color=INK, ha="right")
        if key == "ms_per_step_mean" and vals.max() > 20 * vals.min():
            ax.set_xscale("log")      # TCFM's ~0.5 s/step would flatten every other bar
            ax.set_xlim(vals.min() / 2, vals.max() * 4)
            title = title.replace("↓", "(log) ↓")
        else:
            ax.set_xlim(0, vals.max() * 1.25)
        ax.set_title(title, fontsize=10)
        _style(ax)
        ax.grid(axis="y", visible=False)
    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels([f"{m} ×{int(s.loc[m, 'seeds'])}" if s.loc[m, "seeds"] > 1 else m for m in order],
                           fontsize=8, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in FAMILIES]
    fig.legend(handles, [f for f, _, _ in FAMILIES], loc="lower center", ncol=len(FAMILIES), frameon=False,
               fontsize=9)
    n_ev = int(s["exit_events"].iloc[0])
    fig.suptitle(f"CFM variants and baselines in the particle filter — locked test set, {int(s['episodes'].iloc[0])} episodes, "
                 f"{n_ev} exit events (bars: mean, whiskers: 95% episode-bootstrap CI; ×3 = mean of 3 seeds)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(out, "overview_all_methods.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Summarize a CFM-variant benchmark split")
    p.add_argument("--split", default="test")
    p.add_argument("--old-results", dest="old_results", default=None,
                   help="Dev split: folder of the producing benchmark's CSVs to regress against")
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--config", default=CONFIG_PATH)
    args = p.parse_args()
    cfg = load_config(args.config)
    d = os.path.join(args.out_dir, args.split)
    ep = pd.read_csv(os.path.join(d, "episodes.csv"))
    ev = pd.read_csv(os.path.join(d, "events.csv"))
    for df in (ep, ev):
        df["seed"] = df["seed"].fillna(-1).astype(int)
    rng = np.random.default_rng(0)
    n_boot = cfg["evaluation"]["bootstrap_resamples"]
    s, by_seed, t = summarize(ep, ev, n_boot, rng)
    pr = paired(t, ev.assign(seed=ev.seed.replace(-1, 0)), n_boot, rng)
    first = s.method.iloc[0]
    e0 = ev[(ev.method == first) & (ev.seed.isin([-1, 0]))]
    lane = (int((e0.lane_side == e0.actual_side).sum()), len(e0))

    s.to_csv(os.path.join(d, "summary.csv"), index=False)
    by_seed.to_csv(os.path.join(d, "summary_by_seed.csv"), index=False)
    pr.to_csv(os.path.join(d, "paired.csv"), index=False)
    md = fmt_table(s, lane) + "\n\nPaired differences (method − control, mean [95% episode-bootstrap CI]):\n\n" + fmt_paired(pr)
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write(md + "\n")
    write_latex(os.path.join(d, "table.tex"), s, lane, args.split)
    plots = os.path.join(d, "plots")
    os.makedirs(plots, exist_ok=True)
    plot_occlusion(ev, plots)
    plot_reliability(ev, plots)
    plot_runtime(s, plots)
    plot_overview(s, lane, plots)
    if args.old_results:
        reg = regression(ep, ev, args.old_results)
        reg.to_csv(os.path.join(d, "regression_vs_old.csv"), index=False)
        print("Regression vs producing benchmark:\n" + reg.to_string(index=False))
    print(md)
    print(f"\nwritten to {d}")


if __name__ == "__main__":
    main()
