#!/usr/bin/env python3
"""
summarize.py — ETH-UCY tables and figures for both tracks.

Per split: mean over the three training seeds (± their std). AVG = mean of the
five split means (the standard ETH-UCY convention). Tracking metrics are means
over tracks of per-track values (metres).

    python -m mml_research.motion.ethucy.summarize
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mml_research.motion.cfm_variants.summarize import GRAY, INK, INK2, SERIES, _style
from mml_research.motion.ethucy.common import CONFIG_PATH, DEFAULT_OUT_DIR, load_config
from mml_research.motion.fair_retrain.common import MODEL_LABEL

# Same grouping and slots as plot_benchmark.py's exit-benchmark figures.
FAMILY = {"CFM": ("CFM family (ours)", SERIES[0]), "M4": ("CFM family (ours)", SERIES[0]),
          "NF": ("Earlier baselines", SERIES[1]), "DMMN": ("Earlier baselines", SERIES[1]),
          "DMMN+Gauss": ("Earlier baselines", SERIES[1]),
          "TrajFlow": ("Open-source ports", SERIES[2]), "MoFlow": ("Open-source ports", SERIES[2]),
          "TCFM": ("Open-source ports", SERIES[2]),
          "PF+CV": ("Training-free", GRAY), "EnFF": ("Training-free", GRAY)}


def per_split(df, metrics, splits, order, by="split"):
    """(method, split) seed-mean and seed-std, plus AVG over splits (`by`: the grouping column)."""
    seed_means = df.groupby(["method", by, "seed"], dropna=False)[metrics].mean().reset_index()
    g = seed_means.groupby(["method", by])[metrics]
    mean, std = g.mean(), g.std(ddof=0).fillna(0.0)
    rows = []
    for m in [x for x in order if x in set(df.method)]:
        r = {"method": m}
        for k in metrics:
            for s in splits:
                r[f"{k}_{s}"], r[f"{k}_{s}_sd"] = mean.loc[(m, s), k], std.loc[(m, s), k]
            r[f"{k}_AVG"] = np.mean([r[f"{k}_{s}"] for s in splits])
        rows.append(r)
    return pd.DataFrame(rows)


def md_table(t, metrics, splits, fmt):
    cols = splits + ["AVG"]
    lines = ["| Method | " + " | ".join(c.upper() for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        cells = [" / ".join(fmt.format(r[f"{k}_{c}"]) for k in metrics) for c in cols]
        lines.append(f"| {r.method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bar_panels(t, panels, splits, title, path, legend_note="", dot_label="one split's mean"):
    order = list(t.method)
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 1.6 + 0.42 * len(order)), sharey=True)
    for ax, (key, label, fmt, log) in zip(np.atleast_1d(axes), panels):
        avg = t[f"{key}_AVG"].to_numpy(float)
        ax.barh(y, avg, height=0.62, color=[FAMILY[m][1] for m in order], edgecolor="white", linewidth=2)
        for s in splits:                       # each split's mean as a dot on the AVG bar
            ax.scatter(t[f"{key}_{s}"], y, s=9, color=INK2, zorder=3, linewidths=0)
        hi = np.nanmax(t[[f"{key}_{s}" for s in splits]].to_numpy(float), axis=1)
        for yi, v, h in zip(y, avg, hi):
            ax.text(max(v, h), yi, "  " + fmt.format(v), va="center", fontsize=7, color=INK2)
        if log:
            ax.set_xscale("log")
            ax.set_xlim(np.nanmin(avg) / 2, np.nanmax(hi) * 4)
        else:
            ax.set_xlim(0, np.nanmax(hi) * 1.25)
        ax.set_title(label, fontsize=10)
        _style(ax)
        ax.grid(axis="y", visible=False)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(y)
    ax0.set_yticklabels(order, fontsize=8, color=INK)
    fams = list(dict.fromkeys(FAMILY[m] for m in order))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in fams]
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=4, color=INK2))
    fig.legend(handles, [f for f, _ in fams] + [f"{dot_label} (" + ", ".join(splits) + ")"],
               loc="lower center", ncol=len(handles), frameon=False, fontsize=8.5)
    fig.suptitle(title + legend_note, fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Summarize ETH-UCY forecasting and tracking")
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--config", default=CONFIG_PATH)
    args = p.parse_args()
    cfg = load_config(args.config)
    splits = cfg["splits"]
    order = [MODEL_LABEL[m] for m in cfg["models"]] + list(cfg["training_free"])
    plots = os.path.join(args.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    md = []

    fc = pd.read_csv(os.path.join(args.out_dir, "forecast.csv"))
    fc["method"] = fc.method.map(lambda m: MODEL_LABEL.get(m, m))
    fc["seed"] = fc.seed.fillna(-1)
    ft = per_split(fc, ["minADE", "minFDE"], splits, order)
    ft.to_csv(os.path.join(args.out_dir, "forecast_summary.csv"), index=False)
    md += [f"## Forecasting (best-of-{cfg['forecast']['k']} minADE / minFDE, m; mean of 3 seeds)", "",
           md_table(ft, ["minADE", "minFDE"], splits, "{:.2f}"), ""]
    bar_panels(ft, [("minADE", f"minADE@{cfg['forecast']['k']} (m) ↓", "{:.2f}", False),
                    ("minFDE", f"minFDE@{cfg['forecast']['k']} (m) ↓", "{:.2f}", False)],
               splits, "ETH-UCY forecasting — 8 observed → 12 future frames, every model retrained per split "
                       "under the same protocol (bars: AVG of 5 splits, mean of 3 seeds)",
               os.path.join(plots, "forecast_ethucy.png"))

    tp = os.path.join(args.out_dir, "tracking", "test_tracks.csv")
    if os.path.exists(tp):
        tk = pd.read_csv(tp)
        tk["seed"] = tk.seed.fillna(-1)
        metrics = ["rmse_all", "rmse_occluded", "es_mean", "ms_per_step"]
        tt = per_split(tk, metrics, splits, order)
        tt.to_csv(os.path.join(args.out_dir, "tracking", "summary.csv"), index=False)
        tc = cfg["tracking"]
        md += [f"## Tracking (PF on test tracks; noise {tc['meas_noise']} m, occlusions "
               f"{tc['occluded_run'][0]}-{tc['occluded_run'][1]} frames) — RMSE all / RMSE occluded (m)", "",
               md_table(tt, ["rmse_all", "rmse_occluded"], splits, "{:.3f}"), "",
               "Energy score (m) / ms per step", "", md_table(tt, ["es_mean", "ms_per_step"], splits, "{:.3f}"), ""]
        bar_panels(tt, [("rmse_all", "RMSE all frames (m) ↓", "{:.3f}", False),
                        ("rmse_occluded", "RMSE occluded frames (m) ↓", "{:.3f}", False),
                        ("es_mean", "Energy score (m) ↓", "{:.3f}", False),
                        ("ms_per_step", "Runtime (ms / step, log) ↓", "{:.0f}", True)],
                   splits, "ETH-UCY tracking — particle filter on real pedestrian tracks with simulated noise "
                           "and occlusions (bars: AVG of 5 splits, mean of 3 seeds)",
                   os.path.join(plots, "tracking_ethucy.png"))
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
