#!/usr/bin/env python3
"""
merge_noise_sweeps.py
=====================
Fold a process-noise sweep of additional trackers (sweep_process_noise.py
with --trackers ..., written to its own directory) and the EnFF sweep
(sweep_enff.py, at its selected lambda / sigma_min) into the main
noise_sweep/ directory, so make_benchmark_tables.py sees one sweep.

Every sweep run is seeded by its run index and tracker, so rows for trackers
already in noise_sweep/ are what a joint re-run would produce; they are kept
and only trackers not yet present are appended.

Usage:
    python -m mml_research.motion.merge_noise_sweeps --root <exit_benchmark dir> \
        --add noise_sweep_new --enff enff_sweep
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mml_research.motion.benchmark_exit_prediction import TRACKERS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--add", nargs="*", default=[], help="sweep dirs (relative to --root) to append")
    p.add_argument("--enff", default=None, help="sweep_enff.py output dir (relative to --root)")
    args = p.parse_args()

    main_dir = os.path.join(args.root, "noise_sweep")
    runs = pd.read_csv(os.path.join(main_dir, "noise_sweep_runs.csv"))
    summary = pd.read_csv(os.path.join(main_dir, "noise_sweep.csv"))

    for d in args.add:
        r = pd.read_csv(os.path.join(args.root, d, "noise_sweep_runs.csv"))
        s = pd.read_csv(os.path.join(args.root, d, "noise_sweep.csv"))
        new = sorted(set(r.tracker) - set(runs.tracker))
        runs = pd.concat([runs, r[r.tracker.isin(new)]], ignore_index=True)
        summary = pd.concat([summary, s[s.tracker.isin(new)]], ignore_index=True)
        print(f"appended {new} from {d}")

    if args.enff and "EnFF" not in set(runs.tracker):
        es = pd.read_csv(os.path.join(args.root, args.enff, "enff_sweep.csv"))
        er = pd.read_csv(os.path.join(args.root, args.enff, "enff_sweep_runs.csv"))
        best = es.loc[es.rmse_all.idxmin()]
        at = lambda df: df[np.isclose(df.enff_lambda, best.enff_lambda)
                           & np.isclose(df.enff_sigma_min, best.enff_sigma_min)]
        r = at(er).drop(columns=["enff_lambda", "enff_sigma_min"])
        s = at(es).drop(columns=["enff_lambda", "enff_sigma_min"]).assign(tracker="EnFF")
        runs = pd.concat([runs, r[runs.columns]], ignore_index=True)
        summary = pd.concat([summary, s[summary.columns]], ignore_index=True)
        print(f"appended EnFF at lambda={best.enff_lambda:g}, sigma_min={best.enff_sigma_min:g}")

    order = {t: i for i, t in enumerate(TRACKERS)}
    runs = runs.sort_values(["process_noise", "tracker", "run"], key=lambda c: c.map(order) if c.name == "tracker" else c)
    summary = summary.sort_values(["tracker", "process_noise"], key=lambda c: c.map(order) if c.name == "tracker" else c)
    runs.to_csv(os.path.join(main_dir, "noise_sweep_runs.csv"), index=False)
    summary.to_csv(os.path.join(main_dir, "noise_sweep.csv"), index=False)

    chosen = {t: g.loc[g.rmse_all.idxmin(), "process_noise"] for t, g in summary.groupby("tracker", sort=False)}
    line = " ".join(f"{t}={chosen[t]:g}" for t in TRACKERS if t in chosen)
    with open(os.path.join(main_dir, "chosen_noise.txt"), "w") as f:
        f.write(line + "\n")
    print(f"chosen (lowest RMSE all): {line}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("rmse_all", "RMSE all (m)", True), ("exit_majority_rate", "Exit success (>50% mass)", False),
              ("exit_p_correct", "P(correct exit)", False)]
    for ax, (key, label, log) in zip(axes, panels):
        for t in [t for t in TRACKERS if t in chosen]:
            g = summary[summary.tracker == t].sort_values("process_noise")
            ax.plot(g.process_noise, g[key], marker="o", label=t)
            ax.scatter([chosen[t]], g.loc[g.process_noise == chosen[t], key], s=120,
                       facecolors="none", edgecolors="black", zorder=3)
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("process noise (m per step)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Process-noise sensitivity on validation runs (circle = chosen by lowest RMSE all)")
    fig.tight_layout()
    fig.savefig(os.path.join(main_dir, "noise_sensitivity.png"), dpi=150)
    print(f"wrote {main_dir}")


if __name__ == "__main__":
    main()
