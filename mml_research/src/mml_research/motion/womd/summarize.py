#!/usr/bin/env python3
"""
summarize.py — WOMD tables and figures for both tracks.

Per object type: mean over the three training seeds (± their std). AVG = mean
of the three type means, the official aggregation (forecasting metrics are
already averaged over the 3 / 5 / 8 s measurement steps). Tracking metrics
are means over tracks of per-track values (metres).

    python -m mml_research.motion.womd.summarize
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from mml_research.motion.ethucy.summarize import bar_panels, md_table, per_split
from mml_research.motion.fair_retrain.common import MODEL_LABEL
from mml_research.motion.womd.common import CONFIG_PATH, DEFAULT_OUT_DIR, TYPES, load_config

METRICS = ["minADE", "minFDE", "MR"]


def main():
    p = argparse.ArgumentParser(description="Summarize WOMD forecasting and tracking")
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--config", default=CONFIG_PATH)
    args = p.parse_args()
    cfg = load_config(args.config)
    order = [MODEL_LABEL[m] for m in cfg["models"]] + list(cfg["training_free"])
    plots = os.path.join(args.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    fc_k = cfg["forecast"]["k"]

    fc = pd.read_csv(os.path.join(args.out_dir, "forecast.csv"))
    fc["method"] = fc.method.map(lambda m: MODEL_LABEL.get(m, m))
    fc["seed"] = fc.seed.fillna(-1)
    long = pd.concat([fc[["method", "seed"]].assign(type=t, **{m: fc[f"{m}_{t}"] for m in METRICS})
                      for t in TYPES])
    ft = per_split(long, METRICS, TYPES, order, by="type")
    ft.to_csv(os.path.join(args.out_dir, "forecast_summary.csv"), index=False)
    md = [f"## Forecasting — WOMD v1.3.0 validation, {int(fc.agents.iloc[0])} agents, K = {fc_k}; "
          f"minADE / minFDE (m) / MR, each averaged over 3 / 5 / 8 s; mean of 3 seeds", "",
          md_table(ft, METRICS, TYPES, "{:.3f}"), "",
          "Single-agent, map-free inputs (own 1.1 s history only): not comparable to the leaderboard.", ""]
    bar_panels(ft, [("minADE", f"minADE@{fc_k} (m) ↓", "{:.2f}", False),
                    ("minFDE", f"minFDE@{fc_k} (m) ↓", "{:.2f}", False),
                    ("MR", f"Miss rate@{fc_k} ↓", "{:.3f}", False)],
               TYPES, "WOMD forecasting — 1.1 s history → 8 s future, own history only, every model trained on "
                      "the same rows (bars: mean over types, of 3 seeds)",
               os.path.join(plots, "forecast_womd.png"), dot_label="one type's mean")

    tp = os.path.join(args.out_dir, "tracking", "test_tracks.csv")
    if os.path.exists(tp):
        tk = pd.read_csv(tp)
        tk["seed"] = tk.seed.fillna(-1)
        metrics = ["rmse_all", "rmse_occluded", "es_mean", "ms_per_step"]
        tt = per_split(tk, metrics, TYPES, order, by="type")
        tt.to_csv(os.path.join(args.out_dir, "tracking", "summary.csv"), index=False)
        tc = cfg["tracking"]
        md += [f"## Tracking (PF on {tc['test_per_type']} validation tracks per type; noise {tc['meas_noise']} m, "
               f"occlusions {tc['occluded_run'][0]}-{tc['occluded_run'][1]} frames) — RMSE all / occluded (m)", "",
               md_table(tt, ["rmse_all", "rmse_occluded"], TYPES, "{:.3f}"), "",
               "Energy score (m) / ms per step", "", md_table(tt, ["es_mean", "ms_per_step"], TYPES, "{:.3f}"), ""]
        bar_panels(tt, [("rmse_all", "RMSE all frames (m) ↓", "{:.3f}", False),
                        ("rmse_occluded", "RMSE occluded frames (m) ↓", "{:.3f}", False),
                        ("es_mean", "Energy score (m) ↓", "{:.3f}", False),
                        ("ms_per_step", "Runtime (ms / step, log) ↓", "{:.0f}", True)],
                   TYPES, "WOMD tracking — particle filter on 9.1 s agent tracks with simulated noise and "
                          "occlusions (bars: mean over types, of 3 seeds)",
                   os.path.join(plots, "tracking_womd.png"), dot_label="one type's mean")
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
