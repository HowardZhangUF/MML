#!/usr/bin/env python3
"""
aggregate_recovery_results.py
==============================
Reads all occlusion_recovery_*.csv files from tracking_results/ and prints
a quantitative comparison table of CFM vs Transformer (MultiPFTra).

Primary metric  : Modal-side recovery rate  (fraction of events correct)
Secondary metric: Mean probability score     (calibration — CFM only meaningful)

Also prints a reliability / calibration table and saves a summary CSV and
PNG bar chart to tracking_results/recovery_summary/.

Usage
-----
    python scripts/aggregate_recovery_results.py
    python scripts/aggregate_recovery_results.py --folder tracking_results
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir


import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DISPLAY = {
    "CFM":        "CFM (Flow Matching)",
    "MultiPFTra": "Transformer",
}

COLORS = {
    "CFM":        "#4C72B0",   # blue
    "MultiPFTra": "#DD8452",   # orange
}

N_CAL_BINS = 5


def load_recovery_csvs(folder: str) -> pd.DataFrame:
    """Load and concatenate all occlusion_recovery_*.csv files."""
    frames = []
    for fname in sorted(os.listdir(folder)):
        if "occlusion_recovery" not in fname or not fname.endswith(".csv"):
            continue
        fpath = os.path.join(folder, fname)
        if os.path.getsize(fpath) == 0:
            print(f"  Skipped (empty): {fname}")
            continue
        try:
            df = pd.read_csv(fpath)
        except pd.errors.EmptyDataError:
            print(f"  Skipped (no columns): {fname}")
            continue
        if df.empty:
            print(f"  Skipped (0 rows): {fname}")
            continue
        # Infer model name from filename:  prefix_occlusion_recovery_MODEL_timestamp.csv
        parts = fname.split("_occlusion_recovery_")
        if len(parts) == 2:
            model_ts = parts[1]                          # e.g. "CFM_20260625_143201.csv"
            model = model_ts.rsplit("_", 2)[0]           # e.g. "CFM"
        else:
            model = df.get("model", ["unknown"])[0]
        df["model"] = model
        df["source_file"] = fname
        frames.append(df)
        print(f"  Loaded {fname}  ({len(df)} events, model={model})")

    if not frames:
        print("No occlusion_recovery_*.csv files found in", folder)
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def per_run_stats(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Compute per-run (per-file) summary for one model."""
    rows = []
    subset = df[df["model"] == model]
    for fname, grp in subset.groupby("source_file"):
        scored_modal = grp["modal_correct"].dropna()
        scored_prob  = grp["prob_correct"].dropna()
        rows.append({
            "file":                fname,
            "n_events":            len(grp),
            "modal_recovery_rate": scored_modal.mean() if len(scored_modal) else np.nan,
            "mean_prob_correct":   scored_prob.mean()  if len(scored_prob)  else np.nan,
            "mean_duration":       grp["duration_steps"].mean(),
            "max_duration":        grp["duration_steps"].max(),
        })
    return pd.DataFrame(rows)


def calibration_table(df: pd.DataFrame, model: str, n_bins: int = N_CAL_BINS) -> pd.DataFrame:
    """Bin events by predicted probability and show observed accuracy."""
    subset = df[(df["model"] == model) & df["prob_correct"].notna() & df["modal_correct"].notna()]
    if subset.empty:
        return pd.DataFrame()

    prob  = subset["prob_correct"].values
    modal = subset["modal_correct"].values

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi)
        cnt  = int(mask.sum())
        acc  = float(modal[mask].mean()) if cnt > 0 else np.nan
        rows.append({
            "bin":              f"[{lo:.2f}, {hi:.2f})",
            "bin_center":       (lo + hi) / 2,
            "n_events":         cnt,
            "observed_accuracy": acc,
        })
    return pd.DataFrame(rows)


def print_model_summary(df: pd.DataFrame, model: str, run_stats: pd.DataFrame) -> None:
    display = MODEL_DISPLAY.get(model, model)
    n_runs   = len(run_stats)
    n_events = run_stats["n_events"].sum()
    n_valid  = df[(df["model"] == model) & df["modal_correct"].notna()].shape[0]

    mrr_mean = run_stats["modal_recovery_rate"].mean()
    mrr_std  = run_stats["modal_recovery_rate"].std()
    mpc_mean = run_stats["mean_prob_correct"].mean()
    mpc_std  = run_stats["mean_prob_correct"].std()
    dur_mean = run_stats["mean_duration"].mean()
    dur_max  = run_stats["max_duration"].max()

    print(f"\n{'=' * 62}")
    print(f"  OCCLUSION RECOVERY SUMMARY  [{display}]")
    print(f"{'=' * 62}")
    print(f"  Runs                          : {n_runs}")
    print(f"  Total occlusion events        : {n_events}  ({n_valid} scored)")
    print(f"  Events per run  (mean)        : {n_events/n_runs:.1f}")
    print()
    print(f"  Modal-side recovery rate")
    print(f"    Mean ± SD                   : {mrr_mean*100:.1f}% ± {mrr_std*100:.1f}%")
    print(f"    Per-run values              : " +
          "  ".join(f"{v*100:.0f}%" for v in run_stats["modal_recovery_rate"].dropna()))
    print()
    print(f"  Mean probability score  (calibration)")
    print(f"    Mean ± SD                   : {mpc_mean:.3f} ± {mpc_std:.3f}  (chance = 0.500)")
    print(f"    Per-run values              : " +
          "  ".join(f"{v:.3f}" for v in run_stats["mean_prob_correct"].dropna()))
    print()
    print(f"  Occlusion duration  mean/max  : {dur_mean:.1f} / {dur_max:.0f} steps")

    cal = calibration_table(df, model)
    if not cal.empty and cal["n_events"].sum() > 0:
        print()
        print("  Calibration  (predicted prob → observed accuracy):")
        print(f"  {'Bin':<18} {'n':>5}  {'Accuracy':>9}  {'Bar'}")
        for _, row in cal.iterrows():
            if row["n_events"] > 0:
                acc_str = f"{row['observed_accuracy']*100:.0f}%" if not np.isnan(row["observed_accuracy"]) else "  -"
                bar = "#" * int(row["n_events"])
                print(f"  {row['bin']:<18} {int(row['n_events']):>5}  {acc_str:>9}  {bar}")
    print(f"{'=' * 62}")


def print_comparison_table(stats: dict) -> None:
    """Side-by-side comparison of both models."""
    print(f"\n{'=' * 62}")
    print("  CFM  vs  TRANSFORMER  —  HEAD-TO-HEAD")
    print(f"{'=' * 62}")
    header = f"  {'Metric':<38} {'CFM':>10}  {'Transformer':>12}"
    print(header)
    print(f"  {'-'*38} {'-'*10}  {'-'*12}")

    def row(label, cfm_val, tra_val, fmt="{:.3f}"):
        cfm_s = fmt.format(cfm_val) if cfm_val is not None and not np.isnan(cfm_val) else "  n/a"
        tra_s = fmt.format(tra_val) if tra_val is not None and not np.isnan(tra_val) else "  n/a"
        print(f"  {label:<38} {cfm_s:>10}  {tra_s:>12}")

    cfm = stats.get("CFM", {})
    tra = stats.get("MultiPFTra", {})

    row("Modal recovery rate (mean)",
        cfm.get("mrr_mean"), tra.get("mrr_mean"), fmt="{:.1%}")
    row("Modal recovery rate (std)",
        cfm.get("mrr_std"),  tra.get("mrr_std"),  fmt="{:.1%}")
    row("Mean probability score (mean)",
        cfm.get("mpc_mean"), tra.get("mpc_mean"))
    row("Mean probability score (std)",
        cfm.get("mpc_std"),  tra.get("mpc_std"))
    row("Total events scored",
        cfm.get("n_events"), tra.get("n_events"), fmt="{:.0f}")
    row("Runs",
        cfm.get("n_runs"),   tra.get("n_runs"),   fmt="{:.0f}")

    # Winner
    cfm_mrr = cfm.get("mrr_mean", np.nan)
    tra_mrr = tra.get("mrr_mean", np.nan)
    if not np.isnan(cfm_mrr) and not np.isnan(tra_mrr):
        diff = (cfm_mrr - tra_mrr) * 100
        winner = "CFM" if diff > 0 else "Transformer"
        print(f"\n  Recovery rate delta (CFM − Transformer): {diff:+.1f} pp")
        print(f"  Winner (modal recovery): {winner}")

    print(f"{'=' * 62}")


def save_bar_chart(stats: dict, outdir: str) -> None:
    """Save side-by-side bar chart for modal recovery rate and prob score."""
    models   = [m for m in ["CFM", "MultiPFTra"] if m in stats]
    labels   = [MODEL_DISPLAY[m] for m in models]
    colors   = [COLORS[m] for m in models]

    mrr_means = [stats[m]["mrr_mean"] * 100 for m in models]
    mrr_stds  = [stats[m]["mrr_std"]  * 100 for m in models]
    mpc_means = [stats[m]["mpc_mean"] for m in models]
    mpc_stds  = [stats[m]["mpc_std"]  for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # ── Left: modal recovery rate ─────────────────────────────────────────────
    bars1 = ax1.bar(labels, mrr_means, yerr=mrr_stds, color=colors,
                    capsize=7, width=0.5, edgecolor="black", linewidth=0.8)
    ax1.axhline(50, color="grey", linestyle="--", linewidth=1.2, label="Chance (50%)")
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Modal Recovery Rate (%)", fontsize=12)
    ax1.set_title("Primary Metric\nModal-Side Recovery Rate", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    for bar, val, err in zip(bars1, mrr_means, mrr_stds):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 val + err + 2,
                 f"{val:.1f}%\n±{err:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── Right: mean probability score ─────────────────────────────────────────
    bars2 = ax2.bar(labels, mpc_means, yerr=mpc_stds, color=colors,
                    capsize=7, width=0.5, edgecolor="black", linewidth=0.8)
    ax2.axhline(0.5, color="grey", linestyle="--", linewidth=1.2, label="Chance (0.500)")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Mean Probability Score", fontsize=12)
    ax2.set_title("Secondary Metric\nCalibration (Prob Score)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    for bar, val, err in zip(bars2, mpc_means, mpc_stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + err + 0.02,
                 f"{val:.3f}\n±{err:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Occlusion Recovery: CFM vs Transformer\n"
        f"(runs/method: CFM={stats.get('CFM',{}).get('n_runs',0)}, "
        f"Transformer={stats.get('MultiPFTra',{}).get('n_runs',0)})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out_path = os.path.join(outdir, "occlusion_recovery_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n  Bar chart saved → {out_path}")


def save_calibration_chart(df: pd.DataFrame, outdir: str) -> None:
    """Reliability diagram: predicted prob vs observed accuracy."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    for model in ["CFM", "MultiPFTra"]:
        cal = calibration_table(df, model, n_bins=N_CAL_BINS)
        if cal.empty or cal["n_events"].sum() == 0:
            continue
        valid = cal[cal["n_events"] > 0]
        ax.errorbar(
            valid["bin_center"],
            valid["observed_accuracy"],
            fmt="o-",
            color=COLORS[model],
            label=MODEL_DISPLAY[model],
            linewidth=1.8,
            markersize=7,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (correct side)", fontsize=12)
    ax.set_ylabel("Observed accuracy", fontsize=12)
    ax.set_title("Reliability Diagram\n(CFM calibration check)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(outdir, "calibration_reliability_diagram.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Reliability diagram saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="tracking_results",
                        help="Directory containing occlusion_recovery_*.csv files")
    args = parser.parse_args()

    outdir = os.path.join(args.folder, "recovery_summary")
    os.makedirs(outdir, exist_ok=True)

    print(f"\nLoading recovery CSVs from: {args.folder}/")
    df = load_recovery_csvs(args.folder)

    print(f"\nTotal events loaded: {len(df)}")
    print(f"Models found: {df['model'].unique().tolist()}")

    # ── Per-model stats ───────────────────────────────────────────────────────
    summary_stats = {}
    for model in ["CFM", "MultiPFTra"]:
        if model not in df["model"].values:
            print(f"\n  [warning] No data for model '{model}' — skipping.")
            continue
        run_stats = per_run_stats(df, model)
        print_model_summary(df, model, run_stats)
        summary_stats[model] = {
            "n_runs":   len(run_stats),
            "n_events": run_stats["n_events"].sum(),
            "mrr_mean": run_stats["modal_recovery_rate"].mean(),
            "mrr_std":  run_stats["modal_recovery_rate"].std(),
            "mpc_mean": run_stats["mean_prob_correct"].mean(),
            "mpc_std":  run_stats["mean_prob_correct"].std(),
        }

    # ── Head-to-head comparison ───────────────────────────────────────────────
    if len(summary_stats) == 2:
        print_comparison_table(summary_stats)

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Summary CSV
    summary_rows = []
    for model, s in summary_stats.items():
        summary_rows.append({
            "model":                     MODEL_DISPLAY.get(model, model),
            "n_runs":                    s["n_runs"],
            "n_events":                  s["n_events"],
            "modal_recovery_rate_mean":  s["mrr_mean"],
            "modal_recovery_rate_std":   s["mrr_std"],
            "mean_prob_correct_mean":    s["mpc_mean"],
            "mean_prob_correct_std":     s["mpc_std"],
        })
    summary_df = pd.DataFrame(summary_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(outdir, f"recovery_summary_{ts}.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Summary CSV saved → {csv_path}")

    # Charts
    if len(summary_stats) >= 1:
        save_bar_chart(summary_stats, outdir)
        save_calibration_chart(df, outdir)

    print(f"\n  All outputs in: {outdir}/\n")


if __name__ == "__main__":
    main()
