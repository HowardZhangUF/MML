#!/usr/bin/env python3
"""
make_benchmark_tables.py
========================
Write every exit-prediction benchmark result as LaTeX tables (booktabs +
adjustbox), with mean ± 95% confidence intervals.

Inputs (all under --root, default <motion artifacts>/exit_benchmark):
    results/             final benchmark, per-tracker tuned process noise
    results_noise0.01/   same benchmark, every PF tracker at 0.01 m
    noise_sweep/         validation sweep that chose the tuned noise
    enff_sweep/          validation sweep that chose EnFF's q, lambda, sigma_min
                         (optional; sweep_enff.py)

Confidence intervals:
    RMSE, ms/step   mean over runs ± t_{0.975, n-1} · sd / sqrt(n)
    exit metrics    pooled over events; ± 1.96 · bootstrap SE, resampling
                    whole runs (events in one run share a tracker state)

Usage (after installing the project):
    python -m mml_research.motion.make_benchmark_tables --root <dir>
Also includes the road-network figure (plot_road_network.py output, road_network.pdf
next to the .tex). Needs \\usepackage{booktabs,adjustbox,graphicx} in the preamble.
"""

from mml_core.config import motion_resource_dir

import os
import argparse

import numpy as np
import pandas as pd
from scipy import stats

TRACKERS = ["PF+CFM", "PF+CFM-AS", "PF+CFM-LM", "PF+CFM-P", "PF+CFM-Pnaive", "PF+NF", "DiffPF",
            "PF+CV", "PF+DMMN",
            "PF+TrajFlow", "PF+MoFlow", "PF+TCFM", "EnFF"]
LABELS = {
    "PF+CFM": "PF+CFM (fixed $\\sigma{=}1.0$)",
    "PF+CFM-AS": "PF+CFM-AS (adaptive $\\sigma$)",
    "PF+CFM-LM": "PF+CFM-LM (learned likelihood)",
    "PF+CFM-P": "PF+CFM-P (learned proposal + likelihood)",
    "PF+CFM-Pnaive": "PF+CFM-P, uncorrected weights",
    "PF+NF": "PF+NF",
    "DiffPF": "DiffPF",
    "PF+CV": "PF+CV",
    "PF+DMMN": "PF+DMMN",
    "PF+TrajFlow": "PF+TrajFlow",
    "PF+MoFlow": "PF+MoFlow",
    "PF+TCFM": "PF+TCFM",
    "EnFF": "EnFF",
}
# A \midrule follows each of these rows: our methods / earlier baselines /
# open-source flow-matching baselines (baselines/mml_baselines).
GROUP_ENDS = {"PF+CFM-Pnaive", "PF+DMMN"}
N_BOOT = 2000


# ── Statistics ───────────────────────────────────────────────────────────────

def t_ci(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(half)


def pooled_ci(num, den, seed=0):
    """Pooled ratio sum(num)/sum(den) with a run-level bootstrap 95% half-width."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(N_BOOT, len(num)))
    boot = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(num.sum() / den.sum()), float(1.96 * boot.std(ddof=1))


# ── Formatting ───────────────────────────────────────────────────────────────

def sig3(x):
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        mant, exp = f"{x:.2e}".split("e")
        return f"{float(mant):.3g}\\times10^{{{int(exp)}}}"
    return f"{x:.3g}"


def cell(mean, half, bold=False, percent=False):
    body = (f"{100 * mean:.1f}\\pm{100 * half:.1f}" if percent
            else f"{sig3(mean)}\\pm{sig3(half)}")
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def best_set(means, higher_is_better):
    """Indices of every entry tied for the best mean."""
    arr = np.array(means, dtype=float)
    best = np.nanmax(arr) if higher_is_better else np.nanmin(arr)
    tied = set(np.flatnonzero(np.isclose(arr, best, rtol=0, atol=1e-12)))
    return set() if len(tied) == len(arr) else tied   # nothing to single out


def table(caption, label, header, rows, footnote):
    """rows: list of (row label, [cell strings]); a None row inserts \\midrule."""
    lines = [
        "\\begin{table}[ht]", "\\centering", "\\small",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\begin{adjustbox}{max width=\\linewidth}",
        f"\\begin{{tabular}}{{l{'c' * (len(header) - 1)}}}", "\\toprule",
        " & ".join(header) + " \\\\", "\\midrule",
    ]
    for row in rows:
        lines.append("\\midrule" if row is None else " & ".join([row[0]] + row[1]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}",
              f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\footnotesize {footnote}}}",
              "\\end{table}", ""]
    return "\n".join(lines)


# ── Tables ───────────────────────────────────────────────────────────────────

def benchmark_columns(per_run, events, trackers):
    """Per tracker: dict of metric -> (mean, half-width)."""
    out = {}
    for name in trackers:
        r = per_run[per_run.tracker == name].sort_values("run")
        e = events[events.tracker == name]
        maj = e.assign(m=(e.p_correct > 0.5).astype(int)).groupby("run").m.sum()
        maj = maj.reindex(r.run, fill_value=0).to_numpy()
        out[name] = {
            "rmse_all": t_ci(r.rmse_all), "rmse_visible": t_ci(r.rmse_visible),
            "rmse_occluded": t_ci(r.rmse_occluded), "ms_per_step": t_ci(r.ms_per_step),
            "top1": pooled_ci(r.exit_success, r.exit_events),
            "majority": pooled_ci(maj, r.exit_events),
            "p_correct": pooled_ci(r.exit_p_correct_sum, r.exit_events),
            "noise": r.process_noise.iloc[0] if "process_noise" in r and name != "DiffPF" else None,
            "events": int(r.exit_events.sum()),
        }
    return out


def benchmark_table(cols, caption, label, footnote, with_noise):
    trackers = list(cols)
    metrics = [("rmse_all", False, False), ("rmse_visible", False, False),
               ("rmse_occluded", False, False), ("top1", True, True),
               ("majority", True, True), ("p_correct", True, False), ("ms_per_step", False, False)]
    best = {m: best_set([cols[t][m][0] for t in trackers], hib) for m, hib, _ in metrics}
    rows = []
    for i, name in enumerate(trackers):
        c = cols[name]
        cells = [cell(*c[m], bold=(i in best[m]), percent=pct) for m, _, pct in metrics]
        if with_noise:
            cells = [("--" if c["noise"] is None else f"${float(c['noise']):g}$")] + cells
        rows.append((LABELS[name], cells))
        if name in GROUP_ENDS and name != trackers[-1]:
            rows.append(None)
    header = (["Tracker"] + (["Noise (m)"] if with_noise else []) +
              ["RMSE all (m)", "RMSE visible (m)", "RMSE occluded (m)", "Exit top-1 (\\%)",
               "Exit $>$50\\% (\\%)", "$P$(correct exit)", "Runtime (ms/step)"])
    return table(caption, label, header, rows, footnote)


def sweep_table(runs, metric, caption, label, footnote, higher_is_better, percent):
    grid = sorted(runs.process_noise.unique())
    trackers = [t for t in TRACKERS if t != "DiffPF" and t in set(runs.tracker)]
    means = {}
    for name in trackers:
        for q in grid:
            r = runs[(runs.tracker == name) & (runs.process_noise == q)].sort_values("run")
            if metric == "rmse_all":
                means[name, q] = t_ci(r.rmse_all)
            elif metric == "majority":
                means[name, q] = pooled_ci(r.exit_majority, r.exit_events)
            else:
                means[name, q] = pooled_ci(r.exit_p_correct_sum, r.exit_events)
    rows = []
    for name in trackers:
        b = best_set([means[name, q][0] for q in grid], higher_is_better)
        rows.append((LABELS[name], [cell(*means[name, q], bold=(j in b), percent=percent)
                                    for j, q in enumerate(grid)]))
        if name in GROUP_ENDS and name != trackers[-1]:
            rows.append(None)
    header = ["Tracker"] + [f"$q{{=}}{q:g}$" for q in grid]
    return table(caption, label, header, rows, footnote)


def enff_sweep_table(rows, chosen):
    """Validation RMSE of EnFF over (lambda, sigma_min) at the chosen q, plus the chosen row."""
    q = chosen["process_noise"]
    r = rows[np.isclose(rows.process_noise, q)]
    lams, smins = sorted(r.enff_lambda.unique()), sorted(r.enff_sigma_min.unique())
    best = r.rmse_all.min()
    body = []
    for smin in smins:
        cells = []
        for lam in lams:
            v = r[np.isclose(r.enff_lambda, lam) & np.isclose(r.enff_sigma_min, smin)].rmse_all.iloc[0]
            cells.append(f"$\\mathbf{{{sig3(v)}}}$" if np.isclose(v, best) else f"${sig3(v)}$")
        body.append((f"$\\sigma_{{\\min}}{{=}}{smin:g}$", cells))
    header = ["EnFF"] + [f"$\\lambda{{=}}{sig3(lam)}$" for lam in lams]
    return table(
        f"EnFF hyperparameter selection on validation runs: mean RMSE over all steps (m, lower is "
        f"better) for guidance scale $\\lambda$ and probability-path width $\\sigma_{{\\min}}$ at the "
        f"chosen process noise $q{{=}}{q:g}$\\,m. Same runs as Table~\\ref{{tab:noise-sweep-rmse}}.",
        "tab:enff-sweep", header, body,
        "$q$, $\\lambda$ and $\\sigma_{\\min}$ were searched jointly over "
        f"{r.shape[0] * rows.process_noise.nunique()} settings ($q$ grid as in "
        "Table~\\ref{tab:noise-sweep-rmse}); bold marks the selected setting, the lowest mean "
        "RMSE all. EnFF integrates its flow with "
        f"{int(chosen.get('enff_steps', 5)) - 1} Euler steps (sampling\\_time\\_step\\_count 5, as "
        "in the DAFM benchmarks).")


def road_network_figure(lane_ok, n_ev):
    return "\n".join([
        "\\begin{figure}[ht]", "\\centering",
        "\\includegraphics[width=\\linewidth]{road_network.pdf}",
        "\\caption{Benchmark road network as a Markov chain over road nodes (positions in metres). "
        "The target drives the edges at 0.03\\,m per step and draws its next node on arrival; "
        "unlabelled edges have probability 1. At node 4 it picks the north lane (4$\\to$5$\\to$6, "
        "$y{=}{+}0.15$\\,m) or the south lane (4$\\to$8$\\to$9, $y{=}{-}0.15$\\,m) with probability "
        "0.5 each. The T-junction (nodes 6 and 9) lies inside occlusion 0, so the turn happens while "
        "the target is hidden; it turns the way its lane signals with probability 0.9. These "
        f"junction occlusions are the {n_ev} exit-prediction events of "
        "Tables~\\ref{tab:exit-benchmark-tuned} and~\\ref{tab:exit-benchmark-shared}, where the "
        f"lane cue predicts the exit in {lane_ok} ({100 * lane_ok / n_ev:.1f}\\%). Node 1 is a "
        "visible 50/50 junction; occlusion 1 covers a straight road with a single exit and is not "
        "scored. Right: zoom on the occluded T-junction.}",
        "\\label{fig:road-network}", "\\end{figure}", "",
    ])


def main():
    parser = argparse.ArgumentParser(description="Write the exit-prediction benchmark LaTeX tables")
    parser.add_argument("--root", default=str(motion_resource_dir() / "exit_benchmark"))
    parser.add_argument("--out", default=None, help="Output .tex (default <root>/benchmark_tables.tex)")
    parser.add_argument("--tuned_workers", type=int, default=24,
                        help="Benchmark processes sharing the CPU for results/ (runtime footnote)")
    parser.add_argument("--shared_workers", type=int, default=24,
                        help="Benchmark processes sharing the CPU for results_noise0.01/")
    args = parser.parse_args()
    root = args.root
    out_path = args.out or os.path.join(root, "benchmark_tables.tex")

    tuned_runs = pd.read_csv(os.path.join(root, "results/exit_benchmark_per_run.csv"))
    tuned_events = pd.read_csv(os.path.join(root, "results/exit_benchmark_events.csv"))
    shared_runs = pd.read_csv(os.path.join(root, "results_noise0.01/exit_benchmark_per_run.csv"))
    shared_events = pd.read_csv(os.path.join(root, "results_noise0.01/exit_benchmark_events.csv"))
    sweep_runs = pd.read_csv(os.path.join(root, "noise_sweep/noise_sweep_runs.csv"))
    enff_path = os.path.join(root, "enff_sweep/enff_sweep.csv")
    enff_rows = pd.read_csv(enff_path) if os.path.exists(enff_path) else None
    trackers = [t for t in TRACKERS if t in set(tuned_runs.tracker)]

    lane_ev = tuned_events[tuned_events.tracker == TRACKERS[0]]
    lane_ok, n_ev = int((lane_ev.lane_side == lane_ev.actual_side).sum()), len(lane_ev)
    n_runs = tuned_runs.run.nunique()
    v_runs = sweep_runs.run.unique()
    setup = (f"{n_runs} seeded runs of 2000 steps (runs 0--{n_runs - 1}), identical trajectories, "
             "measurement noise and filter seeds for every tracker; 500 particles per filter. "
             f"Exit metrics are pooled over the {n_ev} junction occlusions (occlusion box with two "
             "possible exits; Fig.~\\ref{fig:road-network}). ")
    exit_def = ("Exit top-1: the side of the box holding the largest share of particle mass, "
                "predicting without measurements until 10 steps after the target re-appears, is "
                "the side the target exits through; Exit $>$50\\%: more than half the mass is "
                "beyond that side; $P$(correct exit): mean mass beyond that side. ")
    ci_def = ("Mean $\\pm$ 95\\% CI: $t$-interval over runs for RMSE and runtime; run-level "
              "bootstrap for exit metrics.")
    ceiling = (f"The approach-lane cue predicts the exit in {lane_ok}/{n_ev} "
               f"({100 * lane_ok / n_ev:.1f}\\%) events, the best achievable exit success; chance is 50\\%. ")
    bold_note = "Bold marks the best mean per column, not a statistically significant difference."

    tuned = benchmark_columns(tuned_runs, tuned_events, trackers)
    shared = benchmark_columns(shared_runs, shared_events,
                               [t for t in trackers if t in set(shared_runs.tracker)])
    enff_note, enff_chosen = "", None
    if enff_rows is not None and "EnFF" in trackers:
        enff_chosen = enff_rows.loc[enff_rows.rmse_all.idxmin()]
        enff_note = (f"EnFF (DAFM Ensemble Flow Filter) uses guidance scale "
                     f"$\\lambda{{=}}{enff_chosen.enff_lambda:g}$ and path width "
                     f"$\\sigma_{{\\min}}{{=}}{enff_chosen.enff_sigma_min:g}$, chosen with its $q$ by the "
                     "same rule (Table~\\ref{tab:enff-sweep}); its constant-velocity forecast plays the "
                     "role of PF+CV's proposal. ")
    new_note = ("PF+TrajFlow, PF+MoFlow and PF+TCFM are ports of the open-source trajectory forecasters "
                "trained on the same measurement stream as the other learned models; each particle "
                "proposes the first step of one forecast drawn by the method's own mode probabilities. "
                if any(t in trackers for t in ("PF+TrajFlow", "PF+MoFlow", "PF+TCFM")) else "")
    parts = [
        "% Generated by mml_research.motion.make_benchmark_tables.",
        "% Needs \\usepackage{booktabs,adjustbox,graphicx}.",
        "",
        road_network_figure(lane_ok, n_ev),
        benchmark_table(
            tuned,
            "Filter-level tracking and exit-prediction benchmark with per-tracker process noise "
            "(lower RMSE and runtime, higher exit metrics are better). " + setup + ci_def,
            "tab:exit-benchmark-tuned",
            "Process noise $q$ (per-step position jitter, m) chosen for each particle filter by "
            f"lowest RMSE all on separate validation runs {v_runs.min()}--{v_runs.max()} "
            "(Table~\\ref{tab:noise-sweep-rmse}); DiffPF uses its built-in propagation noise. "
            + enff_note + new_note + exit_def + ceiling +
            f"Runtime measured with {args.tuned_workers} benchmark processes sharing the CPU, so it is "
            "comparable within this table only. " + bold_note,
            with_noise=True),
        benchmark_table(
            shared,
            "Same benchmark with one shared process noise $q{=}0.01$\\,m for every particle "
            "filter" + (" and EnFF" if "EnFF" in trackers else "")
            + " (lower RMSE and runtime, higher exit metrics are better). " + setup + ci_def,
            "tab:exit-benchmark-shared",
            "PF+DMMN gives every particle the same velocity, so at $q{=}0.01$ the cloud cannot "
            "follow the target through the visible junction and loses track in most runs. "
            + (f"EnFF keeps its tuned $\\lambda$ and $\\sigma_{{\\min}}$. " if enff_note else "")
            + exit_def + ceiling +
            f"Runtime measured with {args.shared_workers} benchmark processes sharing the CPU. " + bold_note,
            with_noise=False),
        sweep_table(
            sweep_runs, "rmse_all",
            "Process-noise sweep on validation runs: RMSE over all steps (m, lower is better) "
            f"for per-step process noise $q$ (m). {len(v_runs)} runs of 2000 steps "
            f"(runs {v_runs.min()}--{v_runs.max()}, disjoint from the benchmark runs); mean "
            "$\\pm$ 95\\% $t$-interval over runs.",
            "tab:noise-sweep-rmse",
            "Bold marks the lowest mean per row: the selection rule, fixed before the sweep, takes it; those "
            "values are used in Table~\\ref{tab:exit-benchmark-tuned}. DiffPF has no process-noise "
            "parameter and is not swept"
            + ("; the EnFF row is at its selected $\\lambda$ and $\\sigma_{\\min}$ "
               "(Table~\\ref{tab:enff-sweep})." if enff_note else "."),
            higher_is_better=False, percent=False),
        sweep_table(
            sweep_runs, "majority",
            "Process-noise sweep on validation runs: exit success with more than half the "
            "particle mass beyond the actual exit side (\\%, higher is better). Same runs as "
            "Table~\\ref{tab:noise-sweep-rmse}; mean $\\pm$ 95\\% run-level bootstrap CI.",
            "tab:noise-sweep-exit",
            "Process noise is added on every step, including occluded ones, so a larger $q$ "
            "widens the forecast while the target is hidden; this was not used for selection. "
            "Bold marks the best mean per row, not a statistically significant difference.",
            higher_is_better=True, percent=True),
        sweep_table(
            sweep_runs, "p_correct",
            "Process-noise sweep on validation runs: mean particle mass beyond the actual exit "
            "side, $P$(correct exit) (higher is better). Same runs as "
            "Table~\\ref{tab:noise-sweep-rmse}; mean $\\pm$ 95\\% run-level bootstrap CI.",
            "tab:noise-sweep-pcorrect",
            "Not used for selection. Bold marks the best mean per row, not a statistically "
            "significant difference.",
            higher_is_better=True, percent=False),
    ]
    if enff_chosen is not None:
        parts.append(enff_sweep_table(enff_rows, enff_chosen))
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
