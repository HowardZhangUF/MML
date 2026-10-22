#!/usr/bin/env python3
"""
plot_road_network.py
====================
Draw the benchmark road network (generate_synthetic_data.py) as a Markov
chain: nodes at their positions, edges with transition probabilities,
occlusion boxes, and a zoom on the occluded T-junction with its lane cue.

Usage (after installing the project):
    python -m mml_research.motion.plot_road_network --out <path without extension>
Writes <out>.png and <out>.pdf.
"""

from mml_core.config import motion_resource_dir

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from mml_research.motion.generate_synthetic_data import (
    NODE_POS, TRANS_MATRIX, OCC_CENTERS, OCC_WIDTHS, LANE_OFFSET, CUE_RELIABILITY, SPEED,
)

INK = "#0b0b0b"
INK_2 = "#52514e"
EDGE = "#8a8984"
NORTH = "#2a78d6"   # categorical slot 1
SOUTH = "#eb6834"   # categorical slot 2
OCC_FILL = "#e9e8e4"

NORTH_NODES, SOUTH_NODES = {5, 6}, {8, 9}


def edge_color(i, j):
    if i in NORTH_NODES or (i == 4 and j in NORTH_NODES):
        return NORTH
    if i in SOUTH_NODES or (i == 4 and j in SOUTH_NODES):
        return SOUTH
    return EDGE


def draw(ax, node_r, label_size, prob_size, show_box_labels):
    for k, (c, w) in enumerate(zip(OCC_CENTERS, OCC_WIDTHS)):
        ax.add_patch(Rectangle(c - w / 2, w, w, facecolor=OCC_FILL, edgecolor=INK_2,
                               linestyle="--", linewidth=1.0, zorder=0))
        if show_box_labels:
            if k == 0:   # below-left of the junction box
                ax.text(c[0] - w / 2 - 0.03, c[1] - w / 2, "occlusion 0\n(T-junction)", ha="right",
                        va="bottom", fontsize=label_size - 2, color=INK_2, zorder=1, clip_on=True)
            else:        # above the box
                ax.text(c[0], c[1] + w / 2 + 0.03, "occlusion 1", ha="center", va="bottom",
                        fontsize=label_size - 2, color=INK_2, zorder=1, clip_on=True)

    for i in range(len(NODE_POS)):
        for j in np.flatnonzero(TRANS_MATRIX[i]):
            p = TRANS_MATRIX[i, j]
            col = edge_color(i, j)
            ax.add_patch(FancyArrowPatch(
                NODE_POS[i], NODE_POS[j], arrowstyle="-|>", mutation_scale=12,
                shrinkA=node_r, shrinkB=node_r, color=col,
                linewidth=2.0 if p >= 0.5 else 1.2,
                linestyle="-" if p >= 0.5 else (0, (4, 2)), zorder=2))
            if p < 1.0:
                a, b = NODE_POS[i], NODE_POS[j]
                t = 0.62 if p >= 0.5 else 0.38
                pos = a + t * (b - a)
                ax.text(*pos, f"{p:g}", fontsize=prob_size, color=INK, ha="center", va="center",
                        clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=col,
                                  linewidth=1.0), zorder=4)

    for i, (x, y) in enumerate(NODE_POS):
        ring = NORTH if i in NORTH_NODES else SOUTH if i in SOUTH_NODES else INK
        ax.plot(x, y, "o", markersize=node_r * 1.15, markerfacecolor="white",
                markeredgecolor=ring, markeredgewidth=1.6, zorder=5)
        ax.text(x, y, str(i), fontsize=label_size, color=INK, ha="center", va="center", zorder=6,
                clip_on=True)


def main():
    parser = argparse.ArgumentParser(description="Plot the benchmark road network")
    parser.add_argument("--out", default=str(motion_resource_dir() / "exit_benchmark" / "road_network"))
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(13, 5.6),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    draw(ax, node_r=11, label_size=11, prob_size=9, show_box_labels=True)
    ax.set_xlim(-2.0, 2.6)
    ax.set_ylim(-1.45, 1.45)
    ax.set_title("Benchmark road network (Markov chain over road nodes)", fontsize=12, color=INK)

    draw(zoom, node_r=13, label_size=12, prob_size=10, show_box_labels=True)
    zoom.set_xlim(-0.05, 2.45)
    zoom.set_ylim(-1.1, 1.1)
    zoom.set_title("Zoom: occluded T-junction and lane cue", fontsize=12, color=INK)
    zoom.text(0.95, LANE_OFFSET + 0.04, f"north lane  y = +{LANE_OFFSET:g} m", ha="center",
              va="bottom", fontsize=10, color=INK)
    zoom.text(0.95, -LANE_OFFSET - 0.04, f"south lane  y = −{LANE_OFFSET:g} m", ha="center",
              va="top", fontsize=10, color=INK)

    for a in (ax, zoom):
        a.set_aspect("equal")
        a.set_xlabel("x (m)", color=INK_2)
        a.set_ylabel("y (m)", color=INK_2)
        a.tick_params(colors=INK_2, labelsize=9)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#c9c8c3")
        a.grid(color="#eeeeec", linewidth=0.6, zorder=-1)

    handles = [
        plt.Line2D([], [], color=NORTH, lw=2, label="north lane (4→5→6)"),
        plt.Line2D([], [], color=SOUTH, lw=2, label="south lane (4→8→9)"),
        plt.Line2D([], [], color=EDGE, lw=2, label="other roads"),
        plt.Line2D([], [], color=INK_2, lw=1.2, ls=(0, (4, 2)), label="low-probability turn"),
        Rectangle((0, 0), 1, 1, facecolor=OCC_FILL, edgecolor=INK_2, ls="--", label="occlusion box"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.text(0.5, 0.07,
             f"Unlabelled edges have probability 1. At node 4 the target picks a lane 50/50; inside "
             f"occlusion 0 it turns the way its lane signals with probability {CUE_RELIABILITY:g}. "
             f"Speed {SPEED:g} m/step; the next node is drawn on arrival.",
             ha="center", fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.out}.png and {args.out}.pdf")


if __name__ == "__main__":
    main()
