#!/usr/bin/env python3
"""Make Figure 1 for the direct/indirect prior-boundary manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig01_workflow.pdf"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def panel(ax, tag, title):
    ax.set_axis_off()
    ax.text(0.02, 0.96, tag, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="top")
    ax.text(0.10, 0.96, title, transform=ax.transAxes, fontsize=8.7, fontweight="bold", va="top")


def box(ax, xy, wh, text, fc="#f7f7f7", ec="#444444", fs=7.2):
    r = Rectangle(xy, wh[0], wh[1], facecolor=fc, edgecolor=ec, lw=0.9)
    ax.add_patch(r)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        linespacing=1.15,
    )
    return r


def arrow(ax, start, end, color="#444444"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, lw=1.0, color=color))


def make():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.linewidth": 0.8,
        }
    )
    fig = plt.figure(figsize=(7.0, 5.9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.10], hspace=0.26, wspace=0.20)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    panel(ax_a, "A", "Bounded synthetic prior and simulator")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    box(ax_a, (0.06, 0.62), (0.34, 0.16), "Tectonic prior\nm", "#e9f2fb")
    box(ax_a, (0.06, 0.36), (0.34, 0.16), "Bounded support", "#fef4df")
    box(ax_a, (0.60, 0.50), (0.33, 0.16), "Surface-wave\nsimulator", "#edf5e9")
    arrow(ax_a, (0.39, 0.70), (0.57, 0.58))
    arrow(ax_a, (0.39, 0.42), (0.57, 0.54))
    ax_a.text(0.53, 0.23, "Training pairs:  m -> d", fontsize=7.2, ha="center")

    panel(ax_b, "B", "Direct learned inversion")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    box(ax_b, (0.06, 0.60), (0.27, 0.16), "Observed\ndispersion d", "#f7f7f7")
    box(ax_b, (0.43, 0.60), (0.40, 0.16), "Direct posterior\nq_theta(m | d)", "#fde8e8")
    box(ax_b, (0.43, 0.27), (0.40, 0.15), "Samples and\nsummaries", "#fde8e8")
    arrow(ax_b, (0.35, 0.67), (0.42, 0.67))
    arrow(ax_b, (0.60, 0.57), (0.60, 0.43), "#b22222")
    ax_b.text(0.53, 0.12, "Risk: prior-boundary bias\nor prior collapse", fontsize=6.9, ha="center", color="#7a1f1f")

    panel(ax_c, "C", "Indirect learned inversion")
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    box(ax_c, (0.06, 0.64), (0.30, 0.14), "Controls\nm_cp", "#f7f7f7")
    box(ax_c, (0.56, 0.64), (0.34, 0.14), "Forward surrogate\nF_phi(m)", "#e6f0fb")
    box(ax_c, (0.18, 0.28), (0.56, 0.16), "Control-point\noptimization", "#edf5e9")
    arrow(ax_c, (0.38, 0.70), (0.53, 0.70))
    arrow(ax_c, (0.70, 0.61), (0.70, 0.44))
    arrow(ax_c, (0.32, 0.43), (0.25, 0.61))
    ax_c.text(0.50, 0.10, "Not prior-free: surrogate domain,\ncontrols, initialization, regularization", fontsize=6.8, ha="center")

    panel(ax_d, "D", "Prior-boundary tests")
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    box(ax_d, (0.10, 0.65), (0.28, 0.12), "In-prior", "#e9f2fb")
    box(ax_d, (0.10, 0.45), (0.28, 0.12), "Boundary", "#fef4df")
    box(ax_d, (0.10, 0.25), (0.28, 0.12), "Out-of-prior", "#fde8e8")
    box(ax_d, (0.57, 0.42), (0.35, 0.19), "Metrics\nerror, bias\ncoverage, pull-in", "#f7f7f7")
    for y in [0.71, 0.50, 0.29]:
        arrow(ax_d, (0.39, y), (0.54, 0.52))

    panel(ax_e, "E", "Local 1-D inversions assembled into regional structure")
    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1)
    for i, x in enumerate([0.10, 0.27, 0.44]):
        box(ax_e, (x, 0.58), (0.12, 0.16), f"Local 1-D\nsite {i+1}", "#f7f7f7", fs=6.5)
        arrow(ax_e, (x + 0.06, 0.57), (x + 0.06, 0.38))
        box(ax_e, (x, 0.22), (0.12, 0.15), "Profile\nresult", "#edf5e9", fs=6.5)
    box(ax_e, (0.70, 0.36), (0.20, 0.22), "Regional 3-D\nmodel products", "#e9f2fb", fs=6.8)
    for x in [0.16, 0.33, 0.50]:
        arrow(ax_e, (x, 0.29), (0.69, 0.45))
    ax_e.text(
        0.50,
        0.08,
        "Local prior-boundary bias can accumulate into regional structural bias if many inversions share the same bounded training support.",
        ha="center",
        fontsize=6.9,
    )

    fig.savefig(OUT, bbox_inches="tight", metadata={"Creator": "make_paper_figures.py"})
    fig.savefig(ROOT / "figures" / "fig01_workflow.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    make()
