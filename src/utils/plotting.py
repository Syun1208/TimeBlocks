from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

ACADEMIC_PALETTE = ("#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3", "#937860")
ACADEMIC_MARKERS = ("o", "s", "^", "D", "v", "P")
ACADEMIC_LINESTYLES = ("-", "--", "-.", ":", "-", "--")
FIGURE_DPI = 300
SINGLE_COLUMN_WIDTH = 3.5
CLIQUE_LINE_WIDTH = 3.0
TEXT_COLOUR = "#222222"
FIGURE_SUFFIXES = (".pdf", ".png")


def apply_academic_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=ACADEMIC_PALETTE, marker=ACADEMIC_MARKERS, linestyle=ACADEMIC_LINESTYLES),
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
        }
    )


def save_figure(
    *,
    figure: "matplotlib.figure.Figure",
    path: Path,
) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in FIGURE_SUFFIXES:
        target = path.with_suffix(suffix)
        figure.savefig(target)
        saved.append(target)
    return saved


def plot_critical_difference(
    *,
    models: Sequence[str],
    average_ranks: Sequence[float],
    cliques: Sequence[Sequence[str]],
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    count = len(models)
    order = list(np.argsort(average_ranks))
    half = (count + 1) // 2
    figure, axis = plt.subplots(figsize=(2 * SINGLE_COLUMN_WIDTH, 0.6 + 0.3 * half + 0.25 * len(cliques)))
    axis.set_xlim(count + 0.5, 0.5)
    axis.set_ylim(-(half + 0.5 * len(cliques) + 1), 1)
    axis.axis("off")
    axis.hlines(0, 1, count, color=TEXT_COLOUR, linewidth=1.0)
    for tick in range(1, count + 1):
        axis.vlines(tick, 0, 0.15, color=TEXT_COLOUR, linewidth=1.0)
        axis.text(tick, 0.3, str(tick), ha="center", va="bottom", color=TEXT_COLOUR)
    for position, index in enumerate(order):
        rank = average_ranks[index]
        best_side = position < half
        level = -(position + 1) if best_side else -(count - position)
        anchor = 0.6 if best_side else count + 0.4
        axis.plot([rank, rank, anchor], [0, level, level], color=TEXT_COLOUR, linewidth=0.8, linestyle="-", marker="")
        axis.text(
            anchor - 0.05 if best_side else anchor + 0.05,
            level,
            f"{models[index]} ({rank:.2f})",
            ha="left" if best_side else "right",
            va="center",
            color=TEXT_COLOUR,
        )
    rank_of = {models[index]: average_ranks[index] for index in range(count)}
    for depth, clique in enumerate(cliques):
        ranks = [rank_of[name] for name in clique]
        level = -(half + 0.5 + depth * 0.5)
        axis.hlines(level, min(ranks) - 0.05, max(ranks) + 0.05, color=TEXT_COLOUR, linewidth=CLIQUE_LINE_WIDTH)
    return figure


def plot_metric_curves(
    *,
    curves: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    x_label: str,
    y_label: str,
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.4))
    for label, (x_values, y_values) in curves.items():
        axis.plot(x_values, y_values, label=label)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(frameon=False)
    return figure
