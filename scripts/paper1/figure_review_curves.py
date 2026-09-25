"""Point events schematically: one probability curve per class, its peak, and the events on one timeline.

Two panels side by side, on :data:`WINDOW_S` of :data:`TRIAL` in the run :data:`RUN`, one row per
class of :data:`CLASSES` (top to bottom), no numbers anywhere:

1. **Model output** — each class's curve on its own baseline, in its class colour.
2. **Peaks** — the same curves with the tallest peak marked and its focus window shaded, and under
   them one timeline carrying every class's event as a tick in its colour: the events share a
   window and a timeline, the curves never do.

Writes ``figure_review_curves.{pdf,svg,png}`` into ``figure_review.ROOT``.

    python scripts/paper1/figure_review_curves.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from figure_combined import paper_style
from figure_review import FOCUS_WINDOW_S, MAPPING, POINT_RUNS, ROOT, save, tint

from ethograph.labels.confidence import curve_stats, window_samples
from ethograph.labels.intervals import load_label_mapping
from ethograph.labels.onset_curves import CURVES_FILE, read_curves
from ethograph.labels.rescore import curve_rate

RUN = "0506"
TRIAL = 1
#: Top to bottom.
CLASSES = (31, 32)
WINDOW_S = (1.5, 2.5)
#: Inches: the whole page, and the timeline under the curves as a share of one curve's row.
FIGSIZE = (8, 2.6)
TIMELINE_RATIO = 0.35
LINE_WIDTH = 1.5
BASELINE_COLOR = "0.75"
PEAK_SIZE = 6
TICK_HEIGHT = 0.6


def load_curves() -> list[tuple[int, np.ndarray, np.ndarray, int]]:
    """Per class: ``(label, time, curve, peak index)`` on :data:`WINDOW_S`, the peak found on the whole curve."""
    time, per_label = read_curves(POINT_RUNS[RUN] / CURVES_FILE)[str(TRIAL)]
    shown = (time >= WINDOW_S[0]) & (time <= WINDOW_S[1])
    out = []
    for label in CLASSES:
        curve = per_label[label]
        stats = curve_stats(curve, window_samples(FOCUS_WINDOW_S, curve_rate(time)))
        if not stats.found or not shown[stats.index]:
            raise ValueError(f"class {label} of trial {TRIAL} has no peak inside {WINDOW_S}")
        out.append((label, time[shown], curve[shown], int(stats.index - np.flatnonzero(shown)[0])))
    return out


def bare(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_curve(ax: plt.Axes, time: np.ndarray, curve: np.ndarray, color, peak: int | None) -> None:
    """One class's curve over a grey baseline; with *peak*, its focus window shaded and the peak dotted."""
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1, zorder=0)
    if peak is not None:
        near = np.abs(time - time[peak]) <= FOCUS_WINDOW_S
        ax.fill_between(time, 0, curve, where=near, facecolor=tint(color), edgecolor="none", zorder=1)
        ax.plot([time[peak]], [curve[peak]], "o", color=color, markersize=PEAK_SIZE, zorder=3)
    ax.plot(time, curve, color=color, linewidth=LINE_WIDTH, zorder=2)
    ax.set_xlim(*WINDOW_S)
    ax.set_ylim(-0.05, 1.15)
    bare(ax)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    mapping = load_label_mapping(MAPPING)
    curves = load_curves()
    for label, time, _, peak in curves:
        print(f"{mapping[label]['name']}: peak at {time[peak]:.3f} s")

    with paper_style():
        fig = plt.figure(figsize=FIGSIZE)
        heights = [1.0] * len(curves) + [TIMELINE_RATIO]
        grid = fig.add_gridspec(len(heights), 2, height_ratios=heights, wspace=0.15, hspace=0.1)
        for row, (label, time, curve, peak) in enumerate(curves):
            color = mapping[label]["color"]
            draw_curve(fig.add_subplot(grid[row, 0]), time, curve, color, peak=None)
            draw_curve(fig.add_subplot(grid[row, 1]), time, curve, color, peak=peak)
        # One timeline for every class's event.
        timeline = fig.add_subplot(grid[-1, 1])
        timeline.axhline(0, color=BASELINE_COLOR, linewidth=1, linestyle="--", zorder=0)
        for label, time, _, peak in curves:
            timeline.vlines(time[peak], -TICK_HEIGHT, TICK_HEIGHT, color=mapping[label]["color"], linewidth=2)
        timeline.set_xlim(*WINDOW_S)
        timeline.set_ylim(-1, 1)
        bare(timeline)
        fig.add_subplot(grid[-1, 0]).set_axis_off()
    save(fig, ROOT / "figure_review_curves")


if __name__ == "__main__":
    main()
