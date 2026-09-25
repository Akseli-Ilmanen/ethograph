"""One reviewed trial over its speed: the review figure's trial panel with a movement trace under it.

Two panels, top to bottom, sharing the time axis of :data:`TRIAL`'s window in
``figure_review.STATE_TRIALS``:

1. The trial as ``figure_review`` draws it, without its knobs — the ground-truth strip, then the
   predicted segments under the per-frame confidence.
2. :data:`TRACE_VAR` of :data:`KEYPOINT`, coloured frame by frame by :data:`COLOR_VAR` (movement
   direction), the way ``figure_combined`` draws its trace.

Writes ``figure_review_speed.{pdf,svg,png}`` into ``figure_review.ROOT``.

    python scripts/paper1/figure_review_speed.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from figure_combined import WIDTH, paper_style
from figure_review import MAPPING, ROOT, clip_segments, draw_state_trial, load_state, save
from matplotlib.collections import LineCollection

import ethograph as eto
from ethograph.labels.intervals import load_label_mapping
from ethograph.labels.plots import plot_label_segments

SESSION = (
    r"C:\Users\aksel\Documents\AK_data\derivatives\sub-03_id-Freddy"
    r"\ses-000_date-20250526_01\behav\Trial_data3.nc"
)
#: The trial drawn; one of ``figure_review.STATE_TRIALS``, whose window it takes.
TRIAL = 97
PIN = {"keypoint": "beakTip", "individual": "Freddy"}
TRACE_VAR = "speed"
#: A per-frame RGB variable: the trace's colour, frame by frame.
COLOR_VAR = "angle_rgb"
#: Inches: the ground-truth strip, the trial panel, the trace.
HEIGHTS = (0.4, 3.5, 2.0)


def load_trace(t0: float, t1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time (T,), :data:`TRACE_VAR` (T,) and :data:`COLOR_VAR` (T, 3) at :data:`PIN`, cut to ``[t0, t1]``."""
    ds = eto.open(SESSION).trial(TRIAL)
    time_coord = eto.get_time_coord(ds[TRACE_VAR])
    if time_coord is None:
        raise ValueError(f"{TRACE_VAR} of trial {TRIAL} has no time coordinate")
    time = np.asarray(time_coord.values, dtype=np.float64)
    trace, _ = eto.sel_valid(ds[TRACE_VAR], PIN)
    colors, _ = eto.sel_valid(ds[COLOR_VAR], PIN)
    shown = (time >= t0) & (time <= t1)
    return time[shown], np.asarray(trace, dtype=np.float64)[shown], np.asarray(colors, dtype=np.float64)[shown]


def draw_trace(ax: plt.Axes, time: np.ndarray, trace: np.ndarray, colors: np.ndarray) -> None:
    """One segment per frame, coloured by the frame it starts on."""
    points = np.column_stack([time, trace])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    valid = np.isfinite(segments).all(axis=(1, 2)) & np.isfinite(colors[:-1]).all(axis=1)
    ax.add_collection(LineCollection(segments[valid], colors=colors[:-1][valid], linewidths=1.2))
    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_ylim(0, float(np.nanmax(trace)) * 1.05)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(TRACE_VAR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    mapping = load_label_mapping(MAPPING)
    state = load_state(mapping)
    reviewed = next(r for r in state.reviewed if r.trial == TRIAL)
    t0, t1 = reviewed.window
    time, trace, colors = load_trace(t0, t1)

    with paper_style():
        fig = plt.figure(figsize=(WIDTH, sum(HEIGHTS)))
        truth_ax, trial_ax, trace_ax = fig.subplots(
            3, 1, sharex=True, height_ratios=HEIGHTS, gridspec_kw={"left": 0.08, "right": 0.88, "hspace": 0.08}
        )
        # Opaque: CorelDRAW mishandles a PDF's transparency and inverts the page's colours.
        truth = clip_segments(state.truth[state.truth["trial"] == TRIAL], t0, t1)
        plot_label_segments(truth_ax, truth, mapping, alpha=1)
        truth_ax.set_yticks([])
        truth_ax.tick_params(bottom=False, labelbottom=False)
        for spine in truth_ax.spines.values():
            spine.set_visible(False)
        draw_state_trial(trial_ax, state.segments, reviewed, mapping, knobs=False)
        trial_ax.set_xlabel("")
        trial_ax.set_yticks([])
        trial_ax.tick_params(labelbottom=False)
        draw_trace(trace_ax, time, trace, colors)
    save(fig, ROOT / "figure_review_speed")


if __name__ == "__main__":
    main()
