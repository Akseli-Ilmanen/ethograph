"""The review figure: what a human controls when checking a model's predictions, per event kind.

One row per reviewed state trial and one for point events, each read left to right from the
model's output to what the reviewer sees in depth (one trial, or one event frame by frame):

1. **State events** (the segmentation model in :data:`STATE_RUN`), one row per entry of
   :data:`STATE_TRIALS`. Left: the model's output on the trial's window — the logits as a grey
   ``C × T`` heatmap (the
   run keeps its softmax, whose log is the logits up to a per-frame constant), the softmax with
   each class's row in its colour and opacity ``p ** γ`` (:data:`GAMMA`, so the small values
   show; the colourbar's ticks read true probability), the argmax strip, and the per-frame
   confidence. Right: the
   trial reviewed trial by trial — the ground truth (:data:`GROUND_TRUTH`) as a strip of class
   colours on top, and under it the per-frame confidence ``1 − H(p) / log C`` over the
   predicted segments on the same window, from :data:`STATE_Y_MIN` up, the two knobs as
   horizontal lines: :data:`FRAME_THRESHOLD` (the trial's mean is judged against it, printed
   here, not drawn; the mean over the shown window is dotted) and :data:`SEGMENT_THRESHOLD`,
   with a red dot on every frame below it.
2. **Point events** (the pixel spotter's runs in :data:`POINT_RUNS`). Left: three events reviewed
   frame by frame (:data:`EXEMPLARS`, each naming its run, trial and class), each its class curve
   — :data:`ZOOM_HALF_S` either side of the peak, or the trial-time window the exemplar names —
   with the focus window shaded and the rival peak marked, and its ``focus``, ``ratio`` and their
   product. Right: the trade-off knob — the confidence each exemplar reads under
   ``ratio × (α + (1 − α) · focus)`` as α runs from the product (0) to the ratio alone (1), against
   :data:`FLAG_THRESHOLD`.

Writes ``figure_review.{pdf,png}`` into :data:`ROOT`.

    python scripts/paper1/figure_review.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_combined import WIDTH, paper_style
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
from matplotlib.figure import SubFigure
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator

from ethograph.labels.confidence import CurveStats, curve_stats, entropy_confidence, segment_confidence, window_samples
from ethograph.labels.intervals import LABELING_AUTOMATED, load_label_mapping
from ethograph.labels.onset_curves import CURVES_FILE, read_curves
from ethograph.labels.plots import plot_label_segments
from ethograph.labels.predictions import PredictionsStore
from ethograph.labels.rescore import curve_rate, rule_value
from ethograph.labels.tsv_store import load_labels_tsv

ROOT = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\paper\figure1")
MAPPING = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\crowlab\mapping.txt")

# --- Row 1: state events ------------------------------------------------------------------------

#: One saved segmentation prediction set: its TSV is the segments, its ``_probs.npz`` the curve.
STATE_RUN = ROOT / "predictions_fold-Trial_data3-12f7f748_20260902-0000_20260902_002243"
#: The session's human labels (a labels TSV), drawn as a strip over each trial panel.
GROUND_TRUTH = STATE_RUN / "Trial_data_groundtruth.tsv"
#: The trials reviewed in depth, each with the stretch (s) its model output and its trial panel
#: draw; one row of the figure per entry.
STATE_TRIALS: dict[int, tuple[float, float]] = {33: (0.5, 3.5), 97: (0.5, 4.45)}
#: The GUI's two state-event knobs (``widgets_io.py``): a frame below the first is marked, and a
#: trial whose mean confidence falls below it, or any of whose segments' mean falls below the
#: second, is flagged for review. The GUI opens at 0.75 / 0.6; this run is confident nearly
#: everywhere, so the figure sets both where they bite.
FRAME_THRESHOLD = 0.9
SEGMENT_THRESHOLD = 0.8
#: The trial panel's y axis starts here: the curve never leaves the top half.
STATE_Y_MIN = 0.5
#: Opacity of a softmax cell is ``p ** GAMMA``: 1 is linear, lower lifts the small values.
GAMMA = 0.3
#: Probabilities below this draw as blank in the logits heatmap (float16 on disk floors at 6e-8).
LOGIT_FLOOR = 1e-6
#: The colourbar's ticks, in probability.
GAMMA_TICKS = (0, 0.001, 0.01, 0.1, 0.5, 1)
BACKGROUND_COLOR = "0.45"

# --- Row 2: point events ------------------------------------------------------------------------

#: Saved pixel-spotter runs by a short name: each TSV is its events, its ``onset_curves.npz`` the
#: curves. The grid pools them all; an exemplar names the run it comes from.
POINT_RUNS: dict[str, Path] = {
    "0506": Path(
        r"C:\Users\aksel\Documents\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250506_02"
        r"\behav\labels\predictions_spot_ctx2s_res10ms_features_msagsm_20260828_120613"
    ),
    "0512": Path(
        r"C:\Users\aksel\Documents\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250512_01"
        r"\behav\labels\predictions_spot_ctx2s_res10ms_features_msagsm_20260828_131154"
    ),
}
#: The focus half-width the run scored its curves with (its ``infer.focus_window_ms``).
FOCUS_WINDOW_S = 0.1
#: The GUI's point-event knob (the Curation section's flag threshold).
FLAG_THRESHOLD = 0.5
#: Name → (run, trial, class, window) of the events reviewed frame by frame: one clean bump, one
#: smeared bump, one with a rival peak, each on a ``(t0, t1)`` window of the trial's clock
#: (``None`` would take :data:`ZOOM_HALF_S` either side of the peak instead).
EXEMPLARS: dict[str, tuple[str, int, int, tuple[float, float] | None]] = {
    "one bump": ("0506", 61, 31, (1.5, 3.5)),
    "smeared": ("0506", 5, 32, (2.3, 4.3)),
    "rival": ("0512", 52, 31, (3.0, 5.0)),
}
#: Seconds drawn either side of the peak, for an exemplar without a window of its own.
ZOOM_HALF_S = 1.0
#: Least vertical distance between two direct labels in the trade-off panel (confidence units).
LABEL_GAP = 0.05
#: The α values the trade-off panel is drawn over.
ALPHAS = np.linspace(0.0, 1.0, 101)

# --- Style --------------------------------------------------------------------------------------

FLAG_COLOR = "red"
#: A class colour is tinted this far towards white where it fills an area under a curve.
TINT = 0.55
CURVE_COLOR = "black"
#: Inches: every state row, then the point row.
STATE_HEIGHT = 4.5
POINT_HEIGHT = 5.0
#: Gap between an arrow's ends and the panels either side, as a fraction of the page width.
ARROW_PAD = 0.006


def tint(color) -> tuple[float, float, float]:
    """*color* mixed :data:`TINT` of the way to white — opaque, so CorelDRAW keeps the page's colours."""
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * TINT)


def arrow(fig: plt.Figure, left: plt.Axes, right: plt.Axes) -> None:
    """A right arrow from *left*'s drawn extent (labels included) to *right*'s, at *right*'s mid-height."""
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    to_figure = fig.transFigure.inverted()
    left_box, right_box = left.get_tightbbox(renderer), right.get_tightbbox(renderer)
    if left_box is None or right_box is None:
        raise ValueError("an arrow's panel has nothing drawn")
    x0 = to_figure.transform(left_box.max)[0] + ARROW_PAD
    x1 = to_figure.transform(right_box.min)[0] - ARROW_PAD
    y = float(to_figure.transform(right.get_window_extent(renderer)).mean(axis=0)[1])
    fig.add_artist(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.5,
            color="black",
        )
    )


# --- Row 1: state events ------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewedTrial:
    """One trial of :data:`STATE_TRIALS` on the run's clock: its softmax, the confidence read off it, its window."""

    trial: int
    time: np.ndarray
    confidence: np.ndarray
    probs: np.ndarray
    window: tuple[float, float]


@dataclass(frozen=True)
class StateData:
    #: Every segment the run predicted; ``confidence`` is the mean of the per-frame curve over it.
    segments: pd.DataFrame
    #: The human state labels.
    truth: pd.DataFrame
    reviewed: list[ReviewedTrial]
    #: The label id of each column of every trial's softmax.
    column_ids: list[int]


def load_state(mapping: dict) -> StateData:
    """The run's segments, each re-scored as its mean of the per-frame entropy confidence.

    The run's TSV carries the mean of the segment's own class probability instead; the review
    overlay draws the entropy curve, and this figure's segment means are read off that same
    curve (:func:`segment_confidence`, as the GUI's review PDF does) so the two agree on the page.
    """
    store = PredictionsStore(STATE_RUN)
    if store.npz_path is None:
        raise FileNotFoundError(f"{STATE_RUN} has no *_probs.npz to read the confidence from")
    segments = load_labels_tsv(store.tsv_path)
    segments["confidence"] = np.nan
    reviewed: dict[int, ReviewedTrial] = {}
    with np.load(store.npz_path) as npz:
        for key in npz.files:
            if key.endswith(("_time", "_boundary")):
                continue
            trial = int(key.split("_trial")[1].split("_")[0])
            # float16 on disk: the entropy's epsilon underflows unless widened first.
            probs = np.asarray(npz[key], dtype=np.float64)
            time = np.asarray(npz[f"{key}_time"], dtype=np.float64)
            confidence = entropy_confidence(probs)
            rows = segments.index[segments["trial"] == trial]
            for idx in rows:
                onset, offset = segments.at[idx, "onset_s"], segments.at[idx, "offset_s"]
                segments.at[idx, "confidence"] = segment_confidence(confidence, time, onset, offset)
            if trial in STATE_TRIALS:
                reviewed[trial] = ReviewedTrial(trial, time, confidence, probs, STATE_TRIALS[trial])
    missing = set(STATE_TRIALS) - set(reviewed)
    if missing:
        raise ValueError(f"{store.npz_path} never predicted trials {sorted(missing)}")
    if segments["confidence"].isna().any():
        raise ValueError(f"{store.tsv_path} has segments of trials the {store.npz_path.name} never predicted")
    column_ids = None
    for trial in STATE_TRIALS:
        ids = probability_columns(
            mapping, reviewed[trial].probs, reviewed[trial].time, segments[segments["trial"] == trial]
        )
        if column_ids is not None and ids != column_ids:
            raise ValueError("the reviewed trials disagree on the probability columns")
        column_ids = ids
    if column_ids is None:
        raise ValueError("STATE_TRIALS is empty")
    truth = load_labels_tsv(GROUND_TRUTH)
    truth = truth[(truth["event_type"] == "state") & (truth["labeling_method"] != LABELING_AUTOMATED)]
    return StateData(
        segments=segments, truth=truth, reviewed=[reviewed[t] for t in STATE_TRIALS], column_ids=column_ids
    )


def probability_columns(mapping: dict, probs: np.ndarray, time: np.ndarray, segments: pd.DataFrame) -> list[int]:
    """The label id each column of *probs* stands for: the mapping's state classes, background first.

    That is the run's own class table (``segment.samples.class_table``), which the run folder
    does not keep — so the order is checked against the run's segments: the argmax over every
    segment must be the segment's label.
    """
    ids = [0, *sorted(k for k, v in mapping.items() if k != 0 and v.get("event_type", "state") != "point")]
    if len(ids) != probs.shape[1]:
        raise ValueError(f"{len(ids)} state classes in the mapping but {probs.shape[1]} probability columns")
    argmax = np.asarray(ids)[probs.argmax(axis=1)]
    for row in segments.itertuples(index=False):
        inside = (time >= row.onset_s) & (time <= row.offset_s)
        if inside.any() and np.bincount(argmax[inside]).argmax() != int(row.labels):
            raise ValueError(f"column order does not reproduce the segment at {row.onset_s:.3f} s")
    return ids


def draw_model_output(subfig: SubFigure, reviewed: ReviewedTrial, column_ids: list[int], mapping: dict) -> plt.Axes:
    """Logits (grey), softmax (class colour, opacity ``p ** γ``), argmax and confidence over the trial's window."""
    window = (reviewed.time >= reviewed.window[0]) & (reviewed.time <= reviewed.window[1])
    probs, time, confidence = reviewed.probs[window], reviewed.time[window], reviewed.confidence[window]
    argmax = probs.argmax(axis=1)
    # Background and every class the argmax visits in the window, in column order.
    shown = sorted({0, *argmax.tolist()})
    colors = np.array([to_rgb(BACKGROUND_COLOR if col == 0 else mapping[column_ids[col]]["color"]) for col in shown])
    p = probs[:, shown]  # (T, C)
    rows = np.broadcast_to(colors[:, None, :], (len(shown), len(p), 3))
    rgba = np.concatenate([rows, (p.T**GAMMA)[..., None]], axis=-1)
    strip = colors[[shown.index(col) for col in argmax]][None]
    extent = (float(time[0]), float(time[-1]), len(shown) - 0.5, -0.5)

    grid = subfig.add_gridspec(4, 1, height_ratios=[4, 4, 0.6, 2], hspace=0.15)
    logit_ax, soft_ax, argmax_ax, conf_ax = (subfig.add_subplot(grid[i]) for i in range(4))
    logits = np.log(np.maximum(p.T, LOGIT_FLOOR))
    logit_ax.imshow(logits, aspect="auto", interpolation="nearest", cmap="Greys", extent=extent)
    soft_ax.imshow(rgba, aspect="auto", interpolation="nearest", extent=extent)
    argmax_ax.imshow(strip, aspect="auto", interpolation="nearest", extent=(extent[0], extent[1], 0, 1))
    conf_ax.plot(time, confidence, color=CURVE_COLOR, linewidth=0.8)
    conf_ax.set_ylim(0, 1.02)
    # No row names: they are set by hand in the vector editor.
    for ax in (logit_ax, soft_ax, argmax_ax, conf_ax):
        ax.set_xlim(extent[0], extent[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Colourbar: white → black stands in for transparent → full colour; ticks read true probability.
    cmap = LinearSegmentedColormap.from_list("alpha", [(1, 1, 1), (0, 0, 0)])
    cax = soft_ax.inset_axes((1.04, 0.0, 0.05, 1.0))
    cbar = subfig.colorbar(ScalarMappable(norm=PowerNorm(GAMMA, vmin=0, vmax=1), cmap=cmap), cax=cax)
    cbar.set_ticks(list(GAMMA_TICKS))
    cbar.ax.tick_params(labelsize=7)
    return cax


def clip_segments(segments: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    """The rows of *segments* that touch ``[t0, t1]``, their ends cut to it — so nothing is drawn past the axes."""
    inside = segments[(segments["offset_s"] > t0) & (segments["onset_s"] < t1)]
    return inside.assign(onset_s=inside["onset_s"].clip(lower=t0), offset_s=inside["offset_s"].clip(upper=t1))


def draw_state_trial(
    ax: plt.Axes, segments: pd.DataFrame, reviewed: ReviewedTrial, mapping: dict, knobs: bool = True
) -> None:
    """The reviewed trial on its window: shaded segments and the curve; with *knobs*, the low frames
    dotted, the two thresholds and the window mean too."""
    segments = segments[segments["trial"] == reviewed.trial]
    t0, t1 = reviewed.window
    for row in clip_segments(segments, t0, t1).itertuples(index=False):
        ax.axvspan(
            row.onset_s, row.offset_s, facecolor=tint(mapping[int(row.labels)]["color"]), edgecolor="none", zorder=0
        )
    shown = np.isfinite(reviewed.confidence) & (reviewed.time >= t0) & (reviewed.time <= t1)
    time, confidence = reviewed.time[shown], reviewed.confidence[shown]
    ax.plot(time, confidence, color=CURVE_COLOR, linewidth=0.8, zorder=2)
    trial_mean = float(np.nanmean(reviewed.confidence))
    if knobs:
        low = confidence < SEGMENT_THRESHOLD
        ax.plot(time[low], confidence[low], "o", color=FLAG_COLOR, markersize=2, linestyle="none", zorder=3)
        # The mean over the stretch on the page, dotted: what the frame threshold judges, on this window.
        window_mean = float(np.mean(confidence))
        ax.axhline(window_mean, color=CURVE_COLOR, linewidth=1, linestyle=":", zorder=1)
        # The higher of the two labels sits above its line, the lower below, so they never collide.
        above = window_mean > FRAME_THRESHOLD
        ax.text(t1, window_mean, " window mean", va="bottom" if above else "top", fontsize=9, color=CURVE_COLOR)
        ax.axhline(FRAME_THRESHOLD, color="0.4", linewidth=1, linestyle=":", zorder=1)
        ax.axhline(SEGMENT_THRESHOLD, color=FLAG_COLOR, linewidth=1, linestyle=":", zorder=1)
        ax.text(t1, FRAME_THRESHOLD, " frame threshold", va="top" if above else "bottom", fontsize=9, color="0.4")
        ax.text(t1, SEGMENT_THRESHOLD, " segment threshold", va="center", fontsize=9, color=FLAG_COLOR)
    n_low = int((segments["confidence"] < SEGMENT_THRESHOLD).sum())
    print(
        f"trial {reviewed.trial}: mean {trial_mean:.2f}, {n_low} of {len(segments)} segments below {SEGMENT_THRESHOLD}"
    )
    ax.set_xlim(t0, t1)
    ax.set_ylim(STATE_Y_MIN, 1.02)
    ax.set_yticks([STATE_Y_MIN, 1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("confidence")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_state(subfig: SubFigure, data: StateData, reviewed: ReviewedTrial, mapping: dict) -> list[plt.Axes]:
    """One trial's row: its panels, left to right; an arrow goes between each neighbouring pair."""
    columns = subfig.add_gridspec(1, 2, width_ratios=[1.1, 2.4], wspace=0.08)
    model, rest = subfig.add_subfigure(columns[0]), subfig.add_subfigure(columns[1])
    cax = draw_model_output(model, reviewed, data.column_ids, mapping)
    truth_ax, trial_ax = rest.subplots(
        2, 1, sharex=True, height_ratios=[0.12, 1], gridspec_kw={"left": 0.16, "right": 0.86, "hspace": 0.05}
    )
    # Opaque: CorelDRAW mishandles a PDF's transparency and inverts the page's colours.
    truth = clip_segments(data.truth[data.truth["trial"] == reviewed.trial], *reviewed.window)
    plot_label_segments(truth_ax, truth, mapping, alpha=1)
    truth_ax.set_yticks([])
    truth_ax.tick_params(bottom=False, labelbottom=False)
    for spine in truth_ax.spines.values():
        spine.set_visible(False)
    draw_state_trial(trial_ax, data.segments, reviewed, mapping)
    return [cax, trial_ax]


# --- Row 2: point events ------------------------------------------------------------------------


@dataclass(frozen=True)
class Exemplar:
    name: str
    run: str
    trial: int
    label: int
    time: np.ndarray
    curve: np.ndarray
    stats: CurveStats
    #: The stretch drawn, on the trial's clock.
    window: tuple[float, float]

    @property
    def peak_time(self) -> float:
        return float(self.time[self.stats.index])


@dataclass(frozen=True)
class PointData:
    #: Every event of every run in :data:`POINT_RUNS`, with the confidence it wrote and a ``run`` column.
    events: pd.DataFrame
    exemplars: list[Exemplar]


def load_point() -> PointData:
    events, curves = [], {}
    for run, folder in POINT_RUNS.items():
        curves[run] = read_curves(folder / CURVES_FILE)
        if not curves[run]:
            raise FileNotFoundError(f"{folder} has no readable {CURVES_FILE}")
        events.append(load_labels_tsv(PredictionsStore(folder).tsv_path).assign(run=run))
    pooled = pd.concat(events, ignore_index=True)
    exemplars = []
    for name, (run, trial, label, window) in EXEMPLARS.items():
        time, per_label = curves[run][str(trial)]
        curve = per_label[label]
        stats = curve_stats(curve, window_samples(FOCUS_WINDOW_S, curve_rate(time)))
        written = pooled[(pooled["run"] == run) & (pooled["trial"] == trial) & (pooled["labels"] == label)][
            "confidence"
        ]
        if len(written) != 1 or not np.isclose(float(written.iloc[0]), stats.shape, atol=1e-3):
            raise ValueError(f"{name}: recomputed confidence {stats.shape:.3f} is not the run's ({written.tolist()})")
        peak_time = float(time[stats.index])
        if window is None:
            window = (peak_time - ZOOM_HALF_S, peak_time + ZOOM_HALF_S)
        elif not window[0] <= peak_time <= window[1]:
            raise ValueError(f"{name}: the peak at {peak_time:.3f} s lies outside its window {window}")
        print(f"{name}: run {run} trial {trial} class {label}: {stats}")
        exemplars.append(Exemplar(name, run, trial, label, time, curve, stats, window))
    return PointData(events=pooled, exemplars=exemplars)


def draw_exemplar(ax: plt.Axes, exemplar: Exemplar, mapping: dict, time_axis: bool) -> None:
    """One event frame by frame: the curve on its window, the focus window shaded, the rival marked.

    An exemplar zoomed on its peak reads in seconds from the peak; one with a window of its own
    reads on the trial's clock.
    """
    stats, curve, peak_t = exemplar.stats, exemplar.curve, exemplar.peak_time
    absolute = exemplar.window != (peak_t - ZOOM_HALF_S, peak_t + ZOOM_HALF_S)
    origin = 0.0 if absolute else peak_t
    shown = (exemplar.time >= exemplar.window[0]) & (exemplar.time <= exemplar.window[1])
    t, curve = exemplar.time[shown] - origin, curve[shown]
    peak = peak_t - origin
    color = mapping[exemplar.label]["color"]
    near = np.abs(t - peak) <= FOCUS_WINDOW_S
    ax.fill_between(t, 0, curve, where=near, facecolor=tint(color), edgecolor="none", zorder=1)
    ax.plot(t, curve, color=color, linewidth=1.2, zorder=2)
    ax.axvline(peak - FOCUS_WINDOW_S, color="0.5", linewidth=0.8, linestyle=":")
    ax.axvline(peak + FOCUS_WINDOW_S, color="0.5", linewidth=0.8, linestyle=":")
    ax.plot([peak], [stats.peak], "o", color=color, markeredgecolor="black", markersize=6, zorder=3)
    rival = 1.0 - stats.ratio
    if rival > 0:
        ax.axhline(rival * stats.peak, color="0.5", linewidth=0.8, linestyle=":", zorder=1)
    where = f"  (trial {exemplar.trial}, {exemplar.window[0]:g}–{exemplar.window[1]:g} s)" if absolute else ""
    ax.text(
        0.02,
        0.95,
        f"{exemplar.name}{where}\nfocus {stats.focus:.2f}  ratio {stats.ratio:.2f}  product {stats.shape:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.set_xlim(exemplar.window[0] - origin, exemplar.window[1] - origin)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 1])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(labelsize=8)
    if time_axis:
        ax.set_xlabel("time (s)" if absolute else "time from peak (s)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_tradeoff(ax: plt.Axes, exemplars: list[Exemplar], mapping: dict) -> None:
    """Each exemplar's confidence as α runs from the product to the ratio alone."""
    ends = []
    for exemplar in exemplars:
        values = [rule_value(exemplar.stats, "custom", float(a)) for a in ALPHAS]
        color = mapping[exemplar.label]["color"]
        ax.plot(ALPHAS, values, color=color, linewidth=1.5)
        ends.append((values[-1], exemplar.name, color))
    # Direct labels at the right edge, pushed apart where two curves end close together.
    ends.sort()
    placed = [ends[0][0]]
    for value, _, _ in ends[1:]:
        placed.append(max(value, placed[-1] + LABEL_GAP))
    for y, (_, name, color) in zip(placed, ends, strict=True):
        ax.text(1.02, y, name, va="center", fontsize=9, color=color)
    ax.axhline(FLAG_THRESHOLD, color=FLAG_COLOR, linewidth=1, linestyle=":")
    ax.text(1.02, FLAG_THRESHOLD, "flag threshold", va="center", fontsize=9, color=FLAG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0\nfocus × ratio", "0.5", "1\nratio"])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("α")
    ax.set_ylabel("confidence")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_point(subfig: SubFigure, data: PointData, mapping: dict) -> list[plt.Axes]:
    """The row's panels, left to right; an arrow goes between each neighbouring pair."""
    grid = subfig.add_gridspec(len(data.exemplars), 2, width_ratios=[1.5, 1], wspace=0.45, left=0.12, right=0.88)
    exemplar_axs = []
    for i, exemplar in enumerate(data.exemplars):
        exemplar_axs.append(subfig.add_subplot(grid[i, 0]))
        draw_exemplar(exemplar_axs[-1], exemplar, mapping, time_axis=i == len(data.exemplars) - 1)
    tradeoff_ax = subfig.add_subplot(grid[:, 1])
    draw_tradeoff(tradeoff_ax, data.exemplars, mapping)
    # The middle exemplar stands for its column: the arrow sits at its mid-height.
    return [exemplar_axs[len(exemplar_axs) // 2], tradeoff_ax]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    mapping = load_label_mapping(MAPPING)
    state, point = load_state(mapping), load_point()
    with paper_style():
        heights = [STATE_HEIGHT] * len(state.reviewed) + [POINT_HEIGHT]
        fig = plt.figure(figsize=(WIDTH, sum(heights)))
        slots = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.08)
        rows = [
            draw_state(fig.add_subfigure(slots[i]), state, reviewed, mapping)
            for i, reviewed in enumerate(state.reviewed)
        ]
        rows.append(draw_point(fig.add_subfigure(slots[-1]), point, mapping))
        # Arrows need the panels' drawn extents, labels included, so the page is laid out first.
        fig.canvas.draw()
        for panels in rows:
            for left, right in zip(panels[:-1], panels[1:], strict=True):
                arrow(fig, left, right)
    save(fig, ROOT / "figure_review")


#: What CorelDRAW opens (``scripts/paper1/README.md``): the 14 core PDF fonts, never embedded;
#: real ``<text>`` elements in the SVG; paths unsimplified.
COREL_RC = {
    "svg.fonttype": "none",
    "pdf.use14corefonts": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "path.simplify": False,
}


def save(fig: plt.Figure, stem: Path) -> None:
    """*fig* as ``{stem}.pdf``, ``.svg`` and ``.png`` — the PDF and SVG in the form CorelDRAW opens."""
    # CorelDRAW drops any group with an SVG clip-path, so every artist inside an axes is unclipped;
    # nothing here draws outside its axes' limits.
    for ax in fig.get_axes():
        for artist in [*ax.patches, *ax.collections, *ax.lines, *ax.images]:
            artist.set_clip_on(False)
    with plt.style.context(COREL_RC), PdfPages(stem.with_suffix(".pdf")) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches=None, metadata={"Creator": "matplotlib"})
        fig.savefig(stem.with_suffix(".svg"), bbox_inches=None)
        fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.pdf')}, .svg and .png")


if __name__ == "__main__":
    main()
