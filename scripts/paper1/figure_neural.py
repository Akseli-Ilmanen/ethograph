"""Neural decoding: curated ground truth over the cross-validated predictions, one column per trial.

For each trial in :data:`TRIALS`, three rows top to bottom, on the trial's clock: the curated
labels as coloured rectangles, the prediction set's labels the same way, and the model's
per-frame confidence. Every trial was predicted by a fold that never saw it. Only the class
legend, the row names and the time axis carry text.

A second figure, ``figure_neural_raster``, draws :data:`RASTER_TRIALS` from the trial's start:
the curated labels, the spikes of that one trial as a raster (only units firing from before the
first of these trials until after the last, so none that appeared or vanished in between), and
the predicted labels.
Units are ordered the way the GUI's heatmap sorts its rows (``gui/heatmap_sort.py``): by the
:data:`SORT_WINDOW_S` window in which the unit fires most, earliest on top — computed on
:data:`SORT_TRIAL` and applied unchanged to every other trial, so a unit keeps its row.

Writes ``figure_neural.{pdf,png}`` and ``figure_neural_raster.{pdf,png}`` into :data:`ROOT`.

    python scripts/paper1/figure_neural.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from ethograph.gui.heatmap_sort import argmax_window_order
from ethograph.io.nwb_alignment import NWBAlignment
from ethograph.labels.intervals import LABELING_AUTOMATED, load_label_mapping
from ethograph.labels.plots import plot_label_segments
from ethograph.labels.predictions import PredictionsStore, prediction_to_labels_and_confidence
from ethograph.labels.tsv_store import load_labels_tsv

ROOT = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\paper\neural")
PREDICTIONS = Path(
    r"C:\Users\aksel\Documents\Code\ethograph\projects\neural"
    r"\predictions_cv_mstcn_rf5s__rate_5ms_gauss25ms_20260903_005802"
)
GROUND_TRUTH = Path(
    r"C:\Users\aksel\Documents\AI_data\derivatives\sub-01_id-Ivy\ses-000_date-20260416_01\behav\Trial_data_labels.tsv"
)
MAPPING = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\crowlab\mapping.txt")
#: The session folder: its alignment NWB is the trial timing, its ``pynapple/units.npz`` the spikes.
SESSION = GROUND_TRUTH.parent
UNITS = SESSION / "pynapple" / "units.npz"
ALIGNMENT = SESSION / ".ethograph" / "alignment.nwb"
TRIALS = (15, 30, 47)
INDIVIDUAL = "Ivy"
#: Seconds drawn before the first and after the last label of either row.
MARGIN_S = 0.2
#: Spacing of the time ticks (s).
TICK_S = 1.0
ROWS = ("ground truth", "model", "confidence")
HEIGHT_RATIOS = (1, 1, 1.5)
WIDTH = 15
HEIGHT = 4.5
#: The class legend sits above the columns, which start this far up the page.
LEGEND_TOP = 0.78
LEGEND_COLUMNS = 10

# --- The raster figure ----------------------------------------------------------------------------

#: The trials drawn, each from its start; the unit order comes from :data:`SORT_TRIAL` alone. Only
#: units present throughout are drawn: a unit's first spike must precede the first of these trials
#: and its last spike follow the last, so a unit that appeared or vanished in between is left out.
RASTER_TRIALS = (30, 47)
SORT_TRIAL = 30
#: The GUI heatmap's sort: spike rate per window of this length, overlapping by this fraction,
#: and a unit ranks by the window holding its largest rate.
SORT_WINDOW_S = 0.1
SORT_OVERLAP = 0.5
#: Spikes are counted in bins this long before the windows are averaged.
SORT_BIN_S = 0.01
RASTER_ROWS = ("ground truth", "spikes", "model")
RASTER_HEIGHT_RATIOS = (1, 10, 1)
RASTER_HEIGHT = 8
SPIKE_COLOR = "black"

STYLE_DIR = Path(__file__).parent / "style"
EXPORT_RC = {
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "path.simplify": False,
}


@dataclass(frozen=True)
class Trial:
    """One trial's rows, on the trial's clock."""

    trial: int
    truth: pd.DataFrame
    predicted: pd.DataFrame
    time: np.ndarray
    confidence: np.ndarray


def state_rows(df: pd.DataFrame, trial: int) -> pd.DataFrame:
    return df[(df["trial"] == trial) & (df["event_type"] == "state") & (df["individual"] == INDIVIDUAL)]


def load_trials(store: PredictionsStore) -> list[Trial]:
    ground_truth = load_labels_tsv(GROUND_TRUTH)
    curated = ground_truth[ground_truth["labeling_method"] != LABELING_AUTOMATED]
    predicted = load_labels_tsv(store.tsv_path)
    if store.npz_path is None:
        raise FileNotFoundError(f"{store.folder} has no *_probs.npz to read the confidence from")

    trials = []
    with np.load(store.npz_path) as npz:
        for trial in TRIALS:
            marker = f"_trial{trial}_"
            key = next((k for k in npz.files if marker in k and not k.endswith(("_time", "_boundary"))), None)
            if key is None:
                raise ValueError(f"{store.npz_path} never predicted trial {trial}")
            _, confidence = prediction_to_labels_and_confidence(np.asarray(npz[key], dtype=np.float64))
            if confidence is None:
                raise ValueError(f"{store.npz_path}: {key!r} is not (T, C) probabilities")
            trials.append(
                Trial(
                    trial=trial,
                    truth=state_rows(curated, trial),
                    predicted=state_rows(predicted, trial),
                    time=np.asarray(npz[f"{key}_time"], dtype=np.float64),
                    confidence=confidence,
                )
            )
    return trials


def bare(ax: plt.Axes, time_axis: bool = False) -> None:
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=time_axis, labelbottom=time_axis)
    for spine in ax.spines.values():
        spine.set_visible(False)


def window(trial: Trial) -> tuple[float, float]:
    labels = pd.concat([trial.truth, trial.predicted])
    return float(labels["onset_s"].min()) - MARGIN_S, float(labels["offset_s"].max()) + MARGIN_S


def draw_column(axs: list[plt.Axes], trial: Trial, mapping: dict, name_rows: bool) -> None:
    t0, t1 = window(trial)
    for ax, row in zip(axs, ROWS, strict=True):
        if row == "ground truth":
            # Opaque: CorelDRAW mishandles a PDF's transparency and inverts the page's colours.
            plot_label_segments(ax, trial.truth, mapping, alpha=1)
        elif row == "model":
            plot_label_segments(ax, trial.predicted, mapping, alpha=1)
        else:
            ax.plot(trial.time, trial.confidence, color="black", lw=0.8)
            ax.set_ylim(0, 1.05)
        if name_rows:
            ax.set_ylabel(row, rotation=0, ha="right", va="center", fontsize=8)
        bare(ax, time_axis=ax is axs[-1])
        ax.set_xlim(t0, t1)
    axs[0].set_title(f"trial {trial.trial}", fontsize=8)
    axs[-1].xaxis.set_major_locator(MultipleLocator(TICK_S))
    axs[-1].xaxis.set_major_formatter(FormatStrFormatter("%g"))


def draw(fig: plt.Figure, trials: list[Trial], mapping: dict) -> None:
    # Columns are as wide as their duration, so time runs at one scale in all of them.
    windows = [window(t) for t in trials]
    grid = fig.add_gridspec(
        len(ROWS),
        len(trials),
        height_ratios=HEIGHT_RATIOS,
        width_ratios=[t1 - t0 for t0, t1 in windows],
        wspace=0.1,
        top=LEGEND_TOP,
    )
    columns = [[fig.add_subplot(grid[row, col]) for row in range(len(ROWS))] for col in range(len(trials))]
    for col, (axs, trial) in enumerate(zip(columns, trials, strict=True)):
        draw_column(axs, trial, mapping, name_rows=col == 0)

    shown = sorted({int(k) for t in trials for k in pd.concat([t.truth, t.predicted])["labels"]})
    fig.legend(
        handles=[Patch(facecolor=mapping[k]["color"], label=mapping[k]["name"]) for k in shown],
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=LEGEND_COLUMNS,
        frameon=False,
        fontsize=7,
    )


# --- The raster figure ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Raster:
    """One trial's spikes on the trial's clock: one array per kept unit, in the units' file order."""

    trial: Trial
    spikes: list[np.ndarray]
    #: The kept units' ids.
    units: list[int]


def load_rasters(trials: list[Trial]) -> list[Raster]:
    """Every unit's spikes of each of *trials*, cut to the trial by the alignment's timing, on the trial's clock."""
    units = nap.load_file(str(UNITS))
    if not isinstance(units, nap.TsGroup):
        raise ValueError(f"{UNITS} is a {type(units).__name__}, expected a TsGroup")
    alignment = NWBAlignment(ALIGNMENT)
    try:
        bounds = {}
        for trial in trials:
            start, stop = alignment.start_time(trial.trial), alignment.stop_time(trial.trial)
            if stop is None:
                raise ValueError(f"{ALIGNMENT} has no stop time for trial {trial.trial}")
            bounds[trial.trial] = (start, stop)
    finally:
        alignment.close()
    first, last = min(b[0] for b in bounds.values()), max(b[1] for b in bounds.values())
    kept = [int(u) for u in units.index if len(units[u]) and units[u].t[0] < first and units[u].t[-1] > last]
    dropped = sorted(set(int(u) for u in units.index) - set(kept))
    print(f"{len(kept)} of {len(units)} units fire from before {first:.1f} s to after {last:.1f} s; dropped {dropped}")
    rasters = []
    for trial in trials:
        start, stop = bounds[trial.trial]
        inside = units.restrict(nap.IntervalSet(start, stop))
        rasters.append(Raster(trial, [np.asarray(inside[u].t, dtype=np.float64) - start for u in kept], kept))
    return rasters


def unit_order(raster: Raster, t1: float) -> np.ndarray:
    """Unit indices, earliest peak window first, from the spikes of *raster* between the trial's start and *t1*.

    A unit that never fires in that stretch has no peak and goes last, in file order.
    """
    edges = np.arange(0.0, t1 + SORT_BIN_S, SORT_BIN_S)
    counts = np.stack([np.histogram(s, bins=edges)[0] for s in raster.spikes], axis=1).astype(float)  # (T, C)
    centres = edges[:-1] + SORT_BIN_S / 2
    order = argmax_window_order(counts, centres, SORT_WINDOW_S, SORT_OVERLAP)
    silent = counts.sum(axis=0) == 0
    return np.concatenate([order[~silent[order]], order[silent[order]]])


def draw_raster_column(axs: list[plt.Axes], raster: Raster, order: np.ndarray, mapping: dict, name_rows: bool) -> None:
    """The trial from its start to the end of its label window: curated labels, the sorted raster, predicted labels."""
    t1 = window(raster.trial)[1]
    for ax, row in zip(axs, RASTER_ROWS, strict=True):
        if row == "ground truth":
            # Opaque: CorelDRAW mishandles a PDF's transparency and inverts the page's colours.
            plot_label_segments(ax, raster.trial.truth, mapping, alpha=1)
        elif row == "model":
            plot_label_segments(ax, raster.trial.predicted, mapping, alpha=1)
        else:
            ax.eventplot([raster.spikes[i] for i in order], colors=SPIKE_COLOR, linelengths=0.8, linewidths=0.5)
            ax.set_ylim(len(order) - 0.5, -0.5)  # the first unit of the order on top
        if name_rows:
            ax.set_ylabel(row, rotation=0, ha="right", va="center", fontsize=8)
        bare(ax, time_axis=ax is axs[-1])
        ax.set_xlim(0.0, t1)
    axs[0].set_title(f"trial {raster.trial.trial}", fontsize=8)
    axs[-1].xaxis.set_major_locator(MultipleLocator(TICK_S))
    axs[-1].xaxis.set_major_formatter(FormatStrFormatter("%g"))


def draw_rasters(fig: plt.Figure, rasters: list[Raster], mapping: dict) -> None:
    sorter = next(r for r in rasters if r.trial.trial == SORT_TRIAL)
    order = unit_order(sorter, window(sorter.trial)[1])
    print(f"unit order from trial {SORT_TRIAL}: {order.tolist()}")
    # Columns are as wide as their duration, so time runs at one scale in all of them.
    grid = fig.add_gridspec(
        len(RASTER_ROWS),
        len(rasters),
        height_ratios=RASTER_HEIGHT_RATIOS,
        width_ratios=[window(r.trial)[1] for r in rasters],
        wspace=0.1,
        top=0.92,
    )
    for col, raster in enumerate(rasters):
        axs = [fig.add_subplot(grid[row, col]) for row in range(len(RASTER_ROWS))]
        draw_raster_column(axs, raster, order, mapping, name_rows=col == 0)


def save(fig: plt.Figure, stem: Path) -> None:
    with plt.style.context(EXPORT_RC), PdfPages(stem.with_suffix(".pdf")) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches=None, metadata={"Creator": "matplotlib"})
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        plt.close(fig)
    print(f"Wrote {stem.with_suffix('.pdf')}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    mapping = load_label_mapping(MAPPING)
    trials = load_trials(PredictionsStore(PREDICTIONS))
    for trial in trials:
        print(f"trial {trial.trial}: {len(trial.truth)} curated, {len(trial.predicted)} predicted, {window(trial)}")
    with plt.style.context([STYLE_DIR / "style.mplstyle", EXPORT_RC]):
        fig = plt.figure(figsize=(WIDTH, HEIGHT))
        draw(fig, trials, mapping)
    save(fig, ROOT / "figure_neural")

    rasters = load_rasters([t for t in trials if t.trial in RASTER_TRIALS])
    if len(rasters) != len(RASTER_TRIALS):
        raise ValueError(f"RASTER_TRIALS {RASTER_TRIALS} must all be in TRIALS {TRIALS}")
    for raster in rasters:
        print(
            f"trial {raster.trial.trial}: {sum(len(s) for s in raster.spikes)} spikes from {len(raster.spikes)} units"
        )
    with plt.style.context([STYLE_DIR / "style.mplstyle", EXPORT_RC]):
        fig = plt.figure(figsize=(WIDTH, RASTER_HEIGHT))
        draw_rasters(fig, rasters, mapping)
    save(fig, ROOT / "figure_neural_raster")


if __name__ == "__main__":
    main()
