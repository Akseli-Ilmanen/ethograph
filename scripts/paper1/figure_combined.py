"""The paper figure: one movement in 3D, one trial's labels and predictions, and the birdpark session.

Three parts, top to bottom, no text but the time-axis numbers, the row names and the class legend:

1. **Movement** — the beak and stick tips' 3D trajectories during the ground-truth
   :data:`MOVEMENT` label, coloured by :data:`COLOR_VAR` (full view with the box, and zoomed on
   the beak without it). Below: curated ground truth as grey lettered boxes, :data:`TRACE_VAR`
   coloured by :data:`COLOR_VAR`, :data:`COLOR_VAR` as a 1D colour strip, and :data:`ZOOM_S` of
   the :data:`MOVEMENT` label above the same trace.
2. **Predictions** — the same trial's rows, chosen per version in :data:`VERSIONS`: the ground
   truth over the saved prediction set as coloured rectangles, :data:`TRACE_VAR`, and the
   model's per-frame confidence.
3. **Birdpark** — one bird's manual labels over one row per entry of :data:`BIRDPARK_TRACE_VARS`,
   one column per class showing that class's first label with :data:`MARGIN_S` either side.

Writes one ``figure_combined{suffix}.{pdf,png}`` per entry of :data:`VERSIONS` into :data:`ROOT`.
:data:`SINGLE_PAGE` stacks the parts on one page, otherwise each part gets a page of its own.

    python scripts/paper1/figure_combined.py
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.figure import FigureBase, SubFigure
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from mpl_toolkits.mplot3d.axes3d import Axes3D

import ethograph as eto
from ethograph.labels.intervals import LABELING_AUTOMATED, load_label_mapping
from ethograph.labels.plots import plot_label_segments
from ethograph.labels.predictions import PredictionsStore, prediction_to_labels_and_confidence
from ethograph.labels.tsv_store import labels_tsv_path, load_labels_tsv

ROOT = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\paper\figure1")
#: One page holding every part, or one page per part.
SINGLE_PAGE = True
WIDTH = 15

# --- Parts 1 and 2: one trial -------------------------------------------------------------------

SESSION = Path(
    r"C:\Users\aksel\Documents\AK_data\derivatives\sub-03_id-Freddy"
    r"\ses-000_date-20250526_01\behav\Trial_data3.nc"
)
MAPPING = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\crowlab\mapping.txt")
#: The trial drawn. ``None`` takes the first trial with ground truth.
TRIAL: int | None = 33
INDIVIDUAL = "Freddy"
PIN = {"keypoint": "stickTip", "individual": INDIVIDUAL}
#: Edges of the movement part's time axis (s); ``None`` runs to that end of the trial. The axis
#: is shifted so ``T_START`` reads 0.
T_START: float | None = 0.35
T_END: float | None = 3.7
TRACE_VAR = "speed"
COLOR_VAR = "angle_rgb"
#: The zoomed trace's time window (s), on the plotted axis (``T_START`` reads 0).
ZOOM_S = (1.5, 2.0)
#: One letter per ground-truth label on the plotted axis, in onset order.
LABEL_LETTERS = ("A", "B", "C", "C", "C", "C", "D", "E", "F", "G")
LABEL_FACE = (0.8, 0.8, 0.8)
LABEL_EDGE = (0.3, 0.3, 0.3)
MOVEMENT_HEIGHT = 13

#: The ground-truth label whose interval the 3D panels draw, by mapping name.
MOVEMENT = "nodding"
#: Keypoint → line alpha in the 3D panels; the first is the one zoomed on.
TRAJECTORIES = {"beakTip": 1.0, "stickTip": 0.6}
#: Padding around the zoomed keypoint's path, in the position's units.
ZOOM_MARGIN = 0.1
ELEVATION = 25
#: The 3D panels' viewing angle (°). Anything from 200 to 245 reads well; adjust by hand.
AZIMUTH = 225
#: The box: its x/y footprint corners and z floor/ceiling, in the position's units.
BOX_XY = np.array([[-7.00, 0.00], [-7.00, 9.80], [6.80, 9.80], [6.80, 0.00]])
BOX_Z = (0.65, 2.75)
#: The four front edges drawn solid; the four receding ones dashed, this far along.
FRONT_EDGES = ((0, 1), (4, 5), (0, 4), (1, 5))
DEPTH_EDGES = ((0, 3), (1, 2), (4, 7), (5, 6))
DEPTH_FRACTION = 1 / 5

#: A folder holding one saved ``predictions_*`` set: its TSV is the model row, its
#: ``_probs.npz`` the confidence. Its post-processing is whatever its ``inference.yaml`` records.
PREDICTIONS_ROOT = ROOT
#: The rows the predictions part can draw, top to bottom in the order a version lists them:
#: ``labels`` is the ground truth over the model as coloured rectangles, ``trace`` is
#: :data:`TRACE_VAR` at :data:`PIN`, ``confidence`` the model's per-frame confidence.
ROW_KINDS = ("labels", "trace", "confidence")
#: Output file suffix → the predictions part's rows in that version.
VERSIONS: dict[str, tuple[str, ...]] = {
    "": ("labels",),
    "_traces": ("labels", "trace", "confidence"),
    "_traces_only": ("trace", "confidence"),
}
#: Edges of the predictions part's time axis (s), on the trial's clock; ``None`` runs to that
#: end of what the model predicted. It starts with the movement part and runs on past it.
PREDICTION_T_START: float | None = T_START
PREDICTION_T_END: float | None = None
#: Inches per row of the predictions part.
ROW_HEIGHT = 1.3

# --- Part 3: birdpark ---------------------------------------------------------------------------

BIRDPARK_SESSION = Path(r"C:\Users\aksel\Documents\Code\ethograph\data\birdpark\copExpBP08")
BIRDPARK_MAPPING = BIRDPARK_SESSION / "mapping.txt.txt"
BIRDPARK_INDIVIDUAL = "female (yellow radio)"
BIRDPARK_TRACE_VARS = ("vibration_bandpass", "vibration")
BIRDPARK_HEIGHT = 6
#: Seconds drawn before and after each label.
MARGIN_S = 0.2
#: Spacing of the time ticks (s).
TICK_S = 0.2
#: A trace with more samples than this is drawn as its min/max envelope in this many bins.
MAX_POINTS = 4000
TRACE_COLOR = "black"

# --- Style --------------------------------------------------------------------------------------

# Vector-editor-friendly export (CorelDRAW, Illustrator), as session.py's save_pdf:
# the paper style's Arial font, TrueType fonts kept as editable text, and paths
# written unsimplified.
STYLE_DIR = Path(__file__).parent / "style"
EXPORT_RC = {
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "path.simplify": False,
}


def movement_style() -> AbstractContextManager[None]:
    return plt.style.context([STYLE_DIR / "style ppt.mplstyle", EXPORT_RC])


def paper_style() -> AbstractContextManager[None]:
    return plt.style.context([STYLE_DIR / "style.mplstyle", EXPORT_RC])


def bare(ax: plt.Axes, time_axis: bool = False) -> None:
    """No spines and no y axis; the time axis only when *time_axis*."""
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=time_axis, labelbottom=time_axis)
    for spine in ax.spines.values():
        spine.set_visible(False)


# --- Part 1: movement ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialData:
    """One trial, read once and shared by the movement and predictions parts."""

    trial: int
    truth: pd.DataFrame
    time: np.ndarray
    trace: np.ndarray
    colors: np.ndarray
    positions: dict[str, np.ndarray]
    movement_colors: dict[str, np.ndarray]
    mapping: dict


def state_rows(df: pd.DataFrame, trial: int) -> pd.DataFrame:
    """*trial*'s state labels — the classes the segmentation model predicts."""
    return df[(df["trial"] == trial) & (df["event_type"] == "state")]


def movement_interval(truth: pd.DataFrame, mapping: dict) -> tuple[float, float]:
    """Onset and offset (s) of the one :data:`MOVEMENT` label in *truth*."""
    class_id = next(k for k, v in mapping.items() if v["name"] == MOVEMENT)
    rows = truth[truth["labels"] == class_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one {MOVEMENT!r} label in this trial, found {len(rows)}")
    return float(rows["onset_s"].iloc[0]), float(rows["offset_s"].iloc[0])


def keypoint_window(ds: xr.Dataset, var: str, keypoint: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Times (T,) and *var* at *keypoint* between *t0* and *t1*, as (T, 3)."""
    da = ds[var].sel(keypoint=keypoint, individual=INDIVIDUAL)
    time_dim = eto.get_time_coord(da).name
    window = da.sel({time_dim: slice(t0, t1)}).transpose(time_dim, ...)
    return np.asarray(window[time_dim].values), np.asarray(window.values, dtype=np.float64)


def load_trial() -> TrialData:
    mapping = load_label_mapping(MAPPING)

    ground_truth = load_labels_tsv(labels_tsv_path(SESSION))
    curated = ground_truth[ground_truth["labeling_method"] != LABELING_AUTOMATED]
    trial = TRIAL if TRIAL is not None else int(sorted(curated["trial"].unique())[0])
    truth = state_rows(curated, trial)

    ds = eto.open(str(SESSION)).trial(trial)
    time = np.asarray(eto.get_time_coord(ds[TRACE_VAR]).values, dtype=np.float64)
    trace, _ = eto.sel_valid(ds[TRACE_VAR], PIN)
    colors, _ = eto.sel_valid(ds[COLOR_VAR], PIN)

    t0, t1 = movement_interval(truth, mapping)
    print(f"{MOVEMENT}: {t0:.3f}-{t1:.3f} s")
    return TrialData(
        trial=trial,
        truth=truth,
        time=time,
        trace=np.asarray(trace, dtype=np.float64),
        colors=np.asarray(colors, dtype=np.float64),
        positions={k: keypoint_window(ds, "position", k, t0, t1)[1] for k in TRAJECTORIES},
        movement_colors={k: keypoint_window(ds, COLOR_VAR, k, t0, t1)[1] for k in TRAJECTORIES},
        mapping=mapping,
    )


def draw_labels(ax: plt.Axes, labels: pd.DataFrame) -> None:
    """Every label as a grey full-height box with a dark grey outline, its ``letter`` on it."""
    for onset, offset, letter in labels[["onset_s", "offset_s", "letter"]].itertuples(index=False):
        ax.axvspan(onset, offset, facecolor=LABEL_FACE, edgecolor=LABEL_EDGE, linewidth=1)
        ax.text(
            (onset + offset) / 2,
            0.5,
            letter,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=9,
            clip_on=True,
        )


def box_vertices() -> np.ndarray:
    (x_min, y_min), (x_max, y_max) = BOX_XY.min(axis=0), BOX_XY.max(axis=0)
    footprint = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    return np.array([(x, y, z) for z in BOX_Z for x, y in footprint])


def draw_box(ax: Axes3D) -> None:
    vertices = box_vertices()
    for a, b in FRONT_EDGES:
        ax.plot(*vertices[[a, b]].T, "k-", linewidth=1)
    for a, b in DEPTH_EDGES:
        end = vertices[a] + DEPTH_FRACTION * (vertices[b] - vertices[a])
        ax.plot(*np.stack([vertices[a], end]).T, "k--", linewidth=1, dashes=(3, 2))


def draw_scene(ax: Axes3D, positions: dict[str, np.ndarray], colors: dict[str, np.ndarray]) -> None:
    """Each keypoint's (T, 3) trajectory, coloured frame by frame."""
    for keypoint, alpha in TRAJECTORIES.items():
        xyz, rgb = positions[keypoint], colors[keypoint]
        for i in range(len(xyz) - 1):
            ax.plot(*xyz[i : i + 2].T, color=rgb[i], linewidth=2, alpha=alpha)
        ax.scatter(*xyz.T, c=rgb, s=10, edgecolors="k", linewidths=0.5, zorder=5, alpha=alpha)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)


def plot_movement(subfig: SubFigure, positions: dict[str, np.ndarray], colors: dict[str, np.ndarray]) -> None:
    full = subfig.add_subplot(121, projection="3d")
    draw_box(full)
    draw_scene(full, positions, colors)

    # The zoom's limits are the zoomed keypoint's whole drawn path. mplot3d never
    # clips lines to the axes, so the box is left out and other trajectories' points beyond the limits are blanked.
    zoomed = positions[next(iter(TRAJECTORIES))]
    lo, hi = np.nanmin(zoomed, axis=0) - ZOOM_MARGIN, np.nanmax(zoomed, axis=0) + ZOOM_MARGIN
    inside = {
        k: np.where(((xyz >= lo) & (xyz <= hi)).all(axis=1, keepdims=True), xyz, np.nan) for k, xyz in positions.items()
    }
    zoom = subfig.add_subplot(122, projection="3d")
    draw_scene(zoom, inside, colors)
    zoom.set_xlim(lo[0], hi[0])
    zoom.set_ylim(lo[1], hi[1])
    zoom.set_zlim(lo[2], hi[2])
    zoom.set_box_aspect(hi - lo)  # equal units on every axis


def plot_trial(subfig: SubFigure, data: TrialData) -> None:
    time, trace, colors = data.time, data.trace, data.colors
    t_start = time[0] if T_START is None else T_START
    t_end = (time[-1] if T_END is None else T_END) - t_start
    time = time - t_start
    truth = data.truth.assign(onset_s=data.truth["onset_s"] - t_start, offset_s=data.truth["offset_s"] - t_start)
    truth = truth[(truth["offset_s"] > 0) & (truth["onset_s"] < t_end)].sort_values("onset_s")
    if len(truth) != len(LABEL_LETTERS):
        raise ValueError(f"{len(truth)} labels on the plotted axis but {len(LABEL_LETTERS)} LABEL_LETTERS")
    truth = truth.assign(letter=LABEL_LETTERS)

    # The fourth row is a spacer between the full-range rows and the zoom.
    axs = subfig.subplots(6, 1, gridspec_kw={"height_ratios": [1, 2, 0.5, 0.4, 1, 8]})
    label_ax, trace_ax, strip_ax, spacer_ax, zoom_label_ax, zoom_ax = axs
    spacer_ax.set_axis_off()
    draw_labels(label_ax, truth)
    movement_id = next(k for k, v in data.mapping.items() if v["name"] == MOVEMENT)
    movement = truth[truth["labels"] == movement_id]
    draw_labels(zoom_label_ax, movement)

    # One segment per frame, coloured by the frame it starts on.
    points = np.column_stack([time, trace])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    valid = np.isfinite(segments).all(axis=(1, 2)) & np.isfinite(colors[:-1]).all(axis=1)
    trace_ax.add_collection(LineCollection(segments[valid], colors=colors[:-1][valid], linewidths=1.2))
    shown = trace[(time >= 0) & (time <= t_end)]
    trace_ax.set_ylim(np.nanmin(shown), np.nanmax(shown) * 1.05)

    strip = np.where(np.isfinite(colors), colors, 1.0)[np.newaxis]
    strip_ax.imshow(strip, aspect="auto", extent=(time[0], time[-1], 0, 1), interpolation="nearest")

    zoom_ax.add_collection(LineCollection(segments[valid], colors=colors[:-1][valid], linewidths=1.5))
    zoomed = trace[(time >= ZOOM_S[0]) & (time <= ZOOM_S[1])]
    zoom_ax.set_ylim(np.nanmin(zoomed), np.nanmax(zoomed) * 1.05)

    # Dotted drops from the zoomed label's edges down to where the curve is at that time.
    finite = np.isfinite(trace)
    for edge in movement[["onset_s", "offset_s"]].to_numpy().ravel():
        drop = ConnectionPatch(
            xyA=(edge, 0),
            coordsA=zoom_label_ax.get_xaxis_transform(),
            xyB=(edge, np.interp(edge, time[finite], trace[finite])),
            coordsB=zoom_ax.transData,
            linestyle=":",
            linewidth=1.5,
            color="black",
        )
        subfig.add_artist(drop)

    for ax in (label_ax, trace_ax, strip_ax, zoom_label_ax, zoom_ax):
        bare(ax, time_axis=ax in (strip_ax, zoom_ax))
        ax.set_xlim(*(ZOOM_S if ax in (zoom_label_ax, zoom_ax) else (0, t_end)))


def draw_movement(fig: FigureBase, data: TrialData) -> None:
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.2, 2.1])
    plot_movement(top, data.positions, data.movement_colors)
    plot_trial(bottom, data)


# --- Part 2: predictions ------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """The saved prediction set's view of one trial, on the trial's clock."""

    labels: pd.DataFrame
    time: np.ndarray
    confidence: np.ndarray


def load_prediction(trial: int) -> Prediction:
    folders = sorted(PREDICTIONS_ROOT.glob("predictions_*"))
    if len(folders) != 1:
        raise FileNotFoundError(f"Expected one predictions_* folder in {PREDICTIONS_ROOT}, found {len(folders)}")
    store = PredictionsStore(folders[0])
    if store.npz_path is None:
        raise FileNotFoundError(f"{store.folder} has no *_probs.npz to read the confidence from")

    marker = f"_trial{trial}_"
    with np.load(store.npz_path) as npz:
        key = next((k for k in npz.files if marker in k and not k.endswith(("_time", "_boundary"))), None)
        if key is None:
            raise ValueError(f"{store.npz_path} never predicted trial {trial}")
        probs = np.asarray(npz[key], dtype=np.float64)
        time = np.asarray(npz[f"{key}_time"], dtype=np.float64)
    _, confidence = prediction_to_labels_and_confidence(probs)
    if confidence is None:
        raise ValueError(f"{store.npz_path}: {key!r} is not (T, C) probabilities")
    return Prediction(labels=state_rows(load_labels_tsv(store.tsv_path), trial), time=time, confidence=confidence)


def prediction_rows(kinds: tuple[str, ...]) -> list[str]:
    """The row names of a version: ``labels`` is two rows, every other kind one."""
    unknown = set(kinds) - set(ROW_KINDS)
    if unknown:
        raise ValueError(f"Unknown row kinds {sorted(unknown)}; expected some of {ROW_KINDS}")
    return [row for kind in kinds for row in (("ground truth", "model") if kind == "labels" else (kind,))]


def draw_predictions(fig: FigureBase, data: TrialData, prediction: Prediction, kinds: tuple[str, ...]) -> None:
    t0 = prediction.time[0] if PREDICTION_T_START is None else PREDICTION_T_START
    t1 = prediction.time[-1] if PREDICTION_T_END is None else PREDICTION_T_END
    rows = prediction_rows(kinds)
    axs = np.atleast_1d(fig.subplots(len(rows), 1, sharex=True))
    for ax, row in zip(axs, rows, strict=True):
        if row == "ground truth":
            # Opaque: CorelDRAW mishandles a PDF's transparency and inverts the page's colours.
            plot_label_segments(ax, data.truth, data.mapping, alpha=1)
        elif row == "model":
            plot_label_segments(ax, prediction.labels, data.mapping, alpha=1)
        elif row == "trace":
            ax.plot(data.time, data.trace, color="tab:blue", lw=0.8)
            shown = data.trace[(data.time >= t0) & (data.time <= t1)]
            ax.set_ylim(np.nanmin(shown), np.nanmax(shown) * 1.05)
        else:
            ax.plot(prediction.time, prediction.confidence, color="black", lw=0.8)
            ax.set_ylim(0, 1.05)
        ax.set_ylabel(TRACE_VAR if row == "trace" else row, rotation=0, ha="right", va="center", fontsize=8)
        bare(ax)
        ax.set_xlim(t0, t1)


# --- Part 3: birdpark ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BirdparkData:
    labels: pd.DataFrame
    ds: xr.Dataset
    mapping: dict


def load_birdpark() -> BirdparkData:
    labels = load_labels_tsv(labels_tsv_path(BIRDPARK_SESSION))
    labels = labels[(labels["labeling_method"] != LABELING_AUTOMATED) & (labels["individual"] == BIRDPARK_INDIVIDUAL)]
    # The one .nc of the session folder, as a flat dataset.
    (nc,) = BIRDPARK_SESSION.glob("*.nc")
    return BirdparkData(
        labels=labels, ds=xr.open_datatree(nc).to_dataset(), mapping=load_label_mapping(BIRDPARK_MAPPING)
    )


def trace_window(ds: xr.Dataset, var: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Times (T,) and values (T,) of *var* for :data:`BIRDPARK_INDIVIDUAL` between *t0* and *t1*."""
    da = ds[var].sel(individual=BIRDPARK_INDIVIDUAL)
    time_dim = eto.get_time_coord(da).name
    window = da.sel({time_dim: slice(t0, t1)})
    return np.asarray(window[time_dim].values), np.asarray(window.values, dtype=np.float64)


def envelope(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A dense trace as the min and max of :data:`MAX_POINTS` equal bins, which draws the same
    picture as every sample but stays a short vector path an editor can open."""
    if len(time) <= 2 * MAX_POINTS:
        return time, values
    # The tail is padded with the last sample so every bin is full and none is dropped.
    pad = -len(time) % MAX_POINTS
    t = np.pad(time, (0, pad), mode="edge").reshape(MAX_POINTS, -1)
    v = np.pad(values, (0, pad), mode="edge").reshape(MAX_POINTS, -1)
    rows = np.arange(MAX_POINTS)
    # Min and max in the order they occur, so the path never doubles back in time.
    picks = np.sort(np.stack([np.nanargmin(v, axis=1), np.nanargmax(v, axis=1)], axis=1), axis=1)
    return t[rows[:, np.newaxis], picks].ravel(), v[rows[:, np.newaxis], picks].ravel()


def draw_column(axs: list[plt.Axes], data: BirdparkData, t0: float, t1: float) -> None:
    """Labels over one row per :data:`BIRDPARK_TRACE_VARS` entry, on *t0*–*t1*; only the last keeps its time axis."""
    label_ax, *trace_axs = axs
    labels = data.labels
    # Opaque: CorelDRAW mishandles a PDF's transparency and inverts the page's colours.
    plot_label_segments(label_ax, labels[(labels["offset_s"] > t0) & (labels["onset_s"] < t1)], data.mapping, alpha=1)
    for ax, var in zip(trace_axs, BIRDPARK_TRACE_VARS, strict=True):
        time, values = trace_window(data.ds, var, t0, t1)
        ax.plot(*envelope(time, values), color=TRACE_COLOR, linewidth=0.6)
    for ax in axs:
        bare(ax, time_axis=ax is axs[-1])
        ax.set_xlim(t0, t1)


def draw_birdpark(fig: FigureBase, data: BirdparkData) -> None:
    mapping = data.mapping

    # The first label of each class, in time order, with the margin either side.
    firsts = data.labels.sort_values("onset_s").drop_duplicates("labels")
    windows = [(float(on) - MARGIN_S, float(off) + MARGIN_S) for on, off in zip(firsts["onset_s"], firsts["offset_s"])]
    for class_id, (t0, t1) in zip(firsts["labels"], windows, strict=True):
        print(f"{mapping[int(class_id)]['name']}: {t0:.3f}-{t1:.3f} s")

    # Columns are as wide as their duration, so time runs at one scale in all of them.
    n_rows = 1 + len(BIRDPARK_TRACE_VARS)
    grid = fig.add_gridspec(
        n_rows,
        len(windows),
        height_ratios=[0.5, *[2] * len(BIRDPARK_TRACE_VARS)],
        width_ratios=[t1 - t0 for t0, t1 in windows],
        wspace=0.1,
    )
    columns = [[fig.add_subplot(grid[row, col]) for row in range(n_rows)] for col in range(len(windows))]
    for axs, (t0, t1) in zip(columns, windows, strict=True):
        draw_column(axs, data, t0, t1)
        axs[-1].xaxis.set_major_locator(MultipleLocator(TICK_S))
        axs[-1].xaxis.set_major_formatter(FormatStrFormatter("%g"))
    # One y-scale per row across the columns.
    for row in zip(*columns, strict=True):
        lows, highs = zip(*(ax.get_ylim() for ax in row), strict=True)
        for ax in row:
            ax.set_ylim(min(lows), max(highs))

    columns[0][0].legend(
        handles=[Patch(facecolor=mapping[int(k)]["color"], label=mapping[int(k)]["name"]) for k in firsts["labels"]],
        loc="lower left",
        bbox_to_anchor=(0, 1.05),
        ncol=len(firsts),
        frameon=False,
    )


# --- Layout -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Part:
    """One part of the figure: its height (in) at :data:`WIDTH`, its style, and its drawing."""

    height: float
    style: Callable[[], AbstractContextManager[None]]
    draw: Callable[[FigureBase], None]


def pages(parts: list[Part]) -> list[plt.Figure]:
    if not SINGLE_PAGE:
        figs = []
        for part in parts:
            with part.style():
                fig = plt.figure(figsize=(WIDTH, part.height))
                part.draw(fig)
            figs.append(fig)
        return figs

    heights = [part.height for part in parts]
    fig = plt.figure(figsize=(WIDTH, sum(heights)))
    for part, subfig in zip(parts, fig.subfigures(len(parts), 1, height_ratios=heights), strict=True):
        with part.style():
            part.draw(subfig)
    return [fig]


def save(figs: list[plt.Figure], stem: Path) -> None:
    """One PDF holding every page — through ``PdfPages`` with the figure's own bounds, which
    CorelDRAW imports cleanly — and a PNG per page."""
    # Fonts as editable text and paths unsimplified are read at save time, not when drawing.
    with plt.style.context(EXPORT_RC), PdfPages(stem.with_suffix(".pdf")) as pdf:
        for number, fig in enumerate(figs, start=1):
            pdf.savefig(fig, dpi=300, bbox_inches=None, metadata={"Creator": "matplotlib"})
            page = "" if len(figs) == 1 else f"_page{number}"
            fig.savefig(stem.with_name(f"{stem.name}{page}.png"), dpi=300)
            plt.close(fig)
    print(f"Wrote {stem.with_suffix('.pdf')}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    trial = load_trial()
    prediction = load_prediction(trial.trial)
    birdpark = load_birdpark()
    for suffix, kinds in VERSIONS.items():
        parts = [
            Part(MOVEMENT_HEIGHT, movement_style, lambda fig: draw_movement(fig, trial)),
            Part(
                ROW_HEIGHT * len(prediction_rows(kinds)),
                paper_style,
                lambda fig, kinds=kinds: draw_predictions(fig, trial, prediction, kinds),
            ),
            Part(BIRDPARK_HEIGHT, paper_style, lambda fig: draw_birdpark(fig, birdpark)),
        ]
        save(pages(parts), ROOT / f"figure_combined{suffix}")


if __name__ == "__main__":
    main()
