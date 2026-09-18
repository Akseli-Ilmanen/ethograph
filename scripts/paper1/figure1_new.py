"""Figure 1: one movement in 3D, above one trial's ground truth.

Class names and colours come from :data:`MAPPING`.

Top: the beak and stick tips' 3D trajectories during the ground-truth
:data:`MOVEMENT` label, coloured by :data:`COLOR_VAR` — full view with the box,
and zoomed on the beak without it. Below, top to bottom: curated ground truth as grey boxes,
:data:`TRACE_VAR` coloured by :data:`COLOR_VAR`, :data:`COLOR_VAR` as a 1D colour
strip, and :data:`ZOOM_S` of the :data:`MOVEMENT` label above :data:`TRACE_VAR` coloured
by :data:`COLOR_VAR`. No text but the time-axis
numbers. Writes one ``figure1_trial{N}_azim{A}.{pdf,png}`` per entry of
:data:`AZIMUTHS` into ``{ROOT}/azimuths``.

    python scripts/paper1/figure1_new.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.figure import SubFigure
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D

import ethograph as eto
from ethograph.labels.intervals import LABELING_AUTOMATED, load_label_mapping
from ethograph.labels.tsv_store import labels_tsv_path, load_labels_tsv

ROOT = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\paper\figure1")
SESSION = Path(
    r"C:\Users\aksel\Documents\AK_data\derivatives\sub-03_id-Freddy"
    r"\ses-000_date-20250526_01\behav\Trial_data3.nc"
)
MAPPING = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\crowlab\mapping.txt")
#: The trial drawn. ``None`` takes the first trial with ground truth.
TRIAL: int | None = 33
INDIVIDUAL = "Freddy"
PIN = {"keypoint": "beakTip", "individual": INDIVIDUAL}
#: Edges of the time axis (s); ``None`` runs to that end of the trial. The axis
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

#: The ground-truth label whose interval the 3D panels draw, by mapping name.
MOVEMENT = "nodding"
#: Keypoint → line alpha in the 3D panels; the first is the one zoomed on.
TRAJECTORIES = {"beakTip": 1.0, "stickTip": 0.6}
#: Padding around the zoomed keypoint's path, in the position's units.
ZOOM_MARGIN = 0.1
ELEVATION = 25
#: One figure per azimuth, in 5° steps from 200° to 250°.
AZIMUTHS = tuple(azim for azim in range(200, 250, 5))
#: The box: its x/y footprint corners and z floor/ceiling, in the position's units.
BOX_XY = np.array([[-7.00, 0.00], [-7.00, 9.80], [6.80, 9.80], [6.80, 0.00]])
BOX_Z = (0.65, 2.75)
#: The four front edges drawn solid; the four receding ones dashed, this far along.
FRONT_EDGES = ((0, 1), (4, 5), (0, 4), (1, 5))
DEPTH_EDGES = ((0, 3), (1, 2), (4, 7), (5, 6))
DEPTH_FRACTION = 1 / 5


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


# Vector-editor-friendly export (CorelDRAW, Illustrator), as session.py's save_pdf:
# the paper style's Arial font, TrueType fonts kept as editable text, and paths
# written unsimplified.
plt.style.use(Path(__file__).parent / "style" / "style ppt.mplstyle")
mpl.rcParams.update(
    {
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "path.simplify": False,
    }
)


def save_pdf(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Write *fig* the way session.py's ``save_pdf`` does, then close it: through
    ``PdfPages`` with the figure's own bounds, which CorelDRAW imports cleanly."""
    with PdfPages(path) as pdf:
        pdf.savefig(fig, dpi=dpi, bbox_inches=None, metadata={"Creator": "matplotlib"})
    plt.close(fig)


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


def draw_scene(ax: Axes3D, positions: dict[str, np.ndarray], colors: dict[str, np.ndarray], azim: float) -> None:
    """Each keypoint's (T, 3) trajectory, coloured frame by frame."""
    for keypoint, alpha in TRAJECTORIES.items():
        xyz, rgb = positions[keypoint], colors[keypoint]
        for i in range(len(xyz) - 1):
            ax.plot(*xyz[i : i + 2].T, color=rgb[i], linewidth=2, alpha=alpha)
        ax.scatter(*xyz.T, c=rgb, s=10, edgecolors="k", linewidths=0.5, zorder=5, alpha=alpha)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=ELEVATION, azim=azim)


def plot_movement(
    subfig: SubFigure,
    positions: dict[str, np.ndarray],
    colors: dict[str, np.ndarray],
    azim: float,
) -> None:
    full = subfig.add_subplot(121, projection="3d")
    draw_box(full)
    draw_scene(full, positions, colors, azim)

    # The zoom's limits are the zoomed keypoint's whole drawn path. mplot3d never
    # clips lines to the axes, so the box is left out and other trajectories' points beyond the limits are blanked.
    zoomed = positions[next(iter(TRAJECTORIES))]
    lo, hi = np.nanmin(zoomed, axis=0) - ZOOM_MARGIN, np.nanmax(zoomed, axis=0) + ZOOM_MARGIN
    inside = {
        k: np.where(((xyz >= lo) & (xyz <= hi)).all(axis=1, keepdims=True), xyz, np.nan) for k, xyz in positions.items()
    }
    zoom = subfig.add_subplot(122, projection="3d")
    draw_scene(zoom, inside, colors, azim)
    zoom.set_xlim(lo[0], hi[0])
    zoom.set_ylim(lo[1], hi[1])
    zoom.set_zlim(lo[2], hi[2])
    zoom.set_box_aspect(hi - lo)  # equal units on every axis


def plot_trial(
    subfig: SubFigure,
    truth: pd.DataFrame,
    time: np.ndarray,
    trace: np.ndarray,
    colors: np.ndarray,
    mapping: dict,
) -> None:
    t_start = time[0] if T_START is None else T_START
    t_end = (time[-1] if T_END is None else T_END) - t_start
    time = time - t_start
    truth = truth.assign(onset_s=truth["onset_s"] - t_start, offset_s=truth["offset_s"] - t_start)
    truth = truth[(truth["offset_s"] > 0) & (truth["onset_s"] < t_end)].sort_values("onset_s")
    if len(truth) != len(LABEL_LETTERS):
        raise ValueError(f"{len(truth)} labels on the plotted axis but {len(LABEL_LETTERS)} LABEL_LETTERS")
    truth = truth.assign(letter=LABEL_LETTERS)

    # The fourth row is a spacer between the full-range rows and the zoom.
    axs = subfig.subplots(6, 1, gridspec_kw={"height_ratios": [1, 2, 0.5, 0.4, 1, 8]})
    label_ax, trace_ax, strip_ax, spacer_ax, zoom_label_ax, zoom_ax = axs
    spacer_ax.set_axis_off()
    draw_labels(label_ax, truth)
    movement_id = next(k for k, v in mapping.items() if v["name"] == MOVEMENT)
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
        ax.set_yticks([])
        with_axis = ax in (strip_ax, zoom_ax)
        ax.tick_params(left=False, bottom=with_axis, labelbottom=with_axis)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(*(ZOOM_S if ax in (zoom_label_ax, zoom_ax) else (0, t_end)))


def keypoint_window(ds: xr.Dataset, var: str, keypoint: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Times (T,) and *var* at *keypoint* between *t0* and *t1*, as (T, 3)."""
    da = ds[var].sel(keypoint=keypoint, individual=INDIVIDUAL)
    time_dim = eto.get_time_coord(da).name
    window = da.sel({time_dim: slice(t0, t1)}).transpose(time_dim, ...)
    return np.asarray(window[time_dim].values), np.asarray(window.values, dtype=np.float64)


def main() -> None:
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
    positions = {k: keypoint_window(ds, "position", k, t0, t1)[1] for k in TRAJECTORIES}
    movement_colors = {k: keypoint_window(ds, COLOR_VAR, k, t0, t1)[1] for k in TRAJECTORIES}

    out_dir = ROOT / "azimuths"
    out_dir.mkdir(exist_ok=True)
    for azim in sorted(AZIMUTHS):
        fig = plt.figure(figsize=(15, 13))
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.2, 2.1])
        plot_movement(top, positions, movement_colors, azim)
        plot_trial(
            bottom,
            truth,
            time,
            np.asarray(trace, dtype=np.float64),
            np.asarray(colors, dtype=np.float64),
            mapping,
        )
        stem = out_dir / f"figure1_trial{trial}_azim{azim}"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        save_pdf(fig, stem.with_suffix(".pdf"))
        print(f"Wrote azimuth {azim}")


if __name__ == "__main__":
    main()
