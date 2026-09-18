"""Figure 1: one movement in 3D, above one trial's ground truth against a saved prediction set.

Nothing is trained or predicted: the model row is read from the newest
``predictions_*/*_predictions.tsv`` under :data:`ROOT`; class names and
colours come from :data:`MAPPING`.

Top: the beak and stick tips' 3D trajectories during the ground-truth
:data:`MOVEMENT` label, coloured by :data:`COLOR_VAR`, with :data:`CONTEXT_S` either
side in grey — full view with the box,
and zoomed on the beak without it. Below, top to bottom: curated ground truth,
model output, :data:`TRACE_VAR` drawn as a curve coloured by :data:`COLOR_VAR`, and
:data:`COLOR_VAR` again as a 1D colour strip; in both, frames outside every
ground-truth label are :data:`CONTEXT_COLOR`. No text but the time-axis
numbers. Writes one ``figure1_trial{N}_azim{A}.{pdf,png}`` per entry of
:data:`AZIMUTHS` into ``{ROOT}/azimuths``.

    python scripts/paper1/figure1_new.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.figure import SubFigure
from mpl_toolkits.mplot3d.axes3d import Axes3D

import ethograph as eto
from ethograph.labels.intervals import LABELING_AUTOMATED, load_label_mapping
from ethograph.labels.plots import plot_label_segments
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

#: The ground-truth label whose interval the 3D panels draw, by mapping name.
MOVEMENT = "nodding"
#: Keypoint → line alpha in the 3D panels; the first is the one zoomed on.
TRAJECTORIES = {"beakTip": 1.0, "stickTip": 0.6}
#: Padding around the zoomed keypoint's path, in the position's units.
ZOOM_MARGIN = 0.1
#: Seconds of trajectory drawn before and after the movement, in :data:`CONTEXT_COLOR`.
CONTEXT_S = 0.1
CONTEXT_COLOR = (0.6, 0.6, 0.6)
ELEVATION = 25
#: One figure per azimuth, in 5° steps from 185° to 270°, skipping 210°.
AZIMUTHS = tuple(azim for azim in range(185, 275, 5) if azim != 210)
#: The box: its x/y footprint corners and z floor/ceiling, in the position's units.
BOX_XY = np.array([[-7.00, 0.00], [-7.00, 9.80], [6.80, 9.80], [6.80, 0.00]])
BOX_Z = (0.65, 2.75)
#: The four front edges drawn solid; the four receding ones dashed, this far along.
FRONT_EDGES = ((0, 1), (4, 5), (0, 4), (1, 5))
DEPTH_EDGES = ((0, 3), (1, 2), (4, 7), (5, 6))
DEPTH_FRACTION = 1 / 5


def newest_predictions(root: Path) -> Path:
    """The ``*_predictions.tsv`` of the newest ``predictions_*`` folder under *root*."""
    # Folder names end in their timestamp, so name order is time order.
    tsvs = sorted(root.glob("predictions_*/*_predictions.tsv"), key=lambda p: p.parent.name)
    if not tsvs:
        raise FileNotFoundError(f"No predictions_*/*_predictions.tsv under {root}")
    return tsvs[-1]


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

    # The zoom's limits are the zoomed keypoint's whole drawn path, context
    # included. mplot3d never clips lines to the axes, so the box is left out and
    # other trajectories' points beyond the limits are blanked.
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
    predicted: pd.DataFrame,
    time: np.ndarray,
    trace: np.ndarray,
    colors: np.ndarray,
    mapping: dict,
) -> None:
    t_start = time[0] if T_START is None else T_START
    t_end = (time[-1] if T_END is None else T_END) - t_start
    time = time - t_start
    truth, predicted = (
        df.assign(onset_s=df["onset_s"] - t_start, offset_s=df["offset_s"] - t_start) for df in (truth, predicted)
    )

    axs = subfig.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 2, 0.5]})
    plot_label_segments(axs[0], truth, mapping)
    plot_label_segments(axs[1], predicted, mapping)

    # One segment per frame, coloured by the frame it starts on.
    points = np.column_stack([time, trace])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    valid = np.isfinite(segments).all(axis=(1, 2)) & np.isfinite(colors[:-1]).all(axis=1)
    axs[2].add_collection(LineCollection(segments[valid], colors=colors[:-1][valid], linewidths=1.2))
    shown = trace[(time >= 0) & (time <= t_end)]
    axs[2].set_ylim(np.nanmin(shown), np.nanmax(shown) * 1.05)

    strip = np.where(np.isfinite(colors), colors, 1.0)[np.newaxis]
    axs[3].imshow(strip, aspect="auto", extent=(time[0], time[-1], 0, 1), interpolation="nearest")

    for ax in axs:
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=ax is axs[-1])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(0, t_end)


def in_labels(time: np.ndarray, labels: pd.DataFrame) -> np.ndarray:
    """Which of *time* fall inside any of *labels*' intervals."""
    onsets, offsets = labels["onset_s"].to_numpy(), labels["offset_s"].to_numpy()
    return ((time[:, np.newaxis] >= onsets) & (time[:, np.newaxis] <= offsets)).any(axis=1)


def keypoint_window(ds: xr.Dataset, var: str, keypoint: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Times (T,) and *var* at *keypoint* between *t0* and *t1*, as (T, 3)."""
    da = ds[var].sel(keypoint=keypoint, individual=INDIVIDUAL)
    time_dim = eto.get_time_coord(da).name
    window = da.sel({time_dim: slice(t0, t1)}).transpose(time_dim, ...)
    return np.asarray(window[time_dim].values), np.asarray(window.values, dtype=np.float64)


def main() -> None:
    tsv = newest_predictions(ROOT)
    print(f"Predictions: {tsv}")
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
    w0, w1 = t0 - CONTEXT_S, t1 + CONTEXT_S
    window_time, _ = keypoint_window(ds, "position", next(iter(TRAJECTORIES)), w0, w1)
    in_movement = (window_time >= t0) & (window_time <= t1)
    positions = {k: keypoint_window(ds, "position", k, w0, w1)[1] for k in TRAJECTORIES}
    movement_colors = {
        k: np.where(in_movement[:, np.newaxis], keypoint_window(ds, COLOR_VAR, k, w0, w1)[1], CONTEXT_COLOR)
        for k in TRAJECTORIES
    }

    predicted = state_rows(load_labels_tsv(tsv), trial)
    out_dir = ROOT / "azimuths"
    out_dir.mkdir(exist_ok=True)
    for azim in sorted(AZIMUTHS):
        fig = plt.figure(figsize=(15, 8))
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.2, 1])
        plot_movement(top, positions, movement_colors, azim)
        plot_trial(
            bottom,
            truth,
            predicted,
            time,
            np.asarray(trace, dtype=np.float64),
            np.where(in_labels(time, truth)[:, np.newaxis], np.asarray(colors, dtype=np.float64), CONTEXT_COLOR),
            mapping,
        )
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"figure1_trial{trial}_azim{azim}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote azimuth {azim}")


if __name__ == "__main__":
    main()
