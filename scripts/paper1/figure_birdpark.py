"""Birdpark figure: one bird's accelerometer vibration under the labels made on it.

Top to bottom: the manual labels of :data:`INDIVIDUAL` coloured by
:data:`MAPPING`, then one row per entry of :data:`TRACE_VARS` — one column per class,
showing that class's first label with :data:`MARGIN_S` either side. No text but the
time-axis numbers and the class legend.
Writes ``figure_birdpark.{pdf,png}`` into :data:`ROOT`.

    python scripts/paper1/figure_birdpark.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import ethograph as eto
from ethograph.labels.intervals import LABELING_AUTOMATED, load_label_mapping
from ethograph.labels.plots import plot_label_segments
from ethograph.labels.tsv_store import labels_tsv_path, load_labels_tsv

ROOT = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\paper\figure1\figure_birdpark")
SESSION = Path(r"C:\Users\aksel\Documents\Code\ethograph\data\birdpark\copExpBP08")
MAPPING = SESSION / "mapping.txt.txt"
INDIVIDUAL = "female (yellow radio)"
TRACE_VARS = ("vibration_bandpass", "vibration")
#: Seconds drawn before and after each label.
MARGIN_S = 0.2
#: Spacing of the time ticks (s).
TICK_S = 0.2
#: A trace with more samples than this is drawn as its min/max envelope in this many bins.
MAX_POINTS = 4000
TRACE_COLOR = "black"

# Vector-editor-friendly export (CorelDRAW, Illustrator), as figure1_new.py: the
# paper style's Arial font and TrueType fonts kept as editable text.
plt.style.use(Path(__file__).parent / "style" / "style.mplstyle")
mpl.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False})


def save_pdf(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Write *fig* through ``PdfPages`` with the figure's own bounds, which CorelDRAW imports cleanly."""
    with PdfPages(path) as pdf:
        pdf.savefig(fig, dpi=dpi, bbox_inches=None, metadata={"Creator": "matplotlib"})


def session_dataset(session: Path) -> xr.Dataset:
    """The one ``.nc`` of the *session* folder, as a flat dataset."""
    (nc,) = session.glob("*.nc")
    return xr.open_datatree(nc).to_dataset()


def trace_window(ds: xr.Dataset, var: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Times (T,) and values (T,) of *var* for :data:`INDIVIDUAL` between *t0* and *t1*."""
    da = ds[var].sel(individual=INDIVIDUAL)
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


def draw_column(axs: list[plt.Axes], ds: xr.Dataset, labels: pd.DataFrame, mapping: dict, t0: float, t1: float) -> None:
    """Labels over one row per :data:`TRACE_VARS` entry, all on *t0*–*t1*; only the last row keeps its time axis."""
    label_ax, *trace_axs = axs
    # Opaque: CorelDRAW mishandles a PDF's transparency and inverts the page's colours.
    plot_label_segments(label_ax, labels[(labels["offset_s"] > t0) & (labels["onset_s"] < t1)], mapping, alpha=1)
    for ax, var in zip(trace_axs, TRACE_VARS, strict=True):
        time, values = trace_window(ds, var, t0, t1)
        ax.plot(*envelope(time, values), color=TRACE_COLOR, linewidth=0.6)
    for ax in axs:
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=ax is axs[-1], labelbottom=ax is axs[-1])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(t0, t1)


def main() -> None:
    mapping = load_label_mapping(MAPPING)
    labels = load_labels_tsv(labels_tsv_path(SESSION))
    labels = labels[(labels["labeling_method"] != LABELING_AUTOMATED) & (labels["individual"] == INDIVIDUAL)]
    ds = session_dataset(SESSION)

    # The first label of each class, in time order, with the margin either side.
    firsts = labels.sort_values("onset_s").drop_duplicates("labels")
    windows = [(float(on) - MARGIN_S, float(off) + MARGIN_S) for on, off in zip(firsts["onset_s"], firsts["offset_s"])]
    for class_id, (t0, t1) in zip(firsts["labels"], windows, strict=True):
        print(f"{mapping[int(class_id)]['name']}: {t0:.3f}-{t1:.3f} s")

    # Columns are as wide as their duration, so time runs at one scale in all of them.
    n_rows = 1 + len(TRACE_VARS)
    fig = plt.figure(figsize=(15, 6))
    grid = fig.add_gridspec(
        n_rows,
        len(windows),
        height_ratios=[0.5, *[2] * len(TRACE_VARS)],
        width_ratios=[t1 - t0 for t0, t1 in windows],
        wspace=0.1,
    )
    columns = [[fig.add_subplot(grid[row, col]) for row in range(n_rows)] for col in range(len(windows))]
    for axs, (t0, t1) in zip(columns, windows, strict=True):
        draw_column(axs, ds, labels, mapping, t0, t1)
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

    ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figure_birdpark.png", dpi=300, bbox_inches="tight")
    save_pdf(fig, ROOT / "figure_birdpark.pdf")
    plt.close(fig)
    print(f"Wrote {ROOT / 'figure_birdpark.pdf'}")


if __name__ == "__main__":
    main()
