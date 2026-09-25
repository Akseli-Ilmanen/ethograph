"""S3D embeddings and the ``s3d`` dataset variable of one trial as GUI-style heatmaps.

Loads the trial from :data:`DATASET`, the matching ``embeddings_s3d/{video}.npy``
(one row per video frame, same clock as the trial), z-scores each row the way
the GUI's heatmap does (``heatmap_normalization = per_channel``), orders rows
by the GUI's "Sort by trial window" (``heatmap_sort_window_s = 0.5``,
``heatmap_sort_overlap = 0.5``) and clips the colour range at the GUI's
exclusion percentile. The manual labels of the trial are drawn as a coloured
strip under each heatmap, as on screen. Two panels, one above the other:
the 1024-d embeddings, then the 20-d ``s3d`` variable.

Writes ``embedding_heatmaps.{pdf,svg,png}`` into :data:`OUT_DIR`.

    python scripts/paper1/embedding_heatmaps.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from ethograph.features.preprocessing import z_normalize
from ethograph.gui.heatmap_sort import argmax_window_order
from ethograph.labels.intervals import load_label_mapping

# Export settings CorelDRAW opens cleanly (same as scripts/paper1/session.py::save_pdf).
mpl.rcParams.update(
    {
        # SVG: real <text> elements (Corel ignores the <use>-referenced glyph outlines "path" writes).
        "svg.fonttype": "none",
        # PDF: the 14 core fonts are never embedded, so Corel does not meet matplotlib's CID subsets.
        "pdf.use14corefonts": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "path.simplify": False,
    }
)

ROOT = Path(r"C:\Users\aksel\.ethograph\cache\example_data\Moll2025")
DATASET = ROOT / "Trial_data.nc"
EMBEDDINGS = ROOT / "embeddings_s3d"
LABELS = ROOT / "labels.tsv"
MAPPING = ROOT / ".ethograph" / "mapping.txt"
OUT_DIR = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\paper\video_feats")
TRIAL = 41
#: The GUI's defaults (``app_state.AppStateSpec``).
SORT_WINDOW_S = 0.5
SORT_OVERLAP = 0.5
EXCLUSION_PERCENTILE = 98.0
COLORMAP = "RdBu_r"
LABEL_STRIP_HEIGHT = 0.08  # of the heatmap's height


def load_trial(trial: int) -> tuple[xr.Dataset, np.ndarray]:
    """The trial's dataset and its S3D embeddings ``(T, 1024)`` on the trial clock."""
    dt = xr.open_datatree(DATASET)
    node = next(n for n in dt.children.values() if int(n.attrs["trial"]) == trial)
    ds = node.ds.load()
    video = pd.read_csv(ROOT / "Trial_data_labels.tsv", sep="\t").query("trial == @trial")["video_cam-1"].iloc[0]
    emb = np.load(EMBEDDINGS / f"{Path(video).stem}.npy")
    n = ds.sizes["time"]
    if emb.shape[0] != n:
        raise ValueError(f"{video}: {emb.shape[0]} embedding frames for {n} trial samples")
    return ds, emb


def gui_heatmap(data: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalise, sort and clip ``(T, C)`` as the GUI heatmap renders it → ``(C, T)``, vmax."""
    normalized = z_normalize(np.asarray(data, dtype=float)).astype(np.float32)
    np.nan_to_num(normalized, copy=False, nan=0.0)
    vmax = float(np.percentile(np.abs(normalized), EXCLUSION_PERCENTILE)) or 1.0
    order = argmax_window_order(normalized, time, SORT_WINDOW_S, SORT_OVERLAP)
    return normalized[:, order].T, vmax


def draw_heatmap(ax: plt.Axes, image: np.ndarray, time: np.ndarray, vmax: float, title: str) -> mpl.image.AxesImage:
    im = ax.imshow(
        image,
        aspect="auto",
        cmap=COLORMAP,
        vmin=-vmax,
        vmax=vmax,
        extent=(float(time[0]), float(time[-1]), image.shape[0], 0),
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_ylabel(f"{title}\n({image.shape[0]} dims, sorted)", fontsize=11)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.spines[["top", "right"]].set_visible(False)
    return im


def draw_labels(ax: plt.Axes, labels: pd.DataFrame, mapping: dict, time: np.ndarray) -> None:
    """The trial's state labels as coloured blocks, point events as lines."""
    for row in labels.itertuples():
        color = mapping[int(row.labels)]["color"]
        if row.event_type == "point":
            ax.axvline(row.onset_s, color=color, linewidth=1.5)
            continue
        ax.add_patch(Rectangle((row.onset_s, 0), row.offset_s - row.onset_s, 1, facecolor=color, linewidth=0))
    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)


def plot(ds: xr.Dataset, emb: np.ndarray, labels: pd.DataFrame, mapping: dict) -> plt.Figure:
    time = ds["time"].to_numpy()
    panels = [("embeddings_s3d", emb), ("s3d", ds["s3d"].to_numpy())]

    ratios = [h for _ in panels for h in (1.0, LABEL_STRIP_HEIGHT)]
    fig, axes = plt.subplots(
        len(panels) * 2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": ratios, "hspace": 0.06},
    )
    for (name, data), ax_heat, ax_lab in zip(panels, axes[0::2], axes[1::2], strict=True):
        image, vmax = gui_heatmap(data, time)
        im = draw_heatmap(ax_heat, image, time, vmax, name)
        cbar = fig.colorbar(im, ax=ax_heat, pad=0.01, fraction=0.03)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label("z-score", fontsize=9)
        draw_labels(ax_lab, labels, mapping, time)
        fig.colorbar(im, ax=ax_lab, pad=0.01, fraction=0.03).ax.set_visible(False)
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.suptitle(f"Trial {TRIAL}", fontsize=12, y=0.93)
    fig.subplots_adjust(top=0.9, bottom=0.08)
    for ax in axes:
        for artist in [*ax.patches, *ax.lines]:
            artist.set_clip_on(False)
    return fig


def main() -> None:
    ds, emb = load_trial(TRIAL)
    labels = pd.read_csv(LABELS, sep="\t").query("trial == @TRIAL")
    mapping = load_label_mapping(MAPPING)
    fig = plot(ds, emb, labels, mapping)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "embedding_heatmaps"
    with PdfPages(stem.with_suffix(".pdf")) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches=None, metadata={"Creator": "matplotlib"})
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    for ext in ("pdf", "svg", "png"):
        print(f"Wrote {stem.with_suffix('.' + ext)}")


if __name__ == "__main__":
    main()
