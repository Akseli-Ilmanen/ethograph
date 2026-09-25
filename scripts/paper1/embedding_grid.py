"""The S3D and FERAL embeddings of two videos as a 2 × 2 grid of GUI-style heatmaps.

Each ``embeddings_{extractor}/{video}.npy`` is ``(frames, dims)`` on the video's
clock; rows are z-scored, ordered by the GUI's "Sort by trial window" and
clipped at its exclusion percentile through :func:`embedding_heatmaps.gui_heatmap`.
Columns are the videos, rows the extractors; no labels, no playhead. The frame
rate comes from the dataset's ``fps`` attribute so the sort window is in seconds.

Writes ``embedding_grid.{pdf,svg,png}`` into :data:`OUT_DIR`.

    python scripts/paper1/embedding_grid.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from embedding_heatmaps import COLORMAP, DATASET, ROOT, gui_heatmap  # sets the CorelDRAW-safe rcParams
from matplotlib.backends.backend_pdf import PdfPages

OUT_DIR = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\paper\video_feats")
#: extractor folder → row title, top to bottom.
EXTRACTORS = {"embeddings_s3d": "S3D", "embeddings_feral": "FERAL"}
#: video stem → column title, left to right.
VIDEOS = {"2024-12-18_041_Crow1-cam-1": "Trial 41", "2024-12-17_115_Crow1-cam-1": "Trial 115"}


def video_fps() -> float:
    with xr.open_datatree(DATASET) as dt:
        node = next(iter(dt.children.values()))
        return float(node.attrs["fps"])


def plot(fps: float) -> plt.Figure:
    fig, axes = plt.subplots(
        len(EXTRACTORS),
        len(VIDEOS),
        figsize=(4.5 * len(VIDEOS) + 0.8, 3.2 * len(EXTRACTORS)),
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
        squeeze=False,
    )
    last_im = None
    for r, (folder, row_title) in enumerate(EXTRACTORS.items()):
        for c, (stem, col_title) in enumerate(VIDEOS.items()):
            emb = np.load(ROOT / folder / f"{stem}.npy").astype(float)
            time = np.arange(emb.shape[0]) / fps
            image, vmax = gui_heatmap(emb, time)
            ax = axes[r, c]
            last_im = ax.imshow(
                image,
                aspect="auto",
                cmap=COLORMAP,
                vmin=-vmax,
                vmax=vmax,
                extent=(0.0, float(time[-1]), image.shape[0], 0),
                interpolation="nearest",
                rasterized=True,
            )
            ax.set_yticks([])
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel(f"{row_title}\n({image.shape[0]} dims, sorted)", fontsize=11)
            if r == len(EXTRACTORS) - 1:
                ax.set_xlabel(f"Time (s)\n{col_title}", fontsize=11)
            else:
                ax.tick_params(axis="x", labelbottom=False)
    assert last_im is not None
    cbar = fig.colorbar(last_im, ax=axes, pad=0.02, fraction=0.025)
    cbar.set_label("z-score (per panel)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    return fig


def main() -> None:
    fig = plot(video_fps())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "embedding_grid"
    with PdfPages(stem.with_suffix(".pdf")) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches=None, metadata={"Creator": "matplotlib"})
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    for ext in ("pdf", "svg", "png"):
        print(f"Wrote {stem.with_suffix('.' + ext)}")


if __name__ == "__main__":
    main()
