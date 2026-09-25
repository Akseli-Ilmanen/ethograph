"""Segmental F1 with the top-20 S3D dimensions vs all 1024, both with changepoints and kinematics.

The values come from the MSc feature-ablation run ``Freddy_feature_ablation_20251021_161907``
(MovFormer, one held-out split per condition, epoch 100, changepoint-corrected
metrics) and are copied from the first cell of ``s3d_ablation.ipynb`` next to
this script; the ``test_results_epoch100.npy`` files it read live on the lab
machine and are not in this repo. Paired bars per metric (F1@50, F1@75,
F1@90). Writes ``s3d_ablation_bars.{pdf,svg,png}`` into :data:`OUT_DIR`.

    python scripts/paper1/s3d_ablation_bars.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Export settings CorelDRAW opens cleanly — see scripts/paper1/README.md.
mpl.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.use14corefonts": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "path.simplify": False,
    }
)

OUT_DIR = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\paper\s3d_ablation")
METRICS = ("F1@50", "F1@75", "F1@90")
#: condition → (subscript of the legend label, colour, scores per metric). Splits 4 and 5 of the run.
CONDITIONS = {
    "top20": ("20", "#2a6fdb", (83.0, 70.5, 54.5)),
    "all1024": ("1024", "#b0b7c3", (81.7, 67.8, 53.1)),
}
FOOTNOTE = "CP = changepoint, Kin = Kinematic"
LEGEND_FONTSIZE = 13


def draw_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    """A two-entry legend of plain text pieces: ``S3D`` + subscript + `` + CP + Kin.``.

    Math text (``$_{20}$``) would embed DejaVu as a CID font in the PDF and
    write ``<use>`` glyphs in the SVG, both of which CorelDRAW refuses.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    x0, y = ax.get_position().x0, ax.get_position().y1 + 0.06
    x = x0
    small = LEGEND_FONTSIZE * 0.7
    for subscript, color, _ in CONDITIONS.values():
        patch = mpl.patches.Rectangle(
            (x, y - 0.012), 0.08, 0.024, transform=fig.transFigure, facecolor=color, edgecolor="black", linewidth=0.5
        )
        fig.add_artist(patch)
        x += 0.095
        pieces = (("S3D", LEGEND_FONTSIZE, 0.0), (subscript, small, -0.008), (" + CP + Kin.", LEGEND_FONTSIZE, 0.0))
        for text, size, dy in pieces:
            t = fig.text(x, y + dy, text, fontsize=size, ha="left", va="center")
            bbox = to_fig.transform(t.get_window_extent(renderer))
            x = bbox[1, 0]
        x += 0.05


def plot() -> plt.Figure:
    bar_w, metric_gap = 0.35, 0.3
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    x = 0.0
    metric_ticks = []
    for m in range(len(METRICS)):
        for i, (_, color, scores) in enumerate(CONDITIONS.values()):
            ax.bar(x + i * bar_w, scores[m], bar_w, color=color, edgecolor="black", linewidth=0.5)
        metric_ticks.append(x + bar_w * (len(CONDITIONS) - 1) / 2)
        x += bar_w * len(CONDITIONS) + metric_gap

    ax.set_xticks(metric_ticks, METRICS, fontsize=10)
    ax.set_ylabel("segmental F1 (%)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.02, FOOTNOTE, ha="center", va="bottom", fontsize=12)
    # CorelDRAW drops SVG groups that carry a clip-path, so nothing may be clipped.
    for artist in [*ax.patches, *ax.collections, *ax.lines]:
        artist.set_clip_on(False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    draw_legend(fig, ax)
    return fig


def main() -> None:
    for subscript, _, scores in CONDITIONS.values():
        print(f"S3D_{subscript:5s} + CP + Kin.", np.array(scores))
    fig = plot()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "s3d_ablation_bars"
    with PdfPages(stem.with_suffix(".pdf")) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches=None, metadata={"Creator": "matplotlib"})
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    for ext in ("pdf", "svg", "png"):
        print(f"Wrote {stem.with_suffix('.' + ext)}")


if __name__ == "__main__":
    main()
