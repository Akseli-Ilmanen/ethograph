"""Frame-wise and segmental F1 per bird, MLP vs. C2F-TCN.

Reads every ``cv_{bird}_{arch}_all_loss`` cross-validation folder under
:data:`ROOT`; each fold in ``folds.tsv`` is one held-out session. Nested
bars: bird → metric (frame F1, F1@50, F1@75, F1@90) → architecture. Bar
height is the mean over folds, the error bar its SEM, and the dots are the
individual folds. Writes ``crossval_bars.{pdf,svg,png}`` into :data:`ROOT`.

    python scripts/paper1/crossval_bars.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Export settings CorelDRAW opens cleanly (same as scripts/paper1/session.py::save_pdf):
# TrueType (42) rather than Type 3 fonts, SVG text as outlines, no path simplification.
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

ROOT = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\paper\crossval_birds")
#: ``raw`` or ``postprocessed`` — which column family of ``folds.tsv`` to plot.
STAGE = "postprocessed"
#: ``folds.tsv`` column suffix → tick label, in plotting order.
METRICS = {"frame_f1": "F1@frame", "f1@50": "F1@50", "f1@75": "F1@75", "f1@90": "F1@90"}
FOLDER_RE = re.compile(r"^cv_(?P<bird>[^_]+)_(?P<arch>.+)_all_loss$")
ARCH_NAMES = {"mlp": "MLP (per-frame baseline)", "c2f_tcn": "C2F-TCN"}
COLORS = {"mlp": "#b0b7c3", "c2f_tcn": "#2a6fdb"}
BIRD_NAMES = {"crow1": "Crow 1", "crow2": "Crow 2", "crow3": "Crow 3"}


def load_folds(root: Path) -> pd.DataFrame:
    """One row per (bird, architecture, fold, metric) with its score."""
    rows = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        match = FOLDER_RE.match(run.name)
        if match is None or match["arch"] not in ARCH_NAMES:
            continue
        folds = pd.read_csv(run / "folds.tsv", sep="\t")
        for metric in METRICS:
            for session, score in zip(folds["session"], folds[f"{STAGE}.{metric}"], strict=True):
                rows.append(
                    {
                        "bird": match["bird"],
                        "arch": match["arch"],
                        "session": session,
                        "metric": metric,
                        "score": score,
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No cv_*_all_loss folders under {root}")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> plt.Figure:
    birds = [b for b in BIRD_NAMES if b in set(df["bird"])] + sorted(set(df["bird"]) - BIRD_NAMES.keys())
    archs = list(ARCH_NAMES)
    bar_w, metric_gap, bird_gap = 0.35, 0.25, 0.8
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(3.4 * len(birds) + 1.5, 4.2))
    bird_centres, metric_ticks, metric_labels = [], [], []
    x = 0.0
    for bird in birds:
        bird_start = x
        for metric, tick in METRICS.items():
            for i, arch in enumerate(archs):
                scores = df.query("bird == @bird and metric == @metric and arch == @arch")["score"].to_numpy()
                pos = x + i * bar_w
                if scores.size == 0:
                    continue
                sem = scores.std(ddof=1) / np.sqrt(scores.size) if scores.size > 1 else 0.0
                ax.bar(
                    pos,
                    scores.mean(),
                    bar_w,
                    yerr=sem,
                    capsize=2,
                    color=COLORS[arch],
                    edgecolor="black",
                    linewidth=0.5,
                    label=ARCH_NAMES[arch],
                    error_kw={"linewidth": 0.8},
                )
                jitter = rng.uniform(-bar_w * 0.2, bar_w * 0.2, scores.size)
                ax.scatter(pos + jitter, scores, s=8, color="black", zorder=3, linewidths=0)
            metric_ticks.append(x + bar_w * (len(archs) - 1) / 2)
            metric_labels.append(tick)
            x += bar_w * len(archs) + metric_gap
        bird_centres.append((bird_start + x - metric_gap - bar_w) / 2)
        x += bird_gap - metric_gap

    ax.set_xticks(metric_ticks, metric_labels, fontsize=9)
    for centre, bird in zip(bird_centres, birds, strict=True):
        ax.annotate(
            BIRD_NAMES.get(bird, bird),
            xy=(centre, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -26),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=12,
        )
    ax.set_ylabel("F1 Score (%)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=len(archs),
        fontsize=12,
    )
    # CorelDRAW drops SVG groups that carry a clip-path, so nothing may be clipped.
    for artist in [*ax.patches, *ax.collections, *ax.lines]:
        artist.set_clip_on(False)
    fig.tight_layout()
    return fig


def main() -> None:
    df = load_folds(ROOT)
    summary = df.groupby(["bird", "metric", "arch"])["score"].agg(["mean", "std", "count"])
    print(summary.round(2).to_string())
    fig = plot(df)
    stem = ROOT / "crossval_bars"
    with PdfPages(stem.with_suffix(".pdf")) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches=None, metadata={"Creator": "matplotlib"})
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    for ext in ("pdf", "svg", "png"):
        print(f"Wrote {stem.with_suffix('.' + ext)}")


if __name__ == "__main__":
    main()
