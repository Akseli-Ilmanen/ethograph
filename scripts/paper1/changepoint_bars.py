"""Segmental F1 with vs. without changepoint features, per architecture.

Reads every ``cv_{dataset}_{arch}_{features}_loss`` cross-validation folder
under :data:`ROOT`; a folder whose name carries ``_no_cp`` was trained without
the changepoint features, its partner with them. Each fold in ``folds.tsv`` is
one held-out session.

Architectures in :data:`EXCLUDE` are skipped. Nested bars: architecture →
metric (F1@50, F1@90) → with / without changepoint features. Bar height is
the mean over folds, the error bar its SEM, and the dots are the individual
folds. Writes ``changepoint_bars.{pdf,png}`` into
:data:`ROOT`.

    python scripts/paper1/changepoint_bars.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\aksel\Documents\Code\ethograph\projects\paper\changepoint")
#: ``raw`` or ``postprocessed`` — which column family of ``folds.tsv`` to plot.
STAGE = "postprocessed"
METRICS = ("f1@50", "f1@90")
FOLDER_RE = re.compile(r"^cv_(?P<dataset>[^_]+)_(?P<arch>.+)_all(?P<no_cp>_no_cp)?_loss$")
CONDITIONS = (("with changepoint features", False), ("without changepoint features", True))
COLORS = {False: "#2a6fdb", True: "#b0b7c3"}
ARCH_NAMES = {"mlp": "MLP", "mstcn": "MS-TCN", "c2f_tcn": "C2F-TCN"}
EXCLUDE = {"c2f_transformer"}


def load_folds(root: Path) -> pd.DataFrame:
    """One row per (architecture, condition, fold, metric) with its score."""
    rows = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        match = FOLDER_RE.match(run.name)
        if match is None or match["arch"] in EXCLUDE:
            continue
        folds = pd.read_csv(run / "folds.tsv", sep="\t")
        for metric in METRICS:
            for session, score in zip(folds["session"], folds[f"{STAGE}.{metric}"], strict=True):
                rows.append(
                    {
                        "arch": match["arch"],
                        "no_cp": match["no_cp"] is not None,
                        "session": session,
                        "metric": metric,
                        "score": score,
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No cv_*_loss folders under {root}")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> plt.Figure:
    archs = [a for a in ARCH_NAMES if a in set(df["arch"])] + sorted(set(df["arch"]) - ARCH_NAMES.keys())
    bar_w, metric_gap, arch_gap = 0.35, 0.25, 0.8
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(1.9 * len(archs) + 1.5, 4.2))
    arch_centres, metric_ticks, metric_labels = [], [], []
    x = 0.0
    for arch in archs:
        arch_start = x
        for metric in METRICS:
            for i, (label, no_cp) in enumerate(CONDITIONS):
                scores = df.query("arch == @arch and metric == @metric and no_cp == @no_cp")["score"].to_numpy()
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
                    color=COLORS[no_cp],
                    edgecolor="black",
                    linewidth=0.5,
                    label=label,
                    error_kw={"linewidth": 0.8},
                )
                jitter = rng.uniform(-bar_w * 0.2, bar_w * 0.2, scores.size)
                ax.scatter(pos + jitter, scores, s=8, color="black", zorder=3, linewidths=0)
            metric_ticks.append(x + bar_w * (len(CONDITIONS) - 1) / 2)
            metric_labels.append(metric.replace("f1", "F1"))
            x += bar_w * len(CONDITIONS) + metric_gap
        arch_centres.append((arch_start + x - metric_gap - bar_w) / 2)
        x += arch_gap - metric_gap

    ax.set_xticks(metric_ticks, metric_labels, fontsize=10)
    for centre, arch in zip(arch_centres, archs, strict=True):
        ax.annotate(
            ARCH_NAMES.get(arch, arch),
            xy=(centre, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -26),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=12,
        )
    ax.set_ylabel("segmental F1 (%)", fontsize=13)
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
        ncols=2,
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    df = load_folds(ROOT)
    summary = df.groupby(["arch", "metric", "no_cp"])["score"].agg(["mean", "std", "count"])
    print(summary.round(2).to_string())
    fig = plot(df)
    for ext in ("pdf", "png"):
        out = ROOT / f"changepoint_bars.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
