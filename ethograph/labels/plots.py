from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def plot_label_segments(
    ax: plt.Axes,
    df: pd.DataFrame,
    label_mappings: Dict[int, Dict],
    individual: Optional[str] = None,
    is_main: bool = True,
    fraction: float = 0.2,
    alpha: float = 0.8,
) -> None:
    """Plot label segments from an intervals DataFrame.

    Args:
        ax: Matplotlib axis to plot on
        df: Intervals DataFrame with columns onset_s, offset_s, labels, individual
        label_mappings: Dict mapping label IDs to color info
        individual: If given, only plot segments for this individual
        is_main: If True, plot full-height rectangles; if False, plot small rectangles at top
        fraction: Height fraction for non-main rectangles

    Example::

        import ethograph as eto
        from ethograph.labels.intervals import load_label_mapping

        dt = eto.open("data.nc")
        label_mappings = load_label_mapping("mapping.txt")

        fig, ax = plt.subplots()
        # df is an intervals DataFrame with onset_s, offset_s, labels, individual
        plot_label_segments(ax, df, label_mappings)
        plt.show()
    """
    if individual is not None:
        df = df[df["individual"] == individual]

    for _, row in df.iterrows():
        draw_label_rectangle(
            ax,
            row["onset_s"],
            row["offset_s"],
            int(row["labels"]),
            label_mappings,
            is_main,
            fraction=fraction,
            alpha=alpha,
        )


def draw_label_rectangle(
    ax: plt.Axes,
    start_time: float,
    end_time: float,
    labels: int,
    label_mappings: Dict[int, Dict],
    is_main: bool = True,
    fraction: Optional[float] = None,
    alpha: float = 0.8,
) -> None:
    """Draw a label rectangle on a matplotlib axis.

    Args:
        ax: Matplotlib axis to plot on
        start_time: Start time of the label
        end_time: End time of the label
        labels: Label class ID for color mapping
        label_mappings: Dict mapping label IDs to color info
        is_main: If True, draw full-height rectangle; if False, draw small rectangle at top
        fraction: Height fraction for non-main rectangles

    Example::

        fig, ax = plt.subplots()
        ax.plot(time, signal)
        draw_label_rectangle(ax, 1.2, 3.5, label_id=1, label_mappings=label_mappings)
    """
    if labels not in label_mappings:
        return

    color = label_mappings[labels]["color"]

    if is_main:
        ax.axvspan(start_time, end_time, alpha=alpha, color=color, zorder=-10)
    else:
        y_min, y_max = ax.get_ylim()
        height = (y_max - y_min) * fraction
        rect = patches.Rectangle(
            (start_time, y_max - height),
            end_time - start_time,
            height,
            color=color,
            alpha=alpha,
            zorder=10,
        )
        ax.add_patch(rect)


def plot_label_segments_multirow(
    ax: plt.Axes,
    df: pd.DataFrame,
    label_mappings: Dict[int, Dict[str, str]],
    row_index: int = 0,
    row_spacing: float = 0.8,
    rect_height: float = 0.7,
    alpha: float = 0.7,
    individual: Optional[str] = None,
) -> None:
    """Plot label segments at a specific row position.

    Useful for comparing ground truth vs. predictions on the same axis by
    placing each on a different row.

    Args:
        ax: Matplotlib axis to plot on
        df: Intervals DataFrame with columns onset_s, offset_s, labels, individual
        label_mappings: Dict mapping label IDs to color info
        row_index: Row number (0-based) for vertical positioning
        row_spacing: Vertical spacing between rows
        rect_height: Height of each rectangle
        alpha: Transparency of rectangles
        individual: If given, only plot segments for this individual

    Example::

        import ethograph as eto
        from ethograph.labels.intervals import load_label_mapping

        dt = eto.open("data.nc")
        pred_dt = eto.open("predictions.nc")
        label_mappings = load_label_mapping("mapping.txt")

        fig, ax = plt.subplots()
        ax.set_yticks([0, 0.8])
        ax.set_yticklabels(["ground truth", "predictions"])

        # gt_df, pred_df are intervals DataFrames with onset_s, offset_s, labels, individual
        gt_df = ...
        pred_df = ...

        plot_label_segments_multirow(ax, gt_df, label_mappings, row_index=0)
        plot_label_segments_multirow(ax, pred_df, label_mappings, row_index=1)
        plt.show()
    """
    if individual is not None:
        df = df[df["individual"] == individual]

    y_base = row_index * row_spacing
    for _, row in df.iterrows():
        _draw_rectangle(
            ax,
            row["onset_s"],
            row["offset_s"],
            y_base,
            rect_height,
            int(row["labels"]),
            label_mappings,
            alpha,
        )


def _draw_rectangle(
    ax: plt.Axes,
    start_time: float,
    end_time: float,
    y_base: float,
    height: float,
    labels: int,
    label_mappings: Dict[int, Dict[str, str]],
    alpha: float,
) -> None:
    if labels not in label_mappings:
        return

    color = label_mappings[labels]["color"]
    rect = patches.Rectangle(
        (start_time, y_base),
        end_time - start_time,
        height,
        color=color,
        alpha=alpha,
        zorder=-10,
    )
    ax.add_patch(rect)


def plot_confidence_pdf(
    confidence_map: dict,
    labels_df: pd.DataFrame,
    dt,
    label_mappings: Dict[int, Dict],
    output_path: str | Path | None = None,
    confidence_threshold: float = 0.75,
    segment_confidence_threshold: float = 0.6,
) -> tuple[Path, dict]:
    """Plot per-trial prediction confidence and save to a PDF.

    Parameters
    ----------
    confidence_map : dict
        {trial: confidence_array (T,)}, built from ``PredictionsStore.get_confidence``
        (see :mod:`ethograph.labels.predictions`).
    labels_df : pd.DataFrame
        Predictions intervals DataFrame (onset_s, offset_s, labels, trial, individual).
    dt : TrialTree
        Used to get per-trial time coordinates.
    label_mappings : dict
        Mapping from label int → {'color': ..., 'name': ...}.
    output_path : Path or None
        Where to save the PDF. If None, a temp file is created.
    confidence_threshold : float
        Frame-level threshold below which frames are marked low-confidence.
    segment_confidence_threshold : float
        Segment-level mean confidence below which the segment is highlighted.

    Returns
    -------
    output_path : Path
        Path to the saved PDF.
    highlighted : dict
        {trial: bool} — True if the trial was highlighted red (low confidence).
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        output_path = Path(tmp.name)
        tmp.close()
    output_path = Path(output_path)

    trials = sorted(confidence_map.keys())
    n_cols = 3
    n_rows = max(1, (len(trials) + n_cols - 1) // n_cols)
    highlighted: dict = {}

    with PdfPages(output_path) as pdf:
        # --- Legend / explanation page ---
        leg_fig, leg_ax = plt.subplots(figsize=(14, 5))
        leg_ax.axis("off")

        lines = [
            (0.92, r"$\bf{Confidence\ score\ (per\ frame):}$", 14, "black"),
            (0.78, r"$c_t = 1 - \frac{H(p_t)}{\log K}$", 13, "black"),
            (
                0.64,
                r"where $p_t \in \mathbb{R}^K$ is the softmax output at frame $t$,"
                r"  $H(p_t) = -\sum_k p_k \log p_k$ is the entropy,"
                r"  and $K$ is the number of classes.",
                10,
                "#333333",
            ),
            (
                0.50,
                r"$c_t = 1.0$ means the model is certain;  $c_t = 0.0$ means the model is maximally uncertain (uniform distribution).",  # noqa: E501
                10,
                "#333333",
            ),
            (0.34, r"$\bf{Thresholds:}$", 12, "black"),
            (
                0.22,
                rf"Frame threshold (orange dashed, $\tau_{{frame}}={confidence_threshold:.2f}$): "
                r"frames with $c_t < \tau_{frame}$ are plotted as red dots.",
                10,
                "darkorange",
            ),
            (
                0.10,
                rf"Segment threshold (red dotted, $\tau_{{seg}}={segment_confidence_threshold:.2f}$): "
                r"segments whose mean frame confidence $\bar{{c}} < \tau_{{seg}}$ are shaded red — "
                r"state events only, a point event has no span to average over. "
                r"A trial is marked low-confidence (red border) if its overall mean $< \tau_{{frame}}$ or any segment $< \tau_{{seg}}$.",  # noqa: E501
                10,
                "red",
            ),
        ]
        for y, txt, size, color in lines:
            leg_ax.text(
                0.02,
                y,
                txt,
                transform=leg_ax.transAxes,
                fontsize=size,
                color=color,
                va="top",
                wrap=True,
            )

        pdf.savefig(leg_fig, bbox_inches="tight")
        plt.close(leg_fig)

        # --- Per-trial subplots ---
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2.5 * n_rows))
        axes = np.array(axes).reshape(n_rows, n_cols).flatten()

        for idx, trial in enumerate(trials):
            ax = axes[idx]
            confidence = confidence_map[trial]

            if confidence is None:
                highlighted[trial] = False
                ax.set_title(f"trial-{trial}\n(no confidence)", fontsize=9)
                ax.axis("off")
                continue

            ds = dt.trial(trial)
            time_coord = ds.time.values if "time" in ds.coords else np.arange(len(confidence))
            n = min(len(confidence), len(time_coord))
            confidence = confidence[:n]
            time_coord = time_coord[:n]

            trial_intervals = labels_df[labels_df["trial"] == trial]

            for _, row in trial_intervals.iterrows():
                label_id = int(row["labels"])
                color = label_mappings.get(label_id, {}).get("color", "gray")
                ax.axvspan(row["onset_s"], row["offset_s"], alpha=0.25, color=color, zorder=-10)

            ax.plot(time_coord, confidence, color="steelblue", lw=0.8, alpha=0.9)
            ax.axhline(confidence_threshold, color="orange", lw=0.8, ls="--", alpha=0.7)
            ax.axhline(segment_confidence_threshold, color="red", lw=0.6, ls=":", alpha=0.6)

            low_mask = confidence < confidence_threshold
            if np.any(low_mask):
                ax.scatter(
                    time_coord[low_mask],
                    confidence[low_mask],
                    color="red",
                    s=4,
                    alpha=0.5,
                    zorder=5,
                )

            has_low_segment = False
            for _, row in trial_intervals.iterrows():
                onset, offset = row["onset_s"], row["offset_s"]
                seg_mask = (time_coord >= onset) & (time_coord <= offset)
                if seg_mask.any():
                    seg_conf = np.mean(confidence[seg_mask])
                    if seg_conf < segment_confidence_threshold:
                        has_low_segment = True
                        ax.axvspan(onset, offset, color="red", alpha=0.2, zorder=4)

            mean_conf = float(np.mean(confidence))
            low = mean_conf < confidence_threshold or has_low_segment
            highlighted[trial] = low

            ax.set_title(
                f"trial-{trial}  mean={mean_conf:.2f}",
                fontsize=9,
                color="red" if low else "black",
                weight="bold" if low else "normal",
            )
            if low:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2)

            ax.set_ylim(0, 1.05)
            ax.set_xlabel("time (s)", fontsize=7)
            ax.set_ylabel("confidence", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        for j in range(len(trials), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return output_path, highlighted
