"""Label grid view: video frames at label times, and verdicts by clicking them.

Opened from the Labels tab's Curation section (**Label grid view…**) on the
labels in scope — or from the Predict dialog on what it just predicted. One
window with two tabs. On *Setup* the label classes are listed read-only (the
scope area in the Labels tab is the one place to pick them), the trials are
the ones the trials table shows (its filters are the one place trials are
included or excluded, for every operation), and the cameras that matter are
picked. *Generate* decodes, for every matching label instance, the
video frame closest to its time — one frame per point event, a start and an
end frame per state event — overlays the pose when a pose file exists for
that (trial, camera), and fills the *Frames* tab with a grid of thumbnails.
Each tile is titled with the label and carries its confidence beside that
title, large and in red once it falls under the threshold; trial, camera, time
and ``labeling_method`` read underneath. The same threshold outlines doubtful
tiles in red and **Histogram…** shows where the scores pile up.

**A double click always navigates**, whatever the mode: it jumps the main
GUI to that trial and time, or — in frame-by-frame curation mode — drops
straight into the review at that boundary
(:meth:`CurationPanel.start_review_at`). Qt opens a double click with a plain
press, which has already toggled the tile; the double click toggles it back,
so navigating never leaves a verdict behind.

A **single** click **tags the label for review** (orange): a batch is
mostly right, so the click marks the bad ones, and **Done** curates every
other automated label on screen — the tagged ones stay automated, to be
looked at. **Tag low-confidence** pre-tags the tiles the confidence
threshold outlines: a low score is a reason to doubt a label, never to
approve it. There is no mode in which a click approves.

The **Label** combo narrows a grid built from several classes to one of them,
and it narrows the operations too: the flagged tiles, **Done** and the PDF
all run over what is on screen, so a scope of several classes is curated one
class at a time without reopening the dialog.

The grid's column count is adjustable and the whole grid exports to a
paginated PDF. The same verdict machinery (:class:`TileVerdicts`) drives the
video grid (``dialog_video_grid.py``).
"""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import QEventLoop, QLocale, QObject, QRect, Qt, QTimer, Signal
from qtpy.QtGui import (
    QColor,
    QDoubleValidator,
    QFont,
    QImage,
    QPageSize,
    QPainter,
    QPdfWriter,
    QPen,
    QPixmap,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.app_constants import MULTIDIM_COLORS
from ethograph.gui.file_dialogs import browse_save_file
from ethograph.gui.notify import notify
from ethograph.gui.pose_fill import VideoFrameSource
from ethograph.gui.pose_render import POSES_DATASET_SUFFIX, PoseRenderData, load_pose_from_file
from ethograph.gui.video_manager import probe_video
from ethograph.io.time_model import TimeRange
from ethograph.labels.curation import subject_str
from ethograph.labels.intervals import (
    EVENT_TYPE_POINT,
    HUMAN_CONFIDENCE,
    LABELING_AUTOMATED,
    LABELING_CURATED,
    LABELING_MANUAL,
)
from ethograph.labels.rescore import RULES
from ethograph.labels.workflow import DEFAULT_CONFIDENCE

logger = logging.getLogger(__name__)

#: Longest side frames are decoded at — thumbnails, not the full video.
THUMB_MAX_SIDE = 512

#: How long the GUI gets to redraw before a panel is screenshotted. The
#: panels paint synchronously (pyqtgraph) — the jump itself returns with the
#: plots updated — so this only needs to cover deferred zero-timers (labels
#: redraw, marker sync) and short range-change debounces; longer after a
#: trial switch, whose refresh cascade schedules more of them.
PANEL_SETTLE_MS = 100
PANEL_TRIAL_SETTLE_MS = 300

#: Size the dialog grows to when the grid first appears — the setup form is
#: narrow, a grid of thumbnails is not.
_GRID_MIN_WIDTH = 1100
_GRID_MIN_HEIGHT = 780

#: Quiet period after a resize before the tiles are rescaled, so dragging the
#: window edge does not rescale every pixmap per pixel.
_REFLOW_DEBOUNCE_MS = 120

#: Outline drawn around a tile whose label falls below the confidence
#: threshold — in the grid (stylesheet) and in the PDF (pen colour).
LOW_CONFIDENCE_COLOR = "#d94040"
_LOW_CONFIDENCE_STYLE = f"QFrame#frameCell {{ border: 2px solid {LOW_CONFIDENCE_COLOR}; border-radius: 3px; }}"

#: The confidence beside a tile's title: big enough to read across a grid, and
#: in the flag red once it falls under the threshold.
CONFIDENCE_OK_COLOR = "#9aa0a6"
CONFIDENCE_FONT_PX = 15

#: Tile outlines for the verdict a click gave: curated (green) or flagged as
#: wrong (orange — distinct from the confidence red, which is a hint, not a
#: verdict).
#: A tile tagged for review — the one verdict a click gives.
TAGGED_COLOR = "#ff9f1c"
_TAGGED_STYLE = f"QFrame#frameCell {{ border: 3px solid {TAGGED_COLOR}; border-radius: 3px; }}"

#: Confidence-histogram popup: dark canvas (the label colours are picked for
#: one), plots per row and each plot's floor.
_HIST_BG = "#1a1d21"
_HIST_COLUMNS = 3
_HIST_MIN_SIZE = (300, 320)
_HIST_BINS = 20

#: A label class drawn in its own colour would hide the flagged part of its
#: histogram if that colour is itself reddish — those bars go neutral instead
#: (the plot title still carries the label's colour).
_HIST_NEUTRAL = "#8a9099"
_HIST_MIN_COLOR_DISTANCE = 90.0


# ----------------------------------------------------------------------
# Pure logic (Qt-free, unit-tested in tests/test_unit/test_label_gridview.py)
# ----------------------------------------------------------------------


#: A tile's kind. A point event is one frame; a state event is one tile
#: holding its onset frame and its offset frame side by side, so a label is
#: one tile and a verdict on it is one click.
TILE_POINT = "point"
TILE_STATE = "state"

#: The review field a frame of a state tile stands for.
FIELD_START = "start"
FIELD_END = "end"


@dataclass
class FrameEntry:
    """One tile of the grid: a label seen by one camera."""

    trial: object
    camera: str | None
    label_id: int
    name: str
    event_type: str
    boundary: str  # TILE_POINT | TILE_STATE
    #: The onset (or the point event's moment), trial-relative.
    t_rel: float
    onset_s: float
    offset_s: float
    individual: object = None
    individual_rec: object = None
    #: How sure the label is: 1.0 for a human label, the model's own score for
    #: a predicted one (see ``labels/onset_model.py``).
    confidence: float = HUMAN_CONFIDENCE
    #: Who vouches for the label (``labels/curation.py``).
    labeling_method: str = LABELING_MANUAL
    color_hex: str = "#ffffff"
    image: np.ndarray | None = None
    frame_idx: int | None = None
    #: A state tile's second frame, at the offset.
    end_image: np.ndarray | None = None
    end_frame_idx: int | None = None
    cropped: bool = False
    error: str | None = None
    #: (panel title, QImage) screenshots of ticked GUI panels around the
    #: label, shared between the cameras of the same label.
    panels: list = field(default_factory=list)

    @property
    def is_state(self) -> bool:
        return self.boundary == TILE_STATE

    @property
    def span(self) -> int:
        """Grid columns the tile takes: a state tile is two frames wide."""
        return 2 if self.is_state else 1

    def frames(self) -> list[tuple[str, float]]:
        """``(review field, time)`` for each frame the tile shows."""
        if self.is_state:
            return [(FIELD_START, self.t_rel), (FIELD_END, self.offset_s)]
        return [(TILE_POINT, self.t_rel)]


def is_low_confidence(entry: "FrameEntry", threshold: float) -> bool:
    """Whether *entry* should be flagged. A threshold of 0 flags nothing."""
    return threshold > 0.0 and entry.confidence < threshold


#: Decimal places a confidence is stored and shown to. A model's scores live
#: at the bottom of the range, where 0.0002 and 0.001 are different answers.
CONFIDENCE_DECIMALS = 12


#: Decimal places a confidence is *displayed* to. Three, not two: an onset
#: model's scores cluster low, where 0.004 and 0.001 are different answers and
#: two decimals shows both as 0.00. (The threshold box keeps full precision —
#: that is what CONFIDENCE_DECIMALS is for.)
CONFIDENCE_DISPLAY_DECIMALS = 3


def format_confidence(value: float) -> str:
    """*value* as it is typed back into a :class:`ConfidenceEdit`."""
    return f"{round(float(value), CONFIDENCE_DECIMALS):.{CONFIDENCE_DECIMALS}f}".rstrip("0").rstrip(".") or "0"


def confidence_display(value: float) -> str:
    """A confidence as it reads anywhere in the GUI that shows one."""
    return f"{float(value):.{CONFIDENCE_DISPLAY_DECIMALS}f}"


class ConfidenceEdit(QLineEdit):
    """A confidence in [0, 1], typed rather than stepped.

    A spin box has to pick a number of decimals, and every choice is wrong
    for a probability: two decimals cannot tell 0.0002 from 0, and twelve
    make the arrows useless. So this is a plain line edit with the spin box's
    API — ``value()``, ``setValue()``, ``valueChanged`` — and callers cannot
    tell the difference. Empty text reads as 0.
    """

    valueChanged = Signal(float)

    def __init__(self, value: float = DEFAULT_CONFIDENCE, parent=None):
        super().__init__(parent)
        self._value = 0.0
        validator = QDoubleValidator(0.0, 1.0, CONFIDENCE_DECIMALS, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setLocale(QLocale.c())
        self.setLocale(QLocale.c())
        self.setValidator(validator)
        self.setMaximumWidth(90)
        self.setPlaceholderText("off")
        self.setValue(value)
        self.textChanged.connect(self._on_text)
        self.editingFinished.connect(self._normalise)

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:
        """Show *value*, announcing it exactly when it is a different number.

        Two boxes bound to each other (the grid's and the histogram's) settle
        after one round trip: the second call finds the value it already
        holds and stays quiet.
        """
        new = min(1.0, max(0.0, float(value)))
        text = format_confidence(new)
        if text != self.text():
            blocked = self.blockSignals(True)
            self.setText(text)
            self.blockSignals(blocked)
        self._announce(new)

    def _on_text(self, text: str) -> None:
        new = self._parse(text)
        if new is not None:
            self._announce(new)

    def _announce(self, value: float) -> None:
        if value == self._value:
            return
        self._value = value
        self.valueChanged.emit(value)

    def _normalise(self) -> None:
        """Once typing is over, show the number that is actually in force."""
        if self._parse(self.text()) is None:
            self.setText(format_confidence(self._value))

    @staticmethod
    def _parse(text: str) -> float | None:
        """*text* as a confidence, or ``None`` while it is still half-typed."""
        stripped = text.strip()
        if not stripped:
            return 0.0
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        return min(1.0, max(0.0, parsed))


def confidence_text(entry) -> str:
    """The confidence as it reads beside a tile's title."""
    return confidence_display(entry.confidence)


def confidence_style(entry, threshold: float) -> str:
    """Stylesheet for that reading — red exactly when the tile is flagged."""
    color = LOW_CONFIDENCE_COLOR if is_low_confidence(entry, threshold) else CONFIDENCE_OK_COLOR
    return f"font-weight: bold; font-size: {CONFIDENCE_FONT_PX}px; color: {color};"


def _mapping_color_hex(info: dict) -> str:
    color = info.get("color")
    if color is None:
        return "#ffffff"
    return "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in color[:3]))


def entry_key(entry) -> tuple:
    """The label an entry belongs to — two cameras share one key, so a
    verdict on either tile is a verdict on the label."""
    return (
        str(entry.trial),
        int(entry.label_id),
        round(float(entry.onset_s), 6),
        subject_str(entry.individual),
        subject_str(entry.individual_rec),
    )


def entry_inst(entry) -> dict:
    """The label row an entry stands for, as the curation helpers want it."""
    return {
        "trial": entry.trial,
        "labels": entry.label_id,
        "onset_s": entry.onset_s,
        "offset_s": entry.offset_s,
        "individual": entry.individual,
        "individual_rec": entry.individual_rec,
        "event_type": entry.event_type,
    }


def build_frame_entries(
    labels_df: pd.DataFrame,
    mappings: dict,
    label_ids: list[int],
    cameras: list[str | None],
    allowed_trials: set[str] | None = None,
    methods: frozenset[str] | None = None,
) -> list[FrameEntry]:
    """Expand matching label rows into grid entries.

    One entry per label — a point event's frame, or a state event's onset
    and offset frames in one tile — times trial-relative, each repeated for
    every selected camera so a label's views sit next to each other in the
    grid. *methods* keeps only the labeling methods named (``None`` keeps
    every label).
    """
    if labels_df is None or labels_df.empty:
        return []
    rows = labels_df[labels_df["labels"].isin(label_ids)]
    if allowed_trials is not None:
        rows = rows[rows["trial"].astype(str).isin(allowed_trials)]
    rows = rows.sort_values(["trial", "onset_s"])

    entries: list[FrameEntry] = []
    for _, row in rows.iterrows():
        if not keep_method(row, methods):
            continue
        label_id = int(row["labels"])
        info = mappings.get(label_id, {})
        name = str(info.get("name", label_id))
        event_type = str(info.get("event_type", "state"))
        onset = float(row["onset_s"])
        offset = float(row["offset_s"])
        is_point = event_type == EVENT_TYPE_POINT or not math.isfinite(offset)
        for camera in cameras:
            entries.append(
                FrameEntry(
                    trial=row["trial"],
                    camera=camera,
                    label_id=label_id,
                    name=name,
                    event_type=EVENT_TYPE_POINT if is_point else "state",
                    boundary=TILE_POINT if is_point else TILE_STATE,
                    t_rel=onset,
                    onset_s=onset,
                    offset_s=offset,
                    individual=row.get("individual"),
                    individual_rec=row.get("individual_rec"),
                    confidence=_row_confidence(row),
                    labeling_method=_row_method(row),
                    color_hex=_mapping_color_hex(info),
                )
            )
    return entries


def seeds_from_entries(entries: list[FrameEntry]) -> list[dict]:
    """Review seeds for *entries* — one per boundary, cameras deduplicated.

    A state tile is two boundaries (its onset, then its offset); a label two
    cameras saw is two tiles but one label, and the review queue must stop
    at each boundary once. Each seed is the label row plus the ``field`` to
    edit, which is what :func:`ethograph.labels.curation.targets_from_seeds`
    consumes.
    """
    seeds: dict[tuple, dict] = {}
    for entry in entries:
        for field_name, _t in entry.frames():
            key = (*entry_key(entry), field_name)
            seeds.setdefault(key, {**entry_inst(entry), "field": field_name})
    return list(seeds.values())


def pack_tiles(entries: list, columns: int) -> list[tuple]:
    """Where each tile goes in a *columns*-wide grid: ``(entry, row, col, span)``.

    Tiles fill a row left to right in the given order; a state tile is two
    columns wide and wraps to the next row when only one column is left, so
    the two frames of one label never straddle a row break. Shared by the
    screen layout and the PDF, so the printed sheet matches the screen.
    """
    columns = max(1, int(columns))
    placed: list[tuple] = []
    row, col = 0, 0
    for entry in entries:
        span = min(entry.span, columns)
        if col + span > columns:
            row, col = row + 1, 0
        placed.append((entry, row, col, span))
        col += span
        if col >= columns:
            row, col = row + 1, 0
    return placed


def flagged_trials(entries: list[FrameEntry], threshold: float) -> set[str]:
    """Trials holding at least one entry below *threshold* (as strings)."""
    return {str(entry.trial) for entry in entries if is_low_confidence(entry, threshold)}


def label_filter_choices(entries: list[FrameEntry]) -> list[tuple[int | None, str]]:
    """The grid's label filter: every class in the grid, plus "all" first.

    Each choice carries its tile count, so a grid built from several classes
    says how much of it each one is. The ``None`` choice is the unfiltered
    grid — the first, and the one a grid opens on.
    """
    counts: dict[int, int] = {}
    names: dict[int, str] = {}
    for entry in entries:
        counts[entry.label_id] = counts.get(entry.label_id, 0) + 1
        names.setdefault(entry.label_id, entry.name)
    choices: list[tuple[int | None, str]] = [(None, f"All labels ({len(entries)})")]
    for label_id in sorted(counts, key=lambda i: (names[i], i)):
        choices.append((label_id, f"{names[label_id]} ({counts[label_id]})"))
    return choices


def filter_entries(entries: list[FrameEntry], label_id: int | None) -> list[FrameEntry]:
    """The entries one filter choice shows; ``None`` shows all of them."""
    if label_id is None:
        return list(entries)
    return [entry for entry in entries if entry.label_id == label_id]


#: How a grid can order what it shows, key -> combo text. Reviewing a model's
#: output, the useful order is by confidence: it puts every doubtful label on
#: one screen instead of scattering them through the trials.
GRID_SORT_ORDERS = {
    "trial": "Trial, then time",
    "confidence_asc": "Confidence: lowest first",
    "confidence_desc": "Confidence: highest first",
}

#: The video grid's own list — its clips play together, so their length is an
#: order in its own right (clips of a similar length end around the same time).
VIDEO_GRID_SORT_ORDERS = {"duration": "Duration: shortest first", **GRID_SORT_ORDERS}


def _trial_key(entry):
    """Trial id sorted numerically where it can be, textually otherwise."""
    trial = str(getattr(entry, "trial", ""))
    try:
        return (0, int(trial), "")
    except ValueError:
        return (1, 0, trial)


def sort_entries(entries: list, order: str) -> list:
    """*entries* in the order *order* names, stably.

    Works on a frame grid's entries and a video grid's clips alike — both
    carry ``confidence``, ``trial`` and ``onset_s``. Ties always fall back to
    (trial, time), so a screenful of equal confidences still reads in a
    sensible order rather than an arbitrary one.
    """
    if order not in (GRID_SORT_ORDERS | VIDEO_GRID_SORT_ORDERS):
        raise ValueError(f"Unknown grid sort order {order!r} (expected one of {', '.join(VIDEO_GRID_SORT_ORDERS)}).")

    def fallback(entry):
        return (_trial_key(entry), float(getattr(entry, "onset_s", 0.0)))

    if order == "trial":
        return sorted(entries, key=fallback)
    if order == "duration":
        return sorted(entries, key=lambda e: (float(getattr(e, "duration", 0.0)), fallback(e)))
    signed = 1.0 if order == "confidence_asc" else -1.0
    return sorted(entries, key=lambda e: (signed * float(e.confidence), fallback(e)))


class TileVerdicts:
    """Which labels were clicked in a curate/uncurate mode, and what Done does.

    Keyed by :func:`entry_key`, so clicking any tile of a label marks the
    label. Shared by the frame grid and the video grid.
    """

    def __init__(self) -> None:
        self.clicked: set[tuple] = set()

    def toggle(self, entry) -> bool:
        """Flip *entry*'s label; returns whether it is clicked now."""
        key = entry_key(entry)
        if key in self.clicked:
            self.clicked.discard(key)
            return False
        self.clicked.add(key)
        return True

    def is_clicked(self, entry) -> bool:
        return entry_key(entry) in self.clicked

    def clear(self) -> None:
        self.clicked.clear()

    def insts_for_done(self, entries) -> list[dict]:
        """The labels Done curates: every automated one *not* tagged, each
        label once. A tagged label stays automated — it is the one to review."""
        out: dict[tuple, dict] = {}
        for entry in entries:
            if entry.labeling_method != LABELING_AUTOMATED:
                continue
            if entry_key(entry) not in self.clicked:
                out.setdefault(entry_key(entry), entry_inst(entry))
        return list(out.values())


@dataclass
class ConfidenceGroup:
    """One histogram's worth of confidences: a label class, one animal's."""

    label_id: int
    name: str
    #: ``None`` when the labels name a single individual — then the label
    #: class alone is the split.
    individual: str | None
    color_hex: str
    values: list[float] = field(default_factory=list)


def confidence_groups(entries: list[FrameEntry]) -> list[ConfidenceGroup]:
    """Group the entries' confidences, one group per histogram.

    A label instance seen by two cameras is two tiles but one score, and a
    state event's start and end tiles share their row's — both collapse here,
    so a histogram counts events, not tiles. Groups split per individual as
    well as per label class whenever more than one individual is labelled: a
    model is rarely equally sure about every animal.
    """
    rows: dict[tuple, FrameEntry] = {}
    for entry in entries:
        rows.setdefault(entry_key(entry), entry)

    per_individual = len({subject_str(e.individual) for e in rows.values()}) > 1
    groups: dict[tuple[int, str], ConfidenceGroup] = {}
    for entry in rows.values():
        individual = subject_str(entry.individual) if per_individual else None
        group = groups.get((entry.label_id, individual or ""))
        if group is None:
            group = ConfidenceGroup(entry.label_id, entry.name, individual, entry.color_hex)
            groups[(entry.label_id, individual or "")] = group
        group.values.append(entry.confidence)
    return [groups[key] for key in sorted(groups)]


def split_histogram(values, threshold: float, bins: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin *values* into ``(edges, below threshold, the rest)``.

    Both halves share one set of edges over [0, 1], so the bin the threshold
    falls inside splits into a red part and a normal part instead of being
    coloured all one way. A threshold of 0 flags nothing, as in the grid.
    """
    data = np.clip(np.asarray(list(values), dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    low = data < threshold if threshold > 0.0 else np.zeros(data.shape, dtype=bool)
    below, _ = np.histogram(data[low], bins=edges)
    above, _ = np.histogram(data[~low], bins=edges)
    return edges, below, above


def _row_confidence(row) -> float:
    """A label row's confidence; fully confident when the column is absent."""
    value = row.get("confidence", HUMAN_CONFIDENCE)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return HUMAN_CONFIDENCE
    return float(value)


def _row_method(row) -> str:
    """A label row's ``labeling_method``; a row without one is read off its
    confidence, exactly as :func:`ensure_labeling_method` would."""
    value = row.get("labeling_method")
    if isinstance(value, str) and value in (LABELING_MANUAL, LABELING_AUTOMATED, LABELING_CURATED):
        return value
    return LABELING_AUTOMATED if _row_confidence(row) < HUMAN_CONFIDENCE else LABELING_MANUAL


#: The grids' "Labeling method" filter: each choice's label and the methods it
#: keeps (``None`` keeps every label). "Manual or curated" stays as its own
#: choice — both mean a human vouched for the label — alongside the two split
#: out separately for a reviewer who wants to see one without the other.
GRID_METHOD_FILTERS: dict[str, tuple[str, frozenset[str] | None]] = {
    "all": ("All labels", None),
    "manual": ("Manual only", frozenset({LABELING_MANUAL})),
    "curated": ("Curated only", frozenset({LABELING_CURATED})),
    "human": ("Manual or curated", frozenset({LABELING_MANUAL, LABELING_CURATED})),
    "automated": ("Automated only", frozenset({LABELING_AUTOMATED})),
}


def methods_for_filter(choice: str) -> frozenset[str] | None:
    """The labeling methods *choice* keeps; ``None`` (also for an unknown
    choice) keeps every label."""
    return GRID_METHOD_FILTERS.get(str(choice), GRID_METHOD_FILTERS["all"])[1]


def keep_method(row, methods: frozenset[str] | None) -> bool:
    """Whether a label row survives a method filter."""
    return methods is None or _row_method(row) in methods


def _hex_to_rgb(color_hex: str) -> tuple[int, int, int]:
    return tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))


def _draw_disc(image: np.ndarray, cx: int, cy: int, radius: int, rgb: tuple[int, int, int]) -> None:
    h, w = image.shape[:2]
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    image[y0:y1, x0:x1][disc] = rgb


def draw_pose_points(
    image: np.ndarray,
    pose: PoseRenderData,
    frame_idx: int,
    scale: float,
    color_by: str,
    *,
    hidden_keypoints: frozenset[str] | set[str] = frozenset(),
    show_text: bool = False,
) -> None:
    """Draw the pose points of one video frame onto a decoded thumbnail.

    ``scale`` is the source→decoded pixel ratio (``VideoFrameSource.scale``);
    colour encodes ``color_by`` (keypoint/individual), matching the video
    overlay's one-axis colour rule, and with ``show_text`` the *other* axis
    is written next to each point, as the overlay does. ``hidden_keypoints``
    is the sidebar's keypoint filter: those are not drawn.
    """
    frame_col = 1 if pose.data.shape[1] > 3 else 0
    frames = np.full(len(pose.data), -1, dtype=int)
    valid = pose.data_not_nan
    frames[valid] = np.round(pose.data[valid, frame_col]).astype(int)
    mask = frames == frame_idx
    if hidden_keypoints and "keypoint" in pose.properties.columns:
        mask &= ~pose.properties["keypoint"].astype(str).isin(hidden_keypoints).to_numpy()
    if not mask.any():
        return

    ys = pose.data[mask, frame_col + 1] / scale
    xs = pose.data[mask, frame_col + 2] / scale
    rows = pose.properties.iloc[np.flatnonzero(mask)]
    if color_by in pose.properties.columns:
        cats = rows[color_by].astype(str).to_numpy()
        order = pose.properties[color_by].astype(str).unique()
    else:
        cats = np.array([""] * len(xs))
        order = [""]
    palette = {val: _hex_to_rgb(MULTIDIM_COLORS[i % len(MULTIDIM_COLORS)]) for i, val in enumerate(order)}
    text_prop = "individual" if color_by == "keypoint" else "keypoint"
    texts = rows[text_prop].astype(str).to_numpy() if show_text and text_prop in rows.columns else None

    h, w = image.shape[:2]
    radius = max(2, round(min(h, w) / 130))
    font_scale = max(0.3, min(h, w) / 600)
    for i, (x, y, cat) in enumerate(zip(xs, ys, cats)):
        colour = palette.get(cat, (255, 255, 255))
        cx, cy = int(round(x)), int(round(y))
        _draw_disc(image, cx, cy, radius, colour)
        if texts is not None:
            cv2.putText(
                image,
                texts[i],
                (cx + radius + 2, cy - radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                colour,
                1,
                cv2.LINE_AA,
            )


def crop_thumbnail(image: np.ndarray, rect: tuple[int, int, int, int], scale: float) -> np.ndarray:
    """Cut a GUI camera crop — ``(x0, y0, x1, y1)`` source pixels, y down —
    out of a thumbnail decoded at ``1/scale``. A rect degenerate after
    clamping returns the image unchanged."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (int(round(v / scale)) for v in rect)
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return image
    return np.ascontiguousarray(image[y0:y1, x0:x1])


# ----------------------------------------------------------------------
# Frame extraction
# ----------------------------------------------------------------------


def _pose_device(alignment, camera: str | None) -> str | None:
    """Which pose stream feeds *camera* — same pairing as the video overlay."""
    pose_keys = list(getattr(alignment, "pose_keys", None) or [])
    if camera in pose_keys:
        return camera
    cameras = list(getattr(alignment, "cameras", None) or [])
    if camera in cameras:
        idx = cameras.index(camera)
        if idx < len(pose_keys):
            return pose_keys[idx]
    return None


def _load_group_pose(
    alignment,
    trial,
    camera: str | None,
    pose_folder: str | None,
    source_software: str | None,
    fps: float,
) -> PoseRenderData | None:
    device = _pose_device(alignment, camera) or camera
    path = alignment.resolve_media_path(trial, "pose", device=device, fallback_folder=pose_folder)
    if not path or not Path(path).exists():
        return None
    if not source_software and Path(path).suffix.lower() != POSES_DATASET_SUFFIX:
        logger.info("Pose file %s skipped — source software unknown.", Path(path).name)
        return None
    try:
        return load_pose_from_file(path, source_software, fps)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Failed to load pose %s: %s", Path(path).name, exc)
        return None


def resolve_video_jobs(
    groups: dict,
    *,
    alignment,
    video_folder: str | None,
    current_trial=None,
    current_video_path: str | None = None,
) -> list[tuple]:
    """Resolve each (trial, camera) group's video path, fps, offset and frame count.

    Sequential on purpose — the alignment NWB (h5py) is not safe to read from
    worker threads. A group whose video cannot be found or probed has its
    entries' ``error`` set and is left out. Shared with the video grid.
    """
    jobs: list[tuple] = []
    for (_, camera), group in groups.items():
        trial = group[0].trial
        path = alignment.resolve_media_path(trial, "video", device=camera, fallback_folder=video_folder)
        if not path and current_video_path and str(trial) == str(current_trial):
            path = current_video_path
        if not path or not Path(path).exists():
            for entry in group:
                entry.error = "video not found"
            continue
        try:
            probe = probe_video(path)
            fps = alignment.get_stream_rate("video", camera) or probe.fps
            offset = alignment.stream_offset_for_trial(trial, "video", camera)
        except (OSError, ValueError) as exc:
            logger.warning("Video probe failed for %s: %s", path, exc)
            for entry in group:
                entry.error = str(exc)
            continue
        jobs.append((group, path, fps, offset, probe.nframes))
    return jobs


def decode_entry_images(
    entries: list[FrameEntry],
    *,
    alignment,
    video_folder: str | None,
    pose_folder: str | None,
    source_software: str | None,
    pose_color_by: str,
    camera_crops: dict[str | None, tuple[int, int, int, int]] | None = None,
    current_trial=None,
    current_video_path: str | None = None,
    progress_cb=None,
    show_keypoints: bool = True,
    hidden_keypoints: frozenset[str] | set[str] = frozenset(),
    show_text: bool = False,
) -> None:
    """Fill each entry's ``image`` (RGB thumbnail with pose overlay) in place.

    The overlay follows the sidebar's Pose section: ``show_keypoints`` off
    draws none, ``hidden_keypoints`` are left out, ``show_text`` names each
    point — the thumbnails show what the video shows.

    Entries are grouped per (trial, camera) so each video is opened once and
    visited in frame order. Media metadata (paths, rates, offsets, pose
    files) resolves sequentially — the alignment NWB is not thread-safe —
    then the groups decode concurrently in a small thread pool, one video
    per worker. A file that cannot be resolved or decoded marks its entries
    with ``error`` instead of raising — missing media is a runtime
    condition, and the grid shows the message on the tile.
    ``camera_crops`` maps a selected camera to its GUI display crop (source
    pixels); a cropped camera's thumbnails show only that region.
    ``progress_cb(done)`` returning False cancels the remaining work.
    """
    groups: dict[tuple[str, str | None], list[FrameEntry]] = {}
    for entry in entries:
        groups.setdefault((str(entry.trial), entry.camera), []).append(entry)

    # Phase 1 — sequential: resolve paths, rates, offsets and pose files.
    jobs: list[tuple] = []
    for group, path, fps, offset, nframes in resolve_video_jobs(
        groups,
        alignment=alignment,
        video_folder=video_folder,
        current_trial=current_trial,
        current_video_path=current_video_path,
    ):
        camera = group[0].camera
        pose = (
            _load_group_pose(alignment, group[0].trial, camera, pose_folder, source_software, fps)
            if show_keypoints
            else None
        )
        jobs.append((group, path, fps, offset, pose, (camera_crops or {}).get(camera), nframes))

    def report() -> bool:
        if progress_cb is None:
            return True
        return progress_cb(sum(1 for e in entries if e.image is not None or e.error is not None))

    if not report() or not jobs:
        return

    # Phase 2 — parallel: pure PyAV decode + numpy pose/crop work, one video
    # per worker (each opens its own container, so no shared decode state).
    cancel = threading.Event()

    def run_job(job) -> None:
        group, path, fps, offset, pose, crop, nframes = job
        try:
            # Every frame the group needs, in frame order, so the video is
            # visited once forwards: a state tile contributes two.
            wanted = [(t, entry, field_name) for entry in group for field_name, t in entry.frames()]
            with VideoFrameSource(path, fps, nframes, max_side=THUMB_MAX_SIDE) as source:
                for t_rel, entry, field_name in sorted(wanted, key=lambda w: w[0]):
                    if cancel.is_set():
                        return
                    frame = int(round((t_rel - offset) * fps))
                    frame = min(max(frame, 0), max(nframes - 1, 0))
                    image = np.ascontiguousarray(source[frame])
                    if pose is not None:
                        draw_pose_points(
                            image,
                            pose,
                            frame,
                            source.scale,
                            pose_color_by,
                            hidden_keypoints=hidden_keypoints,
                            show_text=show_text,
                        )
                    if crop is not None:
                        image = crop_thumbnail(image, crop, source.scale)
                        entry.cropped = True
                    if field_name == FIELD_END:
                        entry.end_frame_idx = frame
                        entry.end_image = image
                    else:
                        entry.frame_idx = frame
                        entry.image = image
        except (OSError, ValueError) as exc:
            logger.warning("Frame extraction failed for %s: %s", path, exc)
            for entry in group:
                if entry.image is None:
                    entry.error = str(exc)

    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        while not all(f.done() for f in futures):
            if not report():
                cancel.set()
            # Keep the progress dialog painting while the workers decode.
            settle(50)
    report()
    for future in futures:
        future.result()


# ----------------------------------------------------------------------
# GUI panel capture
# ----------------------------------------------------------------------


def settle(ms: int) -> None:
    """Let the GUI redraw for *ms* — a local event loop, not a blocking sleep."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def _parent_dock(widget: QWidget) -> QDockWidget | None:
    parent = widget.parent()
    while parent is not None and not isinstance(parent, QDockWidget):
        parent = parent.parent()
    return parent


def _dock_label(dock: QDockWidget | None, fallback: str) -> str:
    """A dock's display title (the slim title bar's label), else *fallback*."""
    if dock is None:
        return fallback
    bar = dock.titleBarWidget()
    label = getattr(bar, "_label", None)
    if label is not None and label.text():
        return label.text()
    return dock.windowTitle() or fallback


def open_gui_panels(meta) -> list[tuple[str, QWidget]]:
    """(title, widget) of every plot panel currently visible in the GUI."""
    panels: list[tuple[str, QWidget]] = []
    data_widget = getattr(meta, "data_widget", None)
    container = getattr(data_widget, "plot_container", None)
    if container is not None:
        for plot in list(getattr(container, "_dyn_panels", []) or []):
            dock = (getattr(container, "_dyn_docks", {}) or {}).get(plot)
            if dock is None or dock.isHidden():
                continue
            panels.append((_dock_label(dock, str(getattr(plot, "panel_type", "panel"))), plot))
        for name, dock in (getattr(container, "_panel_docks", {}) or {}).items():
            if not dock.isHidden():
                panels.append((_dock_label(dock, name), dock.widget()))
    for kind, plots in (
        ("Space", getattr(data_widget, "space_plots", None) or []),
        ("Radial", getattr(data_widget, "radial_plots", None) or []),
    ):
        for i, plot in enumerate(plots, start=1):
            if plot.isVisible():
                panels.append((_dock_label(_parent_dock(plot), f"{kind} plot {i}"), plot))
    return panels


def _panel_viewbox(widget: QWidget):
    """The pyqtgraph ViewBox behind a panel, or None (space/GL widgets)."""
    get_vb = getattr(widget, "getViewBox", None)
    if callable(get_vb):
        return get_vb()
    return getattr(widget, "vb", None)


def capture_panel_images(
    entries: list[FrameEntry],
    *,
    nav,
    panels: list[tuple[str, QWidget]],
    window_s: float,
    autoscale: bool = True,
    progress_cb=None,
) -> int:
    """Screenshot the ticked GUI panels around each entry's time, in place.

    Navigates the real GUI once per unique (trial, time) — the plots redraw
    with the time marker on the label time and the viewport spanning
    ``window_s`` around it — then grabs each panel widget (``QWidget.grab``,
    same capture the screen recorder uses, so pygfx canvases come out too).
    Cameras of the same label share the captures; a state tile's window
    spans the whole label plus ``window_s / 2`` either side.

    ``autoscale`` fits each panel's y-range to the data visible in that
    capture's time window; with it off, the ranges the user has set now stay
    frozen for the duration. The panels' own view state is restored
    afterwards. Returns the number of positions visited;
    ``progress_cb(done)`` returning False stops early.
    """
    keyed: dict[tuple[str, float], list[FrameEntry]] = {}
    for entry in entries:
        keyed.setdefault((str(entry.trial), round(entry.t_rel, 6)), []).append(entry)

    view_boxes = [vb for vb in (_panel_viewbox(widget) for _, widget in panels) if vb is not None]
    saved_states = [vb.getState(copy=True) for vb in view_boxes]
    for vb in view_boxes:
        if autoscale:
            # Fit y to the data visible in the x window, not the whole trace.
            vb.setAutoVisible(y=True)
            vb.enableAutoRange(y=True)
        else:
            vb.enableAutoRange(y=False)

    half = window_s / 2.0
    done = 0
    last_trial: str | None = None
    try:
        for (trial_key, _), group in sorted(keyed.items()):
            lead = group[0]
            view_end = lead.offset_s if lead.is_state else lead.t_rel
            nav.jump_to_label_instance(
                {
                    "trial": lead.trial,
                    "onset_s": lead.onset_s,
                    "offset_s": lead.offset_s,
                    "individual": lead.individual,
                    "individual_rec": lead.individual_rec,
                },
                seek_rel=lead.t_rel,
                play=False,
                view_rel=TimeRange(lead.t_rel - half, view_end + half),
            )
            settle(PANEL_TRIAL_SETTLE_MS if trial_key != last_trial else PANEL_SETTLE_MS)
            last_trial = trial_key

            shots: list[tuple[str, QImage]] = []
            for title, widget in panels:
                try:
                    # repaint() forces the widget to redraw synchronously right
                    # now, unlike update() (which only schedules one) — without
                    # it grab() can catch the backing store between the marker
                    # move and its next paint pass, capturing it without the
                    # red time-marker line.
                    widget.repaint()
                    shots.append((title, widget.grab().toImage().convertToFormat(QImage.Format_RGB888)))
                except RuntimeError:
                    logger.warning("Panel %r was closed during capture — skipped.", title)
            for entry in group:
                entry.panels = shots
            done += 1
            if progress_cb is not None and not progress_cb(done):
                return done
        return done
    finally:
        for vb, state in zip(view_boxes, saved_states):
            vb.setState(state)


# ----------------------------------------------------------------------
# Tile text
# ----------------------------------------------------------------------


def _entry_title(entry: FrameEntry) -> str:
    return f"{entry.name} ({entry.label_id})"


def _entry_info(entry: FrameEntry) -> str:
    """Where the label sits, the video grid's way: ``at 0.500 s`` for a point
    event, ``1.000–1.800 s  ·  0.800 s`` for a state event."""
    parts = [f"trial {entry.trial}"]
    if entry.camera:
        parts.append(str(entry.camera))
    individual = entry.individual
    if individual is not None and not (isinstance(individual, float) and math.isnan(individual)):
        parts.append(str(individual))
    if entry.is_state:
        parts.append(f"{entry.onset_s:.3f}–{entry.offset_s:.3f} s")
        parts.append(f"{entry.offset_s - entry.onset_s:.3f} s")
    else:
        parts.append(f"at {entry.t_rel:.3f} s")
    parts.append(entry.labeling_method)
    if entry.cropped:
        parts.append("cropped")
    return "  ·  ".join(parts)


def _frame_caption(field_name: str, t_rel: float) -> str:
    """The small caption under one frame of a state tile."""
    return f"{'onset' if field_name == FIELD_START else 'offset'}  {t_rel:.3f} s"


# ----------------------------------------------------------------------
# PDF export
# ----------------------------------------------------------------------


def _to_qimage(image: np.ndarray) -> QImage:
    h, w, _ = image.shape
    return QImage(image.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def write_frames_pdf(
    path: str | Path,
    entries: list[FrameEntry],
    columns: int,
    confidence_threshold: float = 0.0,
) -> None:
    """Write the grid as a paginated PDF, *columns* single tiles per row.

    Tiles are placed as on screen (:func:`pack_tiles`): a state tile spans
    two columns, its onset and offset frames side by side. Tiles below
    *confidence_threshold* get the same red outline they carry in the grid,
    so a printed review sheet flags what to check.
    """
    writer = QPdfWriter(str(path))
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setResolution(150)
    painter = QPainter(writer)
    try:
        page_w, page_h = writer.width(), writer.height()
        margin, gap = 40, 18
        cell_w = (page_w - 2 * margin - (columns - 1) * gap) // columns
        body_font = QFont("Helvetica", 7)
        conf_font = QFont("Helvetica", 10, QFont.Bold)
        painter.setFont(conf_font)
        conf_h = painter.fontMetrics().height()
        painter.setFont(body_font)
        line_h = painter.fontMetrics().height()
        text_h = conf_h + line_h + 4

        def tile_width(span: int) -> int:
            return span * cell_w + (span - 1) * gap

        def image_height(image: np.ndarray | None) -> int:
            if image is not None:
                h, w = image.shape[:2]
                return int(cell_w * h / w)
            return int(cell_w * 9 / 16)

        def frames_height(entry: FrameEntry) -> int:
            heights = [image_height(entry.image)]
            if entry.is_state:
                heights.append(image_height(entry.end_image))
                heights[0] += line_h  # the onset / offset captions
                heights[1] += line_h
            return max(heights)

        def cell_height(entry: FrameEntry, span: int) -> int:
            total = text_h + frames_height(entry)
            width = tile_width(span)
            for _, qimg in entry.panels:
                total += line_h + int(width * qimg.height() / max(1, qimg.width()))
            return total

        def draw_frame(x: int, cy: int, image: np.ndarray | None, error: str | None) -> None:
            h_img = image_height(image)
            if image is not None:
                painter.drawImage(QRect(x, cy, cell_w, h_img), _to_qimage(image))
            else:
                painter.drawRect(x, cy, cell_w, h_img)
                painter.drawText(x + 4, cy + line_h, f"(no frame: {error or 'unavailable'})")

        placed = pack_tiles(entries, columns)
        rows: dict[int, list[tuple]] = {}
        for entry, row_idx, col, span in placed:
            rows.setdefault(row_idx, []).append((entry, col, span))

        y = margin
        for row_idx in sorted(rows):
            row = rows[row_idx]
            row_h = max(cell_height(entry, span) for entry, _, span in row)
            if y + row_h > page_h - margin and y > margin:
                writer.newPage()
                y = margin
            for entry, col, span in row:
                x = margin + col * (cell_w + gap)
                width = tile_width(span)
                if is_low_confidence(entry, confidence_threshold):
                    painter.save()
                    painter.setPen(QPen(QColor(LOW_CONFIDENCE_COLOR), 2))
                    painter.drawRect(x - 5, y - 5, width + 10, cell_height(entry, span) + 10)
                    painter.restore()
                painter.save()
                painter.setFont(conf_font)
                if is_low_confidence(entry, confidence_threshold):
                    painter.setPen(QColor(LOW_CONFIDENCE_COLOR))
                conf_text = confidence_text(entry)
                conf_w = painter.fontMetrics().horizontalAdvance(conf_text)
                painter.drawText(QRect(x, y, width, conf_h), Qt.AlignRight | Qt.AlignVCenter, conf_text)
                painter.restore()
                painter.drawText(
                    QRect(x, y, max(1, width - conf_w - 6), conf_h),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    _entry_title(entry),
                )
                painter.drawText(x, y + conf_h + line_h, _entry_info(entry))
                cy = y + text_h
                if entry.is_state and span == 2:
                    for i, (field_name, t_rel) in enumerate(entry.frames()):
                        fx = x + i * (cell_w + gap)
                        painter.drawText(fx, cy + line_h - 2, _frame_caption(field_name, t_rel))
                        draw_frame(fx, cy + line_h, entry.end_image if i else entry.image, entry.error)
                else:
                    draw_frame(x, cy, entry.image, entry.error)
                cy += frames_height(entry)
                for title, qimg in entry.panels:
                    painter.drawText(x, cy + line_h - 2, title)
                    cy += line_h
                    h_panel = int(width * qimg.height() / max(1, qimg.width()))
                    painter.drawImage(QRect(x, cy, width, h_panel), qimg)
                    cy += h_panel
            y += row_h + gap
    finally:
        painter.end()


# ----------------------------------------------------------------------
# Grid dialog
# ----------------------------------------------------------------------


class _ClickableLabel(QLabel):
    clicked = Signal()
    double_clicked = Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)


def histogram_bar_color(color_hex: str) -> str:
    """The colour the unflagged bars take.

    The label's own, unless it sits so close to the flag red that the flagged
    part of the same histogram would not read as a separate colour.
    """
    if math.dist(_hex_to_rgb(color_hex), _hex_to_rgb(LOW_CONFIDENCE_COLOR)) < _HIST_MIN_COLOR_DISTANCE:
        return _HIST_NEUTRAL
    return color_hex


def _group_title(group: ConfidenceGroup, flagged: int) -> str:
    title = f"{group.name} ({group.label_id})"
    if group.individual:
        title += f" — {group.individual}"
    counts = f"n={len(group.values)}"
    if flagged:
        counts += f" · {flagged} flagged"
    return f"{title}   {counts}"


class ConfidenceHistogramsDialog(QDialog):
    """How the confidences of each label class are distributed.

    One histogram per label class — per (class, individual) when more than one
    individual is labelled — with the part below the threshold drawn in the
    same red the grid outlines flagged tiles in, so the threshold can be
    chosen by looking at where the model's scores actually pile up. The
    threshold box is bound both ways to the grid's.
    """

    threshold_changed = Signal(float)

    def __init__(
        self,
        groups: list[ConfidenceGroup],
        threshold: float = 0.0,
        parent=None,
        rule_controller: "ConfidenceRuleController | None" = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Confidence histograms")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._groups = list(groups)
        self._plots: list[tuple[ConfidenceGroup, pg.PlotWidget]] = []
        #: The confidence knob, when the host has curves to re-read: which
        #: reading of a prediction's curve is its confidence. Every change is
        #: previewed on the grid and these bars; Apply confirms it.
        self._controller = rule_controller if (rule_controller is not None and rule_controller.curves()) else None
        self._applied = False

        layout = QVBoxLayout(self)
        if self._controller is not None:
            layout.addWidget(self._build_rule_panel(self._controller))
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Flag confidence below:"))
        self.threshold_edit = ConfidenceEdit(threshold)
        self.threshold_edit.setToolTip("Shared with the grid — moving it here recolours the tiles too.")
        self.threshold_edit.valueChanged.connect(self._on_threshold)
        bar.addWidget(self.threshold_edit)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Bins:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 100)
        self.bins_spin.setValue(_HIST_BINS)
        self.bins_spin.valueChanged.connect(self._redraw)
        bar.addWidget(self.bins_spin)
        bar.addStretch()
        layout.addLayout(bar)

        hint = QLabel(
            "Each event counts once, whatever it was seen by · red is what falls below the threshold · "
            "human labels score 1.0."
        )
        hint.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(8)
        for i, group in enumerate(self._groups):
            plot = pg.PlotWidget()
            plot.setBackground(_HIST_BG)
            plot.setLabel("bottom", "confidence")
            plot.setLabel("left", "events")
            plot.setXRange(0.0, 1.0, padding=0.02)
            plot.setMinimumSize(*_HIST_MIN_SIZE)
            grid.addWidget(plot, i // _HIST_COLUMNS, i % _HIST_COLUMNS)
            self._plots.append((group, plot))
        scroll.setWidget(container)
        layout.addWidget(scroll)

        self._redraw()
        columns = min(_HIST_COLUMNS, max(1, len(self._groups)))
        rows = math.ceil(len(self._groups) / _HIST_COLUMNS) if self._groups else 1
        width = columns * (_HIST_MIN_SIZE[0] + 20) + 40
        height = rows * (_HIST_MIN_SIZE[1] + 20) + 120
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            width = min(width, int(avail.width() * 0.95))
            height = min(height, int(avail.height() * 0.9))
        self.resize(width, height)

    def _build_rule_panel(self, controller: "ConfidenceRuleController") -> QGroupBox:
        rule, alpha, window_ms = controller.settings()
        box = QGroupBox("Confidence rule — what the number is read off the prediction's curve")
        lay = QVBoxLayout(box)
        intro = QLabel(
            "ratio asks whether there was one candidate or two; focus whether the bump is sharp or smeared. "
            "Change the rule and watch the bars move; Apply writes it into the automated labels that have a curve."
        )
        intro.setStyleSheet("color: grey; font-size: 10px;")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QHBoxLayout()
        row.addWidget(QLabel("Rule:"))
        self.rule_combo = QComboBox()
        for key, text in RULES.items():
            self.rule_combo.addItem(text, key)
        self.rule_combo.setCurrentIndex(max(0, self.rule_combo.findData(rule)))
        self.rule_combo.currentIndexChanged.connect(self._on_rule)
        row.addWidget(self.rule_combo)
        row.addSpacing(12)
        row.addWidget(QLabel("α:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(round(alpha * 100)))
        self.alpha_slider.setFixedWidth(140)
        self.alpha_slider.setToolTip("custom rule only — 1 = ratio alone (candidates); 0 = focus × ratio")
        self.alpha_slider.valueChanged.connect(self._on_rule)
        row.addWidget(self.alpha_slider)
        self.alpha_label = QLabel("")
        self.alpha_label.setMinimumWidth(32)
        row.addWidget(self.alpha_label)
        row.addSpacing(12)
        row.addWidget(QLabel("Same event within:"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(1.0, 5000.0)
        self.window_spin.setDecimals(0)
        self.window_spin.setSingleStep(10.0)
        self.window_spin.setSuffix(" ms")
        self.window_spin.setValue(window_ms)
        self.window_spin.setToolTip(
            "± this around the peak counts as the same event: mass inside it is focus,\n"
            "a second peak outside it is a rival. Twice the precision you believe the labels to."
        )
        self.window_spin.valueChanged.connect(self._on_rule)
        row.addWidget(self.window_spin)
        row.addStretch()
        self.apply_btn = QPushButton("Apply to labels")
        self.apply_btn.setAutoDefault(False)
        self.apply_btn.setToolTip(
            "Write the previewed confidences into the automated labels that have a curve\n"
            "(one undo step per trial). Manual and curated labels are never touched.\n"
            "Closing without applying puts the original values back."
        )
        self.apply_btn.clicked.connect(self._apply_rule)
        row.addWidget(self.apply_btn)
        self.copy_btn = QPushButton("Copy for project.yaml")
        self.copy_btn.setAutoDefault(False)
        self.copy_btn.setToolTip(
            "Put the `infer:` lines for this rule on the clipboard — paste them into a spot\n"
            "project.yaml and the next inference() writes confidence the same way."
        )
        self.copy_btn.clicked.connect(self._copy_rule)
        row.addWidget(self.copy_btn)
        lay.addLayout(row)
        scored, total = controller.coverage()
        self.coverage = QLabel(
            f"{scored} of {total} automated labels here have a curve and follow the rule; "
            "the rest keep their confidence."
        )
        self.coverage.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(self.coverage)
        self._sync_alpha()
        return box

    def rule(self) -> tuple[str, float, float]:
        """The rule the panel shows: ``(rule, alpha, window_ms)``."""
        return str(self.rule_combo.currentData()), self.alpha_slider.value() / 100.0, float(self.window_spin.value())

    def _sync_alpha(self) -> None:
        rule, alpha, _ = self.rule()
        self.alpha_slider.setEnabled(rule == "custom")
        self.alpha_label.setText(f"{alpha:.2f}")

    def _on_rule(self, *_args) -> None:
        self._sync_alpha()
        self._applied = False
        assert self._controller is not None
        self._controller.preview(*self.rule())

    def _apply_rule(self) -> None:
        assert self._controller is not None
        self._applied = True
        self._controller.apply()

    def _copy_rule(self) -> None:
        """The rule as the `infer:` lines of a spot project.yaml, on the clipboard."""
        from ethograph.labels.rescore import yaml_snippet

        snippet = yaml_snippet(*self.rule())
        QApplication.clipboard().setText(snippet)
        notify("Copied for project.yaml:\n" + snippet.rstrip())

    @property
    def rule_applied(self) -> bool:
        return self._applied

    def closeEvent(self, event):
        if self._controller is not None and not self._applied:
            self._controller.revert()  # the preview was only ever a preview
        super().closeEvent(event)

    def set_threshold(self, value: float) -> None:
        """Follow the grid's threshold (a no-op when it already matches)."""
        self.threshold_edit.setValue(float(value))

    def set_groups(self, groups: list[ConfidenceGroup]) -> None:
        """Replace the plotted values — the confidence knob previewing another rule."""
        by_key = {(g.label_id, g.individual or ""): g for g in groups}
        for i, (group, _plot) in enumerate(self._plots):
            fresh = by_key.get((group.label_id, group.individual or ""))
            if fresh is not None:
                self._plots[i] = (fresh, _plot)
        self._groups = list(groups)
        self._redraw()

    def _on_threshold(self, value: float) -> None:
        self.threshold_changed.emit(float(value))
        self._redraw()

    def _redraw(self) -> None:
        threshold = self.threshold_edit.value()
        bins = self.bins_spin.value()
        for group, plot in self._plots:
            edges, below, above = split_histogram(group.values, threshold, bins)
            centers = (edges[:-1] + edges[1:]) / 2.0
            width = (edges[1] - edges[0]) * 0.9
            plot.clear()
            plot.addItem(
                pg.BarGraphItem(
                    x=centers,
                    height=above,
                    y0=below,
                    width=width,
                    brush=pg.mkBrush(histogram_bar_color(group.color_hex)),
                    pen=pg.mkPen(None),
                )
            )
            plot.addItem(
                pg.BarGraphItem(
                    x=centers,
                    height=below,
                    width=width,
                    brush=pg.mkBrush(LOW_CONFIDENCE_COLOR),
                    pen=pg.mkPen(None),
                )
            )
            if threshold > 0.0:
                plot.addItem(
                    pg.InfiniteLine(
                        pos=threshold,
                        angle=90,
                        pen=pg.mkPen(LOW_CONFIDENCE_COLOR, width=1, style=Qt.DashLine),
                    )
                )
            plot.setTitle(_group_title(group, int(below.sum())), color=group.color_hex, size="10pt")


def curation_panel_of(meta):
    """The Labels tab's curation panel, when the host GUI has one."""
    return getattr(getattr(meta, "labels_widget", None), "curation_panel", None)


class ConfidenceRuleController(QObject):
    """The confidence knob's arithmetic on a grid: preview on its entries, apply to the labels.

    Both grids own one; the histogram popup drives it. Entries are the
    grid's tiles (``FrameEntry`` / ``ClipEntry``): their ``confidence`` is
    overwritten for the preview and put back by :meth:`revert` when the popup
    closes without Apply. The curves are the session's, every run merged
    newest-per-class (:func:`~ethograph.labels.onset_curves.read_all_curves`).
    """

    changed = Signal()

    def __init__(self, meta, entries_fn, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.app_state = meta.app_state
        self._entries_fn = entries_fn
        self._curves = None
        self._originals: dict[int, float] = {}
        self._rule: tuple[str, float, float] | None = None

    def curves(self) -> dict:
        if self._curves is None:
            from ethograph.labels.onset_curves import read_all_curves

            session = getattr(self.app_state, "nc_file_path", None)
            self._curves = read_all_curves(session) if session else {}
        return self._curves

    def settings(self) -> tuple[str, float, float]:
        """The remembered rule: ``(rule, alpha, window_ms)``."""
        state = self.app_state
        return (
            str(state.get_with_default("grid_confidence_rule")),
            float(state.get_with_default("grid_confidence_alpha")),
            float(state.get_with_default("grid_confidence_window_ms")),
        )

    def coverage(self) -> tuple[int, int]:
        """``(automated labels with a curve, automated labels)`` in the grid — labels, not tiles."""
        curves = self.curves()

        def has_curve(e) -> bool:
            return str(e.trial) in curves and e.label_id in curves[str(e.trial)][1]

        automated = [e for e in self._entries_fn() if e.labeling_method == LABELING_AUTOMATED]
        scored = [e for e in automated if has_curve(e)]
        return len({entry_key(e) for e in scored}), len({entry_key(e) for e in automated})

    def begin(self) -> None:
        """Snapshot the entries' confidences: what :meth:`revert` puts back."""
        self._originals = {id(e): float(e.confidence) for e in self._entries_fn()}

    def preview(self, rule: str, alpha: float, window_ms: float) -> None:
        """Re-score the grid's entries in memory under the rule; nothing reaches the labels."""
        from ethograph.labels.rescore import confidence_of

        self._rule = (rule, alpha, window_ms)
        curves = self.curves()
        for entry in self._entries_fn():
            if entry.labeling_method != LABELING_AUTOMATED:
                continue
            value = confidence_of(curves, entry.trial, entry.label_id, rule, alpha, window_ms / 1000.0)
            entry.confidence = self._originals.get(id(entry), entry.confidence) if value is None else float(value)
        self.changed.emit()

    def revert(self) -> None:
        for entry in self._entries_fn():
            if id(entry) in self._originals:
                entry.confidence = self._originals[id(entry)]
        self.changed.emit()

    def apply(self) -> None:
        """Write the previewed rule into the labels: one undo step per trial touched."""
        from ethograph.labels.rescore import rescore_labels

        if self._rule is None:
            return
        rule, alpha, window_ms = self._rule
        df = getattr(self.app_state, "_all_labels_df", None)
        new_df, touched, n_changed = rescore_labels(df, self.curves(), rule, alpha, window_ms / 1000.0)
        for trial in touched:
            self.app_state.record_label_edit("confidence rule", trial=trial)
        self.app_state.replace_all_labels(new_df)
        self.app_state.grid_confidence_rule = rule
        self.app_state.grid_confidence_alpha = float(alpha)
        self.app_state.grid_confidence_window_ms = float(window_ms)
        self._originals = {id(e): float(e.confidence) for e in self._entries_fn()}
        labels_widget = getattr(self.meta, "labels_widget", None)
        if labels_widget is not None and hasattr(labels_widget, "_on_label_table_changed"):
            labels_widget._on_label_table_changed()
        notify(f"Confidence re-read under '{RULES[rule]}': {n_changed} labels changed in {len(touched)} trials.")
        self.changed.emit()


class GridVerdictBar(QWidget):
    """Tag low-confidence / Clear / Done — the verdict controls both grids share.

    A click tags a tile for review; **Done** curates every automated label
    on screen that is not tagged. The host passes its entries and a
    ``restyle()`` callback; this widget owns the :class:`TileVerdicts` and
    applies Done through the curation panel. ``entries_fn`` returns what is
    *on screen* — a grid filtered to one label class curates that class and
    nothing else.
    """

    def __init__(self, meta, entries_fn, restyle_fn, flagged_fn=None, parent=None):
        super().__init__(parent)
        self.meta = meta
        self._entries_fn = entries_fn
        self._restyle_fn = restyle_fn
        self._flagged_fn = flagged_fn
        self.verdicts = TileVerdicts()

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tag_flagged_btn = QPushButton("Tag low-confidence")
        self.tag_flagged_btn.setAutoDefault(False)
        self.tag_flagged_btn.setToolTip(
            "Tag every tile the confidence threshold outlines in red for review —\n"
            "a low score is a reason to doubt a label, never to approve it."
        )
        self.tag_flagged_btn.clicked.connect(self._tag_flagged)
        self.tag_flagged_btn.setEnabled(flagged_fn is not None)
        lay.addWidget(self.tag_flagged_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setAutoDefault(False)
        self.clear_btn.setToolTip("Forget every tag")
        self.clear_btn.clicked.connect(self.clear)
        lay.addWidget(self.clear_btn)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(self.count_label)
        self.done_btn = QPushButton("Done")
        self.done_btn.setAutoDefault(False)
        self.done_btn.setToolTip(
            "Curate every automated label on screen that is not tagged;\nthe tagged ones stay automated, for review."
        )
        self.done_btn.clicked.connect(self.apply_done)
        lay.addWidget(self.done_btn)

    def _sync_count(self) -> None:
        """Count the tags the host is showing, not every tag ever made —
        a verdict on a hidden label is not what Done is about to apply."""
        keys = {entry_key(entry) for entry in self._entries_fn()}
        n = len(self.verdicts.clicked & keys)
        self.count_label.setText(f"{n} tagged" if n else "")

    def refresh(self) -> None:
        """Re-read the host's entries — what is shown, and so what Done does."""
        self._sync_count()

    def click(self, entry) -> bool:
        """A tile click; returns whether the label is tagged now."""
        marked = self.verdicts.toggle(entry)
        self._restyle_fn()
        self._sync_count()
        return marked

    def clear(self) -> None:
        self.verdicts.clear()
        self._restyle_fn()
        self._sync_count()

    def _tag_flagged(self) -> None:
        if self._flagged_fn is None:
            return
        for entry in self._flagged_fn():
            self.verdicts.clicked.add(entry_key(entry))
        self._restyle_fn()
        self._sync_count()

    def apply_done(self) -> int:
        """Curate every untagged automated label on screen; the entries are
        restamped to match."""
        panel = curation_panel_of(self.meta)
        entries = list(self._entries_fn())
        insts = self.verdicts.insts_for_done(entries)
        if not insts:
            notify("Nothing to curate — every automated label on screen is tagged.", severity="warning")
            return 0
        if panel is None:
            notify("No curation panel to apply the verdicts to.", severity="warning")
            return 0
        n = panel.curate_labels(insts)
        if panel.session_active:
            # These verdicts may have curated labels the frame-by-frame
            # session is reviewing — rebuild its queue against the new state.
            panel.restart_review()
        done_keys = {
            (
                str(i["trial"]),
                int(i["labels"]),
                round(float(i["onset_s"]), 6),
                subject_str(i.get("individual")),
                subject_str(i.get("individual_rec")),
            )
            for i in insts
        }
        for entry in entries:
            if entry_key(entry) in done_keys:
                entry.labeling_method = LABELING_CURATED
        self.verdicts.clear()
        self._restyle_fn()
        self._sync_count()
        return n


class LabelGridView(QWidget):
    """Grid of label frames — the *Frames* tab of the dialog.

    A tile click is what the mode bar says it is: a jump, or a verdict.

    The **Label** combo narrows the grid to one of the classes it was built
    from. It is a filter on the whole tab, not just the view: the tile count,
    **Tag low-confidence**, **Done** and the PDF all run over
    what is on screen, so a scope of several classes can be curated one class
    at a time without reopening the dialog.
    """

    def __init__(self, meta, entries: list[FrameEntry], parent=None):
        super().__init__(parent)
        self.meta = meta
        self.app_state = meta.app_state
        self._entries = entries
        #: Which label class the grid is narrowed to; ``None`` is all of them.
        self._filter_label_id: int | None = None
        #: Filled once the toolbar exists — the mode bar restyles on creation.
        self._cells: list[QFrame] = []
        #: The confidence-histogram popup while it is open.
        self._hist_dialog: ConfidenceHistogramsDialog | None = None
        self._reflow_timer = QTimer(self)
        self._reflow_timer.setSingleShot(True)
        self._reflow_timer.timeout.connect(self._relayout)

        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        choices = label_filter_choices(entries)
        self._filter_row = QWidget(self)
        filter_lay = QHBoxLayout(self._filter_row)
        filter_lay.setContentsMargins(0, 0, 0, 0)
        filter_lay.addWidget(QLabel("Label:"))
        self.label_filter = QComboBox()
        for label_id, text in choices:
            self.label_filter.addItem(text, label_id)
        self.label_filter.setToolTip(
            "Show one label class at a time.\n"
            "The rest of the tab follows the filter: the flagged tiles, Done and the\n"
            "PDF all apply to the class on screen and to no other."
        )
        self.label_filter.currentIndexChanged.connect(self._on_filter_changed)
        filter_lay.addWidget(self.label_filter)
        bar.addWidget(self._filter_row)
        # Nothing to choose between when the grid holds a single class.
        self._filter_row.setVisible(len(choices) > 2)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Sort:"))
        self.sort_combo = QComboBox()
        for key, text in GRID_SORT_ORDERS.items():
            self.sort_combo.addItem(text, key)
        saved = str(self.app_state.get_with_default("label_grid_sort"))
        self.sort_combo.setCurrentIndex(max(0, self.sort_combo.findData(saved)))
        self.sort_combo.setToolTip(
            "The order the tiles are laid out in.\n"
            "\n"
            "By confidence puts the doubtful labels together on the first screens\n"
            "instead of scattering them through the trials — the fastest way to\n"
            "review what a model was least sure about."
        )
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        bar.addWidget(self.sort_combo)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Columns:"))
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, 12)
        # Remembered across sessions and datasets (SCOPE_GLOBAL).
        self.columns_spin.setValue(int(self.app_state.get_with_default("label_grid_columns")))
        self.columns_spin.valueChanged.connect(self._on_columns_changed)
        bar.addWidget(self.columns_spin)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Flag confidence below:"))
        # Shared (SCOPE_GLOBAL) with the video grid's own threshold box.
        self.threshold_edit = ConfidenceEdit(float(self.app_state.get_with_default("grid_confidence_threshold")))
        self.threshold_edit.setToolTip(
            "Outline every tile whose label scores below this in red.\n"
            "Human labels are 1.0; a predicted label carries the model's own score.\n"
            "Type it to as many decimals as the scores need (0.0002); 0 flags nothing."
        )
        self.threshold_edit.valueChanged.connect(self._apply_styles)
        self.threshold_edit.valueChanged.connect(self._on_threshold_changed)
        bar.addWidget(self.threshold_edit)
        self.histogram_btn = QPushButton("Histogram…")
        self.histogram_btn.setAutoDefault(False)
        self.histogram_btn.setToolTip(
            "How the confidences are distributed, one histogram per label class\n"
            "(per individual too, when more than one is labelled). The part below\n"
            "the threshold is red, and the threshold can be set from there — and so\n"
            "can the rule the confidence is read off the prediction's curve by."
        )
        self.histogram_btn.clicked.connect(self._show_histograms)
        bar.addWidget(self.histogram_btn)
        # The confidence rule lives in the histogram popup; this drives it.
        self.rule_controller = ConfidenceRuleController(meta, entries_fn=lambda: self._entries, parent=self)
        self.rule_controller.changed.connect(self._on_confidence_rescored)
        bar.addStretch()
        self.count_label = QLabel("")
        bar.addWidget(self.count_label)
        export_btn = QPushButton("Export PDF…")
        export_btn.setAutoDefault(False)
        export_btn.clicked.connect(self._export_pdf)
        bar.addWidget(export_btn)
        layout.addLayout(bar)

        self.verdict_bar = GridVerdictBar(
            meta,
            entries_fn=self.visible_entries,
            restyle_fn=self._apply_styles,
            flagged_fn=self._flagged_entries,
        )
        layout.addWidget(self.verdict_bar)

        self.hint = QLabel("")
        self.hint.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(self.hint)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget(self._scroll)
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(10)
        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll)

        self._cells = [self._make_cell(entry) for entry in entries]
        self._relayout()
        self._apply_styles()
        self._sync_count()
        self._sync_hint()

    @property
    def entries(self) -> list[FrameEntry]:
        return self._entries

    def visible_entries(self) -> list[FrameEntry]:
        """The entries the label filter shows, in the chosen order.

        What every operation acts on — the grid, Done, the flag count and the
        PDF — so the sort is applied here rather than at layout time.
        """
        return sort_entries(filter_entries(self._entries, self._filter_label_id), self._sort_order())

    def _sort_order(self) -> str:
        combo = getattr(self, "sort_combo", None)
        return combo.currentData() if combo is not None else "trial"

    def _on_sort_changed(self, *_args) -> None:
        """Re-lay the grid in the new order; the clicks survive it.

        Verdicts are keyed by label, not by position, so reordering never
        moves a verdict onto a different tile.
        """
        self.app_state.label_grid_sort = self._sort_order()
        self._relayout()
        self._apply_styles()

    def _on_filter_changed(self, *_args) -> None:
        """Re-show the grid under the new filter. The clicks are kept: a
        verdict on a hidden label is simply out of Done's reach until its
        class is shown again."""
        self._filter_label_id = self.label_filter.currentData()
        self._relayout()
        self._apply_styles()
        self._sync_count()
        self.verdict_bar.refresh()
        self._sync_hint()

    def _sync_count(self) -> None:
        shown = len(self.visible_entries())
        total = len(self._entries)
        self.count_label.setText(f"{shown} labels" if shown == total else f"{shown} of {total} labels")

    def _sync_hint(self, *_args) -> None:
        click = "Click the frames that are wrong to tag them for review; Done curates every other label."
        panel = curation_panel_of(self.meta)
        if panel is not None and panel.reviews_on_jump():
            jump = "Double-click a frame to review that label in the main GUI."
        else:
            jump = "Double-click a frame to jump the GUI to that boundary."
        self.hint.setText(f"{click} {jump}{self._filter_note()}")

    def _filter_note(self) -> str:
        """What the filter restricts Done to — silent when nothing is filtered."""
        if self._filter_label_id is None:
            return ""
        name = next((e.name for e in self._entries if e.label_id == self._filter_label_id), self._filter_label_id)
        return f" Filtered to '{name}': no other label class is touched."

    def _make_cell(self, entry: FrameEntry) -> QFrame:
        # Parented from birth: a parentless widget that is shown — even for
        # the instant before the grid layout adopts it — is a top-level
        # window, and one flashes per tile.
        cell = QFrame(self._container)
        cell.setObjectName("frameCell")
        lay = QVBoxLayout(cell)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel(_entry_title(entry))
        title.setStyleSheet(f"font-weight: bold; color: {entry.color_hex};")
        title.setWordWrap(True)
        header.addWidget(title, 1)
        conf = QLabel(confidence_text(entry))
        conf.setStyleSheet(confidence_style(entry, self.threshold_edit.value()))
        conf.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        conf.setToolTip("Confidence of this label — red below the flag threshold.")
        header.addWidget(conf, 0)
        lay.addLayout(header)
        cell._conf = conf  # type: ignore[attr-defined]
        info = QLabel(_entry_info(entry))
        info.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(info)
        cell._info = info  # type: ignore[attr-defined]

        #: (label, unscaled pixmap) pairs — relayout rescales each to the
        #: current column width.
        pix_labels: list[tuple[QLabel, QPixmap]] = []
        #: (label, unscaled pixmap) pairs that span the whole tile (panel
        #: captures): a state tile rescales these to two columns.
        wide_labels: list[tuple[QLabel, QPixmap]] = []

        def make_frame(image: np.ndarray | None, field_name: str) -> _ClickableLabel:
            image_label = _ClickableLabel()
            image_label.setCursor(Qt.PointingHandCursor)
            image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            if image is not None:
                pixmap = QPixmap.fromImage(_to_qimage(image))
                image_label.setPixmap(pixmap)
                pix_labels.append((image_label, pixmap))
            else:
                image_label.setText(f"(no frame:\n{entry.error or 'unavailable'})")
                image_label.setFrameShape(QFrame.StyledPanel)
                image_label.setMinimumSize(160, 90)
            image_label.clicked.connect(lambda e=entry: self._on_tile_clicked(e))
            image_label.double_clicked.connect(lambda e=entry, f=field_name: self._on_tile_double_clicked(e, f))
            return image_label

        if entry.is_state:
            # Onset and offset side by side, each captioned — without the
            # captions two frames read as two cameras.
            frames_row = QHBoxLayout()
            frames_row.setSpacing(self._grid.spacing())
            for i, (field_name, t_rel) in enumerate(entry.frames()):
                column = QVBoxLayout()
                column.setSpacing(1)
                caption = QLabel(_frame_caption(field_name, t_rel))
                caption.setStyleSheet("color: grey; font-size: 9px;")
                column.addWidget(caption)
                column.addWidget(make_frame(entry.end_image if i else entry.image, field_name))
                column.addStretch()
                frames_row.addLayout(column)
            lay.addLayout(frames_row)
        else:
            lay.addWidget(make_frame(entry.image, TILE_POINT))

        for panel_title, qimage in entry.panels:
            caption = QLabel(panel_title)
            caption.setStyleSheet("color: grey; font-size: 9px;")
            lay.addWidget(caption)
            panel_label = QLabel()
            panel_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            pixmap = QPixmap.fromImage(qimage)
            panel_label.setPixmap(pixmap)
            wide_labels.append((panel_label, pixmap))
            lay.addWidget(panel_label)

        cell._pix_labels = pix_labels  # type: ignore[attr-defined]
        cell._wide_labels = wide_labels  # type: ignore[attr-defined]
        lay.addStretch()
        return cell

    def _flagged_entries(self) -> list[FrameEntry]:
        threshold = self.threshold_edit.value()
        return [e for e in self.visible_entries() if is_low_confidence(e, threshold)]

    def _on_threshold_changed(self, value: float) -> None:
        self.app_state.grid_confidence_threshold = float(value)

    def _apply_styles(self, *_args) -> None:
        """Outline the tiles: verdict colour first, else the confidence red."""
        threshold = self.threshold_edit.value()
        verdicts = self.verdict_bar.verdicts
        for cell, entry in zip(self._cells, self._entries):
            if verdicts.is_clicked(entry):
                cell.setStyleSheet(_TAGGED_STYLE)
            else:
                cell.setStyleSheet(_LOW_CONFIDENCE_STYLE if is_low_confidence(entry, threshold) else "")
            cell._info.setText(_entry_info(entry))
            cell._conf.setText(confidence_text(entry))
            cell._conf.setStyleSheet(confidence_style(entry, threshold))
        if self._hist_dialog is not None:
            self._hist_dialog.set_threshold(threshold)

    def _on_confidence_rescored(self) -> None:
        """The confidence knob changed the entries' values: restyle, and refresh the histogram."""
        self._apply_styles()
        if self._hist_dialog is not None:
            self._hist_dialog.set_groups(confidence_groups(self._entries))

    def _show_histograms(self):
        """Open (or raise) the per-label confidence histograms."""
        if self._hist_dialog is not None:
            self._hist_dialog.show()
            self._hist_dialog.raise_()
            return
        groups = confidence_groups(self._entries)
        if not groups:
            notify("No labels to plot confidences for.", severity="warning")
            return
        self.rule_controller.begin()
        self._hist_dialog = ConfidenceHistogramsDialog(
            groups, self.threshold_edit.value(), parent=self, rule_controller=self.rule_controller
        )
        self._hist_dialog.threshold_changed.connect(self.threshold_edit.setValue)
        self._hist_dialog.destroyed.connect(self._on_histograms_closed)
        self._hist_dialog.show()

    def _on_histograms_closed(self, *_args):
        self._hist_dialog = None

    def resizeEvent(self, event):
        """Refit the thumbnails whenever the width changes — maximizing the
        dialog grows every tile to the new column width."""
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self._reflow_timer.start(_REFLOW_DEBOUNCE_MS)

    def _on_columns_changed(self, value: int) -> None:
        self.app_state.label_grid_columns = int(value)
        self._relayout()

    def _relayout(self):
        columns = self.columns_spin.value()
        spacing = self._grid.spacing()
        viewport_w = max(self._scroll.viewport().width(), 400)
        thumb_w = max(100, (viewport_w - spacing * (columns + 1)) // columns)
        while self._grid.count():
            self._grid.takeAt(0)
        # Laid out in visible_entries() order — the same list Done, the flag
        # count and the PDF read, so the sort and the filter are decided in
        # exactly one place and the screen cannot disagree with them.
        cells = {id(entry): cell for entry, cell in zip(self._entries, self._cells)}
        for cell in self._cells:
            cell.setVisible(False)
        for entry, row, col, span in pack_tiles(self.visible_entries(), columns):
            cell = cells.get(id(entry))
            if cell is None:
                continue
            wide_w = span * thumb_w + (span - 1) * spacing
            for label, pixmap in cell._pix_labels:
                if not pixmap.isNull():
                    label.setPixmap(pixmap.scaledToWidth(min(thumb_w, pixmap.width()), Qt.SmoothTransformation))
            for label, pixmap in cell._wide_labels:
                if not pixmap.isNull():
                    label.setPixmap(pixmap.scaledToWidth(min(wide_w, pixmap.width()), Qt.SmoothTransformation))
            self._grid.addWidget(cell, row, col, 1, span, alignment=Qt.AlignTop)
            cell.setVisible(True)

    def _on_tile_clicked(self, entry: FrameEntry):
        """A single click is the verdict the mode names."""
        self.verdict_bar.click(entry)

    def _on_tile_double_clicked(self, entry: FrameEntry, field_name: str = TILE_POINT):
        """A double click navigates, in every mode. Qt delivers a plain press
        first, which already toggled the tile — toggling again undoes it, so
        navigating leaves the verdicts exactly as they were. On a state tile
        the frame clicked says which boundary: the onset or the offset."""
        self.verdict_bar.click(entry)
        self._jump(entry, field_name)

    def _jump(self, entry: FrameEntry, field_name: str = TILE_POINT):
        """Go there — into the review when the curation panel is in a review
        mode, else a plain jump to that boundary's time."""
        panel = curation_panel_of(self.meta)
        if panel is not None and panel.reviews_on_jump():
            panel.start_review_at(entry_inst(entry), field_name)
            return
        nav = getattr(self.meta, "navigation_widget", None)
        if nav is None:
            return
        seek = entry.offset_s if field_name == FIELD_END else entry.t_rel
        nav.jump_to_label_instance(entry_inst(entry), seek_rel=seek, play=False)

    def _export_pdf(self):
        labels_path = self.app_state.labels_file_path()
        default = f"{labels_path.stem}_frames.pdf" if labels_path else "label_frames.pdf"
        path = browse_save_file(
            self,
            self.app_state,
            "Export label frames PDF",
            default,
            "PDF files (*.pdf)",
            preferred_dir=labels_path,
        )
        if not path:
            return
        write_frames_pdf(path, self.visible_entries(), self.columns_spin.value(), self.threshold_edit.value())
        notify(f"Wrote {Path(path).name}")


# ----------------------------------------------------------------------
# Setup page shared with the video grid
# ----------------------------------------------------------------------


class LabelSetupPage(QWidget):
    """Label classes + metadata filters + cameras — what both grids start from."""

    def __init__(self, meta, *, label_ids=None, trials=None, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.app_state = meta.app_state
        self.labels_widget = meta.labels_widget
        self._restrict_trials = set(trials) if trials else None
        #: The label classes this run is about — chosen elsewhere (the
        #: curation scope, or what the Predict dialog just wrote) and only
        #: *shown* here: the one place to pick labels is the scope area.
        self._label_ids = [int(i) for i in (label_ids or ()) if int(i) != 0]

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        self.layout_ = layout

        labels_group = QGroupBox("Labels in scope")
        labels_lay = QVBoxLayout(labels_group)
        mappings = self.mappings()
        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.label_list.setFocusPolicy(Qt.NoFocus)
        for label_id in self._label_ids:
            info = mappings.get(label_id, {})
            name = info.get("name", str(label_id))
            event_type = info.get("event_type", "state")
            item = QListWidgetItem(f"{label_id} — {name}  ({event_type})")
            item.setData(Qt.UserRole, label_id)
            item.setFlags(Qt.ItemIsEnabled)
            self.label_list.addItem(item)
        self.label_list.setMaximumHeight(max(40, min(160, 22 * len(self._label_ids) + 6)))
        labels_lay.addWidget(self.label_list)
        scope_hint = QLabel(
            "No labels in scope — drag label rows into the Curation section's scope area."
            if not self._label_ids
            else "To change this, drag other label rows into the Curation section's scope area."
        )
        scope_hint.setWordWrap(True)
        scope_hint.setStyleSheet("color: grey; font-size: 10px;")
        labels_lay.addWidget(scope_hint)
        if self._restrict_trials is not None:
            restricted = QLabel(f"Restricted to {len(self._restrict_trials)} trials handed over for review.")
            restricted.setWordWrap(True)
            restricted.setStyleSheet("color: grey; font-size: 10px;")
            labels_lay.addWidget(restricted)
        layout.addWidget(labels_group)

        # Which labels of those classes: everything, or one side of the
        # human/model divide. Reviewing a model's output means looking at what
        # it wrote and nothing else, while checking one's own labelling means
        # the opposite — and both grids fill up with the wrong half otherwise.
        method_group = QGroupBox("Labeling method")
        method_lay = QVBoxLayout(method_group)
        self.method_combo = QComboBox()
        for key, (text, _methods) in GRID_METHOD_FILTERS.items():
            self.method_combo.addItem(text, key)
        saved = str(self.app_state.get_with_default("grid_method_filter"))
        index = self.method_combo.findData(saved)
        self.method_combo.setCurrentIndex(index if index >= 0 else 0)
        self.method_combo.setToolTip(
            "Which labels of those classes the grid shows.\n"
            "Manual and curated are one choice: both mean a human vouched for the label."
        )
        self.method_combo.currentIndexChanged.connect(self._save_method_filter)
        method_lay.addWidget(self.method_combo)
        layout.addWidget(method_group)

        # Which trials: the trials table's filters, and nothing else — the one
        # place trials are included or excluded for every operation.
        self.trials_note = QLabel("")
        self.trials_note.setWordWrap(True)
        self.trials_note.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(self.trials_note)
        self._refresh_trials_note()
        self.app_state.trials_changed.connect(self._refresh_trials_note)

        self.camera_list: QListWidget | None = None
        cameras = list(getattr(getattr(self.app_state, "nwb_alignment", None), "cameras", None) or [])
        if cameras:
            cam_group = QGroupBox("Cameras")
            cam_lay = QVBoxLayout(cam_group)
            self.camera_list = QListWidget()
            # Seed check state from the last-saved preference (gui_settings.yaml);
            # a camera never seen before defaults to checked.
            saved = getattr(self.app_state, "grid_selected_cameras", None)
            for camera in cameras:
                cropped = self.gui_crop_for(camera) is not None
                item = QListWidgetItem(f"{camera}  (cropped)" if cropped else str(camera))
                item.setData(Qt.UserRole, camera)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                checked = str(camera) in saved if saved else True
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                self.camera_list.addItem(item)
            self.camera_list.setMaximumHeight(90)
            self.camera_list.itemChanged.connect(self._save_camera_selection)
            cam_lay.addWidget(self.camera_list)
            layout.addWidget(cam_group)

        crop_hint = QLabel(
            "A camera cropped in the GUI keeps its crop here — crop a camera\n"
            "view first to zoom every frame onto the region of interest."
        )
        crop_hint.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(crop_hint)

    def mappings(self) -> dict:
        return getattr(self.labels_widget, "_mappings", {}) or {}

    def gui_crop_for(self, camera: str | None) -> tuple[int, int, int, int] | None:
        """The display crop the GUI holds for *camera* (source pixels).

        With no named cameras the entries carry ``camera=None`` — that maps
        to whatever camera the primary view currently shows.
        """
        vm = getattr(getattr(self.meta, "data_widget", None), "video_mgr", None)
        if vm is None:
            return None
        name = camera if camera is not None else getattr(vm.primary_view, "camera_name", None)
        return vm.camera_crop(name)

    def camera_crops(self, cameras: list[str | None]) -> dict[str | None, tuple[int, int, int, int]]:
        crops = {}
        for camera in cameras:
            rect = self.gui_crop_for(camera)
            if rect is not None:
                crops[camera] = rect
        return crops

    def _refresh_trials_note(self, *_args) -> None:
        visible = self.visible_trials()
        n = len(visible) if visible is not None else 0
        text = (
            f"Runs over the {n} trial(s) the trials table currently shows — filter there "
            "(Navigation section) to include or exclude trials."
        )
        if self._restrict_trials is not None:
            text += f" Further restricted to the {len(self._restrict_trials)} trial(s) handed over for review."
        self.trials_note.setText(text)

    def visible_trials(self) -> set[str] | None:
        """Trial ids (as strings) the trials table shows; ``None`` when unknown."""
        trials = getattr(self.app_state, "trials", None)
        if not trials:
            return None
        return {str(t) for t in trials}

    def selected_label_ids(self) -> list[int]:
        """The label classes in scope — fixed for the life of the dialog."""
        return list(self._label_ids)

    def selected_cameras(self) -> list[str | None]:
        if self.camera_list is None:
            return [None]
        cameras = []
        for i in range(self.camera_list.count()):
            item = self.camera_list.item(i)
            if item.checkState() == Qt.Checked:
                cameras.append(item.data(Qt.UserRole))
        return cameras

    def selected_methods(self) -> frozenset[str] | None:
        """The labeling methods the grid shows; ``None`` = all of them."""
        return methods_for_filter(self.method_combo.currentData())

    def _save_method_filter(self, *_args) -> None:
        self.app_state.grid_method_filter = str(self.method_combo.currentData())

    def _save_camera_selection(self, *_args) -> None:
        checked = [str(c) for c in self.selected_cameras() if c is not None]
        self.app_state.grid_selected_cameras = checked

    def allowed_trials(self) -> set[str] | None:
        """The trials this run covers: what the trials table shows, further
        narrowed to a handed-over set (the Predict dialog's review)."""
        allowed = self.visible_trials()
        if self._restrict_trials is not None:
            allowed = self._restrict_trials if allowed is None else allowed & self._restrict_trials
        return allowed


# ----------------------------------------------------------------------
# Dialog: Setup tab + Frames tab
# ----------------------------------------------------------------------


class LabelGridViewDialog(QDialog):
    """One window, two tabs: *Setup* picks what to show, *Frames* is the grid.

    *label_ids* pre-ticks label classes (the curation scope) and *trials*
    narrows the run to a set of trial ids on top of the metadata filters —
    how the Predict dialog hands a batch of labels over for review.
    """

    def __init__(self, meta, parent=None, *, label_ids: list[int] | None = None, trials: set[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Label grid view")
        # Minimise/maximise sit next to the close button, so the grid goes
        # full screen in one click.
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
        )
        self.setModal(False)
        self.meta = meta
        self.app_state = meta.app_state
        self.labels_widget = meta.labels_widget
        self.grid_view: LabelGridView | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        self.setup = LabelSetupPage(meta, label_ids=label_ids, trials=trials)
        layout = self.setup.layout_

        panel_group = QGroupBox("GUI panels under each frame")
        panel_lay = QVBoxLayout(panel_group)
        self.panel_list: QListWidget | None = None
        self.window_spin: QDoubleSpinBox | None = None
        self.skip_video_cb: QCheckBox | None = None
        self.axis_auto_cb: QCheckBox | None = None
        open_panels = open_gui_panels(self.meta)
        if open_panels:
            panel_lay.addWidget(QLabel("Tick open panels to screenshot below each frame:"))
            self.panel_list = QListWidget()
            for title, widget in open_panels:
                item = QListWidgetItem(title)
                item.setData(Qt.UserRole, widget)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.panel_list.addItem(item)
            self.panel_list.setMaximumHeight(90)
            panel_lay.addWidget(self.panel_list)
            window_row = QHBoxLayout()
            window_row.addWidget(QLabel("Time window:"))
            self.window_spin = QDoubleSpinBox()
            self.window_spin.setRange(0.01, 600.0)
            self.window_spin.setDecimals(2)
            # Remembered across sessions and datasets (SCOPE_GLOBAL).
            self.window_spin.setValue(float(self.app_state.get_with_default("label_grid_window_s")))
            self.window_spin.setSuffix(" s")
            self.window_spin.setToolTip("Plot window shown around each label time — the marker sits on the label.")
            self.window_spin.valueChanged.connect(lambda v: setattr(self.app_state, "label_grid_window_s", float(v)))
            window_row.addWidget(self.window_spin, stretch=1)
            panel_lay.addLayout(window_row)
            self.axis_auto_cb = QCheckBox("Autoscale y per window")
            self.axis_auto_cb.setChecked(True)
            self.axis_auto_cb.setToolTip(
                "Fit each panel's y-range to the data visible in that capture's\n"
                "time window; unticked freezes the y-ranges as they are set now."
            )
            panel_lay.addWidget(self.axis_auto_cb)
            self.skip_video_cb = QCheckBox("Skip video loading during capture (faster)")
            self.skip_video_cb.setChecked(True)
            self.skip_video_cb.setToolTip(
                "Trial switches during capture skip loading each trial's video and\n"
                "pose — the screenshots only need the plot panels, and spawning a\n"
                "video decoder dominates the per-trial cost. Reloaded afterwards."
            )
            panel_lay.addWidget(self.skip_video_cb)
            panel_hint = QLabel("Capturing navigates the GUI through the selected labels' trials, then returns.")
            panel_hint.setStyleSheet("color: grey; font-size: 10px;")
            panel_lay.addWidget(panel_hint)
        else:
            panel_lay.addWidget(QLabel("No plot panels open in the GUI."))
        layout.addWidget(panel_group)

        generate_btn = QPushButton("Generate")
        generate_btn.setAutoDefault(False)
        generate_btn.clicked.connect(self._generate)
        layout.addWidget(generate_btn)

        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setWidget(self.setup)
        self.tabs.addTab(setup_scroll, "Setup")

        self._frames_placeholder = QLabel("Pick labels on the Setup tab and press Generate.")
        self._frames_placeholder.setAlignment(Qt.AlignCenter)
        self._frames_placeholder.setStyleSheet("color: grey;")
        self.tabs.addTab(self._frames_placeholder, "Frames")
        self.tabs.setTabEnabled(1, False)
        self.resize(460, 620)

    # Setup-page pass-throughs (the tests and the Predict dialog read these).
    @property
    def label_list(self) -> QListWidget:
        return self.setup.label_list

    @property
    def _restrict_trials(self) -> set[str] | None:
        return self.setup._restrict_trials

    def _mappings(self) -> dict:
        return self.setup.mappings()

    def _checked_panels(self) -> list[tuple[str, QWidget]]:
        if self.panel_list is None:
            return []
        panels = []
        for i in range(self.panel_list.count()):
            item = self.panel_list.item(i)
            if item.checkState() == Qt.Checked:
                panels.append((item.text(), item.data(Qt.UserRole)))
        return panels

    def _current_position(self) -> tuple[object, float] | None:
        """Where the user currently is, to jump back after panel capture."""
        video = getattr(self.app_state, "video", None)
        if video is not None:
            hit = self.app_state.from_display(video.frame_to_time(int(self.app_state.current_frame)))
            if hit is not None:
                return hit
        trial = getattr(self.app_state, "trials_sel", None)
        return (trial, 0.0) if trial is not None else None

    def _restore_position(self, position: tuple[object, float] | None):
        nav = getattr(self.meta, "navigation_widget", None)
        if position is None or nav is None:
            return
        trial, t_rel = position
        nav.jump_to_label_instance(
            {"trial": trial, "onset_s": float(t_rel), "offset_s": float("nan")},
            play=False,
        )

    def _capture_panels(self, entries: list[FrameEntry]):
        """Screenshot the ticked panels for every entry, restoring the view."""
        panels = self._checked_panels()
        nav = getattr(self.meta, "navigation_widget", None)
        if not panels or nav is None:
            return
        n_positions = len({(str(e.trial), round(e.t_rel, 6)) for e in entries})
        progress = QProgressDialog("Capturing GUI panels…", "Cancel", 0, n_positions, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        def on_progress(done: int) -> bool:
            progress.setValue(done)
            QApplication.processEvents()
            return not progress.wasCanceled()

        position = self._current_position()
        data_widget = getattr(self.meta, "data_widget", None)
        skip_video = data_widget is not None and self.skip_video_cb is not None and self.skip_video_cb.isChecked()
        if skip_video:
            # Trial switches during capture skip video/pose loading — the
            # panels being screenshotted never show them, and spawning a
            # decode worker dominates the per-trial cost.
            data_widget.suppress_video_load = True
        try:
            capture_panel_images(
                entries,
                nav=nav,
                panels=panels,
                window_s=self.window_spin.value() if self.window_spin is not None else 4.0,
                autoscale=self.axis_auto_cb is None or self.axis_auto_cb.isChecked(),
                progress_cb=on_progress,
            )
        finally:
            progress.close()
            # Jump back while still suppressed (cheap), then reload the media
            # once for wherever we landed — a same-file reload reuses the
            # loaded PlotVideo, so this is cheap when the trial didn't change.
            self._restore_position(position)
            if skip_video:
                data_widget.suppress_video_load = False
                data_widget.update_video()
                data_widget._init_or_update_extra_cameras()
                data_widget.video_mgr.sync_proxies()
                data_widget.update_pose()

    def generate(self):
        """Build the grid as if *Generate* were pressed (a workflow's entry)."""
        self._generate()

    def _generate(self):
        label_ids = self.setup.selected_label_ids()
        if not label_ids:
            notify("No labels in scope — drag label rows into the Curation section's scope area.", severity="warning")
            return
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is None or df.empty:
            notify("No labels loaded.", severity="warning")
            return
        cameras = self.setup.selected_cameras()
        if not cameras:
            notify("Tick at least one camera.", severity="warning")
            return
        entries = build_frame_entries(
            df,
            self._mappings(),
            label_ids,
            cameras,
            self.setup.allowed_trials(),
            self.setup.selected_methods(),
        )
        if not entries:
            notify(
                "No label instances match the selected labels, labeling method and metadata filters.",
                severity="warning",
            )
            return

        progress = QProgressDialog("Extracting frames…", "Cancel", 0, len(entries), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        def on_progress(done: int) -> bool:
            progress.setValue(done)
            QApplication.processEvents()
            return not progress.wasCanceled()

        source_software = self.app_state.source_software or getattr(self.app_state.ds, "source_software", None)
        decode_entry_images(
            entries,
            alignment=self.app_state.nwb_alignment,
            video_folder=self.app_state.video_folder,
            pose_folder=self.app_state.pose_folder,
            source_software=source_software,
            pose_color_by=getattr(self.app_state, "pose_color_by", "keypoint") or "keypoint",
            camera_crops=self.setup.camera_crops(cameras),
            show_keypoints=bool(self.app_state.get_with_default("pose_show_keypoints")),
            hidden_keypoints=frozenset(str(n) for n in self.app_state.get_with_default("pose_hidden_keypoints") or []),
            show_text=bool(self.app_state.get_with_default("pose_show_text")),
            current_trial=getattr(self.app_state, "trials_sel", None),
            current_video_path=getattr(self.app_state, "video_path", None),
            progress_cb=on_progress,
        )
        progress.close()
        if progress.wasCanceled():
            entries = [e for e in entries if e.image is not None or e.error]
            if not entries:
                return

        self._capture_panels(entries)
        self._show_grid(entries)

    def _show_grid(self, entries: list[FrameEntry]):
        """Put a freshly built grid on the *Frames* tab and go there."""
        old = self.tabs.widget(1)
        self.grid_view = LabelGridView(self.meta, entries, parent=self)
        self.tabs.removeTab(1)
        self.tabs.insertTab(1, self.grid_view, f"Frames ({len(entries)})")
        self.tabs.setTabEnabled(1, True)
        if old is not self._frames_placeholder:
            old.deleteLater()
        self.tabs.setCurrentIndex(1)
        if self.width() < _GRID_MIN_WIDTH:
            self.resize(max(self.width(), _GRID_MIN_WIDTH), max(self.height(), _GRID_MIN_HEIGHT))
