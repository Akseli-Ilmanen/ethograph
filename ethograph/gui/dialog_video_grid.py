"""Video grid: the labels in scope played side by side, one label class at a time.

Opened from the Labels tab's Curation section (**Video grid…**). Where the
label grid view shows one frame per boundary, this plays each label's clip —
a state event's whole span, a point event's window around the instant — so
what a behaviour *looks like in motion* can be compared across instances.

The clips are arranged for comparison, not for browsing:

* Only clips of **one label class** are on screen together; **Previous /
  Next label** switch the class (greyed out with a single class). Comparing
  twelve different behaviours at once tells you nothing; twelve instances of
  the same one do.
* Within a class the clips are ordered by **Sort**: by *duration* (the
  default — clips of a similar length share a screen and end around the same
  time) or by *confidence*, which puts the doubtful ones on the first page.
  **Previous / Next clips** page through the class a screenful at a time
  (greyed out when it fits on one). The **Label** box lists every class with
  its clip count, to go straight to one (a particular unit, say).
* **Durations…** opens one duration histogram per state-event class. A click
  on a histogram shows that class, from the screenful of clips nearest in
  length to the duration clicked (switching to a duration sort if another
  was chosen); the span on screen is shaded on the histogram.
* **Scatter…** opens the burst scatter
  (:mod:`~ethograph.gui.dialog_burst_scatter`): a click or a region on it
  **limits the grid** to those bursts — and plays them at once — until **Show all** — paging, sorting
  and the label box then run over that selection only.
* A screenful is **not scrollable** — every tile is always visible, because
  playback starts and stops for all of them at once: one Play button, one
  slider spanning the **longest clip on screen** (shorter clips hold their
  last frame), played once and stopped. **←/→** pause and step every tile
  one frame (of the page's fastest clip) back or forward.
* A point event's clip carries a **red marker** in the bottom-right corner on
  the frame the event falls on.

Frames are decoded one screenful at a time (at :data:`CLIP_MAX_SIDE`), so the
first one is quick and memory stays bounded; while a page is on screen the
**next page decodes ahead** on a worker thread (its videos are resolved on
the GUI thread first — the alignment NWB is not thread-safe), so stepping on
is quick. The verdict bar is the label grid's
(:class:`~ethograph.gui.dialog_label_gridview.GridVerdictBar`): a single click
tags the label for review, **Done** curates every other automated label on
screen, a double click jumps the GUI there.
"""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import QPointF, Qt, QTimer, Signal
from qtpy.QtGui import QBrush, QColor, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.dialog_burst_scatter import BurstScatterDialog, scrolling_combo
from ethograph.gui.dialog_label_gridview import (
    DURATION_SORT_ORDERS,
    LOW_CONFIDENCE_COLOR,
    TAGGED_COLOR,
    VIDEO_GRID_SORT_ORDERS,
    ConfidenceEdit,
    ConfidenceHistogramsDialog,
    ConfidenceRuleController,
    GridVerdictBar,
    LabelSetupPage,
    _mapping_color_hex,
    _row_confidence,
    _row_method,
    confidence_groups,
    confidence_style,
    confidence_text,
    curation_panel_of,
    entry_inst,
    entry_key,
    is_low_confidence,
    keep_method,
    resolve_video_jobs,
    settle,
    sort_entries,
)
from ethograph.gui.notify import notify
from ethograph.gui.pose_fill import VideoFrameSource
from ethograph.labels.intervals import EVENT_TYPE_POINT, HUMAN_CONFIDENCE, LABELING_MANUAL

logger = logging.getLogger(__name__)

#: Longest side clips are decoded at. Smaller than the frame grid's thumbnails:
#: a page holds several seconds of every clip, all in memory at once.
CLIP_MAX_SIDE = 320

#: Radius of the point-event marker as a fraction of the tile's shorter side.
MARKER_RADIUS_FRACTION = 0.06
MARKER_COLOR = "#e53935"

_DUR_BG = "#1a1d21"
_DUR_COLUMNS = 3
_DUR_MIN_SIZE = (300, 260)
_DUR_BINS = 20

_GRID_MIN_WIDTH = 1100
_GRID_MIN_HEIGHT = 760

_TILE_STYLE = "QFrame#clipTile { border: 2px solid #444; border-radius: 3px; }"
_TILE_LOW_STYLE = f"QFrame#clipTile {{ border: 2px solid {LOW_CONFIDENCE_COLOR}; border-radius: 3px; }}"
_TILE_TAGGED_STYLE = f"QFrame#clipTile {{ border: 3px solid {TAGGED_COLOR}; border-radius: 3px; }}"


# ----------------------------------------------------------------------
# Pure logic (Qt-free, unit-tested in tests/test_unit/test_video_grid.py)
# ----------------------------------------------------------------------


@dataclass
class ClipEntry:
    """One clip of the grid: a label instance seen by one camera.

    Carries the same identity fields as the frame grid's ``FrameEntry`` so
    :func:`~ethograph.gui.dialog_label_gridview.entry_key` /
    :func:`~ethograph.gui.dialog_label_gridview.entry_inst` work on it.
    """

    trial: object
    camera: str | None
    label_id: int
    name: str
    event_type: str
    onset_s: float
    offset_s: float
    #: Clip bounds, trial-relative: the label's span, or a window around a
    #: point event (clamped at the trial start).
    t0: float
    t1: float
    individual: object = None
    individual_rec: object = None
    confidence: float = HUMAN_CONFIDENCE
    labeling_method: str = LABELING_MANUAL
    color_hex: str = "#ffffff"
    #: Decoded RGB frames ``(N, H, W, 3)`` from ``t0`` on, at ``fps``.
    frames: np.ndarray | None = None
    fps: float | None = None
    error: str | None = None
    #: Set when the requested window ran past the video's start or end and
    #: was cut: the clip then shows less than ``t0``–``t1``.
    note: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.t1 - self.t0)

    @property
    def is_point(self) -> bool:
        return self.event_type == EVENT_TYPE_POINT

    @property
    def point_t(self) -> float | None:
        """The point event's instant as seconds into the clip, else None."""
        return self.onset_s - self.t0 if self.is_point else None


def build_clip_entries(
    labels_df: pd.DataFrame,
    mappings: dict,
    label_ids: list[int],
    cameras: list[str | None],
    point_window_s: float,
    allowed_trials: set[str] | None = None,
    methods: frozenset[str] | None = None,
) -> list[ClipEntry]:
    """One clip per matching label row per camera.

    A state event's clip is its span; a point event's is ``±point_window_s``
    around the instant, clamped so it never starts before the trial.
    *methods* keeps only the labeling methods named (``None`` keeps every
    label).
    """
    if labels_df is None or labels_df.empty:
        return []
    rows = labels_df[labels_df["labels"].isin(label_ids)]
    if allowed_trials is not None:
        rows = rows[rows["trial"].astype(str).isin(allowed_trials)]
    rows = rows.sort_values(["trial", "onset_s"])

    entries: list[ClipEntry] = []
    for _, row in rows.iterrows():
        if not keep_method(row, methods):
            continue
        label_id = int(row["labels"])
        info = mappings.get(label_id, {})
        event_type = str(info.get("event_type", "state"))
        onset = float(row["onset_s"])
        offset = float(row["offset_s"])
        is_point = event_type == EVENT_TYPE_POINT or not math.isfinite(offset)
        if is_point:
            t0, t1 = max(0.0, onset - point_window_s), onset + point_window_s
        else:
            t0, t1 = onset, offset
        for camera in cameras:
            entries.append(
                ClipEntry(
                    trial=row["trial"],
                    camera=camera,
                    label_id=label_id,
                    name=str(info.get("name", label_id)),
                    event_type=EVENT_TYPE_POINT if is_point else "state",
                    onset_s=onset,
                    offset_s=offset,
                    t0=t0,
                    t1=t1,
                    individual=row.get("individual"),
                    individual_rec=row.get("individual_rec"),
                    confidence=_row_confidence(row),
                    labeling_method=_row_method(row),
                    color_hex=_mapping_color_hex(info),
                )
            )
    return entries


def group_clips(entries: list[ClipEntry], order: str = "duration") -> list[list[ClipEntry]]:
    """One group per label class (by id), each ordered by *order*.

    Clips of one behaviour are what is worth seeing together; how they are
    ordered *within* the class is the user's choice
    (:data:`~ethograph.gui.dialog_label_gridview.VIDEO_GRID_SORT_ORDERS`).
    **Duration** is the default because the clips play together: ones of a
    similar length share a screen and end around the same time. **Confidence**
    trades that away to put the doubtful ones on the first page.
    """
    groups: dict[int, list[ClipEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.label_id, []).append(entry)
    return [sort_entries(groups[label_id], order) for label_id in sorted(groups)]


def paginate(group: list[ClipEntry], per_page: int) -> list[list[ClipEntry]]:
    per_page = max(1, int(per_page))
    return [group[i : i + per_page] for i in range(0, len(group), per_page)] or [[]]


def nearest_window_start(group: list[ClipEntry], duration: float, per_page: int) -> int:
    """Where the screenful of *per_page* clips nearest in length to *duration* starts.

    *group* must be in a duration order (either direction): the nearest clips
    are then a contiguous run, and the run whose farthest clip is closest wins.
    """
    per_page = max(1, int(per_page))
    if len(group) <= per_page:
        return 0
    durations = [e.duration for e in group]

    def spread(start: int) -> float:
        return max(abs(durations[start] - duration), abs(durations[start + per_page - 1] - duration))

    return min(range(len(group) - per_page + 1), key=spread)


@dataclass
class DurationGroup:
    """One duration histogram: a state-event class, each label counted once."""

    label_id: int
    name: str
    color_hex: str
    values: list[float]


def duration_groups(entries: list[ClipEntry]) -> list[DurationGroup]:
    """The durations of every state-event class, one label once whatever
    camera saw it. Point events are left out — their clip length is the
    window setting, not the behaviour."""
    rows: dict[tuple, ClipEntry] = {}
    for entry in entries:
        if not entry.is_point:
            rows.setdefault(entry_key(entry), entry)
    groups: dict[int, DurationGroup] = {}
    for entry in rows.values():
        group = groups.setdefault(entry.label_id, DurationGroup(entry.label_id, entry.name, entry.color_hex, []))
        group.values.append(entry.duration)
    return [groups[label_id] for label_id in sorted(groups)]


def page_duration(page: list[ClipEntry]) -> float:
    """The slider's span: the longest clip on the page."""
    return max((e.duration for e in page), default=0.0)


def page_fps(page: list[ClipEntry]) -> float | None:
    """The tick rate a page plays at — its fastest clip's, so no clip skips frames."""
    rates = [e.fps for e in page if e.fps]
    return max(rates) if rates else None


def frame_index(entry: ClipEntry, t: float) -> int | None:
    """The decoded frame to show *t* seconds into the clip.

    A clip that has ended holds its last frame; ``None`` means nothing was
    decoded for it.
    """
    if entry.frames is None or entry.fps is None or len(entry.frames) == 0:
        return None
    idx = int(round(max(0.0, t) * entry.fps))
    return min(idx, len(entry.frames) - 1)


def marker_visible(entry: ClipEntry, t: float) -> bool:
    """Whether the point-event marker is on at *t* seconds into the clip —
    the frame the event falls on (within half a frame period)."""
    point_t = entry.point_t
    if point_t is None or not entry.fps:
        return False
    return abs(t - point_t) <= 0.5 / entry.fps


def clip_note(wanted_start: int, wanted_stop: int, nframes: int) -> str | None:
    """How a requested frame window ``[wanted_start, wanted_stop)`` was cut.

    ``None`` when the video holds the whole window. A window that ran past
    the video's start or end is what makes a clip show less than it asked
    for — worth saying on the tile instead of leaving the user to wonder.
    """
    if nframes <= 0:
        return "video reports no frames"
    cut = []
    if wanted_start < 0:
        cut.append(f"cut {-wanted_start} frames at video start")
    if wanted_stop > nframes:
        cut.append(f"cut {wanted_stop - nframes} frames at video end")
    if wanted_start >= nframes:
        return "window lies past the video end"
    return "; ".join(cut) or None


def plan_clip_jobs(
    entries: list[ClipEntry],
    *,
    alignment,
    video_folder: str | None,
    current_trial=None,
    current_video_path: str | None = None,
) -> list[tuple]:
    """The GUI-thread half of decoding: one video job per (trial, camera)
    among the clips still lacking frames.

    Sequential on purpose — the alignment NWB is not safe to read from worker
    threads — so a background prefetch plans here and only decodes off-thread.
    A clip whose video cannot be resolved gets its ``error`` set right away.
    """
    todo = [e for e in entries if e.frames is None and e.error is None]
    groups: dict[tuple[str, str | None], list[ClipEntry]] = {}
    for entry in todo:
        groups.setdefault((str(entry.trial), entry.camera), []).append(entry)
    return resolve_video_jobs(
        groups,
        alignment=alignment,
        video_folder=video_folder,
        current_trial=current_trial,
        current_video_path=current_video_path,
    )


def decode_clip_jobs(
    jobs: list[tuple],
    *,
    max_side: int = CLIP_MAX_SIDE,
    cancel: threading.Event | None = None,
    progress_cb=None,
) -> None:
    """Decode planned jobs, one worker per video, filling ``frames`` / ``fps``.

    With *progress_cb* (GUI thread) the wait pumps events and a ``False``
    return cancels; without one the call blocks until done — which is what
    makes it safe to run in a worker thread. *cancel* lets a caller abandon
    a run from outside.
    """
    if not jobs:
        return
    cancel = cancel if cancel is not None else threading.Event()
    entries = [e for job in jobs for e in job[0]]

    def report() -> bool:
        if progress_cb is None:
            return True
        return progress_cb(sum(1 for e in entries if e.frames is not None or e.error is not None))

    def run_job(job) -> None:
        group, path, fps, offset, nframes = job
        try:
            with VideoFrameSource(path, fps, nframes, max_side=max_side) as source:
                for entry in sorted(group, key=lambda e: e.t0):
                    if cancel.is_set():
                        return
                    wanted_start = int(round((entry.t0 - offset) * fps))
                    wanted_stop = int(round((entry.t1 - offset) * fps)) + 1
                    start = min(max(wanted_start, 0), max(nframes - 1, 0))
                    stop = min(max(wanted_stop, start + 1), max(nframes, 1))
                    entry.note = clip_note(wanted_start, wanted_stop, nframes)
                    entry.frames = np.ascontiguousarray(source[start:stop])
                    entry.fps = float(fps)
        except (OSError, ValueError) as exc:
            logger.warning("Clip decode failed for %s: %s", path, exc)
            for entry in group:
                if entry.frames is None:
                    entry.error = str(exc)

    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        if progress_cb is not None:
            while not all(f.done() for f in futures):
                if not report():
                    cancel.set()
                settle(50)
    report()
    for future in futures:
        future.result()


def decode_clips(
    entries: list[ClipEntry],
    *,
    alignment,
    video_folder: str | None,
    current_trial=None,
    current_video_path: str | None = None,
    max_side: int = CLIP_MAX_SIDE,
    progress_cb=None,
) -> None:
    """Plan and decode in one go on the GUI thread (clips already decoded are
    skipped) — the same two-phase shape as the frame grid."""
    todo = [e for e in entries if e.frames is None and e.error is None]
    jobs = plan_clip_jobs(
        entries,
        alignment=alignment,
        video_folder=video_folder,
        current_trial=current_trial,
        current_video_path=current_video_path,
    )
    planned = {id(e) for job in jobs for e in job[0]}
    unplanned = sum(1 for e in todo if id(e) not in planned)  # errored at planning

    def report(done_in_jobs: int) -> bool:
        return progress_cb(unplanned + done_in_jobs)

    if progress_cb is not None and not report(0):
        return
    decode_clip_jobs(jobs, max_side=max_side, progress_cb=report if progress_cb is not None else None)


# ----------------------------------------------------------------------
# Tiles + player
# ----------------------------------------------------------------------


class _ClipTile(QFrame):
    """One clip: its current frame (marker painted on), title and caption."""

    clicked = Signal(object)
    double_clicked = Signal(object)

    def __init__(self, entry: ClipEntry, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.setObjectName("clipTile")
        self.setCursor(Qt.PointingHandCursor)
        self._image_size = (160, 90)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.title = QLabel(f"{entry.name} ({entry.label_id}) · trial {entry.trial}")
        self.title.setStyleSheet(f"font-weight: bold; color: {entry.color_hex}; font-size: 11px;")
        # Word-wrapped so a long name/trial id grows the tile's height, never
        # its width — an unwrapped label's minimum width tracks its full text
        # and can exceed the column width _fit_tiles() assigned, which grows
        # the grid host and feeds back into an ever-larger next resize.
        self.title.setWordWrap(True)
        header = QHBoxLayout()
        header.setSpacing(6)
        header.addWidget(self.title, 1)
        self.confidence = QLabel(confidence_text(entry))
        self.confidence.setStyleSheet(confidence_style(entry, 0.0))
        self.confidence.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.confidence.setToolTip("Confidence of this label — red below the flag threshold.")
        header.addWidget(self.confidence, 0)
        lay.addLayout(header)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setStyleSheet("background: #111;")
        lay.addWidget(self.image, stretch=1)
        self.caption = QLabel()
        self.caption.setStyleSheet("color: grey; font-size: 10px;")
        self.caption.setWordWrap(True)
        lay.addWidget(self.caption)
        self.refresh_caption()

    def refresh_caption(self, threshold: float = 0.0) -> None:
        e = self.entry
        self.confidence.setText(confidence_text(e))
        self.confidence.setStyleSheet(confidence_style(e, threshold))
        parts = []
        if e.camera:
            parts.append(str(e.camera))
        individual = e.individual
        if individual is not None and not (isinstance(individual, float) and math.isnan(individual)):
            parts.append(str(individual))
        # Where in the trial the label sits — a point event predicted at
        # t≈0 looks like "the start of the trial", and this is how to tell.
        parts.append(f"at {e.onset_s:.2f} s" if e.is_point else f"{e.onset_s:.2f}–{e.offset_s:.2f} s")
        parts.append(f"{e.duration:.2f} s")
        parts.append(e.labeling_method)
        if e.note:
            parts.append(e.note)
        if e.error:
            parts.append(f"no video: {e.error}")
        self.caption.setText("  ·  ".join(parts))

    def set_image_size(self, width: int, height: int) -> None:
        self._image_size = (max(40, width), max(30, height))
        self.image.setFixedSize(*self._image_size)

    def show_frame(self, frame: np.ndarray | None, marker: bool) -> None:
        if frame is None:
            self.image.setText("(no frame)")
            return
        h, w = frame.shape[:2]
        qimage = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(*self._image_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        if marker:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            radius = max(4, int(min(pixmap.width(), pixmap.height()) * MARKER_RADIUS_FRACTION))
            painter.setBrush(QBrush(QColor(MARKER_COLOR)))
            painter.setPen(Qt.NoPen)
            margin = radius // 2 + 2
            painter.drawEllipse(
                pixmap.width() - 2 * radius - margin, pixmap.height() - 2 * radius - margin, 2 * radius, 2 * radius
            )
            painter.end()
        self.image.setPixmap(pixmap)

    def mousePressEvent(self, event):
        self.clicked.emit(self.entry)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self.entry)
        super().mouseDoubleClickEvent(event)


class VideoGridPlayer(QWidget):
    """The *Playback* tab: one group's page of clips, played together."""

    def __init__(
        self,
        meta,
        entries: list[ClipEntry],
        *,
        columns: int,
        per_page: int,
        decode_fn,
        prefetch_fn=None,
        parent=None,
        read_only: bool = False,
    ):
        super().__init__(parent)
        self.meta = meta
        self.app_state = meta.app_state
        self._all_entries = entries
        #: The clips come from a read-only label source: the grid only plays
        #: and navigates — no verdicts, no rescoring.
        self._read_only = read_only
        #: Entry keys the grid is limited to (a scatter selection), with the
        #: text that says so in the header; ``None`` shows every clip.
        self._subset: set[tuple] | None = None
        self._subset_text = ""
        self._sort_order = str(self.app_state.get_with_default("video_grid_sort"))
        self._groups = group_clips(entries, self._sort_order)
        self._columns = max(1, columns)
        self._per_page = max(1, per_page)
        self._decode_fn = decode_fn
        #: Called with the page Next would show, once the current one is up —
        #: so it decodes ahead and the step there is quick.
        self._prefetch_fn = prefetch_fn
        self._group_idx = 0
        #: Index in the group of the first clip on screen. Not a page number:
        #: a duration pick starts a screenful wherever its nearest clips are.
        self._start = 0
        self._t = 0.0
        self._tiles: list[_ClipTile] = []
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._slider_from_timer = False
        #: The confidence-histogram popup while it is open.
        self._hist_dialog: ConfidenceHistogramsDialog | None = None
        #: The duration-histogram popup while it is open.
        self._dur_dialog: DurationHistogramsDialog | None = None
        #: The burst-scatter popup while it is open.
        self._scatter_dialog: BurstScatterDialog | None = None

        layout = QVBoxLayout(self)
        head = QHBoxLayout()
        self.header = QLabel("")
        self.header.setStyleSheet("font-weight: bold; font-size: 13px;")
        head.addWidget(self.header, stretch=1)
        head.addWidget(QLabel("Label:"))
        # Every class at a glance, to go straight to one instead of stepping.
        self.label_combo = scrolling_combo()
        self.label_combo.setToolTip("Every label class in scope, with its clip count — pick one to show it")
        self.label_combo.currentIndexChanged.connect(self._on_label_picked)
        head.addWidget(self.label_combo)
        self.durations_btn = QPushButton("Durations…")
        self.durations_btn.setAutoDefault(False)
        self.durations_btn.setToolTip(
            "One duration histogram per state-event class. Click a histogram to show\n"
            "that class's clips nearest in length to the duration clicked."
        )
        self.durations_btn.clicked.connect(self._show_durations)
        head.addWidget(self.durations_btn)
        self.scatter_btn = QPushButton("Scatter…")
        self.scatter_btn.setAutoDefault(False)
        self.scatter_btn.setToolTip(
            "Every state label as a point, two features of your choice as axes (timing, spikes,\n"
            "behaviour). Click or draw a region there to limit the grid to those labels."
        )
        self.scatter_btn.clicked.connect(self._show_scatter)
        head.addWidget(self.scatter_btn)
        self.show_all_btn = QPushButton("Show all")
        self.show_all_btn.setAutoDefault(False)
        self.show_all_btn.setToolTip("Drop the scatter selection: show every clip again")
        self.show_all_btn.clicked.connect(lambda: self.set_subset(None))
        self.show_all_btn.hide()
        head.addWidget(self.show_all_btn)
        layout.addLayout(head)

        top = QHBoxLayout()
        # The threshold box exists before the verdict bar: the bar restyles the
        # tiles as soon as it is built, and the style reads the threshold.
        # Shared (SCOPE_GLOBAL) with the frame grid's own threshold box.
        self.threshold_edit = ConfidenceEdit(float(self.app_state.get_with_default("grid_confidence_threshold")))
        self.threshold_edit.valueChanged.connect(self._apply_styles)
        self.threshold_edit.valueChanged.connect(self._on_threshold_changed)
        self.verdict_bar = GridVerdictBar(
            meta,
            entries_fn=self.visible_entries,
            restyle_fn=self._apply_styles,
            flagged_fn=self._flagged_entries,
            read_only=read_only,
        )
        top.addWidget(self.verdict_bar, stretch=1)
        if read_only:
            self.verdict_bar.hide()
            top.addStretch(1)
        top.addWidget(QLabel("Sort:"))
        self.sort_combo = QComboBox()
        for key, text in VIDEO_GRID_SORT_ORDERS.items():
            self.sort_combo.addItem(text, key)
        self.sort_combo.setCurrentIndex(max(0, self.sort_combo.findData(self._sort_order)))
        self.sort_combo.setToolTip(
            "How the clips of a label class are ordered.\n"
            "\n"
            "Duration keeps clips of a similar length on one screen, so they end\n"
            "around the same time. Confidence puts the doubtful ones on the first\n"
            "page instead — the fastest way to review what a model was least sure of."
        )
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        top.addWidget(self.sort_combo)
        top.addSpacing(12)
        top.addWidget(QLabel("Flag confidence below:"))
        top.addWidget(self.threshold_edit)
        self.histogram_btn = QPushButton("Histogram…")
        self.histogram_btn.setAutoDefault(False)
        self.histogram_btn.setToolTip(
            "How the confidences are distributed, one histogram per label class\n"
            "(per individual too, when more than one is labelled). The part below\n"
            "the threshold is red, and the threshold can be set from there."
        )
        self.histogram_btn.clicked.connect(self._show_histograms)
        top.addWidget(self.histogram_btn)
        # The confidence rule lives in the histogram popup; this drives it.
        self.rule_controller = ConfidenceRuleController(
            meta, entries_fn=lambda: self._all_entries, parent=self, read_only=read_only
        )
        self.rule_controller.changed.connect(self._on_confidence_rescored)
        layout.addLayout(top)

        self.hint = QLabel("")
        self.hint.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(self.hint)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setSpacing(8)
        # _fit_tiles() below reserves exactly `spacing` on every edge (the
        # `+1` gap terms) — the style's own default content margins (often
        # wider than that) are not accounted for. Left at their default, the
        # tiles it sizes end up needing more room than the host currently
        # has, forcing the host (and the window) to grow to fit; the next
        # resizeEvent then computes an even bigger available width from that
        # grown host, growing the tiles again — an unbounded per-resize
        # creep, worse with few columns because the same fixed margin
        # mismatch is divided across fewer tiles. Zeroing the margins here
        # makes `_fit_tiles`'s spacing-only assumption exactly correct.
        self._grid.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._grid_host, stretch=1)

        controls = QHBoxLayout()
        # Two levels of navigation: which LABEL CLASS is on screen, and which
        # CLIPS of it (as many as fit at once). Each pair greys out when there
        # is nowhere to go — one class, or one screenful.
        self.prev_label_btn = QPushButton("◀ Previous label")
        self.prev_label_btn.setAutoDefault(False)
        self.prev_label_btn.setToolTip("Show the clips of the previous label class")
        self.prev_label_btn.clicked.connect(lambda: self._step_label(-1))
        controls.addWidget(self.prev_label_btn)
        self.prev_clips_btn = QPushButton("◀ Previous clips")
        self.prev_clips_btn.setAutoDefault(False)
        self.prev_clips_btn.setToolTip("The previous screenful of this label's clips")
        self.prev_clips_btn.clicked.connect(lambda: self._step_clips(-1))
        controls.addWidget(self.prev_clips_btn)
        self.play_btn = QPushButton("Play")
        self.play_btn.setAutoDefault(False)
        self.play_btn.setFixedWidth(70)
        self.play_btn.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_btn)
        # Playback speed, % of real time — its own sticky setting (SCOPE_GLOBAL),
        # deliberately decoupled from the GUI's own playback_speed_pct so
        # tuning it here for a review never touches main playback.
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(5, 400)
        self.speed_spin.setSingleStep(5)
        self.speed_spin.setSuffix(" %")
        self.speed_spin.setFixedWidth(72)
        self.speed_spin.setToolTip(
            "Playback speed as a % of real time (100 = the recording's own pace).\n"
            "Remembered across sessions, independent of the GUI's own playback speed."
        )
        self.speed_spin.setValue(int(round(float(self.app_state.get_with_default("video_grid_speed_pct")))))
        self.speed_spin.valueChanged.connect(self._on_speed_changed)
        controls.addWidget(self.speed_spin)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider)
        controls.addWidget(self.slider, stretch=1)
        self.time_label = QLabel("0.00 / 0.00 s")
        self.time_label.setMinimumWidth(110)
        controls.addWidget(self.time_label)
        self.next_clips_btn = QPushButton("Next clips ▶")
        self.next_clips_btn.setAutoDefault(False)
        self.next_clips_btn.setToolTip("The next screenful of this label's clips")
        self.next_clips_btn.clicked.connect(lambda: self._step_clips(+1))
        controls.addWidget(self.next_clips_btn)
        self.next_label_btn = QPushButton("Next label ▶")
        self.next_label_btn.setAutoDefault(False)
        self.next_label_btn.setToolTip("Show the clips of the next label class")
        self.next_label_btn.clicked.connect(lambda: self._step_label(+1))
        controls.addWidget(self.next_label_btn)
        layout.addLayout(controls)

        # ←/→ step one frame while anything in the player has focus (the
        # shortcut); the player itself handles them in keyPressEvent. A text
        # field keeps the arrows for its cursor — the line edit claims them
        # before the shortcut sees them.
        self.setFocusPolicy(Qt.StrongFocus)
        for key, direction in ((Qt.Key_Left, -1), (Qt.Key_Right, +1)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda d=direction: self.step_frame(d))

        self._fill_label_combo()
        self._sync_hint()
        self.show_page(0, 0)

    # ------------------------------------------------------------------
    # Paging
    # ------------------------------------------------------------------

    @property
    def group(self) -> list[ClipEntry]:
        return self._groups[self._group_idx] if self._groups else []

    @property
    def page(self) -> list[ClipEntry]:
        return self.group[self._start : self._start + self._per_page]

    def next_page(self) -> list[ClipEntry]:
        """The page **Next clips** (else **Next label**) would show — what is
        worth decoding ahead. Empty at the very end."""
        ahead = self._start + self._per_page
        if ahead < len(self.group):
            return self.group[ahead : ahead + self._per_page]
        if self._group_idx + 1 < len(self._groups):
            return paginate(self._groups[self._group_idx + 1], self._per_page)[0]
        return []

    def _step_label(self, direction: int) -> None:
        """Show the previous / next label class, from its first clips."""
        target = self._group_idx + direction
        if 0 <= target < len(self._groups):
            self.show_page(target, 0)

    def _step_clips(self, direction: int) -> None:
        """The previous / next screenful of the current label's clips."""
        if direction < 0 and self._start > 0:
            self.show_page(self._group_idx, max(0, self._start - self._per_page))
        elif direction > 0 and self._start + self._per_page < len(self.group):
            self.show_page(self._group_idx, self._start + self._per_page)

    def visible_entries(self) -> list[ClipEntry]:
        """The clips the grid runs over: all of them, or the scatter selection."""
        if self._subset is None:
            return self._all_entries
        return [e for e in self._all_entries if entry_key(e) in self._subset]

    def set_subset(self, keys: set[tuple] | None, text: str = "") -> None:
        """Limit the grid to the labels in *keys* (``None`` shows all again).

        The class on screen stays on screen when the selection still holds
        clips of it; an empty selection is refused rather than shown.
        """
        if keys is not None and not any(entry_key(e) in keys for e in self._all_entries):
            notify("That selection holds no clips.", severity="warning")
            return
        on_screen = self.group[0].label_id if self.group else None
        self._subset = None if keys is None else set(keys)
        self._subset_text = text if keys is not None else ""
        self._groups = group_clips(self.visible_entries(), self._sort_order)
        self._fill_label_combo()
        self.show_all_btn.setVisible(self._subset is not None)
        group_idx = self._group_index_of(on_screen) if on_screen is not None else None
        self.show_page(group_idx or 0, 0)

    def _fill_label_combo(self) -> None:
        self.label_combo.blockSignals(True)
        self.label_combo.clear()
        for group in self._groups:
            lead = group[0]
            self.label_combo.addItem(f"{lead.name} ({lead.label_id}) · {len(group)} clips", lead.label_id)
        self.label_combo.blockSignals(False)
        self.label_combo.setEnabled(len(self._groups) > 1)

    def _on_label_picked(self, index: int) -> None:
        if 0 <= index < len(self._groups) and index != self._group_idx:
            self.show_page(index, 0)

    def _group_index_of(self, label_id: int) -> int | None:
        for i, group in enumerate(self._groups):
            if group[0].label_id == label_id:
                return i
        return None

    def show_duration(self, label_id: int, duration: float) -> None:
        """Show *label_id*'s screenful of clips nearest in length to *duration*.

        Nearest-in-length is only a screenful under a duration order, so any
        other order gives way to short-to-long; a chosen direction is kept.
        """
        if self._sort_order not in DURATION_SORT_ORDERS:
            self.sort_combo.blockSignals(True)
            self.sort_combo.setCurrentIndex(self.sort_combo.findData("duration"))
            self.sort_combo.blockSignals(False)
            self._set_sort("duration")
        group_idx = self._group_index_of(label_id)
        if group_idx is None:
            return
        start = nearest_window_start(self._groups[group_idx], duration, self._per_page)
        self.show_page(group_idx, start)

    def show_page(self, group_idx: int, start: int) -> None:
        """Decode (if needed) and lay out the clips from *start* on, rewound."""
        self.stop()
        self._group_idx = group_idx
        self._start = max(0, min(start, max(0, len(self.group) - 1)))
        page = self.page
        self._decode_fn(page)
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._tiles = []
        for i, entry in enumerate(page):
            tile = _ClipTile(entry)
            tile.clicked.connect(self._on_tile_clicked)
            tile.double_clicked.connect(self._on_tile_double_clicked)
            self._grid.addWidget(tile, i // self._columns, i % self._columns)
            self._tiles.append(tile)
        self._fit_tiles()
        self._t = 0.0
        duration = page_duration(page)
        self.slider.blockSignals(True)
        self.slider.setRange(0, int(round(duration * 1000)))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._apply_timer_interval()
        self._render()
        self._apply_styles()
        self._sync_header()
        self._sync_nav_buttons()
        self._sync_durations()
        if self._scatter_dialog is not None:
            self._scatter_dialog.mark({entry_key(e) for e in page})
        if self._prefetch_fn is not None:
            ahead = self.next_page()
            if ahead:
                self._prefetch_fn(ahead)

    def _sync_header(self) -> None:
        if not self._groups:
            self.header.setText("No clips.")
            return
        lead = self.group[0]
        kind = "point" if lead.is_point else "state"
        first = self._start + 1
        last = min(len(self.group), first + len(self.page) - 1)
        parts = [f"{lead.name} ({lead.label_id}, {kind})"]
        if len(self._groups) > 1:
            parts.append(f"label {self._group_idx + 1} of {len(self._groups)}")
        if len(self.group) > len(self.page):
            parts.append(f"clips {first}–{last} of {len(self.group)}")
        else:
            parts.append(f"{len(self.group)} clips")
        if self._subset is not None:
            parts.append(self._subset_text)
        self.header.setText("  ·  ".join(parts))
        self.header.setStyleSheet(f"font-weight: bold; font-size: 13px; color: {lead.color_hex};")

    def _sync_nav_buttons(self) -> None:
        """Grey out whatever has nowhere to go — one label class, one screenful."""
        self.prev_label_btn.setEnabled(self._group_idx > 0)
        self.next_label_btn.setEnabled(self._group_idx < len(self._groups) - 1)
        self.prev_clips_btn.setEnabled(self._start > 0)
        self.next_clips_btn.setEnabled(self._start + self._per_page < len(self.group))
        self.label_combo.blockSignals(True)
        self.label_combo.setCurrentIndex(self._group_idx)
        self.label_combo.blockSignals(False)

    def _sync_hint(self, *_args) -> None:
        click = "Click the clips that are wrong, then Done curates every other label."
        self.hint.setText(
            f"Clips of one label class · ←/→ step a frame · {click} Double-click a clip to jump the GUI there."
        )

    # ------------------------------------------------------------------
    # Layout: every tile stays visible
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_tiles()
        self._render()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.step_frame(-1)
        elif event.key() == Qt.Key_Right:
            self.step_frame(+1)
        else:
            super().keyPressEvent(event)

    def _fit_tiles(self) -> None:
        if not self._tiles:
            return
        rows = math.ceil(len(self._tiles) / self._columns)
        spacing = self._grid.spacing()
        avail_w = max(200, self._grid_host.width() - spacing * (self._columns + 1))
        avail_h = max(150, self._grid_host.height() - spacing * (rows + 1))
        tile_w = avail_w // self._columns
        tile_h = avail_h // rows
        caption_h = 48  # title + caption rows
        for tile in self._tiles:
            tile.set_image_size(tile_w - 10, tile_h - caption_h)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    @property
    def time(self) -> float:
        return self._t

    def toggle_play(self) -> None:
        if self.playing:
            self.stop()
        else:
            self.play()

    def speed_factor(self) -> float:
        """Playback speed as a multiple of real time (the spin's % / 100)."""
        return max(0.01, self.speed_spin.value() / 100.0)

    def _on_speed_changed(self, value: int) -> None:
        self.app_state.video_grid_speed_pct = float(value)
        self._apply_timer_interval()

    def _apply_timer_interval(self) -> None:
        """One tick per frame at the chosen speed, never faster than 100 Hz.

        Slowing down stretches the tick; speeding up shortens it until the
        floor, past which each tick advances more than one frame instead —
        either way :meth:`_tick` advances wall-clock time × speed, so the
        clip's pace is right whatever the tile refresh rate.
        """
        fps = page_fps(self.page)
        if not fps:
            self._timer.setInterval(40)
            return
        self._timer.setInterval(max(10, int(round(1000.0 / (fps * self.speed_factor())))))

    def play(self) -> None:
        duration = page_duration(self.page)
        if duration <= 0.0:
            return
        if self._t >= duration:
            self._t = 0.0  # Play at the end starts the page over
            self._render()
        self._timer.start()
        self.play_btn.setText("Pause")

    def stop(self) -> None:
        self._timer.stop()
        self.play_btn.setText("Play")

    def _tick(self) -> None:
        fps = page_fps(self.page)
        if not fps:
            self.stop()
            return
        duration = page_duration(self.page)
        self._t = min(self._t + self.speed_factor() * self._timer.interval() / 1000.0, duration)
        if self._t >= duration:
            # Play once and stop on the last frames; Play again rewinds.
            self.stop()
        self._sync_slider()
        self._render()

    def step_frame(self, direction: int) -> None:
        """←/→: pause and show the previous / next frame on every tile.

        One frame of the page's fastest clip, snapped to the frame grid so
        repeated steps never drift; clamped to the page's span.
        """
        fps = page_fps(self.page)
        if not fps:
            return
        self.stop()
        duration = page_duration(self.page)
        idx = int(round(self._t * fps)) + direction
        self._t = min(max(idx / fps, 0.0), duration)
        self._sync_slider()
        self._render()

    def _sync_slider(self) -> None:
        """Move the slider to ``_t`` without it seeking back."""
        self._slider_from_timer = True
        self.slider.setValue(int(round(self._t * 1000)))
        self._slider_from_timer = False

    def _on_slider(self, value: int) -> None:
        if self._slider_from_timer:
            return
        self._t = value / 1000.0
        self._render()

    def seek(self, t: float) -> None:
        """Move the page to *t* seconds (tests and programmatic use)."""
        self.slider.setValue(int(round(max(0.0, t) * 1000)))

    def _render(self) -> None:
        for tile in self._tiles:
            entry = tile.entry
            idx = frame_index(entry, self._t)
            frame = entry.frames[idx] if idx is not None else None
            tile.show_frame(frame, marker_visible(entry, self._t))
        self.time_label.setText(f"{self._t:.2f} / {page_duration(self.page):.2f} s")

    # ------------------------------------------------------------------
    # Verdicts + navigation
    # ------------------------------------------------------------------

    def _flagged_entries(self) -> list[ClipEntry]:
        threshold = self.threshold_edit.value()
        return [e for e in self.visible_entries() if is_low_confidence(e, threshold)]

    def _on_threshold_changed(self, value: float) -> None:
        self.app_state.grid_confidence_threshold = float(value)

    def _on_sort_changed(self, *_args) -> None:
        """Regroup and show the first page of the class that was on screen.

        Reordering moves clips between pages, so keeping the page number would
        land somewhere arbitrary; the class is what the user was looking at,
        so that is what is kept.
        """
        self._set_sort(self.sort_combo.currentData())
        self.show_page(min(self._group_idx, max(0, len(self._groups) - 1)), 0)

    def _set_sort(self, order: str) -> None:
        self._sort_order = order
        self.app_state.video_grid_sort = order
        self._groups = group_clips(self.visible_entries(), order)

    def _apply_styles(self, *_args) -> None:
        threshold = self.threshold_edit.value()
        verdicts = self.verdict_bar.verdicts
        for tile in self._tiles:
            entry = tile.entry
            if verdicts.is_clicked(entry):
                tile.setStyleSheet(_TILE_TAGGED_STYLE)
            elif is_low_confidence(entry, threshold):
                tile.setStyleSheet(_TILE_LOW_STYLE)
            else:
                tile.setStyleSheet(_TILE_STYLE)
            tile.refresh_caption(threshold)
        if self._hist_dialog is not None:
            self._hist_dialog.set_threshold(threshold)

    def _on_confidence_rescored(self) -> None:
        """The confidence knob changed the entries' values: restyle, and refresh the histogram."""
        self._apply_styles()
        if self._hist_dialog is not None:
            self._hist_dialog.set_groups(confidence_groups(self._all_entries))

    def _show_histograms(self) -> None:
        """Open (or raise) the per-label confidence histograms, scope-wide."""
        if self._hist_dialog is not None:
            self._hist_dialog.show()
            self._hist_dialog.raise_()
            return
        groups = confidence_groups(self._all_entries)
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

    def _on_histograms_closed(self, *_args) -> None:
        self._hist_dialog = None

    def _show_durations(self) -> None:
        """Open (or raise) the per-class duration histograms."""
        if self._dur_dialog is not None:
            self._dur_dialog.show()
            self._dur_dialog.raise_()
            return
        groups = duration_groups(self._all_entries)
        if not groups:
            notify("No state events in scope — point events have no duration to plot.", severity="warning")
            return
        self._dur_dialog = DurationHistogramsDialog(groups, parent=self)
        self._dur_dialog.picked.connect(self.show_duration)
        self._dur_dialog.destroyed.connect(self._on_durations_closed)
        self._sync_durations()
        self._dur_dialog.show()

    def _on_durations_closed(self, *_args) -> None:
        self._dur_dialog = None

    def _sync_durations(self) -> None:
        """Shade the span of lengths on screen on the duration histograms."""
        if self._dur_dialog is None:
            return
        page = self.page
        if not page or page[0].is_point:
            self._dur_dialog.mark(None, 0.0, 0.0)
            return
        lengths = [e.duration for e in page]
        self._dur_dialog.mark(page[0].label_id, min(lengths), max(lengths))

    def _show_scatter(self) -> None:
        """Open (or raise) the burst scatter over every state label in the grid."""
        if self._scatter_dialog is not None:
            self._scatter_dialog.show()
            self._scatter_dialog.raise_()
            return
        if all(e.is_point for e in self._all_entries):
            notify("No state events in scope — the scatter plots labels that last.", severity="warning")
            return
        self._scatter_dialog = BurstScatterDialog(
            self.meta, self._all_entries, n_nearest_fn=lambda: self._per_page, parent=self
        )
        self._scatter_dialog.selected.connect(self._on_scatter_selected)
        self._scatter_dialog.cleared.connect(lambda: self.set_subset(None))
        self._scatter_dialog.destroyed.connect(self._on_scatter_closed)
        self._scatter_dialog.mark({entry_key(e) for e in self.page})
        self._scatter_dialog.show()

    def _on_scatter_selected(self, keys: set[tuple], text: str) -> None:
        """A pick on the scatter shows its clips and plays them at once — picking is asking to watch."""
        self.set_subset(keys, text)
        if self._subset == keys:
            self.play()

    def _on_scatter_closed(self, *_args) -> None:
        self._scatter_dialog = None

    def _on_tile_clicked(self, entry: ClipEntry) -> None:
        """A single click tags the clip's label for review."""
        self.verdict_bar.click(entry)

    def _on_tile_double_clicked(self, entry: ClipEntry) -> None:
        """A double click navigates, tagged or not. Qt delivers a plain press
        first, which already toggled the tile — toggling again undoes it, so
        navigating leaves the verdicts exactly as they were."""
        self.verdict_bar.click(entry)
        self._jump(entry)

    def _jump(self, entry: ClipEntry) -> None:
        self.stop()
        panel = curation_panel_of(self.meta)
        if panel is not None and panel.mode() == "frame":
            panel.start_review_at(entry_inst(entry), "point" if entry.is_point else "start")
            return
        nav = getattr(self.meta, "navigation_widget", None)
        if nav is None:
            return
        nav.jump_to_label_instance(entry_inst(entry), seek_rel=entry.onset_s, play=False)


class DurationHistogramsDialog(QDialog):
    """How long each state-event class's labels last, one histogram per class.

    Clicking a histogram emits :attr:`picked` with its class and the duration
    under the cursor; the player answers by showing the clips nearest it and
    calls :meth:`mark` to shade what is on screen.
    """

    picked = Signal(int, float)

    def __init__(self, groups: list[DurationGroup], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Duration histograms")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._groups = list(groups)
        self._plots: dict[int, pg.PlotWidget] = {}
        self._regions: dict[int, pg.LinearRegionItem] = {}

        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Bins:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 100)
        self.bins_spin.setValue(_DUR_BINS)
        self.bins_spin.valueChanged.connect(self._redraw)
        bar.addWidget(self.bins_spin)
        bar.addStretch()
        layout.addLayout(bar)
        hint = QLabel(
            "Each label counts once, whatever camera saw it · click a histogram to show that class's "
            "clips nearest in length to where you clicked · shaded: the lengths on screen."
        )
        hint.setStyleSheet("color: grey; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(8)
        for i, group in enumerate(self._groups):
            plot = pg.PlotWidget()
            plot.setBackground(_DUR_BG)
            plot.setTitle(f"{group.name} ({group.label_id})   n={len(group.values)}", color=group.color_hex)
            plot.setLabel("bottom", "duration (s)")
            plot.setLabel("left", "labels")
            plot.setMinimumSize(*_DUR_MIN_SIZE)
            plot.setMouseEnabled(x=False, y=False)
            plot.setMenuEnabled(False)
            plot.scene().sigMouseClicked.connect(lambda ev, g=group, p=plot: self._on_click(ev, g, p))
            grid.addWidget(plot, i // _DUR_COLUMNS, i % _DUR_COLUMNS)
            self._plots[group.label_id] = plot
        scroll.setWidget(container)
        layout.addWidget(scroll)
        self._redraw()

        columns = min(_DUR_COLUMNS, max(1, len(self._groups)))
        rows = math.ceil(len(self._groups) / _DUR_COLUMNS) if self._groups else 1
        width = columns * (_DUR_MIN_SIZE[0] + 20) + 40
        height = rows * (_DUR_MIN_SIZE[1] + 20) + 100
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            width, height = min(width, int(avail.width() * 0.95)), min(height, int(avail.height() * 0.9))
        self.resize(width, height)

    def _redraw(self, *_args) -> None:
        bins = self.bins_spin.value()
        for group in self._groups:
            plot = self._plots[group.label_id]
            plot.clear()
            counts, edges = np.histogram(np.asarray(group.values, dtype=float), bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2.0
            plot.addItem(
                pg.BarGraphItem(
                    x=centers,
                    height=counts,
                    width=(edges[1] - edges[0]) * 0.9,
                    brush=pg.mkBrush(group.color_hex),
                    pen=pg.mkPen(None),
                )
            )
            region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(255, 255, 255, 50))
            region.setZValue(10)
            region.hide()
            plot.addItem(region)
            self._regions[group.label_id] = region

    def _on_click(self, event, group: DurationGroup, plot: pg.PlotWidget) -> None:
        point: QPointF = plot.getPlotItem().getViewBox().mapSceneToView(event.scenePos())
        self.picked.emit(group.label_id, max(0.0, float(point.x())))

    def mark(self, label_id: int | None, lo: float, hi: float) -> None:
        """Shade ``lo``–``hi`` on *label_id*'s histogram (the lengths on
        screen) and clear every other; ``None`` clears them all."""
        for gid, region in self._regions.items():
            if gid == label_id:
                region.setRegion((lo, max(hi, lo + 1e-3)))
                region.show()
            else:
                region.hide()


# ----------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------


@dataclass
class _Prefetch:
    """One page decoding ahead: which clips, the worker's future, its cancel flag."""

    entry_ids: set[int]
    future: Future
    cancel: threading.Event


class VideoGridDialog(QDialog):
    """*Setup* picks labels, filters, cameras and the playback layout;
    *Playback* is the grid player."""

    def __init__(self, meta, parent=None, *, label_ids: list[int] | None = None, trials: set[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Video grid")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
        )
        self.setModal(False)
        self.meta = meta
        self.app_state = meta.app_state
        self.player: VideoGridPlayer | None = None
        #: The page being decoded ahead, if any: planned on the GUI thread,
        #: decoding on the one prefetch worker. A jump onto it waits for the
        #: worker instead of decoding twice; a jump elsewhere abandons it.
        self._prefetch: _Prefetch | None = None
        self._prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-grid-prefetch")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        self.setup = LabelSetupPage(meta, label_ids=label_ids, trials=trials)
        layout = self.setup.layout_

        # The layout choices are remembered across sessions and datasets
        # (SCOPE_GLOBAL): each spin opens at the saved value and writes back.
        state = self.app_state
        play_group = QGroupBox("Playback layout")
        play_lay = QGridLayout(play_group)
        play_lay.addWidget(QLabel("Window around point events:"), 0, 0)
        self.point_window_spin = QDoubleSpinBox()
        self.point_window_spin.setRange(0.05, 30.0)
        self.point_window_spin.setDecimals(2)
        self.point_window_spin.setSingleStep(0.25)
        self.point_window_spin.setValue(float(state.get_with_default("video_grid_point_window_s")))
        self.point_window_spin.setSuffix(" s")
        self.point_window_spin.setToolTip("A point event's clip spans ± this around the instant (red marker on it)")
        self.point_window_spin.valueChanged.connect(lambda v: setattr(state, "video_grid_point_window_s", float(v)))
        play_lay.addWidget(self.point_window_spin, 0, 1)
        play_lay.addWidget(QLabel("Clips on screen:"), 1, 0)
        self.per_page_spin = QSpinBox()
        self.per_page_spin.setRange(1, 24)
        self.per_page_spin.setValue(int(state.get_with_default("video_grid_per_page")))
        self.per_page_spin.setToolTip(
            "How many clips of one label class show (and play) at once — the view never scrolls,\n"
            "so keep it to what fits; Previous / Next clips step through the rest."
        )
        self.per_page_spin.valueChanged.connect(lambda v: setattr(state, "video_grid_per_page", int(v)))
        play_lay.addWidget(self.per_page_spin, 1, 1)
        play_lay.addWidget(QLabel("Columns:"), 2, 0)
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, 8)
        self.columns_spin.setValue(int(state.get_with_default("video_grid_columns")))
        self.columns_spin.valueChanged.connect(lambda v: setattr(state, "video_grid_columns", int(v)))
        play_lay.addWidget(self.columns_spin, 2, 1)
        layout.addWidget(play_group)

        mem_hint = QLabel(
            "Clips decode page by page at a reduced size when a page is shown —\n"
            "long state events and many clips per page take longer to open."
        )
        mem_hint.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(mem_hint)

        generate_btn = QPushButton("Generate")
        generate_btn.setAutoDefault(False)
        generate_btn.clicked.connect(self._generate)
        layout.addWidget(generate_btn)

        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setWidget(self.setup)
        self.tabs.addTab(setup_scroll, "Setup")

        self._placeholder = QLabel("Pick labels on the Setup tab and press Generate.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: grey;")
        self.tabs.addTab(self._placeholder, "Playback")
        self.tabs.setTabEnabled(1, False)
        self.resize(460, 620)

    @property
    def label_list(self):
        return self.setup.label_list

    def _prefetch_page(self, page: list[ClipEntry]) -> None:
        """Start decoding *page* ahead on the prefetch worker (the pending
        one, if for another page, is abandoned)."""
        self._drop_prefetch()
        jobs = plan_clip_jobs(
            page,
            alignment=self.app_state.nwb_alignment,
            video_folder=self.app_state.video_folder,
            current_trial=getattr(self.app_state, "trials_sel", None),
            current_video_path=getattr(self.app_state, "video_path", None),
        )
        if not jobs:
            return
        cancel = threading.Event()
        future = self._prefetch_pool.submit(decode_clip_jobs, jobs, cancel=cancel)
        self._prefetch = _Prefetch({id(e) for job in jobs for e in job[0]}, future, cancel)

    def _drop_prefetch(self) -> None:
        if self._prefetch is not None:
            self._prefetch.cancel.set()
            self._prefetch = None

    def _await_prefetch(self, page: list[ClipEntry]) -> bool:
        """Wait for a prefetch that covers *page*; ``False`` if the user
        cancelled the wait. A prefetch for some other page is abandoned."""
        pre = self._prefetch
        if pre is None:
            return True
        if not pre.entry_ids & {id(e) for e in page}:
            self._drop_prefetch()
            return True
        self._prefetch = None
        if not pre.future.done():
            progress = QProgressDialog("Finishing clips…", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            while not pre.future.done():
                if progress.wasCanceled():
                    pre.cancel.set()
                settle(50)
            progress.close()
            if progress.wasCanceled():
                return False
        pre.future.result()
        return True

    def _decode_page(self, page: list[ClipEntry]) -> None:
        if not self._await_prefetch(page):
            return
        todo = [e for e in page if e.frames is None and e.error is None]
        if not todo:
            return
        progress = QProgressDialog("Decoding clips…", "Cancel", 0, len(todo), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        def on_progress(done: int) -> bool:
            progress.setValue(done)
            QApplication.processEvents()
            return not progress.wasCanceled()

        decode_clips(
            page,
            alignment=self.app_state.nwb_alignment,
            video_folder=self.app_state.video_folder,
            current_trial=getattr(self.app_state, "trials_sel", None),
            current_video_path=getattr(self.app_state, "video_path", None),
            progress_cb=on_progress,
        )
        progress.close()

    def generate(self) -> None:
        """Build the grid as if *Generate* were pressed (a workflow's entry)."""
        self._generate()

    def _generate(self) -> None:
        label_ids = self.setup.selected_label_ids()
        if not label_ids:
            notify("No labels in scope — drag label rows into the Curation section's scope area.", severity="warning")
            return
        df = self.setup.labels_df()
        if df is None or df.empty:
            notify("No labels loaded.", severity="warning")
            return
        cameras = self.setup.selected_cameras()
        if not cameras:
            notify("Tick at least one camera.", severity="warning")
            return
        entries = build_clip_entries(
            df,
            self.setup.mappings(),
            label_ids,
            cameras,
            self.point_window_spin.value(),
            self.setup.allowed_trials(),
            self.setup.selected_methods(),
        )
        if not entries:
            notify(
                "No label instances match the selected labels, labeling method and metadata filters.",
                severity="warning",
            )
            return
        self._show_player(entries)

    def _show_player(self, entries: list[ClipEntry]) -> None:
        old = self.tabs.widget(1)
        if self.player is not None:
            self.player.stop()
        self._drop_prefetch()
        self.player = VideoGridPlayer(
            self.meta,
            entries,
            columns=self.columns_spin.value(),
            per_page=self.per_page_spin.value(),
            decode_fn=self._decode_page,
            prefetch_fn=self._prefetch_page,
            parent=self,
            read_only=not self.setup.source.writable,
        )
        self.tabs.removeTab(1)
        self.tabs.insertTab(1, self.player, f"Playback ({len(entries)})")
        self.tabs.setTabEnabled(1, True)
        if old is not self._placeholder:
            old.deleteLater()
        self.tabs.setCurrentIndex(1)
        self.player.setFocus()  # so ←/→ step frames straight away
        if self.width() < _GRID_MIN_WIDTH:
            self.resize(max(self.width(), _GRID_MIN_WIDTH), max(self.height(), _GRID_MIN_HEIGHT))

    def closeEvent(self, event):
        if self.player is not None:
            self.player.stop()
        self._drop_prefetch()
        self._prefetch_pool.shutdown(wait=False)
        super().closeEvent(event)
