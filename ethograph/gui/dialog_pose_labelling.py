"""Keypoint labelling dialog: label a few frames, let a tracker fill the rest.

One tab per stage of the work — **Define keypoints** (who is labelled and with
which keypoints), **Label & Edit** (the modes, the points table, frame
suggestions), **Detect** (optional marker detection), **Calibrate** (optional
pixel→cm landmark clicking) and **Fill and export**. One column holding every
group at once grew taller than a screen; the split is by stage, so nothing a
stage needs sits on another tab.

Scope: **one video at a time** — a single camera and a single trial, which is
what a drag & drop of a video gives you. The store is keyed by frame index on
that video's own frame grid, the sidecar sits next to that video, and the fill
backends see one continuous clip; there is no trial or camera axis anywhere in
the model, so a multi-trial ``TrialTree`` (or a second camera view) is *not*
supported — the dialog always follows ``app_state.video_path``, the primary
camera's current video, and labels made against another trial are simply
another sidecar. The full design rules live in
``docs/source/advanced/keypoint_labelling/``.

Non-modal, because the whole point is to keep navigating frames with the normal
playhead while labelling. It owns a :class:`~ethograph.gui.pose_annotate.KeypointStore`,
attaches a :class:`~ethograph.gui.pose_edit_mixin.KeypointLabelMode` to the
primary camera view, and renders fill results back through the ordinary pose
overlay so filled keypoints are indistinguishable from imported predictions.

The panel is hierarchical, following SLEAP: one keypoint schema (the skeleton)
shared by every individual, and one branch per individual whose children are
that individual's keypoints on the current frame. Clicking a branch selects the
individual to label; clicking a leaf selects the keypoint. Labelling one
individual is simply the case where there is one branch, and no branches at all
is a legal state too — the tree is empty until an individual is added.

Unticking "Individuals share the same keypoints" gives each individual its own
subset of the schema, for animals that cannot be labelled symmetrically. The
keypoint buttons then edit the selected individual's set instead of the schema
everyone shares.

The canvas has two modes, armed by their buttons (pressing the active one
again turns labelling off), after napari-deeplabcut: **Sequential** labels every
keypoint on one frame and never navigates by itself, **Loop** sweeps one
keypoint across frames. What Loop does after each click is the "Between clicks"
dropdown — stay put, step a frame, or jump to the next suggested frame — rather
than being inferred from whether a suggestion list happens to exist. Editing
needs no mode — clicking an existing point always drags it, ``Backspace``
deletes the selected point and ``Ctrl+Z`` undoes. Clicking a *filled* point pins
it as a label, which is how a prediction is accepted or corrected. See
:mod:`~ethograph.gui.pose_edit_mixin`.

Navigation: plain ``←``/``→`` step one frame (the main window's own binding,
untouched) and ``N`` jumps to the next suggested frame — the ones worth
labelling, which is what you actually move through while annotating. Going
*back* has no key: the suggestions are a queue to work down, and any frame at
all is one click away in the points table.

The points table has one row per ``(frame, individual)`` and an ``x``/``y``/
``conf`` column triple per keypoint, so everything on a frame is visible at
once. The keypoint name is painted once *above* its triple by
:class:`GroupedHeaderView`. Clicking a cell seeks the playhead to that frame and
makes the clicked keypoint active; conversely the playhead's own row is selected
and scrolled to. Rows are multi-selectable and right-click deletes their labels
— or pins their predictions.

Confidence is per keypoint rather than per row, because that is the granularity
the fill actually produces: a row-level average let nine well-tracked points
hide the one the tracker lost, which is the only point worth going back for.

Human labels and filled predictions live in the same table. A ``Source`` column
says which a row is (one hand-placed point makes the row the user's), predicted
coordinates are dimmed, and the ``Individual`` and ``Source`` headers carry the
funnel filters of :mod:`~ethograph.gui.table_filter` — so "show me only what I
labelled", or "only the frames the fill invented", is one click. Confidence
carries no funnel: one per keypoint would AND together ("beak *and* tail below
0.5") when the question is always "*any* point below 0.5", and that question is
what the "Lowest fill confidence" suggestion answers instead.

Before a fill the rows are the labelled frames; afterwards they are every frame
the fill covers — the span between the outermost labels — which is why the table
is a virtual model (:class:`PointTableModel`) rather than a widget grid.

"Load into the GUI" turns the keypoints — and optionally velocity, speed and
acceleration derived from them — into ordinary plottable features, so a fill can
be inspected without writing and reloading a file. The keypoints are themselves a
complete dataset: the time axis is the video's own frame grid (frame index ×
1/fps) and the keypoint/individual names come from this dialog, so it works with
no dataset loaded at all — one is created.

Anchors are project data, not settings: they are persisted to a sidecar next to
the video (``<video>.keypoints.json``), never into ``app_state``. A fill is not
persisted at all, and never feeds the next fill — see
:mod:`~ethograph.gui.pose_annotate`.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from qtpy.QtCore import QAbstractTableModel, QEvent, QModelIndex, QRect, Qt, QTimer
from qtpy.QtGui import QBrush, QColor, QIcon, QImage, QKeySequence, QPalette, QPen, QPixmap, QShortcut
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.dialog_busy_progress import BusyProgressDialog
from ethograph.gui.file_dialogs import browse_open_file
from ethograph.gui.notify import notify
from ethograph.gui.pose_annotate import (
    DEFAULT_INDIVIDUAL,
    KINEMATICS,
    LEARNED,
    MANUAL,
    MIN_CALIBRATION_LANDMARKS,
    RECOMMENDED_LABEL_SHARE,
    AssignmentError,
    KeypointStore,
    KeypointStoreError,
    detections_path,
    keypoints_dataset_path,
    load_world_coordinates,
    refinement_path,
    sidecar_path,
    store_to_dataset,
    store_to_movement_ds,
)
from ethograph.gui.pose_convert import COLOR_BY_INDIVIDUAL, COLOR_BY_KEYPOINT, COLOR_BY_MODES
from ethograph.gui.pose_detect import (
    APRILTAG_DETECTOR,
    TAG_FAMILIES,
    TAG_PARTS,
    PointDetectorError,
    available_detectors,
    build_detector,
    diagnose_frame,
    family_note,
    label_name,
    label_preview,
    learn_assignment,
    run_detector,
)
from ethograph.gui.pose_detect_preview import PreviewPanel
from ethograph.gui.pose_edit_mixin import (
    LOOP_MODE,
    SEQUENTIAL_MODE,
    CalibrationClickMode,
    KeypointLabelMode,
    individual_colors_for,
    keypoint_colors_for,
)
from ethograph.gui.pose_fill import (
    COTRACKER_CHECKPOINT_NAME,
    POSEPAL_BACKEND,
    VideoFrameSource,
    available_backends,
    build_backend,
    cotracker_checkpoint_dir,
)
from ethograph.gui.pose_render import movement_ds_to_pose_render
from ethograph.gui.pose_suggest import suggest_frames
from ethograph.gui.table_filter import (
    SORT_ROLE,
    CategoryFilterDialog,
    FilterHeaderView,
    MultiColumnFilterProxy,
)

logger = logging.getLogger(__name__)


@contextmanager
def _blocked(widget):
    """Mute a widget's signals — these combos are readouts as well as inputs."""
    previous = widget.blockSignals(True)
    try:
        yield widget
    finally:
        widget.blockSignals(previous)


#: Longest image side fed to the tracking backends. Downscaling is the single
#: biggest CPU speedup and costs almost nothing in accuracy at this anchor density.
MAX_SIDE = 512

#: Detection decodes at **full resolution** — ``None`` for no downscaling.
#:
#: The tracking backends can afford 512 px because they follow a texture the
#: user pointed at; a tag decoder has nothing but pixels. A 6-module tag wants
#: ~5 px per module *including its border*, so on a 1920 px video the 3.75×
#: downscale turns a healthy 45 px tag into 12 px — under the ~3 px/module cliff
#: — and the quad finder stops proposing it at all. Every millimetre of sizing
#: advice in `pose_tagsheet` is computed against the camera's real width, so
#: detection has to see that width too or the advice is self-defeating.
DETECT_MAX_SIDE = None

_LABELLED_MARK = "●"
_UNLABELLED_MARK = "·"

#: The sidecar most recently saved by any labelling dialog in this GUI
#: session: what a fresh store's static keypoints are seeded from.
_LAST_SAVED_SIDECAR: dict[str, str] = {}


def template_path(video_path: str) -> Path:
    """The camera's static-keypoint template: ``{video folder}/.ethograph/keypoints_template.json``."""
    return Path(video_path).parent / ".ethograph" / "keypoints_template.json"


#: Side of the colour swatch drawn beside a keypoint in the schema tree.
_SWATCH_PX = 12

#: Keeps the points table from eating the dialog — it scrolls past this.
#: Sized so the dialog's default height shows a useful run of frames rather
#: than capping the table early and leaving the extra space blank.
TABLE_MAX_HEIGHT = 380

#: Columns before the per-keypoint ``x``/``y``/``conf`` triples.
_FIXED_COLUMNS = ("Frame", "Individual", "Source")

#: Column indices of :data:`_FIXED_COLUMNS`, since they read badly as bare
#: numbers in the filter wiring.
FRAME_COLUMN, INDIVIDUAL_COLUMN, SOURCE_COLUMN = range(len(_FIXED_COLUMNS))

#: What each keypoint contributes to the table, in column order. Confidence sits
#: beside the coordinates it scores rather than being averaged into one figure
#: per row: the fill produces a number per point, and the point it lost is
#: precisely the one a row-level mean would bury.
_KEYPOINT_AXES = ("x", "y", "conf")
COLUMNS_PER_KEYPOINT = len(_KEYPOINT_AXES)

#: Provenance of a ``(frame, individual)`` row — see ``KeypointStore.is_human``.
#: Three states, ranked as the store ranks them: a row you touched is yours, a
#: row a detector read off the pixels is evidence, and the rest is inference.
HUMAN_SOURCE = "Human"
DETECTED_SOURCE = "Detected"
FILL_SOURCE = "Fill"

#: Backends that score confidence by forward/backward tracking agreement, and
#: so take a disagreement tolerance. The spline scores by distance instead.
_TRACKING_BACKENDS = ("flow", POSEPAL_BACKEND)

_FIXED_COLUMN_TOOLTIPS = (
    "Video frame. Click a cell to jump the playhead there.",
    "Which individual this row's points belong to.",
    f"{HUMAN_SOURCE} — you placed or corrected at least one point on this row.\n"
    f"{DETECTED_SOURCE} — a marker detector read at least one point off this\n"
    "frame's pixels; the rest, if any, came from the fill.\n"
    f"{FILL_SOURCE} — every point here was interpolated between other frames.\n\n"
    "Filling always rebuilds from the observed points — yours and the\n"
    "detector's — so a *filled* point only survives a re-fill once you pin it\n"
    "(click it, or use the right-click menu here).",
)

#: The scoring scheme, shown on every ``conf`` header. It belongs on the column
#: rather than in a manual: it is the one number here nobody can derive by
#: looking at the video.
_CONFIDENCE_TOOLTIP = (
    "How much the fill trusts this one point.\n"
    "1.00 means you labelled it by hand; low means the fill was lost.\n\n"
    "Spline: decays with distance from the nearest labelled frame.\n"
    "Optical flow and PosePAL: each gap is tracked twice, forwards from\n"
    "the label on its left and backwards from the one on its right — the\n"
    "score falls as the two tracks disagree, and drops to zero where either\n"
    "tracker reports the point as lost.\n\n"
    "It is per keypoint because that is what the fill produces, and because\n"
    "one lost point is what makes a frame worth revisiting — an average over\n"
    "the row would bury it. 'Lowest fill confidence' below ranks frames by\n"
    "their worst point, so it finds those frames without you reading down\n"
    "every column."
)

#: Thumbnails are enough to score frames for suggestions — DeepLabCut clusters
#: at ~30 px wide. Far cheaper to decode than the fill backends' input.
SUGGEST_MAX_SIDE = 64

#: Smallest share of a video the suggestion spin box offers. A tenth of a
#: percent is 60 frames of a 60k-frame recording — already more than anyone
#: labels by hand — and a floor keeps the box from resolving to zero frames.
MIN_SUGGEST_PERCENT = 0.1

#: Each method names the workflow it suits, because they are NOT
#: interchangeable — see the module docstring of ``pose_suggest``. Ordered by
#: **when in the workflow they apply**: the first three need nothing but the
#: video and so are where a new project starts; ``uncertain`` comes last
#: because it can only rank what a fill has already scored.
_SUGGEST_METHODS = (
    (
        "uniform",
        "Evenly spaced  (before fill)",
        "Equally spaced frames, no video scan — instant.\nHard to beat on a short clip of one behaviour.",
    ),
    (
        "motion",
        "Biggest pixel change  (before fill)",
        "Frames whose pixels differ most from the frame before —\n"
        "where the animal moves fastest. Fast motion is where optical\n"
        "flow loses the point and where a spline cuts the corner.",
    ),
    (
        "diverse",
        "Most different frames  (before fill)",
        "Groups the frames by how they look and takes one per group\n"
        "(DeepLabCut's k-means), so the picks are as unlike each\n"
        "other as possible.",
    ),
    (
        "detection_gaps",
        "Where the detector saw nothing  (after detect)",
        "Frames furthest from any detection — occlusion, blur, the animal\n"
        "facing away. A marker detector is not uncertain, it is absent, so\n"
        "its failures are a set of frames rather than a low score, and the\n"
        "middle of the longest blind stretch is where a fill has least to\n"
        "go on.\n\nNeeds a detector run.",
    ),
    (
        "uncertain",
        "Lowest fill confidence  (after fill)",
        "Frames whose *worst* keypoint the last fill scored lowest,\n"
        "where tracking forwards and backwards disagreed most — the\n"
        "ones worth correcting. One lost point is reason enough, so\n"
        "the other keypoints on the frame cannot vote it down.\n"
        "The backends are frozen, so extra labels reset drift rather\n"
        "than teach anything.\n\n"
        "Only frames the fill covered, i.e. between your first and\n"
        "last label — past those there is nothing to correct.\n\n"
        "Needs a fill to have run.",
    ),
)

#: Columns of the Detect tab's assignment table. "Label" carries the detector's
#: own preview — a colour swatch, or the tag itself rendered — as its icon.
#:
#: The last column is **"Set by"**, deliberately *not* "Source": the points table
#: already has a Source column meaning where a coordinate came from
#: (Human/Detected/Fill), and this one means something entirely different —
#: whether the *meaning* of a label was proposed or typed. Two columns of the
#: same name in one dialog, saying different things, is the confusion.
_ASSIGNMENT_COLUMNS = ("Label", "Individual", "Keypoint", "Matched on", "Set by")

#: How :data:`~ethograph.gui.pose_annotate.LEARNED` / ``MANUAL`` read in the
#: "Set by" column — the store's vocabulary is for the sidecar, not the user.
_ASSIGNMENT_SOURCE_LABELS = {LEARNED: "learning", MANUAL: "you"}

_ASSIGNMENT_TOOLTIPS = (
    "What the detector produces — a decoded tag ID.\nThe icon is that tag itself, rendered.",
    "Which individual this label lands on.",
    "Which keypoint of that individual this label lands on.",
    "How many of your labelled frames agreed on this target when it was learned.\n"
    "Two is the minimum — the nearest detection to a single click is always\nsomething.",
    "Where this row's meaning came from — NOT where a coordinate came from\n"
    "(that is the points table's Source column).\n\n"
    "learning — proposed by 'Learn from labels'; a later re-learn may change it.\n"
    "you — you picked it, so no re-learn will ever overwrite it.",
)

#: Where a detector run scans.
_RANGE_WHOLE = "whole"
_RANGE_LABELLED = "labelled"
_RANGE_FILL = "fill"

_DETECT_RANGES = (
    (_RANGE_WHOLE, "The whole video", "Every frame. What you want once the settings are right."),
    (
        _RANGE_LABELLED,
        "Your labelled span",
        "First to last labelled frame — the stretch a fill would cover anyway.\n"
        "Use it to try the settings out cheaply.",
    ),
    (
        _RANGE_FILL,
        "The current fill range",
        "Exactly the frames the last fill covers, to see where detection\n"
        "improves on it. Falls back to the whole video with no fill loaded.",
    ),
)

#: How long the detector preview waits before redrawing. Dragging the playhead
#: emits a frame change per tick and each redraw decodes a frame, so the point
#: is to coalesce a scrub into one — short enough to still feel immediate.
PREVIEW_DEBOUNCE_MS = 120

#: Canvas marker legend. The detected style is only named once a detector has
#: run — a third symbol nobody can see on screen is noise.
_LEGEND_LABEL_AND_FILL = '<span style="opacity:0.7;">●&nbsp;label&nbsp;&nbsp;&nbsp;○&nbsp;prediction</span>'
_LEGEND_WITH_DETECTIONS = (
    '<span style="opacity:0.7;">●&nbsp;label&nbsp;&nbsp;&nbsp;◉&nbsp;detected&nbsp;&nbsp;&nbsp;○&nbsp;filled</span>'
)

#: What Loop mode does after each click — the "Between clicks" dropdown.
AFTER_CLICK_FRAME = "frame"
AFTER_CLICK_SUGGESTION = "suggestion"
AFTER_CLICK_STAY = "stay"

_AFTER_CLICK_CHOICES = (
    (
        AFTER_CLICK_FRAME,
        "Jump +1 frame",
        "Step to the very next frame after each click — a dense sweep of one\nkeypoint through a short clip.",
    ),
    (
        AFTER_CLICK_SUGGESTION,
        "Jump to next suggested frame",
        "Follow the list from 'Which frames to label' below, so each click lands\n"
        "on a frame that was chosen deliberately rather than on the near-identical\n"
        "neighbour. Suggest some frames first.",
    ),
    (
        AFTER_CLICK_STAY,
        "Stay on this frame",
        "Never move the playhead. Navigate yourself with ← / → (single frames),\n"
        "N (next suggested frame) or the points table.",
    ),
)


class GroupedHeaderView(FilterHeaderView):
    """Two-row horizontal header: a group name spanning each keypoint's columns.

    ``QHeaderView`` has no multi-level support, and repeating the keypoint name
    in every column label ("beak x", "beak y", "beak conf") is what made the
    table so wide. Each of a group's sections therefore paints the *same* group
    name across their union rect and only its own sub-label beneath its own
    share — the parts join into one centred label, and painting it repeatedly is
    idempotent, so it survives a partial repaint of any one section.

    The first *fixed_columns* sections are ungrouped, painted normally, and are
    the ones that carry the inherited filter funnels.
    """

    #: Slack around a group name, so ResizeToContents never elides it.
    PADDING = 12

    def __init__(self, fixed_columns: int, sub_labels: Sequence[str], parent=None):
        super().__init__(parent=parent)
        self._fixed = int(fixed_columns)
        #: The label under each column of a group; its length is the group span.
        self._sub_labels = tuple(sub_labels)
        self._groups: list[str] = []
        self._brushes: list[QBrush] = []
        self.setSectionsClickable(True)
        self.setHighlightSections(False)

    @property
    def _span(self) -> int:
        return len(self._sub_labels)

    def groups(self) -> list[str]:
        """The group name over each column group, left to right."""
        return list(self._groups)

    def set_groups(self, groups: list[str], brushes: list[QBrush]) -> None:
        self._groups = list(groups)
        self._brushes = list(brushes)
        self.viewport().update()

    def _group_index(self, section: int) -> int | None:
        """Which group a section belongs to, or ``None`` for a fixed column."""
        if section < self._fixed:
            return None
        index = (section - self._fixed) // self._span
        return index if 0 <= index < len(self._groups) else None

    def sectionSizeFromContents(self, section: int):
        size = super().sectionSizeFromContents(section)
        # The funnel is drawn over the section, so its zone has to be paid for
        # in width or ResizeToContents elides the label underneath it.
        if section in self.filterable:
            size.setWidth(size.width() + self.FILTER_ZONE_W)
        if not self._groups:
            return size
        size.setHeight(size.height() * 2)
        index = self._group_index(section)
        if index is not None:
            # Each column of the group carries its share of the name, so the
            # group's own sections between them are always wide enough for it.
            share = self.fontMetrics().horizontalAdvance(self._groups[index]) // self._span
            size.setWidth(max(size.width(), share + self.PADDING))
        return size

    def paintSection(self, painter, rect, section: int) -> None:
        index = self._group_index(section)
        if index is None:
            super().paintSection(painter, rect, section)  # draws the filter funnel too
            return
        painter.save()
        super().paintSection(painter, rect, section)  # background; the label is empty
        half = rect.height() // 2
        first = self._fixed + self._span * index
        span = QRect(
            self.sectionViewportPosition(first),
            rect.top(),
            sum(self.sectionSize(first + offset) for offset in range(self._span)),
            half,
        )
        painter.setPen(QPen(self._brushes[index].color()))
        painter.drawText(span, Qt.AlignCenter, self._groups[index])
        painter.setPen(QPen(self.palette().color(QPalette.ButtonText)))
        painter.drawText(
            QRect(rect.left(), rect.top() + half, rect.width(), half),
            Qt.AlignCenter,
            self._sub_labels[section - first],
        )
        painter.restore()


class PointTableModel(QAbstractTableModel):
    """A :class:`KeypointStore` as rows of ``(frame, individual)``.

    Columns are ``Frame | Individual | Source`` then an ``x``/``y``/``conf``
    triple per keypoint. Values are read from the store on demand rather than
    copied into cells, for two reasons: once a fill exists there is a row for
    *every* frame it covers, which no item-based table can hold; and a view that
    reads the store cannot disagree with it, which the previous diffing item
    table could.

    Provenance is shown twice over. The ``Source`` cell states whether the row
    is the user's or the backend's — one human point anywhere in the row is
    enough, since correcting a single keypoint is what makes a frame yours — and
    each keypoint's triple is dimmed wherever the coordinate came from the fill,
    so a mixed row still reads correctly.
    """

    def __init__(self, store: KeypointStore, parent=None):
        super().__init__(parent)
        self._store = store
        self._rows: list[tuple[int, str]] = []
        self._columns: list[str] = []
        self._row_of: dict[tuple[int, str], int] = {}
        #: ``frame -> (first row, last row)``; rows of a frame are contiguous, so
        #: a drag can repaint one frame instead of the whole video.
        self._frame_rows: dict[int, tuple[int, int]] = {}
        #: One frame's ``(positions, human mask, detected mask, confidence)``,
        #: so a row costs one lookup rather than one per cell.
        self._cache: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None = None
        #: Set while colour means *individual*: the Individual cell then carries
        #: the colour, since the keypoint column headers no longer can.
        self._individual_brush: Callable[[str], QBrush] | None = None

    def set_individual_brush(self, brush: Callable[[str], QBrush] | None) -> None:
        """Colour the Individual column, or ``None`` to leave it on the palette."""
        self._individual_brush = brush
        self.refresh_all()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    @property
    def rows(self) -> list[tuple[int, str]]:
        return self._rows

    @property
    def keypoint_columns(self) -> list[str]:
        return self._columns

    def set_layout(self, rows: list[tuple[int, str]], columns: list[str]) -> None:
        """Replace the row keys and the keypoints carrying an ``x``/``y`` pair."""
        self.beginResetModel()
        self._rows = list(rows)
        self._columns = list(columns)
        self._row_of = {key: index for index, key in enumerate(self._rows)}
        self._frame_rows = {}
        for index, (frame, _individual) in enumerate(self._rows):
            first, _last = self._frame_rows.get(frame, (index, index))
            self._frame_rows[frame] = (first, index)
        self._cache = None
        self.endResetModel()

    def row_of(self, key: tuple[int, str]) -> int | None:
        return self._row_of.get(key)

    def key_at(self, row: int) -> tuple[int, str] | None:
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def keypoint_at(self, column: int) -> tuple[str, str] | None:
        """``(keypoint, axis)`` a column belongs to, or ``None`` for a fixed one.

        *axis* is one of :data:`_KEYPOINT_AXES` — ``"x"``, ``"y"`` or ``"conf"``.
        """
        offset = column - len(_FIXED_COLUMNS)
        index = offset // COLUMNS_PER_KEYPOINT
        if offset < 0 or index >= len(self._columns):
            return None
        return self._columns[index], _KEYPOINT_AXES[offset % COLUMNS_PER_KEYPOINT]

    def refresh_frame(self, frame: int | None) -> None:
        """Repaint one frame's rows — what a drag or a single placement changes."""
        self._cache = None
        span = self._frame_rows.get(int(frame)) if frame is not None else None
        if span is None:
            return
        first, last = span
        self.dataChanged.emit(self.index(first, 0), self.index(last, self.columnCount() - 1))

    def refresh_all(self) -> None:
        """Repaint everything — after a fill, which rewrites every row."""
        self._cache = None
        if self._rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self._rows) - 1, self.columnCount() - 1))

    # ------------------------------------------------------------------
    # Qt model interface
    # ------------------------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(_FIXED_COLUMNS) + COLUMNS_PER_KEYPOINT * len(self._columns)

    def _frame_cache(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        if self._cache is None or self._cache[0] != frame:
            self._cache = (
                frame,
                self._store.positions(frame),
                self._store.human_mask(frame),
                self._store.detected_mask(frame),
                self._store.confidence_at(frame),
            )
        return self._cache[1:]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        frame, individual = self._rows[index.row()]
        column = index.column()

        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter) if column >= len(_FIXED_COLUMNS) else None

        if column == FRAME_COLUMN:
            if role == Qt.DisplayRole:
                return str(frame)
            if role == SORT_ROLE:
                return float(frame)
            return None
        if column == INDIVIDUAL_COLUMN:
            if role == Qt.ForegroundRole and self._individual_brush is not None:
                return self._individual_brush(individual)
            return individual if role == Qt.DisplayRole else None
        if column == SOURCE_COLUMN:
            return self._source_data(frame, individual, role)
        return self._point_data(frame, individual, column, role)

    def _source_data(self, frame: int, individual: str, role):
        index = self._store.individual_index(individual)
        positions, human_mask, detected_mask, _confidence = self._frame_cache(frame)
        human, detected = human_mask[index], detected_mask[index]
        # An empty row says nothing rather than "Fill": with the predictions
        # deleted there is no source to name.
        empty = not np.any(~np.isnan(positions[index][:, 0]))
        if role == Qt.DisplayRole:
            if human.any():
                return HUMAN_SOURCE
            if empty:
                return ""
            return DETECTED_SOURCE if detected.any() else FILL_SOURCE
        if role == Qt.ForegroundRole and not human.any():
            return self._dim_brush()
        if role == Qt.ToolTipRole:
            total = len(self._store.keypoints_for(individual))
            hand = f"{int(human.sum())} of {total} keypoints placed by hand on frame {frame}."
            return hand if not detected.any() else f"{hand}\n{int(detected.sum())} found by the detector."
        return None

    def _point_data(self, frame: int, individual: str, column: int, role):
        """One keypoint's ``x``, ``y`` or ``conf`` cell.

        The three share a dimming rule — a predicted coordinate and the score
        that judges it are both the backend's, so the whole triple reads as one
        thing. The store composes the score in the same precedence as the
        positions (see ``confidence_at``), so a hand-placed point shows ``1.00``
        and a detected one the detector's own quality rather than the fill's.
        """
        if role not in (Qt.DisplayRole, Qt.ForegroundRole, SORT_ROLE):
            return None
        target = self.keypoint_at(column)
        if target is None:
            return None
        keypoint, axis = target
        i, k = self._store.individual_index(individual), self._store.keypoint_index(keypoint)
        positions, human, _detected, confidence = self._frame_cache(frame)
        if axis == "conf":
            # No position means nothing to score, whatever the fill array still
            # holds: a deleted point must not keep a confidence behind it.
            value = np.nan if np.isnan(positions[i, k, 0]) or confidence is None else confidence[i, k]
            digits = 2
        else:
            value = positions[i, k, _KEYPOINT_AXES.index(axis)]
            digits = 1
        if np.isnan(value):
            return None
        if role == Qt.ForegroundRole:
            return None if human[i, k] else self._dim_brush()
        return f"{value:.{digits}f}" if role == Qt.DisplayRole else float(value)

    @staticmethod
    def _dim_brush() -> QBrush:
        """The palette's disabled text colour — a predicted value, not a label."""
        return QBrush(QApplication.palette().color(QPalette.Disabled, QPalette.Text))

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):
        if orientation != Qt.Horizontal:
            return None
        fixed = section < len(_FIXED_COLUMNS)
        if role == Qt.DisplayRole:
            # The keypoint labels stay empty: GroupedHeaderView paints the name
            # over each pair itself, and repeating it in both would double it.
            return _FIXED_COLUMNS[section] if fixed else ""
        if role == Qt.ToolTipRole:
            if fixed:
                return _FIXED_COLUMN_TOOLTIPS[section]
            target = self.keypoint_at(section)
            if target is None:
                return None
            keypoint, axis = target
            if axis != "conf":
                return f"{keypoint} {axis}"
            return f"{keypoint} — fill confidence\n\n{_CONFIDENCE_TOOLTIP}"
        return None


class PoseLabellingDialog(QDialog):
    """Individual/keypoint tree, canvas labelling, backend choice, fill and export."""

    def __init__(self, data_widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keypoint labelling")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)

        self._data_widget = data_widget
        self.app_state = data_widget.app_state
        #: The main window. Key events land there whenever the user is looking
        #: at the video rather than at this dialog, so it is a key-filter target.
        self._shell = data_widget.shell
        self._view = self._shell.video_area.primary
        self._mode: KeypointLabelMode | None = None
        #: The calibration click mode, attached only while the Calibrate tab is
        #: open. Held apart from ``_mode``: `_lock_wanted`/`_apply_lock` govern
        #: the labelling mode alone, and the two are never attached together.
        self._calib_mode: CalibrationClickMode | None = None
        #: The labelling mode the Calibrate tab suspended — ``(mode, keypoint,
        #: individual)`` — restored when the calibration mode detaches.
        self._resume_label_mode: tuple[str, str | None, str | None] | None = None
        #: Guards the calibration table's ``itemChanged`` against its own
        #: rebuild churn, like the trials table's ``_building``.
        self._calib_building = False
        #: The coordinate space the USER picked, surviving the combo being
        #: forced back to pixels while the calibration is transiently invalid
        #: (a half-edited cell, a landmark being replaced) — restored the
        #: moment the fit is usable again, so "cm" is never silently lost.
        self._space_choice = "pixels"
        #: True while `_refresh_space_combo` sets the combo programmatically,
        #: so only the user's own changes update `_space_choice`.
        self._space_syncing = False
        #: Frames proposed by pose_suggest, and where the user is within them.
        self._suggestions: list[int] = []
        self._suggestion_index = 0
        #: What the table's rows and columns were built from — recomputing the
        #: layout on every drag is what this avoids.
        self._table_signature: tuple | None = None
        #: The refined backend, kept between fills so its fit — minutes of GPU —
        #: is paid once, together with what it was built for.
        self._refined_backend = None
        self._refined_built_for: tuple | None = None
        #: The video the kept fit belongs to — a fit is never valid for another.
        self._refined_video: str | None = None
        #: Tab / Shift+Tab / Shift+H / N, which have to be shortcuts — the tree and
        #: the table swallow them otherwise. See :meth:`_bind_shortcuts`.
        self._shortcuts: list[QShortcut] = []
        #: Whether the pose overlay currently carries our fill — see
        #: :meth:`_push_pose_override`, which hands the canvas to the anchor
        #: overlay while a mode is armed.
        self._override_pushed = False
        #: The point detector, kept between runs because the live preview asks
        #: for it on every redraw.
        self._detector = None
        self._detector_built_for: tuple | None = None
        #: The last run *before* the quality threshold, so the threshold can be
        #: returned without decoding the video again.
        self._raw_detections: tuple[dict, dict, dict] | None = None
        #: A frame source held open for the detector preview — opening a PyAV
        #: container costs far more than the detection, and this runs per scrub.
        self._preview_frames = None
        self._preview_frames_for: tuple | None = None
        #: Whether head direction has already been offered for the current
        #: detections. Offering it once when a run first produces oriented
        #: markers is helpful; re-ticking it on every later refresh would keep
        #: overruling someone who deliberately turned it off.
        self._head_direction_offered = False

        self.store = self._load_store()
        #: The video the current store was loaded for; a change means a new clip.
        self._store_video = self._video_path()
        self._build_ui()
        self._load_detections()
        self._rebuild_tree()
        self._refresh_point_table()
        # Colour is app-wide state, so the dialog opens in whatever mode the
        # pose overlay is already in rather than in its own default.
        self.apply_color_by()

        self._install_key_filter(True)
        self.app_state.current_frame_changed.connect(self._on_frame_changed)
        self.app_state.video_path_changed.connect(self._on_video_changed)

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------

    def _video_path(self) -> str | None:
        return self.app_state.video_path

    def _fps(self) -> float | None:
        fps = self._view.fps or self.app_state.video_fps
        return float(fps) if fps else None

    def _n_frames(self) -> int:
        return self._view.n_frames or int(self.app_state.num_frames or 0)

    def _load_store(self) -> KeypointStore:
        """Reuse the sidecar next to the video when one exists."""
        video = self._video_path()
        n_frames = self._n_frames()
        if video:
            path = sidecar_path(video)
            if path.exists():
                # A damaged sidecar is a runtime condition, not a bug: warn and
                # start empty rather than making the dialog un-openable. The
                # file is left alone so the user can inspect or repair it.
                try:
                    store = KeypointStore.load(path)
                except (KeypointStoreError, ValueError, KeyError, OSError) as e:
                    notify(f"Could not read {path.name}: {e}. Starting with no labels.", "warning")
                else:
                    store.n_frames = n_frames or store.n_frames
                    return store
        store = KeypointStore(
            keypoint_names=list(self.app_state.keypoints or []),
            n_frames=n_frames,
            individual_names=list(self.app_state.labelling_individuals or [DEFAULT_INDIVIDUAL]),
        )
        template = self._last_saved_sidecar()
        if template is not None:
            seeded = store.seed_static_from(template)
            if seeded:
                notify(f"Seeded {seeded} static keypoint(s) from the last labelled clip.", "info")
        return store

    def _last_saved_sidecar(self) -> KeypointStore | None:
        """What a fresh clip's static keypoints are seeded from.

        The sidecar this GUI session last wrote — the previous clip of this
        camera — else the camera's template beside the videos
        (:func:`template_path`), written whenever a clip with static keypoints
        is saved, so the seed survives a restart. Many short clips of one
        fixed camera share their static keypoints; re-clicking four corners in
        every clip is exactly the work "static" exists to remove.
        """
        candidates = [_LAST_SAVED_SIDECAR.get("path")]
        video = self._video_path()
        if video:
            candidates.append(str(template_path(video)))
        for path in candidates:
            if path is None or not os.path.isfile(path):
                continue
            try:
                return KeypointStore.load(path)
            except (KeypointStoreError, ValueError, KeyError, OSError):
                continue
        return None

    def _write_template(self, video: str) -> None:
        """Keep the camera's template current: the schema and the static keypoints only."""
        if not self.store.static_keypoints:
            return
        template = KeypointStore(
            keypoint_names=list(self.store.keypoint_names),
            n_frames=1,
            individual_names=list(self.store.individual_names),
            shared_keypoints=self.store.shared_keypoints,
            keypoint_sets={k: list(v) for k, v in self.store.keypoint_sets.items()},
        )
        template.seed_static_from(self.store)
        path = template_path(video)
        path.parent.mkdir(parents=True, exist_ok=True)
        template.save(path)

    def _save_store(self) -> None:
        video = self._video_path()
        if video:
            path = sidecar_path(video)
            self.store.save(path)
            _LAST_SAVED_SIDECAR["path"] = str(path)
            self._write_template(video)
            if hasattr(self, "clips_label"):
                self._refresh_clip_counter()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """One tab per stage, in the order the work happens.

        One column of every group at once made the dialog taller than most
        screens; the split is by *stage*, so nothing a stage needs lives on
        another tab.
        """
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._label_page = self._build_label_page()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_schema_page(), "Define keypoints")
        self.tabs.addTab(self._label_page, "Label && Edit")  # && escapes the mnemonic
        # Between labelling and filling, because that is where it sits in the
        # pipeline: it produces observations, which the fill then bridges.
        self._detect_page = self._build_detect_page()
        self.tabs.addTab(self._detect_page, "Detect")
        # Before the export, which is what a calibration changes: it maps the
        # exported positions into the user's own cm frame.
        self._calibrate_page = self._build_calibrate_page()
        self.tabs.addTab(self._calibrate_page, "Calibrate")
        self.tabs.addTab(self._build_output_page(), "Fill and export")
        # Opening the tab IS the intent to label, so arm Sequential rather than
        # making the first click a mode choice.
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs, stretch=1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        # Taller by default: the points table is the thing you read most, and a
        # short dialog turns it into a 4-row peephole. Clamped to the screen so
        # it never opens taller than the display it lands on.
        width, height = 480, 800
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, available.width() - 80)
            height = min(height, available.height() - 80)
        self.resize(width, height)

    def _build_label_page(self) -> QWidget:
        """Labelling itself: the modes, the points table and frame suggestions."""
        page = QWidget()
        box = QVBoxLayout(page)
        box.addWidget(self._build_mode_controls())
        box.addLayout(self._build_clips_row())
        box.addWidget(self._build_table_group(), stretch=1)

        # Approve beside Clear, because they are the two bulk verdicts on a run:
        # take all of it, or none of it. Both are confirmed and neither is
        # undoable, so they belong on the same row rather than a step apart.
        clear_row = QHBoxLayout()
        clear_row.addStretch()

        self.approve_detections_btn = QPushButton("Approve all detections…")
        self.approve_detections_btn.setToolTip(
            "Turn every detected point into a label, on every frame — the same\n"
            "as 'Pin detections as labels' on a row, for the whole run.\n\n"
            "Use it once a run looks right: labels are what the fill treats as\n"
            "ground truth and what an export calls human-placed.\n\n"
            "Points you already labelled are left alone."
        )
        self.approve_detections_btn.clicked.connect(self._on_approve_all_detections)
        clear_row.addWidget(self.approve_detections_btn)

        self.approve_fill_btn = QPushButton("Approve all filled points…")
        self.approve_fill_btn.setToolTip(
            "Turn everything the fill produced into labels, across its whole\n"
            "span — the same as 'Pin filled points as labels' on a row.\n\n"
            "Detected points inside the span are pinned too: what you are\n"
            "agreeing with is what is on screen, and the screen shows both.\n\n"
            "Points you already labelled are left alone."
        )
        self.approve_fill_btn.clicked.connect(self._on_approve_all_fill)
        clear_row.addWidget(self.approve_fill_btn)

        clear_all_btn = QPushButton("Clear all labels…")
        clear_all_btn.setToolTip("Delete every labelled point in the table. Filled points are not affected.")
        clear_all_btn.clicked.connect(self._on_clear_all_labels)
        clear_row.addWidget(clear_all_btn)
        box.addLayout(clear_row)
        self._refresh_bulk_approve_buttons()

        box.addWidget(self._build_suggest_group())
        return page

    def _build_output_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.addWidget(self._build_fill_group())
        box.addWidget(self._build_export_group())
        box.addStretch()
        return page

    # ------------------------------------------------------------------
    # Calibrate: pixel → cm from clicked landmarks
    # ------------------------------------------------------------------

    def _build_calibrate_page(self) -> QWidget:
        """Landmark table + clicking instructions; the tab arms its own mode.

        Landmarks are **not keypoints**: they live in the store's
        :class:`~ethograph.gui.pose_annotate.CalibrationTable`, so they never
        join the fill span, the individual axis or the exported dims — and they
        cannot collide with the detector's ``corner_N`` tag keypoints. Opening
        the tab attaches a :class:`~ethograph.gui.pose_edit_mixin.CalibrationClickMode`
        in place of the labelling mode (see :meth:`_on_tab_changed`), so a click
        on the video places the selected landmark; leaving hands the canvas back.
        """
        page = QWidget()
        box = QVBoxLayout(page)

        intro = QLabel(
            "Give each fixed physical landmark its real-world x/y (in cm), then "
            "click it on the video on a few frames — the clicks are averaged, "
            "assuming the camera does not move. With "
            f"{MIN_CALIBRATION_LANDMARKS} landmarks ready the export can "
            "produce cm instead of pixels (3 fit an affine; 4 or more fit a "
            "homography, which also corrects an angled camera)."
        )
        intro.setWordWrap(True)
        box.addWidget(intro)

        group = QGroupBox("Landmarks")
        group_box = QVBoxLayout(group)

        self.calib_table = QTableWidget(0, 5)
        self.calib_table.setHorizontalHeaderLabels(["Landmark", "x (cm)", "y (cm)", "Clicks", "Mean (px)"])
        self.calib_table.horizontalHeader().setStretchLastSection(True)
        self.calib_table.verticalHeader().setVisible(False)
        self.calib_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.calib_table.setSelectionMode(QTableWidget.SingleSelection)
        self.calib_table.itemChanged.connect(self._on_calib_item_changed)
        self.calib_table.currentCellChanged.connect(self._on_calib_row_selected)
        group_box.addWidget(self.calib_table)

        buttons = QHBoxLayout()
        add_btn = QPushButton("Add landmark…")
        add_btn.clicked.connect(self._on_add_landmark)
        buttons.addWidget(add_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.setToolTip("Remove the selected landmark, its cm coordinates and its clicks.")
        remove_btn.clicked.connect(self._on_remove_landmark)
        buttons.addWidget(remove_btn)
        clear_btn = QPushButton("Clear clicks")
        clear_btn.setToolTip("Drop the selected landmark's clicks, keeping its cm coordinates.")
        clear_btn.clicked.connect(self._on_clear_landmark_clicks)
        buttons.addWidget(clear_btn)
        buttons.addStretch()
        load_btn = QPushButton("Load coordinates…")
        load_btn.setToolTip(
            "Read landmark cm coordinates from a table, instead of typing them\n"
            "per session. Two layouts work:\n\n"
            "  name, x, y            — one row per landmark\n"
            "  session, <name>_x, <name>_y   — one row per session; you pick the row\n\n"
            "A z column is ignored: a single camera calibrates a plane, so use\n"
            "landmarks roughly level with where the animal moves. Existing\n"
            "landmarks keep their clicks — re-clicking per session is what\n"
            "absorbs camera drift."
        )
        load_btn.clicked.connect(self._on_load_world_coordinates)
        buttons.addWidget(load_btn)
        group_box.addLayout(buttons)

        self.calib_status = QLabel("")
        self.calib_status.setWordWrap(True)
        group_box.addWidget(self.calib_status)

        box.addWidget(group)

        # The clicked frames, mirrored from the points table: one row per
        # frame, one column per landmark, click a row to go there.
        frames_group = QGroupBox("Clicked frames")
        frames_box = QVBoxLayout(frames_group)
        self.calib_frames_table = QTableWidget(0, 1)
        self.calib_frames_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.calib_frames_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.calib_frames_table.setSelectionMode(QTableWidget.SingleSelection)
        self.calib_frames_table.verticalHeader().setVisible(False)
        self.calib_frames_table.horizontalHeader().setStretchLastSection(True)
        self.calib_frames_table.setToolTip(
            "Every frame carrying a calibration click. Clicking a row seeks the\n"
            "video there; clicking a landmark's cell also makes it the one the\n"
            "next canvas click places. Right-click removes clicks."
        )
        self.calib_frames_table.cellClicked.connect(self._on_calib_frame_clicked)
        self.calib_frames_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.calib_frames_table.customContextMenuRequested.connect(self._on_calib_frames_menu)
        frames_box.addWidget(self.calib_frames_table)
        box.addWidget(frames_group, stretch=1)

        self._refresh_calibration_table()
        return page

    def _enter_calibrate_mode(self) -> None:
        """Swap the canvas from labelling to landmark clicking."""
        if self._calib_mode is not None:
            return
        if self._view.scene() is None:
            return  # the table is still editable; clicking needs a loaded frame
        if self._mode is not None:
            self._resume_label_mode = (self._mode.mode, self._mode.active_keypoint, self._mode.active_individual)
            self._detach_mode()
        self._calib_mode = CalibrationClickMode(
            self._view,
            self.store.calibration,
            on_changed=self._after_calibration_changed,
            point_size=float(self.app_state.labelling_point_size),
        )
        self._calib_mode.set_frame(int(self.app_state.current_frame or 0))
        self._sync_calib_selection()

    def _exit_calibrate_mode(self, restore: bool = True) -> None:
        """Detach the calibration mode; *restore* re-arms the suspended labelling.

        ``restore=False`` is for callers about to install a mode of their own
        (:meth:`set_interaction_mode`) or tearing the canvas down entirely.
        """
        resume, self._resume_label_mode = self._resume_label_mode, None
        if self._calib_mode is None:
            return
        self._calib_mode.detach()
        self._calib_mode = None
        if restore and resume is not None and self._can_label(quiet=True):
            mode, keypoint, individual = resume
            self._attach_mode(mode)
            self._sync_mode_buttons()
            if individual not in self.store.individual_names:
                individual = None
            if keypoint is not None and self.store.has_keypoint(keypoint, individual):
                self._mode.set_active(keypoint, individual)
            self._apply_lock()

    def _after_calibration_changed(self) -> None:
        """Everything a landmark click or table edit touches, in one place."""
        self._save_store()  # clicks and cm coordinates are user intent
        self._refresh_calibration_table()
        self._refresh_space_combo()
        if self._calib_mode is not None:
            self._calib_mode.refresh()
            self._sync_calib_selection()

    def _refresh_calibration_table(self) -> None:
        table = getattr(self, "calib_table", None)
        if table is None:
            return
        self._calib_building = True
        try:
            landmarks = self.store.calibration.landmarks
            table.setRowCount(len(landmarks))
            for row, landmark in enumerate(landmarks):
                world = landmark.world_xy
                mean = landmark.mean_pixel()
                cells = [
                    QTableWidgetItem(landmark.name),
                    QTableWidgetItem("" if world is None else f"{world[0]:g}"),
                    QTableWidgetItem("" if world is None else f"{world[1]:g}"),
                    QTableWidgetItem(str(len(landmark.clicks))),
                    QTableWidgetItem("—" if mean is None else f"({mean[0]:.1f}, {mean[1]:.1f})"),
                ]
                for column, item in enumerate(cells):
                    if column >= 3:  # clicks + mean are read-only readouts
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table.setItem(row, column, item)
        finally:
            self._calib_building = False
        self._refresh_calibration_status()
        self._refresh_calibration_frames()

    def _refresh_calibration_frames(self) -> None:
        """Rebuild the clicked-frames table: one row per frame, one column per landmark."""
        table = getattr(self, "calib_frames_table", None)
        if table is None:
            return
        names = self.store.calibration.names()
        frames = sorted({frame for landmark in self.store.calibration for frame in landmark.clicks})
        table.setColumnCount(1 + len(names))
        table.setHorizontalHeaderLabels(["Frame", *names])
        table.setRowCount(len(frames))
        for row, frame in enumerate(frames):
            table.setItem(row, 0, QTableWidgetItem(str(frame)))
            for column, name in enumerate(names, start=1):
                click = self.store.calibration.get(name).clicks.get(frame)
                text = "" if click is None else f"{click[0]:.1f}, {click[1]:.1f}"
                table.setItem(row, column, QTableWidgetItem(text))
        self._select_calib_frame_row()

    def _select_calib_frame_row(self) -> None:
        """Highlight the playhead's row in the clicked-frames table, without seeking."""
        table = getattr(self, "calib_frames_table", None)
        if table is None:
            return
        frame = int(self.app_state.current_frame or 0)
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None and int(item.text()) == frame:
                table.selectRow(row)
                return
        table.clearSelection()

    def _on_calib_frame_clicked(self, row: int, column: int) -> None:
        """Seek to the clicked row's frame; a landmark cell also selects it."""
        item = self.calib_frames_table.item(row, 0)
        if item is None:
            return
        self._seek(int(item.text()))
        names = self.store.calibration.names()
        if column >= 1 and column - 1 < len(names) and self._calib_mode is not None:
            self._calib_mode.set_active(names[column - 1])
            self._sync_calib_selection()

    def _on_calib_frames_menu(self, position) -> None:
        table = self.calib_frames_table
        item = table.itemAt(position)
        if item is None:
            return
        frame_item = table.item(item.row(), 0)
        if frame_item is None:
            return
        frame = int(frame_item.text())
        names = self.store.calibration.names()
        menu = QMenu(table)
        column = item.column()
        if column >= 1 and column - 1 < len(names):
            name = names[column - 1]
            if frame in self.store.calibration.get(name).clicks:
                remove_one = menu.addAction(f"Remove {name}'s click on frame {frame}")
                remove_one.triggered.connect(
                    lambda: (self.store.calibration.remove_click(name, frame), self._after_calibration_changed())
                )
        remove_all = menu.addAction(f"Remove every click on frame {frame}")

        def _remove_frame() -> None:
            for landmark_name in names:
                self.store.calibration.remove_click(landmark_name, frame)
            self._after_calibration_changed()

        remove_all.triggered.connect(_remove_frame)
        menu.exec_(table.viewport().mapToGlobal(position))

    def _refresh_calibration_status(self) -> None:
        status = getattr(self, "calib_status", None)
        if status is None:
            return
        calibration = self.store.calibration
        ready = len(calibration.ready())
        if not len(calibration):
            status.setText("No landmarks yet — add them by hand or load a coordinates file.")
        elif ready >= MIN_CALIBRATION_LANDMARKS:
            fit_kind = "affine" if ready == MIN_CALIBRATION_LANDMARKS else "homography"
            status.setText(f"Ready: {ready} landmarks → {fit_kind} fit. The export now offers cm.")
        else:
            missing_world = sum(1 for lm in calibration if lm.world_xy is None)
            missing_clicks = sum(1 for lm in calibration if not lm.clicks)
            parts = []
            if missing_world:
                parts.append(f"{missing_world} without cm coordinates")
            if missing_clicks:
                parts.append(f"{missing_clicks} without a click")
            status.setText(f"{ready}/{MIN_CALIBRATION_LANDMARKS} landmarks ready ({', '.join(parts)}).")

    def _sync_calib_selection(self) -> None:
        """Point the table's selected row at the mode's active landmark."""
        if self._calib_mode is None or self._calib_mode.active_landmark is None:
            return
        names = self.store.calibration.names()
        if self._calib_mode.active_landmark not in names:
            return
        row = names.index(self._calib_mode.active_landmark)
        if self.calib_table.currentRow() != row:
            self._calib_building = True
            try:
                self.calib_table.setCurrentCell(row, 0)
            finally:
                self._calib_building = False

    def _on_calib_row_selected(self, row: int, *_args) -> None:
        if self._calib_building or self._calib_mode is None:
            return
        names = self.store.calibration.names()
        if 0 <= row < len(names):
            self._calib_mode.set_active(names[row])

    def _on_calib_item_changed(self, item: QTableWidgetItem) -> None:
        if self._calib_building:
            return
        landmarks = self.store.calibration.landmarks
        row, column = item.row(), item.column()
        if not 0 <= row < len(landmarks):
            return
        landmark = landmarks[row]
        text = item.text().strip()
        if column == 0:
            if text and text != landmark.name:
                try:
                    self.store.calibration.rename(landmark.name, text)
                except KeypointStoreError as e:
                    notify(str(e), "warning")
        elif column in (1, 2):
            x_text = (self.calib_table.item(row, 1).text() or "").strip()
            y_text = (self.calib_table.item(row, 2).text() or "").strip()
            # A blank half falls back to the value the landmark already has:
            # the table is rebuilt after every edit, so treating a lone x as
            # "no position" wiped the y the user had typed a moment before.
            old = landmark.world_xy
            try:
                x_val = float(x_text) if x_text else (old[0] if old else None)
                y_val = float(y_text) if y_text else (old[1] if old else None)
            except ValueError:
                notify("cm coordinates must be numbers.", "warning")
                x_val, y_val = old if old else (None, None)
            if x_val is None or y_val is None:
                self.store.calibration.set_world(landmark.name, None)
            else:
                self.store.calibration.set_world(landmark.name, (x_val, y_val))
        self._after_calibration_changed()

    def _on_add_landmark(self) -> None:
        name, ok = QInputDialog.getText(self, "Add landmark", "Landmark name:")
        name = name.strip()
        if not ok or not name:
            return
        try:
            self.store.calibration.add(name)
        except KeypointStoreError as e:
            notify(str(e), "warning")
            return
        self._after_calibration_changed()
        self.calib_table.setCurrentCell(self.calib_table.rowCount() - 1, 1)

    def _selected_landmark_name(self) -> str | None:
        names = self.store.calibration.names()
        row = self.calib_table.currentRow()
        return names[row] if 0 <= row < len(names) else None

    def _on_remove_landmark(self) -> None:
        name = self._selected_landmark_name()
        if name is None:
            return
        self.store.calibration.remove(name)
        self._after_calibration_changed()

    def _on_clear_landmark_clicks(self) -> None:
        name = self._selected_landmark_name()
        if name is None:
            return
        self.store.calibration.clear_clicks(name)
        self._after_calibration_changed()

    def _on_load_world_coordinates(self) -> None:
        """Fill the landmarks' cm coordinates from a table on disk.

        Existing landmarks keep their clicks — the world layout is the stable
        part, and re-clicking per session is what absorbs camera drift. New
        names become new rows.
        """
        path = browse_open_file(
            self,
            self.app_state,
            "Load landmark coordinates",
            "Tables (*.csv *.tsv *.txt *.xlsx);;All files (*)",
            preferred_dir=self.app_state.calibration_coords_path or None,
        )
        if not path:
            return
        try:
            loaded = load_world_coordinates(path)
            if isinstance(loaded, list):  # session-keyed file: ask which row
                session, ok = QInputDialog.getItem(
                    self, "Pick a session", "This file has one row per session:", loaded, 0, False
                )
                if not ok:
                    return
                loaded = load_world_coordinates(path, session=session)
        except KeypointStoreError as e:
            notify(str(e), "error")
            return
        self.app_state.calibration_coords_path = str(path)
        for name, world in loaded.items():
            if name not in self.store.calibration:
                self.store.calibration.add(name)
            self.store.calibration.set_world(name, world)
        self._after_calibration_changed()
        notify(f"Loaded cm coordinates for {len(loaded)} landmark(s) — now click them on the video.", "info")

    def _build_schema_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)

        self.shared_toggle = QCheckBox("Individuals share the same keypoints")
        self.shared_toggle.setToolTip(
            "On: every individual is an instance of one schema (SLEAP's skeleton).\n"
            "Off: each individual carries its own keypoints, so one animal can be\n"
            "labelled with keypoints another does not have. The keypoint buttons\n"
            "below then edit the selected individual's set."
        )
        self.shared_toggle.setChecked(self.store.shared_keypoints)
        self.shared_toggle.toggled.connect(self._on_shared_toggled)
        box.addWidget(self.shared_toggle)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Name", "This frame", "Static"])
        self.tree.headerItem().setToolTip(
            2,
            "A keypoint that does not move: an arena corner, a fixed landmark.\n"
            "Label it once, on any frame, and it is there on every frame —\n"
            "the fill leaves it alone, the export writes it everywhere, and\n"
            "the next clip of the same camera starts with it already placed.",
        )
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setRootIsDecorated(True)
        self.tree.setMinimumHeight(180)
        self.tree.header().setStretchLastSection(False)
        self.tree.setColumnWidth(0, 220)
        self.tree.currentItemChanged.connect(self._on_tree_item_changed)
        self.tree.itemChanged.connect(self._on_tree_static_toggled)
        box.addWidget(self.tree)

        individual_row = QHBoxLayout()
        add_individual = QPushButton("Add individual…")
        add_individual.setToolTip("Label a second animal with the same keypoints (SLEAP's 'add instance').")
        add_individual.clicked.connect(self._on_add_individual)
        individual_row.addWidget(add_individual)
        remove_individual = QPushButton("Remove individual")
        remove_individual.setToolTip(
            "Remove the selected individual, including the last one — labelling\n"
            "resumes once an individual is added back."
        )
        remove_individual.clicked.connect(self._on_remove_individual)
        individual_row.addWidget(remove_individual)
        box.addLayout(individual_row)

        keypoint_row = QHBoxLayout()
        self._add_keypoint_btn = QPushButton("Add keypoint…")
        self._add_keypoint_btn.clicked.connect(self._on_add_keypoint)
        keypoint_row.addWidget(self._add_keypoint_btn)
        self._remove_keypoint_btn = QPushButton("Remove keypoint")
        self._remove_keypoint_btn.clicked.connect(self._on_remove_keypoint)
        keypoint_row.addWidget(self._remove_keypoint_btn)
        box.addLayout(keypoint_row)

        colour_row = QHBoxLayout()
        colour_row.addWidget(QLabel("Colour by:"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItem("Keypoint", COLOR_BY_KEYPOINT)
        self.color_by_combo.addItem("Individual", COLOR_BY_INDIVIDUAL)
        self.color_by_combo.setToolTip(
            "Which axis colour tells apart, on the canvas and the pose overlay alike.\n\n"
            "Keypoint: one colour per body part, the same on every individual —\n"
            "the labelling default, since a click answers 'which body part is this?'.\n"
            "Individual: one colour per individual, shared by all its keypoints —\n"
            "for pulling two animals apart when they overlap.\n\n"
            "Either way the individual being labelled is drawn at full opacity and\n"
            "the others are dimmed."
        )
        index = self.color_by_combo.findData(self.color_by)
        self.color_by_combo.setCurrentIndex(index if index >= 0 else 0)
        self.color_by_combo.currentIndexChanged.connect(self._on_color_by_changed)
        colour_row.addWidget(self.color_by_combo)
        self._colour_btn = QPushButton("Colour…")
        self._colour_btn.setToolTip(
            "Pick the colour of the selected item, everywhere it is shown — canvas\n"
            "markers, this tree and the points table.\n\n"
            "It edits whichever palette 'Colour by' is drawing: the keypoint's\n"
            "colour, or the individual's. The choice is saved beside the labels in\n"
            "the sidecar, so both palettes survive switching between them."
        )
        self._colour_btn.clicked.connect(self._on_keypoint_color)
        colour_row.addWidget(self._colour_btn)
        self._reset_colours_btn = QPushButton("Reset colours")
        self._reset_colours_btn.setToolTip(
            "Hand every keypoint and individual back to the generated palette,\n"
            "which spreads the schema over distinguishable hues."
        )
        self._reset_colours_btn.clicked.connect(self._on_reset_keypoint_colors)
        colour_row.addWidget(self._reset_colours_btn)
        colour_row.addStretch()
        box.addLayout(colour_row)

        self._refresh_keypoint_hints()
        return page

    def _refresh_keypoint_hints(self) -> None:
        """Say who the keypoint buttons act on — everyone, or one individual."""
        if self.store.shared_keypoints:
            scope = "the schema every individual shares"
        else:
            scope = "the selected individual only"
        self._add_keypoint_btn.setToolTip(f"Add a keypoint to {scope}.")
        self._remove_keypoint_btn.setToolTip(f"Remove the selected keypoint from {scope}.")
        # Nothing to reset while every colour is still the generated one.
        self._reset_colours_btn.setEnabled(bool(self.store.keypoint_color or self.store.individual_color))

    def _build_clips_row(self) -> QHBoxLayout:
        """Where this clip sits among the camera's clips, and the way to the next one."""
        row = QHBoxLayout()
        self.clips_label = QLabel("")
        row.addWidget(self.clips_label)
        row.addStretch(1)
        self.next_clip_btn = QPushButton("Next clip without labels")
        self.next_clip_btn.setToolTip(
            "Jump to the next trial whose clip has no labels yet. The schema and the\n"
            "static keypoints carry over; each clip keeps its own labels."
        )
        self.next_clip_btn.setAutoDefault(False)
        self.next_clip_btn.setDefault(False)
        self.next_clip_btn.clicked.connect(self._on_next_unlabelled_clip)
        row.addWidget(self.next_clip_btn)
        QTimer.singleShot(0, self._refresh_clip_counter)
        return row

    def _build_mode_controls(self) -> QWidget:
        """One compact row — modes plus the target pickers — over the status chip.

        Everything shares a single row so adding the pickers costs no vertical
        space: the table below is what the tab is for.

        Editing needs no mode of its own: clicking an existing point always
        selects and drags it, ``Backspace`` deletes the selected point and
        ``Ctrl+Z`` undoes. **Lock** is the way to stop clicking from labelling
        without stopping the mode, which would take the anchor overlay with it.
        """
        widget = QWidget()
        box = QVBoxLayout(widget)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.sequential_btn = QPushButton("Sequential")
        self.sequential_btn.setCheckable(True)
        self.sequential_btn.setToolTip(
            "Label every keypoint on one frame. Each click places the active\n"
            "keypoint, then moves to the next one this individual still lacks,\n"
            "filling the table left to right; the playhead never moves on its own.\n\n"
            "Click an existing point to drag it. Tab cycles keypoints, 1-9 pick\n"
            "the individual, Backspace deletes the selected point, Ctrl+Z undoes,\n"
            "Shift+drag pans. Press the button again to stop."
        )
        self.sequential_btn.clicked.connect(
            lambda checked: self.set_interaction_mode(SEQUENTIAL_MODE if checked else None)
        )
        row.addWidget(self.sequential_btn)

        self.loop_btn = QPushButton("Loop")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setToolTip(
            "Sweep ONE keypoint across frames. Each click places it and then does\n"
            "whatever 'Between clicks' says — step a frame, jump to the next\n"
            "suggested frame, or stay put and let you navigate.\n\n"
            "This is what the fill backends want: each keypoint is interpolated\n"
            "over its own anchor set, so temporal density per keypoint matters.\n"
            "Press the button again to stop."
        )
        self.loop_btn.clicked.connect(lambda checked: self.set_interaction_mode(LOOP_MODE if checked else None))
        row.addWidget(self.loop_btn)

        # Sits with the mode buttons because it is the third answer to "what
        # does a click on the video do": label, sweep, or nothing at all.
        self.lock_check = QCheckBox("Lock")
        self.lock_check.setToolTip(
            "Look around without labelling: left-drag pans the video again and\n"
            "clicks no longer place, move or pin points.\n\n"
            "The labels stay on screen and the active keypoint is kept, so\n"
            "unticking carries on exactly where you were — unlike stopping the\n"
            "mode, which takes the anchor overlay with it. Backspace and Ctrl+Z\n"
            "still act on the selected point.\n\n"
            "The other tabs lock the pointer on their own, so a click while you\n"
            "are detecting or filling cannot drop a stray point. This tick is\n"
            "left alone by that, and applies again the moment you come back."
        )
        self.lock_check.toggled.connect(self._on_lock_toggled)
        row.addWidget(self.lock_check)

        self.individual_combo = QComboBox()
        self.individual_combo.setToolTip("Which individual the next click labels (1-9 also select it).")
        self.individual_combo.currentIndexChanged.connect(self._on_individual_combo_changed)
        row.addWidget(self.individual_combo, stretch=1)

        # Only Loop mode holds one keypoint still, so only Loop mode has a
        # keypoint to pick; Sequential advances through them on its own.
        self.keypoint_combo = QComboBox()
        self.keypoint_combo.setToolTip("Which keypoint to sweep across frames (Tab also cycles).")
        self.keypoint_combo.currentIndexChanged.connect(self._on_keypoint_combo_changed)
        self.keypoint_combo.hide()
        row.addWidget(self.keypoint_combo, stretch=1)

        box.addLayout(row)

        # Its own row: the mode row is already full, and this choice is the
        # whole substance of Loop mode. It governs approving too, so it is shown
        # once a fill exists as well as in Loop mode.
        self.after_click_row = QWidget()
        after_click = QHBoxLayout(self.after_click_row)
        after_click.setContentsMargins(0, 0, 0, 0)
        after_click.setSpacing(4)
        after_click.addWidget(QLabel("Then go to:"))
        self.after_click_combo = QComboBox()
        for key, label, tip in _AFTER_CLICK_CHOICES:
            self.after_click_combo.addItem(label, key)
            self.after_click_combo.setItemData(self.after_click_combo.count() - 1, tip, Qt.ToolTipRole)
        self.after_click_combo.setToolTip(
            "Where the playhead goes after each click in Loop mode, and after\nShift+H approves a frame."
        )
        after_click.addWidget(self.after_click_combo, stretch=1)
        self.after_click_row.hide()
        box.addWidget(self.after_click_row)

        # What the next click will place, in the canvas's own visual language:
        # the individual's marker glyph, and the keypoint in its marker colour.
        self.active_label = QLabel()
        self.active_label.setTextFormat(Qt.RichText)
        self.active_label.setStyleSheet(
            "QLabel { font-size: 14px; padding: 3px 8px; border-radius: 4px; background: rgba(127,127,127,0.18); }"
        )
        self.active_label.hide()

        # Says what the marker styles on the canvas mean. Only once something
        # unlabelled is on screen: with no predictions there is nothing to tell
        # apart.
        self.legend_label = QLabel(_LEGEND_LABEL_AND_FILL)
        self.legend_label.setTextFormat(Qt.RichText)
        self.legend_label.setToolTip(
            "Filled markers are your labels; hollow ones were not placed by you —\n"
            "drawn empty so you can see the pixels underneath and judge them.\n"
            "A hollow marker with a dot in it was read off THIS frame by the\n"
            "detector; an empty one was interpolated between other frames.\n\n"
            "Click a hollow marker to pin it as a label (it turns solid), or drag\n"
            "it to correct it first."
        )
        self.legend_label.hide()

        # Approving is the other half of reviewing a fill: the legend says which
        # markers are predictions, this accepts them. Shown on the same terms —
        # with no fill on screen there is nothing to approve.
        self.approve_btn = QPushButton("Approve frame")
        self.approve_btn.setToolTip(
            "Shift+H — keep every predicted point on this frame as your own\n"
            "label, for all individuals at once, then go where 'Then go to:'\n"
            "says. Reviewing a fill is mostly agreeing with it, so agreeing is\n"
            "one key; correct the odd point by dragging it first.\n\n"
            "Points you already labelled are left exactly as they are."
        )
        self.approve_btn.clicked.connect(self._approve_frame)
        self.approve_btn.hide()

        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        status_row.addWidget(self.active_label)
        status_row.addWidget(self.legend_label)
        status_row.addWidget(self.approve_btn)
        status_row.addStretch()

        # Marker size is in SCREEN pixels, so it stays put when the canvas is
        # zoomed; big markers are easier to hit, small ones let you see the
        # pixel you are aiming at.
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(4, 60)
        self.point_size_spin.setValue(int(self.app_state.labelling_point_size))
        self.point_size_spin.setSuffix(" px")
        self.point_size_spin.setToolTip(
            "Size of the labelling markers, in screen pixels — constant as you\n"
            "zoom. Shrink it to see the pixel under the point you are placing."
        )
        self.point_size_spin.valueChanged.connect(self._on_point_size_changed)
        self.point_size_spin.hide()
        status_row.addWidget(self.point_size_spin)
        box.addLayout(status_row)
        return widget

    def _refresh_legend(self) -> None:
        """Explain the marker styles, but only while more than one is on screen."""
        self.legend_label.setVisible(self._mode is not None and self._has_predictions())
        self.legend_label.setText(_LEGEND_WITH_DETECTIONS if self.store.detections else _LEGEND_LABEL_AND_FILL)

    def _has_predictions(self) -> bool:
        """Whether anything on screen is not the user's own work."""
        return self.store.filled is not None or bool(self.store.detections)

    def _refresh_approve_button(self) -> None:
        """Offer approving only once there is a prediction to approve.

        Unlike the legend this does not need a mode: reviewing a fill is looking
        at the video and agreeing, which no more requires arming labelling than
        deleting a point does. A detector run counts — it too is a proposal.
        """
        self.approve_btn.setVisible(self._has_predictions())

    def _on_lock_toggled(self, _locked: bool) -> None:
        """Hand the pointer to the camera, or take it back for labelling."""
        self._apply_lock()

    def _lock_wanted(self) -> bool:
        """Whether the pointer should be suspended, for either of two reasons.

        The tick box is the user's standing intent. **Leaving the Label tab is
        the other**: the other tabs are read-and-configure screens, and on them
        a click on the video is far more likely to be someone trying to look at
        a frame — scrubbing to judge a detection, or lining up a fill — than to
        be a label. Dropping a stray keypoint while reviewing is silent, and
        costs an undo the user does not know they need.

        The two are kept apart rather than collapsed into the tick box: syncing
        the box on a tab change would rewrite the user's own setting behind
        their back, so that a trip to Detect and back would leave labelling
        locked with no memory of who locked it.
        """
        return self.lock_check.isChecked() or self.tabs.currentWidget() is not self._label_page

    def _apply_lock(self) -> None:
        """Push the effective lock onto the armed mode, if it has changed."""
        wanted = self._lock_wanted()
        if self._mode is not None and self._mode.locked != wanted:
            self._mode.set_locked(wanted)
        self._refresh_active_label()

    def _on_point_size_changed(self, value: int) -> None:
        self.app_state.labelling_point_size = float(value)
        if self._mode is not None:
            self._mode.set_point_size(float(value))
        if self._calib_mode is not None:
            self._calib_mode.set_point_size(float(value))

    # ------------------------------------------------------------------
    # Target pickers
    # ------------------------------------------------------------------

    def _refresh_target_combos(self) -> None:
        """Rebuild and sync the individual/keypoint pickers from the store.

        Signals stay blocked throughout: each combo is both an input and a
        readout, and the mode writes back whenever a number key, a Tab or a
        click on an existing point moves the target.
        """
        individuals = list(self.store.individual_names)
        wanted = self._mode.active_individual if self._mode else self._combo_individual()
        with _blocked(self.individual_combo):
            if self._combo_items(self.individual_combo) != individuals:
                self.individual_combo.clear()
                for name in individuals:
                    self.individual_combo.addItem(name, name)
            # Re-brushed on every refresh, not only on a rebuild: a recolour or a
            # change of colour mode leaves the item set alone.
            for position, name in enumerate(individuals):
                self.individual_combo.setItemData(position, self._point_brush(name, None), Qt.ForegroundRole)
            position = self.individual_combo.findData(wanted)
            self.individual_combo.setCurrentIndex(position if position >= 0 else 0)
        self.individual_combo.setEnabled(bool(individuals))

        loop = self.interaction_mode == LOOP_MODE
        self.keypoint_combo.setVisible(loop)
        # Also once a fill exists, since it is then what Shift+H (approve this
        # frame) does next — approving needs no mode at all.
        self.after_click_row.setVisible(loop or self.store.filled is not None)
        if not loop or self._mode is None:
            return
        keypoints = self._mode.active_keypoints
        with _blocked(self.keypoint_combo):
            if self._combo_items(self.keypoint_combo) != keypoints:
                self.keypoint_combo.clear()
                for name in keypoints:
                    self.keypoint_combo.addItem(name, name)
            for position, name in enumerate(keypoints):
                brush = self._point_brush(self._mode.active_individual, name)
                self.keypoint_combo.setItemData(position, brush, Qt.ForegroundRole)
            position = self.keypoint_combo.findData(self._mode.active_keypoint)
            self.keypoint_combo.setCurrentIndex(position if position >= 0 else 0)

    @staticmethod
    def _combo_items(combo: QComboBox) -> list:
        return [combo.itemData(i) for i in range(combo.count())]

    def _combo_individual(self) -> str | None:
        """The picker's individual, used to seed the mode when it is armed."""
        return self.individual_combo.currentData() if self.individual_combo.count() else None

    def _on_individual_combo_changed(self, _index: int) -> None:
        name = self.individual_combo.currentData()
        if self._mode is not None and name is not None:
            self._mode.set_active_individual(name)
            self._refresh_active_label()
            self._sync_tree_selection()

    def _on_keypoint_combo_changed(self, _index: int) -> None:
        name = self.keypoint_combo.currentData()
        if self._mode is not None and name is not None:
            self._mode.set_active(name, self._mode.active_individual)
            self._refresh_active_label()
            self._sync_tree_selection()

    def _refresh_active_label(self) -> None:
        """Show the marker the next click will drop, or hide the line when idle."""
        self._refresh_target_combos()
        self._refresh_legend()
        self._refresh_approve_button()
        if self._mode is None:
            self.active_label.hide()
            self.point_size_spin.hide()
            return
        if self._mode.locked:
            # The chip promises what the next click does, so while the pointer
            # belongs to the camera it must say so rather than name a keypoint.
            self.active_label.setText('<span style="opacity:0.75;">🔒 Locked — clicks pan the video</span>')
            self.active_label.show()
            self.point_size_spin.show()
            return
        individual = self._mode.active_individual
        keypoint = self._mode.active_keypoint
        if individual is None or keypoint is None:
            self.active_label.hide()
            return

        # The dot IS the marker the next click drops, in the colour the canvas
        # will draw it — whichever axis that colour is currently encoding.
        colour = self._point_brush(individual, keypoint).color().name()
        mode = "Loop" if self._mode.mode == LOOP_MODE else "Sequential"
        self.active_label.setText(
            f'<span style="color:{colour}; font-size:17px;">●</span>&nbsp;'
            f"<b>{html.escape(individual)}</b>"
            f'&nbsp;·&nbsp;<b style="color:{colour};">{html.escape(keypoint)}</b>'
            f'&nbsp;&nbsp;<span style="opacity:0.6;">— {mode}</span>'
        )
        self.active_label.show()
        self.point_size_spin.show()

    def _build_table_group(self) -> QTableView:
        """The points table — seeks the video when clicked, right-click edits.

        A view over a virtual model rather than a widget table: once a fill
        exists the row set spans every frame it covers, which is far more rows
        than cell widgets can be made for.
        """
        self.point_model = PointTableModel(self.store, self)
        self.point_proxy = MultiColumnFilterProxy(self)
        self.point_proxy.setSourceModel(self.point_model)

        self.point_table = QTableView()
        self.point_table.setModel(self.point_proxy)
        header = GroupedHeaderView(len(_FIXED_COLUMNS), _KEYPOINT_AXES, self.point_table)
        self.point_table.setHorizontalHeader(header)
        # Frame carries no funnel: the rows are already in frame order and the
        # suggestion navigation is how you move between frames. Neither does
        # confidence, now that there is one per keypoint — the criteria are
        # ANDed, so ticking two would ask for "beak *and* tail below 0.5" when
        # the question is only ever "any point below 0.5". That question is what
        # the "Lowest fill confidence" suggestion answers, over the whole video.
        header.set_filterable({INDIVIDUAL_COLUMN, SOURCE_COLUMN}, set())
        header.filter_requested.connect(self._on_filter_requested)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        # ResizeToContents measures rows to size a column; with a fill loaded
        # there are as many rows as frames, so cap what it looks at.
        header.setResizeContentsPrecision(50)

        self.point_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.point_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # Extended selection so a run of frames can be discarded in one go.
        self.point_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.point_table.verticalHeader().setVisible(False)
        self.point_table.setMaximumHeight(TABLE_MAX_HEIGHT)
        self.point_table.setToolTip(
            "Click a cell to jump to that frame and make its keypoint active.\n"
            "Right-click to delete — or pin — the selected rows' points.\n"
            "The funnels in the Individual and Source headers filter the rows.\n"
            "Each keypoint's 'conf' column says how much the fill trusts that\n"
            "one point; 'Lowest fill confidence' below finds the worst frames."
        )
        # clicked, not the selection signal: the table also selects itself when
        # the playhead moves, and that must not seek back.
        self.point_table.clicked.connect(self._on_table_clicked)
        self.point_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.point_table.customContextMenuRequested.connect(self._on_table_context_menu)
        return self.point_table

    # ------------------------------------------------------------------
    # Column filters
    # ------------------------------------------------------------------

    def _filter_values(self, column: int) -> list[str]:
        """The categories offered for a categorical column."""
        if column == INDIVIDUAL_COLUMN:
            return list(self.store.individual_names)
        return [HUMAN_SOURCE, DETECTED_SOURCE, FILL_SOURCE]

    def _on_filter_requested(self, column: int) -> None:
        """A funnel was clicked: edit that column's filter.

        Only the categorical ones carry a funnel here — see
        :meth:`_build_table_group` for why confidence does not.
        """
        header = self.point_table.horizontalHeader()
        if not header.is_categorical(column):
            return
        dialog = CategoryFilterDialog(column, self._filter_values(column), self.point_proxy.cat_filter(column), self)
        if dialog.exec_():
            self.point_proxy.set_cat_filter(column, dialog.get_allowed())
        header.set_active_filters(self.point_proxy.active_filters())
        self._select_table_row_for_frame()

    def _clear_filters(self) -> None:
        """Drop every filter — schema edits can make the categories meaningless."""
        self.point_proxy.clear_filters()
        self.point_table.horizontalHeader().set_active_filters(set())

    # ------------------------------------------------------------------

    def _selected_table_keys(self) -> list[tuple[int, str]]:
        """The ``(frame, individual)`` behind each selected row, in view order."""
        rows = self.point_table.selectionModel().selectedRows()
        keys = [self.point_model.key_at(self.point_proxy.mapToSource(index).row()) for index in rows]
        return [key for key in keys if key is not None]

    def _on_table_context_menu(self, position) -> None:
        """Right-click → delete the selected rows' labels, or pin their fill."""
        clicked = self.point_table.indexAt(position)
        if clicked.isValid() and clicked.row() not in {
            index.row() for index in self.point_table.selectionModel().selectedRows()
        }:
            # Right-clicking outside the selection acts on the row under the
            # cursor, which is what every file manager does.
            self.point_table.selectRow(clicked.row())
        keys = self._selected_table_keys()
        if not keys:
            return
        menu = QMenu(self.point_table)
        suffix = "this frame" if len(keys) == 1 else f"{len(keys)} frames"
        # Each action appears only when the selection holds something for it to
        # act on: "Delete labels" on a row that is entirely a prediction did
        # nothing at all, which reads as a broken menu rather than an empty one.
        if any(self.store.is_human(frame, individual) for frame, individual in keys):
            menu.addAction(f"Delete labels on {suffix}", lambda: self._delete_table_rows(keys))
        if any(self.store.has_fill(frame, individual) for frame, individual in keys):
            clear = menu.addAction(f"Delete filled points on {suffix}", lambda: self._clear_fill_rows(keys))
            clear.setToolTip(
                "Throw these predictions away — for frames where the animal is\n"
                "occluded or out of shot and the backend placed a point anyway.\n"
                "Your labels are kept; a later fill will predict them again."
            )
            pin = menu.addAction(f"Pin filled points as labels on {suffix}", lambda: self._pin_table_rows(keys))
            pin.setToolTip(
                "Keep these predicted positions as your own labels, so the next\n"
                "fill treats them as ground truth instead of re-deriving them."
            )
        if any(self.store.is_detected(frame, individual) for frame, individual in keys):
            reject = menu.addAction(f"Reject detections on {suffix}", lambda: self._clear_detection_rows(keys))
            reject.setToolTip(
                "Throw away what the detector found here — for the frames where\n"
                "it locked onto a reflection or misread a tag. Your labels stay,\n"
                "and re-running the detector will find them again."
            )
            bless = menu.addAction(f"Pin detections as labels on {suffix}", lambda: self._pin_detection_rows(keys))
            bless.setToolTip(
                "Accept these detected positions as your own labels — for freezing\n"
                "a run before retuning the detector, or seeding a training set."
            )
        if menu.isEmpty():
            return
        menu.exec_(self.point_table.viewport().mapToGlobal(position))

    def _delete_table_rows(self, keys: list[tuple[int, str]]) -> None:
        """Drop every labelled point of the given ``(frame, individual)`` rows."""
        for frame, individual in keys:
            self.store.clear_individual(frame, individual)
        self._after_table_edit()

    def _clear_fill_rows(self, keys: list[tuple[int, str]]) -> None:
        """Throw away the given rows' predictions, keeping their labels."""
        removed = sum(self.store.clear_fill_for(frame, individual) for frame, individual in keys)
        if not removed:
            return
        self._push_pose_override()
        self._after_table_edit()
        notify(f"Removed {removed} filled point(s).", "info")

    def _clear_detection_rows(self, keys: list[tuple[int, str]]) -> None:
        """Reject the given rows' detections, keeping their labels."""
        removed = sum(self.store.clear_detections_for(frame, individual) for frame, individual in keys)
        if not removed:
            return
        self._push_pose_override()
        self._after_table_edit()
        notify(f"Rejected {removed} detected point(s).", "info")

    def _pin_detection_rows(self, keys: list[tuple[int, str]]) -> None:
        """Promote the given rows' detections to labels."""
        pinned = sum(self.store.promote_detections(frame, individual) for frame, individual in keys)
        if not pinned:
            notify("Nothing to pin — those points are already labelled.", "info")
            return
        self._after_table_edit()
        notify(f"Pinned {pinned} detected point(s) as labels.", "info")

    def _pin_table_rows(self, keys: list[tuple[int, str]]) -> None:
        """Promote the given rows' filled points to labels ("accept the fill")."""
        pinned = sum(self.store.promote_fill(frame, individual) for frame, individual in keys)
        if not pinned:
            notify("Nothing to pin — those points are already labelled.", "info")
            return
        self._after_table_edit()
        notify(f"Pinned {pinned} filled point(s) as labels.", "info")

    def _refresh_bulk_approve_buttons(self) -> None:
        """Offer each bulk approval only when there is something for it to take."""
        self.approve_detections_btn.setEnabled(bool(self.store.detections))
        self.approve_fill_btn.setEnabled(self.store.fill_range is not None)

    def _on_approve_all_detections(self) -> None:
        """Bless the whole detector run as labels, after confirming."""
        self._approve_all(
            "Approve all detections",
            f"Pin every detected point on {len(self.store.detections)} frame(s) as a label?",
            self.store.promote_all_detections,
        )

    def _on_approve_all_fill(self) -> None:
        """Bless the whole fill as labels, after confirming."""
        span = self.store.fill_range
        if span is None:
            notify("Fill the frames between your labels first.", "warning")
            return
        self._approve_all(
            "Approve all filled points",
            f"Pin every filled point on frames {span[0]}–{span[1]} as a label?\n"
            "Detected points in that range are pinned too.",
            self.store.promote_all_fill,
        )

    def _approve_all(self, title: str, question: str, promote) -> None:
        """Confirm, promote in bulk, and repaint — the shape both buttons share.

        Confirmed because it is not undoable: a bulk promotion discards the undo
        history rather than leaving a stack nobody can walk back (see
        :meth:`~ethograph.gui.pose_annotate.KeypointStore._promote_bulk`), and
        the wording says so in the same words "Clear all labels…" uses.
        """
        if QMessageBox.question(self, title, f"{question}\nThis cannot be undone.") != QMessageBox.Yes:
            return
        promoted = promote()
        if not promoted:
            # Not a failure: everything worth taking was already labelled.
            notify("Nothing to approve — those points are already labels.", "info")
            return
        self._after_table_edit()
        notify(f"Approved {promoted} point(s) as labels.", "info")

    def _on_clear_all_labels(self) -> None:
        """Wipe every labelled point in the table, after confirming."""
        n_frames = len(self.store.anchors)
        if n_frames == 0:
            notify("There are no labels to clear.", "info")
            return
        confirm = QMessageBox.question(
            self,
            "Clear all labels",
            f"Delete every labelled point on {n_frames} frame(s)?\n"
            "Filled (predicted) points are kept. This cannot be undone.",
        )
        if confirm != QMessageBox.Yes:
            return
        self.store.clear_all_labels()
        self._after_table_edit()
        notify("Cleared all labels.", "info")

    def _after_table_edit(self) -> None:
        """A bulk edit touched several frames, so nothing narrow can be repainted."""
        if self._mode is not None:
            self._mode.refresh()
        self._on_store_changed(full=True)
        self._save_store()

    def _build_suggest_group(self) -> QGroupBox:
        """Which frames to label — see :mod:`~ethograph.gui.pose_suggest`."""
        group = QGroupBox("Which frames to label")
        box = QVBoxLayout(group)

        row = QHBoxLayout()
        self.suggest_method_combo = QComboBox()
        for key, label, tip in _SUGGEST_METHODS:
            self.suggest_method_combo.addItem(label, key)
            self.suggest_method_combo.setItemData(self.suggest_method_combo.count() - 1, tip, Qt.ToolTipRole)
        # "uncertain" sits last because it can only rank what a fill has already
        # scored: opening on it would mean the first press of the button can
        # only warn. Before a fill the combo starts at the top of the list.
        opening = "uncertain" if self.store.confidence is not None else _SUGGEST_METHODS[0][0]
        self.suggest_method_combo.setCurrentIndex(self.suggest_method_combo.findData(opening))
        row.addWidget(self.suggest_method_combo, stretch=1)

        # A share of the video, not an absolute count: how many frames are worth
        # labelling scales with how long the clip is, and the resolved count is
        # spelled out beside it so the percentage is never abstract.
        self.suggest_percent_spin = QDoubleSpinBox()
        self.suggest_percent_spin.setRange(MIN_SUGGEST_PERCENT, 100.0)
        self.suggest_percent_spin.setDecimals(1)
        self.suggest_percent_spin.setSingleStep(0.5)
        self.suggest_percent_spin.setSuffix(" %")
        self.suggest_percent_spin.setValue(self._default_suggest_percent())
        self.suggest_percent_spin.setToolTip("What share of the video to propose for labelling.")
        self.suggest_percent_spin.valueChanged.connect(lambda _v: self._refresh_suggest_count_label())
        row.addWidget(self.suggest_percent_spin)
        box.addLayout(row)

        self.suggest_count_label = QLabel()
        self.suggest_count_label.setToolTip("How many frames that percentage works out to.")
        box.addWidget(self.suggest_count_label)

        suggest_btn = QPushButton("Suggest frames")
        suggest_btn.setToolTip(
            "Propose frames spread across the video.\nLabelling neighbouring frames teaches a tracker very little."
        )
        suggest_btn.clicked.connect(self._on_suggest)
        box.addWidget(suggest_btn)

        self.suggestion_label = QLabel("No suggested frames yet.")
        box.addWidget(self.suggestion_label)

        # One direction only: the suggestions are a queue to work down, and any
        # frame — suggested or not — is one click away in the points table.
        next_btn = QPushButton("Next suggested frame  (N)")
        next_btn.setToolTip(
            "Jump to the next suggested frame, wrapping at the end.\n"
            "Plain ← / → step one frame at a time; the points table seeks to any frame."
        )
        next_btn.clicked.connect(self._next_suggestion)
        box.addWidget(next_btn)
        self._refresh_suggest_count_label()
        return group

    def _default_suggest_percent(self) -> float:
        """:data:`RECOMMENDED_LABEL_SHARE` — roughly every 10th frame.

        A spacing, not a count: the backends bridge *gaps*, and a gap is
        measured in frames, so the same share means the same difficulty on a
        200-frame clip and on an hour of footage. The resolved count is spelled
        out beside the spin box, which is where an unreasonable one shows up.
        """
        return RECOMMENDED_LABEL_SHARE

    def _suggest_count(self) -> int:
        """Frames the requested percentage works out to, at least one."""
        return max(1, round(self._n_frames() * self.suggest_percent_spin.value() / 100))

    def _refresh_suggest_count_label(self) -> None:
        n_frames = self._n_frames()
        if not n_frames:
            self.suggest_count_label.setText("Frame count unknown — load a video.")
            return
        self.suggest_count_label.setText(f"{self._suggest_count()} of {n_frames} frames")

    def _build_fill_group(self) -> QGroupBox:
        group = QGroupBox("Fill")
        box = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        for info in available_backends():
            self.backend_combo.addItem(info.label, info.key)
            index = self.backend_combo.count() - 1
            if not info.available:
                item = self.backend_combo.model().item(index)
                item.setEnabled(False)
                self.backend_combo.setItemData(index, f"Not installed — {info.hint}", Qt.ToolTipRole)
        saved = self.app_state.labelling_backend
        position = self.backend_combo.findData(saved)
        self.backend_combo.setCurrentIndex(position if position >= 0 else 0)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        row.addWidget(self.backend_combo, stretch=1)
        box.addLayout(row)

        # Only the tracking backends score by forward/backward agreement; the
        # spline scores by distance from the nearest anchor, so this row hides
        # for it rather than sitting there meaning nothing.
        self.disagreement_row = QWidget()
        disagreement = QHBoxLayout(self.disagreement_row)
        disagreement.setContentsMargins(0, 0, 0, 0)
        disagreement.addWidget(QLabel("Disagreement tolerance:"))
        self.disagreement_spin = QDoubleSpinBox()
        self.disagreement_spin.setRange(0.5, 500.0)
        self.disagreement_spin.setDecimals(1)
        self.disagreement_spin.setSuffix(" px")
        self.disagreement_spin.setValue(float(self.app_state.labelling_disagreement_px))
        self.disagreement_spin.setToolTip(
            "How far the forward and backward tracks may drift apart before a\n"
            "point is called unreliable: this many source-video pixels of\n"
            "disagreement costs a factor 1/e of confidence.\n\n"
            "Raise it for large or fast animals, lower it to be strict. It only\n"
            "changes each keypoint's 'conf' column and which frames 'Lowest\n"
            "fill confidence' proposes — never the positions."
        )
        self.disagreement_spin.valueChanged.connect(self._on_disagreement_changed)
        disagreement.addWidget(self.disagreement_spin, stretch=1)
        box.addWidget(self.disagreement_row)

        # The stock checkpoint is a default, not a constant: a CoTracker3
        # fine-tuned on animal footage is a drop-in state dict, and picking one
        # must not mean editing pose_fill.
        self.checkpoint_row = QWidget()
        checkpoint = QHBoxLayout(self.checkpoint_row)
        checkpoint.setContentsMargins(0, 0, 0, 0)
        checkpoint.addWidget(QLabel("Model weights:"))
        self.checkpoint_edit = QLineEdit(self.app_state.labelling_cotracker_checkpoint)
        self.checkpoint_edit.setPlaceholderText(f"Stock CoTracker3 ({COTRACKER_CHECKPOINT_NAME})")
        self.checkpoint_edit.setToolTip(
            "A CoTracker3 checkpoint to track with. Leave empty for the stock\n"
            "weights, downloaded on first use.\n\n"
            "Point this at weights fine-tuned for your footage — anything sharing\n"
            "the CoTracker3 architecture loads here. A different architecture\n"
            "will not: that needs a new backend."
        )
        self.checkpoint_edit.editingFinished.connect(self._on_checkpoint_edited)
        checkpoint.addWidget(self.checkpoint_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_checkpoint)
        checkpoint.addWidget(browse_btn)
        box.addWidget(self.checkpoint_row)

        # Refinement is the one backend carrying state between fills: the fit is
        # minutes of GPU, so a fill reuses it whenever the labels it was made
        # from still stand. Fill decides that by itself, so there is no fit
        # BUTTON — a second verb for a step the first one already takes reads as
        # a choice about the result, which it never was. What the user cannot
        # infer is which phases the next fill will pay for, and that is text.
        self.refinement_row = QWidget()
        refinement = QVBoxLayout(self.refinement_row)
        refinement.setContentsMargins(0, 0, 0, 0)

        self.refinement_method = QLabel(
            "Filling runs two phases: <b>fit</b> — learn what your keypoints look like "
            "in this video (minutes on a GPU) — then <b>track</b> — follow them across "
            "every gap (fast). Fill does both and skips the fit while it still matches "
            "your labels."
        )
        self.refinement_method.setWordWrap(True)
        self.refinement_method.setToolTip(
            "The fit optimises CoTracker3's per-keypoint appearance features against\n"
            "the frames you labelled, so it tracks YOUR keypoints on THIS animal\n"
            "rather than whatever the query patch happened to look like.\n\n"
            "It depends only on your labels and the video, so it is cached in memory\n"
            "and saved next to the video, and every fill that follows is just the\n"
            "tracking pass. Correct a point and the fit is out of date: the next fill\n"
            "redoes it by itself, from scratch — there is no incremental fitting, so a\n"
            "refit and a first fit are the same work."
        )
        refinement.addWidget(self.refinement_method)

        self.refinement_status = QLabel()
        self.refinement_status.setWordWrap(True)
        self.refinement_status.setToolTip(
            "What the next fill will have to do: fit and track, or track alone.\n"
            "Cancelling it leaves your labels, your fill and the current fit as\n"
            "they are."
        )
        refinement.addWidget(self.refinement_status)
        box.addWidget(self.refinement_row)

        self._refresh_backend_rows()

        fill_btn = QPushButton("Fill frames between labels")
        fill_btn.setToolTip(
            "Fill every frame from the first labelled one to the last.\n\n"
            "Frames before the first label and after the last are left empty:\n"
            "there is no second label to interpolate towards and nothing to\n"
            "track between, so anything put there would be a guess extended\n"
            "from one end. Label further out to extend the range."
        )
        fill_btn.clicked.connect(self._on_fill)
        box.addWidget(fill_btn)

        clear_btn = QPushButton("Clear fill")
        clear_btn.setToolTip("Discard the filled frames; labelled anchors are kept.")
        clear_btn.clicked.connect(self._on_clear_fill)
        box.addWidget(clear_btn)
        return group

    def _build_export_group(self) -> QGroupBox:
        group = QGroupBox("Export")
        box = QVBoxLayout(group)

        space_row = QHBoxLayout()
        space_row.addWidget(QLabel("Coordinate space:"))
        self.space_combo = QComboBox()
        self.space_combo.addItem("pixels", "pixels")
        self.space_combo.addItem("cm (calibrated)", "cm")
        self.space_combo.currentIndexChanged.connect(self._on_space_combo_changed)
        space_row.addWidget(self.space_combo)
        space_row.addStretch()
        box.addLayout(space_row)

        self.invert_y_check = QCheckBox("Flip y so plots are not upside down")
        self.invert_y_check.setChecked(True)
        self.invert_y_check.setToolTip(
            "Pose coordinates count y DOWNWARD from the top-left corner; plots count\n"
            "it upward. Without the flip a keypoint at the top of the video is drawn\n"
            "at the bottom of a space plot.\n\n"
            "Applies to 'Load into the GUI' and the NetCDF export. The video overlay\n"
            "and the DeepLabCut export always keep raw image coordinates."
        )
        box.addWidget(self.invert_y_check)

        box.addWidget(QLabel("Derive from the labelled + filled trajectories:"))
        derive_row = QHBoxLayout()
        self.kinematic_checks: dict[str, QCheckBox] = {}
        for name in KINEMATICS:
            check = QCheckBox(name.capitalize())
            check.setChecked(True)
            check.setToolTip(f"Compute {name} per keypoint (movement.kinematics.compute_{name}).")
            derive_row.addWidget(check)
            self.kinematic_checks[name] = check
        derive_row.addStretch()
        box.addLayout(derive_row)

        box.addWidget(self._build_head_direction_row())

        load_btn = QPushButton("Load into the GUI")
        load_btn.setToolTip(
            "Add the keypoints — and whichever kinematics are ticked — to the\n"
            "current trial as features, so they can be plotted straight away.\n"
            "Filled frames are included, which is the point: it is how you see\n"
            "what the fill actually did. No file is written."
        )
        load_btn.clicked.connect(self._on_load_into_gui)
        box.addWidget(load_btn)

        movement_btn = QPushButton("Export poses (NetCDF)…")
        movement_btn.setToolTip(
            "Write a movement-compatible poses dataset covering every frame of the\n"
            "video; frames outside the filled span are NaN, as movement expects."
        )
        movement_btn.clicked.connect(self._on_export_movement)
        box.addWidget(movement_btn)
        self._refresh_space_combo()
        return group

    def _refresh_space_combo(self) -> None:
        """Offer cm exactly when a usable calibration exists.

        The gate is the same shape as the head-direction row's: enabled when
        the evidence is there, the reason in the tooltip when it is not — and a
        calibration that becomes unusable snaps the choice back to pixels
        rather than exporting something the fit can no longer honour.
        """
        combo = getattr(self, "space_combo", None)
        if combo is None:
            return
        valid = self.store.calibration.is_valid()
        combo.setEnabled(valid)
        # The combo SHOWS pixels while the fit is unusable, but the user's own
        # choice is kept and restored: a calibration goes transiently invalid
        # on every half-edited coordinate cell, and losing "cm" to that churn
        # meant exporting pixels without anyone choosing them.
        target = self._space_choice if valid else "pixels"
        self._space_syncing = True
        try:
            index = combo.findData(target)
            if index >= 0 and combo.currentIndex() != index:
                combo.setCurrentIndex(index)
        finally:
            self._space_syncing = False
        if valid:
            combo.setToolTip(
                "pixels: source-video image coordinates, as clicked.\n"
                "cm: mapped through the Calibrate tab's landmark fit — the\n"
                "user-defined world frame, kinematics in cm/s.\n\n"
                "Applies to 'Load into the GUI' and the NetCDF export. The video\n"
                "overlay always stays in pixels."
            )
        else:
            combo.setToolTip(
                f"Needs at least {MIN_CALIBRATION_LANDMARKS} landmarks, each with cm\n"
                "coordinates and at least one click — see the Calibrate tab."
            )
        self._sync_flip_for_space()

    def _on_space_combo_changed(self, _index: int) -> None:
        if not self._space_syncing:
            self._space_choice = self.space_combo.currentData()
        self._sync_flip_for_space()

    def _sync_flip_for_space(self) -> None:
        """The flip means something different per space; the tooltip says which.

        In pixels it undoes the image's y-down convention (via ``image_height``);
        in cm it mirrors the user's world frame after the calibration (composed
        into the matrix in :meth:`_build_dataset`) — never the pixels, which the
        fit was not made from.
        """
        check = getattr(self, "invert_y_check", None)
        if check is None:
            return
        if self.space_combo.currentData() == "cm":
            check.setToolTip(
                "In cm this mirrors your world frame's y axis (y → −y), applied\n"
                "after the calibration. Untick it if your landmark coordinates\n"
                "already have y pointing the way you want plots to read."
            )
        else:
            check.setToolTip(
                "Pose coordinates count y DOWNWARD from the top-left corner; plots count\n"
                "it upward. Without the flip a keypoint at the top of the video is drawn\n"
                "at the bottom of a space plot.\n\n"
                "Applies to 'Load into the GUI' and the NetCDF export. The video overlay\n"
                "and the DeepLabCut export always keep raw image coordinates."
            )

    def _cm_selected(self) -> bool:
        """Whether to export in cm: asked for, and the fit can honour it."""
        return bool(self.space_combo.currentData() == "cm" and self.store.calibration.is_valid())

    def _build_head_direction_row(self) -> QWidget:
        """Head direction, read off the tags themselves.

        One tick box and nothing to configure: an AprilTag is a square, so the
        detector already measured which way each one faces when it decoded it.
        There is no pair of keypoints to nominate, because the heading belongs
        to the tagged keypoint itself. With no oriented marker in the session
        there is nothing to offer, and the box says so instead of asking.
        """
        widget = QWidget()
        box = QVBoxLayout(widget)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)

        self.head_direction_check = QCheckBox("Head direction (from marker orientation)")
        box.addWidget(self.head_direction_check)

        self._refresh_head_direction_row()
        return widget

    def _refresh_head_direction_row(self) -> None:
        """Offer head direction exactly when an oriented marker was detected.

        Called after a detector run and after schema changes. A run that finds
        tags turns the tick on by itself — measuring their orientation costs
        nothing extra, so having it and not offering it would be perverse —
        while a session with no tags leaves it off and disabled, with the reason
        in the tooltip rather than in a warning after the fact.
        """
        check = getattr(self, "head_direction_check", None)
        if check is None:
            return
        available = self.store.has_orientation
        check.setEnabled(available)
        if available:
            check.setToolTip(
                "Add each tagged keypoint's forward vector and its heading angle\n"
                "in degrees.\n\n"
                "Measured from the tag itself: its printed TOP edge is taken as the\n"
                "front, so the heading is perpendicular to that edge. One tag is one\n"
                "heading — nothing to pick, and no second keypoint needed.\n\n"
                "Frames where the tag did not decode are left empty rather than\n"
                "interpolated: a heading is a measurement, not a prediction.\n\n"
                "Applies to 'Load into the GUI' and the NetCDF export."
            )
            if not self._head_direction_offered:
                check.setChecked(True)
                self._head_direction_offered = True
        else:
            check.setChecked(False)
            check.setToolTip(
                "Needs a marker that has an orientation of its own.\n\n"
                "An AprilTag is a square, so decoding one says which way it faces —\n"
                "run the detector on the Detect tab and this turns on by itself.\n"
                "Hand-placed keypoints are bare points: a single one cannot face\n"
                "anywhere, so there is no head direction to compute."
            )
            self._head_direction_offered = False

    def _head_direction_wanted(self) -> bool:
        """Whether to add a heading: asked for, and there is one to add."""
        return bool(self.head_direction_check.isChecked() and self.store.has_orientation)

    def _export_image_height(self) -> float | None:
        """Image height for the y-flip, or ``None`` to keep image coordinates."""
        if not self.invert_y_check.isChecked():
            return None
        height = self._view.image_height()
        if not height:
            notify("Video height is unknown — exporting in image coordinates (y not flipped).", "warning")
            return None
        return float(height)

    def _selected_kinematics(self) -> list[str]:
        return [name for name, check in self.kinematic_checks.items() if check.isChecked()]

    def _build_dataset(self):
        """The poses dataset both output paths produce, or ``None`` with a reason.

        One builder, so *Load into the GUI* and *Export poses* can never disagree
        about what a labelling session contains.
        """
        fps = self._fps()
        if not fps:
            notify("Video frame rate is unknown — cannot build a poses dataset.", "warning")
            return None
        if not self.store.anchor_frames() and not self.store.detection_frames():
            notify("Nothing to save — no frames are labelled or detected.", "warning")
            return None
        world_transform = None
        if self._cm_selected():
            try:
                world_transform = self.store.calibration.fit()
            except KeypointStoreError as e:
                notify(f"Calibration is not usable: {e}", "error")
                return None
            if self.invert_y_check.isChecked():
                # In cm the flip mirrors the WORLD frame's y, composed after
                # the fit — never a pixel flip, which the fit was not made
                # from. Positions and head direction both ride the one matrix.
                world_transform = np.diag([1.0, -1.0, 1.0]) @ world_transform
        try:
            return store_to_dataset(
                self.store,
                fps,
                # Kinematics and head direction follow the flip, so their y signs
                # match the trajectory the user is looking at. In cm the flip
                # retires — the world frame defines its own orientation.
                image_height=None if world_transform is not None else self._export_image_height(),
                kinematics=self._selected_kinematics(),
                head_direction=self._head_direction_wanted(),
                world_transform=world_transform,
            )
        except (KeypointStoreError, ImportError, ValueError) as e:
            notify(f"Could not build the poses dataset: {e}", "error")
            return None

    def _on_load_into_gui(self) -> None:
        """Serve the session's features from the poses dataset, and save a copy.

        The dataset is what the GUI gets — replacing the feature data rather
        than being merged into it, so ``keypoint`` and ``individual`` arrive as
        ordinary selectable dimensions. The file written beside the video is the
        same thing on disk, byte-for-byte what *Export poses* produces; writing
        it is not how the GUI is fed, so a read-only folder costs a warning and
        nothing else.
        """
        video = self._video_path()
        if not video:
            notify("No video is loaded — there is nothing to attach the keypoints to.", "warning")
            return
        if self.store.filled is None:
            notify("Only the observed frames will be loaded — run Fill to cover the rest.", "info")

        ds = self._build_dataset()
        if ds is None:
            return

        path = keypoints_dataset_path(video)
        try:
            ds.to_netcdf(path)
        except OSError as e:
            notify(f"Loaded, but could not save a copy to {path.name}: {e}", "warning")

        if self._data_widget.load_keypoint_dataset(ds):
            notify(
                f"Loaded {', '.join(ds.data_vars)} in {ds.attrs['space_unit']} — saved as {path.name}.",
                "info",
            )

    # ------------------------------------------------------------------
    # Detect: markers read off the pixels, one frame at a time
    # ------------------------------------------------------------------

    def _build_detect_page(self) -> QWidget:
        """Detector, what its labels mean, and the run itself.

        Its own tab because it is its own stage: it produces *observations*, the
        same kind of thing a click produces, which the fill on the next tab then
        bridges. Running it is optional, and running it twice with different
        parameters never disturbs a label.
        """
        page = QWidget()
        box = QVBoxLayout(page)
        box.addWidget(self._build_detector_group())
        # Directly under the parameters, because it is what they do: every spin
        # box above redraws it on the frame already on screen.
        box.addWidget(self._build_preview_group())
        box.addWidget(self._build_assignment_group(), stretch=1)
        box.addWidget(self._build_run_group())
        return page

    def _build_preview_group(self) -> QGroupBox:
        """What the detector sees on this frame — masks, keeps and near misses.

        Tuning by numbers alone is guesswork: you set a tolerance, run thirty
        thousand frames, and find out afterwards. This costs one decoded frame
        and a few milliseconds, so it can follow the playhead.
        """
        group = QGroupBox("Preview on this frame")
        box = QVBoxLayout(group)

        self.mask_preview = PreviewPanel()
        box.addWidget(self.mask_preview)

        controls = QHBoxLayout()
        self.preview_check = QCheckBox("Show preview")
        self.preview_check.setChecked(True)
        self.preview_check.setToolTip(
            "Redraw as you scrub and as you change the settings above.\n"
            "Untick to leave it alone — nothing else depends on it."
        )
        self.preview_check.toggled.connect(self._on_preview_toggled)
        controls.addWidget(self.preview_check)
        controls.addStretch()
        self.preview_summary = QLabel()
        self.preview_summary.setWordWrap(True)
        controls.addWidget(self.preview_summary, stretch=1)
        box.addLayout(controls)

        # Coalesces a scrub into one redraw: dragging the playhead emits a frame
        # change per tick, and each redraw decodes a frame.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._refresh_preview)
        return group

    def _build_detector_group(self) -> QGroupBox:
        group = QGroupBox("Detector")
        box = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Find:"))
        self.detector_combo = QComboBox()
        for info in available_detectors():
            self.detector_combo.addItem(info.label, info.key)
            index = self.detector_combo.count() - 1
            if not info.available:
                self.detector_combo.model().item(index).setEnabled(False)
                self.detector_combo.setItemData(index, f"Not installed — {info.hint}", Qt.ToolTipRole)
        position = self.detector_combo.findData(self.app_state.detect_detector)
        self.detector_combo.setCurrentIndex(position if position >= 0 else 0)
        # A key from an older version (or a detector since removed) resolves to
        # the first entry — write that back rather than leaving a setting that
        # names something no longer offered.
        self.app_state.detect_detector = self.detector_combo.currentData()
        self.detector_combo.currentIndexChanged.connect(self._on_detector_changed)
        row.addWidget(self.detector_combo, stretch=1)
        box.addLayout(row)

        # The family combo offers only what pose_detect can actually construct:
        # an unlisted AprilTag family aborts the process rather than raising, so
        # the list is the guard, not a suggestion.
        self.tag_row = QWidget()
        tags = QHBoxLayout(self.tag_row)
        tags.setContentsMargins(0, 0, 0, 0)

        tags.addWidget(QLabel("Family:"))
        self.tag_family_combo = QComboBox()
        for name in TAG_FAMILIES:
            self.tag_family_combo.addItem(name, name)
            self.tag_family_combo.setItemData(self.tag_family_combo.count() - 1, family_note(name), Qt.ToolTipRole)
        position = self.tag_family_combo.findData(self.app_state.detect_tag_family)
        self.tag_family_combo.setCurrentIndex(position if position >= 0 else 0)
        self.tag_family_combo.setToolTip(
            "Which AprilTag family you printed. It must match the sheet — a\n"
            "tag16h5 tag will never decode as tag36h11.\n\n"
            "tag36h11 is the default: the most IDs and the widest margin. The\n"
            "smaller families need less paper for the same pixels per module,\n"
            "which is what makes them worth trying on a small animal."
        )
        self.tag_family_combo.currentIndexChanged.connect(self._on_tag_params_changed)
        tags.addWidget(self.tag_family_combo)

        tags.addWidget(QLabel("Downscale:"))
        self.tag_decimate_spin = QDoubleSpinBox()
        self.tag_decimate_spin.setRange(1.0, 4.0)
        self.tag_decimate_spin.setDecimals(1)
        self.tag_decimate_spin.setSingleStep(1)
        self.tag_decimate_spin.setValue(float(self.app_state.detect_quad_decimate))
        self.tag_decimate_spin.setToolTip(
            "quad_decimate: how far the frame is shrunk before tags are looked\n"
            "for. 2.0 runs several times faster and needs tags twice as big —\n"
            "the library's own default, and the reason small tags 'stop working'\n"
            "elsewhere. 1.0 (here) searches the full frame.\n\n"
            "The 'must be ≥N px per side' figure below already includes it."
        )
        self.tag_decimate_spin.valueChanged.connect(self._on_tag_params_changed)
        tags.addWidget(self.tag_decimate_spin)

        tags.addWidget(QLabel("Sharpening:"))
        self.tag_sharpening_spin = QDoubleSpinBox()
        self.tag_sharpening_spin.setRange(0.0, 2.0)
        self.tag_sharpening_spin.setDecimals(2)
        self.tag_sharpening_spin.setSingleStep(0.05)
        self.tag_sharpening_spin.setValue(float(self.app_state.detect_decode_sharpening))
        self.tag_sharpening_spin.setToolTip(
            "decode_sharpening: applied to the sampled bit pattern before it is\n"
            "read. Raise it for motion-blurred or slightly out-of-focus tags."
        )
        self.tag_sharpening_spin.valueChanged.connect(self._on_tag_params_changed)
        tags.addWidget(self.tag_sharpening_spin)

        self.tag_corners_check = QCheckBox("Detect the four corners too")
        self.tag_corners_check.setChecked(bool(self.app_state.detect_tag_corners))
        self.tag_corners_check.setToolTip(
            "One tag becomes four keypoints instead of one, so a tagged animal\n"
            "carries an orientation as well as a position."
        )
        self.tag_corners_check.toggled.connect(self._on_tag_params_changed)
        tags.addWidget(self.tag_corners_check)
        tags.addStretch()
        # Printing the tags is NOT offered here. By the time this tab is
        # reachable there is a video, which means the tags were printed and
        # stuck on the animals weeks ago — the sheet belongs on the cover page's
        # "Pre-recording tools", the one screen that exists before a recording.
        box.addWidget(self.tag_row)

        self._refresh_detector_rows()
        return group

    def _build_assignment_group(self) -> QGroupBox:
        """What each detector label *means* — the core of the feature.

        A tag decodes to ``7``; it does not know it is the beak, or bee 12.
        Learning proposes; the table is where the user overrules, and any row
        they touch is never overwritten by a re-learn.
        """
        group = QGroupBox("What the detector's labels mean")
        box = QVBoxLayout(group)

        self.assignment_table = QTableWidget(0, len(_ASSIGNMENT_COLUMNS))
        self.assignment_table.setHorizontalHeaderLabels(list(_ASSIGNMENT_COLUMNS))
        self.assignment_table.verticalHeader().setVisible(False)
        self.assignment_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.assignment_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.assignment_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.assignment_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.assignment_table.setMaximumHeight(TABLE_MAX_HEIGHT)
        self.assignment_table.setToolTip(
            "One row per label the detector can produce. Pick the individual and\n"
            "keypoint each one lands on; a row you edit is kept through every\n"
            "later 'Learn from labels'."
        )
        for column, tip in enumerate(_ASSIGNMENT_TOOLTIPS):
            self.assignment_table.horizontalHeaderItem(column).setToolTip(tip)
        box.addWidget(self.assignment_table)

        buttons = QHBoxLayout()
        learn_btn = QPushButton("Learn from labels")
        learn_btn.setToolTip(
            "Run the detector on the frames you labelled and match each decoded\n"
            "tag to the nearest labelled point — so labelling the animal a tag is\n"
            "stuck to is what teaches Ethograph whose tag it is.\n\n"
            "At least two labelled frames must agree before a row is proposed."
        )
        learn_btn.clicked.connect(self._on_learn_assignment)
        buttons.addWidget(learn_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Forget every assignment, including the ones you edited by hand.")
        clear_btn.clicked.connect(self._on_clear_assignment)
        buttons.addWidget(clear_btn)
        buttons.addStretch()
        box.addLayout(buttons)

        self.assignment_warning = QLabel()
        self.assignment_warning.setWordWrap(True)
        self.assignment_warning.setToolTip(
            "Keypoints no label claims produce nothing on a run, and a label\n"
            "pointing at a keypoint that no longer exists is simply skipped."
        )
        box.addWidget(self.assignment_warning)
        self._refresh_assignment_table()
        return group

    # -- preview -------------------------------------------------------

    def _preview_wanted(self) -> bool:
        """Whether to spend a decode: only when ticked and actually on screen."""
        return self.preview_check.isChecked() and self.tabs.currentWidget() is self._detect_page

    def _schedule_preview(self) -> None:
        """Redraw shortly — coalescing a scrub or a spin box's repeats."""
        if self._preview_wanted():
            self._preview_timer.start()

    def _on_preview_toggled(self, on: bool) -> None:
        """Unticking must also drop the redraw already queued behind a scrub."""
        if on:
            self._schedule_preview()
        else:
            self._preview_timer.stop()

    def _preview_frame_source(self):
        """A frame source kept open for the preview, or ``None`` without a video.

        Kept rather than opened per redraw: opening a PyAV container costs far
        more than the detection itself, and this runs on every scrub.
        """
        video = self._video_path()
        if not video or not self._fps():
            return None
        wanted = (video, self._n_frames(), self._view.start_frame)
        if self._preview_frames is not None and self._preview_frames_for != wanted:
            self._close_preview_frames()
        if self._preview_frames is None:
            try:
                self._preview_frames = self._open_frames(max_side=DETECT_MAX_SIDE)
            except (ValueError, OSError) as e:
                logger.warning("No preview frames: %s", e)
                return None
            self._preview_frames_for = wanted
        return self._preview_frames

    def _close_preview_frames(self) -> None:
        if self._preview_frames is not None:
            self._preview_frames.close()
        self._preview_frames = None
        self._preview_frames_for = None

    def _refresh_preview(self) -> None:
        """Draw the current frame as the detector sees it.

        Everything shown comes from ``diagnose_frame``, which runs the detector's
        own ``detect`` path — a preview that could disagree with a run would be
        worse than none. A detector that cannot be built yet (no labels to learn
        a colour from) is a *message*, not an error: it is the normal state on
        the way in.
        """
        if not self._preview_wanted():
            return
        frames = self._preview_frame_source()
        if frames is None:
            self.mask_preview.show_message("Load a video to preview the detector.")
            self.preview_summary.setText("")
            return
        frame_index = max(0, min(self._current_frame(), len(frames) - 1))
        try:
            image = np.asarray(frames[frame_index])
            detector = self._current_detector()
            preview = diagnose_frame(detector, image)
        except (PointDetectorError, KeypointStoreError, ValueError, OSError) as e:
            self.mask_preview.show_message(str(e))
            self.preview_summary.setText("")
            return

        names = {shape.label: self._label_text(shape.label) for shape in preview.accepted if shape.label is not None}
        self.mask_preview.show_preview(image, preview, self._preview_colors(preview), names)
        self._refresh_preview_summary(frame_index, preview)

    def _preview_colors(self, preview) -> dict[int, tuple[float, float, float]]:
        """One colour per label — the keypoint's own.

        A tag carries no colour of its own, so it borrows the keypoint it is
        assigned to, which is what makes the preview readable beside the video
        overlay: the same tag is the same colour in both.
        """
        colors: dict[int, tuple[float, float, float]] = {}
        for entry in self.store.assignment:
            if entry.keypoint not in self.store.keypoint_names:
                continue
            colour = self._keypoint_brush(entry.keypoint).color()
            colors[entry.label] = (colour.redF(), colour.greenF(), colour.blueF())
        for shape in preview.shapes:
            if shape.label is not None:
                colors.setdefault(shape.label, (1.0, 1.0, 1.0))
        return colors

    def _refresh_preview_summary(self, frame_index: int, preview) -> None:
        """Name the failure modes apart — that is the point of the panel.

        Three of them, and only the first two are worth changing a setting for:
        a tag read but **not trusted** (bit errors had to be corrected), a tag
        read but assigned to nothing, and nothing found at all. For the last the
        question is always "was it big enough?", so the frame size and the
        pixels a tag needs *at this downscale* are stated outright — those two
        numbers plus the tag on screen settle it without a run.
        """
        assigned = set(self.store.assignment_rows())
        kept = [shape for shape in preview.accepted if shape.label in assigned]
        unassigned = len(preview.accepted) - len(kept)
        parts = [f"Frame {frame_index}: {len(kept)} tag(s) decoded"]
        if unassigned:
            parts.append(f"{unassigned} decoded but unassigned")
        if preview.rejected:
            parts.append(f"{len(preview.rejected)} rejected — misread")
        if not preview.accepted:
            parts.append("no tag decoded")
        width, height = preview.size
        needed = getattr(self._detector, "min_side_px", None)
        budget = f"scanned at {width}×{height}"
        if needed:
            budget += f", so a tag must be ≥{needed:.0f} px per side here"
        parts.append(budget)
        self.preview_summary.setText(" · ".join(parts))

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        box = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Over:"))
        self.detect_range_combo = QComboBox()
        for key, label, tip in _DETECT_RANGES:
            self.detect_range_combo.addItem(label, key)
            self.detect_range_combo.setItemData(self.detect_range_combo.count() - 1, tip, Qt.ToolTipRole)
        row.addWidget(self.detect_range_combo, stretch=1)
        box.addLayout(row)

        quality = QHBoxLayout()
        quality.addWidget(QLabel("Quality threshold:"))
        self.detect_quality_spin = QDoubleSpinBox()
        self.detect_quality_spin.setRange(0.0, 1.0)
        self.detect_quality_spin.setSingleStep(0.05)
        self.detect_quality_spin.setDecimals(2)
        self.detect_quality_spin.setValue(float(self.app_state.detect_quality_min))
        self.detect_quality_spin.setToolTip(
            "Detections scoring below this are dropped. The score is the decode\n"
            "margin — how cleanly the tag's bits separated — as a fraction of a\n"
            "good read. Spurious reads on noise score under 0.15; a real tag\n"
            "scores about 1.0, so 0.3 sits in the gap between them.\n\n"
            "Tags needing any bit correction are already rejected outright,\n"
            "whatever this says.\n\n"
            "Applied as the results are stored, so retuning it after a run costs\n"
            "nothing; only a run from a *previous session* has to be repeated."
        )
        self.detect_quality_spin.valueChanged.connect(self._on_quality_changed)
        quality.addWidget(self.detect_quality_spin, stretch=1)
        box.addLayout(quality)

        run_btn = QPushButton("Run detector")
        run_btn.setToolTip(
            "Find every assigned marker over the chosen range. The result joins\n"
            "your labels as observations, so the fill on the next tab bridges the\n"
            "frames where nothing was found."
        )
        run_btn.clicked.connect(self._on_run_detector)
        box.addWidget(run_btn)

        clear_btn = QPushButton("Clear detections")
        clear_btn.setToolTip("Discard the whole run. Your labels — and the assignments — are kept.")
        clear_btn.clicked.connect(self._on_clear_detections)
        box.addWidget(clear_btn)

        self.detect_summary = QLabel()
        self.detect_summary.setWordWrap(True)
        box.addWidget(self.detect_summary)
        self._refresh_detect_summary()
        return group

    # -- detector construction -----------------------------------------

    def _on_detector_changed(self, _index: int) -> None:
        self.app_state.detect_detector = self.detector_combo.currentData()
        self._detector = None
        self._refresh_detector_rows()
        self._schedule_preview()

    def _on_tag_params_changed(self, _value=None) -> None:
        self.app_state.detect_tag_family = self.tag_family_combo.currentText()
        self.app_state.detect_quad_decimate = float(self.tag_decimate_spin.value())
        self.app_state.detect_decode_sharpening = float(self.tag_sharpening_spin.value())
        self.app_state.detect_tag_corners = bool(self.tag_corners_check.isChecked())
        self._detector = None
        self._schedule_preview()
        # A family change re-reads every label: tag 7 of tag16h5 and tag 7 of
        # tag36h11 share a label but are different physical tags, so the names
        # and thumbnails beside each assignment have to be redrawn.
        self._refresh_assignment_table()

    def _refresh_detector_rows(self) -> None:
        """Show each option only for the detector it applies to."""
        self.tag_row.setVisible(self.detector_combo.currentData() == APRILTAG_DETECTOR)

    def _detector_params(self) -> dict:
        """Constructor arguments for the current detector, from the widgets."""
        return {
            "family": self.tag_family_combo.currentText(),
            "quad_decimate": float(self.tag_decimate_spin.value()),
            "decode_sharpening": float(self.tag_sharpening_spin.value()),
            "parts": TAG_PARTS if self.tag_corners_check.isChecked() else ("centre",),
        }

    def _current_detector(self, progress=None):
        """The detector for the current settings, rebuilt only when they change.

        Kept between runs because the live preview asks for it on every redraw.
        Raises rather than returning ``None`` — it also runs inside
        ``BusyProgressDialog``, which reports the message.
        """
        key = self.detector_combo.currentData()
        params = self._detector_params()
        built_for = (key, params)
        if self._detector is not None and self._detector_built_for == built_for:
            return self._detector
        detector = build_detector(key, **params)
        self._detector = detector
        self._detector_built_for = built_for
        return detector

    # -- assignment ----------------------------------------------------

    def _refresh_assignment_table(self) -> None:
        """Rebuild the label rows from the store's assignment table."""
        table = self.assignment_table
        invalid = self.store.assignment.invalid_labels(self.store)
        entries = self.store.assignment.entries
        table.blockSignals(True)
        table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            item = QTableWidgetItem(self._label_text(entry.label))
            item.setData(Qt.UserRole, entry.label)
            preview = self._label_icon(entry.label)
            if preview is not None:
                item.setIcon(preview)
            if entry.label in invalid:
                item.setForeground(QBrush(QColor("#d05050")))
                item.setToolTip(
                    "Skipped on a run: this target no longer exists in the schema, "
                    "or another label already owns that point."
                )
            table.setItem(row, 0, item)
            table.setCellWidget(row, 1, self._assignment_individual_combo(entry))
            table.setCellWidget(row, 2, self._assignment_keypoint_combo(entry))
            matched = QTableWidgetItem(f"{entry.matched_frames} frames" if entry.matched_frames else "—")
            matched.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
            table.setItem(row, 3, matched)
            set_by = QTableWidgetItem(_ASSIGNMENT_SOURCE_LABELS.get(entry.source, entry.source))
            set_by.setToolTip(_ASSIGNMENT_TOOLTIPS[4])
            table.setItem(row, 4, set_by)
        table.blockSignals(False)
        self._refresh_assignment_warning()

    def _label_text(self, label: int) -> str:
        """The detector's own name for a label, or the bare number without one."""
        return label_name(self._detector, label) if self._detector is not None else f"label {label}"

    def _label_icon(self, label: int):
        """Colour swatch or rendered tag for a label, if the detector offers one."""
        image = label_preview(self._detector, label) if self._detector is not None else None
        if image is None:
            return None
        array = np.ascontiguousarray(image, dtype=np.uint8)
        height, width = array.shape[:2]
        qimage = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        return QIcon(QPixmap.fromImage(qimage))

    def _assignment_individual_combo(self, entry) -> QComboBox:
        """Named individuals only — never a "the first one" entry.

        ``individual=None`` means the first individual throughout the store, but
        offering it *here* would put two spellings of one point in the same
        dropdown, and a label assigned to each would then both write to that
        point. A sidecar carrying ``None`` shows the individual it resolves to,
        and editing anything writes the name.
        """
        combo = QComboBox()
        for name in self.store.individual_names:
            combo.addItem(name, name)
        position = 0 if entry.individual is None else combo.findData(entry.individual)
        if position < 0:
            combo.addItem(f"{entry.individual} (missing)", entry.individual)
            position = combo.count() - 1
        combo.setCurrentIndex(position)
        combo.currentIndexChanged.connect(lambda _i, label=entry.label: self._on_assignment_edited(label))
        return combo

    def _assignment_keypoint_combo(self, entry) -> QComboBox:
        combo = QComboBox()
        # Filtered by the individual's own set: with an asymmetric schema, half
        # the keypoints simply do not exist on this animal.
        for name in self._keypoints_for_assignment(entry.individual):
            combo.addItem(name, name)
        position = combo.findData(entry.keypoint)
        if position < 0:
            combo.addItem(f"{entry.keypoint} (missing)", entry.keypoint)
            position = combo.count() - 1
        combo.setCurrentIndex(position)
        combo.currentIndexChanged.connect(lambda _i, label=entry.label: self._on_assignment_edited(label))
        return combo

    def _keypoints_for_assignment(self, individual: str | None) -> list[str]:
        if not self.store.individual_names:
            return list(self.store.keypoint_names)
        if individual is not None and individual not in self.store.individual_names:
            return list(self.store.keypoint_names)
        return self.store.keypoints_for(individual)

    def _assignment_row(self, label: int) -> int | None:
        for row in range(self.assignment_table.rowCount()):
            item = self.assignment_table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == label:
                return row
        return None

    def _on_assignment_edited(self, label: int) -> None:
        """A combo moved: the row becomes the user's and is never re-learned."""
        row = self._assignment_row(label)
        if row is None:
            return
        individual = self.assignment_table.cellWidget(row, 1).currentData()
        keypoint = self.assignment_table.cellWidget(row, 2).currentData()
        try:
            self.store.assignment.set(label, individual, keypoint, MANUAL)
        except AssignmentError as e:
            notify(str(e), "warning")
        self._save_store()
        # The preview colours labels by the keypoint they land on.
        self._schedule_preview()
        # Deferred: the rebuild replaces the very combo whose signal is running,
        # and Qt deletes a cell widget the moment it is swapped out.
        QTimer.singleShot(0, self._refresh_assignment_table)

    def _refresh_assignment_warning(self) -> None:
        """Name what will silently produce nothing, rather than letting it."""
        if not len(self.store.assignment):
            self.assignment_warning.setText("No labels assigned yet — press 'Learn from labels'.")
            return
        messages = []
        invalid = self.store.assignment.invalid_labels(self.store)
        if invalid:
            messages.append(
                f"{len(invalid)} label(s) point at a keypoint or individual that no longer exists, "
                "or at a point another label already owns."
            )
        # Resolved through the store, so "the first individual" and its name are
        # one row here rather than two spellings.
        claimed = set(self.store.assignment_rows().values())
        unclaimed = [
            f"{individual} · {keypoint}"
            for i, individual in enumerate(self.store.individual_names)
            for keypoint in self.store.keypoints_for(individual)
            if i * self.store.n_keypoints + self.store.keypoint_index(keypoint) not in claimed
        ]
        if unclaimed:
            messages.append("No detector label lands on: " + ", ".join(unclaimed) + ".")
        self.assignment_warning.setText("  ".join(messages))

    def _on_learn_assignment(self) -> None:
        """Propose what each label means by matching detections to the labels."""
        if not self.store.anchor_frames():
            notify("Label a few frames first — this learns by matching detections to your labels.", "warning")
            return
        busy = BusyProgressDialog("Learning what the detector's labels mean…", parent=self)

        def progress(fraction: float) -> bool:
            busy.setLabelText(f"Scanning your labelled frames… {fraction:.0%}")
            busy.pump_events()
            return not busy.wasCanceled()

        learned, error = busy.execute(self._run_learn_assignment, progress)
        if error is not None or learned is None:
            return
        taken = self.store.assignment.learn(learned.proposals)
        self._refresh_assignment_table()
        self._save_store()
        if not learned.proposals:
            notify(
                "Nothing could be matched — the detector found no marker near your labels on at "
                "least two frames. Check the detector's settings, or label the marker itself.",
                "warning",
            )
            return
        kept = len(learned.proposals) - taken
        message = f"Learned {taken} assignment(s) from {learned.frames_scanned} labelled frames."
        if kept:
            message += f" {kept} left alone (yours, or already taken)."
        if learned.unmatched_targets:
            message += f" Nothing detected for: {', '.join(k for _i, k in learned.unmatched_targets)}."
        notify(message, "info")

    def _run_learn_assignment(self, progress):
        detector = self._current_detector(progress)
        frames = self._open_frames(max_side=DETECT_MAX_SIDE)
        try:
            return learn_assignment(detector, frames, self.store, progress=progress)
        finally:
            frames.close()

    def _on_clear_assignment(self) -> None:
        if not len(self.store.assignment):
            return
        confirm = QMessageBox.question(
            self,
            "Clear assignments",
            f"Forget what all {len(self.store.assignment)} detector label(s) mean?\n"
            "Rows you edited by hand go too. Detections are not affected.",
        )
        if confirm != QMessageBox.Yes:
            return
        self.store.assignment.clear()
        self._refresh_assignment_table()
        self._save_store()

    # -- the run itself ------------------------------------------------

    def _detect_span(self) -> tuple[int, int]:
        """The frame range to scan, from the "Over:" combo."""
        n_frames = self._n_frames() or self.store.n_frames
        choice = self.detect_range_combo.currentData()
        if choice == _RANGE_LABELLED:
            labelled = self.store.anchor_frames()
            if labelled:
                return labelled[0], labelled[-1]
        elif choice == _RANGE_FILL and self.store.fill_range is not None:
            return self.store.fill_range
        return 0, max(n_frames - 1, 0)

    def _quality_min(self) -> float:
        return float(self.detect_quality_spin.value())

    def _on_quality_changed(self, value: float) -> None:
        """Retune the threshold without re-running — the run is kept in memory."""
        self.app_state.detect_quality_min = float(value)
        if self._raw_detections is None:
            return
        positions, quality, orientation = self._raw_detections
        self.store.set_detections_from_flat(positions, quality, self._quality_min(), orientation)
        self._after_detections_changed()

    def _on_run_detector(self) -> None:
        if not self.store.assignment_rows():
            notify(
                "Nothing to detect — no usable label is assigned to a keypoint. Press 'Learn from labels'.",
                "warning",
            )
            return
        if not self._n_frames():
            notify("Frame count is unknown — load a video first.", "warning")
            return
        self.store.n_frames = self._n_frames()

        busy = BusyProgressDialog("Detecting markers…", parent=self)
        cancelled = False

        def progress(fraction: float) -> bool:
            nonlocal cancelled
            busy.setLabelText(f"Detecting markers… {fraction:.0%}")
            busy.pump_events()
            cancelled = cancelled or busy.wasCanceled()
            return not cancelled

        result, error = busy.execute(self._run_detection, progress)
        if cancelled:
            # Half a run is not a worse run, it is a different one — and one the
            # user cannot tell apart from a complete one afterwards.
            notify("Detection cancelled — nothing was changed.", "info")
            return
        if error is not None or result is None:
            return

        positions, quality, orientation = result
        self._raw_detections = (positions, quality, orientation)
        found = self.store.set_detections_from_flat(positions, quality, self._quality_min(), orientation)
        self._after_detections_changed()
        first, last = self._detect_span()
        scanned = last - first + 1
        if not found:
            notify(
                "Nothing was detected. Lower the quality threshold, or check the detector's settings.",
                "warning",
            )
            return
        notify(
            f"Detected {found} point(s) on {len(self.store.detection_frames())} of {scanned} frames. "
            "Run Fill to bridge the rest.",
            "info",
        )

    def _run_detection(self, progress):
        detector = self._current_detector(progress)
        frames = self._open_frames(max_side=DETECT_MAX_SIDE)
        try:
            return run_detector(
                detector,
                frames,
                self.store.assignment_rows(),
                self.store.n_points,
                self._detect_span(),
                progress,
            )
        finally:
            frames.close()

    def _on_clear_detections(self) -> None:
        if not self.store.detections:
            return
        self.store.clear_detections()
        self._raw_detections = None
        self._after_detections_changed()
        notify("Cleared the detections; your labels are untouched.", "info")

    def _after_detections_changed(self) -> None:
        """Everything a detector run touches — the canvas, the table, the cache."""
        if self._mode is not None:
            self._mode.refresh()
        self._push_pose_override()
        self._refresh_active_label()
        self._refresh_bulk_approve_buttons()
        self._refresh_point_table(full=True)
        self._refresh_detect_summary()
        # A run is what makes a heading available (or takes it away again).
        self._refresh_head_direction_row()
        self._schedule_preview()
        self._save_detections()

    def _refresh_detect_summary(self) -> None:
        frames = len(self.store.detections)
        if not frames:
            self.detect_summary.setText("No detections. A run adds them beside your labels; it never replaces one.")
            return
        total = sum(int(np.count_nonzero(~np.isnan(points[:, :, 0]))) for points in self.store.detections.values())
        per_keypoint = np.zeros(self.store.n_keypoints, dtype=int)
        for points in self.store.detections.values():
            per_keypoint += np.count_nonzero(~np.isnan(points[:, :, 0]), axis=0)
        worst = ", ".join(
            f"{name} {count}"
            for name, count in sorted(zip(self.store.keypoint_names, per_keypoint), key=lambda item: item[1])[:3]
        )
        self.detect_summary.setText(
            f"{total} point(s) on {frames} frames — {frames * 100 / max(self.store.n_frames, 1):.0f}% "
            f"of the video. Fewest: {worst}."
        )

    # -- the detection cache -------------------------------------------

    def _detection_signature(self) -> str:
        """Identifies a run: the detector, its parameters, and where they land.

        The quality threshold is part of it, because the cache holds what was
        *kept*: lowering the threshold after reopening the dialog has to re-run,
        which is honest — the discarded detections were never written down.
        """
        payload = [
            self.detector_combo.currentData(),
            sorted(self._detector_params().items(), key=str),
            self._quality_min(),
            sorted(self.store.assignment_rows().items()),
            self.store.n_frames,
        ]
        return hashlib.sha1(json.dumps(payload, default=str).encode()).hexdigest()

    def _load_detections(self) -> None:
        """Restore a cached run for this video, if it still applies."""
        video = self._video_path()
        if not video:
            return
        try:
            loaded = self.store.load_detections(detections_path(video), self._detection_signature())
        except Exception:  # noqa: BLE001 - a cache that cannot be read is a cache miss
            logger.warning("Ignoring unreadable detections at %s", detections_path(video), exc_info=True)
            return
        if loaded:
            self._refresh_detect_summary()

    def _save_detections(self) -> None:
        video = self._video_path()
        if video:
            self.store.save_detections(detections_path(video), self._detection_signature())

    # ------------------------------------------------------------------
    # Individual / keypoint tree
    # ------------------------------------------------------------------

    @property
    def color_by(self) -> str:
        """Which axis colour encodes — the app-wide setting, validated."""
        mode = getattr(self.app_state, "pose_color_by", COLOR_BY_KEYPOINT)
        return mode if mode in COLOR_BY_MODES else COLOR_BY_KEYPOINT

    def _keypoint_brush(self, keypoint: str) -> QBrush:
        """The keypoint palette's colour for *keypoint* — pinned or generated."""
        rgba = keypoint_colors_for(self.store)[self.store.keypoint_index(keypoint)]
        return QBrush(QColor.fromRgbF(*(float(c) for c in rgba)))

    def _individual_brush(self, individual: str) -> QBrush:
        """The individual palette's colour for *individual*."""
        rgba = individual_colors_for(self.store)[self.store.individual_index(individual)]
        return QBrush(QColor.fromRgbF(*(float(c) for c in rgba)))

    def _point_brush(self, individual: str | None, keypoint: str | None) -> QBrush:
        """The colour the canvas actually draws this point in.

        The one place the colour mode is resolved for Qt widgets, so the tree,
        the chip, the pickers and the table can never say something the overlay
        does not. Falls back to the other axis when the caller only has one —
        an individual branch has no keypoint, and vice versa.
        """
        if self.color_by == COLOR_BY_INDIVIDUAL and individual is not None:
            return self._individual_brush(individual)
        if keypoint is not None:
            return self._keypoint_brush(keypoint)
        return self._individual_brush(individual) if individual is not None else QBrush()

    def _swatch(self, brush: QBrush) -> QIcon:
        """A filled square of *brush*'s colour, for the tree's name column.

        The mark in the second column is only drawn once the point is labelled
        on this frame, so a swatch is what makes a colour visible while it is
        being chosen.
        """
        pixmap = QPixmap(_SWATCH_PX, _SWATCH_PX)
        pixmap.fill(brush.color())
        return QIcon(pixmap)

    def _group_brushes(self, columns: list[str]) -> list[QBrush]:
        """Header colours for the points table's keypoint columns.

        Only while colour means keypoint: colouring the beak column orange when
        the canvas colours by individual would teach a mapping nothing draws.
        The Individual column carries the colour in that mode instead.
        """
        if self.color_by == COLOR_BY_INDIVIDUAL:
            return [QBrush() for _ in columns]
        return [self._keypoint_brush(name) for name in columns]

    def _rebuild_tree(self) -> None:
        """Recreate the branches; call whenever the schema changes.

        Every item carries a swatch of the colour the canvas draws it in, so the
        tree reads like the canvas in either colour mode: per keypoint (leaves
        differ, branches share) or per individual (branches differ, each one's
        leaves share its colour).
        """
        self.tree.blockSignals(True)
        self.tree.clear()
        for individual in self.store.individual_names:
            branch = QTreeWidgetItem([individual, ""])
            branch.setData(0, Qt.UserRole, (individual, None))
            branch.setIcon(0, self._swatch(self._point_brush(individual, None)))
            for keypoint in self.store.keypoints_for(individual):
                leaf = QTreeWidgetItem([keypoint, "", ""])
                leaf.setData(0, Qt.UserRole, (individual, keypoint))
                leaf.setIcon(0, self._swatch(self._point_brush(individual, keypoint)))
                leaf.setForeground(1, self._point_brush(individual, keypoint))
                leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
                leaf.setCheckState(2, Qt.Checked if self.store.is_static(keypoint) else Qt.Unchecked)
                branch.addChild(leaf)
            self.tree.addTopLevelItem(branch)
            branch.setExpanded(True)
        self.tree.blockSignals(False)
        self._refresh_keypoint_hints()
        self._refresh_head_direction_row()
        self._refresh_tree_marks()
        self._sync_tree_selection()
        self._refresh_active_label()
        self.app_state.labelling_keypoints = list(self.store.keypoint_names)
        self.app_state.labelling_individuals = list(self.store.individual_names)

    def _refresh_tree_marks(self) -> None:
        """Update the per-item "labelled on this frame" column."""
        frame = int(self.app_state.current_frame or 0)
        placed = self.store.anchor_positions(frame)
        for i in range(self.tree.topLevelItemCount()):
            branch = self.tree.topLevelItem(i)
            labelled = 0
            for row in range(branch.childCount()):
                leaf = branch.child(row)
                # A leaf's row is not the schema index once individuals carry
                # different keypoints — always resolve through the name.
                keypoint = leaf.data(0, Qt.UserRole)[1]
                is_set = not np.isnan(placed[i, self.store.keypoint_index(keypoint), 0])
                leaf.setText(1, _LABELLED_MARK if is_set else _UNLABELLED_MARK)
                labelled += int(is_set)
            branch.setText(1, f"{labelled}/{branch.childCount()}")

    def _sync_tree_selection(self) -> None:
        """Highlight the item the canvas is currently writing to."""
        if self._mode is None:
            return
        target = (self._mode.active_individual, self._mode.active_keypoint)
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            branch = self.tree.topLevelItem(i)
            for k in range(branch.childCount()):
                leaf = branch.child(k)
                if leaf.data(0, Qt.UserRole) == target:
                    self.tree.setCurrentItem(leaf)
                    self.tree.blockSignals(False)
                    return
        self.tree.blockSignals(False)

    def _on_tree_static_toggled(self, item: QTreeWidgetItem, column: int) -> None:
        """The Static checkbox: the same keypoint is static on every individual."""
        if column != 2 or item.parent() is None:
            return
        _individual, keypoint = item.data(0, Qt.UserRole)
        static = item.checkState(2) == Qt.Checked
        if static == self.store.is_static(keypoint):
            return
        self.store.set_static(keypoint, static)
        self._rebuild_tree()  # every individual's leaf shows the same state
        self._on_store_changed(full=True)

    def _on_tree_item_changed(self, item: QTreeWidgetItem | None, _previous=None) -> None:
        if item is None or self._mode is None:
            return
        individual, keypoint = item.data(0, Qt.UserRole)
        if keypoint is None:
            self._mode.set_active_individual(individual)
        else:
            self._mode.set_active(keypoint, individual)
        self._refresh_active_label()

    # ------------------------------------------------------------------
    # Schema editing
    # ------------------------------------------------------------------

    def _on_add_individual(self) -> None:
        suggestion = f"individual_{self.store.n_individuals}"
        name, ok = QInputDialog.getText(self, "Add individual", "Individual name:", text=suggestion)
        name = name.strip()
        if not ok or not name:
            return
        if name in self.store.individual_names:
            notify(f"Individual {name!r} already exists.", "warning")
            return
        self._apply_schema(individuals=[*self.store.individual_names, name])

    def _on_remove_individual(self) -> None:
        """Remove the selected individual — including the last one.

        Deleting every individual is allowed: with none left nothing can be
        labelled, which is exactly the state to start renaming from.
        """
        individual = self._selected_individual()
        if individual is None:
            notify("There are no individuals to remove.", "warning")
            return
        labelled = self.store.anchor_frames_for_individual(individual)
        if labelled:
            confirm = QMessageBox.question(
                self,
                "Remove individual",
                f"{individual!r} is labelled on {len(labelled)} frame(s).\n"
                "Removing it discards those points. Continue?",
            )
            if confirm != QMessageBox.Yes:
                return
        self._apply_schema(individuals=[n for n in self.store.individual_names if n != individual])

    def _on_shared_toggled(self, shared: bool) -> None:
        self._apply_schema(shared=shared)

    def _on_add_keypoint(self) -> None:
        name, ok = QInputDialog.getText(self, "Add keypoint", "Keypoint name:")
        name = name.strip()
        if not ok or not name:
            return
        if self.store.shared_keypoints:
            if name in self.store.keypoint_names:
                notify(f"Keypoint {name!r} already exists.", "warning")
                return
            self._apply_schema(keypoints=[*self.store.keypoint_names, name])
            return
        individual = self._require_individual()
        if individual is None:
            return
        if name in self.store.keypoints_for(individual):
            notify(f"{individual!r} already has keypoint {name!r}.", "warning")
            return
        self._apply_schema(individual_keypoints=(individual, [*self.store.keypoints_for(individual), name]))

    def _on_remove_keypoint(self) -> None:
        keypoint = self._selected_keypoint()
        if keypoint is None:
            notify("Select a keypoint to remove.", "warning")
            return
        if self.store.shared_keypoints:
            self._apply_schema(keypoints=[n for n in self.store.keypoint_names if n != keypoint])
            return
        individual = self._selected_individual()
        remaining = [n for n in self.store.keypoints_for(individual) if n != keypoint]
        self._apply_schema(individual_keypoints=(individual, remaining))

    def _on_keypoint_color(self) -> None:
        """Pin a colour on whichever axis is currently being drawn.

        The picker edits what the canvas shows: the selected keypoint's colour
        while colour means keypoint, the selected individual's while it means
        individual. Editing the invisible palette would be a control with no
        visible effect.
        """
        if self.color_by == COLOR_BY_INDIVIDUAL:
            individual = self._selected_individual()
            if individual is None:
                notify("Select an individual to colour.", "warning")
                return
            chosen = QColorDialog.getColor(self._individual_brush(individual).color(), self, f"Colour for {individual}")
            if not chosen.isValid():  # the user cancelled
                return
            self.store.set_individual_color(individual, chosen.name())
            self._apply_keypoint_colors()
            return
        keypoint = self._selected_keypoint()
        if keypoint is None:
            notify("Select a keypoint to colour.", "warning")
            return
        chosen = QColorDialog.getColor(self._keypoint_brush(keypoint).color(), self, f"Colour for {keypoint}")
        if not chosen.isValid():  # the user cancelled
            return
        self.store.set_keypoint_color(keypoint, chosen.name())
        self._apply_keypoint_colors()

    def _on_reset_keypoint_colors(self) -> None:
        self.store.clear_keypoint_colors()
        self._apply_keypoint_colors()

    def _on_color_by_changed(self, _index: int) -> None:
        """Switch the colour axis — app-wide, so the pose overlay follows too."""
        mode = self.color_by_combo.currentData()
        if mode == self.color_by:
            return
        self.app_state.pose_color_by = mode
        self.apply_color_by()
        self._data_widget.update_pose()

    def apply_color_by(self) -> None:
        """Re-read ``app_state.pose_color_by`` and repaint. Also called from the sidebar."""
        with _blocked(self.color_by_combo):
            index = self.color_by_combo.findData(self.color_by)
            self.color_by_combo.setCurrentIndex(index if index >= 0 else 0)
        if self._mode is not None:
            self._mode.set_color_by(self.color_by)
        self._repaint_colors()

    def _apply_keypoint_colors(self) -> None:
        """Repaint everything that draws a point in its colour, and save.

        Not a schema change: the arrays, the fill and the active pair are all
        untouched, so the canvas mode is recoloured in place rather than
        restarted. A pinned colour *is* project data, so it is saved with the
        anchors — unlike the colour *mode*, which is a viewing preference.
        """
        if self._mode is not None:
            self._mode.refresh_colors()
        self._repaint_colors()
        self._save_store()

    def _repaint_colors(self) -> None:
        """Rebuild the Qt surfaces that carry a colour — tree, table header, pickers."""
        self._rebuild_tree()
        header = self.point_table.horizontalHeader()
        header.set_groups(header.groups(), self._group_brushes(header.groups()))
        self.point_model.set_individual_brush(self._individual_brush if self.color_by == COLOR_BY_INDIVIDUAL else None)
        self._refresh_active_label()

    def _require_individual(self) -> str | None:
        """The individual per-individual edits apply to, warning when there is none."""
        individual = self._selected_individual()
        if individual is None:
            notify("Add an individual first — keypoints belong to one when they are not shared.", "warning")
        return individual

    def _selected_individual(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return self.store.individual_names[0] if self.store.individual_names else None
        return item.data(0, Qt.UserRole)[0]

    def _selected_keypoint(self) -> str | None:
        item = self.tree.currentItem()
        return None if item is None else item.data(0, Qt.UserRole)[1]

    def _apply_schema(
        self,
        keypoints: list[str] | None = None,
        individuals: list[str] | None = None,
        shared: bool | None = None,
        individual_keypoints: tuple[str, list[str]] | None = None,
    ) -> None:
        """Change the schema; any existing fill is invalidated by the store."""
        if shared is not None:
            self.store.set_shared_keypoints(shared)
        if individuals is not None:
            self.store.set_individual_names(individuals)
        if keypoints is not None:
            self.store.set_keypoint_names(keypoints)
        if individual_keypoints is not None:
            self.store.set_keypoints_for(*individual_keypoints)
        self._push_pose_override()
        if self._mode is not None:
            if self.store.n_individuals:
                self._restart_mode()
            else:
                # Nothing left to label — drop out of labelling mode rather than
                # leaving a canvas that silently swallows clicks.
                self.set_interaction_mode(None)
        self._rebuild_tree()
        # An assignment can now name a keypoint that is gone, or an individual
        # that is new — the table shows both, so it has to be rebuilt here.
        self._refresh_assignment_table()
        # The individuals a filter names may not exist any more, and a filter
        # nobody can see the cause of looks like a broken table.
        self._clear_filters()
        self._refresh_point_table(full=True)
        self._save_store()

    # ------------------------------------------------------------------
    # Labelling mode
    # ------------------------------------------------------------------

    @property
    def interaction_mode(self) -> str | None:
        """``"label"``, ``"edit"``, or ``None`` when the canvas is not armed."""
        return None if self._mode is None else self._mode.mode

    def set_interaction_mode(self, mode: str | None) -> None:
        """Arm the canvas for labelling or editing, or disarm it (``None``)."""
        if mode is not None:
            # A mode button pressed on the Calibrate tab: the calibration mode
            # must let go of the canvas first, without re-arming the labelling
            # mode itself — that is exactly what this call is about to do.
            self._exit_calibrate_mode(restore=False)
        if mode is not None and not self._can_label():
            mode = None
        if mode is None:
            self._detach_mode()
        elif self._mode is None:
            self._attach_mode(mode)
        else:
            self._mode.set_mode(mode)
        if mode is not None:
            # Arming from the Keypoints tab would otherwise hide the line that
            # says what the next click places.
            self.tabs.setCurrentWidget(self._label_page)
        self._sync_mode_buttons()
        # After the tab switch above, so the lock is decided against the tab
        # actually showing. Refreshes the status chip on the way through.
        self._apply_lock()

    def _can_label(self, quiet: bool = False) -> bool:
        """Whether the canvas can be armed, warning about whatever is missing.

        ``quiet`` suppresses the warnings: arming happens automatically when the
        labelling tab is opened, and a user who has not defined a schema yet
        should not be scolded for looking at the tab.
        """
        if not self.store.n_individuals:
            if not quiet:
                notify("Add an individual before labelling.", "warning")
            return False
        if not self.store.keypoint_names:
            if not quiet:
                notify("Add at least one keypoint before labelling.", "warning")
            return False
        if self._view.scene() is None:
            if not quiet:
                notify("Load a video (or a still frame) before labelling.", "warning")
            return False
        return True

    def _on_tab_changed(self, index: int) -> None:
        """Arm Sequential when the labelling tab is opened with nothing armed.

        The Calibrate tab is the one exception to "the armed mode survives a
        tab switch": it needs the pointer for its own clicks, so entering it
        suspends the labelling mode and attaches the calibration mode, and
        leaving reverses that — restoring the labelling mode with its active
        pair, locked or not per :meth:`_lock_wanted` as usual.
        """
        current = self.tabs.widget(index)
        if current is not self._calibrate_page:
            self._exit_calibrate_mode()
        # Applies to every tab: away from Label, the pointer goes back to the
        # camera (see `_lock_wanted`). The armed mode itself survives, so
        # coming back carries on where it left off.
        self._apply_lock()
        if current is self._calibrate_page:
            self._enter_calibrate_mode()
            return
        if current is self._detect_page:
            # Nothing was drawn while the tab was hidden, so opening it is the
            # first chance to spend a decode on the preview.
            self._schedule_preview()
            return
        if current is not self._label_page or self._mode is not None:
            return
        if self._can_label(quiet=True):
            self.set_interaction_mode(SEQUENTIAL_MODE)

    def _sync_mode_buttons(self) -> None:
        mode = self.interaction_mode
        self.sequential_btn.setChecked(mode == SEQUENTIAL_MODE)
        self.loop_btn.setChecked(mode == LOOP_MODE)
        # There is nothing to lock while the canvas is not armed — clicks pan
        # already — but the tick is remembered for the next time it is.
        self.lock_check.setEnabled(mode is not None)

    def _attach_mode(self, mode: str = SEQUENTIAL_MODE) -> None:
        self.store.n_frames = self._n_frames() or self.store.n_frames
        self._mode = KeypointLabelMode(
            self._view,
            self.store,
            on_changed=self._on_store_changed,
            mode=mode,
            on_advance_frame=self._advance_frame,
            point_size=float(self.app_state.labelling_point_size),
            on_released=self._on_store_changed,
            locked=self._lock_wanted(),
            color_by=self.color_by,
        )
        self._mode.set_frame(int(self.app_state.current_frame or 0))
        self._install_key_filter(True)
        # The anchor overlay now draws the fill too, so the pose overlay must
        # stop drawing it or every point gets two markers.
        self._push_pose_override()
        self._sync_tree_selection()
        self._refresh_active_label()

    def _detach_mode(self) -> None:
        if self._mode is None:
            return
        # The key filter stays: Backspace and Ctrl+Z go on working on whatever
        # the Keypoints tree has selected once labelling is disarmed.
        self._mode.detach()
        self._mode = None
        # Hand the fill back to the pose overlay, which is what shows it once
        # the anchor overlay is gone.
        self._push_pose_override()
        self._save_store()
        self._sync_mode_buttons()
        self._refresh_active_label()

    def _restart_mode(self) -> None:
        mode = self.interaction_mode or SEQUENTIAL_MODE
        keypoint = self._mode.active_keypoint if self._mode else None
        individual = self._mode.active_individual if self._mode else None
        self._detach_mode()
        self._attach_mode(mode)
        self._sync_mode_buttons()
        if individual not in self.store.individual_names:
            individual = None
        if keypoint is not None and self.store.has_keypoint(keypoint, individual):
            self._mode.set_active(keypoint, individual)

    def _on_store_changed(self, full: bool = False, frame: int | None = None) -> None:
        self._refresh_tree_marks()
        self._sync_tree_selection()
        self._refresh_active_label()
        # Here rather than in `_refresh_active_label`: what the bulk approvals
        # need is a detector run or a fill, neither of which is a mode change.
        self._refresh_bulk_approve_buttons()
        self._refresh_point_table(full=full, frame=frame)
        # A correction has to reach the pose overlay too, or the prediction it
        # replaced stays on screen next to it. Not while a drag is running: the
        # poses dataset is rebuilt whole, which is far too much per mouse move,
        # and the release fires this again once the point has settled.
        if self.store.filled is not None and not (self._mode is not None and self._mode.dragging):
            self._push_pose_override()

    def _on_frame_changed(self, frame: int) -> None:
        if self._mode is not None:
            self._mode.set_frame(int(frame))
        if self._calib_mode is not None:
            self._calib_mode.set_frame(int(frame))
            self._select_calib_frame_row()
        self._refresh_tree_marks()
        self._select_table_row_for_frame()
        self._schedule_preview()

    def _on_undo(self) -> None:
        # An undo can land on any frame, not the one being viewed, so the table
        # is repainted where the change actually happened. Through the common
        # path, so an undone correction also reaches the pose overlay rather
        # than leaving the point it replaced on screen.
        frame = self.store.undo()
        if self._mode is not None:
            self._mode.refresh()
        self._on_store_changed(frame=frame)

    # ------------------------------------------------------------------
    # Labelled-points table
    # ------------------------------------------------------------------

    def _table_layout(self) -> tuple[list[tuple[int, str]], list[str]]:
        """``(rows, keypoint columns)`` for what the store currently holds.

        A row is one ``(frame, individual)``, so every keypoint of that
        individual on that frame is visible at once.

        Before a fill only labelled rows exist, and columns cover only the
        keypoints carrying at least one label — an unlabelled 20-keypoint schema
        would otherwise push the ones being worked on off the side. Once a fill
        exists every frame *it covers* has a position for every point, so each of
        those gets a row: a prediction is then visible, filterable and
        correctable from here and not only on the canvas. That is the labelled
        span rather than the whole video — a fill stops at the outermost labels,
        so rows outside it would be empty ones nothing can ever populate.
        """
        span = self.store.fill_range
        if span is not None:
            rows = [
                (frame, individual)
                for frame in range(span[0], span[1] + 1)
                for individual in self.store.individual_names
            ]
            return rows, list(self.store.keypoint_names)
        # No fill: one row per *observed* ``(frame, individual)`` — labelled or
        # detected, since a detection is a position on a frame exactly as a
        # label is, and the same rows are what the next fill will be built from.
        observed = sorted(set(self.store.anchor_frames()) | set(self.store.detection_frames()))
        rows: list[tuple[int, str]] = []
        seen = np.zeros(self.store.n_keypoints, dtype=bool)
        for frame in observed:
            present = ~np.isnan(self.store.observation_positions(frame)[:, :, 0])
            seen |= present.any(axis=0)
            rows.extend((frame, name) for i, name in enumerate(self.store.individual_names) if present[i].any())
        return rows, [name for name, keep in zip(self.store.keypoint_names, seen) if keep]

    def _layout_signature(self) -> tuple:
        """What :meth:`_table_layout` depends on, cheap enough to test per drag.

        With a fill loaded the row set is one row per covered frame per
        individual, so it must not be rebuilt — or even enumerated — on every
        mouse move of a drag; ``fill_range`` is two numbers and stands in for
        all of them.
        """
        if self.store.fill_range is not None:
            return (
                True,
                self.store.fill_range,
                tuple(self.store.individual_names),
                tuple(self.store.keypoint_names),
            )
        # A detector run is the same problem as a fill: it can add a row per
        # frame of the video, so it is summarised by its revision counter rather
        # than enumerated here. The labelled rows stay explicit — there are few
        # of them, and they change one point at a time.
        points = self.store.labelled_points()
        rows = tuple(dict.fromkeys((frame, individual) for frame, individual, _kp, _x, _y in points))
        labelled = tuple(sorted({keypoint for _frame, _individual, keypoint, _x, _y in points}))
        return (False, rows, labelled, self.store.detections_revision, tuple(self.store.keypoint_names))

    def _refresh_point_table(self, full: bool = False, frame: int | None = None) -> None:
        """Sync the table with the store.

        Values are read straight from the store by the model, so a refresh only
        has to say *what changed*: normally one frame (a placement or a drag,
        which fires on every mouse move), or everything after a fill. *frame*
        defaults to the one being edited — pass it only when an edit landed
        somewhere else, as an undo can.
        """
        signature = self._layout_signature()
        if signature != self._table_signature:
            self._table_signature = signature
            rows, columns = self._table_layout()
            self.point_model.set_layout(rows, columns)
            self.point_table.horizontalHeader().set_groups(columns, self._group_brushes(columns))
        elif full:
            self.point_model.refresh_all()
        else:
            self.point_model.refresh_frame(self._current_frame() if frame is None else frame)
        self._select_table_row_for_frame()

    def _current_frame(self) -> int:
        """The frame edits land on — the canvas mode's, else the playhead's."""
        return self._mode.frame if self._mode is not None else int(self.app_state.current_frame or 0)

    def _select_table_row_for_frame(self) -> None:
        """Highlight the playhead's first visible row, without seeking."""
        frame = int(self.app_state.current_frame or 0)
        for individual in self.store.individual_names or [None]:
            source_row = self.point_model.row_of((frame, individual))
            if source_row is None:
                continue
            index = self.point_proxy.mapFromSource(self.point_model.index(source_row, 0))
            if index.isValid():
                self.point_table.selectRow(index.row())
                self.point_table.scrollTo(index)
                return
        self.point_table.clearSelection()

    def _on_table_clicked(self, index) -> None:
        """Seek to the clicked row's frame and make what was clicked active."""
        key = self.point_model.key_at(self.point_proxy.mapToSource(index).row())
        if key is None:
            return
        frame, individual = key
        self._seek(frame)
        if self._mode is None or individual not in self.store.individual_names:
            return
        target = self.point_model.keypoint_at(index.column())
        if target is not None and self.store.has_keypoint(target[0], individual):
            self._mode.set_active(target[0], individual)
        else:
            self._mode.set_active_individual(individual)
        self._sync_tree_selection()
        self._refresh_active_label()

    def _seek(self, frame: int) -> None:
        video = self.app_state.video
        if video is None:
            notify("No video is loaded to seek.", "warning")
            return
        video.seek_to_frame(int(frame))

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _install_key_filter(self, install: bool) -> None:
        """Catch the dialog's keys wherever they are actually pressed.

        Three targets, because focus moves between windows while labelling: the
        dialog, the **video canvas** (which belongs to the main window, so
        clicking to place a point takes focus out of the dialog) and the **main
        window** itself (key events propagate up to it from any widget inside).
        Without the third, the keys pressed while looking at the video —
        ``Backspace``, ``Ctrl+Z``, ``Shift+H``, ``N`` — did nothing at all, or
        hit whatever the main window binds instead.

        The canvas target is :meth:`CameraView.key_target`, **not** the widget
        the canvas is laid out as: rendercanvas nests the render widget that
        takes focus inside a wrapper, and that inner widget swallows every key
        press. Filtering the wrapper saw nothing, so Backspace did nothing at
        the one moment it is wanted — right after clicking the video.

        Only those few keys are claimed from the main window — see
        :meth:`_owned_key`. Everything else there belongs to whatever the user
        is doing over there.

        Installed for the dialog's whole lifetime, not only while a mode is
        armed: deleting the keypoint you have selected must not require arming
        labelling first. Re-installing is safe (Qt drops the earlier
        registration of the same filter) and is what follows the canvas when
        another video replaces it; the canvas only exists once one is loaded.
        """
        for widget in (self, self._view.key_target(), self._shell):
            if widget is None:
                continue
            if install:
                widget.installEventFilter(self)
            else:
                widget.removeEventFilter(self)
        if install and not self._shortcuts:
            self._bind_shortcuts()

    def _on_video_changed(self, _path=None) -> None:
        """Follow the canvas: another video builds a new render widget.

        Deferred, because the signal fires from ``app_state`` and the view has
        not necessarily swapped its canvas yet.
        """
        # The held-open preview source belongs to the video that is going away.
        self._close_preview_frames()
        self._switch_clip()
        # The calibration mode holds pygfx objects in the scene being replaced;
        # rebuilt (deferred, like the key filter) if the tab is still open.
        self._exit_calibrate_mode(restore=False)
        QTimer.singleShot(0, self._sync_calibrate_mode)
        QTimer.singleShot(0, self._reinstall_key_filter)
        self._schedule_preview()

    def _switch_clip(self) -> None:
        """Another clip of the same camera: save this store, load that clip's.

        A session cut into trials is labelled clip by clip without closing the
        dialog — the schema and the static keypoints carry over, the labels
        are each clip's own. Nothing happens when the path did not change (a
        reload of the same video).
        """
        video = self._video_path()
        if video == self._store_video:
            return
        if self._store_video:
            path = sidecar_path(self._store_video)
            self.store.save(path)
            _LAST_SAVED_SIDECAR["path"] = str(path)
            self._write_template(self._store_video)
        self.store = self._load_store()
        self._store_video = video
        if self._mode is not None:
            self._mode.store = self.store
        self._rebuild_tree()
        self._on_store_changed(full=True)
        self._refresh_clip_counter()

    # -- many clips, one camera ---------------------------------------------

    def _clip_videos(self) -> list[tuple[object, str]]:
        """``(trial, video path)`` for every trial the trials table shows, in order."""
        alignment = getattr(self.app_state, "nwb_alignment", None)
        camera = getattr(self.app_state, "primary_camera", None)
        if alignment is None or not camera:
            return []
        out = []
        for trial in list(self.app_state.trials or []):
            path = alignment.resolve_media_path(
                trial, "video", device=camera, fallback_folder=self.app_state.video_folder
            )
            if path:
                out.append((trial, str(path)))
        return out

    def _refresh_clip_counter(self) -> None:
        clips = self._clip_videos()
        if not clips:
            self.clips_label.setText("")
            self.next_clip_btn.setEnabled(False)
            return
        labelled = sum(1 for _, v in clips if sidecar_path(v).exists())
        self.clips_label.setText(f"Clips labelled: {labelled} / {len(clips)}")
        self.next_clip_btn.setEnabled(labelled < len(clips))

    def _on_next_unlabelled_clip(self) -> None:
        """Jump to the next trial (after this one, wrapping) whose clip has no sidecar."""
        clips = self._clip_videos()
        if not clips:
            notify("No trials with a video on this camera.", "warning")
            return
        current = self.app_state.trials_sel
        order = [t for t, _ in clips]
        start = order.index(current) + 1 if current in order else 0
        for k in range(len(clips)):
            trial, video = clips[(start + k) % len(clips)]
            if not sidecar_path(video).exists():
                self._data_widget.navigation_widget.navigate_to_trial(trial)
                return
        notify("Every clip on this camera has labels.", "info")

    def _sync_calibrate_mode(self) -> None:
        """Re-attach the calibration mode if its tab is (still) the open one."""
        if self.tabs.currentWidget() is self._calibrate_page:
            self._enter_calibrate_mode()

    def _reinstall_key_filter(self) -> None:
        self._install_key_filter(True)

    def _bind_shortcuts(self) -> None:
        """Bind the keys an item view would otherwise eat, as real shortcuts.

        The event filter cannot have them, because it sits on the dialog rather
        than on the tree and the table and they never let the key propagate:
        Qt turns **Tab** into focus navigation inside ``QWidget::event`` of
        whatever holds focus, and ``QAbstractItemView`` turns any **printable**
        key — ``Shift+H`` and ``N`` included — into a keyboard type-ahead search
        and accepts it. Both silently did nothing whenever focus was in the tree or
        the table, which is most of the time: you pick the frame to review by
        clicking its row. Shortcuts are dispatched *before* the key press
        reaches the focus widget, so they win.

        Window context, so they are live anywhere in the dialog but never steal
        the key from the main window — where the ``KeyPress`` branch of
        :meth:`eventFilter` handles it instead, this dialog not being active.
        """
        bindings = [
            (QKeySequence(Qt.Key_Tab), lambda: self._cycle_keypoint(1)),
            (QKeySequence(Qt.SHIFT | Qt.Key_Tab), lambda: self._cycle_keypoint(-1)),
            (QKeySequence(Qt.Key_Backtab), lambda: self._cycle_keypoint(-1)),
            (QKeySequence(Qt.SHIFT | Qt.Key_H), self._approve_frame),
            (QKeySequence(Qt.Key_N), self._next_suggestion),
        ]
        for sequence, slot in bindings:
            shortcut = QShortcut(sequence, self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

    def _cycle_keypoint(self, step: int) -> None:
        if self._mode is None:
            return
        self._mode.cycle(step)
        self._on_store_changed()

    def _owned_key(self, event, main_window: bool = False) -> bool:
        """Keys this dialog consumes, given what is armed and where they landed.

        Deletion and undo always count; the target keys only while a mode runs,
        so the main window keeps its `1`-`9` behaviour labels. Events that came
        from the *main window* yield everything except the few keys pressed
        while looking at the video: the user is working over there.
        """
        key = event.key()
        # `N` jumps to the next suggested frame. Nothing in the main window
        # binds it (the behaviour labels are 1-9, 0, QWERTZUIOP and ASDFGHJKL),
        # so it reaches across without taking anything away — and the arrows
        # stay entirely the main window's, single-frame stepping included.
        if key == Qt.Key_N and not event.modifiers():
            return not self._typing()
        # Deleting and undoing reach across from the main window too: you press
        # them right after clicking the video, and whether that click left focus
        # on the canvas or on some other panel is not something to have to know.
        # Neither key is bound over there, so nothing is taken away.
        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            return not self._typing()
        if key == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            return True
        # Shift+H approves the frame on screen, so it reaches across for the same
        # reason: you press it while looking at the video. The main window binds
        # plain `H` (behaviour label 26), which Shift+H does not match.
        if key == Qt.Key_H and self._shift_only(event):
            return not self._typing()
        if main_window:
            return False
        if self._mode is None:
            return False
        if key in (Qt.Key_Tab, Qt.Key_Backtab):
            return True
        return Qt.Key_1 <= key <= Qt.Key_9 and not event.modifiers()

    @staticmethod
    def _shift_only(event) -> bool:
        modifiers = event.modifiers()
        return bool(modifiers & Qt.ShiftModifier) and not modifiers & (Qt.ControlModifier | Qt.AltModifier)

    def _typing(self) -> bool:
        """Whether focus sits in a text field, where Backspace means editing.

        The spin boxes on this dialog are the ones that matter: eating their
        Backspace would delete a keypoint instead of a digit.
        """
        return isinstance(QApplication.focusWidget(), (QAbstractSpinBox, QLineEdit))

    def eventFilter(self, obj, event):
        # Type first: this filter sits on the main window too, so every event
        # there passes through here and must cost as little as possible.
        if event.type() not in (QEvent.ShortcutOverride, QEvent.KeyPress):
            return False
        # The main window binds 1-9 (behaviour labels) as QShortcuts. Accepting
        # the ShortcutOverride keeps those from swallowing the key, so the
        # KeyPress below still reaches us.
        if not self._owned_key(event, main_window=obj is self._shell):
            return False
        key = event.key()
        if event.type() == QEvent.ShortcutOverride:
            # The keys we bind ourselves are the exception: this dialog's own
            # QShortcut is what makes them work at all (see _bind_shortcuts),
            # and accepting the override here would suppress that shortcut
            # exactly like any other — leaving the key press to the focused tree
            # or table, which turn it into focus navigation and type-ahead
            # search respectively. Declining lets our shortcut run.
            if key in (Qt.Key_Tab, Qt.Key_Backtab, Qt.Key_H, Qt.Key_N):
                return False
            event.accept()
            return True
        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            return self._delete_selected_point()
        if key == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self._on_undo()
            return True
        if key == Qt.Key_H:
            self._approve_frame()
            return True
        if key == Qt.Key_N:
            self._next_suggestion()
            return True
        if key in (Qt.Key_Tab, Qt.Key_Backtab):
            self._cycle_keypoint(1 if key == Qt.Key_Tab else -1)
            return True
        if self._mode.select_individual_by_number(key - Qt.Key_1 + 1):
            self._on_store_changed()
            return True
        return False

    def _delete_selected_point(self) -> bool:
        """Backspace/Delete: remove the selected point from the current frame.

        With the canvas armed that is the outlined point, falling back to the
        one under the cursor. Otherwise it is whatever the Keypoints tree
        highlights — the same pair the tree, the combos and the table selection
        all agree on — so a keypoint can be deleted without arming a mode.

        Note that a *filled* point cannot be deleted, only its label: with a
        fill loaded the prediction stays on screen (dimmed, and its table row
        turns to ``Fill``), which is correct — the next fill is free to place it
        again, unconstrained by the label you removed.
        """
        if self._calib_mode is not None:
            # Calibrating: the key removes the active landmark's click on this
            # frame. The fan-out (sidecar save included) runs via on_changed.
            return self._calib_mode.delete_active()
        if self._mode is not None:
            deleted = self._mode.delete_selected()
        else:
            frame = self._current_frame()
            individual, keypoint = self._selected_individual(), self._selected_keypoint()
            deleted = keypoint is not None and self.store.is_anchor(frame, keypoint, individual)
            if deleted:
                self.store.clear_point(frame, keypoint, individual)
                self._on_store_changed(frame=frame)
        if deleted:
            self._save_store()
        return deleted

    # ------------------------------------------------------------------
    # Fill
    # ------------------------------------------------------------------

    def _on_backend_changed(self, _index: int) -> None:
        self.app_state.labelling_backend = self.backend_combo.currentData()
        self._refresh_backend_rows()

    def _on_disagreement_changed(self, value: float) -> None:
        self.app_state.labelling_disagreement_px = float(value)

    def _on_checkpoint_edited(self) -> None:
        self.app_state.labelling_cotracker_checkpoint = self.checkpoint_edit.text().strip()

    def _on_browse_checkpoint(self) -> None:
        start = self.app_state.labelling_cotracker_checkpoint or str(cotracker_checkpoint_dir())
        path, _ = QFileDialog.getOpenFileName(self, "CoTracker3 weights", start, "Checkpoint (*.pth *.pt)")
        if not path:
            return
        self.checkpoint_edit.setText(path)
        self._on_checkpoint_edited()

    def _refresh_backend_rows(self) -> None:
        """Show each option only for the backends it actually applies to."""
        key = self.backend_combo.currentData()
        self.disagreement_row.setVisible(key in _TRACKING_BACKENDS)
        # Custom weights and the fit are both PosePAL's, since it is the only
        # backend loading a CoTracker3 state dict at all.
        self.checkpoint_row.setVisible(key == POSEPAL_BACKEND)
        self.refinement_row.setVisible(key == POSEPAL_BACKEND)
        if key == POSEPAL_BACKEND:
            self._refresh_refinement_status()

    # ------------------------------------------------------------------
    # Test-time refinement
    # ------------------------------------------------------------------

    def _refinement_signature(self) -> str:
        """Identifies the labels a fit was made from.

        Every anchor goes in, so labelling one more frame marks the fit stale —
        it stays perfectly usable, it is simply no longer the best fit available.
        The schema goes in too: the delta is indexed by point row, so renaming or
        adding a keypoint makes an old fit meaningless rather than merely dated.
        """
        payload = [self.store.keypoint_names, self.store.individual_names]
        for frame in sorted(self.store.anchors):
            payload.append([frame, np.round(self.store.anchors[frame], 3).tolist()])
        return hashlib.sha1(json.dumps(payload).encode()).hexdigest()

    def _refinement_status_text(self) -> str:
        """What the next fill will spend its time on — never merely that a fit exists.

        The wait is what the user is deciding about, and it is the fit: whether
        the next fill costs minutes or seconds is exactly whether this fit still
        matches the labels.
        """
        backend = self._refined_backend
        refinement = getattr(backend, "refinement", None)
        if refinement is None or not refinement.fitted:
            return "Not fitted — the next fill fits first (a few minutes), then tracks."
        frames = refinement.n_anchor_frames
        if refinement.matches(self._refinement_signature()):
            return f"Fitted on {frames} labelled frames — the next fill only tracks."
        return (
            f"Fitted on {frames} labelled frames, but your labels changed since — "
            "the next fill fits again first (a few minutes)."
        )

    def _refresh_refinement_status(self) -> None:
        self.refinement_status.setText(self._refinement_status_text())

    def _refined_backend_for(self, progress):
        """Build the refined backend, or hand back the one already loaded.

        Rebuilt only when something it was constructed around changed — the
        weights or the point-row count — since rebuilding drops the fit.
        """
        checkpoint = self.app_state.labelling_cotracker_checkpoint or None
        built_for = (checkpoint, self.store.n_points)
        if self._refined_backend is None or self._refined_built_for != built_for:
            self._refined_backend = build_backend(
                POSEPAL_BACKEND,
                checkpoint=checkpoint,
                progress=progress,
                disagreement_px=float(self.app_state.labelling_disagreement_px),
                n_points=self.store.n_points,
            )
            self._refined_built_for = built_for
            self._refined_video = None
        if self._refined_video != self._video_path():
            # A fit describes one video's pixels; carrying it to the next one
            # would be worse than not fitting. The signature cannot catch this —
            # it is made of labels, which a copied sidecar can match exactly.
            self._refined_backend.refinement.clear()
            self._refined_video = self._video_path()
            self._load_refinement(self._refined_backend)
        self._refined_backend.disagreement_px = float(self.app_state.labelling_disagreement_px)
        return self._refined_backend

    def _load_refinement(self, backend) -> None:
        """Restore a saved fit for this video, if one still applies."""
        video = self._video_path()
        if not video:
            return
        path = refinement_path(video)
        if not path.is_file():
            return
        try:
            backend.refinement.load(path, self._refinement_signature())
        except Exception:  # noqa: BLE001 - a cache that cannot be read is a cache miss
            logger.warning("Ignoring unreadable refinement at %s", path, exc_info=True)

    def _save_refinement(self, backend) -> None:
        video = self._video_path()
        if video and backend.refinement.fitted:
            backend.refinement.save(refinement_path(video))

    def _ready_to_fill(self) -> bool:
        if not self.store.anchor_frames() and not self.store.detection_frames():
            notify("Label at least one frame (or run a detector) before filling.", "warning")
            return False
        n_frames = self._n_frames()
        if not n_frames:
            notify("Frame count is unknown — load a video first.", "warning")
            return False
        self.store.n_frames = n_frames
        return True

    def _on_fill(self) -> None:
        if not self._ready_to_fill():
            return
        key = self.backend_combo.currentData()
        label = self.backend_combo.currentText()
        busy = BusyProgressDialog(f"Filling frames with {label}…", parent=self)
        # A refined fill is two phases and spends most of its wait in the first,
        # so the backend renames the stage as it goes rather than claiming to be
        # filling throughout.
        stage: str | None = None

        def set_stage(text: str) -> None:
            nonlocal stage
            stage = text

        # A cancelled fill must leave the previous one alone. Backends answer a
        # cancel with the spline seed they started from — they have arrays to
        # return — so applying that result would trade a fill the user liked for
        # a plain interpolation they never asked for, and with PosePAL the wait
        # they are cancelling out of is usually the fit.
        cancelled = False

        def report(default: str):
            def progress(fraction: float) -> bool:
                nonlocal cancelled
                busy.setLabelText(f"{stage or default} {fraction:.0%}")
                busy.pump_events()
                cancelled = cancelled or busy.wasCanceled()
                return not cancelled

            return progress

        # The backend is built INSIDE the dialog: CoTracker downloads ~97 MB of
        # weights on first use, and that must be visible and cancellable rather
        # than freezing the UI before any progress bar exists.
        result, error = busy.execute(self._build_and_fill, key, label, report, set_stage)
        self._refresh_backend_rows()
        if cancelled:
            notify("Fill cancelled — nothing was changed.", "info")
            return
        if error is not None or result is None:
            return

        filled, confidence = result
        self.store.set_fill_from_flat(filled, confidence)
        self._push_pose_override()
        if self._mode is not None:
            # Predictions are drawn by the anchor overlay while armed, so the
            # canvas only shows the new fill once the mode redraws.
            self._mode.refresh()
        self._refresh_legend()
        # There is now something to approve, so the "Then go to:" row and the
        # Approve button appear.
        self._refresh_target_combos()
        self._refresh_approve_button()
        self._refresh_bulk_approve_buttons()
        # Every row's coordinates and provenance changed, and the table now
        # covers every frame of the labelled span rather than the labels alone.
        self._refresh_point_table(full=True)
        self._save_store()
        span = self.store.fill_range
        if span is None:
            notify("Nothing was filled — the observed frames carry no point to track.", "warning")
            return
        labelled = len(self.store.anchor_frames())
        detected = len(self.store.detection_frames())
        source = f"{labelled} labelled" + (f" and {detected} detected" if detected else "") + " frames"
        notify(
            f"Filled frames {span[0]}–{span[1]} ({span[1] - span[0] + 1} of {self.store.n_frames}) from {source}.",
            "info",
        )

    def _build_and_fill(self, key: str, label: str, report, set_stage):
        """Backends track flat points — the individual/keypoint split is restored after."""
        download = report("Downloading CoTracker3 weights…")
        if key == POSEPAL_BACKEND:
            backend = self._refined_backend_for(download)
            backend.on_stage = set_stage
            backend.signature = self._refinement_signature()
        else:
            backend = build_backend(
                key,
                checkpoint=self.app_state.labelling_cotracker_checkpoint or None,
                progress=download,
                disagreement_px=float(self.app_state.labelling_disagreement_px),
                n_points=self.store.n_points,
            )
        frames = None
        if backend.requires_video:
            frames = self._open_frames()
        try:
            filled = backend.fill(
                # Observations, not anchors: a detection is evidence from one
                # frame's pixels exactly as a click is, so the backends bridge
                # the gaps between both. Nothing in pose_fill knows the
                # difference — this line is the whole of the coupling.
                self.store.flat_observations(),
                self.store.n_frames,
                frames,
                report(f"Filling frames with {label}…"),
            )
        finally:
            if frames is not None:
                frames.close()
        if key == POSEPAL_BACKEND:
            self._save_refinement(backend)
        if self.store.static_keypoints:
            filled = self.store.pin_static(*filled)
        return filled

    def _open_frames(self, max_side: int = MAX_SIDE) -> VideoFrameSource:
        video = self._video_path()
        fps = self._fps()
        if not video:
            raise ValueError("This needs the video, but no video is loaded.")
        if not fps:
            raise ValueError("Video frame rate is unknown — cannot decode frames.")
        return VideoFrameSource(
            video,
            fps=fps,
            n_frames=self.store.n_frames,
            max_side=max_side,
            start_frame=self._view.start_frame,
        )

    # ------------------------------------------------------------------
    # Which frames to label
    # ------------------------------------------------------------------

    def _on_suggest(self) -> None:
        """Propose frames to label, then jump to the first one.

        Labelling consecutive frames is close to wasted effort — they are nearly
        identical. See :mod:`~ethograph.gui.pose_suggest` for the strategies.
        """
        method = self.suggest_method_combo.currentData()
        n_frames = self._n_frames()
        if not n_frames:
            notify("Frame count is unknown — load a video first.", "warning")
            return
        count = self._suggest_count()

        exclude = set(self.store.anchor_frames())
        if method == "uncertain" and self.store.confidence is None:
            notify("Run Fill first — this suggests the frames the fill was least sure about.", "warning")
            return
        if method == "detection_gaps" and not self.store.detections:
            notify("Run a detector first — this suggests the frames it found nothing on.", "warning")
            return

        # Only the pixel methods decode video; the others are instant.
        if method in ("uniform", "uncertain", "detection_gaps"):
            picks = suggest_frames(
                method,
                count,
                n_frames,
                exclude=exclude,
                confidence=self.store.confidence,
                detected=self.store.detection_frames(),
            )
            error = None
        else:
            busy = BusyProgressDialog("Scanning the video…", parent=self)

            def progress(fraction: float) -> bool:
                busy.setLabelText(f"Scanning the video… {fraction:.0%}")
                busy.pump_events()
                return not busy.wasCanceled()

            picks, error = busy.execute(self._run_suggest, method, count, n_frames, progress)
        if error is not None or not picks:
            if error is None:
                notify("No frames to suggest.", "warning")
            return

        self._suggestions = list(picks)
        self._suggestion_index = 0
        self._go_to_suggestion(0)

    def _run_suggest(self, method: str, count: int, n_frames: int, progress):
        # Thumbnails only — the suggestion scan never needs full-size frames.
        frames = self._open_frames(max_side=SUGGEST_MAX_SIDE)
        try:
            return suggest_frames(
                method,
                count,
                n_frames,
                frames,
                exclude=set(self.store.anchor_frames()),
                progress=progress,
            )
        finally:
            frames.close()

    def _go_to_suggestion(self, index: int) -> None:
        """Seek the playhead to a suggested frame."""
        if not self._suggestions:
            return
        self._suggestion_index = index % len(self._suggestions)
        self._seek(self._suggestions[self._suggestion_index])
        self._refresh_suggestion_label()

    def _approve_frame(self) -> None:
        """``Shift+H``: keep this frame's predictions as labels, then move on.

        Reviewing a fill is mostly *agreeing* with it, so agreeing has to cost
        one key. This is the same promotion as the table's "Pin filled points as
        labels", for every individual on the frame at once — the odd wrong point
        is corrected by dragging it, which pins it too.

        No mode is needed: approving is looking at the video and saying yes, no
        more a labelling action than deleting a point is. Where the playhead
        goes next is the explicit "Then go to:" choice, never inferred from
        whether a suggestion list happens to exist.
        """
        frame = self._current_frame()
        if not self.store.has_predictions(frame) and not self.store.is_human(frame):
            notify(f"Nothing to approve on frame {frame} — it carries no points.", "warning")
            return
        # Zero promotions is not a failure: an already-approved frame is agreed
        # with too, and the point of the key is to keep moving.
        if self.store.promote_fill(frame):
            self._after_table_edit()
        self._advance_frame()

    def _advance_frame(self) -> None:
        """Where the playhead goes next — whatever "Then go to:" says.

        Shared by a Loop-mode click and by ``Shift+H`` (approve this frame).
        Explicit rather than inferred: this used to follow the suggestion list
        whenever one existed, which meant the same click did different things
        depending on state the user could not see from the canvas.
        """
        behaviour = self.after_click_combo.currentData()
        if behaviour == AFTER_CLICK_FRAME:
            self._step_frames(1)
        elif behaviour == AFTER_CLICK_SUGGESTION:
            self._next_suggestion()

    def _step_frames(self, direction: int) -> None:
        """Seek one frame, clamped to the clip."""
        total = self._n_frames() or 0
        if not total:
            return
        frame = int(self.app_state.current_frame or 0) + direction
        self._seek(max(0, min(frame, total - 1)))

    def _next_suggestion(self) -> None:
        """``N``: the next frame worth labelling, wrapping at the end.

        There is deliberately no key for the previous one — the suggestions are
        a queue to work down, and going back to any frame at all (suggested or
        not) is one click in the points table.
        """
        if not self._suggestions:
            notify("No suggestions yet — press Suggest frames.", "warning")
            return
        self._go_to_suggestion(self._suggestion_index + 1)

    def _refresh_suggestion_label(self) -> None:
        if not self._suggestions:
            self.suggestion_label.setText("No suggested frames yet.")
            return
        position = self._suggestion_index + 1
        frame = self._suggestions[self._suggestion_index]
        done = sum(1 for f in self._suggestions if f in self.store.anchors)
        self.suggestion_label.setText(
            f"Suggestion {position}/{len(self._suggestions)} — frame {frame}   ·   {done} labelled"
        )

    def _on_clear_fill(self) -> None:
        self.store.clear_fill()
        self._push_pose_override()
        # Nothing left to approve, so the button and its "Then go to:" row go.
        self._refresh_active_label()
        self._refresh_point_table(full=True)

    def _push_pose_override(self) -> None:
        """Render the store through the normal pose overlay.

        Everything the store holds goes through, labels included and not only a
        fill: they are the same kind of thing to
        :mod:`~ethograph.gui.pose_render`, and pushing them registers the schema
        as this camera's keypoints, so the Pose sidebar's keypoint filter,
        confidence threshold, marker sizing and skeleton act on hand-labelled
        points exactly as on an imported DLC file. Confidence filtering then
        comes for free — low-confidence filled points are hidden by the existing
        "Filter below confidence" spinbox, and a label always scores 1.0.

        **Not while a mode is armed**: the anchor overlay already draws every
        point there, solid for a label and hollow for a prediction, so this
        would put a second marker on each one in a colour scheme that says
        nothing about where the point came from. Disarming hands the pose back
        to the overlay.
        """
        pose_mgr = self._data_widget.pose_mgr
        if pose_mgr is None:
            return
        if self._mode is not None:
            if self._override_pushed:
                pose_mgr.set_pose_override(None)
                self._override_pushed = False
                self._data_widget.update_pose()
            return
        fps = self._fps()
        empty = self.store.filled is None and not self.store.anchor_frames() and not self.store.detection_frames()
        if empty or not fps:
            pose_mgr.set_pose_override(None)
            self._override_pushed = False
        else:
            # No y-flip here: the overlay draws in image coordinates and does
            # its own `y_world = img_height - y` (see pose_overlay). Flipping
            # first would mirror the points off the animal.
            ds = store_to_movement_ds(self.store, fps)
            pose_mgr.set_pose_override(movement_ds_to_pose_render(ds, "labelled keypoints"))
            self._override_pushed = True
        self._data_widget.update_pose()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_movement(self) -> None:
        """Write the same dataset *Load into the GUI* loads, where the user asks."""
        ds = self._build_dataset()
        if ds is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export poses", "keypoints.nc", "NetCDF (*.nc)")
        if not path:
            return
        ds.to_netcdf(path)
        notify(f"Wrote {path} ({ds.attrs['space_unit']})", "info")

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._exit_calibrate_mode(restore=False)
        self._detach_mode()
        self._save_detections()
        self._preview_timer.stop()
        self._close_preview_frames()
        self._install_key_filter(False)
        for signal, slot in (
            (self.app_state.current_frame_changed, self._on_frame_changed),
            (self.app_state.video_path_changed, self._on_video_changed),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)
