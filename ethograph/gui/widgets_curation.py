"""Curation section of the Labels tab — the one place labels get reviewed.

Every label carries a ``labeling_method`` (``labels/curation.py``): *manual*,
*automated* or *curated*. This panel, sitting under the label tables, is where
a user turns automated labels into curated ones:

* **Scope** — which label classes curation acts on. Rows dragged out of the
  label tables above land in the drop area (their ids are listed); *All*
  means every class and *Reset* empties the area again.
* **Mode** — how a label gets curated:

  - *Manual (trial level)*: any label edit stamps that label manual (the
    store does this); **Ctrl+C** curates every automated label in scope of
    the current trial.
  - *Inspect is enough (trial level)*: merely opening a trial curates its
    automated labels in scope.
  - *Segment review*: the state labels in scope become a queue of whole
    labels walked one by one; each jump plays the label with the navigation
    padding around it and arms it for editing exactly as **Ctrl+E** would,
    so two left clicks (new start, new end) re-place it. **Backspace**
    deletes, **B**/**N** go back / next — *N* curates the label it leaves
    when the checkbox says so, and **Enter** does nothing here: a label
    that plays right needs no key but N.
  - *Frame-by-frame review*: the labels in scope become a queue of
    boundaries walked one by one, each centred in a small view window.
    ``←``/``→`` nudge the video, **Enter** commits the frame on screen as the
    boundary (the label becomes manual if it moved), **Backspace** deletes the
    event outright, **B**/**N** go back / next — and *N* also curates the
    boundary it leaves when the checkbox says so. **Automated only** (ticked
    by default) leaves manual/curated boundaries out of the queue — a human
    already vouched for those, so there is nothing to re-review.

* **Order** (next to Mode) — how a review queue is walked
  (:data:`~ethograph.labels.curation.REVIEW_ORDERS`, saved to
  ``gui_settings.yaml``): *Trial-by-trial* finishes every boundary of one
  trial before moving to the next; *Label-by-label* finishes every instance
  of one class, across every trial, before moving to the next class.
  Changing it mid-session rebuilds the queue in place, keeping the current
  boundary in view.

* **Label grid view…** / **Video grid…** open the review grids
  (``dialog_label_gridview.py``, ``dialog_video_grid.py``) on the scope; a
  tile click there navigates, and in frame-by-frame mode it drops straight
  into the review at that label.
* **Model ▸ Curation workflows…** (top bar) opens the saved curation routines
  (``dialog_curation_workflow.py``): filter, predict, scope, grid, review,
  save — recorded once and replayed, rather than set up again each session.
* **Tools ▸ Label bulk editing…** (``dialog_bulk_labels.py``) is where curate /
  delete / purge across many trials at once live — this panel only keeps the
  shortcut (**Ctrl+C**, current trial) and the two review grids, so it stays
  about *reviewing* rather than accumulating every bulk action as a button.

Curate / delete / purge all take an explicit trial scope
(:data:`~ethograph.labels.workflow.TRIAL_SCOPE_CHOICES`: single / all /
filtered / hidden) and an explicit label-class set — the bulk-editing dialog's
own checkbox list, never silently the drag-and-drop scope area above, so a
bulk action's blast radius is always what the dialog shows on screen.

The per-trial verdict (no automated label left) colours the trial combo and
the bottom bar, and is written to the metadata table's ``curated`` column on
a timer (:data:`METADATA_SYNC_MS`) — labelling must never wait on a file
write.

That timer only runs while curation is **active** (``app_state.curation_active``,
never saved): dropping label classes into the scope area or curating anything
arms it via :meth:`CurationPanel.activate`, and a fresh dataset disarms it. A
session that curates nothing therefore touches no file. Arming is also the one
moment a metadata TSV is created — the ``curated`` column is Ethograph's own
state and never goes into a recording or the alignment NWB, so
:func:`~ethograph.io.metadata_edit.ensure_tabular_target` copies the loaded
table to the sidecar TSV, which becomes the metadata table from then on.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.dialog_label_gridview import confidence_display
from ethograph.gui.label_drawing_mixin import draw_key
from ethograph.gui.notify import notify
from ethograph.gui.shortcuts import typing_in_text_field
from ethograph.io.time_model import TimeRange
from ethograph.labels import onset_curves
from ethograph.labels import review_metrics as rm
from ethograph.labels import workflow as wf
from ethograph.labels.curation import (
    CURATED_COLUMN,
    CURATED_NO,
    CURATED_YES,
    FIELD_LABEL,
    REVIEW_ORDER_TRIAL,
    ReviewTarget,
    build_review_queue,
    curate_label,
    curate_trials,
    curated_column_differs,
    delete_labels,
    method_counts,
    purge_short_labels,
    queue_index_of,
    row_mask,
    stitch_labels,
    targets_from_seeds,
)
from ethograph.labels.intervals import (
    HUMAN_CONFIDENCE,
    LABELING_AUTOMATED,
    LABELING_CURATED,
    LABELING_MANUAL,
    delete_interval,
    ensure_labeling_method,
    get_interval_bounds,
)
from ethograph.labels.plots import plot_confidence_pdf
from ethograph.labels.predictions import PredictionsStore
from ethograph.labels.tsv_store import set_trial_in_tsv

logger = logging.getLogger(__name__)

#: How often the per-trial verdicts are pushed into the metadata table's
#: ``curated`` column. Curating is a stream of small edits; writing the
#: table after each would put a file write on the labelling path.
METADATA_SYNC_MS = 5000

#: Curation modes: key → combo text.
CURATION_MODES = {
    "manual": "Manual (trial level)",
    "inspect": "Inspect is enough (trial level)",
    "segment": "Segment review",
    "frame": "Frame-by-frame review",
}

#: The modes that walk a queue (Start review / Stop review).
REVIEW_MODES = ("segment", "frame")

#: Frame-by-frame review order (``labels/curation.REVIEW_ORDERS``): key → combo text.
REVIEW_ORDERS = {
    "trial": "Trial-by-trial",
    "label": "Label-by-label",
}

_REVIEW_ORDER_HINTS = {
    "trial": "Walks every boundary of a trial (all classes in scope, in time order), then the next trial.",
    "label": "Walks every instance of one class across every trial, then moves to the next class.",
}

#: The plain "current trial"/"all trials"/… phrase, without the parenthetical
#: — for a message or dialog title built from :data:`wf.TRIAL_SCOPE_CHOICES`.
_TRIAL_SCOPE_NOUN = {key: text.split(" (")[0].lower() for key, text in wf.TRIAL_SCOPE_CHOICES.items()}

#: Sentinel default for a *label_ids* parameter: "not given, fall back to the
#: curation scope area" — distinct from an explicit ``None`` (every class),
#: which a caller like the bulk-editing dialog's "All" checkbox passes on purpose.
_SCOPE_UNSET = object()


def open_path(path) -> None:
    """Open *path* with the system's default application."""
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


_MODE_HINTS = {
    "manual": "Editing a label makes it manual · Ctrl+C curates every automated label in scope of this trial.",
    "inspect": "Opening a trial curates its automated labels in scope — looking is enough.",
    "segment": (
        "Walk the labels in scope one at a time: each plays, and two clicks (new start, new end) "
        "re-place it. N moves on and curates."
    ),
    "frame": "Walk the labels in scope boundary by boundary and use shortcuts (below) to approve/edit.",
}

_FIELD_TITLES = {"point": "POINT", "start": "START", "end": "END", FIELD_LABEL: "SEGMENT"}

#: What the delta line says while a segment is armed for editing.
_SEGMENT_HINT = "click the new start, then the new end  ·  V replays"

#: Linger on a just-committed boundary this long before jumping to the next
#: seed, so the user sees the label land where they put it.
_CONFIRM_PAUSE_MS = 100

_KEYS_SCHEMATIC = (
    "<table cellspacing='2' style='color:#bbb; font-size:10px;'>"
    "<tr><td><b>←</b> / <b>→</b></td><td>one frame</td>"
    "<td>&nbsp;&nbsp;<b>Enter</b></td><td>confirm this frame</td></tr>"
    "<tr><td><b>B</b> / <b>N</b></td><td>back / next</td>"
    "<td>&nbsp;&nbsp;<b>Backspace</b></td><td>delete the event</td></tr>"
    "</table>"
)

_SEGMENT_KEYS_SCHEMATIC = (
    "<table cellspacing='2' style='color:#bbb; font-size:10px;'>"
    "<tr><td><b>click</b> / <b>click</b></td><td>new start / new end</td>"
    "<td>&nbsp;&nbsp;<b>V</b></td><td>replay the label</td></tr>"
    "<tr><td><b>B</b> / <b>N</b></td><td>back / next</td>"
    "<td>&nbsp;&nbsp;<b>Backspace</b></td><td>delete the label</td></tr>"
    "</table>"
)

_KEY_ROWS = {
    "frame": [
        ("←  /  →", "step the video one frame back / forward"),
        ("Enter", "confirm: the frame on screen becomes the boundary and the review moves on"),
        ("Backspace  /  Delete", "this event should not exist — delete it and move on"),
        ("B", "back to the previous boundary"),
        ("N", "next boundary (curates the one you leave when the box is ticked)"),
        ("Space", "play / pause"),
    ],
    "segment": [
        ("click, click", "re-place the label: the first click is its new start, the second its new end"),
        ("V", "play the label again"),
        ("Backspace  /  Delete", "this label should not exist — delete it and move on"),
        ("B", "back to the previous label"),
        ("N", "next label (curates the one you leave when the box is ticked)"),
        ("Space", "play / pause"),
    ],
}
_KEY_ROWS_ALWAYS = [
    ("Ctrl+C", "curate every automated label in scope of this trial"),
    ("Ctrl+T", "flag this trial as hard (again: back to normal) — shown more often when training"),
]


def drag_label_ids(text: str) -> list[int]:
    """Label ids carried by a drag out of the label tables (``"1,4,8"``)."""
    ids: list[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            return []
        if value not in ids:
            ids.append(value)
    return ids


def _num(value) -> float | None:
    if value is None:
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _close(a: float | None, b: float | None, atol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= atol


def _color_hex(mapping: dict) -> str:
    color = mapping.get("color")
    if color is None:
        return "#ffffff"
    return "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in color[:3]))


# ----------------------------------------------------------------------
# Scope drop area
# ----------------------------------------------------------------------


class ScopeDropArea(QFrame):
    """Where label rows dragged out of the tables land; lists their ids."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(30)
        self.setToolTip(
            "Drag label rows from the tables above here to curate only those classes.\nEmpty means every class."
        )
        self._ids: list[int] = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        self._label = QLabel()
        self._label.setWordWrap(True)
        lay.addWidget(self._label, stretch=1)
        self._refresh()

    def ids(self) -> list[int]:
        return list(self._ids)

    def set_ids(self, ids) -> None:
        self._ids = [int(i) for i in (ids or [])]
        self._refresh()

    def add_ids(self, ids) -> bool:
        """Add *ids* (keeping order, no duplicates); True when something changed."""
        added = False
        for i in ids:
            if int(i) not in self._ids:
                self._ids.append(int(i))
                added = True
        if added:
            self._refresh()
        return added

    def _refresh(self) -> None:
        if self._ids:
            self._label.setText("Labels to curate: " + ", ".join(str(i) for i in self._ids))
            self.setStyleSheet("QFrame { border: 1px solid #ffe066; border-radius: 3px; }")
        else:
            self._label.setText("All labels — drag label rows here to narrow")
            self.setStyleSheet("QFrame { border: 1px dashed #888; border-radius: 3px; color: #aaa; }")

    def dragEnterEvent(self, event):
        if event.mimeData().hasText() and drag_label_ids(event.mimeData().text()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        ids = drag_label_ids(event.mimeData().text())
        if not ids:
            event.ignore()
            return
        # A drop here is a copy into the scope — the row stays in its table,
        # so the default MoveAction must not be reported back to the source.
        event.setDropAction(Qt.CopyAction)
        event.accept()
        if self.add_ids(ids):
            panel = self.parent()
            while panel is not None and not isinstance(panel, CurationPanel):
                panel = panel.parent()
            if panel is not None:
                panel._on_scope_edited(activates=True, dropped=ids)


class ShortcutsPopup(QDialog):
    """A review mode's keys, drawn as a little schematic."""

    def __init__(self, parent=None, mode: str = "frame"):
        super().__init__(parent)
        self.setWindowTitle(f"{CURATION_MODES[mode]} keys")
        self.setModal(False)
        lay = QVBoxLayout(self)
        for key, what in _KEY_ROWS[mode] + _KEY_ROWS_ALWAYS:
            row = QHBoxLayout()
            key_label = QLabel(key)
            key_label.setStyleSheet(
                "QLabel { background: #333; color: #ffe066; border: 1px solid #666; border-radius: 4px;"
                " padding: 2px 8px; font-weight: bold; }"
            )
            key_label.setMinimumWidth(150)
            key_label.setAlignment(Qt.AlignCenter)
            row.addWidget(key_label)
            row.addWidget(QLabel(what), stretch=1)
            lay.addLayout(row)
        close_btn = QPushButton("Close")
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.close)
        lay.addWidget(close_btn, alignment=Qt.AlignRight)


# ----------------------------------------------------------------------
# The panel
# ----------------------------------------------------------------------


class CurationPanel(QGroupBox):
    """Scope + mode + the two review modes, under the label tables."""

    #: A review session (segment or frame-by-frame) ended — finished, stopped
    #: or torn down. How a curation workflow knows the reviewer is done.
    review_finished = Signal()

    def __init__(self, app_state, labels_widget, parent=None):
        super().__init__("Curation", parent)
        self.app_state = app_state
        self.labels_widget = labels_widget
        self.meta = None
        self.nav = None
        self.data_widget = None
        self.plot_container = None
        self._grid_dialog = None
        self._video_dialog = None
        self._workflow_dialog = None
        self._shortcuts_popup: ShortcutsPopup | None = None

        # Review state (one session at a time, in the mode it started in)
        self._targets: list[ReviewTarget] = []
        self._idx = 0
        self._seed_frame: int | None = None
        self._session_active = False
        self._session_mode = "frame"
        self._advance_pending = False
        self._jumping = False
        self._frame_conn = False
        self._n_confirmed = 0
        self._n_deleted = 0
        self._session_shortcuts: list[QShortcut] = []
        #: The onset_curves.npz picked for this review session (None = don't
        #: show curves), resolved once in _begin, plus the (path, mtime) it
        #: was last read at — a fresh prediction run rewrites the file, so
        #: the key is what notices instead of a cross-module invalidation call.
        self._curve_source: Path | None = None
        self._curves: dict[str, onset_curves.TrialCurves] = {}
        self._curves_key: tuple | None = None
        #: The review has run for the current "every trial curated" state;
        #: cleared the moment any trial is automated again (a new prediction
        #: run), so finishing the curation fires it exactly once.
        self._review_done = False
        self._syncing_hard = False

        self._build_ui()

        # Started by activate() — a session that curates nothing writes nothing.
        self._metadata_timer = QTimer(self)
        self._metadata_timer.setInterval(METADATA_SYNC_MS)
        self._metadata_timer.timeout.connect(self.sync_metadata)

        app_state.ready_changed.connect(self._on_ready_changed)
        app_state.trial_changed.connect(self._on_trial_changed)
        app_state.current_frame_changed.connect(self._on_frame_changed)
        app_state.curation_mode_changed.connect(self._sync_mode_from_state)
        app_state.curation_label_ids_changed.connect(self._sync_scope_from_state)
        app_state.curation_review_order_changed.connect(self._sync_order_from_state)
        self._sync_mode_from_state()
        self._sync_scope_from_state()
        self._sync_order_from_state()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Set apart from the label tables above: a gap, and a title that reads
        # as a heading rather than one more group.
        self.setObjectName("curation_panel")
        self.setStyleSheet(
            "QGroupBox#curation_panel { margin-top: 18px; padding-top: 18px; font-size: 11pt; font-weight: bold; }"
        )
        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(4, 4, 4, 4)

        scope_row = QHBoxLayout()
        self.scope_area = ScopeDropArea()
        scope_row.addWidget(self.scope_area, stretch=1)
        self.scope_all_btn = QPushButton("All")
        self.scope_all_btn.setFixedWidth(36)
        self.scope_all_btn.setToolTip("Curate every label class")
        self.scope_all_btn.clicked.connect(self._scope_all)
        scope_row.addWidget(self.scope_all_btn)
        self.scope_reset_btn = QPushButton("Reset")
        self.scope_reset_btn.setFixedWidth(48)
        self.scope_reset_btn.setToolTip("Empty the scope so other labels can be dragged in")
        self.scope_reset_btn.clicked.connect(self._scope_all)
        scope_row.addWidget(self.scope_reset_btn)
        lay.addLayout(scope_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        for key, text in CURATION_MODES.items():
            self.mode_combo.addItem(text, key)
            self.mode_combo.setItemData(self.mode_combo.count() - 1, _MODE_HINTS[key], Qt.ToolTipRole)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_combo)
        mode_row.addWidget(self.mode_combo, stretch=1)
        mode_row.addWidget(QLabel("Order:"))
        self.order_combo = QComboBox()
        for key, text in REVIEW_ORDERS.items():
            self.order_combo.addItem(text, key)
        self.order_combo.setToolTip(
            "Review order — Trial-by-trial walks a trial's labels then moves on;\n"
            "Label-by-label finishes one class across every trial first."
        )
        self.order_combo.currentIndexChanged.connect(self._on_order_combo)
        mode_row.addWidget(self.order_combo, stretch=1)
        lay.addLayout(mode_row)

        # ── Segment / frame-by-frame review ─────────────────────────
        self.frame_group = QWidget()
        frame_lay = QVBoxLayout(self.frame_group)
        frame_lay.setContentsMargins(0, 2, 0, 0)
        frame_lay.setSpacing(3)

        self.target_label = QLabel("")
        self.target_label.setAlignment(Qt.AlignCenter)
        self.target_label.setTextFormat(Qt.PlainText)
        self.target_label.setWordWrap(True)
        frame_lay.addWidget(self.target_label)
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        frame_lay.addWidget(self.info_label)
        self.delta_label = QLabel("")
        self.delta_label.setAlignment(Qt.AlignCenter)
        frame_lay.addWidget(self.delta_label)

        # The view window is the frame review's; the segment review shows the
        # whole label with the navigation padding, so the row hides with it.
        self.window_row = QWidget()
        win_row = QHBoxLayout(self.window_row)
        win_row.setContentsMargins(0, 0, 0, 0)
        win_row.addWidget(QLabel("View window:"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.02, 600.0)
        self.window_spin.setDecimals(2)
        self.window_spin.setSingleStep(0.2)
        self.window_spin.setSuffix(" s")
        self.window_spin.setToolTip(
            "Seconds of time series shown around the boundary being reviewed\n"
            "(seed centred) — independent of the navigation Before/After padding."
        )
        self.window_spin.setValue(float(self.app_state.get_with_default("refine_window_s")))
        self.window_spin.valueChanged.connect(self._on_window_changed)
        # Enter in the spinbox commits the value and hands focus back, so the
        # NEXT Enter is a boundary confirm again — one keypress, one meaning.
        self.window_spin.editingFinished.connect(self.window_spin.clearFocus)
        win_row.addWidget(self.window_spin, stretch=1)
        self.lock_checkbox = QCheckBox("Locked around label")
        self.lock_checkbox.setChecked(False)
        self.lock_checkbox.setToolTip(
            "Ticked: the view stays a small window around the boundary.\n"
            "Unticked: pan/zoom the whole trial freely — Enter still confirms\n"
            "the frame on screen, wherever you navigated."
        )
        self.lock_checkbox.toggled.connect(self._on_lock_toggled)
        win_row.addWidget(self.lock_checkbox)
        frame_lay.addWidget(self.window_row)

        review_opts_row = QHBoxLayout()
        self.next_curates_cb = QCheckBox("Click N curates current")
        self.next_curates_cb.setToolTip(
            "Moving on with N means you looked at the boundary and it is fine:\n"
            "an automated label becomes curated. Untick to only browse."
        )
        self.next_curates_cb.setChecked(bool(self.app_state.get_with_default("curation_next_curates")))
        self.next_curates_cb.toggled.connect(lambda v: setattr(self.app_state, "curation_next_curates", v))
        review_opts_row.addWidget(self.next_curates_cb)

        self.automated_only_cb = QCheckBox("Show automated only")
        self.automated_only_cb.setToolTip(
            "A human already vouched for a manual or curated label, so the queue\n"
            "leaves those out — only automated boundaries need a first look.\n"
            "Untick to walk every label in scope regardless of method."
        )
        self.automated_only_cb.setChecked(bool(self.app_state.get_with_default("frame_review_automated_only")))
        self.automated_only_cb.toggled.connect(lambda v: setattr(self.app_state, "frame_review_automated_only", v))
        review_opts_row.addWidget(self.automated_only_cb)

        self.auto_advance_cb = QCheckBox("Jump to next after Enter/Backspace")
        self.auto_advance_cb.setToolTip(
            "Ticked: confirming (Enter) or deleting (Backspace) a boundary\n"
            "moves on to the next target automatically. Untick to stay put."
        )
        self.auto_advance_cb.setChecked(bool(self.app_state.get_with_default("curation_auto_advance")))
        self.auto_advance_cb.toggled.connect(lambda v: setattr(self.app_state, "curation_auto_advance", v))
        review_opts_row.addWidget(self.auto_advance_cb)
        frame_lay.addLayout(review_opts_row)

        keys_row = QHBoxLayout()
        self.keys_label = QLabel(_KEYS_SCHEMATIC)
        self.keys_label.setTextFormat(Qt.RichText)
        keys_row.addWidget(self.keys_label, stretch=1)
        self.shortcuts_btn = QPushButton("Shortcuts…")
        self.shortcuts_btn.setAutoDefault(False)
        self.shortcuts_btn.clicked.connect(self._show_shortcuts)
        keys_row.addWidget(self.shortcuts_btn, alignment=Qt.AlignTop)
        frame_lay.addLayout(keys_row)

        self.start_stop_btn = QPushButton("Start review")
        self.start_stop_btn.setAutoDefault(False)
        self.start_stop_btn.setDefault(False)
        self.start_stop_btn.clicked.connect(self._toggle_session)
        frame_lay.addWidget(self.start_stop_btn)
        lay.addWidget(self.frame_group)

        # ── Tools ───────────────────────────────────────────────────
        # Curate trial/visible/delete/purge all moved to Tools ▸ Label bulk
        # editing… (dialog_bulk_labels.py) — this panel keeps only the
        # Ctrl+C shortcut (bound in shortcuts.py) and the two review grids.
        tools_row = QHBoxLayout()
        self.grid_btn = QPushButton("Label grid view…")
        self.grid_btn.setAutoDefault(False)
        self.grid_btn.setToolTip("A grid of video frames at the label times in scope — click a tile to go there")
        self.grid_btn.clicked.connect(self.open_grid_view)
        tools_row.addWidget(self.grid_btn)
        self.video_grid_btn = QPushButton("Video grid…")
        self.video_grid_btn.setAutoDefault(False)
        self.video_grid_btn.setToolTip(
            "Play the labels in scope side by side, one label class per group\n(decodes video — slower to build)"
        )
        self.video_grid_btn.clicked.connect(self.open_video_grid)
        tools_row.addWidget(self.video_grid_btn)
        self.curves_pdf_btn = QPushButton("Confidence curves…")
        self.curves_pdf_btn.setAutoDefault(False)
        self.curves_pdf_btn.setToolTip(
            "Each trial's mean frame confidence into the metadata table's model_confidence\n"
            "column — sort or filter the trials table on it to decide which trials to open\n"
            "and which to curate in bulk — and a PDF of every trial's curve with its labels.\n"
            "From the runs behind the labels, else the prediction set selected in the I/O tab."
        )
        self.curves_pdf_btn.clicked.connect(self.export_confidence_pdf)
        tools_row.addWidget(self.curves_pdf_btn)
        lay.addLayout(tools_row)

        # ── Curator feedback: where the human overruled the model ──────
        # Only a human writes the hard flag: by hand, or once from the F1
        # histogram after looking at the distribution. Score now measures
        # and never flags; confidence plays no part at all — a
        # confident-and-wrong trial is exactly the one to catch here.
        feedback_title = QLabel("<b>Curator feedback</b>")
        feedback_title.setToolTip(
            "Where you overruled the model, it should pay extra attention next time.\n"
            "Flag a trial hard by hand (the box), or score every trial against its\n"
            "prediction run and flag the worst from the histogram, once — into the\n"
            "metadata table's difficulty column, which a training run reads back\n"
            "(train.oversample; off unless set)."
        )
        lay.addWidget(feedback_title)
        review_row = QHBoxLayout()
        self.hard_cb = QCheckBox("Hard trial (Ctrl+T)")
        self.hard_cb.setToolTip(
            "Flag this trial as hard in the metadata table's difficulty column —\n"
            "the model barely managed it, or it is just difficult. A training run\n"
            "with train.oversample draws hard trials more often."
        )
        self.hard_cb.toggled.connect(self._on_hard_toggled)
        review_row.addWidget(self.hard_cb)
        review_row.addStretch(1)
        review_row.addWidget(QLabel("Tolerance:"))
        self.review_tolerance_spin = QDoubleSpinBox()
        self.review_tolerance_spin.setRange(0.0, 10.0)
        self.review_tolerance_spin.setDecimals(3)
        self.review_tolerance_spin.setSingleStep(0.01)
        self.review_tolerance_spin.setSpecialValueText("run's own")
        self.review_tolerance_spin.setSuffix(" s")
        self.review_tolerance_spin.setToolTip(
            'Point-event tolerance for the review. Left at "run\'s own", each run is judged\n'
            "at the tolerance its model was trained to (read from the run folder). Set it to\n"
            "compare runs trained at different tolerances, or for a run folder that carries none."
        )
        override = self.app_state.get_with_default("review_tolerance_s")
        self.review_tolerance_spin.setValue(float(override) if override else 0.0)
        self.review_tolerance_spin.valueChanged.connect(
            lambda v: setattr(self.app_state, "review_tolerance_s", float(v) if v > 0 else None)
        )
        self.review_tolerance_spin.editingFinished.connect(self.review_tolerance_spin.clearFocus)
        review_row.addWidget(self.review_tolerance_spin)
        self.review_btn = QPushButton("Score now")
        self.review_btn.setAutoDefault(False)
        self.review_btn.setToolTip(
            "Score the trials the table shows against each one's prediction run — an F1 per\n"
            "trial and event type into the metadata table (state labels at IoU ≥ 0.5, point\n"
            "labels within the run's own tolerance). Measures only; runs by itself once the\n"
            "last trial is curated."
        )
        self.review_btn.clicked.connect(lambda: self.run_review())
        review_row.addWidget(self.review_btn)
        self.review_hist_btn = QPushButton("Histogram…")
        self.review_hist_btn.setAutoDefault(False)
        self.review_hist_btn.setToolTip(
            "The scored trials' F1 as a histogram. Look for the split, set the threshold in\n"
            "the gap, and flag everything below it hard — once, by your decision."
        )
        self.review_hist_btn.clicked.connect(self.open_review_histogram)
        review_row.addWidget(self.review_hist_btn)
        lay.addLayout(review_row)

        self.review_label = QLabel("")
        self.review_label.setTextFormat(Qt.PlainText)
        self.review_label.setWordWrap(True)
        self.review_label.setStyleSheet("font-size: 10px; color: #bbb;")
        lay.addWidget(self.review_label)

        self.status_label = QLabel("")
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setStyleSheet("font-size: 10px;")
        lay.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_data_widget(self, data_widget) -> None:
        self.data_widget = data_widget
        self.nav = getattr(data_widget, "navigation_widget", None) or self.nav

    def set_plot_container(self, plot_container) -> None:
        self.plot_container = plot_container

    def set_meta(self, meta) -> None:
        self.meta = meta
        self.nav = getattr(meta, "navigation_widget", None) or self.nav
        self.data_widget = getattr(meta, "data_widget", None) or self.data_widget

    def _trials_widget(self):
        return getattr(self.meta, "trials_widget", None) or getattr(self.data_widget, "trials_widget", None)

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    def scope(self) -> set[int] | None:
        """Label classes curation acts on; ``None`` = every class."""
        return self.app_state.curation_scope()

    def scope_or_all_ids(self) -> list[int]:
        """The scope as an explicit id list (every mapped class when unset)."""
        ids = self.scope_area.ids()
        if ids:
            return ids
        mappings = getattr(self.labels_widget, "_mappings", {}) or {}
        return sorted(lid for lid in mappings if isinstance(lid, int) and lid != 0)

    def _on_scope_edited(self, *, activates: bool = False, dropped=None) -> None:
        self.app_state.curation_label_ids = self.scope_area.ids() or None
        if activates:
            self.activate("label classes dropped into the curation scope")
        if dropped:
            self._follow_branch(dropped)
        self._refresh_status()

    def _follow_branch(self, label_ids) -> None:
        """Make the branch the dropped classes belong to the editable one.

        Reviewing a class means editing its labels, and only the active
        branch is editable. Classes from several branches name no single
        branch, so nothing changes then.
        """
        mappings = getattr(self.labels_widget, "_mappings", {}) or {}
        branches = {int(mappings[i].get("branch", 0)) for i in label_ids if i in mappings}
        if len(branches) != 1:
            return
        set_active = getattr(self.labels_widget, "set_active_branch", None)
        if callable(set_active):
            set_active(branches.pop())

    def set_scope(self, label_ids, *, reason: str) -> None:
        """Replace the curation scope with *label_ids*, as if dragged in, and activate.

        Public hand-off point for callers outside this module (e.g. the onset
        model's "Review predictions…" button) that want the just-produced
        classes sitting in the scope area rather than opening a dialog.
        """
        self.scope_area.set_ids(label_ids)
        self.app_state.curation_label_ids = self.scope_area.ids() or None
        self.activate(reason)
        self._refresh_status()

    def _scope_all(self) -> None:
        self.scope_area.set_ids([])
        self._on_scope_edited()

    def _sync_scope_from_state(self, *_args) -> None:
        ids = self.app_state.curation_label_ids or []
        if ids != self.scope_area.ids():
            self.scope_area.set_ids(ids)
        self._refresh_status()

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    def mode(self) -> str:
        return str(self.mode_combo.currentData() or "manual")

    def _on_mode_combo(self, _index: int) -> None:
        key = self.mode()
        if self.app_state.curation_mode != key:
            self.app_state.curation_mode = key
        self._apply_mode(key)

    def _sync_mode_from_state(self, *_args) -> None:
        key = str(self.app_state.get_with_default("curation_mode") or "manual")
        idx = self.mode_combo.findData(key)
        if idx < 0:
            idx = 0
        if self.mode_combo.currentIndex() != idx:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(idx)
            self.mode_combo.blockSignals(False)
        self._apply_mode(self.mode())

    def _apply_mode(self, key: str) -> None:
        self.mode_combo.setToolTip(_MODE_HINTS[key])
        self.frame_group.setVisible(key in REVIEW_MODES)
        self.window_row.setVisible(key == "frame")
        self.keys_label.setText(_SEGMENT_KEYS_SCHEMATIC if key == "segment" else _KEYS_SCHEMATIC)
        self.auto_advance_cb.setText(
            "Jump to next after Backspace" if key == "segment" else "Jump to next after Enter/Backspace"
        )
        # The queue is the mode's (boundaries vs whole labels): leaving the
        # mode a session started in ends it.
        if self._session_active and key != self._session_mode:
            self._stop()
        if key == "inspect" and self.app_state.ready:
            self.curate_current_trial(quiet=True)

    def reviews_on_jump(self) -> bool:
        """Whether a grid's double-click should drop into a review session."""
        return self.mode() in REVIEW_MODES

    # ------------------------------------------------------------------
    # Review order
    # ------------------------------------------------------------------

    def order(self) -> str:
        return str(self.order_combo.currentData() or REVIEW_ORDER_TRIAL)

    def _on_order_combo(self, _index: int) -> None:
        key = self.order()
        if self.app_state.curation_review_order != key:
            self.app_state.curation_review_order = key
        self.order_combo.setToolTip(_REVIEW_ORDER_HINTS.get(key, ""))
        if self._session_active:
            self.restart_review()

    def _sync_order_from_state(self, *_args) -> None:
        key = str(self.app_state.get_with_default("curation_review_order") or REVIEW_ORDER_TRIAL)
        idx = self.order_combo.findData(key)
        if idx < 0:
            idx = 0
        if self.order_combo.currentIndex() != idx:
            self.order_combo.blockSignals(True)
            self.order_combo.setCurrentIndex(idx)
            self.order_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Applying method changes
    # ------------------------------------------------------------------

    def _commit(
        self,
        df,
        n: int,
        *,
        restyle: tuple | None = None,
        message: str | None = None,
        activate_reason: str = "labels curated",
    ) -> int:
        """Swap *df* in and refresh what shows a label's method.

        *restyle* is ``(inst, automated)`` for a single-label transition —
        restyled in place on the plots when it is on screen, else a full
        label redraw. A whole trial changing is always a full redraw.
        """
        if not n:
            return 0
        # The per-trial verdict may have changed either way, so it needs a home.
        self.activate(activate_reason)
        self.app_state.replace_all_labels(df)
        self.app_state.changes_saved = False
        restyled = 0
        if restyle is not None and self.plot_container is not None:
            inst, automated = restyle
            key = draw_key(
                inst["labels"],
                self.app_state.to_display(inst["trial"], float(inst["onset_s"])),
                inst.get("individual"),
                inst.get("individual_rec"),
            )
            restyled = self.plot_container.restyle_label(key, automated)
        if not restyled and self.plot_container is not None:
            self.plot_container.schedule_labels_redraw()
        self.app_state.curation_changed.emit()
        self._refresh_status()
        if message:
            notify(message)
        return n

    def trials_for_scope(self, which: str) -> list:
        """Which trials *which* (:data:`wf.TRIAL_SCOPE_CHOICES`) names right now.

        "single" is the current trial. "filtered" is exactly ``app_state.trials``
        (what the table shows). "all"/"hidden" need the full trial list — the
        trials table is the one place that still has it once its filters have
        narrowed ``app_state.trials``.
        """
        if which == wf.TRIAL_SCOPE_SINGLE:
            trial = getattr(self.app_state, "trials_sel", None)
            return [trial] if trial is not None else []
        visible = self.app_state.trials or []
        if which in (wf.TRIAL_SCOPE_ALL, wf.TRIAL_SCOPE_HIDDEN):
            trials_widget = self._trials_widget()
            if trials_widget is None:
                return list(visible) if which == wf.TRIAL_SCOPE_ALL else []
            if which == wf.TRIAL_SCOPE_ALL:
                return trials_widget.all_trials()
            visible_set = {str(t) for t in visible}
            return [t for t in trials_widget.all_trials() if str(t) not in visible_set]
        return list(visible)  # filtered

    def curate_trial_labels(
        self, which: str = wf.TRIAL_SCOPE_FILTERED, label_ids=_SCOPE_UNSET, confirm: bool = False, quiet: bool = False
    ) -> int:
        """Curate every automated label of *label_ids*, across the trials *which* names.

        *label_ids* defaults to the curation scope area (``self.scope()``);
        pass an explicit set (or ``None`` for every class) to bypass it — what
        the bulk-editing dialog's own checkbox list does. Manual labels are
        never rewritten.

        *confirm* asks first: from the GUI this is one click away from marking
        labels nobody looked at as seen, which is the one thing the
        automated/curated split exists to keep apart. A workflow step is
        already a deliberate, written-down choice and does not ask.

        Not a follow-up to a grid's *uncurate* Done: that already curates
        every unclicked automated label in the grid, which is this same set.
        This is for the flows where nothing swept up — a grid browsed in
        navigate mode, a review stopped partway, or a deliberate bulk accept.
        """
        if not self.app_state.ready:
            return 0
        scope = self.scope() if label_ids is _SCOPE_UNSET else label_ids
        trials = self.trials_for_scope(which)
        df, total = curate_trials(self.app_state._all_labels_df, trials, scope)
        if not total:
            if not quiet:
                notify(f"Nothing left to curate in scope across the {_TRIAL_SCOPE_NOUN[which]}.")
            return 0
        if confirm and not self._confirm_bulk_curate(which, scope, total, len(trials)):
            return 0
        message = None if quiet else f"Curated {total} label(s) across {len(trials)} trial(s)."
        return self._commit(df, total, message=message)

    def curate_current_trial(self, quiet: bool = False) -> int:
        """Ctrl+C: every automated label in scope of the current trial → curated."""
        return self.curate_trial_labels(wf.TRIAL_SCOPE_SINGLE, quiet=quiet)

    def curate_visible_trials(self, confirm: bool = False) -> int:
        """Every automated label in scope, in every trial the trials table shows.

        Ctrl+C over the whole visible set rather than the current trial. The
        ``curate_trials`` workflow step's manual twin.
        """
        return self.curate_trial_labels(wf.TRIAL_SCOPE_FILTERED, confirm=confirm)

    def _confirm_bulk_curate(self, which: str, label_ids, total: int, n_trials: int) -> bool:
        """Ask before marking labels across many trials as seen by a human.

        Curating is **not** undoable: ``Ctrl+Z`` walks per-trial snapshots
        recorded by the label handlers, and curation records none — nothing
        runs automated → curated backwards. Saying so is the whole job of this
        dialog, so it says it plainly and defaults to No.
        """
        classes = "every label class" if label_ids is None else f"{len(label_ids)} label class(es)"
        noun = _TRIAL_SCOPE_NOUN[which]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(f"Curate: {noun}")
        box.setText(f"Mark {total} automated label(s) as curated, across {n_trials} {noun}, in {classes}?")
        box.setInformativeText(
            "Curated means a human has approved them — labels you have not looked at "
            "will be marked as though you had.\n\n"
            "This cannot be undone: Ctrl+Z does not take back a curation. Nothing "
            "reaches disk until you save, so closing without saving still discards it."
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        return box.exec() == QMessageBox.Yes

    def delete_trial_labels(
        self, which: str = wf.TRIAL_SCOPE_FILTERED, label_ids=_SCOPE_UNSET, confirm: bool = False
    ) -> int:
        """Delete every label of *label_ids*, in the trials *which* names.

        *label_ids* defaults to the curation scope area (``self.scope()``);
        pass an explicit set (or ``None`` for every class) to bypass it.
        Unlike curation this touches labels of every ``labeling_method``, not
        just automated ones — deleting means the event is gone.

        Each touched trial is snapshotted first, so ``Ctrl+Z`` can take the
        deletion back one trial at a time.

        *confirm* asks first: a workflow step is already a deliberate,
        written-down choice and does not ask.
        """
        if not self.app_state.ready:
            return 0
        all_df = self.app_state._all_labels_df
        scope = self.scope() if label_ids is _SCOPE_UNSET else label_ids
        trials = self.trials_for_scope(which)
        touched = []
        total = 0
        for trial in trials:
            df, n = delete_labels(all_df, [trial], scope)
            if not n:
                continue
            touched.append(trial)
            total += n
        if not total:
            notify(f"Nothing in scope to delete across the {_TRIAL_SCOPE_NOUN[which]}.")
            return 0
        if confirm and not self._confirm_bulk_delete(which, scope, total, len(touched)):
            return 0
        for trial in touched:
            self.app_state.record_label_edit(f"Delete labels: {which} (trial {trial})", trial=trial)
        df, _ = delete_labels(self.app_state._all_labels_df, touched, scope)
        return self._commit(
            df,
            total,
            message=f"Deleted {total} label(s) across {len(touched)} trial(s).",
            activate_reason="labels deleted",
        )

    def _confirm_bulk_delete(self, which: str, label_ids, total: int, n_trials: int) -> bool:
        """Ask before deleting labels across many trials — there is no server-side undo past Ctrl+Z."""
        classes = "every label class" if label_ids is None else f"{len(label_ids)} label class(es)"
        noun = _TRIAL_SCOPE_NOUN[which]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(f"Delete labels: {noun}")
        box.setText(f"Delete {total} label(s), across {n_trials} {noun}, in {classes}?")
        box.setInformativeText(
            "This removes the events outright — manual and curated labels included, not\n"
            "just automated ones.\n\n"
            "Ctrl+Z can take it back one trial at a time while this session is open. Nothing\n"
            "reaches disk until you save, so closing without saving still discards it."
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        return box.exec() == QMessageBox.Yes

    def purge_trial_labels(
        self,
        which: str = wf.TRIAL_SCOPE_FILTERED,
        min_duration_s: float = 0.010,
        label_ids=_SCOPE_UNSET,
        confirm: bool = False,
    ) -> int:
        """Drop state-interval labels of *label_ids* shorter than *min_duration_s*,
        in the trials *which* names. Point events are never touched.

        *label_ids* defaults to the curation scope area (``self.scope()``);
        pass an explicit set (or ``None`` for every class) to bypass it.
        Each touched trial is snapshotted first, so ``Ctrl+Z`` can take the
        purge back one trial at a time.
        """
        if not self.app_state.ready:
            return 0
        all_df = self.app_state._all_labels_df
        scope = self.scope() if label_ids is _SCOPE_UNSET else label_ids
        trials = self.trials_for_scope(which)
        touched = []
        total = 0
        for trial in trials:
            df, n = purge_short_labels(all_df, [trial], min_duration_s, scope)
            if not n:
                continue
            touched.append(trial)
            total += n
        if not total:
            notify(f"Nothing in scope shorter than {min_duration_s:g} s across the {_TRIAL_SCOPE_NOUN[which]}.")
            return 0
        if confirm and not self._confirm_bulk_purge(which, scope, total, len(touched), min_duration_s):
            return 0
        for trial in touched:
            self.app_state.record_label_edit(f"Purge short labels: {which} (trial {trial})", trial=trial)
        df, _ = purge_short_labels(self.app_state._all_labels_df, touched, min_duration_s, scope)
        return self._commit(
            df,
            total,
            message=f"Purged {total} label(s) shorter than {min_duration_s:g} s across {len(touched)} trial(s).",
            activate_reason="labels purged",
        )

    def stitch_trial_labels(
        self,
        which: str = wf.TRIAL_SCOPE_FILTERED,
        max_gap_s: float = 0.015,
        label_ids=_SCOPE_UNSET,
        confirm: bool = False,
    ) -> int:
        """Merge same-class state labels of *label_ids* separated by less than
        *max_gap_s*, in the trials *which* names. Point events are never touched.

        The merge rule is the changepoint correction's own
        (:func:`~ethograph.labels.intervals.stitch_intervals`); this is that
        one step alone, over a chosen scope, without the snap. *label_ids*
        defaults to the curation scope area; each touched trial is
        snapshotted first, so ``Ctrl+Z`` can take the stitch back one trial at
        a time.
        """
        if not self.app_state.ready:
            return 0
        scope = self.scope() if label_ids is _SCOPE_UNSET else label_ids
        trials = self.trials_for_scope(which)
        stitched: dict = {}
        total = 0
        for trial in trials:
            df, n = stitch_labels(self.app_state.get_trial_intervals(trial), max_gap_s, scope)
            if not n:
                continue
            stitched[trial] = df
            total += n
        if not total:
            notify(f"Nothing in scope closer than {max_gap_s:g} s across the {_TRIAL_SCOPE_NOUN[which]}.")
            return 0
        if confirm and not self._confirm_bulk_stitch(which, scope, total, len(stitched), max_gap_s):
            return 0
        all_df = self.app_state._all_labels_df
        for trial, df in stitched.items():
            self.app_state.record_label_edit(f"Stitch labels: {which} (trial {trial})", trial=trial)
            all_df = set_trial_in_tsv(all_df, trial, df)
        return self._commit(
            all_df,
            total,
            message=f"Stitched {total} label(s) closer than {max_gap_s:g} s across {len(stitched)} trial(s).",
            activate_reason="labels stitched",
        )

    def _confirm_bulk_stitch(self, which: str, label_ids, total: int, n_trials: int, max_gap_s: float) -> bool:
        """Ask before stitching — a merged label is one label, and the seam is gone."""
        classes = "every label class" if label_ids is None else f"{len(label_ids)} label class(es)"
        noun = _TRIAL_SCOPE_NOUN[which]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(f"Stitch labels: {noun}")
        box.setText(
            f"Merge {total} label(s) into their neighbours closer than {max_gap_s:g} s, "
            f"across {n_trials} {noun}, in {classes}?"
        )
        box.setInformativeText(
            "Same class, same individual, gap below the threshold. Point events are never touched.\n\n"
            "Ctrl+Z can take it back one trial at a time while this session is open. Nothing\n"
            "reaches disk until you save, so closing without saving still discards it."
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        return box.exec() == QMessageBox.Yes

    def _confirm_bulk_purge(self, which: str, label_ids, total: int, n_trials: int, min_duration_s: float) -> bool:
        """Ask before purging short labels — there is no server-side undo past Ctrl+Z."""
        classes = "every label class" if label_ids is None else f"{len(label_ids)} label class(es)"
        noun = _TRIAL_SCOPE_NOUN[which]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(f"Purge short labels: {noun}")
        box.setText(
            f"Delete {total} label(s) shorter than {min_duration_s:g} s, across {n_trials} {noun}, in {classes}?"
        )
        box.setInformativeText(
            "Point events have no duration and are never touched.\n\n"
            "Ctrl+Z can take it back one trial at a time while this session is open. Nothing\n"
            "reaches disk until you save, so closing without saving still discards it."
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        return box.exec() == QMessageBox.Yes

    def curate_labels(self, insts: list[dict]) -> int:
        """Curate the given labels (grid-view verdicts); manual ones are untouched."""
        df = self.app_state._all_labels_df
        total = 0
        for inst in insts:
            df, n = curate_label(df, inst)
            total += n
        if total:
            self._commit(df, total, message=f"Curated {total} label(s).")
        return total

    def note_labels_edited(self) -> None:
        """A label was placed, moved, deleted or undone — the verdict may have changed."""
        self.app_state.curation_changed.emit()
        self._refresh_status()
        if self._session_active and self._session_mode == "segment":
            self._sync_segment_after_edit()

    # ------------------------------------------------------------------
    # Status + metadata
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        trial = getattr(self.app_state, "trials_sel", None)
        if trial is None or not self.app_state.ready:
            self.status_label.setText("")
            return
        counts = method_counts(self.app_state._all_labels_df, trial)
        automated = counts[LABELING_AUTOMATED]
        colour = "#ff7b72" if automated else "#7ee787"
        self.status_label.setText(
            f"Trial {trial}: <span style='color:{colour};'>{automated} automated</span> · "
            f"{counts[LABELING_CURATED]} curated · {counts[LABELING_MANUAL]} manual"
        )

    # ------------------------------------------------------------------
    # Active / inactive
    # ------------------------------------------------------------------

    def activate(self, reason: str) -> None:
        """Curation has started: give the verdicts a file and start the sync.

        Idempotent, and the only path that creates a metadata TSV. Until it
        runs, a session that never curates anything writes nothing.
        """
        if self.app_state.curation_active:
            return
        self.app_state.curation_active = True
        logger.info("Curation active (%s) — per-trial verdicts will be saved.", reason)
        self._ensure_metadata_file()
        self._review_done = self._all_curated(self.app_state.trial_curation_status())
        self._metadata_timer.start()

    def deactivate(self) -> None:
        """Stop syncing (a fresh dataset curates nothing until asked to)."""
        self._metadata_timer.stop()
        self.app_state.curation_active = False

    def _ensure_metadata_file(self) -> None:
        """Point the metadata table at a TSV, creating it when there is none.

        The ``curated`` column is ours, so it never goes into a recording or
        the alignment NWB (``io/metadata_edit.py``): the loaded table is
        copied to the sidecar TSV, which becomes the metadata table for this
        dataset from here on.
        """
        trials_widget = self._trials_widget()
        if trials_widget is None or not self.app_state.ready:
            return
        trials_widget.ensure_tabular_metadata_file()

    def sync_metadata(self) -> None:
        """Push the per-trial verdicts into the metadata ``curated`` column.

        Runs on :data:`METADATA_SYNC_MS` while curation is active; a no-op
        unless a verdict differs from what the table holds, so the timer is
        cheap when idle.
        """
        if not self.app_state.curation_active:
            return
        if not self.app_state.ready or not self.app_state.trials:
            return
        trials_widget = self._trials_widget()
        if trials_widget is None:
            return
        status = self.app_state.trial_curation_status()
        mdf = getattr(self.app_state, "metadata_df", None)
        if mdf is None:
            mdf = getattr(trials_widget, "_metadata_df", None)
        if mdf is None or mdf.empty:
            return
        if curated_column_differs(mdf, status):
            trials_widget.set_column_values(
                CURATED_COLUMN, {t: (CURATED_YES if v else CURATED_NO) for t, v in status.items()}
            )
        self._review_when_done(status)

    # ------------------------------------------------------------------
    # Curator feedback: hand flags + post-curation review (labels/review_metrics.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _all_curated(status: dict[str, bool]) -> bool:
        return bool(status) and all(status.values())

    def _review_when_done(self, status: dict[str, bool]) -> None:
        """The last trial was just curated → score the session, once."""
        if not self._all_curated(status):
            self._review_done = False
            return
        if self._review_done:
            return
        self._review_done = True
        self.run_review()

    def run_review(self) -> dict[str, rm.TrialReview]:
        """Score every trial the table shows against the run that predicted it.

        Writes ``review_f1_state`` / ``review_f1_point`` into the metadata
        table — a measurement, nothing more. Which of those trials the next
        training run should see more often is decided by a human, from the
        histogram (:meth:`open_review_histogram`) or by hand.
        """
        trials_widget = self._trials_widget()
        session = getattr(self.app_state, "nc_file_path", None)
        if trials_widget is None or not self.app_state.ready or not session:
            return {}
        reviews = rm.review_session(
            session,
            self.app_state._all_labels_df,
            self.app_state.trials or [],
            tolerance_override_s=self.app_state.review_tolerance_s,
        )
        if not reviews:
            message = rm.summary(reviews)
            self.review_label.setText(message)
            notify(message)
            return reviews
        self.activate("review scored")
        for column, values in rm.review_columns(reviews).items():
            scored = {t: v for t, v in values.items() if not math.isnan(v)}
            if scored:
                trials_widget.set_column_values(column, scored)
        message = rm.summary(reviews)
        self.review_label.setText(message)
        notify(message)
        return reviews

    def open_review_histogram(self) -> None:
        """The F1 histogram: the human picks the threshold and flags once."""
        from ethograph.gui.dialog_review_histogram import ReviewHistogramDialog

        dialog = ReviewHistogramDialog(self.app_state, self.flag_trials_hard, parent=self.window())
        dialog.exec()

    def flag_trials_hard(self, trials: set[str]) -> int:
        """Flag *trials* hard in the metadata table — the histogram's one
        press. Only ever adds; ``Ctrl+T`` takes one back. Returns how many
        changed."""
        trials_widget = self._trials_widget()
        if not trials or trials_widget is None or not self.app_state.ready:
            return 0
        self.activate("trials flagged")
        values = rm.difficulty_values(getattr(self.app_state, "metadata_df", None), set(trials), trials)
        if values:
            trials_widget.set_column_values(rm.DIFFICULTY_COLUMN, values)
        notify(f"Flagged {len(values)} trial(s) hard.")
        self._sync_hard_checkbox()
        return len(values)

    # ------------------------------------------------------------------
    # Frame confidence curves: the runs behind the labels
    # ------------------------------------------------------------------

    def _curve_store_for(self, trial):
        """The prediction run whose curve *trial* is judged on: the run the
        labels were imported from, else the run on disk that predicted the
        trial, else the prediction set selected in the I/O tab."""
        store = getattr(self.app_state, "labels_pred_store", None)
        if store is not None:
            return store
        session = getattr(self.app_state, "nc_file_path", None)
        if session:
            source = rm.get_trial_meta(self.app_state._all_labels_df, trial).get("prediction_source")
            folder = rm.resolve_run(session, trial, source if isinstance(source, str) and source else None)
            if folder is not None:
                cached = self._run_stores.get(folder)
                if cached is None:
                    cached = self._run_stores[folder] = PredictionsStore(folder)
                return cached
        return getattr(self.app_state, "pred_store", None)

    def _confidence_curves(self, trials) -> dict[str, np.ndarray | None]:
        """``{trial: frame confidence curve or None}`` for *trials*."""
        self._run_stores: dict = {}
        individual = self.app_state.selected_individual()
        out: dict[str, np.ndarray | None] = {}
        for trial in trials:
            store = self._curve_store_for(trial)
            out[str(trial)] = store.get_confidence(trial, self.app_state.dt, individual=individual) if store else None
        return out

    def export_confidence_pdf(self) -> None:
        """Trial-level confidence, two ways: each trial's mean frame confidence
        into the metadata table (``model_confidence``, a number to sort and
        filter on), and every trial's curve with its labels as a PDF the
        system viewer opens. A review surface, like the grids: it decides
        nothing about a trial, and never touches ``difficulty``."""
        if not self.app_state.ready:
            return
        trials = list(self.app_state.trials or [])
        curves = self._confidence_curves(trials)
        if not any(c is not None for c in curves.values()):
            notify(
                "No frame confidence curves: the labels carry no prediction run with probabilities, "
                "and no run is selected in the I/O tab.",
                severity="warning",
            )
            return
        means = rm.trial_confidence_means(curves)
        trials_widget = self._trials_widget()
        if means and trials_widget is not None:
            self.activate("trial confidence scored")
            trials_widget.set_column_values(rm.MODEL_CONFIDENCE_COLUMN, means)
        mappings = getattr(self.labels_widget, "_mappings", None) or {}
        try:
            pdf_path, _highlighted = plot_confidence_pdf(
                {t: curves.get(str(t)) for t in trials},
                self.app_state._all_labels_df,
                self.app_state.dt,
                mappings,
                confidence_threshold=0.0,
                segment_confidence_threshold=0.0,
            )
        except (OSError, ValueError, KeyError) as exc:
            notify(f"Confidence PDF failed: {exc}", severity="error")
            return
        notify(f"model_confidence written for {len(means)} trial(s); wrote {Path(pdf_path).name}")
        open_path(pdf_path)

    def _trial_is_hard(self, trial) -> bool:
        mdf = getattr(self.app_state, "metadata_df", None)
        if mdf is None or mdf.empty or rm.DIFFICULTY_COLUMN not in mdf.columns:
            return False
        hit = mdf["trial"].astype(str) == str(trial)
        return bool(hit.any()) and rm.is_hard(mdf.loc[hit, rm.DIFFICULTY_COLUMN].iloc[0])

    def _sync_hard_checkbox(self) -> None:
        trial = getattr(self.app_state, "trials_sel", None)
        self._syncing_hard = True
        try:
            self.hard_cb.setEnabled(trial is not None and self.app_state.ready)
            self.hard_cb.setChecked(trial is not None and self._trial_is_hard(trial))
        finally:
            self._syncing_hard = False

    def _on_hard_toggled(self, checked: bool) -> None:
        if not self._syncing_hard:
            self.set_difficulty(bool(checked))

    def set_difficulty(self, hard: bool, trial=None) -> None:
        """Write *trial*'s (default: the current one) difficulty to the metadata table."""
        if trial is None:
            trial = getattr(self.app_state, "trials_sel", None)
        trials_widget = self._trials_widget()
        if trial is None or trials_widget is None or not self.app_state.ready:
            return
        self.activate("trial flagged")
        value = rm.DIFFICULTY_HARD if hard else rm.DIFFICULTY_NORMAL
        trials_widget.set_column_values(rm.DIFFICULTY_COLUMN, {str(trial): value})
        notify(f"Trial {trial}: {value}.")
        self._sync_hard_checkbox()

    def toggle_difficulty(self) -> None:
        """Ctrl+T: the current trial hard ↔ normal."""
        trial = getattr(self.app_state, "trials_sel", None)
        if trial is None or not self.app_state.ready:
            return
        self.set_difficulty(not self._trial_is_hard(trial))

    # ------------------------------------------------------------------
    # Trial changes
    # ------------------------------------------------------------------

    def _on_ready_changed(self, *_args) -> None:
        """A dataset came or went — curation starts off again."""
        self.deactivate()
        self.app_state.curve_run_path = None
        self._review_done = False
        self.review_label.setText("")
        self._sync_hard_checkbox()

    def _on_trial_changed(self) -> None:
        if not self.app_state.ready:
            return
        if self.mode() == "inspect":
            # Deferred: the trial-change cascade (data load, label view) must
            # settle before the trial's labels are restamped and redrawn.
            QTimer.singleShot(0, lambda: self.curate_current_trial(quiet=True))
        self._refresh_status()
        self._sync_hard_checkbox()
        if self._session_active and not self._jumping:
            self._follow_trial()

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------

    def open_grid_view(self):
        """Open (or raise) the label grid on the scope; returns the dialog."""
        from ethograph.gui.dialog_label_gridview import LabelGridViewDialog

        if self.meta is None:
            return None
        if self._grid_dialog is None or not self._grid_dialog.isVisible():
            self._grid_dialog = LabelGridViewDialog(self.meta, parent=self.window(), label_ids=self.scope_or_all_ids())
        self._grid_dialog.show()
        self._grid_dialog.raise_()
        self._grid_dialog.activateWindow()
        return self._grid_dialog

    def open_video_grid(self):
        """Open (or raise) the video grid on the scope; returns the dialog."""
        from ethograph.gui.dialog_video_grid import VideoGridDialog

        if self.meta is None:
            return None
        if self._video_dialog is None or not self._video_dialog.isVisible():
            self._video_dialog = VideoGridDialog(self.meta, parent=self.window(), label_ids=self.scope_or_all_ids())
        self._video_dialog.show()
        self._video_dialog.raise_()
        self._video_dialog.activateWindow()
        return self._video_dialog

    def open_workflows(self) -> None:
        """The saved curation workflows: manage them, or run one from here."""
        from ethograph.gui.dialog_curation_workflow import CurationWorkflowDialog

        if self.meta is None:
            return
        if self._workflow_dialog is None or not self._workflow_dialog.isVisible():
            self._workflow_dialog = CurationWorkflowDialog(self.meta, parent=self.window())
        self._workflow_dialog.show()
        self._workflow_dialog.raise_()
        self._workflow_dialog.activateWindow()

    # ==================================================================
    # Frame-by-frame review session
    # ==================================================================

    @property
    def session_active(self) -> bool:
        return self._session_active

    @property
    def targets(self) -> list[ReviewTarget]:
        return self._targets

    @property
    def current_index(self) -> int:
        return self._idx

    def _video_ready(self) -> bool:
        if getattr(self.app_state, "video", None) is None:
            notify("Frame-by-frame review needs a loaded video.", severity="warning")
            return False
        return True

    def _allowed_trials(self) -> set[str] | None:
        if self.nav is None:
            return None
        return self.nav._visible_trials()

    def build_queue(self) -> list[ReviewTarget]:
        """The labels in scope, in the trials the table shows: every boundary
        (frame-by-frame) or every whole label (segment review)."""
        return build_review_queue(
            self.app_state._all_labels_df,
            self.scope(),
            allowed_trials=self._allowed_trials(),
            automated_only=self.automated_only_cb.isChecked(),
            order=self.order(),
            whole_labels=self.mode() == "segment",
        )

    def _toggle_session(self) -> None:
        if self._session_active:
            self._stop()
        else:
            self.start_review()

    def start_review(self, idx: int = 0) -> bool:
        """Walk the scope's queue from *idx*. Returns whether a session started.

        The frame review nudges video frames, so it needs a video; the
        segment review plays whatever the session has (audio alone will do).
        """
        if self.mode() != "segment" and not self._video_ready():
            return False
        targets = self.build_queue()
        if not targets:
            notify("No labels in scope to review.", severity="warning")
            return False
        self._begin(targets, idx)
        return True

    def start_review_at(self, inst: dict, field: str = "point") -> bool:
        """Drop into the review at *inst* (a grid tile): the scope's queue if
        the label is in it, else a one-label queue.

        In segment review the tile's boundary does not matter — the whole
        label is the target; outside both review modes the frame review is
        entered.
        """
        if self.mode() not in REVIEW_MODES:
            idx = self.mode_combo.findData("frame")
            self.mode_combo.setCurrentIndex(idx)
        if self.mode() == "segment":
            field = FIELD_LABEL
        elif not self._video_ready():
            return False
        targets = self.build_queue()
        idx = queue_index_of(targets, inst, field)
        if idx is None:
            targets = targets_from_seeds([{**inst, "field": field}])
            idx = queue_index_of(targets, inst, field) or 0
        self._begin(targets, idx)
        return True

    def start_from_seeds(self, seeds: list[dict]) -> bool:
        """Review exactly *seeds* (one boundary each) and nothing else."""
        if not self._video_ready():
            return False
        targets = targets_from_seeds(seeds)
        if not targets:
            notify("Nothing to review.", severity="warning")
            return False
        self._begin(targets, 0)
        return True

    def restart_review(self) -> None:
        """Rebuild the queue in place: a Done elsewhere (a grid) may have
        curated labels the current session is reviewing, so what's left to
        review has changed under it."""
        if not self._session_active:
            return
        current = self._targets[self._idx] if self._targets else None
        targets = self.build_queue()
        if not targets:
            self._stop(done=True)
            return
        idx = 0
        if current is not None:
            found = queue_index_of(targets, current.inst, current.field)
            idx = found if found is not None else min(self._idx, len(targets) - 1)
        self._begin(targets, idx)

    def _begin(self, targets: list[ReviewTarget], idx: int) -> None:
        if self._session_active:
            self._teardown()
        self._targets = targets
        self._idx = min(max(idx, 0), len(targets) - 1)
        self._n_confirmed = 0
        self._n_deleted = 0
        self._session_active = True
        self._session_mode = self.mode()
        self._advance_pending = False
        self.start_stop_btn.setText("Stop review")
        self._install_session_shortcuts()
        self._curve_source = self._resolve_curve_source()
        self._curves = {}
        self._curves_key = None
        self._jump_current()

    def _stop(self, done: bool = False) -> None:
        n_confirmed, n_deleted = self._n_confirmed, self._n_deleted
        self._teardown()
        self.start_stop_btn.setText("Start review")
        self.target_label.setText("")
        self.info_label.setText("")
        self.delta_label.setText("")
        self.review_finished.emit()
        if not (n_confirmed or n_deleted):
            return
        parts = []
        if n_confirmed and self._session_mode == "segment":
            parts.append(f"moved {n_confirmed} label{'' if n_confirmed == 1 else 's'}")
        elif n_confirmed:
            parts.append(f"confirmed {n_confirmed} boundar{'y' if n_confirmed == 1 else 'ies'}")
        if n_deleted:
            parts.append(f"deleted {n_deleted} event{'' if n_deleted == 1 else 's'}")
        notify(f"{'Done' if done else 'Stopped'} — {' and '.join(parts)}. Save with Ctrl+S.")

    def _teardown(self) -> None:
        self._disarm_edit()
        self._session_active = False
        self._advance_pending = False
        self._seed_frame = None
        self._curve_source = None
        self._curves = {}
        self._curves_key = None
        self._remove_session_shortcuts()
        if self.plot_container is not None:
            self.plot_container.hide_onset_curves()

    # ------------------------------------------------------------------
    # The model's probability curves, under the label being reviewed
    # ------------------------------------------------------------------

    def _resolve_curve_source(self) -> Path | None:
        """Which run's ``onset_curves.npz`` to draw for this review session.

        Review itself never asks — that popped up on every ``_begin``,
        including the one ``restart_review`` fires after each commit, asking
        again mid-review. The pick lives on ``app_state.curve_run_path``,
        session-only, set once by :func:`~ethograph.gui.dialog_onset_model.
        predict_onsets`'s caller when more than one run exists. With exactly
        one run and nothing chosen yet, that one is used without asking.
        """
        chosen = self.app_state.curve_run_path
        if chosen and Path(chosen).is_file():
            return Path(chosen)
        session = self.app_state.nc_file_path
        if not session:
            return None
        folders = onset_curves.run_dirs(session)
        if len(folders) == 1:
            return folders[0] / onset_curves.CURVES_FILE
        return None

    def _load_curves(self) -> dict[str, onset_curves.TrialCurves]:
        """This session's chosen run's curves, keyed by trial id.

        Re-read when the file's mtime changes, so re-running the same model
        mid-review shows the new curves without any invalidation call.
        """
        if self._curve_source is None:
            return {}
        try:
            mtime = self._curve_source.stat().st_mtime
        except OSError:
            return {}
        key = (str(self._curve_source), mtime)
        if key != self._curves_key:
            self._curves = onset_curves.read_curves(self._curve_source)
            self._curves_key = key
        return self._curves

    def _draw_curves(self) -> None:
        """Draw the classes **in scope** for the trial under review.

        Scope is what the user dragged in, so a review of one class shows
        that class's belief and not every model output in the trial. The
        curves are stored trial-relative; the plot axis may be on the session
        clock, so they are shifted the way every other consumer shifts.
        """
        container = self.plot_container
        if container is None:
            return
        entry = self._load_curves().get(str(self._targets[self._idx].inst["trial"]))
        scope = self.scope()
        wanted = (
            {}
            if entry is None
            else {label: curve for label, curve in entry[1].items() if scope is None or label in scope}
        )
        if not wanted:
            container.hide_onset_curves()
            return
        trial = self._targets[self._idx].inst["trial"]
        mappings = getattr(self.labels_widget, "_mappings", {}) or {}
        colors = {label: _color_hex(mappings.get(label, {})) for label in wanted}
        offset = float(self.app_state.to_display(trial, 0.0))
        if not container.show_onset_curves(entry[0] + offset, wanted, colors):
            logger.info(
                "Onset curves exist for trial %s but no open panel can host them — open a feature panel to see them.",
                trial,
            )

    # ------------------------------------------------------------------
    # Session shortcuts: Enter, Backspace/Delete, B, N
    # ------------------------------------------------------------------

    def _install_session_shortcuts(self) -> None:
        """Bind the verdict keys application-wide for the session's life.

        Disabled while a text field has focus, exactly like the shell's
        guarded shortcuts — an enabled QShortcut would swallow the key before
        the field sees it (see gui/shortcuts.py).
        """
        if self._session_shortcuts:
            return
        bindings = [
            (Qt.Key_Backspace, self._delete_current),
            (Qt.Key_Delete, self._delete_current),
            (Qt.Key_B, self._back),
            (Qt.Key_N, self._next),
        ]
        if self._session_mode == "frame":
            # Enter confirms a frame; a segment is re-placed by clicking, so
            # the key stays free (and cannot commit a half-placed label).
            bindings += [(Qt.Key_Return, self._confirm), (Qt.Key_Enter, self._confirm)]
        for key, slot in bindings:
            shortcut = QShortcut(QKeySequence(key), self.window())
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(slot)
            self._session_shortcuts.append(shortcut)
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._sync_session_shortcuts)
        self._sync_session_shortcuts()

    def _sync_session_shortcuts(self, *_args) -> None:
        enabled = not typing_in_text_field()
        for shortcut in self._session_shortcuts:
            shortcut.setEnabled(enabled)

    def _remove_session_shortcuts(self) -> None:
        if not self._session_shortcuts:
            return
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.disconnect(self._sync_session_shortcuts)
        for shortcut in self._session_shortcuts:
            shortcut.setEnabled(False)
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._session_shortcuts = []

    def _show_shortcuts(self) -> None:
        mode = self._session_mode if self._session_active else self.mode()
        if self._shortcuts_popup is None or not self._shortcuts_popup.isVisible():
            self._shortcuts_popup = ShortcutsPopup(self.window(), mode if mode in REVIEW_MODES else "frame")
        self._shortcuts_popup.show()
        self._shortcuts_popup.raise_()

    # ------------------------------------------------------------------
    # Jumping + display
    # ------------------------------------------------------------------

    def _seed_rel(self, target: ReviewTarget) -> float:
        return target.inst["offset_s"] if target.field == "end" else target.inst["onset_s"]

    def _global_row_idx(self, inst: dict) -> int | None:
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is None or df.empty:
            return None
        mask = (df["trial"].astype(str) == str(inst["trial"])) & row_mask(df, inst)
        pos = np.flatnonzero(mask.to_numpy())
        return int(pos[0]) if len(pos) else None

    def _view_rel(self, seed_rel: float) -> TimeRange:
        """The seed-centred view window, slid (not shrunk) to stay in the trial."""
        size = float(self.app_state.get_with_default("refine_window_s"))
        half = size / 2.0
        t0, t1 = seed_rel - half, seed_rel + half
        tb = self.app_state.trial_bounds
        if tb is not None:
            if t0 < tb.start_s:
                t0, t1 = tb.start_s, min(tb.end_s, tb.start_s + size)
            elif t1 > tb.end_s:
                t0, t1 = max(tb.start_s, tb.end_s - size), tb.end_s
        return TimeRange(t0, t1)

    def _on_window_changed(self, value: float) -> None:
        self.app_state.refine_window_s = value
        if self._session_active and self.lock_checkbox.isChecked() and self.nav is not None:
            target = self._targets[self._idx]
            self.nav.set_view_range(target.inst["trial"], self._view_rel(self._seed_rel(target)))

    def _on_lock_toggled(self, checked: bool) -> None:
        if not self._session_active:
            return
        if checked:
            self._jump_current()
        else:
            self._free_navigation()

    def _free_navigation(self) -> None:
        """Widen the restriction to the normal navigation scope."""
        if self.nav is None:
            return
        self.nav._apply_slider_scope()
        pc = self.nav.plot_container
        if pc is not None:
            pc._apply_all_zoom_constraints()

    def _follow_trial(self) -> None:
        """Normal trial navigation pulls the session to that trial's first boundary."""
        trial = str(getattr(self.app_state, "trials_sel", None))
        if str(self._targets[self._idx].inst["trial"]) == trial:
            return
        for i, target in enumerate(self._targets):
            if str(target.inst["trial"]) == trial:
                self._advance_pending = False
                QTimer.singleShot(0, lambda i=i: self._follow_trial_jump(i))
                return

    def _follow_trial_jump(self, i: int) -> None:
        if not self._session_active:
            return
        self._idx = i
        self._jump_current()

    def _jump_current(self) -> None:
        self._disarm_edit()
        target = self._targets[self._idx]
        if target.field == FIELD_LABEL:
            self._jump_segment(target.inst)
            return
        inst = target.inst
        seed_rel = self._seed_rel(target)
        if self.nav is not None:
            self._jumping = True
            try:
                self.nav.jump_to_label_instance(
                    {**inst, "row_idx": self._global_row_idx(inst)},
                    seek_rel=seed_rel,
                    play=False,
                    view_rel=self._view_rel(seed_rel),
                )
                if not self.lock_checkbox.isChecked():
                    self._free_navigation()
            finally:
                self._jumping = False
        video = getattr(self.app_state, "video", None)
        seed_display = self.app_state.to_display(inst["trial"], seed_rel)
        self._seed_frame = video.time_to_frame(seed_display, round_nearest=True) if video else None
        self._update_target_display()
        self._update_delta()
        self._draw_curves()

    # ------------------------------------------------------------------
    # Segment review: play the label, arm it for a two-click edit
    # ------------------------------------------------------------------

    def _jump_segment(self, inst: dict) -> None:
        """Show the whole label with the navigation padding, play it, and arm it."""
        if self.nav is not None:
            self._jumping = True
            try:
                self.nav.jump_to_label_instance(
                    {**inst, "row_idx": self._global_row_idx(inst)},
                    seek_rel=inst["onset_s"],
                    play=math.isfinite(inst["offset_s"]),
                )
            finally:
                self._jumping = False
        self._seed_frame = None
        self._update_target_display()
        self._arm_edit()
        self._draw_curves()

    def _arm_edit(self) -> None:
        """Select the current label and enter the labels widget's edit mode.

        What ``Ctrl+E`` does after a click on the label, minus the click: the
        next two plot clicks re-place it (start, then end). Edits go by click
        even when the labelling mode is keys, so one gesture serves the whole
        review; a label outside the active branch is refused by
        ``_edit_label`` itself, and stays only playable.
        """
        found = self._current_row()
        if found is None:
            return
        _df, row_idx = found
        lw = self.labels_widget
        lw.current_labels_pos = row_idx
        lw.current_labels = int(self._targets[self._idx].inst["labels"])
        lw.current_labels_is_prediction = False
        lw._edit_label()
        if lw.old_labels_pos is None:
            self.delta_label.setText("")
            return
        lw.ready_for_label_click = True
        self.delta_label.setText(_SEGMENT_HINT)

    def _disarm_edit(self) -> None:
        """Forget a half-placed edit: a label left with one click in keeps
        its old boundaries."""
        if self._session_mode != "segment":
            return  # only the segment review arms the labels widget
        lw = self.labels_widget
        if lw.old_labels_pos is None and not lw.ready_for_label_click:
            return
        lw.old_labels_pos = None
        lw.old_labels = None
        lw._reset_label_clicks()
        lw.ready_for_label_click = False

    def _sync_segment_after_edit(self) -> None:
        """The armed label was re-placed (its row is gone, the new one is
        selected): the target follows it, and the new segment plays so the
        result is seen before N."""
        target = self._targets[self._idx]
        if target.field != FIELD_LABEL:
            return
        df = self.app_state.label_intervals
        if df is None or df.empty or row_mask(df, target.inst).any():
            return
        pos = self.labels_widget.current_labels_pos
        if pos is None or pos not in df.index:
            return
        onset_s, offset_s, _labels = get_interval_bounds(df, pos)
        target.inst["onset_s"] = float(onset_s)
        target.inst["offset_s"] = float(offset_s)
        self._n_confirmed += 1
        self._update_target_display()
        self.delta_label.setText("✓ moved  ·  V replays  ·  N next")
        self.labels_widget._play_segment()

    def _update_target_display(self) -> None:
        target = self._targets[self._idx]
        inst = target.inst
        mappings = getattr(self.labels_widget, "_mappings", {}) or {}
        mapping = mappings.get(inst["labels"], {})
        name = mapping.get("name", str(inst["labels"]))
        # Just the name, in the class colour; the boundary only where it matters
        # (a state event's start or end), then who vouches for it and how sure.
        self.target_label.setText(name)
        self.target_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_color_hex(mapping)};")
        parts = []
        if target.field in ("start", "end"):
            parts.append(_FIELD_TITLES[target.field].lower())
        method = self._current_method()
        if method:
            parts.append(method)
        confidence = self._current_confidence()
        if confidence is not None:
            parts.append(f"conf {confidence_display(confidence)}")
        self.info_label.setText("  ·  ".join(parts))

    def _current_label_row(self):
        """The current target's row in ``_all_labels_df``, or ``None``."""
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is None or df.empty:
            return None
        inst = self._targets[self._idx].inst
        mask = (df["trial"].astype(str) == str(inst["trial"])) & row_mask(df, inst)
        rows = df.loc[mask]
        return rows.iloc[0] if len(rows) else None

    def _current_method(self) -> str | None:
        row = self._current_label_row()
        if row is None or "labeling_method" not in row.index:
            return None
        return str(row["labeling_method"])

    def _current_confidence(self) -> float | None:
        row = self._current_label_row()
        if row is None or "confidence" not in row.index:
            return None
        value = row["confidence"]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)

    def _on_frame_changed(self, *_args) -> None:
        if self._session_active:
            self._update_delta()

    def _update_delta(self) -> None:
        """A frame step clears the feedback line. It used to print a running
        "moved ±N frames"; the video shows where the playhead is, and Enter
        says what it did. The segment review's hint stays up."""
        if self._targets and self._targets[self._idx].field == FIELD_LABEL:
            return
        self.delta_label.setText("")

    # ------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------

    def _current_row(self):
        """(df, row index) of the current target in ``label_intervals``, or None."""
        df = self.app_state.label_intervals
        if df is None or df.empty:
            notify("No labels in the current trial.", severity="warning")
            return None
        mask = row_mask(df, self._targets[self._idx].inst)
        pos = np.flatnonzero(mask.to_numpy())
        if not len(pos):
            notify("This label no longer exists (edited elsewhere?) — use N to skip.", severity="warning")
            return None
        return df, df.index[pos[0]]

    def _redraw_after_edit(self) -> None:
        if self.data_widget is not None:
            self.data_widget.update_main_plot(preserve_x_range=True)
        if self.labels_widget is not None:
            self.labels_widget.refresh_labels_shapes_layer()

    def _confirm(self) -> None:
        """Enter: the frame on screen is the boundary.

        A boundary that moved makes the label **manual** (and fully confident
        — a hand placed it); one confirmed where it stood is **curated**.
        """
        if not self._session_active or self._advance_pending:
            return
        target = self._targets[self._idx]
        if target.field == FIELD_LABEL:
            return  # a segment is re-placed by clicking, never by Enter
        inst = target.inst
        video = getattr(self.app_state, "video", None)
        if video is None:
            notify("No video loaded.", severity="warning")
            return
        t_display = video.frame_to_time(int(self.app_state.current_frame))
        hit = self.app_state.from_display(t_display)
        if hit is None:
            notify("The current frame lies outside any trial.", severity="warning")
            return
        trial_id, t_rel = hit
        if str(trial_id) != str(inst["trial"]):
            notify(
                f"The playhead is in trial {trial_id}, but this boundary belongs to trial {inst['trial']} — "
                "use N, or navigate back.",
                severity="warning",
            )
            return
        if target.field == "start" and t_rel >= inst["offset_s"]:
            notify("The start must stay before the end of the label.", severity="warning")
            return
        if target.field == "end" and t_rel <= inst["onset_s"]:
            notify("The end must stay after the start of the label.", severity="warning")
            return
        found = self._current_row()
        if found is None:
            return
        df, row_idx = found
        df = ensure_labeling_method(df)
        old_onset, old_offset = _num(inst["onset_s"]), _num(inst["offset_s"])

        if target.field == "end":
            df.loc[row_idx, "offset_s"] = t_rel
            inst["offset_s"] = t_rel
        else:
            df.loc[row_idx, "onset_s"] = t_rel
            if target.field == "point" and math.isfinite(inst["offset_s"]):
                df.loc[row_idx, "offset_s"] = t_rel
                inst["offset_s"] = t_rel
            inst["onset_s"] = t_rel
        moved = not (_close(old_onset, _num(inst["onset_s"])) and _close(old_offset, _num(inst["offset_s"])))
        if moved:
            df.loc[row_idx, "confidence"] = HUMAN_CONFIDENCE
            df.loc[row_idx, "labeling_method"] = LABELING_MANUAL
        elif df.loc[row_idx, "labeling_method"] == LABELING_AUTOMATED:
            df.loc[row_idx, "labeling_method"] = LABELING_CURATED

        self.app_state.record_label_edit("confirm boundary", trial=trial_id)
        self.app_state.label_intervals = df
        self.app_state.set_trial_intervals(trial_id, df)
        self.app_state.changes_saved = False
        self._n_confirmed += 1
        self._redraw_after_edit()
        self.app_state.curation_changed.emit()
        self._refresh_status()

        self.delta_label.setText("✓ placed")
        if self.auto_advance_cb.isChecked():
            self._advance_pending = True
            QTimer.singleShot(_CONFIRM_PAUSE_MS, self._advance_after_pause)

    def _delete_current(self) -> None:
        """Backspace: this event does not belong in the trial — drop the label."""
        if not self._session_active or self._advance_pending:
            return
        inst = self._targets[self._idx].inst
        trial = inst["trial"]
        current = getattr(self.app_state, "trials_sel", None)
        if current is not None and str(current) != str(trial):
            notify(
                f"The GUI is on trial {current}, but this boundary belongs to trial {trial} — navigate back first.",
                severity="warning",
            )
            return
        found = self._current_row()
        if found is None:
            return
        df, row_idx = found
        self._disarm_edit()
        self.app_state.record_label_edit("delete event", trial=trial)
        df = delete_interval(df, row_idx)
        self.app_state.label_intervals = df
        self.app_state.set_trial_intervals(trial, df)
        self.app_state.changes_saved = False
        self._n_deleted += 1
        self._redraw_after_edit()
        self.app_state.curation_changed.emit()
        self._refresh_status()
        self.delta_label.setText("✗ deleted")
        if self.auto_advance_cb.isChecked():
            self._advance_past(inst)

    def _next(self) -> None:
        """N: leave this boundary — curating it when the checkbox says so."""
        if not self._session_active or self._advance_pending:
            return
        if self.next_curates_cb.isChecked():
            inst = self._targets[self._idx].inst
            df, n = curate_label(self.app_state._all_labels_df, inst)
            self._commit(df, n, restyle=(inst, False))
        self._advance(+1)

    def _back(self) -> None:
        if not self._session_active:
            return
        self._advance(-1)

    def _advance_past(self, inst: dict) -> None:
        """Jump to the next target that is not part of *inst* (whose row is gone)."""
        idx = self._idx + 1
        while idx < len(self._targets) and self._targets[idx].inst is inst:
            idx += 1
        if idx >= len(self._targets):
            self._stop(done=True)
            return
        self._idx = idx
        self._jump_current()

    def _advance_after_pause(self) -> None:
        if not self._advance_pending or not self._session_active:
            return  # Stop / B / N intervened during the pause
        self._advance(+1)

    def _advance(self, direction: int) -> None:
        self._advance_pending = False
        new_idx = self._idx + direction
        if new_idx >= len(self._targets):
            self._stop(done=True)
            return
        self._idx = max(0, new_idx)
        self._jump_current()
