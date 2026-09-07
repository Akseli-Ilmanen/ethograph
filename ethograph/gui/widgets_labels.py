"""Widget for labeling segments in movement data."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import QMimeData, QSize, Qt, Signal
from qtpy.QtGui import QColor, QDrag
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.notify import notify
from ethograph.io.catalog import INDIVIDUAL_DIMS
from ethograph.io.metadata_table import (
    empty_metadata_df,
    metadata_tsv_path,
    save_metadata_tsv,
)
from ethograph.labels import onset_curves
from ethograph.labels.intervals import (
    EVENT_TYPE_POINT,
    EVENT_TYPE_STATE,
    add_interval,
    add_point,
    delete_interval,
    empty_intervals,
    find_interval_at,
    find_point_at,
    get_interval_bounds,
    load_label_mapping,
    save_label_mapping,
    subject_mask,
)
from ethograph.labels.plots import plot_confidence_pdf
from ethograph.labels.predictions import PredictionsStore, merge_as_labels
from ethograph.labels.tsv_store import get_trial_from_tsv, load_labels_tsv

# Glyphs used to indicate the kind of a label in the table.
#   ━ (horizontal bar) = state event — visually conveys "spans across time"
#   ● (filled circle)  = point event — visually conveys "single moment"
# Chosen so the two are unmistakable at a glance, and don't collide with the
# parenthesised shortcut letter.
_EVENT_TYPE_GLYPH = {
    EVENT_TYPE_STATE: "━",
    EVENT_TYPE_POINT: "●",
}

# Fixed branch -> draw-position mapping. There are never more than 3 branches;
# branch 0 always renders "full" (main), branch 1 always "top1", branch 2
# always "top2". This is a hard rule — not user-configurable.
_BRANCH_POSITION = {0: "main", 1: "top1", 2: "top2"}
_BRANCH_POSITION_LABEL = {0: "Full", 1: "Top1", 2: "Top2"}
MAX_LABEL_BRANCHES = 3
from ethograph.gui.project import project_dir_of  # noqa: E402
from ethograph.utils.paths import defaults_dir, find_mapping_file  # noqa: E402

logger = logging.getLogger(__name__)

from .app_constants import (  # noqa: E402
    DEFAULT_LABEL_OVERLAY_MODES,
    DEFAULT_LAYOUT_SPACING,
    LABEL_OVERLAY_MODE_BOTTOM,
    LABEL_OVERLAY_MODE_FULL,
    LABEL_OVERLAY_MODE_NONE,
    LABEL_OVERLAY_PLOT_TYPES,
    LABELLING_MODE_FRAME,
    LABELLING_MODES,
    LABELS_TABLE_COLOR_COLUMN_WIDTH,
    LABELS_TABLE_ID_COLUMN_WIDTH,
    LABELS_TABLE_ROW_HEIGHT,
    LABELS_WIDGET_SIZE_HINT_HEIGHT,
)
from .file_dialogs import browse_open_dir, browse_open_file  # noqa: E402
from .widgets_curation import CurationPanel, drag_label_ids  # noqa: E402


class BranchTable(QTableWidget):
    """QTableWidget subclass that supports cross-table drag & drop of labels."""

    label_moved = Signal(int, int)  # (label_id, target_branch)

    def __init__(self, branch_idx: int, parent=None):
        super().__init__(parent)
        self.branch_idx = branch_idx
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)

    def startDrag(self, supportedActions):
        """Drag every selected label (comma-separated ids, see :func:`drag_label_ids`).

        A multi-selection is what the curation scope area is for — pull several
        classes in at once; a branch drop moves them all.
        """
        ids = self.selected_label_ids()
        if not ids:
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(",".join(str(i) for i in ids))
        drag.setMimeData(mime)
        drag.exec_(Qt.MoveAction)

    def selected_label_ids(self) -> list[int]:
        """The distinct label ids of the selected cells, current item first."""
        ids: list[int] = []
        current = self.currentItem()
        candidates = ([current] if current is not None else []) + list(self.selectedItems())
        for item in candidates:
            label_id = item.data(Qt.UserRole)
            if label_id is not None and int(label_id) not in ids:
                ids.append(int(label_id))
        return ids

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        source = event.source()
        if source is self:
            event.ignore()
            return
        label_ids = drag_label_ids(event.mimeData().text())
        if not label_ids:
            event.ignore()
            return
        event.acceptProposedAction()
        for label_id in label_ids:
            self.label_moved.emit(label_id, self.branch_idx)


class LabelsWidget(QWidget):
    """Widget for labeling movement labels in time series data."""

    highlight_spaceplot = Signal(float, float)

    def __init__(self, shell, app_state, parent=None):
        super().__init__(parent=parent)
        self.shell = shell
        self.app_state = app_state

        self.data_widget = None  # Will be set after creation

        self.plot_container = None  # Will be set after creation
        self.meta_widget = None  # Will be set after creation
        self.changepoints_widget = None  # Will be set after creation
        self.io_widget = None  # Will be set after creation

        # Make widget focusable for keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Remove Qt event filter and key event logic
        # Instead, rely on napari's @viewer.bind_key for global shortcuts
        # Shortcut bindings are now handled outside the widget

        # Labeling state
        self._mappings: dict[int, dict[str, Any]] = {}
        self.ready_for_label_click = False
        self.ready_for_play_click = False
        self.first_click = None
        self.second_click = None
        self.selected_labels = 0

        # Current  selection for editing (interval DataFrame index)
        self.current_labels_pos: int | None = None  # DataFrame index of selected interval
        self.current_labels: int | None = None  # ID of currently selected
        self.current_labels_is_prediction: bool = False  # Whether selected  is from predictions
        # Edge-triggered: warn once when Predictions loses its Top1/Top2 slot to
        # shown branches, not on every redraw. Reset once a slot is free again.
        self._predictions_slot_warned = False

        # Edit mode state
        self.old_labels_pos: int | None = None  # Original interval index when editing
        self.old_labels: int | None = None  # Original ID when editing

        # Frame tracking for  display
        self.previous_frame: int | None = None

        # UI components — branch tables
        self.labels_table = None  # kept for backward compat; points to active branch table
        self._branch_sections: dict[int, dict] = {}  # branch_idx → {"label", "table", "widget", "checkbox"}
        self._branches_layout = None
        self._mapping_file_path: str | None = None
        self._previous_active_branch: int | None = None
        self._label_table_dialog = None

        self._setup_ui()

        mapping_path = find_mapping_file(project_dir=project_dir_of(self.app_state))
        self._mapping_file_path = str(mapping_path) if mapping_path else None
        self._mappings = load_label_mapping(mapping_path) if mapping_path else {}
        self.app_state._label_mappings = self._mappings
        self.app_state._active_branch = 0
        self.app_state._branch_shown = {0: True}
        self._populate_labels_table()

    def refresh_mapping_for_data_dir(self, data_dir: Path | str):
        """Re-resolve mapping.txt now that a data directory is known.

        Called by DataWidget after loading a .nc file. A session's own
        ``.ethograph/mapping.txt`` overrides the project's; when the two
        disagree the override is said out loud, so two meanings for one id
        never pass unnoticed.
        """
        project = project_dir_of(self.app_state)
        mapping_path = find_mapping_file(data_dir, project_dir=project)
        if mapping_path is None:
            return
        project_mapping = project / "mapping.txt" if project is not None else None
        if project_mapping is not None and project_mapping.is_file() and mapping_path != project_mapping:
            if mapping_path.read_text(encoding="utf-8").split() != project_mapping.read_text(encoding="utf-8").split():
                notify(
                    f"Using this session's own {mapping_path}, which differs from the project's {project_mapping}.",
                    severity="warning",
                )
        current_path = Path(self.io_widget.mapping_file_path_edit.text()) if self.io_widget else None
        if current_path == mapping_path:
            return
        self._reload_mapping(str(mapping_path))
        if self.io_widget:
            self.io_widget.mapping_file_path_edit.setText(str(mapping_path))

    def set_data_widget(self, data_widget):
        """Set reference to the data widget for plot updates."""
        self.data_widget = data_widget
        self.curation_panel.set_data_widget(data_widget)

    def _mark_changes_unsaved(self):
        """Mark that changes have been made and are not saved."""
        self.app_state.changes_saved = False

    def set_plot_container(self, plot_container):
        """Set the plot container reference and connect click handler to all plots."""
        self.plot_container = plot_container
        plot_container.set_label_mappings(self._mappings)
        self.curation_panel.set_plot_container(plot_container)
        self._sync_active_label_ids()
        # A half-placed state label belongs to the trial it was started in — its
        # anchor must not survive into the next one (the second click would be
        # refused as spanning two trials anyway).
        self.app_state.trial_changed.connect(self._reset_label_clicks)

        for plot in [
            *plot_container.spectrogram_plots,
            *plot_container.audio_trace_plots,
            *plot_container.heatmap_plots,
            *plot_container.neo_trace_plots,
            plot_container.ephys_trace_plot,
            *plot_container.line_plots,
        ]:
            if plot is not None:
                plot.plot_clicked.connect(self._on_plot_clicked)
        # Panels created later (any dynamic panel) get the same click handling.
        plot_container.panel_added.connect(lambda p: p.plot_clicked.connect(self._on_plot_clicked))
        # Static-image primary has no frame clock — the marker drives the
        # current-label overlay directly (video sessions use frame_changed).
        plot_container.time_marker_updated.connect(self._on_marker_time_for_overlay)
        plot_container.pending_label_hovered.connect(self._follow_pending_hover)

    def set_meta_widget(self, meta_widget):
        """Set reference to the meta widget for layout refresh."""
        self.meta_widget = meta_widget
        self.curation_panel.set_meta(meta_widget)

    def attach_video_groupbox(self, groupbox):
        """Add the video label-name overlay control to the video context group."""
        groupbox.layout().addWidget(self.hide_label_cb)

    def attach_overlay_groupbox(self, groupbox, row_layout=None):
        """Add the per-plot-type label control into the "Label overlay" groupbox.

        Shares a row with the groupbox's existing toggles instead of taking a
        row of its own, so the button costs no extra vertical space.
        """
        self.labels_per_plot_btn = QPushButton("Per-plot type…")
        self.labels_per_plot_btn.setToolTip(
            "Choose how label rectangles render on each plot type: full plot, bottom strip, or not at all"
        )
        self.labels_per_plot_btn.clicked.connect(self._show_labels_per_plot_dialog)
        target = row_layout if row_layout is not None else groupbox.layout()
        # Insert before the trailing stretch (if any) rather than appending after it.
        insert_at = target.count()
        if insert_at > 0 and target.itemAt(insert_at - 1).spacerItem() is not None:
            insert_at -= 1
        target.insertWidget(insert_at, self.labels_per_plot_btn)

    def _show_labels_per_plot_dialog(self):
        modes = dict(DEFAULT_LABEL_OVERLAY_MODES)
        modes.update(self.app_state.label_overlay_modes or {})
        dialog = LabelsPerPlotDialog(modes, self)
        if dialog.exec_():
            self.app_state.label_overlay_modes = dialog.get_modes()
            if self.plot_container is not None:
                self.plot_container.labels_redraw_needed.emit()

    def open_label_table(self):
        """Open (or raise) the spreadsheet view of the whole label table."""
        from ethograph.gui.dialog_label_table import LabelTableDialog

        if self._label_table_dialog is None or not self._label_table_dialog.isVisible():
            self._label_table_dialog = LabelTableDialog(
                self.app_state,
                mappings=self._mappings,
                on_changed=self._on_label_table_changed,
                parent=self.window(),
            )
        else:
            self._label_table_dialog.set_mappings(self._mappings)
        self._label_table_dialog.show()
        self._label_table_dialog.raise_()
        self._label_table_dialog.activateWindow()
        return self._label_table_dialog

    def _on_label_table_changed(self):
        """A cell edit or row deletion in the label table landed: catch up.

        The click-selection is dropped: it names a row of the trial's frame,
        and the row it named may have just moved or gone.
        """
        self.current_labels_pos = None
        self.current_labels = None
        self.current_labels_is_prediction = False
        self._mark_changes_unsaved()
        self.curation_panel.note_labels_edited()
        nav = getattr(self.app_state, "navigation_widget", None)
        if nav is not None and hasattr(nav, "on_labels_changed"):
            nav.on_labels_changed()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
        self.refresh_labels_shapes_layer()

    def plot_all_labels(self, intervals_df, predictions_df=None):
        """Plot all labels for current trial based on interval data.

        Builds the per-slot draw config (Main / Top1 / Top2) from the
        currently selected sources in app_state and forwards to PlotContainer.

        Args:
            intervals_df: DataFrame with onset_s, offset_s, labels, individual columns
            predictions_df: Optional prediction intervals DataFrame
        """
        if self.plot_container is None:
            return

        slots = self._compute_label_slots(intervals_df, predictions_df)
        self.plot_container.draw_all_labels(slots)

    def _compute_label_slots(self, intervals_df, predictions_df):
        """Build draw-ready slot dicts from shown branches (fixed positions) + predictions.

        Branch 0 always draws "main" (full), branch 1 "top1", branch 2 "top2" —
        each only if that branch exists and its visibility checkbox is on.
        Predictions (toggled separately) fill whichever of top1/top2 isn't
        already occupied by a shown branch.
        """
        state = self.app_state
        slots: list[dict] = []
        if intervals_df is not None and not intervals_df.empty:
            for branch_idx, position in _BRANCH_POSITION.items():
                if branch_idx not in self._branch_sections:
                    continue
                if not state._branch_shown.get(branch_idx, True):
                    continue
                branch_ids = {
                    lid
                    for lid, data in self._mappings.items()
                    if isinstance(lid, int) and lid != 0 and data.get("branch", 0) == branch_idx
                }
                if not branch_ids:
                    continue
                slots.append({"df": intervals_df, "label_ids": branch_ids, "position": position})

        if state._show_predictions_overlay and predictions_df is not None and not predictions_df.empty:
            occupied = {slot["position"] for slot in slots}
            pred_position = "top1" if "top1" not in occupied else ("top2" if "top2" not in occupied else None)
            if pred_position is not None:
                slots.append({"df": predictions_df, "label_ids": None, "position": pred_position})
                self._predictions_slot_warned = False
            elif not self._predictions_slot_warned:
                self._predictions_slot_warned = True
                notify(
                    "Predictions has no free strip — Top1 and Top2 are both taken by shown branches. "
                    "Hide a branch to make room.",
                    severity="warning",
                )

        return slots

    def sizeHint(self):
        return QSize(300, LABELS_WIDGET_SIZE_HINT_HEIGHT)

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Where a new label's boundaries come from: a click on the plots, or
        # the label key pressed at the frame on screen (see activate_label).
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Labelling:"))
        self.labelling_mode_combo = QComboBox()
        for key, text in LABELLING_MODES.items():
            self.labelling_mode_combo.addItem(text, key)
        self.labelling_mode_combo.setToolTip(
            "On the time series: press a label key, then click the plots (twice for a state event).\n"
            "At the current frame: the label key itself places the boundary at the frame on screen — "
            "a point event on one press, a state event on two (start, then end after navigating)."
        )
        idx = self.labelling_mode_combo.findData(self.app_state.get_with_default("labelling_mode"))
        self.labelling_mode_combo.setCurrentIndex(max(0, idx))
        self.labelling_mode_combo.currentIndexChanged.connect(self._on_labelling_mode_changed)
        mode_row.addWidget(self.labelling_mode_combo, stretch=1)
        layout.addLayout(mode_row)

        self.ribbon_auto_cb = QCheckBox("Open a label timeline when no panel is shown")
        self.ribbon_auto_cb.setToolTip(
            "A session with only a video has no panel to draw labels on. With this ticked, an empty "
            "time axis (the Label timeline, also in the ➕ Add panel popup) opens on load in that case."
        )
        self.ribbon_auto_cb.setChecked(bool(self.app_state.get_with_default("label_ribbon_auto")))
        self.ribbon_auto_cb.toggled.connect(self._on_ribbon_auto_toggled)
        layout.addWidget(self.ribbon_auto_cb)
        # The timeline is the frame mode's companion (labelling from the video
        # alone); in plots mode there is always a panel to label on.
        self.ribbon_auto_cb.setVisible(self.frame_labelling)

        # Legend explaining the per-label glyphs (above Branch 0).
        legend = QLabel(
            f"  {_EVENT_TYPE_GLYPH[EVENT_TYPE_STATE]} state event"
            f"   ·   {_EVENT_TYPE_GLYPH[EVENT_TYPE_POINT]} point event"
            "   (right-click a label to change)"
        )
        legend.setStyleSheet("QLabel { color: #aaa; font-size: 11px; padding: 2px 0px; }")
        legend.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(legend)

        # "Hide label" lives in the video context of the right sidebar (see
        # attach_video_groupbox) since it's a video-overlay display setting,
        # not a labeling action.
        self.hide_label_cb = QCheckBox("Hide label")
        self.hide_label_cb.setToolTip("Hide the label-name overlay shown on the video during playback")
        self.hide_label_cb.setChecked(bool(self.app_state.get_with_default("hide_label_text")))
        self.hide_label_cb.toggled.connect(lambda v: setattr(self.app_state, "hide_label_text", v))

        # Scrollable area for branch tables
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll_content = QWidget()
        self._branches_layout = QVBoxLayout(scroll_content)
        self._branches_layout.setSpacing(6)
        self._branches_layout.setContentsMargins(0, 0, 0, 0)
        self._branches_layout.addStretch()
        scroll.setWidget(scroll_content)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(scroll, stretch=1)

        # "+" button to add branches, and the raw table behind these classes.
        button_row = QHBoxLayout()
        add_branch_btn = QPushButton("+")
        add_branch_btn.setToolTip("Add a new label branch")
        add_branch_btn.setFixedWidth(28)
        add_branch_btn.clicked.connect(self._add_new_branch)
        button_row.addWidget(add_branch_btn)

        table_btn = QPushButton("Label table…")
        table_btn.setToolTip("Inspect every trial's labels as a spreadsheet: filter, sort, edit cells, delete rows")
        table_btn.clicked.connect(self.open_label_table)
        button_row.addWidget(table_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        # Curation lives under the label tables: every curation question
        # starts with "which labels", and those are the rows right above —
        # drag them into the scope area (see widgets_curation.py).
        self.curation_panel = CurationPanel(self.app_state, self)
        layout.addWidget(self.curation_panel)

    _TABLE_STYLE = """
        QTableWidget { gridline-color: transparent; background: #444; color: #fff; }
        QTableWidget::item { padding: 0px 2px; color: #fff; }
        QTableWidget::item:selected { background: #ffe066; color: #000; }
        QHeaderView::section { padding: 0px 2px; background: #888; color: #fff; }
    """

    def _create_branch_table(self, branch_idx: int) -> BranchTable:
        """Create a single labels table widget for the given branch."""
        table = BranchTable(branch_idx)
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["ID", "Name (Shortcut)", "C", "ID", "Name (Shortcut)", "C"])
        table.verticalHeader().setVisible(False)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Fixed)

        table.setColumnWidth(0, LABELS_TABLE_ID_COLUMN_WIDTH)
        table.setColumnWidth(2, LABELS_TABLE_COLOR_COLUMN_WIDTH)
        table.setColumnWidth(3, LABELS_TABLE_ID_COLUMN_WIDTH)
        table.setColumnWidth(5, LABELS_TABLE_COLOR_COLUMN_WIDTH)

        table.setSelectionBehavior(QAbstractItemView.SelectItems)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        table.verticalHeader().setDefaultSectionSize(LABELS_TABLE_ROW_HEIGHT)
        table.setStyleSheet(self._TABLE_STYLE)

        table.itemSelectionChanged.connect(self._on_table_selection_changed)
        table.label_moved.connect(self._on_label_dropped)
        table.setContextMenuPolicy(Qt.CustomContextMenu)
        table.customContextMenuRequested.connect(lambda pos, t=table: self._on_label_context_menu(t, pos))
        return table

    def _add_branch_section(self, branch_idx: int):
        """Add a UI section (checkbox + header label + table) for a branch.

        Branch <-> draw-position is fixed (0=Full, 1=Top1, 2=Top2; see
        ``_BRANCH_POSITION``) — at most :data:`MAX_LABEL_BRANCHES` branches
        ever exist. The checkbox controls whether the branch is shown as an
        overlay; clicking the branch name makes it the active (editable) one.
        """
        if branch_idx in self._branch_sections or branch_idx not in _BRANCH_POSITION:
            return

        section_widget = QWidget()
        section_layout = QVBoxLayout(section_widget)
        section_layout.setContentsMargins(0, 4, 0, 0)
        section_layout.setSpacing(1)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        show_checkbox = QCheckBox()
        show_checkbox.setChecked(self.app_state._branch_shown.setdefault(branch_idx, True))
        show_checkbox.setToolTip(
            "Show this branch's labels on the plots.\n"
            "This does NOT choose which branch you edit — click the branch name for that."
        )
        show_checkbox.stateChanged.connect(lambda qt_state, b=branch_idx: self._on_branch_shown_changed(b, qt_state))
        header_row.addWidget(show_checkbox)

        header_label = QLabel()
        header_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        header_label.setCursor(Qt.PointingHandCursor)
        header_label.setToolTip("Click to make this the branch you edit (the checkbox only shows/hides it)")
        header_label.mousePressEvent = lambda _event, b=branch_idx: self.set_active_branch(b)
        header_row.addWidget(header_label)

        if branch_idx > 0:
            delete_btn = QPushButton("x")
            delete_btn.setFixedSize(20, 20)
            delete_btn.setToolTip("Delete this branch (must be empty)")
            delete_btn.setStyleSheet(
                "QPushButton { color: #aaa; background: transparent; border: none; font-weight: bold; } QPushButton:hover { color: #f66; }"  # noqa: E501
            )
            delete_btn.clicked.connect(lambda _, b=branch_idx: self._delete_branch(b))
            header_row.addWidget(delete_btn)
        header_row.addStretch()
        section_layout.addLayout(header_row)

        table = self._create_branch_table(branch_idx)
        section_layout.addWidget(table)

        self._branch_sections[branch_idx] = {
            "label": header_label,
            "table": table,
            "widget": section_widget,
            "checkbox": show_checkbox,
        }

        # Insert before the stretch at the end of _branches_layout
        insert_pos = self._branches_layout.count() - 1
        self._branches_layout.insertWidget(insert_pos, section_widget)

        # Set labels_table to first branch for backward compat
        if self.labels_table is None:
            self.labels_table = table

        self._update_branch_header_styles()

    def _update_branch_header_styles(self):
        """Mark which branch is the editable one.

        The checkbox (display) and the header (editable) were easy to confuse
        when the only difference between an active and an inactive branch was
        the shade of its name, so the active one says so in words.
        """
        active = self.app_state._active_branch
        for branch_idx, section in self._branch_sections.items():
            name = f"Branch {branch_idx} ({_BRANCH_POSITION_LABEL[branch_idx]})"
            if branch_idx == active:
                section["label"].setText(f"✎ {name} — editing")
                section["label"].setStyleSheet("QLabel { color: #ffe066; font-weight: bold; }")
            else:
                section["label"].setText(name)
                section["label"].setStyleSheet("QLabel { color: #999; font-weight: normal; }")

    def set_active_branch(self, branch_idx: int):
        """Make *branch_idx* the active (editable) branch."""
        if branch_idx not in self._branch_sections:
            return
        current = self.app_state._active_branch
        if branch_idx == current:
            return
        self._previous_active_branch = current
        self.app_state._active_branch = branch_idx
        self._update_branch_header_styles()

    def _on_branch_shown_changed(self, branch_idx: int, qt_state):
        """Handle a branch's visibility checkbox being toggled."""
        self.app_state._branch_shown[branch_idx] = Qt.CheckState(qt_state) == Qt.Checked
        self._sync_active_label_ids()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
        self.refresh_labels_shapes_layer()

    def _add_new_branch(self):
        """Add a new empty branch (triggered by '+' button), up to MAX_LABEL_BRANCHES."""
        existing = set(self._branch_sections.keys())
        if len(existing) >= MAX_LABEL_BRANCHES:
            notify(f"Maximum of {MAX_LABEL_BRANCHES} label branches (Full / Top1 / Top2)", severity="warning")
            return
        new_idx = next(b for b in sorted(_BRANCH_POSITION) if b not in existing)
        self._add_branch_section(new_idx)
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)

    def _delete_branch(self, branch_idx: int):
        """Delete a branch. Only allowed if it has no labels."""
        has_labels = any(
            isinstance(lid, int) and lid != 0 and data.get("branch", 0) == branch_idx
            for lid, data in self._mappings.items()
        )
        if has_labels:
            notify("Cannot delete branch — move all labels out first", severity="error")
            return
        if branch_idx not in self._branch_sections:
            return
        section = self._branch_sections.pop(branch_idx)
        section["widget"].setParent(None)
        section["widget"].deleteLater()
        self.app_state._branch_shown.pop(branch_idx, None)
        if self.app_state._active_branch == branch_idx:
            self.app_state._active_branch = 0 if 0 in self._branch_sections else next(iter(self._branch_sections), 0)
        if self._previous_active_branch == branch_idx:
            self._previous_active_branch = None
        if self.labels_table is section["table"]:
            self.labels_table = next((s["table"] for s in self._branch_sections.values()), None)
        self._update_branch_header_styles()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)

    def toggle_branch(self):
        """Swap the active branch with the previously-active one (Shift+B)."""
        target = self._previous_active_branch
        if target is None or target not in self._branch_sections:
            return
        self.set_active_branch(target)

    def _on_label_dropped(self, label_id: int, target_branch: int):
        """Handle a label being dragged from one branch table to another."""
        if label_id not in self._mappings:
            return
        old_branch = self._mappings[label_id].get("branch", 0)
        if old_branch == target_branch:
            return
        self._mappings[label_id]["branch"] = target_branch
        self.app_state._label_mappings = self._mappings
        self._save_current_mapping()
        self._populate_labels_table()
        self._sync_active_label_ids()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
        self.refresh_labels_shapes_layer()
        notify(f"Moved '{self._mappings[label_id]['name']}' to branch {target_branch}")

    def _on_label_context_menu(self, table: QTableWidget, pos):
        """Right-click on a label cell -> show event-type toggle menu."""
        item = table.itemAt(pos)
        if item is None:
            return
        label_id = item.data(Qt.UserRole)
        if label_id is None or label_id not in self._mappings:
            return

        current = self._mappings[label_id].get("event_type", EVENT_TYPE_STATE)
        target = EVENT_TYPE_POINT if current == EVENT_TYPE_STATE else EVENT_TYPE_STATE
        target_label = "Point event" if target == EVENT_TYPE_POINT else "State event"

        menu = QMenu(table)
        action = menu.addAction(f"Mark as {target_label}")
        chosen = menu.exec_(table.viewport().mapToGlobal(pos))
        if chosen is action:
            self._set_label_event_type(label_id, target)

    def _set_label_event_type(self, label_id: int, event_type: str):
        """Switch a label between state and point; persist to mapping.txt."""
        if label_id not in self._mappings:
            return
        if self._mappings[label_id].get("event_type", EVENT_TYPE_STATE) == event_type:
            return
        self._mappings[label_id]["event_type"] = event_type
        self.app_state._label_mappings = self._mappings
        self._save_current_mapping()
        self._populate_labels_table()
        self.refresh_labels_shapes_layer()
        kind_label = "point event" if event_type == EVENT_TYPE_POINT else "state event"
        notify(f"'{self._mappings[label_id]['name']}' is now a {kind_label}")

    def _save_current_mapping(self):
        """Write the current mappings back to the loaded mapping.txt file."""
        path = self._mapping_file_path
        if not path and self.io_widget:
            path = self.io_widget.mapping_file_path_edit.text()
        if not path:
            logger.warning("_save_current_mapping: no path set, skipping save")
            return
        save_label_mapping(path, self._mappings)
        n_point = sum(
            1 for d in self._mappings.values() if isinstance(d, dict) and d.get("event_type") == EVENT_TYPE_POINT
        )
        logger.info(
            "Saved mapping to %s (%d labels, %d point)",
            path,
            len(self._mappings),
            n_point,
        )

    def _sync_active_label_ids(self):
        """Push current active label IDs to plot container."""
        if self.plot_container:
            self.plot_container.set_active_label_ids(self.app_state.active_label_ids)

    def _browse_mapping_file(self):
        """Browse for a mapping.txt file and reload mappings."""
        current = find_mapping_file(project_dir=project_dir_of(self.app_state))
        start_dir = str(current.parent) if current else str(defaults_dir())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select mapping.txt file",
            start_dir,
            "Text files (*.txt);;All Files (*)",
        )
        if file_path:
            self.io_widget.mapping_file_path_edit.setText(file_path)
            self._reload_mapping(file_path)

    def _reload_mapping(self, mapping_path: str):
        """Reload mappings from the specified path."""
        try:
            self._mappings = load_label_mapping(Path(mapping_path))
            self._mapping_file_path = mapping_path
            n_point = sum(
                1 for d in self._mappings.values() if isinstance(d, dict) and d.get("event_type") == EVENT_TYPE_POINT
            )
            logger.info(
                "Loaded mapping from %s (%d labels, %d point)",
                mapping_path,
                len(self._mappings),
                n_point,
            )
            self.app_state._label_mappings = self._mappings
            # Reset branch state — active branch starts on the first branch present in the file.
            new_branches = {data.get("branch", 0) for data in self._mappings.values() if isinstance(data, dict)}
            first_branch = min(new_branches) if new_branches else 0
            self.app_state._active_branch = first_branch
            self.app_state._branch_shown = dict.fromkeys(new_branches, True) if new_branches else {0: True}
            self._previous_active_branch = None
            # Remove stale branch UI sections
            for b in list(self._branch_sections):
                section = self._branch_sections.pop(b)
                section["widget"].setParent(None)
                section["widget"].deleteLater()
            if self.plot_container:
                self.plot_container.set_label_mappings(self._mappings)
            if self.changepoints_widget:
                self.changepoints_widget.set_motif_mappings(self._mappings)
            if self.data_widget and self.data_widget.navigation_widget:
                self.data_widget.navigation_widget.set_mappings(self._mappings)
            if self._label_table_dialog is not None and self._label_table_dialog.isVisible():
                self._label_table_dialog.set_mappings(self._mappings)
            self._populate_labels_table()
            self._sync_active_label_ids()
            self.refresh_labels_shapes_layer()
            if self.data_widget:
                self.data_widget.update_main_plot(preserve_x_range=True)
            notify(f"Loaded {len(self._mappings) - 1} labels from {Path(mapping_path).name}")
        except FileNotFoundError:
            notify(f"Mapping file not found: {mapping_path}", "warning")

    def _create_temporary_labels(self):
        """Open dialog to create temporary labels for this session."""
        dialog = TemporaryLabelsDialog(self)
        if dialog.exec_():
            labels = dialog.get_labels()
            if labels:
                from ethograph.utils.paths import default_config_dir

                data_dir = Path(self.app_state.nc_file_path).parent if self.app_state.nc_file_path else None
                config_dir = default_config_dir(data_dir)
                mapping_path = config_dir / "mapping_temporary.txt"
                mapping_path.parent.mkdir(parents=True, exist_ok=True)
                with open(mapping_path, "w") as f:
                    f.write("0 background\n")
                    for i, label in enumerate(labels, start=1):
                        f.write(f"{i} {label}\n")

                self.io_widget.mapping_file_path_edit.setText(str(mapping_path))
                self._mappings = load_label_mapping(mapping_path)
                self.app_state._label_mappings = self._mappings
                if self.plot_container:
                    self.plot_container.set_label_mappings(self._mappings)
                if self.changepoints_widget:
                    self.changepoints_widget.set_motif_mappings(self._mappings)
                if self.data_widget and self.data_widget.navigation_widget:
                    self.data_widget.navigation_widget.set_mappings(self._mappings)
                self._populate_labels_table()
                self.refresh_labels_shapes_layer()
                if self.data_widget:
                    self.data_widget.update_main_plot(preserve_x_range=True)
                notify(f"Loaded {len(labels)} temporary labels")

    def _import_predictions_from_folder(self):
        """A segmentation run's own output folder — real intervals, confidence curves."""
        folder = browse_open_dir(
            self,
            self.app_state,
            "Select predictions folder",
            preferred_dir=self.app_state.nc_file_path,
        )
        if not folder:
            return
        individual = self.app_state.selected_individual() or "default"
        try:
            store = PredictionsStore(folder)
            labels_df, _confidence_levels = store.load_all(self.app_state.dt, individual)
        except (FileNotFoundError, ValueError, AssertionError) as e:
            notify(str(e), severity="error")
            return
        self._pin_curve_run(folder)
        self._finish_predictions_import(labels_df, store, folder)

    def _pin_curve_run(self, folder: str) -> None:
        """A folder picked here by hand is the frame-review confidence source too.

        Picking a specific folder is a stronger, more explicit signal than
        anything ``CurationPanel._resolve_curve_source`` could infer on its
        own — no reason to leave it to guess (or show nothing) among several
        run folders under ``labels/`` once the user has pointed straight at
        one. A folder without its own :data:`onset_curves.CURVES_FILE`
        (e.g. imported from a plain ``.tsv``) leaves the existing pick alone.
        """
        curves_file = Path(folder) / onset_curves.CURVES_FILE
        if curves_file.is_file():
            self.app_state.curve_run_path = str(curves_file)

    def _import_predictions_from_tsv(self):
        """A plain labels TSV as predictions — e.g. a second annotator's file, for comparison.

        No confidence curve (there is no ``.npz``): the "Confidence" checkbox
        simply has nothing to show for this predictions set.
        """
        path = browse_open_file(
            self,
            self.app_state,
            "Select predictions TSV file",
            "TSV files (*.tsv)",
            preferred_dir=self.app_state.nc_file_path,
        )
        if not path:
            return
        try:
            labels_df = load_labels_tsv(path)
        except (FileNotFoundError, ValueError) as e:
            notify(str(e), severity="error")
            return
        self._finish_predictions_import(labels_df, None, path)

    def _finish_predictions_import(self, labels_df, store, source_text: str):
        """A predictions set was loaded — dispatch on the panel's Load-as combo.

        "Import as labels" writes it into the working labels themselves and
        never touches the overlay state; everything below is the overlay
        ("compare with ground truth") path.
        """
        if self.io_widget.pred_load_mode() == "labels":
            self._import_predictions_as_labels(labels_df, source_text)
            return
        threshold = self.io_widget.pred_confidence_threshold_spin.value()
        self.app_state.pred_labels_df = labels_df
        self.app_state.pred_store = store
        self.app_state.pred_confidence_threshold = threshold

        self.app_state._show_predictions_overlay = True
        if self.data_widget:
            self.data_widget.refresh_trials_confidence()
        self.io_widget.pred_confidence_pdf_btn.setEnabled(store is not None)
        self.io_widget.pred_file_path_edit.setText(source_text)

        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
            self.data_widget._update_confidence_overlay()
            if self.data_widget.plot_container:
                self.data_widget.plot_container.labels_redraw_needed.emit()
        self.refresh_labels_shapes_layer()

    def _import_predictions_as_labels(self, predicted_df, source_text: str) -> None:
        """Write a loaded prediction set straight into the working labels.

        Merge (the panel's checkbox) keeps every existing row and only adds
        predictions for (trial, class, individual) pairs the current labels
        don't already cover (:func:`~ethograph.labels.predictions.merge_as_labels`,
        the onset model's own "never override" rule); unticked, the prediction
        set replaces the working labels outright. What this is about to do is
        said in the panel itself (``IOWidget.pred_labels_warning_label``, red
        text beside the combo) rather than a confirmation popup here — it
        stays visible while the combo and checkbox are still being decided,
        instead of springing the choice back at click time. Not an undoable
        label edit either way, like curating visible trials.
        """
        existing = self.app_state._all_labels_df
        has_existing = existing is not None and not existing.empty
        merge = has_existing and self.io_widget.pred_merge_checkbox.isChecked()
        merged = merge_as_labels(existing, predicted_df) if merge else predicted_df
        self.app_state._all_labels_df = merged
        self.app_state.clear_label_history()
        self.app_state.changes_saved = False
        current = getattr(self.app_state, "trials_sel", None)
        if current is not None:
            self.app_state.label_intervals = get_trial_from_tsv(merged, current)
        if self.data_widget:
            self.data_widget.refresh_individual_choices()
            self.data_widget.update_main_plot(preserve_x_range=True)
            if self.data_widget.plot_container:
                self.data_widget.plot_container.labels_redraw_needed.emit()
        self.refresh_labels_shapes_layer()
        self.curation_panel.note_labels_edited()
        self.io_widget.pred_file_path_edit.setText(source_text)
        verb = "Merged" if merge else "Imported"
        notify(f"{verb} {len(predicted_df)} prediction row(s) as labels. Save with Ctrl+S.")

    def _on_confidence_threshold_changed(self, _value):
        self.app_state.pred_confidence_threshold = self.io_widget.pred_confidence_threshold_spin.value()
        self.app_state.pred_segment_confidence_threshold = self.io_widget.pred_segment_confidence_threshold_spin.value()

    def _plot_confidence_pdf(self):
        store = getattr(self.app_state, "pred_store", None)
        labels_df = getattr(self.app_state, "pred_labels_df", None)
        if labels_df is None or labels_df.empty:
            notify("No predictions loaded.", severity="warning")
            return
        if store is None:
            notify(
                "This predictions set has no confidence curves (loaded from a plain .tsv, not a run folder).",
                severity="warning",
            )
            return
        try:
            # Load all confidence arrays at click time for the PDF (one-off)
            individual = self.app_state.selected_individual()
            confidence_map = {
                trial: store.get_confidence(trial, self.app_state.dt, individual=individual)
                for trial in self.app_state.trials
            }
            pdf_path, highlighted = plot_confidence_pdf(
                confidence_map,
                labels_df,
                self.app_state.dt,
                self._mappings,
                confidence_threshold=self.app_state.pred_confidence_threshold,
                segment_confidence_threshold=self.app_state.pred_segment_confidence_threshold,
            )

            # Update metadata table with mean model confidence per trial
            mdf = getattr(self.app_state, "metadata_df", None)
            if mdf is None or mdf.empty:
                mdf = empty_metadata_df(self.app_state.trials)
            else:
                mdf = mdf.copy()
            if "trial" not in mdf.columns:
                mdf["trial"] = list(self.app_state.trials)

            mdf_index = mdf.set_index("trial", drop=False)
            for trial in self.app_state.trials:
                arr = confidence_map.get(trial)
                try:
                    mean_confidence = float(np.nanmean(arr)) if arr is not None else float("nan")
                except Exception:
                    mean_confidence = float("nan")
                mdf_index.loc[trial, "model_confidence"] = mean_confidence
                mdf_index.loc[trial, "model_confidence_level"] = "low" if highlighted.get(trial, False) else "high"

            mdf_updated = mdf_index.reset_index(drop=True)
            self.app_state.metadata_df = mdf_updated

            # Save to metadata TSV (explicit metadata_path or sidecar next to nc)
            md_path = getattr(self.app_state, "metadata_path", None)
            if not md_path and getattr(self.app_state, "nc_file_path", None):
                md_path = metadata_tsv_path(self.app_state.nc_file_path)
            if md_path:
                try:
                    save_metadata_tsv(md_path, mdf_updated)
                    # ensure app_state knows about the metadata path
                    if not getattr(self.app_state, "metadata_path", None):
                        self.app_state.metadata_path = str(md_path)
                    notify(f"Saved metadata with model confidence to {Path(md_path).name}")
                except Exception as e:
                    notify(f"Failed saving metadata: {e}", severity="warning")

            if self.data_widget:
                self.data_widget.refresh_trials_confidence()
            os.startfile(str(pdf_path))
        except Exception as e:
            notify(str(e), severity="error")

    labels_TO_KEY = {}

    # Row 1: 1-0 (Labels 1-10)
    number_keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    for i, key in enumerate(number_keys):
        _id = i + 1 if key != "0" else 10
        labels_TO_KEY[_id] = key

    # Row 2: Q-P (Labels 11-20)
    qwerty_row = ["q", "w", "e", "r", "t", "z", "u", "i", "o", "p"]
    for i, key in enumerate(qwerty_row):
        _id = i + 11
        labels_TO_KEY[_id] = key.upper()  # Display as uppercase for clarity

    # Row 3: A-; (Labels 21-30)
    home_row = ["a", "s", "d", "f", "g", "h", "j", "k", "l", "y"]
    for i, key in enumerate(home_row):
        _id = i + 21
        labels_TO_KEY[_id] = key.upper() if key != ";" else ";"  # Keep ; as is

    # Row 4: F1-F10 (Labels 31-40)
    fn_row = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
    for i, key in enumerate(fn_row):
        _id = i + 31
        labels_TO_KEY[_id] = key

    # Also provide reverse mapping for key to _id
    KEY_TO_labels = {v.lower(): k for k, v in labels_TO_KEY.items()}

    def _populate_labels_table(self):
        """Populate per-branch tables with loaded mappings."""
        # Collect branches present in the mappings
        branches: set[int] = set()
        clamped = False
        for lid, data in self._mappings.items():
            if isinstance(lid, int):
                b = data.get("branch", 0)
                if b not in _BRANCH_POSITION:
                    data["branch"] = MAX_LABEL_BRANCHES - 1
                    clamped = True
                    b = MAX_LABEL_BRANCHES - 1
                branches.add(b)
        if not branches:
            branches = {0}
        if clamped:
            notify(f"Some labels used a branch beyond the max of {MAX_LABEL_BRANCHES} — clamped", severity="warning")

        # Ensure branch sections exist for all branches in the mapping
        for b in sorted(branches):
            if b not in self._branch_sections:
                self._add_branch_section(b)

        # Remove sections for branches that no longer have any labels,
        # but keep user-added empty branches (those with no mapping entries yet)
        # Only clean up branches when reloading a mapping file (branches come from file)
        # Don't remove branches that the user manually added via "+" button

        # Populate each branch table
        for branch_idx, section in self._branch_sections.items():
            table = section["table"]
            items = [
                (lid, data)
                for lid, data in self._mappings.items()
                if isinstance(lid, int) and lid != 0 and data.get("branch", 0) == branch_idx
            ]
            items.sort(key=lambda kv: kv[0])
            half = max((len(items) + 1) // 2, 1)
            table.clearContents()
            table.setRowCount(half)

            for i, (_id, data) in enumerate(items):
                row = i % half
                col_offset = 0 if i < half else 3

                id_item = QTableWidgetItem(str(_id))
                id_item.setData(Qt.UserRole, _id)
                table.setItem(row, col_offset, id_item)

                shortcut = self.labels_TO_KEY.get(_id, "?")
                event_type = data.get("event_type", EVENT_TYPE_STATE)
                glyph = _EVENT_TYPE_GLYPH.get(event_type, _EVENT_TYPE_GLYPH[EVENT_TYPE_STATE])
                name_with_shortcut = f"{glyph} {data['name']} ({shortcut})"
                name_item = QTableWidgetItem(name_with_shortcut)
                name_item.setData(Qt.UserRole, _id)
                kind_label = "Point event" if event_type == EVENT_TYPE_POINT else "State event"
                name_item.setToolTip(f"{kind_label} — right-click to change")
                table.setItem(row, col_offset + 1, name_item)

                color_item = QTableWidgetItem()
                color = data["color"]
                qcolor = QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                color_item.setBackground(qcolor)
                color_item.setData(Qt.UserRole, _id)
                table.setItem(row, col_offset + 2, color_item)

            # Auto-size table height to fit all rows (no cap)
            row_count = table.rowCount()
            table.setFixedHeight(LABELS_TABLE_ROW_HEIGHT * row_count + table.horizontalHeader().height() + 4)

        # Keep backward-compat labels_table pointing to first branch
        if self._branch_sections:
            first_branch = min(self._branch_sections)
            self.labels_table = self._branch_sections[first_branch]["table"]

        self._sync_active_label_ids()

    def _on_table_selection_changed(self):
        """Handle table cell selection changes by activating the selected label."""
        sender = self.sender()
        if not isinstance(sender, QTableWidget):
            return
        selected = sender.selectedItems()
        if selected:
            item = selected[0]
            _id = item.data(Qt.UserRole)
            if _id is not None:
                self.activate_label(_id)

    def _on_labelling_mode_changed(self, _index: int) -> None:
        mode = self.labelling_mode_combo.currentData()
        if mode == self.app_state.get_with_default("labelling_mode"):
            return
        self.app_state.labelling_mode = mode
        self.ribbon_auto_cb.setVisible(self.frame_labelling)
        if self.frame_labelling and self.meta_widget is not None and self.app_state.ready:
            self.meta_widget.ensure_label_ribbon()
        # A half-placed label belongs to the mode it was started in.
        self._reset_label_clicks()
        self.ready_for_label_click = False

    def _on_ribbon_auto_toggled(self, checked: bool) -> None:
        self.app_state.label_ribbon_auto = bool(checked)
        if checked and self.meta_widget is not None and self.app_state.ready:
            self.meta_widget.ensure_label_ribbon()

    @property
    def frame_labelling(self) -> bool:
        """Whether the label key places the boundary at the frame on screen."""
        return self.app_state.get_with_default("labelling_mode") == LABELLING_MODE_FRAME

    def activate_label(self, _key):
        """Activate a label by shortcut: select cell, set up for labeling, and scroll to it.

        In the "at the current frame" mode the key does not arm a plot click —
        it *is* the placement (:meth:`_place_at_current_frame`).
        """
        _id = self.KEY_TO_labels.get(str(_key).lower(), _key)
        if _id not in self._mappings:
            return

        # Only labels in the active branch can be edited.
        label_branch = self._mappings[_id].get("branch", 0)
        if label_branch != self.app_state._active_branch:
            return

        if self.frame_labelling:
            self._place_at_current_frame(_id)
            return

        self.selected_labels = _id
        self.ready_for_label_click = True
        self._reset_label_clicks()
        self._select_label_row(_id)

    def current_display_time(self) -> float | None:
        """Where the playhead is, in the display clock.

        With a video it is the frame on screen — the same reading
        frame-by-frame review commits — so a label placed from the video lands
        on a frame, never between two. Without one it is the time marker.
        """
        video = getattr(self.app_state, "video", None)
        if video is not None:
            return float(video.frame_to_time(int(self.app_state.current_frame)))
        if self.plot_container is None:
            return None
        for plot in self.plot_container._visible_plots():
            return float(plot.time_marker.value())
        return None

    def _place_at_current_frame(self, _id: int) -> None:
        """The label key pressed in "at the current frame" mode.

        A point class lands on the frame on screen at once. A state class
        starts there on the first press (the pending preview shows it) and
        ends on the second, wherever the user has navigated to meanwhile —
        playback, the timeline or frame stepping. A different key while a
        state label is half placed abandons it and starts that class.
        """
        if self.first_click is not None and _id != self.selected_labels:
            pending = self._mappings.get(self.selected_labels, {}).get("name", self.selected_labels)
            notify(f"Abandoned the half-placed '{pending}' label", severity="warning")
            self._reset_label_clicks()
        self.selected_labels = _id
        self._select_label_row(_id)

        t_display = self.current_display_time()
        if t_display is None:
            notify("No playhead to place the label at — open a panel or a video first", severity="warning")
            return
        placed = self.app_state.from_display(t_display, strict=True)
        if placed is None:
            notify("The playhead falls between trials — no label placed", severity="warning")
            self._reset_label_clicks()
            return
        p_trial, p_rel = placed
        if p_trial != self.app_state.trials_sel:
            notify("The playhead is in another trial — no label placed", severity="warning")
            self._reset_label_clicks()
            return

        if self._active_label_is_point():
            self._apply_point(p_rel)
        elif self.first_click is None:
            self.first_click = p_rel
            self._show_pending_label(p_rel)
        elif p_rel == self.first_click:
            key = self.labels_TO_KEY.get(_id, "the label key")
            notify(f"Start and end are the same frame — move to the end frame and press {key} again")
        else:
            self.second_click = p_rel
            self._apply_label()

    def _select_label_row(self, _id: int) -> None:
        """Highlight the label's row in its branch table and scroll to it."""
        for section in self._branch_sections.values():
            table = section["table"]
            table.blockSignals(True)
            for row in range(table.rowCount()):
                for col in [0, 3]:
                    item = table.item(row, col)
                    if item and item.data(Qt.UserRole) == _id:
                        table.setCurrentItem(item)
                        table.scrollToItem(item)
                        table.blockSignals(False)
                        return
            table.blockSignals(False)

    def _current_individual(self) -> str:
        """Return the currently selected individual name for interval operations."""
        ind = self.app_state.selected_individual()
        if ind is not None:
            return ind
        _ds = self.app_state.ds
        _ind_dim = next((n for n in INDIVIDUAL_DIMS if _ds is not None and n in _ds.coords), None)
        if _ind_dim is not None:
            return str(_ds.coords[_ind_dim].values[0])
        return "default"

    def _current_receiver(self) -> str:
        """The receiver the labels being drawn are about, "" for a solo one.

        Together with :meth:`_current_individual` this is the label subject:
        every label is created, found and drawn for exactly one pair.
        """
        return self.app_state.selected_receiver()

    @property
    def ready_for_label_click(self) -> bool:
        """Whether the next plot click places a label boundary."""
        return self._ready_for_label_click

    @ready_for_label_click.setter
    def ready_for_label_click(self, armed: bool) -> None:
        # Mirrored onto app_state so BasePlot._handle_click can deliver a
        # closing click that Qt reports as a double-click.
        self._ready_for_label_click = bool(armed)
        self.app_state.label_drawing_armed = bool(armed)

    def _on_plot_clicked(self, click_info):
        """Handle mouse clicks on the lineplot widget.

        Args:
            click_info: dict with 'x' (time coordinate), 'button' (Qt button
                constant) and, from a real plot, 'plot' (the panel clicked).
        """

        t_display = click_info["x"]
        button = click_info["button"]
        clicked_plot = click_info.get("plot")

        if t_display is None or not self.app_state.ready:
            return

        individual = self._current_individual()

        # The click arrives in the display clock; selection/storage work in
        # trial-relative time on ONE trial. Resolve the trial under the click
        # once — in trial basis this is the identity on the current trial.
        resolved = self.app_state.from_display(t_display)
        click_trial, t_rel = resolved if resolved is not None else (self.app_state.trials_sel, t_display)

        try:
            # Left-click outside label-drawing mode selects the segment under the
            # cursor, so pressing "V" plays it back. Right-click stays minimal
            # (seek only) so it's fast.
            if button == Qt.LeftButton and not self.ready_for_label_click:
                if click_trial != self.app_state.trials_sel:
                    self._switch_trial_for_click(click_trial)
                self._check_labels_click(t_rel, individual)

        except (KeyError, IndexError, ValueError, AttributeError) as e:
            logger.error("Error in plot click handling: %s", e)
            return

        # Without video, any click (not in label mode) jumps the time marker
        if getattr(self.app_state, "video", None) is None and not self.ready_for_label_click:
            if self.plot_container is not None:
                self.plot_container.update_time_marker_by_time(t_display)
            if button == Qt.LeftButton:
                return

        # When a label is armed (label-drawing mode), a click places the onset
        # then the offset (point events: a single click). Either mouse button
        # works. Otherwise right-click seeks the video; the red cursor follows
        # automatically via the video's frame_changed signal.
        if self.ready_for_label_click:
            # Snap to the nearest changepoint the clicked panel draws (display
            # clock — the changepoint times come back on the plot axis).
            if self.changepoints_widget and self.changepoints_widget.is_changepoint_correction_enabled():
                t_display = self._snap_to_changepoint_time(t_display, clicked_plot)

            # A placed label belongs to exactly ONE trial: strict resolution
            # refuses inter-trial gaps, and the two clicks of a state event
            # must land in the same trial.
            placed = self.app_state.from_display(t_display, strict=True)
            if placed is None:
                notify("Click falls between trials — no label placed", severity="warning")
                self._reset_label_clicks()
                return
            p_trial, p_rel = placed
            if self.first_click is not None and p_trial != self.app_state.trials_sel:
                notify("Label would span two trials — cancelled", severity="warning")
                self._reset_label_clicks()
                return
            if p_trial != self.app_state.trials_sel:
                self._switch_trial_for_click(p_trial)

            # Point events: single click drops a marker at the snapped time.
            # State events: two clicks bound the interval.
            if self._active_label_is_point():
                self._apply_point(p_rel)
            elif self.first_click is None:
                self.first_click = p_rel
                self._show_pending_label(p_rel)
            else:
                self.second_click = p_rel
                self._apply_label()

        elif button == Qt.RightButton and self.app_state.video:
            # Seek to the nearest frame but keep the playhead on the exact click.
            self._seek_to_frame(t_display)

    def _switch_trial_for_click(self, trial_id):
        """Session-basis click on another trial's span: make it current.

        The session axis already shows the right place, so the view must not
        re-center (``_preserve_x_range_next``); the trial-change machinery
        loads that trial's data/video underneath the click.
        """
        state = self.app_state
        state._preserve_x_range_next = True
        state._marker_driven_trial_switch = True
        state.trials_sel = trial_id
        nav = getattr(state, "navigation_widget", None)
        combo = getattr(nav, "trials_combo", None)
        if combo is not None:
            combo.blockSignals(True)
            combo.setCurrentText(str(trial_id))
            combo.blockSignals(False)
        state.trial_changed.emit()

    def _active_label_is_point(self) -> bool:
        """True iff the currently selected label class is declared as a point."""
        lid = self.selected_labels
        if not lid or lid not in self._mappings:
            return False
        return self._mappings[lid].get("event_type", EVENT_TYPE_STATE) == EVENT_TYPE_POINT

    def _check_labels_click(self, t_clicked: float, individual: str) -> bool:
        """Check if the click is on an existing interval or point, and select it.

        **Any label the user can see is selectable** — the gate is the shown
        branches (``active_label_ids``), not the active one.  Selection is a
        read affordance: it drives playback (V), the space-plot highlight and
        the heatmap sort, and gating it on the active branch made all three
        silently do nothing on a label that was plainly visible.  Mutation
        stays branch-scoped instead: :meth:`_delete_label` and
        :meth:`_edit_label` refuse a selection outside the active branch, and
        the clicked class is only adopted for drawing when it is editable.

        A click that misses every branch falls through to the shown
        Predictions overlay (:meth:`_check_predictions_click`) — clicking a
        predicted segment selects it too, so V plays back *that* segment.

        Args:
            t_clicked: Time in seconds of the click
            individual: Individual name to check
        """
        df = self.app_state.label_intervals
        if df is not None and not df.empty:
            active_ids = self.app_state.active_label_ids

            # Points take precedence over overlapping state intervals: they're
            # smaller targets, so a near-hit is almost certainly intentional.
            # State intervals are only selected when no point matches.
            # Each lookup falls back to ANY individual: a loaded file may store a
            # different individual name than the current selection (e.g. a TSV
            # from another session), and a label the user can see and click must
            # be selectable — otherwise playback (V) silently fails on it.
            tolerance_s = self._point_click_tolerance_s()
            idx = find_point_at(df, t_clicked, individual, tolerance_s, label_ids=active_ids)
            if idx is None:
                idx = find_point_at(df, t_clicked, None, tolerance_s, label_ids=active_ids)
            if idx is not None:
                row = df.loc[idx]
                t = float(row["onset_s"])
                labels = int(row["labels"])
                self.current_labels = labels
                self.current_labels_pos = idx
                self.current_labels_is_prediction = False
                self.highlight_spaceplot.emit(self._to_display(t), self._to_display(t))
                self._adopt_clicked_class(labels)
                return True

            # No point near the click — fall through to state intervals.
            idx = find_interval_at(df, t_clicked, individual, label_ids=active_ids)
            if idx is None:
                idx = find_interval_at(df, t_clicked, None, label_ids=active_ids)
            if idx is not None:
                onset_s, offset_s, labels = get_interval_bounds(df, idx)
                self.current_labels = labels
                self.current_labels_pos = idx
                self.current_labels_is_prediction = False
                self.highlight_spaceplot.emit(self._to_display(onset_s), self._to_display(offset_s))
                self._adopt_clicked_class(labels)
                return True

        return self._check_predictions_click(t_clicked, individual)

    def _check_predictions_click(self, t_clicked: float, individual: str) -> bool:
        """Same lookup as :meth:`_check_labels_click`, over the shown Predictions overlay.

        Predictions are read-only — no branch/mapping, no class adoption,
        and :meth:`_delete_label`/:meth:`_edit_label` refuse a prediction
        selection — but they select and drive V playback just the same.
        Only searched while the overlay is actually shown: a click where a
        hidden prediction sits should not select it.
        """
        if not self.app_state._show_predictions_overlay:
            return False
        df = self.app_state.pred_labels_df
        if df is None or df.empty:
            return False
        if "trial" in df.columns:
            df = df[df["trial"] == self.app_state.trials_sel]
            if df.empty:
                return False

        tolerance_s = self._point_click_tolerance_s()
        idx = find_point_at(df, t_clicked, individual, tolerance_s)
        if idx is None:
            idx = find_point_at(df, t_clicked, None, tolerance_s)
        if idx is not None:
            row = df.loc[idx]
            t = float(row["onset_s"])
            self.current_labels = int(row["labels"])
            self.current_labels_pos = idx
            self.current_labels_is_prediction = True
            self.highlight_spaceplot.emit(self._to_display(t), self._to_display(t))
            return True

        idx = find_interval_at(df, t_clicked, individual)
        if idx is None:
            idx = find_interval_at(df, t_clicked, None)
        if idx is not None:
            onset_s, offset_s, labels = get_interval_bounds(df, idx)
            self.current_labels = labels
            self.current_labels_pos = idx
            self.current_labels_is_prediction = True
            self.highlight_spaceplot.emit(self._to_display(onset_s), self._to_display(offset_s))
            return True
        return False

    def _is_editable_label(self, label_id: int | None) -> bool:
        """True iff *label_id*'s class belongs to the currently active branch."""
        if label_id is None:
            return False
        editable = self.app_state.editable_label_ids
        return editable is None or int(label_id) in editable

    def _adopt_clicked_class(self, label_id: int) -> None:
        """Make the clicked label's class the one new labels are drawn with.

        Only for the active branch: clicking a label of another branch selects
        it (for playback) but must not change what the next drawn label is.
        """
        if self._is_editable_label(label_id):
            self.selected_labels = label_id

    def _refuse_foreign_branch(self, label_id: int | None) -> bool:
        """Warn and return True when *label_id* is outside the active branch."""
        if self._is_editable_label(label_id):
            return False
        name = self._mappings.get(label_id, {}).get("name", label_id)
        notify(
            f"'{name}' belongs to another branch — activate that branch to edit it.",
            severity="warning",
        )
        return True

    def _point_click_tolerance_s(self) -> float:
        """Click tolerance for picking a point event, in seconds.

        Scales with the visible time range so a few pixels of slop translate
        to a sensible time window at any zoom level.  Falls back to 0.05 s
        when the plot's current x-range cannot be read.
        """
        plot = getattr(self.plot_container, "current_plot", None)
        if plot is not None and hasattr(plot, "get_current_xlim"):
            try:
                xmin, xmax = plot.get_current_xlim()
                visible = float(xmax) - float(xmin)
                if visible > 0:
                    return visible * 0.005
            except (AttributeError, TypeError, ValueError):
                pass
        return 0.05

    def _snap_to_changepoint_time(self, t_clicked: float, plot=None) -> float:
        """Snap *t_clicked* (display clock) to the nearest changepoint *plot* draws.

        Unbounded: the nearest candidate wins however far away it is. The
        candidates are :meth:`_snap_candidates` — what is on the clicked
        panel, so a click never lands on a mark the user cannot see.
        """
        cp_times = self._snap_candidates(plot)
        if len(cp_times) == 0:
            logger.debug("snap: no changepoints drawn for the click at %.4f s", t_clicked)
            return t_clicked
        snapped = float(cp_times[np.argmin(np.abs(cp_times - t_clicked))])
        logger.debug("snap: %.4f s -> %.4f s (%d candidates)", t_clicked, snapped, len(cp_times))
        return snapped

    def _snap_candidates(self, plot) -> np.ndarray:
        """The changepoint times (display clock) drawn on *plot*.

        An audio panel draws the trial's audio changepoints. A feature panel
        draws the masks targeting **its** feature at **its** keypoint /
        individual selections — the same reading ``XarrayLoader.select``
        renders, so drawn and snapped are one set. A panel that draws no
        changepoints (ephys, raster, heatmap, or a click with no panel) takes
        the active feature panel's, as the changepoint jump keys do.
        """
        container = self.plot_container
        loader = getattr(self.app_state, "data_loader", None)
        if container is None or loader is None:
            return np.array([], dtype=np.float64)

        if plot is not None and plot in [*container.spectrogram_plots, *container.audio_trace_plots]:
            ds = self.app_state.ds
            if ds is None or "audio_cp_onsets" not in ds.data_vars or "audio_cp_offsets" not in ds.data_vars:
                return np.array([], dtype=np.float64)
            onsets = ds["audio_cp_onsets"].values.astype(np.float64)
            offsets = ds["audio_cp_offsets"].values.astype(np.float64)
            return np.unique(np.concatenate([onsets, offsets]))

        feature_plot = plot if plot in container.line_plots else container.get_current_plot()
        feature = feature_plot._effective_feature()
        if not feature:
            return np.array([], dtype=np.float64)
        wb = self.app_state.window_bounds
        t0, t1 = (wb.start_s, wb.end_s) if wb is not None else (None, None)
        return loader.get_cp_times(feature, feature_plot._effective_selections(), t0=t0, t1=t1)

    def _post_label_cleanup(self, placed_onset: float, placed_offset: float, individual: str, receiver: str):
        """Purge only the slivers ``add_interval`` just created.

        Scoped cleanup: when the new interval cuts through an existing one,
        ``add_interval`` leaves left/right remnants whose boundary touches the
        new interval at ``±1e-3`` s.  Those remnants are the only intervals we
        consider here — the just-placed interval and everything else in the
        trial are left untouched.

        Runs in both new-label and Ctrl+E edit mode.  To opt out of snapping
        entirely (and therefore any boundary changes from snap), toggle
        changepoint correction off with Ctrl+B.
        """
        cw = self.changepoints_widget
        if cw is None:
            return
        min_duration_s, _ = cw.get_apply_label_cleanup_params()
        if min_duration_s <= 0:
            return

        df = self.app_state.label_intervals
        if df is None or df.empty:
            return

        eps = 1e-3
        durations = df["offset_s"] - df["onset_s"]
        same_ind = subject_mask(df, individual)
        touches_left = np.isclose(df["offset_s"], placed_onset - eps, atol=eps / 10)
        touches_right = np.isclose(df["onset_s"], placed_offset + eps, atol=eps / 10)
        sliver = same_ind & (touches_left | touches_right) & (durations < min_duration_s)

        to_drop = df.index[sliver].tolist()
        if not to_drop:
            return
        df = df.drop(index=to_drop).reset_index(drop=True)
        trial = self.app_state.trials_sel
        self.app_state.set_trial_intervals(trial, df)
        self.app_state.label_intervals = df

    def _apply_label(self):
        """Apply the selected label to the selected time range using intervals."""
        if self.first_click is None or self.second_click is None:
            return

        onset_s = min(self.first_click, self.second_click)
        offset_s = max(self.first_click, self.second_click)
        individual = self._current_individual()
        receiver = self._current_receiver()

        self.app_state.record_label_edit("move label" if self.old_labels_pos is not None else "place label")

        self.highlight_spaceplot.emit(self._to_display(onset_s), self._to_display(offset_s))

        df = self.app_state.label_intervals
        if df is None:
            df = empty_intervals()

        # If editing, delete the old interval first
        was_edit = self.old_labels_pos is not None
        if was_edit:
            if self.old_labels_pos in df.index:
                df = delete_interval(df, self.old_labels_pos)
            self.old_labels_pos = None
            self.old_labels = None

        # Only the active branch's labels are editable — every other branch's
        # labels must be protected from trimming/overwriting, whether or not
        # that branch is currently shown as an overlay.
        editable_ids = self.app_state.editable_label_ids
        if editable_ids is not None:
            all_ids = {lid for lid, d in self._mappings.items() if isinstance(lid, int) and lid != 0}
            protected = all_ids - editable_ids
        else:
            protected = None
        df = add_interval(
            df,
            onset_s,
            offset_s,
            self.selected_labels,
            individual,
            protected_label_ids=protected,
            individual_rec=receiver,
        )
        self.app_state.label_intervals = df
        self.app_state.set_trial_intervals(self.app_state.trials_sel, df)
        self._post_label_cleanup(onset_s, offset_s, individual, receiver)
        if self.changepoints_widget:
            self.changepoints_widget.cp_correction_from_labelling()
        df = self.app_state.label_intervals

        # Auto-select the newly created interval for immediate playback
        new_idx = find_interval_at(df, (onset_s + offset_s) / 2, individual, individual_rec=receiver)
        self.current_labels_pos = new_idx
        self.current_labels = self.selected_labels
        self.current_labels_is_prediction = False

        self._reset_label_clicks()
        self.ready_for_label_click = False

        self._mark_changes_unsaved()
        # add_interval / add_point stamp the new row LABELING_MANUAL; the
        # panel only has to re-read the trial's verdict.
        self.curation_panel.note_labels_edited()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
        # Placed from the frame on screen, the playhead is already where the
        # user wants to continue from; jumping back to the onset would undo
        # the navigation that found the end.
        if not self.frame_labelling:
            self._seek_to_frame(self._to_display(onset_s))
        self.refresh_labels_shapes_layer()

    def _apply_point(self, t_clicked: float):
        """Insert a point event at *t_clicked* for the active label class.

        Single-click counterpart to :meth:`_apply_label`.  Snapping happens
        upstream in :meth:`_on_plot_clicked`, so *t_clicked* is already
        snapped if changepoint correction is enabled.

        If we entered label-click mode via :meth:`_edit_label`
        (``old_labels_pos`` set), the previous row is deleted first so the
        point effectively moves to the new time.
        """
        individual = self._current_individual()
        receiver = self._current_receiver()

        self.app_state.record_label_edit("move point" if self.old_labels_pos is not None else "place point")

        df = self.app_state.label_intervals
        if df is None:
            df = empty_intervals()

        if self.old_labels_pos is not None:
            if self.old_labels_pos in df.index:
                df = delete_interval(df, self.old_labels_pos)
            self.old_labels_pos = None
            self.old_labels = None

        df = add_point(df, t_clicked, self.selected_labels, individual, individual_rec=receiver)
        self.app_state.label_intervals = df
        self.app_state.set_trial_intervals(self.app_state.trials_sel, df)
        # Point events don't cut through other intervals → no slivers to purge.
        df = self.app_state.label_intervals

        self.current_labels_pos = None  # selecting an existing point comes later
        self.current_labels = self.selected_labels
        self.current_labels_is_prediction = False

        self._reset_label_clicks()
        self.ready_for_label_click = False

        self._mark_changes_unsaved()
        # add_interval / add_point stamp the new row LABELING_MANUAL; the
        # panel only has to re-read the trial's verdict.
        self.curation_panel.note_labels_edited()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
        if not self.frame_labelling:
            self._seek_to_frame(self._to_display(t_clicked))
        self.refresh_labels_shapes_layer()

    def _to_display(self, t_rel: float) -> float:
        """Current trial's trial-relative time → the plot axis's clock."""
        return self.app_state.to_display(self.app_state.trials_sel, t_rel)

    def _show_pending_label(self, t_rel: float) -> None:
        """Show where a state label started, until its second click lands."""
        if self.plot_container is None:
            return
        mapping = self._mappings.get(self.selected_labels)
        if mapping is None:
            return
        color_rgb = tuple(int(c * 255) for c in mapping["color"])
        self.plot_container.show_pending_label(self._to_display(t_rel), color_rgb)

    def _follow_pending_hover(self, t_display: float) -> None:
        """Scrub the video to the cursor while a state label's end is being aimed.

        The preview rectangle says where the second click will land on the
        trace; the video says what the animal is doing there. The seek moves
        the playhead with it (as any seek does); the anchor stays put on the
        dashed line. A frame already on screen is not re-decoded, and playback
        is never interrupted.
        """
        if self.first_click is None:
            return
        video = getattr(self.app_state, "video", None)
        if video is None or video.is_playing:
            return
        frame = video.time_to_frame(t_display, round_nearest=True)
        if frame == video.current_frame:
            return
        video.seek_to_frame(frame)

    def _reset_label_clicks(self) -> None:
        """Forget a half-placed state label and take its preview off the plots."""
        self.first_click = None
        self.second_click = None
        if self.plot_container is not None:
            self.plot_container.clear_pending_label()

    def _seek_to_frame(self, time_s: float):
        """Seek to display-clock *time_s*, keeping the red marker on the EXACT
        (sub-frame) time.

        The video can only display discrete frames, so it shows the nearest one,
        but the playhead stays on the true clicked/label time. This lets audio
        syllable boundaries be placed precisely even between video frames. See
        docs/source/advanced/playback.md.
        """
        if hasattr(self.app_state, "video") and self.app_state.video:
            # VideoSync.time_to_frame speaks the display clock too.
            video_frame = self.app_state.video.time_to_frame(time_s, round_nearest=True)
            # seek_to_frame fires frame_changed (syncs pose/extra cameras and
            # snaps the marker to the frame); we then override the marker so it
            # sits on the exact time, not the frame grid.
            self.app_state.video.seek_to_frame(video_frame)
        if self.plot_container:
            self.plot_container.update_time_marker_by_time(time_s)

    def _delete_label(self):
        if self.current_labels_pos is None:
            return
        if self.current_labels_is_prediction:
            notify("Predictions are read-only — cannot delete.", severity="warning")
            return

        df = self.app_state.label_intervals
        if df is None or df.empty:
            return

        if self.current_labels_pos not in df.index:
            self.current_labels_pos = None
            return

        _, _, labels = get_interval_bounds(df, self.current_labels_pos)
        if self._refuse_foreign_branch(int(labels)):
            return

        self.app_state.record_label_edit("delete label")
        df = delete_interval(df, self.current_labels_pos)
        self.app_state.label_intervals = df
        self.app_state.set_trial_intervals(self.app_state.trials_sel, df)

        self.current_labels_pos = None
        self.current_labels = None
        self.current_labels_is_prediction = False

        self._mark_changes_unsaved()
        self.curation_panel.note_labels_edited()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
        self.refresh_labels_shapes_layer()

    def undo_last_label_edit(self):
        """``Ctrl+Z``: take back the last label placed, moved or deleted.

        A half-placed state label is the one thing undone in place: its first
        click has changed nothing yet, so cancelling it is what the key means
        there — popping a finished edit instead would take back the wrong one.
        """
        if self.first_click is not None and self.second_click is None:
            self._reset_label_clicks()
            notify("Cancelled the label being placed")
            return

        edit = self.app_state.undo_label_edit()
        if edit is None:
            notify("Nothing to undo", severity="warning")
            return

        self.current_labels_pos = None
        self.current_labels = None
        self.current_labels_is_prediction = False
        self.old_labels_pos = None
        self.old_labels = None
        self._reset_label_clicks()
        self.ready_for_label_click = False

        trial = self.app_state.trials_sel
        # An undo can land on a trial the user has navigated away from; going
        # there is what makes it visible, and the trial change redraws for us.
        if trial is not None and str(edit.trial) != str(trial):
            self._switch_trial_for_click(edit.trial)
        elif self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)

        self._mark_changes_unsaved()
        self.curation_panel.note_labels_edited()
        self.refresh_labels_shapes_layer()
        notify(f"Undo: {edit.description}")

    def _edit_label(self):
        """Enter edit mode for adjusting interval boundaries."""
        if self.current_labels_pos is None:
            logger.warning("No label selected. Click on a label first to select it.")
            return
        if self.current_labels_is_prediction:
            notify("Predictions are read-only — cannot edit.", severity="warning")
            return

        if self._refuse_foreign_branch(self.current_labels):
            return

        self.old_labels_pos = self.current_labels_pos
        self.old_labels = self.current_labels
        self.selected_labels = self.current_labels

        # At the current frame, the label's own key re-places it (start,
        # then end) — a plot click must not.
        self.ready_for_label_click = not self.frame_labelling
        self._reset_label_clicks()
        if self.frame_labelling:
            key = self.labels_TO_KEY.get(self.selected_labels, "its key")
            notify(f"Navigate to the new start and press {key}, then to the new end and press {key} again")

    def _play_segment(self):
        if self.current_labels_pos is None:
            logger.warning("No label selected for playback")
            return

        if self.current_labels_is_prediction:
            df = self.app_state.pred_labels_df
            if df is not None and "trial" in df.columns:
                df = df[df["trial"] == self.app_state.trials_sel]
        else:
            df = self.app_state.label_intervals
        if df is None or self.current_labels_pos not in df.index:
            return

        onset_s, offset_s, _ = get_interval_bounds(df, self.current_labels_pos)

        # Point events have offset_s = NaN; there's nothing to play back.
        if not np.isfinite(offset_s) or offset_s <= onset_s:
            notify(
                "Playback only works for state events, not point events.",
                severity="warning",
            )
            return

        if self.app_state.video:
            # Round to the nearest frame so the marker lands on the label
            # boundary instead of truncating up to a frame short. time_to_frame
            # speaks the display clock; the stored bounds are trial-relative.
            onset_d, offset_d = self._to_display(onset_s), self._to_display(offset_s)
            start_frame = self.app_state.video.time_to_frame(onset_d, round_nearest=True)
            end_frame = self.app_state.video.time_to_frame(offset_d, round_nearest=True)
            # Video shows nearest frames; audio and the marker use the exact
            # label bounds, so the tail isn't clipped to the frame grid and the
            # playhead stops on the offset itself.
            self.app_state.video.play_segment(start_frame, end_frame, exact_t0=onset_d, exact_t1=offset_d)
        else:
            self._play_audio_segment(onset_s, offset_s)

    def _play_audio_segment(self, onset_s: float, offset_s: float):
        # The audio player runs on the display clock (it feeds the marker).
        if self.plot_container and hasattr(self.plot_container, "audio_player"):
            self.plot_container.audio_player.play_segment(self._to_display(onset_s), self._to_display(offset_s))

    def _get_canvas_widget(self):
        """Return the stable primary CameraView (host for the label overlay).

        The wgpu render canvas is recreated on every video load, which would
        destroy an overlay parented to it — the CameraView itself persists."""
        video_area = getattr(self.shell, "video_area", None)
        if video_area is not None:
            return video_area.primary
        return self.shell.canvas_widget()

    def _add_labels_shapes_layer(self):
        """Add a Qt QLabel overlay on the video canvas.

        setText() is a trivial Qt repaint — the overlay follows the current
        video frame via the VideoSync ``frame_changed`` signal.
        """
        if getattr(self, "_label_overlay", None) is not None:
            return

        canvas_widget = self._get_canvas_widget()
        if canvas_widget is None:
            return

        overlay = QLabel(canvas_widget)
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        overlay.setAlignment(Qt.AlignCenter)
        overlay.setStyleSheet(self._overlay_stylesheet("white"))
        overlay.setFixedHeight(36)
        overlay.hide()
        self._label_overlay = overlay
        self._label_overlay_last_text = ""

        def _reposition_overlay():
            if self._label_overlay is None:
                return
            parent = self._label_overlay.parent()
            if parent is None:
                return
            pw = parent.width()
            ph = parent.height()
            ow = self._label_overlay.sizeHint().width()
            oh = self._label_overlay.sizeHint().height()
            self._label_overlay.move(pw - ow - 8, ph - oh - 8)

        self._reposition_overlay = _reposition_overlay

        # Track canvas resizes to reposition overlay
        orig_resize = canvas_widget.resizeEvent

        def _on_canvas_resize(event):
            orig_resize(event)
            _reposition_overlay()

        canvas_widget.resizeEvent = _on_canvas_resize

        def _update_labels_text(video_frame=None, time_s=None):
            overlay = getattr(self, "_label_overlay", None)
            if overlay is None:
                return
            if self.app_state.hide_label_text:
                overlay.hide()
                return
            if time_s is None:
                video = getattr(self.app_state, "video", None)
                if video_frame is None:
                    video_frame = int(getattr(self.app_state, "current_frame", 0) or 0)
                if video:
                    # frame_to_time is display-clock; the interval lookup
                    # below runs on the current trial's trial-relative rows.
                    resolved = self.app_state.from_display(video.frame_to_time(video_frame))
                    if resolved is None:
                        return
                    time_s = resolved[1]
                elif hasattr(self.app_state, "video_fps") and self.app_state.video_fps:
                    time_s = video_frame / self.app_state.video_fps
                else:
                    return
            df = self.app_state.label_intervals
            ind = self._current_individual()
            mappings = self._mappings

            text = ""
            css_color = "white"
            active_ids = self.app_state.active_label_ids
            if df is not None and not df.empty:
                idx = find_interval_at(df, time_s, ind, label_ids=active_ids)
                if idx is not None:
                    _, _, labels = get_interval_bounds(df, idx)
                    if labels in mappings and labels != 0:
                        text = mappings[labels]["name"]
                        color = mappings[labels]["color"]
                        if hasattr(color, "tolist"):
                            color = color.tolist()
                        r, g, b = (int(c * 255) for c in color[:3])
                        css_color = f"rgb({r},{g},{b})"

            if text == self._label_overlay_last_text:
                return
            self._label_overlay_last_text = text

            if not text:
                overlay.hide()
                return
            overlay.setStyleSheet(self._overlay_stylesheet(css_color))
            overlay.setText(text)
            overlay.adjustSize()
            overlay.setFixedHeight(36)
            _reposition_overlay()
            overlay.show()
            overlay.raise_()

        self.app_state.current_frame_changed.connect(_update_labels_text)
        self.app_state.hide_label_text_changed.connect(self._on_hide_label_text_changed)
        self._update_labels_text = _update_labels_text
        _update_labels_text()

    def _on_hide_label_text_changed(self, *_):
        """React to the video context's "Hide label" checkbox."""
        overlay = getattr(self, "_label_overlay", None)
        if overlay is None:
            return
        if self.app_state.hide_label_text:
            overlay.hide()
        else:
            self._label_overlay_last_text = ""
            if hasattr(self, "_update_labels_text"):
                self._update_labels_text()

    @staticmethod
    def _overlay_stylesheet(color: str) -> str:
        return (
            "QLabel {"
            "  background: rgba(0, 0, 0, 160);"
            f"  color: {color};"
            "  font-size: 22px;"
            "  font-weight: bold;"
            "  padding: 4px 16px;"
            "  border-radius: 6px;"
            "}"
        )

    def refresh_labels_shapes_layer(self):
        """Refresh: ensure overlay exists, then force an update."""
        # Every label mutation ends here, so it is also where an open label
        # table hears that the frame under it was replaced (its own edits leave
        # it current, so this costs nothing on that path).
        if self._label_table_dialog is not None and self._label_table_dialog.isVisible():
            self._label_table_dialog.refresh_if_stale()
        if getattr(self, "_label_overlay", None) is None:
            self._add_labels_shapes_layer()
            return
        self._label_overlay_last_text = ""
        if hasattr(self, "_update_labels_text"):
            self._update_labels_text()

    def _on_marker_time_for_overlay(self, time_s: float):
        """Update the current-label overlay from marker time when no video runs."""
        if getattr(self.app_state, "video", None) is not None:
            return
        updater = getattr(self, "_update_labels_text", None)
        if updater is not None:
            # The marker is on the display clock; the overlay matches against
            # the current trial's (trial-relative) intervals.
            resolved = self.app_state.from_display(time_s)
            updater(time_s=resolved[1] if resolved is not None else time_s)


class LabelsPerPlotDialog(QDialog):
    """Dialog controlling how label rectangles render on each plot type."""

    _MODE_OPTIONS = [
        ("Full", LABEL_OVERLAY_MODE_FULL),
        ("Bottom", LABEL_OVERLAY_MODE_BOTTOM),
        ("None", LABEL_OVERLAY_MODE_NONE),
    ]

    def __init__(self, current_modes: dict[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Show label overlays per plot type")
        self._combos: dict[str, QComboBox] = {}

        layout = QVBoxLayout(self)
        info = QLabel("Applies to every plot instance of the type.")
        info.setStyleSheet("QLabel { color: #aaa; }")
        layout.addWidget(info)

        form = QFormLayout()
        for type_key, display_name in LABEL_OVERLAY_PLOT_TYPES.items():
            combo = QComboBox()
            for option_label, mode in self._MODE_OPTIONS:
                combo.addItem(option_label, mode)
            idx = combo.findData(current_modes.get(type_key, DEFAULT_LABEL_OVERLAY_MODES[type_key]))
            combo.setCurrentIndex(max(idx, 0))
            form.addRow(f"{display_name}:", combo)
            self._combos[type_key] = combo
        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_modes(self) -> dict[str, str]:
        return {type_key: combo.currentData() for type_key, combo in self._combos.items()}


class TemporaryLabelsDialog(QDialog):
    """Dialog for creating temporary labels for the current session."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Temporary Labels")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel("Enter label names (one per line):")
        layout.addWidget(info_label)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            "label1\nlabel2\nlabel3\n...\n\n(background is added automatically as label 0)"
        )
        layout.addWidget(self.text_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_labels(self):
        """Parse and return the list of label names."""
        text = self.text_edit.toPlainText()
        labels = [line.strip().replace(" ", "_") for line in text.split("\n") if line.strip()]
        return labels
