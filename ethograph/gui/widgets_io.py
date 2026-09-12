"""Widget for input/output controls and data loading."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QAction
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.project import project_dir_of
from ethograph.io.catalog import INDIVIDUAL_DIMS
from ethograph.io.metadata_table import metadata_tsv_path
from ethograph.io.validation import EPHYS_FILE_FILTER
from ethograph.labels.tsv_store import labels_tsv_path, load_labels_tsv
from ethograph.utils.paths import (
    default_config_dir,
    find_mapping_file,
)
from ethograph.utils.qt import populate_if_exists

from .app_state import AppStateSpec
from .dialog_select_template import TemplateDialog
from .file_dialogs import browse_open_dir, browse_open_file, browse_save_file
from .notify import notify, notify_dialog
from .top_bar import SectionPopup
from .wizard_overview import NCWizardDialog

logger = logging.getLogger(__name__)


class IOWidget(QWidget):
    """Widget to control I/O paths, device selection, and data loading."""

    def __init__(self, app_state, data_widget, labels_widget, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.data_widget = data_widget
        self.labels_widget = labels_widget

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self.combos = {}
        self.controls = []

        self._create_load_panel(main_layout)
        self._create_controls_panel(main_layout)
        self._create_export_panel(main_layout)
        self._wire_app_state_path_signals()
        self._wire_path_edit_signals()

        # Restore UI text fields from app state
        if self.app_state.nc_file_path:
            self.nc_file_path_edit.setText(self.app_state.nc_file_path)
        if self.app_state.video_folder:
            self.video_folder_edit.setText(self.app_state.video_folder)
        if self.app_state.audio_folder:
            self.audio_folder_edit.setText(self.app_state.audio_folder)
        if self.app_state.pose_folder:
            self.pose_folder_edit.setText(self.app_state.pose_folder)
        if self.app_state.ephys_path:
            self.ephys_path_edit.setText(self.app_state.ephys_path)
        if self.app_state.neurons_path:
            self.neurons_path_edit.setText(self.app_state.neurons_path)

        # Auto-discover metadata file on startup
        self._auto_discover_metadata()

    def _wire_app_state_path_signals(self):
        self.app_state.nc_file_path_changed.connect(lambda value: self.nc_file_path_edit.setText(value or ""))
        self.app_state.video_folder_changed.connect(lambda value: self.video_folder_edit.setText(value or ""))
        self.app_state.audio_folder_changed.connect(lambda value: self.audio_folder_edit.setText(value or ""))
        self.app_state.pose_folder_changed.connect(lambda value: self.pose_folder_edit.setText(value or ""))
        self.app_state.ephys_path_changed.connect(lambda value: self.ephys_path_edit.setText(value or ""))
        self.app_state.neurons_path_changed.connect(lambda value: self.neurons_path_edit.setText(value or ""))
        self.app_state.nwb_file_path_changed.connect(lambda value: self.nwb_file_path_edit.setText(value or ""))
        self.app_state.metadata_path_changed.connect(lambda value: self.metadata_path_edit.setText(value or ""))
        self.app_state.nc_file_path_changed.connect(lambda _: self._auto_discover_metadata())

    def _wire_path_edit_signals(self):
        self.nc_file_path_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.nc_file_path_edit, "nc_file_path")
        )
        self.nwb_file_path_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.nwb_file_path_edit, "nwb_file_path")
        )
        self.video_folder_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.video_folder_edit, "video_folder")
        )
        self.audio_folder_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.audio_folder_edit, "audio_folder")
        )
        self.pose_folder_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.pose_folder_edit, "pose_folder")
        )
        self.ephys_path_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.ephys_path_edit, "ephys_path")
        )
        self.neurons_path_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.neurons_path_edit, "neurons_path")
        )

    def _sync_line_edit_to_state(self, line_edit, attr_name):
        value = line_edit.text().strip() or None
        if getattr(self.app_state, attr_name, None) != value:
            setattr(self.app_state, attr_name, value)

    def restore_subpanel(self, widget):
        """Return a sub-panel borrowed by a top-bar popup to its home slot.

        The File menu pops up ``labels_group`` / ``pred_group`` /
        ``export_panel`` individually; this re-inserts the borrowed widget
        when the popup closes.
        """
        if widget is self.labels_group:
            self._controls_layout.insertRow(0, self.labels_group)
        elif widget is self.pred_group:
            self._controls_layout.insertRow(1, self.pred_group)
        elif widget is self.export_panel:
            # Main layout order: load, controls, export.
            self.layout().insertWidget(2, self.export_panel)
            self.export_panel.setVisible(False)
        else:
            raise ValueError(f"not an IO sub-panel: {widget!r}")

    # ------------------------------------------------------------------
    # Load panel
    # ------------------------------------------------------------------

    def _create_load_panel(self, main_layout):
        self.load_panel = QWidget()
        self._load_layout = QFormLayout()
        self._load_layout.setSpacing(2)
        self._load_layout.setContentsMargins(0, 0, 0, 0)
        self.load_panel.setLayout(self._load_layout)

        # Button row
        self.reset_button = QPushButton("Reset gui_settings.yaml")
        self.reset_button.setObjectName("reset_button")
        self.reset_button.clicked.connect(self._on_reset_gui_clicked)

        self.create_nc_button = QPushButton("🧙Data wizard")
        self.create_nc_button.setObjectName("create_nc_button")
        self.create_nc_button.clicked.connect(self._on_create_nc_clicked)

        self.template_button = QPushButton("💡Select templates")
        self.template_button.setObjectName("template_button")
        self.template_button.clicked.connect(self._on_select_template_clicked)

        # Wrapped in a QWidget so the cover page can hide this row while it
        # hosts the load panel (the cover page has its own wizard/template buttons).
        self.load_buttons_row = QWidget()
        button_row = QHBoxLayout(self.load_buttons_row)
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.create_nc_button)
        button_row.addWidget(self.template_button)
        self._load_layout.addRow(self.load_buttons_row)

        # Path widgets
        self.nc_file_path_edit = self._create_path_widget(
            self._load_layout,
            label="Get session (required):",
            object_name="nc_file_path",
            browse_callback=lambda: self.on_browse_clicked("file", "data"),
        )

        # Alignment row: NWB path + ephys offset in a single row.
        alignment_row = QHBoxLayout()
        self.nwb_file_path_edit = QLineEdit()
        self.nwb_file_path_edit.setObjectName("nwb_file_path_edit")
        alignment_row.addWidget(self.nwb_file_path_edit)

        nwb_browse_button = QPushButton("Browse")
        nwb_browse_button.setObjectName("nwb_file_path_browse_button")
        nwb_browse_button.clicked.connect(self._browse_nwb_file)
        alignment_row.addWidget(nwb_browse_button)

        nwb_clear_button = QPushButton("Clear")
        nwb_clear_button.setObjectName("nwb_file_path_clear_button")
        nwb_clear_button.clicked.connect(lambda: self._on_clear_path_clicked("nwb_file_path", self.nwb_file_path_edit))
        alignment_row.addWidget(nwb_clear_button)

        self._load_layout.addRow("Alignment:", alignment_row)

        meta_template_btn = QPushButton("Template")
        meta_template_btn.setObjectName("metadata_template_button")
        meta_template_btn.setToolTip(
            "Save a template metadata file (TSV) with example columns.\n"
            "Open it in Excel, fill in your conditions, and save."
        )
        meta_template_btn.clicked.connect(self._save_metadata_template)

        self.metadata_path_edit = self._create_path_widget(
            self._load_layout,
            label="Metadata:",
            object_name="metadata_path",
            browse_callback=lambda: self._browse_metadata_file(),
            extra_buttons=[meta_template_btn],
        )
        self.video_folder_edit = self._create_path_widget(
            self._load_layout,
            label="Video folder:",
            object_name="video_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "video"),
        )
        self.pose_folder_edit = self._create_path_widget(
            self._load_layout,
            label="Pose folder:",
            object_name="pose_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "pose"),
        )

        self.audio_folder_edit = self._create_path_widget(
            self._load_layout,
            label="Audio folder:",
            object_name="audio_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "audio"),
        )
        self.ephys_path_edit = self._create_path_widget(
            self._load_layout,
            label="Ephys file:",
            object_name="ephys_path",
            browse_callback=lambda: self.on_browse_clicked("file", "ephys"),
        )

        self.neurons_path_edit = self._create_path_widget(
            self._load_layout,
            label="Units (Kilosort/Pynapple):",
            object_name="neurons_path",
            browse_callback=self._browse_neurons,
        )

        # Downsample + Load button
        self._create_load_button(self._load_layout)

        main_layout.addWidget(self.load_panel)

    # ------------------------------------------------------------------
    # Controls panel
    # ------------------------------------------------------------------

    def _create_controls_panel(self, main_layout):
        self.controls_panel = QWidget()
        self._controls_layout = QFormLayout()
        self._controls_layout.setSpacing(2)
        self._controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_panel.setLayout(self._controls_layout)

        self.labels_group = QGroupBox("Labels")
        self._labels_group_layout = QFormLayout()
        self._labels_group_layout.setSpacing(2)
        self._labels_group_layout.setContentsMargins(4, 4, 4, 4)
        self.labels_group.setLayout(self._labels_group_layout)

        self._create_mapping_row(self._labels_group_layout)
        # Labels row inserted here dynamically by create_device_controls()
        self._labels_row_index = self._labels_group_layout.rowCount()

        self._controls_layout.addRow(self.labels_group)
        self._create_predictions_row(self._controls_layout)

        main_layout.addWidget(self.controls_panel)

    # ------------------------------------------------------------------
    # Export panel
    # ------------------------------------------------------------------

    def _create_export_panel(self, main_layout):
        self.export_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.export_panel.setLayout(layout)

        # Offset correction and short-label purging moved to Tools ▸ Label
        # bulk editing… (gui/dialog_bulk_labels.py, CurationPanel.correct_offsets
        # / .purge_trial_labels) — scoped by the same trial-scope vocabulary
        # every other bulk action uses, not a single_trial/all_trials pair
        # local to this panel.

        # Save button
        self.save_labels_button = QPushButton("Save labels (Ctrl+S)")
        self.save_labels_button.setToolTip("Save labels TSV + local backup + optional remote backup")
        self.save_labels_button.clicked.connect(self._save_labels)
        layout.addWidget(self.save_labels_button)

        # Local backup (read-only display)
        local_backup_row = QHBoxLayout()
        local_backup_label = QLabel("Local backup:")
        local_backup_label.setFixedWidth(90)
        local_backup_row.addWidget(local_backup_label)
        self.local_backup_edit = QLineEdit()
        self.local_backup_edit.setReadOnly(True)
        self.local_backup_edit.setPlaceholderText("labels/backups/")
        if self.app_state.nc_file_path:
            backup_dir = str(Path(self.app_state.nc_file_path).parent / "labels" / "backups")
            self.local_backup_edit.setText(backup_dir)
        local_backup_row.addWidget(self.local_backup_edit)
        layout.addLayout(local_backup_row)

        # Remote backup (optional)
        remote_group = QGroupBox("Remote backup")
        remote_group_layout = QVBoxLayout()
        remote_group_layout.setSpacing(4)
        remote_group.setLayout(remote_group_layout)

        self.remote_backup_enabled_cb = QCheckBox("Enable remote backup")
        self.remote_backup_enabled_cb.setChecked(self.app_state.remote_backup_enabled)
        remote_group_layout.addWidget(self.remote_backup_enabled_cb)

        remote_path_row = QHBoxLayout()
        self.remote_backup_edit = QLineEdit()
        self.remote_backup_edit.setPlaceholderText("cloud/git folder...")
        self.remote_backup_edit.setToolTip(
            "Path to a remote folder for label backups.\n"
            "Useful for syncing labels via cloud storage or a git repository."
        )
        if self.app_state.remote_backup_path:
            self.remote_backup_edit.setText(self.app_state.remote_backup_path)
        self.remote_backup_edit.editingFinished.connect(
            lambda: setattr(
                self.app_state,
                "remote_backup_path",
                self.remote_backup_edit.text().strip() or None,
            )
        )
        remote_path_row.addWidget(self.remote_backup_edit)
        remote_browse_btn = QPushButton("Browse folder")
        remote_browse_btn.clicked.connect(self._browse_remote_backup)
        remote_path_row.addWidget(remote_browse_btn)
        remote_group_layout.addLayout(remote_path_row)

        remote_options_row = QHBoxLayout()
        self.remote_save_mode_combo = QComboBox()
        self.remote_save_mode_combo.addItem("Save with timestamp")
        self.remote_save_mode_combo.addItem("Overwrite file")
        self.remote_save_mode_combo.addItem("Overwrite + git commit")
        self.remote_save_mode_combo.setToolTip(
            "Timestamp: each save creates a new file (safe, auditable).\n"
            "Overwrite: saves a single file, no version control.\n"
            "Overwrite + git commit: saves a single file and auto-commits.\n"
            "  Requires the remote folder to be inside a git repo."
        )
        mode_map = {"timestamp": 0, "overwrite": 1, "git": 2}
        self.remote_save_mode_combo.setCurrentIndex(mode_map.get(self.app_state.remote_backup_mode, 0))

        def _on_mode_changed(text):
            if text == "Overwrite + git commit":
                self.app_state.remote_backup_mode = "git"
            elif text == "Overwrite file":
                self.app_state.remote_backup_mode = "overwrite"
            else:
                self.app_state.remote_backup_mode = "timestamp"

        self.remote_save_mode_combo.currentTextChanged.connect(_on_mode_changed)
        remote_options_row.addWidget(self.remote_save_mode_combo)

        self.remote_depth_combo = QComboBox()
        self.remote_depth_combo.setToolTip(
            "Controls the subfolder structure inside the remote backup root.\n"
            "Flat: all files land directly in the remote root.\n"
            "Higher levels mirror parent directories to avoid filename collisions."
        )
        self._populate_remote_depth_combo()
        self.remote_depth_combo.currentIndexChanged.connect(
            lambda idx: setattr(self.app_state, "remote_path_depth", idx)
        )
        remote_options_row.addWidget(self.remote_depth_combo)
        remote_group_layout.addLayout(remote_options_row)

        self._remote_backup_controls = [
            self.remote_backup_edit,
            remote_browse_btn,
            self.remote_save_mode_combo,
            self.remote_depth_combo,
        ]

        def _on_remote_enabled(checked: bool):
            self.app_state.remote_backup_enabled = checked
            for w in self._remote_backup_controls:
                w.setEnabled(checked)

        self.remote_backup_enabled_cb.toggled.connect(_on_remote_enabled)
        _on_remote_enabled(self.app_state.remote_backup_enabled)

        layout.addWidget(remote_group)

        self.app_state.nc_file_path_changed.connect(self._populate_remote_depth_combo)

        main_layout.addWidget(self.export_panel)

    # ------------------------------------------------------------------
    # Export panel handlers
    # ------------------------------------------------------------------

    def _save_labels(self):
        remote_path = self.remote_backup_edit.text().strip() or None
        remote_mode = self.app_state.remote_backup_mode
        try:
            self.app_state.save_labels(remote_path=remote_path, remote_mode=remote_mode)
        except Exception as e:
            notify_dialog(str(e), "error", "Save Error", self)

    def _populate_remote_depth_combo(self):
        if not hasattr(self, "remote_depth_combo"):
            return
        nc_path = self.app_state.nc_file_path
        combo = self.remote_depth_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Flat (no subfolders)")
        if nc_path:
            parts = Path(nc_path).parent.parts[1:]
            for i, _ in enumerate(parts):
                subfolder = "/".join(parts[len(parts) - i - 1 :])
                combo.addItem(subfolder)
        saved_depth = self.app_state.remote_path_depth
        combo.setCurrentIndex(min(saved_depth, combo.count() - 1))
        combo.blockSignals(False)

    def _browse_remote_backup(self):
        folder = browse_open_dir(
            self,
            self.app_state,
            "Select remote backup folder",
            preferred_dir=self.app_state.remote_backup_path,
        )
        if folder:
            self.remote_backup_edit.setText(folder)
            self.app_state.remote_backup_path = folder

    def _create_mapping_row(self, target_layout):
        mapping_row = QWidget()
        mapping_layout = QHBoxLayout()
        mapping_layout.setContentsMargins(0, 0, 0, 0)
        mapping_row.setLayout(mapping_layout)

        self.mapping_file_path_edit = QLineEdit()
        default_mapping = find_mapping_file(project_dir=project_dir_of(self.app_state))
        self.mapping_file_path_edit.setText(str(default_mapping) if default_mapping else "")
        self.mapping_file_path_edit.setToolTip("Path to mapping.txt file")
        mapping_layout.addWidget(self.mapping_file_path_edit)

        self.browse_mapping_btn = QPushButton("Browse")
        mapping_layout.addWidget(self.browse_mapping_btn)

        target_layout.addRow("Name mapping:", mapping_row)

        self.temp_labels_button = QPushButton("Create temporary labels")
        self.temp_labels_button.setToolTip("Create custom labels for this session only")
        target_layout.addRow("", self.temp_labels_button)

    def _create_predictions_row(self, target_layout):
        self.pred_group = QGroupBox("Predictions")
        pred_group_layout = QVBoxLayout()
        pred_group_layout.setContentsMargins(4, 4, 4, 4)
        pred_group_layout.setSpacing(2)
        self.pred_group.setLayout(pred_group_layout)

        # Row 0: load-as — overlay (the default, non-destructive) or
        # straight into the working labels, plus (import-as-labels only, and
        # only once a labels table already exists to merge onto) whether to
        # keep the existing labels and only add what they're missing.
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(QLabel("Load as:"))
        self.pred_load_mode_combo = QComboBox()
        self.pred_load_mode_combo.addItem("Overlay (compare with ground truth)", "overlay")
        self.pred_load_mode_combo.addItem("Import as labels", "labels")
        self.pred_load_mode_combo.setToolTip(
            "Overlay: draws the prediction set beside the ground truth for comparison — "
            "the working labels are untouched.\n"
            "Import as labels: writes the prediction set into the working labels themselves."
        )
        saved_mode = self.app_state.pred_load_mode
        idx = self.pred_load_mode_combo.findData(saved_mode)
        if idx >= 0:
            self.pred_load_mode_combo.setCurrentIndex(idx)
        self.pred_load_mode_combo.currentIndexChanged.connect(self._on_pred_load_mode_changed)
        mode_row.addWidget(self.pred_load_mode_combo, stretch=1)
        pred_group_layout.addLayout(mode_row)

        self.pred_merge_checkbox = QCheckBox("Merge with existing labels")
        self.pred_merge_checkbox.setToolTip(
            "Ticked: keep the current labels and only add predictions for (trial, class,\n"
            "individual) combinations they don't already cover.\n"
            "Unticked: the imported predictions replace the working labels outright."
        )
        self.pred_merge_checkbox.setChecked(bool(self.app_state.merge_imported_predictions))
        self.pred_merge_checkbox.toggled.connect(self._on_pred_merge_toggled)
        pred_group_layout.addWidget(self.pred_merge_checkbox)
        self.pred_merge_checkbox.setVisible(False)

        # What "Import as labels" is actually about to do — said here, up
        # front, instead of a confirmation popup at click time.
        self.pred_labels_warning_label = QLabel("")
        self.pred_labels_warning_label.setWordWrap(True)
        self.pred_labels_warning_label.setStyleSheet("color: #ff5555; font-size: 10px;")
        pred_group_layout.addWidget(self.pred_labels_warning_label)
        self.pred_labels_warning_label.setVisible(False)

        # Row 1: path + import (two ways in: a run's folder, or a plain .tsv)
        folder_row = QHBoxLayout()
        folder_row.setContentsMargins(0, 0, 0, 0)
        self.pred_file_path_edit = QLineEdit()
        self.pred_file_path_edit.setReadOnly(True)
        self.pred_file_path_edit.setPlaceholderText("No predictions loaded")
        folder_row.addWidget(self.pred_file_path_edit)
        self.import_predictions_btn = QPushButton("Import…")
        self.import_predictions_btn.setToolTip("Import a prediction set — from a run's folder, or a plain .tsv")
        self.import_predictions_menu = QMenu(self.import_predictions_btn)
        self.import_predictions_from_folder_action = QAction(
            "From folder (segmentation run)…", self.import_predictions_menu
        )
        self.import_predictions_from_folder_action.setToolTip(
            "Select a segmentation run's prediction folder (labels/predictions_{run}_{timestamp}/)"
        )
        self.import_predictions_menu.addAction(self.import_predictions_from_folder_action)
        self.import_predictions_from_tsv_action = QAction("From .tsv file…", self.import_predictions_menu)
        self.import_predictions_from_tsv_action.setToolTip(
            "Load a plain labels TSV as predictions — e.g. a second annotator's labels, for comparison"
        )
        self.import_predictions_menu.addAction(self.import_predictions_from_tsv_action)
        self.import_predictions_btn.setMenu(self.import_predictions_menu)
        folder_row.addWidget(self.import_predictions_btn)
        pred_group_layout.addLayout(folder_row)

        # Row 2: show checkbox + threshold + PDF button
        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        # Exposed so MetaWidget can move the "Confidence" overlay checkbox here.
        self._pred_controls_row = controls_row

        controls_row.addWidget(QLabel("Frame thr:"))
        self.pred_confidence_threshold_spin = QDoubleSpinBox()
        self.pred_confidence_threshold_spin.setRange(0.0, 1.0)
        self.pred_confidence_threshold_spin.setSingleStep(0.05)
        self.pred_confidence_threshold_spin.setDecimals(2)
        self.pred_confidence_threshold_spin.setValue(0.75)
        self.pred_confidence_threshold_spin.setToolTip(
            "Frame-level confidence threshold — frames below this are marked red."
        )
        controls_row.addWidget(self.pred_confidence_threshold_spin)

        controls_row.addWidget(QLabel("Segment thr (state labels only):"))
        self.pred_segment_confidence_threshold_spin = QDoubleSpinBox()
        self.pred_segment_confidence_threshold_spin.setRange(0.0, 1.0)
        self.pred_segment_confidence_threshold_spin.setSingleStep(0.05)
        self.pred_segment_confidence_threshold_spin.setDecimals(2)
        self.pred_segment_confidence_threshold_spin.setValue(0.6)
        self.pred_segment_confidence_threshold_spin.setToolTip(
            "Segment-level mean confidence threshold — segments below this are highlighted red.\n"
            "Meaningful for state labels only: a point event has no span to average over, so this "
            "threshold has no effect on it."
        )
        controls_row.addWidget(self.pred_segment_confidence_threshold_spin)

        self.pred_confidence_pdf_btn = QPushButton("Update confidence (+ PDF)")
        self.pred_confidence_pdf_btn.setToolTip(
            "Regenerate confidence PDF with current thresholds and update low/high confidence classification.\n"
            "The segment threshold only affects state labels — point events have no span to average over."
        )
        self.pred_confidence_pdf_btn.setEnabled(False)
        controls_row.addWidget(self.pred_confidence_pdf_btn)

        pred_group_layout.addLayout(controls_row)
        target_layout.addRow(self.pred_group)
        # The saved mode may already be "labels" — reflect that immediately
        # rather than waiting for the combo's first change.
        self._update_pred_merge_visibility()

    def pred_load_mode(self) -> str:
        """``"overlay"`` or ``"labels"`` — the Predictions panel's Load-as combo."""
        return self.pred_load_mode_combo.currentData()

    def _on_pred_load_mode_changed(self, *_args) -> None:
        """Remembered globally (gui_settings.yaml) — a viewing/import habit,
        not something tied to one dataset."""
        self.app_state.pred_load_mode = self.pred_load_mode()
        self._update_pred_merge_visibility()

    def _update_pred_merge_visibility(self, *_args) -> None:
        """The merge checkbox only matters for "Import as labels" onto a
        session that already has some — a fresh session has nothing to merge
        onto, so "replace" and "import" are the same thing.

        What clicking Import is about to do is said here, live, instead of a
        confirmation popup at click time — so it stays visible while the
        combo and checkbox are still being decided, not sprung afterwards.
        """
        existing = getattr(self.app_state, "_all_labels_df", None)
        has_existing = existing is not None and not existing.empty
        show_merge = self.pred_load_mode() == "labels" and has_existing
        self.pred_merge_checkbox.setVisible(show_merge)

        if self.pred_load_mode() != "labels":
            self.pred_labels_warning_label.setVisible(False)
            return
        if show_merge and self.pred_merge_checkbox.isChecked():
            text = (
                "Adds predictions only for (trial, class, individual) combinations the "
                "current labels don't already cover — nothing existing is overridden."
            )
        else:
            text = "Replaces the working labels outright. Nothing reaches disk until you save."
        self.pred_labels_warning_label.setText(text)
        self.pred_labels_warning_label.setVisible(True)

    def _on_pred_merge_toggled(self, checked: bool) -> None:
        self.app_state.merge_imported_predictions = checked
        self._update_pred_merge_visibility()

    def _create_labels_row_at_index(self):
        """Create two labels rows: input (browse) and output (auto-generated TSV)."""
        # Row 1: format combo + input path + browse
        input_row = QWidget()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_row.setLayout(input_layout)

        self.labels_format_combo = QComboBox()
        self.labels_format_combo.addItem(".tsv")
        self.labels_format_combo.addItem(".nc (legacy)")
        self.labels_format_combo.addItem("pynapple (.npz)")
        self.labels_format_combo.addItem("pynapple (.nwb)")
        if self.app_state.audio_folder:
            from ethograph.labels.converters import CROWSETTA_SEQ_FORMATS

            for fmt in CROWSETTA_SEQ_FORMATS:
                self.labels_format_combo.addItem(fmt)
        self.labels_format_combo.setToolTip("Label file format to import")
        self.labels_format_combo.currentTextChanged.connect(self._on_labels_format_changed)
        input_layout.addWidget(self.labels_format_combo)

        self.label_file_path_edit = QLineEdit()
        if self.import_labels_checkbox.isChecked():
            path = self._resolve_import_labels_path()
            if path:
                self.label_file_path_edit.setText(path)
        input_layout.addWidget(self.label_file_path_edit)

        self.labels_browse_btn = QPushButton("Browse")
        self.labels_browse_btn.clicked.connect(self._on_labels_browse_clicked)
        input_layout.addWidget(self.labels_browse_btn)

        self._labels_input_label = QLabel("Labels path:")
        self._labels_group_layout.insertRow(self._labels_row_index, self._labels_input_label, input_row)

        # Row 2: output TSV path (read-only, no browse, greyed out for .tsv)
        self.labels_output_edit = QLineEdit()
        self.labels_output_edit.setReadOnly(True)
        self.labels_output_edit.setPlaceholderText("Converted .tsv output will appear here")
        self._labels_output_label = QLabel("Labels output:")
        self._labels_group_layout.insertRow(
            self._labels_row_index + 1,
            self._labels_output_label,
            self.labels_output_edit,
        )

        # Initial state: .tsv selected → output row disabled
        self._set_labels_output_enabled(False)

    def _set_labels_output_enabled(self, enabled: bool):
        self.labels_output_edit.setEnabled(enabled)
        self._labels_output_label.setEnabled(enabled)

    def _on_labels_format_changed(self, fmt: str):
        is_tsv = fmt == ".tsv"
        self._set_labels_output_enabled(not is_tsv)
        if is_tsv:
            self._labels_input_label.setText("Labels path:")
            self.labels_output_edit.clear()
        else:
            self._labels_input_label.setText("Labels input:")

    def _on_labels_browse_clicked(self):
        fmt = self.labels_format_combo.currentText()
        if fmt == ".tsv":
            self._import_tsv_labels()
            return
        if fmt == ".nc (legacy)":
            self.on_browse_clicked("file", "labels")
            return
        if fmt.startswith("pynapple"):
            self._import_pynapple_labels()
            return
        self._import_crowsetta_labels(fmt)

    def _import_tsv_labels(self):
        file_path = browse_open_file(
            self,
            self.app_state,
            "Open labels TSV file",
            "TSV files (*.tsv)",
            preferred_dir=self.app_state.nc_file_path,
        )
        if not file_path:
            return

        try:
            labels_df = load_labels_tsv(file_path)
        except (FileNotFoundError, ValueError) as e:
            notify(str(e), severity="error")
            return

        self.app_state._all_labels_df = labels_df
        self.app_state.clear_label_history()
        if self.data_widget:
            # With no individual dimension the selector's names come from the
            # labels, which have just been replaced.
            self.data_widget.refresh_individual_choices()
        self.app_state._labels_file_path = file_path  # Track which file is active
        # Remember this exact file for future loads of this dataset — an
        # explicit choice must never be re-guessed from the .nc filename.
        self.app_state.labels_import_path = file_path
        self.import_labels_checkbox.setChecked(True)
        self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)
        self.label_file_path_edit.setText(file_path)

        if hasattr(self, "changepoints_widget") and self.changepoints_widget:
            self.changepoints_widget._update_cp_status()
        if self.labels_widget:
            self.labels_widget._mark_changes_unsaved()
            self.labels_widget.refresh_labels_shapes_layer()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
            if self.data_widget.plot_container:
                self.data_widget.plot_container.labels_redraw_needed.emit()
        self._close_labels_popup()

    def _merge_tsv_labels(self):
        """Fuse a second labels TSV into the one already loaded in the GUI.

        Unlike :meth:`_import_tsv_labels`, this keeps whatever is currently in
        :attr:`app_state._all_labels_df` and appends the rows from the picked
        file — nothing is replaced or de-duplicated. If a label class (the
        ``labels`` column) shows up in both files, that is surfaced as a
        warning before anything is merged, so the user can back out.
        """
        file_path = browse_open_file(
            self,
            self.app_state,
            "Open labels TSV to merge",
            "TSV files (*.tsv)",
            preferred_dir=self.app_state.nc_file_path,
        )
        if not file_path:
            return

        try:
            incoming = load_labels_tsv(file_path)
        except (FileNotFoundError, ValueError) as e:
            notify(str(e), severity="error")
            return
        existing = self.app_state._all_labels_df

        if existing is not None and not existing.empty and not incoming.empty:
            shared_ids = set(existing["labels"].unique()) & set(incoming["labels"].unique())
            if shared_ids:
                names = self.labels_widget._mappings if self.labels_widget else {}
                shown = ", ".join(str(names.get(lid, {}).get("name", lid)) for lid in sorted(shared_ids))
                answer = QMessageBox.question(
                    self,
                    "Merge labels",
                    f"Both files contain label class(es): {shown}.\n"
                    "Merging keeps every row from both files as-is (no de-duplication).\n\n"
                    "Merge anyway?",
                )
                if answer != QMessageBox.Yes:
                    return

        if existing is None or existing.empty:
            merged = incoming
        else:
            merged = pd.concat([existing, incoming], ignore_index=True)

        self.app_state._all_labels_df = merged
        self.app_state.clear_label_history()
        if self.data_widget:
            # With no individual dimension the selector's names come from the
            # labels, which have just gained rows.
            self.data_widget.refresh_individual_choices()
        self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)

        if hasattr(self, "changepoints_widget") and self.changepoints_widget:
            self.changepoints_widget._update_cp_status()
        if self.labels_widget:
            self.labels_widget._mark_changes_unsaved()
            self.labels_widget.refresh_labels_shapes_layer()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
            if self.data_widget.plot_container:
                self.data_widget.plot_container.labels_redraw_needed.emit()

        notify(f"Merged {len(incoming)} label row(s) from {Path(file_path).name}")

    def _close_labels_popup(self):
        """Close the top-bar "Import labels" popup hosting ``labels_group``.

        Only relevant after a direct .tsv import; conversion formats
        (crowsetta, pynapple) keep the popup open so the user can see the
        converted-TSV output path.
        """
        popup = self.labels_group.window()
        if isinstance(popup, SectionPopup):
            popup.close()

    def _import_crowsetta_labels(self, format_name):
        filter_map = {
            "aud-seq": "Text files (*.txt)",
            "simple-seq": "CSV/Text files (*.csv *.txt)",
            "generic-seq": "CSV files (*.csv)",
            "notmat": "NotMat files (*.not.mat)",
            "textgrid": "TextGrid files (*.TextGrid)",
            "timit": "PHN files (*.phn)",
            "yarden": "Yarden annotation files (*.mat)",
        }
        file_filter = filter_map.get(format_name, "All files (*)")

        file_path = browse_open_file(
            self,
            self.app_state,
            f"Open {format_name} annotation file",
            file_filter,
            preferred_dir=self.app_state.nc_file_path,
        )
        if not file_path:
            return

        self.label_file_path_edit.setText(file_path)
        self._do_crowsetta_import(format_name, file_path)

    def _do_crowsetta_import(self, format_name, file_path):
        from ethograph.labels.converters import (
            crowsetta_to_intervals,
            resolve_crowsetta_mapping,
        )

        data_dir = Path(self.app_state.nc_file_path).parent if self.app_state.nc_file_path else None
        configs_dir = default_config_dir(data_dir)
        mapping_path = self.mapping_file_path_edit.text()

        try:
            name_to_id, new_mapping_path, warning = resolve_crowsetta_mapping(
                file_path,
                format_name,
                mapping_path,
                configs_dir,
            )
        except (OSError, ValueError, KeyError) as e:
            logger.exception("Crowsetta mapping resolution failed")
            notify_dialog(str(e), "error", "Mapping error", self)
            return

        if warning:
            notify_dialog(warning, "warning", "Mapping warning", self)

        if new_mapping_path:
            self.mapping_file_path_edit.setText(new_mapping_path)
            if self.labels_widget:
                self.labels_widget._reload_mapping(new_mapping_path)

        individual = "ind0"
        ds = getattr(self.app_state, "ds", None)
        _ind_dim = next((n for n in INDIVIDUAL_DIMS if ds is not None and n in ds.coords), None)
        if _ind_dim is not None:
            individual = str(ds.coords[_ind_dim].values[0])

        try:
            intervals_df = crowsetta_to_intervals(
                file_path,
                format_name,
                name_to_id,
                individual,
            )
        except (OSError, ValueError, KeyError) as e:
            logger.exception("Failed to parse %s file", format_name)
            notify_dialog(
                f"Failed to parse {format_name} file:\n{e}",
                "error",
                "Import error",
                self,
            )
            return

        if intervals_df.empty:
            notify_dialog("No non-background labels found in file.", "info", "No labels", self)
            return

        self._apply_imported_intervals(intervals_df)

    def _import_pynapple_labels(self):
        """Import labels from a pynapple .npz or .nwb file."""
        import pynapple as nap

        fmt = self.labels_format_combo.currentText()
        ext_filter = {
            "pynapple (.npz)": "NPZ files (*.npz);;All files (*)",
            "pynapple (.nwb)": "NWB files (*.nwb);;All files (*)",
        }.get(fmt, "All files (*)")

        file_path = browse_open_file(
            self,
            self.app_state,
            f"Open {fmt} file for labels",
            ext_filter,
            preferred_dir=self.app_state.nc_file_path,
        )
        if not file_path:
            return

        self.label_file_path_edit.setText(file_path)

        try:
            data = nap.load_file(file_path)
        except Exception as e:
            logger.exception("Failed to load pynapple file")
            notify_dialog(f"Failed to load pynapple file:\n{e}", "error", "Import error", self)
            return

        intervalsets = {}
        for key in data.keys():
            try:
                val = data[key]
            except Exception:
                continue
            if isinstance(val, nap.IntervalSet) and key.lower() != "trials":
                intervalsets[key] = val

        if not intervalsets:
            notify_dialog("No IntervalSets found (excluding 'trials').", "info", "No labels", self)
            return

        from ethograph.labels.converters import (
            build_mapping_from_labels,
            write_mapping_file,
        )
        from ethograph.labels.intervals import _rows_to_df

        label_names = sorted(intervalsets.keys())
        name_to_id = build_mapping_from_labels(label_names)

        data_dir = Path(self.app_state.nc_file_path).parent if self.app_state.nc_file_path else None
        configs_dir = default_config_dir(data_dir)
        mapping_path = configs_dir / "mapping_pynapple.txt"
        write_mapping_file(mapping_path, name_to_id)
        self.mapping_file_path_edit.setText(str(mapping_path))
        if self.labels_widget:
            self.labels_widget._reload_mapping(str(mapping_path))

        individual = "ind0"
        ds = getattr(self.app_state, "ds", None)
        _ind_dim = next((n for n in INDIVIDUAL_DIMS if ds is not None and n in ds.coords), None)
        if _ind_dim is not None:
            individual = str(ds.coords[_ind_dim].values[0])

        rows: list[dict] = []
        for name, iset in intervalsets.items():
            label_id = name_to_id.get(name, 0)
            if label_id == 0:
                continue
            starts = np.asarray(iset.start)
            ends = np.asarray(iset.end)
            for s, e in zip(starts, ends):
                rows.append(
                    {
                        "onset_s": float(s),
                        "offset_s": float(e),
                        "labels": label_id,
                        "individual": individual,
                    }
                )

        intervals_df = _rows_to_df(rows)
        if intervals_df.empty:
            notify_dialog("No intervals found in IntervalSets.", "info", "No labels", self)
            return

        self._apply_imported_intervals(intervals_df)

    def _apply_imported_intervals(self, intervals_df):
        """Common post-import: save converted TSV, load into app state, refresh UI."""
        from ethograph.labels.tsv_store import (
            save_labels_tsv,
        )

        # Set the complete imported dataframe as the active labels
        self.app_state._all_labels_df = intervals_df
        self.app_state.clear_label_history()
        if self.data_widget:
            self.data_widget.refresh_individual_choices()

        trial = getattr(self.app_state, "trials_sel", None)
        if trial is not None:
            self.app_state.label_intervals = self.app_state.get_trial_intervals(trial)

        # For non-.tsv formats, save the converted TSV and show in output row
        fmt = self.labels_format_combo.currentText()
        if fmt != ".tsv" and self.app_state.nc_file_path:
            tsv_out = labels_tsv_path(self.app_state.nc_file_path)
            if intervals_df is not None and not intervals_df.empty:
                save_labels_tsv(tsv_out, intervals_df)
            self.app_state._labels_file_path = str(tsv_out)  # Track the converted TSV as active
            self.labels_output_edit.setText(str(tsv_out))

        if hasattr(self, "changepoints_widget") and self.changepoints_widget:
            self.changepoints_widget._update_cp_status()
        if self.labels_widget:
            self.labels_widget._mark_changes_unsaved()
            self.labels_widget.refresh_labels_shapes_layer()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
            if self.data_widget.plot_container:
                self.data_widget.plot_container.labels_redraw_needed.emit()

    # ------------------------------------------------------------------
    # Post-load behavior
    # ------------------------------------------------------------------

    def on_load_complete(self):
        """Disable the load panel (a dataset is active) and prep import panels."""
        for child in self.load_panel.findChildren(QWidget):
            child.setEnabled(False)

        self._ensure_crowsetta_formats()

        canary_path = getattr(self, "_canary_labels_path", None)
        if canary_path:
            self.labels_format_combo.setCurrentText("aud-seq")
            self.label_file_path_edit.setText(canary_path)
            del self._canary_labels_path

        self._auto_populate_nwb_video_folder()
        self._auto_discover_nwb()
        self._auto_discover_metadata()
        self._auto_import_crowsetta_labels()
        self._apply_nwb_epoch_mapping()

    def _auto_populate_nwb_video_folder(self):
        """If the loaded NWB file downloaded trial clips, auto-fill the video folder field."""
        dt = getattr(self.app_state, "dt", None)
        if dt is None:
            return
        video_folder = getattr(self.app_state, "nwb_video_folder", None)
        if not video_folder:
            return
        video_folder = str(video_folder)
        self.video_folder_edit.setText(video_folder)
        self.app_state.video_folder = video_folder
        logger.info("NWB auto-set video folder: %s", video_folder)

    def _apply_nwb_epoch_mapping(self):
        """If NWB epochs were imported, write mapping file and load into labels widget.

        Writes the mapping only on first load (when the file doesn't already
        exist).  On subsequent loads the existing file is reused so user edits
        (branch assignments, ``event_type`` toggles) are preserved.
        """
        dt = getattr(self.app_state, "dt", None)
        if dt is None:
            return
        epoch_mapping = getattr(self.app_state, "nwb_epoch_mapping", None)
        if not epoch_mapping or not isinstance(epoch_mapping, dict):
            return

        from ethograph.labels.intervals import (
            EVENT_TYPE_STATE,
            save_label_mapping,
        )

        data_dir = Path(self.app_state.nc_file_path).parent if self.app_state.nc_file_path else None
        mapping_path = default_config_dir(data_dir) / "mapping_nwb_epochs.txt"

        if not mapping_path.exists():
            save_label_mapping(
                mapping_path,
                {
                    label_id: {
                        "name": name,
                        "branch": 0,
                        "event_type": EVENT_TYPE_STATE,
                    }
                    for name, label_id in epoch_mapping.items()
                },
            )
            n_labels = len(epoch_mapping) - 1  # exclude background
            logger.info(
                "NWB auto-created mapping with %d epoch labels: %s",
                n_labels,
                mapping_path,
            )
        else:
            logger.info(
                "Reusing existing NWB epoch mapping (preserving user edits): %s",
                mapping_path,
            )

        self.mapping_file_path_edit.setText(str(mapping_path))
        if self.labels_widget:
            self.labels_widget._reload_mapping(str(mapping_path))

    def _ensure_crowsetta_formats(self):
        """Add crowsetta formats to labels combo if not already present."""
        from ethograph.labels.converters import CROWSETTA_SEQ_FORMATS

        existing = [self.labels_format_combo.itemText(i) for i in range(self.labels_format_combo.count())]
        for fmt in CROWSETTA_SEQ_FORMATS:
            if fmt not in existing:
                self.labels_format_combo.addItem(fmt)

    def _auto_import_crowsetta_labels(self):
        """If a crowsetta format and path are set, auto-import after load."""
        fmt = self.labels_format_combo.currentText()
        file_path = self.label_file_path_edit.text().strip()
        if fmt in (".tsv", ".nc (legacy)") or not file_path or not Path(file_path).exists():
            return
        self._do_crowsetta_import(fmt, file_path)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _on_reset_gui_clicked(self):
        self.downsample_checkbox.setChecked(False)
        self.import_labels_checkbox.setChecked(False)
        # Global scope only — Help ▸ "Reset local settings" owns the dataset's
        # local_settings.yaml.
        self.app_state.delete_yaml(str(self.app_state._global_settings_path()))

        for var in AppStateSpec.VARS:
            default = AppStateSpec.get_default(var)
            setattr(self.app_state, var, default)

        for attr in list(dir(self.app_state)):
            if attr.endswith("_sel"):
                try:
                    delattr(self.app_state, attr)
                except AttributeError:
                    pass

        self._clear_all_line_edits()
        self._clear_combo_boxes()

        global_settings = default_config_dir() / "gui_settings.yaml"
        self.app_state._yaml_path = str(global_settings)
        self.app_state.save_to_yaml()

    def _on_create_nc_clicked(self):
        self._clear_all_line_edits()
        dialog = NCWizardDialog(self.app_state, self, self)
        dialog.exec_()

    def _on_select_template_clicked(self):
        self._clear_all_line_edits()
        dialog = TemplateDialog(self)
        if dialog.exec_() and dialog.selected_template:
            t = dialog.selected_template
            if t["nc_file_path"]:
                self.nc_file_path_edit.setText(t["nc_file_path"])
                self.app_state.nc_file_path = t["nc_file_path"]
            if t["video_folder"]:
                self.video_folder_edit.setText(t["video_folder"])
                self.app_state.video_folder = t["video_folder"]
            if t["audio_folder"]:
                self.audio_folder_edit.setText(t["audio_folder"])
                self.app_state.audio_folder = t["audio_folder"]
            if t["pose_folder"]:
                self.pose_folder_edit.setText(t["pose_folder"])
                self.app_state.pose_folder = t["pose_folder"]
            if t.get("import_labels"):
                self.import_labels_checkbox.setChecked(True)
            if t.get("downsample"):
                self.downsample_checkbox.setChecked(True)
                self.downsample_spin.setValue(int(t["downsample"]))
            if t.get("labels_file"):
                self._canary_labels_path = t["labels_file"]
            if t.get("library_geometry"):
                self.app_state.space_library_geometry = t["library_geometry"]

            self._on_load_clicked()

    def _on_load_clicked(self):
        from .dialog_busy_progress import BusyProgressDialog

        dialog = BusyProgressDialog("Loading data...", parent=self)

        def _update(msg: str) -> None:
            dialog.setLabelText(msg)
            dialog.pump_events()

        self.app_state._progress_callback = _update
        dialog.execute_blocking(self.data_widget.on_load_clicked)

    def _clear_all_line_edits(self):
        for attr in (
            "nc_file_path_edit",
            "nwb_file_path_edit",
            "metadata_path_edit",
            "video_folder_edit",
            "audio_folder_edit",
            "pose_folder_edit",
            "ephys_path_edit",
            "neurons_path_edit",
            "label_file_path_edit",
            "pred_file_path_edit",
        ):
            widget = getattr(self, attr, None)
            if widget:
                widget.clear()
            state_key = attr.removesuffix("_edit")
            if hasattr(self.app_state, state_key):
                setattr(self.app_state, state_key, None)
        self.app_state.ephys_offset = 0.0
        self.downsample_checkbox.setChecked(False)

    def _clear_combo_boxes(self):
        for combo in self.combos.values():
            combo.clear()
            combo.addItems(["None"])
            combo.setCurrentText("None")

    def _create_path_widget(self, target_layout, label, object_name, browse_callback, extra_buttons=None):
        line_edit = QLineEdit()
        line_edit.setObjectName(f"{object_name}_edit")
        if object_name == "nc_file_path":
            line_edit.setPlaceholderText("Path to .nc / .nwb / .npz file or pynapple folder")

        browse_button = QPushButton("Browse")
        browse_button.setObjectName(f"{object_name}_browse_button")
        browse_button.clicked.connect(browse_callback)

        if object_name == "nc_file_path":
            self.import_labels_checkbox = QCheckBox("Import labels")
            self.import_labels_checkbox.setObjectName("import_labels_checkbox")
            self.import_labels_checkbox.setToolTip("Load labels from {name}_labels.tsv alongside the .nc file.\n")
            self.import_labels_checkbox.stateChanged.connect(self._on_import_labels_checked)
            self.import_labels_checkbox.setChecked(bool(self.app_state.import_labels_nc_data))

        clear_button = QPushButton("Clear")
        clear_button.setObjectName(f"{object_name}_clear_button")
        clear_button.clicked.connect(lambda: self._on_clear_path_clicked(object_name, line_edit))

        row_layout = QHBoxLayout()
        row_layout.addWidget(line_edit)
        if extra_buttons:
            for btn in extra_buttons:
                row_layout.addWidget(btn)
        row_layout.addWidget(browse_button)
        if object_name == "nc_file_path":
            browse_folder_button = QPushButton("Browse folder")
            browse_folder_button.setObjectName(f"{object_name}_browse_folder_button")
            browse_folder_button.setToolTip("Browse for a pynapple data folder")
            browse_folder_button.clicked.connect(self._browse_data_folder)
            row_layout.addWidget(browse_folder_button)
        row_layout.addWidget(clear_button)
        target_layout.addRow(label, row_layout)

        return line_edit

    def _on_import_labels_checked(self, state):
        checked = Qt.CheckState(state) == Qt.Checked
        self.app_state.import_labels_nc_data = checked
        if checked:
            path = self._resolve_import_labels_path()
            if path and hasattr(self, "label_file_path_edit"):
                self.label_file_path_edit.setText(path)

    def _resolve_import_labels_path(self) -> str | None:
        """Resolve + persist the explicit "Import labels" path.

        ``app_state.labels_import_path`` (SCOPE_LOCAL) is the single source of
        truth once set — remembered per dataset instead of re-derived from the
        ``.nc`` filename on every load, which silently found nothing whenever
        a labels file didn't happen to match the ``{stem}_labels.tsv``
        convention. Seeded once, on first use for a dataset that has never
        set it, from that same canonical guess (still correct for datasets
        that DO follow it, e.g. downloaded templates) — after that the guess
        is never repeated. The guess is only seeded when the file exists:
        the checkbox itself is a global preference, so a dataset with no
        labels file must resolve to nothing (and load without labels) rather
        than pin a nonexistent path that would error every future load.
        """
        if self.app_state.labels_import_path:
            return self.app_state.labels_import_path
        if not self.app_state.nc_file_path:
            return None
        guess = labels_tsv_path(self.app_state.nc_file_path)
        if not guess.exists():
            return None
        self.app_state.labels_import_path = str(guess)
        return str(guess)

    def _on_clear_path_clicked(self, object_name, line_edit):
        line_edit.setText("")
        attr_map = {
            "nc_file_path": "nc_file_path",
            "nwb_file_path": "nwb_file_path",
            "video_folder": "video_folder",
            "audio_folder": "audio_folder",
            "pose_folder": "pose_folder",
            "ephys_path": "ephys_path",
            "neurons_path": "neurons_path",
        }
        attr = attr_map.get(object_name)
        if attr:
            setattr(self.app_state, attr, None)

    # Device controls (populated after load)
    # ------------------------------------------------------------------

    def create_device_controls(self, catalog):
        self._create_labels_row_at_index()
        self.controls.append(self.label_file_path_edit)

    def _expand_ephys_with_streams(self, ephys_path, ds):
        """Discover Neo streams from the ephys file for the Neo-Viewer."""
        from ..io.ephys_loader import load_ephys

        self.app_state.ephys_source_map.clear()
        feature_names = []

        if not ephys_path:
            return feature_names

        filepath = os.path.normpath(str(ephys_path))

        try:
            loader = load_ephys(filepath, stream_id="0")
            streams = loader.stream_info

            if streams and len(streams) > 1:
                for sid, info in streams.items():
                    display_name = info["name"]
                    self.app_state.ephys_source_map[display_name] = (filepath, str(sid), 0)
                    feature_names.append(display_name)
            else:
                display_name = "Ephys Waveform"
                self.app_state.ephys_source_map[display_name] = (filepath, "0", 0)
                feature_names.append(display_name)
        except (OSError, IOError, ValueError) as e:
            logger.error("Skipping ephys file %s: %s", Path(filepath).name, e)

        return feature_names

    def _create_combo_widget(self, key, vars):
        combo = QComboBox()
        combo.setObjectName(f"{key}_combo")
        combo.currentTextChanged.connect(self._on_combo_changed)
        combo.addItems([str(var) for var in vars])

        self._controls_layout.addRow(f"{key.capitalize()}:", combo)
        self.combos[key] = combo
        self.controls.append(combo)
        return combo

    def _on_combo_changed(self):
        if hasattr(self.data_widget, "_on_combo_changed"):
            self.data_widget._on_combo_changed()

    def set_controls_enabled(self, enabled):
        for control in self.controls:
            control.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Load button + downsample
    # ------------------------------------------------------------------

    def _create_load_button(self, target_layout):
        controls_layout = QHBoxLayout()

        self.downsample_checkbox = QCheckBox("Downsample:")
        self.downsample_checkbox.setObjectName("downsample_checkbox")
        self.downsample_checkbox.setChecked(self.app_state.downsample_enabled)
        self.downsample_checkbox.setToolTip("Downsample data on load for faster display")
        self.downsample_checkbox.toggled.connect(self._on_downsample_toggled)

        self.downsample_spin = QSpinBox()
        self.downsample_spin.setObjectName("downsample_spin")
        self.downsample_spin.setRange(2, 1000)
        self.downsample_spin.setValue(self.app_state.downsample_factor)
        self.downsample_spin.setEnabled(self.app_state.downsample_enabled)
        self.downsample_spin.setToolTip("Downsample factor (e.g., 100 = keep 1 in 100 samples)")
        self.downsample_spin.setFixedWidth(70)
        self.downsample_spin.valueChanged.connect(self._on_downsample_value_changed)

        self.load_button = QPushButton("Load")
        self.load_button.setObjectName("load_button")
        self.load_button.clicked.connect(self._on_load_clicked)

        controls_layout.addWidget(self.import_labels_checkbox)
        controls_layout.addWidget(self.downsample_checkbox)
        controls_layout.addWidget(self.downsample_spin)
        controls_layout.addStretch()

        load_button_layout = QHBoxLayout()
        load_button_layout.setContentsMargins(0, 0, 0, 0)
        load_button_layout.addWidget(self.load_button)
        self.load_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        target_layout.addRow(controls_layout)
        target_layout.addRow(load_button_layout)

    def _on_downsample_toggled(self, checked):
        self.downsample_spin.setEnabled(checked)
        self.app_state.downsample_enabled = checked

    def _on_downsample_value_changed(self, value):
        self.app_state.downsample_factor = value

    def disable_downsample_controls(self):
        self.downsample_checkbox.setEnabled(False)
        self.downsample_spin.setEnabled(False)

    def get_downsample_factor(self):
        if self.downsample_checkbox.isChecked():
            return self.downsample_spin.value()
        return None

    # ------------------------------------------------------------------
    # Browse dialogs
    # ------------------------------------------------------------------

    def _browse_nwb_file(self):
        """Browse for an NWB session/alignment file."""
        path = browse_open_file(
            None,
            self.app_state,
            "Open NWB session file",
            "NWB files (*.nwb);;All files (*)",
            preferred_dir=self.app_state.nc_file_path,
        )
        if path:
            self.nwb_file_path_edit.setText(path)
            self.app_state.nwb_file_path = path  # auto-syncs to app_state.nwb_alignment

    def _browse_metadata_file(self):
        """Browse for a metadata source file (TSV, CSV, Excel, NWB, or NPZ)."""
        path = browse_open_file(
            None,
            self.app_state,
            "Open metadata file",
            (
                "Metadata files (*.tsv *.csv *.xlsx *.xls *.nwb *.npz);;"
                "TSV/CSV files (*.tsv *.csv);;"
                "Excel files (*.xlsx *.xls);;"
                "NWB files (*.nwb);;"
                "Pynapple files (*.npz);;"
                "All files (*)"
            ),
            preferred_dir=self.app_state.nc_file_path,
        )
        if path:
            self.metadata_path_edit.setText(path)
            self.app_state.metadata_path = path

    def _save_metadata_template(self):
        """Save a metadata template TSV that users can fill in with Excel."""
        import pandas as pd

        nc_path = self.app_state.nc_file_path
        default_name = Path(nc_path).stem + "_metadata.tsv" if nc_path else "metadata.tsv"

        path = browse_save_file(
            None,
            self.app_state,
            "Save metadata template",
            default_name,
            "TSV files (*.tsv);;CSV files (*.csv);;All files (*)",
            preferred_dir=nc_path,
        )
        if not path:
            return

        trials = getattr(self.app_state, "trials", None)
        if trials:
            trial_ids = list(trials)
        else:
            trial_ids = [1, 2, 3]

        df = pd.DataFrame(
            {
                "trial": trial_ids,
                "start_time": [""] * len(trial_ids),
                "stop_time": [""] * len(trial_ids),
                "condition": [""] * len(trial_ids),
            }
        )

        path = Path(path)
        if path.suffix.lower() in (".csv",):
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, sep="\t", index=False)

        self.metadata_path_edit.setText(str(path))
        self.app_state.metadata_path = str(path)
        logger.info("Saved metadata template to %s", path)

        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError:
            logger.warning("Could not open %s in default application", path)

    def _auto_discover_nwb(self):
        """Auto-discover alignment.nwb near the loaded data file."""
        from ethograph.utils.paths import find_nwb_file

        # Skip if data_loader already provided a valid alignment
        if self.app_state.nwb_alignment is not None:
            return

        nc_path = self.app_state.nc_file_path
        if not nc_path:
            return

        source = Path(nc_path)
        if source.suffix.lower() == ".nwb" and source.exists():
            self.nwb_file_path_edit.setText(str(source))
            self.app_state.nwb_file_path = str(source)
            logger.info("Using source NWB for alignment: %s", source)
            return

        # For project directories, search inside the dir itself
        data_dir = str(source) if source.is_dir() else str(source.parent)
        nwb = find_nwb_file(data_dir)
        if nwb is not None:
            self.nwb_file_path_edit.setText(str(nwb))
            self.app_state.nwb_file_path = str(nwb)
            logger.info("Auto-discovered NWB alignment: %s", nwb)

    def _auto_discover_metadata(self):
        """Auto-discover {stem}_metadata.tsv near the loaded data file."""
        nc_path = self.app_state.nc_file_path
        if not nc_path:
            return
        tsv = metadata_tsv_path(nc_path)
        populate_if_exists(self.metadata_path_edit, tsv)
        if tsv.exists():
            self.app_state.metadata_path = str(tsv)

    def _try_auto_populate_alignment(self, selected_path: str) -> None:
        """Populate the alignment field from .ethograph/alignment.nwb if not already set."""
        if self.nwb_file_path_edit.text().strip():
            return
        from ethograph.utils.paths import find_nwb_file

        p = Path(selected_path)
        data_dir = str(p) if p.is_dir() else str(p.parent)
        nwb = find_nwb_file(data_dir)
        if nwb is not None:
            self.nwb_file_path_edit.setText(str(nwb))
            logger.info("Auto-populated alignment from browse: %s", nwb)

    def _browse_data_file(self):
        """Browse for a data file (.nc, .nwb, .npz)."""
        path = browse_open_file(
            None,
            self.app_state,
            "Open data file",
            "Data files (*.nc *.nwb *.npz);;All files (*)",
            preferred_dir=self.nc_file_path_edit.text().strip() or None,
        )
        if path:
            self.nc_file_path_edit.setText(path)
            self.app_state.nc_file_path = path
            self._try_auto_populate_alignment(path)

    def _browse_data_folder(self):
        """Browse for a pynapple data folder."""
        path = browse_open_dir(
            None,
            self.app_state,
            "Open pynapple data folder",
            preferred_dir=self.nc_file_path_edit.text().strip() or None,
        )
        if path:
            self.nc_file_path_edit.setText(path)
            self.app_state.nc_file_path = path
            self._try_auto_populate_alignment(path)

    def on_browse_clicked(self, browse_type="file", media_type=None):
        if browse_type == "file":
            if media_type == "data":
                self._browse_data_file()
                return

            elif media_type == "labels":
                labels_file_path = browse_open_file(
                    None,
                    self.app_state,
                    "Load labels TSV file",
                    "TSV files (*.tsv)",
                    preferred_dir=self.app_state.nc_file_path,
                )
                if not labels_file_path:
                    return

                try:
                    labels_df = load_labels_tsv(labels_file_path)
                except (FileNotFoundError, ValueError) as e:
                    notify(str(e), severity="error")
                    return

                self.app_state._all_labels_df = labels_df
                self.app_state.clear_label_history()
                if self.data_widget:
                    self.data_widget.refresh_individual_choices()

                self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)
                self.label_file_path_edit.setText(labels_file_path)

                if hasattr(self, "changepoints_widget") and self.changepoints_widget:
                    self.changepoints_widget._update_cp_status()
                if self.labels_widget:
                    self.labels_widget._mark_changes_unsaved()
                    self.labels_widget.refresh_labels_shapes_layer()
                if self.data_widget:
                    self.data_widget.update_main_plot(preserve_x_range=True)
                    if self.data_widget.plot_container:
                        self.data_widget.plot_container.labels_redraw_needed.emit()

            elif media_type == "ephys":
                ephys_path = browse_open_file(
                    None,
                    self.app_state,
                    "Open ephys recording file",
                    EPHYS_FILE_FILTER,
                    preferred_dir=self.app_state.ephys_path or self.app_state.nc_file_path,
                )
                if not ephys_path:
                    return

                self.ephys_path_edit.setText(ephys_path)
                self.app_state.ephys_path = ephys_path
                self._auto_detect_neurons(ephys_path)

        elif browse_type == "folder":
            if media_type == "video":
                caption = "Open folder with video files (e.g. mp4, mov)."
                current = self.app_state.video_folder
            elif media_type == "audio":
                caption = "Open folder with audio files (e.g. wav, mp3, mp4)."
                current = self.app_state.audio_folder
            elif media_type == "pose":
                caption = "Open folder with pose files (e.g. .csv, .h5)."
                current = self.app_state.pose_folder

            folder_path = browse_open_dir(
                None,
                self.app_state,
                caption,
                preferred_dir=current or self.app_state.nc_file_path,
            )
            # A cancelled dialog must leave the current folder alone — writing
            # the empty result back would silently unset it.
            if not folder_path:
                return

            if media_type == "video":
                self.video_folder_edit.setText(folder_path)
                self.app_state.video_folder = folder_path
            elif media_type == "audio":
                self.audio_folder_edit.setText(folder_path)
                self.app_state.audio_folder = folder_path
                if hasattr(self.data_widget, "clear_audio_checkbox"):
                    self.data_widget.clear_audio_checkbox.setChecked(False)
            elif media_type == "pose":
                self.pose_folder_edit.setText(folder_path)
                self.app_state.pose_folder = folder_path

    def _browse_neurons(self):
        """Browse for a Kilosort folder or Pynapple file (.npz, .nwb)."""
        preferred = self.app_state.neurons_path or self.app_state.ephys_path or self.app_state.nc_file_path

        dialog = QDialog(self)
        dialog.setWindowTitle("Load neuron data")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select the type of neuron data to load:"))

        btn_ks = QPushButton("Kilosort Folder")
        btn_ks.setToolTip("Select a Kilosort output folder (spike_times.npy, etc.)")
        btn_nap = QPushButton("Pynapple File (.npz / .nwb)")
        btn_nap.setToolTip("Select a Pynapple-compatible file with a 'units' TsGroup")

        layout.addWidget(btn_ks)
        layout.addWidget(btn_nap)

        chosen_path = [None]

        def _on_kilosort():
            folder = browse_open_dir(
                dialog,
                self.app_state,
                "Select Kilosort output folder",
                preferred_dir=preferred,
            )
            if folder:
                chosen_path[0] = folder
                dialog.accept()

        def _on_pynapple():
            path = browse_open_file(
                dialog,
                self.app_state,
                "Select Pynapple file",
                "Pynapple files (*.npz *.nwb);;All files (*)",
                preferred_dir=preferred,
            )
            if path:
                chosen_path[0] = path
                dialog.accept()

        btn_ks.clicked.connect(_on_kilosort)
        btn_nap.clicked.connect(_on_pynapple)

        if dialog.exec_() == QDialog.Accepted and chosen_path[0]:
            self.neurons_path_edit.setText(chosen_path[0])
            self.app_state.neurons_path = chosen_path[0]
            self.neurons_path_edit.returnPressed.emit()

    def _auto_detect_neurons(self, ephys_path: str):
        ephys_parent = Path(ephys_path).parent
        for folder_name in ("kilosort4", "kilosort"):
            ks_folder = ephys_parent / folder_name
            if ks_folder.is_dir():
                self.neurons_path_edit.setText(str(ks_folder))
                self.app_state.neurons_path = str(ks_folder)
                return
        self.neurons_path_edit.clear()
        self.app_state.neurons_path = None

    def get_nc_file_path(self):
        return self.nc_file_path_edit.text().strip()

    def get_import_labels_path(self) -> str | None:
        """Explicit labels TSV override for "Import labels", resolved at Load time.

        ``app_state.labels_import_path`` (SCOPE_LOCAL) is the single source of
        truth: set explicitly (cover-page labels drop, "Import labels…" browse)
        or seeded once from the canonical ``{stem}_labels.tsv`` guess the first
        time it resolves for a dataset that has never set it (only if that
        file exists — the checkbox is a global preference, so datasets without
        labels resolve to ``None`` and load without labels). An explicitly-set
        path that has gone missing is a load error, not a silent no-op (see
        ``_phase_load_data``).
        """
        if not self.import_labels_checkbox.isChecked():
            return None
        path = self._resolve_import_labels_path()
        if path is None:
            logger.info("'Import labels' is checked but no labels file resolved — loading without labels.")
        return path

    # ------------------------------------------------------------------
    # Wire signals to other widgets (called from MetaWidget)
    # ------------------------------------------------------------------

    def wire_label_signals(self):
        """Connect mapping/predictions UI to LabelsWidget methods."""
        self.mapping_file_path_edit.returnPressed.connect(
            lambda: self.labels_widget._reload_mapping(self.mapping_file_path_edit.text())
        )
        self.browse_mapping_btn.clicked.connect(self.labels_widget._browse_mapping_file)
        self.temp_labels_button.clicked.connect(self.labels_widget._create_temporary_labels)
        self.import_predictions_from_folder_action.triggered.connect(self.labels_widget._import_predictions_from_folder)
        self.import_predictions_from_tsv_action.triggered.connect(self.labels_widget._import_predictions_from_tsv)
        self.pred_confidence_pdf_btn.clicked.connect(self.labels_widget._plot_confidence_pdf)
        self.pred_confidence_threshold_spin.valueChanged.connect(self.labels_widget._on_confidence_threshold_changed)
        self.pred_segment_confidence_threshold_spin.valueChanged.connect(
            self.labels_widget._on_confidence_threshold_changed
        )

    def wire_ephys_signals(self, ephys_widget):
        """Connect neurons UI to EphysWidget methods."""
        self.neurons_path_edit.returnPressed.connect(ephys_widget._load_neurons)
