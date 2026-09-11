"""Widget for selecting start/stop times and playing a segment."""

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import ethograph as eto
from ethograph.gui.notify import notify, notify_dialog
from ethograph.gui.pose_convert import COLOR_BY_INDIVIDUAL, COLOR_BY_KEYPOINT, individual_color_map
from ethograph.io.catalog import INDIVIDUAL_DIMS, ComboSpec
from ethograph.io.data_loader import load_features_dataset
from ethograph.io.derived import DerivedLoader
from ethograph.io.plot_sources import FileSource
from ethograph.io.time_model import compute_trial_video_bounds
from ethograph.labels.intervals import get_interval_bounds, select_subject
from ethograph.utils.qt import (
    ElidedDelegate,
    find_combo_index,
    get_combo_value,
    make_searchable,
    set_combo_to_value,
)

from ..io.ephys_loader import load_ephys
from .app_constants import (
    DEFAULT_LAYOUT_MARGIN,
    DEFAULT_LAYOUT_SPACING,
    SESSION_LABELS_MAX_DRAWN,
    SIDEBAR_AFTER_LOAD_WIDTH_RATIO,
)
from .dialog_keypoint_filter import KeypointFilterDialog
from .make_pretty import clean_display_labels
from .plots_radial import RadialPlot
from .plots_space import SpacePlot
from .plots_spectrogram import SharedAudioCache
from .pose_render import (
    PoseDisplayManager,
    strip_common_prefix,
)
from .video_manager import VideoManager, is_url
from .widgets_transform import ENERGY_DISPLAY_NAMES, compute_energy_envelope_multichannel

logger = logging.getLogger(__name__)


def _color_swatch_icon(color_hex: str) -> QIcon:
    """Return a small filled-square icon for a color-picker button."""
    pix = QPixmap(20, 20)
    pix.fill(QColor(color_hex))
    return QIcon(pix)


def _rgba_to_hex(rgba: tuple) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(round(c * 255)) for c in rgba[:3]))


def _sync_spin(spin: QDoubleSpinBox, value: float) -> None:
    """Mirror *value* into a twin widget without re-firing its handler."""
    if spin.value() != value:
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)


def _sync_check(box: QCheckBox, checked: bool) -> None:
    if box.isChecked() != checked:
        box.blockSignals(True)
        box.setChecked(checked)
        box.blockSignals(False)


def _detect_nwb_pose_keys(nwb_path: str | None) -> list[str] | None:
    """Detect PoseEstimation container names from NWB processing modules."""
    if not nwb_path or not Path(nwb_path).exists():
        return None
    try:
        from pynwb import NWBHDF5IO

        keys = []
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            for mod in nwb.processing.values():
                for name, di in mod.data_interfaces.items():
                    if hasattr(di, "pose_estimation_series"):
                        keys.append(name)
        return keys if keys else None
    except Exception:
        return None


class _LoadError(Exception):
    """Raised during any loading phase to abort with a user-visible message."""


@dataclass
class _LoadContext:
    """Accumulated load state -- app_state is not mutated until _apply_to_state.

    Each loading phase reads from and writes to this context.  If any phase
    raises ``_LoadError``, the load is cancelled and app_state remains clean.
    """

    result: object  # LoadResult from data_loader.load_dataset
    nc_file_path: str
    catalog: object  # DataCatalog

    # Inferred media availability
    has_audio: bool = False
    has_neo: bool = False
    has_neurons: bool = False
    cameras: list = field(default_factory=list)
    nwb_local: str | None = None

    # NWB-embedded ephys
    nwb_ephys_display: str | None = None
    nwb_ephys_entry: tuple | None = None

    # Dataset (possibly downsampled)
    dt: object = None
    ds: object = None
    trials: list = field(default_factory=list)
    all_labels_df: object = None
    data_loader: object = None
    downsample_factor: int | None = None
    video_folder_override: str | None = None


class DataPanel(QWidget):
    """Visible panel for the 'Data' collapsible section in the sidebar.

    Organised into three tabs: Main, Pose, Audio.
    """

    ENERGY_DISPLAY_NAMES = ENERGY_DISPLAY_NAMES

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self._update_pose_callback = None
        layout = QVBoxLayout()
        layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        layout.setContentsMargins(
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
        )
        self.setLayout(layout)

        self._create_toggle_buttons(layout)

        self.main_panel = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.main_panel.setLayout(main_layout)
        self._create_main_section(main_layout)
        layout.addWidget(self.main_panel)

        self.pose_panel = QWidget()
        pose_layout = QVBoxLayout()
        pose_layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        pose_layout.setContentsMargins(2, 2, 2, 2)
        self.pose_panel.setLayout(pose_layout)
        self._create_video_crop_section(pose_layout)
        self._create_video_label_section(pose_layout)
        self._create_pose_section(pose_layout)
        layout.addWidget(self.pose_panel)

        self.audio_panel = QWidget()
        audio_layout = QVBoxLayout()
        audio_layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        audio_layout.setContentsMargins(2, 2, 2, 2)
        self.audio_panel.setLayout(audio_layout)
        self._create_audio_section(audio_layout)
        layout.addWidget(self.audio_panel)

        self._show_panel("main")

    def _create_toggle_buttons(self, parent_layout):
        toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_widget.setLayout(toggle_layout)

        self.main_toggle = QPushButton("Main")
        self.main_toggle.setCheckable(True)
        self.main_toggle.clicked.connect(lambda: self._show_panel("main"))
        toggle_layout.addWidget(self.main_toggle)

        self.pose_toggle = QPushButton("Pose")
        self.pose_toggle.setCheckable(True)
        self.pose_toggle.clicked.connect(lambda: self._show_panel("pose"))
        toggle_layout.addWidget(self.pose_toggle)

        self.audio_toggle = QPushButton("Audio")
        self.audio_toggle.setCheckable(True)
        self.audio_toggle.clicked.connect(lambda: self._show_panel("audio"))
        toggle_layout.addWidget(self.audio_toggle)

        parent_layout.addWidget(toggle_widget)

    def _show_panel(self, panel_name):
        panels = {
            "main": (self.main_panel, self.main_toggle),
            "pose": (self.pose_panel, self.pose_toggle),
            "audio": (self.audio_panel, self.audio_toggle),
        }
        for name, (panel, toggle) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
            else:
                panel.hide()
                toggle.setChecked(False)

    def _create_main_section(self, parent_layout):
        self._create_individual_section(parent_layout)

        self.coords_groupbox = QGroupBox("Xarray coords")
        self.coords_groupbox_layout = QFormLayout()
        self.coords_groupbox_layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        self.coords_groupbox_layout.setContentsMargins(
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
        )
        self.coords_groupbox.setLayout(self.coords_groupbox_layout)
        parent_layout.addWidget(self.coords_groupbox)

        self.slot_groupbox = QGroupBox("Space/Cameras")
        slot_vbox = QVBoxLayout()
        slot_vbox.setSpacing(2)
        slot_vbox.setContentsMargins(4, 4, 4, 4)
        self.slot_layout = QHBoxLayout()
        self.slot_layout.setSpacing(5)
        self.slot_row2_layout = QHBoxLayout()
        self.slot_row2_layout.setSpacing(5)
        slot_vbox.addLayout(self.slot_layout)
        slot_vbox.addLayout(self.slot_row2_layout)
        self.slot_groupbox.setLayout(slot_vbox)
        self.slot_groupbox.hide()
        parent_layout.addWidget(self.slot_groupbox)

        self.panels_groupbox = QGroupBox("Plot panels")
        panels_vbox = QVBoxLayout()
        panels_vbox.setSpacing(2)
        panels_vbox.setContentsMargins(4, 4, 4, 4)
        self.panels_groupbox.setLayout(panels_vbox)

        for i in range(1, 6):
            row = QHBoxLayout()
            row.setSpacing(10)
            setattr(self, f"panels_row{i}_layout", row)
            panels_vbox.addLayout(row)

        parent_layout.addWidget(self.panels_groupbox)

        self.overlays_groupbox = QGroupBox("Overlays")
        # Two rows: row 1 holds the label-slot dropdowns, row 2 the
        # secondary toggles (Confidence, Envelope, ...).
        self.overlays_layout = QVBoxLayout()
        self.overlays_layout.setSpacing(2)
        self.overlays_layout.setContentsMargins(4, 4, 4, 4)
        self.overlays_row1_layout = QHBoxLayout()
        self.overlays_row1_layout.setSpacing(15)
        self.overlays_row2_layout = QHBoxLayout()
        self.overlays_row2_layout.setSpacing(15)
        self.overlays_layout.addLayout(self.overlays_row1_layout)
        self.overlays_layout.addLayout(self.overlays_row2_layout)
        self.overlays_groupbox.setLayout(self.overlays_layout)
        parent_layout.addWidget(self.overlays_groupbox)

        parent_layout.addStretch()

    def _create_individual_section(self, parent_layout):
        """Who the panels and the labels are about — shown above every plot's
        own settings, whatever backend the data came from.

        The actor combo is the dataset's individual dimension when it has one
        (created by ``DataWidget._create_combo_widget`` into this layout, so it
        keeps its dim name), and a plain list of the individuals the labels use
        when it does not.  The receiver combo turns a solo behaviour into a
        dyadic one: (actor, receiver) is the label subject.
        """
        self.individual_groupbox = QGroupBox("Individual")
        self.individual_layout = QFormLayout()
        self.individual_layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        self.individual_layout.setContentsMargins(
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
        )
        self.individual_groupbox.setLayout(self.individual_layout)

        self.individual_rec_combo = QComboBox()
        self.individual_rec_combo.setObjectName("individual_rec_combo")
        self.individual_rec_combo.setToolTip(
            "Receiver of the next label you place: who the behaviour is directed "
            "at, for dyadic interactions (e.g. one bird mounting another).\n"
            "Recorded on the label (individual_rec) and shown as a tag on it; "
            "None means a solo behaviour. It never filters what is shown."
        )
        self.individual_rec_combo.addItem("None", "")
        self.individual_layout.addRow("Receiver:", self.individual_rec_combo)

        parent_layout.addWidget(self.individual_groupbox)

    def _create_video_crop_section(self, parent_layout):
        """Display crop for the clicked camera view (borrowed into the video
        context of the right sidebar, like the pose group)."""
        self.videocrop_groupbox = QGroupBox("Crop")
        row = QHBoxLayout()
        row.setSpacing(5)
        row.setContentsMargins(4, 4, 4, 4)
        self.videocrop_groupbox.setLayout(row)

        self.crop_video_btn = QPushButton("Crop video")
        self.crop_video_btn.setToolTip(
            "Show only a rectangular region of this camera's video:\n"
            "click a corner on the video, drag, then click (or release) again.\n"
            "The crop is display-only and follows the camera across trials."
        )
        row.addWidget(self.crop_video_btn)

        self.uncrop_video_btn = QPushButton("Uncrop video")
        self.uncrop_video_btn.setToolTip("Show this camera's full video frame again.")
        row.addWidget(self.uncrop_video_btn)
        row.addStretch()

        parent_layout.addWidget(self.videocrop_groupbox)

    def _create_video_label_section(self, parent_layout):
        """Host for the video's label-name overlay control.

        The checkbox itself is owned by ``LabelsWidget`` (it edits label state)
        and is added here via ``LabelsWidget.attach_video_groupbox``.
        """
        self.videolabel_groupbox = QGroupBox("Label overlay")
        video_label_layout = QVBoxLayout()
        video_label_layout.setSpacing(4)
        video_label_layout.setContentsMargins(4, 4, 4, 4)
        self.videolabel_groupbox.setLayout(video_label_layout)

        parent_layout.addWidget(self.videolabel_groupbox)

    def _create_pose_section(self, parent_layout):
        self.pose_groupbox = QGroupBox("Pose overlay")
        pose_layout = QVBoxLayout()
        pose_layout.setSpacing(4)
        pose_layout.setContentsMargins(4, 4, 4, 4)
        self.pose_groupbox.setLayout(pose_layout)

        # ── Filter keypoints ──
        filter_box = QGroupBox("Filter keypoints")
        filter_row = QHBoxLayout()
        filter_row.setSpacing(5)
        filter_row.setContentsMargins(4, 4, 4, 4)
        filter_row.addWidget(QLabel("Filter below confidence:"))
        self.pose_hide_threshold_spin = QDoubleSpinBox()
        self.pose_hide_threshold_spin.setObjectName("pose_hide_threshold_spin")
        self.pose_hide_threshold_spin.setRange(0.0, 1.0)
        self.pose_hide_threshold_spin.setSingleStep(0.1)
        self.pose_hide_threshold_spin.setDecimals(1)
        self.pose_hide_threshold_spin.setFixedWidth(60)
        self.pose_hide_threshold_spin.setToolTip(
            "Hide pose markers below this confidence (0.0-1.0). A skeleton edge "
            "is also hidden on any frame where one of its two keypoints is hidden."
        )
        self.pose_hide_threshold_spin.setValue(self.app_state.pose_hide_threshold)
        filter_row.addWidget(self.pose_hide_threshold_spin)
        self.filter_keypoints_btn = QPushButton("Filter individual keypoints…")
        self.filter_keypoints_btn.setToolTip(
            "Open a popup to show/hide individual keypoints. Hidden keypoints "
            "also drop any skeleton edge touching them."
        )
        filter_row.addWidget(self.filter_keypoints_btn)
        filter_row.addStretch()
        filter_box.setLayout(filter_row)
        pose_layout.addWidget(filter_box)

        # ── Design ──
        design_box = QGroupBox("Design")
        design_vbox = QVBoxLayout()
        design_vbox.setSpacing(4)
        design_vbox.setContentsMargins(4, 4, 4, 4)
        grid = QGridLayout()
        grid.setSpacing(5)
        for col, header in enumerate(["", "Show", "Show text", "Size / width", "Base colour"]):
            grid.addWidget(QLabel(f"<b>{header}</b>" if header else ""), 0, col)

        # Points row
        grid.addWidget(QLabel("Points"), 1, 0)
        self.pose_show_keypoints_checkbox = QCheckBox()
        self.pose_show_keypoints_checkbox.setChecked(self.app_state.pose_show_keypoints)
        self.pose_show_keypoints_checkbox.setToolTip("Show/hide keypoint markers (skeleton edges are unaffected)")
        grid.addWidget(self.pose_show_keypoints_checkbox, 1, 1)
        self.pose_show_text_checkbox = QCheckBox()
        self.pose_show_text_checkbox.setChecked(self.app_state.pose_show_text)
        self.pose_show_text_checkbox.setToolTip("Show keypoint/individual labels on pose markers")
        grid.addWidget(self.pose_show_text_checkbox, 1, 2)
        self.pose_point_size_spin = QDoubleSpinBox()
        self.pose_point_size_spin.setObjectName("pose_point_size_spin")
        self.pose_point_size_spin.setRange(1.0, 50.0)
        self.pose_point_size_spin.setSingleStep(1.0)
        self.pose_point_size_spin.setDecimals(0)
        self.pose_point_size_spin.setFixedWidth(55)
        self.pose_point_size_spin.setValue(self.app_state.pose_point_size)
        grid.addWidget(self.pose_point_size_spin, 1, 3)
        points_color_cell = QHBoxLayout()
        points_color_cell.setSpacing(2)
        self.pose_points_color_btn = QPushButton()
        self.pose_points_color_btn.setObjectName("pose_points_color_btn")
        self.pose_points_color_btn.setFixedWidth(40)
        self.pose_points_color_btn.setToolTip("Uniform colour for all pose points")
        self.pose_points_color_btn.setIcon(_color_swatch_icon(self.app_state.pose_points_base_color or "#FF3333"))
        points_color_cell.addWidget(self.pose_points_color_btn)
        self.pose_points_use_base_checkbox = QCheckBox()
        self.pose_points_use_base_checkbox.setChecked(self.app_state.pose_points_use_base)
        self.pose_points_use_base_checkbox.setToolTip(
            "Checked: all points use the base colour.\nUnchecked: per-keypoint colours (turbo colormap)."
        )
        points_color_cell.addWidget(self.pose_points_use_base_checkbox)
        grid.addLayout(points_color_cell, 1, 4)

        # Skeleton row
        grid.addWidget(QLabel("Skeleton"), 2, 0)
        self.pose_show_skeleton_checkbox = QCheckBox()
        self.pose_show_skeleton_checkbox.setChecked(self.app_state.pose_show_skeleton)
        self.pose_show_skeleton_checkbox.setToolTip(
            "Draw skeleton edges (from the NWB ndx-pose Skeleton or the skeleton editor)"
        )
        grid.addWidget(self.pose_show_skeleton_checkbox, 2, 1)
        self.pose_skeleton_width_spin = QDoubleSpinBox()
        self.pose_skeleton_width_spin.setObjectName("pose_skeleton_width_spin")
        self.pose_skeleton_width_spin.setRange(0.5, 20.0)
        self.pose_skeleton_width_spin.setSingleStep(0.5)
        self.pose_skeleton_width_spin.setDecimals(1)
        self.pose_skeleton_width_spin.setFixedWidth(55)
        self.pose_skeleton_width_spin.setValue(self.app_state.pose_skeleton_width)
        grid.addWidget(self.pose_skeleton_width_spin, 2, 3)
        skeleton_color_cell = QHBoxLayout()
        skeleton_color_cell.setSpacing(2)
        self.pose_skeleton_color_btn = QPushButton()
        self.pose_skeleton_color_btn.setObjectName("pose_skeleton_color_btn")
        self.pose_skeleton_color_btn.setFixedWidth(40)
        self.pose_skeleton_color_btn.setToolTip("Uniform colour for all skeleton edges")
        self.pose_skeleton_color_btn.setIcon(_color_swatch_icon(self.app_state.skeleton_base_color or "#00CC66"))
        skeleton_color_cell.addWidget(self.pose_skeleton_color_btn)
        self.pose_skeleton_use_base_checkbox = QCheckBox()
        self.pose_skeleton_use_base_checkbox.setChecked(self.app_state.skeleton_use_base)
        self.pose_skeleton_use_base_checkbox.setToolTip(
            "Checked: all edges use the base colour.\n"
            "Unchecked: per-edge custom colours from the skeleton editor / NWB."
        )
        skeleton_color_cell.addWidget(self.pose_skeleton_use_base_checkbox)
        grid.addLayout(skeleton_color_cell, 2, 4)
        grid.setColumnStretch(5, 1)
        design_vbox.addLayout(grid)

        color_by_row = QHBoxLayout()
        color_by_row.addWidget(QLabel("Colour by"))
        self.pose_color_by_combo = QComboBox()
        self.pose_color_by_combo.setObjectName("pose_color_by_combo")
        self.pose_color_by_combo.addItem("Keypoint", COLOR_BY_KEYPOINT)
        self.pose_color_by_combo.addItem("Individual", COLOR_BY_INDIVIDUAL)
        self.pose_color_by_combo.setToolTip(
            "Keypoint: one colour per body part, the same on every animal.\n"
            "Individual: one colour per animal, the same for all its keypoints.\n\n"
            "Text labels carry whichever axis the colours are not saying.\n"
            "Also styles the keypoint labelling canvas."
        )
        index = self.pose_color_by_combo.findData(self.app_state.pose_color_by)
        self.pose_color_by_combo.setCurrentIndex(index if index >= 0 else 0)
        color_by_row.addWidget(self.pose_color_by_combo)
        color_by_row.addStretch()
        design_vbox.addLayout(color_by_row)

        text_size_row = QHBoxLayout()
        text_size_row.addWidget(QLabel("Text size"))
        self.pose_text_size_spin = QDoubleSpinBox()
        self.pose_text_size_spin.setObjectName("pose_text_size_spin")
        self.pose_text_size_spin.setRange(4.0, 72.0)
        self.pose_text_size_spin.setSingleStep(1.0)
        self.pose_text_size_spin.setDecimals(0)
        self.pose_text_size_spin.setFixedWidth(55)
        self.pose_text_size_spin.setValue(self.app_state.pose_text_size)
        text_size_row.addWidget(self.pose_text_size_spin)
        text_size_row.addStretch()
        design_vbox.addLayout(text_size_row)
        design_box.setLayout(design_vbox)
        pose_layout.addWidget(design_box)

        # ── Actions ──
        self.create_skeleton_btn = QPushButton("Create / edit skeleton…")
        self.create_skeleton_btn.setToolTip(
            "Open an editor to draw skeleton connections on real pose data:\n"
            "drag between keypoints to connect, then assign color categories."
        )
        pose_layout.addWidget(self.create_skeleton_btn)

        self.label_keypoints_btn = QPushButton("Label keypoints…")
        self.label_keypoints_btn.setToolTip(
            "Label a handful of frames by clicking the video, then let a point\n"
            "tracker fill the rest. No training and no GPU required."
        )
        pose_layout.addWidget(self.label_keypoints_btn)

        self.pose_match_btn = QPushButton("Match Pose ↔ Video")
        self.pose_match_btn.setToolTip(
            "Open dialog to match NWB PoseEstimation containers to video cameras.\n"
            "Required when the NWB has multiple pose containers (e.g. LeftCamera, RightCamera)."
        )
        self.pose_match_btn.clicked.connect(self._on_pose_match_clicked)
        pose_layout.addWidget(self.pose_match_btn)
        pose_layout.addStretch()

        self.pose_groupbox.hide()
        parent_layout.addWidget(self.pose_groupbox, stretch=1)
        self._create_bbox_section(parent_layout)

    def _create_bbox_section(self, parent_layout):
        """The video's controls when the pose file holds bounding boxes.

        Shown instead of the Pose section (``RightContextPanel``): a box has no
        skeleton and no keypoints, and its colour says which individual it is
        — so the controls are the confidence filter, the text, the line width
        and one colour swatch per individual. The threshold/text/width widgets
        drive the same state as the Pose section's, kept in step by
        ``DataWidget``.
        """
        self.bbox_groupbox = QGroupBox("Bounding boxes")
        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)
        self.bbox_groupbox.setLayout(layout)

        grid = QGridLayout()
        grid.setSpacing(5)
        grid.addWidget(QLabel("Filter below confidence"), 0, 0)
        self.bbox_hide_threshold_spin = QDoubleSpinBox()
        self.bbox_hide_threshold_spin.setObjectName("bbox_hide_threshold_spin")
        self.bbox_hide_threshold_spin.setRange(0.0, 1.0)
        self.bbox_hide_threshold_spin.setSingleStep(0.1)
        self.bbox_hide_threshold_spin.setDecimals(2)
        self.bbox_hide_threshold_spin.setFixedWidth(60)
        self.bbox_hide_threshold_spin.setToolTip("Hide boxes whose detector confidence is below this (0.0-1.0).")
        self.bbox_hide_threshold_spin.setValue(self.app_state.pose_hide_threshold)
        grid.addWidget(self.bbox_hide_threshold_spin, 0, 1)
        grid.addWidget(QLabel("Show text"), 1, 0)
        self.bbox_show_text_checkbox = QCheckBox()
        self.bbox_show_text_checkbox.setChecked(self.app_state.pose_show_text)
        self.bbox_show_text_checkbox.setToolTip("Write the individual's name at each box")
        grid.addWidget(self.bbox_show_text_checkbox, 1, 1)
        grid.addWidget(QLabel("Text size"), 2, 0)
        self.bbox_text_size_spin = QDoubleSpinBox()
        self.bbox_text_size_spin.setObjectName("bbox_text_size_spin")
        self.bbox_text_size_spin.setRange(4.0, 72.0)
        self.bbox_text_size_spin.setSingleStep(1.0)
        self.bbox_text_size_spin.setDecimals(0)
        self.bbox_text_size_spin.setFixedWidth(55)
        self.bbox_text_size_spin.setValue(self.app_state.pose_text_size)
        grid.addWidget(self.bbox_text_size_spin, 2, 1)
        grid.addWidget(QLabel("Line width"), 3, 0)
        self.bbox_line_width_spin = QDoubleSpinBox()
        self.bbox_line_width_spin.setObjectName("bbox_line_width_spin")
        self.bbox_line_width_spin.setRange(0.5, 20.0)
        self.bbox_line_width_spin.setSingleStep(0.5)
        self.bbox_line_width_spin.setDecimals(1)
        self.bbox_line_width_spin.setFixedWidth(55)
        self.bbox_line_width_spin.setValue(self.app_state.pose_skeleton_width)
        grid.addWidget(self.bbox_line_width_spin, 3, 1)
        grid.setColumnStretch(2, 1)
        layout.addLayout(grid)

        colors_box = QGroupBox("Colour per individual")
        colors_box.setToolTip("An individual keeps its colour on every camera, whether or not the others are shown.")
        self.bbox_colors_form = QFormLayout()
        self.bbox_colors_form.setSpacing(4)
        self.bbox_colors_form.setContentsMargins(4, 4, 4, 4)
        colors_box.setLayout(self.bbox_colors_form)
        layout.addWidget(colors_box)
        #: name -> swatch button, rebuilt by ``DataWidget.populate_bbox_colors``.
        self.bbox_color_buttons: dict[str, QPushButton] = {}
        layout.addStretch()

        self.bbox_groupbox.hide()
        parent_layout.addWidget(self.bbox_groupbox, stretch=1)

    def _on_pose_match_clicked(self):
        from .dialog_pose_video_matcher import PoseVideoMatcherDialog

        sio = getattr(self.app_state, "nwb_alignment", None)

        # Saved mapping from local_settings, then NWB acquisition, then detection
        pose_keys = list(getattr(self.app_state, "nwb_pose_keys", None) or [])
        if not pose_keys and sio:
            pose_keys = sio.pose_keys
        if not pose_keys:
            nwb_local = getattr(self.app_state, "nwb_local", None)
            if nwb_local:
                pose_keys = _detect_nwb_pose_keys(nwb_local) or []

        if not pose_keys:
            from ethograph.gui.notify import notify

            notify("No PoseEstimation containers found in the NWB file.", "warning")
            return

        dialog = PoseVideoMatcherDialog(
            video_folder=getattr(self.app_state, "video_folder", "") or "",
            pose_folder=getattr(self.app_state, "pose_folder", "") or "",
            trial_ids=getattr(self.app_state, "trials", []),
            parent=self,
            pose_items=pose_keys,
        )
        if dialog.exec_():
            mapping = dialog.get_mapping()
            ordered_keys = [pose_key for _, pose_key in mapping]
            self.app_state.nwb_pose_keys = ordered_keys
            if self._update_pose_callback:
                self._update_pose_callback()

    def _create_audio_section(self, parent_layout):
        group = QGroupBox("Energy envelope")
        self.energy_group = group  # exposed for the context-sensitive sidebar
        grid = QGridLayout()
        group.setLayout(grid)

        grid.addWidget(QLabel("Energy metric:"), 0, 0)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(self.ENERGY_DISPLAY_NAMES.values())
        grid.addWidget(self.metric_combo, 0, 1, 1, 2)

        self.energy_configure_btn = QPushButton("Configure...")
        self.energy_configure_btn.setToolTip("Open parameter editor for selected energy metric")
        grid.addWidget(self.energy_configure_btn, 1, 0, 1, 3)

        parent_layout.addWidget(group)
        parent_layout.addStretch()

        self._restore_energy_selections()

    def _restore_energy_selections(self):
        metric = self.app_state.get_with_default("energy_metric")
        display = self.ENERGY_DISPLAY_NAMES.get(metric, "SOS lowpass envelope")
        self.metric_combo.setCurrentText(display)

    def energy_display_to_key(self, display_text: str) -> str:
        for key, val in self.ENERGY_DISPLAY_NAMES.items():
            if val == display_text:
                return key
        return "energy_lowpass"


class DataWidget(QWidget):
    """Orchestrator widget — loads data, manages selections, updates plots."""

    def __init__(
        self,
        shell,
        app_state,
        meta_widget,
        io_widget,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.shell = shell
        layout = QFormLayout()
        layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        layout.setContentsMargins(
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
        )
        self.setLayout(layout)
        self.app_state = app_state
        self.meta_widget = meta_widget
        self.io_widget = io_widget
        self.plot_container = None
        self.labels_widget = None
        self.plot_settings_widget = None
        self.ephys_widget = None
        self.audio_player = None
        self.video_path = None
        self.audio_path = None
        self.space_plots: list[SpacePlot] = []
        self.active_space_plot: SpacePlot | None = None
        self._space_signals_connected = False
        self._space_plot_autocreated = False
        self.radial_plots: list[RadialPlot] = []
        self.active_radial_plot: RadialPlot | None = None
        self._radial_signals_connected = False
        #: While True, trial changes skip video/extra-camera/pose loading —
        #: set by the label-frames dialog while it captures panel screenshots
        #: (the plots need the trial's data, never its video). Whoever sets it
        #: reloads the media after clearing it.
        self.suppress_video_load = False

        self.combos = {}
        #: The widget occupying each combo's row in the coords form. Kept so a
        #: row can be hidden when a new catalog lacks that dimension — removing
        #: it would delete widgets the right sidebar and MetaWidget hold too.
        self._combo_row_fields: dict[str, QWidget] = {}
        #: The form each combo's row lives in — coords for most dims, the
        #: individual group for the individual one.
        self._combo_row_layouts: dict[str, QFormLayout] = {}
        self.all_checkboxes = {}
        self.controls = []
        self._keypoint_names: list[str] = []

        self.source_software = None
        self.file_path = None

        self.video_mgr = VideoManager(shell.video_area, app_state)
        self.video_mgr.set_frame_changed_callback(self._on_primary_frame_changed)
        self.video_mgr.set_video_reloaded_callback(self.update_pose)
        shell.video_area.camera_view_removed.connect(self._on_camera_view_removed)

        # Session-basis auto-follow: after the marker settles in another
        # trial's span, switch to it (debounced — a trial switch can mean a
        # ~2 s video decoder respawn, so never per marker move).
        self._marker_follow_timer = QTimer(self)
        self._marker_follow_timer.setSingleShot(True)
        self._marker_follow_timer.setInterval(300)
        self._marker_follow_timer.timeout.connect(self._follow_marker_trial)
        self._follow_pending_time: float | None = None
        self._follow_pending_trial = None
        self.pose_mgr: PoseDisplayManager | None = None  # created after set_data_panel
        self._keypoint_labelling_dialog = None
        self.app_state.audio_video_sync = None
        self.catalog = None  # DataCatalog set after load

    def set_data_panel(self, panel: DataPanel):
        self.data_panel = panel
        self.coords_groupbox = panel.coords_groupbox
        self.coords_groupbox_layout = panel.coords_groupbox_layout
        self.individual_groupbox = panel.individual_groupbox
        self.individual_layout = panel.individual_layout
        self.individual_rec_combo = panel.individual_rec_combo
        panel.individual_rec_combo.currentIndexChanged.connect(self._on_receiver_changed)
        self.slot_groupbox = panel.slot_groupbox
        self.slot_layout = panel.slot_layout
        self.slot_row2_layout = panel.slot_row2_layout
        self.panels_groupbox = panel.panels_groupbox
        self.panels_row1_layout = panel.panels_row1_layout
        self.panels_row2_layout = panel.panels_row2_layout
        self.panels_row3_layout = panel.panels_row3_layout
        self.panels_row4_layout = panel.panels_row4_layout
        self.panels_row5_layout = panel.panels_row5_layout
        self.overlays_groupbox = panel.overlays_groupbox
        self.overlays_layout = panel.overlays_layout
        self.overlays_row1_layout = panel.overlays_row1_layout
        self.overlays_row2_layout = panel.overlays_row2_layout
        self.pose_groupbox = panel.pose_groupbox
        self.pose_hide_threshold_spin = panel.pose_hide_threshold_spin
        self.pose_show_text_checkbox = panel.pose_show_text_checkbox
        self.pose_point_size_spin = panel.pose_point_size_spin
        self.pose_text_size_spin = panel.pose_text_size_spin
        self.pose_show_skeleton_checkbox = panel.pose_show_skeleton_checkbox
        self.pose_skeleton_width_spin = panel.pose_skeleton_width_spin
        self.pose_skeleton_color_btn = panel.pose_skeleton_color_btn
        self.pose_skeleton_use_base_checkbox = panel.pose_skeleton_use_base_checkbox
        self.pose_points_color_btn = panel.pose_points_color_btn
        self.pose_points_use_base_checkbox = panel.pose_points_use_base_checkbox
        self.pose_color_by_combo = panel.pose_color_by_combo
        self.create_skeleton_btn = panel.create_skeleton_btn
        self.label_keypoints_btn = panel.label_keypoints_btn
        self.pose_show_keypoints_checkbox = panel.pose_show_keypoints_checkbox
        self.filter_keypoints_btn = panel.filter_keypoints_btn
        self.bbox_groupbox = panel.bbox_groupbox
        self.bbox_hide_threshold_spin = panel.bbox_hide_threshold_spin
        self.bbox_show_text_checkbox = panel.bbox_show_text_checkbox
        self.bbox_text_size_spin = panel.bbox_text_size_spin
        self.bbox_line_width_spin = panel.bbox_line_width_spin
        self.bbox_colors_form = panel.bbox_colors_form
        self.bbox_color_buttons = panel.bbox_color_buttons

        self.pose_mgr = PoseDisplayManager(self.shell.video_area, self.app_state, self.video_mgr, self)
        self.app_state.keypoints_changed.connect(self.populate_keypoints)

        panel.pose_hide_threshold_spin.valueChanged.connect(self._on_pose_hide_threshold_changed)
        panel.pose_show_keypoints_checkbox.stateChanged.connect(self._on_pose_show_keypoints_toggled)
        panel.filter_keypoints_btn.clicked.connect(self._on_filter_keypoints_clicked)
        panel.pose_show_text_checkbox.stateChanged.connect(self._on_pose_text_toggled)
        panel.pose_point_size_spin.valueChanged.connect(self._on_pose_point_size_changed)
        panel.pose_text_size_spin.valueChanged.connect(self._on_pose_text_size_changed)
        panel.pose_show_skeleton_checkbox.stateChanged.connect(self._on_pose_show_skeleton_toggled)
        panel.pose_skeleton_width_spin.valueChanged.connect(self._on_pose_skeleton_width_changed)
        panel.pose_skeleton_color_btn.clicked.connect(self._on_skeleton_color_clicked)
        panel.pose_skeleton_use_base_checkbox.stateChanged.connect(self._on_skeleton_use_base_toggled)
        panel.pose_points_color_btn.clicked.connect(self._on_points_color_clicked)
        panel.pose_points_use_base_checkbox.stateChanged.connect(self._on_points_use_base_toggled)
        panel.pose_color_by_combo.currentIndexChanged.connect(self._on_pose_color_by_changed)
        panel.create_skeleton_btn.clicked.connect(self._on_create_skeleton_clicked)
        panel.label_keypoints_btn.clicked.connect(self.open_keypoint_labelling)
        panel.bbox_hide_threshold_spin.valueChanged.connect(self._on_bbox_hide_threshold_changed)
        panel.bbox_show_text_checkbox.stateChanged.connect(self._on_bbox_text_toggled)
        panel.bbox_text_size_spin.valueChanged.connect(self._on_bbox_text_size_changed)
        panel.bbox_line_width_spin.valueChanged.connect(self._on_bbox_line_width_changed)
        panel._update_pose_callback = self.update_pose

        self.videocrop_groupbox = panel.videocrop_groupbox
        self.videolabel_groupbox = panel.videolabel_groupbox
        panel.crop_video_btn.clicked.connect(self._on_crop_video_clicked)
        panel.uncrop_video_btn.clicked.connect(self._on_uncrop_video_clicked)

        panel.energy_configure_btn.clicked.connect(self._open_energy_params)

    # ------------------------------------------------------------------
    # Video display crop
    # ------------------------------------------------------------------

    def _crop_target_view(self):
        """The camera view the crop buttons act on: the active video panel,
        falling back to the primary view."""
        manager = getattr(self.meta_widget, "active_panels", None)
        reg = getattr(manager, "active", None)
        if reg is not None and reg.kind == "video" and getattr(reg.widget, "has_video", False):
            return reg.widget
        primary = self.video_mgr.primary_view
        return primary if primary.has_video else None

    def _on_crop_video_clicked(self):
        view = self._crop_target_view()
        if view is None:
            notify("No video is loaded.", "warning")
            return
        if view.crop_selection_active:
            view.cancel_crop_selection()
            notify("Crop selection cancelled.")
            return
        if view.start_crop_selection(lambda rect, v=view: self._on_crop_selected(v, rect)):
            notify("Click a corner of the region on the video, drag, then click again to crop.")

    def _on_crop_selected(self, view, rect):
        camera = getattr(view, "camera_name", None)
        if rect is None or not camera:
            notify("Crop selection cancelled.", "warning")
            return
        self.video_mgr.set_camera_crop(camera, rect)
        notify(
            f"Cropped {camera} to {rect[2] - rect[0]}×{rect[3] - rect[1]} px "
            f"at ({rect[0]}, {rect[1]})-({rect[2]}, {rect[3]})."
        )

    def _on_uncrop_video_clicked(self):
        view = self._crop_target_view()
        if view is not None and view.crop_selection_active:
            view.cancel_crop_selection()
        camera = getattr(view, "camera_name", None) if view is not None else None
        if not camera or self.video_mgr.camera_crop(camera) is None:
            notify("No crop is set for this camera.", "warning")
            return
        self.video_mgr.clear_camera_crop(camera)
        notify(f"Removed crop from {camera}.")

    def populate_keypoints(self, keypoint_names: list[str]) -> None:
        self._keypoint_names = [str(n) for n in keypoint_names]
        self.pose_groupbox.show()

    def get_hidden_keypoints(self) -> set[str]:
        return set(self.app_state.pose_hidden_keypoints)

    def _on_pose_show_keypoints_toggled(self, state: int):
        self.app_state.pose_show_keypoints = self.pose_show_keypoints_checkbox.isChecked()
        self.pose_mgr.apply_pose_style()

    def _on_filter_keypoints_clicked(self):
        names = self._keypoint_names or (self.pose_mgr.all_keypoints if self.pose_mgr else [])
        if not names:
            notify("No pose keypoints loaded yet.", "warning")
            return
        dialog = KeypointFilterDialog(names, self.get_hidden_keypoints(), parent=self)
        dialog.hidden_changed.connect(self._on_hidden_keypoints_changed)
        dialog.exec_()

    def _on_hidden_keypoints_changed(self, hidden: set):
        self.app_state.pose_hidden_keypoints = sorted(str(n) for n in hidden)
        self.update_pose()

    def _on_pose_hide_threshold_changed(self, value: float):
        self.app_state.pose_hide_threshold = value
        _sync_spin(self.bbox_hide_threshold_spin, value)
        self.update_pose()

    # ── Bounding boxes section: the same state as the Pose section's widgets ──

    def _on_bbox_hide_threshold_changed(self, value: float):
        self.app_state.pose_hide_threshold = value
        _sync_spin(self.pose_hide_threshold_spin, value)
        self.update_pose()

    def _on_bbox_text_toggled(self, state: int):
        checked = self.bbox_show_text_checkbox.isChecked()
        self.app_state.pose_show_text = checked
        _sync_check(self.pose_show_text_checkbox, checked)
        self.pose_mgr.apply_pose_style()

    def _on_bbox_text_size_changed(self, value: float):
        self.app_state.pose_text_size = value
        _sync_spin(self.pose_text_size_spin, value)
        self.pose_mgr.apply_pose_style()

    def _on_bbox_line_width_changed(self, value: float):
        self.app_state.pose_skeleton_width = value
        _sync_spin(self.pose_skeleton_width_spin, value)
        self.pose_mgr.apply_skeleton_style()

    def populate_bbox_colors(self, individuals: list[str]) -> None:
        """One swatch per individual, in the dataset's order, showing the
        colour the overlay gives that name (the user's pick, else the palette)."""
        form = self.bbox_colors_form
        while form.rowCount():
            form.removeRow(0)
        self.bbox_color_buttons.clear()
        overrides = dict(self.app_state.pose_individual_colors or {})
        colors = individual_color_map([str(n) for n in individuals], overrides)
        for name, rgba in colors.items():
            button = QPushButton()
            button.setObjectName(f"bbox_color_btn_{name}")
            button.setFixedWidth(40)
            button.setIcon(_color_swatch_icon(_rgba_to_hex(rgba)))
            button.setToolTip(f"Colour of the boxes of {name}")
            button.clicked.connect(lambda _checked=False, n=name: self._on_bbox_color_clicked(n))
            form.addRow(QLabel(name), button)
            self.bbox_color_buttons[name] = button

    def _on_bbox_color_clicked(self, name: str):
        overrides = dict(self.app_state.pose_individual_colors or {})
        palette = individual_color_map(list(self.bbox_color_buttons), overrides)
        current = overrides.get(name) or _rgba_to_hex(palette[name])
        color = QColorDialog.getColor(QColor(current), self, f"Colour of {name}")
        if not color.isValid():
            return
        overrides[name] = color.name().upper()
        self.app_state.pose_individual_colors = overrides
        self.bbox_color_buttons[name].setIcon(_color_swatch_icon(overrides[name]))
        self.pose_mgr.refresh_skeleton()

    def _on_pose_text_toggled(self, state: int):
        self.app_state.pose_show_text = self.pose_show_text_checkbox.isChecked()
        _sync_check(self.bbox_show_text_checkbox, self.app_state.pose_show_text)
        self.pose_mgr.apply_pose_style()

    def _on_pose_color_by_changed(self, _index: int):
        """Switch the colour axis — for the overlay and any open labelling canvas.

        One setting styles both, so the animal that is blue on the canvas is the
        animal that is blue on the pose overlay.
        """
        self.app_state.pose_color_by = self.pose_color_by_combo.currentData()
        self.update_pose()
        dialog = getattr(self, "_keypoint_labelling_dialog", None)
        if dialog is not None:
            dialog.apply_color_by()

    def _on_pose_point_size_changed(self, value: float):
        self.app_state.pose_point_size = value
        self.pose_mgr.apply_pose_style()

    def _on_pose_text_size_changed(self, value: float):
        self.app_state.pose_text_size = value
        _sync_spin(self.bbox_text_size_spin, value)
        self.pose_mgr.apply_pose_style()

    def _on_pose_show_skeleton_toggled(self, state: int):
        self.app_state.pose_show_skeleton = self.pose_show_skeleton_checkbox.isChecked()
        self.update_pose()

    def _on_pose_skeleton_width_changed(self, value: float):
        self.app_state.pose_skeleton_width = value
        _sync_spin(self.bbox_line_width_spin, value)
        self.pose_mgr.apply_skeleton_style()

    def _on_skeleton_color_clicked(self):
        current = self.app_state.skeleton_base_color or "#00CC66"
        color = QColorDialog.getColor(QColor(current), self, "Skeleton base colour")
        if not color.isValid():
            return
        self.app_state.skeleton_base_color = color.name().upper()
        self.pose_skeleton_color_btn.setIcon(_color_swatch_icon(self.app_state.skeleton_base_color))
        self.pose_mgr.refresh_skeleton()

    def _on_skeleton_use_base_toggled(self, state: int):
        self.app_state.skeleton_use_base = self.pose_skeleton_use_base_checkbox.isChecked()
        self.pose_mgr.refresh_skeleton()

    def _on_points_color_clicked(self):
        current = self.app_state.pose_points_base_color or "#FF3333"
        color = QColorDialog.getColor(QColor(current), self, "Points base colour")
        if not color.isValid():
            return
        self.app_state.pose_points_base_color = color.name().upper()
        self.pose_points_color_btn.setIcon(_color_swatch_icon(self.app_state.pose_points_base_color))
        self.pose_mgr.refresh_skeleton()

    def _on_points_use_base_toggled(self, state: int):
        self.app_state.pose_points_use_base = self.pose_points_use_base_checkbox.isChecked()
        self.pose_mgr.refresh_skeleton()

    def _on_create_skeleton_clicked(self):
        from .dialog_skeleton_editor import SkeletonEditorDialog

        data = self.pose_mgr.primary_pose_for_editor()
        if data is None:
            notify("No pose data available for the current camera/trial.", "warning")
            return
        keypoints, positions = data
        if positions.shape[0] == 0:
            notify("Pose data has no frames to edit.", "warning")
            return
        existing = getattr(self.app_state, "skeleton_config_override", None)
        dialog = SkeletonEditorDialog(keypoints, positions, existing_config=existing, parent=self)
        if dialog.exec_():
            self.app_state.skeleton_config_override = dialog.get_config()
            self.pose_show_skeleton_checkbox.setChecked(True)
            self.update_pose()

    def open_keypoint_labelling(self):
        """Open (or raise) the keypoint labelling dialog.

        Non-modal and kept as a single instance: the user labels while
        navigating frames, and both the Tools menu and the Pose sidebar button
        route here.
        """
        from .dialog_pose_labelling import PoseLabellingDialog

        existing = getattr(self, "_keypoint_labelling_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return existing

        dialog = PoseLabellingDialog(self, parent=self.shell)
        dialog.finished.connect(lambda _=0: setattr(self, "_keypoint_labelling_dialog", None))
        self._keypoint_labelling_dialog = dialog
        dialog.show()
        return dialog

    def open_box_labelling(self):
        """Open (or raise) the box labelling dialog — OCTRON's workflow over every open camera."""
        from .dialog_box_labelling import BoxLabellingDialog

        existing = getattr(self, "_box_labelling_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return existing

        dialog = BoxLabellingDialog(self, parent=self.shell)
        dialog.finished.connect(lambda _=0: setattr(self, "_box_labelling_dialog", None))
        self._box_labelling_dialog = dialog
        dialog.show()
        return dialog

    def open_pose_refinement(self):
        """Open (or raise) the pose refinement dialog.

        Non-modal single instance, like the labelling dialog: the user corrects
        an imported pose file while navigating trials, and each trial's
        ``_refined`` copy is flushed as they move on.
        """
        from .dialog_pose_refinement import PoseRefinementDialog

        existing = getattr(self, "_pose_refinement_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return existing

        dialog = PoseRefinementDialog(self, parent=self.shell)
        dialog.finished.connect(lambda _=0: setattr(self, "_pose_refinement_dialog", None))
        self._pose_refinement_dialog = dialog
        dialog.show()
        return dialog

    def set_references(
        self,
        plot_container,
        labels_widget,
        plot_settings_widget,
        navigation_widget,
        changepoints_widget=None,
        ephys_widget=None,
        layout_mgr=None,
        trials_widget=None,
    ):
        self.plot_container = plot_container
        self.labels_widget = labels_widget
        self.plot_settings_widget = plot_settings_widget
        self.navigation_widget = navigation_widget
        self.changepoints_widget = changepoints_widget
        self.ephys_widget = ephys_widget
        self.layout_mgr = layout_mgr
        self.trials_widget = trials_widget

        if trials_widget is not None:
            trials_widget.trials_filtered.connect(self._on_trials_filtered)

        plot_container.time_marker_updated.connect(self._on_time_marker_updated)

        if changepoints_widget is not None:
            changepoints_widget.request_plot_update.connect(self._on_plot_update_request)

        plot_container.plot_changed.connect(self._on_feature_plot_type_changed)

    def _on_plot_update_request(self):
        if not self.app_state.ready or not self.plot_container:
            return
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def _on_feature_plot_type_changed(self, plot_type: str):
        self._update_sort_button_state()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _cleanup_load_state(self):
        dt = getattr(self.app_state, "dt", None)
        if dt is not None:
            dt.close()
            self.app_state.dt = None
        self.app_state.ds = None
        self.app_state.data_loader = None
        self.app_state.source_collection = None
        self.app_state._all_labels_df = None
        self.app_state.clear_label_history()
        self.app_state.labels_confidence_ds = None
        self.catalog = None
        self.app_state.ready = False

    def _cancel_load(self, reason: str):
        notify_dialog(reason, "warning", "Load cancelled", self)
        self._cleanup_load_state()

    def on_load_clicked(self):
        if not self.app_state.nc_file_path:
            notify_dialog(
                "Please select a data file (.nc, .nwb, .npz) or folder",
                "warning",
                "Load cancelled",
                self,
            )
            return

        nc_file_path = self.io_widget.get_nc_file_path()

        # Phases 1-4: pure computation — app_state is untouched.
        try:
            ctx = self._phase_load_data(nc_file_path)
            self._phase_infer_media(ctx)
            self._phase_prepare_dataset(ctx)
            self._phase_validate(ctx)
        except _LoadError as e:
            self._cancel_load(str(e))
            return

        # Phase 5: single commit point — mutate app_state.
        try:
            self._apply_to_state(ctx)
        except _LoadError as e:
            self._cancel_load(str(e))
            return

        # Phase 6: UI setup (app_state is now consistent).
        self._setup_ui_after_load(ctx)

    # ------------------------------------------------------------------
    # Loading phases
    # ------------------------------------------------------------------

    def _phase_load_data(self, nc_file_path: str) -> _LoadContext:
        """Phase 1: Load dataset from disk."""
        labels_path = self.io_widget.get_import_labels_path()
        if labels_path and not Path(labels_path).exists():
            raise _LoadError(
                f"'Import labels' is checked but the labels file was not found:\n{labels_path}\n\n"
                "Uncheck 'Import labels', or point it at the correct file via "
                "File ▸ Import labels…"
            )
        try:
            result = load_features_dataset(
                nc_file_path,
                progress_callback=getattr(self.app_state, "_progress_callback", None),
                metadata_path=self.app_state.metadata_path,
                alignment_path=getattr(self.app_state, "nwb_file_path", None),
                labels_path=labels_path,
            )
        except (OSError, ValueError, KeyError) as e:
            logger.exception("load_features_dataset failed")
            raise _LoadError(f"Failed to load feature dataset: {type(e).__name__}: {e}") from e

        video_folder_override = None
        if result.nwb_video_folder and not self.app_state.video_folder:
            video_folder_override = str(result.nwb_video_folder)

        return _LoadContext(
            result=result,
            nc_file_path=nc_file_path,
            catalog=result.catalog,
            dt=result.dt,
            all_labels_df=result.all_labels_df,
            nwb_local=result.nwb_local,
            video_folder_override=video_folder_override,
        )

    def _phase_infer_media(self, ctx: _LoadContext) -> None:
        """Phase 2: Detect video/audio/pose/ephys availability."""
        result = ctx.result
        sio = result.nwb_alignment

        ctx.has_audio = bool(sio.mics) if sio else False

        # NWB-embedded ephys — detect directly from the NWB file
        if sio:
            eseries = sio.electrical_series()
            if eseries:
                first = eseries[0]
                ctx.nwb_ephys_display = f"{first['name']} (NWB)"
                ctx.nwb_ephys_entry = (first["path"], "0", 0)

        ctx.has_neo = bool(self.app_state.ephys_path) or bool(ctx.nwb_ephys_entry)
        ctx.has_neurons = bool(self.app_state.neurons_path)

    def _phase_prepare_dataset(self, ctx: _LoadContext) -> None:
        """Phase 3: Downsample, resolve trials, build first ds + DataLoader."""
        downsample_factor = self.io_widget.get_downsample_factor()
        if downsample_factor is not None and ctx.dt is not None:
            ctx.dt = eto.downsample_trialtree(ctx.dt, downsample_factor)
            ctx.downsample_factor = downsample_factor
            logger.info("Downsampled data by factor %d", downsample_factor)

        ctx.trials = sorted(ctx.result.trial_ids)
        if ctx.dt is not None:
            ctx.ds = ctx.dt.trial(ctx.trials[0])
        else:
            ctx.ds = xr.Dataset()

        store = ctx.result.data_loader
        if store is None:
            from ethograph.io.catalog import XarrayLoader, catalog_from_xarray

            cat = catalog_from_xarray(ctx.ds, ctx.dt)
            store = XarrayLoader(ctx.ds, cat)
        ctx.data_loader = store

    def _phase_validate(self, ctx: _LoadContext) -> None:
        """Phase 4: Validate media files for the first trial."""
        video_folder = ctx.video_folder_override or self.app_state.video_folder
        missing = self._validate_media_files(
            nwb_alignment=ctx.result.nwb_alignment,
            first_trial=ctx.trials[0],
            video_folder=video_folder,
            audio_folder=self.app_state.audio_folder,
            pose_folder=self.app_state.pose_folder,
        )
        if missing:
            raise _LoadError("Missing media files for first trial:\n" + "\n".join(missing))

    def _apply_to_state(self, ctx: _LoadContext) -> None:
        """Phase 5: Single commit — mutate app_state from accumulated context."""
        # Suppress signal handlers during bulk mutation so that handlers
        # checking app_state.ready don't react to partially-set state.
        self.app_state.ready = False

        result = ctx.result

        self.app_state.dt = ctx.dt
        self.app_state.metadata_df = result.metadata_df
        self.app_state.metadata_path = result.metadata_path
        self.catalog = ctx.catalog

        if ctx.video_folder_override:
            self.app_state.video_folder = ctx.video_folder_override

        self.app_state.trial_conditions = ctx.catalog.trial_conditions
        self.app_state.source_collection = result.source_collection
        self.app_state.nwb_alignment = result.nwb_alignment

        self.app_state.has_audio = ctx.has_audio

        if ctx.nwb_local:
            self.app_state.nwb_local = ctx.nwb_local

        # NWB-embedded ephys
        if ctx.nwb_ephys_display and ctx.nwb_ephys_entry:
            if not self.app_state.ephys_path:
                self.app_state.ephys_path = ctx.nwb_ephys_entry[0]
            self.app_state.ephys_source_map[ctx.nwb_ephys_display] = ctx.nwb_ephys_entry
            self.app_state.ephys_stream_sel = ctx.nwb_ephys_display

        self.app_state.has_neo = ctx.has_neo
        self.app_state.has_neurons = ctx.has_neurons

        # Expand ephys streams (mutates io_widget + app_state.ephys_source_map)
        if self.app_state.ephys_path:
            try:
                self.io_widget._expand_ephys_with_streams(
                    self.app_state.ephys_path,
                    ctx.ds,
                )
            except (OSError, ValueError, KeyError) as e:
                logger.exception("Ephys stream expansion failed")
                raise _LoadError(f"Failed to load ephys features: {type(e).__name__}: {e}") from e

        self.io_widget.disable_downsample_controls()
        self.app_state.downsample_factor_used = ctx.downsample_factor

        self.app_state._all_labels_df = ctx.all_labels_df
        self.app_state.clear_label_history()
        self.app_state._labels_file_path = ctx.result.labels_file_path
        self.app_state.trials = ctx.trials if ctx.trials else [1]
        self.app_state.ds = ctx.ds
        # Wrapped once, here, so every consumer holds the same object whether or
        # not the user ever opens the console (DerivedLoader forwards everything
        # it does not define to the real loader).
        self.app_state.data_loader = DerivedLoader(ctx.data_loader) if ctx.data_loader is not None else None
        self._install_display_offset_provider()

        # Set trials_sel early so _expand_mics_with_channels / get_media
        # can resolve filenames during UI creation.
        trial = getattr(self.app_state, "trials_sel", None)
        try:
            is_nan = np.isnan(trial)
        except (TypeError, ValueError):
            is_nan = False
        if not trial or is_nan or trial not in self.app_state.trials:
            self.app_state.trials_sel = self.app_state.trials[0]

    def _setup_ui_after_load(self, ctx: _LoadContext) -> None:
        """Phase 6: Create controls, restore defaults, enable UI."""
        self._create_trial_controls()

        # TrialsWidget._apply_filters may have overwritten app_state.trials
        # with an empty list (e.g. when metadata_df mismatches trial_ids).
        # Restore from the authoritative LoadResult.
        if not self.app_state.trials and ctx.trials:
            self.app_state.trials = ctx.trials

        self._set_controls_enabled(True)
        self.app_state.ready = True

        self._restore_or_set_defaults()
        # The restored individual decides who is available as a receiver.
        self._populate_receiver_combo()

        self.io_widget.on_load_complete()
        self.labels_widget.refresh_mapping_for_data_dir(Path(ctx.nc_file_path).parent)
        self.changepoints_widget.setEnabled(True)
        self.plot_settings_widget.set_enabled_state()
        if self.ephys_widget:
            self.ephys_widget.setEnabled(True)
            self.ephys_widget.populate_ephys_default_path()

        self.meta_widget.configure_layout_for_data()

        # Re-apply sidebar cap after load-triggered dock/layout rearrangement.
        if getattr(self, "layout_mgr", None) is not None:
            QTimer.singleShot(
                0,
                lambda: self.layout_mgr.set_sidebar_default_width(self.meta_widget, SIDEBAR_AFTER_LOAD_WIDTH_RATIO),
            )

        self.update_trials_combo()
        self._load_trial_with_fallback()
        self._disable_empty_panels()
        self._apply_video_dock_default()
        # After the trial created the camera views, and after the primary dock
        # knows whether it is shown: both are cells of the grid.
        self.meta_widget.arrange_camera_grid_if_default()

        if self.navigation_widget:
            self.navigation_widget.set_mappings(self.labels_widget._mappings)
            self.navigation_widget.refresh_after_load()

        self.view_mode_combo.show()

    # ------------------------------------------------------------------
    # Trials combo
    # ------------------------------------------------------------------

    def update_trials_combo(self) -> None:
        if not self.app_state.ready:
            return

        combo = self.navigation_widget.trials_combo
        combo.blockSignals(True)
        combo.clear()

        trial_status = self._collect_trial_status()

        for trial in self.app_state.trials:
            combo.addItem(str(trial))
            index = combo.count() - 1
            # Green: every label of the trial is manual or curated; red: some
            # are still a model's unreviewed output (labels/curation.py).
            is_curated = trial_status.get(str(trial), True)
            bg_color = QColor(144, 238, 144) if is_curated else QColor(255, 182, 193)
            combo.setItemData(index, bg_color, Qt.BackgroundRole)
            text_color = QColor(0, 100, 0) if is_curated else QColor(139, 0, 0)
            combo.setItemData(index, text_color, Qt.ForegroundRole)

        combo.setCurrentText(str(self.app_state.trials_sel))
        combo.blockSignals(False)
        self.navigation_widget._sync_trials_combo_color()

    def _collect_trial_status(self) -> Dict[str, bool]:
        return self.app_state.trial_curation_status()

    def _on_trials_filtered(self, filtered_trials: list) -> None:
        """Handle TrialsWidget filter changes."""
        if not self.app_state.ready:
            return
        self.update_trials_combo()
        if self.app_state.trials_sel not in filtered_trials and filtered_trials:
            self.app_state.set_key_sel("trials", filtered_trials[0])
            self.app_state.trial_changed.emit()
        # Label / sequence navigation walks instances across trials — the ones
        # the table just hid must drop out of that walk too.
        self.navigation_widget.on_trials_filtered()
        self.update_main_plot()

    # ------------------------------------------------------------------
    # Create controls (populates main panel groupboxes)
    # ------------------------------------------------------------------

    def refresh_trials_confidence(self) -> None:
        """Refresh the Trials tab with current metadata."""
        if getattr(self, "trials_widget", None) is None:
            return
        mdf = self.app_state.metadata_df
        if mdf is None or mdf.empty:
            mdf = pd.DataFrame({"trial": self.app_state.trials})
        self.trials_widget.setup(mdf)

    def _create_trial_controls(self):
        self.io_widget.create_device_controls(self.catalog)
        self.navigation_widget.setup_trial_conditions(self.catalog)
        self.navigation_widget.set_data_widget(self)

        if getattr(self, "trials_widget", None) is not None:
            mdf = (
                self.app_state.metadata_df
                if self.app_state.metadata_df is not None
                else pd.DataFrame({"trial": self.app_state.trials})
            )
            self.trials_widget.setup(mdf)
            self.refresh_trials_confidence()

        for combo_name, combo_spec in self.catalog.combos.items():
            if not combo_spec.values:
                continue
            self._create_combo_widget(combo_name, list(combo_spec.values))

        self._create_colors_combo()
        self.refresh_individual_choices()

        # Restore camera combos
        cameras = self.app_state.nwb_alignment.cameras
        slot_layout = self.slot_layout

        # Slot 1: Layers / Space Plot toggle
        self.space_view_combo = QComboBox()
        self.space_view_combo.setObjectName("space_view_combo")
        view_items = ["Layers", "Space Plot"]
        self.space_view_combo.addItems(view_items)
        self.space_view_combo.currentTextChanged.connect(self._on_space_view_changed)
        self.controls.append(self.space_view_combo)
        slot_layout.addWidget(self.space_view_combo)

        # Slot 2: Camera selection (first camera)
        self.primary_camera_combo = QComboBox()
        self.primary_camera_combo.setObjectName("primary_camera_combo")
        self.primary_camera_combo.addItems([str(c) for c in cameras])
        self.primary_camera_combo.setItemDelegate(ElidedDelegate(parent=self.primary_camera_combo))
        self.primary_camera_combo.currentTextChanged.connect(self._on_primary_camera_changed)
        self.controls.append(self.primary_camera_combo)
        slot_layout.addWidget(self.primary_camera_combo)

        if len(cameras) > 1:
            cam_names = [str(c) for c in cameras]
            n_extra = len(cameras) - 1  # no cap — one slot per non-primary camera
            self._extra_camera_combos: list[QComboBox] = []
            for i in range(n_extra):
                combo = QComboBox()
                combo.setObjectName(f"extra_camera_combo_{i}")
                combo.addItems(["None"] + cam_names)
                combo.setItemDelegate(ElidedDelegate(parent=combo))
                combo.setCurrentIndex(0)
                combo.currentTextChanged.connect(lambda _text, idx=i: self._on_extra_camera_combo_changed(idx))
                self._extra_camera_combos.append(combo)
                self.controls.append(combo)
                slot_layout.addWidget(combo)

        if "keypoint" in self.app_state.ds.coords:
            keypoint_names = strip_common_prefix([str(k) for k in self.app_state.ds.coords["keypoint"].values])
            self.populate_keypoints(keypoint_names)

        slot_layout.addStretch()
        # The Space/Cameras group itself is hidden (cameras are added via
        # drag-drop now), but its combos stay alive as headless state driven
        # programmatically elsewhere (pose_render, shortcuts, PCA auto-switch).
        self.slot_groupbox.hide()

        self._setup_panel_controls()

    def _setup_panel_controls(self):
        """Apply initial panel visibility from data availability and build the
        per-stream controls (mic / view-mode / neo / neural combos).

        Panels are layout instances: shown when their data exists, removed via
        a panel's ✕ button, re-added via the add-panel popup (➕ / Shift+N).
        There is no per-plot-type on/off toggle state.
        """
        self._audio_row_widgets = []
        pc = self.plot_container

        has_audio = bool(self.app_state.has_audio or self.app_state.audio_path)

        # Expand mics → channels first so the default audio panels can pin to the
        # first channel of each mic (needs audio_source_map / audio_mic_channels).
        mic_names = self.catalog.mics if (self.app_state.has_audio and self.catalog) else []
        expanded = self._expand_mics_with_channels(mic_names) if self.app_state.has_audio else []

        if has_audio:
            self._create_default_audio_panels(mic_names)
        else:
            pc.set_audiotrace_visible(False)
            pc.set_spectrogram_visible(False)
        if self.catalog and self.catalog.features and not pc.line_plots:
            pc.add_lineplot()
        # Neo + Phy trace panels are heavy; they are NOT shown automatically.
        # The user adds them on demand from the "➕ Add panel" popup. Just make
        # sure the Phy loader's stream is resolvable.
        if self.app_state.ephys_source_map:
            self._ensure_default_ephys_stream()

        # Seed the default mic/channel selection. There is no visible global
        # "Mic:" combo — playback follows the last-clicked audio panel and each
        # panel's own "Channel:" combo (see docs/source/advanced/playback.md).
        if self.app_state.has_audio and expanded:
            self.app_state.set_key_sel("mics", expanded[0])

        # Row 3: feature view controls
        self.panels_row3_layout.addWidget(QLabel("View:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.setObjectName("view_mode_combo")
        self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
        self.view_mode_combo.hide()
        self.controls.append(self.view_mode_combo)
        self.panels_row3_layout.addWidget(self.view_mode_combo)
        self.sort_channels_btn = QPushButton("Sort channels")
        self.sort_channels_btn.setToolTip("Sort heatmap channels by activity in selected label interval")
        self.sort_channels_btn.setEnabled(False)
        self.sort_channels_btn.clicked.connect(self._on_sort_channels_clicked)
        self.panels_row3_layout.addWidget(self.sort_channels_btn)
        self.panels_row3_layout.addStretch()

        # Row 5: neural view combo
        self._neural_view_label = QLabel("View:")
        self.neural_view_combo = QComboBox()
        self.neural_view_combo.setObjectName("neural_view_combo")
        self.neural_view_combo.addItems(["Multi Trace", "Raster"])
        self.neural_view_combo.currentTextChanged.connect(self._on_neural_view_changed)
        self.panels_row5_layout.addWidget(self._neural_view_label)
        self.panels_row5_layout.addWidget(self.neural_view_combo)
        self._neural_view_label.hide()
        self.neural_view_combo.hide()
        self.panels_row5_layout.addStretch()

        if self.app_state.has_neurons and self.ephys_widget:
            self._neural_view_label.show()
            self.neural_view_combo.show()
            # Wire the Phy loader/source so it renders instantly when the user
            # adds the Phy viewer from the popup — but keep the panel hidden.
            self.ephys_widget.configure_ephys_trace_plot()

        self.video_mgr.set_audio_row_widgets(self._audio_row_widgets)

        # Overlays row 1 — label branches are shown/hidden via their own
        # checkboxes in the Labels panel (branch 0 = Full, 1 = Top1, 2 =
        # Top2, fixed). This row carries the one Predictions toggle: it both
        # occupies whichever of Top1/Top2 isn't already used by a shown
        # branch, and gates the dotted prediction-confidence curve on every
        # feature plot (`PanelStateMixin.show_predictions_enabled`) — there is
        # no separate per-panel predictions checkbox.
        row1 = self.overlays_row1_layout
        row1.setSpacing(2)

        self.show_predictions_overlay_checkbox = QCheckBox("Predictions")
        self.show_predictions_overlay_checkbox.setChecked(False)
        self.show_predictions_overlay_checkbox.setToolTip(
            "Show imported predictions: as a top strip on the labels track (fills Top1, or "
            "Top2 if Top1 is used by a branch) and as the dotted confidence curve on feature plots"
        )
        self.show_predictions_overlay_checkbox.stateChanged.connect(self._on_show_predictions_overlay_changed)
        row1.addWidget(self.show_predictions_overlay_checkbox)

        row1.addStretch()

        # Overlays row 2 — secondary scalar overlays.
        row2 = self.overlays_row2_layout

        self.show_confidence_checkbox = QCheckBox("Confidence")
        self.show_confidence_checkbox.setChecked(True)
        self.show_confidence_checkbox.stateChanged.connect(self._update_confidence_overlay)
        row2.addWidget(self.show_confidence_checkbox)

        self.show_envelope_checkbox = QCheckBox("Show Envelope")
        self.show_envelope_checkbox.setChecked(False)
        self.show_envelope_checkbox.stateChanged.connect(self._on_envelope_overlay_changed)
        self.show_envelope_checkbox.hide()
        row2.addWidget(self.show_envelope_checkbox)

        row2.addStretch()

        self._set_controls_enabled(False)

    # ------------------------------------------------------------------
    # Combo / checkbox handlers
    # ------------------------------------------------------------------

    def refresh_lineplot(self):
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def _update_confidence_overlay(self):
        if not self.app_state.ready or self.plot_container is None:
            return
        if not self.show_confidence_checkbox.isChecked():
            self.plot_container.hide_confidence_plot()
            return
        host = self.plot_container.get_current_plot()
        if hasattr(host, "show_predictions_enabled") and not host.show_predictions_enabled():
            self.plot_container.hide_confidence_plot()
            return
        trial = self.app_state.trials_sel
        store = getattr(self.app_state, "pred_store", None)
        if store is not None:
            trial_confidence = store.get_confidence(
                trial, self.app_state.dt, individual=self.app_state.selected_individual()
            )
            if trial_confidence is not None:
                time_coord = self.app_state.time_coord.values
                n = min(len(trial_confidence), len(time_coord))
                self.plot_container.show_confidence_plot(trial_confidence[:n], time_coord[:n])
                return
        label_ds = getattr(self.app_state, "labels_confidence_ds", None)
        if label_ds is not None and "labels_confidence" in getattr(label_ds, "data_vars", {}):
            ds_kwargs = self.app_state.get_ds_kwargs()
            try:
                label_confidence, _ = eto.sel_valid(label_ds.labels_confidence, ds_kwargs)
            except (KeyError, AttributeError, ValueError):
                label_confidence = None
            if label_confidence is not None and len(label_confidence) > 0:
                self.plot_container.show_confidence_plot(label_confidence)
                return
        self.plot_container.hide_confidence_plot()

    # ------------------------------------------------------------------
    # Predictions overlay toggle
    # ------------------------------------------------------------------
    # Label branches themselves are shown/hidden via their own checkboxes in
    # the Labels panel (fixed position: branch 0 = Full, 1 = Top1, 2 = Top2).

    def _on_show_predictions_overlay_changed(self, qt_state):
        """User toggled the Predictions checkbox — the single control for both
        the labels-track interval strip and every feature plot's dotted
        prediction-confidence curve. Persist + redraw both."""
        self.app_state._show_predictions_overlay = Qt.CheckState(qt_state) == Qt.Checked
        if self.app_state.ready:
            self.update_label_plot()
            self._update_confidence_overlay()
        if self.labels_widget is not None:
            self.labels_widget.refresh_labels_shapes_layer()

    def cycle_neural_view(self):
        if not hasattr(self, "neural_view_combo") or not self.neural_view_combo.isVisible():
            return
        next_index = (self.neural_view_combo.currentIndex() + 1) % self.neural_view_combo.count()
        self.neural_view_combo.setCurrentIndex(next_index)

    def _on_neural_view_changed(self, mode: str):
        if not self.app_state.ready or not mode:
            return
        if self.ephys_widget:
            self.ephys_widget.set_neural_view(mode)

    # ------------------------------------------------------------------
    # Neo trace panels (one instance per stream/modality)
    # ------------------------------------------------------------------

    def _kilosort_params(self) -> dict | None:
        return getattr(self.ephys_widget, "_kilosort_params", None) if self.ephys_widget else None

    def _neo_stream_meta(self, filepath, stream_id) -> tuple[int, float] | None:
        """(n_channels, rate) for a stream, cached so opening the add-panel
        popup doesn't re-parse the ephys header for every stream each time."""
        cache = self.__dict__.setdefault("_stream_meta_cache", {})
        key = (str(filepath), str(stream_id))
        if key not in cache:
            try:
                loader = load_ephys(filepath, str(stream_id))
                cache[key] = (int(loader.n_channels), float(loader.rate))
            except Exception:
                cache[key] = None
        return cache[key]

    def _stream_matches_kilosort(self, filepath, stream_id) -> bool:
        """True if a Neo stream is the probe stream shown in the Phy trace."""
        ks = self._kilosort_params()
        if not ks:
            return False
        meta = self._neo_stream_meta(filepath, stream_id)
        if meta is None:
            return False
        n_ch, rate = meta
        return n_ch == ks.get("n_channels_dat", 0) and abs(rate - ks.get("sample_rate", 0)) < 1.0

    def neo_stream_names(self, exclude_kilosort: bool = True) -> list[str]:
        """Display names of Neo streams (modalities), optionally excluding the
        stream that feeds the Phy trace (matches kilosort params)."""
        source_map = getattr(self.app_state, "ephys_source_map", {}) or {}
        names = []
        for name, entry in source_map.items():
            if not entry or len(entry) < 2:
                continue
            filepath, stream_id = entry[0], entry[1]
            if exclude_kilosort and self._stream_matches_kilosort(filepath, stream_id):
                continue
            names.append(name)
        return names

    def neo_stream_channel_count(self, stream_name: str) -> int:
        source_map = getattr(self.app_state, "ephys_source_map", {}) or {}
        entry = source_map.get(stream_name)
        if entry is None:
            return 0
        filepath, stream_id, _ch = entry
        meta = self._neo_stream_meta(filepath, stream_id)
        return meta[0] if meta is not None else 0

    def _ensure_default_ephys_stream(self) -> None:
        """Ensure ephys_stream_sel points at a valid stream so the Phy trace
        loader (get_ephys_source) resolves — even when every stream is neo."""
        source_map = getattr(self.app_state, "ephys_source_map", {}) or {}
        if source_map and self.app_state.ephys_stream_sel not in source_map:
            self.app_state.ephys_stream_sel = next(iter(source_map))

    def add_neo_panel(self, stream_name: str, channels: list[int] | None = None):
        return self.plot_container.add_panel("neo", stream_name=stream_name, channels=channels)

    def configure_neo_plot(self, plot) -> None:
        """Load the plot's stream and render it (called back from add_panel)."""
        stream_name = getattr(plot, "neo_stream_name", None)
        source_map = getattr(self.app_state, "ephys_source_map", {}) or {}
        entry = source_map.get(stream_name)
        if entry is None:
            return
        filepath, stream_id, channel_idx = entry
        try:
            loader = load_ephys(filepath, str(stream_id))
        except Exception:
            return
        neo_starting_time = float(getattr(loader, "starting_time", 0.0) or 0.0)
        plot.set_loader(loader, channel_idx or 0)
        plot.set_source(FileSource("neo", loader, start_time=neo_starting_time))
        channels = getattr(plot, "neo_channels", None)
        if channels is not None:
            plot.set_custom_channel_set(np.asarray(channels, dtype=int))
        if self.app_state.ready:
            xmin, xmax = self.plot_container.get_current_xlim()
            plot.update_plot_content(xmin, xmax)
            plot.auto_channel_spacing()
            plot.auto_gain()
            plot.autoscale()

    def refresh_neo_panels(self) -> None:
        """Re-render every Neo trace instance (e.g. on trial change)."""
        if not self.plot_container:
            return
        for plot in self.plot_container.neo_trace_plots:
            self.configure_neo_plot(plot)

    def on_kilosort_loaded(self) -> None:
        """After Kilosort loads, the Phy trace owns the probe stream — drop any
        Neo panel that now duplicates it, and keep the Phy loader stream valid."""
        self._ensure_default_ephys_stream()
        pc = self.plot_container
        if pc is None:
            return
        source_map = getattr(self.app_state, "ephys_source_map", {}) or {}
        for plot in list(pc.neo_trace_plots):
            entry = source_map.get(getattr(plot, "neo_stream_name", None))
            if entry and self._stream_matches_kilosort(entry[0], entry[1]):
                pc.remove_panel(plot)

    def _create_default_audio_panels(self, mic_names: list) -> None:
        """Data-availability default for audio: an audio trace + spectrogram per mic.

        One mic → a single global-following pair (unpinned), unchanged behaviour.
        Multiple mics (e.g. several audio files dropped) → one pair per mic, each
        pinned to that mic's first channel, so every audio file is visualised at
        once. Saved layouts override this (``apply_layout_state`` rebuilds panels).
        """
        pc = self.plot_container
        if len(mic_names) > 1:
            for mic in mic_names:
                channels = self.app_state.audio_mic_channels.get(str(mic))
                key = channels[0] if channels else None
                pc.add_audio_panel("audiotrace", mic_name=key)
                pc.add_audio_panel("spectrogram", mic_name=key)
        else:
            pc.set_audiotrace_visible(True)
            pc.set_spectrogram_visible(True)
        pc.update_audio_panels()

    def _get_audio_channel_count(self, audio_path):
        try:
            from audioio import AudioLoader

            from ..io.audio_extract import resolve_audio_path

            with AudioLoader(resolve_audio_path(audio_path)) as loader:
                if loader.shape is None:
                    return 1
                return (
                    loader.channels
                    if hasattr(loader, "channels")
                    else (loader.shape[1] if len(loader.shape) > 1 else 1)
                )
        except (ImportError, OSError, ValueError):
            return 1

    def _expand_mics_with_channels(self, mic_labels):
        self.app_state.audio_source_map.clear()
        self.app_state.audio_mic_channels.clear()
        expanded_items = []
        audio_folder = self.app_state.audio_folder
        trial_id = getattr(self.app_state, "trials_sel", None)
        if trial_id is None and self.app_state.trials:
            trial_id = self.app_state.trials[0]

        # No dt/backend gate here: file resolution below goes through the
        # alignment (get_media / resolve_media_path), which answers for
        # pynapple- and NWB-backed sessions too — a pure-media drag & drop
        # loads its tmp alignment via pynapple, and gating on dt left its
        # multichannel audio as a single un-expanded mic.
        if not audio_folder:
            for mic in mic_labels:
                display_name = str(mic)
                self.app_state.audio_source_map[display_name] = (str(mic), 0)
                self.app_state.audio_mic_channels[str(mic)] = [display_name]
                expanded_items.append(display_name)
            return expanded_items

        for mic_label in mic_labels:
            mic_file = self.app_state.nwb_alignment.get_media(trial_id, "audio", str(mic_label))
            if mic_file:
                audio_path = os.path.join(audio_folder, mic_file)
            else:
                # Stream-based alignments (e.g. drag & drop tmp alignment) keep
                # file references only in ImageSeries — no trials-table columns.
                audio_path = self.app_state.nwb_alignment.resolve_media_path(
                    trial_id, "audio", device=str(mic_label), fallback_folder=audio_folder
                )
                if not audio_path:
                    continue
                mic_file = Path(audio_path).name
            try:
                n_channels = self._get_audio_channel_count(audio_path)
                if n_channels > 1:
                    channel_keys = []
                    for ch in range(n_channels):
                        display_name = f"{mic_file} (Ch {ch + 1})"
                        self.app_state.audio_source_map[display_name] = (mic_file, ch)
                        channel_keys.append(display_name)
                        expanded_items.append(display_name)
                    self.app_state.audio_mic_channels[str(mic_label)] = channel_keys
                else:
                    self.app_state.audio_source_map[mic_file] = (mic_file, 0)
                    self.app_state.audio_mic_channels[str(mic_label)] = [mic_file]
                    expanded_items.append(mic_file)
            except (OSError, ValueError):
                self.app_state.audio_source_map[mic_file] = (mic_file, 0)
                self.app_state.audio_mic_channels[str(mic_label)] = [mic_file]
                expanded_items.append(mic_file)
        return expanded_items

    def refresh_audio_sources_for_trial(self, ds):
        """Rebuild ``audio_source_map`` for the current trial (media paths can be
        per-trial) and keep the ``mics_sel`` selection valid."""
        new_items = self.app_state.nwb_alignment.mics
        if not new_items:
            return
        expanded = self._expand_mics_with_channels(np.array(new_items, dtype=str))
        if expanded and getattr(self.app_state, "mics_sel", None) not in expanded:
            self.app_state.set_key_sel("mics", expanded[0])

    def _update_device_sels_for_trial(self, ds):
        cameras = self.app_state.nwb_alignment.cameras

        primary = getattr(self, "primary_camera_combo", None)
        if primary is not None and cameras:
            prev_index = primary.currentIndex()
            primary.blockSignals(True)
            primary.clear()
            primary.addItems(cameras)
            if prev_index < primary.count():
                primary.setCurrentIndex(prev_index)
            else:
                primary.setCurrentIndex(0)
            primary.blockSignals(False)
            self.app_state.primary_camera = primary.currentText()

        self._update_extra_camera_combo_items(cameras)

    def _is_autoscale_on(self) -> bool:
        return self.plot_settings_widget is not None and self.plot_settings_widget.autoscale_checkbox.isChecked()

    def show_neural_panel(self):
        """Show the neural panel (trace or raster, per the neural view combo)."""
        if not self.plot_container:
            return
        mode = self.neural_view_combo.currentText() if hasattr(self, "neural_view_combo") else "Multi Trace"
        self.plot_container.set_neural_panel_mode("raster" if mode == "Raster" else "trace")
        if self._is_autoscale_on():
            self.plot_container.ephys_trace_plot.vb.enableAutoRange(x=False, y=True)

    def _update_view_mode_items(self, feature_sel: str):
        """Update view_mode_combo items based on available data.

        Feature view controls what the bottom (feature) panel shows.
        Audio/Ephys heatmap modes compute envelope from raw data.
        """
        current_text = self.view_mode_combo.currentText()
        self.view_mode_combo.blockSignals(True)
        self.view_mode_combo.clear()

        has_audio = self.app_state.has_audio or bool(self.app_state.audio_path)
        items = ["LinePlot", "Heatmap"]
        if has_audio:
            items.append("Heatmap (Audio)")
        if self.app_state.has_neo:
            items.append("Heatmap (Ephys)")
        self.view_mode_combo.addItems(items)

        idx = self.view_mode_combo.findText(current_text)
        if idx >= 0:
            self.view_mode_combo.setCurrentIndex(idx)
        self.view_mode_combo.blockSignals(False)

    def _on_view_mode_changed(self, mode: str):
        if not self.app_state.ready or not self.plot_container:
            return

        self.app_state.feature_view_mode = mode

        if mode.startswith("Heatmap"):
            self.plot_container.switch_to_heatmap()
            for heatmap in self.plot_container.heatmap_plots:
                heatmap._clear_buffer()
        else:
            self.plot_container.switch_to_lineplot()

        self._update_sort_button_state()

        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def _update_sort_button_state(self):
        btn = getattr(self, "sort_channels_btn", None)
        if btn is None:
            return
        enabled = self.plot_container is not None and self.plot_container.is_heatmap()
        btn.setEnabled(enabled)

    def _on_sort_channels_clicked(self):
        if not self.labels_widget or not self.plot_container:
            return

        idx = self.labels_widget.current_labels_pos
        if idx is None:
            notify("Select a label interval first", "warning")
            return

        df = self.app_state.label_intervals
        if df is None or idx not in df.index:
            notify("Select a label interval first", "warning")
            return

        onset_s, offset_s, _ = get_interval_bounds(df, idx)

        heatmap = self.plot_container.heatmap_plot
        if heatmap is None:
            notify("Open a heatmap panel first", "warning")
            return
        data = heatmap.get_normalized_data_for_range(onset_s, offset_s)
        if data is None or data.size == 0:
            notify("No heatmap data available for the selected interval", "warning")
            return

        channel_sums = np.nansum(np.abs(data), axis=0)
        sort_order = np.argsort(channel_sums)[::-1]
        heatmap.set_sort_order(sort_order)

    def _configure_ephys_trace_plot(self):
        if self.ephys_widget:
            self.ephys_widget.configure_ephys_trace_plot()

    def _on_envelope_overlay_changed(self):
        if not self.plot_container:
            return
        if self.show_envelope_checkbox.isChecked():
            self.plot_container.show_envelope_overlay()
        else:
            self.plot_container.hide_envelope_overlay()

    def _open_energy_params(self):
        from .dialog_function_params import open_function_params_dialog

        key = self.data_panel.energy_display_to_key(
            self.data_panel.metric_combo.currentText(),
        )
        if key:
            # Parent to the shell, not data_panel (which is now a hidden shell).
            result = open_function_params_dialog(key, self.app_state, parent=self.shell)
            if result is not None:
                self._on_energy_apply()

    def _on_energy_apply(self):
        metric_key = self.data_panel.energy_display_to_key(
            self.data_panel.metric_combo.currentText(),
        )
        self.app_state.energy_metric = metric_key

        if not self.plot_container:
            return

        from .dialog_busy_progress import BusyProgressDialog

        self.plot_container.hide_envelope_overlay()
        dialog = BusyProgressDialog("Computing energy envelope (all channels)...", parent=self.shell)
        feature_name, error = dialog.execute_blocking(self._compute_envelope_feature)

        if hasattr(self, "show_envelope_checkbox") and not self.show_envelope_checkbox.isChecked():
            self.show_envelope_checkbox.setChecked(True)  # triggers overlay redraw
        else:
            self.plot_container.show_envelope_overlay()

        if error is None and feature_name:
            notify(
                f'Envelope saved as feature "{feature_name}" — add it via ➕ Add panel (heatmap shows all channels).',
                "info",
            )

    def _current_mic_device(self) -> str | None:
        """Mic device label for the current channel-expanded mic selection."""
        mic_channels = self.app_state.audio_mic_channels
        mics_sel = getattr(self.app_state, "mics_sel", None)
        for device, keys in mic_channels.items():
            if mics_sel in keys:
                return device
        return next(iter(mic_channels), None)

    def _resolve_trial_audio_path(self, trial_id, mic_label: str) -> str | None:
        align = self.app_state.nwb_alignment
        if align is None:
            return getattr(self.app_state, "audio_path", None)
        audio_folder = self.app_state.audio_folder
        mic_file = align.get_media(trial_id, "audio", str(mic_label))
        if mic_file and audio_folder:
            path = os.path.join(audio_folder, mic_file)
            if os.path.exists(path):
                return path
        return align.resolve_media_path(trial_id, "audio", device=str(mic_label), fallback_folder=audio_folder)

    def _compute_envelope_feature(self) -> str | None:
        """Compute the energy envelope for ALL channels of the current mic and
        store it as a per-trial feature named after the metric, so it can be
        added via ➕ Add panel (e.g. as a heatmap across channels)."""
        app_state = self.app_state
        dt = getattr(app_state, "dt", None)
        store = app_state.data_loader
        if dt is None or store is None or not hasattr(store, "update_ds") or getattr(dt, "_is_continuous", False):
            notify("Saving the envelope as a feature requires an xarray (.nc) dataset.", "warning")
            return None

        mic_label = self._current_mic_device()
        if mic_label is None:
            return None

        metric = app_state.get_with_default("energy_metric")
        feature_name = ENERGY_DISPLAY_NAMES.get(metric, metric)
        # Per-metric dim names: envelope metrics have different output rates,
        # so sharing one time dim across features would make xarray raise an
        # AlignmentError on conflicting sizes.
        time_dim = f"time_{metric}"
        channel_dim = f"{metric}_channel"

        env_by_path: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        computed = 0
        for trial_id in app_state.trials:
            audio_path = self._resolve_trial_audio_path(trial_id, mic_label)
            if not audio_path:
                continue
            if audio_path not in env_by_path:
                loader = SharedAudioCache.get_loader(audio_path)
                if loader is None:
                    continue
                raw = np.asarray(loader[0 : len(loader)], dtype=np.float64)
                env_by_path[audio_path] = compute_energy_envelope_multichannel(raw, loader.rate, metric, app_state)
            env_time, envelopes = env_by_path[audio_path]
            da = xr.DataArray(
                envelopes,
                dims=(time_dim, channel_dim),
                coords={
                    time_dim: env_time,
                    channel_dim: [f"ch{i + 1}" for i in range(envelopes.shape[1])],
                },
                attrs={"ylabel": feature_name, "mic": str(mic_label)},
            )

            def _assign(ds: xr.Dataset, da: xr.DataArray = da) -> xr.Dataset:
                ds = ds.drop_vars([feature_name, time_dim, channel_dim], errors="ignore")
                return ds.assign({feature_name: da})

            dt.update_trial(trial_id, _assign)
            computed += 1

        if not computed:
            notify(f"No audio found for mic {mic_label} — envelope feature not created.", "warning")
            return None

        trials_sel = app_state.trials_sel
        if trials_sel is not None:
            app_state.ds = dt.trial(trials_sel)
            store.update_ds(app_state.ds)

        self._register_feature(feature_name)
        return feature_name

    def load_keypoint_dataset(self, ds: xr.Dataset) -> bool:
        """Serve features from *ds* instead of the current dataset.

        The keypoints **are** a dataset — a movement poses set whose ``keypoint``
        and ``individual`` are ordinary dimensions — so they replace the feature
        data rather than being grafted onto whatever was open. Merging was the
        previous approach and it was wrong twice over: it *crashes* on any trial
        that already has a variable named ``individual`` (the labels table has
        one), since xarray cannot tell whether the merged name is a coordinate
        or a data variable; and the grafted dims reached no combo, so the
        features could not be reduced to something a plot can draw.

        Only the data layer moves. This is deliberately **not** a file load:
        that re-runs the whole startup pipeline — media re-resolution, dock
        registration, ``apply_saved_panel_layout`` and several deferred timers —
        over a window that is already built, and re-applying a ``restoreState``
        blob across live pygfx canvases is a native crash. Media, alignment,
        panels and layout are all left exactly as they are, so nothing here can
        reach them.
        """
        from ethograph.io.catalog import XarrayLoader, catalog_from_xarray
        from ethograph.io.trialtree import TrialTree

        app_state = self.app_state
        trial_id = app_state.trials_sel if app_state.trials_sel is not None else 1
        try:
            # validate=False: a poses-only tree is a legal session but not the
            # shape the full validator expects.
            dt = TrialTree.from_datasets([ds.assign_attrs(trial=trial_id)], validate=False)
        except (ValueError, KeyError) as e:
            notify(f"Could not build a dataset from the keypoints: {e}", "error")
            return False

        app_state.dt = dt
        app_state.ds = dt.trial(trial_id)
        if not app_state.trials:
            app_state.trials = [trial_id]
        app_state.trials_sel = trial_id
        self.catalog = catalog_from_xarray(app_state.ds, dt)
        app_state.data_loader = DerivedLoader(XarrayLoader(app_state.ds, self.catalog))
        self._install_display_offset_provider()

        self._rebuild_coord_controls()
        # The panels' own selections were valid for the previous data; one that
        # names an individual or keypoint this dataset does not have raises
        # KeyError out of `.sel()` the moment the panel next renders.
        for plot in [*self.plot_container.line_plots, *self.plot_container.heatmap_plots]:
            plot.resync_selections()
        if "keypoint" in app_state.ds.coords:
            self.populate_keypoints([str(k) for k in app_state.ds.coords["keypoint"].values])
        self.plot_container.schedule_labels_redraw()
        return True

    def _rebuild_coord_controls(self) -> None:
        """Point the "Xarray coords" combos at the catalog now serving features.

        These are built once, when a session loads, so a catalog arriving later
        has dimensions with no control at all — and a feature whose dims cannot
        be pinned is one ``sel_valid`` refuses to return, which reaches the user
        as a permanently empty panel.

        Updated in place and **never torn down**. This form holds widgets this
        widget does not own — ``MetaWidget`` inserts "Feature plot type:" at row
        0, and the right sidebar borrows the whole group box — and
        ``QFormLayout.removeRow`` *deletes* what it removes, leaving live
        references to deleted C++ objects behind. A row for a dimension the new
        catalog does not have is hidden, not removed, for the same reason.
        """
        wanted = {name: [str(v) for v in spec.values] for name, spec in self.catalog.combos.items() if spec.values}

        for key, values in wanted.items():
            combo = self.combos.get(key)
            if combo is None:
                self._create_combo_widget(key, values)
            else:
                self._refill_combo(combo, values)
            self._set_combo_row_visible(key, True)
            self.app_state.set_key_sel(key, get_combo_value(self.combos[key]))

        for key in set(self.combos) - set(wanted) - {"colors"}:
            self._set_combo_row_visible(key, False)

        self._populate_colors_combo(
            self.combos["colors"],
            list(self.catalog.features),
            rgb_filter=self._colors_rgb_checkbox.isChecked(),
        )
        self.refresh_individual_choices()

    @staticmethod
    def _refill_combo(combo, values: list[str]) -> None:
        """Replace a combo's items, keeping the selection when it still exists.

        Signals stay blocked: repopulating would otherwise fire one panel update
        per item, each one routed at the active plot.
        """
        previous = get_combo_value(combo)
        combo.blockSignals(True)
        combo.clear()
        for display, raw in zip(clean_display_labels(values), values):
            combo.addItem(display, raw)
        index = find_combo_index(combo, previous)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def _set_combo_row_visible(self, key: str, visible: bool) -> None:
        """Show or hide a dimension's whole row, label included."""
        field = self._combo_row_fields.get(key)
        if field is None:
            return
        layout = self._combo_row_layouts.get(key, self.coords_groupbox_layout)
        label = layout.labelForField(field)
        field.setVisible(visible)
        if label is not None:
            label.setVisible(visible)

    def _register_feature(self, feature_name: str) -> None:
        """Add a computed feature to the catalog + features combo so the
        add-panel popup and sidebar offer it everywhere."""
        cat = self.catalog or getattr(self.app_state.data_loader, "catalog", None)
        if cat is not None:
            if feature_name not in cat.features:
                cat.features.append(feature_name)
            choices = cat.feature_choices()
            if feature_name not in choices:
                choices.append(feature_name)
            cat.combos["features"] = ComboSpec("features", tuple(choices))
        combo = self.combos.get("features")
        if combo is not None and find_combo_index(combo, feature_name) < 0:
            combo.blockSignals(True)
            combo.addItem(feature_name, feature_name)
            combo.blockSignals(False)

    def refresh_feature_choices(self) -> None:
        """Re-fill the features combo from the catalog.

        Used when features appear or disappear after load — today that means
        the console panel registering/forgetting a derived feature. The combo
        is refilled rather than appended to, so ``forget()`` takes effect too.
        """
        cat = self.catalog or getattr(self.app_state.data_loader, "catalog", None)
        combo = self.combos.get("features")
        if cat is None or combo is None:
            return
        self._refill_combo(combo, cat.feature_choices())

    def _set_controls_enabled(self, enabled: bool):
        for control in self.controls:
            control.setEnabled(enabled)
        self.io_widget.set_controls_enabled(enabled)
        self.app_state.ready = enabled

    def _create_colors_combo(self):
        """Create the Colors combo populated with all features, filtered by rgb suffix."""
        all_features = list(self.catalog.features)
        if not all_features:
            return

        combo = QComboBox()
        combo.setObjectName("colors_combo")
        combo.currentIndexChanged.connect(self._on_combo_changed)

        rgb_checkbox = QCheckBox("rgb suffix")
        rgb_checkbox.setObjectName("colors_rgb_suffix_checkbox")
        rgb_checkbox.setToolTip("Only show features with 'rgb' in the name")
        rgb_checkbox.setChecked(True)
        rgb_checkbox.stateChanged.connect(self._on_rgb_suffix_changed)
        self._colors_rgb_checkbox = rgb_checkbox

        self._populate_colors_combo(combo, all_features, rgb_filter=True)
        make_searchable(combo)

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(5)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_layout.addWidget(combo)
        row_layout.addWidget(rgb_checkbox)

        self.coords_groupbox_layout.addRow("Colors:", row_widget)
        self.combos["colors"] = combo
        self.controls.append(combo)
        self.controls.append(rgb_checkbox)

    def _populate_colors_combo(self, combo: QComboBox, features: list[str], rgb_filter: bool):
        prev = get_combo_value(combo) if combo.count() > 0 else "None"
        combo.blockSignals(True)
        combo.clear()
        raw_items = ["None"]
        if rgb_filter:
            raw_items += [f for f in features if "rgb" in f.lower()]
        else:
            raw_items += features
        display_items = clean_display_labels(raw_items)
        for display, raw in zip(display_items, raw_items):
            combo.addItem(display, raw)
        # Restore previous selection if still available
        for i in range(combo.count()):
            if combo.itemData(i) == prev:
                combo.setCurrentIndex(i)
                break
        combo.blockSignals(False)

    def _on_rgb_suffix_changed(self, _state: int):
        combo = self.combos.get("colors")
        if combo is None:
            return
        rgb_filter = self._colors_rgb_checkbox.isChecked()
        self._populate_colors_combo(combo, list(self.catalog.features), rgb_filter)
        self.app_state.set_key_sel("colors", get_combo_value(combo))
        if self.app_state.ready and self.plot_container:
            current_plot = self.plot_container.get_current_plot()
            xmin, xmax = current_plot.get_current_xlim()
            self.update_main_plot(t0=xmin, t1=xmax)

    def _create_combo_widget(self, key, vars):
        excluded_from_all = {*INDIVIDUAL_DIMS, "features", "cameras", "mics"}
        show_all_checkbox = key not in excluded_from_all

        combo = QComboBox()
        combo.setObjectName(f"{key}_combo")
        combo.currentIndexChanged.connect(self._on_combo_changed)
        raw_items = [str(var) for var in vars]
        display_items = clean_display_labels(raw_items)
        for display, raw in zip(display_items, raw_items):
            combo.addItem(display, raw)

        make_searchable(combo)

        # The individual lives above every plot's own settings, not among the
        # coords: which animal is shown is a question every panel type answers,
        # not one only feature plots have.
        is_individual = key in INDIVIDUAL_DIMS
        target_layout = self.individual_layout if is_individual else self.coords_groupbox_layout

        if is_individual:
            # The combo is the sidebar's individual — what every panel that
            # follows it shows. The pin button beside it switches the panel
            # last clicked between following and a pinned individual.
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(5)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_layout.addWidget(combo)

            pin_btn = QToolButton()
            pin_btn.setObjectName("individual_pin_button")
            pin_btn.setText("\U0001f4cc")
            pin_btn.setToolTip(
                "The panel you last clicked (a plot or a camera view): follow this combo, or pin it to one individual"
            )
            pin_btn.setPopupMode(QToolButton.InstantPopup)
            pin_menu = QMenu(pin_btn)
            pin_menu.aboutToShow.connect(lambda m=pin_menu: self._fill_pin_menu(m))
            pin_btn.setMenu(pin_menu)
            row_layout.addWidget(pin_btn)

            self.individual_pin_button = pin_btn
            target_layout.insertRow(0, "Individual:", row_widget)
            self.combos[key] = combo
            self._combo_row_fields[key] = row_widget
            self._combo_row_layouts[key] = target_layout
            self.controls.append(combo)
            return combo

        if show_all_checkbox:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(5)

            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_layout.addWidget(combo)

            all_checkbox = QCheckBox("All")
            all_checkbox.setObjectName(f"{key}_all_checkbox")
            all_checkbox.setToolTip(f"Show all {key} traces on the plot")
            all_checkbox.stateChanged.connect(lambda state, k=key: self._on_all_checkbox_changed(k, state))
            row_layout.addWidget(all_checkbox)

            self.all_checkboxes[key] = all_checkbox
            self.controls.append(all_checkbox)

            target_layout.addRow(f"{key.capitalize()}:", row_widget)
            field = row_widget
        else:
            target_layout.addRow(f"{key.capitalize()}:", combo)
            field = combo

        self.combos[key] = combo
        # The widget occupying the row, so the row can later be hidden without
        # being removed — see `_rebuild_coord_controls`.
        self._combo_row_fields[key] = field
        self._combo_row_layouts[key] = target_layout
        self.controls.append(combo)

        return combo

    # ------------------------------------------------------------------
    # Individual (actor) + receiver
    # ------------------------------------------------------------------

    def _individual_actor_key(self) -> str:
        """The selection key the actor combo writes to.

        The dataset's own individual dim when it has one (movement is
        singular, older wizard data plural — never renamed), and the singular
        spelling otherwise, where it is an inert key: labels use it, loaders
        ignore it, exactly like any selection naming a dim a feature lacks.
        """
        catalog = self.catalog or getattr(self.app_state.data_loader, "catalog", None)
        return (catalog.individual_combo if catalog is not None else None) or INDIVIDUAL_DIMS[0]

    def refresh_individual_choices(self) -> None:
        """Point the Individual / Receiver combos at this session's individuals.

        Run on load, on a catalog swap, and after a label import: with no
        individual dimension the names come from the labels themselves, and
        those arrive from a file the user can pick at any time.
        """
        if getattr(self, "individual_rec_combo", None) is None:
            return
        key = self._individual_actor_key()
        catalog = self.catalog or getattr(self.app_state.data_loader, "catalog", None)
        is_dim = catalog is not None and key in catalog.combos
        combo = self.combos.get(key)
        if combo is None:
            combo = self._create_combo_widget(key, self.app_state.label_individuals())
        elif not is_dim:
            # Not a dim: the values are the individuals the labels name, which
            # a dim combo must never be refilled with.
            self._refill_combo(combo, self.app_state.label_individuals())
        self._set_combo_row_visible(key, True)
        self.populate_bbox_colors(self.app_state.label_individuals())
        # One animal: nothing to pin, so the control is not there to puzzle over.
        pin_btn = getattr(self, "individual_pin_button", None)
        if pin_btn is not None:
            pin_btn.setVisible(len(self.app_state.label_individuals()) > 1)
        self.app_state.set_key_sel(key, get_combo_value(combo))
        # Exactly one spelling carries a value: a stale `individuals_sel` from
        # a previous dataset would answer `selected_individual()` first.
        for other in INDIVIDUAL_DIMS:
            if other != key:
                self.app_state.set_key_sel(other, None)
        self._populate_receiver_combo()

    def _populate_receiver_combo(self) -> None:
        """Offer every individual except the actor — nothing is its own receiver."""
        combo = getattr(self, "individual_rec_combo", None)
        if combo is None:
            return
        actor = self.app_state.selected_individual()
        names = [n for n in self.app_state.label_individuals() if n != actor]
        wanted = self.app_state.selected_receiver()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("None", "")
        for name in names:
            combo.addItem(str(name), str(name))
        idx = find_combo_index(combo, wanted) if wanted else 0
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
        combo.setEnabled(bool(names))
        # A receiver this actor cannot have is dropped, not kept as a filter
        # that silently matches nothing.
        self.app_state.individual_receiver = get_combo_value(combo) or ""

    # ------------------------------------------------------------------
    # Per-panel individual: follow the sidebar, or a pin — and the subject
    # ------------------------------------------------------------------

    def _pinnable_widget(self):
        """The last clicked panel, if it is one that can be pinned: a feature plot or a camera view."""
        from .active_panel import PanelKind

        manager = getattr(self.plot_container, "active_panels", None)
        reg = manager.active if manager is not None else None
        if reg is None:
            return None
        if reg.kind in PanelKind.FEATURE:
            return reg.plot
        if reg.kind == PanelKind.VIDEO:
            return reg.widget
        return None

    def _fill_pin_menu(self, menu: QMenu) -> None:
        from .plots_container import add_pin_choices

        menu.clear()
        target = self._pinnable_widget()
        names = self.app_state.label_individuals()
        if len(names) < 2:
            menu.addAction("One individual \u2014 nothing to pin").setEnabled(False)
            return
        if target is None:
            menu.addAction("Click a plot or camera panel first").setEnabled(False)
        else:
            title = self.plot_container.panel_title(target) if hasattr(target, "set_pinned_individual") else "camera"
            menu.addAction(f"Panel: {title}").setEnabled(False)
            add_pin_choices(
                menu,
                names,
                self.app_state.pinned_individual_of(target),
                self.app_state.sidebar_individual(),
                lambda n, t=target: self.pin_panel(t, n),
            )
        menu.addSeparator()
        unpin_all = menu.addAction("Unpin all panels (follow sidebar)", self.unpin_all_panels)
        unpin_all.setEnabled(self._any_pinned())

    def _pinned_widgets(self) -> list:
        pc = self.plot_container
        plots = list(pc._panels_of_group("feature")) if pc is not None else []
        views = [self.video_mgr.primary_view, *self.video_mgr.extra_widgets.values()] if self.video_mgr else []
        return [w for w in [*plots, *views] if self.app_state.pinned_individual_of(w) is not None]

    def _any_pinned(self) -> bool:
        return bool(self._pinned_widgets())

    def unpin_all_panels(self) -> None:
        """Every panel back to following the sidebar — the way to show one individual everywhere."""
        for widget in self._pinned_widgets():
            self.pin_panel(widget, None)

    def pin_panel(self, widget, individual: str | None) -> None:
        """Pin *widget* — a feature plot or a camera view — to *individual* (``None`` unpins)."""
        if hasattr(widget, "set_pinned_individual"):
            self.plot_container.pin_panel(widget, individual)
        else:
            self.pin_camera_view(widget, individual)
        self.app_state.refresh_labelling_subject()
        self.on_labelling_subject_changed()

    @staticmethod
    def _camera_pin_key(view) -> str:
        dock = getattr(view, "dock_widget", None)
        key = getattr(dock, "_camera_key", None)
        return str(key) if key else "primary"

    def pin_camera_view(self, view, individual: str | None) -> None:
        """Pin a camera view: its pose overlay shows that individual only; remembered in ``camera_individuals``."""
        view.pinned_individual = individual or None
        pins = dict(self.app_state.camera_individuals or {})
        key = self._camera_pin_key(view)
        if individual:
            pins[key] = str(individual)
        else:
            pins.pop(key, None)
        self.app_state.camera_individuals = pins
        self.video_mgr.refresh_view_title(view)
        if self.app_state.ready:
            self.update_pose()

    def _restore_camera_pins(self) -> None:
        """Put the saved pins back on the camera views the layout just rebuilt."""
        pins = self.app_state.camera_individuals or {}
        views = [self.video_mgr.primary_view, *self.video_mgr.extra_widgets.values()]
        for view in views:
            view.pinned_individual = pins.get(self._camera_pin_key(view))
            self.video_mgr.refresh_view_title(view)
        if pins and self.app_state.ready:
            self.update_pose()

    def refresh_following_panels(self) -> None:
        """Re-render every panel that follows the sidebar's individual (pinned ones stay put)."""
        pc = self.plot_container
        if pc is None or not self.app_state.ready:
            return
        for plot in pc._panels_of_group("feature"):
            if self.app_state.pinned_individual_of(plot) is None:
                plot.update_plot()
                pc.panel_content_changed.emit(plot)
        pc.refresh_panel_titles()
        self.video_mgr.refresh_view_titles()
        self.update_pose()

    def on_labelling_subject_changed(self) -> None:
        """The subject a new label lands on moved: the receiver choices and any half-placed label follow."""
        self._populate_receiver_combo()
        if self.labels_widget is not None:
            self.labels_widget._reset_label_clicks()

    def _on_receiver_changed(self, _index: int) -> None:
        if not self.app_state.ready:
            return
        self.app_state.individual_receiver = get_combo_value(self.individual_rec_combo) or ""
        if self.labels_widget:
            # A half-placed label was anchored for the previous subject.
            self.labels_widget._reset_label_clicks()
            self.labels_widget.refresh_labels_shapes_layer()
        self.update_label_plot()

    def sync_sidebar_from_active_plot(self):
        """Repopulate EVERY data-panel selection control (feature, dim combos,
        colours, and 'All' checkboxes) from the active plot's own state, so the
        sidebar always shows the settings of the plot the user last clicked."""
        if not self.app_state.ready:
            return
        plot = getattr(self.plot_container, "active_feature_plot", None)
        if plot is None or not hasattr(plot, "_effective_feature"):
            return
        feature = plot._effective_feature()
        selections = plot._effective_selections()
        color = plot._effective_color() or "None"

        def _set(ckey, value):
            combo = self.combos.get(ckey)
            if combo is None or value is None:
                return
            idx = find_combo_index(combo, str(value))
            if idx < 0 and ckey == "features":
                # The combo must always be able to display the active plot's
                # feature — a dropdown showing another plot's value is leakage.
                combo.blockSignals(True)
                combo.addItem(str(value), str(value))
                combo.blockSignals(False)
                idx = combo.count() - 1
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

        _set("features", feature)
        for ckey in list(self.combos.keys()):
            # The individual combo is the sidebar's, never a mirror of the
            # active panel: a pinned panel shows its pin, everything else this.
            if ckey in ("features", "colors") or ckey in INDIVIDUAL_DIMS:
                continue
            if ckey in selections:
                _set(ckey, selections[ckey])
        _set("colors", color)

        # 'All' checkboxes: a dimension absent from this plot's selections means
        # "show all values" for it → the box is checked and its combo disabled.
        all_checkboxes = getattr(self, "all_checkboxes", {})
        for akey, checkbox in all_checkboxes.items():
            is_all = akey not in selections
            checkbox.blockSignals(True)
            checkbox.setChecked(is_all)
            checkbox.blockSignals(False)
            combo = self.combos.get(akey)
            if combo is not None:
                combo.setEnabled(not is_all)

    def apply_panel_control(self, key: str, value):
        """Generic entry point for EVERY data-panel selection control (feature
        dropdown, dimension combos, colours, and 'All' checkboxes).

        The control only ever affects the *active* plot: it writes to that plot's
        own ``panel_state`` and re-renders just it. The value is also mirrored to
        global ``app_state`` so the shared consumers (label overlays,
        changepoints, feature view-mode) follow the active plot — safe because
        every plot has already forked its own state. The space plot renders
        purely from its own catalog-driven combos and does not follow this.
        """
        if key in INDIVIDUAL_DIMS:
            # The sidebar's individual is global: every unpinned panel follows
            # it, a pinned one does not, and no panel records it as its own.
            self.app_state.set_key_sel(key, value)
            self.app_state.refresh_labelling_subject()
            self.on_labelling_subject_changed()
            self.refresh_following_panels()
            if self.labels_widget is not None:
                self.labels_widget.refresh_labels_shapes_layer()
            self.update_label_plot()
            return

        active = getattr(self.plot_container, "active_feature_plot", None)
        if active is not None and hasattr(active, "set_panel_control"):
            active.set_panel_control(key, value)
            active.update_plot()
            if key == "features" and value:
                self.plot_container.set_panel_title(active, self.plot_container.panel_title(active))
            # What this panel shows just changed; anything tracking its contents
            # (the console) must follow, and no click will tell it.
            self.plot_container.panel_content_changed.emit(active)

        self.app_state.set_key_sel(key, value)

        # A panel never changes type behind the user's back: changing the
        # feature must NOT auto-switch the lineplot/heatmap view (that hid
        # other open panels). Only the explicit View combo switches views.
        if key == "features" and active is self.plot_container.get_current_plot():
            self._update_view_mode_items(value)
            self.view_mode_combo.show()
        if key == "cluster_id" and self.ephys_widget:
            try:
                self.ephys_widget.select_cluster_in_table(int(value))
            except (ValueError, TypeError):
                pass
        self.update_label_plot()

    def _on_combo_changed(self):
        if not self.app_state.ready:
            return
        combo = self.sender()
        name = combo.objectName()
        key = name[:-6] if name.endswith("_combo") else None
        if not key:
            return
        self.apply_panel_control(key, get_combo_value(combo))

    def _on_all_checkbox_changed(self, key: str, state: int):
        if not self.app_state.ready:
            return

        combo = self.combos.get(key)
        if combo is None:
            return

        is_checked = Qt.CheckState(state) == Qt.Checked

        # "All" checkboxes are mutually exclusive: turning one on resets the others.
        if is_checked:
            for other_key, other_checkbox in self.all_checkboxes.items():
                if other_key != key and other_checkbox.isChecked():
                    other_checkbox.blockSignals(True)
                    other_checkbox.setChecked(False)
                    other_checkbox.blockSignals(False)
                    other_combo = self.combos.get(other_key)
                    if other_combo:
                        other_combo.setEnabled(True)
                        self.apply_panel_control(other_key, get_combo_value(other_combo))
                    self._update_all_checkbox_state(other_key, False)

        combo.setEnabled(not is_checked)
        self._update_all_checkbox_state(key, is_checked)

        # None ⇒ "show all values for this dimension" (routed to the active plot).
        self.apply_panel_control(key, None if is_checked else get_combo_value(combo))

    def _update_all_checkbox_state(self, key: str, is_checked: bool):
        states = self.app_state.all_checkbox_states.copy()
        if is_checked:
            states[key] = True
        else:
            states.pop(key, None)
        self.app_state.all_checkbox_states = states

    def _restore_or_set_defaults(self):
        if not self.catalog:
            return
        for key, spec in self.catalog.combos.items():
            combo = self.io_widget.combos.get(key) or self.combos.get(key)
            vals = list(spec.values)

            if combo is not None and vals:
                saved_value = self.app_state.get_key_sel(key) if self.app_state.key_sel_exists(key) else None
                vals_str = [str(v) for v in vals]

                if saved_value in vals_str:
                    set_combo_to_value(combo, saved_value)
                elif saved_value and key == "mics":
                    match = next((v for v in vals_str if v.startswith(str(saved_value))), None)
                    if match:
                        set_combo_to_value(combo, match)
                        self.app_state.set_key_sel(key, match)
                    else:
                        set_combo_to_value(combo, vals_str[0])
                        self.app_state.set_key_sel(key, vals_str[0])
                else:
                    if key == "features" and "speed" in vals_str:
                        set_combo_to_value(combo, "speed")
                        self.app_state.set_key_sel(key, "speed")
                    else:
                        set_combo_to_value(combo, vals_str[0])
                        self.app_state.set_key_sel(key, vals_str[0])

        # Restore colors combo (not in catalog.combos, created separately)
        colors_combo = self.combos.get("colors")
        if colors_combo is not None and self.app_state.key_sel_exists("colors"):
            saved_color = self.app_state.get_key_sel("colors")
            if saved_color is not None:
                idx = find_combo_index(colors_combo, str(saved_color))
                if idx < 0 and hasattr(self, "_colors_rgb_checkbox"):
                    # Saved value not visible with rgb filter on — disable it
                    self._colors_rgb_checkbox.blockSignals(True)
                    self._colors_rgb_checkbox.setChecked(False)
                    self._colors_rgb_checkbox.blockSignals(False)
                    self._populate_colors_combo(colors_combo, list(self.catalog.features), rgb_filter=False)
                    idx = find_combo_index(colors_combo, str(saved_color))
                if idx >= 0:
                    colors_combo.blockSignals(True)
                    colors_combo.setCurrentIndex(idx)
                    colors_combo.blockSignals(False)
                    self.app_state.set_key_sel("colors", saved_color)

        if self.app_state.key_sel_exists("trials"):
            saved_trial = self.app_state.get_key_sel("trials")
            self.app_state.set_key_sel("trials", saved_trial)
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials_sel))
        else:
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials[0]))
            self.app_state.trials_sel = self.app_state.trials[0]

        space_plot_type = getattr(self.app_state, "space_plot_type", "Layers")
        # Migrate old values to simplified combo
        if space_plot_type in (
            "Space 2D",
            "Space 3D",
            "space_2D",
            "space_3D",
            "PCA 2D",
            "PCA 3D",
        ):
            space_plot_type = "Space Plot"
        if hasattr(self, "space_view_combo"):
            self.space_view_combo.setCurrentText(space_plot_type)

        saved_camera = self.app_state.primary_camera
        if saved_camera and hasattr(self, "primary_camera_combo"):
            idx = self.primary_camera_combo.findText(saved_camera)
            if idx >= 0:
                self.primary_camera_combo.setCurrentIndex(idx)

        saved_extra = self.app_state.extra_cameras
        if saved_extra and hasattr(self, "_extra_camera_combos"):
            for i, cam_name in enumerate(saved_extra):
                if i >= len(self._extra_camera_combos):
                    break
                combo = self._extra_camera_combos[i]
                idx = combo.findText(cam_name)
                if idx >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)

        # Normalize stale *_sel values against currently loaded options.
        self._normalize_saved_sel_values()

        checkbox_states = self.app_state.all_checkbox_states or {}
        for key, is_checked in checkbox_states.items():
            checkbox = self.all_checkboxes.get(key)
            combo = self.combos.get(key)
            if checkbox and is_checked:
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
                if combo:
                    combo.setEnabled(False)
                self.app_state.set_key_sel(key, None)

    def _normalize_saved_sel_values(self) -> None:
        def _normalize_from_combo(key: str, combo: QComboBox) -> None:
            if combo is None or combo.count() == 0:
                return
            saved_value = self.app_state.get_key_sel(key) if self.app_state.key_sel_exists(key) else None
            if saved_value is None:
                self.app_state.set_key_sel(key, get_combo_value(combo))
                return
            if find_combo_index(combo, str(saved_value)) >= 0:
                return
            combo.setCurrentIndex(0)
            fallback = get_combo_value(combo)
            logger.warning(
                "Saved %s_sel '%s' not found; reverting to '%s'",
                key,
                saved_value,
                fallback,
            )
            self.app_state.set_key_sel(key, fallback)

        for key, combo in self.combos.items():
            _normalize_from_combo(key, combo)
        for key, combo in self.io_widget.combos.items():
            _normalize_from_combo(key, combo)

        primary_combo = getattr(self, "primary_camera_combo", None)
        if isinstance(primary_combo, QComboBox) and primary_combo.count() > 0:
            saved = self.app_state.primary_camera
            if saved is None or find_combo_index(primary_combo, saved) < 0:
                self.app_state.primary_camera = get_combo_value(primary_combo)

    # ------------------------------------------------------------------
    # Trial change
    # ------------------------------------------------------------------

    def _load_trial_with_fallback(self) -> None:
        first_trial = self.app_state.trials[0]
        current_trial = getattr(self.app_state, "trials_sel", None)

        try:
            is_nan = np.isnan(current_trial)
        except (TypeError, ValueError):
            is_nan = False

        if not current_trial or is_nan or current_trial not in self.app_state.trials:
            if current_trial and not is_nan:
                logger.warning(
                    "Saved trial %s not in dataset, using %s",
                    current_trial,
                    first_trial,
                )
            self.app_state.trials_sel = first_trial

        self.on_trial_changed()

    def _apply_video_dock_default(self):
        """Show the video dock only when a video actually loaded (data-availability
        default after a dataset load — mirrors the audio/feature panel defaults)."""
        shell = getattr(self.meta_widget, "shell", None)
        if shell is None:
            return
        has_video = bool(getattr(self.app_state, "video", None)) or self.video_mgr.primary_view.has_video
        toggle = getattr(shell, "_video_toggle", None)
        if toggle is not None:
            toggle.setChecked(has_video)
        shell.set_video_viewer_visible(has_video)

    def _disable_empty_panels(self):
        """Hide plot panels that have no data after the first trial load."""
        pc = self.plot_container
        if pc is None:
            return
        if not self.app_state.audio_path:
            pc.set_audiotrace_visible(False)
            pc.set_spectrogram_visible(False)
        if not (self.catalog and self.catalog.features):
            for plot in list(pc.line_plots):
                pc.remove_lineplot(plot)
            pc.set_heatmap_visible(False)
        for plot in list(pc.neo_trace_plots):
            if getattr(plot, "_source", None) is None:
                pc.remove_panel(plot)
        if not self.app_state.has_neurons:
            pc.set_ephys_visible(False)

    def _validate_media_files(
        self,
        nwb_alignment=None,
        first_trial=None,
        video_folder=None,
        audio_folder=None,
        pose_folder=None,
    ) -> list[str]:
        sio = nwb_alignment if nwb_alignment is not None else self.app_state.nwb_alignment
        if first_trial is None:
            first_trial = self.app_state.trials[0]
        if video_folder is None:
            video_folder = self.app_state.video_folder
        if audio_folder is None:
            audio_folder = self.app_state.audio_folder
        if pose_folder is None:
            pose_folder = self.app_state.pose_folder

        missing = []

        if video_folder:
            for cam in sio.cameras:
                vid = sio.get_media(first_trial, "video", device=cam)
                if not vid or is_url(vid):
                    continue
                path = os.path.join(video_folder, vid)
                if not os.path.isfile(path):
                    missing.append(f"Video: {path}")

        if audio_folder:
            mics = sio.mics
            if not mics:
                notify(
                    "You selected an audio folder, although the .nc contains no audio media entries.",
                    "warning",
                )
            else:
                for mic in mics:
                    aud = sio.get_media(first_trial, "audio", device=mic)
                    if not aud:
                        continue
                    path = os.path.join(audio_folder, aud)
                    if not os.path.isfile(path):
                        missing.append(f"Audio: {path}")

        if pose_folder:
            cameras = sio.cameras
            if not cameras:
                notify(
                    "You selected a pose folder, although the .nc contains no pose data.",
                    "warning",
                )
            else:
                for cam in cameras:
                    pose_file = sio.get_media(first_trial, "pose", device=cam)
                    if not pose_file:
                        continue
                    path = os.path.join(pose_folder, pose_file)
                    if not os.path.isfile(path):
                        missing.append(f"Pose: {path}")

        return missing

    def _install_display_offset_provider(self) -> None:
        """Give the loader its display→native clock bridge, per backend.

        The offset is *pulled* per select call, so trial and scope changes
        need no re-sync. Pynapple data is natively session-absolute, xarray
        natively trial-local — each gets the provider that shifts a
        display-clock query into its own clock.
        """
        loader = self.app_state.data_loader
        if loader is None or not hasattr(loader, "set_display_offset_provider"):
            return
        if getattr(loader, "backend", None) == "pynapple":
            loader.set_display_offset_provider(self._pynapple_display_offset)
        else:
            loader.set_display_offset_provider(self._xarray_display_offset)

    def _trial_session_start(self) -> float:
        sc = getattr(self.app_state, "source_collection", None)
        trial = getattr(self.app_state, "trials_sel", None)
        if sc is None or trial is None:
            return 0.0
        return float(sc.to_session(trial, 0.0))

    def _pynapple_display_offset(self) -> float:
        """Display→absolute offset for the pynapple loader.

        Pynapple sources are session-absolute, so trial-basis queries shift
        forward by the current trial's session start; in session basis the
        axis already matches. The basis itself comes from
        ``app_state.display_basis`` — never re-derived here.
        """
        if self.app_state.display_basis == "session":
            return 0.0
        return self._trial_session_start()

    def _xarray_display_offset(self) -> float:
        """Display→trial-local offset for the xarray loader.

        Xarray time coords are trial-local, so session-basis queries shift
        back by the trial's session start — the current trial renders at its
        true session position (other trials are simply absent; multi-trial
        stitching is out of scope, see the time-slider docs).
        """
        if self.app_state.display_basis != "session":
            return 0.0
        return -self._trial_session_start()

    def _build_trial_alignment(self, trial_id) -> None:
        self.app_state.trial_alignment = compute_trial_video_bounds(
            self.app_state.nwb_alignment,
            trial_id,
            self.app_state.ds,
            video_folder=self.app_state.video_folder,
            audio_folder=self.app_state.audio_folder,
            cameras_sel=self.app_state.primary_camera,
            source_collection=getattr(self.app_state, "source_collection", None),
        )

    def _build_restrict_window(self, trial_id) -> None:
        if self.navigation_widget is not None:
            self.navigation_widget._apply_slider_scope()

    def on_trial_changed(self):
        trials_sel = self.app_state.trials_sel

        if trials_sel not in self.app_state.trials:
            logger.warning(
                "Selected trial '%s' not found in dataset, reverting to first trial '%s'",
                trials_sel,
                self.app_state.trials[0],
            )
            trials_sel = self.app_state.trials[0]
            self.app_state.trials_sel = trials_sel
            return

        if self.app_state.dt is not None:
            self.app_state.ds = self.app_state.dt.trial(trials_sel)

        self.pose_frame_path = self.app_state.ds.attrs.get("frame_path")

        # Update data_loader (xarray only: swap the backing dataset)
        store = self.app_state.data_loader
        trial_idx = self.app_state.trials.index(trials_sel)
        if store is not None and hasattr(store, "update_ds"):
            store.update_ds(self.app_state.ds)

        # Sync SourceCollection trial index for time-range queries
        sc = getattr(self.app_state, "source_collection", None)
        if sc is not None:
            for src in sc.sources.values():
                if hasattr(src, "set_trial"):
                    src.set_trial(trial_idx)

        self._update_device_sels_for_trial(self.app_state.ds)
        self.refresh_audio_sources_for_trial(self.app_state.ds)

        features_combo = self.combos.get("features")
        fallback_feature = features_combo.itemText(0) if features_combo and features_combo.count() else None
        feature_sel = getattr(self.app_state, "features_sel", fallback_feature)

        if hasattr(self, "view_mode_combo"):
            self._update_view_mode_items(feature_sel)

        self.app_state.label_intervals = self.app_state.get_trial_intervals(trials_sel)

        self._build_trial_alignment(trials_sel)
        self._build_restrict_window(trials_sel)

        self.app_state.current_frame = 0
        if not self.suppress_video_load:
            self.update_video()
            self._init_or_update_extra_cameras()
            # Reconcile background proxy jobs to the new trial's visible videos
            # (cancel stale ones, start/swap for the current set).
            self.video_mgr.sync_proxies()
        self.update_audio()
        if not self.suppress_video_load:
            self.update_pose()
        self.update_label()
        if self.ephys_widget:
            self.ephys_widget.on_trial_changed()

        preserve = getattr(self.app_state, "_preserve_x_range_next", False)
        self.app_state._preserve_x_range_next = False
        self.update_main_plot(preserve_x_range=preserve)
        try:
            self.update_space_plot()
        except Exception:
            logger.debug("update_space_plot failed", exc_info=True)
        self.refresh_radial_plots()

        # Trial start in the display clock — 0.0 only in trial basis; in
        # session basis the trial starts at its session offset. Marker-driven
        # trial switches (auto-follow, session-scope label clicks) keep the
        # marker where the user put it instead.
        marker_follow = getattr(self.app_state, "_marker_driven_trial_switch", False)
        self.app_state._marker_driven_trial_switch = False
        if not marker_follow:
            self.plot_container.update_time_marker_by_time(self.app_state.to_display(trials_sel, 0.0))

        self._update_confidence_overlay()

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------

    def update_main_plot(self, **kwargs):
        if not self.app_state.ready:
            return

        self.plot_container.clear_amplitude_envelope()

        self.plot_container.update_feature_plots(**kwargs)

        if self.show_envelope_checkbox.isChecked():
            self.plot_container.show_envelope_overlay()

        self.update_label_plot()

    def _subject_intervals(self, df, panel=None):
        """*df* reduced to the labels of one actor and the selected receiver.

        The actor is *panel*'s individual — its pin, else the sidebar's — so
        every panel draws the labels of the animal it shows; with no panel,
        the labelling subject (the last clicked panel's). Read from
        ``app_state``, never from the xarray kwargs: a pynapple session has
        no ``ds_kwargs`` at all, and whose label a row is was never an xarray
        question.
        """
        if df is None or df.empty:
            return df
        actor = self.app_state.panel_individual(panel) if panel is not None else self.app_state.selected_individual()
        if not self.app_state.labels_name_our_individuals(df):
            actor = None
        # Every receiver: the receiver is a tag on a label, never a filter.
        return select_subject(df, actor)

    def update_label_plot(self):
        # Labels are hidden when no branch is shown and predictions aren't toggled on.
        state = self.app_state
        any_slot = bool(state._branch_shown and any(state._branch_shown.values())) or state._show_predictions_overlay
        if not any_slot:
            if self.plot_container:
                for plot in self.plot_container._get_all_plots():
                    self.plot_container._clear_labels_on_plot(plot)
            return

        # Display-basis view: session basis shows EVERY trial's labels at
        # their session positions; trial basis is the current trial verbatim.
        intervals_df = self.app_state.get_display_intervals()

        # Session basis can hold thousands of rows across many panels — draw
        # only what intersects the viewport (padded half a window each side;
        # zoom/pan re-derives the set via the container's debounced redraw),
        # and nothing at all above the cap: at that density the rectangles are
        # sub-pixel mush and the GUI would crawl. Zooming in brings them back.
        if (
            intervals_df is not None
            and not intervals_df.empty
            and self.app_state.display_basis == "session"
            and self.plot_container is not None
        ):
            try:
                t0, t1 = self.plot_container.get_current_xlim()
            except (AttributeError, TypeError):
                t0 = t1 = None
            if t0 is not None and t1 is not None and t1 > t0:
                pad = (t1 - t0) * 0.5
                ends = intervals_df["offset_s"].fillna(intervals_df["onset_s"])
                visible = (intervals_df["onset_s"] <= t1 + pad) & (ends >= t0 - pad)
                intervals_df = intervals_df[visible]
            if len(intervals_df) > SESSION_LABELS_MAX_DRAWN:
                logger.debug(
                    "Skipping label overlay: %d labels in view (max %d) — zoom in",
                    len(intervals_df),
                    SESSION_LABELS_MAX_DRAWN,
                )
                intervals_df = intervals_df.iloc[0:0]

        # Filtered per panel while drawing (`LabelDrawingMixin.draw_all_labels`),
        # so a pinned panel shows its own individual's labels.
        self.plot_container.subject_filter = self._subject_intervals

        predictions_df = None
        if self.app_state.pred_labels_df is not None:
            trial = self.app_state.trials_sel
            df = self.app_state.pred_labels_df
            predictions_df = df[df["trial"] == trial] if "trial" in df.columns else df
            if predictions_df is not None and self.app_state.display_basis == "session":
                shift = self.app_state.to_display(trial, 0.0)
                if shift:
                    predictions_df = predictions_df.copy()
                    predictions_df["onset_s"] = predictions_df["onset_s"] + shift
                    predictions_df["offset_s"] = predictions_df["offset_s"] + shift

        self.labels_widget.plot_all_labels(intervals_df, predictions_df)

    # ------------------------------------------------------------------
    # Video / audio / pose / space
    # ------------------------------------------------------------------

    def update_video(self):
        if not self.app_state.ready:
            return
        self.show_envelope_checkbox.show()
        self.video_mgr.update_video(plot_container=self.plot_container)

    def set_video_quality(self, proxy: bool):
        """Switch video decoding between full-res and a low-res proxy.

        Non-blocking: enabling kicks off background proxy generation for every
        visible video and swaps each in when ready; disabling cancels the jobs
        and reverts to full-res. Frame timing is identical between source and
        proxy, so labels/alignment are unaffected.
        """
        if proxy and self.pose_mgr.has_active_skeleton_overlay():
            notify(
                "Pose/skeleton overlay may be misaligned on the proxy video (different resolution than the source).",
                "warning",
            )
        self.app_state.video_quality_mode = "proxy" if proxy else "full"
        self.video_mgr.sync_proxies()

    def update_audio(self):
        if not self.app_state.ready:
            return
        self.video_mgr.update_audio(plot_container=self.plot_container)

    def update_label(self):
        self.labels_widget.refresh_labels_shapes_layer()

    def toggle_pause_resume(self):
        self.video_mgr.toggle_pause_resume(self.plot_container)

    def _on_time_marker_updated(self, time_s: float):
        # Session basis: once the marker sits in ANOTHER trial's span, follow
        # it — switch the current trial after a short debounce. The timer only
        # (re)starts when the TARGET trial changes: a continuously moving
        # marker (gap-run playback) must still trigger the follow, so the
        # debounce must not be reset by every tick within the same span.
        if self.app_state.display_basis == "session":
            hit = self.app_state.from_display(time_s, strict=True)
            target = hit[0] if hit is not None else None
            if target is not None and target != self.app_state.trials_sel:
                self._follow_pending_time = time_s
                if target != self._follow_pending_trial or not self._marker_follow_timer.isActive():
                    self._follow_pending_trial = target
                    self._marker_follow_timer.start()
            else:
                # Back in the current trial (or a gap): nothing to follow.
                self._follow_pending_trial = None
                self._marker_follow_timer.stop()
        self._update_video_blanking(time_s)
        # Static-image views have no frame clock — the marker animates their
        # pose overlay directly (overlay time is trial-local).
        image_views = self.video_mgr.image_views()
        if image_views:
            resolved = self.app_state.from_display(time_s)
            if resolved is not None:
                for view in image_views:
                    view.set_overlay_time(resolved[1])
        visible = [sp for sp in self.space_plots if sp.isVisible()]
        if not visible:
            return
        for sp in visible:
            sp.update_time_marker(time_s)
        self._highlight_label_at_time(time_s)

    def _update_video_blanking(self, time_s: float) -> None:
        """Black out the camera views when the marker has no video under it.

        Session basis only: in an inter-trial gap, or inside another trial's
        span while that trial's video hasn't loaded yet, the views show "no
        input" (black cover) instead of freezing on the last frame. As soon
        as the marker's trial is the loaded one, the cover lifts.
        """
        state = self.app_state
        blank = False
        if state.display_basis == "session" and getattr(state, "video", None) is not None:
            hit = state.from_display(time_s, strict=True)
            blank = hit is None or hit[0] != state.trials_sel

        views = [self.shell.video_area.primary, *self.video_mgr.extra_widgets.values()]
        for view in views:
            if getattr(view, "static_image_path", None):
                continue  # still images are timeless — never blanked
            if hasattr(view, "set_blanked") and view.is_blanked != blank:
                view.set_blanked(blank)

    def _follow_marker_trial(self):
        """Debounce target: make the trial under the marker current (session basis).

        Strict resolution — a marker resting in an inter-trial gap follows
        nothing. The view is preserved and the marker is restored to where the
        user put it (``on_trial_changed`` would otherwise reset it to the
        trial start).
        """
        state = self.app_state
        self._follow_pending_trial = None
        if state.display_basis != "session" or not state.ready:
            return
        time_s = getattr(self, "_follow_pending_time", None)
        if time_s is None:
            return
        hit = state.from_display(time_s, strict=True)
        if hit is None or hit[0] == state.trials_sel:
            return
        trial_id = hit[0]
        old_video = getattr(state, "video", None)
        was_playing = bool(old_video is not None and old_video.is_playing)
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
        # Land the video (and marker) on the followed time, not the trial start.
        self.plot_container.update_time_marker_by_time(time_s)
        video = getattr(state, "video", None)
        if video is not None:
            video.seek_to_frame(video.time_to_frame(time_s, round_nearest=True))
            if was_playing:
                # Playback survived the trial hop: the new decoder streams
                # frames in as they arrive ("freeze until loaded, then go").
                video.start()

    def _on_xrange_for_space_plot(self, _time_s: float):
        """Debounced re-render of space plots when lineplot x-range changes."""
        for sp in self.space_plots:
            if sp.isVisible():
                sp.on_xrange_changed()

    _space_highlight_key: tuple | None = None

    def _clear_space_highlight(self):
        """Drop the sticky label highlight on every space plot.

        Without this, a highlight applied while the marker was inside a label
        would be re-applied by windowed re-renders forever after leaving it.
        """
        if self._space_highlight_key is None:
            return
        self._space_highlight_key = None
        for sp in self.space_plots:
            if sp.isVisible():
                sp.clear_time_highlight()

    def _highlight_label_at_time(self, time_s: float):
        """If the current time falls inside a label, highlight that segment.

        Only redraws when entering a different label interval.
        """
        label_intervals = self._subject_intervals(self.app_state.get_display_intervals())
        if label_intervals is None or label_intervals.empty:
            self._clear_space_highlight()
            return
        mask = (label_intervals["onset_s"] <= time_s) & (label_intervals["offset_s"] >= time_s)
        hits = label_intervals[mask]
        if hits.empty:
            self._clear_space_highlight()
            return
        row = hits.iloc[0]
        key = (float(row["onset_s"]), float(row["offset_s"]), int(row["labels"]))
        if key == self._space_highlight_key:
            return
        self._space_highlight_key = key

        color = (255, 102, 0)
        mappings = getattr(self.labels_widget, "_mappings", {})
        color = mappings.get(key[2], {}).get("color", color)
        for sp in self.space_plots:
            if sp.isVisible():
                sp.highlight_time_segment(key[0], key[1], color)

    def _on_primary_frame_changed(self, frame_number: int):
        self.plot_container.update_time_marker_and_window(frame_number)

        video = getattr(self.app_state, "video", None)
        if video:
            current_time = video.frame_to_time(frame_number)
        else:
            current_time = self.app_state.to_display(
                getattr(self.app_state, "trials_sel", None), frame_number / self.app_state.video_fps
            )

        xlim = self.plot_container.get_current_xlim()
        if getattr(self.app_state, "center_playback", False) or current_time < xlim[0] or current_time > xlim[1]:
            self.plot_container.set_x_range(mode="center", center_on_frame=frame_number)

    def update_pose(self):
        """Refresh primary and extra camera pose layers through PoseDisplayManager.

        No master on/off gate here: marker and skeleton visibility are the
        "Show" checkboxes in the Pose section. (The old napari-era
        ``pose_markers_visible`` flag lost its checkbox in a refactor and then
        silently blocked the whole pose pipeline for any dataset whose local
        settings had it persisted as False.)
        """
        if self.pose_mgr is None:
            return
        if not hasattr(self.app_state, "trials_sel") or self.app_state.trials_sel is None:
            return
        self.pose_mgr.update_pose(self.get_hidden_keypoints())

    def cleanup(self) -> None:
        """Release what a loaded session holds: caches, the video, the cycles.

        Qt delivers ``closeEvent`` only to the window, never to its children,
        so this has to be driven from :meth:`EthographMainWindow.closeEvent`.
        Covered by tests/test_unit/test_session_cleanup.py.
        """
        SharedAudioCache.clear_cache()
        if getattr(self.app_state, "video", None):
            self.app_state.video.stop()
        dt = getattr(self.app_state, "dt", None)
        if dt is not None:
            dt.close()
        # A closed window's widgets, plots and buffers reference each other, so
        # refcounting alone frees none of them — a session that builds and
        # closes windows grows by ~80 MB each time until a large native (HDF5)
        # allocation fails with an access violation. Only a generational pass
        # breaks those cycles.
        gc.collect()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def _on_space_view_changed(self, text):
        if not self.app_state.ready:
            return
        self.app_state.space_plot_type = text

        if text == "Space Plot":
            if self.space_plots:
                self.update_space_plot()
            else:
                self.add_space_plot()
        else:
            for sp in self.space_plots:
                sp.hide()

    def _on_primary_camera_changed(self, camera_name):
        if not self.app_state.ready or not camera_name:
            return
        self.app_state.primary_camera_previous = self.app_state.primary_camera
        self.app_state.primary_camera = camera_name
        # The alignment is computed for the primary camera's stream, so it is
        # stale the moment that camera changes — without this the new primary
        # would be clipped and offset by the previous camera's numbers and
        # drift out of sync with the other camera views.
        self._build_trial_alignment(self.app_state.trials_sel)
        self.update_video()
        self.update_pose()

    def _on_extra_camera_combo_changed(self, combo_idx: int):
        if not self.app_state.ready:
            return
        self._apply_extra_cameras()
        self._save_extra_cameras()

    def _apply_extra_cameras(self):
        desired = self._get_desired_extra_cameras()
        # Compare by camera name (dict keys are unique per view instance —
        # duplicates of the same camera are allowed). Static-image views are
        # not cameras — the combos never name them, so reconciling against
        # `desired` would wrongly remove them.
        current = {
            getattr(view, "camera_name", key)
            for key, view in self.video_mgr.extra_widgets.items()
            if not getattr(view, "static_image_path", None)
        }

        for name in current - desired:
            self.video_mgr.remove_camera(name)
            if self.pose_mgr is not None:
                self.pose_mgr.on_camera_removed(name)

        to_add: dict[str, str] = {}
        for name in desired - current:
            video_path = self.video_mgr._resolve_video_path(name, self.app_state.video_folder)
            if video_path:
                to_add[name] = video_path

        from .video_manager import VideoManager

        readers = VideoManager.open_readers_parallel(to_add)

        for name, video_path in to_add.items():
            self.video_mgr.add_camera(
                camera_name=name,
                video_path=video_path,
                layout_mgr=self.layout_mgr,
                meta_widget=self.meta_widget,
                reader=readers.get(name),
            )
            if self.pose_mgr is not None:
                self.pose_mgr.update_extra_camera_pose(name, self.get_hidden_keypoints())

        # Video panels opened/closed → reconcile background proxy jobs.
        self.video_mgr.sync_proxies()
        self._restore_camera_pins()

    def _on_camera_view_removed(self, view):
        """A camera view was removed (its dock's ✕, or programmatically).

        When it was the LAST view of that camera, drop its pose layers and
        clear any extra-camera combo still naming it — otherwise the next
        combo re-apply would resurrect the view."""
        name = getattr(view, "camera_name", None)
        if not name or self.video_mgr.views_for_camera(name):
            return
        if self.pose_mgr is not None:
            self.pose_mgr.on_camera_removed(name)
        combos = getattr(self, "_extra_camera_combos", [])
        changed = False
        for combo in combos:
            if combo.currentText() == name:
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
                changed = True
        if changed:
            self._save_extra_cameras()
        # A video panel closed → cancel its now-orphaned proxy job.
        self.video_mgr.sync_proxies()

    def _get_desired_extra_cameras(self) -> set[str]:
        if not hasattr(self, "_extra_camera_combos"):
            return set()
        names = set()
        for combo in self._extra_camera_combos:
            text = combo.currentText()
            if text and text != "None":
                names.add(text)
        return names

    def _save_extra_cameras(self):
        if not hasattr(self, "_extra_camera_combos"):
            return
        values = [combo.currentText() for combo in self._extra_camera_combos]
        self.app_state.extra_cameras = [v for v in values if v and v != "None"]

    def _update_extra_camera_combo_items(self, cameras: list[str]):
        if not hasattr(self, "_extra_camera_combos"):
            return
        cam_names = [str(c) for c in cameras]
        for combo in self._extra_camera_combos:
            prev_text = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(["None"] + cam_names)
            idx = combo.findText(prev_text)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    def _init_or_update_extra_cameras(self):
        # Reload the views that actually exist first. Most are created by
        # dragging a camera out of the add-panel popup, which touches no combo,
        # so anything driven off `_extra_camera_combos` alone would leave them
        # showing whichever trial was open when they were added.
        self.video_mgr.refresh_extra_videos()

        if not hasattr(self, "_extra_camera_combos"):
            return
        # The combos only still create views: the saved `extra_cameras` restore
        # path. A camera that already has a view was handled above.
        desired = {name for name in self._get_desired_extra_cameras() if not self.video_mgr.views_for_camera(name)}
        to_add: dict[str, str] = {}
        for camera_name in desired:
            video_path = self.video_mgr._resolve_video_path(camera_name, self.app_state.video_folder)
            if video_path:
                to_add[camera_name] = video_path

        from .video_manager import VideoManager

        readers = VideoManager.open_readers_parallel(to_add)

        for camera_name, video_path in to_add.items():
            self.video_mgr.add_camera(
                camera_name=camera_name,
                video_path=video_path,
                layout_mgr=self.layout_mgr,
                meta_widget=self.meta_widget,
                reader=readers.get(camera_name),
            )
            if self.pose_mgr is not None:
                self.pose_mgr.update_extra_camera_pose(camera_name, self.get_hidden_keypoints())

    @property
    def space_plot(self) -> SpacePlot | None:
        """The active space-plot instance (backwards-compat accessor for
        shortcuts / settings code that acts on "the" space plot)."""
        if self.active_space_plot is not None and self.active_space_plot in self.space_plots:
            return self.active_space_plot
        return self.space_plots[-1] if self.space_plots else None

    def _space_store(self):
        store = getattr(self.app_state, "data_loader", None)
        if store is None and self.app_state.ds is not None:
            from ethograph.io.catalog import XarrayLoader, catalog_from_xarray

            cat = catalog_from_xarray(self.app_state.ds, self.app_state.dt)
            store = XarrayLoader(self.app_state.ds, cat)
        return store

    def add_space_plot(
        self,
        feature: str | None = None,
        view_3d: bool | None = None,
        focus: bool = True,
        default_width: bool = True,
    ) -> SpacePlot:
        """Create a new space-plot panel. Space plots are instances like line
        plots: any number can be open at once, each in its own dock.

        ``default_width=False`` skips the deferred 20%-of-window dock resize —
        used when a saved layout is about to dictate the dock geometry."""
        sp = SpacePlot(self.shell, self.app_state)
        sp._apply_default_width = default_width
        sp.set_plot_container(self.plot_container)
        sp.closed.connect(self.remove_space_plot)
        sp.view_changed.connect(self._on_space_plot_view_changed)
        self.space_plots.append(sp)
        # Canonical objectName BEFORE the dock exists: dock creation tries
        # shell.restoreDockWidget() with it, so a saved window state places
        # the dock exactly like any other late-created dock.
        self._canonicalize_space_dock_names()
        sp.dock_object_name = f"SpacePlotDock_{len(self.space_plots) - 1}"

        if not self._space_signals_connected:
            self._space_signals_connected = True
            self.plot_container.time_marker_updated.connect(self._on_xrange_for_space_plot)
            self.app_state.space_sync_views_changed.connect(self._on_space_sync_views_toggled)
            if self.labels_widget:
                self.labels_widget.highlight_spaceplot.connect(self._highlight_positions_in_space_plot)

        # Each instance's X/Y/Z + 3D controls live in the sidebar's Space
        # context; only the active instance's controls are shown.
        ps = getattr(self, "plot_settings_widget", None)
        if ps is not None and getattr(ps, "spaceplot_panel", None) is not None:
            ps.spaceplot_panel.layout().insertWidget(0, sp.controls_widget)
        # Register with the active-panel manager (green edge + Space context
        # on click), like every other panel type.
        mgr = getattr(self.meta_widget, "active_panels", None)
        if mgr is not None:
            mgr.register(sp, "space", clicked_signal=sp.clicked)

        sp.set_store(self._space_store())
        sp.show()
        if feature is not None or view_3d is not None:
            sp.configure(feature=feature, view_3d=view_3d)
        else:
            sp.refresh()

        # With view sync on, a new plot adopts the coordinate frame of the
        # most recent open plot of the same type (2D/3D) instead of its own
        # auto-range.
        if getattr(self.app_state, "space_sync_views", True):
            mode = "3d" if sp.is_3d else "2d"
            for other in reversed(self.space_plots[:-1]):
                state = other.capture_view_state()
                if state is not None and state.get("mode") == mode:
                    sp.apply_view_state(state)
                    break

        self.set_active_space_plot(sp)
        if focus and self.meta_widget is not None and hasattr(self.meta_widget, "_on_plot_focus"):
            self.meta_widget._on_plot_focus("space")
        return sp

    def _on_space_plot_view_changed(self, source: SpacePlot):
        """Mirror an interactive view change (zoom/pan/orbit) onto every other
        open space plot of the same type — ``apply_view_state`` no-ops on a
        2D/3D mismatch, so 2D and 3D plots never sync with each other."""
        if not getattr(self.app_state, "space_sync_views", True):
            return
        state = source.capture_view_state()
        if state is None:
            return
        for sp in self.space_plots:
            if sp is not source:
                sp.apply_view_state(state)

    def _on_space_sync_views_toggled(self, *_args):
        """Turning sync ON immediately aligns all plots to the active one."""
        if getattr(self.app_state, "space_sync_views", True) and self.space_plot is not None:
            self._on_space_plot_view_changed(self.space_plot)

    def _canonicalize_space_dock_names(self):
        """Name space docks by list position so the shell's saveState /
        restoreDockWidget can match them across sessions."""
        for i, sp in enumerate(self.space_plots):
            if sp.dock_widget is not None:
                sp.dock_widget.setObjectName(f"SpacePlotDock_{i}")

    def space_layout_state(self) -> list[dict]:
        """Serializable state of all open space plots (stored in
        ``app_state.panel_layout["space_plots"]``)."""
        self._canonicalize_space_dock_names()
        return [sp.space_settings() for sp in self.space_plots]

    def apply_space_layout_state(self, entries) -> None:
        if not isinstance(entries, list):
            return
        # Reuse existing instances (the load may have auto-created one via
        # update_space_plot): destroying and recreating live GL views
        # (space_3d) mid-load can crash natively on Windows.
        while len(self.space_plots) > len(entries):
            self.remove_space_plot(self.space_plots[-1])
        while len(self.space_plots) < len(entries):
            self.add_space_plot(focus=False, default_width=False)
        for sp, e in zip(self.space_plots, entries):
            sp._apply_default_width = False  # saved layout owns the size now
            sp.apply_space_settings(e)
        if entries:
            self._space_plot_autocreated = True

    def remove_space_plot(self, sp: SpacePlot):
        """Drop a space-plot instance (its dock was closed)."""
        if sp not in self.space_plots:
            return
        self.space_plots.remove(sp)
        mgr = getattr(self.meta_widget, "active_panels", None)
        if mgr is not None:
            mgr.unregister(sp)
        controls = sp.controls_widget
        if controls is not None:
            controls.setParent(None)
        dock = sp.dock_widget
        if dock is not None:
            # hide + deleteLater, NOT shell.removeDockWidget(): removing a
            # dock whose widget holds a GL view leaves Qt in a state where the
            # next shell.show() crashes natively (access violation) on Windows.
            dock.hide()
            dock.deleteLater()
        sp.deleteLater()
        if self.active_space_plot is sp:
            self.set_active_space_plot(self.space_plots[-1] if self.space_plots else None)

    # ------------------------------------------------------------------
    # Radial (compass) plots — instances, exactly like space plots
    # ------------------------------------------------------------------

    def add_radial_plot(self, feature: str | None = None, focus: bool = True, default_width: bool = True):
        """Create a new radial-plot panel showing one heading as an arrow."""
        rp = RadialPlot(self.shell, self.app_state)
        rp._apply_default_width = default_width
        rp.closed.connect(self.remove_radial_plot)
        self.radial_plots.append(rp)
        self._canonicalize_radial_dock_names()
        rp.dock_object_name = f"RadialPlotDock_{len(self.radial_plots) - 1}"

        if not self._radial_signals_connected:
            self._radial_signals_connected = True
            # A compass shows one instant, so it follows the time marker rather
            # than an x-range.
            self.plot_container.time_marker_updated.connect(self._on_time_for_radial_plots)

        ps = getattr(self, "plot_settings_widget", None)
        if ps is not None and getattr(ps, "radialplot_panel", None) is not None:
            ps.radialplot_panel.layout().insertWidget(0, rp.controls_widget)
        mgr = getattr(self.meta_widget, "active_panels", None)
        if mgr is not None:
            mgr.register(rp, "radial", clicked_signal=rp.clicked)

        rp.set_store(self._space_store())
        rp.show()
        rp.configure(feature=feature)

        self.set_active_radial_plot(rp)
        if focus and self.meta_widget is not None and hasattr(self.meta_widget, "_on_plot_focus"):
            self.meta_widget._on_plot_focus("radial")
        return rp

    def remove_radial_plot(self, rp):
        if rp not in self.radial_plots:
            return
        self.radial_plots.remove(rp)
        mgr = getattr(self.meta_widget, "active_panels", None)
        if mgr is not None:
            mgr.unregister(rp)
        if rp.controls_widget is not None:
            rp.controls_widget.setParent(None)
        dock = rp.dock_widget
        if dock is not None:
            dock.hide()
            dock.deleteLater()
        rp.dock_widget = None
        rp.deleteLater()
        if self.active_radial_plot is rp:
            self.set_active_radial_plot(self.radial_plots[-1] if self.radial_plots else None)
        self._canonicalize_radial_dock_names()

    def set_active_radial_plot(self, rp) -> None:
        self.active_radial_plot = rp
        for other in self.radial_plots:
            other.controls_widget.setVisible(other is rp)

    def _canonicalize_radial_dock_names(self):
        for i, rp in enumerate(self.radial_plots):
            if rp.dock_widget is not None:
                rp.dock_widget.setObjectName(f"RadialPlotDock_{i}")

    def _on_time_for_radial_plots(self, time_s: float):
        for rp in self.radial_plots:
            if rp.isVisible():
                rp.set_time(time_s)

    def refresh_radial_plots(self) -> None:
        """Re-read the data (trial change, new derived feature, …)."""
        for rp in self.radial_plots:
            rp.set_store(self._space_store())
            rp.refresh()

    def radial_layout_state(self) -> list[dict]:
        self._canonicalize_radial_dock_names()
        return [rp.radial_settings() for rp in self.radial_plots]

    def apply_radial_layout_state(self, entries) -> None:
        if not isinstance(entries, list):
            return
        while len(self.radial_plots) > len(entries):
            self.remove_radial_plot(self.radial_plots[-1])
        while len(self.radial_plots) < len(entries):
            self.add_radial_plot(focus=False, default_width=False)
        for rp, entry in zip(self.radial_plots, entries):
            rp._apply_default_width = False
            rp.apply_radial_settings(entry)

    def set_active_space_plot(self, sp: SpacePlot | None):
        """Track the active instance and show only its controls in the
        sidebar's Space context."""
        self.active_space_plot = sp
        for other in self.space_plots:
            other.controls_widget.setVisible(other is sp)

    def update_space_plot(self):
        """Refresh every open space-plot panel; lazily create the first one
        when the saved view type asks for it."""
        if not self.app_state.ready:
            return

        plot_type = self.app_state.get_with_default("space_plot_type")
        if plot_type != "Space Plot":
            for sp in self.space_plots:
                sp.hide()
            return

        if not self.space_plots:
            if not self._space_plot_autocreated:
                self._space_plot_autocreated = True
                self.add_space_plot()
            return

        store = self._space_store()
        for sp in self.space_plots:
            sp.set_store(store)
            sp.refresh()
            sp.show()

    def _highlight_positions_in_space_plot(self, start_time: float, end_time: float):
        visible = [sp for sp in self.space_plots if sp.dock_widget is not None and sp.dock_widget.isVisible()]
        if not visible:
            return

        color = (255, 102, 0)
        label_intervals = self._subject_intervals(self.app_state.get_display_intervals())
        active_ids = self.app_state.active_label_ids
        if label_intervals is not None and not label_intervals.empty:
            mid = (start_time + end_time) / 2.0
            mask = (label_intervals["onset_s"] <= mid) & (label_intervals["offset_s"] >= mid)
            hits = label_intervals[mask]
            if not hits.empty:
                label_id = int(hits.iloc[0]["labels"])
                if active_ids is None or label_id in active_ids:
                    mappings = getattr(self.labels_widget, "_mappings", {})
                    color = mappings.get(label_id, {}).get("color", color)

        for sp in visible:
            sp.highlight_time_segment(start_time, end_time, color)
