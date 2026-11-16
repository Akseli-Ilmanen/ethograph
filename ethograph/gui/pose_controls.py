"""The pose controls: one widget, shown wherever a pose is drawn.

The video's Pose section and every skeleton panel show this same widget, bound
to the same ``app_state`` variables. A widget writes its variable; each
variable's ``*_changed`` signal writes every copy back. Every surface therefore
shows the same threshold, sizes, colours, colour-by and hidden keypoints, and a
change in one view shows in all of them. Surfaces redraw from the signals,
never from a particular copy's widgets.

The keypoint filter lists the keypoints of the surface it sits on
(*keypoints*): a keypoint that only the 2D data has cannot be toggled from the
3D panel, but a keypoint hidden anywhere is hidden everywhere it exists.
"""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.dialog_keypoint_filter import KeypointFilterDialog
from ethograph.gui.notify import notify
from ethograph.gui.pose_convert import COLOR_BY_INDIVIDUAL, COLOR_BY_KEYPOINT

_POINTS_DEFAULT = "#FF3333"
_SKELETON_DEFAULT = "#00CC66"


def color_swatch_icon(color_hex: str) -> QIcon:
    """A small filled square for a colour-picker button."""
    pix = QPixmap(20, 20)
    pix.fill(QColor(color_hex))
    return QIcon(pix)


def _spin(lo: float, hi: float, step: float, decimals: int, width: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setFixedWidth(width)
    return spin


class PoseControls(QWidget):
    """Filter + Design boxes over the shared pose ``app_state`` variables."""

    def __init__(self, app_state, keypoints: Callable[[], list[str]], parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._keypoints = keypoints

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ── Filter keypoints ──
        filter_box = QGroupBox("Filter keypoints")
        filter_row = QHBoxLayout(filter_box)
        filter_row.setSpacing(5)
        filter_row.setContentsMargins(4, 4, 4, 4)
        filter_row.addWidget(QLabel("Filter below confidence:"))
        self.pose_hide_threshold_spin = _spin(0.0, 1.0, 0.1, 1, 60)
        self.pose_hide_threshold_spin.setObjectName("pose_hide_threshold_spin")
        self.pose_hide_threshold_spin.setToolTip(
            "Hide pose markers below this confidence (0.0-1.0). A skeleton edge "
            "is also hidden on any frame where one of its two keypoints is hidden."
        )
        filter_row.addWidget(self.pose_hide_threshold_spin)
        self.filter_keypoints_btn = QPushButton("Filter individual keypoints…")
        self.filter_keypoints_btn.setToolTip(
            "Show/hide individual keypoints — in every view that has them. Hidden "
            "keypoints also drop any skeleton edge touching them."
        )
        filter_row.addWidget(self.filter_keypoints_btn)
        filter_row.addStretch()
        layout.addWidget(filter_box)

        # ── Design ──
        design_box = QGroupBox("Design")
        design_vbox = QVBoxLayout(design_box)
        design_vbox.setSpacing(4)
        design_vbox.setContentsMargins(4, 4, 4, 4)
        grid = QGridLayout()
        grid.setSpacing(5)
        for col, header in enumerate(["", "Show", "Show text", "Size / width", "Base colour"]):
            grid.addWidget(QLabel(f"<b>{header}</b>" if header else ""), 0, col)

        grid.addWidget(QLabel("Points"), 1, 0)
        self.pose_show_keypoints_checkbox = QCheckBox()
        self.pose_show_keypoints_checkbox.setToolTip("Show/hide keypoint markers (skeleton edges are unaffected)")
        grid.addWidget(self.pose_show_keypoints_checkbox, 1, 1)
        self.pose_show_text_checkbox = QCheckBox()
        self.pose_show_text_checkbox.setToolTip("Show keypoint/individual labels on pose markers")
        grid.addWidget(self.pose_show_text_checkbox, 1, 2)
        self.pose_point_size_spin = _spin(1.0, 50.0, 1.0, 0, 55)
        self.pose_point_size_spin.setObjectName("pose_point_size_spin")
        grid.addWidget(self.pose_point_size_spin, 1, 3)
        points_cell = QHBoxLayout()
        points_cell.setSpacing(2)
        self.pose_points_color_btn = QPushButton()
        self.pose_points_color_btn.setObjectName("pose_points_color_btn")
        self.pose_points_color_btn.setFixedWidth(40)
        self.pose_points_color_btn.setToolTip("Uniform colour for all pose points")
        points_cell.addWidget(self.pose_points_color_btn)
        self.pose_points_use_base_checkbox = QCheckBox()
        self.pose_points_use_base_checkbox.setToolTip(
            "Checked: all points use the base colour.\nUnchecked: per-keypoint colours (turbo colormap)."
        )
        points_cell.addWidget(self.pose_points_use_base_checkbox)
        grid.addLayout(points_cell, 1, 4)

        grid.addWidget(QLabel("Skeleton"), 2, 0)
        self.pose_show_skeleton_checkbox = QCheckBox()
        self.pose_show_skeleton_checkbox.setToolTip(
            "Draw skeleton edges (from the NWB ndx-pose Skeleton or the skeleton editor)"
        )
        grid.addWidget(self.pose_show_skeleton_checkbox, 2, 1)
        self.pose_skeleton_width_spin = _spin(0.5, 20.0, 0.5, 1, 55)
        self.pose_skeleton_width_spin.setObjectName("pose_skeleton_width_spin")
        grid.addWidget(self.pose_skeleton_width_spin, 2, 3)
        skeleton_cell = QHBoxLayout()
        skeleton_cell.setSpacing(2)
        self.pose_skeleton_color_btn = QPushButton()
        self.pose_skeleton_color_btn.setObjectName("pose_skeleton_color_btn")
        self.pose_skeleton_color_btn.setFixedWidth(40)
        self.pose_skeleton_color_btn.setToolTip("Uniform colour for all skeleton edges")
        skeleton_cell.addWidget(self.pose_skeleton_color_btn)
        self.pose_skeleton_use_base_checkbox = QCheckBox()
        self.pose_skeleton_use_base_checkbox.setToolTip(
            "Checked: all edges use the base colour.\n"
            "Unchecked: per-edge custom colours from the skeleton editor / NWB."
        )
        skeleton_cell.addWidget(self.pose_skeleton_use_base_checkbox)
        grid.addLayout(skeleton_cell, 2, 4)
        grid.setColumnStretch(5, 1)
        design_vbox.addLayout(grid)
        speed_hint = QLabel("Tip: navigating between trials is faster with points and skeleton hidden.")
        speed_hint.setStyleSheet("color: grey; font-size: 10px;")
        speed_hint.setWordWrap(True)
        design_vbox.addWidget(speed_hint)

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
        color_by_row.addWidget(self.pose_color_by_combo)
        color_by_row.addStretch()
        design_vbox.addLayout(color_by_row)

        text_size_row = QHBoxLayout()
        text_size_row.addWidget(QLabel("Text size"))
        self.pose_text_size_spin = _spin(4.0, 72.0, 1.0, 0, 55)
        self.pose_text_size_spin.setObjectName("pose_text_size_spin")
        text_size_row.addWidget(self.pose_text_size_spin)
        text_size_row.addStretch()
        design_vbox.addLayout(text_size_row)
        layout.addWidget(design_box)

        # widget, its app_state variable, how to read it, how to write it back.
        spins = (
            (self.pose_hide_threshold_spin, "pose_hide_threshold"),
            (self.pose_point_size_spin, "pose_point_size"),
            (self.pose_skeleton_width_spin, "pose_skeleton_width"),
            (self.pose_text_size_spin, "pose_text_size"),
        )
        checks = (
            (self.pose_show_keypoints_checkbox, "pose_show_keypoints"),
            (self.pose_show_text_checkbox, "pose_show_text"),
            (self.pose_points_use_base_checkbox, "pose_points_use_base"),
            (self.pose_show_skeleton_checkbox, "pose_show_skeleton"),
            (self.pose_skeleton_use_base_checkbox, "skeleton_use_base"),
        )
        for spin, name in spins:
            spin.valueChanged.connect(lambda value, n=name: setattr(self.app_state, n, float(value)))
            getattr(app_state, f"{name}_changed").connect(lambda value, s=spin: self._show_spin(s, value))
            self._show_spin(spin, getattr(app_state, name))
        for box, name in checks:
            box.toggled.connect(lambda checked, n=name: setattr(self.app_state, n, bool(checked)))
            getattr(app_state, f"{name}_changed").connect(lambda value, b=box: self._show_check(b, value))
            self._show_check(box, getattr(app_state, name))

        self.pose_color_by_combo.currentIndexChanged.connect(self._on_color_by)
        app_state.pose_color_by_changed.connect(self._show_color_by)
        self._show_color_by(app_state.pose_color_by)

        self.pose_points_color_btn.clicked.connect(
            lambda: self._pick_color("pose_points_base_color", _POINTS_DEFAULT, "Points base colour")
        )
        self.pose_skeleton_color_btn.clicked.connect(
            lambda: self._pick_color("skeleton_base_color", _SKELETON_DEFAULT, "Skeleton base colour")
        )
        app_state.pose_points_base_color_changed.connect(self._show_colors)
        app_state.skeleton_base_color_changed.connect(self._show_colors)
        self._show_colors()

        self.filter_keypoints_btn.clicked.connect(self._open_filter)

    def set_keypoints_source(self, keypoints: Callable[[], list[str]]) -> None:
        """Which keypoints this view's filter lists (the surface's own)."""
        self._keypoints = keypoints

    # -- app_state → widgets --------------------------------------------------

    @staticmethod
    def _show_spin(spin: QDoubleSpinBox, value) -> None:
        if value is not None and spin.value() != float(value):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    @staticmethod
    def _show_check(box: QCheckBox, value) -> None:
        if box.isChecked() != bool(value):
            box.blockSignals(True)
            box.setChecked(bool(value))
            box.blockSignals(False)

    def _show_color_by(self, value) -> None:
        index = self.pose_color_by_combo.findData(value)
        if index >= 0 and index != self.pose_color_by_combo.currentIndex():
            self.pose_color_by_combo.blockSignals(True)
            self.pose_color_by_combo.setCurrentIndex(index)
            self.pose_color_by_combo.blockSignals(False)

    def _show_colors(self, *_args) -> None:
        self.pose_points_color_btn.setIcon(color_swatch_icon(self.app_state.pose_points_base_color or _POINTS_DEFAULT))
        self.pose_skeleton_color_btn.setIcon(color_swatch_icon(self.app_state.skeleton_base_color or _SKELETON_DEFAULT))

    # -- widgets → app_state --------------------------------------------------

    def _on_color_by(self, _index: int) -> None:
        self.app_state.pose_color_by = str(self.pose_color_by_combo.currentData())

    def _pick_color(self, name: str, default: str, title: str) -> None:
        color = QColorDialog.getColor(QColor(getattr(self.app_state, name) or default), self, title)
        if color.isValid():
            setattr(self.app_state, name, color.name().upper())

    def _open_filter(self) -> None:
        names = [str(n) for n in self._keypoints()]
        if not names:
            notify("No pose keypoints loaded yet.", "warning")
            return
        dialog = KeypointFilterDialog(names, set(self.app_state.pose_hidden_keypoints), parent=self)
        dialog.hidden_changed.connect(lambda hidden: self._on_hidden(names, hidden))
        dialog.exec()

    def _on_hidden(self, names: list[str], hidden: set) -> None:
        """This view's keypoints take the dialog's answer; others' entries are kept as they were."""
        kept = {k for k in self.app_state.pose_hidden_keypoints if k not in names}
        self.app_state.pose_hidden_keypoints = sorted(kept | {str(n) for n in hidden})
