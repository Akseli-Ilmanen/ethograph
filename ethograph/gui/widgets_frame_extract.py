"""Extract Frames: the Labels section while a pose project's videos are open.

The extract stage of a DeepLabCut / LightningPose project
(:mod:`ethograph.labels.pose_project`) is ordinary labelling with a
two-class vocabulary: **ExtractSegment** (``1``, a state event: a stretch
worth sampling frames from) and **ExtractFrame** (``2``, a point event:
this exact frame). The model's confidence and speed curves are what the
stretches are chosen from. This panel replaces the Labels body with those
two classes, the sampling choice for segments — evenly spaced, or one frame
per k-means cluster of thumbnails as DeepLabCut does — and the share of a
segment's frames to take. **Extract** writes the frames of the current
video into ``labeled-data/<video>/`` with the model's predictions as their
starting labels, ready for the Refine Pose stage.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.labels.pose_project import (
    EXTRACT_FRAME_LABEL,
    EXTRACT_SEGMENT_LABEL,
    METHOD_DIVERSE,
    METHOD_UNIFORM,
)

_METHODS = (
    (METHOD_UNIFORM, "Uniform (evenly spaced)", "Equally spaced frames within each segment — no decoding."),
    (
        METHOD_DIVERSE,
        "Diverse (k-means)",
        "One frame per k-means cluster of the segment's thumbnails — DeepLabCut's\n"
        "method: distinct postures rather than whatever the animal did most.",
    ),
)

_CLASSES = (
    (
        EXTRACT_SEGMENT_LABEL,
        "━  ExtractSegment",
        "1",
        "State event: click twice on a plot to mark a stretch worth sampling from.",
    ),
    (EXTRACT_FRAME_LABEL, "●  ExtractFrame", "2", "Point event: the frame on screen, taken as it is."),
)


class FrameExtractPanel(QWidget):
    """The two extract classes, the sampling options and the Extract button."""

    def __init__(self, app_state, labels_widget, mode, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.labels_widget = labels_widget
        self.mode = mode
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)

        hint = QLabel(
            "Mark what to extract from this video: a stretch (<b>1</b>, then two clicks on a plot) "
            "or the frame on screen (<b>2</b>). Read the confidence and speed curves to decide where."
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.RichText)
        box.addWidget(hint)

        self.table = QTableWidget(len(_CLASSES), 2)
        self.table.setHorizontalHeaderLabels(["Label", "Key"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for row, (label_id, name, key, tip) in enumerate(_CLASSES):
            item = QTableWidgetItem(name)
            item.setData(Qt.UserRole, label_id)
            item.setToolTip(tip)
            self.table.setItem(row, 0, item)
            self.table.setItem(row, 1, QTableWidgetItem(key))
        self.table.setFixedHeight(self.table.horizontalHeader().height() + self.table.rowHeight(0) * len(_CLASSES) + 4)
        self.table.cellClicked.connect(self._on_class_clicked)
        box.addWidget(self.table)

        self.extract_frame_btn = QPushButton("Extract current frame (2)")
        self.extract_frame_btn.setToolTip("Place an ExtractFrame point at the frame on screen.")
        self.extract_frame_btn.clicked.connect(lambda: self.labels_widget.place_label_now(EXTRACT_FRAME_LABEL))
        box.addWidget(self.extract_frame_btn)

        options = QGroupBox("Frames within a segment")
        form = QFormLayout(options)
        self.method_combo = QComboBox()
        for key, text, tip in _METHODS:
            self.method_combo.addItem(text, key)
            self.method_combo.setItemData(self.method_combo.count() - 1, tip, Qt.ToolTipRole)
        current = self.method_combo.findData(self.app_state.pose_extract_method)
        self.method_combo.setCurrentIndex(max(0, current))
        self.method_combo.currentIndexChanged.connect(
            lambda _i: setattr(self.app_state, "pose_extract_method", str(self.method_combo.currentData()))
        )
        form.addRow("Pick:", self.method_combo)
        self.coverage_spin = QDoubleSpinBox()
        self.coverage_spin.setRange(1.0, 100.0)
        self.coverage_spin.setDecimals(0)
        self.coverage_spin.setSuffix(" %")
        self.coverage_spin.setValue(float(self.app_state.pose_extract_coverage))
        self.coverage_spin.setToolTip("Share of each segment's frames to extract. Point events are always taken.")
        self.coverage_spin.valueChanged.connect(lambda v: setattr(self.app_state, "pose_extract_coverage", float(v)))
        form.addRow("Coverage:", self.coverage_spin)
        box.addWidget(options)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: rgba(255,255,255,150);")
        box.addWidget(self.summary)

        row = QHBoxLayout()
        self.extract_btn = QPushButton("Extract frames from this video")
        self.extract_btn.setToolTip(
            "Write the chosen frames of the current video into labeled-data/<video>/\n"
            "as PNGs, with the model's predictions as their starting labels."
        )
        self.extract_btn.clicked.connect(self.mode.extract_current_video)
        row.addWidget(self.extract_btn)
        box.addLayout(row)
        box.addStretch(1)

        self.app_state.trial_changed.connect(self.refresh_summary)
        self.app_state.label_intervals_changed.connect(self.refresh_summary)
        self.refresh_summary()

    def _on_class_clicked(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.labels_widget.activate_label(int(item.data(Qt.UserRole)))

    def refresh_summary(self, *_args) -> None:
        """How much of the current video is marked, and where it would go."""
        segments, points = self.mode.marked_frames()
        trial = self.app_state.trials_sel
        where = f"labeled-data/{trial}/" if trial is not None else "labeled-data/"
        self.summary.setText(
            f"{len(segments)} segment(s), {len(points)} single frame(s) marked in this video → {where}"
        )
        self.extract_btn.setEnabled(bool(segments or points))
