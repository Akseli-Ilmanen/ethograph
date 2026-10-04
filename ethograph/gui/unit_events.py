"""Under the neuron table: which units' periods the review grids show.

**Tools ▸ Neural: Neuronal firing / burst detection…** read unit by unit
gives a read-only label set with one class per unit. This
area is where such a set is looked at: the units selected in the neuron table
— or Ctrl+clicked in the raster or the firing rates, or dragged in — are held
here, and the label grid or the video grid opens on just those. Empty means every unit the table's filters let through,
which is also what the set's own panel shows.

Until a per-unit set is loaded the area is greyed out and says where to make
one; the neuron table only lets rows be dragged while there is somewhere to
drop them, so sweeping over rows keeps selecting them the rest of the time.

The grids themselves are the Curation section's
(:meth:`~ethograph.gui.widgets_curation.CurationPanel.open_grid_view`), opened
on these classes instead of its scope.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from ethograph.gui.notify import notify
from ethograph.labels.feature_events import classes_along
from ethograph.labels.predictions import PredictionSet

_HINT = "Tools ▸ Neural: Neuronal firing / burst detection…"

_STYLE_OFF = "QFrame#unitEventsDrop { border: 1px dashed #666; border-radius: 3px; } QLabel { color: #777; }"
_STYLE_EMPTY = "QFrame#unitEventsDrop { border: 1px dashed #888; border-radius: 3px; } QLabel { color: #aaa; }"
_STYLE_FILLED = "QFrame#unitEventsDrop { border: 1px solid #ffe066; border-radius: 3px; }"


class UnitEventsArea(QFrame):
    """The units selected in the neuron table, and the grids opened on their periods."""

    def __init__(self, app_state, ephys: Any, unit_dim: str, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.ephys = ephys
        self.unit_dim = unit_dim
        #: The units selected or dragged in, as the set's classes name them; empty = every unit.
        self._units: list[str] = []
        self.setAcceptDrops(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(3)
        self._drop = QFrame()
        self._drop.setObjectName("unitEventsDrop")
        self._drop.setMinimumHeight(30)
        drop_lay = QHBoxLayout(self._drop)
        drop_lay.setContentsMargins(6, 2, 6, 2)
        self._label = QLabel()
        self._label.setWordWrap(True)
        drop_lay.addWidget(self._label, stretch=1)
        drop_row = QHBoxLayout()
        drop_row.setSpacing(4)
        drop_row.addWidget(self._drop, stretch=1)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip(
            "Unselect every unit: nothing highlighted in the table or the panels, and the area empty —\n"
            "the grids show every unit the table's filters let through."
        )
        self.reset_btn.clicked.connect(self.reset)
        drop_row.addWidget(self.reset_btn)
        lay.addLayout(drop_row)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        self.grid_btn = QPushButton("Label grid…")
        self.grid_btn.setToolTip("Every period of these units side by side, as plots.")
        self.grid_btn.clicked.connect(self.open_label_grid)
        buttons.addWidget(self.grid_btn)
        self.video_btn = QPushButton("Video grid…")
        self.video_btn.setToolTip("Play the periods of these units side by side (decodes video — slower to build).")
        self.video_btn.clicked.connect(self.open_video_grid)
        buttons.addWidget(self.video_btn)
        buttons.addStretch(1)
        lay.addLayout(buttons)

        # The units highlighted in the table (or Ctrl+clicked in a panel) are the ones to review.
        ephys.cluster_table.selectionModel().selectionChanged.connect(self._follow_selection)
        app_state.prediction_sets_changed.connect(self.refresh)
        app_state.label_source_path_changed.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------
    # The set being reviewed
    # ------------------------------------------------------------------

    def unit_set(self) -> PredictionSet | None:
        """The loaded set with a class per unit: the one the tools read now, else the newest."""
        sets = [s for s in self.app_state.prediction_sets if self._classes(s)]
        reading = next((s for s in sets if str(s.path) == self.app_state.label_source_path), None)
        return reading or (sets[-1] if sets else None)

    def _classes(self, prediction_set: PredictionSet | None) -> dict[str, int]:
        if prediction_set is None or prediction_set.mappings is None:
            return {}
        return classes_along(prediction_set.mappings, self.unit_dim)

    def units(self) -> list[str]:
        return list(self._units)

    def label_ids(self) -> list[int]:
        """The classes the grids open on: the dragged units', or every filtered unit's when none was dragged."""
        classes = self._classes(self.unit_set())
        units = self._units or self.ephys.firing_rate_units()
        return [classes[unit] for unit in units if unit in classes]

    def set_units(self, units: Iterable[object]) -> None:
        self._units = []
        self.add_units(units)

    def add_units(self, units: Iterable[object]) -> None:
        """Add *units* (keeping order, no duplicates); a unit the set has no class for is named and left out."""
        target = self.unit_set()
        classes = self._classes(target)
        missing = []
        for unit in map(str, units):
            if unit not in classes:
                missing.append(unit)
            elif unit not in self._units:
                self._units.append(unit)
        if missing and target is not None:
            notify(f"“{target.name}” was not made for unit(s) {', '.join(missing)}.", severity="warning")
        self.refresh()

    def _follow_selection(self, *_args) -> None:
        """Hold the units selected in the neuron table; one the set has no class for is left out, unremarked."""
        classes = self._classes(self.unit_set())
        self._units = [unit for unit in map(str, self.ephys.selected_unit_ids()) if unit in classes]
        self.refresh()

    def reset(self) -> None:
        """Unselect every unit everywhere and empty the area."""
        self.ephys._unselect_clusters()
        self.set_units([])

    def refresh(self, *_args) -> None:
        """Follow the loaded sets: greyed out with nowhere to read from, else the units in the area."""
        target = self.unit_set()
        classes = self._classes(target)
        self._units = [unit for unit in self._units if unit in classes]
        self.setEnabled(target is not None)
        # Rows are draggable only while there is somewhere to drop them.
        self.ephys.cluster_table.setDragEnabled(target is not None)
        if target is None:
            self._label.setText(f"Review periods per unit: make them first with {_HINT}")
            self._drop.setStyleSheet(_STYLE_OFF)
        elif self._units:
            self._label.setText(f"Units to review in “{target.name}”: {', '.join(self._units)}")
            self._drop.setStyleSheet(_STYLE_FILLED)
        else:
            self._label.setText(
                f"All filtered units of “{target.name}” — select units (table, or Ctrl+click a row) to narrow"
            )
            self._drop.setStyleSheet(_STYLE_EMPTY)

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------

    def open_label_grid(self):
        """Open the label grid on the units' periods; returns the dialog."""
        return self._open("open_grid_view")

    def open_video_grid(self):
        """Open the video grid on the units' periods; returns the dialog."""
        return self._open("open_video_grid")

    def _open(self, opener: str):
        target = self.unit_set()
        if target is None:
            return None
        # The grids read the label source: make it this set, whatever was being read.
        self.app_state.label_source_path = str(target.path)
        curation = self.ephys.meta_widget.labels_widget.curation_panel
        return getattr(curation, opener)(label_ids=self.label_ids())

    # ------------------------------------------------------------------
    # Drops from the neuron table
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.source() is self.ephys.cluster_table:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        if event.source() is not self.ephys.cluster_table:
            event.ignore()
            return
        # The rows stay in the table: a copy, never a move.
        event.setDropAction(Qt.CopyAction)
        event.accept()
        # A drag carries the selected rows.
        self.add_units(self.ephys.selected_unit_ids())
