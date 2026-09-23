"""Radial (compass) plot — an arrow showing a heading at the current time.

A fourth plot type alongside lineplot / heatmap / space, and like the space
plot it is **not** a time-series panel: it lives in its own shell dock, has no
x-axis to link, and shows one instant — the value under the time marker.

Columns are picked the way every other panel picks them: every dim the catalog
offers for the feature (keypoint, individual, …) gets its own combo plus an
"All" checkbox, and — as everywhere else — **at most one dim may be free**.
Pinned, the compass shows one arrow; free, it shows one arrow per value of that
dim, colour-coded with a legend, so two keypoints or two individuals can be
compared at a glance. Colours come from the shared ``MULTIDIM_COLORS``, so a
keypoint keeps the same colour as its line-plot trace. Past the end of that
palette every arrow is drawn in one colour — recycling hues would claim the
first and eleventh individual are the same, and the legend says how many
arrows there are instead.

The gate is narrow: a feature is only offered when its dims pin down to a
column whose values cover a full turn (``max - min ≈ 360`` degrees, or ``≈ 2π``
radians), which is what tells a heading apart from any other 1-D signal; the
unit is read off that same span, never assumed. Everything else the popup
simply does not offer, so a radial plot can never be pointed at data that has
no direction to show.

Which value points up is the user's call (the "Up" control in the right
sidebar): datasets differ on whether 0 means north, east, or the arena's own
reference, and there is no way to infer it from the numbers.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ethograph.io.catalog import INDIVIDUAL_DIMS
from ethograph.io.time_model import TimeRange

from .app_constants import MEDIA_VIEW_MIN_HEIGHT, MEDIA_VIEW_MIN_WIDTH, MULTIDIM_COLORS
from .plots_base import IndividualPinMixin

logger = logging.getLogger(__name__)

#: How far a variable's span may fall short of (or overshoot) a full turn and
#: still count as angular. Real headings rarely sweep exactly 360°, and a trial
#: the animal never fully turned in would otherwise be rejected.
FULL_TURN_TOLERANCE = 0.15

_FULL_TURN_DEG = 360.0
_FULL_TURN_RAD = 2.0 * np.pi


def angular_unit(values) -> str | None:
    """``"deg"``, ``"rad"``, or ``None`` if *values* are not a heading.

    The test is the span of the data, exactly as specified: a full turn in
    degrees or its radian equivalent. Nothing else about the variable is
    consulted — no name matching, no metadata.
    """
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    if finite.size < 2:
        return None
    span = float(np.max(finite) - np.min(finite))
    if abs(span - _FULL_TURN_DEG) <= _FULL_TURN_DEG * FULL_TURN_TOLERANCE:
        return "deg"
    if abs(span - _FULL_TURN_RAD) <= _FULL_TURN_RAD * FULL_TURN_TOLERANCE:
        return "rad"
    return None


def _candidate_bounds(app_state) -> list:
    """Every range worth probing, widest first.

    Each is *tried*, not trusted. "Is this variable a heading?" is a property
    of the variable, not of the viewport, so a narrow ``window_bounds`` must
    not be the only answer — but the session ranges are session-absolute while
    an xarray loader slices trial-relative time, so on a session whose trials
    carry offsets they can select nothing at all. Neither candidate is right on
    its own; whichever actually returns a full turn decides.

    A pynapple loader carrying a display offset (trial-local window) answers
    queries in display coordinates, so the session-absolute candidates are
    also offered shifted into display coordinates — again tried, not trusted.
    """
    collection = getattr(app_state, "source_collection", None)
    candidates = [
        getattr(collection, "session_range", None),
        getattr(collection, "union_range", None),
        getattr(app_state, "window_bounds", None),
    ]

    loader = getattr(app_state, "data_loader", None)
    offset = loader.display_offset() if hasattr(loader, "display_offset") else 0.0
    if offset:
        candidates += [TimeRange(c.start_s - offset, c.end_s - offset) for c in candidates[:2] if c is not None]

    out: list = []
    for candidate in candidates:
        if candidate is not None and candidate not in out:
            out.append(candidate)
    return out


def default_selections(app_state, feature: str) -> dict[str, str]:
    """Every dim the catalog offers for *feature*, pinned to its first value.

    A heading almost never lives on a bare ``(time,)`` variable: it comes with
    a keypoint or individual dim like any other feature. Requiring the *raw*
    variable to be one column hid the option from exactly the datasets that
    have headings, so the gate pins the dims first and judges the column that
    comes out — the same column a lineplot would draw.
    """
    loader = getattr(app_state, "data_loader", None)
    if loader is None or not hasattr(loader, "feature_dims"):
        return {}
    return {dim: values[0] for dim, values in loader.feature_dims(feature).items() if values}


def probe_angular_unit(app_state, feature: str, selections: dict[str, str]) -> str | None:
    """The angular unit of one *selected* column, over its whole extent."""
    loader = getattr(app_state, "data_loader", None)
    if loader is None:
        return None

    spans: list[float] = []
    for bounds in _candidate_bounds(app_state):
        plot_data = loader.select(feature, dict(selections), t0=bounds.start_s, t1=bounds.end_s)
        if plot_data is None or plot_data.data is None:
            continue
        data = np.asarray(plot_data.data, dtype=float)
        if data.ndim > 1 and data.shape[1] != 1:
            # A compass shows ONE heading. Every dim the catalog knows about is
            # already pinned, so anything still wider than a column has no
            # single direction to point in.
            logger.debug("radial: %s selects as %s, not one column", feature, data.shape)
            return None
        values = data.ravel()
        unit = angular_unit(values)
        if unit is not None:
            return unit
        finite = values[np.isfinite(values)]
        spans.append(float(np.max(finite) - np.min(finite)) if finite.size >= 2 else float("nan"))
    logger.debug("radial: %s spans %s — neither a full 360 nor 2*pi", feature, spans)
    return None


def feature_angular_unit(app_state, feature: str, selections: dict[str, str] | None = None) -> str | None:
    """The angular unit of *feature*, or ``None``.

    Used to gate the "Radial" entry in the add-panel popup, so it is only
    offered for variables that actually carry a direction. With no
    *selections*, the first value of every dim stands for the feature.
    """
    if selections is None:
        selections = default_selections(app_state, feature)
    return probe_angular_unit(app_state, feature, selections)


class RadialPlot(IndividualPinMixin, QWidget):
    """One heading, drawn as an arrow on a compass rose.

    Instances behave like space plots: any number can be open, each in its own
    dock, each rendering purely from its own controls (feature + "Up") rather
    than from any global ``*_sel`` state. The one exception is the individual
    combo, which is the panel's pin: it shows the pin, else the sidebar's
    individual, and picking a name in it pins the panel (:pyattr:`pin_changed`).
    Freeing the individual dim ("All") draws every individual and leaves the
    pin — whose labels a click on the panel creates — untouched.
    """

    #: Emitted on any mouse press → switches the sidebar to the Radial context.
    clicked = Signal()
    #: Emitted with ``self`` when the user closes this panel's dock.
    closed = Signal(object)
    #: Emitted with ``self`` when the user pinned this panel through its own
    #: individual combo; the owner re-titles it and moves the labelling subject.
    pin_changed = Signal(object)

    #: Monotonic counter so every instance's dock gets a unique objectName.
    _dock_seq = 0

    _RADIUS = 1.0
    _LIMIT = 1.35

    def __init__(self, shell, app_state):
        super().__init__()
        self.shell = shell
        self.app_state = app_state
        self.dock_widget = None
        self.dock_object_name: str | None = None
        self._dock_name = "Radial Plot"
        self._apply_default_width = True

        self._store = None
        self._unit: str | None = None
        self._time: np.ndarray | None = None
        self._values: np.ndarray | None = None  # (T, D), always degrees
        self._labels: list[str] = []
        self._cache_key: tuple | None = None
        self._t: float | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 0)
        root.setSpacing(4)

        # --- Controls (re-parented into the sidebar's Radial context) ---
        self.controls_widget = QWidget()
        controls = QVBoxLayout(self.controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)

        feature_row = QHBoxLayout()
        feature_row.setContentsMargins(0, 0, 0, 0)
        feature_row.setSpacing(6)
        feature_row.addWidget(QLabel("Feature"))
        self.feature_combo = QComboBox()
        self.feature_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        feature_row.addWidget(self.feature_combo)
        controls.addLayout(feature_row)

        # One combo per dim of the selected feature (keypoint, individual, …),
        # rebuilt whenever the feature changes — catalog-driven, like the space
        # plot's — each with an "All" checkbox that frees that dim into one
        # arrow per value.
        self._dim_rows = QVBoxLayout()
        self._dim_rows.setContentsMargins(0, 0, 0, 0)
        self._dim_rows.setSpacing(4)
        controls.addLayout(self._dim_rows)
        self._dim_combos: dict[str, QComboBox] = {}
        self._dim_all_checks: dict[str, QCheckBox] = {}

        up_row = QHBoxLayout()
        up_row.setContentsMargins(0, 0, 0, 0)
        up_row.setSpacing(6)
        up_row.addWidget(QLabel("Up ="))
        # Always degrees, whatever the data's unit: "up = 90°" is readable in a
        # way "up = 1.5708" is not, and the conversion is exact either way.
        self.up_spin = QDoubleSpinBox()
        self.up_spin.setRange(-360.0, 360.0)
        self.up_spin.setDecimals(1)
        self.up_spin.setSingleStep(15.0)
        self.up_spin.setSuffix("°")
        self.up_spin.setToolTip("The data value that points straight up")
        up_row.addWidget(self.up_spin)
        self.cw_check = QCheckBox("Clockwise")
        self.cw_check.setToolTip("Increasing angle turns clockwise (compass bearings)")
        up_row.addWidget(self.cw_check)
        up_row.addStretch()
        controls.addLayout(up_row)

        self.unit_label = QLabel("")
        self.unit_label.setStyleSheet("color: rgba(255,255,255,140);")
        controls.addWidget(self.unit_label)

        # --- Plot area ---
        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setAspectLocked(True)
        self.plot_item.hideAxis("bottom")
        self.plot_item.hideAxis("left")
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.plot_item.hideButtons()
        self.plot_item.setRange(
            xRange=(-self._LIMIT, self._LIMIT),
            yRange=(-self._LIMIT, self._LIMIT),
            padding=0,
        )
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.plot_widget, 1)

        self._rose_items: list = []
        self._arrow_items: list = []
        self._readout = pg.TextItem("", anchor=(0.5, 0.5), color=(220, 220, 220))
        self._readout.setPos(0.0, -self._LIMIT + 0.12)
        self.plot_item.addItem(self._readout)

        # Only populated when a dim is free — with one arrow the readout under
        # the rose already says everything a legend would.
        self._legend = pg.LegendItem(offset=(-8, 8), labelTextSize="8pt")
        self._legend.setParentItem(self.plot_item.getViewBox())
        self._legend.hide()

        self.feature_combo.currentIndexChanged.connect(self._on_feature_changed)
        self.up_spin.valueChanged.connect(self._on_orientation_changed)
        self.cw_check.toggled.connect(self._on_orientation_changed)

        self._draw_rose()

    # ------------------------------------------------------------------
    # Docking (mirrors SpacePlot)
    # ------------------------------------------------------------------

    def show(self):
        if not self.dock_widget:
            RadialPlot._dock_seq += 1
            seq = RadialPlot._dock_seq
            name = "Radial Plot" if seq == 1 else f"Radial Plot {seq}"
            self._dock_name = name
            self.dock_widget = self.shell.add_dock_widget(
                self, area="top", name=name, object_name=self.dock_object_name
            )
            self.setMinimumSize(MEDIA_VIEW_MIN_WIDTH, MEDIA_VIEW_MIN_HEIGHT)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            if not self.dock_widget.restored_from_state and self._apply_default_width:
                QTimer.singleShot(0, self._apply_default_dock_width)
            self.dock_widget.installEventFilter(self)
            self.refresh_title()
        else:
            self.dock_widget.setVisible(True)
        super().show()

    def refresh_title(self) -> None:
        """Re-title the dock: its name, then which individual it shows and why."""
        if self.dock_widget is not None:
            self.dock_widget.setWindowTitle(self._dock_name + self.app_state.panel_mode_suffix(self))

    # ------------------------------------------------------------------
    # The panel's individual
    # ------------------------------------------------------------------

    def set_pinned_individual(self, individual: str | None) -> None:
        super().set_pinned_individual(individual)
        self.sync_individual(unfree=individual is not None)

    def sync_individual(self, *, unfree: bool = False) -> None:
        """Point the individual combo at this panel's individual (its pin, else the sidebar's).

        A freed individual dim ("All") is left free unless *unfree*: an
        explicit pin names one individual to draw.
        """
        target = self.app_state.panel_individual(self)
        if target is None:
            return
        changed = False
        for dim, combo in self._dim_combos.items():
            if dim not in INDIVIDUAL_DIMS:
                continue
            check = self._dim_all_checks[dim]
            if check.isChecked():
                if not unfree:
                    continue
                check.blockSignals(True)
                check.setChecked(False)
                check.blockSignals(False)
                combo.setEnabled(True)
                changed = True
            index = combo.findText(str(target))
            if index >= 0 and combo.currentIndex() != index:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
                changed = True
        if changed:
            self._on_selection_changed()

    def _pin_through_combo(self, individual: str) -> None:
        """Picking an individual in the panel's own combo pins the panel to it."""
        if individual == self.app_state.panel_individual(self):
            return
        IndividualPinMixin.set_pinned_individual(self, individual)
        self.refresh_title()
        self.pin_changed.emit(self)

    def _apply_default_dock_width(self):
        dock = self.dock_widget
        try:
            if not self._apply_default_width:
                return
            if dock is None or dock.isFloating() or self.shell.dockWidgetArea(dock) == Qt.NoDockWidgetArea:
                return
            self.shell.resizeDocks([dock], [int(self.shell.width() * 0.2)], Qt.Horizontal)
        except RuntimeError:
            pass  # dock's C++ object deleted before the timer fired

    def eventFilter(self, obj, event):
        if obj is self.dock_widget and event.type() == event.Type.Close:
            self.closed.emit(self)
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_store(self, store) -> None:
        self._store = store
        self.refresh_features()

    def radial_features(self) -> list[str]:
        """Features this plot can show: 1-D and spanning a full turn."""
        store = self._store
        catalog = getattr(store, "catalog", None) if store is not None else None
        if catalog is None:
            return []
        return [f for f in catalog.feature_choices() if feature_angular_unit(self.app_state, f) is not None]

    def refresh_features(self) -> None:
        wanted = self.radial_features()
        current = self.feature_combo.currentText()
        # A trial change re-enters here; rebuilding from scratch would silently
        # re-pin a dim the user had freed. No combos yet means no state to keep.
        preset = self.selections() if self._dim_combos else None
        self.feature_combo.blockSignals(True)
        self.feature_combo.clear()
        self.feature_combo.addItems(wanted)
        index = self.feature_combo.findText(current)
        self.feature_combo.setCurrentIndex(index if index >= 0 else 0)
        self.feature_combo.blockSignals(False)
        self._rebuild_dim_combos(preset)
        self._invalidate()

    def configure(self, feature: str | None = None, selections: dict | None = None) -> None:
        if feature:
            index = self.feature_combo.findText(feature)
            if index >= 0:
                self.feature_combo.blockSignals(True)
                self.feature_combo.setCurrentIndex(index)
                self.feature_combo.blockSignals(False)
        self._rebuild_dim_combos(selections)
        self._invalidate()
        self.refresh()

    def _on_feature_changed(self, _index=None):
        self._rebuild_dim_combos()
        self._invalidate()
        self.refresh()

    def _on_selection_changed(self, _index=None):
        self._invalidate()
        self.refresh()

    def _on_dim_combo_changed(self, dim: str) -> None:
        self._on_selection_changed()
        combo = self._dim_combos.get(dim)
        if dim in INDIVIDUAL_DIMS and combo is not None and combo.currentText():
            self._pin_through_combo(combo.currentText())

    # ------------------------------------------------------------------
    # Dim combos
    # ------------------------------------------------------------------

    def _feature_dims(self) -> dict[str, list[str]]:
        feature = self.feature_combo.currentText()
        store = self._store
        if store is None or not feature or not hasattr(store, "feature_dims"):
            return {}
        return store.feature_dims(feature)

    def _clear_dim_combos(self) -> None:
        while self._dim_rows.count():
            item = self._dim_rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._dim_combos = {}
        self._dim_all_checks = {}

    def _rebuild_dim_combos(self, preset: dict | None = None) -> None:
        """One row per dim: value combo + "All".

        *preset* is a selections dict as saved by :meth:`radial_settings` —
        a dim missing from it is a dim that was left free, which is exactly the
        "absence means All" convention the feature panels use. ``None`` means
        "no saved state", where every dim starts pinned.
        """
        self._clear_dim_combos()
        # The individual is never one of the panel's own selections: it is
        # the pin, or the sidebar's individual for a panel that follows it.
        individual = self.app_state.panel_individual(self)
        for dim, values in self._feature_dims().items():
            if not values:
                continue
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            layout.addWidget(QLabel(dim.capitalize()))

            combo = QComboBox()
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            combo.addItems(values)
            preset_value = (preset or {}).get(dim, "")
            # A saved layout that left the individual dim free keeps it free.
            if dim in INDIVIDUAL_DIMS and individual is not None and (preset is None or dim in preset):
                preset_value = individual
            index = combo.findText(str(preset_value))
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.currentIndexChanged.connect(lambda _i, d=dim: self._on_dim_combo_changed(d))
            layout.addWidget(combo)

            all_check = QCheckBox("All")
            all_check.setToolTip(f"One arrow per {dim}, colour-coded")
            is_free = preset is not None and dim not in preset
            all_check.setChecked(is_free)
            combo.setEnabled(not is_free)
            all_check.toggled.connect(lambda checked, d=dim: self._on_all_toggled(d, checked))
            layout.addWidget(all_check)

            self._dim_rows.addWidget(row)
            self._dim_combos[dim] = combo
            self._dim_all_checks[dim] = all_check

    def _on_all_toggled(self, dim: str, checked: bool) -> None:
        """Freeing a dim pins every other one — at most ONE free dim, the same
        invariant every feature panel keeps: two free dims have no single
        column to draw."""
        if checked:
            for other, check in self._dim_all_checks.items():
                if other != dim and check.isChecked():
                    check.blockSignals(True)
                    check.setChecked(False)
                    check.blockSignals(False)
                    self._dim_combos[other].setEnabled(True)
        self._dim_combos[dim].setEnabled(not checked)
        self._on_selection_changed()

    def selections(self) -> dict[str, str]:
        """The pinned dims. A dim left free ("All") is simply absent."""
        return {
            dim: combo.currentText()
            for dim, combo in self._dim_combos.items()
            if combo.currentText() and not self._dim_all_checks[dim].isChecked()
        }

    def _pinned_selections(self) -> dict[str, str]:
        """:meth:`selections` with the free dim pinned to its first value.

        The unit is a property of the variable, so it is judged on one column
        even when several are drawn — probing a free dim would see a ``(T, D)``
        block and refuse to answer.
        """
        pinned = self.selections()
        for dim, combo in self._dim_combos.items():
            if dim not in pinned and combo.count():
                pinned[dim] = combo.itemText(0)
        return pinned

    def _on_orientation_changed(self, _value=None):
        self._draw_rose()
        self._draw_arrow()

    def _invalidate(self):
        self._cache_key = None

    def _ensure_data(self) -> bool:
        """Load the whole window once and index into it per frame — a select()
        per time-marker tick would re-query the backend at frame rate."""
        feature = self.feature_combo.currentText()
        store = self._store
        bounds = getattr(self.app_state, "window_bounds", None)
        if not feature or store is None or bounds is None:
            self._time = self._values = None
            return False

        selections = self.selections()
        key = (
            feature,
            tuple(sorted(selections.items())),
            bounds.start_s,
            bounds.end_s,
            getattr(self.app_state, "trials_sel", None),
        )
        if key == self._cache_key and self._values is not None:
            return True

        plot_data = store.select(feature, dict(selections), t0=bounds.start_s, t1=bounds.end_s)
        if plot_data is None or plot_data.data is None:
            self._time = self._values = None
            self._cache_key = None
            return False

        data = np.asarray(plot_data.data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        # The unit is a property of the selected column, judged over its whole
        # extent. Re-deriving it from this window would blank the arrow as soon
        # as the user zoomed into a stretch without a full turn in it.
        unit = probe_angular_unit(self.app_state, feature, self._pinned_selections())
        self._unit = unit
        self._values = np.degrees(data) if unit == "rad" else data
        self._time = np.asarray(plot_data.time, dtype=float)
        self._labels = self._column_labels(plot_data, data.shape[1])
        self._cache_key = key
        self.unit_label.setText(f"{feature} — {unit or 'not angular'}")
        return unit is not None

    def _column_labels(self, plot_data, n_columns: int) -> list[str]:
        labels = list(getattr(plot_data, "dim_labels", None) or [])
        if len(labels) == n_columns:
            return [str(label) for label in labels]
        if n_columns == 1:
            return [self.feature_combo.currentText()]
        return [str(i) for i in range(n_columns)]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _screen_angle(self, value_deg: float) -> float:
        """Data value → screen angle in degrees, CCW from the +x axis.

        ``up`` is the value that points straight up (screen 90°); the
        clockwise flag chooses the handedness, which no amount of offset can
        express on its own.
        """
        sign = -1.0 if self.cw_check.isChecked() else 1.0
        return 90.0 + sign * (value_deg - self.up_spin.value())

    def _draw_rose(self):
        for item in self._rose_items:
            self.plot_item.removeItem(item)
        self._rose_items.clear()

        angles = np.linspace(0, 2 * np.pi, 181)
        circle = pg.PlotCurveItem(
            self._RADIUS * np.cos(angles),
            self._RADIUS * np.sin(angles),
            pen=pg.mkPen((130, 130, 130), width=1),
        )
        self.plot_item.addItem(circle)
        self._rose_items.append(circle)

        # Ticks every 30°, labelled every 90°, in the data's own values so the
        # ring reads as the variable rather than as screen geometry.
        for step in range(0, 360, 30):
            value = self.up_spin.value() + step
            screen = np.radians(self._screen_angle(value))
            major = step % 90 == 0
            inner = 0.88 if major else 0.94
            tick = pg.PlotCurveItem(
                [inner * np.cos(screen), self._RADIUS * np.cos(screen)],
                [inner * np.sin(screen), self._RADIUS * np.sin(screen)],
                pen=pg.mkPen((130, 130, 130), width=2 if major else 1),
            )
            self.plot_item.addItem(tick)
            self._rose_items.append(tick)
            if major:
                shown = value if self._unit != "rad" else np.radians(value)
                text = f"{shown:.0f}" if self._unit != "rad" else f"{shown:.2f}"
                label = pg.TextItem(text, anchor=(0.5, 0.5), color=(170, 170, 170))
                label.setPos(1.18 * np.cos(screen), 1.18 * np.sin(screen))
                self.plot_item.addItem(label)
                self._rose_items.append(label)

    def _draw_arrow(self):
        for item in self._arrow_items:
            self.plot_item.removeItem(item)
        self._arrow_items.clear()
        self._legend.clear()

        values = self.current_values()
        if not values:
            self._legend.hide()
            self._readout.setText("—")
            return

        # Past the palette, colour stops identifying anything: recycling hues
        # would claim the 1st and 11th individual are the same. Every arrow
        # goes one colour instead, and the legend says how many there are.
        distinct = len(values) <= len(MULTIDIM_COLORS)

        for index, (label, value) in enumerate(values):
            color = pg.mkColor(MULTIDIM_COLORS[index] if distinct else MULTIDIM_COLORS[0])
            screen = np.radians(self._screen_angle(value))
            tip = (0.82 * np.cos(screen), 0.82 * np.sin(screen))
            shaft = pg.PlotCurveItem([0.0, tip[0]], [0.0, tip[1]], pen=pg.mkPen(color, width=3))
            self.plot_item.addItem(shaft)
            self._arrow_items.append(shaft)
            # pyqtgraph's ArrowItem angle is measured clockwise from +x and
            # points *towards* its position, hence the negation and 180° flip.
            head = pg.ArrowItem(
                angle=180.0 - np.degrees(screen),
                headLen=18,
                tipAngle=32,
                brush=pg.mkBrush(color),
                pen=None,
            )
            head.setPos(*tip)
            self.plot_item.addItem(head)
            self._arrow_items.append(head)
            if len(values) > 1 and distinct:
                # The legend carries the value too, so several arrows stay
                # readable without a readout line per arrow.
                self._legend.addItem(
                    pg.PlotDataItem(pen=pg.mkPen(color, width=3)),
                    f"{label}: {self._format(value)}",
                )

        if len(values) > 1 and not distinct:
            free_dim = self._free_dim() or "value"
            self._legend.addItem(
                pg.PlotDataItem(pen=pg.mkPen(pg.mkColor(MULTIDIM_COLORS[0]), width=3)),
                f"{len(values)} {free_dim}s",
            )

        self._legend.setVisible(len(values) > 1)
        self._readout.setText(self._format(values[0][1]) if len(values) == 1 else "")

    def _free_dim(self) -> str | None:
        """The dim left on "All", if any — at most one can be."""
        return next((dim for dim, check in self._dim_all_checks.items() if check.isChecked()), None)

    def _format(self, value_deg: float) -> str:
        """A data value, in the data's own unit."""
        if self._unit == "rad":
            return f"{np.radians(value_deg):.2f} rad"
        return f"{value_deg:.1f}°"

    def current_values(self) -> list[tuple[str, float]]:
        """``(label, degrees)`` per drawn arrow, under the time marker.

        One entry per value of the free dim, or a single entry when every dim
        is pinned. Columns whose value is not finite are dropped rather than
        drawn pointing nowhere.
        """
        if not self._ensure_data() or self._t is None:
            return []
        time, values = self._time, self._values
        if time is None or values is None or len(time) == 0:
            return []
        index = int(np.clip(np.searchsorted(time, self._t), 0, len(time) - 1))
        row = np.atleast_1d(np.asarray(values)[index])
        labels = self._labels or [""] * row.size
        return [(labels[i], float(v)) for i, v in enumerate(row) if np.isfinite(v)]

    def current_value(self) -> float | None:
        """The first drawn heading, in degrees (``None`` if unknown)."""
        values = self.current_values()
        return values[0][1] if values else None

    def set_time(self, t: float) -> None:
        self._t = t
        self._draw_arrow()

    def refresh(self) -> None:
        self._ensure_data()
        self._draw_rose()
        self._draw_arrow()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def radial_settings(self) -> dict:
        return {
            "feature": self.feature_combo.currentText() or None,
            "selections": self.selections(),
            "up": self.up_spin.value(),
            "clockwise": self.cw_check.isChecked(),
            **({"individual": self.pinned_individual} if self.pinned_individual else {}),
        }

    def apply_radial_settings(self, settings: dict) -> None:
        for widget, setter, value in (
            (self.up_spin, self.up_spin.setValue, settings.get("up")),
            (self.cw_check, self.cw_check.setChecked, settings.get("clockwise")),
        ):
            if value is None:
                continue
            widget.blockSignals(True)
            setter(value)
            widget.blockSignals(False)
        # Before the combos: the individual combo is built from the pin.
        IndividualPinMixin.set_pinned_individual(self, settings.get("individual"))
        self.configure(feature=settings.get("feature"), selections=settings.get("selections"))
