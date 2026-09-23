"""Space plot widget for displaying arbitrary 2D/3D scatter trajectories.

Users pick which feature (and sub-dimension) to plot on each axis via
combo boxes embedded in the dock widget itself.  Data is fetched through
the DataLoader so xarray, pynapple, and NWB sources all work.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import yaml
from qtpy.QtCore import QEvent, Qt, QTimer, Signal
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

from ethograph.features.preprocessing import interpolate_nans
from ethograph.gui.app_constants import MEDIA_VIEW_MIN_HEIGHT, MEDIA_VIEW_MIN_WIDTH
from ethograph.gui.plots_base import IndividualPinMixin
from ethograph.gui.plots_lineplot import MultiColoredLineItem
from ethograph.io.catalog import INDIVIDUAL_DIMS, DataLoader
from ethograph.utils.paths import defaults_dir, seed_defaults

logger = logging.getLogger(__name__)

SEPARATOR = " · "

#: Dim name that gets priority as the default axis dimension.
SPACE_DIM_NAME = "space"

#: Default color-combo entry — trajectory colored by label highlight.
LABELS_COLOR_MODE = "Labels"


# ---------------------------------------------------------------------------
# Axis selection helper
# ---------------------------------------------------------------------------


def _select_axis(
    store: DataLoader,
    feature: str,
    selections: dict[str, str],
    t0: float | None = None,
    t1: float | None = None,
    color_variable: str | None = None,
):
    """Fetch 1-D numpy array + time (+ optional color data) for a single axis.

    ``selections`` pins the axis dimension to one value (e.g. ``space="x"``)
    plus the values of every extra dim combo; ``store.select`` uses
    ``sel_valid`` internally and ignores dimensions the feature doesn't have.

    Returns ``(time, data, color_data)`` or ``(None, None, None)`` on failure.
    """
    pd = store.select(feature, selections, t0=t0, t1=t1, color_variable=color_variable)
    if pd is None:
        return None, None, None

    data = pd.data
    if data.ndim == 2:
        data = data[:, 0]

    return pd.time, data.astype(np.float64), pd.color_data


# ---------------------------------------------------------------------------
# Reference geometry: vertices + edges
# ---------------------------------------------------------------------------


@dataclass
class ReferenceGeometry:
    """A set of vertices connected by indexed edges."""

    name: str
    vertices: np.ndarray  # (N, 2) or (N, 3)
    edges: list[tuple[int, int]]
    color: str = "black"


def load_geometry_yaml(path: Path) -> Optional[dict]:
    """Load a geometry YAML file. Returns None if the file is absent."""
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


#: User library of reference geometries. Drop a ``*.yaml`` file here (a
#: ``references:`` list of vertices/edges) to make it selectable — by file
#: stem — in the Space controls / persist-able as a default via
#: ``space_library_geometry`` in gui_settings.yaml or local_settings.yaml.
GEOMETRY_LIBRARY_DIR = defaults_dir("config") / "space"


def ensure_geometry_library() -> Path:
    """Seed the geometry library (with every other bundled default) and return it.

    ``seed_defaults`` copies each shipped file only when it is missing, so a
    user's edits to a bundled geometry stick; a deleted one comes back.
    """
    seed_defaults()
    GEOMETRY_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return GEOMETRY_LIBRARY_DIR


def load_library_geometries(lib_dir: Path | None = None) -> dict[str, list["ReferenceGeometry"]]:
    """Parse every YAML file in the geometry library, keyed by file stem.

    One file = one selectable geometry (e.g. ``moll2025.yaml`` →
    ``"moll2025"``); all of a file's ``references`` are drawn together.
    Unparsable files are skipped with a log message (user-supplied input).
    """
    lib_dir = GEOMETRY_LIBRARY_DIR if lib_dir is None else Path(lib_dir)
    geometries: dict[str, list[ReferenceGeometry]] = {}
    if not lib_dir.is_dir():
        return geometries
    for path in sorted(lib_dir.glob("*.y*ml")):
        cfg = load_geometry_yaml(path)
        if not cfg:
            continue
        try:
            refs = _parse_references(cfg)
        except Exception:
            logger.exception("Failed to parse geometry library file %s", path)
            continue
        if refs:
            geometries[path.stem] = refs
    return geometries


def _parse_references(cfg: dict) -> list[ReferenceGeometry]:
    """Parse a geometry config's ``references`` list (name/vertices/edges/color)
    into :class:`ReferenceGeometry` objects."""
    refs: list[ReferenceGeometry] = []
    for entry in cfg.get("references", []):
        verts = np.array(entry["vertices"], dtype=np.float64)
        edges = [tuple(e) for e in entry["edges"]]
        refs.append(
            ReferenceGeometry(
                name=entry.get("name", "ref"),
                vertices=verts,
                edges=edges,
                color=entry.get("color", "black"),
            )
        )
    return refs


def _color_to_rgba(color_str: str) -> tuple:
    """Convert color name/hex to (r, g, b, a) float tuple for GL."""
    try:
        qc = pg.mkColor(color_str)
        return (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
    except Exception:
        return (0.0, 0.0, 0.0, 1.0)


def _render_reference_2d(plot_item, ref: ReferenceGeometry):
    """Draw a ReferenceGeometry on a 2D PlotWidget."""
    verts = ref.vertices
    for i0, i1 in ref.edges:
        line = pg.PlotCurveItem(
            x=np.array([verts[i0, 0], verts[i1, 0]]),
            y=np.array([verts[i0, 1], verts[i1, 1]]),
            pen=pg.mkPen(color=ref.color, width=2),
        )
        line._is_reference = True
        plot_item.addItem(line)


def _render_reference_3d(gl_widget, ref: ReferenceGeometry):
    """Draw a ReferenceGeometry on a 3D GLViewWidget."""
    verts = ref.vertices
    if verts.shape[1] < 3:
        verts = np.column_stack([verts, np.zeros(len(verts))])

    # GL cannot render NaN vertex positions, so draw each edge as a
    # separate line segment using mode='lines' (vertex pairs).
    pairs = []
    for i0, i1 in ref.edges:
        pairs.extend([verts[i0], verts[i1]])
    if not pairs:
        return

    color = _color_to_rgba(ref.color)
    wireframe = gl.GLLinePlotItem(
        pos=np.array(pairs, dtype=np.float32),
        color=color,
        width=2,
        antialias=True,
        mode="lines",
        glOptions="opaque",
    )
    wireframe._is_reference = True
    gl_widget.addItem(wireframe)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_2d(plot_widget, X, Y, color_data=None):
    """Plot 2D trajectory on a PlotWidget. Returns the line item."""
    if color_data is not None and color_data.ndim == 2 and color_data.shape[1] >= 3:
        line = MultiColoredLineItem(x=X, y=Y, colors=color_data, width=3)
    else:
        line = pg.PlotCurveItem(x=X, y=Y, pen=pg.mkPen(color="b", width=3))
    line._is_trajectory = True
    plot_widget.addItem(line)
    return line


def _render_3d(gl_widget, X, Y, Z, color_data=None):
    """Plot 3D trajectory on a GLViewWidget. Returns the line item."""
    xyz = np.column_stack([X, Y, Z]).astype(np.float32)
    if color_data is not None and color_data.ndim == 2 and color_data.shape[1] >= 3:
        if color_data.shape[1] == 3:
            alpha = np.ones((color_data.shape[0], 1), dtype=color_data.dtype)
            color_data = np.concatenate([color_data, alpha], axis=1)
        if color_data.max() > 1.0:
            color_data = color_data / 255.0
        line = gl.GLLinePlotItem(pos=xyz, color=color_data, width=3, antialias=True)
    else:
        line = gl.GLLinePlotItem(pos=xyz, color=(0, 0, 1, 1), width=3, antialias=True)
    line._is_trajectory = True
    gl_widget.addItem(line)
    return line


def _auto_camera_3d(gl_widget, X, Y, Z):
    """Set a reasonable default camera for 3D data."""
    cx, cy, cz = float(np.nanmean(X)), float(np.nanmean(Y)), float(np.nanmean(Z))
    extent = (
        float(
            max(
                np.nanmax(X) - np.nanmin(X),
                np.nanmax(Y) - np.nanmin(Y),
                np.nanmax(Z) - np.nanmin(Z),
            )
        )
        * 1.5
    )
    gl_widget.setCameraPosition(
        pos=pg.Vector(cx, cy, cz),
        distance=max(extent, 1.0),
        elevation=30,
        azimuth=200,
    )


# ---------------------------------------------------------------------------
# SpacePlot widget
# ---------------------------------------------------------------------------


class SpacePlot(IndividualPinMixin, QWidget):
    """Dock widget for displaying spatial plots with user-selectable axes.

    Space plots are instances like line plots: any number can be open at
    once (same or different features / 2D/3D), each in its own shell dock.
    Closing the dock emits :pyattr:`closed` so the owner can drop the
    instance (``DataWidget.remove_space_plot``).

    The individual combo is the panel's pin: it shows the pin, else the
    sidebar's individual, and picking a name in it pins the panel
    (:pyattr:`pin_changed`), exactly as a feature panel never keeps the
    individual among its own selections.
    """

    #: Emitted on any mouse press in the plot → switches the sidebar to the
    #: Space context (parity with the pyqtgraph plots' ``plot_clicked``).
    clicked = Signal()

    #: Emitted with ``self`` when the user closes this panel's dock.
    closed = Signal(object)

    #: Emitted with ``self`` when the user pinned this panel through its own
    #: individual combo; the owner re-titles it and moves the labelling subject.
    pin_changed = Signal(object)

    #: Emitted with ``self`` after the user interactively changes the view
    #: (2D zoom/pan or 3D camera orbit/zoom) — DataWidget mirrors the new
    #: view onto every other open space plot of the same type when
    #: ``app_state.space_sync_views`` is on.
    view_changed = Signal(object)

    #: Monotonic counter so every instance's dock gets a unique objectName.
    _dock_seq = 0

    def __init__(self, shell, app_state):
        super().__init__()
        self.shell = shell
        self.app_state = app_state
        self.dock_widget = None
        self.dock_object_name: str | None = None
        self._apply_default_width = True
        self._dock_name = "Space Plot"

        self._store: DataLoader | None = None

        # --- Layout ---
        root = QVBoxLayout()
        root.setContentsMargins(4, 4, 4, 0)
        root.setSpacing(4)
        self.setLayout(root)

        # Row 1: 3D checkbox + feature combo
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(6)

        self.cb_3d = QCheckBox("3D")
        row1.addWidget(self.cb_3d)

        row1.addWidget(QLabel("Feature"))
        self.feature_combo = QComboBox()
        self.feature_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row1.addWidget(self.feature_combo)

        # Row 2: axis-dimension combo
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(6)

        row2.addWidget(QLabel("Space dim:"))
        self.space_dim_combo = QComboBox()
        self.space_dim_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row2.addWidget(self.space_dim_combo)

        # Row 3: axis combos — values along the selected space dim
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)

        toolbar.addWidget(QLabel("X"))
        self.x_combo = QComboBox()
        self.x_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(self.x_combo)

        toolbar.addWidget(QLabel("Y"))
        self.y_combo = QComboBox()
        self.y_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(self.y_combo)

        self.z_label = QLabel("Z")
        toolbar.addWidget(self.z_label)
        self.z_combo = QComboBox()
        self.z_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(self.z_combo)

        # Dynamic rows: one combo per remaining (non-space) dimension of the
        # selected feature — catalog-driven, replaces the hardcoded keypoint combo.
        self._dim_rows = QVBoxLayout()
        self._dim_rows.setContentsMargins(0, 0, 0, 0)
        self._dim_rows.setSpacing(4)
        self._dim_combos: dict[str, QComboBox] = {}
        self._current_dim_values: dict[str, str | None] = {}

        # Last row: color combo
        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.setSpacing(6)

        color_row.addWidget(QLabel("Color"))
        self.color_combo = QComboBox()
        self.color_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        color_row.addWidget(self.color_combo)

        # Time window: only ± this many seconds of trajectory around the
        # marker (0 = whole window). Session-long trajectories are slow to
        # draw and unreadable; the window slides with the marker.
        window_row = QHBoxLayout()
        window_row.setContentsMargins(0, 0, 0, 0)
        window_row.setSpacing(6)
        window_row.addWidget(QLabel("± Window"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.0, 100000.0)
        self.window_spin.setDecimals(1)
        self.window_spin.setSingleStep(1.0)
        self.window_spin.setSuffix(" s")
        self.window_spin.setSpecialValueText("All")
        self.window_spin.setValue(float(getattr(app_state, "space_window_s", 0.0) or 0.0))
        self.window_spin.setToolTip(
            "Show only this many seconds of trajectory either side of the time marker.\n"
            "'All' draws the whole window — slow and cluttered on session-long data."
        )
        self.window_spin.valueChanged.connect(self._on_window_spin_changed)
        window_row.addWidget(self.window_spin, 1)

        # These controls live in the sidebar's Space context, not on the plot;
        # they are re-parented there when the space plot is created
        # (see DataWidget.update_space_plot).
        self.controls_widget = QWidget()
        controls_v = QVBoxLayout(self.controls_widget)
        controls_v.setContentsMargins(0, 0, 0, 0)
        controls_v.setSpacing(4)
        controls_v.addLayout(row1)
        controls_v.addLayout(row2)
        controls_v.addLayout(toolbar)
        controls_v.addLayout(self._dim_rows)
        controls_v.addLayout(color_row)
        controls_v.addLayout(window_row)

        # Plot area — stable container that stays in the layout; the actual
        # PlotWidget / GLViewWidget is swapped inside it.
        self._plot_holder = QWidget()
        self._plot_holder_layout = QVBoxLayout()
        self._plot_holder_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_holder.setLayout(self._plot_holder_layout)
        self._plot_holder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._plot_holder, 1)

        self.space_widget = None
        self.is_3d = False
        self._plot_container = None
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.timeout.connect(self._update_plot)

        # Trajectory state for highlight / time marker
        self._trajectory_pos: tuple | None = None
        self._trajectory_times: np.ndarray | None = None
        self._time_marker_item = None
        self._locked_ranges: dict | None = None  # saved axis ranges when lock is on
        # Sliding time window: last marker time + the range last fetched, so
        # playback re-fetches only when the marker leaves the middle half.
        self._last_marker_t: float | None = None
        self._fetch_range: tuple[float, float] | None = None
        # Sticky label highlight (start, end, rgb) — re-applied after every
        # re-render, cleared via clear_time_highlight().
        self._last_highlight: tuple[float, float, tuple] | None = None
        self._slide_timer = QTimer()
        self._slide_timer.setSingleShot(True)
        self._slide_timer.setInterval(300)
        self._slide_timer.timeout.connect(self._update_plot)

        # View-sync: emit ``view_changed`` AFTER the widget has processed the
        # interaction (the 3D events are seen in the event filter *before*
        # GLViewWidget moves its camera), collapsing bursts within one
        # event-loop iteration.
        self._view_sync_timer = QTimer()
        self._view_sync_timer.setSingleShot(True)
        self._view_sync_timer.setInterval(0)
        self._view_sync_timer.timeout.connect(lambda: self.view_changed.emit(self))

        # Connect combo/checkbox signals
        self.feature_combo.currentIndexChanged.connect(self._on_feature_changed)
        self.space_dim_combo.currentIndexChanged.connect(self._on_space_dim_changed)
        self.x_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.y_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.z_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.color_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.cb_3d.toggled.connect(self._on_3d_toggled)

        # Listen for settings changes via app_state
        app_state.space_percentile_xyzlim_changed.connect(self._on_settings_changed)
        app_state.space_limit_to_window_changed.connect(self._on_settings_changed)
        app_state.space_window_s_changed.connect(self._on_window_state_changed)
        app_state.space_hide_zeros_changed.connect(self._on_settings_changed)
        app_state.space_show_references_changed.connect(self._on_settings_changed)
        app_state.space_library_geometry_changed.connect(self._on_settings_changed)

        self._set_3d_visible(False)
        super().hide()

    # --- Public API --------------------------------------------------------

    def set_plot_container(self, plot_container):
        """Wire up the main plot container for x-range queries."""
        self._plot_container = plot_container

    def set_store(self, store: DataLoader | None):
        """Set the feature store and repopulate axis combos."""
        if store is self._store:
            return
        self._store = store
        self._populate_combos()

    def show(self):
        if not self.dock_widget:
            SpacePlot._dock_seq += 1
            name = "Space Plot" if SpacePlot._dock_seq == 1 else f"Space Plot {SpacePlot._dock_seq}"
            self._dock_name = name
            # Dock in the top area — the same row (and height) as the video —
            # instead of the left edge, where the dock title collided with the
            # top bar and had to be dragged into place manually.
            self.dock_widget = self.shell.add_dock_widget(
                self, area="top", name=name, object_name=self.dock_object_name
            )
            self.setMinimumSize(MEDIA_VIEW_MIN_WIDTH, MEDIA_VIEW_MIN_HEIGHT)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            # Skipped when the saved window state placed the dock — this
            # deferred resize would stomp the restored width.
            if not self.dock_widget.restored_from_state and getattr(self, "_apply_default_width", True):
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

    # --- The panel's individual -----------------------------------------------

    def set_pinned_individual(self, individual: str | None) -> None:
        super().set_pinned_individual(individual)
        self.sync_individual()

    def _individual_combos(self) -> dict[str, QComboBox]:
        return {dim: combo for dim, combo in self._dim_combos.items() if dim in INDIVIDUAL_DIMS}

    def sync_individual(self) -> None:
        """Point the individual combo at this panel's individual (its pin, else the sidebar's) and re-render."""
        target = self.app_state.panel_individual(self)
        if target is None:
            return
        changed = False
        for dim, combo in self._individual_combos().items():
            index = combo.findText(str(target))
            if index < 0 or combo.currentIndex() == index:
                continue
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
            self._current_dim_values[dim] = combo.currentText()
            changed = True
        if changed and self._store is not None:
            self._update_plot()

    def _apply_default_dock_width(self):
        """Deferred default sizing for a NEW dock. By firing time the panel
        may have been removed (e.g. an auto-created plot replaced by the
        saved layout) — resizing a dock outside the layout warns, and the
        wrapped QDockWidget may already be deleted."""
        dock = self.dock_widget
        try:
            # _apply_default_width may have been cleared after scheduling —
            # a saved layout adopted this instance and now owns its size.
            if not self._apply_default_width:
                return
            if dock is None or dock.isFloating() or self.shell.dockWidgetArea(dock) == Qt.NoDockWidgetArea:
                return
            self.shell.resizeDocks([dock], [int(self.shell.width() * 0.2)], Qt.Horizontal)
        except RuntimeError:
            pass  # dock's C++ object deleted before the timer fired

    def hide(self):
        if self.dock_widget:
            self.dock_widget.setVisible(False)
        super().hide()

    def refresh(self):
        """Re-render with current axis selections."""
        self._update_plot()

    def configure(self, feature: str | None = None, view_3d: bool | None = None):
        """Programmatically set feature and/or 2D/3D mode, then re-render once.

        Used when a panel is created via drag-drop ("Space (2D)" / "Space (3D)"
        in the plot-type picker).
        """
        if feature is not None:
            idx = self.feature_combo.findText(feature)
            if idx >= 0 and idx != self.feature_combo.currentIndex():
                self.feature_combo.blockSignals(True)
                self.feature_combo.setCurrentIndex(idx)
                self.feature_combo.blockSignals(False)
                self._populate_space_dim_combo()
                self._populate_axis_combos()
                self._rebuild_dim_combos()
        if view_3d is not None:
            self.cb_3d.blockSignals(True)
            self.cb_3d.setChecked(view_3d)
            self.cb_3d.blockSignals(False)
            self._set_3d_visible(view_3d)
        if self._store is not None:
            self._save_to_app_state()
            self._update_plot()

    # --- Combo population --------------------------------------------------

    def _populate_combos(self):
        """Fill all combos from the current store: feature → space dim →
        axis values → extra dim combos → color."""
        self.feature_combo.blockSignals(True)
        self.feature_combo.clear()

        if self._store is None:
            self.feature_combo.blockSignals(False)
            self._clear_dim_combos()
            return

        self.feature_combo.addItems(self._store.features)
        saved = getattr(self.app_state, "space_feature", None)
        idx = self.feature_combo.findText(saved) if saved else -1
        if idx >= 0:
            self.feature_combo.setCurrentIndex(idx)
        else:
            default = self._default_feature()
            if default:
                self.feature_combo.setCurrentIndex(self.feature_combo.findText(default))
        self.feature_combo.blockSignals(False)

        self._populate_space_dim_combo()
        self._populate_axis_combos()
        self._rebuild_dim_combos()
        self._populate_color_combo()

    def _default_feature(self) -> str | None:
        """Prefer a feature with a dim named "space"; else the first feature
        with any multi-valued dim; else the first feature."""
        feats = self._store.features
        if not feats:
            return None
        for feat in feats:
            if any(d.lower() == SPACE_DIM_NAME for d in self._store.feature_dims(feat)):
                return feat
        for feat in feats:
            if any(len(v) >= 2 for v in self._store.feature_dims(feat).values()):
                return feat
        return feats[0]

    def _feature_dims(self) -> dict[str, list[str]]:
        feat = self.feature_combo.currentText()
        if self._store is None or not feat:
            return {}
        return self._store.feature_dims(feat)

    def _populate_space_dim_combo(self):
        """Fill the space-dim combo with the selected feature's dims.

        A dim literally named "space" gets priority as default (movement
        datasets); otherwise the first dim (e.g. "pca").
        """
        combo = self.space_dim_combo
        prev = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        dims = list(self._feature_dims())
        combo.addItems(dims)

        saved = getattr(self.app_state, "space_dim", None)
        space_named = next((d for d in dims if d.lower() == SPACE_DIM_NAME), None)
        for candidate in (prev, saved, space_named):
            idx = combo.findText(candidate) if candidate else -1
            if idx >= 0:
                combo.setCurrentIndex(idx)
                break
        combo.blockSignals(False)

    def _populate_axis_combos(self):
        """Fill X/Y/Z combos with the space dim's values.

        Defaults: values named x/y/z if present, else the first three values
        (the user can always pick any other value manually).
        """
        vals = self._feature_dims().get(self.space_dim_combo.currentText(), [])
        lower = [v.lower() for v in vals]

        saved = (
            getattr(self.app_state, "space_x_axis", None),
            getattr(self.app_state, "space_y_axis", None),
            getattr(self.app_state, "space_z_axis", None),
        )
        axis_combos = (self.x_combo, self.y_combo, self.z_combo)
        prev = tuple(c.currentText() for c in axis_combos)

        for i, (combo, name) in enumerate(zip(axis_combos, ("x", "y", "z"))):
            if name in lower:
                default = vals[lower.index(name)]
            elif vals:
                default = vals[min(i, len(vals) - 1)]
            else:
                default = None
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(vals)
            for candidate in (prev[i], saved[i], default):
                idx = combo.findText(candidate) if candidate else -1
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    break
            combo.blockSignals(False)

    def _clear_dim_combos(self):
        while self._dim_rows.count():
            item = self._dim_rows.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._dim_combos = {}
        self._current_dim_values = {}

    def _rebuild_dim_combos(self):
        """One combo per non-space dimension of the selected feature
        (keypoints, individuals, …) — whatever the catalog exposes."""
        self._clear_dim_combos()
        space_dim = self.space_dim_combo.currentText()
        global_sels = self.app_state.get_selections() if hasattr(self.app_state, "get_selections") else {}
        # The individual is never one of the panel's own selections: it is
        # the pin, or the sidebar's individual for a panel that follows it.
        individual = self.app_state.panel_individual(self)
        for dim in INDIVIDUAL_DIMS:
            if individual is not None:
                global_sels[dim] = individual

        for dim, vals in self._feature_dims().items():
            if dim == space_dim or not vals:
                continue
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            layout.addWidget(QLabel(dim.capitalize()))

            combo = QComboBox()
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            combo.addItems(vals)
            preset = global_sels.get(dim)
            if preset is not None:
                idx = combo.findText(str(preset))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(lambda _i, d=dim: self._on_dim_changed(d))
            layout.addWidget(combo)

            self._dim_rows.addWidget(row)
            self._dim_combos[dim] = combo
            self._current_dim_values[dim] = combo.currentText()

    def _populate_color_combo(self):
        """Color combo: "Labels" (default label-highlight mode) + all features."""
        combo = self.color_combo
        prev = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(LABELS_COLOR_MODE)
        if self._store is not None:
            combo.addItems(self._store.features)
        saved = getattr(self.app_state, "space_color", None)
        for candidate in (prev, saved):
            idx = combo.findText(candidate) if candidate else -1
            if idx >= 0:
                combo.setCurrentIndex(idx)
                break
        combo.blockSignals(False)

    def _get_dim_selections(self) -> dict[str, str]:
        """Selection dict from the extra dim combos (non-space dims)."""
        return {dim: c.currentText() for dim, c in self._dim_combos.items() if c.currentText()}

    def color_variable(self) -> str | None:
        """Feature used for trajectory coloring, or None in the default
        label-highlight ("Labels") mode."""
        text = self.color_combo.currentText()
        return text if text and text != LABELS_COLOR_MODE else None

    # --- Change handlers -----------------------------------------------------

    def _on_feature_changed(self, *_args):
        self._populate_space_dim_combo()
        self._populate_axis_combos()
        self._rebuild_dim_combos()
        if self._store is not None:
            self._save_to_app_state()
            self._update_plot()

    def _on_space_dim_changed(self, *_args):
        self._populate_axis_combos()
        self._rebuild_dim_combos()
        if self._store is not None:
            self._save_to_app_state()
            self._update_plot()

    def _on_dim_changed(self, dim: str):
        combo = self._dim_combos.get(dim)
        if combo is not None:
            new_text = combo.currentText()
            current = self._current_dim_values.get(dim)
            if new_text and new_text != current:
                self._current_dim_values[dim] = new_text
        if self._store is not None:
            self._update_plot()
        if dim in INDIVIDUAL_DIMS and combo is not None and combo.currentText():
            self._pin_through_combo(combo.currentText())

    def _pin_through_combo(self, individual: str) -> None:
        """Picking an individual in the panel's own combo pins the panel to it."""
        if individual == self.app_state.panel_individual(self):
            return
        IndividualPinMixin.set_pinned_individual(self, individual)
        self.refresh_title()
        self.pin_changed.emit(self)

    def _on_axis_changed(self, *_args):
        if self._store is not None:
            self._save_to_app_state()
            self._update_plot()

    def _on_3d_toggled(self, checked: bool):
        self._set_3d_visible(checked)
        self._save_to_app_state()
        if self._store is not None:
            self._update_plot()

    def _on_settings_changed(self, *_args):
        """Re-render when a plot-settings value changes (debounced)."""
        if self._store is not None and self.isVisible():
            self._debounce_timer.start()

    def _on_window_spin_changed(self, value: float):
        self.app_state.space_window_s = float(value)
        if self._store is not None and self.isVisible():
            self._debounce_timer.start()

    def _on_window_state_changed(self, value):
        """Another instance (or a restore) changed the shared window setting."""
        spin = float(self.window_spin.value())
        value = float(value or 0.0)
        if abs(spin - value) > 1e-9:
            self.window_spin.blockSignals(True)
            self.window_spin.setValue(value)
            self.window_spin.blockSignals(False)
        if self._store is not None and self.isVisible():
            self._debounce_timer.start()

    def on_xrange_changed(self):
        """Called by DataWidget when the lineplot x-range changes."""
        if getattr(self.app_state, "space_limit_to_window", False) and self.isVisible():
            self._debounce_timer.start()

    def _save_to_app_state(self):
        self.app_state.space_feature = self.feature_combo.currentText() or None
        self.app_state.space_dim = self.space_dim_combo.currentText() or None
        self.app_state.space_x_axis = self.x_combo.currentText() or None
        self.app_state.space_y_axis = self.y_combo.currentText() or None
        self.app_state.space_z_axis = self.z_combo.currentText() or None
        self.app_state.space_3d = self.cb_3d.isChecked()
        self.app_state.space_color = self.color_combo.currentText() or None

    # --- Per-instance settings (layout persistence) --------------------------

    def space_settings(self) -> dict:
        """This instance's full combo state in serializable form. Dock
        placement is not stored here — it lives in the shell's window-state
        blob like every other dock."""
        return {
            "feature": self.feature_combo.currentText() or None,
            "view_3d": self.cb_3d.isChecked(),
            "space_dim": self.space_dim_combo.currentText() or None,
            "x": self.x_combo.currentText() or None,
            "y": self.y_combo.currentText() or None,
            "z": self.z_combo.currentText() or None,
            "dims": {d: c.currentText() for d, c in self._dim_combos.items() if c.currentText()},
            "color": self.color_combo.currentText() or None,
            **({"individual": self.pinned_individual} if self.pinned_individual else {}),
        }

    def apply_space_settings(self, settings: dict) -> None:
        """Restore combo state captured by :meth:`space_settings`, then render."""

        def _set(combo, value):
            idx = combo.findText(value) if value else -1
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

        # Before the combos: the individual combo is built from the pin.
        IndividualPinMixin.set_pinned_individual(self, settings.get("individual"))
        _set(self.feature_combo, settings.get("feature"))
        self._populate_space_dim_combo()
        _set(self.space_dim_combo, settings.get("space_dim"))
        self._populate_axis_combos()
        self._rebuild_dim_combos()
        _set(self.x_combo, settings.get("x"))
        _set(self.y_combo, settings.get("y"))
        _set(self.z_combo, settings.get("z"))
        for dim, val in (settings.get("dims") or {}).items():
            combo = self._dim_combos.get(dim)
            if combo is not None:
                _set(combo, val)
                self._current_dim_values[dim] = combo.currentText()
        _set(self.color_combo, settings.get("color"))
        view_3d = settings.get("view_3d")
        if view_3d is not None:
            self.cb_3d.blockSignals(True)
            self.cb_3d.setChecked(bool(view_3d))
            self.cb_3d.blockSignals(False)
            self._set_3d_visible(bool(view_3d))
        if self._store is not None:
            self._save_to_app_state()
            self._update_plot()

    def _set_3d_visible(self, visible: bool):
        self.z_label.setVisible(visible)
        self.z_combo.setVisible(visible)

    # --- Core plot logic ---------------------------------------------------

    def _get_window_time_range(self) -> tuple[float | None, float | None]:
        """Return (t0, t1) from the lineplot x-range if limit-to-window is on."""
        if not getattr(self.app_state, "space_limit_to_window", False):
            return None, None
        if self._plot_container is None:
            return None, None
        try:
            return self._plot_container.get_current_xlim()
        except Exception:
            return None, None

    def _update_plot(self):
        """Fetch data for selected axes and render."""
        store = self._store
        if store is None:
            return

        feature = self.feature_combo.currentText()
        space_dim = self.space_dim_combo.currentText()
        x_val = self.x_combo.currentText()
        y_val = self.y_combo.currentText()
        if not feature or not space_dim or not x_val or not y_val:
            return

        view_3d = self.cb_3d.isChecked()
        z_val = self.z_combo.currentText() if view_3d else None
        color_var = self.color_variable()

        selections = self._get_dim_selections()
        t0, t1 = self._get_window_time_range()
        if t0 is None or t1 is None:
            wb = self.app_state.window_bounds
            if wb is not None:
                t0, t1 = wb.start_s, wb.end_s
        if t0 is None or t1 is None:
            return

        # Sliding time window: clip the fetch to ± window around the marker.
        # Session-long trajectories are slow to draw and unreadable; the
        # window re-fetches as the marker approaches its edges (see
        # _maybe_slide_window).
        window_s = float(getattr(self.app_state, "space_window_s", 0.0) or 0.0)
        if window_s > 0:
            center = self._current_marker_time()
            t0 = max(t0, center - window_s)
            t1 = min(t1, center + window_s)
            if t1 <= t0:
                return
            self._fetch_range = (t0, t1)
        else:
            self._fetch_range = None

        time_x, data_x, color_data = _select_axis(
            store, feature, {**selections, space_dim: x_val}, t0=t0, t1=t1, color_variable=color_var
        )
        time_y, data_y, _ = _select_axis(store, feature, {**selections, space_dim: y_val}, t0=t0, t1=t1)
        if time_x is None or time_y is None:
            return

        n = min(len(data_x), len(data_y))
        data_x, data_y = data_x[:n], data_y[:n]
        times = time_x[:n]
        if color_data is not None:
            color_data = color_data[:n]

        data_z = None
        if view_3d and z_val:
            _, dz, _ = _select_axis(store, feature, {**selections, space_dim: z_val}, t0=t0, t1=t1)
            if dz is not None:
                data_z = dz[:n]

        # Mask points where all dimensions are exactly zero
        if getattr(self.app_state, "space_hide_zeros", False):
            zero_mask = (data_x == 0) & (data_y == 0)
            if data_z is not None:
                zero_mask &= data_z == 0
            data_x = np.where(zero_mask, np.nan, data_x)
            data_y = np.where(zero_mask, np.nan, data_y)
            if data_z is not None:
                data_z = np.where(zero_mask, np.nan, data_z)

        use_3d = view_3d and data_z is not None
        locked = getattr(self.app_state, "space_lock_axes", False)

        # GL cannot render NaN positions — interpolate before 3D rendering
        if use_3d:
            data_x = interpolate_nans(data_x)
            data_y = interpolate_nans(data_y)
            data_z = interpolate_nans(data_z)

        # Save current ranges before touching the widget
        saved_ranges = self._capture_ranges() if locked else None

        # Rebuild the widget only when the 2D/3D type changes — rebuilding on
        # every update was the dominant render cost. Otherwise the trajectory
        # items are swapped on the live widget, keeping the user's zoom/camera.
        rebuilt = self._ensure_plot_widget(use_3d)
        self._clear_dynamic_items()

        if use_3d:
            _render_3d(self.space_widget, data_x, data_y, data_z, color_data)
            if rebuilt:
                _auto_camera_3d(self.space_widget, data_x, data_y, data_z)
        else:
            _render_2d(self.space_widget, data_x, data_y, color_data)
            plot_item = self.space_widget.getPlotItem()
            plot_item.setLabel("bottom", f"{feature}{SEPARATOR}{x_val}")
            plot_item.setLabel("left", f"{feature}{SEPARATOR}{y_val}")

        if rebuilt:
            if locked and saved_ranges:
                self._restore_ranges(saved_ranges)
            else:
                self._apply_percentile_limits(data_x, data_y, data_z)
        elif self._fetch_range is None and not locked:
            # Non-windowed refresh (settings/axis change): re-fit as before.
            # A sliding window keeps the current view — re-fitting every
            # slide would make the axes jump with the animal.
            self._apply_percentile_limits(data_x, data_y, data_z)

        self._draw_references()

        self._trajectory_pos = (data_x, data_y, data_z)
        self._trajectory_times = times
        self.is_3d = use_3d

        if self._last_highlight is not None and self.color_variable() is None:
            self.highlight_time_segment(*self._last_highlight)

        self.update_time_marker(self._current_marker_time())

    def _current_marker_time(self) -> float:
        """The time marker's position on the display clock."""
        if self._last_marker_t is not None:
            return float(self._last_marker_t)
        frame = getattr(self.app_state, "current_frame", 0)
        video = getattr(self.app_state, "video", None)
        if video:
            return float(video.frame_to_time(frame))
        fps = getattr(self.app_state, "video_fps", None)
        t_rel = frame / fps if fps else 0.0
        return float(self.app_state.to_display(getattr(self.app_state, "trials_sel", None), t_rel))

    def _maybe_slide_window(self, t: float):
        """Re-fetch the sliding window when the marker nears its edges.

        Hysteresis: the fetch spans ± window, a slide fires only once the
        marker leaves the middle half — so playback re-renders periodically,
        not per marker tick (and the render itself no longer rebuilds the
        widget).
        """
        if self._fetch_range is None or not self.isVisible():
            return
        t0, t1 = self._fetch_range
        span = t1 - t0
        if span <= 0:
            return
        if (t < t0 + span * 0.25 or t > t1 - span * 0.25) and not self._slide_timer.isActive():
            self._slide_timer.start()

    def _ensure_plot_widget(self, view_3d: bool) -> bool:
        """(Re)create the plot widget only when the 2D/3D type changes."""
        if self.space_widget is not None and isinstance(self.space_widget, gl.GLViewWidget) == view_3d:
            return False
        self._rebuild_plot_widget(view_3d)
        self._time_marker_item = None
        return True

    def _clear_dynamic_items(self):
        """Remove trajectory/highlight items, keeping references and marker."""
        if self.space_widget is None:
            return
        if isinstance(self.space_widget, gl.GLViewWidget):
            for item in list(self.space_widget.items):
                if getattr(item, "_is_trajectory", False) or getattr(item, "_is_highlight", False):
                    self.space_widget.removeItem(item)
        else:
            plot_item = self.space_widget.getPlotItem()
            for item in list(plot_item.items):
                if getattr(item, "_is_trajectory", False) or getattr(item, "_is_highlight", False):
                    plot_item.removeItem(item)

    def _rebuild_plot_widget(self, view_3d: bool):
        """Remove old widget and create the right type inside the holder."""
        if self.space_widget is not None:
            self._plot_holder_layout.removeWidget(self.space_widget)
            self.space_widget.hide()
            self.space_widget.deleteLater()

        if view_3d:
            try:
                self.space_widget = gl.GLViewWidget()
                self.space_widget.setBackgroundColor("w")
                self.space_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            except Exception:
                logger.warning("OpenGL unavailable, falling back to 2D view")
                self.cb_3d.blockSignals(True)
                self.cb_3d.setChecked(False)
                self.cb_3d.blockSignals(False)
                self.space_widget = pg.PlotWidget()
                self.space_widget.setBackground("w")
        else:
            self.space_widget = pg.PlotWidget()
            self.space_widget.setBackground("w")

        if not isinstance(self.space_widget, gl.GLViewWidget):
            # Manually only: programmatic setRange (percentile limits, restore,
            # incoming sync) must not re-broadcast, or two synced plots loop.
            vb = self.space_widget.getPlotItem().vb
            vb.sigRangeChangedManually.connect(self._on_view_interacted)

        self._plot_holder_layout.addWidget(self.space_widget)
        self._install_click_filter()

    def _install_click_filter(self):
        """Emit ``clicked`` on any mouse press inside the plot (2D or 3D)."""
        self.installEventFilter(self)
        if self.space_widget is not None:
            self.space_widget.installEventFilter(self)
            for child in self.space_widget.findChildren(QWidget):
                child.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.dock_widget:
            if event.type() == QEvent.Close:
                self.closed.emit(self)
            return False
        if event.type() == QEvent.MouseButtonPress:
            self.clicked.emit()
        elif event.type() in (QEvent.MouseMove, QEvent.Wheel, QEvent.MouseButtonRelease):
            # GLViewWidget has no view-changed signal, so 3D camera changes
            # (orbit drag / wheel zoom) are inferred from input events. Mouse
            # moves arrive here only while a button is held (no tracking).
            if isinstance(self.space_widget, gl.GLViewWidget) and obj is self.space_widget:
                self._on_view_interacted()
        return False

    # --- View sync across space plots ----------------------------------------

    def _on_view_interacted(self, *_args):
        """The user zoomed/panned/orbited — schedule a ``view_changed`` emit."""
        self._view_sync_timer.start()

    def capture_view_state(self) -> dict | None:
        """Current coordinate frame: 2D axis ranges or 3D camera parameters."""
        return self._capture_ranges()

    def apply_view_state(self, state: dict) -> None:
        """Adopt another plot's coordinate frame. No-ops when the state's
        2D/3D mode doesn't match this plot's widget — sync never crosses
        the 2D/3D boundary."""
        self._restore_ranges(state)

    def _load_references(self) -> list[ReferenceGeometry]:
        """Reference geometry to overlay, resolved from the geometry library.

        ``app_state.space_library_geometry`` (chosen in the Space controls, or
        set as a default in gui_settings.yaml / local_settings.yaml) is the
        stem of a YAML file in ``~/.ethograph/defaults/config/space/``; all of that
        file's references are drawn.
        """
        selected = getattr(self.app_state, "space_library_geometry", None)
        if not selected:
            return []
        refs = load_library_geometries().get(selected)
        if refs is None:
            logger.warning("Geometry file %r not found in %s", selected, GEOMETRY_LIBRARY_DIR)
            return []
        return refs

    def _draw_references(self):
        """(Re)draw all reference geometry items.

        Idempotent: old reference items are removed first, so this runs on
        every render — a library-geometry / show-references change re-renders
        every open space plot without a widget rebuild.
        """
        self._clear_reference_items()
        if not getattr(self.app_state, "space_show_references", True):
            return
        refs = self._load_references()
        if not refs:
            return

        is_gl = isinstance(self.space_widget, gl.GLViewWidget)
        for ref in refs:
            try:
                if is_gl:
                    _render_reference_3d(self.space_widget, ref)
                else:
                    plot_item = self.space_widget.getPlotItem()
                    _render_reference_2d(plot_item, ref)
            except Exception:
                logger.exception("Failed to draw reference %s", ref.name)

    def _clear_reference_items(self):
        """Remove previously drawn reference geometry items."""
        if self.space_widget is None:
            return
        if isinstance(self.space_widget, gl.GLViewWidget):
            for item in list(self.space_widget.items):
                if getattr(item, "_is_reference", False):
                    self.space_widget.removeItem(item)
        else:
            plot_item = self.space_widget.getPlotItem()
            for item in list(plot_item.items):
                if getattr(item, "_is_reference", False):
                    plot_item.removeItem(item)

    # --- Percentile axis limits (zoom constraints) --------------------------

    def _apply_percentile_limits(self, data_x, data_y, data_z=None):
        """Constrain zoom to per-axis percentile range using vb.setLimits()."""
        percentile = getattr(self.app_state, "space_percentile_xyzlim", 100.0)
        if percentile >= 100.0 or self.space_widget is None:
            return

        lo = (100 - percentile) / 2
        hi = 100 - lo

        x_lo, x_hi = np.nanpercentile(data_x, [lo, hi])
        y_lo, y_hi = np.nanpercentile(data_y, [lo, hi])

        x_range = x_hi - x_lo
        y_range = y_hi - y_lo
        if x_range <= 0 or y_range <= 0:
            return

        x_buf = x_range * 0.2
        y_buf = y_range * 0.2

        is_gl = isinstance(self.space_widget, gl.GLViewWidget)
        if is_gl:
            cx = float((x_lo + x_hi) / 2)
            cy = float((y_lo + y_hi) / 2)
            extent = max(x_range, y_range)
            if data_z is not None:
                z_lo, z_hi = np.nanpercentile(data_z, [lo, hi])
                cz = float((z_lo + z_hi) / 2)
                extent = max(extent, z_hi - z_lo)
            else:
                cz = 0.0
            self.space_widget.setCameraPosition(
                pos=pg.Vector(cx, cy, cz),
                distance=max(float(extent) * 1.5, 1.0),
                elevation=30,
                azimuth=200,
            )
        else:
            vb = self.space_widget.getPlotItem().vb
            vb.setLimits(
                xMin=x_lo - x_buf,
                xMax=x_hi + x_buf,
                yMin=y_lo - y_buf,
                yMax=y_hi + y_buf,
                minXRange=x_range * 0.1,
                maxXRange=x_range + x_buf,
                minYRange=y_range * 0.1,
                maxYRange=y_range + y_buf,
            )
            vb.setRange(xRange=(x_lo, x_hi), yRange=(y_lo, y_hi), padding=0.05)

    def _capture_ranges(self) -> dict | None:
        """Snapshot the current axis ranges (2D) or camera position (3D)."""
        if self.space_widget is None:
            return None
        if isinstance(self.space_widget, gl.GLViewWidget):
            opts = self.space_widget.cameraParams()
            return {"mode": "3d", "camera": opts}
        vb = self.space_widget.getPlotItem().vb
        xr, yr = vb.viewRange()
        return {"mode": "2d", "x": tuple(xr), "y": tuple(yr)}

    def _restore_ranges(self, ranges: dict):
        """Restore previously captured axis ranges."""
        if self.space_widget is None:
            return
        if ranges["mode"] == "3d" and isinstance(self.space_widget, gl.GLViewWidget):
            cam = ranges["camera"]
            self.space_widget.setCameraPosition(
                pos=cam.get("center"),
                distance=cam.get("distance"),
                elevation=cam.get("elevation"),
                azimuth=cam.get("azimuth"),
            )
        elif ranges["mode"] == "2d" and not isinstance(self.space_widget, gl.GLViewWidget):
            vb = self.space_widget.getPlotItem().vb
            vb.setRange(xRange=ranges["x"], yRange=ranges["y"], padding=0)

    # --- Highlight / time marker -------------------------------------------

    def highlight_time_segment(self, start_time: float, end_time: float, color=(255, 102, 0)):
        """Highlight a time segment of the trajectory.

        Only applies in the default "Labels" color mode — when the trajectory
        is colored by another feature, the label highlight must not repaint it.
        The segment is remembered and re-applied after every re-render —
        sliding-window updates redraw the trajectory, which would otherwise
        silently wipe the highlight (and the sender dedups per label, so it
        never came back until the marker left and re-entered the label).
        """
        if self.color_variable() is not None:
            return
        self._last_highlight = (float(start_time), float(end_time), tuple(np.ravel(color)[:3]))
        if not self.space_widget or self._trajectory_pos is None or self._trajectory_times is None:
            return

        X, Y, Z = self._trajectory_pos
        times = self._trajectory_times

        i0 = int(np.searchsorted(times, start_time))
        i1 = int(np.searchsorted(times, end_time))
        if i1 <= i0:
            return

        # Normalize color to 0-255 int tuple regardless of input format
        c = np.asarray(color, dtype=np.float64).ravel()[:3]
        if c.max() <= 1.0:
            c = c * 255
        r8, g8, b8 = int(c[0]), int(c[1]), int(c[2])
        rf, gf, bf = r8 / 255.0, g8 / 255.0, b8 / 255.0

        is_gl = isinstance(self.space_widget, gl.GLViewWidget)

        if is_gl:
            for item in list(self.space_widget.items):
                if getattr(item, "_is_trajectory", False) or getattr(item, "_is_highlight", False):
                    self.space_widget.removeItem(item)

            z_arr = Z if Z is not None else np.zeros_like(X)
            xyz = np.column_stack([X, Y, z_arr]).astype(np.float32)
            bg = gl.GLLinePlotItem(pos=xyz, color=(0.7, 0.7, 0.7, 0.5), width=2, antialias=True)
            bg._is_trajectory = True
            self.space_widget.addItem(bg)

            seg = xyz[i0 : i1 + 1]
            if len(seg) > 1:
                hl = gl.GLLinePlotItem(pos=seg, color=(rf, gf, bf, 1), width=5, antialias=True)
                hl._is_highlight = True
                self.space_widget.addItem(hl)
        else:
            plot_item = self.space_widget.getPlotItem()
            for item in list(plot_item.items):
                if getattr(item, "_is_trajectory", False) or getattr(item, "_is_highlight", False):
                    plot_item.removeItem(item)

            bg = pg.PlotCurveItem(x=X, y=Y, pen=pg.mkPen(color=(180, 180, 180, 128), width=2))
            bg._is_trajectory = True
            plot_item.addItem(bg)

            x_seg, y_seg = X[i0 : i1 + 1], Y[i0 : i1 + 1]
            if len(x_seg) > 1:
                hl = pg.PlotCurveItem(
                    x=x_seg,
                    y=y_seg,
                    pen=pg.mkPen(color=(r8, g8, b8), width=4),
                )
                hl._is_highlight = True
                plot_item.addItem(hl)

    def clear_time_highlight(self):
        """Forget the sticky label highlight and remove its items.

        The grey background trajectory (if a highlight drew one) stays until
        the next re-render — matching the pre-window behaviour, where a
        highlight also persisted until the next full render.
        """
        self._last_highlight = None
        if self.space_widget is None:
            return
        if isinstance(self.space_widget, gl.GLViewWidget):
            for item in list(self.space_widget.items):
                if getattr(item, "_is_highlight", False):
                    self.space_widget.removeItem(item)
        else:
            plot_item = self.space_widget.getPlotItem()
            for item in list(plot_item.items):
                if getattr(item, "_is_highlight", False):
                    plot_item.removeItem(item)

    def update_time_marker(self, time_position: float):
        """Show a red circle at the current time position on the trajectory."""
        self._last_marker_t = float(time_position)
        self._maybe_slide_window(time_position)
        if not getattr(self.app_state, "space_marker_visible", True):
            self._remove_time_marker()
            return
        if not self.space_widget or self._trajectory_pos is None or self._trajectory_times is None:
            return

        times = self._trajectory_times
        X, Y, Z = self._trajectory_pos
        if len(times) == 0 or len(X) == 0:
            self._remove_time_marker()
            return

        idx = int(np.searchsorted(times, time_position, side="right") - 1)
        idx = int(np.clip(idx, 0, len(X) - 1))

        x, y = float(X[idx]), float(Y[idx])

        is_gl = isinstance(self.space_widget, gl.GLViewWidget)

        if is_gl:
            z = float(Z[idx]) if Z is not None else 0.0
            pos_arr = np.array([[x, y, z]], dtype=np.float32)
            color_arr = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            if self._time_marker_item is not None:
                self._time_marker_item.setData(pos=pos_arr, color=color_arr)
            else:
                self._time_marker_item = gl.GLScatterPlotItem(
                    pos=pos_arr,
                    color=color_arr,
                    size=20,
                    pxMode=True,
                    glOptions="translucent",
                )
                self.space_widget.addItem(self._time_marker_item)
        else:
            if self._time_marker_item is not None:
                self._time_marker_item.setData([x], [y])
            else:
                self._time_marker_item = pg.ScatterPlotItem(
                    [x],
                    [y],
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(255, 0, 0),
                    size=12,
                    symbol="o",
                )
                self._time_marker_item.setZValue(1000)
                plot_item = self.space_widget.getPlotItem()
                plot_item.addItem(self._time_marker_item)

    def _remove_time_marker(self):
        if self._time_marker_item is not None and self.space_widget is not None:
            is_gl = isinstance(self.space_widget, gl.GLViewWidget)
            if is_gl:
                self.space_widget.removeItem(self._time_marker_item)
            else:
                self.space_widget.getPlotItem().removeItem(self._time_marker_item)
            self._time_marker_item = None
