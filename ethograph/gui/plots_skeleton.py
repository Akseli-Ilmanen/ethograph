"""Skeleton plot — the pose at the time marker, drawn in the data's own space.

A space plot draws where a point *went*; this panel draws where the animal
*is*: every keypoint of the dataset's ``position`` at the current instant, with
the skeleton's bones between them, on the same 2D / 3D canvas the space plot
uses (:func:`~ethograph.gui.plots_space.make_space_widget`). It is offered only
for ``position`` — a variable with a ``space`` dim holding x/y(/z) and a
keypoint dim — because that is the one variable whose columns *are* a body.

Nothing here is a trajectory: no window, no colour-by-feature, no highlight.
The controls are what a pose needs and nothing else — which individuals, which
keypoints, and whether colour says the body part or the animal
(``pose_color_by``, the video overlay's toggle). The bones come from the same
resolution the video overlay uses (:func:`~ethograph.gui.pose_render.resolve_skeleton_config`),
the colours from the same palette (:func:`~ethograph.gui.pose_render.pose_color_map`),
the marker size and edge width from the same Pose settings — so the panel and
the overlay always agree on what the pose looks like.

Instances behave like space and radial plots: any number can be open, each in
its own dock; the individual combo is the panel's pin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from qtpy.QtCore import QEvent, Qt, QTimer, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.app_constants import MEDIA_VIEW_MIN_HEIGHT, MEDIA_VIEW_MIN_WIDTH
from ethograph.gui.individual_colors import decorate_individual_combo
from ethograph.gui.plots_base import IndividualPinMixin
from ethograph.gui.plots_space import auto_camera_3d, draw_reference_geometry, make_space_widget
from ethograph.gui.pose_controls import PoseControls
from ethograph.gui.pose_convert import COLOR_BY_INDIVIDUAL, COLOR_BY_KEYPOINT
from ethograph.gui.pose_render import pose_color_map, resolve_skeleton_config
from ethograph.gui.project import project_dir_of
from ethograph.io.catalog import INDIVIDUAL_DIMS, KEYPOINT_DIMS, SPACE_DIM
from ethograph.skeleton.config import hex_to_rgba
from ethograph.skeleton.mesh import (
    MeshModel,
    available_meshes,
    fit_scale,
    load_mesh,
    place_mesh,
)

logger = logging.getLogger(__name__)

#: The one variable a skeleton plot draws. Any other variable is a trajectory
#: for the space plot, however it is shaped.
POSE_FEATURE = "position"
#: Its companion, when the dataset has one: points below ``pose_hide_threshold``
#: are dropped, exactly as the video overlay drops them.
CONFIDENCE_FEATURE = "confidence"

SKELETON_2D = "Skeleton (2D)"
SKELETON_3D = "Skeleton (3D)"

#: Points whose colour axis has no entry in the map (a keypoint the palette
#: was not built for) — visible, never silently skipped.
_FALLBACK_RGBA = (1.0, 0.2, 0.2, 1.0)
#: Bone colour when the points already say the body part: neutral, so twelve
#: keypoint hues are not fighting twelve more on the bones.
_BONE_RGBA = (0.0, 0.0, 0.0, 1.0)
#: A bone-ivory skull, so the mesh reads as bone against any background.
_MESH_RGBA = (0.93, 0.9, 0.82, 1.0)

#: ``app_state`` variables that restyle the pose without changing the data.
_STYLE_SETTINGS = (
    "pose_point_size",
    "pose_skeleton_width",
    "pose_points_use_base",
    "pose_points_base_color",
    "pose_individual_colors",
    "pose_show_keypoints",
    "pose_show_skeleton",
    "pose_show_text",
    "pose_text_size",
    "pose_color_by",
    "pose_hidden_keypoints",
)
#: Variables that change the reference geometry drawn behind the pose.
_REFERENCE_SETTINGS = (
    "space_show_references",
    "space_library_geometry",
    "space_geometry_preview",
)
#: Variables that change which bones exist or what colour they are.
_SKELETON_SETTINGS = (
    "skeleton_use_base",
    "skeleton_base_color",
    "skeleton_config_override",
    "skeleton_source",
    "skeleton_name",
)


# ---------------------------------------------------------------------------
# What a pose variable looks like — the gate and the axis bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseDims:
    """The dims of a pose variable, as the catalog spells them.

    ``axes`` is the space dim's values in x, y(, z) order; ``individuals`` is
    empty when the variable has no individual dim (a single unnamed animal).
    ``extra`` is every other dim with its values (e.g. a ``position_type`` of
    aligned/absolute): a pose is drawn at ONE value of each, chosen by the user.
    """

    space_dim: str
    axes: tuple[str, ...]
    keypoint_dim: str
    keypoints: tuple[str, ...]
    individual_dim: str | None
    individuals: tuple[str, ...]
    extra: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def has_z(self) -> bool:
        return len(self.axes) == 3


def pose_dims(store, feature: str = POSE_FEATURE) -> PoseDims | None:
    """The pose dims of *feature* in *store*, or ``None`` if it cannot be a pose.

    A pose has a ``space`` dim with at least x and y, and a keypoint dim. The
    catalog's ``feature_dims`` is the source, so xarray and pynapple sessions
    answer alike.
    """
    if store is None or not hasattr(store, "feature_dims"):
        return None
    dims = store.feature_dims(feature)
    if not dims:
        return None
    space_dim = next((d for d in dims if d.lower() == SPACE_DIM), None)
    keypoint_dim = next((d for d in KEYPOINT_DIMS if d in dims), None)
    if space_dim is None or keypoint_dim is None or not dims[keypoint_dim]:
        return None
    by_name = {v.lower(): v for v in dims[space_dim]}
    if "x" not in by_name or "y" not in by_name:
        return None
    axes = [by_name["x"], by_name["y"]]
    if "z" in by_name:
        axes.append(by_name["z"])
    individual_dim = next((d for d in INDIVIDUAL_DIMS if d in dims), None)
    known = {space_dim, keypoint_dim, individual_dim}
    return PoseDims(
        space_dim=space_dim,
        axes=tuple(axes),
        keypoint_dim=keypoint_dim,
        keypoints=tuple(dims[keypoint_dim]),
        individual_dim=individual_dim,
        individuals=tuple(dims[individual_dim]) if individual_dim else (),
        extra=tuple((d, tuple(values)) for d, values in dims.items() if d not in known and values),
    )


def skeleton_plot_types(app_state, feature: str) -> list[str]:
    """The skeleton plot types the add-panel popup offers for *feature*.

    Only ``position`` qualifies; 3D only when the pose has a z axis.
    """
    if feature != POSE_FEATURE:
        return []
    dims = pose_dims(getattr(app_state, "data_loader", None), feature)
    if dims is None:
        return []
    return [SKELETON_2D, SKELETON_3D] if dims.has_z else [SKELETON_2D]


# ---------------------------------------------------------------------------
# One instant of a pose, as vertex arrays
# ---------------------------------------------------------------------------


@dataclass
class PoseFrame:
    """Vertex arrays for one instant: points ``(N, 3)`` and edge pairs ``(2E, 3)``."""

    points: np.ndarray
    point_colors: np.ndarray
    edges: np.ndarray
    edge_colors: np.ndarray
    #: One text per point: the axis the colours are not saying (the overlay's rule).
    labels: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return len(self.points) == 0 and len(self.edges) == 0


def pose_frame(
    coords: np.ndarray,
    keypoints: tuple[str, ...] | list[str],
    individuals: tuple[str, ...] | list[str],
    *,
    shown_keypoints: set[str],
    color_by: str,
    color_map: dict[str, tuple],
    connections: list[tuple[str, str]],
    bone_colors: dict[tuple[str, str], tuple] | None = None,
) -> PoseFrame:
    """Build the vertex arrays for one instant.

    *coords* is ``(I, K, D)`` — individuals × keypoints × x/y(/z) — at one
    time index; a point is drawn when it is finite and its keypoint is shown.
    An edge is drawn when both of its keypoints are drawn, for each individual.
    *connections* are ``(start, end)`` by keypoint name. Bones are black when
    colour encodes the keypoint, and the animal's colour when it encodes the
    individual — the bones then say whose body it is. *bone_colors*, when
    given, colours a bone by its ``(start, end)`` instead — the skeleton
    config's colours, exactly as the video overlay draws them.
    """
    n_ind, n_kp, n_dim = coords.shape
    xyz = np.zeros((n_ind, n_kp, 3), dtype=np.float32)
    xyz[:, :, : min(n_dim, 3)] = coords[:, :, :3]
    shown = np.array([k in shown_keypoints for k in keypoints], dtype=bool)
    visible = np.isfinite(coords).all(axis=2) & shown[None, :]
    kp_index = {name: i for i, name in enumerate(keypoints)}

    points: list[np.ndarray] = []
    point_colors: list[tuple] = []
    labels: list[str] = []
    edges: list[np.ndarray] = []
    edge_colors: list[tuple] = []
    by_keypoint = color_by == COLOR_BY_KEYPOINT
    for i, individual in enumerate(individuals):
        bone = _BONE_RGBA if by_keypoint else tuple(color_map.get(individual, _FALLBACK_RGBA))
        for k, keypoint in enumerate(keypoints):
            if not visible[i, k]:
                continue
            key = keypoint if by_keypoint else individual
            points.append(xyz[i, k])
            point_colors.append(tuple(color_map.get(key, _FALLBACK_RGBA)))
            labels.append(individual if by_keypoint else keypoint)
        for start, end in connections:
            a, b = kp_index.get(start), kp_index.get(end)
            if a is None or b is None or not (visible[i, a] and visible[i, b]):
                continue
            colour = bone_colors.get((start, end), bone) if bone_colors else bone
            edges.extend((xyz[i, a], xyz[i, b]))
            edge_colors.extend((colour, colour))

    return PoseFrame(
        points=np.array(points, dtype=np.float32).reshape(-1, 3),
        point_colors=np.array(point_colors, dtype=np.float32).reshape(-1, 4),
        edges=np.array(edges, dtype=np.float32).reshape(-1, 3),
        edge_colors=np.array(edge_colors, dtype=np.float32).reshape(-1, 4),
        labels=labels,
    )


def _keypoint_columns(plot_data, dims: PoseDims) -> np.ndarray:
    """``(T, K)`` in the dims' keypoint order, from a select that left the keypoint dim free."""
    data = np.asarray(plot_data.data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    labels = list(getattr(plot_data, "dim_labels", None) or [])
    keypoints = list(dims.keypoints)
    if len(labels) == data.shape[1] and set(labels) == set(keypoints):
        data = data[:, [labels.index(k) for k in keypoints]]
    if data.shape[1] != len(keypoints):
        raise ValueError(f"{POSE_FEATURE} selected as {data.shape}, expected {len(keypoints)} keypoint columns")
    return data


def skeleton_connections(config: dict | None) -> list[tuple[str, str]]:
    """``(start, end)`` per bone of a resolved skeleton config.

    The config's own edge colours are the video overlay's; here a bone's colour
    follows the colour-by choice (see :func:`pose_frame`).
    """
    if not config:
        return []
    return [(conn["start"], conn["end"]) for conn in config.get("connections") or []]


def skeleton_bone_colors(config: dict | None) -> dict[tuple[str, str], tuple]:
    """RGBA per ``(start, end)`` bone of a resolved config — the overlay's own edge colours."""
    if not config:
        return {}
    return {
        (conn["start"], conn["end"]): hex_to_rgba(conn["color"])
        for conn in config.get("connections") or []
        if conn.get("color")
    }


def pose_for_editor(app_state) -> tuple[list[str], np.ndarray] | None:
    """``(keypoints, positions (T, K, 2))`` from the dataset's ``position``, for the skeleton editor.

    The editor is a 2D canvas, so a 3D pose is shown in the projection with the
    widest spread — the view that separates the keypoints best. The first
    individual is drawn, at the first value of any extra dim, over the current window.
    Returns ``None`` when the dataset has no pose variable.
    """
    store = getattr(app_state, "data_loader", None)
    dims = pose_dims(store, POSE_FEATURE)
    bounds = getattr(app_state, "window_bounds", None)
    if store is None or dims is None or bounds is None:
        return None
    selection = {dim: values[0] for dim, values in dims.extra}
    if dims.individual_dim:
        selection[dims.individual_dim] = dims.individuals[0]
    columns = []
    for axis in dims.axes:
        plot_data = store.select(POSE_FEATURE, {**selection, dims.space_dim: axis}, t0=bounds.start_s, t1=bounds.end_s)
        if plot_data is None or plot_data.data is None:
            return None
        columns.append(_keypoint_columns(plot_data, dims))
    n = min(len(c) for c in columns)
    cube = np.stack([c[:n] for c in columns], axis=2)  # (T, K, D)
    if cube.shape[2] > 2:
        spread = np.nanmax(cube, axis=(0, 1)) - np.nanmin(cube, axis=(0, 1))
        spread = np.nan_to_num(spread, nan=0.0)
        pair = sorted(np.argsort(spread)[-2:])
        cube = cube[:, :, pair]
    return list(dims.keypoints), cube


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


class SkeletonPlot(IndividualPinMixin, QWidget):
    """The pose at the time marker on a 2D or 3D canvas.

    The individual combo is the panel's pin: it shows the pin, else the
    sidebar's individual, and picking a name pins the panel
    (:pyattr:`pin_changed`); "All" draws every animal at once and leaves the
    pin — whose labels a click on the panel creates — untouched.
    """

    #: Emitted on any mouse press → switches the sidebar to the Skeleton context.
    clicked = Signal()
    #: Emitted with ``self`` when the user closes this panel's dock.
    closed = Signal(object)
    #: Emitted with ``self`` when the user pinned this panel through its own
    #: individual combo; the owner re-titles it and moves the labelling subject.
    pin_changed = Signal(object)

    #: Monotonic counter so every instance's dock gets a unique objectName.
    _dock_seq = 0

    def __init__(self, shell, app_state):
        super().__init__()
        self.shell = shell
        self.app_state = app_state
        self.dock_widget = None
        self.dock_object_name: str | None = None
        self._dock_name = "Skeleton Plot"
        self._apply_default_width = True

        self._store = None
        self._dims: PoseDims | None = None
        #: ``(T, I, K, D)`` for the shown individuals over the window, and its clock.
        self._coords: np.ndarray | None = None
        self._time: np.ndarray | None = None
        self._shown: tuple[str | None, ...] = ()
        self._cache_key: tuple | None = None
        self._t: float | None = None
        self._connections: list[tuple[str, str]] | None = None
        self._bone_colors: dict[tuple[str, str], tuple] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 0)
        root.setSpacing(4)

        # --- Controls (re-parented into the sidebar's Skeleton context) ---
        self.controls_widget = QWidget()
        controls = QVBoxLayout(self.controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)

        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.setSpacing(6)
        self.cb_3d = QCheckBox("3D")
        self.cb_3d.setToolTip("Orbit the pose in three dimensions (needs a z axis)")
        view_row.addWidget(self.cb_3d)
        self.cb_flip_y = QCheckBox("Flip Y")
        self.cb_flip_y.setToolTip("Image coordinates grow downwards; flip so the pose is upright (2D only)")
        view_row.addWidget(self.cb_flip_y)
        view_row.addStretch()
        controls.addLayout(view_row)

        mesh_row = QHBoxLayout()
        mesh_row.setContentsMargins(0, 0, 0, 0)
        mesh_row.setSpacing(6)
        mesh_row.addWidget(QLabel("Meshes"))
        self.mesh_button = QToolButton()
        self.mesh_button.setPopupMode(QToolButton.InstantPopup)
        self.mesh_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mesh_menu = QMenu(self.mesh_button)
        # Read the folder each time the menu opens, so added or renamed files show at once.
        self.mesh_menu.aboutToShow.connect(self._refill_mesh_menu)
        self.mesh_button.setMenu(self.mesh_menu)
        self.mesh_button.setToolTip(
            "Draw parts from the project's meshes/ folder on their keypoints, "
            "instead of the bones between them (3D only)"
        )
        mesh_row.addWidget(self.mesh_button)
        controls.addLayout(mesh_row)
        self._parts: dict[str, MeshModel] = {}
        #: One scale per (part, animal) for the loaded window, so a rigid mesh keeps its size.
        self._scales: dict[tuple[str, int], float | None] = {}
        self._scales_for: object = None
        self._mesh_items: list[gl.GLMeshItem] = []
        self._update_mesh_button()

        individual_row = QHBoxLayout()
        individual_row.setContentsMargins(0, 0, 0, 0)
        individual_row.setSpacing(6)
        self.individual_label = QLabel("Individual")
        individual_row.addWidget(self.individual_label)
        self.individual_combo = QComboBox()
        self.individual_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        individual_row.addWidget(self.individual_combo)
        self.cb_all_individuals = QCheckBox("All")
        self.cb_all_individuals.setToolTip("Draw every individual at once")
        individual_row.addWidget(self.cb_all_individuals)
        controls.addLayout(individual_row)

        # One combo per dim the pose has beyond space/keypoint/individual
        # (e.g. ``position_type``): a pose is drawn at one value of each.
        self._extra_rows = QVBoxLayout()
        self._extra_rows.setContentsMargins(0, 0, 0, 0)
        self._extra_rows.setSpacing(4)
        controls.addLayout(self._extra_rows)
        self._extra_combos: dict[str, QComboBox] = {}

        # The video's own pose controls, synced through app_state: what is
        # hidden, sized or coloured here is so on the video and every panel.
        self.pose_controls = PoseControls(app_state, keypoints=lambda: list(self._dims.keypoints) if self._dims else [])
        controls.addWidget(self.pose_controls)

        # --- Canvas: a stable holder; the 2D/3D widget is swapped inside it ---
        self._canvas_holder = QWidget()
        self._canvas_layout = QVBoxLayout(self._canvas_holder)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas_holder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._canvas_holder, 1)
        self.canvas = None
        self.is_3d = False
        self._points_item = None
        self._edge_items: dict[tuple, pg.PlotCurveItem] = {}
        self._gl_edges_item = None
        self._text_items: list = []

        self._restyle_timer = QTimer(self)
        self._restyle_timer.setSingleShot(True)
        self._restyle_timer.setInterval(50)
        self._restyle_timer.timeout.connect(self._draw)

        self.cb_3d.toggled.connect(self._on_view_changed)
        self.cb_flip_y.toggled.connect(self._on_view_changed)
        self.individual_combo.currentIndexChanged.connect(self._on_individual_combo)
        self.cb_all_individuals.toggled.connect(self._on_all_individuals)

        if app_state is not None:
            for name in _STYLE_SETTINGS:
                getattr(app_state, f"{name}_changed").connect(self._on_style_changed)
            for name in _SKELETON_SETTINGS:
                getattr(app_state, f"{name}_changed").connect(self._on_skeleton_changed)
            for name in _REFERENCE_SETTINGS:
                getattr(app_state, f"{name}_changed").connect(self._draw_references)
            app_state.pose_hide_threshold_changed.connect(self._on_data_changed)

        self._ensure_canvas(False)

    # ------------------------------------------------------------------
    # Docking (mirrors SpacePlot / RadialPlot)
    # ------------------------------------------------------------------

    def show(self):
        if not self.dock_widget:
            SkeletonPlot._dock_seq += 1
            seq = SkeletonPlot._dock_seq
            self._dock_name = "Skeleton Plot" if seq == 1 else f"Skeleton Plot {seq}"
            self.dock_widget = self.shell.add_dock_widget(
                self, area="top", name=self._dock_name, object_name=self.dock_object_name
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

    def hide(self):
        if self.dock_widget:
            self.dock_widget.setVisible(False)
        super().hide()

    def refresh_title(self) -> None:
        """Re-title the dock: its name, then which individual it shows and why."""
        if self.dock_widget is not None:
            self.dock_widget.setWindowTitle(self._dock_name + self.app_state.panel_mode_suffix(self))

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
        if obj is self.dock_widget:
            if event.type() == QEvent.Close:
                self.closed.emit(self)
            return False
        if event.type() == QEvent.MouseButtonPress:
            self.clicked.emit()
        return False

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    # The panel's individual
    # ------------------------------------------------------------------

    def set_pinned_individual(self, individual: str | None) -> None:
        super().set_pinned_individual(individual)
        if individual is not None and self.cb_all_individuals.isChecked():
            # An explicit pin names one individual to draw.
            self.cb_all_individuals.blockSignals(True)
            self.cb_all_individuals.setChecked(False)
            self.cb_all_individuals.blockSignals(False)
            self.individual_combo.setEnabled(True)
        self.sync_individual()

    def sync_individual(self) -> None:
        """Point the individual combo at this panel's individual (its pin, else the sidebar's) and redraw."""
        target = self.app_state.panel_individual(self)
        if target is None:
            return
        index = self.individual_combo.findText(str(target))
        if index >= 0 and index != self.individual_combo.currentIndex():
            self.individual_combo.blockSignals(True)
            self.individual_combo.setCurrentIndex(index)
            self.individual_combo.blockSignals(False)
        self._on_data_changed()

    def _pin_through_combo(self, individual: str) -> None:
        """Picking an individual in the panel's own combo pins the panel to it."""
        if individual == self.app_state.panel_individual(self):
            return
        IndividualPinMixin.set_pinned_individual(self, individual)
        self.refresh_title()
        self.pin_changed.emit(self)

    def _on_individual_combo(self, _index: int) -> None:
        self._on_data_changed()
        name = self.individual_combo.currentText()
        if name:
            self._pin_through_combo(name)

    def _on_all_individuals(self, checked: bool) -> None:
        self.individual_combo.setEnabled(not checked)
        self._on_data_changed()

    def shown_individuals(self) -> tuple[str | None, ...]:
        """The individuals drawn: all of them, the combo's one, or ``(None,)`` for a pose without that dim."""
        dims = self._dims
        if dims is None or dims.individual_dim is None:
            return (None,)
        if self.cb_all_individuals.isChecked():
            return dims.individuals
        name = self.individual_combo.currentText()
        return (name,) if name else ()

    # ------------------------------------------------------------------
    # Store, dims, controls
    # ------------------------------------------------------------------

    def set_store(self, store) -> None:
        """Bind the loader (again on every trial: the dataset behind it changed)."""
        self._store = store
        dims = pose_dims(store)
        if dims != self._dims:
            self._dims = dims
            self._rebuild_controls()
        self._cache_key = None
        self._connections = None

    def _refill_mesh_menu(self) -> None:
        """The project's parts as checkable entries, the checked ones by name."""
        project = project_dir_of(self.app_state) if self.app_state is not None else None
        self.mesh_menu.clear()
        paths = available_meshes(project)
        if not paths:
            self.mesh_menu.addAction("No meshes in the project's meshes/ folder").setEnabled(False)
        for path in paths:
            action = self.mesh_menu.addAction(path.stem)
            action.setCheckable(True)
            action.setChecked(path.stem in self._parts)
            action.toggled.connect(lambda on, path=path: self._toggle_part(path, on))

    def _toggle_part(self, path: Path, on: bool) -> None:
        if on:
            self._parts[path.stem] = load_mesh(path)
        else:
            self._parts.pop(path.stem, None)
        self._scales.clear()
        self._connections = None
        self._update_mesh_button()
        self._draw()

    def set_parts(self, names: list[str]) -> None:
        """Check *names* (those still in the project's folder)."""
        project = project_dir_of(self.app_state) if self.app_state is not None else None
        by_name = {p.stem: p for p in available_meshes(project)}
        self._parts = {n: load_mesh(by_name[n]) for n in names if n in by_name}
        self._scales.clear()
        self._connections = None
        self._update_mesh_button()

    def _update_mesh_button(self) -> None:
        self.mesh_button.setText(", ".join(self._parts) or "None")

    def _part_keypoints(self) -> set[str]:
        """The keypoints the checked parts stand on: the bones among them are not drawn."""
        names: set[str] = set()
        for part in self._parts.values():
            names |= set(part.landmarks)
        return names

    def _rebuild_controls(self) -> None:
        """Keypoint list and individual combo from the dims, keeping the user's choices by name."""
        dims = self._dims
        current = self.individual_combo.currentText()
        self.individual_combo.blockSignals(True)
        self.individual_combo.clear()
        if dims is not None and dims.individual_dim is not None:
            self.individual_combo.addItems(list(dims.individuals))
            if self.app_state is not None:
                decorate_individual_combo(self.individual_combo, self.app_state)
            target = self.app_state.panel_individual(self) if self.app_state is not None else None
            for candidate in (target, current):
                index = self.individual_combo.findText(str(candidate)) if candidate else -1
                if index >= 0:
                    self.individual_combo.setCurrentIndex(index)
                    break
        self.individual_combo.blockSignals(False)
        several = dims is not None and len(dims.individuals) > 1
        for widget in (self.individual_label, self.individual_combo, self.cb_all_individuals):
            widget.setVisible(several)

        previous = self.extra_selections()
        while self._extra_rows.count():
            item = self._extra_rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._extra_combos = {}
        for name, values in dims.extra if dims else ():
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            layout.addWidget(QLabel(name.capitalize()))
            combo = QComboBox()
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            combo.addItems(list(values))
            index = combo.findText(previous.get(name, ""))
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.currentIndexChanged.connect(self._on_data_changed)
            layout.addWidget(combo)
            self._extra_rows.addWidget(row)
            self._extra_combos[name] = combo

        has_z = dims is not None and dims.has_z
        self.cb_3d.setEnabled(has_z)
        if not has_z and self.cb_3d.isChecked():
            self.cb_3d.setChecked(False)

    def extra_selections(self) -> dict[str, str]:
        """The value pinned for each dim beyond space/keypoint/individual."""
        return {name: combo.currentText() for name, combo in self._extra_combos.items() if combo.currentText()}

    def hidden_keypoints(self) -> set[str]:
        """This pose's keypoints hidden anywhere — the one list every view shares."""
        hidden = set(self.app_state.pose_hidden_keypoints)
        return {k for k in self._dims.keypoints if k in hidden} if self._dims else set()

    def shown_keypoints(self) -> set[str]:
        return set(self._dims.keypoints) - self.hidden_keypoints() if self._dims else set()

    def color_by(self) -> str:
        return str(self.app_state.pose_color_by or COLOR_BY_KEYPOINT)

    def configure(self, view_3d: bool | None = None) -> None:
        if view_3d is not None:
            self.cb_3d.blockSignals(True)
            self.cb_3d.setChecked(bool(view_3d) and self.cb_3d.isEnabled())
            self.cb_3d.blockSignals(False)
            self._ensure_canvas(self.cb_3d.isChecked())
        self.refresh()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _on_data_changed(self, *_args) -> None:
        self._cache_key = None
        self._draw()

    def _on_skeleton_changed(self, *_args) -> None:
        self._connections = None
        self._restyle_timer.start()

    def _on_style_changed(self, *_args) -> None:
        if self.app_state is not None and self.individual_combo.count():
            decorate_individual_combo(self.individual_combo, self.app_state)
        self._restyle_timer.start()

    def _draw_references(self, *_args) -> None:
        draw_reference_geometry(self.canvas, self.app_state)

    def _ensure_data(self) -> bool:
        """Load the whole window once and index into it per tick — a select()
        per time-marker tick would re-query the backend at frame rate."""
        dims, store = self._dims, self._store
        bounds = getattr(self.app_state, "window_bounds", None)
        shown = self.shown_individuals()
        if dims is None or store is None or bounds is None or not shown:
            self._coords = self._time = None
            return False
        extra = self.extra_selections()
        key = (
            shown,
            tuple(sorted(extra.items())),
            bounds.start_s,
            bounds.end_s,
            getattr(self.app_state, "trials_sel", None),
            dims,
        )
        if key == self._cache_key and self._coords is not None:
            return True

        def selection_for(individual: str | None) -> dict[str, str]:
            """Everything pinned but the keypoint dim, which stays free: one column per keypoint."""
            selection = dict(extra)
            if individual is not None and dims.individual_dim:
                selection[dims.individual_dim] = individual
            return selection

        blocks: dict[tuple[int, int], np.ndarray] = {}
        time: np.ndarray | None = None
        for i, individual in enumerate(shown):
            for d, axis in enumerate(dims.axes):
                selection = {**selection_for(individual), dims.space_dim: axis}
                plot_data = store.select(POSE_FEATURE, selection, t0=bounds.start_s, t1=bounds.end_s)
                if plot_data is None or plot_data.data is None:
                    self._coords = self._time = None
                    return False
                blocks[(i, d)] = _keypoint_columns(plot_data, dims)
                if time is None:
                    time = np.asarray(plot_data.time, dtype=float)
        if time is None:
            return False
        cube = np.full((len(time), len(shown), len(dims.keypoints), len(dims.axes)), np.nan)
        for (i, d), columns in blocks.items():
            n = min(len(time), len(columns))
            cube[:n, i, :, d] = columns[:n]

        # The video overlay's confidence filter, applied to the same points.
        threshold = float(getattr(self.app_state, "pose_hide_threshold", 0.0) or 0.0)
        if threshold > 0.0 and CONFIDENCE_FEATURE in list(getattr(store, "features", [])):
            for i, individual in enumerate(shown):
                conf = store.select(CONFIDENCE_FEATURE, selection_for(individual), t0=bounds.start_s, t1=bounds.end_s)
                if conf is None or conf.data is None:
                    continue
                columns = _keypoint_columns(conf, dims)
                n = min(len(time), len(columns))
                cube[:n, i][columns[:n] < threshold] = np.nan

        self._coords, self._time, self._shown = cube, time, shown
        self._cache_key = key
        self._fit_view()
        return True

    def _connections_now(self) -> list[tuple[str, str]]:
        if self._connections is None:
            config = resolve_skeleton_config(self.app_state, None)
            self._connections = skeleton_connections(config)
            if self._parts and self.is_3d:
                # A part stands in for the bones it spans.
                covered = self._part_keypoints()
                self._connections = [(a, b) for a, b in self._connections if not (a in covered and b in covered)]
            self._bone_colors = skeleton_bone_colors(config)
        return self._connections

    def _bone_colors_now(self) -> dict[tuple[str, str], tuple]:
        self._connections_now()
        return self._bone_colors

    def _marker_time(self) -> float:
        """The time marker's position on the display clock.

        Until the first marker tick reaches this panel, the video's frame is
        the marker (the same reading the space plot takes), so a panel opened
        mid-trial draws the pose that is on screen rather than nothing.
        """
        if self._t is not None:
            return self._t
        state = self.app_state
        frame = int(getattr(state, "current_frame", 0) or 0)
        video = getattr(state, "video", None)
        if video:
            return float(video.frame_to_time(frame))
        fps = getattr(state, "video_fps", None)
        t_rel = frame / fps if fps else 0.0
        return float(state.to_display(getattr(state, "trials_sel", None), t_rel))

    def _frame_index(self, time: np.ndarray) -> int:
        return int(np.clip(np.searchsorted(time, self._marker_time(), side="right") - 1, 0, len(time) - 1))

    def current_frame(self) -> PoseFrame | None:
        """The vertex arrays under the time marker, or ``None`` without data."""
        if not self._ensure_data():
            return None
        dims, coords, time = self._dims, self._coords, self._time
        if dims is None or coords is None or time is None or len(time) == 0:
            return None
        index = self._frame_index(time)
        individuals = tuple(name or "" for name in self._shown)
        color_by = self.color_by()
        if color_by == COLOR_BY_INDIVIDUAL:
            # The dataset's own list, in its order, so an animal keeps its
            # colour whether drawn alone or with the others.
            values = list(self.app_state.label_individuals())
            values += [v for v in dims.individuals if v not in values]
        else:
            values = list(dims.keypoints)
        color_map = pose_color_map(self.app_state, color_by, values)
        show_points = bool(self.app_state.pose_show_keypoints)
        show_skeleton = bool(self.app_state.pose_show_skeleton)
        anything = show_points or show_skeleton or bool(self.app_state.pose_show_text)
        return pose_frame(
            coords[index],
            dims.keypoints,
            individuals,
            shown_keypoints=self.shown_keypoints() if anything else set(),
            color_by=color_by,
            color_map=color_map,
            connections=self._connections_now() if show_skeleton else [],
            bone_colors=self._bone_colors_now(),
        )

    def set_time(self, t: float) -> None:
        self._t = float(t)
        self._draw()

    def refresh(self) -> None:
        self._cache_key = None
        self._draw()

    # ------------------------------------------------------------------
    # Canvas + rendering
    # ------------------------------------------------------------------

    def _on_view_changed(self, *_args) -> None:
        self._ensure_canvas(self.cb_3d.isChecked())
        self._fit_view()
        self._draw()

    def _ensure_canvas(self, view_3d: bool) -> bool:
        """(Re)create the canvas only when the 2D/3D type changes."""
        if self.canvas is not None and self.is_3d == view_3d:
            if not view_3d:
                self.canvas.getPlotItem().vb.invertY(self.cb_flip_y.isChecked())
            return False
        if self.canvas is not None:
            self._canvas_layout.removeWidget(self.canvas)
            self.canvas.hide()
            self.canvas.deleteLater()
        self.canvas, self.is_3d = make_space_widget(view_3d)
        self._mesh_items = []
        self._connections = None
        if view_3d and not self.is_3d:
            self.cb_3d.blockSignals(True)
            self.cb_3d.setChecked(False)
            self.cb_3d.blockSignals(False)
        if not self.is_3d:
            plot_item = self.canvas.getPlotItem()
            plot_item.setAspectLocked(True)
            plot_item.vb.invertY(self.cb_flip_y.isChecked())
        self._points_item = None
        self._gl_edges_item = None
        self._edge_items = {}
        self._text_items = []
        self._canvas_layout.addWidget(self.canvas)
        self.canvas.installEventFilter(self)
        for child in self.canvas.findChildren(QWidget):
            child.installEventFilter(self)
        self._draw_references()
        return True

    def _fit_view(self) -> None:
        """Frame the whole window's pose once per data load, so the view does
        not chase the animal from tick to tick."""
        cube = self._coords
        if cube is None or self.canvas is None or not np.isfinite(cube).any():
            return
        flat = cube.reshape(-1, cube.shape[-1])
        if self.is_3d:
            z = flat[:, 2] if flat.shape[1] > 2 else np.zeros(len(flat))
            auto_camera_3d(self.canvas, flat[:, 0], flat[:, 1], z)
        else:
            x_lo, x_hi = np.nanmin(flat[:, 0]), np.nanmax(flat[:, 0])
            y_lo, y_hi = np.nanmin(flat[:, 1]), np.nanmax(flat[:, 1])
            self.canvas.getPlotItem().vb.setRange(xRange=(x_lo, x_hi), yRange=(y_lo, y_hi), padding=0.1)

    def _draw(self, *_args) -> None:
        if self.canvas is None:
            return
        frame = self.current_frame()
        if frame is None:
            empty3, empty4 = np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)
            frame = PoseFrame(empty3, empty4, empty3, empty4)
        labelled = (frame.points, frame.labels)
        if not self.app_state.pose_show_keypoints:
            frame.points = frame.points[:0]
            frame.point_colors = frame.point_colors[:0]
        if self.is_3d:
            self._draw_gl(frame)
        else:
            self._draw_2d(frame)
        points, labels = labelled
        self._draw_text(points, labels if self.app_state.pose_show_text else [])

    def _point_size(self) -> float:
        return float(getattr(self.app_state, "pose_point_size", 10.0) or 10.0)

    def _edge_width(self) -> float:
        return float(getattr(self.app_state, "pose_skeleton_width", 2.0) or 2.0)

    def _draw_text(self, points: np.ndarray, labels: list[str]) -> None:
        """A label beside each point, sized by the shared text size."""
        for item in self._text_items:
            if self.is_3d:
                self.canvas.removeItem(item)
            else:
                self.canvas.getPlotItem().removeItem(item)
        self._text_items = []
        size = int(getattr(self.app_state, "pose_text_size", 12.0) or 12.0)
        for point, label in zip(points, labels):
            if not label:
                continue
            if self.is_3d:
                font = QFont()
                font.setPointSize(size)
                item = gl.GLTextItem(pos=point, text=label, color=(255, 255, 255, 255), font=font)
                self.canvas.addItem(item)
            else:
                item = pg.TextItem(label, color=(230, 230, 230), anchor=(0, 1))
                item.setFont(QFont("", size))
                item.setPos(float(point[0]), float(point[1]))
                item.setZValue(20)
                self.canvas.getPlotItem().addItem(item)
            self._text_items.append(item)

    def _draw_gl(self, frame: PoseFrame) -> None:
        if self._points_item is None:
            self._points_item = gl.GLScatterPlotItem(pxMode=True, glOptions="translucent")
            self.canvas.addItem(self._points_item)
        if self._gl_edges_item is None:
            self._gl_edges_item = gl.GLLinePlotItem(mode="lines", antialias=True, glOptions="opaque")
            self.canvas.addItem(self._gl_edges_item)
        if len(frame.points):
            self._points_item.setData(pos=frame.points, color=frame.point_colors, size=self._point_size())
            self._points_item.setVisible(True)
        else:
            self._points_item.setVisible(False)
        if len(frame.edges):
            self._gl_edges_item.setData(pos=frame.edges, color=frame.edge_colors, width=self._edge_width())
            self._gl_edges_item.setVisible(True)
        else:
            self._gl_edges_item.setVisible(False)
        self._draw_meshes()

    def _scale(self, name: str, part: MeshModel, animal: int, keypoints: tuple[str, ...]) -> float | None:
        """The part's size on this animal over the loaded window: one number, so it never breathes."""
        if self._scales_for is not self._coords:
            self._scales, self._scales_for = {}, self._coords
        key = (name, animal)
        if key not in self._scales and self._coords is not None:
            self._scales[key] = fit_scale(part, keypoints, self._xyz(self._coords[:, animal]))
        return self._scales.get(key)

    @staticmethod
    def _xyz(coords: np.ndarray) -> np.ndarray:
        xyz = np.zeros(coords.shape[:-1] + (3,))
        xyz[..., : min(coords.shape[-1], 3)] = coords[..., :3]
        return xyz

    def _draw_meshes(self) -> None:
        """Each checked part on each shown animal at the marker; hidden where its keypoints are missing."""
        placed: list[tuple[np.ndarray, MeshModel, tuple]] = []
        dims, coords, time = self._dims, self._coords, self._time
        if self._parts and dims is not None and coords is not None and time is not None and len(time):
            frame = self._xyz(coords[self._frame_index(time)])
            for name, part in self._parts.items():
                for i, animal in enumerate(frame):
                    scale = self._scale(name, part, i, dims.keypoints)
                    vertices = place_mesh(part, dims.keypoints, animal, scale)
                    if vertices is not None:
                        placed.append((vertices, part, _MESH_RGBA))
        while len(self._mesh_items) < len(placed):
            item = gl.GLMeshItem(smooth=True, shader="shaded", glOptions="translucent")
            self.canvas.addItem(item)
            self._mesh_items.append(item)
        for i, item in enumerate(self._mesh_items):
            if i < len(placed):
                vertices, part, rgba = placed[i]
                if part.face_colors is not None:
                    # Vertex colours: in smooth mode pyqtgraph reads face colours as if per vertex.
                    item.setMeshData(vertexes=vertices, faces=part.faces, vertexColors=part.vertex_colors)
                else:
                    item.setMeshData(vertexes=vertices, faces=part.faces)
                    item.setColor(rgba)
            item.setVisible(i < len(placed))

    def _draw_2d(self, frame: PoseFrame) -> None:
        plot_item = self.canvas.getPlotItem()
        if self._points_item is None:
            self._points_item = pg.ScatterPlotItem(pxMode=True, pen=None)
            self._points_item.setZValue(10)
            plot_item.addItem(self._points_item)
        brushes = [pg.mkBrush(*(int(c * 255) for c in rgba)) for rgba in frame.point_colors]
        self._points_item.setData(pos=frame.points[:, :2], brush=brushes, size=self._point_size())

        # One curve per bone colour: a PlotCurveItem has one pen, and the
        # skeleton config may colour each bone differently.
        by_color: dict[tuple, list[np.ndarray]] = {}
        for k in range(0, len(frame.edges), 2):
            by_color.setdefault(tuple(frame.edge_colors[k]), []).append(frame.edges[k : k + 2, :2])
        for rgba, item in self._edge_items.items():
            if rgba not in by_color:
                item.setData(x=[], y=[])
        for rgba, segments in by_color.items():
            xy = np.concatenate(segments)
            item = self._edge_items.get(rgba)
            if item is None:
                pen = pg.mkPen(pg.mkColor(*(int(c * 255) for c in rgba)), width=self._edge_width())
                item = pg.PlotCurveItem(connect="pairs", pen=pen)
                item.setZValue(5)
                plot_item.addItem(item)
                self._edge_items[rgba] = item
            else:
                item.setPen(pg.mkPen(pg.mkColor(*(int(c * 255) for c in rgba)), width=self._edge_width()))
            item.setData(x=xy[:, 0], y=xy[:, 1])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def skeleton_settings(self) -> dict:
        return {
            "view_3d": self.cb_3d.isChecked(),
            "flip_y": self.cb_flip_y.isChecked(),
            "all_individuals": self.cb_all_individuals.isChecked(),
            "dims": self.extra_selections(),
            "meshes": list(self._parts),
            **({"individual": self.pinned_individual} if self.pinned_individual else {}),
        }

    def apply_skeleton_settings(self, settings: dict) -> None:
        for widget, value in (
            (self.cb_flip_y, settings.get("flip_y")),
            (self.cb_all_individuals, settings.get("all_individuals")),
        ):
            if value is not None:
                widget.blockSignals(True)
                widget.setChecked(bool(value))
                widget.blockSignals(False)
        self.individual_combo.setEnabled(not self.cb_all_individuals.isChecked())
        # Before the combo: the individual combo is built from the pin.
        IndividualPinMixin.set_pinned_individual(self, settings.get("individual"))
        self._rebuild_controls()
        for name, value in (settings.get("dims") or {}).items():
            combo = self._extra_combos.get(name)
            if combo is None:
                continue
            index = combo.findText(str(value))
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
        self.set_parts(list(settings.get("meshes") or []))
        self.configure(view_3d=settings.get("view_3d"))
