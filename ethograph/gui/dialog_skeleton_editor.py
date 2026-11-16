"""Interactive skeleton editor dialog.

Build/edit a skeleton on top of *real* pose data:

- A frame slider scrubs through frames; the canvas shows the keypoint XY
  positions at that frame (image coordinates, y pointing down).
- Drag from one keypoint to another to create a connection (edge).
- A side panel holds color categories. Select edges (click, or rubber-band
  drag over several) and assign the active category to color them and tag
  their segment name.

``get_config()`` returns the standard skeleton config dict consumed by
``ethograph.skeleton`` (and convertible to NWB nodes/edges via the config
layer), so the result renders through the same ``PrecomputedRenderer`` path.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from ethograph.skeleton.shapes import SHAPE_TEMPLATES, shape_outline_for_frame

_DEFAULT_COLOR = "#CCCCCC"
_SHAPE_COLOR = "#FFCC00"
_HIT_RADIUS_PX = 14.0


def _hex(color: QColor) -> str:
    return color.name().upper()


def _color_swatch_icon(color_hex: str) -> QIcon:
    pix = QPixmap(16, 16)
    pix.fill(QColor(color_hex))
    return QIcon(pix)


class _SkeletonCanvas(pg.PlotWidget):
    """pyqtgraph canvas with drag-to-connect and rubber-band edge selection.

    The view box's own mouse handling is disabled so press/move/release drive
    edge creation and selection instead of pan/zoom.
    """

    def __init__(self, keypoints: list[str], positions: np.ndarray, parent=None):
        super().__init__(parent)
        self.keypoints = keypoints
        self.positions = positions  # (n_frames, n_keypoints, 2) -> (x, y)
        self.n_frames = positions.shape[0]
        self.frame = 0

        # Model (owned by the dialog, shared by reference).
        self.edges: list[tuple[int, int]] = []
        self.edge_colors: list[str] = []
        self.edge_segments: list[str] = []
        self.shapes: list[dict] = []  # anchored shape definitions (preview)
        self.selected: set[int] = set()
        self.on_selection_changed = lambda: None

        self._vb = self.getPlotItem().getViewBox()
        self._vb.setMouseEnabled(x=False, y=False)
        self._vb.setMenuEnabled(False)
        self.setBackground("k")
        self.getPlotItem().invertY(True)  # image coordinates: y grows downward
        self.getPlotItem().setAspectLocked(True)
        self.getPlotItem().hideButtons()

        self._scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen("w"), brush=pg.mkBrush(80, 160, 255))
        self.addItem(self._scatter)
        self._labels: list[pg.TextItem] = []
        self._edge_items: list[pg.PlotCurveItem] = []
        self._shape_items: list[pg.PlotCurveItem] = []
        self._temp_line: pg.PlotCurveItem | None = None
        self._rubber: pg.PlotCurveItem | None = None

        self._drag_from: int | None = None
        self._press_view: tuple[float, float] | None = None
        self._rubber_origin: tuple[float, float] | None = None
        self._pan_last: tuple[float, float] | None = None

        self._fit_view()
        self.redraw()

    # ── coordinate helpers ──

    def _event_view(self, ev) -> tuple[float, float]:
        pt = self._vb.mapSceneToView(self.mapToScene(ev.pos()))
        return float(pt.x()), float(pt.y())

    def _px_per_unit(self) -> tuple[float, float]:
        sx, sy = self._vb.viewPixelSize()  # view units per pixel
        return (1.0 / sx if sx else 1.0, 1.0 / sy if sy else 1.0)

    def _pixel_dist(self, ax, ay, bx, by) -> float:
        ppx, ppy = self._px_per_unit()
        return float(np.hypot((ax - bx) * ppx, (ay - by) * ppy))

    def _frame_xy(self) -> np.ndarray:
        return self.positions[self.frame]

    def _valid(self, idx: int) -> bool:
        return not np.any(np.isnan(self.positions[self.frame, idx]))

    def _nearest_node(self, vx: float, vy: float) -> int | None:
        xy = self._frame_xy()
        best, best_d = None, _HIT_RADIUS_PX
        for i in range(len(self.keypoints)):
            if np.any(np.isnan(xy[i])):
                continue
            d = self._pixel_dist(vx, vy, xy[i, 0], xy[i, 1])
            if d < best_d:
                best, best_d = i, d
        return best

    def _nearest_edge(self, vx: float, vy: float) -> int | None:
        xy = self._frame_xy()
        best, best_d = None, _HIT_RADIUS_PX
        for ei, (a, b) in enumerate(self.edges):
            if np.any(np.isnan(xy[a])) or np.any(np.isnan(xy[b])):
                continue
            d = self._point_segment_px(vx, vy, xy[a], xy[b])
            if d < best_d:
                best, best_d = ei, d
        return best

    def _point_segment_px(self, vx, vy, p0, p1) -> float:
        ppx, ppy = self._px_per_unit()
        ax, ay = (vx - p0[0]) * ppx, (vy - p0[1]) * ppy
        bx, by = (p1[0] - p0[0]) * ppx, (p1[1] - p0[1]) * ppy
        seg2 = bx * bx + by * by
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, (ax * bx + ay * by) / seg2))
        return float(np.hypot(ax - t * bx, ay - t * by))

    # ── mouse handling ──

    def wheelEvent(self, ev):
        """Zoom about the cursor; the view box's own handling is off, so this is the only zoom."""
        steps = ev.angleDelta().y() / 120.0
        if not steps:
            return
        factor = 0.85**steps
        pt = self._vb.mapSceneToView(self.mapToScene(ev.position().toPoint()))
        self._vb.scaleBy((factor, factor), center=pt)
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self._fit_view()
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan_last = self._event_view(ev)
            ev.accept()
            return
        if ev.button() != Qt.LeftButton:
            return
        vx, vy = self._event_view(ev)
        self._press_view = (vx, vy)
        node = self._nearest_node(vx, vy)
        if node is not None:
            self._drag_from = node
            self._temp_line = pg.PlotCurveItem(pen=pg.mkPen("y", width=2, style=Qt.DashLine))
            self.addItem(self._temp_line)
        else:
            self._rubber_origin = (vx, vy)
            self._rubber = pg.PlotCurveItem(pen=pg.mkPen("w", width=1, style=Qt.DashLine))
            self.addItem(self._rubber)
        ev.accept()

    def mouseMoveEvent(self, ev):
        vx, vy = self._event_view(ev)
        if self._pan_last is not None:
            px, py = self._pan_last
            self._vb.translateBy(x=px - vx, y=py - vy)
            # After the shift the cursor maps back onto the grabbed point.
            ev.accept()
            return
        if self._drag_from is not None and self._temp_line is not None:
            xy = self._frame_xy()[self._drag_from]
            self._temp_line.setData([xy[0], vx], [xy[1], vy])
        elif self._rubber_origin is not None and self._rubber is not None:
            ox, oy = self._rubber_origin
            self._rubber.setData([ox, vx, vx, ox, ox], [oy, oy, vy, vy, oy])
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if ev.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan_last = None
            ev.accept()
            return
        if ev.button() != Qt.LeftButton:
            return
        vx, vy = self._event_view(ev)
        moved = self._press_view and self._pixel_dist(vx, vy, *self._press_view) > 4

        if self._drag_from is not None:
            target = self._nearest_node(vx, vy)
            if target is not None and target != self._drag_from:
                self._add_edge(self._drag_from, target)
            self._clear_temp()
        elif self._rubber_origin is not None:
            if moved:
                self._select_in_rect(self._rubber_origin, (vx, vy), ev)
            else:
                self._click_edge(vx, vy, ev)
            self._clear_rubber()
        self._press_view = None
        ev.accept()

    def _clear_temp(self):
        self._drag_from = None
        if self._temp_line is not None:
            self.removeItem(self._temp_line)
            self._temp_line = None

    def _clear_rubber(self):
        self._rubber_origin = None
        if self._rubber is not None:
            self.removeItem(self._rubber)
            self._rubber = None

    # ── model edits ──

    def _add_edge(self, a: int, b: int):
        pair = (a, b)
        if pair in self.edges or (b, a) in self.edges:
            return
        self.edges.append(pair)
        self.edge_colors.append(_DEFAULT_COLOR)
        self.edge_segments.append("")
        self.redraw()

    def _click_edge(self, vx, vy, ev):
        ei = self._nearest_edge(vx, vy)
        if ei is None:
            if not (ev.modifiers() & Qt.ShiftModifier):
                self.selected.clear()
        else:
            if ev.modifiers() & Qt.ShiftModifier:
                self.selected.symmetric_difference_update({ei})
            else:
                self.selected = {ei}
        self.redraw()
        self.on_selection_changed()

    def _select_in_rect(self, p0, p1, ev):
        x0, x1 = sorted((p0[0], p1[0]))
        y0, y1 = sorted((p0[1], p1[1]))
        xy = self._frame_xy()
        hits: set[int] = set()
        for ei, (a, b) in enumerate(self.edges):
            if np.any(np.isnan(xy[a])) or np.any(np.isnan(xy[b])):
                continue
            ts = np.linspace(0, 1, 12)
            pts = xy[a][None, :] * (1 - ts)[:, None] + xy[b][None, :] * ts[:, None]
            inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
            if np.any(inside):
                hits.add(ei)
        if ev.modifiers() & Qt.ShiftModifier:
            self.selected |= hits
        else:
            self.selected = hits
        self.redraw()
        self.on_selection_changed()

    def remove_selected(self):
        if not self.selected:
            return
        keep = [i for i in range(len(self.edges)) if i not in self.selected]
        self.edges = [self.edges[i] for i in keep]
        self.edge_colors = [self.edge_colors[i] for i in keep]
        self.edge_segments = [self.edge_segments[i] for i in keep]
        self.selected.clear()
        self.redraw()
        self.on_selection_changed()

    def apply_category(self, color_hex: str, segment: str):
        for ei in self.selected:
            self.edge_colors[ei] = color_hex
            self.edge_segments[ei] = segment
        self.redraw()

    # ── drawing ──

    def set_frame(self, frame: int):
        self.frame = int(frame)
        self.redraw()

    def _fit_view(self):
        finite = self.positions[np.isfinite(self.positions).all(axis=2)]
        if len(finite):
            self._vb.setRange(
                xRange=(finite[:, 0].min(), finite[:, 0].max()),
                yRange=(finite[:, 1].min(), finite[:, 1].max()),
                padding=0.1,
            )

    def redraw(self):
        xy = self._frame_xy()
        valid = np.isfinite(xy).all(axis=1)
        self._scatter.setData(xy[valid, 0], xy[valid, 1])

        for t in self._labels:
            self.removeItem(t)
        self._labels = []
        for i, name in enumerate(self.keypoints):
            if not valid[i]:
                continue
            t = pg.TextItem(name, color="w", anchor=(0, 1))
            t.setPos(xy[i, 0], xy[i, 1])
            self.addItem(t)
            self._labels.append(t)

        for it in self._edge_items:
            self.removeItem(it)
        self._edge_items = []
        for ei, (a, b) in enumerate(self.edges):
            if not (valid[a] and valid[b]):
                continue
            selected = ei in self.selected
            width = 5 if selected else 3
            pen = pg.mkPen(self.edge_colors[ei], width=width)
            item = pg.PlotCurveItem([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], pen=pen)
            self.addItem(item)
            self._edge_items.append(item)
            if selected:
                halo = pg.PlotCurveItem(
                    [xy[a, 0], xy[b, 0]],
                    [xy[a, 1], xy[b, 1]],
                    pen=pg.mkPen("w", width=1, style=Qt.DashLine),
                )
                self.addItem(halo)
                self._edge_items.append(halo)

        self._draw_shapes(xy)

    def _draw_shapes(self, xy: np.ndarray) -> None:
        for it in self._shape_items:
            self.removeItem(it)
        self._shape_items = []
        kp_index = {n: i for i, n in enumerate(self.keypoints)}
        for shape in self.shapes:
            template = SHAPE_TEMPLATES.get(shape.get("type", ""))
            anchors = shape.get("anchors", [])
            if template is None or len(anchors) < 2:
                continue
            if any(a["keypoint"] not in kp_index for a in anchors):
                continue
            anchor_names = [a["point"] for a in anchors]
            dst = np.array([xy[kp_index[a["keypoint"]]] for a in anchors])
            scale = tuple(shape.get("scale", (1.0, 1.0)))
            outline = shape_outline_for_frame(template, anchor_names, dst, scale)
            if outline is None:
                continue
            closed = np.vstack([outline, outline[0]])
            item = pg.PlotCurveItem(
                closed[:, 0],
                closed[:, 1],
                pen=pg.mkPen(shape.get("color", "#FFCC00"), width=2),
            )
            self.addItem(item)
            self._shape_items.append(item)


class ShapeAnchorDialog(QDialog):
    """Bind keypoints to control points of a shape template (visual picker).

    The template is drawn with its control points (corners, edge midpoints,
    centre). Click a control point, pick a keypoint, and Bind. Two anchors give
    a rigid/angle-preserving fit; three or more allow affine deformation.
    """

    def __init__(self, shape_type: str, keypoints: list[str], parent=None, existing=None):
        super().__init__(parent)
        self.setWindowTitle(f"Anchor shape: {shape_type}")
        self.setMinimumSize(620, 420)
        self.shape_type = shape_type
        self.template = SHAPE_TEMPLATES[shape_type]
        self.keypoints = keypoints
        self._bindings: list[dict] = list(existing.get("anchors", [])) if existing else []
        self._color = existing.get("color", _SHAPE_COLOR) if existing else _SHAPE_COLOR
        self._scale: list[float] = list(existing.get("scale", [1.0, 1.0])) if existing else [1.0, 1.0]
        self._selected_point: str | None = None
        self._template_items: list = []
        self._point_scatter: pg.ScatterPlotItem | None = None

        layout = QVBoxLayout(self)
        help_label = QLabel(
            "Click a yellow control point on the template, choose a keypoint, "
            "then <b>Bind point → keypoint</b>. Bind ≥2 points and the shape "
            "will follow those keypoints each frame (2 = rigid rotate/scale, "
            "3+ = deform). Use Proportions to resize the template."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #aaa;")
        layout.addWidget(help_label)
        body = QHBoxLayout()
        layout.addLayout(body, stretch=1)

        # Left: template canvas with clickable control points.
        self.canvas = pg.PlotWidget()
        self.canvas.getPlotItem().invertY(True)
        self.canvas.getPlotItem().setAspectLocked(True)
        self.canvas.getPlotItem().hideButtons()
        self.canvas.setBackground("k")
        self.canvas.getViewBox().setMouseEnabled(False, False)
        body.addWidget(self.canvas, stretch=2)

        # Right: binding controls.
        right = QVBoxLayout()
        self.sel_label = QLabel("Selected point: —")
        right.addWidget(self.sel_label)
        right.addWidget(QLabel("Keypoint:"))
        self.kp_combo = QComboBox()
        self.kp_combo.addItems(keypoints)
        right.addWidget(self.kp_combo)
        bind_btn = QPushButton("Bind point → keypoint")
        bind_btn.clicked.connect(self._bind)
        right.addWidget(bind_btn)
        right.addWidget(QLabel("<b>Anchors</b>"))
        self.bind_list = QListWidget()
        right.addWidget(self.bind_list, stretch=1)
        unbind_btn = QPushButton("Remove anchor")
        unbind_btn.clicked.connect(self._unbind)
        right.addWidget(unbind_btn)
        self.color_btn = QPushButton("Colour")
        self.color_btn.setIcon(_color_swatch_icon(self._color))
        self.color_btn.clicked.connect(self._pick_color)
        right.addWidget(self.color_btn)

        right.addWidget(QLabel("Proportions (template width × height):"))
        scale_row = QHBoxLayout()
        self.width_spin = QDoubleSpinBox()
        self.height_spin = QDoubleSpinBox()
        for spin, value in ((self.width_spin, self._scale[0]), (self.height_spin, self._scale[1])):
            spin.setRange(0.05, 20.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.setValue(value)
            spin.valueChanged.connect(self._on_scale_changed)
        scale_row.addWidget(QLabel("W"))
        scale_row.addWidget(self.width_spin)
        scale_row.addWidget(QLabel("H"))
        scale_row.addWidget(self.height_spin)
        right.addLayout(scale_row)
        body.addLayout(right, stretch=1)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._draw_template()
        self._refresh_bindings()

    def _on_scale_changed(self, *_):
        self._scale = [self.width_spin.value(), self.height_spin.value()]
        self._draw_template()

    def _draw_template(self):
        for it in self._template_items:
            self.canvas.removeItem(it)
        self._template_items = []
        sx, sy = self._scale
        outline = self.template.outline_array() * np.array([sx, sy])
        closed = np.vstack([outline, outline[0]])
        curve = pg.PlotCurveItem(closed[:, 0], closed[:, 1], pen=pg.mkPen("#888", width=2))
        self.canvas.addItem(curve)
        self._template_items.append(curve)
        names = self.template.anchor_point_names()
        spots = [
            {
                "pos": (self.template.control_points[n][0] * sx, self.template.control_points[n][1] * sy),
                "data": n,
                "size": 14,
                "brush": pg.mkBrush(255, 200, 0),
                "pen": pg.mkPen("w"),
            }
            for n in names
        ]
        self._point_scatter = pg.ScatterPlotItem(spots=spots)
        self._point_scatter.sigClicked.connect(self._on_point_clicked)
        self.canvas.addItem(self._point_scatter)
        self._template_items.append(self._point_scatter)
        for n in names:
            x, y = self.template.control_points[n]
            t = pg.TextItem(n, color="w", anchor=(0, 1))
            t.setPos(x * sx, y * sy)
            self.canvas.addItem(t)
            self._template_items.append(t)

    def _on_point_clicked(self, _scatter, points):
        if not len(points):
            return
        self._selected_point = points[0].data()
        self.sel_label.setText(f"Selected point: {self._selected_point}")

    def _bind(self):
        if self._selected_point is None:
            return
        keypoint = self.kp_combo.currentText()
        self._bindings = [b for b in self._bindings if b["point"] != self._selected_point]
        self._bindings.append({"point": self._selected_point, "keypoint": keypoint})
        self._refresh_bindings()

    def _unbind(self):
        row = self.bind_list.currentRow()
        if 0 <= row < len(self._bindings):
            self._bindings.pop(row)
            self._refresh_bindings()

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._color), self, "Shape colour")
        if color.isValid():
            self._color = _hex(color)
            self.color_btn.setIcon(_color_swatch_icon(self._color))

    def _refresh_bindings(self):
        self.bind_list.clear()
        for b in self._bindings:
            self.bind_list.addItem(f"{b['point']} → {b['keypoint']}")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(len(self._bindings) >= 2)

    def get_shape(self) -> dict:
        return {
            "type": self.shape_type,
            "anchors": list(self._bindings),
            "color": self._color,
            "scale": list(self._scale),
        }


class SkeletonEditorDialog(QDialog):
    """Modal skeleton editor over real pose data.

    Parameters
    ----------
    keypoints
        Keypoint names (order indexes ``positions``).
    positions
        Array ``(n_frames, n_keypoints, 2)`` of image-space ``(x, y)``.
    existing_config
        Optional skeleton config dict to load for editing.
    """

    def __init__(self, keypoints, positions, existing_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create / edit skeleton")
        self.setMinimumSize(900, 600)
        self.resize(1400, 950)

        self.canvas = _SkeletonCanvas(keypoints, positions)
        self.canvas.on_selection_changed = self._sync_assign_enabled

        layout = QVBoxLayout(self)
        body = QHBoxLayout()
        layout.addLayout(body, stretch=1)

        # Left: canvas + frame slider.
        left = QVBoxLayout()
        left.addWidget(self.canvas, stretch=1)
        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, self.canvas.n_frames - 1))
        self.slider.valueChanged.connect(self._on_frame)
        self.frame_label = QLabel(f"Frame 0 / {self.canvas.n_frames - 1}")
        slider_row.addWidget(QLabel("Frame:"))
        slider_row.addWidget(self.slider, stretch=1)
        slider_row.addWidget(self.frame_label)
        left.addLayout(slider_row)
        hint = QLabel(
            "Scroll to zoom • right/middle-drag to pan • double right-click to fit • "
            "drag between two keypoints to connect • click or rubber-band drag "
            "to select edges (Shift to add) • assign a color category • "
            "add anchored shapes that follow the pose (2 anchors keep angles, 3+ deform)"
        )
        hint.setWordWrap(True)
        left.addWidget(hint)
        body.addLayout(left, stretch=5)

        # Right: color categories.
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Color categories</b>"))
        self.cat_list = QListWidget()
        self.cat_list.currentRowChanged.connect(self._sync_assign_enabled)
        right.addWidget(self.cat_list, stretch=1)
        cat_btns = QHBoxLayout()
        add_btn = QPushButton("Add")
        recolor_btn = QPushButton("Recolor")
        del_cat_btn = QPushButton("Remove")
        add_btn.clicked.connect(self._add_category)
        recolor_btn.clicked.connect(self._recolor_category)
        del_cat_btn.clicked.connect(self._remove_category)
        cat_btns.addWidget(add_btn)
        cat_btns.addWidget(recolor_btn)
        cat_btns.addWidget(del_cat_btn)
        right.addLayout(cat_btns)
        self.assign_btn = QPushButton("Assign to selected edges")
        self.assign_btn.clicked.connect(self._assign_to_selected)
        right.addWidget(self.assign_btn)
        self.set_color_btn = QPushButton("Set colour of selected edges…")
        self.set_color_btn.setToolTip(
            "Give the selected edges an explicit colour (kept as-is; the pose "
            "panel's base colour only recolours edges left uncoloured)."
        )
        self.set_color_btn.clicked.connect(self._set_edge_color)
        right.addWidget(self.set_color_btn)
        self.del_edges_btn = QPushButton("Delete selected edges")
        self.del_edges_btn.clicked.connect(self.canvas.remove_selected)
        right.addWidget(self.del_edges_btn)

        # Shapes section
        right.addWidget(QLabel("<b>Shapes</b>"))
        shape_add_row = QHBoxLayout()
        self.shape_type_combo = QComboBox()
        self.shape_type_combo.addItems(sorted(SHAPE_TEMPLATES.keys()))
        shape_add_row.addWidget(self.shape_type_combo)
        add_shape_btn = QPushButton("Add…")
        add_shape_btn.clicked.connect(self._add_shape)
        shape_add_row.addWidget(add_shape_btn)
        right.addLayout(shape_add_row)
        self.shape_list = QListWidget()
        self.shape_list.itemDoubleClicked.connect(lambda *_: self._edit_shape())
        right.addWidget(self.shape_list, stretch=1)
        shape_btns = QHBoxLayout()
        edit_shape_btn = QPushButton("Edit")
        del_shape_btn = QPushButton("Remove")
        edit_shape_btn.clicked.connect(self._edit_shape)
        del_shape_btn.clicked.connect(self._remove_shape)
        shape_btns.addWidget(edit_shape_btn)
        shape_btns.addWidget(del_shape_btn)
        right.addLayout(shape_btns)
        body.addLayout(right, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._categories: list[tuple[str, str]] = []  # (name, hex)
        self._shapes: list[dict] = self.canvas.shapes  # shared reference
        if existing_config:
            self._load_config(existing_config)
        self._sync_assign_enabled()

    # ── frame ──

    def _on_frame(self, value: int):
        self.canvas.set_frame(value)
        self.frame_label.setText(f"Frame {value} / {self.canvas.n_frames - 1}")

    # ── categories ──

    def _append_category(self, name: str, color_hex: str):
        self._categories.append((name, color_hex))
        item = QListWidgetItem(name)
        item.setIcon(self._swatch(color_hex))
        self.cat_list.addItem(item)

    def _swatch(self, color_hex: str) -> QIcon:
        pix = QPixmap(16, 16)
        pix.fill(QColor(color_hex))
        return QIcon(pix)

    def _add_category(self):
        name, ok = QInputDialog.getText(self, "New category", "Name:")
        if not ok or not name:
            return
        color = QColorDialog.getColor(QColor("#FFFFFF"), self, "Pick color")
        if not color.isValid():
            return
        self._append_category(name, _hex(color))
        self.cat_list.setCurrentRow(self.cat_list.count() - 1)

    def _recolor_category(self):
        row = self.cat_list.currentRow()
        if row < 0:
            return
        name, current = self._categories[row]
        color = QColorDialog.getColor(QColor(current), self, "Pick color")
        if not color.isValid():
            return
        self._categories[row] = (name, _hex(color))
        self.cat_list.item(row).setIcon(self._swatch(_hex(color)))

    def _remove_category(self):
        row = self.cat_list.currentRow()
        if row < 0:
            return
        self._categories.pop(row)
        self.cat_list.takeItem(row)

    def _assign_to_selected(self):
        row = self.cat_list.currentRow()
        if row < 0 or not self.canvas.selected:
            return
        name, color = self._categories[row]
        self.canvas.apply_category(color, name)

    def _set_edge_color(self):
        if not self.canvas.selected:
            return
        color = QColorDialog.getColor(QColor(_DEFAULT_COLOR), self, "Edge colour")
        if not color.isValid():
            return
        hex_color = _hex(color)
        # Tag the segment with the hex so _resolve_skeleton_colors keeps this
        # colour instead of overriding it with the pose panel's base colour.
        self.canvas.apply_category(hex_color, hex_color)

    def _sync_assign_enabled(self, *_):
        has_selection = bool(self.canvas.selected)
        self.assign_btn.setEnabled(self.cat_list.currentRow() >= 0 and has_selection)
        self.set_color_btn.setEnabled(has_selection)
        self.del_edges_btn.setEnabled(has_selection)

    # ── shapes ──

    def _add_shape(self):
        shape_type = self.shape_type_combo.currentText()
        dialog = ShapeAnchorDialog(shape_type, self.canvas.keypoints, parent=self)
        if dialog.exec_():
            self._shapes.append(dialog.get_shape())
            self._refresh_shapes()

    def _edit_shape(self):
        row = self.shape_list.currentRow()
        if not (0 <= row < len(self._shapes)):
            return
        shape = self._shapes[row]
        dialog = ShapeAnchorDialog(shape["type"], self.canvas.keypoints, parent=self, existing=shape)
        if dialog.exec_():
            self._shapes[row] = dialog.get_shape()
            self._refresh_shapes()

    def _remove_shape(self):
        row = self.shape_list.currentRow()
        if 0 <= row < len(self._shapes):
            self._shapes.pop(row)
            self._refresh_shapes()

    def _refresh_shapes(self):
        self.shape_list.clear()
        for shape in self._shapes:
            item = QListWidgetItem(f"{shape['type']} ({len(shape.get('anchors', []))} anchors)")
            item.setIcon(self._swatch(shape.get("color", _SHAPE_COLOR)))
            self.shape_list.addItem(item)
        self.canvas.redraw()

    # ── config I/O ──

    def _load_config(self, config: dict):
        names = self.canvas.keypoints
        existing_names = {n for n, _ in self._categories}
        for conn in config.get("connections", []):
            start, end = conn.get("start"), conn.get("end")
            if start not in names or end not in names:
                continue
            a, b = names.index(start), names.index(end)
            self.canvas.edges.append((a, b))
            self.canvas.edge_colors.append(conn.get("color", _DEFAULT_COLOR))
            segment = conn.get("segment", "")
            self.canvas.edge_segments.append(segment)
            if segment and segment not in existing_names:
                self._append_category(segment, conn.get("color", _DEFAULT_COLOR))
                existing_names.add(segment)
        for shape in config.get("shapes", []):
            self._shapes.append(dict(shape))
        self._refresh_shapes()
        self.canvas.redraw()

    def get_config(self) -> dict:
        """Return a skeleton config dict from the current edges/colors/shapes."""
        # Plain str: keypoint names often arrive as numpy strings, which yaml.safe_dump refuses.
        names = [str(k) for k in self.canvas.keypoints]
        connections = []
        for (a, b), color, segment in zip(self.canvas.edges, self.canvas.edge_colors, self.canvas.edge_segments):
            connections.append(
                {
                    "start": names[a],
                    "end": names[b],
                    "color": str(color),
                    "width": 2.0,
                    "segment": str(segment),
                }
            )
        return {
            "keypoints": names,
            "connections": connections,
            "shapes": list(self._shapes),
        }
