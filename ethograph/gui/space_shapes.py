"""Parametric shapes for a space geometry file: a box is a centre and three sizes, not eight vertices.

A geometry YAML (see :mod:`ethograph.gui.plots_space`) holds two lists. Its
``references`` are raw wireframes — vertices plus edges by index — and its
``shapes`` are parametric solids edited in Settings ▸ Edit space geometry::

    shapes:
      - name: perch
        type: cylinder
        center: [0.0, 4.9, 1.2]
        radius: 0.02
        height: 1.5
        axis: x
        color: brown

Every shape becomes a wireframe (:func:`shape_wireframe`) and is drawn exactly
like a reference, so the 2D space plot shows its x/y projection and the 3D one
the solid itself. Units are the data's own: whatever the ``position`` variable
is in, the shapes are too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Every shape type → its size parameters, in the order the editor shows them.
#: Flat shapes (square, rectangle, circle) lie in the x/y plane at the centre's z.
SHAPE_PARAMS: dict[str, tuple[str, ...]] = {
    "square": ("side", "rotation"),
    "rectangle": ("width", "length", "rotation"),
    "box": ("width", "length", "height", "rotation"),
    "circle": ("radius",),
    "cylinder": ("radius", "height"),
    "sphere": ("radius",),
}
SHAPE_TYPES: tuple[str, ...] = tuple(SHAPE_PARAMS)

#: A new shape's size: unit-sized, unrotated.
PARAM_DEFAULTS: dict[str, float] = {
    "side": 1.0,
    "width": 1.0,
    "length": 1.0,
    "height": 1.0,
    "radius": 0.5,
    "rotation": 0.0,
}

#: Parameters in degrees rather than data units.
ANGLE_PARAMS = frozenset({"rotation"})

#: The axis a cylinder's height runs along.
AXES: tuple[str, ...] = ("z", "x", "y")

#: Segments per circle: smooth enough at any zoom, few enough to redraw per keystroke.
_SEGMENTS = 48
#: Lines along a cylinder's side, and meridians on a sphere.
_SIDE_LINES = 8


@dataclass
class Shape:
    """One parametric shape. ``params`` holds exactly :data:`SHAPE_PARAMS` of its type."""

    name: str
    type: str
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    params: dict[str, float] = field(default_factory=dict)
    axis: str = "z"
    color: str = "black"

    def __post_init__(self) -> None:
        if self.type not in SHAPE_PARAMS:
            raise ValueError(f"Unknown shape type {self.type!r}; expected one of {', '.join(SHAPE_TYPES)}")
        if self.axis not in AXES:
            raise ValueError(f"Unknown cylinder axis {self.axis!r}; expected one of {', '.join(AXES)}")
        self.params = {p: float(self.params.get(p, PARAM_DEFAULTS[p])) for p in SHAPE_PARAMS[self.type]}

    def retyped(self, shape_type: str) -> Shape:
        """The same shape as another type, keeping every size the two types share."""
        return Shape(self.name, shape_type, self.center, dict(self.params), self.axis, self.color)

    def to_dict(self) -> dict:
        """The YAML entry, as :func:`shape_from_dict` reads it back."""
        entry: dict = {"name": self.name, "type": self.type, "center": [float(c) for c in self.center]}
        entry.update(self.params)
        if self.type == "cylinder":
            entry["axis"] = self.axis
        entry["color"] = self.color
        return entry


def shape_from_dict(entry: dict) -> Shape:
    """A shape from its YAML entry; a missing size falls back to :data:`PARAM_DEFAULTS`."""
    center = [float(c) for c in entry.get("center", (0.0, 0.0, 0.0))]
    if len(center) not in (2, 3):
        raise ValueError(f"A shape's center is [x, y] or [x, y, z], got {center}")
    if len(center) == 2:
        center.append(0.0)
    shape_type = str(entry["type"])
    params = {p: entry[p] for p in SHAPE_PARAMS.get(shape_type, ()) if p in entry}
    return Shape(
        name=str(entry.get("name", shape_type)),
        type=shape_type,
        center=(center[0], center[1], center[2]),
        params=params,
        axis=str(entry.get("axis", "z")),
        color=str(entry.get("color", "black")),
    )


def parse_shapes(cfg: dict) -> list[Shape]:
    """A geometry config's ``shapes`` list."""
    return [shape_from_dict(entry) for entry in cfg.get("shapes") or []]


# ---------------------------------------------------------------------------
# Wireframes
# ---------------------------------------------------------------------------


def _ring(radius: float, n: int = _SEGMENTS) -> np.ndarray:
    """``(n, 3)`` points of a circle in the x/y plane, about the origin."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)])


def _loop(start: int, n: int) -> list[tuple[int, int]]:
    """Edges closing ``n`` consecutive vertices from *start* into a loop."""
    return [(start + i, start + (i + 1) % n) for i in range(n)]


def _rect_corners(width: float, length: float) -> np.ndarray:
    """The four corners of a ``width`` (x) × ``length`` (y) rectangle about the origin."""
    w, h = width / 2.0, length / 2.0
    return np.array([[-w, -h, 0.0], [w, -h, 0.0], [w, h, 0.0], [-w, h, 0.0]])


def _rotate_z(vertices: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return vertices @ rotation.T


def _z_to_axis(vertices: np.ndarray, axis: str) -> np.ndarray:
    """Turn geometry built along z so that it runs along *axis*."""
    if axis == "x":
        return vertices[:, [2, 0, 1]]
    if axis == "y":
        return vertices[:, [1, 2, 0]]
    return vertices


def shape_wireframe(shape: Shape) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """``(vertices (N, 3), edges)`` outlining *shape*, in data coordinates."""
    p = shape.params
    edges: list[tuple[int, int]]
    if shape.type == "square":
        vertices = _rect_corners(p["side"], p["side"])
        edges = _loop(0, 4)
    elif shape.type == "rectangle":
        vertices = _rect_corners(p["width"], p["length"])
        edges = _loop(0, 4)
    elif shape.type == "box":
        base = _rect_corners(p["width"], p["length"])
        half = p["height"] / 2.0
        vertices = np.vstack([base + [0.0, 0.0, -half], base + [0.0, 0.0, half]])
        edges = _loop(0, 4) + _loop(4, 4) + [(i, i + 4) for i in range(4)]
    elif shape.type == "circle":
        vertices = _ring(p["radius"])
        edges = _loop(0, _SEGMENTS)
    elif shape.type == "cylinder":
        ring = _ring(p["radius"])
        half = p["height"] / 2.0
        vertices = np.vstack([ring + [0.0, 0.0, -half], ring + [0.0, 0.0, half]])
        step = _SEGMENTS // _SIDE_LINES
        edges = _loop(0, _SEGMENTS) + _loop(_SEGMENTS, _SEGMENTS)
        edges += [(i, i + _SEGMENTS) for i in range(0, _SEGMENTS, step)]
        vertices = _z_to_axis(vertices, shape.axis)
    elif shape.type == "sphere":
        # Three latitude rings and a fan of meridians: reads as a ball from any angle.
        r = p["radius"]
        rings = [_ring(r * np.cos(lat)) + [0.0, 0.0, r * np.sin(lat)] for lat in np.deg2rad([-45.0, 0.0, 45.0])]
        meridian = _ring(r)[:, [0, 2, 1]]  # a great circle in the x/z plane
        meridians = [_rotate_z(meridian, 180.0 * k / _SIDE_LINES) for k in range(_SIDE_LINES // 2)]
        vertices = np.vstack(rings + meridians)
        edges = [e for k in range(len(rings) + len(meridians)) for e in _loop(k * _SEGMENTS, _SEGMENTS)]
    else:
        raise ValueError(f"Unknown shape type {shape.type!r}")

    if "rotation" in p:
        vertices = _rotate_z(vertices, p["rotation"])
    return vertices + np.asarray(shape.center, dtype=np.float64), edges


# ---------------------------------------------------------------------------
# The axis cross drawn through the shape being edited
# ---------------------------------------------------------------------------


def nice_step(span: float, target: int = 6) -> float:
    """A 1/2/5 × 10^k step putting about *target* ticks across *span*."""
    if span <= 0:
        return 1.0
    raw = span / target
    magnitude = 10.0 ** np.floor(np.log10(raw))
    return float(next(m * magnitude for m in (1.0, 2.0, 5.0, 10.0) if m * magnitude >= raw))


def axis_ticks(center: float, half_length: float) -> list[float]:
    """Round coordinates within ``center ± half_length``, where the axis cross writes its numbers."""
    step = nice_step(2.0 * half_length)
    first = np.ceil((center - half_length) / step)
    last = np.floor((center + half_length) / step)
    # + 0.0 turns a -0.0 into 0.0, so no tick reads "-0".
    return [float(round(k * step, 10)) + 0.0 for k in np.arange(first, last + 1)]


def axes_half_length(shape: Shape) -> float:
    """How far the axis cross reaches from the centre: past the shape, so its ends are visible."""
    vertices, _ = shape_wireframe(shape)
    extent = float((vertices.max(axis=0) - vertices.min(axis=0)).max())
    return max(extent, 1e-6)
