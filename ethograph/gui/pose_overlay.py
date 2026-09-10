"""Pygfx pose overlay: streaming per-frame points, skeleton, bboxes, shapes.

Replaces the napari Points/Shapes/Vectors layers. All graphics live in the
video view's pygfx scene and are updated per frame by writing into
fixed-capacity GPU buffers (same streaming pattern as pynaviz's PlotPoints)
instead of precomputing per-frame napari arrays.

Coordinates: pose data is in image space (y down). The video texture is
rendered y-flipped (pynaviz convention, y up), so all overlay positions are
mapped with ``y_world = img_height - y_img``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pygfx as gfx

from ethograph.skeleton.shapes import SHAPE_TEMPLATES, shape_outline_for_frame

# Sentinel position far outside any video frame, used to hide masked points
# (pygfx does not skip NaN positions reliably across materials).
_OFFSCREEN = -1.0e6

_Z_SHAPES = 0.5
_Z_SKELETON = 1.0
_Z_BBOX = 1.5
_Z_POINTS = 2.0
_Z_TEXT = 3.0


@dataclass
class OverlayStyle:
    """Display styling for a pose overlay (napari-style semantics)."""

    color_prop: str = "keypoint"  # which property drives point colors
    text_prop: str = "keypoint"  # which property drives labels
    color_map: dict = field(default_factory=dict)  # value -> rgba tuple
    point_size: float = 10.0
    points_visible: bool = True
    text_size: float = 12.0
    text_visible: bool = False
    edge_width: float = 2.0


def _tracks_from_properties(properties: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Factorize rows into track ids by (individual, keypoint).

    Returns (track_idx per row, track table with one row per track).
    """
    n = len(properties)
    ind = properties["individual"].astype(str).to_numpy() if "individual" in properties.columns else ["ind_0"] * n
    kp = properties["keypoint"].astype(str).to_numpy() if "keypoint" in properties.columns else [""] * n
    # Two columns, never one joined string: a pyarrow-backed string column
    # drops a NUL separator, and a name may contain any other character.
    codes, uniques = pd.factorize(pd.MultiIndex.from_arrays([ind, kp]))
    track_table = pd.DataFrame(
        {"individual": list(uniques.get_level_values(0)), "keypoint": list(uniques.get_level_values(1))}
    )
    return codes, track_table


class PoseOverlayData:
    """Dense per-frame pose cube built once from a PoseRenderData.

    positions : (n_frames, n_tracks, 2) float32 in display coords (x, y_img)
    shown     : (n_frames, n_tracks) bool
    track_table : DataFrame with columns individual, keypoint (one row per track)
    """

    def __init__(self, pr):
        coords = pr.data[:, -3:]  # (frame, y, x)
        frames = coords[:, 0].astype(int)
        ys = coords[:, 1]
        xs = coords[:, 2]

        track_idx, self.track_table = _tracks_from_properties(pr.properties)
        self.n_tracks = len(self.track_table)
        self.n_frames = int(frames.max()) + 1 if len(frames) else 0

        self.positions = np.full((self.n_frames, self.n_tracks, 2), np.nan, dtype=np.float32)
        self.shown = np.zeros((self.n_frames, self.n_tracks), dtype=bool)
        self.confidence = np.full((self.n_frames, self.n_tracks), np.nan, dtype=np.float32)

        valid = pr.data_not_nan & ~np.isnan(xs) & ~np.isnan(ys)
        f, t = frames[valid], track_idx[valid]
        self.positions[f, t, 0] = xs[valid]
        self.positions[f, t, 1] = ys[valid]
        self.shown[f, t] = True
        if "confidence" in pr.properties.columns:
            self.confidence[f, t] = pr.properties["confidence"].to_numpy()[valid]

        # Bboxes: (N, 4, 3+) corner rows of ([track_id,] frame, y, x)
        self.bbox_corners = None  # (n_frames, n_boxes, 4, 2)
        self.bbox_shown = None
        self.bbox_track = None
        if pr.bbox_data is not None and len(pr.bbox_data):
            bb = pr.bbox_data[:, :, -3:]  # (N, 4, (frame, y, x))
            bframes = bb[:, 0, 0].astype(int)
            bvalid = pr.data_not_nan & ~np.any(np.isnan(bb[:, :, 1:]), axis=(1, 2))
            btracks = track_idx
            n_boxes = self.n_tracks
            self.bbox_corners = np.full((self.n_frames, n_boxes, 4, 2), np.nan, dtype=np.float32)
            self.bbox_shown = np.zeros((self.n_frames, n_boxes), dtype=bool)
            f, t = bframes[bvalid], btracks[bvalid]
            self.bbox_corners[f, t, :, 0] = bb[bvalid][:, :, 2]  # x
            self.bbox_corners[f, t, :, 1] = bb[bvalid][:, :, 1]  # y
            self.bbox_shown[f, t] = True

    def keypoint_track_index(self, individual: str | None = None) -> dict[str, int]:
        """Map keypoint name -> track index (optionally for one individual)."""
        table = self.track_table
        if individual is not None:
            table = table[table["individual"] == individual]
        return {row.keypoint: idx for idx, row in table.iterrows()}


class PoseOverlay:
    """Manages pygfx objects for one camera view's pose display."""

    def __init__(self, scene: gfx.Scene):
        self._scene = scene
        self._img_height: float = 0.0
        self._data: PoseOverlayData | None = None
        self._style = OverlayStyle()
        self._skeleton_config: dict | None = None
        self._frame: int = 0

        self._points: gfx.Points | None = None
        self._texts: list[gfx.Text] = []
        self._skeleton: gfx.Line | None = None
        self._edges: list[tuple[int, int, tuple]] = []  # (track_a, track_b, rgba)
        self._bbox_lines: gfx.Line | None = None
        self._shape_lines: list[gfx.Line] = []
        self._shape_defs: list[dict] = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def set_data(
        self,
        data: PoseOverlayData | None,
        style: OverlayStyle,
        img_height: float,
        skeleton_config: dict | None = None,
    ) -> None:
        """(Re)build all graphics for a new pose dataset."""
        self.clear()
        self._data = data
        self._style = style
        self._img_height = img_height
        self._skeleton_config = skeleton_config
        if data is None or data.n_tracks == 0:
            return

        n = data.n_tracks
        positions = np.full((n, 3), _OFFSCREEN, dtype=np.float32)
        colors = np.ones((n, 4), dtype=np.float32)
        for t in range(n):
            colors[t] = self._track_color(t)

        self._points = gfx.Points(
            gfx.Geometry(positions=positions, colors=colors),
            gfx.PointsMaterial(size=style.point_size, color_mode="vertex"),
        )
        self._points.local.z = _Z_POINTS
        self._points.visible = style.points_visible
        self._scene.add(self._points)

        # Text labels: one per track
        for t in range(n):
            label = str(data.track_table.iloc[t][style.text_prop] or "")
            text = gfx.Text(
                text=label,
                font_size=style.text_size,
                screen_space=True,
                anchor="bottom-left",
                material=gfx.TextMaterial(color=gfx.Color(*self._track_color(t)[:3])),
            )
            text.local.z = _Z_TEXT
            text.visible = False
            self._scene.add(text)
            self._texts.append(text)

        self._build_skeleton()
        self._build_bboxes()
        self._build_shapes()
        self.set_frame(self._frame)

    def _track_color(self, track: int) -> tuple:
        table = self._data.track_table
        value = table.iloc[track][self._style.color_prop]
        rgba = self._style.color_map.get(value)
        if rgba is None:
            return (1.0, 0.2, 0.2, 1.0)
        rgba = tuple(float(c) for c in rgba)
        return rgba if len(rgba) == 4 else rgba + (1.0,)

    def _build_skeleton(self) -> None:
        config = self._skeleton_config
        self._edges = []
        if not config:
            return
        connections = config.get("connections") or []
        if not connections:
            return
        individuals = self._data.track_table["individual"].unique().tolist()
        for ind in individuals:
            kp_index = self._data.keypoint_track_index(ind)
            for conn in connections:
                a = kp_index.get(conn.get("start"))
                b = kp_index.get(conn.get("end"))
                if a is None or b is None:
                    continue
                rgba = _parse_color(conn.get("color", "#00FF00"))
                self._edges.append((a, b, rgba))
        if not self._edges:
            return
        n_verts = 2 * len(self._edges)
        positions = np.full((n_verts, 3), _OFFSCREEN, dtype=np.float32)
        colors = np.ones((n_verts, 4), dtype=np.float32)
        for i, (_, _, rgba) in enumerate(self._edges):
            colors[2 * i] = rgba
            colors[2 * i + 1] = rgba
        self._skeleton = gfx.Line(
            gfx.Geometry(positions=positions, colors=colors),
            gfx.LineSegmentMaterial(thickness=self._style.edge_width, color_mode="vertex"),
        )
        self._skeleton.local.z = _Z_SKELETON
        self._scene.add(self._skeleton)

    def _build_bboxes(self) -> None:
        if self._data.bbox_corners is None:
            return
        n_boxes = self._data.bbox_corners.shape[1]
        # 4 segments per box -> 8 vertices
        positions = np.full((8 * n_boxes, 3), _OFFSCREEN, dtype=np.float32)
        colors = np.ones((8 * n_boxes, 4), dtype=np.float32)
        for b in range(n_boxes):
            colors[8 * b : 8 * b + 8] = self._track_color(b)
        self._bbox_lines = gfx.Line(
            gfx.Geometry(positions=positions, colors=colors),
            gfx.LineSegmentMaterial(thickness=self._style.edge_width, color_mode="vertex"),
        )
        self._bbox_lines.local.z = _Z_BBOX
        self._scene.add(self._bbox_lines)

    def _build_shapes(self) -> None:
        """Anchored shapes (from the skeleton editor) rendered as polylines."""
        config = self._skeleton_config or {}
        shapes = config.get("shapes") or []
        self._shape_defs = []
        kp_index = self._data.keypoint_track_index()
        for shape in shapes:
            template = SHAPE_TEMPLATES.get(shape.get("type", ""))
            anchors = shape.get("anchors", [])
            if template is None or len(anchors) < 2:
                continue
            if any(a["keypoint"] not in kp_index for a in anchors):
                continue
            n_outline = len(template.outline)
            rgba = _parse_color(shape.get("color", "#FFCC00"))
            positions = np.full((n_outline + 1, 3), _OFFSCREEN, dtype=np.float32)
            line = gfx.Line(
                gfx.Geometry(positions=positions),
                gfx.LineMaterial(thickness=self._style.edge_width, color=gfx.Color(*rgba)),
            )
            line.local.z = _Z_SHAPES
            self._scene.add(line)
            self._shape_lines.append(line)
            self._shape_defs.append(
                {
                    "template": template,
                    "anchor_names": [a["point"] for a in anchors],
                    "anchor_tracks": [kp_index[a["keypoint"]] for a in anchors],
                    "scale": tuple(shape.get("scale", (1.0, 1.0))),
                }
            )

    # ------------------------------------------------------------------
    # Per-frame streaming
    # ------------------------------------------------------------------

    def _to_world(self, xy: np.ndarray) -> np.ndarray:
        """Map (x, y_img) image coords to world coords (y flipped)."""
        out = xy.copy()
        out[..., 1] = self._img_height - out[..., 1]
        return out

    def set_frame(self, frame: int) -> None:
        self._frame = int(frame)
        data = self._data
        if data is None or data.n_tracks == 0:
            return
        f = min(max(self._frame, 0), data.n_frames - 1) if data.n_frames else 0
        in_range = 0 <= self._frame < data.n_frames

        pos = data.positions[f] if in_range else np.full((data.n_tracks, 2), np.nan)
        shown = data.shown[f] if in_range else np.zeros(data.n_tracks, dtype=bool)
        world = self._to_world(pos)

        if self._points is not None:
            buf = self._points.geometry.positions
            arr = buf.data
            arr[:, 2] = 0.0
            arr[:, :2] = np.where(shown[:, None], world, _OFFSCREEN)
            buf.update_full()

        for t, text in enumerate(self._texts):
            visible = bool(shown[t]) and self._style.text_visible and self._style.points_visible
            text.visible = visible
            if visible:
                text.local.position = (float(world[t, 0]) + 3, float(world[t, 1]) + 3, _Z_TEXT)

        if self._skeleton is not None and self._edges:
            arr = self._skeleton.geometry.positions.data
            arr[:, 2] = 0.0
            for i, (a, b, _) in enumerate(self._edges):
                if shown[a] and shown[b]:
                    arr[2 * i, :2] = world[a]
                    arr[2 * i + 1, :2] = world[b]
                else:
                    arr[2 * i, :2] = _OFFSCREEN
                    arr[2 * i + 1, :2] = _OFFSCREEN
            self._skeleton.geometry.positions.update_full()

        if self._bbox_lines is not None and data.bbox_corners is not None:
            arr = self._bbox_lines.geometry.positions.data
            arr[:, 2] = 0.0
            corners = data.bbox_corners[f] if in_range else None
            bshown = data.bbox_shown[f] if in_range else None
            n_boxes = data.bbox_corners.shape[1]
            for b in range(n_boxes):
                base = 8 * b
                if corners is not None and bshown[b]:
                    cw = self._to_world(corners[b])  # (4, 2)
                    for s in range(4):
                        arr[base + 2 * s, :2] = cw[s]
                        arr[base + 2 * s + 1, :2] = cw[(s + 1) % 4]
                else:
                    arr[base : base + 8, :2] = _OFFSCREEN
            self._bbox_lines.geometry.positions.update_full()

        for line, sdef in zip(self._shape_lines, self._shape_defs):
            arr = line.geometry.positions.data
            anchor_pos = pos[sdef["anchor_tracks"]]  # image coords (x, y)
            anchor_ok = shown[sdef["anchor_tracks"]].all() and not np.any(np.isnan(anchor_pos))
            outline = None
            if anchor_ok:
                outline = shape_outline_for_frame(sdef["template"], sdef["anchor_names"], anchor_pos, sdef["scale"])
            if outline is not None:
                closed = np.vstack([outline, outline[:1]])  # (K+1, 2) in (x, y)
                arr[:, :2] = self._to_world(closed.astype(np.float32))
                arr[:, 2] = 0.0
            else:
                arr[:, :2] = _OFFSCREEN
            line.geometry.positions.update_full()

    # ------------------------------------------------------------------
    # Style updates (no rebuild)
    # ------------------------------------------------------------------

    def set_point_size(self, size: float) -> None:
        self._style.point_size = size
        if self._points is not None:
            self._points.material.size = size

    def set_points_visible(self, visible: bool) -> None:
        self._style.points_visible = visible
        if self._points is not None:
            self._points.visible = visible
        self.set_frame(self._frame)

    def set_text_visible(self, visible: bool) -> None:
        self._style.text_visible = visible
        self.set_frame(self._frame)

    def set_text_size(self, size: float) -> None:
        self._style.text_size = size
        for text in self._texts:
            text.font_size = size

    def set_edge_width(self, width: float) -> None:
        self._style.edge_width = width
        if self._skeleton is not None:
            self._skeleton.material.thickness = width
        if self._bbox_lines is not None:
            self._bbox_lines.material.thickness = width
        for line in self._shape_lines:
            line.material.thickness = width

    def clear(self) -> None:
        for obj in [self._points, self._skeleton, self._bbox_lines, *self._texts, *self._shape_lines]:
            if obj is not None:
                self._scene.remove(obj)
        self._points = None
        self._texts = []
        self._skeleton = None
        self._edges = []
        self._bbox_lines = None
        self._shape_lines = []
        self._shape_defs = []
        self._data = None


def _parse_color(color) -> tuple:
    """Parse hex/name/tuple color into an RGBA float tuple."""
    if isinstance(color, (tuple, list, np.ndarray)):
        rgba = tuple(float(c) for c in color)
        return rgba if len(rgba) == 4 else rgba[:3] + (1.0,)
    c = gfx.Color(color)
    return (c.r, c.g, c.b, c.a)
