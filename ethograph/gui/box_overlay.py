"""Masks and prompt points on a pygfx camera view, and the click mode behind them.

The overlay is one translucent RGBA :class:`pygfx.Image` per view, composited
from every object's mask on the current frame in the object's OCTRON colour,
plus a points layer for the clicks (green = include, red = exclude), which is
OCTRON's own look. The mode presents the duck interface ``CameraView`` already
dispatches to (``locked``, ``handle_click``, ``handle_right_click``,
``handle_move``, ``handle_release``, ``detach``, ``set_frame``); the dialog
supplies the callbacks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pygfx as gfx

_Z_MASK = 0.3
_Z_POINTS = 4.0
_OFFSCREEN = -1e6
_POSITIVE = (0.24, 0.86, 0.52, 1.0)
_NEGATIVE = (1.0, 0.32, 0.32, 1.0)
_POINT_SIZE = 9.0
#: Nearest-neighbour downscale step so the overlay texture never exceeds this side.
MAX_OVERLAY_SIDE = 1024


@dataclass
class MaskLayer:
    mask: np.ndarray  # (H, W) uint8 in source pixels
    color: tuple[float, float, float, float]


def composite(layers: list[MaskLayer], height: int, width: int, opacity: float = 0.4) -> np.ndarray:
    """RGBA float32 ``(h, w, 4)`` at a bounded size; later layers paint over earlier ones."""
    step = max(1, int(np.ceil(max(height, width) / MAX_OVERLAY_SIDE)))
    h, w = (height + step - 1) // step, (width + step - 1) // step
    out = np.zeros((h, w, 4), dtype=np.float32)
    for layer in layers:
        m = layer.mask[::step, ::step] > 0
        if m.shape != (h, w):
            mm = np.zeros((h, w), dtype=bool)
            mm[: m.shape[0], : m.shape[1]] = m[:h, :w]
            m = mm
        r, g, b, _ = layer.color
        out[m] = (r, g, b, opacity)
    return out


class MaskOverlay:
    """One view's mask image and prompt points, in the view's image space."""

    def __init__(self, scene: gfx.Scene, view_size: tuple[int, int], source_size: tuple[int, int]):
        self._scene = scene
        self._view_w, self._view_h = view_size
        self._src_w, self._src_h = source_size
        self._image: gfx.Image | None = None
        self._points = gfx.Points(
            gfx.Geometry(positions=np.full((1, 3), _OFFSCREEN, dtype=np.float32), colors=np.ones((1, 4), np.float32)),
            gfx.PointsMarkerMaterial(
                size=_POINT_SIZE,
                size_space="screen",
                marker="circle",
                color_mode="vertex",
                edge_width=1.5,
                edge_color="#fff",
            ),
        )
        self._points.local.z = _Z_POINTS
        scene.add(self._points)

    @property
    def scale(self) -> float:
        """Source pixels per view image pixel (the view may show a proxy)."""
        return self._src_w / max(1, self._view_w)

    def to_source(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    def set_masks(self, layers: list[MaskLayer], opacity: float = 0.4) -> None:
        if self._image is not None:
            self._scene.remove(self._image)
            self._image = None
        if not layers:
            return
        rgba = composite(layers, self._src_h, self._src_w, opacity)
        texture = gfx.Texture(rgba[::-1].copy(), dim=2)  # the video texture is y-flipped too
        image = gfx.Image(gfx.Geometry(grid=texture), gfx.ImageBasicMaterial(clim=(0, 1)))
        sx = self._view_w / rgba.shape[1]
        sy = self._view_h / rgba.shape[0]
        image.local.scale = (sx, sy, 1.0)
        image.local.z = _Z_MASK
        self._scene.add(image)
        self._image = image

    def set_points(self, points: list[tuple[float, float, bool]]) -> None:
        """Clicks in source pixels: ``(x, y, positive)``."""
        n = max(1, len(points))
        positions = np.full((n, 3), _OFFSCREEN, dtype=np.float32)
        colors = np.ones((n, 4), dtype=np.float32)
        for i, (x, y, positive) in enumerate(points):
            positions[i] = (x / self.scale, self._view_h - y / self.scale, 0.0)
            colors[i] = _POSITIVE if positive else _NEGATIVE
        self._scene.remove(self._points)
        self._points = gfx.Points(
            gfx.Geometry(positions=positions, colors=colors),
            gfx.PointsMarkerMaterial(
                size=_POINT_SIZE,
                size_space="screen",
                marker="circle",
                color_mode="vertex",
                edge_width=1.5,
                edge_color="#fff",
            ),
        )
        self._points.local.z = _Z_POINTS
        self._scene.add(self._points)

    def clear(self) -> None:
        if self._image is not None:
            self._scene.remove(self._image)
            self._image = None
        self._scene.remove(self._points)


class BoxLabelMode:
    """Pointer handling for one camera view: left = include, right = exclude.

    The view calls ``handle_click`` for an unmodified left press and
    ``handle_right_click`` for a right press; both are forwarded to the
    dialog's ``on_point(camera, x_src, y_src, positive)``. Everything else is
    inert, so the pan/zoom controls keep working.
    """

    def __init__(
        self,
        view,
        camera: str,
        overlay: MaskOverlay,
        on_point: Callable[[str, float, float, bool], None],
        on_hover: Callable[[str], None] | None = None,
    ):
        self._view = view
        self.camera = camera
        self.overlay = overlay
        self._on_point = on_point
        self._on_hover = on_hover
        self.locked = False
        view.set_label_mode(self)

    def handle_click(self, x: float, y: float) -> None:
        sx, sy = self.overlay.to_source(x, y)
        self._on_point(self.camera, sx, sy, True)

    def handle_right_click(self, x: float, y: float) -> None:
        sx, sy = self.overlay.to_source(x, y)
        self._on_point(self.camera, sx, sy, False)

    def handle_move(self, x: float, y: float) -> None:
        if self._on_hover is not None:
            self._on_hover(self.camera)

    def handle_release(self, x: float, y: float) -> None:
        return None

    def set_frame(self, frame: int) -> None:
        return None

    def detach(self) -> None:
        self.overlay.clear()
        try:
            self._view.set_label_mode(None)
            self._view.request_draw()
        except (RuntimeError, AttributeError):
            pass
