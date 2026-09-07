"""Pygfx-backed camera view widgets (napari replacement for video display).

``CameraView`` hosts one camera's video via pynaviz's :class:`PlotVideo`
(shared-memory worker process decoding, pygfx rendering) plus a
:class:`~ethograph.gui.pose_overlay.PoseOverlay` in the same scene.

Seeking model:
- Programmatic seeks come from :class:`~ethograph.gui.video_sync.VideoSync`
  through ``seek_video_frame`` / ``seek_to_time`` and are *suppressed* — they
  do not re-emit ``time_changed``.
- User interaction on the canvas (scroll = frame step, arrow keys) triggers
  pynaviz sync events on the plot's renderer; the view forwards those as
  ``time_changed(video_time)`` so VideoSync can propagate to plots and other
  cameras.
"""

from __future__ import annotations

import logging
import queue
import time
from types import MethodType
from typing import Optional

import numpy as np
import pygfx as gfx
from pynaviz.audiovideo import PlotVideo
from pynaviz.utils import RenderTriggerSource
from qtpy.QtCore import QEvent, Qt, QTimer, Signal
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from .app_constants import MEDIA_VIEW_MIN_HEIGHT, MEDIA_VIEW_MIN_WIDTH
from .pose_overlay import PoseOverlay

logger = logging.getLogger(__name__)


# Heartbeat age past which the render chain is considered dead (see
# install_animate_guard / nudge_stalled_render). animate() normally re-arms
# itself every frame, so one full second without a beat is a stall.
ANIMATE_STALL_S = 1.0

#: Minimum crop side length in image pixels — a thinner rectangle is a
#: misclick, not a crop.
MIN_CROP_SIZE_PX = 2

#: z of the crop-selection preview rectangle (above the pose overlay's text).
_Z_CROP_PREVIEW = 4.0


def snap_crop_rect(
    x0: float, y0: float, x1: float, y1: float, width: float, height: float
) -> tuple[int, int, int, int] | None:
    """Normalize a dragged rectangle to whole-pixel edges inside the image.

    Returns ``(x0, y0, x1, y1)`` ints with ``x0 < x1`` and ``y0 < y1`` in image
    coordinates (y down), each snapped to the closest pixel edge and clamped to
    the image, or ``None`` when the snapped rectangle is degenerate.
    """
    xa, xb = sorted((float(x0), float(x1)))
    ya, yb = sorted((float(y0), float(y1)))
    xa = int(np.clip(round(xa), 0, int(width)))
    xb = int(np.clip(round(xb), 0, int(width)))
    ya = int(np.clip(round(ya), 0, int(height)))
    yb = int(np.clip(round(yb), 0, int(height)))
    if xb - xa < MIN_CROP_SIZE_PX or yb - ya < MIN_CROP_SIZE_PX:
        return None
    return (xa, ya, xb, yb)


def square_corner(ax: float, ay: float, x: float, y: float) -> tuple[float, float]:
    """The cursor corner constrained so the box from ``(ax, ay)`` is square.

    The side is the larger of the two drag extents, so the box grows with
    whichever axis the mouse moved further along; the drag's direction on
    each axis is kept, so dragging up-left still draws up-left.
    """
    side = max(abs(x - ax), abs(y - ay))
    sx = 1.0 if x >= ax else -1.0
    sy = 1.0 if y >= ay else -1.0
    return ax + sx * side, ay + sy * side


def crop_clip_planes(rect: tuple[int, int, int, int], img_height: float) -> list[tuple[float, float, float, float]]:
    """pygfx world-space clipping planes hiding everything outside *rect*.

    The video texture is rendered y-flipped (world y up), so the image-space
    rows ``[y0, y1]`` become world y ``[img_height - y1, img_height - y0]``.

    Sign convention per pygfx's ``clipping_planes.wgsl`` (mode "ANY"): a
    fragment is DISCARDED where ``dot(world_pos, plane.xyz) < plane.w`` —
    i.e. kept where ``ax + by + cz >= d``. Note ``d`` is a threshold on the
    dot product, NOT the ``+d`` of the plane-equation convention the material
    docstring suggests; with that sign flipped the right/top planes exclude
    the entire image and the video disappears.
    """
    x0, y0, x1, y1 = (float(v) for v in rect)
    bottom, top = img_height - y1, img_height - y0
    return [
        (1.0, 0.0, 0.0, x0),  # keep x >= x0
        (-1.0, 0.0, 0.0, -x1),  # keep x <= x1
        (0.0, 1.0, 0.0, bottom),  # keep world y >= bottom
        (0.0, -1.0, 0.0, -top),  # keep world y <= top
    ]


def _present_disarmed(canvas) -> bool:
    """True when :func:`_disarm_present` has neutralised this canvas.

    Disarming installs the no-op as an *instance* attribute; a healthy canvas
    only has the class-level method, so the instance dict is the tell. A
    disarmed canvas must never be re-armed — its widget is dying, and driving
    draws at it would recreate exactly the freeze signature being guarded
    against.
    """
    widget = getattr(canvas, "_subwidget", canvas)
    return "_rc_request_paint" in getattr(widget, "__dict__", {})


def install_animate_guard(plot, clock=time.perf_counter) -> None:
    """Keep *plot*'s render chain alive across exceptions, and stamp a heartbeat.

    pynaviz's ``PlotVideo.animate`` continues only because its own last line
    re-arms it (``self.canvas.request_draw(self.animate)``), and its ``try``
    catches nothing but ``queue.Empty`` — any other raise (texture update,
    time-text, a present dropped by the Qt scheduler) ends the chain
    permanently: audio and the playhead keep moving while the image freezes.

    The wrapper is set as an *instance* attribute, so the original's own
    re-arm (``self.animate``) resolves back to the wrapper and the chain stays
    wrapped. Every call stamps ``plot._eto_last_animate`` for the watchdog.
    """
    original = plot.animate  # the bound class method, captured pre-override

    def _guarded_animate():
        plot._eto_last_animate = clock()
        try:
            original()
        except Exception:
            logger.warning("PlotVideo.animate raised; re-arming the render loop.", exc_info=True)
            if not _present_disarmed(plot.canvas):
                plot.canvas.request_draw(plot.animate)

    plot.animate = _guarded_animate
    plot._eto_last_animate = clock()


def nudge_stalled_render(plot, max_age_s: float = ANIMATE_STALL_S, clock=time.perf_counter) -> bool:
    """Re-arm a render chain whose heartbeat went stale; ``True`` if nudged.

    Covers the stalls the guard cannot: a paint silently dropped by the
    rendercanvas Qt scheduler never re-enters ``animate`` at all, so only an
    external re-arm can restart the chain.
    """
    last = getattr(plot, "_eto_last_animate", None)
    if last is None or clock() - last <= max_age_s:
        return False
    if _present_disarmed(plot.canvas):
        return False
    plot.canvas.request_draw(plot.animate)
    return True


def _disarm_present(canvas) -> None:
    """Stop a canvas whose widget is about to die from finishing its present.

    wgpu hands the rendered bitmap back asynchronously: ``_finish_present``
    runs a frame or more after the draw and ends in ``QWidget.update()`` on
    rendercanvas's inner ``QRenderWidget``. Closing the canvas deletes that
    widget (``WA_DeleteOnClose``), so a present still in flight lands on a
    deleted C++ object — rendercanvas swallows it, but logs a "Present finish
    error / wrapped C/C++ object of type QRenderWidget has been deleted"
    traceback on every video teardown. Neutralising the repaint request first
    makes the pending present a no-op instead.
    """
    widget = getattr(canvas, "_subwidget", canvas)
    widget._rc_request_paint = lambda: None


def _detach_canvas(layout, canvas) -> None:
    """Unparent *canvas* from *layout*, tolerating an already-deleted C++ object.

    A canvas whose Qt side died with the dock that held it (video panel closed
    and re-added) leaves a live Python wrapper behind, and every call on it
    raises ``RuntimeError: wrapped C/C++ object ... has been deleted``. That
    used to propagate out of ``VideoManager._cleanup_primary_video`` and abort
    the whole trial change; a canvas Qt already destroyed simply has nothing
    left to detach.
    """
    try:
        layout.removeWidget(canvas)
        _disarm_present(canvas)
        canvas.setParent(None)
    except RuntimeError:
        logger.debug("Canvas was already deleted; nothing to detach.", exc_info=True)


class _StaticImagePlot:
    """Minimal pygfx scene showing a single still image (no video worker)."""

    def __init__(self, img: np.ndarray, parent: QWidget):
        from rendercanvas.qt import RenderCanvas

        self.canvas = RenderCanvas(parent=parent)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.scene.add(gfx.Background.from_color("black"))

        data = np.asarray(img)
        if data.ndim == 2:
            data = np.repeat(data[:, :, None], 3, axis=2)
        data = data[::-1].astype("float32")
        if data.max() > 1.0:
            data = data / 255.0
        self.texture = gfx.Texture(data, dim=2)
        self.image = gfx.Image(
            gfx.Geometry(grid=self.texture),
            gfx.ImageBasicMaterial(clim=(0, 1)),
        )
        self.scene.add(self.image)
        self.camera = gfx.OrthographicCamera(maintain_aspect=True)
        self.camera.show_object(self.scene)
        self.controller = gfx.PanZoomController(self.camera, register_events=self.renderer)
        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

    def request_draw(self):
        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

    def close(self):
        _disarm_present(self.canvas)
        try:
            self.canvas.close()
        except Exception:
            pass


class CameraView(QWidget):
    """One camera: pygfx video canvas + pose overlay + Qt label overlays."""

    time_changed = Signal(float)  # video-native time, from user interaction
    clicked = Signal()  # any mouse press in this camera view (for active-panel)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        # 2px margin so the active-panel green edge is visible around the video.
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        self.setLayout(layout)

        self._plot: Optional[PlotVideo] = None
        self._static: Optional[_StaticImagePlot] = None
        self._overlay: Optional[PoseOverlay] = None
        #: Active keypoint labelling mode (ethograph.gui.pose_edit_mixin), if any.
        self._label_mode = None
        self._pan_control = None  # saved controls["mouse1"] while labelling
        #: Display crop ``(x0, y0, x1, y1)`` in image pixels (y down), or None.
        self._crop: tuple[int, int, int, int] | None = None
        #: Callback receiving the selected crop rect (rectangle tool armed).
        self._crop_select_cb = None
        self._crop_anchor: tuple[float, float] | None = None
        #: Constrain the drag to a square (``start_crop_selection(square=True)``).
        self._crop_square = False
        self._crop_preview: Optional[gfx.Line] = None
        #: Set for static-image views: source file + fps of the pose shown on top.
        self.static_image_path: Optional[str] = None
        self.static_pose_fps: float = 0.0
        self._fps: float = 0.0
        #: Path currently decoded by ``_plot`` — reloading it reuses the plot.
        self._video_path: Optional[str] = None
        self._time_offset: float = 0.0
        self._start_frame: int = 0
        self._end_frame: int = 0
        self._suppress_sync = False
        #: True once the async decode worker has served a request (see
        #: decoder_ready); reset when a new PlotVideo/worker is spawned.
        self._decoder_live = False
        #: Wheel-zoom only acts on the active (last-clicked) view. The
        #: ActivePanelManager toggles this via widgets_meta; defaults True so a
        #: standalone view (tests, no manager) zooms without a prior click.
        self.selected = True
        self.setMinimumSize(MEDIA_VIEW_MIN_WIDTH, MEDIA_VIEW_MIN_HEIGHT)

        # Proxy-generation badge (top-left overlay): ⏳ while a low-res proxy
        # is being generated for this view, ✓/⚠ briefly on finish, hidden
        # otherwise. A floating child so it sits over the video canvas.
        self._proxy_badge = QLabel(self)
        self._proxy_badge.setStyleSheet(
            "QLabel { background: rgba(0,0,0,160); color: #e6e6e6;"
            " padding: 2px 6px; border-radius: 4px; font-size: 11px; }"
        )
        self._proxy_badge.hide()
        self._proxy_badge_timer = QTimer(self)
        self._proxy_badge_timer.setSingleShot(True)
        self._proxy_badge_timer.timeout.connect(self._proxy_badge.hide)

    def set_proxy_badge(self, state: Optional[str]) -> None:
        """Show the proxy-generation indicator. *state* ∈ {generating, ready,
        failed, None}; ``None`` hides it immediately."""
        self._proxy_badge_timer.stop()
        if state == "generating":
            self._proxy_badge.setText("⏳ proxy…")
            self._proxy_badge.setToolTip("Generating a low-resolution proxy for smooth navigation")
        elif state == "ready":
            self._proxy_badge.setText("✓ proxy")
            self._proxy_badge.setToolTip("Playing the low-resolution proxy")
            self._proxy_badge_timer.start(2500)
        elif state == "failed":
            self._proxy_badge.setText("⚠ proxy")
            self._proxy_badge.setToolTip("Proxy generation failed — using full resolution")
            self._proxy_badge_timer.start(4000)
        else:
            self._proxy_badge.hide()
            return
        self._proxy_badge.adjustSize()
        self._position_proxy_badge()
        self._proxy_badge.show()
        self._proxy_badge.raise_()

    def _position_proxy_badge(self) -> None:
        self._proxy_badge.move(6, 6)

    def set_blanked(self, blanked: bool) -> None:
        """Cover the video with black ("no input") or reveal it again.

        Used in session basis when the time marker sits where the current
        trial has no video (inter-trial gap, or another trial's span while
        its video is still loading). A floating cover widget — never
        ``clear()`` — so the decoder stays alive and unblanking is free.
        """
        cover = getattr(self, "_blank_cover", None)
        self._blanked = bool(blanked)
        if blanked:
            if cover is None:
                cover = QWidget(self)
                cover.setAutoFillBackground(True)
                cover.setStyleSheet("background-color: black;")
                cover.setAttribute(Qt.WA_TransparentForMouseEvents)
                self._blank_cover = cover
            cover.setGeometry(self.rect())
            cover.show()
            cover.raise_()
            self._proxy_badge.raise_()
        elif cover is not None:
            cover.hide()

    @property
    def is_blanked(self) -> bool:
        return bool(getattr(self, "_blanked", False))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._proxy_badge.isVisible():
            self._position_proxy_badge()
        if self.is_blanked and getattr(self, "_blank_cover", None) is not None:
            self._blank_cover.setGeometry(self.rect())

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def time_offset(self) -> float:
        return self._time_offset

    @property
    def start_frame(self) -> int:
        return self._start_frame

    @property
    def n_frames(self) -> int:
        """Number of frames in the trial-clipped range."""
        return max(0, self._end_frame - self._start_frame)

    @property
    def has_video(self) -> bool:
        return self._plot is not None

    @property
    def plot(self) -> Optional[PlotVideo]:
        return self._plot

    def canvas_widget(self) -> QWidget | None:
        """The canvas as laid out — what Qt overlays are parented to."""
        if self._plot is not None:
            return self._plot.canvas
        if self._static is not None:
            return self._static.canvas
        return None

    def key_target(self) -> QWidget | None:
        """The widget a key press pressed over the video actually lands on.

        ``rendercanvas``'s ``RenderCanvas`` is a *wrapper*: the inner render
        widget is the one with a focus policy, and its ``keyPressEvent`` neither
        ignores the event nor calls the base class, so nothing propagates out to
        the wrapper or the main window. An event filter installed on the wrapper
        alone therefore never sees a key pressed while the video has focus —
        which is exactly when the labelling dialog needs Backspace and Ctrl+Z.
        """
        canvas = self.canvas_widget()
        if canvas is None:
            return None
        focusable = [w for w in canvas.findChildren(QWidget) if w.focusPolicy() != Qt.NoFocus]
        return focusable[0] if focusable else canvas

    def set_video(
        self,
        video_path: str,
        fps: float,
        time_offset: float = 0.0,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        """Load a video file. Frame indices used by callers are trial frames
        (0 = ``start_frame`` in the underlying video).

        Reloading the file that is already decoded (trial change, camera
        re-applied, pose reload) keeps the existing ``PlotVideo`` and only
        re-clips the frame range. Rebuilding it would close one pynaviz decoder
        process and spawn another within seconds, and on Windows that races the
        new worker: spawn makes it re-import ``av``/``pygfx``/``pynapple``
        (~1.5-2 s) before it attaches to the shared-memory frame buffer, while
        ``PlotVideo.close()`` waits only ``join(timeout=2)`` before dropping the
        parent's handle — which is what destroys the mapping on Windows. The
        loser dies with ``FileNotFoundError: [WinError 2] ... 'wnsm_…'``.
        """
        video_path = str(video_path)
        reuse = self._plot is not None and self._video_path == video_path
        if reuse:
            self._detach_load_state()
        else:
            self.clear()
            self._plot = PlotVideo(video=video_path, parent=self)
            install_animate_guard(self._plot)
            self.layout().addWidget(self._plot.canvas)
            # Fresh worker: arm the liveness sentinel. The shm buffer is
            # zero-filled at creation, which would read as "frame 0 served";
            # -1 is only ever overwritten by the worker's first served frame.
            self._decoder_live = False
            if getattr(self._plot, "shared_index", None) is not None:
                self._plot.shared_index[0] = -1.0
        self._video_path = video_path
        self._fit_image_to_canvas(self._plot)
        self._fps = float(fps)
        total = self._plot.data.shape[0]
        self._start_frame = max(0, int(start_frame))
        self._end_frame = int(end_frame) if end_frame is not None else total
        self._end_frame = min(self._end_frame, total)
        self._time_offset = float(time_offset)
        if reuse:
            # Renderer handlers and the overlay hook below belong to the plot,
            # which survived — re-adding them would fire each of them twice.
            self.request_draw()
            return

        self._plot.renderer.add_event_handler(self._on_sync_event, "sync")
        # Route worker-thread frame updates into the pose overlay as well.
        original_update_extra = self._plot._update_extra_objects

        def _update_extra(frame_index, event_type=None):
            original_update_extra(frame_index, event_type)
            if self._overlay is not None:
                self._overlay.set_frame(frame_index - self._start_frame)

        self._plot._update_extra_objects = _update_extra
        # Clicks on the pygfx canvas go through the wgpu renderer, NOT Qt — so a
        # Qt event filter never sees them. Hook the renderer's pointer event to
        # emit `clicked` (used by the active-panel manager for the green edge).
        self._plot.renderer.add_event_handler(self._on_pointer_down, "pointer_down")
        self._plot.renderer.add_event_handler(self._on_pointer_move, "pointer_move")
        self._plot.renderer.add_event_handler(self._on_pointer_up, "pointer_up")
        self._install_click_filter(self._plot.canvas)
        self._enable_scroll_zoom(self._plot.controller)

    def _enable_scroll_zoom(self, controller) -> None:
        """Rebind the scroll wheel from frame-stepping to camera zoom.

        pynaviz's ``GetController`` repurposes zoom-to-cursor into stepping one
        frame per wheel notch. Frame navigation happens via the slider / play
        controls here, so the wheel is freed for zoom-to-cursor — matching the
        static-image views (plain ``PanZoomController``). Restore the base pygfx
        behaviour by delegating to it instead of the frame-step override; the
        base does not emit a pynaviz sync event, so zoom stays local and never
        moves the playhead. Gated on :pyattr:`selected` so only the active view
        zooms. Bound via ``MethodType`` because pygfx's action dispatcher
        introspects ``func.__func__.__code__`` (needs a real bound method)."""
        view = self

        def _zoom_to_point(ctrl, delta, *, screen_pos, rect):
            if not view.selected:
                return
            gfx.PanZoomController._update_zoom_to_point(ctrl, delta, screen_pos=screen_pos, rect=rect)
            ctrl.renderer_request_draw()

        controller._update_zoom_to_point = MethodType(_zoom_to_point, controller)

    def set_static_image(self, img: np.ndarray) -> None:
        """Show a still frame (pose-only mode, no video)."""
        self.clear()
        self._static = _StaticImagePlot(img, parent=self)
        self.layout().addWidget(self._static.canvas)
        self._static.renderer.add_event_handler(self._on_pointer_down, "pointer_down")
        self._static.renderer.add_event_handler(self._on_pointer_move, "pointer_move")
        self._static.renderer.add_event_handler(self._on_pointer_up, "pointer_up")
        self._install_click_filter(self._static.canvas)

    @staticmethod
    def _fit_image_to_canvas(plot) -> None:
        """Frame the exact video rectangle so it fills the canvas (up to the
        aspect-ratio letterbox). pygfx's ``show_object`` fits the bounding SPHERE
        (half-diagonal), which always leaves black padding; ``show_rect`` frames
        the image rectangle itself."""
        try:
            w, h = int(plot.texture.size[0]), int(plot.texture.size[1])
            plot.camera.show_rect(0, w, 0, h)
            plot.controller.renderer_request_draw()
        except Exception:  # noqa: BLE001 - framing is best-effort
            pass

    def _on_pointer_down(self, event=None) -> None:
        self.clicked.emit()
        if self._crop_select_cb is not None:
            self._handle_crop_press(event)
            return
        self._dispatch_label(event, "handle_click")

    def _on_pointer_move(self, event=None) -> None:
        if self._crop_select_cb is not None:
            if self._crop_anchor is not None and event is not None:
                xy = self.screen_to_image(event.x, event.y)
                if xy is not None:
                    self._update_crop_preview(xy)
            return
        self._dispatch_label(event, "handle_move")

    def _on_pointer_up(self, event=None) -> None:
        if self._crop_select_cb is not None:
            self._handle_crop_release(event)
            return
        self._dispatch_label(event, "handle_release")

    def _dispatch_label(self, event, method: str) -> None:
        """Forward a canvas pointer event to the labelling mode, if active.

        Press/release act on the left button only, so right-drag zoom and
        middle-click quickzoom keep working while labelling. A locked mode is
        skipped here as well as inside it, so panning costs no unprojection.

        A press carrying **any modifier belongs to the camera**: ``Shift`` +
        left-drag is what pans while a mode is armed (see
        :meth:`_bind_pan_to_shift`), and forwarding it as well panned the view
        *and* dragged the point under the cursor at the same time. Only the
        press is filtered — a drag can only have been started by an unmodified
        press, so moves and releases stay unconditional; a release dropped
        because a modifier happened to be down would leave the point stuck to
        the cursor.
        """
        if self._label_mode is None or event is None or self._label_mode.locked:
            return
        button = getattr(event, "button", 1)
        if method == "handle_click" and button == 2 and hasattr(self._label_mode, "handle_right_click"):
            # A mode that declares a right-click handler (box labelling's
            # "exclude" point) gets the unmodified right press; every other
            # mode keeps right-drag zoom untouched.
            if getattr(event, "modifiers", ()):
                return
            image_xy = self.screen_to_image(event.x, event.y)
            if image_xy is not None:
                self._label_mode.handle_right_click(*image_xy)
            return
        if method != "handle_move" and button != 1:
            return
        if method == "handle_click" and getattr(event, "modifiers", ()):
            return
        image_xy = self.screen_to_image(event.x, event.y)
        if image_xy is not None:
            getattr(self._label_mode, method)(*image_xy)

    def _install_click_filter(self, canvas) -> None:
        """Also catch Qt clicks (e.g. on the 2px margin around the canvas)."""
        try:
            canvas.installEventFilter(self)
        except (RuntimeError, AttributeError):
            pass

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            self.clicked.emit()
        return False

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    @property
    def overlay(self) -> PoseOverlay | None:
        return self._overlay

    def ensure_overlay(self) -> PoseOverlay | None:
        if self._overlay is not None:
            return self._overlay
        scene = None
        if self._plot is not None:
            scene = self._plot.scene
        elif self._static is not None:
            scene = self._static.scene
        if scene is None:
            return None
        self._overlay = PoseOverlay(scene)
        return self._overlay

    def image_height(self) -> float:
        return self.image_size()[1]

    def image_size(self) -> tuple[float, float]:
        """``(width, height)`` of the texture on screen, in its own pixels.

        Under proxy playback this is the proxy's size, not the source's:
        every rectangle the view reports (``screen_to_image``, a crop) is in
        these units, and a consumer speaking source pixels rescales.
        """
        plot = self._plot if self._plot is not None else self._static
        if plot is None:
            return 0.0, 0.0
        return float(plot.texture.size[0]), float(plot.texture.size[1])

    # ------------------------------------------------------------------
    # Keypoint labelling
    # ------------------------------------------------------------------

    def scene(self) -> gfx.Scene | None:
        """The pygfx scene backing this view (video or static image)."""
        if self._plot is not None:
            return self._plot.scene
        if self._static is not None:
            return self._static.scene
        return None

    def _render_target(self):
        """``(renderer, camera, controller)`` of whichever plot is loaded."""
        plot = self._plot if self._plot is not None else self._static
        if plot is None:
            return None, None, None
        return plot.renderer, plot.camera, plot.controller

    def screen_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        """Unproject canvas coordinates to texture pixels.

        Uses the pygfx camera, so it stays correct under pan and zoom. Returns
        image-space ``(x, y)`` with y pointing *down* — the video texture is
        rendered y-flipped, matching the convention in
        :mod:`~ethograph.gui.pose_overlay`.
        """
        renderer, camera, _ = self._render_target()
        if renderer is None:
            return None
        width, height = renderer.logical_size
        if not width or not height:
            return None
        ndc = np.array([2.0 * x / width - 1.0, 1.0 - 2.0 * y / height, 0.0, 1.0])
        world = np.linalg.inv(np.asarray(camera.camera_matrix)) @ ndc
        world = world / world[3]
        return float(world[0]), self.image_height() - float(world[1])

    def image_units_per_pixel(self) -> float:
        """Image pixels spanned by one screen pixel at the current zoom."""
        origin = self.screen_to_image(0.0, 0.0)
        offset = self.screen_to_image(1.0, 0.0)
        if origin is None or offset is None:
            return 1.0
        return abs(offset[0] - origin[0]) or 1.0

    def set_label_mode(self, mode) -> None:
        """Attach/detach a keypoint labelling mode (``None`` detaches).

        Left-drag is handed over to labelling while a mode is attached; panning
        moves to ``Shift`` + left-drag so navigation stays available.
        """
        self._label_mode = mode
        self._bind_pan_to_shift(mode is not None and not mode.locked)

    def set_label_locked(self, locked: bool) -> None:
        """Give left-drag back to panning without detaching the labelling mode.

        A locked mode keeps drawing its anchors and keeps its active keypoint —
        only the pointer changes hands — so the user can look around a frame and
        carry straight on labelling afterwards.
        """
        self._bind_pan_to_shift(self._label_mode is not None and not locked)

    def _bind_pan_to_shift(self, to_shift: bool) -> None:
        """Move the pan control between ``mouse1`` and ``shift+mouse1``."""
        _, _, controller = self._render_target()
        if controller is None:
            return
        if to_shift and self._pan_control is None:
            self._pan_control = controller.controls.pop("mouse1", None)
            if self._pan_control is not None:
                controller.controls["shift+mouse1"] = self._pan_control
        elif not to_shift and self._pan_control is not None:
            controller.controls.pop("shift+mouse1", None)
            controller.controls["mouse1"] = self._pan_control
            self._pan_control = None

    def clear_overlay(self) -> None:
        if self._overlay is not None:
            self._overlay.clear()
        self.request_draw()

    # ------------------------------------------------------------------
    # Display crop
    # ------------------------------------------------------------------

    @property
    def crop(self) -> tuple[int, int, int, int] | None:
        return self._crop

    @property
    def crop_selection_active(self) -> bool:
        return self._crop_select_cb is not None

    def set_crop(self, rect: tuple[int, int, int, int] | None) -> None:
        """Show only *rect* — ``(x0, y0, x1, y1)`` image pixels, y down — of the
        video; ``None`` reverts to the full frame.

        Display-only: the pixels outside are hidden with world-space clipping
        planes on the image material and the camera is framed on the rect, so
        the decoder, frame math and pose overlay are untouched.
        """
        self._crop = tuple(int(v) for v in rect) if rect is not None else None
        plot = self._plot
        if plot is None:
            return
        material = plot.image.material
        if self._crop is None:
            if material.clipping_plane_count:
                material.clipping_planes = []
                self._fit_image_to_canvas(plot)
            return
        h = float(plot.texture.size[1])
        material.clipping_planes = crop_clip_planes(self._crop, h)
        x0, y0, x1, y1 = self._crop
        try:
            plot.camera.show_rect(x0, x1, h - y1, h - y0)
            plot.controller.renderer_request_draw()
        except Exception:  # noqa: BLE001 - framing is best-effort
            pass

    def start_crop_selection(self, on_done, *, square: bool = False) -> bool:
        """Arm the rectangle tool: click a corner, drag, click (or release)
        again to select. *on_done* receives the snapped rect, or ``None`` for a
        degenerate selection. Panning moves to ``Shift`` + left-drag meanwhile.
        With *square* the cursor is held to a square from the anchor
        (:func:`square_corner`) — the preview and the result alike.
        Returns False when no video is loaded."""
        if self._plot is None:
            return False
        self.cancel_crop_selection()
        self._crop_select_cb = on_done
        self._crop_square = bool(square)
        self._bind_pan_to_shift(True)
        return True

    def _crop_corner(self, xy: tuple[float, float]) -> tuple[float, float]:
        """The cursor as the box's far corner, squared when the tool says so."""
        if not self._crop_square or self._crop_anchor is None:
            return xy
        return square_corner(self._crop_anchor[0], self._crop_anchor[1], xy[0], xy[1])

    def cancel_crop_selection(self) -> None:
        if self._crop_select_cb is None:
            return
        self._crop_select_cb = None
        self._crop_anchor = None
        self._remove_crop_preview()
        # Left-drag goes back to whoever held it before the tool was armed.
        self._bind_pan_to_shift(self._label_mode is not None and not self._label_mode.locked)

    def _handle_crop_press(self, event) -> None:
        # Modified presses belong to the camera (Shift+drag pans, see
        # _dispatch_label for the same rule while labelling).
        if event is None or getattr(event, "button", 1) != 1 or getattr(event, "modifiers", ()):
            return
        xy = self.screen_to_image(event.x, event.y)
        if xy is None:
            return
        if self._crop_anchor is None:
            self._crop_anchor = xy
            self._update_crop_preview(xy)
        else:
            self._finish_crop_selection(xy)

    def _handle_crop_release(self, event) -> None:
        """Finish on release only after a real two-dimensional drag.

        A release still near the anchor is the first half of the two-click
        gesture — the selection stays armed and the next click finishes it.
        """
        if self._crop_anchor is None or event is None or getattr(event, "button", 1) != 1:
            return
        xy = self.screen_to_image(event.x, event.y)
        if xy is None:
            return
        min_drag = 5.0 * self.image_units_per_pixel()
        ax, ay = self._crop_anchor
        xy = self._crop_corner(xy)
        if abs(xy[0] - ax) >= min_drag and abs(xy[1] - ay) >= min_drag:
            self._finish_crop_selection(xy)

    def _finish_crop_selection(self, xy: tuple[float, float]) -> None:
        cb = self._crop_select_cb
        ax, ay = self._crop_anchor
        xy = self._crop_corner(xy)
        w = float(self._plot.texture.size[0])
        h = float(self._plot.texture.size[1])
        rect = snap_crop_rect(ax, ay, xy[0], xy[1], w, h)
        self.cancel_crop_selection()
        cb(rect)

    def _update_crop_preview(self, xy: tuple[float, float]) -> None:
        scene = self.scene()
        if scene is None or self._crop_anchor is None:
            return
        ax, ay = self._crop_anchor
        xy = self._crop_corner(xy)
        x0, x1 = sorted((ax, xy[0]))
        y0, y1 = sorted((ay, xy[1]))
        h = self.image_height()
        pts = np.array(
            [
                [x0, h - y0, _Z_CROP_PREVIEW],
                [x1, h - y0, _Z_CROP_PREVIEW],
                [x1, h - y1, _Z_CROP_PREVIEW],
                [x0, h - y1, _Z_CROP_PREVIEW],
                [x0, h - y0, _Z_CROP_PREVIEW],
            ],
            dtype=np.float32,
        )
        if self._crop_preview is None:
            self._crop_preview = gfx.Line(
                gfx.Geometry(positions=pts),
                gfx.LineMaterial(thickness=2.0, color="#2ecc71"),
            )
            scene.add(self._crop_preview)
        else:
            self._crop_preview.geometry.positions.data[:] = pts
            self._crop_preview.geometry.positions.update_full()
        self.request_draw()

    def _remove_crop_preview(self) -> None:
        if self._crop_preview is None:
            return
        scene = self.scene()
        if scene is not None:
            scene.remove(self._crop_preview)
        self._crop_preview = None
        self.request_draw()

    # ------------------------------------------------------------------
    # Seeking
    # ------------------------------------------------------------------

    def current_video_frame(self) -> int:
        if self._plot is None:
            return 0
        return int(self._plot.controller.frame_index)

    def seek_video_frame(self, video_frame: int, synchronous: bool = False) -> None:
        """Seek to an absolute video frame without re-emitting time_changed."""
        if self._plot is None:
            return
        total = self._plot.data.shape[0]
        video_frame = max(0, min(int(video_frame), total - 1))
        self._suppress_sync = True
        try:
            if synchronous or not hasattr(self._plot, "request_queue"):
                self._plot.controller.frame_index = video_frame
                self._plot._update_buffer(video_frame, RenderTriggerSource.SET_FRAME)
                self._plot.controller.renderer_request_draw()
            else:
                # Async path: hand the request to the decoder worker process.
                self._plot.frame_ready.clear()
                while not self._plot.request_queue.empty():
                    try:
                        self._plot.request_queue.get_nowait()
                    except queue.Empty:
                        break
                self._plot.request_queue.put((video_frame, None, RenderTriggerSource.UNKNOWN))
                self._plot.controller.frame_index = video_frame
        finally:
            self._suppress_sync = False

    def seek_trial_frame(self, trial_frame: int, synchronous: bool = False) -> None:
        self.seek_video_frame(trial_frame + self._start_frame, synchronous=synchronous)

    def decoder_ready(self) -> bool:
        """True once the async decode worker is proven to serve requests.

        A freshly spawned worker re-imports ``av``/``pygfx`` before it can
        answer (~2 s on Windows) and async seeks issued meanwhile are
        superseded rather than served. ``set_video`` arms a ``-1`` sentinel in
        ``shared_index`` at spawn; the worker's first served frame overwrites
        it, which this notices and remembers. Views without an async worker
        (no plot, sync-only plot) are trivially ready.
        """
        plot = self._plot
        if plot is None or not hasattr(plot, "request_queue"):
            return True
        if not self._decoder_live:
            shared = getattr(plot, "shared_index", None)
            if shared is not None and float(shared[0]) >= 0.0:
                self._decoder_live = True
        return self._decoder_live

    def seek_to_time(self, t_trial: float) -> None:
        """Seek from a trial-relative time (used for follower cameras)."""
        if self._plot is None or self._fps <= 0:
            self.set_overlay_time(t_trial)
            return
        video_t = t_trial - self._time_offset + self._start_frame / self._fps
        frame = int(round(video_t * self._fps))
        self.seek_video_frame(frame)

    def set_overlay_time(self, t_trial: float) -> None:
        """Animate the pose overlay on a static-image view from marker time.

        Video views ignore this — their overlay is driven by the decoder's
        frame updates; a static image has no frame clock, so the pose frame
        is derived from the trial time and the pose's own fps.
        """
        if self._static is None or self._overlay is None or self.static_pose_fps <= 0:
            return
        self._overlay.set_frame(int(round(t_trial * self.static_pose_fps)))
        self.request_draw()

    def request_draw(self) -> None:
        if self._plot is not None:
            self._plot.controller.renderer_request_draw()
        elif self._static is not None:
            self._static.request_draw()

    def nudge_render_if_stalled(self) -> bool:
        """Re-arm a stalled video render chain; ``True`` if a nudge was needed.

        Called by VideoSync's playback watchdog — see
        :func:`nudge_stalled_render`."""
        if self._plot is None:
            return False
        return nudge_stalled_render(self._plot)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_sync_event(self, event) -> None:
        """User interaction on the canvas changed the frame → propagate."""
        if self._suppress_sync:
            return
        kwargs = getattr(event, "kwargs", {})
        t = kwargs.get("current_time")
        if t is None and "cam_state" in kwargs:
            return  # camera pan/zoom, not a time change
        if t is not None:
            self.time_changed.emit(float(t))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _detach_load_state(self) -> None:
        """Drop everything tied to one load — labelling mode and pose overlay —
        while keeping the plot itself. The half of :meth:`clear` that a reusing
        :meth:`set_video` still needs; the overlay is rebuilt lazily by
        :meth:`ensure_overlay` on the same scene."""
        self.cancel_crop_selection()
        self.set_label_mode(None)
        if self._overlay is not None:
            self._overlay.clear()
            self._overlay = None

    def clear(self) -> None:
        self._detach_load_state()
        self._crop = None
        if self._plot is not None:
            _detach_canvas(self.layout(), self._plot.canvas)
            try:
                self._plot.close()
            except Exception:
                pass
            self._plot = None
        if self._static is not None:
            _detach_canvas(self.layout(), self._static.canvas)
            try:
                self._static.close()
            except RuntimeError:
                logger.debug("Static image canvas was already deleted.", exc_info=True)
            self._static = None
        self.static_image_path = None
        self.static_pose_fps = 0.0
        self._fps = 0.0
        self._video_path = None
        self._time_offset = 0.0
        self._start_frame = 0
        self._end_frame = 0

    def closeEvent(self, event):
        self.clear()
        super().closeEvent(event)
