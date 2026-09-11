"""Video lifecycle management over pygfx camera views.

``VideoArea`` hosts the primary :class:`CameraView` (inside the video
dock); every extra camera view is its own closable shell dock, so single
views can be removed individually. ``VideoManager`` keeps its old public
surface (update_video, add_camera, extra_widgets, …) but creates pygfx views
instead of napari layers.

Every view — primary included — carries the camera it shows on
``view.camera_name`` and is titled by :func:`camera_dock_title`, so no panel
is ever labelled generically ("Video") while its neighbours name a camera.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
from qtpy.QtCore import QEvent, Qt, QTimer, Signal
from qtpy.QtWidgets import QSplitter, QVBoxLayout, QWidget

from ethograph.io.time_model import trial_frame_window
from ethograph.io.validation import IMAGE_EXTENSIONS
from ethograph.io.video_probe import VideoProbe, probe_video  # noqa: F401  (re-exported for GUI callers)
from ethograph.io.video_proxy import proxy_cache_path
from ethograph.utils.paths import cache_dir

from .app_constants import MEDIA_VIEW_MIN_HEIGHT, MEDIA_VIEW_MIN_WIDTH
from .notify import notify
from .proxy_manager import ProxyManager
from .pygfx_video import CameraView
from .video_sync import VideoSync

#: Share of the window height the camera grid takes (the plots keep the rest).
CAMERA_GRID_HEIGHT_RATIO = 0.6


def camera_grid_shape(n_views: int) -> tuple[int, int]:
    """``(rows, cols)`` for *n_views* camera panels with no saved layout.

    Up to three panels stay in one row; beyond that the grid is as square as
    it gets, wide before tall: 4 → 2×2, 6 → 2 rows × 3, 9 → 3×3, 5 → 3 + 2.
    """
    if n_views < 1:
        raise ValueError("a grid needs at least one view")
    if n_views <= 3:
        return 1, n_views
    cols = math.ceil(math.sqrt(n_views))
    rows = math.ceil(n_views / cols)
    return rows, cols


def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def camera_dock_title(camera_name: str | None, media_path: str | None, mode_suffix: str = "") -> str:
    """Panel title for a camera view: ``"cam-1 (front.mp4)"`` plus the individual mode suffix.

    The camera name alone is ambiguous once several views are open (and says
    nothing about which trial's file is on screen), so the file name is
    appended whenever one is loaded, and *mode_suffix*
    (:meth:`~ethograph.gui.app_state.ObservableAppState.panel_mode_suffix`)
    after that.
    """
    name = str(camera_name or "").strip() or "Camera"
    if media_path:
        file_name = media_path.rstrip("/").rsplit("/", 1)[-1] if is_url(media_path) else Path(media_path).name
        if file_name:
            name = f"{name} ({file_name})"
    return name + mode_suffix


def proxy_cache_dir(video_path: str | None = None) -> Path:
    """Central directory holding all cached video proxies.

    One shared location (``~/.ethograph/cache/proxies``) rather than a folder
    beside each source, so the whole cache is easy to find and clear when disk
    space is tight, and so proxies can be written even when the source lives on
    read-only or network media. Filenames are keyed by source identity
    (path + size + mtime), so a single flat folder never collides. The
    *video_path* argument is accepted for call-site compatibility but ignored.
    """
    return cache_dir("proxies")


class VideoArea(QWidget):
    """Primary camera view; extra camera views live in their own shell docks.

    The primary view is hosted here (inside the shell's video dock, titled
    after the camera it shows). Every extra camera view is its own closable
    :class:`QDockWidget` in the shell's top dock area (like space plots) — the
    user removes any single view via its dock's ✕. Without a shell (tests),
    extras fall back into the local splitter."""

    #: Emitted on any mouse press inside the video area (→ video context sidebar).
    clicked = Signal()
    #: Emitted with a CameraView when an extra camera is added (for active-panel).
    camera_added = Signal(object)
    #: Emitted with a CameraView after its view was removed (dock ✕ or programmatic).
    camera_view_removed = Signal(object)
    #: The primary video dock's ✕ was clicked → VideoManager tears the video down.
    primary_close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        #: The shell main window (set by EthographMainWindow); hosts the
        #: per-camera docks.
        self.shell = None
        #: Set during VideoManager.cleanup() so shutdown-driven removals
        #: don't fire camera_view_removed handlers (combo resets, saves).
        self._suppress_removed_signal = False

        self._splitter = QSplitter(Qt.Horizontal)
        self.primary = CameraView()
        self.primary.camera_name = None
        self._splitter.addWidget(self.primary)
        layout.addWidget(self._splitter)
        self._extras: dict[str, CameraView] = {}

    @property
    def extras(self) -> dict[str, CameraView]:
        return self._extras

    def add_extra(self, name: str) -> CameraView:
        """Add a new view for camera *name*. Duplicates are allowed — each
        call creates a new view (its own dock); the dict key is made unique,
        and the real camera name is kept on ``view.camera_name``."""
        key = name
        n = 2
        while key in self._extras:
            key = f"{name} ({n})"
            n += 1
        view = CameraView()
        view.camera_name = name
        self._extras[key] = view
        shell = self.shell
        if shell is not None and hasattr(shell, "add_dock_widget"):
            view.setMinimumSize(MEDIA_VIEW_MIN_WIDTH, MEDIA_VIEW_MIN_HEIGHT)
            dock = shell.add_dock_widget(view, area="top", name=camera_dock_title(name, None))
            # The objectName keys layout persistence and must stay the raw
            # instance key — the title carries the (mutable) file name.
            dock.setObjectName(f"CameraViewDock {key}")
            dock._camera_key = key
            view.dock_widget = dock
            dock.installEventFilter(self)
            desired_width = max(200, int(shell.width() * 0.2))
            QTimer.singleShot(0, lambda: shell.resizeDocks([dock], [desired_width], Qt.Horizontal))
        else:
            self._splitter.addWidget(view)
            self._equalize()
        self.camera_added.emit(view)
        return view

    def camera_docks(self) -> list:
        """Every docked camera panel, primary first, then extras in creation
        order — the order :meth:`arrange_grid` fills rows in.

        A panel counts as soon as it exists: a dock added to the shell is not
        shown until the event loop runs, so asking Qt whether it is visible
        would miss every view the load just created. The primary is the one
        panel whose existence is conditional (a session without a video gets
        no video dock), and ``_video_dock_enabled`` is that answer."""
        shell = self.shell
        if shell is None:
            return []
        docks = []
        primary = getattr(shell, "_video_dock", None)
        if primary is not None and getattr(shell, "_video_dock_enabled", True):
            docks.append(primary)
        docks += [dock for view in self._extras.values() if (dock := getattr(view, "dock_widget", None)) is not None]
        return [d for d in docks if not d.isFloating()]

    def arrange_grid(self) -> tuple[int, int] | None:
        """Lay the camera docks out as the grid :func:`camera_grid_shape`
        picks for their count, instead of the one long row Qt gives docks
        added to the same area. Returns the ``(rows, cols)`` applied, or
        ``None`` when a single row is already right."""
        shell = self.shell
        docks = self.camera_docks()
        if shell is None or len(docks) <= 3:
            return None
        rows, cols = camera_grid_shape(len(docks))
        cell = lambda r, c: docks[r * cols + c] if r * cols + c < len(docks) else None  # noqa: E731
        for r in range(1, rows):
            for c in range(cols):
                below = cell(r, c)
                if below is not None:
                    shell.splitDockWidget(cell(r - 1, c), below, Qt.Vertical)

        def _size():
            # Equal cells: every dock asks for the same width and height, so
            # each splitter divides its own row/column evenly.
            width = max(MEDIA_VIEW_MIN_WIDTH, shell.width() // cols)
            height = max(MEDIA_VIEW_MIN_HEIGHT, int(shell.height() * CAMERA_GRID_HEIGHT_RATIO) // rows)
            shell.resizeDocks(docks, [width] * len(docks), Qt.Horizontal)
            shell.resizeDocks(docks, [height] * len(docks), Qt.Vertical)

        QTimer.singleShot(0, _size)
        return rows, cols

    def eventFilter(self, obj, event):
        key = getattr(obj, "_camera_key", None)
        if key is not None and event.type() == QEvent.Close:
            # The dock's ✕ was clicked. Defer the teardown — removing the
            # dock while it handles its own close event is unsafe.
            QTimer.singleShot(0, lambda: self.remove_extra(key))
        elif getattr(obj, "_is_primary_video_dock", False) and event.type() == QEvent.Close:
            # Same deal for the primary video dock: closing it must unload
            # the video (plot, worker, sync), not merely hide the dock.
            QTimer.singleShot(0, self.primary_close_requested.emit)
        return False

    def remove_extra(self, name: str) -> None:
        view = self._extras.pop(name, None)
        if view is None:
            return
        view.clear()
        dock = getattr(view, "dock_widget", None)
        if dock is not None and self.shell is not None:
            dock.removeEventFilter(self)
            # hide + deleteLater, NOT shell.removeDockWidget(): removing a
            # dock holding a GL canvas makes the next shell.show() crash
            # natively on Windows (same fix as remove_space_plot).
            dock.hide()
            dock.deleteLater()
        else:
            view.setParent(None)
            view.deleteLater()
            self._equalize()
        if not self._suppress_removed_signal:
            self.camera_view_removed.emit(view)

    def _equalize(self) -> None:
        """Give every camera panel an equal share of the width."""
        n = self._splitter.count()
        if n > 0:
            total = max(1, self._splitter.width())
            self._splitter.setSizes([total // n] * n)


class VideoManager:
    """Manages primary and extra camera views, audio path resolution, sync."""

    def __init__(self, video_area: VideoArea, app_state):
        self.video_area = video_area
        self.app_state = app_state
        self._video_format_warned = False
        self._audio_row_widgets: list = []
        self.proxy_mgr = ProxyManager(proxy_cache_dir, parent=video_area)
        self.proxy_mgr.proxy_ready.connect(self._on_proxy_ready)
        self.proxy_mgr.proxy_started.connect(lambda s: self._set_proxy_badge(s, "generating"))
        self.proxy_mgr.proxy_failed.connect(lambda s: self._set_proxy_badge(s, "failed"))
        video_area.primary_close_requested.connect(self.close_primary_video)
        #: Display crop per camera name (see CameraView.set_crop). Session
        #: state, not saved: re-applied to every view of the camera on each
        #: trial load, so a crop follows the camera across trial navigation.
        self._camera_crops: dict[str, tuple[int, int, int, int]] = {}
        self._video_reloaded_callback = None

    # ------------------------------------------------------------------
    # Display crop (per camera)
    # ------------------------------------------------------------------

    def camera_crop(self, camera_name: str | None) -> tuple[int, int, int, int] | None:
        return self._camera_crops.get(str(camera_name)) if camera_name else None

    def set_camera_crop(self, camera_name: str, rect: tuple[int, int, int, int]) -> None:
        """Crop every view of *camera_name* (now and on later trial loads)."""
        self._camera_crops[str(camera_name)] = tuple(int(v) for v in rect)
        self._apply_camera_crop(camera_name)

    def clear_camera_crop(self, camera_name: str) -> None:
        self._camera_crops.pop(str(camera_name), None)
        self._apply_camera_crop(camera_name)

    def _apply_camera_crop(self, camera_name: str) -> None:
        rect = self._camera_crops.get(str(camera_name))
        views = self.views_for_camera(camera_name)
        if getattr(self.primary_view, "camera_name", None) == camera_name:
            views.insert(0, self.primary_view)
        for view in views:
            view.set_crop(rect)

    @property
    def primary_view(self) -> CameraView:
        return self.video_area.primary

    @property
    def extra_widgets(self) -> dict[str, CameraView]:
        return self.video_area.extras

    # ------------------------------------------------------------------
    # Panel titles
    # ------------------------------------------------------------------

    def refresh_view_titles(self) -> None:
        """Re-title every camera view — after the sidebar's individual changed."""
        for view in [self.primary_view, *self.extra_widgets.values()]:
            self.refresh_view_title(view)

    def refresh_view_title(self, view: CameraView) -> None:
        """Re-title *view*'s panel from the camera it shows and its file.

        The primary lives in the shell's video dock and every extra in its
        own dock, but both are titled the same way — a panel must never read
        as a generic "Video" while its neighbours name a camera.
        """
        title = camera_dock_title(
            getattr(view, "camera_name", None),
            getattr(view, "source_video_path", None) or getattr(view, "static_image_path", None),
            self.app_state.panel_mode_suffix(view),
        )
        dock = getattr(view, "dock_widget", None)
        if dock is not None:
            dock.setWindowTitle(title)
            return
        shell = getattr(self.video_area, "shell", None)
        if view is self.primary_view and shell is not None and hasattr(shell, "set_video_dock_title"):
            shell.set_video_dock_title(title)

    # ------------------------------------------------------------------
    # Primary video
    # ------------------------------------------------------------------

    def update_video(self, plot_container):
        if not self.app_state.ready:
            return
        camera = self.app_state.primary_camera
        sio = getattr(self.app_state, "nwb_alignment", None)
        # The primary is a camera view like any other: it must carry the camera
        # it shows, so titles, pose lookup and proxy handling can treat it the
        # same as the extras instead of special-casing "the Video panel".
        self.primary_view.camera_name = camera
        if camera and sio is not None:
            self.app_state.video_path = sio.resolve_media_path(
                self.app_state.trials_sel,
                "video",
                device=camera,
                fallback_folder=self.app_state.video_folder,
            )
        else:
            self.app_state.video_path = None
        shell = getattr(self.video_area, "shell", None)
        if not self.app_state.video_path:
            # No video in this session/trial → no Video panel slot at all.
            self._cleanup_primary_video()
            if shell is not None:
                shell.set_video_dock_visible(False)
            return
        if shell is not None:
            shell.set_video_dock_visible(True)
        restore_frame = max(0, int(getattr(self.app_state, "current_frame", 0) or 0))
        if Path(self.app_state.video_path).suffix.lower() in IMAGE_EXTENSIONS:
            # Pose-only session: the "camera" is a still image (no playback).
            self._cleanup_primary_video()
            self._setup_primary_image()
            return
        self._warn_video_format()
        # Only the VideoSync is dropped here: clearing the view too would close
        # its decoder process, and set_video reuses it when the trial change
        # kept the same file (see CameraView.set_video).
        self._teardown_primary_sync()
        self._setup_primary_video(restore_frame)

    def _setup_primary_image(self):
        import imageio.v3 as iio

        try:
            img = iio.imread(self.app_state.video_path)
        except (OSError, ValueError) as e:
            notify(f"Image file could not be loaded: {e}", "warning")
            return
        view = self.primary_view
        view.set_static_image(img)
        view.static_image_path = self.app_state.video_path
        self.refresh_view_title(view)

    def _teardown_primary_sync(self):
        """Drop the ``VideoSync`` driving the primary view, leaving the view
        itself loaded (see :meth:`_cleanup_primary_video` to also unload it)."""
        sync = getattr(self.app_state, "video", None)
        if sync is not None:
            try:
                sync.frame_changed.disconnect(self._on_primary_frame_changed)
            except (RuntimeError, TypeError):
                pass
            # Drop the proxy-swap hook BEFORE cleanup() — stopping the sync
            # emits playback_stopped, which would otherwise re-enter
            # _apply_ready_proxies and loop back into another reload.
            try:
                sync.playback_stopped.disconnect(self._apply_ready_proxies)
            except (RuntimeError, TypeError):
                pass
            sync.cleanup()
            self.app_state.video = None

    def _cleanup_primary_video(self):
        self._teardown_primary_sync()
        self.primary_view.clear()
        self.primary_view.source_video_path = None
        self.primary_view.decode_video_path = None
        self.refresh_view_title(self.primary_view)

    def close_primary_video(self):
        """The primary video dock's ✕: tear the video down like an extra's close.

        Without this the dock merely hid while the view kept a live plot,
        decode worker and canvas. ``has_video`` goes False here, so re-adding
        the camera from the popup takes the primary path again instead of
        forking an extra view over an invisible primary. A later trial change
        re-resolves the camera and brings the panel back — the close removes
        the loaded video, not the camera selection.
        """
        self._cleanup_primary_video()
        shell = getattr(self.video_area, "shell", None)
        if shell is not None and hasattr(shell, "set_video_dock_visible"):
            shell.set_video_dock_visible(False)

    def _trial_clip(self, fps: float, time_offset: float, nframes: int) -> tuple[int, int, float]:
        """Compute (start_frame, end_frame, effective_offset) for the trial."""
        alignment = getattr(self.app_state, "trial_alignment", None)
        if alignment and alignment.trial_range:
            start_frame, end_frame = trial_frame_window(alignment.trial_range, fps, time_offset)
            end_frame = min(end_frame, nframes)
            if start_frame > 0 or end_frame < nframes:
                return start_frame, end_frame, 0.0
        return 0, nframes, time_offset

    def _decode_path(self, video_path: str) -> str:
        """Return the path the DECODER should read for *video_path*.

        In "proxy" quality mode, substitute a cached low-res proxy when one
        already exists; otherwise fall back to the source (never generate
        synchronously — that would freeze the GUI for minutes). Alignment and
        frame math elsewhere always use the source, which is safe because the
        proxy has identical fps and frame count.
        """
        if getattr(self.app_state, "video_quality_mode", "full") != "proxy":
            return video_path
        if Path(video_path).suffix.lower() in IMAGE_EXTENSIONS or is_url(video_path):
            return video_path
        try:
            proxy = proxy_cache_path(video_path, proxy_cache_dir(video_path))
        except OSError:
            return video_path
        return str(proxy) if proxy.exists() else video_path

    def visible_video_sources(self) -> list[str]:
        """Source paths of every view currently showing a (non-image) video.

        De-duplicated: the same physical file shown in several views yields a
        single proxy job.
        """
        sources: list[str] = []
        seen: set[str] = set()
        for view in [self.primary_view, *self.extra_widgets.values()]:
            if not getattr(view, "has_video", False):
                continue
            src = getattr(view, "source_video_path", None)
            if not src or src in seen:
                continue
            if Path(src).suffix.lower() in IMAGE_EXTENSIONS or is_url(src):
                continue
            seen.add(src)
            sources.append(src)
        return sources

    def sync_proxies(self) -> None:
        """Reconcile background proxy jobs against the on-screen video set.

        The single choke point for proxy lifecycle: call it whenever the set
        of visible videos may have changed (panel open/close, trial change,
        quality toggle). Starts jobs for newly-visible videos, cancels jobs
        for videos that went away (so the thread count never grows unbounded),
        and swaps in any proxy that is already available.
        """
        if getattr(self.app_state, "video_quality_mode", "full") == "proxy":
            self.proxy_mgr.sync(self.visible_video_sources())
        else:
            self.proxy_mgr.cancel_all()
            self._clear_proxy_badges()
        self._apply_ready_proxies()

    def _on_proxy_ready(self, source: str) -> None:
        self._apply_ready_proxies()
        self._set_proxy_badge(source, "ready")

    def _set_proxy_badge(self, source: str, state: str | None) -> None:
        for view in [self.primary_view, *self.extra_widgets.values()]:
            if getattr(view, "source_video_path", None) == source and hasattr(view, "set_proxy_badge"):
                view.set_proxy_badge(state)

    def _clear_proxy_badges(self) -> None:
        for view in [self.primary_view, *self.extra_widgets.values()]:
            if hasattr(view, "set_proxy_badge"):
                view.set_proxy_badge(None)

    def _apply_ready_proxies(self) -> None:
        """Reload any view whose decode path no longer matches the desired one.

        Idempotent: swaps a view onto its proxy once available, or back to the
        source when proxy mode is off. The primary is skipped while playing so
        playback isn't interrupted; it re-applies when playback stops or on the
        next sync.
        """
        # Reloading a view tears down its VideoSync, which emits signals that
        # can re-enter here — guard against recursion.
        if getattr(self, "_applying_proxies", False):
            return
        self._applying_proxies = True
        reloaded = False
        try:
            view = self.primary_view
            if getattr(view, "has_video", False):
                src = getattr(view, "source_video_path", None)
                if src and self._decode_path(src) != getattr(view, "decode_video_path", None):
                    video = getattr(self.app_state, "video", None)
                    if video is None or not video.is_playing:
                        self._reload_primary()
                        reloaded = True
            for view in list(self.extra_widgets.values()):
                if not getattr(view, "has_video", False):
                    continue
                src = getattr(view, "source_video_path", None)
                if src and self._decode_path(src) != getattr(view, "decode_video_path", None):
                    self._reload_extra(view)
                    reloaded = True
        finally:
            self._applying_proxies = False
        if reloaded and self._video_reloaded_callback is not None:
            # A rebuilt view is a new scene at a new resolution: the pose
            # overlay has to be drawn again, scaled to it.
            self._video_reloaded_callback()

    def _reload_primary(self) -> None:
        frame = max(0, int(getattr(self.app_state, "current_frame", 0) or 0))
        self._cleanup_primary_video()
        self._setup_primary_video(frame)

    def reset_primary_video(self) -> None:
        """Tools ▸ Reset video view — rebuild the primary ``PlotVideo`` in place.

        The proven recovery for a dead render chain (a frozen image while
        audio and the playhead keep moving) without closing and re-adding the
        panel. Safe against the cold-worker shared-memory race documented in
        ``CameraView.set_video``: a *frozen* plot's worker is warm, so
        ``close()``'s join succeeds before the new worker spawns.
        """
        video = getattr(self.app_state, "video", None)
        if video is not None and video.is_playing:
            video.stop()
        if not self.primary_view.has_video:
            notify("No video is loaded.", "warning")
            return
        self._reload_primary()
        notify("Video view was rebuilt.")

    def _reload_extra(self, view: CameraView) -> None:
        camera = getattr(view, "camera_name", None)
        src = getattr(view, "source_video_path", None)
        if not camera or not src:
            return
        try:
            probe = probe_video(src)
        except (OSError, ValueError, av.AVError):
            return
        self._load_extra_video(view, camera, src, probe)
        self._sync_widget_to_current_time(view)

    def _setup_primary_video(self, restore_frame: int):
        try:
            probe = probe_video(self.app_state.video_path)
        except (OSError, ValueError, av.AVError) as e:
            # The caller no longer pre-clears the view (so set_video can reuse a
            # loaded plot), so an abort has to unload whatever is still shown.
            self.primary_view.clear()
            notify(f"Video file could not be loaded: {e}", "warning")
            return

        camera = self.app_state.primary_camera
        if probe.fps and self.app_state.dt is not None:
            self.app_state.nwb_alignment.set_stream_rate(probe.fps, "video", camera)

        # Offset from this camera's own stream, exactly like the extras get
        # theirs. `trial_alignment` is rebuilt for the primary camera, but it
        # is one indirection away from the file actually being opened here, so
        # reading the stream directly keeps the primary from drifting out of
        # sync with the other views when the primary camera changes.
        sio = getattr(self.app_state, "nwb_alignment", None)
        if sio is not None and camera:
            video_time_offset = sio.stream_offset_for_trial(self.app_state.trials_sel, "video", camera)
        else:
            alignment = getattr(self.app_state, "trial_alignment", None)
            video_time_offset = alignment.video_offset if alignment else 0.0
        # The probe is the ground truth for the file being decoded; the stored
        # stream rate can still describe the previously selected camera.
        fps = probe.fps
        start_frame, end_frame, effective_offset = self._trial_clip(fps, video_time_offset, probe.nframes)

        view = self.primary_view
        decode = self._decode_path(self.app_state.video_path)
        try:
            view.set_video(
                decode,
                fps=fps,
                time_offset=effective_offset,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        except (OSError, ValueError) as e:
            view.clear()
            notify(f"Video file could not be loaded: {e}", "warning")
            return
        view.source_video_path = self.app_state.video_path
        view.decode_video_path = decode
        view.source_size = (probe.width, probe.height)
        view.set_crop(self._camera_crops.get(camera))
        self.refresh_view_title(view)

        sync = VideoSync(
            app_state=self.app_state,
            view=view,
            video_source=self.app_state.video_path,
            audio_source=self.app_state.audio_path,
        )
        self.app_state.video = sync
        self.app_state.num_frames = sync.total_frames

        sync.frame_changed.connect(self._on_primary_frame_changed)
        sync.frame_changed.connect(self._sync_extra_cameras)
        # Apply any proxy swap that was deferred while playing, once stopped.
        sync.playback_stopped.connect(self._apply_ready_proxies)
        restore_frame = min(restore_frame, max(0, sync.total_frames - 1))
        sync.seek_to_frame(restore_frame)
        self.app_state.current_frame = restore_frame

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def update_audio(self, plot_container):
        if not self.app_state.ready:
            return
        self._update_audio_path()
        self._update_audio_ui(plot_container)

    def _update_audio_path(self) -> None:
        self.app_state.audio_path = None
        if self.app_state.audio_folder and hasattr(self.app_state, "mics_sel"):
            audio_path, _ = self.app_state.get_audio_source()
            if audio_path:
                self.app_state.audio_path = audio_path

    def _update_audio_ui(self, plot_container):
        has_audio = bool(self.app_state.audio_path)
        for w in self._audio_row_widgets:
            w.setVisible(has_audio)
        if has_audio:
            plot_container.update_audio_panels()

    def set_audio_row_widgets(self, widgets):
        self._audio_row_widgets = widgets

    def _warn_video_format(self):
        video_path = self.app_state.video_path
        if not video_path or is_url(video_path):
            return
        ext = Path(video_path).suffix.lower()
        if ext in (".avi", ".mov") and not self._video_format_warned:
            self._video_format_warned = True
            notify(
                f"Video format '{ext}' may have inaccurate frame seeking. "
                f"See https://akseli-ilmanen.github.io/ethograph/advanced/troubleshooting.html",
                "warning",
            )

    # ------------------------------------------------------------------
    # Frame sync
    # ------------------------------------------------------------------

    def set_frame_changed_callback(self, callback):
        self._frame_changed_callback = callback

    def set_video_reloaded_callback(self, callback) -> None:
        """Called after a view is rebuilt in place (proxy on/off): redraw what sits on it."""
        self._video_reloaded_callback = callback

    def _on_primary_frame_changed(self, frame_number: int):
        if hasattr(self, "_frame_changed_callback"):
            self._frame_changed_callback(frame_number)

    def _sync_extra_cameras(self, frame_number: int):
        video = getattr(self.app_state, "video", None)
        if video is None or not self.extra_widgets:
            return
        # frame_to_time is display-clock; extras' seek_to_time is trial-local.
        resolved = self.app_state.from_display(video.frame_to_time(frame_number))
        if resolved is None:
            return
        for view in self.extra_widgets.values():
            view.seek_to_time(resolved[1])

    def toggle_pause_resume(self, plot_container):
        video = getattr(self.app_state, "video", None)
        if video:
            video.toggle_pause_resume()
        else:
            plot_container.toggle_pause_resume()

    # ------------------------------------------------------------------
    # Extra cameras
    # ------------------------------------------------------------------

    def views_for_camera(self, camera_name: str) -> list[CameraView]:
        """Every extra view showing *camera_name* (duplicates included)."""
        return [view for key, view in self.extra_widgets.items() if getattr(view, "camera_name", key) == camera_name]

    def refresh_extra_videos(self) -> None:
        """Reload every extra camera view for the current trial.

        Extra views are layout instances created by the add-panel popup, so the
        camera combos are NOT a record of what is on screen — a view dropped
        from the popup appears in no combo. Iterating the live views (exactly
        what the pose overlay does) is the only way a view follows the trial;
        driving this off the combos left drag-dropped panels frozen on the
        trial that was open when they were created.
        """
        views = [view for view in self.extra_widgets.values() if not getattr(view, "static_image_path", None)]
        if not views:
            return

        video_folder = getattr(self.app_state, "video_folder", None)
        paths: dict[str, str] = {}
        for view in views:
            name = getattr(view, "camera_name", None)
            if not name or name in paths:
                continue
            path = self._resolve_video_path(name, video_folder)
            if path:
                paths[name] = path

        probes = self.open_readers_parallel(paths)
        for view in views:
            name = getattr(view, "camera_name", None)
            video_path = paths.get(name)
            probe = probes.get(name)
            if video_path is None or probe is None:
                continue
            self._store_camera_fps_in_session(name, probe.fps)
            self._load_extra_video(view, name, video_path, probe)
            self._sync_widget_to_current_time(view)

    def add_camera(
        self,
        camera_name: str,
        video_path: str,
        layout_mgr=None,
        meta_widget=None,
        *,
        reader=None,
        duplicate: bool = False,
    ):
        """Show *camera_name* as an extra view.

        Without *duplicate*, an existing view of that camera is reloaded
        (trial change / combo re-apply). With *duplicate*, a NEW view is
        always created — the same camera can be shown any number of times.
        """
        if not duplicate and self.views_for_camera(camera_name):
            self._update_existing_camera(camera_name, video_path, reader=reader)
            return

        probe = reader if isinstance(reader, VideoProbe) else None
        if probe is None:
            try:
                probe = probe_video(video_path)
            except (OSError, ValueError, av.AVError) as e:
                notify(f"Could not open camera '{camera_name}': {e}", "warning")
                return
        self._store_camera_fps_in_session(camera_name, probe.fps)

        view = self.video_area.add_extra(camera_name)
        self._load_extra_video(view, camera_name, video_path, probe)
        self._sync_widget_to_current_time(view)

    def _update_existing_camera(self, camera_name: str, video_path: str, *, reader=None):
        probe = reader if isinstance(reader, VideoProbe) else None
        if probe is None:
            try:
                probe = probe_video(video_path)
            except (OSError, ValueError, av.AVError) as e:
                notify(f"Could not open camera '{camera_name}': {e}", "warning")
                return
        self._store_camera_fps_in_session(camera_name, probe.fps)
        for view in self.views_for_camera(camera_name):
            self._load_extra_video(view, camera_name, video_path, probe)
            view.show()
            self._sync_widget_to_current_time(view)

    def _load_extra_video(self, view: CameraView, camera_name: str, video_path: str, probe: VideoProbe):
        sio = getattr(self.app_state, "nwb_alignment", None)
        time_offset = 0.0
        if sio is not None:
            trial_id = self.app_state.trials_sel
            time_offset = sio.stream_offset_for_trial(trial_id, "video", camera_name)
        start_frame, end_frame, effective_offset = self._trial_clip(probe.fps, time_offset, probe.nframes)
        decode = self._decode_path(video_path)
        try:
            view.set_video(
                decode,
                fps=probe.fps,
                time_offset=effective_offset,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        except (OSError, ValueError) as e:
            notify(f"Camera '{camera_name}' failed to load: {e}", "warning")
            return
        view.camera_name = camera_name
        view.source_video_path = video_path
        view.decode_video_path = decode
        view.source_size = (probe.width, probe.height)
        view.set_crop(self._camera_crops.get(camera_name))
        self.refresh_view_title(view)

    def _sync_widget_to_current_time(self, view: CameraView):
        video = getattr(self.app_state, "video", None)
        if video is not None:
            resolved = self.app_state.from_display(video.frame_to_time(video.current_frame))
            view.seek_to_time(resolved[1] if resolved is not None else 0.0)
        else:
            view.seek_video_frame(0)

    def add_image_view(self, image_path: str):
        """Show a still image (.png/.jpg …) as a static camera-like view.

        The view lives in its own closable shell dock like any extra camera;
        duplicates are allowed. The pose overlay (primary camera's pose) is
        attached separately by :class:`PoseDisplayManager`.
        """
        import imageio.v3 as iio

        try:
            img = iio.imread(image_path)
        except (OSError, ValueError) as e:
            notify(f"Image '{Path(image_path).name}' could not be loaded: {e}", "warning")
            return None
        view = self.video_area.add_extra(Path(image_path).stem)
        view.set_static_image(img)
        view.static_image_path = str(image_path)
        self.refresh_view_title(view)
        return view

    def image_views(self) -> list[CameraView]:
        """Every static-image view (primary included, if it shows an image)."""
        views = [view for view in self.extra_widgets.values() if getattr(view, "static_image_path", None)]
        if getattr(self.primary_view, "static_image_path", None):
            views.insert(0, self.primary_view)
        return views

    def remove_camera(self, camera_name: str):
        """Remove every view of *camera_name* (duplicates included)."""
        for key, view in list(self.video_area.extras.items()):
            if getattr(view, "camera_name", key) == camera_name:
                self.video_area.remove_extra(key)

    def remove_all_cameras(self):
        for name in list(self.extra_widgets.keys()):
            self.video_area.remove_extra(name)

    def _store_camera_fps_in_session(self, camera_name: str, fps: float):
        sio = getattr(self.app_state, "nwb_alignment", None)
        if sio is None:
            return
        sio.set_stream_rate(fps, "video", camera_name)

    def cleanup(self):
        self.proxy_mgr.cancel_all()
        if getattr(self.app_state, "video", None):
            self.app_state.video.stop()
            self.app_state.video = None
        self._cleanup_primary_video()
        self.video_area._suppress_removed_signal = True
        try:
            self.remove_all_cameras()
        finally:
            self.video_area._suppress_removed_signal = False

    @staticmethod
    def open_readers_parallel(paths: dict[str, str]) -> dict[str, VideoProbe]:
        """Probe video metadata for *paths* concurrently.

        Returns ``{camera_name: VideoProbe}`` for every path that probed
        successfully. Failed probes are silently skipped.
        """

        def _probe(video_path: str) -> VideoProbe | None:
            try:
                return probe_video(video_path)
            except Exception:
                return None

        if not paths:
            return {}

        results: dict[str, VideoProbe] = {}
        with ThreadPoolExecutor(max_workers=len(paths)) as pool:
            futures = {name: pool.submit(_probe, path) for name, path in paths.items()}
            for name, future in futures.items():
                probe = future.result()
                if probe is not None:
                    results[name] = probe
        return results

    def _resolve_video_path(self, camera_name: str, video_folder: str | None) -> str | None:
        if is_url(camera_name):
            return camera_name
        return self.app_state.nwb_alignment.resolve_media_path(
            self.app_state.trials_sel,
            "video",
            device=camera_name,
            fallback_folder=video_folder,
        )
