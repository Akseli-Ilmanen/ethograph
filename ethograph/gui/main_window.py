"""Standalone ethograph main window (napari-free shell).

Layout
------
- **Central area** — :class:`~ethograph.gui.video_manager.VideoArea`: primary
  pygfx camera view + optional extra camera stack.
- **Bottom dock** — the synced plots (``UnifiedPanelContainer``).
- **Right dock** — the control sidebar (``MetaWidget``), collapsible via the
  toolbar button.
- Panels are added via the add-panel popup (bottom bar ➕ button or Shift+N):
  drag a source onto the plot area, or press Enter for default placement.

Layout persistence (no JSON files): ALL layout — the plot-panel layout and
the shell dock arrangement (``shell_dock_state_b64``: space-plot / camera
dock positions) — lives in ``app_state.panel_layout`` → the dataset's
``.ethograph/local_settings.yaml``, so it travels with the dataset.
``app_state.window_state`` → ``gui_settings.yaml`` holds only machine-local
window prefs: geometry (this screen). Both are auto-saved (10 s timer +
close) and restored automatically — there are no save/load layout actions.
The right sidebar always starts visible (its visibility is not persisted).
"""

from __future__ import annotations

import base64
import logging

import numpy as np
from qtpy.QtCore import QByteArray, Qt, QTimer
from qtpy.QtGui import QAction, QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
    QScrollArea,
    QWidget,
)

from .notify import set_toast_host
from .shortcuts import typing_in_text_field
from .video_manager import VideoArea

logger = logging.getLogger(__name__)

_DOCK_AREAS = {
    "left": Qt.LeftDockWidgetArea,
    "right": Qt.RightDockWidgetArea,
    "top": Qt.TopDockWidgetArea,
    "bottom": Qt.BottomDockWidgetArea,
}

# Bump when the dock structure changes so stale saved layouts are ignored.
_LAYOUT_VERSION = 2


class EthographMainWindow(QMainWindow):
    """Top-level window hosting video, plots, and the control sidebar."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ethograph")
        self.setObjectName("EthographMainWindow")
        self.resize(1400, 900)
        self.setDockNestingEnabled(True)

        self._apply_corner_ownership()

        # Video lives in a top dock (added in attach_meta_widget); the plot
        # container becomes the central widget. Do NOT set video_area as the
        # central widget here — setCentralWidget() deletes the previous central
        # widget, which would destroy the CameraView C++ object.
        self.video_area = VideoArea()
        # Extra camera views are created as per-view docks on this shell.
        self.video_area.shell = self

        set_toast_host(self)

        self.meta_widget = None  # set by attach_meta_widget
        self._sidebar_dock: QDockWidget | None = None
        self._plot_dock: QDockWidget | None = None
        self._shortcuts: list[QShortcut] = []
        self._guarded_shortcuts: list[QShortcut] = []
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._sync_guarded_shortcuts)
        self._extra_lineplot_count = 0
        self._window_state_restored = False
        self._pending_dock_state_b64: str | None = None
        self._video_dock_enabled = True
        # Folders / .npy files dropped anywhere on the window become video features.
        self.setAcceptDrops(True)

        self._create_menus()

    # ------------------------------------------------------------------
    # Drops: video features from files
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and self.meta_widget is not None:
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls() and self.meta_widget is not None:
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
        data_widget = getattr(self.meta_widget, "data_widget", None)
        if not paths or data_widget is None:
            return
        event.acceptProposedAction()
        data_widget.add_video_features(paths)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def attach_meta_widget(self, meta_widget) -> None:
        """Dock the sidebar (MetaWidget) and the plot container."""
        self.meta_widget = meta_widget

        scroll = QScrollArea()
        scroll.setWidget(meta_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._sidebar_dock = self.add_dock_widget(scroll, area="right", name="ethograph GUI")
        self._sidebar_dock.setObjectName("SidebarDock")

        # Plots are the central (expanding) widget; the video sits in a compact
        # top dock so it no longer dominates the window (pynaviz-style stack).
        self.setCentralWidget(meta_widget.plot_container)
        # Titled after the camera it shows once one is loaded (see
        # VideoManager.refresh_view_title) — never a generic "Video".
        self._video_dock = self.add_dock_widget(self.video_area, area="top", name="Camera")
        self._video_dock.setObjectName("VideoDock")
        # The dock's ✕ must tear the video down like an extra's close does
        # (VideoArea.eventFilter): hiding alone left a live plot, decode
        # worker and canvas behind an invisible dock.
        self._video_dock._is_primary_video_dock = True
        self._video_dock.installEventFilter(self.video_area)
        QTimer.singleShot(0, lambda: self.resizeDocks([self._video_dock], [300], Qt.Vertical))

        # Bottom playback bar — no dock title bar (an empty widget removes the
        # "Playback" text and the line it occupies).
        from .widgets_bottom_bar import BottomBarScrollHost, BottomPlaybackBar

        bottom_bar = BottomPlaybackBar(meta_widget.app_state)
        # Docked inside a scrollable host: the bar's own width would otherwise
        # become the window's minimum width, leaving no slack to drag the
        # sidebar separator on small screens.
        self._bottom_bar_host = BottomBarScrollHost(bottom_bar)
        self._bottom_bar_dock = self.add_dock_widget(self._bottom_bar_host, area="bottom", name="Playback")
        self._bottom_bar_dock.setObjectName("BottomBarDock")
        self._bottom_bar_dock.setTitleBarWidget(QWidget())
        self._bottom_bar_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.bottom_bar = bottom_bar

        # Add-panel popup: opened from the bottom bar's ➕ button (or Shift+N),
        # anchored above the button.
        bottom_bar.add_panel_btn.clicked.connect(lambda: meta_widget.show_source_popup(bottom_bar.add_panel_btn))

        # Wire bottom bar to data_widget and video sync
        data_widget = getattr(meta_widget, "data_widget", None)
        if data_widget is not None and hasattr(self, "bottom_bar"):
            self.bottom_bar.set_data_widget(data_widget)

            # Connect video sync when it becomes available
            def _wire_video_sync():
                if data_widget.video_mgr and hasattr(data_widget.video_mgr, "video_sync"):
                    self.bottom_bar.connect_video_sync(data_widget.video_mgr.video_sync)

            QTimer.singleShot(100, _wire_video_sync)

        # Build the reorganised top menu bar now that the sidebar sections exist.
        from .top_bar import build_menu_bar

        build_menu_bar(self)

        # Clicking the video shows the pose + playback context in the sidebar.
        if hasattr(self.video_area, "clicked") and hasattr(meta_widget, "focus_video_context"):
            self.video_area.clicked.connect(meta_widget.focus_video_context)

        self._sidebar_toggle.setChecked(True)
        # Geometry now; the dock arrangement comes from the dataset's
        # panel_layout via apply_dock_state_b64 (applied at show — see there
        # for why it must not run while hidden).
        QTimer.singleShot(0, self._restore_window_geometry)

    def _apply_corner_ownership(self):
        """The right control sidebar spans the full window height; the
        video/plots docks sit beside it. Re-applied after any state restore,
        which can otherwise reset corner ownership."""
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

    def add_dock_widget(
        self, widget: QWidget, area: str = "right", name: str = "", object_name: str | None = None
    ) -> QDockWidget:
        dock = QDockWidget(name, self)
        dock.setWidget(widget)
        dock.setObjectName(object_name or name or widget.__class__.__name__)
        # Standard Qt handling for docks created after restoreState(): try the
        # saved placeholder first, fall back to the default area. This must
        # happen INSTEAD of addDockWidget — restoreDockWidget silently fails
        # on a dock that is already in the layout.
        dock.restored_from_state = bool(object_name) and self.restoreDockWidget(dock)
        if dock.restored_from_state:
            # Creation means the panel is open: a placeholder saved closed or
            # unplaced (e.g. by an earlier layout bug) must not resurrect the
            # dock hidden or outside every dock area.
            if not dock.isFloating() and self.dockWidgetArea(dock) == Qt.NoDockWidgetArea:
                dock.restored_from_state = False
            elif dock.isHidden():
                dock.show()
        if not dock.restored_from_state:
            self.addDockWidget(_DOCK_AREAS.get(area, Qt.RightDockWidgetArea), dock)
        return dock

    def _create_menus(self):
        """Create window-level shortcut actions.

        The visible menu bar (File / Changepoints / Neural / Help) is
        built by :func:`ethograph.gui.top_bar.build_menu_bar` once the sidebar
        sections exist.  Here we only register the standalone actions whose
        keyboard shortcuts must work regardless of the menu bar.
        """
        self._sidebar_toggle = QAction("Show sidebar", self, checkable=True, checked=True)
        self._sidebar_toggle.toggled.connect(self._set_sidebar_visible)
        self.addAction(self._sidebar_toggle)

        self._video_toggle = QAction("Show video", self, checkable=True, checked=True)
        self._video_toggle.toggled.connect(self.set_video_viewer_visible)
        self.addAction(self._video_toggle)

        self._zen_mode = False

    # ------------------------------------------------------------------
    # Sidebar / video visibility
    # ------------------------------------------------------------------

    def _set_sidebar_visible(self, visible: bool):
        if self._sidebar_dock is not None:
            self._sidebar_dock.setVisible(visible)
        self._sync_sidebar_button(visible)

    def _sync_sidebar_button(self, visible: bool):
        """Mirror sidebar visibility onto the menu-bar corner button."""
        btn = getattr(self, "_sidebar_corner_btn", None)
        if btn is not None and btn.isChecked() != visible:
            btn.blockSignals(True)
            btn.setChecked(visible)
            btn.blockSignals(False)

    def toggle_sidebar(self):
        self._sidebar_toggle.setChecked(not self._sidebar_toggle.isChecked())

    def set_video_viewer_visible(self, visible: bool):
        # Video now lives in its own top dock.
        dock = getattr(self, "_video_dock", None)
        if dock is not None:
            dock.setVisible(visible)

    def set_zen_mode(self, on: bool):
        """Hide the right sidebar (zen mode). Sidebar updates are skipped.

        Toggled via ``Shift+Z``.  While on, the right control sidebar is
        hidden and ``app_state.zen_mode`` is set so widgets can skip
        expensive refreshes.
        """
        self._zen_mode = on
        if self._sidebar_dock is not None:
            self._sidebar_dock.setVisible(not on)
        self._sync_sidebar_button(not on)
        if self.meta_widget is not None:
            app_state = getattr(self.meta_widget, "app_state", None)
            if app_state is not None and hasattr(app_state, "zen_mode"):
                app_state.zen_mode = on

    # ------------------------------------------------------------------
    # napari-Viewer-replacement services
    # ------------------------------------------------------------------

    def canvas_widget(self) -> QWidget | None:
        """The primary video canvas widget (for Qt overlays)."""
        return self.video_area.primary.canvas_widget() or self.video_area.primary

    def screenshot(self, canvas_only: bool = True, flash: bool = False) -> np.ndarray:
        """Grab the video canvas (or whole window) as an RGBA uint8 array."""
        target = self.video_area if canvas_only else self
        pixmap = target.grab()
        image = pixmap.toImage()
        image = image.convertToFormat(image.Format.Format_RGBA8888)
        w, h = image.width(), image.height()
        ptr = image.constBits()
        try:
            ptr.setsize(h * w * 4)  # PyQt
        except AttributeError:
            pass  # PySide memoryview already sized
        return np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()

    def bind_shortcut(self, key_sequence: str, callback, guarded: bool = False) -> QShortcut:
        """Bind an application-wide shortcut.

        *guarded* shortcuts (plain letters, arrow keys) are **disabled** while
        the user types in a text field rather than firing a no-op callback: an
        enabled QShortcut consumes the key press before the focus widget sees
        it, which left arrow keys dead in every input — including the ↑/↓
        selection walk in the add-panel popup's filter box.
        """
        shortcut = QShortcut(QKeySequence(key_sequence), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)
        if guarded:
            self._guarded_shortcuts.append(shortcut)
            shortcut.setEnabled(not typing_in_text_field())
        return shortcut

    def _sync_guarded_shortcuts(self, *_args):
        typing = typing_in_text_field()
        for shortcut in self._guarded_shortcuts:
            shortcut.setEnabled(not typing)

    def clear_shortcuts(self):
        for shortcut in self._shortcuts:
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._shortcuts = []
        self._guarded_shortcuts = []

    # ------------------------------------------------------------------
    # Window-state persistence (via app_state → gui_settings.yaml; no JSON)
    # ------------------------------------------------------------------

    def capture_window_state(self) -> dict:
        """Machine-specific window prefs only: geometry (where the window sits
        on THIS screen). ALL layout — panels, shell dock arrangement — is
        per-dataset state in app_state.panel_layout →
        .ethograph/local_settings.yaml. Sidebar visibility is deliberately
        NOT persisted: the sidebar always starts visible."""
        return {
            "version": _LAYOUT_VERSION,
            "geometry_b64": base64.b64encode(bytes(self.saveGeometry())).decode("ascii"),
        }

    def capture_dock_state_b64(self) -> str:
        """The shell's dock arrangement as a portable base64 saveState blob.
        Stored per dataset (panel_layout → local_settings.yaml), so space-plot
        / camera dock positions travel with the dataset across machines."""
        return base64.b64encode(bytes(self.saveState(version=_LAYOUT_VERSION))).decode("ascii")

    def _saved_window_state(self) -> dict | None:
        payload = getattr(getattr(self.meta_widget, "app_state", None), "window_state", None)
        if not payload:
            return None
        # Ignore state saved by an incompatible (older) window structure — it
        # would clobber the new video-top-dock / full-height-sidebar arrangement.
        if payload.get("version") != _LAYOUT_VERSION:
            logger.info("Ignoring stale window layout (version mismatch).")
            return None
        return payload

    def _restore_window_geometry(self):
        """Restore the outer window geometry at startup (size/position only —
        no dock layout involved, so it is safe on the hidden window)."""
        payload = self._saved_window_state()
        if not payload:
            return
        try:
            self.restoreGeometry(QByteArray.fromBase64(payload["geometry_b64"].encode("ascii")))
        except (KeyError, Exception) as e:
            logger.warning("Could not restore window geometry: %s", e)

    def _restore_window_prefs(self):
        """Enforce startup window prefs on first show. The sidebar is always
        shown at startup regardless of how the last session ended (geometry
        was already restored at startup)."""
        self._sidebar_toggle.setChecked(True)
        self._set_sidebar_visible(True)

    def apply_dock_state_b64(self, blob: str) -> None:
        """Apply a dataset's dock-arrangement blob (panel_layout →
        local_settings.yaml). Deferred to the next show when hidden:
        restoreState on a hidden window leaves a pending state that Qt
        applies at the first show, evicting — or crashing on — every dock
        created in between (GL docks like space plots and extra camera views
        ended up squished at the top-left corner). Applying while visible is
        synchronous and matches docks by objectName as real widgets."""
        if not self.isVisible():
            self._pending_dock_state_b64 = blob
            return
        self._apply_dock_state(blob)

    def _apply_dock_state(self, blob: str | None) -> None:
        if not blob:
            return
        visible_before = [d for d in self.findChildren(QDockWidget) if not d.isHidden()]
        try:
            self.restoreState(QByteArray.fromBase64(blob.encode("ascii")), _LAYOUT_VERSION)
        except Exception as e:
            logger.warning("Could not restore dock state: %s", e)
        # The blob dictates placement, the session dictates existence: a dock
        # the load created open must not come back hidden (restoreState
        # sporadically restores one as hidden).
        for dock in visible_before:
            if dock.isHidden():
                dock.show()
        # Same principle in reverse: a session without a primary video must
        # not get its Video dock back from a blob saved with one.
        if not self._video_dock_enabled and getattr(self, "_video_dock", None) is not None:
            self._video_dock.hide()
        # restoreState can reset corner ownership — re-assert full-height sidebars.
        self._apply_corner_ownership()

    def set_video_dock_visible(self, visible: bool) -> None:
        """Show/hide the primary Video dock.

        Sessions without a primary video (or static image) get no video panel
        slot at all — the dock's existence follows the data, like every other
        panel."""
        self._video_dock_enabled = visible
        dock = getattr(self, "_video_dock", None)
        if dock is not None:
            dock.setVisible(visible)

    def set_video_dock_title(self, title: str) -> None:
        """Re-title the primary camera dock (``"cam-1 (front.mp4)"``).

        Only the visible title changes — the objectName that keys layout
        persistence stays ``VideoDock``."""
        dock = getattr(self, "_video_dock", None)
        if dock is not None:
            dock.setWindowTitle(title)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._window_state_restored:
            self._window_state_restored = True
            self._restore_window_prefs()
        if self._pending_dock_state_b64:
            blob, self._pending_dock_state_b64 = self._pending_dock_state_b64, None
            self._apply_dock_state(blob)

    # ------------------------------------------------------------------
    # Close handling
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.meta_widget is not None:
            if not self.meta_widget._check_unsaved_changes(event):
                return
            self.meta_widget.flush_pending_writes()
            # Final layout snapshot: window state → gui_settings.yaml,
            # panel layout → the dataset's local_settings.yaml.
            self.meta_widget.app_state.save_to_yaml()
            if hasattr(self.meta_widget.app_state, "stop_auto_save"):
                self.meta_widget.app_state.stop_auto_save()
            data_widget = getattr(self.meta_widget, "data_widget", None)
            if data_widget is not None:
                if getattr(data_widget, "video_mgr", None) is not None:
                    data_widget.video_mgr.cleanup()
                data_widget.cleanup()
        set_toast_host(None)
        super().closeEvent(event)
