"""Unified flexible panel container for all layout scenarios.

Every panel is a QDockWidget inside a nested QMainWindow (pynaviz-style), so
panels can be arranged freely: side by side, stacked, tabbed, or floated.
The default arrangement is a vertical stack in ``_PANEL_ORDER`` with line
plots at the bottom; drag a panel's title bar to rearrange.

Panels (every one optional):
  - Dynamic panels (audiotrace, spectrogram, lineplot, heatmap): any number,
    ALL equal instances created via the generic :meth:`add_panel` and removed
    via each panel's ✕ / :meth:`remove_panel`. Adding NEVER dedups — what
    already exists doesn't matter, every call creates another instance.
    Audio instances may pin their own mic/channel; feature instances render
    from their own ``panel_state``.
  - EphysTrace / Raster / Neo (fixed singletons, only if neural data)
"""

import base64
from typing import Any, Dict

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QByteArray, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QActionGroup, QCursor
from qtpy.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.io.time_model import TimeRange

from ..io.plot_sources import build_audio_source
from .app_constants import (
    ENVELOPE_OVERLAY_COLOR,
    ENVELOPE_OVERLAY_DEBOUNCE_MS,
    ENVELOPE_OVERLAY_WIDTH,
    PANEL_MIN_HEIGHT,
    PLOT_CONTAINER_SIZE_HINT_HEIGHT,
)
from .audio_player import AudioPlayer
from .label_drawing_mixin import LabelDrawingMixin
from .plots_audiotrace import AudioTracePlot
from .plots_base import ThrottleDebounce, right_gutter_width
from .plots_console import ConsolePanel
from .plots_ephystrace import EphysTracePlot
from .plots_heatmap import HeatmapPlot
from .plots_labelribbon import LabelRibbonPlot
from .plots_lineplot import LinePlot
from .plots_overlay import OverlayManager
from .plots_raster import RasterPlot
from .plots_spectrogram import SharedAudioCache, SpectrogramPlot
from .widgets_transform import compute_energy_envelope

#: Overlay-name prefix of the per-class onset-model probability curves.
ONSET_CURVE_PREFIX = "onset_curve:"

# Panel size ratios keyed by (has_audio, has_neurons_or_neo)
# Values: dict mapping panel_name -> fraction of splitter height
_PANEL_RATIOS = {
    # audio + ephys
    (True, True): {
        "audiotrace": 0.10,
        "spectrogram": 0.15,
        "neo": 0.15,
        "ephys": 0.20,
        "raster": 0.10,
        "feature": 0.30,
    },
    # audio only
    (True, False): {"audiotrace": 0.20, "spectrogram": 0.30, "feature": 0.50},
    # ephys only
    (False, True): {"neo": 0.20, "ephys": 0.30, "raster": 0.15, "feature": 0.35},
    # nothing extra
    (False, False): {"feature": 1.0},
}

# Ordered list of (panel_name, app_state_guard_attr | None) for the fixed
# singleton panels. Dynamic panels (lineplot/heatmap/audiotrace/spectrogram)
# are not listed: they are instances managed by add_panel/remove_panel.
# guard_attr: app_state boolean that must be True for the panel to appear; None = always allowed
_PANEL_ORDER = [
    ("ephys", "has_neurons"),
    ("raster", "has_neurons"),
]

# Maps fixed panel name -> widget attribute name on the container
_PANEL_PLOT_ATTR = {
    "ephys": "ephys_trace_plot",
    "raster": "raster_plot",
}

# The generic dynamic-panel table: ONE mechanism for every user-addable panel
# type. General rule: adding a panel ALWAYS creates a new instance — what
# already exists never matters (no dedup, duplicates welcome); the user
# removes extras via each panel's ✕. Group semantics:
#   "audio"   — stacked at the top of the default layout, wired to an audio
#               source (may pin a mic/channel via ``plot.mic_name``)
#   "feature" — stacked at the bottom, renders from its own ``panel_state``
# ``overlay_rescale`` connects the OverlayManager y-rescale hook.
_DYNAMIC_PANEL_SPECS = {
    "audiotrace": {"cls": AudioTracePlot, "group": "audio", "overlay_rescale": True},
    "spectrogram": {"cls": SpectrogramPlot, "group": "audio", "overlay_rescale": False},
    "lineplot": {"cls": LinePlot, "group": "feature", "overlay_rescale": True},
    "heatmap": {"cls": HeatmapPlot, "group": "feature", "overlay_rescale": False},
    # Neo trace: one instance per stream/modality (EMG, accelerometer, …),
    # each showing a chosen channel subset. Configured by DataWidget via the
    # ``configure_neo_plot`` callback (needs ephys_source_map + load_ephys).
    "neo": {"cls": EphysTracePlot, "group": "neo", "overlay_rescale": True},
    # Label timeline: an empty axis carrying only the label overlay, for a
    # session (video-only) that would otherwise have no panel to show labels on.
    "labels": {"cls": LabelRibbonPlot, "group": "labels", "overlay_rescale": False},
}

#: Share of the dock host a label timeline takes by default — a ribbon, not a plot.
_LABEL_RIBBON_RATIO = 0.08


class CurrentLabelIndicator(QLabel):
    """Floating badge showing the label name + color at the current time position."""

    _MARGIN = 8
    _PAD_H = 10
    _PAD_V = 4

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.hide()

    def update_label(self, name: str, color_rgb: tuple | list | None):
        if not name:
            self.hide()
            return
        self._apply_style(name, color_rgb)
        self.show()
        self.raise_()
        self._reposition()

    def _apply_style(self, text: str, color_rgb: tuple | list | None):
        if color_rgb is not None:
            scale = max(color_rgb[:3]) <= 1.0
            r, g, b = (int(c * 255) if scale else int(c) for c in color_rgb[:3])
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            fg = "#000" if lum > 140 else "#fff"
            bg = f"rgb({r},{g},{b})"
        else:
            fg, bg = "#000", "rgba(255,255,255,200)"
        self.setText(text)
        self.setStyleSheet(
            f"QLabel {{ color: {fg}; background: {bg}; border: 1px solid #555;"
            f" border-radius: 4px; padding: {self._PAD_V}px {self._PAD_H}px;"
            f" font-size: 13px; font-weight: bold; }}"
        )
        self.adjustSize()

    def _reposition(self):
        p = self.parent()
        if p is None:
            return
        x = p.width() - self.width() - self._MARGIN
        self.move(max(0, x), self._MARGIN)


def add_pin_choices(menu: QMenu, names: list[str], pinned: str | None, sidebar: str | None, choose) -> None:
    """One radio choice for a panel's individual: follow the sidebar, or one name.

    Exactly one entry is checked, so the menu itself says the two are
    alternatives. *choose* is called with ``None`` (follow) or a name.
    """
    group = QActionGroup(menu)
    group.setExclusive(True)
    follow = menu.addAction(f"Follow sidebar ({sidebar})" if sidebar else "Follow sidebar")
    follow.setCheckable(True)
    follow.setChecked(pinned is None)
    follow.triggered.connect(lambda _=False: choose(None))
    group.addAction(follow)
    menu.addSeparator()
    for name in names:
        action = menu.addAction(name)
        action.setCheckable(True)
        action.setChecked(pinned == name)
        action.triggered.connect(lambda _=False, n=name: choose(n))
        group.addAction(action)


class _PanelDockTitleBar(QWidget):
    """Slim dock title bar: drag handle + panel name + move (⠿) + close (✕)."""

    _BTN_STYLE = (
        "QPushButton {{ color:#ddd; background:rgba(40,40,40,160); border:none;"
        " border-radius:3px; font-size:9px; }}"
        "QPushButton:hover {{ color:#fff; background:{hover}; }}"
    )

    def __init__(self, dock: QDockWidget, title: str, on_close, on_move):
        super().__init__(dock)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 4, 1)
        layout.setSpacing(4)

        self._label = QLabel(title)
        self._label.setStyleSheet("color: rgba(255,255,255,130); font-size: 8pt;")
        layout.addWidget(self._label)
        layout.addStretch()

        move_btn = QPushButton("⠿")
        move_btn.setObjectName("panel_move_btn")
        move_btn.setFixedSize(14, 14)
        move_btn.setToolTip("Move this panel next to another panel…")
        move_btn.setStyleSheet(self._BTN_STYLE.format(hover="rgba(80,120,200,200)"))
        move_btn.clicked.connect(on_move)
        layout.addWidget(move_btn)

        close_btn = QPushButton("✕")
        close_btn.setObjectName("panel_close_btn")
        close_btn.setFixedSize(14, 14)
        close_btn.setToolTip("Remove this panel")
        close_btn.setStyleSheet(self._BTN_STYLE.format(hover="rgba(200,60,60,200)"))
        close_btn.clicked.connect(on_close)
        layout.addWidget(close_btn)
        self.setFixedHeight(17)

    def title(self) -> str:
        return self._label.text()

    def set_title(self, text: str):
        self._label.setText(str(text))


class UnifiedPanelContainer(LabelDrawingMixin, QWidget):
    """Unified container with dynamic panel visibility.

    All panels share the same x-axis via pyqtgraph linking.
    Labels, changepoints, and time markers are drawn on all visible panels.
    """

    plot_changed = Signal(str)
    labels_redraw_needed = Signal()
    spectrogram_overlay_shown = Signal()
    time_marker_updated = Signal(float)
    #: Display-clock time under the cursor while a state label is half-placed
    #: (between its first and second click).
    pending_label_hovered = Signal(float)
    #: Emitted with the plot widget whenever a dynamic panel (line plot or
    #: audio panel) is created.
    panel_added = Signal(object)
    #: Emitted with a feature panel whenever what it shows may have changed —
    #: it was clicked (including while already active) or the sidebar edited
    #: its feature/selections. ``active_changed`` announces clicks but not
    #: sidebar edits, so anything that must track a panel's current contents
    #: (the console) listens here instead — and ONLY here, since a click can
    #: fire both signals.
    panel_content_changed = Signal(object)
    #: Relays bufferUpdated from every spectrogram instance (auto-levels).
    spectrogram_buffer_updated = Signal()

    #: MIME type used by the add-panel popup's drag-and-drop panel creator.
    SOURCE_MIME = "application/x-ethograph-source"

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._data_widget = None  # set by widgets_meta after construction

        # Drag-and-drop panel creation: the add-panel popup drags a
        # Media/Feature source onto this container; ``_on_source_drop(kind,
        # name)`` (set by MetaWidget) opens the plot-type picker and creates
        # a panel.
        self._on_source_drop = None
        self.setAcceptDrops(True)

        # Coalesces deferred label redraws (see schedule_labels_redraw).
        self._labels_redraw_scheduled = False
        # Coalesces deferred x-axis alignment (see schedule_axis_align).
        self._axis_align_scheduled = False
        # Session basis draws only the labels intersecting the viewport, so a
        # zoom/pan must re-derive that set — debounced, label redraws are not
        # per-scroll-tick cheap.
        self._labels_zoom_timer = QTimer(self)
        self._labels_zoom_timer.setSingleShot(True)
        self._labels_zoom_timer.setInterval(350)
        self._labels_zoom_timer.timeout.connect(self.schedule_labels_redraw)

        # --- Plots ---
        # Fixed singleton panels; everything else (lineplot/heatmap/
        # audiotrace/spectrogram) is a dynamic instance (all equal,
        # duplicates allowed) managed by add_panel/remove_panel.
        self.ephys_trace_plot = EphysTracePlot(app_state)  # Phy-Viewer panel
        self.raster_plot = RasterPlot(app_state)

        #: All dynamic panel instances in creation order; each carries
        #: ``panel_type`` and ``panel_group`` attributes set by add_panel.
        self._dyn_panels: list = []
        self._dyn_docks: dict = {}
        self._dyn_counter = 0
        # The console is the one panel type that is a singleton (see
        # add_console_panel) and is not a plot, so it stays out of _dyn_panels.
        self.console_panel = None
        self._console_dock = None
        # Hidden stand-in so get_current_plot() never returns None when no
        # feature panel exists (audio-/video-only sessions). Created lazily.
        self._fallback = None

        # The feature plot the right sidebar currently controls (last clicked).
        self.active_feature_plot = None

        # --- Panel visibility state (fixed panels; dynamic panels exist or don't) ---
        self._panel_visible: dict[str, bool] = {
            "ephys": False,
            "raster": False,
        }

        # --- Mixin state ---
        self.label_mappings: Dict[int, Dict[str, Any]] = {}
        self.audio_overlay_type = None
        self.audio_cp_items: list = []
        self.osc_event_items: list = []
        self._pending_label_items: list = []
        self._pending_label_regions: list = []
        self._pending_hover_conns: list = []
        self._pending_label_anchor: float | None = None

        self.overlay_manager = OverlayManager()
        #: Overlay names of the onset curves currently drawn, so they can be
        #: taken down again without the manager growing a query API.
        self._onset_curve_names: list[str] = []
        self.ephys_trace_plot.vb.sigYRangeChanged.connect(
            lambda: self.overlay_manager.rescale_for_plot(self.ephys_trace_plot)
        )

        # Envelope throttle+debounce for x-range data refresh
        self._envelope_td = None
        self._envelope_xrange_updater = None
        self._envelope_y_updater = None
        self._envelope_host = None

        # --- Layout ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

        # Dock host: a nested QMainWindow whose docks are the panels
        # (pynaviz-style free arrangement: side-by-side, tabs, floating).
        self._dock_host = QMainWindow()
        self._dock_host.setWindowFlags(Qt.Widget)
        self._dock_host.setDockNestingEnabled(True)
        self._dock_host.setDockOptions(
            QMainWindow.AnimatedDocks | QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks
        )
        # Its layout minimum is the sum of every open panel's minimum, which
        # would propagate outwards and stop the video/plots separator early.
        self._dock_host.setMinimumHeight(PANEL_MIN_HEIGHT)
        main_layout.addWidget(self._dock_host)

        # Audio playback (no-video mode)
        self.audio_player = AudioPlayer(
            app_state,
            get_xlim=self.get_current_xlim,
            get_visible_time=self._get_first_visible_time,
            update_marker=self.update_time_marker_by_time,
        )

        # --- Current-label floating indicator (top-right corner) ---
        self._label_indicator = CurrentLabelIndicator(self)

        # --- X-axis sync: every panel's x-range change is copied verbatim to
        # all other open panels (see _sync_panel_xrange). _xlink_master is the
        # first open panel — kept as the conventional target for programmatic
        # range setting (slider/navigation); any panel drives the rest equally.
        self._xlink_master = None
        self._xsync_guard = False

        # Connect zoom events for changepoint line style updates
        # (dynamic panels connect per instance in add_panel)
        for plot in (
            self.ephys_trace_plot,
            self.raster_plot,
        ):
            plot.vb.sigRangeChanged.connect(self._on_plot_zoom)

        # Bidirectional y-axis sync between ephys trace and raster
        self._syncing_y = False
        self.ephys_trace_plot.vb.sigYRangeChanged.connect(self._sync_raster_y_from_ephys)
        self.raster_plot.y_range_changed.connect(self._sync_ephys_y_from_raster)
        self.ephys_trace_plot.y_space_changed.connect(self._on_ephys_y_space_changed)
        self.ephys_trace_plot.seek_time_requested.connect(self._on_seek_time_requested)

        # Track which panel was last clicked (for changepoint navigation).
        # The green-edge highlight + sidebar context is handled by the
        # ActivePanelManager (set as self.active_panels by MetaWidget).
        self._last_clicked_panel = "feature"
        self.active_panels = None
        self.ephys_trace_plot.plot_clicked.connect(lambda _: setattr(self, "_last_clicked_panel", "ephys"))
        self.raster_plot.plot_clicked.connect(lambda _: setattr(self, "_last_clicked_panel", "raster"))

        # Create a dock per fixed panel, hidden, in the default vertical stack.
        # Dynamic panels get their docks in add_panel().
        self._create_panel_docks()

    def _create_panel_docks(self):
        """One QDockWidget per fixed panel; ✕ hides it (dynamic-panel ✕ removes)."""
        closers = {
            "ephys": lambda: self.set_ephys_visible(False),
            "raster": lambda: self.set_raster_visible(False),
        }
        self._panel_docks: dict[str, QDockWidget] = {}
        prev = None
        for name, _ in _PANEL_ORDER:
            dock = self._make_dock(name, self._get_panel_widget(name), closers[name])
            dock.setObjectName(f"panel_{name}")
            self._panel_docks[name] = dock
            if prev is None:
                self._dock_host.addDockWidget(Qt.LeftDockWidgetArea, dock)
            else:
                self._dock_host.splitDockWidget(prev, dock, Qt.Vertical)
            dock.hide()
            prev = dock

    def _make_dock(self, title: str, widget: QWidget, on_close) -> QDockWidget:
        dock = QDockWidget(title, self._dock_host)
        # An explicit minimum overrides the widget's minimumSizeHint, which for
        # a pyqtgraph view is large enough to block the video/plots separator
        # long before the plots reach ~10% of the window.
        widget.setMinimumHeight(PANEL_MIN_HEIGHT)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setTitleBarWidget(_PanelDockTitleBar(dock, title, on_close, on_move=lambda: self._show_move_menu(dock)))
        return dock

    # ------------------------------------------------------------------
    # Dynamic-panel registry (generic; every type is an instance list)
    # ------------------------------------------------------------------

    def panels_of_type(self, panel_type: str) -> list:
        return [p for p in self._dyn_panels if p.panel_type == panel_type]

    def _panels_of_group(self, group: str) -> list:
        return [p for p in self._dyn_panels if p.panel_group == group]

    @property
    def line_plots(self) -> list:
        return self.panels_of_type("lineplot")

    @property
    def heatmap_plots(self) -> list:
        return self.panels_of_type("heatmap")

    @property
    def audio_trace_plots(self) -> list:
        return self.panels_of_type("audiotrace")

    @property
    def spectrogram_plots(self) -> list:
        return self.panels_of_type("spectrogram")

    @property
    def neo_trace_plots(self) -> list:
        return self.panels_of_type("neo")

    def _audio_plots(self) -> list:
        return self._panels_of_group("audio")

    def _neo_plots(self) -> list:
        return self._panels_of_group("neo")

    def _label_ribbons(self) -> list:
        return self._panels_of_group("labels")

    def has_open_plots(self) -> bool:
        """Whether any time-axis panel is on screen (the console is not one)."""
        return any(True for _ in self._visible_plots())

    @property
    def _fallback_plot(self):
        """Hidden stand-in feature plot: keeps get_current_plot() non-None
        when no feature panel exists (audio-/video-only sessions)."""
        if self._fallback is None:
            plot = LinePlot(self.app_state)
            plot.panel_type = "lineplot"
            plot.panel_group = "feature"
            plot.hide()
            self._fallback = plot
        return self._fallback

    def _open_docks(self) -> list[QDockWidget]:
        docks = [self._dyn_docks[p] for p in self._audio_plots() if not self._dyn_docks[p].isHidden()]
        docks += [self._dyn_docks[p] for p in self._neo_plots() if not self._dyn_docks[p].isHidden()]
        docks += [self._panel_docks[n] for n, _ in _PANEL_ORDER if not self._panel_docks[n].isHidden()]
        docks += [self._dyn_docks[p] for p in self._panels_of_group("feature") if not self._dyn_docks[p].isHidden()]
        docks += [self._dyn_docks[p] for p in self._label_ribbons() if not self._dyn_docks[p].isHidden()]
        return docks

    def _show_move_menu(self, dock: QDockWidget):
        """Click-driven panel placement: pick a target panel and a side."""
        targets = [d for d in self._open_docks() if d is not dock]
        if not targets:
            return
        menu = QMenu(self)

        def _place(target: QDockWidget, orient=None, tab=False):
            dock.setFloating(False)
            if tab:
                self._dock_host.tabifyDockWidget(target, dock)
            else:
                self._dock_host.splitDockWidget(target, dock, orient)
            dock.show()
            dock.raise_()

        for target in targets:
            sub = menu.addMenu(target.titleBarWidget().title())
            sub.addAction("Below", lambda _=False, t=target: _place(t, Qt.Vertical))
            sub.addAction("Right of", lambda _=False, t=target: _place(t, Qt.Horizontal))
            sub.addAction("Tab with", lambda _=False, t=target: _place(t, tab=True))
        self._add_pin_actions(menu, dock)
        menu.exec_(QCursor.pos())

    def _add_pin_actions(self, menu: QMenu, dock: QDockWidget) -> None:
        """Pin / unpin entries for a feature panel's dock, when the dataset has several individuals."""
        plot = next((p for p, d in self._dyn_docks.items() if d is dock), None)
        if plot is None or not hasattr(plot, "set_pinned_individual"):
            return
        names = self.app_state.label_individuals()
        if len(names) < 2:
            return
        menu.addSeparator()
        sub = menu.addMenu("Individual")
        add_pin_choices(
            sub, names, plot.pinned_individual, self.app_state.sidebar_individual(), lambda n: self.pin_panel(plot, n)
        )

    def pin_panel(self, plot, individual: str | None) -> None:
        """Pin *plot* to *individual* (``None`` unpins): it re-renders, re-titles and redraws its labels."""
        plot.set_pinned_individual(individual)
        self.set_panel_title(plot, self.panel_title(plot))
        if self.app_state.ready:
            plot.update_plot()
        self.panel_content_changed.emit(plot)
        self.app_state.refresh_labelling_subject()
        self.schedule_labels_redraw()

    def panel_title(self, plot) -> str:
        """A feature panel's title: its feature, then which individual it shows and why."""
        feature = plot._effective_feature() if hasattr(plot, "_effective_feature") else None
        title = str(feature) if feature else str(getattr(plot, "panel_type", "panel"))
        return title + self.app_state.panel_mode_suffix(plot)

    def refresh_panel_titles(self) -> None:
        """Re-title every feature panel — after the sidebar's individual changed."""
        for plot in self._panels_of_group("feature"):
            self.set_panel_title(plot, self.panel_title(plot))

    def _dock_of(self, plot) -> QDockWidget | None:
        for name, dock in self._panel_docks.items():
            if self._get_panel_widget(name) is plot:
                return dock
        return self._dyn_docks.get(plot)

    def set_panel_title(self, plot, title: str) -> None:
        """Update a panel dock's title (e.g. when its feature changes)."""
        dock = self._dock_of(plot)
        if dock is not None:
            dock.titleBarWidget().set_title(title)

    # ------------------------------------------------------------------
    # Drag-and-drop panel creation (add-panel popup → plot area)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(self.SOURCE_MIME):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(self.SOURCE_MIME):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(self.SOURCE_MIME):
            return
        payload = bytes(event.mimeData().data(self.SOURCE_MIME)).decode("utf-8")
        kind, _, name = payload.partition("|")
        if callable(self._on_source_drop):
            self._on_source_drop(kind, name)
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Generic dynamic-panel creation/removal (all types, all equal)
    # ------------------------------------------------------------------

    def _available_features(self) -> list[str]:
        data_widget = self._data_widget
        catalog = getattr(data_widget, "catalog", None) if data_widget else None
        if catalog is not None:
            choices = catalog.feature_choices()
            if choices:
                return choices
        ds = getattr(self.app_state, "ds", None)
        if ds is not None:
            return list(ds.data_vars)
        return []

    def add_panel(
        self,
        panel_type: str,
        *,
        feature: str | None = None,
        mic_name: str | None = None,
        stream_name: str | None = None,
        channels: list[int] | None = None,
    ):
        """Create a NEW panel instance of any dynamic type ("lineplot",
        "heatmap", "audiotrace", "spectrogram", "neo").

        General rule: what already exists never matters — every call creates
        another instance (duplicates included); the user removes extras via
        each panel's ✕. Feature panels take *feature*; audio panels take
        *mic_name* (an ``audio_source_map`` key; ``None`` follows the global
        Mic combo).
        """
        spec = _DYNAMIC_PANEL_SPECS[panel_type]
        group = spec["group"]
        if panel_type == "lineplot" and not self._available_features():
            return None

        plot = spec["cls"](self.app_state)
        plot.panel_type = panel_type
        plot.panel_group = group

        if group == "feature":
            if feature in self._available_features():
                plot.set_panel_control("features", feature)
            title = feature or panel_type
        elif group == "neo":
            plot.neo_stream_name = stream_name
            plot.neo_channels = list(channels) if channels is not None else None
            title = f"Neo — {stream_name}" if stream_name else "Neo"
        elif group == "labels":
            title = "Labels"
        else:
            plot.mic_name = mic_name
            title = f"{panel_type} — {mic_name}" if mic_name else panel_type

        self._dyn_counter += 1
        dock = self._make_dock(title, plot, lambda: self.remove_panel(plot))
        dock.setObjectName(f"panel_{panel_type}_{self._dyn_counter}")
        self._dyn_docks[plot] = dock
        anchor = self._anchor_dock_for_group(group)
        if anchor is None:
            self._dock_host.addDockWidget(Qt.LeftDockWidgetArea, dock)
        else:
            self._dock_host.splitDockWidget(anchor, dock, Qt.Vertical)
        dock.show()
        self._dyn_panels.append(plot)

        plot.vb.sigRangeChanged.connect(self._on_plot_zoom)
        if spec["overlay_rescale"]:
            plot.vb.sigYRangeChanged.connect(lambda *_, p=plot: self.overlay_manager.rescale_for_plot(p))
        if panel_type == "spectrogram":
            plot.bufferUpdated.connect(self.spectrogram_buffer_updated)
        clicked_key = {"feature": "feature", "neo": "neo", "labels": "feature"}.get(group, "audio")
        plot.plot_clicked.connect(lambda _: setattr(self, "_last_clicked_panel", clicked_key))
        if group == "feature":
            plot.plot_clicked.connect(lambda _, p=plot: self.panel_content_changed.emit(p))
        # Register with the active-panel manager so it highlights + shows controls.
        if self.active_panels is not None:
            self.active_panels.register(
                plot,
                panel_type,
                clicked_signal=plot.plot_clicked,
                plot=plot if group == "feature" else None,
            )
        if group == "feature" and self.active_feature_plot is None:
            self.active_feature_plot = plot

        self._update_panel_visibility()

        if group == "audio":
            source = build_audio_source(self.app_state, plot.mic_name)
            plot.set_source(source)
            if self.app_state.ready and source is not None:
                t0, t1 = self.get_current_xlim()
                plot.update_plot(t0=t0, t1=t1)
        elif group == "neo":
            dw = self._data_widget
            if dw is not None and hasattr(dw, "configure_neo_plot"):
                dw.configure_neo_plot(plot)
        elif self.app_state.ready:
            plot.update_plot()
        self.panel_added.emit(plot)
        self.schedule_labels_redraw()
        return plot

    def remove_panel(self, plot) -> None:
        """Remove any dynamic panel instance (they are all removable)."""
        if plot not in self._dyn_panels:
            return
        if self.active_panels is not None:
            self.active_panels.unregister(plot)
        # The throttle/debounce QTimers are not parented to the widget — stop
        # them so no callback fires into the deleted plot. Stopping alone is
        # not enough: the plot's own viewbox range signal would re-arm the
        # debounce (via _on_view_range_changed) between deleteLater() and the
        # actual C++ deletion, firing _do_range_update on a dead object. So
        # first sever that self-retrigger, then stop the timers.
        handler = getattr(plot, "_on_view_range_changed", None)
        if handler is not None:
            for sig_name in ("sigRangeChanged", "sigXRangeChanged"):
                sig = getattr(plot.vb, sig_name, None)
                if sig is not None:
                    try:
                        sig.disconnect(handler)
                    except (TypeError, RuntimeError):
                        pass
        if getattr(plot, "_xsync_connected", False):
            plot._xsync_connected = False
            try:
                plot.plotItem.vb.sigXRangeChanged.disconnect(self._sync_panel_xrange)
            except (TypeError, RuntimeError):
                pass
        # A panel that renders nothing (the label timeline) has no debounce.
        debounce = getattr(plot, "_td", None)
        if debounce is not None:
            debounce.stop()
        if plot.panel_group in ("audio", "neo"):
            plot.set_source(None)
        self._dyn_panels.remove(plot)
        dock = self._dyn_docks.pop(plot, None)
        if self.active_feature_plot is plot:
            feature_panels = self._panels_of_group("feature")
            self.active_feature_plot = feature_panels[0] if feature_panels else None
        if dock is not None:
            self._dock_host.removeDockWidget(dock)
            dock.deleteLater()
        else:
            plot.setParent(None)
            plot.deleteLater()
        self._update_panel_visibility()

    # ------------------------------------------------------------------
    # Python console (singleton — one namespace per session)
    # ------------------------------------------------------------------

    def add_console_panel(self) -> ConsolePanel:
        """Create (or re-show) the console panel and return it.

        Unlike every other panel type the console is a singleton: its value is
        the namespace it accumulates, so a second instance would be a second
        set of variables with the same names. Re-adding brings the existing one
        back instead of starting over.
        """
        if self.console_panel is not None:
            dock = self._console_dock
            if dock is not None:
                dock.show()
                dock.raise_()
            return self.console_panel

        panel = ConsolePanel(self.app_state)
        dock = self._make_dock("Python console", panel, self.remove_console_panel)
        dock.setObjectName("ConsoleDock")
        anchor = self._last_open_dock()
        if anchor is None:
            self._dock_host.addDockWidget(Qt.LeftDockWidgetArea, dock)
        else:
            self._dock_host.splitDockWidget(anchor, dock, Qt.Vertical)
        dock.show()
        self.console_panel = panel
        self._console_dock = dock
        if self.active_panels is not None:
            self.active_panels.register(panel, "console", clicked_signal=panel.plot_clicked)
        return panel

    def remove_console_panel(self) -> None:
        panel, dock = self.console_panel, self._console_dock
        self.console_panel = None
        self._console_dock = None
        if panel is None:
            return
        if self.active_panels is not None:
            self.active_panels.unregister(panel)
        if dock is not None:
            self._dock_host.removeDockWidget(dock)
            dock.deleteLater()

    def _anchor_dock_for_group(self, group: str) -> QDockWidget | None:
        """Audio panels stay grouped at the top of the default vertical stack
        (anchor = last open audio dock, else first position); feature panels
        go to the bottom (below the last open dock)."""
        if group == "audio":
            for plot in reversed(self._audio_plots()):
                dock = self._dyn_docks[plot]
                if not dock.isHidden() and not dock.isFloating():
                    return dock
            return None
        if group == "neo":
            for plot in reversed(self._neo_plots()):
                dock = self._dyn_docks[plot]
                if not dock.isHidden() and not dock.isFloating():
                    return dock
        return self._last_open_dock()

    def _last_open_dock(self) -> QDockWidget | None:
        """The bottom anchor for a new feature dock (default vertical stack)."""
        for docks in (
            [self._dyn_docks[p] for p in self._label_ribbons()],
            [self._dyn_docks[p] for p in self._panels_of_group("feature")],
            [self._panel_docks[n] for n, _ in _PANEL_ORDER],
            [self._dyn_docks[p] for p in self._neo_plots()],
            [self._dyn_docks[p] for p in self._audio_plots()],
        ):
            for dock in reversed(docks):
                if not dock.isHidden() and not dock.isFloating():
                    return dock
        return None

    # --- Backwards-compat wrappers (all delegate to the generic path) ---

    def add_lineplot(self, feature: str | None = None):
        return self.add_panel("lineplot", feature=feature)

    def add_audio_panel(self, panel_type: str, mic_name: str | None = None):
        return self.add_panel(panel_type, mic_name=mic_name)

    def remove_lineplot(self, plot) -> None:
        self.remove_panel(plot)

    def remove_audio_panel(self, plot) -> None:
        self.remove_panel(plot)

    @property
    def audio_trace_plot(self) -> AudioTracePlot | None:
        """First audio-trace instance (backwards-compat accessor)."""
        plots = self.audio_trace_plots
        return plots[0] if plots else None

    @property
    def spectrogram_plot(self) -> SpectrogramPlot | None:
        """First spectrogram instance (backwards-compat accessor)."""
        plots = self.spectrogram_plots
        return plots[0] if plots else None

    @property
    def heatmap_plot(self) -> HeatmapPlot | None:
        """Backwards-compat accessor: the active heatmap instance if the
        active feature plot is a heatmap, else the first one, else None."""
        active = self.active_feature_plot
        if getattr(active, "panel_type", None) == "heatmap":
            return active
        plots = self.heatmap_plots
        return plots[0] if plots else None

    def set_audio_panel_mic(self, plot, mic_name: str | None) -> None:
        """Re-pin an audio panel to another mic/channel and refresh it."""
        if getattr(plot, "panel_group", None) != "audio":
            return
        panel_type = plot.panel_type
        plot.mic_name = mic_name
        plot.set_source(build_audio_source(self.app_state, mic_name))
        dock = self._dyn_docks.get(plot)
        if dock is not None:
            title = f"{panel_type} — {mic_name}" if mic_name else panel_type
            dock.setWindowTitle(title)
            bar = dock.titleBarWidget()
            if bar is not None:
                bar.set_title(title)
        t0, t1 = self.get_current_xlim()
        plot.update_plot(t0=t0, t1=t1)
        self.schedule_labels_redraw()

    # ------------------------------------------------------------------
    # Panel layout persistence (app_state.panel_layout → local_settings.yaml)
    # ------------------------------------------------------------------

    def _canonicalize_dock_names(self):
        """Name dynamic-panel docks by per-type list position so
        QMainWindow.saveState / restoreState blobs match across sessions."""
        counters: dict[str, int] = {}
        for plot in self._dyn_panels:
            i = counters.get(plot.panel_type, 0)
            counters[plot.panel_type] = i + 1
            self._dyn_docks[plot].setObjectName(f"panel_{plot.panel_type}_{i}")

    def layout_state(self) -> dict:
        """Serializable panel layout: the open panels (type + full per-panel
        settings: feature, dim selections — a dim absent means "All" — and
        color) plus the dock-host state blob that encodes the free 2D
        arrangement (positions, sizes, tabs, floating)."""
        self._canonicalize_dock_names()
        panels = []
        for plot in self._audio_plots():
            panels.append({"type": plot.panel_type, "mic": plot.mic_name})
        for plot in self._neo_plots():
            panels.append(
                {
                    "type": "neo",
                    "stream_name": getattr(plot, "neo_stream_name", None),
                    "channels": getattr(plot, "neo_channels", None),
                }
            )
        for name, _ in _PANEL_ORDER:
            if self._panel_visible[name]:
                panels.append({"type": name})
        for plot in self._panels_of_group("feature"):
            panels.append({"type": plot.panel_type, **plot.panel_settings()})
        for plot in self._label_ribbons():
            panels.append({"type": plot.panel_type})
        return {
            "panels": panels,
            "dock_state_b64": base64.b64encode(bytes(self._dock_host.saveState())).decode("ascii"),
        }

    def apply_layout_state(self, state: dict) -> None:
        """Recreate the panel layout captured by :meth:`layout_state`."""
        entries = state.get("panels")
        if not isinstance(entries, list):
            return

        types = {e.get("type") for e in entries}

        for plot in list(self._dyn_panels):
            self.remove_panel(plot)
        for e in entries:
            if e.get("type") in ("audiotrace", "spectrogram"):
                self.add_panel(e["type"], mic_name=e.get("mic"))
            elif e.get("type") == "neo":
                self.add_panel("neo", stream_name=e.get("stream_name"), channels=e.get("channels"))
            elif e.get("type") == "labels":
                self.add_panel("labels")
        if "raster" in types:
            self.set_neural_panel_mode("raster")
        elif "ephys" in types:
            self.set_neural_panel_mode("trace")
        else:
            self.set_ephys_visible(False)

        for e in entries:
            if e.get("type") in ("lineplot", "heatmap"):
                plot = self.add_panel(e["type"], feature=e.get("feature"))
                if plot is not None:
                    plot.apply_panel_settings(e)
                    self.set_panel_title(plot, self.panel_title(plot))
                    if self.app_state.ready:
                        plot.update_plot()
        self._canonicalize_dock_names()

        # Showing an audio panel requires re-wiring its source.
        if "audiotrace" in types or "spectrogram" in types:
            self.update_audio_panels()

        blob = state.get("dock_state_b64")
        if blob:
            # Deferred so it runs after the _apply_panel_sizes singleShot that
            # update_audio_panels / visibility changes schedule.
            data = QByteArray(base64.b64decode(blob))
            QTimer.singleShot(0, lambda: self._dock_host.restoreState(data))

    def update_feature_plots(self, **kwargs) -> None:
        """Re-render every feature panel: all line plots and heatmaps.

        A label timeline has no content, but its x-extent follows the trial
        the same way.
        """
        for plot in self._panels_of_group("feature") + self._label_ribbons():
            plot.update_plot(**kwargs)

    def _get_all_plots(self) -> list:
        return super()._get_all_plots() + list(self.line_plots) + self._label_ribbons()

    def sizeHint(self):
        return QSize(self.width(), PLOT_CONTAINER_SIZE_HINT_HEIGHT)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._label_indicator._reposition()
        self.schedule_axis_align()

    def _update_label_indicator(self, time_s: float):
        # Label indicator (text badge) is only shown on video, not on plots
        self._label_indicator.hide()

    def _on_plot_zoom(self):
        self.update_audio_changepoint_styles()
        self.update_oscillatory_event_styles()

    # ------------------------------------------------------------------
    # Panel configuration
    # ------------------------------------------------------------------

    def configure_panels(self):
        """Called after data load to set up which panels are available."""
        self._update_panel_visibility()

    def _get_panel_widget(self, name: str):
        return getattr(self, _PANEL_PLOT_ATTR[name])

    def _visible_panel_names(self) -> list[str]:
        result = []
        for name, guard in _PANEL_ORDER:
            if guard and not getattr(self.app_state, guard, False):
                continue
            if self._panel_visible[name]:
                result.append(name)
        return result

    def _visible_panel_widgets(self) -> list:
        """All open panels in visual order: audio + fixed panels + feature panels."""
        return (
            self._audio_plots()
            + self._neo_plots()
            + [self._get_panel_widget(n) for n in self._visible_panel_names()]
            + self._panels_of_group("feature")
            + self._label_ribbons()
        )

    def _update_panel_visibility(self):
        """Show/hide fixed panel docks in-place; never reparents widgets."""
        visible_names = self._visible_panel_names()
        visible_set = set(visible_names)

        for name, guard in _PANEL_ORDER:
            self._panel_docks[name].setVisible(name in visible_set)

        self._setup_xlinks_from_visible(visible_names)

        # Every panel keeps its own x-axis ticks: with free 2D arrangement
        # there is no single "bottom" panel to delegate them to.
        for widget in self._visible_panel_widgets():
            widget.plotItem.getAxis("bottom").setStyle(showValues=True)
            if not getattr(widget, "_align_yhook", False):
                widget._align_yhook = True
                widget.plotItem.vb.sigYRangeChanged.connect(self.schedule_axis_align)

        self._apply_all_zoom_constraints()
        QTimer.singleShot(0, self._apply_panel_sizes)
        self.schedule_labels_redraw()
        self.schedule_axis_align()

    def schedule_labels_redraw(self) -> None:
        """Redraw labels once the pending panel-content renders are done.

        General rule: any path that creates or shows a panel must end with a
        label redraw that runs AFTER the panel's content render (update_plot /
        set_source), because "bottom"-strip overlay modes position rectangles
        from the plot's y viewRange — drawing before the content sets the real
        range leaves the labels invisible. Deferring to the next event-loop
        tick (coalesced) guarantees that ordering for every creation path.
        """
        if self._labels_redraw_scheduled:
            return
        self._labels_redraw_scheduled = True
        QTimer.singleShot(0, self._emit_labels_redraw)

    def _emit_labels_redraw(self) -> None:
        self._labels_redraw_scheduled = False
        self.labels_redraw_needed.emit()

    # ------------------------------------------------------------------
    # X-axis alignment across stacked panels
    # ------------------------------------------------------------------

    def schedule_axis_align(self) -> None:
        """Line up the plotting rectangle (left + right insets) of every open
        panel so their time axes start/end at the same pixel and one second is
        the same width everywhere.

        Two things break vertical alignment of stacked, x-linked panels:
          * left y-axis width varies with tick-label length (a value with more
            decimal places pushes the plot area right), so panels start at
            different x positions and, for the same time range, have different
            pixels-per-second;
          * a colorbar (heatmap) or a right axis (confidence / envelope scale)
            takes space on the right that plain panels don't, so their right
            edges differ. Every panel therefore keeps a fixed right gutter of
            the colorbar's footprint (``PANEL_RIGHT_GUTTER_PX``), trimmed by
            whatever its right axis already takes.

        Deferred (coalesced, next tick) because axis widths are only accurate
        after the pending content render + paint has updated the tick text.
        """
        if self._axis_align_scheduled:
            return
        self._axis_align_scheduled = True
        QTimer.singleShot(0, self._align_axes_left)

    def _align_axes_left(self) -> None:
        """Pass 1: give every open panel a common left-axis width."""
        self._axis_align_scheduled = False
        panels = list(self._visible_plots())
        if len(panels) < 2:
            for plot in panels:
                plot.plotItem.getAxis("left").setWidth()
                if getattr(plot, "_align_left_forced", False):
                    plot.plotItem.hideAxis("left")
                    plot._align_left_forced = False
            QTimer.singleShot(0, self._align_axes_right)
            return

        naturals = []
        for plot in panels:
            axis = plot.plotItem.getAxis("left")
            if not axis.isVisible():
                # e.g. the raster hides its left axis; show a valueless gutter
                # so its plot area still lines up with the others.
                axis.show()
                axis.setStyle(showValues=False)
                plot._align_left_forced = True
            axis.setWidth()  # reset to auto → maximumWidth() is the natural width
            naturals.append(axis.maximumWidth())

        target = max(naturals)
        for plot in panels:
            plot.plotItem.getAxis("left").setWidth(target)

        # Right-inset measurement needs the new left widths laid out first.
        QTimer.singleShot(0, self._align_axes_right)

    def _align_axes_right(self) -> None:
        """Pass 2: every panel ends its plotting rectangle ``PANEL_RIGHT_GUTTER_PX``
        from its right edge — the gutter, less what a visible right axis takes."""
        for plot in self._visible_plots():
            plot.reserve_right_gutter(right_gutter_width(plot))

    def _setup_xlinks_from_visible(self, visible_names: list[str] | None = None):
        """Keep all panels' x-ranges in sync via explicit setXRange.

        pyqtgraph's ``setXLink`` propagates by aligning pixel geometries,
        which is only meaningful for viewboxes sharing one scene — across
        separate dock widgets it shifts/scales the propagated range (and
        needs two hops through a master for non-master drags). Instead,
        any panel's x-range change is copied verbatim to every other open
        panel, making all panels equal.
        """
        widgets = self._visible_panel_widgets()
        self._xlink_master = widgets[0] if widgets else None
        for widget in widgets:
            if getattr(widget, "_xsync_connected", False):
                continue
            widget._xsync_connected = True
            widget.plotItem.vb.sigXRangeChanged.connect(self._sync_panel_xrange)

    def _sync_panel_xrange(self, vb, xrange):
        """Copy one panel's new x-range to all other open panels."""
        if self._xsync_guard:
            return
        self._xsync_guard = True
        try:
            t0, t1 = xrange
            for widget in self._visible_panel_widgets():
                other = widget.plotItem.vb
                if other is vb:
                    continue
                other.setXRange(t0, t1, padding=0)
        finally:
            self._xsync_guard = False
        # Session basis: the drawn label set depends on the viewport, so a
        # zoom/pan must refresh it (debounced).
        if getattr(self.app_state, "display_basis", "trial") == "session":
            self._labels_zoom_timer.start()

    def _apply_panel_sizes(self):
        """Default vertical sizing from `_PANEL_RATIOS` via resizeDocks.

        Only a best-effort hint: the dock system preserves whatever
        arrangement/sizes the user drags afterwards.
        """
        total = self._dock_host.height()
        if total <= 0:
            return

        has_audio_panel = self.app_state.has_audio and bool(self._audio_plots())
        has_neural_panel = bool(self._neo_plots()) or (
            self.app_state.has_neurons and (self._panel_visible["ephys"] or self._panel_visible["raster"])
        )
        ratios = _PANEL_RATIOS.get((has_audio_panel, has_neural_panel), {"feature": 1.0})

        visible_names = self._visible_panel_names()

        # The "feature" ratio is shared equally by the feature group:
        # every line plot and heatmap instance.
        feature_panels = self._panels_of_group("feature")
        n_feature = len(feature_panels)
        feature_share = ratios.get("feature", 0.3) * total if n_feature else 0.0

        # Every audio / neo instance gets its group's ratio share.
        audio_raw = [(self._dyn_docks[plot], ratios.get(plot.panel_type, 0.2) * total) for plot in self._audio_plots()]
        neo_raw = [(self._dyn_docks[plot], ratios.get("neo", 0.15) * total) for plot in self._neo_plots()]
        ribbon_raw = [(self._dyn_docks[plot], _LABEL_RIBBON_RATIO * total) for plot in self._label_ribbons()]

        raw = {}
        for name in visible_names:
            raw[name] = ratios.get(name, 0.2) * total

        instance_raw = audio_raw + neo_raw + ribbon_raw
        total_alloc = sum(raw.values()) + feature_share + sum(size for _, size in instance_raw)
        if total_alloc <= 0:
            return
        scale = total / total_alloc
        member_size = max(1, int(feature_share / n_feature * scale)) if n_feature else 0

        docks, sizes = [], []
        for dock, size in instance_raw:
            docks.append(dock)
            sizes.append(max(1, int(size * scale)))
        for name in visible_names:
            docks.append(self._panel_docks[name])
            sizes.append(max(1, int(raw[name] * scale)))
        for plot in feature_panels:
            docks.append(self._dyn_docks[plot])
            sizes.append(member_size)
        if docks:
            self._dock_host.resizeDocks(docks, sizes, Qt.Vertical)

    # ------------------------------------------------------------------
    # Panel visibility toggles
    # ------------------------------------------------------------------

    def _set_panel_visible(self, name: str, visible: bool):
        if self._panel_visible[name] == visible:
            return
        self._panel_visible[name] = visible
        self._update_panel_visibility()

    def _set_type_present(self, panel_type: str, present: bool) -> None:
        """Backwards-compat shim for dynamic panel types: True ensures at
        least one instance exists; False removes every instance."""
        if present:
            if not self.panels_of_type(panel_type):
                self.add_panel(panel_type)
        else:
            for plot in self.panels_of_type(panel_type):
                self.remove_panel(plot)

    def set_audiotrace_visible(self, visible: bool):
        self._set_type_present("audiotrace", visible)

    def set_spectrogram_visible(self, visible: bool):
        self._set_type_present("spectrogram", visible)

    def set_heatmap_visible(self, visible: bool):
        self._set_type_present("heatmap", visible)

    def set_feature_view(self, mode: str):
        """Make *mode* ("lineplot" or "heatmap") the current feature view:
        an instance of that type becomes the active feature plot, created if
        none exists. Other panels are untouched — instances are only ever
        removed via their ✕."""
        if mode not in ("lineplot", "heatmap"):
            return
        if getattr(self.active_feature_plot, "panel_type", None) == mode:
            plot = self.active_feature_plot
        else:
            plots = self.panels_of_type(mode)
            plot = plots[0] if plots else self.add_panel(mode)
        if plot is None:
            return
        self.active_feature_plot = plot
        self.plot_changed.emit(mode)
        self.schedule_labels_redraw()

    # ------------------------------------------------------------------
    # Ephys panel show/hide
    # ------------------------------------------------------------------

    def set_ephys_visible(self, visible: bool):
        if self._panel_visible["ephys"] == visible:
            return
        self._panel_visible["ephys"] = visible
        if not visible:
            self._panel_visible["raster"] = False
            self.ephys_trace_plot.buffer.loader = None
            self.ephys_trace_plot.set_source(None)
        self._update_panel_visibility()

    def set_raster_visible(self, visible: bool):
        self._set_panel_visible("raster", visible)

    def set_neural_panel_mode(self, mode: str):
        """Switch between 'trace' and 'raster' for the neural panel slot."""
        if mode == "trace":
            show_ephys, show_raster = True, False
        elif mode == "raster":
            show_ephys, show_raster = False, True
        else:
            return

        v = self._panel_visible
        if v["ephys"] != show_ephys or v["raster"] != show_raster:
            v["ephys"] = show_ephys
            v["raster"] = show_raster
            self._update_panel_visibility()

    # ------------------------------------------------------------------
    # Bidirectional y-axis sync: ephys <-> raster
    # ------------------------------------------------------------------

    def _sync_raster_y_from_ephys(self):
        if self._syncing_y or not self._panel_visible["raster"]:
            return
        self._syncing_y = True
        try:
            y_lo, y_hi = self.ephys_trace_plot.vb.viewRange()[1]
            self.raster_plot.vb.setYRange(y_lo, y_hi, padding=0)
        finally:
            self._syncing_y = False

    def _sync_ephys_y_from_raster(self):
        if self._syncing_y or not self._panel_visible["raster"]:
            return
        self._syncing_y = True
        try:
            y_lo, y_hi = self.raster_plot.vb.viewRange()[1]
            self.ephys_trace_plot.vb.setYRange(y_lo, y_hi, padding=0)
        finally:
            self._syncing_y = False

    def _on_ephys_y_space_changed(self):
        ep = self.ephys_trace_plot
        total = len(ep._total_ordered_channels)
        if total == 0:
            return
        spacing = ep.buffer.channel_spacing
        self.raster_plot.sync_y_axis(ep._hw_to_global_y, spacing, total)

    # ------------------------------------------------------------------
    # Public API (compatible with PlotContainer + MultiPanelContainer)
    # ------------------------------------------------------------------

    def get_current_plot(self):
        """The feature plot the user is working with: the active (last
        clicked) feature panel, else the first open feature panel, else a
        hidden stand-in line plot (so callers never get None)."""
        active = self.active_feature_plot
        if active is not None and active in self._dyn_panels:
            return active
        feature_panels = self._panels_of_group("feature")
        if feature_panels:
            return feature_panels[0]
        return self._fallback_plot

    @property
    def _feature_plot(self):
        return self.get_current_plot()

    @property
    def _feature_type(self) -> str:
        return "heatmap" if getattr(self.get_current_plot(), "panel_type", None) == "heatmap" else "lineplot"

    @property
    def current_plot(self):
        return self.get_current_plot()

    def get_current_xlim(self):
        master = self._xlink_master or self.get_current_plot()
        return master.get_current_xlim()

    def set_x_range(self, mode="default", curr_xlim=None, center_on_frame=None):
        master = self._xlink_master or self.get_current_plot()
        return master.set_x_range(
            mode=mode,
            curr_xlim=curr_xlim,
            center_on_frame=center_on_frame,
        )

    @property
    def vb(self):
        return self.get_current_plot().vb

    def get_hovered_plot(self):
        for plot in self._visible_plots():
            if plot.underMouse():
                return plot
        return self.get_current_plot()

    def _visible_plots(self):
        for plot in self._audio_plots():
            if not self._dyn_docks[plot].isHidden():
                yield plot
        for plot in self._neo_plots():
            if not self._dyn_docks[plot].isHidden():
                yield plot
        for name, _ in _PANEL_ORDER:
            if not self._panel_docks[name].isHidden():
                yield self._get_panel_widget(name)
        for plot in self._panels_of_group("feature") + self._label_ribbons():
            if not self._dyn_docks[plot].isHidden():
                yield plot

    def _pending_hover_moved(self, t_display: float) -> None:
        self.pending_label_hovered.emit(t_display)

    def update_time_marker_by_time(self, time_s: float):
        for plot in self._visible_plots():
            plot.update_time_marker(time_s)
        self._update_label_indicator(time_s)
        self.time_marker_updated.emit(time_s)

    def _on_seek_time_requested(self, time_s: float):
        self.update_time_marker_by_time(time_s)
        video = getattr(self.app_state, "video", None)
        if video:
            frame = video.time_to_frame(time_s, round_nearest=True)
            video.blockSignals(True)
            video.seek_to_frame(frame)
            video.blockSignals(False)
            self.app_state.current_frame = frame

    def update_time_marker_and_window(self, frame_number):
        video = getattr(self.app_state, "video", None)
        # During audio-clock playback the video posts the exact (sub-frame)
        # marker time so the playhead isn't quantized to the displayed frame.
        override = getattr(video, "marker_time_override", None) if video is not None else None
        if override is not None:
            current_time = override
        elif video:
            current_time = video.frame_to_time(frame_number)
        else:
            current_time = self.app_state.to_display(
                getattr(self.app_state, "trials_sel", None), frame_number / self.app_state.video_fps
            )
        for plot in self._visible_plots():
            plot.update_time_marker(current_time)
        self._update_label_indicator(current_time)
        self.time_marker_updated.emit(current_time)

    def apply_y_range(self, ymin, ymax):
        return self.get_current_plot().apply_y_range(ymin, ymax)

    def toggle_axes_lock(self):
        bounds = self._trial_bounds_tuple()
        for plot in self._visible_plots():
            plot.toggle_axes_lock(x_bounds_override=bounds)

    def _apply_all_zoom_constraints(self):
        bounds = self._trial_bounds_tuple()
        for plot in self._visible_plots():
            plot._apply_zoom_constraints(x_bounds_override=bounds)

    def _trial_bounds_tuple(self):
        tr = self.app_state.padded_bounds
        return (tr.start_s, tr.end_s) if tr is not None else None

    # --- Feature view switching ---

    def switch_to_lineplot(self):
        self.set_feature_view("lineplot")

    def switch_to_heatmap(self):
        self.set_feature_view("heatmap")

    # --- Type checking ---

    def is_lineplot(self):
        return self._feature_type == "lineplot"

    def is_heatmap(self):
        return self._feature_type == "heatmap"

    def is_spectrogram(self):
        return False  # spectrogram is always its own panel when audio loaded

    def is_ephystrace(self):
        return self._panel_visible["ephys"] or self._panel_visible["raster"]

    def has_spectrogram_overlay(self) -> bool:
        return False  # no overlay system — dedicated panel instead

    # --- Audio overlay stubs (dedicated panels instead) ---

    def update_audio_overlay(self):
        pass

    def apply_overlay_levels(self, vmin: float, vmax: float):
        pass

    def apply_overlay_colormap(self, colormap_name: str):
        pass

    # --- Audio panel updates ---

    def update_audio_panels(self):
        """Refresh audio-driven panels (waveform + spectrogram) after mic change.

        Panels pinned to a mic/channel (``plot.mic_name``) keep their own
        source; unpinned panels follow the global Mic combo.
        """
        audio_plots = self._audio_plots()
        for plot in audio_plots:
            plot.set_source(build_audio_source(self.app_state, plot.mic_name))

        t0, t1 = self.get_current_xlim()
        time = self.app_state.time

        if time is not None:
            vals = np.asarray(time)
            data_t0, data_t1 = float(vals[0]), float(vals[-1])
            if t1 - t0 < 0.01 or t0 < data_t0 - 1000 or t1 > data_t1 + 1000:
                view_span = self.app_state.view_span
                t0 = data_t0
                t1 = min(data_t0 + view_span, data_t1)
                master = self._xlink_master or self._feature_plot
                master.vb.setXRange(t0, t1, padding=0)

        for plot in audio_plots:
            plot.update_plot(t0=t0, t1=t1)

        self._apply_all_zoom_constraints()
        QTimer.singleShot(0, self._apply_panel_sizes)
        self.schedule_labels_redraw()
        self.schedule_axis_align()

    # --- Time slider ---

    def _on_slider_time(self, time_s: float):
        self.update_time_marker_by_time(time_s)
        center = getattr(self.app_state, "center_playback", False)
        visible = TimeRange(*self.get_current_xlim())
        if center or not visible.contains(time_s):
            half = self.app_state.view_span / 2.0
            master = self._xlink_master or self._feature_plot
            master.vb.setXRange(time_s - half, time_s + half, padding=0)

    # --- Audio playback (space key) ---

    def toggle_pause_resume(self):
        self.audio_player.toggle()

    def _get_first_visible_time(self) -> float:
        for plot in self._visible_plots():
            return plot.time_marker.value()
        return 0.0

    # --- Confidence overlay ---

    def show_confidence_plot(self, confidence_data, time_coord=None):
        self.overlay_manager.remove_overlay("confidence")

        if confidence_data is None or len(confidence_data) == 0:
            return

        if time_coord is None:
            time_coord = self.app_state.time_coord.values

        item = pg.PlotCurveItem(pen=pg.mkPen(color="k", width=2, style=Qt.PenStyle.DashLine))
        self.overlay_manager.add_scaled_overlay(
            "confidence",
            self.current_plot,
            item,
            time_coord,
            np.asarray(confidence_data, dtype=np.float64),
            tick_format="{:.2f}",
        )

    def hide_confidence_plot(self):
        self.overlay_manager.remove_overlay("confidence")

    # --- Onset-model probability curves ---

    def show_onset_curves(self, time, curves: dict, colors: dict | None = None) -> int:
        """One dashed curve per label class over the current plot.

        What an onset model believed frame by frame, in each class's own
        colour. Every class is scaled against a fixed 0–1 range rather than
        its own extent, so their heights mean the same thing — the point is
        seeing one class's belief rise where another class's event sits.

        Returns how many were drawn — 0 when no open panel can host them.
        """
        self.hide_onset_curves()
        host = self.overlay_host()
        if host is None or time is None or len(time) == 0:
            return 0
        for label, curve in curves.items():
            color = (colors or {}).get(label) or "k"
            item = pg.PlotCurveItem(pen=pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine))
            name = f"{ONSET_CURVE_PREFIX}{label}"
            self.overlay_manager.add_scaled_overlay(
                name,
                host,
                item,
                np.asarray(time, dtype=np.float64),
                np.asarray(curve, dtype=np.float64),
                data_min=0.0,
                data_max=1.0,
                tick_format="{:.2f}",
            )
            self._onset_curve_names.append(name)
        return len(self._onset_curve_names)

    def overlay_host(self):
        """An **open** panel to hang a time overlay on, or ``None``.

        :meth:`get_current_plot` never returns None — with no feature panel
        open it hands back a hidden stand-in, which silently swallows an
        overlay. Anything the user is meant to *see* asks here instead: the
        active feature panel when it is open, else any open panel that can
        host one, else nothing at all.
        """
        current = self.get_current_plot()
        if current in self._dyn_panels:
            return current
        return next((p for p in self._dyn_panels if hasattr(p, "vb") and hasattr(p, "plot_item")), None)

    def hide_onset_curves(self) -> None:
        for name in self._onset_curve_names:
            self.overlay_manager.remove_overlay(name)
        self._onset_curve_names = []

    # --- Amplitude envelope ---

    def draw_amplitude_envelope(
        self,
        time: np.ndarray,
        envelope: np.ndarray,
        threshold: float | None = None,
        thresholds: list[tuple[float, Any]] | None = None,
    ):
        self.clear_amplitude_envelope()

        host = self._get_amp_envelope_host()
        if host is None:
            return

        if thresholds is None and threshold is not None:
            default_pen = pg.mkPen(color=(255, 50, 50, 200), width=2, style=Qt.DashLine)
            if isinstance(threshold, (tuple, list)):
                thresholds = [(v, default_pen) for v in threshold]
            else:
                thresholds = [(threshold, default_pen)]

        vb = self.overlay_manager.add_viewbox_overlay(
            "amplitude_envelope",
            host,
            axis_label="Envelope",
            axis_color=ENVELOPE_OVERLAY_COLOR,
        )
        vb.setZValue(1000)

        item = pg.PlotDataItem(
            time,
            envelope,
            pen=pg.mkPen(color=ENVELOPE_OVERLAY_COLOR, width=2),
            downsample=10,
            downsampleMethod="peak",
        )
        vb.addItem(item)

        max_thresh = 0.0
        if thresholds:
            for value, pen in thresholds:
                vb.addItem(pg.InfiniteLine(pos=value, angle=0, pen=pen))
                max_thresh = max(max_thresh, float(value))

        env_max = max(float(envelope.max()), max_thresh * 1.5) if max_thresh > 0 else float(envelope.max())
        env_min = float(envelope.min())
        if env_min >= env_max:
            env_max = env_min + 1.0
        vb.setYRange(env_min, env_max, padding=0.05)

        t0, t1 = host.get_current_xlim()
        vb.setXRange(t0, t1, padding=0)

    def clear_amplitude_envelope(self):
        self.overlay_manager.remove_overlay("amplitude_envelope")

    def _get_amp_envelope_host(self):
        host = self._get_envelope_host_plot()
        if host is not None:
            return host
        if self._panel_visible["ephys"] and self.ephys_trace_plot.isVisible():
            return self.ephys_trace_plot
        plot = self.get_current_plot()
        if plot in self.line_plots:
            return plot
        return None

    # --- Envelope sibling trace ---

    def _get_envelope_host_plot(self):
        for plot in self.audio_trace_plots:
            if plot.isVisible():
                return plot
        return None

    def show_envelope_overlay(self):
        host = self._get_envelope_host_plot()
        if host is None:
            return

        self.hide_envelope_overlay()

        t0, t1 = host.get_current_xlim()
        signal_data, fs, buf_t0 = self._load_envelope_data(host, t0, t1)
        if signal_data is None:
            return

        metric = self.app_state.get_with_default("energy_metric")
        env_time, env_data = compute_energy_envelope(signal_data, fs, metric, self.app_state)

        if env_data is None or len(env_data) == 0:
            return

        env_time = env_time + buf_t0

        item = pg.PlotCurveItem(
            env_time,
            env_data,
            pen=pg.mkPen(color=ENVELOPE_OVERLAY_COLOR, width=ENVELOPE_OVERLAY_WIDTH),
        )

        vb = self.overlay_manager.add_viewbox_overlay(
            "energy_envelope",
            host,
            host_items=[item],
            axis_label="Envelope",
            axis_color=ENVELOPE_OVERLAY_COLOR,
        )
        host.addItem(item)

        self._sync_envelope_axis_to_host(host, vb)

        def on_host_y_changed():
            env_vb = self.overlay_manager.get_viewbox("energy_envelope")
            if env_vb is not None:
                self._sync_envelope_axis_to_host(host, env_vb)

        host.vb.sigYRangeChanged.connect(on_host_y_changed)
        self._envelope_y_updater = on_host_y_changed

        self._envelope_td = ThrottleDebounce(
            debounce_ms=ENVELOPE_OVERLAY_DEBOUNCE_MS,
            throttle_cb=self._refresh_envelope_data,
            debounce_cb=self._refresh_envelope_data,
        )

        def on_x_range_changed():
            if self.overlay_manager.has_overlay("energy_envelope"):
                self._envelope_td.trigger()

        host.vb.sigXRangeChanged.connect(on_x_range_changed)
        self._envelope_xrange_updater = on_x_range_changed
        self._envelope_host = host

    @staticmethod
    def _sync_envelope_axis_to_host(host, env_vb):
        ymin, ymax = host.vb.viewRange()[1]
        if ymax > ymin:
            env_vb.setYRange(ymin, ymax, padding=0)

    def hide_envelope_overlay(self):
        host = self._envelope_host

        updater = self._envelope_xrange_updater
        if updater and host:
            try:
                host.vb.sigXRangeChanged.disconnect(updater)
            except (RuntimeError, TypeError):
                pass
        self._envelope_xrange_updater = None

        y_updater = self._envelope_y_updater
        if y_updater and host:
            try:
                host.vb.sigYRangeChanged.disconnect(y_updater)
            except (RuntimeError, TypeError):
                pass
        self._envelope_y_updater = None
        self._envelope_host = None

        td = self._envelope_td
        if td:
            td.stop()
            self._envelope_td = None

        self.overlay_manager.remove_overlay("energy_envelope")

    def _compute_current_envelope(self):
        if not self.overlay_manager.has_overlay("energy_envelope"):
            return None

        host = self._get_envelope_host_plot()
        if host is None:
            return None

        t0, t1 = host.get_current_xlim()
        signal_data, fs, buf_t0 = self._load_envelope_data(host, t0, t1)
        if signal_data is None:
            return None

        metric = self.app_state.get_with_default("energy_metric")
        env_time, env_data = compute_energy_envelope(signal_data, fs, metric, self.app_state)

        if env_data is None or len(env_data) == 0:
            return None

        env_time = env_time + buf_t0
        return env_time, env_data

    def _refresh_envelope_data(self):
        result = self._compute_current_envelope()
        if result is None:
            return
        env_time, env_data = result

        vb_entry = self.overlay_manager._vb_entries.get("energy_envelope")
        if vb_entry and vb_entry.host_items:
            vb_entry.host_items[0].setData(env_time, env_data)

        env_vb = self.overlay_manager.get_viewbox("energy_envelope")
        if env_vb is not None:
            host = self._get_envelope_host_plot()
            if host is not None:
                self._sync_envelope_axis_to_host(host, env_vb)

    def _load_envelope_data(self, host, t0, t1):
        # Respect the host panel's pinned mic/channel; fall back to the
        # global Mic combo selection.
        mic_name = getattr(host, "mic_name", None)
        audio_path, channel_idx = self.app_state.get_audio_source(mic_name)
        if not audio_path:
            audio_path = getattr(self.app_state, "audio_path", None)
        if not audio_path:
            return None, None, None
        loader = SharedAudioCache.get_loader(audio_path)
        if loader is None:
            return None, None, None
        fs = loader.rate
        start_idx = max(0, int(t0 * fs))
        stop_idx = min(len(loader), int(t1 * fs))
        if stop_idx <= start_idx:
            return None, None, None
        audio_data = np.array(loader[start_idx:stop_idx], dtype=np.float64)
        if audio_data.ndim > 1:
            ch = min(channel_idx, audio_data.shape[1] - 1)
            audio_data = audio_data[:, ch]
        return audio_data, fs, t0
