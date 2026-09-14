"""Shared base class for plot widgets with sync and marker functionality."""

import logging
from typing import Optional, Tuple

import pyqtgraph as pg
from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from qtpy.QtWidgets import QGraphicsItem, QGraphicsWidget

from ethograph.io.catalog import INDIVIDUAL_DIMS

logger = logging.getLogger(__name__)


from .app_constants import (  # noqa: E402
    AXIS_LIMIT_PADDING_RATIO,
    LOCKED_RANGE_MAX_FACTOR,
    LOCKED_RANGE_MIN_FACTOR,
    PANEL_RIGHT_GUTTER_PX,
    PANEL_RIGHT_SPACER_PX,
    Z_INDEX_TIME_MARKER,
)


# -------------------------------
# Worker helper
# -------------------------------
class WorkerSignals(QObject):
    """Signals for worker completion."""

    finished = Signal(object)  # emit computation result


class Worker(QRunnable):
    """Run a function in a background thread."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        result = self.fn(*self.args, **self.kwargs)
        self.signals.finished.emit(result)


# -------------------------------
# ThrottleDebounce (GUI-thread safe)
# -------------------------------
class ThrottleDebounce(QObject):
    """Throttle + debounce helper for rate-limiting expensive plot updates.

    All callbacks are invoked on the GUI (main) thread — safe to call any
    Qt operation inside them.

    For callbacks that do heavy computation (numpy/IO), split them into a
    pure-compute function and a render function, then use ``run_async``
    inside the callback so that only the Qt rendering touches the main thread.

    Usage::

        self._td = ThrottleDebounce(
            throttle_ms=16,
            debounce_ms=40,
            throttle_cb=self._on_throttle,  # called on main thread
            debounce_cb=self._on_debounce,  # called on main thread
        )
        self._td.trigger()  # call from any GUI event
    """

    def __init__(
        self,
        throttle_ms: int = 16,
        debounce_ms: int = 40,
        throttle_cb=None,
        debounce_cb=None,
    ):
        super().__init__()
        self._throttle_cb = throttle_cb
        self._debounce_cb = debounce_cb

        # Timers live in the main thread (created here, which is the main thread).
        self._throttle_timer = QTimer(self)
        self._throttle_timer.setInterval(max(throttle_ms, 1))
        self._throttle_timer.timeout.connect(self._run_throttle)

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(debounce_ms)
        self._debounce_timer.timeout.connect(self._on_debounce)

    def trigger(self):
        """Call from a GUI-thread event (e.g. sigRangeChanged handler)."""
        if not self._throttle_timer.isActive():
            self._throttle_timer.start()
        self._debounce_timer.start()

    def _run_throttle(self):
        # Runs on main thread via QTimer — safe for any Qt operation.
        if self._throttle_cb is not None:
            self._throttle_cb()

    def _on_debounce(self):
        # Drag stopped: stop throttle, fire final update.
        self._throttle_timer.stop()
        if self._debounce_cb is not None:
            self._debounce_cb()

    def stop(self):
        self._throttle_timer.stop()
        self._debounce_timer.stop()


def run_async(compute_fn, render_fn):
    """Run ``compute_fn`` in a background thread; deliver its return value to
    ``render_fn`` on the main (GUI) thread.

    ``compute_fn`` must not touch any Qt objects.
    ``render_fn(result)`` may freely update the GUI.

    Example inside a plot callback::

        def _do_range_update(self):
            if self._busy:
                return
            self._busy = True
            t0, t1 = self.get_current_xlim()
            run_async(
                lambda: self._compute_data(t0, t1),  # background thread
                lambda data: self._render(data),  # main thread
            )
    """
    worker = Worker(compute_fn)
    # WorkerSignals was created in the main thread, so this connection is
    # automatically a Qt.QueuedConnection → render_fn runs on the main thread.
    worker.signals.finished.connect(render_fn, Qt.QueuedConnection)
    QThreadPool.globalInstance().start(worker)


class TimeAxisItem(pg.AxisItem):
    """Custom axis that displays time in min:sec format."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        """Convert seconds to a human-readable format that adapts to zoom level."""
        strings = []
        for v in values:
            sign = "-" if v < 0 else ""
            a = abs(v)
            if spacing >= 60:
                minutes = int(a // 60)
                seconds = a % 60
                strings.append(f"{sign}{minutes}:{int(seconds):02d}")
            elif spacing >= 1:
                minutes = int(a // 60)
                seconds = a % 60
                if minutes > 0:
                    strings.append(f"{sign}{minutes}:{seconds:05.2f}")
                elif seconds == int(seconds):
                    strings.append(f"{sign}{int(seconds)}s")
                else:
                    strings.append(f"{sign}{seconds:.2f}s")
            elif spacing >= 0.001:
                ms = a * 1000
                if spacing >= 0.1:
                    strings.append(f"{sign}{ms:.0f}ms")
                elif spacing >= 0.01:
                    strings.append(f"{sign}{ms:.1f}ms")
                else:
                    strings.append(f"{sign}{ms:.2f}ms")
            else:
                us = a * 1_000_000
                strings.append(f"{sign}{us:.1f}µs")
        return strings


class PanelStateMixin:
    """Per-panel selection state shared by every feature plot (line plots AND
    the heatmap), so each panel is an independent instance.

    ``panel_state`` keys ("feature", "selections", "color") OVERRIDE the global
    app_state; missing keys fall back to it. The right sidebar reads and writes
    the *active* plot's state only (``DataWidget.apply_panel_control`` /
    ``sync_sidebar_from_active_plot``) — no feature plot may read
    ``app_state.features_sel`` / ``get_selections()`` directly for rendering.
    """

    #: Initial feature for a panel created with an explicit feature (drag-drop).
    feature_override: str | None = None
    #: Dims a fresh panel leaves "All": none for a line plot (one trace until
    #: the user ticks All), one for the heatmap, whose rows are that dim.
    default_free_dims: int = 0

    @property
    def panel_state(self) -> dict:
        try:
            return self._panel_state
        except AttributeError:
            self._panel_state = {}
            return self._panel_state

    def _effective_feature(self):
        if "feature" in self.panel_state:
            return self.panel_state["feature"]
        if self.feature_override is not None:
            return self.feature_override
        return getattr(self.app_state, "features_sel", None)

    @property
    def pinned_individual(self) -> str | None:
        """The individual this panel is pinned to; ``None`` = follow the sidebar."""
        pin = self.panel_state.get("individual")
        return str(pin) if pin else None

    def set_pinned_individual(self, individual: str | None) -> None:
        if individual:
            self.panel_state["individual"] = str(individual)
        else:
            self.panel_state.pop("individual", None)

    def _effective_selections(self) -> dict:
        if "selections" in self.panel_state:
            sels = dict(self.panel_state["selections"])
        else:
            sels = self.app_state.get_selections()
        # The individual is never one of the panel's own selections: it is
        # the panel's pin, or — for every unpinned panel — the sidebar's
        # individual, so switching the sidebar switches every panel that
        # follows it while a pinned one stays put.
        individual = self.app_state.panel_individual(self)
        if individual is not None:
            for dim in self._individual_dims():
                sels[dim] = individual
        return sels

    def _individual_dims(self) -> list[str]:
        """The spellings of the individual dim this panel's feature carries a value for."""
        loader = getattr(self.app_state, "data_loader", None)
        feature = self._effective_feature()
        if loader is None or not feature:
            return []
        dims = loader.feature_dims(feature)
        return [d for d in INDIVIDUAL_DIMS if d in dims]

    def _effective_color(self):
        if "color" in self.panel_state:
            c = self.panel_state["color"]
        else:
            c = getattr(self.app_state, "colors_sel", None)
        return c if c and c != "None" else None

    def set_panel_control(self, key: str, value) -> None:
        """Record a sidebar control change into this panel's own state.

        ``value is None`` for a dimension key means "All": the dim is removed
        from this panel's selections. The 'All' checkbox state is therefore
        stored per panel as the absence of that dim.
        """
        if key == "features":
            old_dims = self._panel_feature_dims()
            self.panel_state["feature"] = value
            # A new feature brings its own dims. Selections carried over from the
            # old one can leave two or more of them free, and `sel_valid` then
            # returns more than `(time, dim)` — which every plot silently drops.
            # Re-reducing here is what makes a feature with dims the session did
            # not start with (`keypoints_position`: space/keypoint/individual)
            # plottable the moment it is picked.
            sels = self._effective_selections()
            new_dims = {d: v for d, v in self._panel_feature_dims().items() if d not in old_dims}
            self.panel_state["selections"] = self._sanitize_selections(self._pin_defaults(sels, new_dims))
        elif key == "colors":
            self.panel_state["color"] = value
        else:
            sels = dict(self.panel_state.get("selections") or self.app_state.get_selections())
            if value in (None, "", "None"):
                sels.pop(key, None)
            else:
                sels[key] = value
            # Sanitize here too, not only on a feature change: setting a dim to
            # "All" (or inheriting globals mid-rebuild) is the other way a panel
            # ends up with two free dims, and the very next update_plot() then
            # trips sel_valid's (time,)/(time, dim) assertion.
            self.panel_state["selections"] = self._sanitize_selections(sels)

    def panel_settings(self) -> dict:
        """This panel's coords-section settings in serializable form (used by
        layout persistence and panel-type conversion)."""
        settings: dict = {}
        feature = self._effective_feature()
        if feature:
            settings["feature"] = str(feature)
        if "selections" in self.panel_state:
            settings["selections"] = {
                str(k): (v.item() if hasattr(v, "item") else v) for k, v in self.panel_state["selections"].items()
            }
        color = self.panel_state.get("color")
        if color and color != "None":
            settings["color"] = str(color)
        if self.pinned_individual:
            settings["individual"] = self.pinned_individual
        return settings

    def apply_panel_settings(self, settings: dict) -> None:
        """Restore settings captured by :meth:`panel_settings` into this
        panel's own state. A dim absent from ``selections`` means "All"."""
        if settings.get("feature"):
            self.panel_state["feature"] = settings["feature"]
        if isinstance(settings.get("selections"), dict):
            self.panel_state["selections"] = self._sanitize_selections(settings["selections"])
        if settings.get("color"):
            self.panel_state["color"] = settings["color"]
        self.set_pinned_individual(settings.get("individual"))

    def _panel_feature_dims(self) -> dict:
        loader = getattr(self.app_state, "data_loader", None)
        feature = self._effective_feature()
        if loader is None or not feature:
            return {}
        return loader.feature_dims(feature)

    def _pin_defaults(self, selections: dict, dims: dict) -> dict:
        """*selections* with the multi-value *dims* it leaves free pinned to their first value,
        keeping :attr:`default_free_dims` of them "All".

        Only a dim with a sidebar combo is pinned: one without (a ``stack``'s
        columns) could never be set back to "All".
        """
        sels = dict(selections)
        combos = self.app_state.data_loader.catalog.combos if dims else {}
        free = [d for d, vals in dims.items() if d not in sels and len(vals) > 1 and d in combos]
        for d in free[self.default_free_dims :]:
            sels[d] = dims[d][0]
        return sels

    def _sanitize_selections(self, selections: dict) -> dict:
        """Make *selections* valid for this panel's feature.

        Two rules. A selection naming a value the feature does not have is
        **dropped** — `.sel()` raises `KeyError` on it, so a panel carrying one
        renders nothing at all; this is what a selection left over from a
        previous dataset looks like. Then at most ONE dim is left "All"
        (absent), because sel_valid output must stay (time,) or (time, dim): a
        loaded layout violating that keeps the FIRST missing multi-value dim as
        "All" and every later one is pinned to its first value. Single-value
        dims squeeze away and may stay absent.
        """
        sels = dict(selections)
        loader = getattr(self.app_state, "data_loader", None)
        feature = self._effective_feature()
        if loader is None or not feature:
            return sels
        dims = loader.feature_dims(feature)

        # A dim the feature lacks entirely is harmless — sel_valid ignores it.
        # Judge exactly as `_selections_for_var` does: a key is a dim or it is
        # inert. Do not add a plural→singular fallback on one side only — a key
        # counted as pinning here but ignored there leaves an extra dim free,
        # and sel_valid then returns more than (time,)/(time, dim).
        sels = {k: v for k, v in sels.items() if k not in dims or str(v) in dims[k]}
        missing_multi = [d for d, vals in dims.items() if d not in sels and len(vals) > 1]
        for d in missing_multi[1:]:
            sels[d] = dims[d][0]
        return sels

    def resync_selections(self) -> None:
        """Re-validate this panel's selections against the data now loaded.

        Called when the dataset under the panels is replaced: the selections
        were valid for the *previous* data, and one naming a value that no
        longer exists raises `KeyError` out of `.sel()` on the next render.
        """
        if "selections" in self.panel_state:
            self.panel_state["selections"] = self._sanitize_selections(self.panel_state["selections"])

    def _ensure_panel_state(self):
        """Fork any still-missing state keys from the current globals on first
        render, so later global changes can never leak into this panel."""
        if self._effective_feature() is None:
            return
        ps = self.panel_state
        ps.setdefault("feature", self._effective_feature())
        # Sanitized on the way in: the globals are a mirror of whichever panel
        # was last edited, so they can leave this panel's feature with more than
        # one free dim — which renders as nothing at all.
        if "selections" not in ps:
            defaults = self._pin_defaults(self.app_state.get_selections(), self._panel_feature_dims())
            ps["selections"] = self._sanitize_selections(defaults)
        ps.setdefault("color", getattr(self.app_state, "colors_sel", None))


def right_gutter_width(plot: "BasePlot") -> int:
    """The gutter a panel should reserve now: the colorbar footprint, less
    what a visible right axis (confidence / envelope scale) already takes, so
    every panel's plotting rectangle ends on the same pixel."""
    axis = plot.plot_item.getAxis("right")
    taken = axis.geometry().width() if axis.isVisible() else 0.0
    return max(0, int(round(PANEL_RIGHT_GUTTER_PX - taken)))


class BasePlot(pg.PlotWidget):
    """Base class for plot widgets with shared sync and marker functionality.

    Handles:
    - Time marker for video sync
    - Stream/label mode switching
    - Axes locking
    - X-axis range management
    - Common plot interactions

    Subclasses must implement the display-specific methods.
    """

    plot_clicked = Signal(object)

    #: PlotItem layout cell right of the right axis. A colorbar lands there
    #: (``ColorBarItem.setImageItem(insert_in=)``); on every other panel the
    #: legend host does.
    _GUTTER_CELL = (2, 5)

    def __init__(self, app_state, parent=None, **kwargs):
        time_axis = TimeAxisItem(orientation="bottom")
        super().__init__(parent, background="white", axisItems={"bottom": time_axis}, **kwargs)
        self.app_state = app_state
        self._gutter_host: QGraphicsWidget | None = None
        self.reserve_right_gutter(PANEL_RIGHT_GUTTER_PX)

        self.setLabel("bottom", "Time")

        # Time marker with enhanced styling
        self.time_marker = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", width=2), movable=False)
        self.addItem(self.time_marker)
        self.time_marker.setZValue(Z_INDEX_TIME_MARKER)

        # Setup viewbox and interaction
        self.plot_item = self.plotItem
        self.vb = self.plot_item.vb
        self.vb.setMenuEnabled(False)

        # Store interaction state
        self._interaction_enabled = True

        # Last-rendered time range; subclasses update this to know when to re-render
        self.current_range: tuple[float, float] | None = None

        # Connect click handler
        self.scene().sigMouseClicked.connect(self._handle_click)

    def reserve_right_gutter(self, width: int) -> None:
        """Fix the empty space right of the plotting rectangle to ``width`` px.

        Every stacked panel reserves the same gutter (a colorbar's footprint),
        so their time axes end on one pixel; the container trims it by
        whatever a visible right axis already takes.
        """
        layout = self.plotItem.layout
        layout.setColumnFixedWidth(4, PANEL_RIGHT_SPACER_PX)
        layout.setColumnFixedWidth(5, max(0, width))

    def legend_host(self) -> QGraphicsWidget | None:
        """The gutter item a legend anchors to; ``None`` when a colorbar holds
        the cell. Created on first use, after a subclass has had its chance to
        place a colorbar there."""
        if self._gutter_host is not None:
            return self._gutter_host
        row, col = self._GUTTER_CELL
        layout = self.plot_item.layout
        # itemAt() warns on stderr for a cell outside the grid, which the
        # gutter cell is until something is placed there.
        in_grid = row < layout.rowCount() and col < layout.columnCount()
        if in_grid and layout.itemAt(row, col) is not None:
            return None
        host = QGraphicsWidget()
        host.setFlag(QGraphicsItem.ItemClipsChildrenToShape, True)
        layout.addItem(host, row, col)
        self._gutter_host = host
        return host

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        """Update the specific plot content (line plot, spectrogram, etc.).

        Subclasses should override this method.
        """
        logger.debug(
            "update_plot_content called in %s (id=%s) t0=%s, t1=%s",
            self.__class__.__name__,
            id(self),
            t0,
            t1,
        )
        raise NotImplementedError("Subclasses must implement update_plot_content")

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        """Apply y-axis range specific to the plot type.

        Subclasses should override this method.
        """
        raise NotImplementedError("Subclasses must implement apply_y_range")

    def update_plot(
        self,
        t0: Optional[float] = None,
        t1: Optional[float] = None,
        preserve_x_range: bool = False,
    ):
        """Update plot with current data and time window."""
        if not hasattr(self.app_state, "ds") or self.app_state.ds is None:
            return

        if preserve_x_range:
            saved_xlim = self.get_current_xlim()

        self.update_plot_content(t0, t1)

        if preserve_x_range:
            self.set_x_range(mode="preserve", curr_xlim=saved_xlim)
        elif t0 is not None and t1 is not None:
            self.set_x_range(mode="preserve", curr_xlim=(t0, t1))
        else:
            self.set_x_range(mode="default")

        # Only apply axis lock after setting the desired range
        is_new_trial = t0 is None and t1 is None and not preserve_x_range
        self.toggle_axes_lock(preserve_default_range=is_new_trial)

    def update_time_marker(self, time_position: float):
        """Update time marker position for video sync."""
        self.time_marker.setValue(time_position)
        self.time_marker.show()

    def update_time_marker_and_window(self, frame_number: int):
        """Update time marker position and window for video sync."""
        video = getattr(self.app_state, "video", None)
        if video:
            current_time = video.frame_to_time(frame_number)
        else:
            current_time = frame_number / self.app_state.video_fps
        self.update_time_marker(current_time)

        if hasattr(self.app_state, "ds") and self.app_state.ds is not None:
            if getattr(self.app_state, "center_playback", False):
                self.set_x_range(mode="center", center_on_frame=frame_number)
                t0, t1 = self.get_current_xlim()
                self.update_plot_content(t0, t1)
            else:
                t0, t1 = self.get_current_xlim()
                self.update_plot_content(t0, t1)

    def set_x_range(self, mode="default", curr_xlim=None, center_on_frame=None):
        """Set plot x-range with different behaviors."""
        if not hasattr(self.app_state, "ds") or self.app_state.ds is None:
            return

        tr = self.app_state.window_bounds
        bounds = (tr.start_s, tr.end_s) if tr is not None else None
        if bounds is None:
            if mode == "preserve" and curr_xlim:
                self.vb.setXRange(curr_xlim[0], curr_xlim[1], padding=0)
            return
        data_tmin, data_tmax = bounds

        if mode == "center":
            video = getattr(self.app_state, "video", None)
            frame = center_on_frame if center_on_frame is not None else self.app_state.current_frame
            if video:
                current_time = video.frame_to_time(frame)
            else:
                # No video: frames tick on the trial clock; the axis speaks
                # the display clock.
                current_time = self.app_state.to_display(
                    getattr(self.app_state, "trials_sel", None), frame / self.app_state.video_fps
                )

            xlim = self.get_current_xlim()
            half_window = (xlim[1] - xlim[0]) / 2.0
            t0 = current_time - half_window
            t1 = current_time + half_window

        elif mode == "preserve" and curr_xlim:
            t0 = curr_xlim[0]
            t1 = curr_xlim[1]

            if t0 < data_tmin:
                t0 = data_tmin
            elif t1 > data_tmax:
                t1 = data_tmax

        else:  # mode == 'default'
            view_span = self.app_state.view_span
            t0 = data_tmin
            # view_span (before_s + after_s) is 0 in trial mode → show the
            # whole window instead of collapsing to a zero-width range.
            t1 = min(t0 + view_span, data_tmax) if view_span > 0 else data_tmax

        self.vb.setXRange(t0, t1, padding=0)

    def get_current_xlim(self) -> Tuple[float, float]:
        """Get current x-axis limits."""
        return self.vb.viewRange()[0]

    def toggle_axes_lock(self, preserve_default_range=False, x_bounds_override=None):
        """Enable or disable axes locking to prevent zoom but allow panning."""
        locked = self.app_state.lock_axes

        if locked:
            current_xlim = self.vb.viewRange()[0]
            current_ylim = self.vb.viewRange()[1]
            x_range = current_xlim[1] - current_xlim[0]

            tr = self.app_state.padded_bounds
            bounds = x_bounds_override or ((tr.start_s, tr.end_s) if tr is not None else None)
            if hasattr(self.app_state, "ds") and self.app_state.ds is not None and bounds is not None:
                data_xmin, data_xmax = bounds
                data_range = data_xmax - data_xmin
                padding = min(data_range * AXIS_LIMIT_PADDING_RATIO, 5)

                if self.app_state.get_with_default("xlim_mode") == "fixed":
                    # Fixed window: span stays locked to fixed_window_s so
                    # dragging pans the window instead of resizing it.
                    min_range = max_range = min(self.app_state.view_span, data_range + 2 * padding)
                elif preserve_default_range:
                    view_span = self.app_state.view_span
                    # view_span is 0 in trial mode → the default view is the
                    # whole window, so lock relative to its full range.
                    default_span = view_span if view_span > 0 else data_range
                    min_range = default_span * LOCKED_RANGE_MIN_FACTOR
                    max_range = default_span * LOCKED_RANGE_MAX_FACTOR
                else:
                    min_range = x_range
                    max_range = x_range

                self.vb.setLimits(
                    xMin=data_xmin - padding,
                    xMax=data_xmax + padding,
                    minXRange=min_range,
                    maxXRange=max_range,
                    yMin=current_ylim[0],
                    yMax=current_ylim[1],
                )

            self.vb.setMouseEnabled(x=True, y=False)
        else:
            self._apply_zoom_constraints(x_bounds_override=x_bounds_override)
            self.vb.setMouseEnabled(x=True, y=True)

    def _apply_zoom_constraints(self, x_bounds_override=None):
        """Apply data-aware zoom constraints to the plot viewbox.

        Parameters
        ----------
        x_bounds_override
            Optional ``(xMin, xMax)`` to use instead of this plot's own
            ``app_state.window_bounds``.  The container passes the tightest
            bounds across all visible panels so that no panel scrolls past
            another's data.
        """
        self.vb.setLimits(
            xMin=None,
            xMax=None,
            yMin=None,
            yMax=None,
            minXRange=None,
            maxXRange=None,
            minYRange=None,
            maxYRange=None,
        )

        if hasattr(self.app_state, "ds") and self.app_state.ds is not None:
            tr = self.app_state.padded_bounds
            bounds = x_bounds_override or ((tr.start_s, tr.end_s) if tr is not None else None)
            if bounds is not None:
                xMin, xMax = bounds
                xRange = xMax - xMin
                padding = xRange * AXIS_LIMIT_PADDING_RATIO

                if self.app_state.get_with_default("xlim_mode") == "fixed":
                    # Fixed window: lock the visible span so dragging pans the
                    # window across the data instead of resizing it.
                    span = min(self.app_state.view_span, xRange + 2 * padding)
                    min_x_range = max_x_range = span
                else:
                    min_x_range = None
                    max_x_range = xRange * (1 + AXIS_LIMIT_PADDING_RATIO)

                self.vb.setLimits(
                    xMin=xMin - padding,
                    xMax=xMax + padding,
                    minXRange=min_x_range,
                    maxXRange=max_x_range,
                )

        self._apply_y_constraints()

    def _apply_y_constraints(self):
        """Apply y-axis constraints specific to the plot type.

        Subclasses should override this method.
        """
        pass  # Default implementation does nothing

    def _handle_click(self, event):
        """Handle mouse clicks on plot."""
        if not self._interaction_enabled:
            return

        from qtpy.QtCore import Qt

        # A left double-click autoscales — unless a label is being drawn, when
        # the "double" click is its closing boundary (two quick clicks inside
        # the system's double-click time and distance) and must be delivered.
        if event.double() and event.button() == Qt.LeftButton and not self.app_state.label_drawing_armed:
            self.autoscale()
            return

        pos = self.plot_item.vb.mapSceneToView(event.scenePos())

        click_info = {"x": pos.x(), "button": event.button(), "plot": self}
        self.plot_clicked.emit(click_info)

    def autoscale(self):
        """Reset Y-axis to auto-fit visible data."""
        self.vb.enableAutoRange(x=False, y=True)
