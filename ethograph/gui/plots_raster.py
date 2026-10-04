"""Spike raster — the spikes of every unit the cluster table lets through, one row each.

A tick per spike while the view is sparse, spike counts per pixel once it is
crowded (``raster_render.choose_render``). Either way only what the viewport
needs is drawn, rebuilt on a debounce so zoom/pan never floods the Qt event
queue. Spikes are black on white; units selected in the cluster table are
drawn in their colours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QColor

from .app_constants import (
    BUFFER_COVERAGE_MARGIN,
    DEFAULT_BUFFER_MULTIPLIER_EPHYS,
    RASTER_DEBOUNCE_MS,
    Z_INDEX_TIME_MARKER,
)
from .plots_base import BasePlot, ThrottleDebounce
from .raster_render import (
    MAX_TICKS,
    TICK_ROW_FRACTION,
    TICK_WIDTH_AUTO,
    SpikeGroup,
    auto_tick_width,
    choose_render,
    density_image,
    tick_segments,
)

if TYPE_CHECKING:
    from ethograph.io.plot_sources import PlotSource

_RENDER_LABEL_COLOR = "#666666"
#: A tick is never shorter than this, so rows thinner than a pixel still show their spikes.
_MIN_TICK_PX = 3.0
<<<<<<< HEAD
#: Row labels on the left axis are at least this far apart; rows closer than that share one.
_MIN_ROW_LABEL_PX = 14.0
=======
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4


@dataclass(frozen=True)
class _Drawn:
    """What is on screen: how it was rendered, over which time buffer, for which view."""

    render: str
    t0: float
    t1: float
    x_span: float
    y_range: tuple[float, float]
    size: tuple[int, int]

    def covers(self, x_lo: float, x_hi: float, y_range: tuple[float, float], size: tuple[int, int]) -> bool:
        """Whether a pan to this view needs nothing redrawn; any zoom or resize does."""
        if y_range != self.y_range or size != self.size:
            return False
        span = x_hi - x_lo
        if not np.isclose(span, self.x_span, rtol=1e-3):
            return False
        margin = span * BUFFER_COVERAGE_MARGIN
        return self.t0 <= x_lo - margin and self.t1 >= x_hi + margin


class RasterPlot(BasePlot):
    """Spike raster: each spike at ``(spike_time, y of its row)``.

    A row is whatever ``sync_y_axis`` maps a row key to: a probe channel's
    depth in the ephys trace's y-space, or a unit's own row.
    """

    y_range_changed = Signal()

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)

        self.time_marker.setPen(pg.mkPen("#FF4444", width=2, style=Qt.PenStyle.DotLine))

        self.setLabel("left", "Unit", Fontsize="14pt")

        self._hw_to_global_y: dict[int, float] = {}
<<<<<<< HEAD
        #: What the left axis writes next to a row, by row key.
        self._row_labels: dict[int, str] = {}
        self._row_ticks: list[tuple[float, str]] | None = None
=======
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
        self._y_lookup: NDArray = np.empty(0, dtype=np.float64)
        self._channel_spacing: float = 1.0
        self._total_channels: int = 0
        #: True while the rows are the ephys trace's channels, so the two panels share a y-range.
        self.follows_trace_y: bool = True

        self._tick_items: list[pg.PlotCurveItem] = []
        self._image_item = pg.ImageItem()
        self._image_item.setZValue(Z_INDEX_TIME_MARKER - 2)
        self._image_item.hide()
        self.vb.addItem(self._image_item, ignoreBounds=True)
        # Says which rendering is on screen whenever it is not the ticks a raster is expected to show.
        self._render_label = pg.TextItem(color=_RENDER_LABEL_COLOR, anchor=(0, 0))
        self._render_label.setParentItem(self.vb)
        self._render_label.setPos(4, 2)
        self._render_label.hide()
        self._drawn: _Drawn | None = None
        #: Pixel width of the ticks last drawn: the user's, or the one picked for the view.
        self.tick_width: int = 1

        self._source: PlotSource | None = None

        # One (times, row keys, colour) entry per colour, times sorted (source of truth).
        self._multi_entries: list[SpikeGroup] = []

        # Debounce viewport-driven rebuilds so rapid zoom/pan doesn't flood
        # the Qt event queue with expensive setData() calls.
        self._td = ThrottleDebounce(
            debounce_ms=RASTER_DEBOUNCE_MS,
            throttle_cb=self._redraw,
            debounce_cb=self._redraw,
        )

        self.vb.sigRangeChanged.connect(self._on_range_changed)
        self.vb.sigResized.connect(self._on_range_changed)
        self.vb.sigYRangeChanged.connect(self._emit_y_range)

    def _on_range_changed(self):
        if self._multi_entries:
            self._td.trigger()

    def _emit_y_range(self):
        self.y_range_changed.emit()

    @property
    def render(self) -> str | None:
        """``"ticks"`` or ``"density"`` — what is on screen now; ``None`` when nothing is."""
        return None if self._drawn is None else self._drawn.render

    # ------------------------------------------------------------------
    # Y-axis sync
    # ------------------------------------------------------------------

    def sync_y_axis(
        self,
        hw_to_global_y: dict[int, float],
        spacing: float,
        total_channels: int,
    ):
        """Set the rows: the y of each row key, ``spacing`` apart, starting at y = 0.

        The y-range is reset only when the row space itself changed, so a
        redraw of the same rows keeps the user's zoom.
        """
        resized = (total_channels, spacing) != (self._total_channels, self._channel_spacing)
        self._hw_to_global_y = hw_to_global_y
        self._channel_spacing = spacing
        self._total_channels = total_channels
        self._y_lookup = np.full(max(hw_to_global_y, default=-1) + 1, np.nan, dtype=np.float64)
        for key, y in hw_to_global_y.items():
            self._y_lookup[key] = y

        if total_channels > 0:
            margin = spacing * 1.0
            y_max = (total_channels - 1) * spacing + margin
            self.vb.setLimits(yMin=-margin, yMax=y_max)
            if resized:
                self.vb.setYRange(-margin, y_max, padding=0)

        self.refresh()
<<<<<<< HEAD

    def set_row_labels(self, labels: dict[int, str]) -> None:
        """Name the rows on the left axis: the label of each row key that has one."""
        self._row_labels = labels
        self._update_row_ticks()

    def row_at(self, y: float) -> int | None:
        """The key of the row drawn at *y*; ``None`` where there is no row."""
        if not self._hw_to_global_y:
            return None
        key, row_y = min(self._hw_to_global_y.items(), key=lambda item: abs(item[1] - y))
        return key if abs(row_y - y) <= self._channel_spacing / 2 else None

    def _update_row_ticks(self) -> None:
        """Label the rows in view, top row first, skipping those too close to the last one written."""
        y_lo, y_hi = self.vb.viewRange()[1]
        min_gap = _MIN_ROW_LABEL_PX * (y_hi - y_lo) / max(self.vb.height(), 1.0)
        in_view = sorted(
            (
                (y, self._row_labels[key])
                for key, y in self._hw_to_global_y.items()
                if key in self._row_labels and y_lo <= y <= y_hi
            ),
            reverse=True,
        )
        ticks: list[tuple[float, str]] = []
        for y, label in in_view:
            if not ticks or ticks[-1][0] - y >= min_gap:
                ticks.append((y, label))
        if ticks != self._row_ticks:
            self._row_ticks = ticks
            self.plot_item.getAxis("left").setTicks([ticks])
=======
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4

    # ------------------------------------------------------------------
    # Spike data API
    # ------------------------------------------------------------------

    def set_multi_cluster_spike_data(self, entries: list[SpikeGroup]):
        # Pre-sort each colour group by time so searchsorted works correctly.
        sorted_entries = []
        for times, rows, color in entries:
            if len(times) == 0:
                continue
            order = np.argsort(times, kind="stable")
            sorted_entries.append((times[order], rows[order], color))
        self._multi_entries = sorted_entries
        self.refresh()

    def clear_spike_data(self):
        self._multi_entries = []
        self._clear_drawn()

    def refresh(self) -> None:
        """Redraw now, whatever is already on screen (new spikes, rows or render mode)."""
        self._drawn = None
        self._redraw()

    # ------------------------------------------------------------------
    # BasePlot overrides
    # ------------------------------------------------------------------

    def update_plot_content(self, t0=None, t1=None):
        pass  # range changes handled via sigRangeChanged → ThrottleDebounce

    def apply_y_range(self, ymin=None, ymax=None):
        if ymin is not None and ymax is not None:
            self.vb.setYRange(ymin, ymax, padding=0)

    def _apply_y_constraints(self):
        if self._total_channels > 0:
            margin = self._channel_spacing * 1.0
            y_max = (self._total_channels - 1) * self._channel_spacing + margin
            self.vb.setLimits(yMin=-margin, yMax=y_max)

    def set_source(self, source: PlotSource | None):
        self._source = source

    # ------------------------------------------------------------------
    # Internal – viewport-culled drawing (called via ThrottleDebounce or directly)
    # ------------------------------------------------------------------

    def _clear_ticks(self):
        for item in self._tick_items:
            self.vb.removeItem(item)
        self._tick_items.clear()

    def _clear_drawn(self):
        self._clear_ticks()
        self._image_item.hide()
        self._render_label.hide()
        self._drawn = None

    def _rows_to_y(self, rows: NDArray) -> tuple[NDArray, NDArray]:
        """The y of each row key, and which keys have a row at all."""
        known = (rows >= 0) & (rows < len(self._y_lookup))
        y = self._y_lookup[np.where(known, rows, 0)]
        return y, known & ~np.isnan(y)

    def _spikes_in_buffer(self, t0: float, t1: float, y_lo: float, y_hi: float) -> list[SpikeGroup]:
        """Per colour, the spikes inside the time buffer whose rows touch the visible y-range."""
        half_row = self._channel_spacing / 2
        groups = []
        for times, rows, color in self._multi_entries:
            # X-cull via searchsorted (O(log n), times is pre-sorted).
            i0 = int(np.searchsorted(times, t0, side="left"))
            i1 = int(np.searchsorted(times, t1, side="right"))
            y, has_row = self._rows_to_y(rows[i0:i1])
            on_screen = has_row & (y >= y_lo - half_row) & (y <= y_hi + half_row)
            if on_screen.any():
                groups.append((times[i0:i1][on_screen], y[on_screen], color))
        return groups

    def _redraw(self):
<<<<<<< HEAD
        self._update_row_ticks()
=======
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
        if not self._hw_to_global_y or not self._multi_entries:
            self._clear_drawn()
            return

        (x_lo, x_hi), (y_lo, y_hi) = self.vb.viewRange()
        size = (max(int(self.vb.width()), 1), max(int(self.vb.height()), 1))
        if self._drawn is not None and self._drawn.covers(x_lo, x_hi, (y_lo, y_hi), size):
            return

        span = x_hi - x_lo
        pad = span * DEFAULT_BUFFER_MULTIPLIER_EPHYS / 2
        t0, t1 = x_lo - pad, x_hi + pad
        groups = self._spikes_in_buffer(t0, t1, y_lo, y_hi)

        n_in_view = sum(int(np.count_nonzero((t >= x_lo) & (t <= x_hi))) for t, _, _ in groups)
        rows_in_view = min(max((y_hi - y_lo) / self._channel_spacing, 1.0), float(self._total_channels))
        # Rows thinner than a pixel cannot be told apart, so a pixel row is the cell then.
        n_cells = int(size[0] * min(rows_in_view, size[1]))
        mode = self.app_state.get_with_default("raster_render_mode")
        render = choose_render(mode, self.render or "ticks", n_in_view, n_cells)

        if render == "ticks":
            self._draw_ticks(groups, (y_hi - y_lo) / size[1], n_in_view, n_cells)
            self._render_label.hide()
        else:
            self._draw_density(groups, t0, t1, max(round(size[0] * (t1 - t0) / span), 1), y_lo, y_hi, size[1])
            refused = mode == "ticks" and n_in_view > MAX_TICKS
            self._render_label.setText("Density — too many spikes in view for ticks" if refused else "Density")
            self._render_label.show()
        self._drawn = _Drawn(render, t0, t1, span, (y_lo, y_hi), size)

    def _draw_ticks(self, groups: list[SpikeGroup], y_per_px: float, n_in_view: int, n_cells: int):
        self._clear_ticks()
        self._image_item.hide()
        half_height = max(self._channel_spacing * TICK_ROW_FRACTION, _MIN_TICK_PX * y_per_px) / 2
        width = self.app_state.get_with_default("raster_tick_width")
        if width == TICK_WIDTH_AUTO:
            width = auto_tick_width(n_in_view, n_cells, 2 * half_height / y_per_px)
        self.tick_width = width
        for times, y, color in groups:
            x, ys = tick_segments(times, y, half_height)
            pen = pg.mkPen(QColor(*color), width=width)
            # Flat, so a wide tick is no taller than a thin one.
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            item = pg.PlotCurveItem(x, ys, connect="pairs", pen=pen, skipFiniteCheck=True)
            item.setZValue(Z_INDEX_TIME_MARKER - 1)
            self.vb.addItem(item, ignoreBounds=True)
            self._tick_items.append(item)

    def _draw_density(
        self,
        groups: list[SpikeGroup],
        t0: float,
        t1: float,
        n_x: int,
        y_lo: float,
        y_hi: float,
        height_px: int,
    ):
        self._clear_ticks()
        spacing = self._channel_spacing
        last_row = self._total_channels - 1
        # Row k is the band [k - 1/2, k + 1/2] * spacing; the image spans the visible ones.
        k_lo = int(np.clip(np.floor(y_lo / spacing + 0.5), 0, last_row))
        k_hi = int(np.clip(np.floor(y_hi / spacing + 0.5), k_lo, last_row))
        n_rows = k_hi - k_lo + 1
        y0 = (k_lo - 0.5) * spacing
        # One cell per row while rows are at least a pixel tall, else one per pixel.
        n_y = min(n_rows, height_px)
        image = density_image(groups, t0, t1, n_x, y0, n_rows * spacing / n_y, n_y)
        self._image_item.setImage(image, autoLevels=False)
        self._image_item.setRect(QRectF(t0, y0, t1 - t0, n_rows * spacing))
        self._image_item.show()
