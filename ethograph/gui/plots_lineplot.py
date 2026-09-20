"""Enhanced line plot inheriting from BasePlot."""

import logging
from typing import Optional

import matplotlib
import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt

import ethograph as eto
from ethograph.io.catalog import PlotData

logger = logging.getLogger(__name__)

from ethograph.io.plot_sources import WindowedBuffer, XarraySource  # noqa: E402

from .app_constants import (  # noqa: E402
    DEFAULT_BUFFER_MULTIPLIER,
    HLINE_COLOR,
    HLINE_WIDTH,
    LINEPLOT_DEBOUNCE_MS,
    MULTIDIM_COLORS,
)
from .make_pretty import clean_display_labels  # noqa: E402
from .plots_base import BasePlot, PanelStateMixin, ThrottleDebounce  # noqa: E402


class LinePlot(PanelStateMixin, BasePlot):
    """Line plot with lazy loading and shared sync/marker functionality."""

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)
        self.setLabel("left", "Value")

        self.plot_items = []
        self.label_items = []
        #: User-drawn horizontal reference lines. Kept apart from
        #: ``plot_items`` (which every re-render clears), so a line outlives
        #: trial changes and feature switches for as long as the panel lives.
        self.hline_items: list[pg.InfiniteLine] = []

        self._buffer = WindowedBuffer(buffer_multiplier=DEFAULT_BUFFER_MULTIPLIER)
        self._current_feature = None
        self._current_trial = None
        self._current_ds_kwargs_hash = None

        self._td = ThrottleDebounce(
            debounce_ms=LINEPLOT_DEBOUNCE_MS,
            throttle_cb=self._do_range_update,
            debounce_cb=self._do_range_update,
        )

        self.vb.sigRangeChanged.connect(self._on_view_range_changed)

    def _get_selections_hash(self) -> str:
        selections = self._effective_selections()
        color = self._effective_color()
        return str(sorted(selections.items())) + f"|color={color}"

    def _context_changed(self) -> bool:
        feature = self._effective_feature()
        trial = getattr(self.app_state, "trials_sel", None)
        sel_hash = self._get_selections_hash()

        return (
            feature != self._current_feature or trial != self._current_trial or sel_hash != self._current_ds_kwargs_hash
        )

    def _update_context(self):
        self._current_feature = self._effective_feature()
        self._current_trial = getattr(self.app_state, "trials_sel", None)
        self._current_ds_kwargs_hash = self._get_selections_hash()

    def _ensure_source(self):
        """Create/update the XarraySource from current app_state."""
        ds = self.app_state.ds
        time_coord = self.app_state.time_coord
        if ds is None or time_coord is None:
            self._buffer.set_source(None)
            return
        bounds = self.app_state.window_bounds
        source = XarraySource(ds, time_coord.name)
        self._buffer.set_source(source, bounds=bounds)

    def _get_buffered_ds(self, t0: float, t1: float):
        """Get buffered dataset slice for the visible time range."""
        if self._context_changed():
            self._buffer.invalidate()
            self._update_context()
            self._ensure_source()

        if self._buffer.source is None:
            self._ensure_source()

        return self._buffer.get(t0, t1)

    @property
    def _store(self):
        return getattr(self.app_state, "data_loader", None)

    def _on_view_range_changed(self):
        if not self.isVisible():
            return
        if self._store is None and (not hasattr(self.app_state, "ds") or self.app_state.ds is None):
            return
        self._td.trigger()

    def _do_range_update(self):
        if not self.isVisible():
            return
        t0, t1 = self.get_current_xlim()
        self._update_plot(t0, t1)

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        clear_plot_items(self.plot_item, self.plot_items)

        if t0 is None or t1 is None:
            t0, t1 = self.get_current_xlim()

        self._update_plot(t0, t1)

    def _update_plot(self, t0: float, t1: float):
        clear_plot_items(self.plot_item, self.plot_items)

        self._ensure_panel_state()
        if self._effective_feature() is None:
            return

        feature_sel = self._effective_feature()
        selections = self._effective_selections()
        color_var = self._effective_color()
        show_cp = getattr(self.app_state, "show_changepoints", False)

        store = self._store
        if store is not None:
            plot_data = store.select(
                feature_sel,
                selections,
                t0=t0,
                t1=t1,
                color_variable=color_var,
            )
        else:
            buffered_ds = self._get_buffered_ds(t0, t1)
            if buffered_ds is None:
                self.app_state.plot_has_changepoints = False
                return
            plot_data = select_feature(buffered_ds, feature_sel, selections, color_var)

        if plot_data is None:
            self.app_state.plot_has_changepoints = False
            return

        self.app_state.plot_has_changepoints = bool(plot_data.changepoints)
        self.plot_items = render_plot_data(
            self.plot_item, plot_data, show_changepoints=show_cp, legend_host=self.legend_host()
        )

        for item in self.plot_items:
            if hasattr(item, "setDownsampling"):
                # Safe against NaN gaps because curves are created with
                # forward-filled data + a `connect` mask (_nan_safe_curve_args);
                # raw NaNs would blank every bin containing one when zoomed out.
                item.setDownsampling(auto=True, method="peak")

    def add_hline(self, value: float) -> pg.InfiniteLine:
        """Draw a horizontal reference line at *value* on this panel."""
        line = pg.InfiniteLine(
            pos=value,
            angle=0,
            movable=False,
            pen=pg.mkPen(color=HLINE_COLOR, width=HLINE_WIDTH, style=Qt.DashLine),
            label=f"{value:g}",
            labelOpts={"position": 0.04, "color": HLINE_COLOR, "fill": (255, 255, 255, 180)},
        )
        # ignoreBounds: a reference line is not data, so it must not stretch
        # the autoranged y-view towards itself.
        self.plot_item.addItem(line, ignoreBounds=True)
        self.hline_items.append(line)
        return line

    def hline_values(self) -> list[float]:
        """Values of this panel's horizontal reference lines, in draw order."""
        return [float(line.value()) for line in self.hline_items]

    def clear_hlines(self) -> None:
        for line in self.hline_items:
            self.plot_item.removeItem(line)
        self.hline_items.clear()

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        if ymin is None and ymax is None:
            return
        if ymin is None or ymax is None:
            cur_lo, cur_hi = self.vb.viewRange()[1]
            ymin = cur_lo if ymin is None else ymin
            ymax = cur_hi if ymax is None else ymax
        self.plot_item.setYRange(ymin, ymax)

    def _apply_y_constraints(self):
        """Apply y-axis constraints based on current feature data."""
        if self._effective_feature() is None:
            return

        feature_sel = self._effective_feature()
        selections = self._effective_selections()

        try:
            store = self._store
            if store is not None:
                wb = self.app_state.window_bounds
                if wb is None:
                    return
                pd = store.select(feature_sel, selections, t0=wb.start_s, t1=wb.end_s)
                if pd is None:
                    return
                data = pd.data
            else:
                data, _ = eto.sel_valid(self.app_state.ds[feature_sel], selections)

            percentile_ylim = self.panel_state.get("percentile")
            if percentile_ylim is None:
                percentile_ylim = self.app_state.get_with_default("percentile_ylim")
            y_min = np.nanpercentile(data, 100 - percentile_ylim)
            y_max = np.nanpercentile(data, percentile_ylim)
            y_range = y_max - y_min
            y_buffer = y_range * 0.2

            if y_range > 0:
                self.vb.setLimits(
                    yMin=y_min - y_buffer,
                    yMax=y_max + y_buffer,
                    minYRange=y_range * 0.1,
                    maxYRange=y_range + y_buffer,
                )
        except (KeyError, AttributeError, ValueError):
            pass


class MultiColoredLineItem(pg.GraphicsObject):
    """Efficient multi-colored line for PyQtGraph.

    Segments are grouped by color and drawn as one QPainterPath per unique
    color, so build/paint cost scales with the number of colors rather than
    the number of samples. Continuous color gradients are quantized to keep
    the path count bounded.
    """

    _MAX_UNIQUE_COLORS = 256

    def __init__(self, x, y, colors, width=2):
        super().__init__()
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.colors = colors
        self.width = width
        self._paths: list[tuple[pg.QtGui.QPen, pg.QtGui.QPainterPath]] = []
        self._bounds = pg.QtCore.QRectF()
        self._build_paths()

    def _segment_colors(self, n_seg: int) -> np.ndarray:
        """Per-segment colors as (n_seg, 3) uint8, white-padded if short."""
        rgb = np.atleast_2d(np.asarray(self.colors, dtype=float))[:n_seg, :3]
        finite = rgb[np.isfinite(rgb).all(axis=1)]
        if finite.size and finite.max() <= 1:
            rgb = rgb * 255
        rgb = np.nan_to_num(rgb, nan=255).clip(0, 255).astype(np.uint8)
        if len(rgb) < n_seg:
            pad = np.full((n_seg - len(rgb), 3), 255, dtype=np.uint8)
            rgb = np.vstack([rgb, pad])
        return rgb

    def _build_paths(self):
        x, y = self.x, self.y
        n_seg = len(x) - 1
        if n_seg < 1:
            return

        rgb = self._segment_colors(n_seg)
        unique, inverse = np.unique(rgb, axis=0, return_inverse=True)
        if len(unique) > self._MAX_UNIQUE_COLORS:
            rgb = (rgb >> 3) << 3
            unique, inverse = np.unique(rgb, axis=0, return_inverse=True)

        # Each segment becomes an independent point pair; connect mask
        # [T, F, T, F, ...] joins only within pairs.
        for k, color in enumerate(unique):
            idx = np.flatnonzero(inverse == k)
            xk = np.empty(2 * len(idx))
            yk = np.empty(2 * len(idx))
            xk[0::2], xk[1::2] = x[idx], x[idx + 1]
            yk[0::2], yk[1::2] = y[idx], y[idx + 1]
            connect = np.zeros(len(xk), dtype=bool)
            connect[0::2] = True
            path = pg.functions.arrayToQPath(xk, yk, connect=connect)
            pen = pg.mkPen(color=tuple(int(c) for c in color), width=self.width)
            self._paths.append((pen, path))

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            xf, yf = x[finite], y[finite]
            self._bounds = pg.QtCore.QRectF(
                pg.QtCore.QPointF(xf.min(), yf.min()),
                pg.QtCore.QPointF(xf.max(), yf.max()),
            )

    def paint(self, painter, *args):
        for pen, path in self._paths:
            painter.setPen(pen)
            painter.drawPath(path)

    def boundingRect(self):
        return self._bounds


def _nan_safe_curve_args(y):
    """Prepare curve data so pyqtgraph's 'peak' downsampling survives NaN gaps.

    Peak (and mean) downsampling propagate NaN — one NaN blanks its whole bin,
    so NaN-gapped curves vanish when zoomed out. Forward-filling keeps min/max
    within real data values, while the returned `connect` mask makes pyqtgraph
    skip the gap segments; its downsampling bins the mask too, so gaps stay
    gaps at any zoom. Returns ``(y, None)`` when no filling is needed.
    """
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.all() or not finite.any():
        return y, None
    idx = np.where(finite, np.arange(len(y)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = y[idx]
    first = np.argmax(finite)
    filled[:first] = y[first]
    connect = finite.copy()
    connect[:-1] &= finite[1:]
    return filled, connect


def plot_multidim(plot_item, time, data, coord_labels=None, existing_curves=None):
    """
    Plot multi-dimensional data (e.g., pos, vel) over time using PyQtGraph.

    Args:
        plot_item: PyQtGraph PlotItem to plot on
        time: time array
        data: shape (time, space)
        coord_labels: list of labels for each dimension (e.g., ['x', 'y', 'z'])
        existing_curves: list to append created curves to

    Returns:
        list of PlotDataItem objects
    """
    if existing_curves is None:
        existing_curves = []

    colors = MULTIDIM_COLORS

    for i in range(data.shape[1]):
        label = coord_labels[i] if coord_labels is not None else f"dim {i}"
        color = colors[i % len(colors)]

        y, connect = _nan_safe_curve_args(data[:, i])
        opts = {"connect": connect} if connect is not None else {}
        curve = plot_item.plot(time, y, pen=pg.mkPen(color=color, width=2), name=label, **opts)
        existing_curves.append(curve)

    return existing_curves


def plot_singledim(
    plot_item,
    time,
    data,
    color_data=None,
    changepoints_dict=None,
    existing_items=None,
    show_changepoints=True,
):
    if existing_items is None:
        existing_items = []

    if color_data is not None and color_data.ndim == 2 and color_data.shape[1] == 3:
        multi_line = MultiColoredLineItem(time, data, color_data)
        plot_item.addItem(multi_line)
        existing_items.append(multi_line)
    else:
        y, connect = _nan_safe_curve_args(data)
        opts = {"connect": connect} if connect is not None else {}
        curve = plot_item.plot(
            time,
            y,
            pen=pg.mkPen(color="k", width=2),
            **opts,
        )
        existing_items.append(curve)

    # Add changepoints as scatter plots, each with its own color and label
    if changepoints_dict is not None and show_changepoints:
        cmap = matplotlib.colormaps.get_cmap("tab10")

        colors = [tuple(int(c * 255) for c in cmap.colors[i][:3]) for i in range(len(cmap.colors))]

        for i, (cp_name, cp_array) in enumerate(changepoints_dict.items()):
            idxs = np.where(cp_array)[0]
            color = colors[(i + 5) % len(colors)]  # offset to match original
            if len(idxs) > 0:
                scatter = pg.ScatterPlotItem(
                    x=time[idxs],
                    y=data[idxs],
                    pen=pg.mkPen(color=color, width=2),
                    brush=None,
                    symbol="o",
                    size=10,
                    name=cp_name,
                )
                plot_item.addItem(scatter)
                existing_items.append(scatter)

    return existing_items


def select_feature(ds, variable, ds_kwargs, color_variable=None) -> PlotData | None:
    """Extract plot-ready data from an xarray Dataset via XarrayLoader."""
    from ethograph.io.catalog import XarrayLoader

    return XarrayLoader(ds).select(variable, ds_kwargs, color_variable=color_variable)


def drop_legend(plot_item) -> None:
    """Take the panel's legend out of the scene.

    ``PlotItem.removeItem`` ignores the legend (it is a child of the viewbox,
    never one of ``items``), so it has to be unparented by hand — otherwise
    each render stacks a fresh legend on top of the last.
    """
    legend = getattr(plot_item, "legend", None)
    if legend is None:
        return
    scene = legend.scene()
    if scene is not None:
        scene.removeItem(legend)  # detaches from its parent too (C++ side)
    else:
        pg.GraphicsWidget.setParentItem(legend, None)  # LegendItem's own override re-anchors
    plot_item.legend = None


def add_legend(plot_item, legend_host=None) -> pg.LegendItem:
    """Give the panel a legend — in the right gutter when the panel offers
    one (``BasePlot.legend_host``), else over the plotting rectangle."""
    if legend_host is None:
        plot_item.legend = plot_item.addLegend(offset=(10, 10))
    else:
        legend = pg.LegendItem(offset=(2, 2))
        legend.setParentItem(legend_host)
        plot_item.legend = legend
    return plot_item.legend


def render_plot_data(plot_item, plot_data: PlotData, show_changepoints=True, legend_host=None) -> list:
    """Render a PlotData to a pyqtgraph PlotItem. Source-agnostic."""
    drop_legend(plot_item)

    items = []
    dim_labels = plot_data.dim_labels
    if dim_labels:
        dim_labels = clean_display_labels(dim_labels)

    if plot_data.data.ndim == 2:
        add_legend(plot_item, legend_host)
        items = plot_multidim(
            plot_item,
            plot_data.time,
            plot_data.data,
            dim_labels,
            items,
        )
    elif plot_data.data.ndim == 1:
        if plot_data.changepoints and show_changepoints:
            add_legend(plot_item, legend_host)

        items = plot_singledim(
            plot_item,
            plot_data.time,
            plot_data.data,
            color_data=plot_data.color_data,
            changepoints_dict=plot_data.changepoints,
            existing_items=items,
            show_changepoints=show_changepoints,
        )
    else:
        logger.warning("Data ndim=%d not supported for plotting", plot_data.data.ndim)

    plot_item.setLabel("left", plot_data.ylabel, Fontsize="14pt")
    plot_item.setTitle(plot_data.title)

    return items


def plot_ds_variable(plot_item, ds, ds_kwargs, variable, color_variable=None, show_changepoints=True):
    """Plot a variable from an xarray Dataset.

    Delegates to :func:`select_feature` + :func:`render_plot_data`.
    """
    plot_data = select_feature(ds, variable, ds_kwargs, color_variable)
    if plot_data is None:
        return []
    return render_plot_data(plot_item, plot_data, show_changepoints=show_changepoints)


def clear_plot_items(plot_item, items_list):
    """Helper function to clear specific plot items from a plot with proper cleanup."""
    for item in items_list:
        plot_item.removeItem(item)

        if isinstance(item, MultiColoredLineItem):
            item._paths.clear()
            item.x = None
            item.y = None
            item.colors = None

        elif isinstance(item, pg.ScatterPlotItem):
            item.clear()
            item.setData([], [])

        elif isinstance(item, (pg.PlotDataItem, pg.PlotCurveItem)):
            item.clear()

        elif isinstance(item, pg.InfiniteLine):
            if hasattr(item, "sigPositionChanged"):
                try:
                    item.sigPositionChanged.disconnect()
                except (TypeError, RuntimeError):
                    pass

        item.setParentItem(None)
        if hasattr(item, "deleteLater"):
            item.deleteLater()

    items_list.clear()
