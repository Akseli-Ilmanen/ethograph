"""Every stacked panel ends its plotting rectangle on the same pixel.

A heatmap carries a colorbar on its right; a line plot does not. Rather than
measure and match, every panel reserves the colorbar's footprint as a fixed
right gutter (``PANEL_RIGHT_GUTTER_PX``), where a line plot parks its legend.
"""

import numpy as np
import pyqtgraph as pg
import pytest

from ethograph.gui.app_constants import (
    COLORBAR_WIDTH_PX,
    PANEL_RIGHT_GUTTER_PX,
    PANEL_RIGHT_SPACER_PX,
)
from ethograph.gui.plots_base import right_gutter_width
from ethograph.gui.plots_heatmap import HeatmapPlot
from ethograph.gui.plots_lineplot import LinePlot, PlotData, add_legend, drop_legend, render_plot_data

WIDTH = 800


def _settle(qapp, *widgets):
    for w in widgets:
        w.resize(WIDTH, 300)
        w.show()
    for _ in range(3):
        qapp.processEvents()


def _vb_right(plot) -> float:
    return plot.plot_item.vb.geometry().right()


def test_gutter_is_the_colorbar_footprint(qapp):
    """Contract with pyqtgraph: a ColorBarItem inserted into a PlotItem takes
    exactly the spacer + gutter we reserve — whatever its tick labels say."""
    widget = pg.PlotWidget()
    image = pg.ImageItem()
    widget.plotItem.addItem(image)
    bar = pg.ColorBarItem(
        values=(-1, 1), colorMap=pg.colormap.get("viridis"), interactive=False, width=COLORBAR_WIDTH_PX
    )
    bar.setImageItem(image, insert_in=widget.plotItem)
    _settle(qapp, widget)

    layout = widget.plotItem.layout
    assert layout.columnMinimumWidth(4) == PANEL_RIGHT_SPACER_PX
    assert bar.geometry().width() == PANEL_RIGHT_GUTTER_PX

    bar.setLevels(values=(-0.123456, 0.98765))
    _settle(qapp, widget)
    assert bar.geometry().width() == PANEL_RIGHT_GUTTER_PX


def test_heatmap_and_lineplot_end_on_the_same_pixel(qapp, app_state):
    line = LinePlot(app_state)
    heat = HeatmapPlot(app_state)
    _settle(qapp, line, heat)
    assert _vb_right(line) == _vb_right(heat)
    assert _vb_right(line) == pytest.approx(WIDTH - PANEL_RIGHT_GUTTER_PX - PANEL_RIGHT_SPACER_PX, abs=2)


def test_visible_right_axis_trims_the_gutter(qapp, app_state):
    plain = LinePlot(app_state)
    scaled = LinePlot(app_state)
    scaled.plot_item.showAxis("right")
    _settle(qapp, plain, scaled)
    assert right_gutter_width(plain) == PANEL_RIGHT_GUTTER_PX
    assert right_gutter_width(scaled) < PANEL_RIGHT_GUTTER_PX

    scaled.reserve_right_gutter(right_gutter_width(scaled))
    _settle(qapp, plain, scaled)
    assert _vb_right(plain) == _vb_right(scaled)


def test_colorbar_panel_offers_no_legend_host(qapp, app_state):
    assert HeatmapPlot(app_state).legend_host() is None
    line = LinePlot(app_state)
    host = line.legend_host()
    assert host is not None
    assert line.legend_host() is host


def test_legend_lives_in_the_gutter_and_never_stacks(qapp, app_state):
    line = LinePlot(app_state)
    _settle(qapp, line)
    t = np.linspace(0.0, 1.0, 50)
    data = PlotData(time=t, data=np.stack([t, t * 2], axis=1), dim_labels=["a", "b"], title="f", ylabel="y")
    render_plot_data(line.plot_item, data, legend_host=line.legend_host())
    legend = line.plot_item.legend
    assert legend.parentItem() is line.legend_host()

    render_plot_data(line.plot_item, data, legend_host=line.legend_host())
    assert line.plot_item.legend is not legend
    assert legend.scene() is None  # the old one left the scene


def test_legend_without_host_overlays_the_viewbox(qapp, app_state):
    line = LinePlot(app_state)
    add_legend(line.plot_item)
    assert line.plot_item.legend.parentItem() is line.plot_item.vb
    drop_legend(line.plot_item)
    assert line.plot_item.legend is None


# ---------------------------------------------------------------------------
# A legend wider than the gutter widens every panel's gutter, never one alone.
# Changepoint names such as ``distance_decrease_onset_cp`` are far wider than
# the colorbar footprint; the panel showing them once ended its time axis
# 140 px left of its neighbours.
# ---------------------------------------------------------------------------

_LONG_CPS = {
    "distance_decrease_onset_cp": np.array([0.2, 0.5]),
    "distance_decrease_end_cp": np.array([0.3]),
}


def _render_changepoints(line: LinePlot) -> None:
    t = np.linspace(0.0, 1.0, 50)
    data = PlotData(time=t, data=t, title="f", ylabel="y", changepoints=_LONG_CPS)
    render_plot_data(line.plot_item, data, legend_host=line.legend_host())


def _align(container, qapp, *panels) -> None:
    """Run the container's right-inset pass over ``panels`` as its visible set."""
    container._visible_plots = lambda: iter(panels)
    container._align_axes_right()
    for _ in range(3):
        qapp.processEvents()


@pytest.fixture
def container():
    from ethograph.gui.plots_container import UnifiedPanelContainer

    return UnifiedPanelContainer.__new__(UnifiedPanelContainer)


def test_long_legend_reports_its_need_and_does_not_grow_alone(qapp, app_state):
    line = LinePlot(app_state)
    plain = LinePlot(app_state)
    _settle(qapp, line, plain)
    assert line.gutter_need() == 0

    _render_changepoints(line)
    _settle(qapp, line, plain)
    assert line.gutter_need() > PANEL_RIGHT_GUTTER_PX
    assert _vb_right(line) == _vb_right(plain)  # the gutter column stays fixed

    drop_legend(line.plot_item)
    assert line.gutter_need() == 0


def test_legend_growth_announces_itself(qapp, app_state):
    line = LinePlot(app_state)
    _settle(qapp, line)
    fired = []
    line.gutter_need_changed.connect(lambda: fired.append(True))
    _render_changepoints(line)
    assert fired


@pytest.mark.parametrize(
    "others",
    [
        pytest.param(("line",), id="line+line"),
        pytest.param(("heatmap",), id="line+heatmap"),
        pytest.param(("line", "heatmap"), id="line+line+heatmap"),
        pytest.param(("right_axis",), id="line+right-axis-line"),
        pytest.param(("line", "right_axis", "heatmap"), id="all"),
    ],
)
def test_changepoint_legend_keeps_every_panel_type_aligned(qapp, app_state, container, others):
    def make(kind):
        if kind == "heatmap":
            return HeatmapPlot(app_state)
        plot = LinePlot(app_state)
        if kind == "right_axis":
            plot.plot_item.showAxis("right")
        return plot

    with_cps = LinePlot(app_state)
    panels = [with_cps, *(make(k) for k in others)]
    _settle(qapp, *panels)
    _align(container, qapp, *panels)
    before = {_vb_right(p) for p in panels}
    assert len(before) == 1

    _render_changepoints(with_cps)
    _align(container, qapp, *panels)
    rights = {_vb_right(p) for p in panels}
    assert len(rights) == 1, [(_vb_right(p), type(p).__name__) for p in panels]
    # The common gutter grew to fit the legend, so nothing is clipped.
    assert rights.pop() < before.pop()
    assert with_cps.plot_item.legend.width() <= with_cps.legend_host().geometry().width()

    drop_legend(with_cps.plot_item)
    _align(container, qapp, *panels)
    assert len({_vb_right(p) for p in panels}) == 1
