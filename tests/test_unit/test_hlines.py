"""Horizontal reference lines on a line plot.

A line drawn from the sidebar is a *reference*, not data: it must survive
every re-render the panel does (trial change, feature switch, zoom) and it
must not drag the autoranged y-view towards itself.
"""

import numpy as np
import pytest

pg = pytest.importorskip("pyqtgraph")

import xarray as xr  # noqa: E402

import ethograph as eto  # noqa: E402
from ethograph.gui.plots_lineplot import LinePlot  # noqa: E402
from ethograph.io.catalog import XarrayLoader, catalog_from_xarray  # noqa: E402
from ethograph.io.time_model import TimeRange  # noqa: E402


@pytest.fixture
def plot(qtbot):
    time = np.linspace(0.0, 10.0, 1001)
    ds = xr.Dataset(
        {"speed": (("time", "individuals"), np.sin(time)[:, None])},
        coords={"time": time, "individuals": ["a"]},
        attrs={"trial": 1},
    )
    catalog = catalog_from_xarray(ds, eto.from_datasets([ds]))

    class _State:
        ds_ = ds
        data_loader = XarrayLoader(ds, catalog)
        window_bounds = TimeRange(0.0, 10.0)
        features_sel = "speed"
        colors_sel = None
        trials_sel = 1
        lock_axes = False
        plot_has_changepoints = False
        time_coord = ds["time"]

        def get_selections(self):
            return {}

        def panel_individual(self, _panel):
            return None

    state = _State()
    state.ds = ds
    widget = LinePlot(state)
    qtbot.addWidget(widget)
    widget.update_plot_content(0.0, 10.0)
    return widget


def test_a_line_is_drawn_at_the_requested_value(plot):
    plot.add_hline(0.5)
    plot.add_hline(-1.25)

    assert plot.hline_values() == [0.5, -1.25]
    assert all(line in plot.plot_item.items for line in plot.hline_items)


def test_lines_survive_a_re_render(plot):
    """Trial changes re-render the panel; the reference lines stay put."""
    plot.add_hline(0.5)
    line = plot.hline_items[0]

    plot.app_state.trials_sel = 2
    plot.update_plot_content(0.0, 10.0)

    assert plot.hline_values() == [0.5]
    assert line in plot.plot_item.items
    assert line.scene() is plot.plot_item.scene()


def test_a_line_does_not_stretch_the_autoranged_view(plot):
    """The data spans [-1, 1]; a line at 500 must not blow the y-range open."""
    plot.add_hline(500.0)
    plot.autoscale()
    plot.vb.updateAutoRange()

    _, (ylo, yhi) = plot.vb.viewRange()
    assert yhi < 10.0
    assert ylo > -10.0


def test_clearing_removes_every_line(plot):
    plot.add_hline(0.5)
    plot.add_hline(1.5)
    lines = list(plot.hline_items)

    plot.clear_hlines()

    assert plot.hline_values() == []
    assert not any(line in plot.plot_item.items for line in lines)
