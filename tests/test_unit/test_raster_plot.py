"""The raster panel picks ticks or the density image from what is in view."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")

from ethograph.gui.plots_raster import RasterPlot  # noqa: E402

GREY = (180, 180, 180, 200)


def _raster(qtbot, app_state, n_rows: int, spikes_per_row: int, t_end: float = 100.0) -> RasterPlot:
    plot = RasterPlot(app_state)
    qtbot.addWidget(plot)
    plot.resize(800, 300)
    rng = np.random.default_rng(0)
    times = rng.uniform(0.0, t_end, n_rows * spikes_per_row)
    rows = np.repeat(np.arange(n_rows), spikes_per_row)
    plot.sync_y_axis({i: float(n_rows - 1 - i) for i in range(n_rows)}, 1.0, n_rows)
    plot.set_multi_cluster_spike_data([(times, rows, GREY)])
    return plot


def _show(plot: RasterPlot, x_range: tuple[float, float], y_range: tuple[float, float] | None = None) -> None:
    plot.vb.setXRange(*x_range, padding=0)
    if y_range is not None:
        plot.vb.setYRange(*y_range, padding=0)
    plot._redraw()


def test_a_sparse_view_is_ticks_and_a_crowded_one_is_density(qtbot, app_state):
    plot = _raster(qtbot, app_state, n_rows=20, spikes_per_row=20_000)

    _show(plot, (0.0, 100.0))
    assert plot.render == "density"
    assert plot._image_item.isVisible() and not plot._tick_items

    _show(plot, (50.0, 50.5))
    assert plot.render == "ticks"
    assert plot._tick_items and not plot._image_item.isVisible()


def test_zooming_into_fewer_rows_brings_the_ticks_back(qtbot, app_state):
    """With rows thinner than a pixel the cell is a pixel row, so fewer rows means fewer spikes per cell."""
    plot = _raster(qtbot, app_state, n_rows=3000, spikes_per_row=60)

    _show(plot, (0.0, 100.0))
    assert plot.render == "density"

    _show(plot, (0.0, 100.0), y_range=(1000.0, 1100.0))
    assert plot.render == "ticks"


def test_the_users_mode_overrides_the_view(qtbot, app_state):
    plot = _raster(qtbot, app_state, n_rows=20, spikes_per_row=200)
    _show(plot, (0.0, 100.0))
    assert plot.render == "ticks"

    app_state.raster_render_mode = "density"
    plot.refresh()
    assert plot.render == "density"


def test_a_set_tick_width_is_used_as_it_is_and_auto_follows_the_view(qtbot, app_state):
    plot = _raster(qtbot, app_state, n_rows=10, spikes_per_row=2_000)

    _show(plot, (50.0, 50.5))
    sparse = plot.tick_width
    _show(plot, (0.0, 10.0))
    assert plot.render == "ticks"
    assert plot.tick_width < sparse

    app_state.raster_tick_width = 5
    plot.refresh()
    assert plot.tick_width == 5
    assert all(item.opts["pen"].width() == 5 for item in plot._tick_items)


def test_redrawing_the_same_rows_keeps_the_zoom(qtbot, app_state):
    plot = _raster(qtbot, app_state, n_rows=50, spikes_per_row=10)
    plot.vb.setYRange(10.0, 20.0, padding=0)

    plot.sync_y_axis({i: float(49 - i) for i in range(50)}, 1.0, 50)

    assert plot.vb.viewRange()[1] == pytest.approx([10.0, 20.0])
