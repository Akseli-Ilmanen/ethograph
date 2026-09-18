"""The heatmap applies the peak-window order: on demand for the visible window,
automatically per trial in trial mode, and drops it in "none" mode."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")

from ethograph.gui.plots_heatmap import HeatmapPlot  # noqa: E402
from ethograph.gui.widgets_plot_settings import PlotSettingsWidget  # noqa: E402


def _make_heatmap(qtbot, app_state, monkeypatch) -> HeatmapPlot:
    plot = HeatmapPlot(app_state)
    qtbot.addWidget(plot)
    # No loader in this test: the buffer filled below stands in for the data.
    monkeypatch.setattr(plot, "_get_buffered_data", lambda t0, t1: (plot._buffered_data, plot._buffered_time))
    monkeypatch.setattr(plot, "_effective_feature", lambda: "speed")
    monkeypatch.setattr(plot, "_get_selections_hash", lambda: "")
    return plot


@pytest.fixture
def heatmap(qtbot, app_state, monkeypatch):
    app_state.heatmap_normalization = "none"
    app_state.heatmap_sort_window_s = 1.0
    app_state.heatmap_sort_overlap = 0.5
    return _make_heatmap(qtbot, app_state, monkeypatch)


def _fill_buffer(plot, peaks_at: list[float], t_end: float = 10.0):
    time = np.linspace(0.0, t_end, int(t_end * 100) + 1)
    data = np.stack([np.exp(-((time - p) ** 2) / 0.1) for p in peaks_at], axis=1)
    plot._buffered_data = data
    plot._buffered_time = time
    plot._buffer_t0, plot._buffer_t1 = 0.0, t_end
    plot._n_channels = data.shape[1]
    plot._channel_labels = [f"ch{i}" for i in range(data.shape[1])]
    plot._normalize_buffer()


def test_sort_for_current_window_orders_by_the_visible_range_only(heatmap):
    _fill_buffer(heatmap, peaks_at=[8.0, 1.0, 5.0])
    # Row 1 peaks at 1 s but rises slowly after: over the trial its peak is
    # early, inside [4, 9] its largest window is the last one.
    heatmap._buffered_data[:, 1] += 0.05 * heatmap._buffered_time
    heatmap._normalize_buffer()

    heatmap.plot_item.setXRange(0.0, 10.0, padding=0)
    assert heatmap.sort_by_visible_window()
    assert heatmap._sort_order.tolist() == [1, 2, 0]

    heatmap.plot_item.setXRange(4.0, 9.0, padding=0)
    assert heatmap.sort_by_visible_window()
    assert heatmap._sort_order.tolist() == [2, 0, 1]


def test_trial_mode_resorts_once_per_context(heatmap, app_state, monkeypatch):
    app_state.heatmap_sort_mode = "trial"
    _fill_buffer(heatmap, peaks_at=[8.0, 1.0, 5.0])
    monkeypatch.setattr(heatmap, "_trial_sort_range", lambda: (0.0, 10.0))

    heatmap._render_heatmap(0.0, 2.0)
    assert heatmap._sort_order.tolist() == [1, 2, 0]

    # Same context: a pan does not re-sort even if the buffer changed.
    _fill_buffer(heatmap, peaks_at=[1.0, 5.0, 8.0])
    heatmap._render_heatmap(2.0, 4.0)
    assert heatmap._sort_order.tolist() == [1, 2, 0]

    # A new trial does.
    app_state.trials_sel = "other"
    heatmap._render_heatmap(0.0, 2.0)
    assert heatmap._sort_order.tolist() == [0, 1, 2]


def test_none_mode_drops_the_order(heatmap):
    _fill_buffer(heatmap, peaks_at=[8.0, 1.0])
    heatmap.set_sort_order(np.array([1, 0]))
    heatmap.set_sort_order(None)
    assert heatmap._sort_order is None


def test_row_window_takes_neighbours_in_sorted_order(heatmap, app_state):
    _fill_buffer(heatmap, peaks_at=[8.0, 1.0, 5.0, 3.0])
    heatmap.set_sort_order(np.array([1, 3, 2, 0]))
    app_state.heatmap_row_percent = 50.0
    app_state.heatmap_row_position = 1.0
    heatmap.refresh_row_window()
    assert heatmap._last_visible_labels == ["ch2", "ch0"]
    assert heatmap.image_item.image.shape[1] == 2


def test_sort_button_sorts_only_the_active_heatmap(heatmap, qtbot, app_state, monkeypatch):
    other = _make_heatmap(qtbot, app_state, monkeypatch)
    _fill_buffer(heatmap, peaks_at=[8.0, 1.0, 5.0])
    _fill_buffer(other, peaks_at=[1.0, 5.0, 8.0])
    for plot in (heatmap, other):
        plot.plot_item.setXRange(0.0, 10.0, padding=0)
    container = SimpleNamespace(heatmap_plots=[heatmap, other], heatmap_plot=heatmap)
    settings = SimpleNamespace(plot_container=container)

    PlotSettingsWidget._on_heatmap_sort_now_clicked(settings)
    assert heatmap._sort_order.tolist() == [1, 2, 0]
    assert other._sort_order is None

    # Sorting the next panel leaves the first one's order alone.
    _fill_buffer(heatmap, peaks_at=[1.0, 5.0, 8.0])
    container.heatmap_plot = other
    PlotSettingsWidget._on_heatmap_sort_now_clicked(settings)
    assert other._sort_order.tolist() == [0, 1, 2]
    assert heatmap._sort_order.tolist() == [1, 2, 0]
