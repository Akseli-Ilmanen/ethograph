"""Periods laid under the raster's rows: one set at a time, behind the spikes."""

from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.plots_raster import RasterPlot


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def raster(qapp, tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    plot = RasterPlot(state)
    plot.sync_y_axis({0: 2.0, 1: 1.0, 2: 0.0}, 1.0, 3)
    yield plot
    plot.close()


def test_bands_sit_on_their_rows_behind_the_spikes_and_skip_rows_not_shown(raster):
    raster.set_intervals(
        np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.7, 1.0]), np.array([0, 2, 7]), [(70, 130, 230, 60)] * 3
    )
    opts = raster._interval_item.opts
    assert raster.n_intervals == 2  # row 7 is not on the raster
    assert list(opts["y0"]) == pytest.approx([2.0 - 0.45, 0.0 - 0.45])
    assert raster._interval_item.zValue() < min(item.zValue() for item in [raster._image_item])


def test_a_new_set_replaces_the_bands_and_rows_moving_moves_them(raster):
    raster.set_intervals(np.array([0.1]), np.array([0.2]), np.array([1]), [(0, 0, 0, 60)])
    raster.set_intervals(np.array([0.3, 0.4]), np.array([0.35, 0.45]), np.array([0, 1]), [(0, 0, 0, 60)] * 2)
    assert raster.n_intervals == 2
    raster.sync_y_axis({0: 0.0, 1: 1.0}, 1.0, 2)  # rows reordered
    assert list(raster._interval_item.opts["y0"]) == pytest.approx([-0.45, 0.55])
    raster.clear_intervals()
    assert raster.n_intervals == 0
