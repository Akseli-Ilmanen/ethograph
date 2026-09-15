"""Heatmap row order: earliest peak window on top, by a windowed argmax."""

import numpy as np
import pytest

from ethograph.gui.heatmap_sort import argmax_window_order, window_starts


def test_windows_tile_the_range_with_the_requested_overlap():
    starts = window_starts(0.0, 10.0, window_s=2.0, overlap=0.5)
    np.testing.assert_allclose(starts, np.arange(0.0, 8.5, 1.0))
    assert len(window_starts(0.0, 0.5, window_s=2.0, overlap=0.5)) == 1


def test_rows_sort_by_where_their_peak_window_is():
    time = np.linspace(0.0, 10.0, 1001)
    peaks_at = [8.0, 1.0, 5.0]
    data = np.stack([np.exp(-((time - p) ** 2) / 0.1) for p in peaks_at], axis=1)
    order = argmax_window_order(data, time, window_s=1.0, overlap=0.5)
    assert order.tolist() == [1, 2, 0]


def test_windowed_argmax_ignores_a_single_outlier_sample():
    time = np.linspace(0.0, 10.0, 1001)
    early_bump = np.exp(-((time - 2.0) ** 2) / 0.5)
    late_spike = np.zeros_like(time)
    late_spike[900] = 50.0
    data = np.stack([early_bump + late_spike, np.exp(-((time - 6.0) ** 2) / 0.5)], axis=1)
    order = argmax_window_order(data, time, window_s=1.0, overlap=0.5)
    assert order.tolist() == [0, 1]


def test_ties_keep_original_order_and_nan_rows_go_last():
    time = np.linspace(0.0, 4.0, 41)
    same = np.exp(-((time - 1.0) ** 2))
    data = np.stack([np.full_like(time, np.nan), same, same], axis=1)
    order = argmax_window_order(data, time, window_s=1.0, overlap=0.5)
    assert order.tolist() == [1, 2, 0]


def test_bad_parameters_are_refused():
    with pytest.raises(ValueError):
        window_starts(0.0, 1.0, window_s=0.0, overlap=0.5)
    with pytest.raises(ValueError):
        window_starts(0.0, 1.0, window_s=1.0, overlap=1.0)
