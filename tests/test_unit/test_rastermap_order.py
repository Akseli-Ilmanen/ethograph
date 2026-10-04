"""Rastermap as a row order: similar rows end up neighbours, whatever the matrix holds."""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.gui.heatmap_sort import RASTERMAP_MIN_ROWS, rastermap_order


def _phase_shifted_rows(n_rows: int, n_samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """``(T, C)`` sines whose phase is the only thing telling the rows apart, in shuffled order."""
    rng = np.random.default_rng(0)
    phase = rng.uniform(0.0, 2.0 * np.pi, n_rows)
    time = np.linspace(0.0, 20.0, n_samples)
    data = np.sin(time[:, None] + phase[None, :]) + 0.2 * rng.standard_normal((n_samples, n_rows))
    return data, phase


def _mean_neighbour_distance(phase: np.ndarray) -> float:
    return float(np.abs(np.angle(np.exp(1j * np.diff(phase)))).mean())


@pytest.mark.parametrize("n_rows", [40, 300])
def test_neighbours_in_the_order_have_similar_activity(n_rows):
    """Both of Rastermap's paths: rows sorted directly, and rows clustered first."""
    data, phase = _phase_shifted_rows(n_rows)

    order = rastermap_order(data)

    assert sorted(order.tolist()) == list(range(n_rows))
    assert _mean_neighbour_distance(phase[order]) < 0.5 * _mean_neighbour_distance(phase)


def test_silent_and_missing_rows_are_ordered_last_not_refused():
    data, _ = _phase_shifted_rows(30)
    data[:, 4] = 0.0
    data[:, 11] = np.nan
    data[::7, 2] = np.nan  # gaps in an otherwise live row

    order = rastermap_order(data)

    assert sorted(order.tolist()) == list(range(30))
    assert set(order[-2:].tolist()) == {4, 11}


def test_too_few_rows_are_refused():
    data, _ = _phase_shifted_rows(RASTERMAP_MIN_ROWS - 1)
    with pytest.raises(ValueError, match="at least"):
        rastermap_order(data)
    data, _ = _phase_shifted_rows(RASTERMAP_MIN_ROWS + 2)
    data[:, :5] = 1.0  # constant: nothing to place them by
    with pytest.raises(ValueError, match="vary over time"):
        rastermap_order(data)
