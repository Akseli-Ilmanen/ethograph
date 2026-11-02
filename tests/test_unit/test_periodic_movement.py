"""Periodic-movement scores: a frame-rate alternation reads high, a smooth sweep reads low."""

import numpy as np

from ethograph.features.movement import alternation, flip_rate, high_band_ratio, jitter_features, periodic_valid_mask

WINDOW = 14


def _signals() -> tuple[np.ndarray, np.ndarray]:
    n = 200
    rng = np.random.default_rng(0)
    smooth = np.sin(np.linspace(0, 4 * np.pi, n)) + 0.01 * rng.standard_normal(n)
    quiver = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) + 0.05 * rng.standard_normal(n)
    return smooth, quiver


def test_alternation_separates_quiver_from_sweep() -> None:
    smooth, quiver = _signals()
    assert np.nanmedian(alternation(quiver, WINDOW)) > 0.8
    assert np.nanmedian(alternation(smooth, WINDOW)) < 0.2


def test_jitter_score_separates_quiver_from_sweep() -> None:
    smooth, quiver = _signals()
    assert np.nanmedian(jitter_features(quiver, WINDOW)["jitter_score"]) > 0.7
    assert np.nanmedian(jitter_features(smooth, WINDOW)["jitter_score"]) < 0.1
    assert np.nanmedian(flip_rate(quiver, WINDOW)) > 0.9
    assert np.nanmedian(high_band_ratio(quiver, WINDOW)) > 0.9


def test_nan_gap_is_masked_with_half_window_margin() -> None:
    _, quiver = _signals()
    quiver[100] = np.nan
    valid = periodic_valid_mask(quiver, WINDOW)
    half = WINDOW // 2 + 1
    assert not valid[100 - half : 100 + half + 1].any()
    assert valid[100 - half - 1] and valid[100 + half + 1]
    assert np.isnan(alternation(quiver, WINDOW)[100]) and np.isnan(jitter_features(quiver, WINDOW)["flip_rate"][100])
