"""The shared curve statistics, and choosing one on a held-out record.

What earns a test: the shape terms must move for a rival or a smear where the
height does not; the AUC must rank a separating statistic above a useless
one; and the choice must not flip on a coin — the default stays unless
something clearly beats it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.labels.confidence import (
    STATISTICS,
    CurveStats,
    choose_statistic,
    curve_stats,
    entropy_confidence,
    focus_window_s,
    rank_auc,
    rank_statistics,
    segment_confidence,
    window_samples,
)


def _bump(length, centre, height, width=3):
    x = np.arange(length)
    return height * np.exp(-0.5 * ((x - centre) / width) ** 2)


class TestFrameWise:
    def test_entropy_confidence_is_one_when_certain_and_zero_when_uniform(self):
        probs = np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.0]])
        conf = entropy_confidence(probs)
        assert conf.dtype == np.float32
        assert conf[0] == pytest.approx(1.0, abs=1e-6)
        assert conf[1] == pytest.approx(0.0, abs=1e-6)
        assert 0.0 < conf[2] < 1.0

    def test_segment_confidence_is_the_mean_inside_else_the_peak(self):
        time = np.arange(5, dtype=float)
        conf = np.array([0.1, 0.2, 0.4, 0.6, 0.9])
        assert segment_confidence(conf, time, 1.0, 3.0) == pytest.approx(0.4)
        assert segment_confidence(conf, time, 10.0, 11.0) == pytest.approx(0.9)


class TestStats:
    def test_one_clean_bump_scores_high_on_every_statistic(self):
        s = curve_stats(_bump(500, 250, 0.9), window=10)
        assert s.index == pytest.approx(250, abs=1)
        assert all(s.statistic(name) > 0.8 for name in STATISTICS)

    def test_a_rival_lowers_ratio_and_shape_but_not_peak(self):
        alone = curve_stats(_bump(500, 250, 0.9), window=10)
        rival = curve_stats(_bump(500, 250, 0.9) + _bump(500, 100, 0.85), window=10)
        assert rival.peak == pytest.approx(alone.peak, abs=0.02)
        assert rival.ratio < 0.1 and rival.shape < alone.shape

    def test_a_broad_lone_bump_has_no_rival(self):
        broad = curve_stats(_bump(500, 250, 0.9, width=60), window=10)
        assert broad.ratio > 0.95  # its own flank outside the window is not another candidate
        assert broad.focus < 0.5

    def test_a_smear_lowers_focus_but_not_peak(self):
        sharp = curve_stats(_bump(500, 250, 0.9, width=3), window=10)
        broad = curve_stats(_bump(500, 250, 0.9, width=60), window=10)
        assert sharp.peak == pytest.approx(broad.peak, abs=1e-6)
        assert broad.focus < sharp.focus

    def test_unknown_statistic_is_refused(self):
        with pytest.raises(ValueError, match="unknown confidence statistic"):
            CurveStats(0, 1.0, 1.0, 1.0).statistic("mass")

    def test_a_curve_near_zero_everywhere_reads_zero_whatever_its_shape(self):
        blip = np.zeros(500)
        blip[250] = 0.02  # the cleanest bump imaginable, and nothing behind it
        s = curve_stats(blip, window=10)
        assert s.peak == pytest.approx(0.02) and not s.found
        assert all(s.statistic(name) == 0.0 for name in STATISTICS)  # even "peak" reads 0: nothing was found

    def test_a_curve_still_climbing_at_the_edge_is_not_an_event(self):
        rising = np.linspace(0.0, 0.9, 500)
        s = curve_stats(rising, window=10)
        assert s.focus == 0.0 and s.ratio == 0.0 and s.shape == 0.0

    def test_an_edge_rising_above_an_interior_bump_is_a_rival(self):
        curve = _bump(500, 250, 0.4) + np.linspace(0.0, 0.9, 500) ** 8  # small bump, then climbing to 0.9 at the end
        s = curve_stats(curve, window=10)
        assert s.index == pytest.approx(250, abs=1)  # the event is still the interior peak
        assert not s.found and all(s.statistic(name) == 0.0 for name in STATISTICS)  # every rule reads 0

    def test_empty_and_flat_curves_are_zero(self):
        assert curve_stats(np.array([]), 5).shape == 0.0
        assert curve_stats(np.zeros(50), 5).peak == 0.0

    def test_window_is_a_duration(self):
        assert window_samples(0.1, 200.0) == 20 and window_samples(0.1, 25.0) == 2

    def test_focus_window_follows_the_label_tolerance(self):
        assert focus_window_s(0.05) == pytest.approx(0.1)
        with pytest.raises(ValueError):
            focus_window_s(0.0)


class TestRanking:
    def test_auc_of_a_separating_score_is_one_and_of_a_useless_one_a_half(self):
        hits = np.array([1, 1, 1, 0, 0, 0], bool)
        assert rank_auc(np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]), hits) == 1.0
        assert rank_auc(np.array([0.5] * 6), hits) == pytest.approx(0.5)
        assert np.isnan(rank_auc(np.array([1.0, 2.0]), np.array([True, True])))

    def test_rank_statistics_scores_every_name(self):
        stats = [
            CurveStats(0, p, f, r) for p, f, r in ((0.9, 0.9, 0.9), (0.8, 0.2, 0.2), (0.7, 0.9, 0.8), (0.9, 0.1, 0.1))
        ]
        hits = np.array([1, 0, 1, 0], bool)
        aucs = rank_statistics(stats, hits)
        assert set(aucs) == set(STATISTICS)
        assert aucs["focus"] == 1.0 and aucs["peak"] < 1.0

    def test_the_default_holds_unless_clearly_beaten(self):
        assert (
            choose_statistic({"peak": 0.70, "shape": 0.72, "focus": 0.60, "ratio": 0.5, "shape_peak": 0.71}) == "peak"
        )
        assert (
            choose_statistic({"peak": 0.58, "shape": 0.82, "focus": 0.81, "ratio": 0.79, "shape_peak": 0.82}) == "shape"
        )

    def test_a_record_with_one_outcome_keeps_the_default(self):
        assert choose_statistic({name: float("nan") for name in STATISTICS}) == "peak"


class TestSeveralEvents:
    """A curve read as several events: the gap suppresses, the cap stops, and
    a neighbour is neither a rival nor a smear."""

    def test_one_event_is_curve_stats_exactly(self):
        from ethograph.labels.confidence import curve_events

        curve = _bump(500, 250, 0.9) + _bump(500, 100, 0.85)
        assert curve_events(curve, window=10, gap=50, max_events=1) == [curve_stats(curve, window=10)]

    def test_peaks_within_the_gap_are_one_event_and_the_cap_holds(self):
        from ethograph.labels.confidence import event_peaks

        curve = _bump(500, 100, 0.9) + _bump(500, 120, 0.8) + _bump(500, 300, 0.7) + _bump(500, 450, 0.6)
        assert event_peaks(curve, gap=50, max_events=10) == [100, 300, 450]  # 120 is within 50 of a taller peak
        assert event_peaks(curve, gap=50, max_events=2) == [100, 300]  # tallest first, reported in time order
        assert event_peaks(_bump(500, 250, 0.02), gap=50, max_events=3) == []  # a blip is not an event

    def test_a_neighbouring_event_is_not_a_rival_or_a_smear(self):
        from ethograph.labels.confidence import curve_events

        curve = _bump(500, 100, 0.9) + _bump(500, 350, 0.85)
        alone = curve_stats(_bump(500, 100, 0.9), window=10)
        first, second = curve_events(curve, window=10, gap=50, max_events=3)
        assert (first.index, second.index) == (100, 350)
        assert first.focus == pytest.approx(alone.focus, abs=0.02) and first.ratio == pytest.approx(1.0)
        assert second.found and second.focus > 0.8

    def test_a_curve_with_no_peak_still_reads_as_one_not_found_event(self):
        from ethograph.labels.confidence import curve_events

        events = curve_events(np.linspace(0.0, 0.9, 500), window=10, gap=50, max_events=3)
        assert len(events) == 1 and not events[0].found
