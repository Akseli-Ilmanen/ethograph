"""Burst scatter: one number per burst, and a selection on the plot limits the video grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ethograph.features.burst_stats import (
    default_percentiles,
    nearest_points,
    points_in_rect,
    scale_between_percentiles,
    shift_onsets,
    spike_features,
    time_from_nearest,
    window_summary,
)
from ethograph.gui.dialog_burst_scatter import (
    FROM_FIRST_LABEL,
    FROM_LAST_LABEL,
    FROM_TRIAL_START,
    REF,
    covering_labels,
    label_timing_column,
    reference_labels,
    shifted_bursts,
    timing_columns,
    unique_bursts,
)


class TestSpikeFeatures:
    def test_a_regular_burst(self):
        f = spike_features(np.array([1.0, 1.1, 1.2, 1.3]))
        assert f["n_spikes"] == 4
        assert f["mean_rate"] == pytest.approx(10.0)
        assert f["first_isi"] == pytest.approx(0.1)
        assert f["isi_cv"] == pytest.approx(0.0, abs=1e-9)
        assert f["rate_trend"] == pytest.approx(0.0, abs=1e-6)

    def test_an_accelerating_burst_has_a_positive_trend_and_a_late_peak(self):
        f = spike_features(np.array([0.0, 0.4, 0.6, 0.7]))
        assert f["rate_trend"] > 0
        assert f["peak_position"] > 0.5
        assert spike_features(np.array([0.0, 0.1, 0.3, 0.7]))["rate_trend"] < 0

    def test_too_few_spikes_leave_the_feature_undefined_not_zero(self):
        one = spike_features(np.array([2.0]))
        assert one["n_spikes"] == 1 and np.isnan(one["mean_rate"])
        two = spike_features(np.array([2.0, 2.5]))
        assert two["mean_rate"] == pytest.approx(2.0) and np.isnan(two["isi_cv"]) and np.isnan(two["rate_trend"])


class TestWindowSummary:
    TIME = np.arange(0.0, 10.0, 1.0)
    VALUES = np.array([0, 0, 0, 0, 2, 4, 6, 0, 0, 0], dtype=float)

    def test_each_summary_reads_the_window(self):
        assert window_summary(self.TIME, self.VALUES, 4, 6, "mean") == pytest.approx(4.0)
        assert window_summary(self.TIME, self.VALUES, 4, 6, "peak") == pytest.approx(6.0)
        assert window_summary(self.TIME, self.VALUES, 4.5, 6, "onset") == pytest.approx(3.0)
        # Mean during (4) minus the mean of the equally long stretch before it (t = 2, 3 → 0).
        assert window_summary(self.TIME, self.VALUES, 4, 6, "change") == pytest.approx(4.0)

    def test_a_window_the_series_does_not_cover_is_nan(self):
        assert np.isnan(window_summary(self.TIME, self.VALUES, 20, 21, "mean"))
        assert np.isnan(window_summary(self.TIME, self.VALUES, 20, 21, "onset"))
        with pytest.raises(ValueError, match="Unknown summary"):
            window_summary(self.TIME, self.VALUES, 0, 1, "median")


class TestShiftOnsets:
    def test_every_onset_moves_by_one_offset_and_comes_back_in_the_order_given(self):
        onsets = np.array([9.0, 1.0, 2.0])  # unsorted on purpose
        for _ in range(20):
            moved = shift_onsets(onsets, 10.0)
            offsets = (moved - onsets) % 10.0
            assert offsets == pytest.approx(np.full(3, offsets[0]))
            assert ((moved >= 0.0) & (moved <= 10.0)).all()

    def test_nothing_to_shift(self):
        assert len(shift_onsets(np.array([]), 10.0)) == 0


class TestColourScale:
    def test_the_scale_opens_wider_at_the_bottom_only_for_values_that_cannot_go_negative(self):
        assert default_percentiles(np.array([0.0, 2.0, np.nan])) == (0.0, 95.0)  # a speed
        assert default_percentiles(np.array([-1.0, 2.0])) == (5.0, 95.0)  # a velocity

    def test_values_spread_between_two_percentiles_and_past_them_are_clipped(self):
        values = np.append(np.arange(100.0), [1e6, np.nan])  # one wild value, one missing
        scaled, lo, hi = scale_between_percentiles(values, 0.0, 99.0)
        assert lo == 0.0 and hi < 200.0  # the wild value does not set the scale
        assert scaled[0] == 0.0 and scaled[100] == 1.0 and np.isnan(scaled[101])
        assert scaled[50] == pytest.approx(50.0 / hi)

    def test_no_spread_and_no_values(self):
        assert list(scale_between_percentiles(np.array([3.0, 3.0]))[0]) == [0.0, 0.0]
        assert np.isnan(scale_between_percentiles(np.array([np.nan]))[0]).all()


class TestPicking:
    def test_nearest_is_measured_on_the_axes_own_ranges_and_skips_nan(self):
        # x spans 1000, y spans 1: unscaled, the y offset would never matter.
        x = np.array([0.0, 100.0, 1000.0, np.nan])
        y = np.array([0.0, 1.0, 0.0, 0.0])
        assert list(nearest_points(x, y, 0.0, 0.9, 2)) == [1, 0]
        assert len(nearest_points(np.array([np.nan]), np.array([1.0]), 0, 0, 3)) == 0

    def test_rect_takes_corners_in_any_order(self):
        x = np.array([0.0, 1.0, 2.0, np.nan])
        y = np.array([0.0, 1.0, 2.0, 1.0])
        assert list(points_in_rect(x, y, 1.5, 0.5, 1.5, 0.5)) == [False, True, False, False]

    def test_time_from_nearest_is_signed(self):
        assert time_from_nearest(1.0, np.array([0.2, 1.3, 5.0])) == pytest.approx(-0.3)
        assert time_from_nearest(1.0, np.array([0.8, np.nan])) == pytest.approx(0.2)
        assert np.isnan(time_from_nearest(1.0, np.array([])))


@dataclass
class _Burst:
    trial: object
    label_id: int
    onset_s: float
    offset_s: float
    name: str = "unit_1"
    is_point: bool = False
    individual: object = None
    individual_rec: object = None
    camera: str | None = None

    @property
    def duration(self) -> float:
        return self.offset_s - self.onset_s


class TestTimingColumns:
    SYLLABLES = pd.DataFrame(
        {
            "trial": [1, 1, 1, 2],
            "labels": [5, 6, 5, 5],
            "onset_s": [1.0, 2.0, 4.0, 0.5],
            "offset_s": [1.5, np.nan, 4.5, 0.9],
        }
    )

    def _state(self):
        return SimpleNamespace(_all_labels_df=self.SYLLABLES, source_collection=None)

    def test_onset_is_measured_from_the_trials_first_and_last_label(self):
        bursts = [_Burst(1, 1, 3.0, 3.2), _Burst("2", 1, 0.1, 0.2), _Burst(3, 1, 1.0, 1.1)]
        cols = timing_columns(self._state(), bursts)
        assert list(cols[FROM_TRIAL_START]) == [3.0, 0.1, 1.0]
        assert cols[FROM_FIRST_LABEL][:2] == pytest.approx([2.0, -0.4])
        assert cols[FROM_LAST_LABEL][:2] == pytest.approx([-1.5, -0.8])
        assert np.isnan(cols[FROM_FIRST_LABEL][2])  # a trial with no labels has no reference

    def test_time_from_a_label_class_uses_the_nearest_edge_in_the_same_trial(self):
        bursts = [_Burst(1, 1, 3.8, 3.9), _Burst(2, 1, 3.8, 3.9)]
        assert label_timing_column(self._state(), bursts, "5 (5)", "onset") == pytest.approx([-0.2, 3.3])
        assert label_timing_column(self._state(), bursts, "5 (5)", "offset") == pytest.approx([-0.7, 2.9])
        assert np.isnan(label_timing_column(self._state(), bursts, "6 (6)", "offset")).all()  # point labels have none

    def test_labels_of_another_loaded_set_are_references_but_never_the_grids_own(self):
        """Syllables kept as a prediction set are still something to time a burst against."""
        rows = pd.DataFrame({"trial": [1], "labels": [5], "onset_s": [3.0], "offset_s": [3.5]})
        syllables = SimpleNamespace(path=Path("syllables"), labels_df=rows, mappings={5: {"name": "A"}})
        bursts_set = SimpleNamespace(path=Path("bursts"), labels_df=rows, mappings={5: {"name": "unit_5"}})
        state = SimpleNamespace(
            _all_labels_df=None, prediction_sets=[syllables, bursts_set], label_source_path="bursts"
        )
        assert list(reference_labels(state)[REF].unique()) == ["A (5) · syllables"]
        # Another set of units (an earlier burst detection) is never a reference, shown or not.
        other_units = SimpleNamespace(path=Path("bursts_old"), labels_df=rows, mappings={5: {"name": "unit_7"}})
        state.prediction_sets.append(other_units)
        assert list(reference_labels(state)[REF].unique()) == ["A (5) · syllables"]
        column = label_timing_column(state, [_Burst(1, 1, 2.0, 2.1)], "A (5) · syllables", "onset")
        assert column == pytest.approx([-1.0])

    def test_a_burst_takes_whichever_has_the_largest_share_of_it_label_or_gap(self):
        """Trial 1: label 5 over 0.0–0.3, label 7 over 0.6–1.0, a point label 6 at 2.0."""
        labels = pd.DataFrame(
            {
                "trial": [1, 1, 1],
                "labels": [5, 7, 6],
                "onset_s": [0.0, 0.6, 2.0],
                "offset_s": [0.3, 1.0, np.nan],
            }
        )
        state = SimpleNamespace(_all_labels_df=labels, source_collection=None)
        bursts = [
            _Burst(1, 1, 0.0, 1.0),  # 30% label 5, 30% gap, 40% label 7: 7 wins without half of it
            _Burst(1, 1, 0.2, 0.6),  # 25% label 5, 75% gap: the gap wins
            _Burst(1, 1, 2.0, 2.1),  # over the point label only
            _Burst(9, 1, 0.0, 0.2),  # a trial without labels
        ]
        covering = covering_labels(state, bursts)
        assert [c and c[0] for c in covering] == ["7 (7)", None, None, None]
        assert covering[0][1] == "#ffffff"  # no vocabulary: the mapping's default colour

    def test_shifted_bursts_keep_their_class_and_length_inside_the_trial(self):
        bursts = [_Burst(1, 1, 1.0, 1.5), _Burst(1, 2, 4.0, 4.2), _Burst(2, 1, 0.5, 0.6), _Burst(3, 1, 0.0, 0.1)]
        shifted = shifted_bursts(bursts, {"1": 10.0, "2": 5.0}, repeats=20)
        # Trial 3 has no timing and is left out; a shift that leaves a burst no room drops it.
        assert 0 < len(shifted) <= 60 and {str(s.trial) for s in shifted} == {"1", "2"}
        lengths = {(s.label_id, str(s.trial)): round(s.duration, 6) for s in shifted}
        assert lengths == {(1, "1"): 0.5, (2, "1"): 0.2, (1, "2"): 0.1}  # never cut short
        assert all(s.offset_s <= {"1": 10.0, "2": 5.0}[str(s.trial)] for s in shifted)

    def test_a_label_seen_by_two_cameras_is_one_burst_and_points_are_left_out(self):
        a, b = _Burst(1, 1, 1.0, 2.0, camera="top"), _Burst(1, 1, 1.0, 2.0, camera="side")
        assert unique_bursts([a, b, _Burst(1, 2, 1.0, 1.0, is_point=True)]) == [a]
