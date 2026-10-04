"""Labels read off a feature: threshold periods, their clean-up, and their classes."""

from types import SimpleNamespace

import numpy as np
import pynapple as nap
import pytest

from ethograph.labels.feature_events import (
    BELOW,
    BURST_DETECTORS,
    LOG_ISI,
    MAX_INTERVAL,
    PERCENTILE,
    ZSCORE,
    BurstRule,
    FeatureQuery,
    ThresholdRule,
    burst_events,
    burst_parameters,
    classes_along,
    clean_intervals,
    event_mappings,
    feature_events,
    sample_series,
    threshold_intervals,
)
from ethograph.labels.intervals import _rows_to_df, purge_short_intervals, stitch_intervals

TIME = np.arange(10) / 10.0


def test_a_period_runs_from_the_first_sample_past_to_the_first_sample_back():
    values = np.array([0, 5, 5, 0, 0, 0, 0, 5, 5, 5.0])
    assert threshold_intervals(TIME, values, ThresholdRule(1.0)).tolist() == [[0.1, 0.3], [0.7, 0.9]]


def test_below_and_nan():
    values = np.array([5, 0, 0, np.nan, 0, 0, 5, 5, 5, 5.0])
    # A NaN is on neither side of the threshold, so it ends the period.
    assert threshold_intervals(TIME, values, ThresholdRule(1.0, BELOW)).tolist() == [[0.1, 0.3], [0.4, 0.6]]


def test_stitching_comes_before_the_duration_check():
    # Two 0.2 s periods 0.1 s apart: too short each, long enough once joined.
    values = np.array([0, 5, 5, 0, 5, 5, 0, 0, 0, 0.0])
    rule = ThresholdRule(1.0, stitch_gap_s=0.15, min_duration_s=0.4)
    assert threshold_intervals(TIME, values, rule).tolist() == [[0.1, 0.6]]
    assert len(threshold_intervals(TIME, values, ThresholdRule(1.0, min_duration_s=0.4))) == 0


@pytest.mark.parametrize("seed", range(5))
def test_clean_up_agrees_with_the_label_operations(seed):
    """The array clean-up and the labels' own stitch + purge make the same comparisons."""
    rng = np.random.default_rng(seed)
    edges = np.cumsum(rng.uniform(0.01, 0.5, size=80))
    intervals = edges.reshape(-1, 2)
    gap, shortest = 0.2, 0.3

    df = _rows_to_df([{"onset_s": a, "offset_s": b, "labels": 1, "individual": "A"} for a, b in intervals])
    expected = purge_short_intervals(stitch_intervals(df, gap), shortest)

    got = clean_intervals(intervals, gap, shortest)
    assert np.allclose(got, expected[["onset_s", "offset_s"]].to_numpy())


class _Loader:
    """A two-unit feature, one trial: both units are active over 0.3–0.5 s."""

    def select(self, feature, selections, t0, t1):
        data = np.zeros((10, 2))
        data[1:6, 0] = 30.0
        data[3:8, 1] = 30.0
        if "unit" in selections:
            column = ["0", "7"].index(selections["unit"])
            return SimpleNamespace(time=TIME + 100.0, data=data[:, column], dim_labels=None)
        return SimpleNamespace(time=TIME + 100.0, data=data, dim_labels=["0", "7"])


def _windows():
    return [("t1", _Loader(), 100.0, 101.0, 100.0)]


def test_one_class_per_value_of_the_dim_and_overlaps_are_kept():
    query = FeatureQuery("rate", {}, "unit", ("0", "7"))
    rule = ThresholdRule(20.0)
    events = feature_events(_windows(), query, rule, "crow")

    # Unit 0 is class 1: 0 is background wherever labels are read.
    mappings = event_mappings(query.class_names(rule))
    assert {lid: m["name"] for lid, m in mappings.items()} == {1: "unit_0", 2: "unit_7"}
    # Trial-relative, and the two units overlap — which one actor's labels may not.
    assert np.allclose(events[["labels", "onset_s", "offset_s"]].to_numpy(), [[1, 0.1, 0.6], [2, 0.3, 0.8]])
    assert set(events["individual"]) == {"crow"}
    assert set(events["labeling_method"]) == {"automated"}


def test_a_pinned_series_is_one_class_named_after_the_rule():
    query = FeatureQuery("rate", {"unit": "7"})
    rule = ThresholdRule(20.0)
    events = feature_events(_windows(), query, rule, "crow")
    assert query.class_names(rule) == ["rate>20"]
    assert np.allclose(events[["labels", "onset_s", "offset_s"]].to_numpy(), [[1, 0.3, 0.8]])


def test_an_individual_dim_names_the_actor():
    assert FeatureQuery("speed", {}, "individual", ("a", "b")).individuals("crow") == ["a", "b"]
    assert FeatureQuery("speed", {"individuals": "b"}, "keypoint", ("x", "y")).individuals("crow") == ["b", "b"]


def test_a_dim_left_free_is_refused():
    with pytest.raises(ValueError, match="pin every dim"):
        feature_events(_windows(), FeatureQuery("rate"), ThresholdRule(20.0), "crow")


class _TwoTrials:
    """Unit 0 idles at 1 Hz and unit 7 at 50 Hz; each doubles for 0.3 s — in the second trial only."""

    def __init__(self, burst: bool):
        self._burst = burst

    def select(self, feature, selections, t0, t1):
        data = np.tile([1.0, 50.0], (10, 1)) + np.tile([0.0, 0.1], 5)[:, None]
        if self._burst:
            data[2:5] *= 2.0
        return SimpleNamespace(time=TIME, data=data, dim_labels=["0", "7"])


def _two_trials():
    return [("quiet", _TwoTrials(False), 0.0, 1.0, 0.0), ("burst", _TwoTrials(True), 0.0, 1.0, 0.0)]


def test_a_zscored_threshold_is_per_series_and_across_trials():
    query = FeatureQuery("rate", {}, "unit", ("0", "7"))

    # No raw threshold serves both units: 1.5 Hz is every sample of unit 7.
    raw = feature_events(_two_trials(), query, ThresholdRule(1.5), "crow")
    assert set(raw.loc[raw["labels"] == 2, "trial"]) == {"quiet", "burst"}

    # In standard deviations the same burst is found in both units, and the
    # quiet trial — where z-scoring trial by trial would invent periods — has none.
    events = feature_events(_two_trials(), query, ThresholdRule(1.5, scale=ZSCORE), "crow")
    assert events[["trial", "labels"]].to_numpy().tolist() == [["burst", 1], ["burst", 2]]
    assert np.allclose(events[["onset_s", "offset_s"]].to_numpy(), [[0.2, 0.5], [0.2, 0.5]])


def test_the_sample_carries_exact_statistics_whatever_it_thins_out():
    query = FeatureQuery("rate", {}, "unit", ("0", "7"))
    everything = np.concatenate([w[1].select("rate", {}, 0, 1).data for w in _two_trials()])

    sample = sample_series(_two_trials(), query, max_values=8)
    assert len(sample.values) < len(everything), "nothing was thinned out"
    assert np.allclose(sample.mean, everything.mean(axis=0))
    assert np.allclose(sample.std, everything.std(axis=0))


def test_a_flat_series_never_passes_a_zscored_threshold():
    flat = SimpleNamespace(select=lambda *_a: SimpleNamespace(time=TIME, data=np.full(10, 3.0), dim_labels=None))
    events = feature_events([("t", flat, 0.0, 1.0, 0.0)], FeatureQuery("rate"), ThresholdRule(-1.0, scale=ZSCORE), "c")
    assert events.empty


def test_only_one_dim_opens_read_value_by_value_and_it_is_not_the_axis_or_the_animal():
    pytest.importorskip("qtpy")
    from ethograph.gui.dialog_feature_labels import default_every_dim

    position = {"space": ["x", "y"], "keypoint": ["beak", "tail"], "individual": ["a", "b"]}
    assert default_every_dim(position) == "keypoint"
    assert default_every_dim({"space": ["x", "y"], "individual": ["a"]}) == "space"
    assert default_every_dim({"individual": ["a"]}) is None


def test_a_percentile_takes_the_same_share_of_every_series():
    """Each series' own top 15 % — whatever its baseline — and no assumption about its distribution."""
    query = FeatureQuery("rate", {}, "unit", ("0", "7"))
    rule = ThresholdRule(85.0, scale=PERCENTILE)

    events = feature_events(_two_trials(), query, rule, "crow")
    # The burst is 3 of each unit's 20 samples: exactly what lies above its 85th percentile.
    assert events[["trial", "labels"]].to_numpy().tolist() == [["burst", 1], ["burst", 2]]
    assert np.allclose(events[["onset_s", "offset_s"]].to_numpy(), [[0.2, 0.5], [0.2, 0.5]])

    with pytest.raises(ValueError, match="between 0 and 100"):
        ThresholdRule(120.0, scale=PERCENTILE)


def test_a_class_is_found_again_by_the_value_it_was_named_after():
    query = FeatureQuery("rate", {}, "unit", ("0", "7"))
    mappings = event_mappings(query.class_names(ThresholdRule(20.0)))
    assert classes_along(mappings, "unit") == {"0": 1, "7": 2}
    assert classes_along(mappings, "keypoint") == {}


#: Two trials that touch at 110 s, each on its own clock.
_SPANS = [("a", 100.0, 110.0, 100.0), ("b", 110.0, 120.0, 110.0)]


def test_bursts_are_one_class_per_unit_on_each_trials_own_clock():
    burst = np.arange(5) * 0.01
    trains = {
        "4": nap.Ts(t=np.concatenate([101.0 + burst, 112.0 + burst])),
        "silent": nap.Ts(t=[105.0, 108.0]),
        "9": nap.Ts(t=113.0 + burst),
    }
    events = burst_events(trains, _SPANS, BurstRule(MAX_INTERVAL), "crow")

    # Class ids follow the order of the trains, whether or not a unit bursts.
    assert events[["trial", "labels"]].to_numpy().tolist() == [["a", 1], ["b", 1], ["b", 3]]
    assert np.allclose(events[["onset_s", "offset_s"]].to_numpy(), [[1.0, 1.04], [2.0, 2.04], [3.0, 3.04]])
    assert set(events["individual"]) == {"crow"}


def test_a_burst_is_cut_where_one_trial_ends_and_the_next_begins():
    # 10 ms apart straight through the boundary at 110 s: one burst in the train, one per trial here.
    trains = {"0": nap.Ts(t=np.arange(109.95, 110.051, 0.01))}
    events = burst_events(trains, _SPANS, BurstRule(MAX_INTERVAL), "crow")
    assert events["trial"].tolist() == ["a", "b"]
    assert (events["onset_s"] >= 0).all() and (events["offset_s"] <= 10.0).all()


def test_a_burst_rule_takes_only_its_detectors_own_parameters():
    rule = BurstRule(LOG_ISI, {"fallback": False})
    # One interval length has no valley: without the fallback there is nothing to find.
    assert burst_events({"0": nap.Ts(t=100.0 + np.arange(50) * 0.02)}, _SPANS, rule, "crow").empty

    with pytest.raises(ValueError, match="no parameter"):
        BurstRule(LOG_ISI, {"max_isi_start": 0.1})
    with pytest.raises(ValueError, match="method must be one of"):
        BurstRule("poisson_surprise")
    assert all(burst_parameters(method) for method in BURST_DETECTORS)


@pytest.mark.parametrize("method", BURST_DETECTORS)
def test_a_unit_with_no_spikes_in_the_trials_read_has_no_bursts(method):
    burst = np.arange(5) * 0.01
    # Unit "away" fires only outside the trials: cut to them, its train is empty.
    trains = {"away": nap.Ts(t=50.0 + burst), "here": nap.Ts(t=101.0 + burst)}
    events = burst_events(trains, _SPANS, BurstRule(method), "crow")
    assert events["labels"].tolist() == [2]
