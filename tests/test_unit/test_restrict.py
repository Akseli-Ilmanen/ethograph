"""Tests for ethograph.io.time_model restriction builders and ethograph.utils.sequences."""

import numpy as np
import pandas as pd
import pytest

from ethograph.io.time_model import (
    TimeRange,
    TrialVideoBounds,
    build_label_window,
    build_sequence_window,
    build_trial_window,
    restrict_xarray,
)
from ethograph.utils.sequences import (
    get_label_instances,
    get_unique_sequences,
    match_sequences,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trial_alignment():
    return TrialVideoBounds(
        trial_id=1,
        trial_range=TimeRange(0.0, 10.0),
    )


@pytest.fixture
def sample_labels_df():
    return pd.DataFrame(
        {
            "trial": [1, 1, 1, 2, 2, 2, 3, 3],
            "onset_s": [0.5, 1.5, 3.0, 0.2, 1.0, 2.0, 0.1, 1.0],
            "offset_s": [1.0, 2.5, 4.0, 0.8, 1.5, 3.0, 0.5, 2.0],
            "labels": [1, 2, 3, 1, 2, 3, 2, 3],
            "individual": ["ind1"] * 8,
        }
    )


# ---------------------------------------------------------------------------
# build_trial_window
# ---------------------------------------------------------------------------


def test_build_trial_window_basic(trial_alignment):
    rw = build_trial_window(trial_alignment, trial_id=1)
    assert rw.mode == "trial"
    assert rw.core_range == TimeRange(0.0, 10.0)
    assert rw.time_range == TimeRange(0.0, 10.0)
    assert rw.trial_id == 1


def test_build_trial_window_with_extra(trial_alignment):
    rw = build_trial_window(trial_alignment, trial_id=1, extra_t0=1.0, extra_t1=2.0)
    assert rw.time_range.start_s == -1.0  # extends before trial for ephys context
    assert rw.time_range.end_s == 12.0
    assert rw.core_range == TimeRange(0.0, 10.0)


# ---------------------------------------------------------------------------
# build_label_window
# ---------------------------------------------------------------------------


def test_build_label_window_with_context(sample_labels_df):
    trial_bounds = TimeRange(0.0, 10.0)
    rw = build_label_window(
        sample_labels_df,
        label_idx=0,
        trial_bounds=trial_bounds,
        extra_t0=0.3,
        extra_t1=0.5,
    )
    assert rw.time_range.start_s == pytest.approx(0.2)
    assert rw.time_range.end_s == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# build_sequence_window
# ---------------------------------------------------------------------------


def test_build_sequence_window():
    match = {
        "trial": 1,
        "onset_s": 0.5,
        "offset_s": 4.0,
        "pattern": "1-2-3",
        "match_rows": [0, 1, 2],
    }
    trial_bounds = TimeRange(0.0, 10.0)
    rw = build_sequence_window(match, trial_bounds, extra_t0=0.2, extra_t1=0.3)
    assert rw.mode == "sequence"
    assert rw.core_range == TimeRange(0.5, 4.0)
    assert rw.time_range.start_s == pytest.approx(0.3)
    assert rw.time_range.end_s == pytest.approx(4.3)


# ---------------------------------------------------------------------------
# restrict_xarray
# ---------------------------------------------------------------------------


def test_restrict_xarray():
    import xarray as xr

    ds = xr.Dataset({"x": ("time", np.arange(100.0))}, coords={"time": np.linspace(0, 10, 100)})
    tr = TimeRange(2.0, 5.0)
    restricted = restrict_xarray(ds, tr)
    assert restricted.time.values.min() >= 2.0
    assert restricted.time.values.max() <= 5.0


# ---------------------------------------------------------------------------
# SourceCollection
# ---------------------------------------------------------------------------


def test_source_collection_union_intersection():
    from ethograph.io.time_model import SourceCollection

    class _FakeSource:
        def __init__(self, name, start, end):
            self.name = name
            self.time_range = TimeRange(start, end)
            self.sampling_rate = 100.0

        def get_data(self, t0, t1):
            return np.array([]), np.array([])

    sc = SourceCollection()
    sc.add(_FakeSource("a", 0.0, 10.0))
    sc.add(_FakeSource("b", 5.0, 15.0))

    assert sc.union_range == TimeRange(0.0, 15.0)
    assert sc.intersection_range == TimeRange(5.0, 10.0)


def test_source_collection_trials():
    from ethograph.io.time_model import SourceCollection

    sc = SourceCollection()
    sc.set_trials(
        ids=[1, 2, 3],
        starts=[0.0, 10.0, 25.0],
        stops=[8.0, 20.0, 30.0],
    )
    assert sc.n_trials == 3
    assert sc.trial_range(0) == TimeRange(0.0, 8.0)
    assert sc.trial_offset(2) == 25.0
    assert sc.session_range == TimeRange(0.0, 30.0)
    assert sc.find_trial(5.0) == 0
    assert sc.find_trial(15.0) == 1


def test_session_range_reaches_data_past_the_trials():
    """Session scope spans the recording, not just first trial start to last
    trial end: a source running past the trials widens it, an empty one never
    drags it to zero."""
    from ethograph.io.time_model import SourceCollection

    class _Source:
        def __init__(self, name, start, end):
            self.name = name
            self.time_range = TimeRange(start, end)
            self.sampling_rate = None

        def get_data(self, t0, t1):
            return np.array([]), np.array([])

    sc = SourceCollection()
    sc.set_trials(ids=[1, 2], starts=[100.0, 200.0], stops=[150.0, 250.0])
    sc.add(_Source("spikes", 50.0, 4000.0))
    sc.add(_Source("empty", 0.0, 0.0))
    assert sc.session_range == TimeRange(50.0, 4000.0)


def test_tsgroup_source_extent_is_its_time_support():
    import numpy as np
    import pynapple as nap

    from ethograph.io.time_sources import TsGroupSource

    group = nap.TsGroup({1: nap.Ts(np.array([10.0, 20.0]))}, time_support=nap.IntervalSet(0.0, 3600.0))
    src = TsGroupSource("units", group)
    assert src.time_range == TimeRange(0.0, 3600.0)
    t, units = src.get_data(5.0, 15.0)
    assert t.tolist() == [10.0] and units.tolist() == [1]


def test_source_collection_infer_stops():
    from ethograph.io.time_model import SourceCollection

    class _FakeSource:
        def __init__(self, name, start, end):
            self.name = name
            self.time_range = TimeRange(start, end)
            self.sampling_rate = None

        def get_data(self, t0, t1):
            return np.array([]), np.array([])

    sc = SourceCollection()
    sc.add(_FakeSource("sig", 0.0, 40.0))
    sc.set_trials(ids=[1, 2], starts=[0.0, 15.0])

    assert sc.trial_range(0) == TimeRange(0.0, 15.0)
    assert sc.trial_range(1) == TimeRange(15.0, 40.0)


# ---------------------------------------------------------------------------
# get_label_instances
# ---------------------------------------------------------------------------


def test_get_label_instances(sample_labels_df):
    instances = get_label_instances(sample_labels_df, label_id=2)
    assert len(instances) == 3  # trials 1, 2, 3
    assert all(inst["trial"] in [1, 2, 3] for inst in instances)


def test_get_label_instances_with_individual(sample_labels_df):
    instances = get_label_instances(sample_labels_df, label_id=1, individual="ind1")
    assert len(instances) == 2  # trials 1, 2
    instances_none = get_label_instances(sample_labels_df, label_id=1, individual="nonexistent")
    assert len(instances_none) == 0


def test_get_label_instances_empty():
    assert get_label_instances(pd.DataFrame(), label_id=1) == []
    assert get_label_instances(None, label_id=1) == []


# ---------------------------------------------------------------------------
# get_unique_sequences
# ---------------------------------------------------------------------------


def test_get_unique_sequences(sample_labels_df):
    seqs = get_unique_sequences(sample_labels_df)
    assert "1-2-3" in seqs
    assert "2-3" in seqs


# ---------------------------------------------------------------------------
# match_sequences
# ---------------------------------------------------------------------------


def test_match_sequences_exact(sample_labels_df):
    matches = match_sequences(sample_labels_df, "1-2-3")
    assert len(matches) == 2  # trials 1 and 2
    assert matches[0]["trial"] == 1
    assert matches[1]["trial"] == 2


def test_match_sequences_subsequence(sample_labels_df):
    matches = match_sequences(sample_labels_df, "2-3")
    assert len(matches) == 3  # all three trials


def test_match_sequences_no_match(sample_labels_df):
    matches = match_sequences(sample_labels_df, "5-6-7")
    assert len(matches) == 0


def test_match_sequences_empty():
    assert match_sequences(pd.DataFrame(), "1-2") == []
    assert match_sequences(None, "1-2") == []
    assert match_sequences(pd.DataFrame({"labels": [1], "trial": [1], "onset_s": [0], "offset_s": [1]}), "") == []
