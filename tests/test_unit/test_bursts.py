"""Burst detectors: what the R comparison (``tests/_test_burst_r_reference.py``) does not exercise."""

import numpy as np
import pynapple as nap
import pytest

from ethograph.features.bursts import (
    detect_bursts_instantaneous_rate,
    detect_bursts_log_isi,
    detect_bursts_max_interval,
)


def test_close_bursts_merge_before_small_ones_are_dropped():
    # Two pairs 0.15 s apart: neither has three spikes, together they are one burst of four.
    spikes = nap.Ts(t=[0.0, 0.05, 0.2, 0.25, 5.0, 5.05, 9.0])
    bursts = detect_bursts_max_interval(spikes, max_isi_start=0.1, max_isi_burst=0.1)
    assert bursts.values.tolist() == [[0.0, 0.25]]
    assert bursts.n_spikes.tolist() == [4]


def test_a_burst_never_spans_a_gap_in_the_time_support():
    epochs = nap.IntervalSet(start=[0.0, 10.0], end=[5.0, 20.0])
    # 50 ms apart throughout: one burst without the gap, cut at it with.
    spikes = nap.Ts(t=np.array([4.9, 4.95, 5.0, 10.0, 10.05, 10.1]), time_support=epochs)
    bursts = detect_bursts_max_interval(spikes, min_burst_duration=0.0)
    assert bursts.values.tolist() == [[4.9, 5.0], [10.0, 10.1]]


def test_spikes_at_one_instant_are_no_burst_and_cost_the_others_nothing():
    """A zero-length interval makes pynapple drop the metadata of every burst."""
    spikes = nap.Ts(t=[0.0, 0.001, 0.002, 1.0, 1.0, 2.0, 2.001])
    bursts = detect_bursts_instantaneous_rate(spikes)
    assert bursts.values.tolist() == [[0.0, 0.002], [2.0, 2.001]]
    assert bursts.n_spikes.tolist() == [3, 2]


def test_log_isi_without_a_valley_falls_back_only_when_asked():
    # One interval length: a within-burst peak, and nothing to separate it from.
    spikes = nap.Ts(t=np.arange(50) * 0.02)
    assert len(detect_bursts_log_isi(spikes)) == 1
    assert len(detect_bursts_log_isi(spikes, fallback=False)) == 0


def test_a_group_is_read_unit_by_unit():
    train = np.concatenate([[0.0, 0.01, 0.02], [3.0]])
    group = nap.TsGroup({2: nap.Ts(t=train), 7: nap.Ts(t=train + 1.0)}, time_support=nap.IntervalSet(0.0, 5.0))
    bursts = detect_bursts_max_interval(group)
    assert {unit: b.start.tolist() for unit, b in bursts.items()} == {2: [0.0], 7: [1.0]}


def test_a_series_that_is_not_spikes_is_refused():
    with pytest.raises(TypeError, match="Ts"):
        detect_bursts_max_interval(np.arange(5.0))
    with pytest.raises(ValueError, match="at least two spikes"):
        detect_bursts_max_interval(nap.Ts(t=[0.0, 1.0]), min_spikes_in_burst=1)
