"""A burst rule as a spike filter: the spikes a detector puts in a burst, and the rule's name."""

import numpy as np
import pynapple as nap
import pytest

from ethograph.labels.feature_events import (
    BURST_DETECTORS,
    INSTANTANEOUS_RATE,
    LOG_ISI,
    MAX_INTERVAL,
    BurstRule,
    burst_spike_times,
)

#: Five spikes 5 ms apart (200 Hz), a lone spike, then four spikes 5 ms apart.
_BURST_A = 1.0 + np.arange(5) * 0.005
_LONE = np.array([2.0])
_BURST_B = 3.0 + np.arange(4) * 0.005
_TRAIN = nap.Ts(t=np.concatenate([_BURST_A, _LONE, _BURST_B]))
_EPOCH = nap.IntervalSet(0.0, 10.0)


@pytest.mark.parametrize("method", BURST_DETECTORS)
def test_only_the_spikes_inside_a_burst_are_kept(method):
    kept = burst_spike_times(_TRAIN, _EPOCH, BurstRule(method))
    assert np.allclose(kept, np.concatenate([_BURST_A, _BURST_B])), method


def test_a_burst_is_read_within_the_epochs_given():
    # Burst B lies outside the epoch, so none of its spikes are a burst's.
    kept = burst_spike_times(_TRAIN, nap.IntervalSet(0.0, 2.5), BurstRule(MAX_INTERVAL))
    assert np.allclose(kept, _BURST_A)


def test_a_train_with_no_bursts_keeps_nothing():
    kept = burst_spike_times(nap.Ts(t=np.array([1.0, 2.0, 3.0])), _EPOCH, BurstRule(INSTANTANEOUS_RATE))
    assert kept.shape == (0,)


def test_the_instantaneous_rate_threshold_is_the_rules_own():
    # 200 Hz spikes: a burst below a 150 Hz threshold, nothing above 250 Hz.
    assert len(burst_spike_times(_TRAIN, _EPOCH, BurstRule(INSTANTANEOUS_RATE, {"rate_threshold": 150.0}))) == 9
    assert len(burst_spike_times(_TRAIN, _EPOCH, BurstRule(INSTANTANEOUS_RATE, {"rate_threshold": 250.0}))) == 0


def test_a_rules_label_names_only_what_was_changed():
    assert BurstRule(LOG_ISI).label() == "bursts log_isi"
    assert BurstRule(LOG_ISI, {"max_cutoff": 0.1, "fallback": False}).label() == "bursts log_isi (fallback=False)"
    rule = BurstRule(MAX_INTERVAL, {"max_isi_start": 0.05, "min_spikes_in_burst": 4})
    assert rule.label() == "bursts max_interval (max_isi_start=0.05, min_spikes_in_burst=4)"


def test_a_rule_survives_the_settings_file():
    rule = BurstRule(INSTANTANEOUS_RATE, {"rate_threshold": 80.0})
    assert BurstRule.from_dict(rule.to_dict()) == rule
    assert BurstRule.from_dict({"method": LOG_ISI}) == BurstRule(LOG_ISI)
