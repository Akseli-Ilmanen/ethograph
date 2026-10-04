"""Firing rates binned from a pre-built ``TsGroup``."""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.features.neural import SpikeTable, firing_rate_to_xarray

nap = pytest.importorskip("pynapple")


def _group(n_units: int = 5) -> nap.TsGroup:
    rng = np.random.default_rng(0)
    return nap.TsGroup({uid: nap.Ts(t=np.sort(rng.uniform(0.0, 10.0, size=50 * (uid + 1)))) for uid in range(n_units)})


def test_a_window_of_the_spike_table_is_every_units_restriction():
    group = _group()
    table = SpikeTable.from_tsgroup(group)

    times, units = table.window(2.0, 6.5)

    assert np.all(np.diff(times) >= 0)
    for uid in group.keys():
        expected = group[uid].restrict(nap.IntervalSet(2.0, 6.5)).times()
        np.testing.assert_array_equal(times[units == uid], expected)


def test_spike_table_counts_match_pynapple_in_the_column_order_asked_for():
    group = _group()
    table = SpikeTable.from_tsgroup(group)

    counts, centers = table.count([3, 1], t_start=1.0, t_stop=9.0, bin_size=0.5)

    assert counts.shape == (16, 2)
    np.testing.assert_allclose(centers, 1.25 + 0.5 * np.arange(16))
    for column, uid in enumerate((3, 1)):
        expected = group[uid].count(0.5, ep=nap.IntervalSet(1.0, 9.0)).values
        np.testing.assert_array_equal(counts[:, column], expected)


def test_cluster_subset_of_a_prebuilt_tsgroup_bins_only_those_units():
    rng = np.random.default_rng(0)
    group = nap.TsGroup({uid: nap.Ts(t=np.sort(rng.uniform(0.0, 10.0, size=50 * (uid + 1)))) for uid in range(5)})
    empty = np.empty(0)

    picked = np.array([3, 1])

    da = firing_rate_to_xarray(empty, empty, 0.5, t_start=0.0, t_stop=10.0, cluster_ids=picked, _tsgroup=group)

    assert da.sizes["cluster_id"] == 2
    for uid in (1, 3):
        expected = group[uid].count(0.5, ep=nap.IntervalSet(0.0, 10.0)).values / 0.5
        np.testing.assert_allclose(da.sel(cluster_id=uid).values, expected)


def test_the_instantaneous_rate_is_one_over_the_interval_a_time_falls_in():
    from ethograph.features.neural import instantaneous_rate

    spikes = np.array([1.0, 1.1, 1.3, 2.3])
    at = np.array([0.5, 1.0, 1.05, 1.2, 1.3, 2.0, 2.3, 3.0])
    # From each spike until the next; nothing before the first spike or from the last one on.
    expected = [0.0, 10.0, 10.0, 5.0, 1.0, 1.0, 0.0, 0.0]
    assert np.allclose(instantaneous_rate(spikes, at), expected)
    assert not instantaneous_rate(np.array([1.0]), at).any()
