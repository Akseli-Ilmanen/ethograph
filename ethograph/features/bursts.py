"""Burst detection in spike trains, one function per detector, all of them pynapple's.

``nap.detect_spike_bursts`` does the detecting (logISI, MaxInterval, an
instantaneous-rate threshold and Poisson surprise; Cotterill & Eglen 2018 compare
the first two, https://arxiv.org/abs/1802.01287). This module only gives each
detector a function of its own, whose keyword defaults are what the GUI's form
opens at (``labels.feature_events.burst_parameters`` reads them off the
signatures), and keeps every parameter a plain number:

* logISI opens with the published fallback on (``fallback=True``), where
  pynapple's default is off;
* Poisson surprise's ``max_isi_burst=0`` stands for pynapple's ``None`` (twice the
  train's mean ISI), as a form row cannot hold ``None``.

Every detector takes a ``Ts`` or a ``TsGroup`` and returns bursts as an
``IntervalSet`` from first to last spike with an ``n_spikes`` metadata column (a
dict keyed by unit for a ``TsGroup``). Detection runs within each epoch of the
time support, so a burst never spans a gap in it.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import pynapple as nap

Bursts: TypeAlias = nap.IntervalSet | dict[Any, nap.IntervalSet]


def detect_bursts_max_interval(
    spikes: nap.Ts | nap.TsGroup,
    max_isi_start: float = 0.17,
    max_isi_burst: float = 0.3,
    min_inter_burst_interval: float = 0.2,
    min_burst_duration: float = 0.01,
    min_spikes_in_burst: int = 3,
) -> Bursts:
    """Bursts by the MaxInterval method: fixed limits on the intervals between spikes (seconds)."""
    return nap.detect_spike_bursts(
        spikes,
        method="max_interval",
        max_isi_start=max_isi_start,
        max_isi_burst=max_isi_burst,
        min_inter_burst_interval=min_inter_burst_interval,
        min_burst_duration=min_burst_duration,
        min_spikes_in_burst=min_spikes_in_burst,
    )


def compute_log_isi_threshold(
    spikes: nap.Ts | nap.TsGroup, max_cutoff: float = 0.1
) -> float | None | dict[Any, float | None]:
    """The logISI threshold (s): the valley after the log-ISI histogram's intra-burst peak; ``None`` without one."""
    return nap.compute_log_isi_threshold(spikes, max_cutoff=max_cutoff)


def detect_bursts_log_isi(
    spikes: nap.Ts | nap.TsGroup,
    max_cutoff: float = 0.1,
    min_spikes_in_burst: int = 3,
    fallback: bool = True,
) -> Bursts:
    """Bursts by the logISI method of Pasquale et al. (2010): a threshold each train reads off its own ISIs.

    *fallback* (on here, off in pynapple) reads a train with no clear valley at
    *max_cutoff*, as the published method does; off, such a train has no bursts.
    """
    return nap.detect_spike_bursts(
        spikes, method="log_isi", max_cutoff=max_cutoff, min_spikes_in_burst=min_spikes_in_burst, fallback=fallback
    )


def detect_bursts_instantaneous_rate(
    spikes: nap.Ts | nap.TsGroup, rate_threshold: float = 125.0, min_spikes_in_burst: int = 2
) -> Bursts:
    """Runs of spikes whose instantaneous rate (1 / ISI) stays above *rate_threshold* (Hz).

    125 Hz is the birdsong field's criterion (Leonardo & Fee 2005).
    """
    return nap.detect_spike_bursts(
        spikes, method="instantaneous_rate", rate_threshold=rate_threshold, min_spikes_in_burst=min_spikes_in_burst
    )


def detect_bursts_poisson_surprise(
    spikes: nap.Ts | nap.TsGroup, min_surprise: float = 5.0, max_isi_burst: float = 0.0
) -> Bursts:
    """Bursts by Poisson surprise (Legendy & Salcman 1985): runs too dense for chance at the train's mean rate.

    *max_isi_burst* (s) is the interval beyond which spikes are no longer added;
    0 means twice the train's mean ISI (pynapple's ``None``).
    """
    return nap.detect_spike_bursts(
        spikes,
        method="poisson_surprise",
        min_surprise=min_surprise,
        max_isi_burst=None if max_isi_burst == 0 else max_isi_burst,
    )
