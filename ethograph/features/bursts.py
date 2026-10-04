"""Burst detection in spike trains: MaxInterval, logISI and an instantaneous-rate threshold.

The two detectors that came out ahead in Cotterill et al. (2016, J Neurophysiol
116:306, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4969396/), as described in
Cotterill & Eglen (2019, https://arxiv.org/abs/1802.01287), plus the birdsong
field's definition. Checked against the R reference implementations by
``tests/_test_burst_r_reference.py``.

Every detector takes a ``Ts`` or a ``TsGroup`` and returns bursts as an
``IntervalSet`` from first to last spike with an ``n_spikes`` metadata column
(a dict keyed by unit for a ``TsGroup``). Detection runs within each epoch of
the time support, so a burst never spans a gap in it.

These are proposed for pynapple; once it ships them this module is an import
of theirs.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from typing import Any

import numpy as np
import pynapple as nap

#: A run of spikes as (first, last) spike indices, both inclusive.
Run = tuple[int, int]
#: What a detector does with one epoch's spike times.
_FindBursts = Callable[[np.ndarray], list[Run]]

#: A void parameter at or above this separates two peaks of the log-ISI histogram.
_VOID_THRESHOLD = 0.7
#: A valley at or beyond this (seconds) separates bursts from pauses, not spikes within bursts.
_MAX_VALLEY_S = 1.0


def _check_spikes(spikes: Any) -> None:
    if not isinstance(spikes, (nap.Ts, nap.TsGroup)):
        raise TypeError(f"`spikes` must be `Ts` or `TsGroup`, got {type(spikes)}")


def _check_positive(name: str, value: Any, strict: bool = True) -> None:
    if not isinstance(value, numbers.Real):
        raise TypeError(f"`{name}` must be a number, got {type(value)}")
    if strict and value <= 0:
        raise ValueError(f"`{name}` must be > 0")
    if not strict and value < 0:
        raise ValueError(f"`{name}` must be >= 0")


def _check_min_spikes(name: str, value: Any) -> None:
    if not isinstance(value, numbers.Integral):
        raise TypeError(f"`{name}` must be an integer, got {type(value)}")
    if value < 2:
        raise ValueError(f"`{name}` must be >= 2, a burst has at least two spikes")


def _find_runs(times: np.ndarray, max_isi_start: float, max_isi_burst: float) -> list[Run]:
    """Phase 1: a run starts at the first of two spikes closer than *max_isi_start* and
    ends at the last spike before an interval longer than *max_isi_burst*."""
    runs: list[Run] = []
    start: int | None = None
    for i in range(1, len(times)):
        isi = times[i] - times[i - 1]
        if start is None:
            if isi < max_isi_start:
                start = i - 1
        elif isi > max_isi_burst:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(times) - 1))
    return runs


def _merge_close_runs(times: np.ndarray, runs: list[Run], min_inter_burst_interval: float) -> list[Run]:
    """Phase 2: consecutive runs separated by less than *min_inter_burst_interval*
    (last spike of one to first spike of the next) become one run."""
    merged: list[Run] = []
    for first, last in runs:
        if merged and times[first] - times[merged[-1][1]] < min_inter_burst_interval:
            merged[-1] = (merged[-1][0], last)
        else:
            merged.append((first, last))
    return merged


def _drop_small_runs(
    times: np.ndarray, runs: list[Run], min_burst_duration: float, min_spikes_in_burst: int
) -> list[Run]:
    """Phase 3: a run shorter than *min_burst_duration* or with fewer than
    *min_spikes_in_burst* spikes is not a burst."""
    return [
        (first, last)
        for first, last in runs
        if times[last] - times[first] >= min_burst_duration and last - first + 1 >= min_spikes_in_burst
    ]


def _detect_per_epoch(spikes: nap.Ts, find_bursts: _FindBursts) -> nap.IntervalSet:
    """Run *find_bursts* on the spikes of each epoch of the time support, so no burst
    spans a gap in it, and return the bursts as one IntervalSet."""
    starts, ends, n_spikes = [], [], []
    for epoch in spikes.time_support:
        times = spikes.restrict(epoch).t
        for first, last in find_bursts(times):
            # Spikes with identical timestamps span no time and cannot be an interval.
            if times[last] == times[first]:
                continue
            starts.append(times[first])
            ends.append(times[last])
            n_spikes.append(last - first + 1)
    return nap.IntervalSet(start=starts, end=ends, metadata={"n_spikes": n_spikes})


def _apply(spikes: nap.Ts | nap.TsGroup, find_bursts: _FindBursts) -> nap.IntervalSet | dict[Any, nap.IntervalSet]:
    if isinstance(spikes, nap.TsGroup):
        return {k: _detect_per_epoch(spikes[k], find_bursts) for k in spikes}
    return _detect_per_epoch(spikes, find_bursts)


def detect_bursts_max_interval(
    spikes: nap.Ts | nap.TsGroup,
    max_isi_start: float = 0.17,
    max_isi_burst: float = 0.3,
    min_inter_burst_interval: float = 0.2,
    min_burst_duration: float = 0.01,
    min_spikes_in_burst: int = 3,
) -> nap.IntervalSet | dict[Any, nap.IntervalSet]:
    """Bursts by the MaxInterval method of NeuroExplorer.

    A burst starts at the first of two spikes closer than *max_isi_start* and
    continues while consecutive spikes are no further apart than
    *max_isi_burst*. Bursts separated by less than *min_inter_burst_interval*
    are merged; one shorter than *min_burst_duration* or with fewer than
    *min_spikes_in_burst* spikes is discarded. All times are in seconds. The
    parameters are those of figure 3 of Cotterill & Eglen (2019), the
    defaults table 1 of Cotterill et al. (2016).
    """
    _check_spikes(spikes)
    _check_positive("max_isi_start", max_isi_start)
    _check_positive("max_isi_burst", max_isi_burst)
    _check_positive("min_inter_burst_interval", min_inter_burst_interval, strict=False)
    _check_positive("min_burst_duration", min_burst_duration, strict=False)
    _check_min_spikes("min_spikes_in_burst", min_spikes_in_burst)

    def find_bursts(times: np.ndarray) -> list[Run]:
        runs = _find_runs(times, max_isi_start, max_isi_burst)
        runs = _merge_close_runs(times, runs, min_inter_burst_interval)
        return _drop_small_runs(times, runs, min_burst_duration, min_spikes_in_burst)

    return _apply(spikes, find_bursts)


def _log_isi_histogram(isi_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The interspike intervals (ms) on a logarithmic axis: 10 bin edges per decade from
    1 ms to the decade above the longest interval, intervals under 1 ms left out.

    Returns the bins' left edges (ms) and the counts normalised to sum to one.
    """
    decades = max(1, int(np.ceil(np.log10(isi_ms.max()))))
    edges = np.logspace(0, decades, 10 * decades)
    isi_ms = isi_ms[isi_ms >= 1]
    # Bin k holds edges[k] < isi <= edges[k + 1]; 1 ms itself goes to bin 0.
    bins = np.clip(np.searchsorted(edges, isi_ms, side="left") - 1, 0, None)
    counts = np.bincount(bins, minlength=len(edges) - 1)
    return edges[:-1], counts / counts.sum()


def _find_peaks(density: np.ndarray) -> list[int]:
    """Bins strictly higher than every bin within two bins of them, first and last bin excluded."""
    peaks = []
    for k in range(1, len(density) - 1):
        neighbours = np.delete(density[max(0, k - 2) : k + 3], min(k, 2))
        if np.all(density[k] > neighbours):
            peaks.append(k)
    return peaks


def _log_isi_valley(isi_s: np.ndarray, max_cutoff: float) -> tuple[bool, float | None]:
    """The threshold of Pasquale et al. (2010): the ISI at the first valley that separates
    the intra-burst peak of the log-ISI histogram from a later peak.

    Returns whether the histogram has an intra-burst peak at all, and the
    threshold in seconds (``None`` without a well-separated valley).
    """
    isi_ms = isi_s * 1000
    if len(isi_ms) == 0 or not np.any(isi_ms >= 1):
        return False, None
    edges, density = _log_isi_histogram(isi_ms)
    peaks = _find_peaks(density)
    intraburst = [k for k in peaks if edges[k] < max_cutoff * 1000]
    if not intraburst:
        return False, None
    peak = max(intraburst, key=lambda k: density[k])
    for later in peaks:
        if later <= peak:
            continue
        between = density[peak : later + 1]
        valley = peak + int(np.argmin(between))
        void = 1 - between.min() / np.sqrt(density[peak] * density[later])
        if void >= _VOID_THRESHOLD:
            return True, float(edges[valley] / 1000)
    return True, None


def _interspike_intervals(spikes: nap.Ts) -> np.ndarray:
    """The intervals between consecutive spikes, within each epoch of the time support only."""
    # pynapple's time_diff() cannot read a train with no spikes at all.
    if len(spikes) == 0:
        return np.empty(0)
    return spikes.time_diff().values


def compute_log_isi_threshold(
    spikes: nap.Ts | nap.TsGroup, max_cutoff: float = 0.1
) -> float | None | dict[Any, float | None]:
    """The burst threshold of the logISI method, in seconds.

    The interval at the valley between the intra-burst peak of the log-ISI
    histogram (its largest peak below *max_cutoff*) and the first later peak
    well separated from it. ``None`` when the histogram has no intra-burst
    peak or no such valley.
    """
    _check_spikes(spikes)
    _check_positive("max_cutoff", max_cutoff)
    if isinstance(spikes, nap.TsGroup):
        return {k: _log_isi_valley(_interspike_intervals(spikes[k]), max_cutoff)[1] for k in spikes}
    return _log_isi_valley(_interspike_intervals(spikes), max_cutoff)[1]


def _extend_to_related_spikes(times: np.ndarray, cores: list[Run], threshold: float) -> list[Run]:
    """Extend each core to the run of spikes with consecutive intervals no longer than
    *threshold* that contains it; cores inside the same run become one burst."""
    runs = _find_runs(times, threshold, threshold)
    extended: list[Run] = []
    for first, last in cores:
        container = next((r for r in runs if r[0] <= first and last <= r[1]), (first, last))
        if container not in extended:
            extended.append(container)
    return extended


def detect_bursts_log_isi(
    spikes: nap.Ts | nap.TsGroup,
    max_cutoff: float = 0.1,
    min_spikes_in_burst: int = 3,
    fallback: bool = True,
) -> nap.IntervalSet | dict[Any, nap.IntervalSet]:
    """Bursts by the logISI method of Pasquale et al. (2010).

    The threshold between intra- and inter-burst intervals is read off each
    train's own log-ISI histogram (:func:`compute_log_isi_threshold`), over
    every epoch; a burst is a run of at least *min_spikes_in_burst* spikes
    with consecutive intervals no longer than it. A threshold above
    *max_cutoff* finds burst cores at *max_cutoff*, merges cores closer than
    the threshold and extends each to the spikes within the threshold of its
    ends. No intra-burst peak means no bursts. No clear valley (or one at a
    second or more) falls back to *max_cutoff* as the threshold — unless
    *fallback* is off, which returns no bursts for such a train and is not
    part of the published method.
    """
    _check_spikes(spikes)
    _check_positive("max_cutoff", max_cutoff)
    _check_min_spikes("min_spikes_in_burst", min_spikes_in_burst)
    if not isinstance(fallback, bool):
        raise TypeError(f"`fallback` must be a bool, got {type(fallback)}")

    if isinstance(spikes, nap.TsGroup):
        return {k: detect_bursts_log_isi(spikes[k], max_cutoff, min_spikes_in_burst, fallback) for k in spikes}

    has_peak, threshold = _log_isi_valley(_interspike_intervals(spikes), max_cutoff)

    def find_bursts(times: np.ndarray) -> list[Run]:
        if not has_peak:
            return []
        if threshold is None or threshold >= _MAX_VALLEY_S:
            if not fallback:
                return []
            runs = _find_runs(times, max_cutoff, max_cutoff)
            return _drop_small_runs(times, runs, 0, min_spikes_in_burst)
        if threshold > max_cutoff:
            cores = _find_runs(times, max_cutoff, max_cutoff)
            cores = _merge_close_runs(times, cores, threshold)
            cores = _drop_small_runs(times, cores, 0, min_spikes_in_burst)
            return _extend_to_related_spikes(times, cores, threshold)
        runs = _find_runs(times, threshold, threshold)
        return _drop_small_runs(times, runs, 0, min_spikes_in_burst)

    return _detect_per_epoch(spikes, find_bursts)


def detect_bursts_instantaneous_rate(
    spikes: nap.Ts | nap.TsGroup, rate_threshold: float = 125.0, min_spikes_in_burst: int = 2
) -> nap.IntervalSet | dict[Any, nap.IntervalSet]:
    """Bursts as the runs of spikes over which the instantaneous rate exceeds *rate_threshold* (Hz).

    The instantaneous rate between two spikes is the inverse of their
    interval — the burst definition of the birdsong literature, 125 Hz in
    Leonardo & Fee (2005, https://doi.org/10.1523/JNEUROSCI.3036-04.2005).
    """
    _check_spikes(spikes)
    _check_positive("rate_threshold", rate_threshold)
    _check_min_spikes("min_spikes_in_burst", min_spikes_in_burst)
    max_isi = 1.0 / rate_threshold

    def find_bursts(times: np.ndarray) -> list[Run]:
        runs = _find_runs(times, max_isi, max_isi)
        return _drop_small_runs(times, runs, 0, min_spikes_in_burst)

    return _apply(spikes, find_bursts)
