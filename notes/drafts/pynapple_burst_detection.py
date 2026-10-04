"""
Functions to detect bursts in spike trains
https://github.com/pynapple-org/pynapple/issues/668
"""

import numbers

import numpy as np

import pynapple as nap


def _check_spikes(spikes):
    if not isinstance(spikes, (nap.Ts, nap.TsGroup)):
        raise TypeError(f"`spikes` must be `Ts` or `TsGroup`, got {type(spikes)}")


def _check_positive(name, value, strict=True):
    if not isinstance(value, numbers.Real):
        raise TypeError(f"`{name}` must be a number, got {type(value)}")
    if strict and value <= 0:
        raise ValueError(f"`{name}` must be > 0")
    if not strict and value < 0:
        raise ValueError(f"`{name}` must be >= 0")


def _check_min_spikes(name, value):
    if not isinstance(value, numbers.Integral):
        raise TypeError(f"`{name}` must be an integer, got {type(value)}")
    if value < 2:
        raise ValueError(f"`{name}` must be >= 2, a burst has at least two spikes")


def _find_runs(times, max_isi_start, max_isi_burst):
    """
    Phase 1: runs of spikes. A run starts at the first of two spikes closer than
    `max_isi_start` and ends at the last spike before an interval longer than
    `max_isi_burst`. Returns (first, last) spike indices, both inclusive.
    """
    runs = []
    start = None
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


def _merge_close_runs(times, runs, min_inter_burst_interval):
    """
    Phase 2: two consecutive runs separated by less than `min_inter_burst_interval`
    (last spike of one to first spike of the next) become one run.
    """
    merged = []
    for first, last in runs:
        if merged and times[first] - times[merged[-1][1]] < min_inter_burst_interval:
            merged[-1] = (merged[-1][0], last)
        else:
            merged.append((first, last))
    return merged


def _drop_small_runs(times, runs, min_burst_duration, min_spikes_in_burst):
    """
    Phase 3: a run shorter than `min_burst_duration` or with fewer than
    `min_spikes_in_burst` spikes is not a burst.
    """
    return [
        (first, last)
        for first, last in runs
        if times[last] - times[first] >= min_burst_duration
        and last - first + 1 >= min_spikes_in_burst
    ]


def _detect_per_epoch(spikes, find_bursts):
    """
    Run `find_bursts(times)` on the spikes of each epoch of the time support,
    so no burst spans a gap in it, and return the bursts as one IntervalSet.
    """
    starts, ends, n_spikes = [], [], []
    for epoch in spikes.time_support:
        times = spikes.restrict(epoch).t
        for first, last in find_bursts(times):
            # Skip identical spike times (artifacts)
            if times[last] == times[first]:
                continue
            starts.append(times[first])
            ends.append(times[last])
            n_spikes.append(last - first + 1)
    return nap.IntervalSet(start=starts, end=ends, metadata={"n_spikes": n_spikes})


def _apply(spikes, find_bursts):
    if isinstance(spikes, nap.TsGroup):
        return {k: _detect_per_epoch(spikes[k], find_bursts) for k in spikes}
    return _detect_per_epoch(spikes, find_bursts)


def detect_bursts_max_interval(
    spikes,
    max_isi_start=0.17,
    max_isi_burst=0.3,
    min_inter_burst_interval=0.2,
    min_burst_duration=0.01,
    min_spikes_in_burst=3,
):
    """
    Detect bursts with the MaxInterval method of NeuroExplorer.

    A burst starts at the first of two spikes closer than `max_isi_start`
    and continues while consecutive spikes are no further apart than
    `max_isi_burst`. Bursts separated by less than `min_inter_burst_interval`
    are merged. A burst shorter than `min_burst_duration` or with fewer than
    `min_spikes_in_burst` spikes is discarded. The parameters are those of
    figure 3 of Cotterill & Eglen (2019); the defaults are the values used
    by Cotterill et al. (2016).

    Bursts are detected within each epoch of the time support separately, so
    a burst never spans a gap in the time support.

    Parameters
    ----------
    spikes : Ts or TsGroup
        Spike times.
    max_isi_start : float, optional
        Maximum interval (in seconds) between the first two spikes of a burst.
    max_isi_burst : float, optional
        Maximum interval (in seconds) between two spikes within a burst.
    min_inter_burst_interval : float, optional
        Bursts closer than this (in seconds, last spike to first spike) are merged.
    min_burst_duration : float, optional
        Minimum duration (in seconds) of a burst, first spike to last spike.
    min_spikes_in_burst : int, optional
        Minimum number of spikes in a burst.

    Returns
    -------
    IntervalSet or dict of IntervalSet
        Bursts, from first spike to last spike, with the metadata column
        `n_spikes`. A dictionary keyed by unit for a `TsGroup`.

    Raises
    ------
    TypeError
        If `spikes` is not a `Ts` or a `TsGroup`, or a parameter has the wrong type.
    ValueError
        If a parameter is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> t = np.concatenate([[0.0, 0.1, 0.2, 0.3], [2.0, 2.05, 2.1], [5.0]])
    >>> spikes = nap.Ts(t=t)
    >>> nap.detect_bursts_max_interval(spikes)
      index    start    end    n_spikes
          0        0    0.3           4
          1        2    2.1           3
    shape: (2, 2), time unit: sec.

    Spikes in a `TsGroup` are processed unit by unit:

    >>> group = nap.TsGroup({0: spikes, 1: nap.Ts(t=t + 0.5)})
    >>> bursts = nap.detect_bursts_max_interval(group)
    >>> bursts[1]
      index    start    end    n_spikes
          0      0.5    0.8           4
          1      2.5    2.6           3
    shape: (2, 2), time unit: sec.

    References
    ----------
    .. [1] Cotterill, E., & Eglen, S. J. (2019). Burst detection methods.
       In: In Vitro Neuronal Networks, Springer, pp. 185-206.
       https://arxiv.org/abs/1802.01287
    .. [2] Cotterill, E., Charlesworth, P., Thomas, C. W., Paulsen, O., & Eglen,
       S. J. (2016). A comparison of computational methods for detecting bursts
       in neuronal spike trains and their application to human stem cell-derived
       neuronal networks. Journal of Neurophysiology, 116(2), 306-321.
       https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4969396/
    """
    _check_spikes(spikes)
    _check_positive("max_isi_start", max_isi_start)
    _check_positive("max_isi_burst", max_isi_burst)
    _check_positive("min_inter_burst_interval", min_inter_burst_interval, strict=False)
    _check_positive("min_burst_duration", min_burst_duration, strict=False)
    _check_min_spikes("min_spikes_in_burst", min_spikes_in_burst)

    def find_bursts(times):
        runs = _find_runs(times, max_isi_start, max_isi_burst)
        runs = _merge_close_runs(times, runs, min_inter_burst_interval)
        return _drop_small_runs(times, runs, min_burst_duration, min_spikes_in_burst)

    return _apply(spikes, find_bursts)


def _log_isi_histogram(isi_ms):
    """
    Histogram of the interspike intervals (in milliseconds) on a logarithmic
    axis: 10 bin edges per decade from 1 ms to the decade above the longest
    interval, intervals under 1 ms left out, counts normalised to sum to one.
    Returns (left edges in ms, normalised counts).
    """
    decades = max(1, int(np.ceil(np.log10(isi_ms.max()))))
    edges = np.logspace(0, decades, 10 * decades)
    isi_ms = isi_ms[isi_ms >= 1]
    # Bin k holds edges[k] < isi <= edges[k + 1]; 1 ms itself goes to bin 0.
    bins = np.clip(np.searchsorted(edges, isi_ms, side="left") - 1, 0, None)
    counts = np.bincount(bins, minlength=len(edges) - 1)
    return edges[:-1], counts / counts.sum()


def _find_peaks(density):
    """
    Bins strictly higher than every bin within two bins of them, excluding the
    first and last bin.
    """
    peaks = []
    for k in range(1, len(density) - 1):
        neighbours = np.delete(density[max(0, k - 2) : k + 3], min(k, 2))
        if np.all(density[k] > neighbours):
            peaks.append(k)
    return peaks


def _log_isi_valley(isi_s, max_cutoff):
    """
    The threshold of Pasquale et al. (2010): the ISI at the first valley that
    separates the intra-burst peak of the log-ISI histogram from a later peak
    with a void parameter of at least 0.7.

    Returns (has_intraburst_peak, threshold in seconds or None).
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
        if void >= 0.7:
            return True, float(edges[valley] / 1000)
    return True, None


def _interspike_intervals(spikes):
    """
    The intervals between consecutive spikes, within each epoch of the time
    support only.
    """
    # time_diff() cannot read a train with no spikes at all.
    if len(spikes) == 0:
        return np.empty(0)
    return spikes.time_diff().values


def compute_log_isi_threshold(spikes, max_cutoff=0.1):
    """
    Compute the burst threshold of the logISI method of Pasquale et al. (2010).

    The histogram of the interspike intervals on a logarithmic axis typically
    has one peak for the intervals within bursts and one for the intervals
    between them. The threshold is the interval at the valley between the
    intra-burst peak (the largest peak below `max_cutoff`) and the first later
    peak that is well separated from it, i.e. whose void parameter
    ``1 - min / sqrt(peak1 * peak2)`` is at least 0.7. The histogram has ten
    bins per decade and is not smoothed: at the span the reference
    implementations use (5 % of at most 60 bins) their smoothing leaves it
    unchanged.

    Parameters
    ----------
    spikes : Ts or TsGroup
        Spike times.
    max_cutoff : float, optional
        Largest interval (in seconds) at which the intra-burst peak may lie.

    Returns
    -------
    float, None or dict
        The threshold in seconds, or None when the histogram has no intra-burst
        peak or no valley well separated from it. A dictionary keyed by unit
        for a `TsGroup`.

    Raises
    ------
    TypeError
        If `spikes` is not a `Ts` or a `TsGroup`, or `max_cutoff` is not a number.
    ValueError
        If `max_cutoff` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> starts = np.arange(200) * 1.5
    >>> t = np.concatenate([s + np.arange(5) * 0.01 for s in starts])
    >>> spikes = nap.Ts(t=t)
    >>> round(nap.compute_log_isi_threshold(spikes), 4)
    0.0106

    References
    ----------
    .. [1] Pasquale, V., Martinoia, S., & Chiappalone, M. (2010). A self-adapting
       approach for the detection of bursts and network bursts in neuronal
       cultures. Journal of Computational Neuroscience, 29(1-2), 213-229.
       https://doi.org/10.1007/s10827-009-0175-1
    .. [2] Cotterill, E., & Eglen, S. J. (2019). Burst detection methods.
       In: In Vitro Neuronal Networks, Springer, pp. 185-206.
       https://arxiv.org/abs/1802.01287
    """
    _check_spikes(spikes)
    _check_positive("max_cutoff", max_cutoff)
    if isinstance(spikes, nap.TsGroup):
        return {k: compute_log_isi_threshold(spikes[k], max_cutoff) for k in spikes}
    return _log_isi_valley(_interspike_intervals(spikes), max_cutoff)[1]


def detect_bursts_log_isi(spikes, max_cutoff=0.1, min_spikes_in_burst=3, fallback=True):
    """
    Detect bursts with the logISI method of Pasquale et al. (2010).

    The threshold between intra- and inter-burst intervals is read off the
    log-ISI histogram of each spike train (see `compute_log_isi_threshold`).
    A burst is a run of at least `min_spikes_in_burst` spikes with consecutive
    intervals no longer than the threshold.

    When the threshold lies above `max_cutoff`, the burst cores are first
    found with `max_cutoff` as the threshold, cores closer than the threshold
    are merged, and each core is then extended to the spikes within the
    threshold of its first and last spike (the "burst-related spikes").
    When the histogram has no intra-burst peak the train has no bursts.
    When it has a peak but no clear valley, or the valley lies at one second
    or more, `max_cutoff` is used as the threshold (the fallback of the
    published method) unless `fallback` is False, in which case no bursts
    are returned.

    Bursts are detected within each epoch of the time support separately, so
    a burst never spans a gap in the time support; the histogram is built
    from the intervals within epochs.

    Parameters
    ----------
    spikes : Ts or TsGroup
        Spike times.
    max_cutoff : float, optional
        Largest interval (in seconds) at which the intra-burst peak may lie,
        and the fallback threshold.
    min_spikes_in_burst : int, optional
        Minimum number of spikes in a burst.
    fallback : bool, optional
        Whether to fall back to `max_cutoff` when the histogram has no clear
        valley. The published method always does; `False` returns no bursts
        for such a train instead, which is the safer choice when the method
        is applied to units without first looking at their histograms.

    Returns
    -------
    IntervalSet or dict of IntervalSet
        Bursts, from first spike to last spike, with the metadata column
        `n_spikes`. A dictionary keyed by unit for a `TsGroup`.

    Raises
    ------
    TypeError
        If `spikes` is not a `Ts` or a `TsGroup`, or a parameter has the wrong type.
    ValueError
        If a parameter is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> starts = np.arange(200) * 1.5
    >>> t = np.concatenate([s + np.arange(5) * 0.01 for s in starts])
    >>> spikes = nap.Ts(t=t)
    >>> bursts = nap.detect_bursts_log_isi(spikes)
    >>> len(bursts), int(bursts.n_spikes.min()), int(bursts.n_spikes.max())
    (200, 5, 5)

    References
    ----------
    .. [1] Pasquale, V., Martinoia, S., & Chiappalone, M. (2010). A self-adapting
       approach for the detection of bursts and network bursts in neuronal
       cultures. Journal of Computational Neuroscience, 29(1-2), 213-229.
       https://doi.org/10.1007/s10827-009-0175-1
    .. [2] Cotterill, E., & Eglen, S. J. (2019). Burst detection methods.
       In: In Vitro Neuronal Networks, Springer, pp. 185-206.
       https://arxiv.org/abs/1802.01287
    .. [3] Cotterill, E., Charlesworth, P., Thomas, C. W., Paulsen, O., & Eglen,
       S. J. (2016). A comparison of computational methods for detecting bursts
       in neuronal spike trains and their application to human stem cell-derived
       neuronal networks. Journal of Neurophysiology, 116(2), 306-321.
       https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4969396/
    """
    _check_spikes(spikes)
    _check_positive("max_cutoff", max_cutoff)
    _check_min_spikes("min_spikes_in_burst", min_spikes_in_burst)
    if not isinstance(fallback, bool):
        raise TypeError(f"`fallback` must be a bool, got {type(fallback)}")

    if isinstance(spikes, nap.TsGroup):
        return {
            k: detect_bursts_log_isi(spikes[k], max_cutoff, min_spikes_in_burst, fallback)
            for k in spikes
        }

    has_peak, threshold = _log_isi_valley(_interspike_intervals(spikes), max_cutoff)
    no_valley = threshold is None or threshold >= 1.0

    def find_bursts(times):
        if not has_peak or (no_valley and not fallback):
            return []
        if no_valley:
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


def _extend_to_related_spikes(times, cores, threshold):
    """
    Extend each core to the run of spikes with consecutive intervals no longer
    than `threshold` that contains it. Cores inside the same run become one burst.
    """
    runs = _find_runs(times, threshold, threshold)
    extended = []
    for first, last in cores:
        container = next((r for r in runs if r[0] <= first and last <= r[1]), (first, last))
        if container not in extended:
            extended.append(container)
    return extended


def detect_bursts_instantaneous_rate(spikes, rate_threshold=125.0, min_spikes_in_burst=2):
    """
    Detect bursts as the intervals over which the instantaneous firing rate
    exceeds a threshold.

    The instantaneous rate between two spikes is the inverse of their interval,
    so a burst is a run of spikes whose consecutive intervals are all shorter
    than ``1 / rate_threshold``. This is the burst definition of the birdsong
    literature, with the 125 Hz of Leonardo & Fee (2005) as the default.

    Bursts are detected within each epoch of the time support separately, so
    a burst never spans a gap in the time support.

    Parameters
    ----------
    spikes : Ts or TsGroup
        Spike times.
    rate_threshold : float, optional
        Instantaneous firing rate (in Hz) above which spikes belong to a burst.
    min_spikes_in_burst : int, optional
        Minimum number of spikes in a burst.

    Returns
    -------
    IntervalSet or dict of IntervalSet
        Bursts, from first spike to last spike, with the metadata column
        `n_spikes`. A dictionary keyed by unit for a `TsGroup`.

    Raises
    ------
    TypeError
        If `spikes` is not a `Ts` or a `TsGroup`, or a parameter has the wrong type.
    ValueError
        If a parameter is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> spikes = nap.Ts(t=[0.0, 0.005, 0.01, 0.1, 0.2, 0.202, 0.3])
    >>> nap.detect_bursts_instantaneous_rate(spikes, rate_threshold=125.0)
      index    start    end    n_spikes
          0      0    0.01            3
          1      0.2  0.202           2
    shape: (2, 2), time unit: sec.

    References
    ----------
    .. [1] Leonardo, A., & Fee, M. S. (2005). Ensemble coding of vocal control
       in birdsong. Journal of Neuroscience, 25(3), 652-661.
       https://doi.org/10.1523/JNEUROSCI.3036-04.2005
    """
    _check_spikes(spikes)
    _check_positive("rate_threshold", rate_threshold)
    _check_min_spikes("min_spikes_in_burst", min_spikes_in_burst)
    max_isi = 1.0 / rate_threshold

    def find_bursts(times):
        runs = _find_runs(times, max_isi, max_isi)
        return _drop_small_runs(times, runs, 0, min_spikes_in_burst)

    return _apply(spikes, find_bursts)
