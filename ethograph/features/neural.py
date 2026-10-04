"""Compute firing rates from spike times using pynapple.

Assumes spike_times are in seconds and sorted ascending (standard for Kilosort/Phy).

The segmentation pipeline's spike input lives here too (:func:`transform_units`,
:func:`sliding_window`): a session's units arrive as a :class:`pynapple.TsGroup`
— spike *times*, not a time series — so nothing can read them as a feature
until they are binned, and how (bin size, smoothing, a rate versus a count)
is a modelling choice worth sweeping. A project config spells it as a list
of pynapple expressions rather than baking it into a file::

    features:
      neural:
        units: units                 # the TsGroup's key in the session
        name: rate                   # the feature the transform produces
        transform:
          - x.count(0.005) / 0.005   # spikes per second in 5 ms bins
          - sliding_window(x, window_size=0.025)

Each step is evaluated with ``x`` bound to the previous step's result (the
``TsGroup`` for the first), and ``nap``, ``np`` and :func:`sliding_window`
in scope. The last step must leave a :class:`pynapple.TsdFrame` — one column
per unit — which then is a feature like any other in the session. The raw
spikes are never written out as a feature file; the transform runs at
session open, every time.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pynapple as nap
import xarray as xr


def build_tsgroup(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    cluster_ids: np.ndarray | None = None,
    time_support: nap.IntervalSet | None = None,
) -> nap.TsGroup:
    """Build a pynapple TsGroup from flat spike arrays.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times in seconds, shape ``(N,)``.
    spike_clusters : numpy.ndarray
        Cluster ID for each spike, shape ``(N,)``.
    cluster_ids : numpy.ndarray, optional
        Subset of clusters to include. Defaults to all unique.
    time_support : pynapple.IntervalSet, optional
        Epoch boundaries. Defaults to full data range.

    Returns
    -------
    pynapple.TsGroup
        One :class:`pynapple.Ts` per cluster.
    """
    spike_times = spike_times.ravel()
    spike_clusters = spike_clusters.ravel()

    if cluster_ids is None:
        cluster_ids = np.unique(spike_clusters)

    if time_support is None:
        time_support = nap.IntervalSet(spike_times[0], spike_times[-1])

    units = {}
    for cid in cluster_ids:
        mask = spike_clusters == cid
        units[int(cid)] = nap.Ts(t=spike_times[mask], time_support=time_support)

    return nap.TsGroup(units, time_support=time_support)


@dataclass(frozen=True)
class SpikeTable:
    """Every spike of a ``TsGroup`` in one time-sorted table.

    A window of it is one binary search, however many units there are —
    restricting the group unit by unit is one pynapple call each.
    """

    times: np.ndarray
    units: np.ndarray

    @classmethod
    def from_tsgroup(cls, tsgroup: nap.TsGroup) -> "SpikeTable":
        unit_ids = [int(uid) for uid in tsgroup.keys()]
        trains = [np.asarray(tsgroup[uid].times(), dtype=np.float64) for uid in unit_ids]
        if not trains:
            return cls(np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64))
        times = np.concatenate(trains)
        units = np.repeat(np.asarray(unit_ids, dtype=np.int64), [len(train) for train in trains])
        order = np.argsort(times, kind="stable")
        return cls(times[order], units[order])

    def window(self, t_start: float, t_stop: float) -> tuple[np.ndarray, np.ndarray]:
        """Spike times and their unit ids inside ``[t_start, t_stop]``."""
        i0 = int(np.searchsorted(self.times, t_start, side="left"))
        i1 = int(np.searchsorted(self.times, t_stop, side="right"))
        return self.times[i0:i1], self.units[i0:i1]

    def count(
        self, unit_ids: Sequence[int], t_start: float, t_stop: float, bin_size: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spike counts of ``unit_ids`` in bins of ``bin_size`` seconds from ``t_start``.

        Returns the counts, shape ``(n_bins, n_units)`` with the columns in
        the order of ``unit_ids``, and the centre of each bin. Only whole bins
        are counted; a remainder shorter than a bin is left out.
        """
        if bin_size <= 0:
            raise ValueError(f"bin_size must be positive, got {bin_size}")
        columns = np.asarray(unit_ids, dtype=np.int64)
        n_bins = int(np.floor((t_stop - t_start) / bin_size + 1e-9))
        centers = t_start + (np.arange(n_bins) + 0.5) * bin_size
        if n_bins <= 0 or len(columns) == 0:
            return np.zeros((max(n_bins, 0), len(columns)), dtype=np.int64), centers

        times, units = self.window(t_start, t_start + n_bins * bin_size)
        by_id = np.argsort(columns, kind="stable")
        found = np.searchsorted(columns, units, sorter=by_id)
        found[found == len(columns)] = 0
        column = by_id[found]
        bin_index = np.floor((times - t_start) / bin_size).astype(np.int64)
        keep = (columns[column] == units) & (bin_index < n_bins)
        flat = bin_index[keep] * len(columns) + column[keep]
        counts = np.bincount(flat, minlength=n_bins * len(columns))
        return counts.reshape(n_bins, len(columns)), centers


def firing_rate_by_cluster(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    bin_size: float,
    t_start: float | None = None,
    t_stop: float | None = None,
    cluster_ids: np.ndarray | None = None,
    _tsgroup: nap.TsGroup | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin spike times per cluster into firing rate curves via pynapple.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times in seconds, sorted ascending, shape ``(N,)``.
    spike_clusters : numpy.ndarray
        Cluster ID for each spike, shape ``(N,)``.
    bin_size : float
        Bin width in seconds.
    t_start : float, optional
        Start of time range. Defaults to first spike time.
    t_stop : float, optional
        End of time range. Defaults to last spike time.
    cluster_ids : numpy.ndarray, optional
        Which clusters to include. Defaults to all unique.
    _tsgroup : pynapple.TsGroup, optional
        Pre-built TsGroup; pass to skip reconstruction.

    Returns
    -------
    rates : numpy.ndarray
        Firing rate in Hz, shape ``(n_clusters, n_bins)``.
    bin_centers : numpy.ndarray
        Time of each bin center, shape ``(n_bins,)``.
    cluster_ids : numpy.ndarray
        Cluster ID for each row, shape ``(n_clusters,)``.
    """
    if _tsgroup is not None:
        if cluster_ids is not None:
            # pynapple refuses a subset whose keys are not in its own (sorted) order.
            _tsgroup = _tsgroup[np.sort(np.asarray(cluster_ids))]
        cluster_ids = np.array(list(_tsgroup.keys()))
        if t_start is None:
            t_start = float(_tsgroup.time_support.start[0])
        if t_stop is None:
            t_stop = float(_tsgroup.time_support.end[-1])
        time_support = nap.IntervalSet(t_start, t_stop)
        _tsgroup = _tsgroup.restrict(time_support)
    else:
        if cluster_ids is None:
            cluster_ids = np.unique(spike_clusters.ravel())
        if t_start is None:
            t_start = float(spike_times.ravel()[0])
        if t_stop is None:
            t_stop = float(spike_times.ravel()[-1])
        time_support = nap.IntervalSet(t_start, t_stop)
        _tsgroup = build_tsgroup(
            spike_times,
            spike_clusters,
            cluster_ids,
            time_support,
        )

    counts = _tsgroup.count(bin_size=bin_size)
    rates = (counts.values / bin_size).T
    bin_centers = counts.times()

    return rates, bin_centers, cluster_ids


def firing_rate_to_xarray(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    bin_size: float,
    t_start: float | None = None,
    t_stop: float | None = None,
    cluster_ids: np.ndarray | None = None,
    _tsgroup: nap.TsGroup | None = None,
):
    """Compute firing rates and return as an xarray DataArray.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times in seconds, shape ``(N,)``.
    spike_clusters : numpy.ndarray
        Cluster ID for each spike, shape ``(N,)``.
    bin_size : float
        Bin width in seconds.
    t_start : float, optional
        Start of time range.
    t_stop : float, optional
        End of time range.
    cluster_ids : numpy.ndarray, optional
        Which clusters to include.
    _tsgroup : pynapple.TsGroup, optional
        Pre-built TsGroup.

    Returns
    -------
    xarray.DataArray
        Firing rates with dims ``("cluster_id", "time_fr")`` and
        ``attrs["bin_size"]``.
    """
    rates, bin_centers, cluster_ids = firing_rate_by_cluster(
        spike_times,
        spike_clusters,
        bin_size,
        t_start,
        t_stop,
        cluster_ids,
        _tsgroup,
    )

    return xr.DataArray(
        data=rates,
        dims=("cluster_id", "time_fr"),
        coords={
            "cluster_id": cluster_ids,
            "time_fr": bin_centers,
        },
        attrs={"bin_size": bin_size, "units": "Hz"},
    )


<<<<<<< HEAD
#: The instantaneous rate above which the birdsong literature calls spikes a burst:
#: "A burst was defined as the interval over which the instantaneous firing rate
#: exceeded a threshold of 125 Hz" — Leonardo A, Fee MS (2005). Ensemble coding of
#: vocal control in birdsong. J Neurosci 25(3):652-661,
#: https://doi.org/10.1523/JNEUROSCI.3036-04.2005. Other songbird studies use 100 Hz.
#: A starting point for a threshold the user sets, tuned to RA neurons — not a constant
#: of nature.
BURST_INSTANTANEOUS_RATE_HZ = 125.0


=======
>>>>>>> 4757530969aedb060f3f6bf02ad1d359e363cab4
def instantaneous_rate(spike_times: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The instantaneous firing rate at each of *at*: ``1 / ISI`` of the inter-spike interval it falls in.

    The birdsong field's definition — from each spike until the next, the
    rate is the reciprocal of that interval — so the rate is a step function
    with no bin and no smoothing, and a burst reads as high from its first
    spike. Before the first spike and from the last one on there is no
    interval to read, and the rate is 0. *spike_times* is sorted ascending;
    spikes at the same instant count once.
    """
    spikes = np.unique(np.asarray(spike_times, dtype=np.float64))
    at = np.asarray(at, dtype=np.float64)
    rate = np.zeros(len(at), dtype=np.float64)
    if len(spikes) < 2:
        return rate
    interval = np.searchsorted(spikes, at, side="right") - 1
    inside = (interval >= 0) & (interval < len(spikes) - 1)
    rate[inside] = 1.0 / np.diff(spikes)[interval[inside]]
    return rate


# ---------------------------------------------------------------------------
# The segmentation pipeline's spike input: ``features.neural.transform``
# ---------------------------------------------------------------------------

_STEP_VARIABLE = "x"


def sliding_window(
    binned: nap.Tsd | nap.TsdFrame,
    window_size: float,
    step_size: float | None = None,
    reduction: Literal["sum", "mean"] = "mean",
) -> Any:
    """Smooth a binned ``TsdFrame`` with a boxcar of *window_size* seconds.

    The window is expressed in seconds and turned into bins off the frame's
    own spacing, so the same config line means the same thing at any bin
    size. ``reduction="mean"`` keeps the units (a rate stays a rate);
    ``"sum"`` turns counts per bin into counts per window. *step_size*
    (seconds) additionally decimates the result to that spacing — leave it
    unset to keep the bin grid, which is what a frame-wise model wants.
    Spike times themselves cannot be windowed; bin them first (``x.count``).
    """
    if not isinstance(binned, (nap.Tsd, nap.TsdFrame)):
        raise TypeError(
            f"sliding_window expects a binned Tsd/TsdFrame (x.count(bin_size) first), got {type(binned).__name__}"
        )
    if reduction not in ("sum", "mean"):
        raise ValueError(f"sliding_window reduction must be 'sum' or 'mean', got {reduction!r}")
    t = np.asarray(binned.t, dtype=np.float64)
    if t.size < 2:
        raise ValueError("sliding_window needs at least two bins to read a bin size off")
    bin_size = float(np.median(np.diff(t)))
    window_bins = max(1, int(round(window_size / bin_size)))
    kernel = np.ones(window_bins, dtype=np.float64)
    if reduction == "mean":
        kernel /= window_bins
    smoothed = binned.convolve(kernel)
    if step_size is None:
        return smoothed
    step = max(1, int(round(step_size / bin_size)))
    return smoothed[::step]


def transform_namespace() -> dict[str, Any]:
    """What a transform step can name besides ``x``."""
    return {"nap": nap, "np": np, "sliding_window": sliding_window}


def transform_units(units: nap.TsGroup, steps: Sequence[str], *, what: str = "features.neural.transform") -> Any:
    """Run *steps* over *units* and return the ``TsdFrame`` they end in.

    *what* names the config key in every error, since the step that failed
    is a line the user wrote. A step that raises, or a chain that does not
    end in a ``TsdFrame``, is a ``ValueError`` naming the step.
    """
    if not isinstance(units, nap.TsGroup):
        raise ValueError(f"{what}: expected a pynapple TsGroup to start from, got {type(units).__name__}")
    if not steps:
        raise ValueError(f"{what} is empty — at least bin the spikes, e.g. 'x.count(0.005)'")
    namespace = transform_namespace()
    x: Any = units
    for i, step in enumerate(steps):
        label = f"{what}[{i}] {step!r}"
        try:
            code = compile(step, f"<{what}[{i}]>", "eval")
        except SyntaxError as exc:
            raise ValueError(f"{label} is not a Python expression: {exc}") from exc
        try:
            x = eval(code, {**namespace, _STEP_VARIABLE: x})  # noqa: S307 — the user's own config line
        except Exception as exc:
            raise ValueError(f"{label} failed: {exc}") from exc
    if not isinstance(x, nap.TsdFrame):
        raise ValueError(
            f"{what} must end in a pynapple TsdFrame (one column per unit), got {type(x).__name__} — "
            "a TsGroup still needs binning (x.count(bin_size)); a Tsd is one unit, not a frame."
        )
    if x.shape[0] < 2:
        raise ValueError(f"{what} produced {x.shape[0]} time bin(s) — nothing to read a sampling rate off")
    return x
