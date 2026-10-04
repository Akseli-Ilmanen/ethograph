"""Labels read off the data: the periods a feature spends past a threshold, or a unit spends bursting.

A question asked of the data ("when is this unit above 20 Hz?", "when is the
beak tip below the perch?") answered as an ordinary labels table, so every
tool that reads labels — navigation, playback, the review grids — works on
the answer. The table is a **read-only label source** held in memory
(:class:`~ethograph.labels.predictions.PredictionSet`), never the session's
``labels.tsv``: nobody judged these periods, and series that are past the
threshold at the same time overlap, which one actor's labels may not.

One feature gives one class, named after the rule. A feature thresholded
along a dim (every unit, every keypoint) gives one class per value of that
dim, named ``{dim}_{value}`` — the vocabulary lives beside the table
(:func:`event_mappings`), not in ``mapping.txt``.

The threshold is in the feature's own units, or on a scale of each series'
own (``ThresholdRule.scale``): standard deviations from its mean, or a
percentile of its values, across the trials read (:func:`sample_series`).
Along a dim that is what makes one number mean the same thing for every
series: units with different baseline rates share no raw threshold. A
z-score asks "how unusual is this for the series"; a percentile asks for a
fixed share of each series' time and assumes nothing about its distribution.

Short blips are removed while the periods are built (:func:`clean_intervals`):
periods separated by less than ``stitch_gap_s`` become one, then whatever is
still shorter than ``min_duration_s`` is dropped — the comparisons
:func:`~ethograph.labels.intervals.stitch_intervals` and
:func:`~ethograph.labels.intervals.purge_short_intervals` make, on arrays.

A threshold is one method; a burst detector is the other
(:func:`burst_events`). It reads spike trains instead of a feature — events
too short for a rate to resolve — and gives the same table, one class per
unit. Which detector, and with what parameters, is a :class:`BurstRule`.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pynapple as nap

from ethograph.features.bursts import detect_bursts_log_isi, detect_bursts_max_interval
from ethograph.io.catalog import INDIVIDUAL_DIMS
from ethograph.labels.intervals import (
    EVENT_TYPE_STATE,
    HUMAN_CONFIDENCE,
    INTERVAL_COLUMNS,
    INTERVAL_DTYPES,
    LABELING_AUTOMATED,
    NO_RECIPIENT,
    label_color,
)

ABOVE = "above"
BELOW = "below"
DIRECTIONS = (ABOVE, BELOW)

_SYMBOL = {ABOVE: ">", BELOW: "<"}

#: What the threshold is measured in.
RAW = "raw"
ZSCORE = "zscore"
PERCENTILE = "percentile"
SCALES = (RAW, ZSCORE, PERCENTILE)

_SCALE_SUFFIX = {RAW: "", ZSCORE: "z", PERCENTILE: "%"}

#: The columns of an events table: a labels table, trial first.
EVENT_COLUMNS = ["trial", *INTERVAL_COLUMNS]


@dataclass(frozen=True)
class ThresholdRule:
    """Which samples count, and how short a period or a gap may be."""

    threshold: float
    direction: str = ABOVE
    #: Periods closer together than this are one period.
    stitch_gap_s: float = 0.0
    #: A period shorter than this (after stitching) is dropped.
    min_duration_s: float = 0.0
    #: What *threshold* is measured in: the feature's units (:data:`RAW`),
    #: standard deviations from each series' own mean (:data:`ZSCORE`), or a
    #: percentile of each series' own values (:data:`PERCENTILE`) — the last
    #: two taken across the trials read.
    scale: str = RAW

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}, got {self.scale!r}")
        if self.scale == PERCENTILE and not 0.0 <= self.threshold <= 100.0:
            raise ValueError(f"A percentile is between 0 and 100, got {self.threshold:g}")
        if self.stitch_gap_s < 0 or self.min_duration_s < 0:
            raise ValueError("stitch_gap_s and min_duration_s must not be negative")

    def describe(self) -> str:
        """``">20"`` / ``"<0.5"`` / ``">2z"`` / ``">95%"`` — the rule as it reads in a class or set name."""
        return f"{_SYMBOL[self.direction]}{self.threshold:g}{_SCALE_SUFFIX[self.scale]}"

    def passes(self, values: np.ndarray, cut: float | np.ndarray | None = None) -> np.ndarray:
        """Which of *values* are past *cut* (the threshold itself by default); a NaN never is.

        *cut* is the threshold in the values' own units — what a series'
        z-score or percentile comes to (:meth:`SeriesSample.cuts`).
        """
        cut = self.threshold if cut is None else cut
        with np.errstate(invalid="ignore"):
            return values > cut if self.direction == ABOVE else values < cut


def clean_intervals(intervals: np.ndarray, stitch_gap_s: float = 0.0, min_duration_s: float = 0.0) -> np.ndarray:
    """Stitch ``(n, 2)`` onset/offset rows across short gaps, then drop the short ones.

    *intervals* is sorted by onset and non-overlapping. Stitching comes first,
    so a period broken up by brief dips is judged by its whole length.
    """
    intervals = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    if len(intervals) == 0:
        return intervals
    onsets, offsets = intervals[:, 0], intervals[:, 1]
    if stitch_gap_s > 0 and len(onsets) > 1:
        starts_run = np.concatenate([[True], (onsets[1:] - offsets[:-1]) >= stitch_gap_s])
        first = np.flatnonzero(starts_run)
        last = np.concatenate([first[1:] - 1, [len(onsets) - 1]])
        onsets, offsets = onsets[first], offsets[last]
    keep = (offsets - onsets) >= min_duration_s
    return np.column_stack([onsets[keep], offsets[keep]])


def threshold_intervals(
    time: np.ndarray, values: np.ndarray, rule: ThresholdRule, cut: float | None = None
) -> np.ndarray:
    """The periods *values* spends past the threshold, as ``(n, 2)`` onset/offset times.

    A period runs from the first sample past the threshold to the first
    sample back on the other side (the last sample, when the series ends
    past it), on the clock *time* is given in. *cut* is the threshold in the
    values' units when the rule's own is on another scale.
    """
    time = np.asarray(time, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if time.ndim != 1 or values.shape != time.shape:
        raise ValueError(f"time {time.shape} and values {values.shape} must be the same 1-D shape")
    passing = rule.passes(values, cut)
    if not passing.any():
        return np.empty((0, 2), dtype=np.float64)
    edges = np.diff(passing.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    stops = np.minimum(np.flatnonzero(edges == -1), len(time) - 1)
    intervals = np.column_stack([time[starts], time[stops]])
    # A single passing sample at the very end spans no time at all.
    intervals = intervals[intervals[:, 1] > intervals[:, 0]]
    return clean_intervals(intervals, rule.stitch_gap_s, rule.min_duration_s)


@dataclass(frozen=True)
class FeatureQuery:
    """Which series of a feature are thresholded.

    Every dim of the feature is either pinned to one value (*selections*) or
    is the one *dim* read value by value — each of *values* a class of its
    own, in that order.
    """

    feature: str
    selections: dict[str, str] = field(default_factory=dict)
    dim: str | None = None
    values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dim is not None and not self.values:
            raise ValueError(f"No values given along {self.dim!r}")
        if self.dim is not None and self.dim in self.selections:
            raise ValueError(f"{self.dim!r} is both pinned and read value by value")

    def class_names(self, rule: ThresholdRule) -> list[str]:
        """One name per class: ``{dim}_{value}`` along a dim, the rule itself for a single series."""
        if self.dim is None:
            return [f"{self.feature}{rule.describe()}"]
        return [f"{self.dim}_{value}" for value in self.values]

    def individuals(self, fallback: str) -> list[str]:
        """Who each class's events belong to.

        The individual the data itself names — the value along an individual
        dim, read or pinned — else *fallback*, the animal being labelled.
        """
        if self.dim in INDIVIDUAL_DIMS:
            return [str(v) for v in self.values]
        pinned = next((self.selections[d] for d in INDIVIDUAL_DIMS if d in self.selections), fallback)
        return [str(pinned)] * max(1, len(self.values))


def event_mappings(names: Sequence[str]) -> dict[int, dict]:
    """The in-memory vocabulary of an events table: class ``i + 1`` is ``names[i]``.

    Ids start at 1 — 0 is background everywhere labels are read — so a unit
    or keypoint numbered 0 keeps its number in the name, never as the id.
    The entries are shaped like :func:`~ethograph.labels.intervals.load_label_mapping`'s.
    """
    return {
        i + 1: {"name": str(name), "color": label_color(i + 1), "branch": 0, "event_type": EVENT_TYPE_STATE}
        for i, name in enumerate(names)
    }


def classes_along(mappings: Mapping[int, Mapping[str, Any]], dim: str) -> dict[str, int]:
    """Which class each value of *dim* is, in a vocabulary named ``{dim}_{value}``.

    The inverse of :meth:`FeatureQuery.class_names`; empty for a vocabulary
    that was not read along *dim*.
    """
    prefix = f"{dim}_"
    return {
        str(entry["name"]).removeprefix(prefix): label_id
        for label_id, entry in mappings.items()
        if str(entry.get("name", "")).startswith(prefix)
    }


def trial_events(
    trial: object,
    time: np.ndarray,
    data: np.ndarray,
    rule: ThresholdRule,
    label_ids: Sequence[int],
    individuals: Sequence[str],
    cuts: Sequence[float] | None = None,
) -> pd.DataFrame:
    """One trial's events: column ``i`` of *data* is class ``label_ids[i]``, acted by ``individuals[i]``.

    *data* is ``(T,)`` or ``(T, D)`` on the trial-relative clock *time*;
    ``cuts[i]`` is column ``i``'s threshold in its own units (the rule's own
    for every column when not given).
    """
    columns = np.asarray(data, dtype=np.float64).reshape(len(time), -1)
    if not columns.shape[1] == len(label_ids) == len(individuals):
        raise ValueError(f"{columns.shape[1]} series, {len(label_ids)} classes, {len(individuals)} individuals")
    parts = []
    for i, (label_id, individual) in enumerate(zip(label_ids, individuals)):
        intervals = threshold_intervals(time, columns[:, i], rule, None if cuts is None else cuts[i])
        if len(intervals):
            parts.append(_event_rows(trial, intervals, label_id, individual))
    if not parts:
        return empty_events()
    return pd.concat(parts, ignore_index=True)


#: One trial to read: ``(trial, loader, t0, t1, shift)`` — the trial's
#: :class:`~ethograph.io.catalog.DataLoader`, the window to read on the
#: loader's own clock, and what to subtract to make its times trial-relative.
TrialWindow = tuple[object, Any, float | None, float | None, float]


def _trial_series(
    windows: Iterable[TrialWindow], query: FeatureQuery
) -> Iterator[tuple[object, np.ndarray, np.ndarray, list[int]]]:
    """Per trial: its trial-relative time, its ``(T, n)`` series and the class of each.

    A trial the feature has no data for is left out; so is a value of the
    dim a trial does not have.
    """
    class_of = {value: i + 1 for i, value in enumerate(query.values)}
    for trial, loader, t0, t1, shift in windows:
        plot = loader.select(query.feature, query.selections, t0, t1)
        if plot is None:
            continue
        time = np.asarray(plot.time, dtype=np.float64) - shift
        columns = np.asarray(plot.data, dtype=np.float64).reshape(len(time), -1)
        if query.dim is None:
            if columns.shape[1] != 1:
                raise ValueError(
                    f"{query.feature!r} came back as {columns.shape[1]} series: "
                    "pin every dim, or read one of them value by value"
                )
            yield trial, time, columns, [1]
            continue
        labels = plot.dim_labels if plot.dim_labels is not None else list(query.values)
        if len(labels) != columns.shape[1]:
            raise ValueError(f"{query.feature!r}: {columns.shape[1]} series but {len(labels)} values of {query.dim!r}")
        present = [i for i, label in enumerate(labels) if str(label) in class_of]
        yield trial, time, columns[:, present], [class_of[str(labels[i])] for i in present]


@dataclass(frozen=True)
class SeriesSample:
    """What each class's series looks like across the trials read.

    *mean* and *std* are exact, one per class (NaN for a class with no
    data); *values* is a thinned-out ``(N, n_classes)`` sample of the raw
    samples, enough to draw their distribution, NaN where a trial lacks a
    class.
    """

    mean: np.ndarray
    std: np.ndarray
    values: np.ndarray

    def cuts(self, rule: ThresholdRule) -> np.ndarray:
        """*rule*'s threshold in each class's own units, one per class.

        A z-score becomes ``mean + threshold × std`` (NaN for a flat series:
        nothing is unusual in it), a percentile that percentile of the
        class's sampled values. A NaN cut lets nothing through.
        """
        n_classes = len(self.mean)
        if rule.scale == ZSCORE:
            return self.mean + rule.threshold * np.where(self.std > 0, self.std, np.nan)
        if rule.scale == PERCENTILE:
            cuts = np.full(n_classes, np.nan)
            for i in range(n_classes):
                column = self.values[:, i]
                column = column[np.isfinite(column)]
                if column.size:
                    cuts[i] = np.percentile(column, rule.threshold)
            return cuts
        return np.full(n_classes, float(rule.threshold))

    def zscored_values(self) -> np.ndarray:
        """The sample in standard deviations from each class's mean; a flat series is all NaN."""
        return (self.values - self.mean) / np.where(self.std > 0, self.std, np.nan)


def sample_series(windows: Iterable[TrialWindow], query: FeatureQuery, max_values: int = 2_000_000) -> SeriesSample:
    """Each class's mean and spread over every trial of *windows*, and a sample of its values.

    One pass. The statistics count every finite sample; the sample keeps
    every k-th row of each trial so that it holds about *max_values* numbers
    however long the session is.
    """
    windows = list(windows)
    n_classes = max(1, len(query.values))
    count = np.zeros(n_classes)
    total = np.zeros(n_classes)
    total_sq = np.zeros(n_classes)
    rows_per_trial = max(1, max_values // (max(1, len(windows)) * n_classes))
    kept: list[np.ndarray] = []
    for _trial, _time, columns, label_ids in _trial_series(windows, query):
        idx = np.asarray(label_ids, dtype=int) - 1
        finite = np.isfinite(columns)
        clean = np.where(finite, columns, 0.0)
        count[idx] += finite.sum(axis=0)
        total[idx] += clean.sum(axis=0)
        total_sq[idx] += (clean**2).sum(axis=0)
        step = max(1, int(np.ceil(len(columns) / rows_per_trial)))
        rows = np.full((len(columns[::step]), n_classes), np.nan, dtype=np.float32)
        rows[:, idx] = columns[::step]
        kept.append(rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean**2, 0.0))
    values = np.concatenate(kept) if kept else np.empty((0, n_classes), dtype=np.float32)
    return SeriesSample(mean, std, values)


def feature_events(
    windows: Iterable[TrialWindow],
    query: FeatureQuery,
    rule: ThresholdRule,
    individual: str,
) -> pd.DataFrame:
    """Every trial's events for *query*, as one table in trial then onset order.

    On a scale of each series' own the trials are read twice: once for the
    series' mean and spread (or a sample of its values, for a percentile)
    across all of them, once for the periods. Scaling trial by trial would
    find "active" periods in a trial where the series does nothing.
    """
    windows = list(windows)
    individuals = query.individuals(individual)
    cuts = None
    if rule.scale == ZSCORE:
        cuts = sample_series(windows, query, max_values=0).cuts(rule)
    elif rule.scale == PERCENTILE:
        cuts = sample_series(windows, query).cuts(rule)
    parts = []
    for trial, time, columns, label_ids in _trial_series(windows, query):
        events = trial_events(
            trial,
            time,
            columns,
            rule,
            label_ids,
            [individuals[label_id - 1] for label_id in label_ids],
            None if cuts is None else [cuts[label_id - 1] for label_id in label_ids],
        )
        if not events.empty:
            parts.append(events)
    if not parts:
        return empty_events()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["trial", "onset_s", "labels"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bursts: the same table, read off spike trains
# ---------------------------------------------------------------------------

MAX_INTERVAL = "max_interval"
LOG_ISI = "log_isi"

#: The burst detectors a rule can name: each takes a spike train and its own keyword parameters.
BURST_DETECTORS: dict[str, Callable[..., nap.IntervalSet]] = {
    MAX_INTERVAL: detect_bursts_max_interval,
    LOG_ISI: detect_bursts_log_isi,
}

#: One trial of a spike train: ``(trial, t0, t1, shift)`` — the trial's window on
#: the spike clock, and what to subtract to make a time in it trial-relative.
TrialSpan = tuple[object, float, float, float]


def burst_parameters(method: str) -> dict[str, float | int | bool]:
    """A detector's parameters and what each defaults to, read off the detector itself."""
    parameters = inspect.signature(BURST_DETECTORS[method]).parameters.values()
    return {p.name: p.default for p in parameters if p.default is not p.empty}


@dataclass(frozen=True)
class BurstRule:
    """Which burst detector runs, and the parameters given in place of its own defaults."""

    method: str
    params: Mapping[str, float | int | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in BURST_DETECTORS:
            raise ValueError(f"method must be one of {tuple(BURST_DETECTORS)}, got {self.method!r}")
        unknown = set(self.params) - set(burst_parameters(self.method))
        if unknown:
            raise ValueError(f"{self.method} has no parameter(s) {sorted(unknown)}")

    def describe(self) -> str:
        """``"bursts max_interval"`` — the rule as it reads in a set name."""
        return f"bursts {self.method}"

    def detect(self, spikes: nap.Ts) -> nap.IntervalSet:
        """*spikes*' bursts, within each epoch of its time support."""
        return BURST_DETECTORS[self.method](spikes, **self.params)


def burst_events(
    trains: Mapping[str, nap.Ts | nap.Tsd], spans: Iterable[TrialSpan], rule: BurstRule, individual: str
) -> pd.DataFrame:
    """Every trial's bursts, as one table in trial then onset order: class ``i + 1`` is the ``i``-th of *trains*.

    Each train is cut to the trials before it is read, so a burst never
    spans two trials, and a detector that adapts to the train (logISI's
    threshold) adapts to the spikes of the trials read — as a z-score reads
    its mean across them, not trial by trial.
    """
    spans = sorted(spans, key=lambda span: span[1])
    if not spans:
        return empty_events()
    with warnings.catch_warnings():
        # Trials that touch are still two epochs; pynapple says so every time.
        warnings.simplefilter("ignore", UserWarning)
        epochs = nap.IntervalSet(start=[t0 for _, t0, _, _ in spans], end=[t1 for _, _, t1, _ in spans])
    parts = []
    for label_id, train in enumerate(trains.values(), start=1):
        # A unit that carries values (a Tsd) is read for its spike times alone.
        spikes = nap.Ts(t=train.t, time_support=train.time_support).restrict(epochs)
        bursts = rule.detect(spikes)
        for trial, t0, t1, shift in spans:
            inside = (bursts.start >= t0) & (bursts.end <= t1)
            if inside.any():
                intervals = np.column_stack([bursts.start[inside], bursts.end[inside]]) - shift
                parts.append(_event_rows(trial, intervals, label_id, individual))
    if not parts:
        return empty_events()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["trial", "onset_s", "labels"], kind="stable").reset_index(drop=True)


def empty_events() -> pd.DataFrame:
    dtypes = {"trial": object, **INTERVAL_DTYPES}
    return pd.DataFrame({col: pd.Series(dtype=dtypes[col]) for col in EVENT_COLUMNS})


def _event_rows(trial: object, intervals: np.ndarray, label_id: int, individual: str) -> pd.DataFrame:
    n = len(intervals)
    df = pd.DataFrame(
        {
            "trial": pd.Series([trial] * n, dtype=object),
            "onset_s": intervals[:, 0],
            "offset_s": intervals[:, 1],
            "labels": np.full(n, label_id),
            "individual": str(individual),
            "individual_rec": NO_RECIPIENT,
            "event_type": EVENT_TYPE_STATE,
            # A threshold crossing or a detected burst is not a guess: there is no score to doubt.
            "confidence": HUMAN_CONFIDENCE,
            "labeling_method": LABELING_AUTOMATED,
        }
    )
    for col, dtype in INTERVAL_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df[EVENT_COLUMNS]
