"""One number per burst: what its spikes did, and what a signal did while it lasted.

A burst is compared with other bursts through scalars — a scatter of two of
them puts similar bursts side by side. Two families:

* :func:`spike_features` reads the burst's own spike times (count, rate,
  how sharply it starts, how regular it is, whether it speeds up).
* :func:`window_summary` reduces any time series to one value over the
  burst's span (its level, its peak, where it stood at onset, how it changed).

Qt-free; the scatter dialog (``gui/dialog_burst_scatter.py``) is the caller.
"""

from __future__ import annotations

import numpy as np
import pynapple as nap

#: Spike feature key -> axis title.
SPIKE_FEATURES = {
    "n_spikes": "Spike count",
    "mean_rate": "Mean rate (Hz)",
    "first_isi": "First ISI (s)",
    "isi_cv": "ISI irregularity (CV)",
    "rate_trend": "Rate trend (>0 accelerating)",
    "peak_position": "Peak-rate position (0 start, 1 end)",
}

#: Window summary key -> how it reads in an axis title.
SUMMARIES = {
    "mean": "mean",
    "peak": "peak",
    "onset": "at onset",
    "change": "change from before",
}


def spike_features(times: np.ndarray) -> dict[str, float]:
    """The :data:`SPIKE_FEATURES` of one burst's spike times (seconds, any origin).

    A feature the burst has too few spikes for is NaN: a rate needs two
    spikes, irregularity and a trend need three (two intervals).
    """
    t = np.sort(np.asarray(times, dtype=float))
    out = dict.fromkeys(SPIKE_FEATURES, float("nan"))
    out["n_spikes"] = float(len(t))
    if len(t) < 2:
        return out
    isi = np.diff(t)
    span = float(t[-1] - t[0])
    if span <= 0.0 or np.any(isi <= 0.0):
        return out
    out["mean_rate"] = (len(t) - 1) / span
    out["first_isi"] = float(isi[0])
    # Where each interval sits in the burst, 0 (start) to 1 (end).
    position = ((t[:-1] + t[1:]) / 2.0 - t[0]) / span
    out["peak_position"] = float(position[int(np.argmin(isi))])
    if len(isi) >= 2:
        out["isi_cv"] = float(np.std(isi) / np.mean(isi))
        # Slope of log instantaneous rate across the burst: scale-free, so a
        # fast unit and a slow one accelerating alike read alike.
        out["rate_trend"] = float(np.polyfit(position, -np.log(isi), 1)[0])
    return out


def window_summary(time: np.ndarray, values: np.ndarray, t0: float, t1: float, how: str) -> float:
    """*values* over ``t0``–``t1`` as one number; NaN where the series has nothing there.

    ``change`` is the mean during the window minus the mean over an equally
    long stretch just before it — "bursts when it starts moving" rather than
    "bursts while moving".
    """
    if how not in SUMMARIES:
        raise ValueError(f"Unknown summary {how!r} (expected one of {', '.join(SUMMARIES)}).")
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    def mean_between(a: float, b: float, right_closed: bool) -> float:
        mask = (time >= a) & ((time <= b) if right_closed else (time < b)) & np.isfinite(values)
        return float(values[mask].mean()) if mask.any() else float("nan")

    if how == "mean":
        return mean_between(t0, t1, True)
    if how == "change":
        return mean_between(t0, t1, True) - mean_between(t0 - (t1 - t0), t0, False)
    inside = (time >= t0) & (time <= t1) & np.isfinite(values)
    if how == "peak":
        return float(values[inside].max()) if inside.any() else float("nan")
    finite = np.isfinite(values)
    if not finite.any() or t0 < time[finite][0] or t0 > time[finite][-1]:
        return float("nan")
    return float(np.interp(t0, time[finite], values[finite]))


def _scaled(values: np.ndarray) -> np.ndarray:
    """*values* over their own range, so two axes in different units weigh alike."""
    finite = values[np.isfinite(values)]
    extent = float(finite.max() - finite.min()) if len(finite) else 0.0
    return values / extent if extent > 0.0 else values


def nearest_points(x: np.ndarray, y: np.ndarray, x0: float, y0: float, n: int) -> np.ndarray:
    """Indices of the *n* points nearest ``(x0, y0)``, nearest first.

    Distances are measured with each axis scaled to its own range — what
    "near" looks like on the plot. Points with a NaN coordinate are never near.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if len(valid) == 0:
        return valid
    finite_x, finite_y = x[valid], y[valid]
    extent_x = float(finite_x.max() - finite_x.min()) or 1.0
    extent_y = float(finite_y.max() - finite_y.min()) or 1.0
    distance = np.hypot((finite_x - x0) / extent_x, (finite_y - y0) / extent_y)
    return valid[np.argsort(distance, kind="stable")[: max(1, int(n))]]


def points_in_rect(x: np.ndarray, y: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Boolean mask of the points inside the rectangle (NaN coordinates are outside)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(invalid="ignore"):
        return (x >= min(x0, x1)) & (x <= max(x0, x1)) & (y >= min(y0, y1)) & (y <= max(y0, y1))


def time_from_nearest(t: float, references: np.ndarray) -> float:
    """*t* minus the reference nearest it: negative before it, positive after; NaN with none."""
    references = np.asarray(references, dtype=float)
    references = references[np.isfinite(references)]
    if len(references) == 0:
        return float("nan")
    return float(t - references[int(np.argmin(np.abs(references - t)))])


def shift_onsets(onsets: np.ndarray, trial_duration_s: float) -> np.ndarray:
    """*onsets* of one trial, all moved by one random offset and wrapped at the trial's end.

    ``pynapple.shift_timestamps`` does the shift (a circular one, so the
    events keep their spacing and only their alignment with everything else
    is broken); it hands the times back sorted, so the offset is read back
    off them to return each event's new time in the order given.
    """
    onsets = np.clip(np.asarray(onsets, dtype=float), 0.0, trial_duration_s)
    if len(onsets) == 0 or trial_duration_s <= 0.0:
        return onsets
    order = np.argsort(onsets, kind="stable")
    ordered = onsets[order]
    train = nap.Ts(t=ordered, time_support=nap.IntervalSet(0.0, trial_duration_s))
    shifted = np.asarray(nap.shift_timestamps(train, mode="wrap").t, dtype=float)
    if len(shifted) != len(ordered):
        raise RuntimeError("pynapple dropped events while shifting them.")
    # The shifted times are the originals plus one offset, rotated by however many wrapped.
    for wrapped in range(len(ordered)):
        offset = (shifted[0] - ordered[(len(ordered) - wrapped) % len(ordered)]) % trial_duration_s
        moved = (ordered + offset) % trial_duration_s
        if np.allclose(np.sort(moved), shifted):
            out = np.empty_like(moved)
            out[order] = moved
            return out
    raise RuntimeError("Could not read the offset back off the shifted events.")


def default_percentiles(values: np.ndarray) -> tuple[float, float]:
    """The percentiles a colour scale opens between: 0–95, or 5–95 for values that go negative.

    A quantity that cannot be negative (a speed) has a real floor worth
    showing, and only its top needs trimming; one that can (a velocity) has
    extremes at both ends.
    """
    values = np.asarray(values, dtype=float)
    return (5.0, 95.0) if np.any(values[np.isfinite(values)] < 0.0) else (0.0, 95.0)


def scale_between_percentiles(
    values: np.ndarray, low: float = 0.0, high: float = 95.0
) -> tuple[np.ndarray, float, float]:
    """*values* spread linearly over 0–1 between two of their percentiles, with the bounds used.

    What lies past a bound is clipped onto it, so a few extreme values do not
    squeeze the rest into one end of a colour scale. NaN stays NaN; values
    with no spread all sit at 0.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.full(len(values), np.nan), float("nan"), float("nan")
    lo, hi = (float(v) for v in np.percentile(finite, [low, high]))
    if hi <= lo:
        return np.where(np.isfinite(values), 0.0, np.nan), lo, hi
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0), lo, hi
