"""Row order for a ``(T, C)`` matrix: earliest peak window on top, or Rastermap's.

Qt-free. Every sort here is one contract — the matrix in, a permutation of its
rows out — so the heatmap and the spike raster order their rows the same way.

Peak window: tile the time axis with windows of ``window_s`` seconds
overlapping by ``overlap`` (0.5 = each window starts halfway into the previous
one), take each row's mean per window, and rank rows by the window holding
their largest mean — an argmax over windows rather than over samples, so one
outlier sample does not decide a row's place.

Rastermap: rows with similar activity become neighbours. It is a fit, not a
formula, so it runs on demand and its order is kept.
"""

from __future__ import annotations

import warnings

import numpy as np
from rastermap import Rastermap

SORT_MODES: dict[str, str] = {
    "none": "Original order",
    "trial": "Sort by trial window",
    "visible": "Sort by visible window",
    "rastermap": "Rastermap (fitted on demand)",
}

#: Below this many rows an order by similarity says nothing a glance does not.
RASTERMAP_MIN_ROWS = 10
#: Rastermap sorts the rows themselves up to this many; above it, it clusters them first.
_RASTERMAP_MAX_UNCLUSTERED_ROWS = 200
#: Samples the fit is given at most; a longer matrix is averaged in time first.
RASTERMAP_MAX_SAMPLES = 5_000


def rastermap_order(data: np.ndarray) -> np.ndarray:
    """Row indices of ``data`` ``(T, C)`` in Rastermap's order.

    Rows with similar activity over time end up next to each other. A row
    with no variance cannot be placed and goes last; missing samples count as
    the row's mean.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"data must be (T, C), got shape {data.shape}")
    n_samples, n_rows = data.shape
    if n_rows < RASTERMAP_MIN_ROWS:
        raise ValueError(f"Rastermap needs at least {RASTERMAP_MIN_ROWS} rows, got {n_rows}")

    rows = data.T
    finite = np.isfinite(rows)
    counts = finite.sum(axis=1)
    sums = np.where(finite, rows, 0.0).sum(axis=1)
    means = np.divide(sums, counts, out=np.zeros(n_rows, dtype=np.float32), where=counts > 0)
    rows = np.where(finite, rows, means[:, None])
    n_varying = int(np.count_nonzero(rows.std(axis=1) > 0))
    if n_varying < RASTERMAP_MIN_ROWS:
        raise ValueError(f"Rastermap needs at least {RASTERMAP_MIN_ROWS} rows that vary over time, got {n_varying}")

    model = Rastermap(
        # Its own default above the limit; below it there are too few rows to cluster.
        **({"n_clusters": None} if n_rows <= _RASTERMAP_MAX_UNCLUSTERED_ROWS else {}),
        time_bin=int(np.ceil(n_samples / RASTERMAP_MAX_SAMPLES)),
        verbose=False,
    )
    with warnings.catch_warnings():
        # Rastermap warns for a row without variance and for a matrix small
        # enough to sort unclustered — both are handled above, neither is news.
        warnings.simplefilter("ignore")
        model.fit(rows)
    return np.asarray(model.isort, dtype=np.int64)


def window_starts(t_start: float, t_end: float, window_s: float, overlap: float) -> np.ndarray:
    """Start times of the windows tiling ``[t_start, t_end]``.

    The last window may run past ``t_end``; there is always at least one.
    """
    if window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    step = window_s * (1.0 - overlap)
    span = max(t_end - t_start - window_s, 0.0)
    n = int(np.floor(span / step + 1e-9)) + 1
    return t_start + step * np.arange(n)


def argmax_window_order(data: np.ndarray, time: np.ndarray, window_s: float, overlap: float = 0.5) -> np.ndarray:
    """Row indices of ``data`` ``(T, C)`` ordered by where each row peaks.

    Rows whose peak window comes earlier sort first; ties keep their original
    order; rows with no finite sample go last.
    """
    data = np.asarray(data, dtype=float)
    time = np.asarray(time, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be (T, C), got shape {data.shape}")
    if time.shape != (data.shape[0],):
        raise ValueError(f"time has shape {time.shape}, data has {data.shape[0]} samples")
    n_rows = data.shape[1]
    if data.shape[0] == 0 or n_rows == 0:
        return np.arange(n_rows)

    starts = window_starts(float(time[0]), float(time[-1]), window_s, overlap)
    finite = np.isfinite(data)
    filled = np.where(finite, data, 0.0)
    means = np.full((len(starts), n_rows), np.nan)
    for i, s in enumerate(starts):
        mask = (time >= s) & (time <= s + window_s)
        counts = finite[mask].sum(axis=0)
        sums = filled[mask].sum(axis=0)
        means[i] = np.divide(sums, counts, out=np.full(n_rows, np.nan), where=counts > 0)

    has_value = np.isfinite(means).any(axis=0)
    peak_window = np.zeros(n_rows, dtype=float)
    peak_window[has_value] = np.nanargmax(means[:, has_value], axis=0)
    peak_window[~has_value] = np.inf
    return np.argsort(peak_window, kind="stable")


def row_window(n_rows: int, percent: float, position: float) -> slice:
    """The contiguous block of display rows kept by the row-window control.

    ``percent`` of ``n_rows`` (at least one row) are kept; ``position`` in
    ``[0, 1]`` slides the block from the top row to the bottom one. The block
    is taken from the displayed (possibly sorted) order, so neighbouring rows
    are neighbours in that order, not in the original index.
    """
    if not 0.0 < percent <= 100.0:
        raise ValueError(f"percent must be in (0, 100], got {percent}")
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"position must be in [0, 1], got {position}")
    n_shown = min(n_rows, max(1, int(np.ceil(n_rows * percent / 100.0))))
    start = int(round((n_rows - n_shown) * position))
    return slice(start, start + n_shown)
