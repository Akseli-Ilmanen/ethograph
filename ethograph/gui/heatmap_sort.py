"""Row order for the heatmap: earliest peak window on top.

Qt-free. Tile the time axis with windows of ``window_s`` seconds overlapping
by ``overlap`` (0.5 = each window starts halfway into the previous one), take
each row's mean per window, and rank rows by the window holding their
largest mean — an argmax over windows rather than over samples, so one
outlier sample does not decide a row's place.
"""

from __future__ import annotations

import numpy as np

SORT_MODES: dict[str, str] = {
    "none": "Original order",
    "trial": "Sort by trial window",
    "visible": "Sort by visible window",
}


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
