"""Trial-concatenated PCA of the behavioural features: one low-dimensional "behavioural state" per moment.

Every feature is averaged over a sliding window on each trial's own clock;
the windows of all trials are stacked into one table (windows x columns),
z-scored per column, its gaps filled with zero — the column's mean — and
decomposed. A moment of any trial then has coordinates on the same
components, so events from different trials can be compared by where in
behaviour space they fall.

Qt-free; the burst scatter (``gui/dialog_burst_scatter.py``) is the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def window_starts(duration_s: float, window_s: float, step_s: float) -> np.ndarray:
    """Start times of the sliding windows over ``0``–*duration_s*: every *step_s*, each *window_s* long.

    At least one window, so a trial shorter than the window still has a value.
    """
    if window_s <= 0.0 or step_s <= 0.0:
        raise ValueError("The window and its step must both be longer than zero.")
    # The tolerance keeps a window that ends exactly at the trial's end (0.6 / 0.2 is 2.999… in floats).
    count = max(1, int(np.floor((duration_s - window_s) / step_s + 1e-9)) + 1)
    return np.arange(count, dtype=float) * step_s


def window_means(time: np.ndarray, values: np.ndarray, starts: np.ndarray, window_s: float) -> np.ndarray:
    """*values* ``(T, D)`` averaged over each window ``[start, start + window_s)``, as ``(W, D)``.

    NaN samples are skipped; a window with no finite sample of a column is NaN there.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float).reshape(len(time), -1)
    finite = np.isfinite(values)
    zero = np.zeros((1, values.shape[1]))
    sums = np.vstack([zero, np.cumsum(np.where(finite, values, 0.0), axis=0)])
    counts = np.vstack([zero, np.cumsum(finite, axis=0)])
    lo = np.searchsorted(time, starts, side="left")
    hi = np.searchsorted(time, np.asarray(starts) + window_s, side="left")
    total, n = sums[hi] - sums[lo], counts[hi] - counts[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / n, np.nan)


@dataclass
class BehaviourPCA:
    """A fitted trial-concatenated PCA.

    ``scores[trial]`` is ``(W, k)`` — the trial's windows on the components —
    and ``centers[trial]`` the trial-relative time each of those windows is
    centred on.
    """

    columns: list[str]
    components: np.ndarray  # (k, D)
    explained: np.ndarray  # (k,) fraction of the variance
    centers: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]

    def top_loadings(self, component: int, n: int = 3) -> list[tuple[str, float]]:
        """The *n* columns weighing most on *component*, heaviest first."""
        weights = self.components[component]
        order = np.argsort(-np.abs(weights))[:n]
        return [(self.columns[i], float(weights[i])) for i in order]

    def at(self, trial: object, t0: float, t1: float, component: int) -> float:
        """*component*'s score over ``t0``–``t1`` of *trial*: the mean of the
        windows centred inside, else the window nearest the span's middle."""
        centers = self.centers.get(str(trial))
        if centers is None or len(centers) == 0:
            return float("nan")
        scores = self.scores[str(trial)][:, component]
        inside = (centers >= t0) & (centers <= t1)
        if inside.any():
            return float(scores[inside].mean())
        return float(scores[int(np.argmin(np.abs(centers - (t0 + t1) / 2.0)))])


def fit_behaviour_pca(
    blocks: dict[str, np.ndarray],
    centers: dict[str, np.ndarray],
    columns: list[str],
    n_components: int,
) -> BehaviourPCA:
    """PCA over the windows of every trial at once.

    ``blocks[trial]`` is ``(W, D)`` window means. Columns are z-scored on
    their finite values across all trials; a NaN then becomes 0, the column's
    mean, so a feature missing somewhere neither drops the window nor pulls
    it. A column with no spread carries nothing and stays at 0.
    """
    if not blocks:
        raise ValueError("No trial has any of these features.")
    trials = list(blocks)
    table = np.vstack([blocks[t] for t in trials])
    if table.shape[1] != len(columns):
        raise ValueError(f"{table.shape[1]} feature columns but {len(columns)} names.")
    finite = np.isfinite(table)
    n = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.where(finite, table, 0.0).sum(axis=0) / n, 0.0)
        spread = np.sqrt(np.where(n > 0, (np.where(finite, table - mean, 0.0) ** 2).sum(axis=0) / n, 0.0))
        z = np.where(finite & (spread > 0), (table - mean) / spread, 0.0)
    k = max(1, min(int(n_components), z.shape[1]))
    values, vectors = np.linalg.eigh(z.T @ z / max(1, len(z) - 1))
    order = np.argsort(values)[::-1][:k]
    components = vectors[:, order].T
    # An eigenvector's sign is arbitrary: fix it so a rerun draws the same picture.
    heaviest = components[np.arange(k), np.argmax(np.abs(components), axis=1)]
    components = components * np.where(heaviest < 0, -1.0, 1.0)[:, None]
    total = float(values.sum())
    explained = values[order] / total if total > 0 else np.zeros(k)
    scores = z @ components.T
    bounds = np.cumsum([0, *(len(blocks[t]) for t in trials)])
    return BehaviourPCA(
        columns=list(columns),
        components=components,
        explained=explained,
        centers={t: np.asarray(centers[t], dtype=float) for t in trials},
        scores={t: scores[bounds[i] : bounds[i + 1]] for i, t in enumerate(trials)},
    )
