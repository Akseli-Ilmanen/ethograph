"""Where to seed and how far to track: pure numpy over one confidence trace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ethograph.samfill.config import PixelExclusion


@dataclass(frozen=True)
class FillRun:
    """One SAM propagation: seed at ``seed``, track ``frames``, overwrite ``fill``.

    ``frames`` starts right after the seed and ends where confidence has
    recovered for ``margin`` frames (exclusive). ``fill`` are the frames inside
    it whose confidence was low; the rest are tracked for continuity but keep
    the tracker's own estimate.
    """

    seed: int
    frames: range
    fill: tuple[int, ...]
    #: Further confident frames to seed from, in order of preference, should the seed's mask be rejected.
    alternates: tuple[int, ...] = ()

    @property
    def n_tracked(self) -> int:
        return len(self.frames)


def _find_seed(confident: np.ndarray, gap_start: int, margin: int) -> int | None:
    """The nearest confident frame at least ``margin`` frames before the gap, or ``None``."""
    for frame in range(gap_start - margin, -1, -1):
        if confident[frame]:
            return frame
    return None


def _find_end(confident: np.ndarray, gap_start: int, window_end: int, margin: int) -> int:
    """First frame from which ``margin`` consecutive confident frames follow, else ``window_end``."""
    run = 0
    for frame in range(gap_start, window_end):
        run = run + 1 if confident[frame] else 0
        if run >= margin:
            return frame - margin + 1
    return window_end


def _spaced(candidates: np.ndarray, seed: int, spacing: int, n: int) -> tuple[int, ...]:
    """Up to *n* of *candidates* (in order), each at least *spacing* frames from the seed and from each other."""
    picked: list[int] = []
    last = seed
    for frame in candidates:
        if abs(int(frame) - last) >= spacing:
            picked.append(int(frame))
            last = int(frame)
        if len(picked) >= n:
            break
    return tuple(picked)


def plan_runs(
    confidence: np.ndarray,
    window: range,
    low_confidence: float,
    seed_confidence: float,
    margin: int,
    max_tries: int = 10,
) -> list[FillRun]:
    """Plan the SAM runs for one keypoint of one individual inside *window*.

    Walks the window; every stretch opening with a low frame becomes a run,
    seeded at the nearest confident frame at least ``margin`` frames before it,
    tracked until confidence stays at ``seed_confidence`` for ``margin`` frames
    or the window ends. A gap with no confident frame before it is left alone.
    The seed may precede the window; nothing before the window is ever
    overwritten. NaN confidence counts as low. Each run also names up to
    ``max_tries - 1`` earlier confident frames as alternates, at least
    ``margin`` frames apart so that a rejected seed's near-identical
    neighbours are not tried again.
    """
    confidence = np.asarray(confidence, dtype=float)
    if confidence.ndim != 1:
        raise ValueError(f"confidence must be 1-D, got shape {confidence.shape}")
    if margin < 1:
        raise ValueError(f"margin must be >= 1, got {margin}")
    n = len(confidence)
    start, stop = max(0, window.start), min(n, window.stop)
    low = np.isnan(confidence) | (confidence < low_confidence)
    confident = ~np.isnan(confidence) & (confidence >= seed_confidence)
    runs: list[FillRun] = []
    i = start
    while i < stop:
        if not low[i]:
            i += 1
            continue
        end = _find_end(confident, i, stop, margin)
        seed = _find_seed(confident, i, margin)
        if seed is not None:
            fill = tuple(int(f) for f in range(i, end) if low[f])
            alternates = _spaced(np.flatnonzero(confident[:seed])[::-1], seed, margin, max_tries - 1)
            runs.append(FillRun(seed=seed, frames=range(seed + 1, end), fill=fill, alternates=alternates))
        i = max(end, i + 1)
    return runs


def plan_anchor_run(
    confidence: np.ndarray,
    window: range,
    low_confidence: float,
    seed_confidence: float,
    max_tries: int = 10,
    spacing: int = 1,
) -> list[FillRun]:
    """One run for the whole *window*, seeded at its first confident frame.

    For a keypoint that starts every trial in a known place: SAM is shown it
    once, while the tracker is sure, and keeps that frame as its anchor for
    the rest of the segment. Low frames anywhere after the seed are filled.
    Empty when no frame in the window reaches ``seed_confidence``. The next
    confident frames, up to ``max_tries - 1`` of them and at least ``spacing``
    frames apart, are the alternates.
    """
    confidence = np.asarray(confidence, dtype=float)
    if confidence.ndim != 1:
        raise ValueError(f"confidence must be 1-D, got shape {confidence.shape}")
    start, stop = max(0, window.start), min(len(confidence), window.stop)
    low = np.isnan(confidence) | (confidence < low_confidence)
    confident = np.flatnonzero(~np.isnan(confidence[start:stop]) & (confidence[start:stop] >= seed_confidence))
    if len(confident) == 0:
        return []
    seed = start + int(confident[0])
    fill = tuple(int(f) for f in range(seed + 1, stop) if low[f])
    if not fill:
        return []
    alternates = _spaced(start + confident[1:], seed, spacing, max_tries - 1)
    return [FillRun(seed=seed, frames=range(seed + 1, stop), fill=fill, alternates=alternates)]


def excluded_frames(x: np.ndarray, y: np.ndarray, exclusion: PixelExclusion) -> np.ndarray:
    """Boolean per frame: the keypoint is somewhere it cannot be, or within ``pad_frames`` of such a frame."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    inside = np.zeros(len(x), dtype=bool)
    for x0, y0, x1, y1 in exclusion.boxes:
        inside |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if exclusion.y_max is not None:
        inside |= y > exclusion.y_max
    padded = inside.copy()
    for shift in range(1, exclusion.pad_frames + 1):
        padded[shift:] |= inside[:-shift]
        padded[:-shift] |= inside[shift:]
    return padded
