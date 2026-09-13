"""How much to trust an event read off a per-frame curve — the statistics, shared.

Every point-event model here ends the same way: a per-frame curve per class,
the tallest peak is the event, and one number beside it says how much to
trust it. Which number is **an empirical question per model**, and the
answer must stay readable off the curve the review draws. Measured on this
repo's models (AUC for "the event is misplaced"):

* the LightGBM lightgbm model's curve is shape-constrained by construction — a
  Gaussian-weighted target smoothed with the matching kernel — so its bumps
  all look alike and the **peak height** carries the information (0.67–0.76);
  the shape terms tie or hurt;
* E2E-Spot's per-frame softmax normalises across classes, never across time,
  so a class can sit moderately high for a stretch and its peak still reads
  as confident: height is near chance (0.58) and the **shape** — one bump or
  many, a rival or not — is the signal (0.82).

So this module offers the candidates and a way to pick among them on a
held-out record (:func:`rank_statistics`); each model chooses. Nothing here
rewards *more* evidence: total mass, width and candidate count all came out
inverted (0.32–0.41).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import rankdata

#: The neighbourhood of the peak that counts as "the same event" for
#: :attr:`CurveStats.focus` and :attr:`CurveStats.ratio`, as a multiple of
#: the tolerance the user declared for their labels: a bump wider than twice
#: the label precision is smeared by the user's own definition, a peak
#: further away than that is a rival. At a 50 ms tolerance this is the
#: 100 ms that measured flat-optimal (50–200 ms); the user's timescale sets
#: it, not a constant. Every model resolves its own — the lightgbm model from
#: ``tolerance_s``, the pixel spotter from ``infer.focus_window_ms``.
FOCUS_WINDOW_TOLERANCES = 2.0


def focus_window_s(tolerance_s: float) -> float:
    """The focus half-width, in seconds, for labels believed to *tolerance_s*."""
    if tolerance_s <= 0:
        raise ValueError(f"tolerance_s must be positive, got {tolerance_s!r}")
    return FOCUS_WINDOW_TOLERANCES * float(tolerance_s)


#: Below this peak height a curve is "nearly nothing everywhere": its
#: ``focus`` and ``ratio`` read 0, so every confidence built on them is 0
#: — flagged for review, never dropped. Without it a single surviving
#: 3-frame blip at 0.02 would be the cleanest bump imaginable and read 1.0.
MIN_PEAK = 0.05

#: Every statistic a model may write as its confidence, by name.
STATISTICS = ("peak", "focus", "ratio", "shape", "shape_peak")

#: What each name means, for a report.
DESCRIPTIONS = {
    "peak": "height of the tallest peak",
    "focus": "share of the curve's mass within the focus window of the peak",
    "ratio": "1 - tallest rival peak / peak",
    "shape": "focus x ratio (the product, the plain confidence)",
    "shape_peak": "focus x ratio x peak",
}


def tallest_peak(curve: np.ndarray) -> tuple[int, float]:
    """The curve's tallest local maximum: ``(frame index, height)``.

    ``find_peaks`` rather than ``argmax`` so a curve still climbing at the
    trial's edge does not report its last frame as a confident event — an edge
    is not a peak. With no local maximum anywhere (a flat or monotone curve)
    the argmax stands in, which for a flat curve is a height near 0 and says
    exactly what it should.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if curve.size == 0:
        return 0, 0.0
    peaks = find_peaks(curve)[0]
    index = int(peaks[np.argmax(curve[peaks])]) if peaks.size else int(np.argmax(curve))
    return index, float(np.clip(curve[index], 0.0, 1.0))


@dataclass(frozen=True)
class CurveStats:
    """One class's curve in one trial, summarised where the peak is."""

    #: Index of the tallest peak, on the curve's own clock.
    index: int
    #: Height there — the model's own per-frame score.
    peak: float
    #: Share of the class's total mass lying within the focus window of the
    #: peak. 1.0 = one clean bump; low = smeared, or evidence elsewhere.
    focus: float
    #: ``1 - (tallest rival / peak)``, the rival being the tallest *other
    #: local maximum* outside the focus window — another candidate moment,
    #: never the peak's own shoulder. 1.0 = no rival; 0.0 = a tie.
    ratio: float
    #: False when the curve did not find an event at all: nearly nothing
    #: anywhere, no interior peak, or an edge rising above the peak (the
    #: event may lie past the trial's end). Every statistic then reads 0 —
    #: whatever a model or a reviewer's rule would otherwise pick.
    found: bool = True

    @property
    def shape(self) -> float:
        return float(np.clip(self.focus * self.ratio, 0.0, 1.0))

    @property
    def shape_peak(self) -> float:
        return float(np.clip(self.focus * self.ratio * self.peak, 0.0, 1.0))

    def statistic(self, name: str) -> float:
        """The named statistic (:data:`STATISTICS`); 0 when the curve found no event."""
        if name not in STATISTICS:
            raise ValueError(f"unknown confidence statistic {name!r}; one of {STATISTICS}")
        return float(getattr(self, name)) if self.found else 0.0


def window_samples(window_s: float, fs: float) -> int:
    """The focus half-width in samples of a curve sampled at *fs*."""
    if fs <= 0:
        raise ValueError(f"Frame rate must be positive, got {fs!r}")
    return max(1, int(round(window_s * fs)))


def curve_stats(curve: np.ndarray, window: int) -> CurveStats:
    """Summarise *curve* around its tallest peak.

    *window* is the focus half-width **in samples of this curve's own clock**;
    the caller resolves it from a duration (:func:`window_samples`), because a
    count of frames means different things at different rates.

    Two curves read as "not found" whatever their shape — ``focus`` and
    ``ratio`` both 0: one whose tallest peak is below :data:`MIN_PEAK`, and
    one with no interior local maximum (still climbing at the trial's edge).
    A curve that has an interior peak *and* rises higher at an edge keeps the
    peak but reads the edge as a rival, so it is flagged rather than trusted.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if curve.size == 0:
        return CurveStats(index=0, peak=0.0, focus=0.0, ratio=0.0, found=False)
    index, peak = tallest_peak(curve)
    if peak < MIN_PEAK or not find_peaks(curve)[0].size:
        # Nearly nothing anywhere, or no interior maximum at all (a curve
        # still climbing at the trial's edge): the event was not found, and
        # no statistic may say otherwise.
        return CurveStats(index=index, peak=peak, focus=0.0, ratio=0.0, found=False)
    positions = np.arange(curve.size)
    near = np.abs(positions - index) <= window
    mass = float(np.clip(curve, 0.0, None).sum())
    focus = float(np.clip(curve[near], 0.0, None).sum()) / mass if mass > 0 else 0.0
    # A rival is another *peak*, not the tallest sample outside the window:
    # a broad bump's own flank would otherwise read as a second candidate,
    # and width is ``focus``'s job. The trial's two ends count as candidates
    # too — a curve still climbing at the edge is the model saying "maybe
    # after the end", which a smaller interior bump must not outrank.
    others = [p for p in find_peaks(curve)[0] if not near[p]]
    edges = [edge for edge in (0, curve.size - 1) if not near[edge]]
    if edges and float(curve[edges].max()) > peak:
        # The curve is higher at the trial's end than at any peak inside it:
        # the event may lie past the end — not found, confidence 0.
        return CurveStats(index=index, peak=peak, focus=0.0, ratio=0.0, found=False)
    others += edges
    rival = float(curve[others].max()) if others else 0.0
    return CurveStats(
        index=index,
        peak=peak,
        focus=float(np.clip(focus, 0.0, 1.0)),
        ratio=float(np.clip(1.0 - rival / peak, 0.0, 1.0)),
    )


def rank_auc(scores: np.ndarray, hits: np.ndarray) -> float:
    """AUC of *scores* for separating hits from misses (Mann–Whitney), ``nan`` if only one class."""
    scores = np.asarray(scores, dtype=np.float64)
    hits = np.asarray(hits, dtype=bool)
    if hits.all() or not hits.any():
        return float("nan")
    ranks = rankdata(scores)  # average ranks, so ties do not pretend to separate
    n1, n0 = hits.sum(), (~hits).sum()
    return float((ranks[hits].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rank_statistics(stats: list[CurveStats], hits: np.ndarray) -> dict[str, float]:
    """AUC per statistic on a held-out record: which number tells a hit from a miss."""
    hits = np.asarray(hits, dtype=bool)
    return {name: rank_auc(np.array([s.statistic(name) for s in stats]), hits) for name in STATISTICS}


def choose_statistic(aucs: dict[str, float], default: str = "peak") -> str:
    """The best-separating statistic, the *default* unless something clearly beats it.

    A record of a few dozen trials cannot tell 0.71 from 0.69, and the
    simplest readable number wins ties; only a margin worth a whole trial's
    worth of ranking moves the choice.
    """
    base = aucs.get(default)
    if base is None or not np.isfinite(base):
        finite = {k: v for k, v in aucs.items() if np.isfinite(v)}
        return max(finite, key=finite.get) if finite else default
    best = max((k for k in aucs if np.isfinite(aucs[k])), key=lambda k: aucs[k])
    return best if aucs[best] > base + MIN_AUC_GAIN else default


#: How much better than the default a statistic must separate hits from misses
#: to be chosen — the AUC change from moving one held-out trial in a record of
#: about twenty.
MIN_AUC_GAIN = 0.05
