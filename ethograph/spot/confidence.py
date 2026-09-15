"""How confident a spotted event is: the curve's shape, not its height.

The statistics themselves live in :mod:`ethograph.labels.curve_confidence`,
shared with the lightgbm model; this module fixes which one the pixel spotter
writes and adds the one helper only it needs (:func:`densify`).

E2E-Spot's curve is a per-frame softmax over ``K + 1`` classes. It normalises
across *classes at each frame* and nothing normalises across time, so a class
can sit moderately high for a long stretch and its peak still reads as
confident. Measured on a held-out session, AUC for detecting an event
misplaced by more than 50 ms:

============================  =====
statistic                      AUC
============================  =====
peak height                    0.58
ratio                          0.79
focus                          0.81
``shape`` (focus x ratio)      0.82
============================  =====

The three shape statistics tie within the sample size. The number written
is :data:`STATISTIC` — ``shape``, the plain product ``focus x ratio``, with
no weight baked in: how much a smeared bump should count against an event
versus a second candidate is a review preference, and the GUI is where it is
set, with the histogram in front of the user. Both halves stay readable off
the curve the review draws: *is it one bump or many*, and *is there a rival*.
"""

from __future__ import annotations

import numpy as np

from ethograph.labels.curve_confidence import (
    CurveStats,
    curve_events,
    curve_stats,
    focus_window_s,
    tallest_peak,
    window_samples,
)
from ethograph.labels.rescore import DEFAULT_RULE, rule_value

__all__ = [
    "DEFAULT_RULE",
    "CurveStats",
    "confidence_of",
    "curve_events",
    "curve_stats",
    "densify",
    "focus_window_s",
    "tallest_peak",
    "window_samples",
]


def confidence_of(stats: CurveStats, rule: str = DEFAULT_RULE, alpha: float = 0.5) -> float:
    """The number written beside a spotted event, under *rule* (``infer.confidence``)."""
    return rule_value(stats, rule, alpha)


def densify(frames: np.ndarray, scores: np.ndarray, length: int) -> np.ndarray:
    """A sparse candidate list as a dense curve of *length* samples.

    E2E-Spot writes only the frames scoring above a low threshold, which is
    the curve with its zeros left out. Everything downstream — the statistics
    and the curves the GUI draws — wants it back.
    """
    curve = np.zeros(int(length), dtype=np.float32)
    frames = np.asarray(frames, dtype=int)
    inside = (frames >= 0) & (frames < length)
    curve[frames[inside]] = np.asarray(scores, dtype=np.float32)[inside]
    return curve
