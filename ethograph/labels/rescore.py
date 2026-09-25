"""Re-reading a run's curves under a rule the reviewer chose — the confidence knob, Qt-free.

A point-event model writes one ``confidence`` per label from its curve
(:mod:`ethograph.labels.confidence`), under a rule the model fixed. How
much a smeared bump should count against a second candidate is a review
preference, not a model constant, and the reviewer has the histogram in
front of them — so the rule is set in the grids' ``Histogram…`` popup, where
every change redraws the bars and restyles the tiles, and **Apply** confirms
it into the labels. This module is the arithmetic of that knob.

Only an **automated** label that **has a curve** is ever re-scored: a
manual or curated label is a human's word (``1.0``), and a label imported
from a run that kept no curves keeps the confidence it came with. Curves stay
an aid to review, never something a session depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ethograph.labels.confidence import CurveStats, curve_stats, window_samples
from ethograph.labels.intervals import LABELING_AUTOMATED
from ethograph.labels.onset_curves import TrialCurves

#: The rules a reviewer can pick, by key, with how each reads on screen.
#: ``custom`` is one slider between two of them: ``ratio × (α + (1 − α)·focus)``
#: is ``ratio`` at α = 1 and ``focus × ratio`` at α = 0.
RULES: dict[str, str] = {
    "product": "focus × ratio",
    "ratio": "ratio  (one candidate or two)",
    "focus": "focus  (one bump or a smear)",
    "peak": "peak height",
    "custom": "ratio × (α + (1 − α) · focus)",
}


DEFAULT_RULE = "product"


def yaml_snippet(rule: str, alpha: float, window_ms: float) -> str:
    """The ``infer:`` lines that make the pixel pipeline write this rule — what the popup copies.

    Pasted into a spot ``project.yaml``, the next ``inference()`` reads the
    confidence the way the review settled on; ``confidence_alpha`` is
    written only for the custom rule, where it means something.
    """
    if rule not in RULES:
        raise ValueError(f"unknown confidence rule {rule!r}; one of {list(RULES)}")
    lines = ["infer:", f"  confidence: {rule}"]
    if rule == "custom":
        lines.append(f"  confidence_alpha: {float(alpha):g}")
    lines.append(f"  focus_window_ms: {float(window_ms):g}")
    return "\n".join(lines) + "\n"


def rule_value(stats: CurveStats, rule: str, alpha: float = 0.5) -> float:
    """The confidence *rule* reads off *stats*."""
    if not stats.found:
        return 0.0
    if rule == "product":
        return stats.shape
    if rule == "custom":
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return float(np.clip(stats.ratio * (alpha + (1.0 - alpha) * stats.focus), 0.0, 1.0))
    if rule in ("ratio", "focus", "peak"):
        return stats.statistic(rule)
    raise ValueError(f"unknown confidence rule {rule!r}; one of {list(RULES)}")


def curve_rate(time: np.ndarray) -> float:
    """The curve's own sampling rate, from its time vector — never a setting."""
    time = np.asarray(time, dtype=np.float64)
    if time.size < 2:
        raise ValueError("a curve needs at least two samples to have a rate")
    step = float(np.median(np.diff(time)))
    if step <= 0:
        raise ValueError("the curve's time vector does not increase")
    return 1.0 / step


def confidence_of(curves: dict[str, TrialCurves], trial, label: int, rule: str, alpha: float, window_s: float):
    """The confidence *rule* gives (trial, label) — ``None`` when no curve was written for it."""
    entry = curves.get(str(trial))
    if entry is None:
        return None
    time, per_label = entry
    curve = per_label.get(int(label))
    if curve is None or len(curve) < 2:
        return None
    stats = curve_stats(np.asarray(curve, dtype=np.float64), window_samples(window_s, curve_rate(time)))
    return rule_value(stats, rule, alpha)


def rescore_labels(
    df: pd.DataFrame, curves: dict[str, TrialCurves], rule: str, alpha: float, window_s: float
) -> tuple[pd.DataFrame, list, int]:
    """*df* with every automated label that has a curve re-scored under *rule*.

    Returns ``(new frame, the trials touched, how many rows changed)``. Rows
    that are not automated, or whose (trial, class) no run kept a curve for,
    are left exactly as they were.
    """
    if df is None or df.empty or "labels" not in df.columns:
        return df, [], 0
    out = df.copy()
    if "confidence" not in out.columns:
        out["confidence"] = 1.0
    methods = out["labeling_method"] if "labeling_method" in out.columns else pd.Series(index=out.index, dtype=object)
    touched: list = []
    changed = 0
    for idx in out.index:
        if methods.get(idx) != LABELING_AUTOMATED:
            continue
        trial = out.at[idx, "trial"]
        value = confidence_of(curves, trial, int(out.at[idx, "labels"]), rule, alpha, window_s)
        if value is None:
            continue
        if not np.isclose(float(out.at[idx, "confidence"]), value):
            out.at[idx, "confidence"] = float(value)
            changed += 1
            if trial not in touched:
                touched.append(trial)
    return out, touched, changed
