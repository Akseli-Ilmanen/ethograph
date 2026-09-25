"""The confidence knob's arithmetic: which labels it may touch, and what the rules read.

What earns a test: only an automated label with a curve changes, and a
manual one never does; the custom rule's slider ends at the two named rules;
the window is a duration resolved against the curve's own rate; and the
frame comes back with the trials it touched, so the undo step can be
recorded per trial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethograph.labels.confidence import CurveStats
from ethograph.labels.intervals import LABELING_AUTOMATED, LABELING_CURATED, LABELING_MANUAL
from ethograph.labels.rescore import RULES, confidence_of, curve_rate, rescore_labels, rule_value, yaml_snippet


def _bump(t, c, h, w=0.03):
    return h * np.exp(-0.5 * ((t - c) / w) ** 2)


def _curves():
    t = np.arange(0, 4.0, 0.01)  # 100 Hz
    sharp = _bump(t, 2.0, 0.9)
    twin = _bump(t, 2.0, 0.9) + _bump(t, 0.8, 0.85)
    return {"1": (t, {31: sharp, 32: twin}), "2": (t, {31: sharp})}


def _df():
    return pd.DataFrame(
        {
            "trial": [1, 1, 2, 2],
            "labels": [31, 32, 31, 33],
            "onset_s": [2.0, 2.0, 2.0, 1.0],
            "offset_s": [np.nan] * 4,
            "confidence": [0.5, 0.5, 1.0, 0.4],
            "labeling_method": [LABELING_AUTOMATED, LABELING_AUTOMATED, LABELING_MANUAL, LABELING_AUTOMATED],
        }
    )


class TestRules:
    def test_custom_ends_at_the_named_rules(self):
        s = CurveStats(index=0, peak=0.9, focus=0.4, ratio=0.8)
        assert rule_value(s, "custom", 1.0) == pytest.approx(s.ratio)
        assert rule_value(s, "custom", 0.0) == pytest.approx(rule_value(s, "product"))
        assert rule_value(s, "custom", 0.5) == pytest.approx(0.8 * 0.7)

    def test_every_rule_is_readable(self):
        s = CurveStats(index=0, peak=0.9, focus=0.4, ratio=0.8)
        assert {k: rule_value(s, k) for k in RULES} == pytest.approx(
            {"product": 0.32, "ratio": 0.8, "focus": 0.4, "peak": 0.9, "custom": 0.56}
        )
        with pytest.raises(ValueError, match="unknown confidence rule"):
            rule_value(s, "entropy")

    def test_the_window_is_a_duration_on_the_curves_clock(self):
        curves = _curves()
        assert curve_rate(curves["1"][0]) == pytest.approx(100.0)
        # the twin peak is 1.2 s away: within a 2 s window it is "the same event", within 100 ms a rival
        assert confidence_of(curves, 1, 32, "ratio", 0.5, window_s=2.0) == pytest.approx(1.0)
        assert confidence_of(curves, 1, 32, "ratio", 0.5, window_s=0.1) < 0.1
        assert confidence_of(curves, 1, 99, "ratio", 0.5, 0.1) is None  # no curve for that class


class TestRescore:
    def test_only_automated_labels_with_a_curve_change(self):
        out, touched, n = rescore_labels(_df(), _curves(), "ratio", 0.5, 0.1)
        assert n == 2 and touched == [1]
        assert out.loc[0, "confidence"] == pytest.approx(1.0)  # sharp lone bump
        assert out.loc[1, "confidence"] < 0.1  # twin peak
        assert out.loc[2, "confidence"] == 1.0  # manual: untouched, even though a curve exists
        assert out.loc[3, "confidence"] == 0.4  # automated but no curve for class 33

    def test_unchanged_values_do_not_count_and_the_input_is_not_mutated(self):
        df = _df()
        first, _, _ = rescore_labels(df, _curves(), "ratio", 0.5, 0.1)
        again, touched, n = rescore_labels(first, _curves(), "ratio", 0.5, 0.1)
        assert n == 0 and touched == []
        assert df.loc[0, "confidence"] == 0.5

    def test_curated_labels_are_a_humans_word(self):
        df = _df()
        df.loc[0, "labeling_method"] = LABELING_CURATED
        out, _, n = rescore_labels(df, _curves(), "product", 0.5, 0.1)
        assert n == 1 and out.loc[0, "confidence"] == 0.5

    def test_empty_frames_pass_through(self):
        empty = pd.DataFrame()
        assert rescore_labels(empty, _curves(), "ratio", 0.5, 0.1) == (empty, [], 0)


class TestSnippet:
    def test_the_popup_copies_the_infer_lines_and_alpha_only_for_custom(self):
        assert yaml_snippet("ratio", 0.5, 100.0) == "infer:\n  confidence: ratio\n  focus_window_ms: 100\n"
        assert "confidence_alpha: 0.8" in yaml_snippet("custom", 0.8, 60.0)
        with pytest.raises(ValueError, match="unknown confidence rule"):
            yaml_snippet("entropy", 0.5, 100.0)
