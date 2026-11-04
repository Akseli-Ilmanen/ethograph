"""Trial bars: one per visible trial on the session clock, culled to the viewport."""

from __future__ import annotations

from ethograph.gui.trial_bars import trial_bars
from ethograph.io.time_model import SourceCollection, TimeRange


def _collection() -> SourceCollection:
    sc = SourceCollection()
    sc.set_trials([1, 2, 3, 4], starts=[0.0, 10.0, 20.0, 30.0], stops=[9.0, 19.0, 29.0, 39.0])
    return sc


def test_bars_follow_the_trial_filter_and_the_session_clock():
    bars = trial_bars(_collection(), [2, 4])
    assert [(b.trial_id, b.start_s, b.end_s) for b in bars] == [(2, 10.0, 19.0), (4, 30.0, 39.0)]
    assert bars[0].color != bars[1].color


def test_bars_outside_the_padded_viewport_are_culled():
    # Viewport 10–19 padded by half a window: trial 1 (ends 9.0) and 2 stay,
    # trial 3 (starts 20.0, within 4.5 s of the edge) stays, trial 4 goes.
    bars = trial_bars(_collection(), [1, 2, 3, 4], TimeRange(10.0, 19.0))
    assert [b.trial_id for b in bars] == [1, 2, 3]


def test_unknown_trial_ids_are_skipped():
    assert trial_bars(_collection(), [99]) == []
