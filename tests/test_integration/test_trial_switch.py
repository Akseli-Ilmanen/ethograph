"""A trial change's view options belong to that change and no other.

``app_state.switching_trial`` scopes ``preserve_x_range`` / ``keep_marker`` to
one trial change. The options must reach ``DataWidget.on_trial_changed`` and
must never survive into a later, unrelated change — not when the block switched
nothing, and not when the change raised half-way.
"""

import pytest

from ethograph.gui.app_state import TrialSwitch


def _spy(meta, monkeypatch) -> tuple[list[bool], list[float]]:
    """Record what each trial change asks of the plot and the marker."""
    preserved: list[bool] = []
    marker_times: list[float] = []
    monkeypatch.setattr(
        meta.data_widget,
        "update_main_plot",
        lambda preserve_x_range=False, **_: preserved.append(preserve_x_range),
    )
    monkeypatch.setattr(meta.plot_container, "update_time_marker_by_time", marker_times.append)
    return preserved, marker_times


def _other_trial(state):
    return next(t for t in state.trials if t != state.trials_sel)


def _switch_to(state, trial) -> None:
    state.trials_sel = trial
    state.trial_changed.emit()


def test_options_end_with_the_block_and_nest(app_state):
    assert app_state.trial_switch == TrialSwitch()
    with app_state.switching_trial(preserve_x_range=True):
        with app_state.switching_trial(keep_marker=True):
            assert app_state.trial_switch == TrialSwitch(keep_marker=True)
        assert app_state.trial_switch == TrialSwitch(preserve_x_range=True)
    assert app_state.trial_switch == TrialSwitch()


def test_plain_switch_recentres_and_resets_the_marker(moll2025_gui, monkeypatch):
    _, meta = moll2025_gui
    state = meta.app_state
    preserved, marker_times = _spy(meta, monkeypatch)

    other = _other_trial(state)
    _switch_to(state, other)

    assert preserved == [False]
    assert state.to_display(other, 0.0) in marker_times


def test_scoped_switch_keeps_the_view_and_the_marker(moll2025_gui, monkeypatch):
    _, meta = moll2025_gui
    state = meta.app_state
    preserved, marker_times = _spy(meta, monkeypatch)

    with state.switching_trial(preserve_x_range=True, keep_marker=True):
        _switch_to(state, _other_trial(state))

    assert preserved == [True]
    assert marker_times == []


def test_a_block_that_switched_nothing_does_not_reach_the_next_change(moll2025_gui, monkeypatch):
    _, meta = moll2025_gui
    state = meta.app_state
    preserved, _ = _spy(meta, monkeypatch)

    # Navigating to the trial already shown fires no trial change.
    with state.switching_trial(preserve_x_range=True):
        meta.navigation_widget.navigate_to_trial(state.trials_sel)
    assert preserved == []

    _switch_to(state, _other_trial(state))
    assert preserved == [False]


def test_a_change_that_raised_does_not_reach_the_next_change(moll2025_gui, monkeypatch):
    _, meta = moll2025_gui
    state = meta.app_state
    preserved, _ = _spy(meta, monkeypatch)

    def _boom():
        raise RuntimeError("audio failed")

    with monkeypatch.context() as patch:
        patch.setattr(meta.data_widget, "update_audio", _boom)
        with pytest.raises(RuntimeError), state.switching_trial(preserve_x_range=True, keep_marker=True):
            meta.data_widget.on_trial_changed()
    assert preserved == []

    _switch_to(state, _other_trial(state))
    assert preserved == [False]
