"""A list-valued setting survives a restart.

The save filter once wrote only scalars and dicts, so every ``list`` setting the
spec declares as saved — the excluded files among them — was dropped on the way
to disk and had to be entered again each session.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from ethograph.gui.app_state import AppStateSpec, ObservableAppState  # noqa: E402


def _restart() -> ObservableAppState:
    state = ObservableAppState(auto_save_interval=999999)
    state.load_from_yaml()
    return state


def test_excluded_files_are_remembered_across_a_restart(app_state, qtbot):
    app_state.ignore_files = ["Trial_data.nc", "*_old.nc"]
    app_state.save_to_yaml()

    reopened = _restart()
    try:
        assert reopened.ignored_files() == ("Trial_data.nc", "*_old.nc")
    finally:
        reopened.stop_auto_save()


def test_clearing_the_list_is_remembered_too(app_state, qtbot):
    app_state.ignore_files = ["Trial_data.nc"]
    app_state.save_to_yaml()
    app_state.ignore_files = []
    app_state.save_to_yaml()

    reopened = _restart()
    try:
        assert reopened.ignored_files() == ()
    finally:
        reopened.stop_auto_save()


def test_every_saved_list_setting_reaches_the_file(app_state):
    """The spec says ``save``; the writer must agree for every list, not one by name."""
    list_settings = [
        key
        for key in AppStateSpec.saveable_attributes()
        if "list" in str(AppStateSpec.get_type(key))
    ]
    assert list_settings
    for key in list_settings:
        setattr(app_state, key, ["a", "b"])

    saved = app_state.get_saveable_state_dict()
    assert {key: saved.get(key) for key in list_settings} == dict.fromkeys(list_settings, ["a", "b"])
