"""The editing branch is a global setting: remembered across sessions, falling
back to the first branch present when the remembered one does not exist."""

import pytest

pytest.importorskip("qtpy")


def test_switching_branch_writes_the_global_setting(gui):
    _viewer, meta = gui
    labels = meta.labels_widget
    labels._add_branch_section(1)
    labels.set_active_branch(1)
    assert meta.app_state.active_branch == 1
    assert meta.app_state._active_branch == 1


def test_restore_uses_the_remembered_branch_when_present(gui):
    _viewer, meta = gui
    labels = meta.labels_widget
    labels._add_branch_section(1)
    meta.app_state.active_branch = 1
    labels.restore_active_branch()
    assert meta.app_state._active_branch == 1


def test_restore_falls_back_to_the_first_branch(gui):
    _viewer, meta = gui
    labels = meta.labels_widget
    meta.app_state.active_branch = 2  # no such branch in this mapping
    labels.restore_active_branch()
    assert meta.app_state._active_branch == min(labels._branch_sections)


def test_deleting_the_active_branch_updates_the_setting(gui):
    _viewer, meta = gui
    labels = meta.labels_widget
    labels._add_branch_section(1)
    labels.set_active_branch(1)
    labels._delete_branch(1)
    assert meta.app_state.active_branch == 0
