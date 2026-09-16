"""File ▸ Open project folder… follows ``app_state.project_path``: live while a
project folder is set and exists, greyed out otherwise."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QObject, Signal

pytest.importorskip("qtpy")

from ethograph.gui.main_window import EthographMainWindow  # noqa: E402
from ethograph.gui.top_bar import build_menu_bar  # noqa: E402


class _State(QObject):
    project_path_changed = Signal(object)

    def __init__(self, project_path):
        super().__init__()
        self.project_path = project_path

    def save_to_yaml(self, *_args, **_kwargs):
        pass

    def set_project(self, value):
        self.project_path = value
        self.project_path_changed.emit(value)


def _file_action(shell, text):
    file_menu = next(a.menu() for a in shell.menuBar().actions() if a.text() == "&File")
    return next(a for a in file_menu.actions() if a.text() == text)


def _shell(qtbot, project_path):
    shell = EthographMainWindow()
    qtbot.addWidget(shell)
    meta = MagicMock()
    meta.shell = shell
    meta.app_state = _State(project_path)
    shell.meta_widget = meta
    build_menu_bar(shell)
    return shell, meta.app_state


def test_greyed_out_without_a_project_and_live_once_one_is_set(qtbot, tmp_path):
    shell, state = _shell(qtbot, None)
    action = _file_action(shell, "Open project folder…")
    assert not action.isEnabled()

    state.set_project(str(tmp_path))
    assert action.isEnabled() and action.toolTip() == str(tmp_path)

    state.set_project(str(tmp_path / "gone"))  # set, but no longer on disk
    assert not action.isEnabled()


def test_opens_the_project_folder(qtbot, tmp_path, monkeypatch):
    shell, _state = _shell(qtbot, str(tmp_path))
    opened = []
    monkeypatch.setattr("ethograph.gui.top_bar.TopBarBuilder._open_folder", staticmethod(opened.append))
    _file_action(shell, "Open project folder…").trigger()
    assert opened == [tmp_path]
