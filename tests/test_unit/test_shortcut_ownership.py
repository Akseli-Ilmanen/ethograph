"""Every key sequence has exactly ONE owner.

``Ctrl+S`` was bound twice — as an application-context QShortcut in
``gui/shortcuts.py`` and again on the File menu's "Save labels" QAction. Qt
answers an ambiguous overload by firing *neither*, so clicking the menu entry
saved while the key did nothing at all.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("qtpy")

from ethograph.gui.main_window import EthographMainWindow  # noqa: E402
from ethograph.gui.shortcuts import bind_global_shortcuts  # noqa: E402
from ethograph.gui.top_bar import build_menu_bar  # noqa: E402


def _menu_shortcuts(shell) -> list[str]:
    keys = []
    for menu_action in shell.menuBar().actions():
        menu = menu_action.menu()
        if menu is None:
            continue
        for action in menu.actions():
            text = action.shortcut().toString()
            if text:
                keys.append(text)
    return keys


def _shell_with_menu_and_shortcuts(qtbot):
    shell = EthographMainWindow()
    qtbot.addWidget(shell)
    meta = MagicMock()
    meta.shell = shell
    shell.meta_widget = meta
    build_menu_bar(shell)
    bind_global_shortcuts(meta)
    return shell, meta


def test_menu_actions_never_duplicate_a_global_shortcut(qtbot):
    shell, _meta = _shell_with_menu_and_shortcuts(qtbot)

    bound = {s.key().toString() for s in shell._shortcuts}
    bound.update(a.shortcut().toString() for a in shell.actions() if a.shortcut().toString())
    clashes = [key for key in _menu_shortcuts(shell) if key in bound]
    assert clashes == []


def test_ctrl_s_reaches_the_save_handler(qtbot):
    """The one Ctrl+S owner is the global QShortcut wired to the save handler.

    Fired through the shortcut itself: an application-context QShortcut is
    only consulted while the shell is the OS-active window, which a headless
    pytest process on Windows is never granted.
    """
    shell, meta = _shell_with_menu_and_shortcuts(qtbot)

    owners = [s for s in shell._shortcuts if s.key().toString() == "Ctrl+S"]
    assert len(owners) == 1
    owners[0].activated.emit()
    assert meta.io_widget._save_labels.call_count == 1
