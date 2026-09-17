"""Window menu: every titled open dialog is listed and can be pulled back onto
the main window's screen. Dialogs are owned windows with no taskbar entry, so
this menu is the only way to find one that was dragged away."""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QDialog, QMainWindow

pytest.importorskip("qtpy")

from ethograph.gui.top_bar import move_onto_screen, open_dialog_windows  # noqa: E402


@pytest.fixture
def shell(qapp):
    win = QMainWindow()
    win.setWindowTitle("shell")
    win.resize(400, 300)
    win.show()
    yield win
    win.close()


def test_lists_only_titled_visible_dialogs(shell):
    titled = QDialog(shell)
    titled.setWindowTitle("Label table")
    titled.show()
    untitled = QDialog(shell)
    untitled.show()
    hidden = QDialog(shell)
    hidden.setWindowTitle("Hidden")

    found = open_dialog_windows(shell)

    assert titled in found
    assert untitled not in found
    assert hidden not in found
    assert shell not in found


def test_gather_moves_offscreen_dialog_onto_shell_screen(shell):
    dlg = QDialog(shell)
    dlg.setWindowTitle("Lost")
    dlg.resize(200, 100)
    dlg.show()
    avail = shell.screen().availableGeometry()
    dlg.move(avail.right() + 5000, avail.bottom() + 5000)

    move_onto_screen(dlg, shell)

    assert avail.contains(dlg.frameGeometry().center())
