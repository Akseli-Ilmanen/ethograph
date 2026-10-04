"""A dialog taller than the screen still fits it: its content scrolls, its buttons stay put."""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QDialog, QDialogButtonBox, QLabel, QScrollArea, QVBoxLayout  # noqa: E402

from ethograph.gui.dialog_fit import fit_dialog  # noqa: E402


def _tall_dialog() -> tuple[QDialog, QDialogButtonBox]:
    dialog = QDialog()
    layout = QVBoxLayout(dialog)
    for i in range(200):
        layout.addWidget(QLabel(f"row {i}"))
    buttons = QDialogButtonBox(QDialogButtonBox.Close)
    layout.addWidget(buttons)
    return dialog, buttons


def test_tall_dialog_scrolls_within_the_screen(qapp):
    dialog, buttons = _tall_dialog()
    content_height = dialog.sizeHint().height()

    fit_dialog(dialog)

    available = dialog.screen().availableGeometry()
    assert content_height > available.height()
    assert dialog.height() <= available.height()
    assert dialog.minimumSizeHint().height() < available.height()
    scroll = dialog.findChild(QScrollArea)
    assert scroll is not None
    assert not scroll.widget().isAncestorOf(buttons)
    assert scroll.widget().findChildren(QLabel)
