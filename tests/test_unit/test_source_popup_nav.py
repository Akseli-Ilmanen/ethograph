"""Keyboard navigation in the add-panel popup.

Typing in the filter box narrows the list; ↑/↓ must walk the entries that
survived the filter. The keys only reach the popup because guarded global
shortcuts (``Up``/``Down`` = prev/next trial) are *disabled* while a text
field has focus — an enabled application-context QShortcut swallows the key
press before the focus widget sees it.
"""

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QLineEdit

pytest.importorskip("qtpy")

from ethograph.gui.main_window import EthographMainWindow  # noqa: E402
from ethograph.gui.shortcuts import bind_global_shortcuts  # noqa: E402
from ethograph.gui.source_popup import PlotTypePicker, SourcePopup  # noqa: E402


class _StubState:
    nwb_alignment = None
    ds = None
    data_loader = None
    image_paths: list[str] = []
    audio_mic_channels: dict = {}
    audio_source_map: dict = {}


class _StubCatalog:
    def feature_choices(self):
        return ["speed", "aux_speed", "position", "heading"]


def _visible_labels(popup):
    return [popup._list.item(i).text().strip() for i in popup._visible_data_rows()]


@pytest.fixture
def popup(qtbot):
    p = SourcePopup(_StubState())
    qtbot.addWidget(p)
    p.refresh(catalog=_StubCatalog())
    return p


def test_filter_narrows_to_matching_features(popup):
    popup._filter.setText("speed")
    assert _visible_labels(popup) == ["speed", "aux_speed"]


def test_arrow_keys_walk_filtered_entries(qtbot, popup):
    popup.show()
    qtbot.waitExposed(popup)
    popup._filter.setFocus()
    qtbot.keyClicks(popup._filter, "speed")
    assert popup._list.currentItem().text().strip() == "speed"

    qtbot.keyClick(popup._filter, Qt.Key_Down)
    assert popup._list.currentItem().text().strip() == "aux_speed"

    qtbot.keyClick(popup._filter, Qt.Key_Up)
    assert popup._list.currentItem().text().strip() == "speed"

    # Wrapping: ↑ from the first entry lands on the last visible one.
    qtbot.keyClick(popup._filter, Qt.Key_Up)
    assert popup._list.currentItem().text().strip() == "aux_speed"


def test_enter_activates_the_highlighted_entry(qtbot, popup):
    chosen = []
    popup.on_activate = lambda kind, name: chosen.append((kind, name))
    popup.show()
    qtbot.waitExposed(popup)
    popup._filter.setFocus()
    qtbot.keyClicks(popup._filter, "speed")
    qtbot.keyClick(popup._filter, Qt.Key_Down)
    qtbot.keyClick(popup._filter, Qt.Key_Return)
    assert chosen == [("feature", "aux_speed")]


def test_guarded_shortcuts_release_arrow_keys_to_text_fields(qtbot):
    """A guarded shortcut must be *disabled* (not a no-op) while typing —
    otherwise it consumes the key and the focused widget never sees it."""
    shell = EthographMainWindow()
    qtbot.addWidget(shell)
    edit = QLineEdit(shell)
    shell.setCentralWidget(edit)
    shell.show()
    qtbot.waitExposed(shell)

    fired = []
    guarded = shell.bind_shortcut("Down", lambda: fired.append("guarded"), guarded=True)
    plain = shell.bind_shortcut("Ctrl+Down", lambda: fired.append("plain"))

    edit.setFocus()
    qtbot.waitUntil(lambda: edit.hasFocus())
    shell._sync_guarded_shortcuts()
    assert not guarded.isEnabled()
    assert plain.isEnabled()

    shell.setFocus()
    qtbot.waitUntil(lambda: not edit.hasFocus())
    shell._sync_guarded_shortcuts()
    assert guarded.isEnabled()

    shell.clear_shortcuts()
    assert shell._guarded_shortcuts == []


def test_text_editing_shortcuts_are_guarded(qtbot):
    """Keys a text editor owns are guarded even when the call site doesn't ask.

    An unguarded ``Ctrl+C`` would eat the copy out of a metadata cell and run
    the action behind it (curate the trial) — which then asks to save labels
    on close after a metadata-only edit.
    """
    shell = EthographMainWindow()
    qtbot.addWidget(shell)
    meta = MagicMock()
    meta.shell = shell

    bind_global_shortcuts(meta)

    guarded = {s.key().toString() for s in shell._guarded_shortcuts}
    assert {"Ctrl+C", "Ctrl+A", "Ctrl+Z", "Ctrl+Left", "Ctrl+Right"} <= guarded
    assert "Ctrl+S" not in guarded  # saving stays available while typing
    shell.clear_shortcuts()  # application-context shortcuts outlive the hidden shell


def test_plot_type_picker_arrows_move_its_selection_not_the_trial(qtbot):
    """The picker is a modal list with focus; the global ``Up``/``Down``
    (prev/next trial) shortcuts must be released while it is open, or they
    swallow the arrows and the highlighted plot type never moves."""
    shell = EthographMainWindow()
    qtbot.addWidget(shell)
    shell.show()
    qtbot.waitExposed(shell)
    fired = []
    shell.bind_shortcut("Down", lambda: fired.append("next_trial"), guarded=True)

    picker = PlotTypePicker(["Lineplot", "Heatmap", "Space (2D)"], parent=shell)
    qtbot.addWidget(picker)
    picker.show()  # non-blocking; still registers as the active modal
    qtbot.waitExposed(picker)
    picker._list.setFocus()
    qtbot.waitUntil(lambda: QApplication.activeModalWidget() is picker)
    shell._sync_guarded_shortcuts()

    qtbot.keyClick(picker._list, Qt.Key_Down)
    assert picker._list.currentItem().text() == "Heatmap"
    assert fired == []

    qtbot.keyClick(picker._list, Qt.Key_Return)
    assert picker.choice == "Heatmap"
