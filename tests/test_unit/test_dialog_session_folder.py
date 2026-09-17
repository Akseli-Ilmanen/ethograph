"""The multi-folder drop popup: preselection, remembering a kind, "Other folder"."""

from pathlib import Path

from ethograph.gui.dialog_session_folder import SessionFolderDialog, choose_session_folder
from ethograph.gui.session_folder import SourceFolder


class _State:
    session_folder_kinds: list[str] = []


def _folders(tmp_path: Path) -> list[SourceFolder]:
    return [
        SourceFolder(tmp_path / "cams", {"video": 3}),
        SourceFolder(tmp_path / "mics", {"audio": 5}),
    ]


def test_single_folder_needs_no_dialog(qapp, tmp_path: Path):
    only = [SourceFolder(tmp_path / "cams", {"video": 3})]
    assert choose_session_folder(only, _State()) == tmp_path / "cams"


def test_preselects_by_hierarchy_and_remembers_the_chosen_kind(qapp, tmp_path: Path):
    state = _State()
    dlg = SessionFolderDialog(_folders(tmp_path), state)
    assert dlg.chosen_folder() == tmp_path / "cams"  # video outranks audio
    dlg._radios[1].setChecked(True)
    dlg.accept()
    assert dlg.result() == 1
    assert dlg.chosen_folder() == tmp_path / "mics"
    assert state.session_folder_kinds == ["audio"]

    again = SessionFolderDialog(_folders(tmp_path), state)
    assert again.chosen_folder() == tmp_path / "mics"  # the remembered kind wins next time


def test_unticked_remember_stores_nothing(qapp, tmp_path: Path):
    state = _State()
    dlg = SessionFolderDialog(_folders(tmp_path), state)
    dlg._remember.setChecked(False)
    dlg._radios[1].setChecked(True)
    dlg.accept()
    assert state.session_folder_kinds == []


def test_other_folder_requires_a_path(qapp, tmp_path: Path):
    dlg = SessionFolderDialog(_folders(tmp_path), _State())
    dlg._other_radio.setChecked(True)
    dlg.accept()
    assert dlg.result() == 0  # refused: nothing picked
    dlg._other_edit.setText(str(tmp_path / "elsewhere"))
    dlg.accept()
    assert dlg.result() == 1 and dlg.chosen_folder() == tmp_path / "elsewhere"
