"""The "which file?" dialog writes the user's exclusion list; it never moves a file.

The list is ``ignore_files`` in ``gui_settings.yaml``, not a project file: a
project folder can be copied between machines, but which of two datasets in a
folder is the current one is answered by whoever is sitting in front of it. So
the dialog works with no project folder at all.
"""

from pathlib import Path

from qtpy.QtWidgets import QPushButton

from ethograph.gui.dialog_ambiguous_session import AmbiguousSessionDialog
from ethograph.io.data_loader import AmbiguousSessionError


class _State:
    def __init__(self) -> None:
        self.project_path = None
        self.ignore_files: list[str] = []

    def get_with_default(self, name):
        return getattr(self, name)


def _error(tmp_path: Path) -> AmbiguousSessionError:
    folder = tmp_path / "sess"
    folder.mkdir()
    a, b = folder / "Trial_data.nc", folder / "Trial_data3.nc"
    a.touch()
    b.touch()
    return AmbiguousSessionError(folder, [a, b])


def _exclude_buttons(dlg: AmbiguousSessionDialog) -> list[QPushButton]:
    return [b for b in dlg.findChildren(QPushButton) if b.text() == "Never load this"]


def test_the_button_excludes_the_file_and_moves_nothing(qapp, tmp_path: Path):
    state = _State()
    error = _error(tmp_path)
    dlg = AmbiguousSessionDialog(error, state)

    buttons = _exclude_buttons(dlg)
    assert len(buttons) == 2, "one per candidate, with no project folder anywhere"

    dlg._ignore("Trial_data.nc", buttons[0])
    assert dlg.ignored == ["Trial_data.nc"]
    assert state.ignore_files == ["Trial_data.nc"]
    assert not buttons[0].isEnabled() and buttons[0].text() == "Excluded"
    assert all(p.exists() for p in error.candidates), "the session folder is never touched"

    dlg._ignore("Trial_data.nc", buttons[0])  # a second click adds nothing
    assert state.ignore_files == ["Trial_data.nc"]
