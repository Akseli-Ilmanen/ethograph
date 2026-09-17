"""The "which file?" dialog writes the project's ignore list; it never moves a file."""

from pathlib import Path

from qtpy.QtWidgets import QPushButton

from ethograph.gui.dialog_ambiguous_session import AmbiguousSessionDialog
from ethograph.gui.project import load_project_settings
from ethograph.io.data_loader import AmbiguousSessionError


class _State:
    def __init__(self, project: Path | None) -> None:
        self.project_path = str(project) if project else None


def _error(tmp_path: Path) -> AmbiguousSessionError:
    folder = tmp_path / "sess"
    folder.mkdir()
    a, b = folder / "Trial_data.nc", folder / "Trial_data3.nc"
    a.touch()
    b.touch()
    return AmbiguousSessionError(folder, [a, b])


def _ignore_buttons(dlg: AmbiguousSessionDialog) -> list[QPushButton]:
    return [b for b in dlg.findChildren(QPushButton) if b.text() == "Ignore in this project"]


def test_ignore_button_appends_to_project_yaml_and_moves_nothing(qapp, tmp_path: Path):
    project = tmp_path / "study"
    project.mkdir()
    error = _error(tmp_path)
    dlg = AmbiguousSessionDialog(error, _State(project))
    buttons = _ignore_buttons(dlg)
    assert len(buttons) == 2
    dlg._ignore("Trial_data.nc", buttons[0])
    assert dlg.ignored == ["Trial_data.nc"]
    assert load_project_settings(project).ignore == ("Trial_data.nc",)
    assert not buttons[0].isEnabled() and buttons[0].text() == "Ignored"
    assert all(p.exists() for p in error.candidates)


def test_without_a_project_nothing_can_be_ignored(qapp, tmp_path: Path):
    dlg = AmbiguousSessionDialog(_error(tmp_path), _State(None))
    assert dlg.project is None
    assert all(not b.isEnabled() for b in _ignore_buttons(dlg))
