"""A session folder with several ``.nc`` files: which one is current?

The answer is a line in the project's ``project.yaml`` ``ignore`` list, not a
file move — nothing on disk is touched except that file, and only on a click.
"""

from __future__ import annotations

from datetime import datetime

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.project import PROJECT_SETTINGS_FILENAME, add_ignore, project_dir_of
from ethograph.io.data_loader import AmbiguousSessionError


class AmbiguousSessionDialog(QDialog):
    """One row per candidate with its date and an *Ignore in this project* button."""

    def __init__(self, error: AmbiguousSessionError, app_state, parent: QWidget | None = None):
        super().__init__(parent)
        self.app_state = app_state
        self.project = project_dir_of(app_state)
        self.ignored: list[str] = []
        self.setWindowTitle("Which file is this session's dataset?")
        lay = QVBoxLayout(self)
        lay.addWidget(
            QLabel(f"<b>{error.folder}</b> holds several datasets. One session has one; the others are old versions.")
        )
        if self.project is None:
            lay.addWidget(
                QLabel("Set a project folder on the start page first: the files to skip go in its project.yaml.")
            )
        for path in sorted(error.candidates, key=lambda p: p.stat().st_mtime, reverse=True):
            row = QHBoxLayout()
            stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            row.addWidget(QLabel(f"{path.name}  <span style='color:#888'>({stamp})</span>"), 1)
            button = QPushButton("Ignore in this project")
            button.setAutoDefault(False)
            button.setEnabled(self.project is not None)
            button.clicked.connect(lambda _=False, name=path.name, b=button: self._ignore(name, b))
            row.addWidget(button)
            lay.addLayout(row)

        open_button = QPushButton(f"Open {PROJECT_SETTINGS_FILENAME}…")
        open_button.setAutoDefault(False)
        open_button.setEnabled(self.project is not None)
        open_button.clicked.connect(self._open_project_yaml)
        lay.addWidget(open_button)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _ignore(self, name: str, button: QPushButton) -> None:
        assert self.project is not None
        add_ignore(self.project, name)
        self.ignored.append(name)
        button.setText("Ignored")
        button.setEnabled(False)

    def _open_project_yaml(self) -> None:
        from ethograph.gui.wizard_overview import open_with_chooser

        assert self.project is not None
        path = self.project / PROJECT_SETTINGS_FILENAME
        if not path.exists():
            add_ignore(self.project, "")  # creates the file with an empty list
        open_with_chooser(path)


def resolve_ambiguous_session(error: AmbiguousSessionError, app_state, parent: QWidget | None = None) -> bool:
    """Show the dialog; ``True`` when the ignore list changed and the load should be retried."""
    dlg = AmbiguousSessionDialog(error, app_state, parent)
    dlg.exec_()
    return bool(dlg.ignored)
