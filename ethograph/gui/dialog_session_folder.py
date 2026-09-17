"""The one question a multi-folder drop asks: where do the session files go?"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.file_dialogs import browse_open_dir
from ethograph.gui.session_folder import SourceFolder, propose, remember


class SessionFolderDialog(QDialog):
    """Radio per source folder, labelled by what came from it, plus "Other folder…".

    The preselected radio follows the kind hierarchy and the user's remembered
    choice; ticking *Remember my choice* (on by default) promotes the chosen
    folder's kind for next time. Only a kind is ever remembered, never a path.
    """

    def __init__(self, folders: list[SourceFolder], app_state, parent: QWidget | None = None):
        super().__init__(parent)
        self.app_state = app_state
        self._folders = folders
        self.setWindowTitle("Session folder")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("<b>Where to put session files (alignment, labels, metadata)?</b>"))
        lay.addWidget(QLabel("Media stay where they are; only the chosen folder gets an .ethograph/ subfolder."))

        preferred = list(getattr(app_state, "session_folder_kinds", None) or [])
        proposed = propose(folders, preferred)
        self._group = QButtonGroup(self)
        self._radios: list[QRadioButton] = []
        for i, f in enumerate(folders):
            radio = QRadioButton(f.label())
            radio.setChecked(f is proposed)
            self._group.addButton(radio, i)
            self._radios.append(radio)
            lay.addWidget(radio)

        other_row = QHBoxLayout()
        self._other_radio = QRadioButton("Other folder:")
        self._group.addButton(self._other_radio, len(folders))
        self._other_edit = QLineEdit()
        self._other_edit.setReadOnly(True)
        browse = QPushButton("Browse…")
        browse.setAutoDefault(False)
        browse.clicked.connect(self._browse)
        other_row.addWidget(self._other_radio)
        other_row.addWidget(self._other_edit, 1)
        other_row.addWidget(browse)
        lay.addLayout(other_row)

        self._remember = QCheckBox("Remember my choice (the kind of folder, not this path)")
        self._remember.setChecked(True)
        lay.addWidget(self._remember)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _browse(self) -> None:
        path = browse_open_dir(self, self.app_state, "Session folder")
        if path:
            self._other_edit.setText(path)
            self._other_radio.setChecked(True)

    def chosen_folder(self) -> Path | None:
        """The folder picked, or ``None`` when "Other" is selected with no path."""
        idx = self._group.checkedId()
        if 0 <= idx < len(self._folders):
            return self._folders[idx].folder
        text = self._other_edit.text().strip()
        return Path(text) if text else None

    def accept(self) -> None:
        if self.chosen_folder() is None:
            self._other_edit.setPlaceholderText("Pick a folder first")
            return
        idx = self._group.checkedId()
        if self._remember.isChecked() and 0 <= idx < len(self._folders):
            preferred = list(getattr(self.app_state, "session_folder_kinds", None) or [])
            self.app_state.session_folder_kinds = remember(preferred, self._folders[idx].best_kind)
        super().accept()


def choose_session_folder(folders: list[SourceFolder], app_state, parent: QWidget | None = None) -> Path | None:
    """The session folder for a drop: the one folder when there is one, else ask.

    Returns ``None`` when the user cancels.
    """
    if len(folders) == 1:
        return folders[0].folder
    dlg = SessionFolderDialog(folders, app_state, parent)
    if dlg.exec_():
        return dlg.chosen_folder()
    return None
