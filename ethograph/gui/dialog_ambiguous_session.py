"""A session folder with several ``.nc`` files: which one is current?

The answer is a line in the user's own ``ignore_files`` list
(``gui_settings.yaml``, also editable on the cover page), not a file move —
nothing in the session folder is touched.
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

from ethograph.io.data_loader import AmbiguousSessionError


class AmbiguousSessionDialog(QDialog):
    """One row per candidate with its date and a *Never load this* button."""

    def __init__(self, error: AmbiguousSessionError, app_state, parent: QWidget | None = None):
        super().__init__(parent)
        self.app_state = app_state
        self.ignored: list[str] = []
        self.setWindowTitle("Which file is this session's dataset?")
        lay = QVBoxLayout(self)
        lay.addWidget(
            QLabel(f"<b>{error.folder}</b> holds several datasets. One session has one; the others are old versions.")
        )
        for path in sorted(error.candidates, key=lambda p: p.stat().st_mtime, reverse=True):
            row = QHBoxLayout()
            stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            row.addWidget(QLabel(f"{path.name}  <span style='color:#888'>({stamp})</span>"), 1)
            button = QPushButton("Never load this")
            button.setAutoDefault(False)
            button.clicked.connect(lambda _=False, name=path.name, b=button: self._ignore(name, b))
            row.addWidget(button)
            lay.addLayout(row)

        edit_button = QPushButton("Edit the exclusion list…")
        edit_button.setAutoDefault(False)
        edit_button.setToolTip("Every file pattern never loaded from a session folder (also on the start page)")
        edit_button.clicked.connect(self._edit_exclusions)
        lay.addWidget(edit_button)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _ignore(self, name: str, button: QPushButton) -> None:
        """Add this file name to the user's exclusion list, once."""
        globs = [str(g) for g in self.app_state.get_with_default("ignore_files")]
        if name not in globs:
            self.app_state.ignore_files = globs + [name]
        self.ignored.append(name)
        button.setText("Excluded")
        button.setEnabled(False)

    def _edit_exclusions(self) -> None:
        from ethograph.gui.dialog_settings import IgnoredFilesDialog

        before = list(self.app_state.get_with_default("ignore_files"))
        IgnoredFilesDialog(self.app_state, parent=self).exec_()
        after = list(self.app_state.get_with_default("ignore_files"))
        # A retry is worth offering only when the list actually gained something.
        self.ignored.extend(g for g in after if g not in before)


def resolve_ambiguous_session(error: AmbiguousSessionError, app_state, parent: QWidget | None = None) -> bool:
    """Show the dialog; ``True`` when the ignore list changed and the load should be retried."""
    dlg = AmbiguousSessionDialog(error, app_state, parent)
    dlg.exec_()
    return bool(dlg.ignored)
