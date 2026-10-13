"""**More tools → Import BORIS projects…**: a batch of ``.boris`` files becomes sessions.

A thin front for :func:`ethograph.labels.boris.import_boris_project`: pick the
project files, the folder they go to (the ethograph project folder by default)
and, when BORIS's stored media paths are stale, the folder the videos are in.
Every MEDIA observation of every file becomes one session folder; the mapping
is shared across all of them. The dialog never loads anything: the sessions
appear in the project's reopen list and open like any other folder.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.file_dialogs import browse_open_dir, browse_open_file
from ethograph.gui.notify import notify_dialog
from ethograph.gui.project import DropRecord, chosen_project_dir, register_session
from ethograph.labels.boris import (
    BorisSessionResult,
    import_boris_project,
    list_media_observations,
    load_boris_project,
    observation_media_files,
    unique_behavior_codes,
)

logger = logging.getLogger(__name__)

BORIS_FILE_FILTER = "BORIS project (*.boris *.boris.gz);;All files (*)"


class BorisImportDialog(QDialog):
    """Convert one or more BORIS projects into session folders under one project folder."""

    def __init__(self, app_state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app_state = app_state
        self.setWindowTitle("Import BORIS projects")
        self.setMinimumWidth(720)
        self.results: list[BorisSessionResult] = []
        self._setup_ui()
        project = chosen_project_dir(app_state)
        if project is not None:
            self._out_edit.setText(str(project))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "<b>Migrate from BORIS</b><br>"
                "Every MEDIA observation becomes one session folder (one trial per media file in "
                "Player 1); all of them share the project's <code>mapping.txt</code>. "
                "Media files are not moved, only referenced."
            )
        )
        layout.addSpacing(6)

        files_row = QHBoxLayout()
        self._files = QListWidget()
        self._files.setMinimumHeight(90)
        files_row.addWidget(self._files, 1)
        file_btns = QVBoxLayout()
        add_btn = QPushButton("Add .boris…")
        add_btn.clicked.connect(self._on_add_file)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove_file)
        file_btns.addWidget(add_btn)
        file_btns.addWidget(remove_btn)
        file_btns.addStretch()
        files_row.addLayout(file_btns)
        layout.addLayout(files_row)

        form = QFormLayout()
        self._out_edit, row_out = self._folder_row(
            "The ethograph project folder (mapping.txt + sessions)", self._on_browse_output
        )
        form.addRow("Project folder:", row_out)
        self._video_edit, row_vid = self._folder_row(
            "Optional — where the media is now, if BORIS's stored paths no longer exist", self._on_browse_video
        )
        form.addRow("Media folder:", row_vid)
        self._pose_edit, row_pose = self._folder_row(
            "Optional — pose files matched to each video by filename", self._on_browse_pose
        )
        form.addRow("Pose folder:", row_pose)
        layout.addLayout(form)

        self._register_cb = QCheckBox("Add the sessions to the project's reopen list")
        self._register_cb.setChecked(True)
        layout.addWidget(self._register_cb)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #888; padding-top: 6px;")
        layout.addWidget(self._summary)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Close")
        cancel_btn.clicked.connect(self.reject)
        self._import_btn = QPushButton("Import")
        self._import_btn.clicked.connect(self._on_import)
        self._import_btn.setEnabled(False)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._import_btn)
        layout.addLayout(btn_row)

    @staticmethod
    def _folder_row(placeholder: str, browse_cb) -> tuple[QLineEdit, QHBoxLayout]:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("Browse…")
        btn.clicked.connect(browse_cb)
        row.addWidget(edit)
        row.addWidget(btn)
        return edit, row

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def boris_paths(self) -> list[Path]:
        items = (self._files.item(i) for i in range(self._files.count()))
        return [Path(item.text()) for item in items if item is not None]

    def add_boris_file(self, path: str | Path) -> None:
        """Add a project file to the batch (the file browser's callback, also used by tests)."""
        path = Path(path)
        if path in self.boris_paths():
            return
        try:
            project = load_boris_project(path)
        except (OSError, ValueError) as exc:
            notify_dialog(f"Not a BORIS project file:\n{path}\n\n{exc}", "error", "Error", self)
            return
        self._files.addItem(str(path))
        if not self._out_edit.text():
            self._out_edit.setText(str(path.parent))
        self._describe(path, project)
        self._import_btn.setEnabled(True)

    def _on_add_file(self) -> None:
        path = browse_open_file(self, self.app_state, "Select a BORIS project", BORIS_FILE_FILTER)
        if path:
            self.add_boris_file(path)

    def _on_remove_file(self) -> None:
        for item in self._files.selectedItems():
            self._files.takeItem(self._files.row(item))
        self._import_btn.setEnabled(self._files.count() > 0)

    def _on_browse_output(self) -> None:
        folder = browse_open_dir(self, self.app_state, "Project folder the sessions go to")
        if folder:
            self._out_edit.setText(folder)

    def _on_browse_video(self) -> None:
        folder = browse_open_dir(self, self.app_state, "Folder holding the media files")
        if folder:
            self._video_edit.setText(folder)

    def _on_browse_pose(self) -> None:
        folder = browse_open_dir(self, self.app_state, "Pose folder (optional)")
        if folder:
            self._pose_edit.setText(folder)

    def _describe(self, path: Path, project: dict) -> None:
        obs_keys = list_media_observations(project)
        n_files = sum(len(observation_media_files(project["observations"][k])) for k in obs_keys)
        n_events = sum(len(project["observations"][k].get("events", [])) for k in obs_keys)
        self._summary.setText(
            f"<b>{path.name}</b>: {len(obs_keys)} MEDIA observation(s) → {len(obs_keys)} session(s), "
            f"{n_files} media file(s) → trials, {n_events} event rows, "
            f"{len(unique_behavior_codes(project))} behaviours."
        )

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _on_import(self) -> None:
        out_dir = Path(self._out_edit.text())
        if not self._out_edit.text():
            notify_dialog("Choose the project folder the sessions go to.", "warning", "Project folder", self)
            return
        video_folder = self._video_edit.text() or None
        pose_folder = self._pose_edit.text() or None
        self.results = []
        failed: list[str] = []
        for boris_path in self.boris_paths():
            try:
                self.results += import_boris_project(
                    boris_path, out_dir, video_folder=video_folder, pose_folder=pose_folder
                )
            except Exception as exc:  # noqa: BLE001 - outermost GUI boundary
                logger.exception("BORIS import failed for %s", boris_path)
                failed.append(f"{boris_path.name}: {exc}")

        if self._register_cb.isChecked():
            for r in self.results:
                _record_session(r, pose_folder)
                register_session(out_dir, r.session_dir)

        missing = [m for r in self.results for m in r.missing_media]
        lines = [f"{len(self.results)} session(s) written under {out_dir}:"]
        lines += [f"  • {r.session_dir.name}  ({r.n_trials} trial(s), {r.n_labels} label(s))" for r in self.results]
        if missing:
            lines.append("")
            lines.append(
                f"{len(missing)} media file(s) were not found; set the Media folder and import again, "
                "or point the GUI to them on load:"
            )
            lines += [f"  • {m}" for m in missing[:10]]
        if failed:
            lines.append("")
            lines.append("Failed:")
            lines += [f"  • {f}" for f in failed]
        severity = "error" if failed and not self.results else ("warning" if failed or missing else "info")
        notify_dialog("\n".join(lines), severity, "BORIS import", self)
        if self.results and not failed:
            self.accept()


def _record_session(result: BorisSessionResult, pose_folder: str | None) -> None:
    """The reopen list reads drop records, so an imported session gets one: the folder is the session."""
    media_dirs = {str(Path(p).parent) for p in result.media_paths if Path(p).is_file()}
    state: dict[str, object] = {
        "nc_file_path": str(result.session_dir),
        "video_folder": next(iter(sorted(media_dirs)), None),
        "pose_folder": pose_folder,
    }
    DropRecord(created=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), files=[result.observation], state=state).save(
        result.session_dir
    )
