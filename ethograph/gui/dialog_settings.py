"""The Settings menu's dialogs: the label vocabulary, the individuals, the skeleton.

Three things a study decides once and every session inherits, each edited here
instead of in a text editor:

* **Label mapping** — the rows of ``mapping.txt``: ``id name branch event_type``.
  The id column is the label's identity and is never edited; a new label takes
  the next free id. Until something is saved the file in use is whatever
  :func:`~ethograph.utils.paths.find_mapping_file` resolved (a session's own, the
  project's, or the home defaults); a save goes to the project's ``mapping.txt``
  unless the loaded file is the session's, which stays the session's.
* **Individuals** — the names the data does not declare, kept in
  ``gui_settings.yaml`` (``extra_individuals``) and *added* to the data's own, so
  a video-only session can label somebody with no pose data and no project folder.
* **Skeleton** — ``project.yaml``'s ``pose.skeleton``, drawn with the same
  :class:`~ethograph.gui.dialog_skeleton_editor.SkeletonEditorDialog` as before,
  plus which source wins when the data carries a skeleton too.

The mapping and the skeleton are the *study's*, so they need a project folder and
say so when there is none. The individuals are the *user's*, so they never do.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ethograph.gui.app_constants import MAX_LABEL_BRANCHES
from ethograph.gui.file_dialogs import browse_open_file
from ethograph.gui.notify import notify
from ethograph.gui.project import project_dir_of
from ethograph.io.session_layout import session_dir_of
from ethograph.labels.intervals import EVENT_TYPE_POINT, EVENT_TYPE_STATE, load_label_mapping, save_label_mapping
from ethograph.utils.paths import SETTINGS_DIR, find_mapping_file


def require_project(app_state, parent) -> Path | None:
    """The chosen project folder, or ``None`` after telling the user why nothing opened."""
    project = project_dir_of(app_state)
    if project is None:
        QMessageBox.information(
            parent,
            "No project folder",
            "These are study-wide settings, so they live in the project folder's "
            "project.yaml.\n\nChoose a project folder on the start page first.",
        )
    return project


# ──────────────────────────────────────────────────────────────────────────
# Label mapping
# ──────────────────────────────────────────────────────────────────────────


class LabelMappingDialog(QDialog):
    """Edit ``mapping.txt`` as a table: id (fixed), name, branch, event type."""

    _COLUMNS = ("ID", "Label name", "Branch", "Event type")

    def __init__(self, app_state, labels_widget=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label mapping")
        self.app_state = app_state
        self.labels_widget = labels_widget
        self._source: Path | None = None

        layout = QVBoxLayout(self)
        self._where = QLabel("")
        self._where.setWordWrap(True)
        self._where.setStyleSheet("color: rgba(255,255,255,150);")
        layout.addWidget(self._where)

        self.table = QTableWidget(0, len(self._COLUMNS), self)
        self.table.setHorizontalHeaderLabels(self._COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.table)

        buttons_row = QHBoxLayout()
        add_btn = QPushButton("Add label")
        add_btn.clicked.connect(lambda: self._add_row())  # clicked passes `checked`, not an id
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected)
        open_btn = QPushButton("Open mapping.txt…")
        open_btn.setToolTip("Load another mapping file and edit it here")
        open_btn.clicked.connect(self._open_file)
        for btn in (add_btn, remove_btn, open_btn):
            buttons_row.addWidget(btn)
        buttons_row.addStretch()
        layout.addLayout(buttons_row)

        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self)
        box.accepted.connect(self._save)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

        self._load(self._initial_path())
        self.resize(560, 520)

    # -- loading ---------------------------------------------------------

    def _session_dir(self) -> Path | None:
        source = getattr(self.app_state, "nc_file_path", None) or getattr(self.app_state, "nwb_file_path", None)
        return session_dir_of(source) if source else None

    def _initial_path(self) -> Path | None:
        """The mapping in use: the loaded session's, else the project's, else the home defaults."""
        loaded = getattr(self.labels_widget, "_mapping_file_path", None)
        if loaded:
            return Path(loaded)
        return find_mapping_file(self._session_dir(), project_dir=project_dir_of(self.app_state))

    def _load(self, path: Path | None) -> None:
        mappings = load_label_mapping(path) if path is not None and path.is_file() else {}
        self._source = path if path is not None and path.is_file() else None
        self.table.setRowCount(0)
        for label_id, data in sorted((k, v) for k, v in mappings.items() if isinstance(k, int)):
            self._add_row(label_id, data.get("name", ""), int(data.get("branch", 0)), data.get("event_type"))
        self._where.setText(
            f"Editing {self._source}" if self._source else "No mapping file yet — add labels and save to create one."
        )

    # -- rows ------------------------------------------------------------

    def _add_row(self, label_id: int | None = None, name="", branch=0, event_type=None) -> None:
        row = self.table.rowCount()
        if label_id is None:  # "Add label" with no id of its own: the next free one
            label_id = 1 + max((self._row_id(r) for r in range(row)), default=-1)
            name = f"label{label_id}"
        self.table.insertRow(row)

        id_item = QTableWidgetItem(str(label_id))
        id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
        id_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 0, id_item)
        self.table.setItem(row, 1, QTableWidgetItem(str(name)))

        spin = QSpinBox(self.table)
        spin.setRange(0, MAX_LABEL_BRANCHES - 1)
        spin.setValue(max(0, min(int(branch), MAX_LABEL_BRANCHES - 1)))
        spin.setToolTip(f"Branch: 0 draws full, 1 top1, 2 top2 ({MAX_LABEL_BRANCHES} branches is the renderer's limit)")
        self.table.setCellWidget(row, 2, spin)

        combo = QComboBox(self.table)
        combo.addItem("state", EVENT_TYPE_STATE)
        combo.addItem("point", EVENT_TYPE_POINT)
        combo.setCurrentIndex(1 if event_type == EVENT_TYPE_POINT else 0)
        self.table.setCellWidget(row, 3, combo)

    def _row_id(self, row: int) -> int:
        return int(self.table.item(row, 0).text())

    def _remove_selected(self) -> None:
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)

    def _open_file(self) -> None:
        path = browse_open_file(self, self.app_state, "Select mapping.txt", "Mapping (*.txt);;All files (*)")
        if path:
            self._load(Path(path))

    # -- saving ----------------------------------------------------------

    def _destination(self) -> Path | None:
        """Where a save goes: the session's own file when that is what we loaded, else the project's."""
        session = self._session_dir()
        if self._source is not None and session is not None and self._source == session / SETTINGS_DIR / "mapping.txt":
            return self._source
        project = require_project(self.app_state, self)
        return None if project is None else project / "mapping.txt"

    def _collect(self) -> dict[int, dict] | None:
        mappings: dict[int, dict] = {}
        for row in range(self.table.rowCount()):
            name = (self.table.item(row, 1).text() if self.table.item(row, 1) else "").strip()
            if not name or " " in name:
                notify(f"Row {row + 1}: a label name is one word, and never empty.", "warning")
                return None
            mappings[self._row_id(row)] = {
                "name": name,
                "branch": self.table.cellWidget(row, 2).value(),
                "event_type": self.table.cellWidget(row, 3).currentData(),
            }
        return mappings

    def _save(self) -> None:
        mappings = self._collect()
        if mappings is None:
            return
        destination = self._destination()
        if destination is None:
            return
        save_label_mapping(destination, mappings)
        reload = getattr(self.labels_widget, "_reload_mapping", None)
        if reload is not None:
            reload(str(destination))
        else:
            notify(f"Saved {len(mappings)} labels to {destination}")
        self.accept()


# ──────────────────────────────────────────────────────────────────────────
# Individuals
# ──────────────────────────────────────────────────────────────────────────


class IndividualsDialog(QDialog):
    """Who can be labelled: the data's names, plus your own.

    Two lists, and the split is the whole design. The top one is what the loaded
    data declares — read-only, because renaming it here would only disagree with
    the file. The bottom one is yours, kept in ``gui_settings.yaml`` so it follows
    you across projects and a video-only session needs no project folder to name
    two crows. Additive, always: a name in one list never hides a name in the other.
    """

    def __init__(self, app_state, on_changed=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Individuals")
        self.app_state = app_state
        self._on_changed = on_changed

        layout = QVBoxLayout(self)
        declared = app_state.declared_individuals()
        layout.addWidget(QLabel("From the data (read-only — it is what the file says):"))
        self._declared = QPlainTextEdit(self)
        self._declared.setPlainText(
            "\n".join(declared) if declared else "nothing — this data has no individual dimension"
        )
        self._declared.setReadOnly(True)
        self._declared.setMaximumHeight(90)
        self._declared.setStyleSheet("color: rgba(255,255,255,120);")
        layout.addWidget(self._declared)

        layout.addWidget(QLabel("Yours, one per line (added to the above, for every project):"))
        self._names = QPlainTextEdit(self)
        self._names.setPlainText("\n".join(app_state.get_with_default("extra_individuals")))
        self._names.textChanged.connect(self._sync_effect)
        layout.addWidget(self._names)

        #: The resolved answer, so the dialog says what it will do before it does it.
        self._effect = QLabel("")
        self._effect.setWordWrap(True)
        self._effect.setStyleSheet("color: rgba(255,255,255,150);")
        layout.addWidget(self._effect)

        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self)
        box.accepted.connect(self._save)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self._sync_effect()
        self.resize(460, 460)

    def _typed(self) -> list[str]:
        return [line.strip() for line in self._names.toPlainText().splitlines() if line.strip()]

    def _resolved(self) -> list[str]:
        """What :meth:`AppState.label_individuals` would answer with the list as typed."""
        names: list[str] = []
        for name in (*self.app_state.declared_individuals(), *self._typed()):
            if name not in names:
                names.append(name)
        return names

    def _sync_effect(self) -> None:
        resolved = self._resolved()
        self._effect.setText("You will be able to label: " + (", ".join(resolved) if resolved else "default"))

    def _save(self) -> None:
        names = self._typed()
        if len(set(names)) != len(names):
            notify("An individual is named twice — every name is one animal.", "warning")
            return
        self.app_state.extra_individuals = names
        # Order matters: the sidebar's combos are repopulated first, so the subject
        # re-announced after them is the one the user can now actually see.
        if self._on_changed is not None:
            self._on_changed()
        self.app_state.refresh_labelling_subject()
        notify(f"{len(self.app_state.label_individuals())} individual(s) can be labelled")
        self.accept()


# ──────────────────────────────────────────────────────────────────────────
# Skeleton
# ──────────────────────────────────────────────────────────────────────────


class SkeletonSettingsDialog(QDialog):
    """The skeleton library: pick one, draw one, and say which source wins."""

    def __init__(self, app_state, pose_mgr=None, on_changed=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Skeleton")
        self.app_state = app_state
        self.pose_mgr = pose_mgr
        self._on_changed = on_changed

        layout = QVBoxLayout(self)
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("Skeleton:"))
        self._name = QComboBox(self)
        self._name.setToolTip("One file per skeleton in config/skeleton/ — a study with two rigs has two")
        self._name.currentIndexChanged.connect(lambda _i: self._refresh_connections())
        pick_row.addWidget(self._name, stretch=1)
        layout.addLayout(pick_row)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("When the data has one too, draw:"))
        self._source = QComboBox(self)
        self._source.addItem("the data's skeleton", "nwb")
        self._source.addItem("the one picked above", "library")
        self._source.setCurrentIndex(1 if getattr(app_state, "skeleton_source", "nwb") == "library" else 0)
        self._source.setToolTip("Pick the library's to edit a skeleton without touching an NWB file")
        source_row.addWidget(self._source)
        source_row.addStretch()
        layout.addLayout(source_row)

        self._connections = QListWidget(self)
        layout.addWidget(self._connections)

        buttons_row = QHBoxLayout()
        edit_btn = QPushButton("Draw / edit…")
        edit_btn.setToolTip("Draw connections on this trial's pose data and save them under a name")
        edit_btn.clicked.connect(self._edit)
        delete_btn = QPushButton("Delete")
        delete_btn.setToolTip("Remove the selected skeleton's file from the library")
        delete_btn.clicked.connect(self._delete)
        buttons_row.addWidget(edit_btn)
        buttons_row.addWidget(delete_btn)
        buttons_row.addStretch()
        layout.addLayout(buttons_row)

        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self)
        box.accepted.connect(self._save)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self._reload_library()
        self.resize(480, 480)

    # -- the library -----------------------------------------------------

    def _project(self) -> Path | None:
        """The project whose library is written — ``None`` writes the user's own, which always works."""
        return project_dir_of(self.app_state)

    def _reload_library(self, select: str | None = None) -> None:
        from ethograph.skeleton.library import load_skeletons

        self._library = load_skeletons(self._project())
        wanted = select or getattr(self.app_state, "skeleton_name", None)
        self._name.blockSignals(True)
        self._name.clear()
        for name in sorted(self._library):
            self._name.addItem(name, name)
        index = self._name.findData(wanted)
        self._name.setCurrentIndex(index if index >= 0 else 0)
        self._name.blockSignals(False)
        self._refresh_connections()

    def _selected(self) -> str | None:
        return self._name.currentData()

    def _refresh_connections(self) -> None:
        connections = list((self._library.get(self._selected() or "") or {}).get("connections") or [])
        where = "this project's" if self._project() is not None else "your own"
        self._summary.setText(
            f"{len(connections)} connection(s) in {where} library ({len(self._library)} skeleton(s))."
            if self._library
            else f"No skeleton in {where} library yet — Draw / edit… makes one. The data's own is drawn."
        )
        self._connections.clear()
        for edge in connections:
            self._connections.addItem(f"{edge.get('start', '?')} → {edge.get('end', '?')}")

    # -- actions ---------------------------------------------------------

    def _edit(self) -> None:
        from qtpy.QtWidgets import QInputDialog

        from ethograph.gui.dialog_skeleton_editor import SkeletonEditorDialog
        from ethograph.skeleton.library import save_skeleton

        data = self.pose_mgr.primary_pose_for_editor() if self.pose_mgr is not None else None
        if data is None:
            notify("No pose data available for the current camera/trial to draw a skeleton on.", "warning")
            return
        keypoints, positions = data
        if positions.shape[0] == 0:
            notify("Pose data has no frames to edit.", "warning")
            return
        selected = self._selected()
        dialog = SkeletonEditorDialog(
            keypoints, positions, existing_config=self._library.get(selected or ""), parent=self
        )
        if not dialog.exec_():
            return
        name, ok = QInputDialog.getText(self, "Save skeleton as", "Name (one file):", text=selected or "skeleton")
        if not ok or not name.strip():
            return
        try:
            path = save_skeleton(name.strip(), dialog.get_config(), self._project())
        except (OSError, ValueError) as exc:
            notify(f"Could not save the skeleton: {exc}", "warning")
            return
        # The drawing is now a library file; a stale per-session override would
        # outrank it, so the edit replaces it rather than hiding behind it.
        self.app_state.skeleton_config_override = None
        self.app_state.skeleton_name = name.strip()
        notify(f"Saved {path}")
        self._reload_library(select=name.strip())

    def _delete(self) -> None:
        from ethograph.skeleton.library import delete_skeleton

        name = self._selected()
        if not name:
            return
        if QMessageBox.question(self, "Delete skeleton", f"Delete {name}.yaml from the library?") != (
            QMessageBox.StandardButton.Yes
        ):
            return
        delete_skeleton(name, self._project())
        if getattr(self.app_state, "skeleton_name", None) == name:
            self.app_state.skeleton_name = None
        self._reload_library()

    def _save(self) -> None:
        self.app_state.skeleton_source = self._source.currentData()
        self.app_state.skeleton_name = self._selected()
        if self._on_changed is not None:
            self._on_changed()
        self.accept()


# ──────────────────────────────────────────────────────────────────────────
# Ignored files
# ──────────────────────────────────────────────────────────────────────────


class IgnoredFilesDialog(QDialog):
    """The file-name globs no session folder's root is ever read from.

    A folder holding two ``.nc`` files is refused rather than guessed
    (``AmbiguousSessionError``); this list is how the old one is named, once, for
    every folder. Like the individuals it is the user's (``gui_settings.yaml``):
    a project folder may be copied from one machine to the next, but which file is
    current is answered by whoever is sitting in front of it.
    """

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Excluded files")
        self.app_state = app_state

        layout = QVBoxLayout(self)
        hint = QLabel(
            "File names or globs never loaded from a session folder — an old version "
            "of a dataset beside the current one. One per line, e.g. Trial_data.nc or *_old.nc."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._globs = QPlainTextEdit(self)
        self._globs.setPlainText("\n".join(app_state.get_with_default("ignore_files")))
        layout.addWidget(self._globs)

        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self)
        box.accepted.connect(self._save)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self.resize(460, 340)

    def _save(self) -> None:
        globs: list[str] = []
        for line in self._globs.toPlainText().splitlines():
            glob = line.strip()
            if glob and glob not in globs:
                globs.append(glob)
        self.app_state.ignore_files = globs
        notify(f"{len(self.app_state.ignored_files())} file pattern(s) excluded")
        self.accept()
