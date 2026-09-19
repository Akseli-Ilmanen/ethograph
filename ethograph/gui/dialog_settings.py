"""The Settings menu's dialogs: the label vocabulary, the individuals, the skeleton.

Three things a study decides once and every session inherits, each edited here
instead of in a text editor:

* **Label mapping** — the rows of ``mapping.txt``: ``id name branch event_type``.
  The id column is the label's identity and is never edited; a new label takes
  the next free id. Until something is saved the file in use is whatever
  :func:`~ethograph.utils.paths.find_mapping_file` resolved (a session's own, the
  project's, or the home defaults); a save goes to the project's ``mapping.txt``
  unless the loaded file is the session's, which stays the session's.
* **Individuals** — ``project.yaml``'s ``individuals`` and ``individuals_mode``,
  the one answer to who can be labelled (:meth:`AppState.label_individuals`), so
  a video-only session can label somebody without any pose data at all.
* **Skeleton** — ``project.yaml``'s ``pose.skeleton``, drawn with the same
  :class:`~ethograph.gui.dialog_skeleton_editor.SkeletonEditorDialog` as before,
  plus which source wins when the data carries a skeleton too.

Every dialog needs a project folder, because its answer is the *study's*: with
none chosen the dialog says so and opens nothing.
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
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ethograph.gui.file_dialogs import browse_open_file
from ethograph.gui.notify import notify
from ethograph.gui.project import (
    INDIVIDUALS_MODES,
    project_dir_of,
    project_settings_of,
    update_project_settings,
)

#: Branch count is a hard rule of the label renderer (``_BRANCH_POSITION``), not a
#: preference: 0 draws full, 1 top1, 2 top2. The spin box says the same thing.
from ethograph.gui.widgets_labels import MAX_LABEL_BRANCHES
from ethograph.io.session_layout import session_dir_of
from ethograph.labels.intervals import EVENT_TYPE_POINT, EVENT_TYPE_STATE, load_label_mapping, save_label_mapping
from ethograph.utils.paths import SETTINGS_DIR, find_mapping_file

_MODE_LABELS = {
    "inherit": "Inherit from the pose / feature data",
    "define": "Define them here",
    "both": "Inherit, plus the names below",
}


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
        add_btn.clicked.connect(self._add_row)
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

    def _add_row(self, label_id=None, name="", branch=0, event_type=None) -> None:
        row = self.table.rowCount()
        if label_id is None:  # a click on "Add label": the next free id
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
    """Who can be labelled: inherit from the data, define here, or both."""

    def __init__(self, app_state, on_changed=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Individuals")
        self.app_state = app_state
        self._on_changed = on_changed
        settings = project_settings_of(app_state)

        layout = QVBoxLayout(self)
        inherited = self._inherited()
        layout.addWidget(
            QLabel("The data declares: " + (", ".join(inherited) if inherited else "nothing — no individual dimension"))
        )

        self._radios: dict[str, QRadioButton] = {}
        for mode in INDIVIDUALS_MODES:
            radio = QRadioButton(_MODE_LABELS[mode], self)
            radio.setChecked(mode == settings.individuals_mode)
            radio.toggled.connect(self._sync_enabled)
            self._radios[mode] = radio
            layout.addWidget(radio)
        self._radios["inherit"].setToolTip("The default: data that names its individuals wins")
        self._radios["both"].setToolTip("Never a default — the data's names plus the ones you add below")

        layout.addWidget(QLabel("Names, one per line:"))
        self._names = QPlainTextEdit(self)
        self._names.setPlainText("\n".join(settings.individuals))
        layout.addWidget(self._names)

        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self)
        box.accepted.connect(self._save)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self._sync_enabled()
        self.resize(420, 400)

    def _inherited(self) -> list[str]:
        declared = [str(v) for v in getattr(getattr(self.app_state, "nwb_alignment", None), "individuals", [])]
        if declared:
            return declared
        catalog = getattr(getattr(self.app_state, "data_loader", None), "catalog", None)
        if catalog is not None and catalog.individual_combo:
            return [str(v) for v in catalog.combo_values(catalog.individual_combo)]
        return []

    def _mode(self) -> str:
        return next(mode for mode, radio in self._radios.items() if radio.isChecked())

    def _sync_enabled(self) -> None:
        self._names.setEnabled(self._mode() != "inherit")

    def _save(self) -> None:
        project = require_project(self.app_state, self)
        if project is None:
            return
        names = [line.strip() for line in self._names.toPlainText().splitlines() if line.strip()]
        if len(set(names)) != len(names):
            notify("An individual is named twice — every name is one animal.", "warning")
            return
        if self._mode() != "inherit" and not names:
            notify(f"'{_MODE_LABELS[self._mode()]}' needs at least one name.", "warning")
            return
        update_project_settings(project, individuals=names, individuals_mode=self._mode())
        self.app_state.refresh_labelling_subject()
        if self._on_changed is not None:
            self._on_changed()
        notify(f"{len(self.app_state.label_individuals())} individual(s) can be labelled")
        self.accept()


# ──────────────────────────────────────────────────────────────────────────
# Skeleton
# ──────────────────────────────────────────────────────────────────────────


class SkeletonSettingsDialog(QDialog):
    """The project's skeleton, and which source wins when the data has one too."""

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

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("When both exist, draw:"))
        self._source = QComboBox(self)
        self._source.addItem("the data's skeleton", "nwb")
        self._source.addItem("the project's skeleton", "project")
        self._source.setCurrentIndex(1 if getattr(app_state, "skeleton_source", "nwb") == "project" else 0)
        self._source.setToolTip("Pick the project's to edit the skeleton without touching the NWB file")
        source_row.addWidget(self._source)
        source_row.addStretch()
        layout.addLayout(source_row)

        self._connections = QListWidget(self)
        layout.addWidget(self._connections)

        buttons_row = QHBoxLayout()
        edit_btn = QPushButton("Edit skeleton…")
        edit_btn.setToolTip("Draw connections on this trial's pose data; the result becomes the project's skeleton")
        edit_btn.clicked.connect(self._edit)
        clear_btn = QPushButton("Clear project skeleton")
        clear_btn.clicked.connect(self._clear)
        buttons_row.addWidget(edit_btn)
        buttons_row.addWidget(clear_btn)
        buttons_row.addStretch()
        layout.addLayout(buttons_row)

        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self)
        box.accepted.connect(self._save)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self._refresh()
        self.resize(480, 460)

    def _project_skeleton(self) -> dict | None:
        return project_settings_of(self.app_state).skeleton

    def _refresh(self) -> None:
        skeleton = self._project_skeleton()
        connections = list((skeleton or {}).get("connections") or [])
        self._summary.setText(
            f"The project draws {len(connections)} connection(s)."
            if connections
            else "The project has no skeleton — the data's own is drawn."
        )
        self._connections.clear()
        for edge in connections:
            self._connections.addItem(f"{edge.get('start', '?')} → {edge.get('end', '?')}")

    def _pose_block(self, skeleton: dict | None) -> dict | None:
        """``project.yaml``'s ``pose:`` with *skeleton* swapped in, the software it already names kept."""
        software = project_settings_of(self.app_state).pose_software
        block = {}
        if software:
            block["source_software"] = software
        if skeleton is not None:
            block["skeleton"] = skeleton
        return block or None

    def _edit(self) -> None:
        from ethograph.gui.dialog_skeleton_editor import SkeletonEditorDialog

        project = require_project(self.app_state, self)
        if project is None:
            return
        data = self.pose_mgr.primary_pose_for_editor() if self.pose_mgr is not None else None
        if data is None:
            notify("No pose data available for the current camera/trial to draw a skeleton on.", "warning")
            return
        keypoints, positions = data
        if positions.shape[0] == 0:
            notify("Pose data has no frames to edit.", "warning")
            return
        dialog = SkeletonEditorDialog(keypoints, positions, existing_config=self._project_skeleton(), parent=self)
        if not dialog.exec_():
            return
        update_project_settings(project, pose=self._pose_block(skeleton=dialog.get_config()))
        # The project's skeleton is now the edited one; a stale per-session drawing
        # would outrank it, so the edit replaces it rather than hiding behind it.
        self.app_state.skeleton_config_override = None
        self._refresh()

    def _clear(self) -> None:
        project = require_project(self.app_state, self)
        if project is None:
            return
        update_project_settings(project, pose=self._pose_block(skeleton=None))
        self.app_state.skeleton_config_override = None
        self._refresh()

    def _save(self) -> None:
        self.app_state.skeleton_source = self._source.currentData()
        if self._on_changed is not None:
            self._on_changed()
        self.accept()
