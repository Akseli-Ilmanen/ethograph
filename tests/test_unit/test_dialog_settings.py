"""The Settings dialogs write the files the rest of the app reads.

Two things nothing else forces to agree: the mapping table's rows and
``mapping.txt``'s columns, and the individuals dialog's mode and
``project.yaml``'s.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QPushButton

pytest.importorskip("qtpy")

from ethograph.gui.dialog_settings import IndividualsDialog, LabelMappingDialog  # noqa: E402
from ethograph.labels.intervals import EVENT_TYPE_POINT, load_label_mapping  # noqa: E402


class _State(QObject):
    project_path_changed = Signal(object)

    def __init__(self, project: Path):
        super().__init__()
        self.project_path = str(project)
        self.nc_file_path = None
        self.nwb_file_path = None
        self.nwb_alignment = None
        self.data_loader = None
        self.labelling_subject = ""
        self.extra_individuals: list[str] = []
        self.ignore_files: list[str] = []

    def refresh_labelling_subject(self):
        self.labelling_subject = self.label_individuals()[0]

    def get_with_default(self, name):
        return getattr(self, name)

    def declared_individuals(self):
        from ethograph.gui.app_state import ObservableAppState

        return ObservableAppState.declared_individuals(self)

    def label_individuals(self):
        from ethograph.gui.app_state import ObservableAppState

        return ObservableAppState.label_individuals(self)


class _Alignment:
    individuals = ["crow1"]


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "study"
    project.mkdir()
    return project


def test_mapping_edits_round_trip_through_mapping_txt(qapp, tmp_path):
    project = _project(tmp_path)
    (project / "mapping.txt").write_text("0 Background\n1 Walk\n2 Peck 1 point\n", encoding="utf-8")
    dialog = LabelMappingDialog(_State(project))

    assert dialog.table.rowCount() == 3
    assert [dialog._row_id(r) for r in range(3)] == [0, 1, 2]
    assert dialog.table.cellWidget(2, 2).value() == 1
    assert dialog.table.cellWidget(2, 3).currentData() == EVENT_TYPE_POINT

    dialog.table.item(1, 1).setText("Fly")  # rename
    dialog._add_row()  # a new label takes the next free id, never an editable one
    dialog._save()

    mappings = load_label_mapping(project / "mapping.txt")
    assert mappings[1]["name"] == "Fly"
    assert mappings[2]["branch"] == 1 and mappings[2]["event_type"] == EVENT_TYPE_POINT
    assert mappings[3]["name"] == "label3"


def test_a_name_with_a_space_is_refused_and_nothing_is_written(qapp, tmp_path):
    """mapping.txt is whitespace-separated: a two-word name would read back as a branch."""
    project = _project(tmp_path)
    dialog = LabelMappingDialog(_State(project))
    dialog._add_row()
    dialog.table.item(0, 1).setText("two words")
    dialog._save()
    assert not (project / "mapping.txt").exists()


def test_the_dialog_greys_out_the_data_and_adds_to_it(qapp, tmp_path):
    """Two lists: the file's names, read-only, and the user's own, editable."""
    state = _State(_project(tmp_path))
    state.nwb_alignment = _Alignment()  # the data names crow1
    dialog = IndividualsDialog(state)

    assert dialog._declared.isReadOnly()
    assert dialog._declared.toPlainText() == "crow1"

    dialog._names.setPlainText("hi")
    assert dialog._resolved() == ["crow1", "hi"]
    assert "crow1, hi" in dialog._effect.text()


def test_individuals_need_no_project_folder(qapp, tmp_path):
    """Naming two crows must not require a project: the list is the user's, not the study's."""
    state = _State(_project(tmp_path))
    state.project_path = None
    dialog = IndividualsDialog(state)
    dialog._names.setPlainText("crow1\ncrow2")
    dialog._save()

    assert state.extra_individuals == ["crow1", "crow2"]
    assert state.label_individuals() == ["crow1", "crow2"]
    assert not (tmp_path / "study" / "project.yaml").exists()


def test_a_name_given_twice_is_refused(qapp, tmp_path):
    state = _State(_project(tmp_path))
    dialog = IndividualsDialog(state)
    dialog._names.setPlainText("crow1\ncrow1")
    dialog._save()
    assert state.extra_individuals == []


def test_add_label_takes_the_next_id_not_qts_checked_flag(qapp, tmp_path):
    """``clicked`` carries a bool; wired straight to ``_add_row`` it landed in ``label_id``.

    The new row's id read "False", and every later row collided with it.
    """
    project = _project(tmp_path)
    (project / "mapping.txt").write_text("0 background\n1 walk\n", encoding="utf-8")
    dialog = LabelMappingDialog(_State(project))

    add_btn = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Add label")
    add_btn.click()
    add_btn.click()

    assert [dialog._row_id(r) for r in range(dialog.table.rowCount())] == [0, 1, 2, 3]
