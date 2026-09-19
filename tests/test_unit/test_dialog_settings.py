"""The Settings dialogs write the files the rest of the app reads.

Two things nothing else forces to agree: the mapping table's rows and
``mapping.txt``'s columns, and the individuals dialog's mode and
``project.yaml``'s.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy.QtCore import QObject, Signal

pytest.importorskip("qtpy")

from ethograph.gui.dialog_settings import IndividualsDialog, LabelMappingDialog  # noqa: E402
from ethograph.gui.project import load_project_settings  # noqa: E402
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

    def refresh_labelling_subject(self):
        self.labelling_subject = self.label_individuals()[0]

    def label_individuals(self):
        from ethograph.gui.app_state import ObservableAppState

        return ObservableAppState.label_individuals(self)


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


def test_individuals_dialog_writes_the_mode_and_the_names(qapp, tmp_path):
    project = _project(tmp_path)
    state = _State(project)
    dialog = IndividualsDialog(state)

    assert dialog._mode() == "inherit"  # the default, and the names are not editable under it
    assert not dialog._names.isEnabled()

    dialog._radios["define"].setChecked(True)
    dialog._names.setPlainText("crow1\ncrow2\n")
    assert dialog._names.isEnabled()
    dialog._save()

    settings = load_project_settings(project)
    assert settings.individuals == ("crow1", "crow2") and settings.individuals_mode == "define"
    assert state.label_individuals() == ["crow1", "crow2"]
