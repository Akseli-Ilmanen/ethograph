"""The BORIS dialog hands every file to the converter and leaves the sessions reopenable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("qtpy")

from ethograph.gui.dialog_boris_import import BorisImportDialog  # noqa: E402
from ethograph.gui.project import DropRecord, list_drops  # noqa: E402


def _boris(tmp_path: Path, name: str, video: Path, code: str) -> Path:
    project = {
        "subjects_conf": {},
        "behaviors_conf": {"0": {"code": code, "type": "State event"}},
        "observations": {
            name: {
                "type": "MEDIA",
                "file": {"1": [str(video)]},
                "media_info": {"length": {str(video): 4.0}, "fps": {str(video): 30.0}, "hasVideo": {str(video): True}},
                "events": [[1.0, "", code, "", ""], [2.0, "", code, "", ""]],
            }
        },
    }
    path = tmp_path / f"{name}.boris"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


def test_batch_import_registers_reopenable_sessions_sharing_one_mapping(qtbot, app_state, tmp_path: Path):
    video = tmp_path / "cam.mp4"
    video.touch()
    project = tmp_path / "study"
    project.mkdir()
    app_state.project_path = str(project)

    dlg = BorisImportDialog(app_state)
    qtbot.addWidget(dlg)
    dlg.add_boris_file(_boris(tmp_path, "morning", video, "walk"))
    dlg.add_boris_file(_boris(tmp_path, "evening", video, "peck"))
    dlg._on_import()

    assert [r.session_dir.name for r in dlg.results] == ["morning", "evening"]
    mapping_lines = (project / "mapping.txt").read_text(encoding="utf-8").splitlines()
    assert mapping_lines[:3] == ["0 background 0", "1 walk 0", "2 peck 0"]
    reopenable = {folder.name: record for folder, record in list_drops(project)}
    assert set(reopenable) == {"morning", "evening"}
    record: DropRecord = reopenable["morning"]
    assert record.state["nc_file_path"] == str(project / "morning")
    assert record.state["video_folder"] == str(tmp_path)
