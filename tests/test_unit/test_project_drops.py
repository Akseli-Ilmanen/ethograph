"""Drops are recorded in the session folder they made, and a project lists them.

The record and the restore must agree on which fields make a drop loadable
again, and nothing but this contract forces them to.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from ethograph.gui.project import (
    DROP_STATE_FIELDS,
    DropRecord,
    drop_record_path,
    is_foreign_session,
    list_drops,
    record_drop,
    register_session,
    restore_drop,
)


def _state(**overrides) -> SimpleNamespace:
    base = {name: None for name in DROP_STATE_FIELDS}
    base["image_paths"] = []
    base["extra_cameras"] = []
    base["metadata_path"] = "stale.tsv"
    base.update(overrides)
    return SimpleNamespace(**base)


class TestRecord:
    def test_round_trip_restores_every_field(self, tmp_path: Path):
        state = _state(
            nc_file_path=str(tmp_path / ".ethograph" / "alignment.nwb"),
            nwb_file_path=str(tmp_path / ".ethograph" / "alignment.nwb"),
            video_folder="D:/raw",
            image_paths=["D:/raw/still.png"],
            primary_camera="cam-1",
            extra_cameras=["cam-2"],
            source_software="DeepLabCut",
        )
        record = record_drop(tmp_path, ["D:/raw/cam0.mp4", "D:/raw/cam1.mp4"], state, datetime(2026, 9, 6, 21, 47, 12))
        assert drop_record_path(tmp_path) == tmp_path / ".ethograph" / "drop.yaml"
        assert drop_record_path(tmp_path).is_file()

        fresh = _state(video_folder="E:/elsewhere", metadata_path="stale.tsv")
        restore_drop(DropRecord.load(tmp_path), fresh)
        for name in DROP_STATE_FIELDS:
            assert getattr(fresh, name) == getattr(state, name), name
        assert fresh.metadata_path is None
        assert record.title == "2026-09-06 21:47 — cam0.mp4, cam1.mp4"

    def test_a_field_missing_from_an_old_record_restores_as_none(self, tmp_path: Path):
        DropRecord(created="2026-09-06_21-47-12", files=[], state={"nc_file_path": "x.nc"}).save(tmp_path)
        fresh = _state(video_folder="E:/elsewhere")
        restore_drop(DropRecord.load(tmp_path), fresh)
        assert fresh.nc_file_path == "x.nc"
        assert fresh.video_folder is None


class TestForeignSession:
    def test_a_session_without_a_drop_record_is_foreign(self, tmp_path: Path):
        assert not is_foreign_session(tmp_path)  # no .ethograph at all: free
        (tmp_path / ".ethograph").mkdir()
        assert not is_foreign_session(tmp_path)  # settings alone make no session
        (tmp_path / ".ethograph" / "alignment.nwb").touch()
        assert is_foreign_session(tmp_path)  # the wizard's or a notebook's
        record_drop(tmp_path, [], _state())
        assert not is_foreign_session(tmp_path)  # a drop's own: regenerable


class TestListing:
    def test_newest_first_and_unrecorded_folders_skipped(self, tmp_path: Path):
        project = tmp_path / "study"
        project.mkdir()
        old, new, failed = (tmp_path / n for n in ("old", "new", "failed"))
        for d in (old, new, failed):
            d.mkdir()
        record_drop(old, ["a.mp4"], _state())
        record_drop(new, ["b.mp4"], _state())
        for d in (old, new, failed):  # a drop that failed before its record
            register_session(project, d)
        assert [folder for folder, _ in list_drops(project)] == [new.resolve(), old.resolve()]

    def test_re_registering_moves_a_session_to_the_front(self, tmp_path: Path):
        project = tmp_path / "study"
        project.mkdir()
        a, b = tmp_path / "a", tmp_path / "b"
        for d in (a, b):
            d.mkdir()
            record_drop(d, [], _state())
        register_session(project, a)
        register_session(project, b)
        register_session(project, a)
        assert [folder for folder, _ in list_drops(project)] == [a.resolve(), b.resolve()]

    def test_no_project_registry_is_empty(self, tmp_path: Path):
        assert list_drops(tmp_path / "nowhere") == []
