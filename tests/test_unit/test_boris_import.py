"""A BORIS observation becomes a session: one trial per media file, events in trial time.

Qt-free. The project dict below mirrors BORIS's own JSON layout (``file["1"]``,
``media_info.length/fps``, events as ``[t, subject, code, modifier, comment]``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pynwb
import pytest

from ethograph.io.nwb_alignment import read_individuals
from ethograph.labels.boris import (
    extract_intervals,
    import_boris_project,
    match_pose_files,
    session_folder_name,
    write_mapping,
)
from ethograph.labels.intervals import EVENT_TYPE_POINT, EVENT_TYPE_STATE, load_label_mapping
from ethograph.labels.tsv_store import TSV_COLUMNS, load_labels_tsv

A_LEN, B_LEN = 10.0, 8.0


def _project(videos: list[str], events: list[list], *, fps: dict[str, float] | None = None) -> dict:
    lengths = dict(zip(videos, [A_LEN, B_LEN]))
    return {
        "subjects_conf": {"0": {"key": "1", "name": "Crow1"}},
        "behaviors_conf": {
            "0": {"code": "walk", "type": "State event"},
            "1": {"code": "peck", "type": "Point event"},
        },
        "observations": {
            "obs/1": {
                "type": "MEDIA",
                "file": {"1": videos, "2": []},
                "media_info": {
                    "length": lengths,
                    "fps": fps or {v: 25.0 for v in videos},
                    "hasVideo": {v: True for v in videos},
                },
                "events": events,
            },
            "live": {"type": "LIVE", "events": []},
        },
    }


@pytest.fixture
def media(tmp_path: Path) -> list[str]:
    folder = tmp_path / "videos"
    folder.mkdir()
    paths = [str(folder / "a.mp4"), str(folder / "b.mp4")]
    for p in paths:
        Path(p).touch()
    return paths


def _write(tmp_path: Path, project: dict) -> Path:
    path = tmp_path / "study.boris"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Events -> trial-local labels
# ---------------------------------------------------------------------------


def test_events_split_at_the_file_boundary(media):
    events = [
        [1.0, "Crow1", "walk", "", ""],
        [3.0, "Crow1", "walk", "", ""],  # closes in file a
        [12.0, "Crow1", "walk", "", ""],  # starts in file b: 2 s into it
        [14.5, "Crow1", "walk", "", ""],
        [11.0, "Crow1", "peck", "", ""],  # a point event, 1 s into file b
    ]
    df = extract_intervals(
        _project(media, events)["observations"]["obs/1"], {"walk": 1, "peck": 2}, {"walk": "state", "peck": "point"}
    )

    assert list(df.columns) == TSV_COLUMNS
    by_trial = {t: g for t, g in df.groupby("trial")}
    assert by_trial[1][["onset_s", "offset_s"]].values.tolist() == [[1.0, 3.0]]
    b = by_trial[2].sort_values("onset_s")
    assert b["event_type"].tolist() == [EVENT_TYPE_POINT, EVENT_TYPE_STATE]
    assert b["onset_s"].tolist() == [1.0, 2.0]
    assert np.isnan(b["offset_s"].iloc[0]) and b["offset_s"].iloc[1] == pytest.approx(4.5)
    assert set(df["individual"]) == {"Crow1"}
    assert set(df["labeling_method"]) == {"manual"}


def test_state_spanning_a_boundary_is_clipped_and_an_open_one_closed(media):
    events = [
        [8.0, "Crow1", "walk", "", ""],
        [12.0, "Crow1", "walk", "", ""],  # stop lands in file b -> clipped at a's end
        [15.0, "Crow1", "walk", "", ""],  # never closed -> b's end
    ]
    df = extract_intervals(_project(media, events)["observations"]["obs/1"], {"walk": 1}, {"walk": "state"})

    assert df[["trial", "onset_s", "offset_s"]].values.tolist() == [[1, 8.0, A_LEN], [2, 5.0, B_LEN]]


# ---------------------------------------------------------------------------
# The whole project -> sessions
# ---------------------------------------------------------------------------


def test_import_writes_one_session_per_media_observation(tmp_path: Path, media):
    boris = _write(tmp_path, _project(media, [[1.0, "Crow1", "walk", "", ""], [2.0, "Crow1", "walk", "", ""]]))
    out = tmp_path / "project"

    results = import_boris_project(boris, out)

    assert [r.observation for r in results] == ["obs/1"]  # the LIVE observation is skipped
    r = results[0]
    assert r.session_dir == out / "obs_1"
    assert r.alignment == out / "obs_1" / ".ethograph" / "alignment.nwb"
    assert r.labels == out / "obs_1" / "labels.tsv"
    assert r.missing_media == []
    assert len(load_labels_tsv(r.labels)) == 1

    with pynwb.NWBHDF5IO(str(r.alignment), "r") as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        assert read_individuals(nwb) == ["Crow1"]
    assert trials["start_time"].tolist() == [0.0, A_LEN]
    assert trials["stop_time"].tolist() == [A_LEN, A_LEN + B_LEN]
    assert trials["video_cam-1"].tolist() == ["a.mp4", "b.mp4"]


def test_stale_media_paths_resolve_through_the_video_folder(tmp_path: Path, media):
    stale = [p.replace("videos", "gone") for p in media]
    boris = _write(tmp_path, _project(stale, []))

    unresolved = import_boris_project(boris, tmp_path / "p1")[0]
    resolved = import_boris_project(boris, tmp_path / "p2", video_folder=tmp_path / "videos")[0]

    assert unresolved.missing_media == stale
    assert resolved.missing_media == []
    assert resolved.media_paths == media


def test_mixed_frame_rates_are_refused(tmp_path: Path, media):
    boris = _write(tmp_path, _project(media, [], fps={media[0]: 25.0, media[1]: 30.0}))
    with pytest.raises(ValueError, match="different rates"):
        import_boris_project(boris, tmp_path / "project")


# ---------------------------------------------------------------------------
# One mapping for the whole project
# ---------------------------------------------------------------------------


def test_second_import_keeps_ids_and_a_behaviour_keeps_its_kind(tmp_path: Path):
    mapping = tmp_path / "mapping.txt"

    def conf(*codes: tuple[str, str]) -> dict:
        return {"behaviors_conf": {str(i): {"code": c, "type": t} for i, (c, t) in enumerate(codes)}}

    first = conf(("walk", "State event"), ("peck", "Point event"))
    second = conf(("peck", "State event"), ("eat", "State event"))

    ids_first = write_mapping(mapping, first)
    ids_second = write_mapping(mapping, second)

    assert ids_second["peck"] == ids_first["peck"]
    assert ids_second["eat"] == max(ids_first.values()) + 1
    reread = load_label_mapping(mapping)
    assert reread[ids_first["peck"]]["event_type"] == EVENT_TYPE_POINT  # the first import's kind wins
    assert reread[ids_second["eat"]]["event_type"] == EVENT_TYPE_STATE


# ---------------------------------------------------------------------------
# Small contracts
# ---------------------------------------------------------------------------


def test_session_folder_name_is_filesystem_safe():
    assert session_folder_name("obs/1: morning?") == "obs_1_ morning"
    assert session_folder_name("...") == "observation"


def test_pose_matching_does_not_confuse_cam1_with_cam10(tmp_path: Path):
    for name in ("cam10DLC.csv", "cam1DLC.csv"):
        (tmp_path / name).touch()

    matched = match_pose_files(tmp_path, ["v/cam1.mp4", "v/cam10.mp4", "v/cam2.mp4"])

    assert [Path(m).name if m else m for m in matched] == ["cam1DLC.csv", "cam10DLC.csv", ""]
