"""Where session files go when files are dropped: kind hierarchy + remembered preference."""

from pathlib import Path

from ethograph.gui.session_folder import KIND_ORDER, propose, ranked_kinds, remember, source_folders


def _buckets(tmp_path: Path) -> dict[str, list[str]]:
    vid = tmp_path / "cams"
    aud = tmp_path / "mics"
    feat = tmp_path / "feat"
    for d in (vid, aud, feat):
        d.mkdir()
    return {
        "video": [str(vid / f"t{i}.mp4") for i in range(3)],
        "audio": [str(aud / f"t{i}.wav") for i in range(5)],
        "session": [str(feat / "session.nc"), str(tmp_path / "kilosort")],  # a folder has no kind
        "pose": [str(vid / "t0.h5")],  # pose beside the videos: one folder, two kinds
    }


def test_source_folders_group_by_folder_and_rank_by_best_kind(tmp_path: Path):
    folders = source_folders(_buckets(tmp_path))
    assert [f.folder.name for f in folders] == ["feat", "cams", "mics"]
    cams = folders[1]
    assert cams.counts == {"video": 3, "pose": 1}
    assert cams.best_kind == "video"
    assert "video ×3" in cams.label() and "pose ×1" in cams.label()


def test_default_proposal_prefers_data_over_media(tmp_path: Path):
    folders = source_folders(_buckets(tmp_path))
    assert propose(folders).folder.name == "feat"


def test_remembered_kind_wins_and_falls_back_when_absent(tmp_path: Path):
    folders = source_folders(_buckets(tmp_path))
    assert propose(folders, preferred=["audio"]).folder.name == "mics"
    assert propose(folders, preferred=["npy"]).folder.name == "feat"  # nothing of that kind: default order


def test_ranked_kinds_promotes_and_ignores_unknown():
    assert ranked_kinds(["audio", "bogus", "audio"])[:2] == ("audio", "nc")
    assert set(ranked_kinds(["pose"])) == set(KIND_ORDER)


def test_remember_moves_choice_to_front():
    assert remember(["audio", "video"], "video") == ["video", "audio"]
    assert remember([], "pose") == ["pose"]


def test_no_folders_proposes_nothing():
    assert propose([]) is None
