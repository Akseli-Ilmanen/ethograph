"""The skeleton library: one YAML per skeleton, the project's shadowing the user's.

A study with a crow rig and a mouse rig has two skeletons, which is the thing an
inline setting could not hold. Nothing here needs a project folder: without one
the user's own library under ``~/.ethograph/defaults/`` answers.
"""

from pathlib import Path

import pytest
import yaml

from ethograph.skeleton.library import (
    delete_skeleton,
    load_skeletons,
    resolve_skeleton,
    save_skeleton,
)


def _skeleton(*names: str) -> dict:
    return {"keypoints": list(names), "connections": [{"start": names[0], "end": names[-1]}]}


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    """An isolated ``~/.ethograph``, so the user's real library is never read or written."""
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("ETHOGRAPH_HOME", str(folder))
    return folder


def test_a_skeleton_round_trips_without_a_project(home: Path):
    path = save_skeleton("crow", _skeleton("beak", "tail"))
    assert path == home / "defaults" / "config" / "skeleton" / "crow.yaml"
    assert load_skeletons()["crow"]["keypoints"] == ["beak", "tail"]

    assert delete_skeleton("crow") is True
    assert load_skeletons() == {}
    assert delete_skeleton("crow") is False


def test_a_study_can_hold_several_and_the_project_shadows_the_user(home: Path, tmp_path: Path):
    project = tmp_path / "study"
    save_skeleton("crow", _skeleton("beak", "tail"))  # the user's
    save_skeleton("crow", _skeleton("bill", "rump"), project)  # the study's, same name
    save_skeleton("mouse", _skeleton("snout", "base"), project)

    library = load_skeletons(project)
    assert sorted(library) == ["crow", "mouse"]
    assert library["crow"]["keypoints"] == ["bill", "rump"], "the nearer library wins"
    assert load_skeletons()["crow"]["keypoints"] == ["beak", "tail"], "the user's is untouched"


def test_one_skeleton_needs_no_choosing(home: Path):
    assert resolve_skeleton(None) is None
    save_skeleton("crow", _skeleton("beak", "tail"))
    assert resolve_skeleton(None)["keypoints"] == ["beak", "tail"]

    save_skeleton("mouse", _skeleton("snout", "base"))
    assert resolve_skeleton(None) is None, "two skeletons: the user has to say which"
    assert resolve_skeleton("mouse")["keypoints"] == ["snout", "base"]
    assert resolve_skeleton("gone") is None


def test_a_file_that_is_not_a_skeleton_is_skipped_not_raised(home: Path):
    folder = home / "defaults" / "config" / "skeleton"
    folder.mkdir(parents=True)
    (folder / "notes.yaml").write_text("just: a note\n", encoding="utf-8")
    (folder / "broken.yaml").write_text("{[unparsable\n", encoding="utf-8")
    (folder / "crow.yaml").write_text(yaml.safe_dump(_skeleton("beak", "tail")), encoding="utf-8")

    assert sorted(load_skeletons()) == ["crow"]


def test_a_name_is_one_file_name(home: Path):
    for bad in ("", "../escape", "nested/crow"):
        with pytest.raises(ValueError, match="one file name"):
            save_skeleton(bad, _skeleton("beak", "tail"))
