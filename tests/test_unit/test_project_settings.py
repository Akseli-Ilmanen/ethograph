"""project.yaml: the study's defaults, overridden by a session's record or dataset, never the reverse."""

from pathlib import Path

import pytest
import yaml

from ethograph.gui.project import ProjectSettings, add_ignore, find_project_dir, load_project_settings


def _write(project: Path, **settings) -> Path:
    project.mkdir(exist_ok=True)
    (project / "project.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    return project


def test_no_project_or_no_file_means_no_defaults(tmp_path: Path):
    assert load_project_settings(None) == ProjectSettings()
    assert load_project_settings(tmp_path) == ProjectSettings()


def test_every_key_reads_back(tmp_path: Path):
    skeleton = {"keypoints": ["beak", "head"], "connections": [{"start": "beak", "end": "head"}]}
    project = _write(
        tmp_path / "study",
        individuals=["crow1", "crow2"],
        cameras=["cam-1"],
        mics=["mic-1"],
        rig="CrowBench",
        pose={"source_software": "SLEAP", "skeleton": skeleton},
    )
    s = load_project_settings(project)
    assert s.individuals == ("crow1", "crow2")
    assert s.cameras == ("cam-1",) and s.mics == ("mic-1",)
    assert s.rig == "CrowBench" and s.pose_software == "SLEAP"
    assert s.skeleton == skeleton  # inline, the shape the skeleton editor saves


def test_unknown_keys_and_bad_shapes_are_refused_by_name(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown keys \\['remote_backup'\\]"):
        load_project_settings(_write(tmp_path / "a", remote_backup={"path": "Z:/"}))
    with pytest.raises(ValueError, match="repeats a name"):
        load_project_settings(_write(tmp_path / "b", individuals=["crow1", "crow1"]))
    with pytest.raises(ValueError, match="pose.skeleton"):
        load_project_settings(_write(tmp_path / "c", pose={"skeleton": "config/skeleton.yaml"}))


def test_label_individuals_precedence(app_state, tmp_path: Path):
    """Session record > dataset dim > project default > labels > "default"."""
    project = _write(tmp_path / "study", individuals=["crow1", "crow2"])
    app_state.project_path = str(project)
    app_state.nwb_alignment = None
    app_state.data_loader = None
    assert app_state.label_individuals() == ["crow1", "crow2"]

    class _Alignment:
        individuals = ["only_this_one"]

    app_state.nwb_alignment = _Alignment()
    assert app_state.label_individuals() == ["only_this_one"]


def test_ignore_reads_back_and_add_ignore_appends_once(tmp_path: Path):
    project = _write(tmp_path / "study", individuals=["crow1"], ignore=["Trial_data.nc"])
    assert load_project_settings(project).ignore == ("Trial_data.nc",)
    add_ignore(project, "*_old.nc")
    add_ignore(project, "*_old.nc")
    s = load_project_settings(project)
    assert s.ignore == ("Trial_data.nc", "*_old.nc")
    assert s.individuals == ("crow1",)  # the rest of the file survives the edit
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    add_ignore(fresh, "x.nc")  # creates project.yaml
    assert load_project_settings(fresh).ignore == ("x.nc",)


def test_find_project_dir_walks_up(tmp_path: Path):
    project = _write(tmp_path / "study")
    assert find_project_dir(project / "config" / "deep") == project
    assert find_project_dir(tmp_path) is None
