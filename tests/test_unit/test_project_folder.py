"""The project folder holds files, never a settings file.

``project.yaml`` is gone: the individuals and the excluded-file globs are the
user's (``gui_settings.yaml``), the skeleton is one YAML per skeleton under
``config/skeleton/``, and the rest were defaults the GUI remembers by itself.
An old file is folded into the settings that replaced it, once.
"""

from pathlib import Path

import yaml

from ethograph.gui.project import migrate_project_yaml


def _write(project: Path, **settings) -> Path:
    project.mkdir(exist_ok=True)
    (project / "project.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    return project


class _Alignment:
    individuals = ["only_this_one"]


def test_a_bare_drag_and_drop_still_labels_somebody(app_state):
    """No project folder, nothing declared: the selector is never empty."""
    app_state.project_path = None
    app_state.nwb_alignment = None
    app_state.data_loader = None
    assert app_state.label_individuals() == ["default"]

    app_state.nwb_alignment = _Alignment()  # a dataset's own dim needs no settings to be used
    assert app_state.label_individuals() == ["only_this_one"]


def test_your_list_only_adds_to_what_the_data_declares(app_state):
    """No modes, no overrides, and no project folder involved at all."""
    app_state.project_path = None
    app_state.data_loader = None
    app_state.nwb_alignment = _Alignment()

    app_state.extra_individuals = ["mine", "only_this_one"]  # named twice, listed once
    assert app_state.label_individuals() == ["only_this_one", "mine"]

    app_state.nwb_alignment = None  # a video-only session: your list is the whole answer
    assert app_state.label_individuals() == ["mine", "only_this_one"]


def test_ignored_files_is_the_users_own_list(app_state):
    app_state.ignore_files = ["*_old.nc", "Trial_data.nc"]
    assert app_state.ignored_files() == ("*_old.nc", "Trial_data.nc")


def test_an_old_project_yaml_is_folded_in_once(app_state, tmp_path: Path):
    """Nothing reads the file any more, so its lists must not be silently lost."""
    project = _write(tmp_path / "study", individuals=["crow1"], ignore=["Trial_data.nc"], rig="CrowBench")
    app_state.project_path = str(project)
    app_state.extra_individuals = ["mine"]
    app_state.ignore_files = []

    assert sorted(migrate_project_yaml(app_state)) == ["extra_individuals", "ignore_files"]
    assert app_state.extra_individuals == ["mine", "crow1"]
    assert app_state.ignore_files == ["Trial_data.nc"]
    assert (project / "project.yaml").is_file(), "the file is the user's; we read it, we do not delete it"

    assert migrate_project_yaml(app_state) == []  # nothing new the second time
    assert app_state.extra_individuals == ["mine", "crow1"]


def test_migration_survives_a_file_that_is_not_project_settings(app_state, tmp_path: Path):
    """Older docs told people to name a pipeline config project.yaml."""
    project = _write(tmp_path / "study", sessions=["a"], features={})
    app_state.project_path = str(project)
    assert migrate_project_yaml(app_state) == []

    (project / "project.yaml").write_text("just a string", encoding="utf-8")
    assert migrate_project_yaml(app_state) == []
