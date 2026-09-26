"""Reference geometries are a library resolved nearest first: session, project, starter project.

The same rule as ``mapping.txt`` and the skeleton library; the Moll 2025
template writes its arena into its own session rather than into the home folder.
"""

from pathlib import Path

import pytest
import yaml

from ethograph.gui.plots_space import geometry_dirs, load_library_geometries
from ethograph.utils.download import write_example_configs
from ethograph.utils.paths import SETTINGS_DIR


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("ETHOGRAPH_HOME", str(folder))
    return folder


def _write(folder: Path, name: str, x: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    geometry = {"references": [{"name": name, "vertices": [[x, 0.0], [x, 1.0]], "edges": [[0, 1]]}]}
    (folder / f"{name}.yaml").write_text(yaml.safe_dump(geometry), encoding="utf-8")


def test_the_nearer_library_shadows_the_farther(home: Path, tmp_path: Path):
    session = tmp_path / "ses-01"
    project = tmp_path / "study"
    (session / "session.nc").parent.mkdir()
    _write(home / "defaults" / "space", "arena", 0.0)
    _write(project / "space", "arena", 1.0)
    _write(project / "space", "perch", 1.0)
    _write(session / SETTINGS_DIR / "space", "arena", 2.0)

    dirs = geometry_dirs(session / "session.nc", project)
    assert dirs == [session / SETTINGS_DIR / "space", project / "space", home / "defaults" / "space"]

    library = load_library_geometries(dirs)
    assert sorted(library) == ["arena", "perch"]
    assert library["arena"][0].vertices[0, 0] == 2.0, "the session's file wins"
    assert library["perch"][0].vertices[0, 0] == 1.0

    assert load_library_geometries(geometry_dirs(None, None))["arena"][0].vertices[0, 0] == 0.0


def test_the_template_writes_its_arena_into_its_own_session(tmp_path: Path):
    write_example_configs("moll2025", tmp_path)

    written = tmp_path / SETTINGS_DIR / "space" / "moll2025.yaml"
    assert written.is_file()
    assert "moll2025" in load_library_geometries(geometry_dirs(tmp_path, None))

    written.write_text("references: []\n", encoding="utf-8")
    write_example_configs("moll2025", tmp_path)
    assert written.read_text(encoding="utf-8") == "references: []\n", "an existing file is never overwritten"
