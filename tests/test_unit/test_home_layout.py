"""The home folder's layout: caches under ``cache/``, study fallbacks under ``defaults/``.

Two things must agree and nothing forces them to: the accessors every module
reads its path from, and the move table that relocates an older home folder.
"""

from pathlib import Path

import pytest

from ethograph.utils.paths import (
    HOME_LAYOUT_MOVES,
    cache_dir,
    defaults_dir,
    find_config,
    logs_dir,
    migrate_home_layout,
    session_id,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("ETHOGRAPH_HOME", str(tmp_path))
    return tmp_path


class TestAccessors:
    def test_layout(self, home: Path):
        assert cache_dir() == home / "cache"
        assert cache_dir("proxies") == home / "cache" / "proxies"
        assert defaults_dir() == home / "defaults"
        assert defaults_dir("mapping.txt") == home / "defaults" / "mapping.txt"
        assert logs_dir() == home / "logs"

    def test_every_module_reads_the_accessors(self, home: Path):
        from ethograph.datasets import DOWNLOAD_BASE  # module constant: bound at import, so checked by name
        from ethograph.gui.pose_fill import cotracker_checkpoint_dir
        from ethograph.io.audio_extract import audio_cache_dir
        from ethograph.labels.onset_model import models_root
        from ethograph.labels.workflow import workflows_root

        assert DOWNLOAD_BASE.parts[-2:] == ("cache", "example_data")
        assert audio_cache_dir() == cache_dir("audio_tracks")
        assert cotracker_checkpoint_dir() == cache_dir("weights") / "cotracker"
        assert models_root() == defaults_dir("runs") / "lightgbm"
        assert workflows_root() == defaults_dir("workflows")

    def test_move_table_targets_are_the_layout(self):
        for new_rel in HOME_LAYOUT_MOVES.values():
            assert new_rel.split("/")[0] in {"cache", "defaults"}

    def test_global_config_fallback_is_defaults(self, home: Path):
        (home / "mapping.txt").write_text("0 Background\n")
        assert find_config("mapping.txt") is None
        defaults_dir().mkdir()
        target = defaults_dir("mapping.txt")
        target.write_text("0 Background\n")
        assert find_config("mapping.txt") == target


class TestMigration:
    def test_moves_old_folders_and_files(self, home: Path):
        (home / "proxies").mkdir()
        (home / "proxies" / "a.mp4").write_bytes(b"x")
        (home / "models" / "cotracker").mkdir(parents=True)
        (home / "models" / "cotracker" / "w.pth").write_bytes(b"w")
        (home / "models" / "pecks").mkdir()
        (home / "models" / "pecks" / "model.joblib").write_bytes(b"m")
        (home / "geometries").mkdir()
        (home / "geometries" / "arena.yaml").write_text("references: []\n")
        (home / "mapping.txt").write_text("0 Background\n")
        (home / "gui_settings.yaml").write_text("{}\n")

        moved = migrate_home_layout()

        assert (home / "cache" / "proxies" / "a.mp4").exists()
        assert (home / "cache" / "weights" / "cotracker" / "w.pth").exists()
        assert (home / "defaults" / "runs" / "lightgbm" / "pecks" / "model.joblib").exists()
        assert not (home / "defaults" / "runs" / "lightgbm" / "cotracker").exists()
        assert (home / "defaults" / "config" / "space" / "arena.yaml").exists()
        assert (home / "defaults" / "mapping.txt").exists()
        assert not (home / "models").exists()
        assert not (home / "proxies").exists()
        assert (home / "gui_settings.yaml").exists()
        assert len(moved) == 5

    def test_never_overwrites_an_existing_destination(self, home: Path):
        (home / "workflows").mkdir()
        (home / "workflows" / "old.yaml").write_text("old")
        (home / "defaults" / "workflows").mkdir(parents=True)
        (home / "defaults" / "workflows" / "new.yaml").write_text("new")

        moved = migrate_home_layout()

        assert moved == []
        assert (home / "workflows" / "old.yaml").read_text() == "old"
        assert not (home / "defaults" / "workflows" / "old.yaml").exists()

    def test_fresh_home_is_a_no_op(self, home: Path):
        assert migrate_home_layout() == []
        assert list(home.iterdir()) == []


class TestSeed:
    """The bundled defaults reach a fresh install, and every shipped config builds."""

    def test_fresh_home_gets_every_bundled_file(self, home: Path):
        from ethograph.utils.download import DEFAULT_MAPPING_PATH
        from ethograph.utils.paths import BUNDLED_DEFAULTS_DIR, seed_defaults

        written = seed_defaults()
        expected = {p.relative_to(BUNDLED_DEFAULTS_DIR) for p in BUNDLED_DEFAULTS_DIR.rglob("*") if p.is_file()}
        expected.discard(Path("README.md"))
        assert {p.relative_to(defaults_dir()) for p in written} == expected
        assert (defaults_dir() / "mapping.txt").read_text() == DEFAULT_MAPPING_PATH.read_text()
        assert (defaults_dir() / "config" / "space" / "moll2025.yaml").is_file()
        assert find_config("mapping.txt") == defaults_dir("mapping.txt")

    def test_an_edited_file_is_kept_and_a_missing_one_returns(self, home: Path):
        from ethograph.utils.paths import seed_defaults

        seed_defaults()
        mapping = defaults_dir("mapping.txt")
        mapping.write_text("0 Background\n1 Mine\n")
        (defaults_dir() / "config" / "segment.yaml").unlink()

        written = seed_defaults()

        assert mapping.read_text() == "0 Background\n1 Mine\n"
        assert [p.name for p in written] == ["segment.yaml"]

    def test_shipped_segment_config_builds(self, home: Path):
        from ethograph.segment.config import load_config
        from ethograph.utils.paths import seed_defaults

        seed_defaults()
        cfg = load_config(defaults_dir("config") / "segment.yaml")
        assert cfg.features.labels.mapping == defaults_dir("mapping.txt")
        assert cfg.root == defaults_dir()

    def test_shipped_spot_config_builds(self, home: Path):
        from ethograph.spot.config import load_config
        from ethograph.utils.paths import seed_defaults

        seed_defaults()
        cfg = load_config(defaults_dir("config") / "spot.yaml")
        assert cfg.labels.classes == [11]
        assert cfg.root == defaults_dir()


class TestMappingResolution:
    """Most specific wins: the session's own copy, then the project's, then the bundled backup."""

    def test_session_beats_project_beats_defaults(self, home: Path, tmp_path: Path):
        project = tmp_path / "study"
        session = tmp_path / "data" / "session_01"
        (session / ".ethograph").mkdir(parents=True)
        defaults_dir().mkdir()
        defaults_dir("mapping.txt").write_text("0 Background\n")

        assert find_config("mapping.txt", session, project_dir=project) == defaults_dir("mapping.txt")

        project.mkdir()
        (project / "mapping.txt").write_text("0 Background\n1 Study\n")
        assert find_config("mapping.txt", session, project_dir=project) == project / "mapping.txt"
        assert find_config("mapping.txt", session) == defaults_dir("mapping.txt")

        (session / ".ethograph" / "mapping.txt").write_text("0 Background\n1 Mine\n")
        assert find_config("mapping.txt", session, project_dir=project) == session / ".ethograph" / "mapping.txt"


def test_session_id_stable_and_distinct(tmp_path):
    a = tmp_path / "sess_a.nc"
    b = tmp_path / "sub" / "sess_a.nc"
    assert session_id(a) == session_id(a)
    assert session_id(a) != session_id(b)
    assert session_id(a).startswith("sess_a-")
