"""A session is a folder: every session file is named by the folder, never by the .nc stem."""

from pathlib import Path

from ethograph.io.session_layout import (
    adopt_legacy_files,
    alignment_path,
    is_session_folder,
    labels_path,
    metadata_path,
    session_dir_of,
)
from ethograph.labels.tsv_store import labels_tsv_path


def test_session_dir_is_the_folder_or_the_files_folder(tmp_path: Path):
    (tmp_path / "session.nc").touch()
    assert session_dir_of(tmp_path) == tmp_path.resolve()
    assert session_dir_of(tmp_path / "session.nc") == tmp_path.resolve()
    assert session_dir_of(tmp_path / "missing.nwb") == tmp_path.resolve()  # a file need not exist


def test_labels_and_metadata_names_do_not_depend_on_the_source_file(tmp_path: Path):
    for source in (tmp_path, tmp_path / "a.nc", tmp_path / "b.nc"):
        assert labels_path(source) == tmp_path.resolve() / "labels.tsv"
        assert metadata_path(source) == tmp_path.resolve() / "metadata.tsv"
    assert labels_tsv_path(tmp_path / "a.nc", "_downsampled_4x").name == "labels_downsampled_4x.tsv"
    assert alignment_path(tmp_path / "a.nc") == tmp_path.resolve() / ".ethograph" / "alignment.nwb"


def test_a_session_folder_has_an_alignment_or_a_root_nwb(tmp_path: Path):
    assert not is_session_folder(tmp_path)
    (tmp_path / "session.nwb").touch()
    assert is_session_folder(tmp_path)
    (tmp_path / "session.nwb").unlink()
    (tmp_path / ".ethograph").mkdir()
    (tmp_path / ".ethograph" / "alignment.nwb").touch()
    assert is_session_folder(tmp_path)


def test_a_lone_legacy_labels_file_is_adopted_once(tmp_path: Path):
    legacy = tmp_path / "Trial_data_labels.tsv"
    legacy.write_text("onset_s\n")
    renamed = adopt_legacy_files(tmp_path / "Trial_data.nc")
    assert renamed == [(legacy, tmp_path.resolve() / "labels.tsv")]
    assert not legacy.exists() and (tmp_path / "labels.tsv").read_text() == "onset_s\n"
    assert adopt_legacy_files(tmp_path) == []  # nothing left to adopt


def test_several_legacy_files_are_left_alone(tmp_path: Path):
    """Two stems in one folder is a folder of sessions, not a session: nothing is guessed."""
    for stem in ("day1", "day2"):
        (tmp_path / f"{stem}_labels.tsv").touch()
    assert adopt_legacy_files(tmp_path) == []
    assert not (tmp_path / "labels.tsv").exists()


def test_an_existing_canonical_file_is_never_replaced(tmp_path: Path):
    (tmp_path / "labels.tsv").write_text("keep\n")
    (tmp_path / "old_labels.tsv").write_text("stale\n")
    assert adopt_legacy_files(tmp_path) == []
    assert (tmp_path / "labels.tsv").read_text() == "keep\n"


def test_an_explicitly_named_file_is_not_adopted(tmp_path: Path):
    (tmp_path / "li_labels.tsv").touch()
    assert adopt_legacy_files(tmp_path, labels=False) == []
    assert (tmp_path / "li_labels.tsv").exists()
