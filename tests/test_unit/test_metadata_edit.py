"""Write-back of edited trial metadata (ethograph.io.metadata_edit)."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from ethograph.io.metadata_edit import (
    TARGET_NWB,
    TARGET_TABULAR,
    MetadataTarget,
    blank_column,
    coerce_value,
    ensure_tabular_target,
    fits_dtype,
    resolve_metadata_target,
    save_metadata_table,
    write_metadata,
    write_trials_metadata,
)
from ethograph.labels.curation import CURATED_COLUMN


@pytest.fixture
def trials_nwb(tmp_path: Path) -> Path:
    import pynwb
    from dateutil.tz import tzlocal
    from pynwb import NWBHDF5IO

    nwbfile = pynwb.NWBFile(
        session_description="test",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )
    nwbfile.add_trial_column(name="trial", description="Trial number")
    nwbfile.add_trial_column(name="condition", description="condition")
    nwbfile.add_trial_column(name="score", description="score")
    for i in range(3):
        nwbfile.add_trial(
            start_time=float(i),
            stop_time=float(i + 1),
            trial=i + 1,
            condition="ctrl",
            score=float(i),
        )
    path = tmp_path / "alignment.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    return path


def _read_trials(path: Path) -> pd.DataFrame:
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r") as io:
        return io.read().trials.to_dataframe()


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def test_explicit_tabular_metadata_path_wins(tmp_path):
    tsv = tmp_path / "session_metadata.tsv"
    target = resolve_metadata_target(tmp_path / "session.nc", metadata_path=tsv)
    assert target == MetadataTarget(tsv, TARGET_TABULAR)


def test_nwb_source_writes_its_own_trials_table(tmp_path):
    nwb = tmp_path / "session.nwb"
    nwb.write_bytes(b"")
    target = resolve_metadata_target(nwb)
    assert target.kind == TARGET_NWB
    assert target.path == nwb


def test_alignment_sidecar_beats_a_new_tsv(tmp_path, trials_nwb):
    """The alignment NWB outranks a sidecar TSV on load, so it must be written."""
    target = resolve_metadata_target(tmp_path / "session.nc", alignment_path=trials_nwb)
    assert target == MetadataTarget(trials_nwb, TARGET_NWB)


def test_falls_back_to_sidecar_tsv(tmp_path):
    target = resolve_metadata_target(tmp_path / "session.nc")
    assert target.kind == TARGET_TABULAR
    assert target.path == tmp_path / "session_metadata.tsv"


def test_pynapple_folder_falls_back_to_sidecar_inside_it(tmp_path):
    folder = tmp_path / "session"
    folder.mkdir()
    target = resolve_metadata_target(folder, metadata_path=folder)
    assert target == MetadataTarget(folder / "session_metadata.tsv", TARGET_TABULAR)


def test_no_source_has_no_target():
    assert resolve_metadata_target(None) is None


# ---------------------------------------------------------------------------
# Value typing
# ---------------------------------------------------------------------------


def test_value_is_typed_to_match_its_column():
    ints = pd.Series([1, 2, 3])
    floats = pd.Series([1.5, 2.5])
    words = pd.Series(["hit", "miss"])

    assert coerce_value("4", ints) == 4
    assert isinstance(coerce_value("4", ints), int)
    assert coerce_value("4", floats) == 4.0
    assert coerce_value("aborted", words) == "aborted"
    # Text in a numeric column stays text — the column widens, it never lies.
    assert coerce_value("n/a", ints) == "n/a"
    # No column yet: read the type off the text.
    assert coerce_value("3", None) == 3
    assert coerce_value("", None) == ""


def test_blank_column_accepts_any_type_later():
    df = pd.DataFrame({"trial": [1, 2]})
    df["scored"] = blank_column(df)

    assert df["scored"].dtype == object
    assert fits_dtype(df["scored"], 3)
    df.loc[0, "scored"] = 3  # would raise on pandas' inferred string dtype
    assert df.loc[0, "scored"] == 3


def test_fits_dtype_guards_a_numeric_column():
    numbers = pd.Series([1, 2, 3])
    assert fits_dtype(numbers, 4)
    assert not fits_dtype(numbers, "n/a")


# ---------------------------------------------------------------------------
# Tabular write-back
# ---------------------------------------------------------------------------


def test_tabular_write_round_trips(tmp_path):
    df = pd.DataFrame({"trial": [1, 2], "outcome": ["hit", "miss"]})
    path = tmp_path / "session_metadata.tsv"
    write_metadata(MetadataTarget(path, TARGET_TABULAR), df)
    assert pd.read_csv(path, sep="\t").equals(df)


def test_tabular_write_leaves_media_filenames_out(tmp_path):
    """Filenames are the alignment's; a copy in the file would go stale."""
    df = pd.DataFrame({"trial": [1], "outcome": ["hit"], "video_cam-1": ["a.mp4"]})
    path = tmp_path / "session_metadata.tsv"
    write_metadata(MetadataTarget(path, TARGET_TABULAR), df)
    assert list(pd.read_csv(path, sep="	").columns) == ["trial", "outcome"]


def test_csv_target_is_written_comma_separated(tmp_path):
    df = pd.DataFrame({"trial": [1], "outcome": ["hit"]})
    path = tmp_path / "meta.csv"
    save_metadata_table(path, df)
    assert path.read_text().splitlines()[0] == "trial,outcome"


# ---------------------------------------------------------------------------
# NWB write-back
# ---------------------------------------------------------------------------


def test_nwb_updates_existing_column(trials_nwb):
    df = pd.DataFrame({"trial": [1, 2, 3], "condition": ["ctrl", "drug", "a-much-longer-value"]})
    written = write_trials_metadata(trials_nwb, df, columns=["condition"])

    assert written == ["condition"]
    assert list(_read_trials(trials_nwb)["condition"]) == ["ctrl", "drug", "a-much-longer-value"]


def test_nwb_appends_an_unknown_column(trials_nwb):
    df = pd.DataFrame({"trial": [1, 2, 3], "scored": ["", "good", "bad"]})
    write_trials_metadata(trials_nwb, df, columns=["scored"])

    trials = _read_trials(trials_nwb)
    assert list(trials["scored"]) == ["", "good", "bad"]
    # Untouched columns survive.
    assert list(trials["condition"]) == ["ctrl", "ctrl", "ctrl"]


def test_nwb_write_joins_on_trial_id_not_row_order(trials_nwb):
    df = pd.DataFrame({"trial": [3, 1, 2], "condition": ["three", "one", "two"]})
    write_trials_metadata(trials_nwb, df, columns=["condition"])
    assert list(_read_trials(trials_nwb)["condition"]) == ["one", "two", "three"]


def test_nwb_write_ignores_structural_columns(trials_nwb):
    df = pd.DataFrame({"trial": [1, 2, 3], "start_time": [9.0, 9.0, 9.0]})
    assert write_trials_metadata(trials_nwb, df, columns=["start_time", "trial"]) == []
    assert list(_read_trials(trials_nwb)["start_time"]) == [0.0, 1.0, 2.0]


def test_nwb_write_rejects_text_in_a_numeric_column(trials_nwb):
    df = pd.DataFrame({"trial": [1, 2, 3], "score": ["n/a", "n/a", "n/a"]})
    with pytest.raises(ValueError, match="numeric"):
        write_trials_metadata(trials_nwb, df, columns=["score"])


def test_open_alignment_must_be_reloaded_around_a_write(trials_nwb):
    """pynwb keeps the file open for reading; HDF5 refuses a second handle."""
    from ethograph.io.nwb_alignment import NWBAlignment

    alignment = NWBAlignment(trials_nwb)
    assert list(alignment.trials_df["condition"]) == ["ctrl", "ctrl", "ctrl"]  # opens + caches

    alignment.reload()
    write_trials_metadata(trials_nwb, pd.DataFrame({"trial": [1, 2, 3], "condition": ["a", "b", "c"]}))
    alignment.reload()

    assert list(alignment.trials_df["condition"]) == ["a", "b", "c"]
    alignment.close()


def test_nwb_write_is_repeatable(trials_nwb):
    """A column added in one pass is editable in the next."""
    df = pd.DataFrame({"trial": [1, 2, 3], "scored": ["a", "b", "c"]})
    write_trials_metadata(trials_nwb, df, columns=["scored"])
    df.loc[0, "scored"] = "z"
    write_trials_metadata(trials_nwb, df, columns=["scored"])
    assert list(_read_trials(trials_nwb)["scored"]) == ["z", "b", "c"]


# ---------------------------------------------------------------------------
# Derived columns (the curation verdict) stay out of NWB
# ---------------------------------------------------------------------------


def test_nwb_write_refuses_the_curated_column(trials_nwb):
    df = pd.DataFrame({"trial": [1, 2, 3], CURATED_COLUMN: ["yes", "yes", "no"], "condition": ["a", "b", "c"]})
    assert write_trials_metadata(trials_nwb, df, columns=[CURATED_COLUMN]) == []
    assert CURATED_COLUMN not in _read_trials(trials_nwb).columns

    # A mixed write still carries the columns that do belong there.
    assert write_trials_metadata(trials_nwb, df, columns=[CURATED_COLUMN, "condition"]) == ["condition"]
    trials = _read_trials(trials_nwb)
    assert CURATED_COLUMN not in trials.columns
    assert list(trials["condition"]) == ["a", "b", "c"]


def test_ensure_tabular_target_copies_an_nwb_table_to_the_sidecar(tmp_path, trials_nwb):
    nc = tmp_path / "session.nc"
    nc.write_bytes(b"")
    df = pd.DataFrame({"trial": [1, 2, 3], "condition": ["a", "b", "c"]})

    target = ensure_tabular_target(nc, df, alignment_path=trials_nwb)

    assert target is not None and target.kind == TARGET_TABULAR
    assert target.path == tmp_path / "session_metadata.tsv"
    assert list(pd.read_csv(target.path, sep="\t")["condition"]) == ["a", "b", "c"]
    # The NWB is left exactly as it was.
    assert list(_read_trials(trials_nwb)["condition"]) == ["ctrl", "ctrl", "ctrl"]


def test_ensure_tabular_target_never_overwrites_an_existing_sidecar(tmp_path, trials_nwb):
    nc = tmp_path / "session.nc"
    nc.write_bytes(b"")
    sidecar = tmp_path / "session_metadata.tsv"
    save_metadata_table(sidecar, pd.DataFrame({"trial": [1], "condition": ["kept"]}))

    target = ensure_tabular_target(nc, pd.DataFrame({"trial": [1], "condition": ["new"]}), alignment_path=trials_nwb)

    assert target.path == sidecar
    assert list(pd.read_csv(sidecar, sep="\t")["condition"]) == ["kept"]


def test_ensure_tabular_target_keeps_an_explicit_tabular_target(tmp_path):
    nc = tmp_path / "session.nc"
    nc.write_bytes(b"")
    explicit = tmp_path / "chosen.tsv"
    save_metadata_table(explicit, pd.DataFrame({"trial": [1]}))

    target = ensure_tabular_target(nc, pd.DataFrame({"trial": [1]}), metadata_path=explicit)

    assert target == MetadataTarget(explicit, TARGET_TABULAR)
    assert not (tmp_path / "session_metadata.tsv").exists()


def test_ensure_tabular_target_writes_a_sidecar_that_does_not_exist_yet(tmp_path):
    """No alignment NWB either — the derived column still needs a file."""
    nc = tmp_path / "session.nc"
    nc.write_bytes(b"")

    target = ensure_tabular_target(nc, pd.DataFrame({"trial": [1, 2]}))

    assert target.path == tmp_path / "session_metadata.tsv"
    assert list(pd.read_csv(target.path, sep="\t")["trial"]) == [1, 2]


def test_ensure_tabular_target_falls_back_to_one_row_per_trial(tmp_path, trials_nwb):
    nc = tmp_path / "session.nc"
    nc.write_bytes(b"")

    target = ensure_tabular_target(nc, None, alignment_path=trials_nwb, trials=["0", "1"])

    assert list(pd.read_csv(target.path, sep="\t")["trial"].astype(str)) == ["0", "1"]
