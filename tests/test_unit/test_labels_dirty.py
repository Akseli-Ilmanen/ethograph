"""The close prompt asks the labels file, not just the ``changes_saved`` flag.

Anything label-adjacent clears the flag, so a session that only edited trial
metadata used to be asked to save labels it never touched. ``labels_dirty()``
settles it against what is actually on disk.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ethograph.labels.tsv_store import labels_equal, load_labels_tsv, save_labels_tsv


def _labels(offset: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": [1, 1, 2],
            "individual": ["crow1", "crow1", "crow1"],
            "labels": [1, 2, 1],
            "onset_s": [0.4, 0.9, 0.2],
            "offset_s": [0.8, offset, 0.5],
            "event_type": ["state"] * 3,
            "human_verified": [0, 0, 0],
            "changepoint_corrected": [0, 0, 0],
            "prediction_source": ["", "", ""],
            "n_samples": [0, 0, 0],
        }
    )


# ---------------------------------------------------------------------------
# labels_equal
# ---------------------------------------------------------------------------


def test_computed_columns_and_row_order_are_not_content(tmp_path):
    df = _labels()
    enriched = df.copy()
    enriched["duration"] = enriched["offset_s"] - enriched["onset_s"]
    enriched["onset_global"] = enriched["onset_s"] + 100.0
    shuffled = enriched.iloc[::-1].reset_index(drop=True)

    assert labels_equal(df, shuffled)


def test_a_moved_boundary_is_content():
    assert not labels_equal(_labels(), _labels(offset=1.5))


def test_a_dropped_row_is_content():
    assert not labels_equal(_labels(), _labels().iloc[:2])


def test_round_trip_through_the_tsv_compares_equal(tmp_path):
    path = tmp_path / "session_labels.tsv"
    save_labels_tsv(path, _labels())
    assert labels_equal(_labels(), load_labels_tsv(path))


def test_empty_matches_missing_file(tmp_path):
    assert labels_equal(None, load_labels_tsv(tmp_path / "absent_labels.tsv"))


# ---------------------------------------------------------------------------
# app_state.labels_dirty
# ---------------------------------------------------------------------------


@pytest.fixture
def state_with_saved_labels(app_state, tmp_path):
    """Labels in memory, identical to the ones on disk, flag says otherwise."""
    nc_path = tmp_path / "session.nc"
    nc_path.write_bytes(b"")
    app_state.nc_file_path = str(nc_path)
    app_state._all_labels_df = _labels()
    save_labels_tsv(tmp_path / "labels.tsv", _labels())  # one labels file per session folder
    app_state.changes_saved = False
    return app_state


def test_a_metadata_only_session_is_not_dirty(state_with_saved_labels):
    assert not state_with_saved_labels.labels_dirty()


def test_an_edited_label_is_dirty(state_with_saved_labels):
    state_with_saved_labels._all_labels_df = _labels(offset=1.5)
    assert state_with_saved_labels.labels_dirty()


def test_labels_with_no_file_yet_are_dirty(app_state, tmp_path):
    nc_path = tmp_path / "session.nc"
    nc_path.write_bytes(b"")
    app_state.nc_file_path = str(nc_path)
    app_state._all_labels_df = _labels()
    app_state.changes_saved = False
    assert app_state.labels_dirty()


def test_the_saved_flag_short_circuits(state_with_saved_labels):
    state_with_saved_labels._all_labels_df = _labels(offset=1.5)
    state_with_saved_labels.changes_saved = True
    assert not state_with_saved_labels.labels_dirty()


# ---------------------------------------------------------------------------
# individual_rec (recipient) round-trip
# ---------------------------------------------------------------------------


def test_recipient_survives_a_save_load_round_trip(tmp_path):
    df = _labels()
    df["individual_rec"] = ["crow2", "", "crow2"]
    path = tmp_path / "pairs_labels.tsv"
    save_labels_tsv(path, df)

    loaded = load_labels_tsv(path)
    assert sorted(loaded["individual_rec"]) == ["", "crow2", "crow2"]
    assert labels_equal(df, loaded)


def test_a_file_without_the_column_reads_back_as_solo_labels(tmp_path):
    """Every label written before recipients existed is a solo behaviour."""
    path = tmp_path / "legacy_labels.tsv"
    _labels().drop(columns=["event_type"]).to_csv(path, sep="\t", index=False)

    loaded = load_labels_tsv(path)
    assert (loaded["individual_rec"] == "").all()
