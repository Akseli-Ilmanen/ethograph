"""The label column schema is the one declaration; every list derives from it.

These are contract guards: nothing in the code forces the derived lists to
agree with the schema, so a column added to one and not the other would
otherwise only show up as a malformed file.
"""

from __future__ import annotations

import pandas as pd

from ethograph.labels.intervals import (
    INTERVAL_COLUMNS,
    INTERVAL_DTYPES,
    KIND_LABEL,
    LABEL_SCHEMA,
    empty_intervals,
    schema_column,
    schema_columns,
    write_order,
)
from ethograph.labels.tsv_store import (
    REQUIRED_COLUMNS,
    REQUIRED_NONNULL_COLUMNS,
    TRIAL_META_COLUMNS,
    TRIAL_META_DEFAULTS,
    TSV_COLUMNS,
    load_labels_tsv,
    save_labels_tsv,
)


class TestSchemaIsConsistent:
    def test_names_are_unique(self):
        names = [col.name for col in LABEL_SCHEMA]
        assert len(names) == len(set(names))

    def test_interval_columns_are_exactly_the_label_kind(self):
        """Only the *order* is stated separately; the membership is the schema's."""
        assert set(INTERVAL_COLUMNS) == set(schema_columns(KIND_LABEL))

    def test_every_stored_column_declares_a_dtype(self):
        for name in INTERVAL_COLUMNS:
            assert INTERVAL_DTYPES[name] is not None

    def test_every_interval_column_is_written_to_the_file(self):
        assert set(INTERVAL_COLUMNS) <= set(TSV_COLUMNS)

    def test_every_trial_meta_column_has_a_default(self):
        assert set(TRIAL_META_COLUMNS) == set(TRIAL_META_DEFAULTS)
        for name in TRIAL_META_COLUMNS:
            assert TRIAL_META_DEFAULTS[name] is not None

    def test_required_and_nonnull_columns_are_stored(self):
        assert REQUIRED_COLUMNS <= set(TSV_COLUMNS)
        assert set(REQUIRED_NONNULL_COLUMNS) <= set(TSV_COLUMNS)

    def test_offset_s_is_required_but_may_be_blank(self):
        """A point event legitimately stores no offset."""
        assert "offset_s" in REQUIRED_COLUMNS
        assert "offset_s" not in REQUIRED_NONNULL_COLUMNS

    def test_empty_intervals_matches_the_schema_dtypes(self):
        df = empty_intervals()
        assert list(df.columns) == INTERVAL_COLUMNS
        for name in INTERVAL_COLUMNS:
            assert df[name].dtype == pd.Series(dtype=INTERVAL_DTYPES[name]).dtype

    def test_unknown_column_is_not_declared(self):
        assert schema_column("poscat") is None
        assert schema_column("onset_s") is not None


class TestWriteOrder:
    def test_known_columns_come_in_schema_order(self):
        assert write_order(["labels", "trial", "onset_s"]) == ["trial", "labels", "onset_s"]

    def test_unknown_columns_follow_in_their_own_order(self):
        assert write_order(["pulse_onsets", "labels", "poscat"]) == ["labels", "pulse_onsets", "poscat"]

    def test_every_column_survives(self):
        cols = ["poscat", "trial", "labels", "duration"]
        assert sorted(write_order(cols)) == sorted(cols)


class TestRoundTrip:
    def test_save_then_load_preserves_column_order_and_extras(self, tmp_path):
        df = pd.DataFrame(
            {
                "labels": [1],
                "poscat": ["a"],
                "trial": [1],
                "individual": ["c1"],
                "onset_s": [0.5],
                "offset_s": [1.5],
            }
        )
        path = tmp_path / "labels.tsv"
        save_labels_tsv(path, df)

        back = load_labels_tsv(path)
        assert list(back.columns)[:2] == ["trial", "individual"]
        assert "poscat" in back.columns
        assert back["poscat"].iloc[0] == "a"
