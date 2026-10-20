"""What a labels file carries: a prediction column is written only once something set it."""

from __future__ import annotations

import pandas as pd

from ethograph.labels.tsv_store import TRIAL_META_DEFAULTS, load_labels_tsv, save_labels_tsv


def _labels(**meta) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "trial": [1, 2],
            "individual": ["c1", "c1"],
            "labels": [1, 2],
            "onset_s": [0.5, 0.2],
            "offset_s": [1.0, 0.9],
        }
    )
    for col, default in TRIAL_META_DEFAULTS.items():
        df[col] = meta.get(col, default)
    return df


def test_all_default_prediction_columns_are_left_out(tmp_path):
    path = tmp_path / "labels.tsv"
    save_labels_tsv(path, _labels())
    written = pd.read_csv(path, sep="	")
    assert not set(TRIAL_META_DEFAULTS) & set(written.columns)
    # Loading restores the defaults, so the store is unchanged by the omission.
    loaded = load_labels_tsv(path)
    for col, default in TRIAL_META_DEFAULTS.items():
        assert loaded[col].tolist() == [default, default]


def test_a_set_prediction_column_is_written(tmp_path):
    path = tmp_path / "labels.tsv"
    save_labels_tsv(path, _labels(prediction_source=["", "predictions_segment_1"]))
    written = pd.read_csv(path, sep="	")
    assert "prediction_source" in written.columns
    assert "changepoint_corrected" not in written.columns
