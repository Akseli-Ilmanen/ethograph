"""Labels imported from an annotation file, which names no trial."""

from ethograph.labels.intervals import _rows_to_df


def test_imported_annotation_rows_land_on_the_current_trial(moll2025_gui):
    _viewer, meta = moll2025_gui
    io = meta.io_widget
    state = meta.app_state
    io.labels_format_combo.setCurrentText(".tsv")  # never write into the example session
    other_trials = state._all_labels_df[state._all_labels_df["trial"] != state.trials_sel]
    imported = _rows_to_df([{"onset_s": 0.1, "offset_s": 0.2, "labels": 1, "individual": "ind0"}])

    io._apply_imported_intervals(imported)

    assert list(state.label_intervals["onset_s"]) == [0.1]
    assert set(state.label_intervals["trial"]) == {state.trials_sel}
    kept = state._all_labels_df[state._all_labels_df["trial"] != state.trials_sel]
    assert len(kept) == len(other_trials)
