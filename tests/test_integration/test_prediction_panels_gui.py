"""Imported prediction files each get one panel, stacked above the time series."""

from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication

pytestmark = pytest.mark.usefixtures("gui")


def test_each_prediction_file_gets_one_panel_on_top(moll2025_gui):
    _viewer, meta = moll2025_gui
    pc = meta.plot_container
    labels = meta.labels_widget
    labels.io_widget.pred_load_mode = lambda: "overlay"
    rows = meta.app_state._all_labels_df.copy()
    time_series = [d for d in pc._open_docks() if not d.isFloating()]

    labels._finish_predictions_import(rows, None, Path("run_a_predictions.tsv"))
    labels._finish_predictions_import(rows, None, Path("run_b_predictions.tsv"))
    labels._finish_predictions_import(rows, None, Path("run_a_predictions.tsv"))
    QApplication.processEvents()

    panels = pc.prediction_panels()
    assert [p.prediction_path.name for p in panels] == ["run_a_predictions.tsv", "run_b_predictions.tsv"]
    assert len(meta.app_state.prediction_sets) == 2

    meta._create_panel_for_source("predictions", "run_b_predictions.tsv", "Prediction timeline")
    assert len(pc.prediction_panels()) == 2, "a file already shown got a second panel"

    pc.window().resize(1200, 900)
    pc.window().show()
    QApplication.processEvents()

    def top(dock):
        return dock.mapTo(pc._dock_host, dock.rect().topLeft()).y()

    prediction_tops = [top(pc._dyn_docks[p]) for p in panels]
    assert prediction_tops == sorted(prediction_tops), "prediction panels are not stacked in import order"
    assert all(max(prediction_tops) < top(d) for d in time_series), "predictions are not above the plots"
