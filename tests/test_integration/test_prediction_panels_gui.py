"""Imported prediction files each get one panel, stacked above the time series."""

from pathlib import Path

import numpy as np
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


class _Store:
    """A run folder's store, reduced to the curve it draws."""

    def __init__(self, time, confidence):
        self._curve = (time, confidence)

    def get_confidence_curve(self, trial, dt, individual=None):
        return self._curve


def test_run_folder_sets_carry_their_folder_and_draw_confidence_in_their_panel(moll2025_gui):
    _viewer, meta = moll2025_gui
    pc = meta.plot_container
    labels = meta.labels_widget
    io = labels.io_widget
    io.pred_load_mode = lambda: "overlay"
    rows = meta.app_state._all_labels_df.copy()
    time = np.linspace(0.0, 1.0, 20)
    store = _Store(time, np.full(20, 0.5))

    # Two runs write the same file name — the folder tells them apart.
    labels._finish_predictions_import(rows, store, Path("labels/predictions_run_a/moll_predictions.tsv"))
    labels._finish_predictions_import(rows, None, Path("labels/other/moll_predictions.tsv"))
    QApplication.processEvents()

    names = [meta.app_state.prediction_sets[0].name, meta.app_state.prediction_sets[1].name]
    assert names == ["predictions_run_a/moll_predictions.tsv", "moll_predictions.tsv"]
    assert [io.pred_sets_list.item(i).text() for i in range(io.pred_sets_list.count())] == names
    run_panel, tsv_panel = pc.prediction_panels()
    assert pc._dyn_docks[run_panel].windowTitle().endswith(names[0])

    # The curve is in the run's own panel, and nowhere else.
    assert run_panel._confidence_item.isVisible()
    assert len(run_panel._confidence_item.getData()[0]) == 20
    assert not tsv_panel._confidence_item.isVisible()
    assert meta.data_widget.show_confidence_checkbox.isChecked()
    meta.data_widget.show_confidence_checkbox.setChecked(False)
    assert not run_panel._confidence_item.isVisible()

    # The list's selection is the PDF source; Remove unloads set and panel.
    io.pred_sets_list.setCurrentRow(0)
    assert meta.app_state.pred_store is store
    assert io.pred_confidence_pdf_btn.isEnabled()
    labels._remove_selected_prediction_set()
    QApplication.processEvents()
    assert [s.name for s in meta.app_state.prediction_sets] == ["moll_predictions.tsv"]
    assert [p.prediction_path.name for p in pc.prediction_panels()] == ["moll_predictions.tsv"]
    assert meta.app_state.pred_store is None
    assert not io.pred_confidence_pdf_btn.isEnabled()


def test_prediction_panel_is_a_strip_without_a_time_axis(moll2025_gui):
    _viewer, meta = moll2025_gui
    labels = meta.labels_widget
    labels.io_widget.pred_load_mode = lambda: "overlay"
    labels._finish_predictions_import(meta.app_state._all_labels_df.copy(), None, Path("run_predictions.tsv"))
    QApplication.processEvents()

    (panel,) = meta.plot_container.prediction_panels()
    assert not panel.plot_item.getAxis("bottom").isVisible()
    assert panel.vb.viewRange()[1] == [0.0, 1.0]


def _write_run(folder: Path, rows) -> Path:
    folder.mkdir(parents=True)
    rows.to_csv(folder / "moll_predictions.tsv", sep="	", index=False)
    return folder


def test_several_run_folders_each_become_a_set(moll2025_gui, tmp_path):
    """Picking several folders at once loads each as its own overlay set."""
    _viewer, meta = moll2025_gui
    labels = meta.labels_widget
    labels.io_widget.pred_load_mode = lambda: "overlay"
    rows = meta.app_state._all_labels_df.copy()
    run_a = _write_run(tmp_path / "predictions_a", rows)
    run_b = _write_run(tmp_path / "predictions_b", rows)
    (tmp_path / "empty").mkdir()  # no TSV: reported and skipped, never aborts the others

    labels._import_prediction_folders([str(run_a), str(tmp_path / "empty"), str(run_b)])
    QApplication.processEvents()

    assert [s.path.parent.name for s in meta.app_state.prediction_sets] == ["predictions_a", "predictions_b"]
    assert len(meta.plot_container.prediction_panels()) == 2


def test_several_run_folders_as_labels_import_all_runs(moll2025_gui, tmp_path):
    """ "Import as labels" with several runs concatenates them, so no run replaces another."""
    _viewer, meta = moll2025_gui
    labels = meta.labels_widget
    labels.io_widget.pred_load_mode = lambda: "labels"
    labels.io_widget.pred_merge_checkbox.setChecked(False)
    rows = meta.app_state._all_labels_df.copy()
    run_a = _write_run(tmp_path / "predictions_a", rows.iloc[: len(rows) // 2])
    run_b = _write_run(tmp_path / "predictions_b", rows.iloc[len(rows) // 2 :])

    labels._import_prediction_folders([str(run_a), str(run_b)])

    assert len(meta.app_state._all_labels_df) == len(rows)
    assert meta.app_state.prediction_sets == []


def test_run_imported_as_labels_draws_its_curve_on_the_feature_plot(moll2025_gui):
    """ "Import as labels" opens no panel, so the run's curve goes on the feature plots."""
    _viewer, meta = moll2025_gui
    pc = meta.plot_container
    labels = meta.labels_widget
    labels.io_widget.pred_load_mode = lambda: "labels"
    labels.io_widget.pred_merge_checkbox.setChecked(False)
    rows = meta.app_state._all_labels_df.copy()
    store = _Store(np.linspace(0.0, 1.0, 20), np.full(20, 0.5))

    labels._finish_predictions_import(rows, store, Path("labels/predictions_run_a/moll_predictions.tsv"))
    QApplication.processEvents()

    assert pc.prediction_panels() == []
    assert meta.app_state.labels_pred_store is store
    assert "confidence" in pc.overlay_manager._entries

    # A label timeline opened afterwards carries the same curve.
    ribbon = pc.add_panel("labels")
    pc.labels_redraw_needed.emit()
    QApplication.processEvents()
    assert ribbon._confidence_item.isVisible()
    meta.data_widget.show_confidence_checkbox.setChecked(False)
    assert "confidence" not in pc.overlay_manager._entries
    assert not ribbon._confidence_item.isVisible()
