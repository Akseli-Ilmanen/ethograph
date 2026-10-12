"""File > Reload data from disk: new data, same setup.

The user edits the session's ``.nc`` in a notebook, saves, and reloads. Only
the data layer moves — catalog, loader, combos, panel selections. The media,
the alignment, the labels and the panels are left alone, for the same reason
`load_keypoint_dataset` never re-runs the file load (see
test_keypoint_dataset_load.py). Trial timing is the alignment's, so a file
whose trials changed is refused rather than half-applied.

The template dataset is shared, so each test edits a copy in tmp_path.
"""

from __future__ import annotations

import shutil

import pytest
from qtpy.QtWidgets import QApplication

import ethograph as eto
from ethograph.io.session_layout import SETTINGS_DIRNAME
from tests.conftest import BIRDPARK_DIR, BIRDPARK_NC, _apply_template, _skip_if_not_downloaded


@pytest.fixture
def copied_birdpark_gui(gui, tmp_path):
    """Birdpark loaded from a private copy of its ``.nc`` + alignment sidecar."""
    _skip_if_not_downloaded("birdpark")
    session = tmp_path / "session"
    session.mkdir()
    nc = session / BIRDPARK_NC.name
    shutil.copy(BIRDPARK_NC, nc)
    shutil.copytree(BIRDPARK_DIR / SETTINGS_DIRNAME, session / SETTINGS_DIRNAME)

    viewer, meta = gui
    _apply_template(meta, "birdpark")  # media folders and label import, as the template does
    # ...but the dataset itself is the copy.
    meta.io_widget.nc_file_path_edit.setText(str(nc))
    meta.app_state.nc_file_path = str(nc)
    meta.data_widget.on_load_clicked()
    QApplication.processEvents()
    assert meta.app_state.ready
    assert meta.data_widget._loaded_source_path == str(nc)
    return viewer, meta, nc


def _items(combo) -> list:
    return [combo.itemData(i) for i in range(combo.count())]


def _add_variable(nc, name: str) -> None:
    """What the user does in a notebook: open, change the trials in place, save."""
    dt = eto.open(str(nc))
    first = dt.trial(dt.trials[0])
    source = next(v for v in first.data_vars if "time" in first[v].dims)
    for trial in dt.trials:
        dt.update_trial(trial, lambda ds: ds.assign({name: ds[source] * 2}))
    dt.save()


def test_a_variable_added_on_disk_reaches_the_catalog_and_the_combo(copied_birdpark_gui):
    _viewer, meta, nc = copied_birdpark_gui
    dw = meta.data_widget
    assert "doubled" not in dw.catalog.feature_choices()

    _add_variable(nc, "doubled")

    assert dw.reload_dataset()
    QApplication.processEvents()
    assert "doubled" in dw.catalog.feature_choices()
    assert "doubled" in _items(dw.combos["features"])
    assert "doubled" in meta.app_state.ds.data_vars
    assert meta.app_state.data_loader.select("doubled", {}) is not None


def test_everything_but_the_data_is_left_alone(copied_birdpark_gui):
    _viewer, meta, nc = copied_birdpark_gui
    app_state = meta.app_state
    alignment_before = app_state.nwb_alignment
    labels_before = app_state._all_labels_df
    panels_before = list(meta.plot_container.line_plots)
    trial_before = app_state.trials_sel
    fps_before = app_state.video_fps
    dt_before = app_state.dt

    _add_variable(nc, "doubled")
    assert meta.data_widget.reload_dataset()

    assert app_state.nwb_alignment is alignment_before
    assert app_state._all_labels_df is labels_before
    assert list(meta.plot_container.line_plots) == panels_before
    assert app_state.trials_sel == trial_before
    assert app_state.video_fps == fps_before
    assert app_state.dt is not dt_before


def test_a_file_whose_trials_changed_is_refused(copied_birdpark_gui, monkeypatch):
    """Trial timing is the alignment's; reload swaps data, never trials."""
    _viewer, meta, nc = copied_birdpark_gui
    dt = eto.open(str(nc))
    first = dt.trial(dt.trials[0]).load()
    dt.close()
    eto.from_datasets([first.assign_attrs(trial=99)]).save(str(nc))

    monkeypatch.setattr("ethograph.gui.widgets_data.notify_dialog", lambda *a, **k: None)
    trials_before = list(meta.app_state.trials)
    dt_before = meta.app_state.dt

    assert not meta.data_widget.reload_dataset()
    assert list(meta.app_state.trials) == trials_before
    assert meta.app_state.dt is dt_before
