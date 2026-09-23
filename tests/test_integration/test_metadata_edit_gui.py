"""Editing trial metadata in the trials table, end to end in the real shell.

Covers the loop the feature exists for: tick the checkbox, double-click the
current trial's cell, type a value, and have the table and the metadata file
both say so — while every other trial stays locked.
"""

from __future__ import annotations

import pandas as pd
import pytest
from qtpy.QtCore import Qt

from ethograph.gui.widget_trials import _CURRENT_ROW_COLOR
from ethograph.io.session_layout import metadata_path


@pytest.fixture
def scored_gui(gui, tmp_path):
    """A shell with a three-trial metadata table backed by a sidecar TSV."""
    shell, meta = gui
    app_state = meta.app_state

    nc_path = tmp_path / "session.nc"
    nc_path.write_bytes(b"")  # only its name is used, to derive the sidecar
    app_state.nc_file_path = str(nc_path)
    app_state.trials = [1, 2, 3]
    app_state.trials_sel = 1
    app_state.metadata_df = pd.DataFrame({"trial": [1, 2, 3], "outcome": ["hit", "miss", "hit"]})
    meta.trials_widget.setup(app_state.metadata_df)
    app_state.ready = True
    # No dataset is loaded here, so the data widget has nothing to re-render on
    # a trial change; only the trials table is under test.
    app_state.trial_changed.disconnect(meta.data_widget.on_trial_changed)

    return shell, meta, metadata_path(tmp_path)


def _cell(trials_widget, trial, column: str):
    item = trials_widget._cell_item(trial, column)
    assert item is not None
    return item


def _outcome_col(trials_widget) -> int:
    return list(trials_widget._metadata_df.columns).index("outcome")


# ---------------------------------------------------------------------------
# Who may be edited
# ---------------------------------------------------------------------------


def test_editing_is_off_until_the_checkbox_is_ticked(scored_gui):
    from qtpy.QtWidgets import QTableWidget

    shell, meta, sidecar = scored_gui
    trials = meta.trials_widget

    assert not trials._edit_checkbox.isChecked()
    assert trials._table.editTriggers() == QTableWidget.NoEditTriggers
    assert not _cell(trials, 1, "outcome").flags() & Qt.ItemIsEditable

    trials._edit_checkbox.setChecked(True)
    assert trials._table.editTriggers() == QTableWidget.DoubleClicked
    assert _cell(trials, 1, "outcome").flags() & Qt.ItemIsEditable


def test_only_the_current_trials_row_is_editable_and_marked(scored_gui):
    shell, meta, sidecar = scored_gui
    trials = meta.trials_widget
    trials._edit_checkbox.setChecked(True)

    assert _cell(trials, 1, "outcome").flags() & Qt.ItemIsEditable
    assert not _cell(trials, 2, "outcome").flags() & Qt.ItemIsEditable
    assert _cell(trials, 1, "outcome").data(Qt.BackgroundRole) == _CURRENT_ROW_COLOR
    assert _cell(trials, 2, "outcome").data(Qt.BackgroundRole) is None

    # The mark and the permission follow the open trial.
    meta.app_state.trials_sel = 2
    meta.app_state.trial_changed.emit()

    assert not _cell(trials, 1, "outcome").flags() & Qt.ItemIsEditable
    assert _cell(trials, 2, "outcome").flags() & Qt.ItemIsEditable
    assert _cell(trials, 2, "outcome").data(Qt.BackgroundRole) == _CURRENT_ROW_COLOR


def test_the_trial_column_is_never_editable(scored_gui):
    """It is the join key — editing it would orphan the row."""
    shell, meta, sidecar = scored_gui
    trials = meta.trials_widget
    trials._edit_checkbox.setChecked(True)

    assert not _cell(trials, 1, "trial").flags() & Qt.ItemIsEditable


# ---------------------------------------------------------------------------
# The edit itself
# ---------------------------------------------------------------------------


def test_committing_a_cell_updates_the_table_and_the_frame(scored_gui):
    shell, meta, sidecar = scored_gui
    trials = meta.trials_widget
    trials._edit_checkbox.setChecked(True)

    # What the delegate does on commit.
    _cell(trials, 1, "outcome").setData(Qt.DisplayRole, "aborted")

    assert list(meta.app_state.metadata_df["outcome"]) == ["aborted", "miss", "hit"]
    assert _cell(trials, 1, "outcome").data(Qt.DisplayRole) == "aborted"


def test_a_new_value_becomes_an_autocomplete_option(scored_gui):
    shell, meta, sidecar = scored_gui
    trials = meta.trials_widget
    trials._edit_checkbox.setChecked(True)
    col = _outcome_col(trials)

    assert trials._column_options(col) == ["hit", "miss"]
    _cell(trials, 1, "outcome").setData(Qt.DisplayRole, "aborted")
    assert trials._column_options(col) == ["aborted", "hit", "miss"]


def test_numeric_looking_values_stay_numeric(scored_gui):
    shell, meta, sidecar = scored_gui
    trials = meta.trials_widget
    trials._edit_checkbox.setChecked(True)

    trials.set_metadata_value(1, "quality", "3")

    assert meta.app_state.metadata_df.loc[0, "quality"] == 3


def test_edit_is_written_to_the_sidecar(scored_gui):
    shell, meta, sidecar = scored_gui
    trials = meta.trials_widget
    trials._edit_checkbox.setChecked(True)

    trials.set_metadata_value(2, "outcome", "aborted")
    trials.flush_metadata()

    assert list(pd.read_csv(sidecar, sep="\t")["outcome"]) == ["hit", "aborted", "hit"]


def test_add_column_appears_in_the_table_and_turns_editing_on(scored_gui, monkeypatch):
    from qtpy.QtWidgets import QInputDialog

    shell, meta, sidecar = scored_gui
    trials = meta.trials_widget
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("usable", True))

    trials._on_add_column()

    assert "usable" in meta.app_state.metadata_df.columns
    assert "usable" in list(trials._metadata_df.columns)
    assert trials._edit_checkbox.isChecked()
    assert _cell(trials, 1, "usable").flags() & Qt.ItemIsEditable


def test_editing_controls_stay_available_without_metadata(gui):
    """With no metadata yet is exactly when the first column is added."""
    shell, meta = gui
    trials = meta.trials_widget
    trials.setup(pd.DataFrame({"trial": [1, 2]}))

    assert not trials._edit_checkbox.isHidden()
    assert not trials._add_column_button.isHidden()
    assert not trials._empty_label.isHidden()
    assert trials._table.isHidden()


# ---------------------------------------------------------------------------
# Row order
# ---------------------------------------------------------------------------


def _visible_trial_order(trials_widget) -> list[int]:
    table = trials_widget._table
    col = list(trials_widget._metadata_df.columns).index("trial")
    return [table.item(r, col).data(Qt.DisplayRole) for r in range(table.rowCount())]


def test_table_opens_ascending_by_trial(gui):
    """Installing the filter header must not flip the table to descending.

    ``setHorizontalHeader`` re-runs ``setSortingEnabled`` internally, and a
    fresh ``QHeaderView`` indicates section 0 *descending* — which used to
    re-sort the rows 12, 11, 10, … right after ``setup`` asked for ascending.
    """
    shell, meta = gui
    trials = meta.trials_widget
    ids = list(range(1, 13))
    trials.setup(pd.DataFrame({"trial": ids, "outcome": ["hit", "miss"] * 6}))

    assert _visible_trial_order(trials) == ids


def test_reloading_metadata_keeps_the_users_sort(gui):
    """A new column rebuilds the header; the chosen sort must survive it."""
    shell, meta = gui
    trials = meta.trials_widget
    df = pd.DataFrame({"trial": [1, 2, 3], "outcome": ["b", "a", "c"]})
    trials.setup(df)

    trials._table.sortByColumn(1, Qt.AscendingOrder)
    assert _visible_trial_order(trials) == [2, 1, 3]

    meta.app_state.metadata_df = df.assign(usable=["", "", ""])
    trials.reload_metadata()

    assert _visible_trial_order(trials) == [2, 1, 3]
