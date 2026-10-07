"""The label source: what label navigation and the review grids read.

The working labels by default; a loaded prediction set when one is picked —
then read-only, in the set's own vocabulary. Nothing may write through it.
"""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("qtpy")

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QWidget  # noqa: E402

from ethograph.gui.app_state import ObservableAppState  # noqa: E402
from ethograph.gui.dialog_label_gridview import (  # noqa: E402
    GridVerdictBar,
    LabelGridView,
    LabelSetupPage,
    build_frame_entries,
)
from ethograph.gui.plots_labelribbon import PredictionPanelPlot  # noqa: E402
from ethograph.gui.widgets_navigation import NavigationWidget  # noqa: E402
from ethograph.labels.feature_events import event_mappings  # noqa: E402
from ethograph.labels.predictions import PredictionSet  # noqa: E402

SESSION_MAPPINGS = {1: {"name": "peck", "color": (1.0, 0.0, 0.0), "branch": 0, "event_type": "state"}}
UNITS = Path("rate >20 (per unit)")


def _rows(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["trial", "onset_s", "offset_s", "labels"])
    return df.assign(
        individual="crow", individual_rec="", event_type="state", confidence=1.0, labeling_method="automated"
    )


@pytest.fixture
def state(qtbot, tmp_path):
    """One working label, and a set of two overlapping unit classes."""
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state._label_mappings = SESSION_MAPPINGS
    state._all_labels_df = _rows([(1, 0.0, 1.0, 1)])
    units = _rows([(1, 0.1, 0.6, 1), (1, 0.3, 0.8, 2), (2, 0.2, 0.4, 2)])
    state.prediction_sets = [PredictionSet(UNITS, units, mappings=event_mappings(["unit_0", "unit_7"]))]
    state.trials = [1, 2]
    yield state
    state.stop_auto_save()


class _Meta:
    def __init__(self, state, nav=None):
        self.app_state = state
        self.labels_widget = type("_Labels", (), {"_mappings": SESSION_MAPPINGS, "curation_panel": None})()
        self.navigation_widget = nav
        self.data_widget = None
        self.io_widget = None


def test_the_source_is_the_working_labels_until_a_set_is_picked(state):
    assert state.label_source().writable
    assert state.label_source().df is state._all_labels_df

    state.label_source_path = str(UNITS)
    source = state.label_source()
    assert not source.writable
    assert [source.mappings[i]["name"] for i in source.label_ids()] == ["unit_0", "unit_7"]
    with pytest.raises(RuntimeError, match="read-only"):
        source.require_writable("curate")

    # The set is removed while it is being read: back to the working labels.
    state.prediction_sets = []
    assert state.label_source().writable


def test_label_navigation_walks_the_picked_source(state, qtbot):
    nav = NavigationWidget(QWidget(), state)
    qtbot.addWidget(nav)
    nav.set_mappings(SESSION_MAPPINGS)
    state.ready = True
    state.navigate_mode = "label"
    assert [nav.label_combo.itemText(i) for i in range(nav.label_combo.count())] == ["1 (peck)"]

    state.label_source_path = str(UNITS)
    assert [nav.label_combo.itemText(i) for i in range(nav.label_combo.count())] == ["1 (unit_0)", "2 (unit_7)"]
    nav.label_combo.setCurrentIndex(1)
    assert [(i["trial"], i["onset_s"]) for i in nav._label_instances] == [(1, 0.3), (2, 0.2)]

    state.label_source_path = None
    assert [nav.label_combo.itemText(i) for i in range(nav.label_combo.count())] == ["1 (peck)"]


def test_the_grids_are_built_from_the_source_and_its_classes_are_picked_on_the_setup_page(state, qtbot):
    state.label_source_path = str(UNITS)
    page = LabelSetupPage(_Meta(state), label_ids=state.label_source().label_ids())
    qtbot.addWidget(page)
    assert page.selected_label_ids() == [1, 2]
    assert page.selected_methods() is None

    page.label_list.item(0).setCheckState(Qt.Unchecked)
    entries = build_frame_entries(page.labels_df(), page.mappings(), page.selected_label_ids(), [None])
    assert {e.name for e in entries} == {"unit_7"}
    assert state._all_labels_df["labels"].tolist() == [1], "the working labels were touched"


class _Nav:
    def __init__(self):
        self.jumps: list[dict] = []

    def jump_to_label_instance(self, inst, **_kwargs):
        self.jumps.append(inst)


def test_a_read_only_grid_navigates_and_never_curates(state, qtbot):
    state.label_source_path = str(UNITS)
    source = state.label_source()
    entries = build_frame_entries(source.df, source.mappings, [1], [None])
    nav = _Nav()
    grid = LabelGridView(_Meta(state, nav), entries, read_only=True)
    qtbot.addWidget(grid)

    assert grid.verdict_bar.isHidden()
    grid._on_tile_clicked(entries[0])
    assert [j["onset_s"] for j in nav.jumps] == [0.1]
    assert not grid.verdict_bar.verdicts.clicked, "a click on a read-only grid left a verdict"

    bar = GridVerdictBar(_Meta(state), entries_fn=lambda: entries, restyle_fn=lambda: None, read_only=True)
    qtbot.addWidget(bar)
    with pytest.raises(RuntimeError, match="read-only"):
        bar.apply_done()


def test_a_set_with_its_own_classes_gets_one_row_per_class(state, qtbot):
    panel = PredictionPanelPlot(state)
    qtbot.addWidget(panel)
    assert panel.lane_label_at(0.9) is None

    assert panel.set_lanes(state.prediction_sets[0].mappings) is True
    assert panel.set_lanes(state.prediction_sets[0].mappings) is False, "unchanged rows were rebuilt"
    # First class on top; the row under the click says which class was meant.
    assert (panel.lane_label_at(0.9), panel.lane_label_at(0.1)) == (1, 2)
