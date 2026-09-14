"""Clicking a visible label selects it, whatever branch is active.

Selection is what playback (V), the space-plot highlight and the heatmap sort
read, so gating it on the *active* branch made all three silently do nothing on
a label the user could plainly see and had just clicked.  The gate is now the
*shown* branches; only mutation (Ctrl+D delete, Ctrl+E edit) and the class new
labels are drawn with stay scoped to the active branch.
"""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("qtpy")

from ethograph.gui.app_state import ObservableAppState  # noqa: E402
from ethograph.gui.widgets_labels import LabelsWidget  # noqa: E402
from ethograph.labels.intervals import empty_intervals  # noqa: E402
from ethograph.labels.predictions import PredictionSet  # noqa: E402

# Branch 0 holds label 1, branch 1 holds label 2.
MAPPINGS = {
    1: {"name": "hop", "color": (1.0, 0.0, 0.0), "branch": 0},
    2: {"name": "call", "color": (0.0, 1.0, 0.0), "branch": 1},
}


class _State:
    """Just enough app_state, reusing the real branch-set properties."""

    active_label_ids = ObservableAppState.active_label_ids
    editable_label_ids = ObservableAppState.editable_label_ids
    _shown_branches = ObservableAppState._shown_branches

    def __init__(self, active_branch: int, shown: dict[int, bool]):
        self._label_mappings = MAPPINGS
        self._branch_shown = shown
        self._active_branch = active_branch
        self.label_intervals = _intervals()
        self.trials_sel = 0
        self.video = None
        self.prediction_sets: list[PredictionSet] = []

    def set_trial_intervals(self, _trial, df):
        self.label_intervals = df

    def record_label_edit(self, _description, trial=None):
        pass


class _Signal:
    def __init__(self):
        self.emitted: list[tuple[float, float]] = []

    def emit(self, *args):
        self.emitted.append(args)


class _Labels:
    """LabelsWidget's click/selection logic without the Qt widget."""

    _check_labels_click = LabelsWidget._check_labels_click
    _check_predictions_click = LabelsWidget._check_predictions_click
    _prediction_rows = LabelsWidget._prediction_rows
    _is_editable_label = LabelsWidget._is_editable_label
    _adopt_clicked_class = LabelsWidget._adopt_clicked_class
    _refuse_foreign_branch = LabelsWidget._refuse_foreign_branch
    _delete_label = LabelsWidget._delete_label
    _edit_label = LabelsWidget._edit_label

    def __init__(self, state: _State):
        self.app_state = state
        self._mappings = MAPPINGS
        self.plot_container = None
        self.data_widget = None
        self.selected_labels = 1
        self.current_labels = None
        self.current_labels_pos = None
        self.current_labels_is_prediction = False
        self._selected_prediction_path = None
        self.old_labels_pos = None
        self.old_labels = None
        self.ready_for_label_click = False
        self.highlight_spaceplot = _Signal()
        # Every mutation tells the curation panel the trial's verdict may have moved.
        self.curation_panel = type("_Panel", (), {"note_labels_edited": lambda self: None})()

    # Stubs for the surroundings the click path touches.
    def _to_display(self, t_rel):
        return t_rel

    def _current_receiver(self):
        return ""

    def _point_click_tolerance_s(self):
        return 0.05

    def _mark_changes_unsaved(self):
        self.unsaved = True

    def refresh_labels_shapes_layer(self):
        pass

    def _reset_label_clicks(self):
        pass


def _intervals() -> pd.DataFrame:
    rows = [
        {"onset_s": 1.0, "offset_s": 2.0, "labels": 1, "individual": "bird1", "event_type": "state"},
        {"onset_s": 3.0, "offset_s": 4.0, "labels": 2, "individual": "bird1", "event_type": "state"},
    ]
    return pd.concat([empty_intervals(), pd.DataFrame(rows)], ignore_index=True)


@pytest.fixture
def widget():
    """Branch 0 active, both branches shown."""
    return _Labels(_State(active_branch=0, shown={0: True, 1: True}))


def test_a_shown_label_of_another_branch_is_selectable(widget):
    assert widget._check_labels_click(3.5, "bird1") is True
    assert widget.current_labels_pos == 1, "the branch-1 label under the click was not selected"
    assert widget.current_labels == 2


def test_selecting_it_does_not_change_the_drawing_class(widget):
    widget._check_labels_click(3.5, "bird1")
    assert widget.selected_labels == 1, "clicking another branch's label changed what would be drawn"


def test_the_active_branch_still_sets_the_drawing_class(widget):
    widget.selected_labels = 0
    widget._check_labels_click(1.5, "bird1")
    assert widget.current_labels_pos == 0
    assert widget.selected_labels == 1


def test_a_hidden_branch_stays_unclickable():
    w = _Labels(_State(active_branch=0, shown={0: True, 1: False}))
    assert w._check_labels_click(3.5, "bird1") is False
    assert w.current_labels_pos is None


def test_deleting_another_branchs_label_is_refused(widget):
    widget._check_labels_click(3.5, "bird1")
    widget._delete_label()
    assert len(widget.app_state.label_intervals) == 2, "a branch-1 label was deleted while branch 0 was active"
    assert widget.current_labels_pos == 1, "the refused delete dropped the selection"


def test_deleting_the_active_branchs_label_still_works(widget):
    widget._check_labels_click(1.5, "bird1")
    widget._delete_label()
    assert len(widget.app_state.label_intervals) == 1
    assert widget.current_labels_pos is None


def test_editing_another_branchs_label_is_refused(widget):
    widget._check_labels_click(3.5, "bird1")
    widget._edit_label()
    assert widget.old_labels_pos is None, "edit mode was entered for a non-editable branch"
    assert widget.ready_for_label_click is False


# ---------------------------------------------------------------------------
# Predictions: each imported file is its own panel. A click on that panel
# selects among that file's predictions only; a click elsewhere never does;
# selecting one never lets it be deleted or edited.
# ---------------------------------------------------------------------------

RUN_A = Path("run_a_predictions.tsv")
RUN_B = Path("run_b_predictions.tsv")


class _PredictionPanel:
    panel_type = "predictions"

    def __init__(self, path: Path):
        self.prediction_path = path


def _predictions(label: int = 1) -> pd.DataFrame:
    rows = [
        {"trial": 0, "onset_s": 6.0, "offset_s": 7.0, "labels": label, "individual": "bird1", "event_type": "state"}
    ]
    return pd.concat([empty_intervals(), pd.DataFrame(rows)], ignore_index=True)


@pytest.fixture
def with_predictions(widget):
    widget.app_state.prediction_sets = [PredictionSet(RUN_A, _predictions(1)), PredictionSet(RUN_B, _predictions(2))]
    return widget


def test_a_click_on_a_prediction_panel_selects_that_files_prediction(with_predictions):
    assert with_predictions._check_labels_click(6.5, "bird1", _PredictionPanel(RUN_B)) is True
    assert with_predictions.current_labels_is_prediction is True
    assert with_predictions.current_labels == 2, "the click selected another file's prediction"


def test_a_prediction_is_unclickable_from_a_label_panel(with_predictions):
    assert with_predictions._check_labels_click(6.5, "bird1") is False
    assert with_predictions.current_labels_pos is None


def test_a_selected_prediction_cannot_be_deleted_or_edited(with_predictions):
    widget = with_predictions
    widget._check_labels_click(6.5, "bird1", _PredictionPanel(RUN_A))

    widget._delete_label()
    assert widget.current_labels_pos is not None, "a prediction was deleted"

    widget._edit_label()
    assert widget.old_labels_pos is None, "edit mode was entered for a prediction"
    assert widget.ready_for_label_click is False
