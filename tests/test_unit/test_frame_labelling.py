"""Labelling at the current frame: the label key itself places the boundary.

With ``labelling_mode == "frame"`` a label key no longer arms a plot click.
A point class lands on the frame on screen at once; a state class starts on
the first press and ends on the second, wherever the user navigated to in
between. Everything after the time is read — undo snapshot, overlap
resolution, the subject pair, the TSV — is the same path a plot click takes.
"""

import pandas as pd
import pytest

pytest.importorskip("qtpy")

from ethograph.gui.app_constants import LABELLING_MODE_FRAME, LABELLING_MODE_PLOTS  # noqa: E402
from ethograph.gui.widgets_labels import LabelsWidget  # noqa: E402
from ethograph.labels.intervals import EVENT_TYPE_POINT, EVENT_TYPE_STATE, empty_intervals  # noqa: E402

FPS = 50.0

MAPPINGS = {
    1: {"name": "peck", "color": (1.0, 0.0, 0.0), "branch": 0, "event_type": EVENT_TYPE_POINT},
    2: {"name": "hop", "color": (0.0, 1.0, 0.0), "branch": 0, "event_type": EVENT_TYPE_STATE},
    3: {"name": "call", "color": (0.0, 0.0, 1.0), "branch": 0, "event_type": EVENT_TYPE_STATE},
}


class _Video:
    def frame_to_time(self, frame: int) -> float:
        return frame / FPS


class _State:
    """The app_state surface the placement path reads (trial basis)."""

    def __init__(self, mode: str):
        self.labelling_mode = mode
        self.video = _Video()
        self.current_frame = 0
        self.trials_sel = 0
        self.label_intervals = empty_intervals()
        self.changes_saved = True
        self.label_drawing_armed = False
        self.editable_label_ids = None
        self.ready = True
        self.edits: list[str] = []

    def get_with_default(self, key):
        return getattr(self, key)

    def from_display(self, t_display, *, strict=False):
        return self.trials_sel, t_display

    def to_display(self, _trial, t_rel):
        return t_rel

    def set_trial_intervals(self, _trial, df):
        self.label_intervals = df

    def record_label_edit(self, description, trial=None):
        self.edits.append(description)

    def label_individuals(self):
        return ["bird"]

    def selected_individual(self):
        return "bird"

    def selected_receiver(self):
        return ""


class _Curation:
    def __init__(self):
        self.edited = 0

    def note_labels_edited(self):
        self.edited += 1


class _Signal:
    def emit(self, *_args):
        pass


class _Labels:
    """LabelsWidget's placement logic without the Qt widget."""

    activate_label = LabelsWidget.activate_label
    can_label = LabelsWidget.can_label
    has_individuals = LabelsWidget.has_individuals
    has_labels = LabelsWidget.has_labels
    frame_labelling = LabelsWidget.frame_labelling
    current_display_time = LabelsWidget.current_display_time
    _place_at_current_frame = LabelsWidget._place_at_current_frame
    _select_label_row = LabelsWidget._select_label_row
    _active_label_is_point = LabelsWidget._active_label_is_point
    _apply_point = LabelsWidget._apply_point
    _apply_label = LabelsWidget._apply_label
    _show_pending_label = LabelsWidget._show_pending_label
    _reset_label_clicks = LabelsWidget._reset_label_clicks
    _post_label_cleanup = LabelsWidget._post_label_cleanup
    _to_display = LabelsWidget._to_display
    _current_individual = LabelsWidget._current_individual
    _current_receiver = LabelsWidget._current_receiver
    _mark_changes_unsaved = LabelsWidget._mark_changes_unsaved
    _edit_label = LabelsWidget._edit_label
    _refuse_foreign_branch = LabelsWidget._refuse_foreign_branch
    _is_editable_label = LabelsWidget._is_editable_label
    ready_for_label_click = LabelsWidget.ready_for_label_click
    labels_TO_KEY = LabelsWidget.labels_TO_KEY
    KEY_TO_labels = LabelsWidget.KEY_TO_labels
    highlight_spaceplot = _Signal()

    def __init__(self, state: _State):
        self.app_state = state
        self._mappings = MAPPINGS
        self.app_state._active_branch = 0
        self._branch_sections = {}
        self.plot_container = None
        self.data_widget = None
        self.changepoints_widget = None
        self.curation_panel = _Curation()
        self.first_click = None
        self.second_click = None
        self.selected_labels = 0
        self.old_labels_pos = None
        self.old_labels = None
        self.current_labels_pos = None
        self.current_labels = None
        self.current_labels_is_prediction = False
        self.ready_for_label_click = False
        self.seeks: list[float] = []

    def _seek_to_frame(self, time_s):
        self.seeks.append(time_s)

    def refresh_labels_shapes_layer(self):
        pass


@pytest.fixture
def frame_mode():
    state = _State(LABELLING_MODE_FRAME)
    return state, _Labels(state)


def _rows(state):
    """(label, onset, offset) per row; a point event has no offset (NaN → None)."""
    df = state.label_intervals
    return [
        (int(r["labels"]), float(r["onset_s"]), None if pd.isna(r["offset_s"]) else float(r["offset_s"]))
        for _, r in df.iterrows()
    ]


def test_point_class_lands_on_the_frame_on_screen(frame_mode):
    state, labels = frame_mode
    state.current_frame = 10

    labels.activate_label(1)

    assert _rows(state) == [(1, 10 / FPS, None)]
    assert labels.ready_for_label_click is False, "the key placed the label; no plot click is armed"
    assert state.edits == ["place point"]
    assert labels.curation_panel.edited == 1


def test_state_class_takes_two_presses_around_navigation(frame_mode):
    state, labels = frame_mode

    state.current_frame = 4
    labels.activate_label(2)
    assert _rows(state) == []
    assert labels.first_click == 4 / FPS, "the first press anchors the start"

    state.current_frame = 20
    labels.activate_label(2)

    assert _rows(state) == [(2, 4 / FPS, 20 / FPS)]
    assert labels.first_click is None and labels.second_click is None
    assert labels.seeks == [], "the playhead stays where the user navigated to"


def test_end_before_start_still_makes_an_interval(frame_mode):
    state, labels = frame_mode
    state.current_frame = 20
    labels.activate_label(2)
    state.current_frame = 4
    labels.activate_label(2)

    assert _rows(state) == [(2, 4 / FPS, 20 / FPS)]


def test_same_frame_twice_keeps_waiting_for_the_end(frame_mode):
    state, labels = frame_mode
    state.current_frame = 7
    labels.activate_label(2)
    labels.activate_label(2)

    assert _rows(state) == []
    assert labels.first_click == 7 / FPS


def test_another_key_abandons_the_half_placed_label(frame_mode):
    state, labels = frame_mode
    state.current_frame = 4
    labels.activate_label(2)

    state.current_frame = 8
    labels.activate_label(3)

    assert _rows(state) == []
    assert labels.selected_labels == 3
    assert labels.first_click == 8 / FPS, "the new class starts at this frame"


def test_point_key_while_a_state_is_pending_places_the_point_only(frame_mode):
    state, labels = frame_mode
    state.current_frame = 4
    labels.activate_label(2)
    state.current_frame = 9
    labels.activate_label(1)

    assert _rows(state) == [(1, 9 / FPS, None)]
    assert labels.first_click is None


def test_plots_mode_only_arms_the_click():
    state = _State(LABELLING_MODE_PLOTS)
    labels = _Labels(state)
    state.current_frame = 10

    labels.activate_label(1)

    assert _rows(state) == []
    assert labels.ready_for_label_click is True
    assert state.label_drawing_armed is True


def test_without_a_video_the_time_marker_is_the_playhead(frame_mode):
    state, labels = frame_mode
    state.video = None

    class _Marker:
        def value(self):
            return 1.25

    class _Plot:
        time_marker = _Marker()

    class _Container:
        def _visible_plots(self):
            yield _Plot()

        def clear_pending_label(self):
            pass

    labels.plot_container = _Container()
    labels.activate_label(1)

    assert _rows(state) == [(1, 1.25, None)]


def test_edit_re_places_with_the_key_not_a_click(frame_mode):
    state, labels = frame_mode
    state.current_frame = 4
    labels.activate_label(2)
    state.current_frame = 10
    labels.activate_label(2)
    assert _rows(state) == [(2, 4 / FPS, 10 / FPS)]

    labels.current_labels_pos = state.label_intervals.index[0]
    labels.current_labels = 2
    labels._edit_label()
    assert labels.ready_for_label_click is False, "a plot click must not move the label in frame mode"

    state.current_frame = 6
    labels.activate_label(2)
    state.current_frame = 30
    labels.activate_label(2)

    assert _rows(state) == [(2, 6 / FPS, 30 / FPS)]
    assert state.edits[-1] == "move label"
