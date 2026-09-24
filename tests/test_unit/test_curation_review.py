"""The Labels tab's curation panel: scope, modes, Ctrl+C, frame-by-frame review."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QWidget

from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.widgets_curation import CurationPanel, drag_label_ids
from ethograph.gui.widgets_navigation import NavigationWidget
from ethograph.labels import onset_curves
from ethograph.labels import review_metrics as rm
from ethograph.labels.curation import CURATED_COLUMN
from ethograph.labels.intervals import (
    HUMAN_CONFIDENCE,
    LABELING_AUTOMATED,
    LABELING_CURATED,
    LABELING_MANUAL,
    add_interval,
    empty_intervals,
)
from ethograph.labels.tsv_store import save_labels_tsv


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeVideo:
    """Minimal display-clock video: 50 fps, trial basis."""

    fps = 50.0

    def time_to_frame(self, t: float, round_nearest: bool = False) -> int:
        return int(round(t * self.fps)) if round_nearest else int(t * self.fps)

    def frame_to_time(self, frame: int) -> float:
        return frame / self.fps

    def seek_to_frame(self, frame: int) -> None:
        self.sought = frame


class _LabelsStub(QWidget):
    def __init__(self, mappings):
        super().__init__()
        self._mappings = mappings
        self.active_branch: int | None = None

    def refresh_labels_shapes_layer(self):
        pass

    def set_active_branch(self, branch_idx: int) -> None:
        self.active_branch = branch_idx


class _TrialsStub:
    def __init__(self, metadata_df):
        self._metadata_df = metadata_df
        self.written: list[tuple[str, dict]] = []
        self.ensured = 0
        #: Every trial the dataset has, filters or no — defaults to the
        #: metadata table's own trials (nothing filtered out).
        self._all_trials = list(metadata_df["trial"])

    def set_column_values(self, column, values):
        self.written.append((column, dict(values)))

    def ensure_tabular_metadata_file(self):
        self.ensured += 1
        return None

    def all_trials(self):
        return list(self._all_trials)


class _Meta:
    def __init__(self, app_state, nav, labels_widget, trials_widget=None):
        self.app_state = app_state
        self.navigation_widget = nav
        self.labels_widget = labels_widget
        self.data_widget = None
        self.io_widget = None
        self.trials_widget = trials_widget


MAPPINGS = {
    4: {"name": "peck", "color": (1.0, 0.0, 0.0), "event_type": "point"},
    6: {"name": "hop", "color": (0.0, 1.0, 0.0), "event_type": "point"},
    8: {"name": "mount", "color": (0.0, 0.0, 1.0), "event_type": "state"},
}


def _labels_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset_s": [1.0, 2.0, 2.5, 0.5],
            "offset_s": [np.nan, 3.0, np.nan, np.nan],
            "labels": [4, 8, 6, 4],
            "individual": ["a", "a", "a", "a"],
            "individual_rec": ["", "", "", ""],
            "event_type": ["point", "state", "point", "point"],
            "confidence": [0.3, 1.0, 0.6, 0.9],
            "labeling_method": [LABELING_AUTOMATED, LABELING_MANUAL, LABELING_AUTOMATED, LABELING_AUTOMATED],
            "trial": ["0", "0", "0", "1"],
        }
    )


def _row(state, trial, labels, onset):
    df = state._all_labels_df
    mask = (df["trial"].astype(str) == str(trial)) & (df["labels"] == labels) & np.isclose(df["onset_s"], onset)
    return df[mask].iloc[0]


@pytest.fixture()
def panel(qapp, tmp_path):
    state = ObservableAppState()
    state._yaml_path = str(tmp_path / "gui_settings.yaml")
    state._all_labels_df = _labels_df()
    state.trials = ["0", "1"]
    state.trials_sel = "0"
    state.label_intervals = state.get_trial_intervals("0")
    state.ready = True
    state.video = _FakeVideo()
    # Most of this module exercises queue-walking mechanics across a mix of
    # automated and manual labels — the default-on "automated only" filter
    # (frame_review_automated_only) is covered on its own in TestAutomatedOnlyFilter.
    state.frame_review_automated_only = False
    nav = NavigationWidget(QWidget(), state)
    labels = _LabelsStub(MAPPINGS)
    trials = _TrialsStub(pd.DataFrame({"trial": ["0", "1"], "genotype": ["wt", "ko"]}))
    widget = CurationPanel(state, labels)
    widget.set_meta(_Meta(state, nav, labels, trials_widget=trials))
    widget._trials_stub = trials
    yield widget
    if widget.session_active:
        widget._stop()
    widget.close()
    nav.close()


def _set_mode(panel, key):
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData(key))


# ---------------------------------------------------------------------------
# Scope + trial-level modes
# ---------------------------------------------------------------------------


class TestScope:
    def test_drag_text_parses_to_ids(self):
        assert drag_label_ids("1,4, 8") == [1, 4, 8]
        assert drag_label_ids("4,4") == [4]
        assert drag_label_ids("x") == []
        assert drag_label_ids("") == []

    def test_dropped_ids_become_the_scope(self, panel):
        assert panel.scope() is None
        panel.scope_area.add_ids([4, 8])
        panel._on_scope_edited()
        assert panel.app_state.curation_label_ids == [4, 8]
        assert panel.scope() == {4, 8}
        panel._scope_all()
        assert panel.scope() is None
        assert panel.app_state.curation_label_ids is None

    def test_scope_follows_the_state(self, panel):
        panel.app_state.curation_label_ids = [6]
        assert panel.scope_area.ids() == [6]

    def test_a_drop_from_one_branch_makes_it_the_editable_one(self, panel):
        """Reviewing a class means editing it, and only the active branch
        is editable — so the branch the dropped classes live in is selected.
        Classes from several branches name no single branch: nothing moves."""
        labels = panel.labels_widget
        labels._mappings = {
            4: {**MAPPINGS[4], "branch": 1},
            6: {**MAPPINGS[6], "branch": 1},
            8: {**MAPPINGS[8], "branch": 2},
        }
        panel.scope_area.add_ids([4, 6])
        panel._on_scope_edited(activates=True, dropped=[4, 6])
        assert labels.active_branch == 1
        labels.active_branch = None
        panel.scope_area.add_ids([8])
        panel._on_scope_edited(activates=True, dropped=[8, 4])
        assert labels.active_branch is None
        panel._on_scope_edited(activates=True, dropped=[8])
        assert labels.active_branch == 2


class TestActive:
    """Nothing is written until someone actually starts curating."""

    def test_a_fresh_panel_is_inactive_and_writes_nothing(self, panel):
        assert panel.app_state.curation_active is False
        assert not panel._metadata_timer.isActive()
        panel.sync_metadata()
        assert panel._trials_stub.written == []
        assert panel._trials_stub.ensured == 0

    def test_dropping_label_classes_into_the_scope_activates(self, panel):
        panel.scope_area.add_ids([4])
        panel._on_scope_edited(activates=True)
        assert panel.app_state.curation_active is True
        assert panel._metadata_timer.isActive()
        assert panel._trials_stub.ensured == 1

    def test_curating_activates(self, panel):
        panel.curate_current_trial()
        assert panel.app_state.curation_active is True
        assert panel._trials_stub.ensured == 1
        panel.curate_labels([])  # already active — the file is ensured once
        assert panel._trials_stub.ensured == 1

    def test_a_fresh_dataset_disarms_curation(self, panel):
        panel.activate("test")
        panel.app_state.ready = False
        assert panel.app_state.curation_active is False
        assert not panel._metadata_timer.isActive()


class TestTrialLevel:
    def test_ctrl_c_curates_the_automated_labels_of_the_trial(self, panel):
        emitted = []
        panel.app_state.curation_changed.connect(lambda: emitted.append(True))
        assert panel.curate_current_trial() == 2  # labels 4 and 6 of trial 0
        df = panel.app_state._all_labels_df
        assert _row(panel.app_state, "0", 4, 1.0)["labeling_method"] == LABELING_CURATED
        assert _row(panel.app_state, "0", 8, 2.0)["labeling_method"] == LABELING_MANUAL
        assert _row(panel.app_state, "1", 4, 0.5)["labeling_method"] == LABELING_AUTOMATED  # other trial
        assert emitted
        assert panel.app_state.changes_saved is False
        assert len(panel.app_state.label_intervals) == len(df[df["trial"] == "0"])

    def test_ctrl_c_respects_the_scope(self, panel):
        panel.app_state.curation_label_ids = [6]
        assert panel.curate_current_trial() == 1
        assert _row(panel.app_state, "0", 4, 1.0)["labeling_method"] == LABELING_AUTOMATED
        assert _row(panel.app_state, "0", 6, 2.5)["labeling_method"] == LABELING_CURATED

    def test_curate_visible_trials_reaches_every_visible_trial(self, panel):
        """The workflow step's manual twin: Ctrl+C over the whole visible set."""
        assert panel.curate_visible_trials() == 3  # 4 and 6 in trial 0, 4 in trial 1
        assert _row(panel.app_state, "0", 4, 1.0)["labeling_method"] == LABELING_CURATED
        assert _row(panel.app_state, "1", 4, 0.5)["labeling_method"] == LABELING_CURATED
        assert _row(panel.app_state, "0", 8, 2.0)["labeling_method"] == LABELING_MANUAL

    def test_confirm_asks_before_curating_what_nobody_looked_at(self, panel, monkeypatch):
        """confirm=True (the bulk-editing dialog's own button) must not silently
        vouch for labels across every trial."""
        asked = []
        monkeypatch.setattr(
            panel, "_confirm_bulk_curate", lambda which, label_ids, total, n: asked.append((which, total, n)) or False
        )
        assert panel.curate_visible_trials(confirm=True) == 0
        assert asked == [("filtered", 3, 2)]
        # Declined: nothing moved.
        assert _row(panel.app_state, "0", 4, 1.0)["labeling_method"] == LABELING_AUTOMATED

    def test_a_workflow_step_does_not_ask(self, panel, monkeypatch):
        """A recorded step is already a deliberate choice."""
        monkeypatch.setattr(panel, "_confirm_bulk_curate", lambda *a: pytest.fail("must not ask"))
        assert panel.curate_visible_trials() == 3

    def test_curating_is_not_an_undoable_label_edit(self, panel):
        """Why the bulk button warns: Ctrl+Z takes back edits, not curations.

        Curation records no undo snapshot — the history is per-trial and
        nothing runs curated → automated — so the confirmation dialog saying
        "this cannot be undone" is a statement about the code, not caution.
        """
        panel.curate_current_trial()
        assert _row(panel.app_state, "0", 4, 1.0)["labeling_method"] == LABELING_CURATED
        assert panel.app_state.undo_label_edit() is None
        assert _row(panel.app_state, "0", 4, 1.0)["labeling_method"] == LABELING_CURATED

    def test_inspect_mode_curates_a_trial_on_arrival(self, panel):
        _set_mode(panel, "inspect")
        QApplication.processEvents()  # the deferred curate of the trial already open
        assert _row(panel.app_state, "0", 4, 1.0)["labeling_method"] == LABELING_CURATED
        panel.nav.navigate_to_trial("1")
        QApplication.processEvents()
        assert _row(panel.app_state, "1", 4, 0.5)["labeling_method"] == LABELING_CURATED

    def test_manual_mode_leaves_a_visited_trial_alone(self, panel):
        assert panel.mode() == "manual"
        panel.nav.navigate_to_trial("1")
        QApplication.processEvents()
        assert _row(panel.app_state, "1", 4, 0.5)["labeling_method"] == LABELING_AUTOMATED

    def test_the_mode_is_remembered_in_the_state(self, panel):
        _set_mode(panel, "frame")
        assert panel.app_state.curation_mode == "frame"
        assert panel.frame_group.isVisibleTo(panel)
        panel.app_state.curation_mode = "manual"
        assert panel.mode() == "manual"
        assert not panel.frame_group.isVisibleTo(panel)

    def test_metadata_sync_writes_the_curated_column_only_when_it_changed(self, panel):
        panel.activate("test")
        panel.sync_metadata()
        assert panel._trials_stub.written == [(CURATED_COLUMN, {"0": "no", "1": "no"})]
        panel._trials_stub._metadata_df[CURATED_COLUMN] = ["no", "no"]
        panel.sync_metadata()
        assert len(panel._trials_stub.written) == 1  # nothing changed
        panel.curate_current_trial()
        panel.sync_metadata()
        assert panel._trials_stub.written[-1] == (CURATED_COLUMN, {"0": "yes", "1": "no"})


class TestDeleteScope:
    """Bulk delete picks a trial scope (single/all/filtered/hidden) and, since
    the bulk-editing dialog moved away, an explicit label-class set — never
    silently the curation scope area, unless the caller asks for that."""

    def test_filtered_deletes_only_the_shown_trials(self, panel):
        # Trial 1 filtered out — only trial 0's labels are in scope.
        panel.app_state.trials = ["0"]
        assert panel.delete_trial_labels("filtered") == 3  # labels 4, 8, 6 of trial 0
        df = panel.app_state._all_labels_df
        assert df["trial"].tolist() == ["1"]

    def test_hidden_deletes_the_filtered_out_trials(self, panel):
        # Trial 1 filtered out of the table, but its labels are what get swept.
        panel.app_state.trials = ["0"]
        assert panel.delete_trial_labels("hidden") == 1  # trial 1's one label
        df = panel.app_state._all_labels_df
        assert df["trial"].unique().tolist() == ["0"]

    def test_hidden_is_empty_when_nothing_is_filtered_out(self, panel):
        assert panel.app_state.trials == ["0", "1"]
        assert panel.delete_trial_labels("hidden") == 0
        assert len(panel.app_state._all_labels_df) == 4  # nothing touched

    def test_single_acts_on_only_the_current_trial(self, panel):
        panel.app_state.trials_sel = "1"
        assert panel.delete_trial_labels("single") == 1  # trial 1's one label
        df = panel.app_state._all_labels_df
        assert df["trial"].unique().tolist() == ["0"]

    def test_all_ignores_the_table_s_filters(self, panel):
        panel.app_state.trials = ["0"]  # trial 1 filtered out
        assert panel.delete_trial_labels("all") == 4  # every label, both trials
        assert panel.app_state._all_labels_df.empty

    def test_deletion_falls_back_to_the_curation_scope_area(self, panel):
        panel.app_state.curation_label_ids = [6]
        panel.app_state.trials = ["0"]
        assert panel.delete_trial_labels("filtered") == 1  # only label 6
        df = panel.app_state._all_labels_df
        assert sorted(df["labels"].tolist()) == [4, 4, 8]

    def test_an_explicit_label_ids_overrides_the_curation_scope_area(self, panel):
        panel.app_state.curation_label_ids = [6]  # the scope area names label 6...
        panel.app_state.trials = ["0"]
        assert panel.delete_trial_labels("filtered", label_ids={4}) == 1  # ...but the caller names 4
        df = panel.app_state._all_labels_df
        # Trial 0's label-4 row is gone; trial 1's (out of scope) and 6/8 survive.
        assert sorted(df["labels"].tolist()) == [4, 6, 8]

    def test_an_explicit_none_means_every_class_regardless_of_the_scope_area(self, panel):
        panel.app_state.curation_label_ids = [6]
        panel.app_state.trials = ["0"]
        assert panel.delete_trial_labels("filtered", label_ids=None) == 3

    def test_deletion_touches_manual_labels_too(self, panel):
        """Unlike curation, deleting is not method-selective."""
        panel.app_state.trials = ["0"]
        panel.delete_trial_labels("filtered")
        assert LABELING_MANUAL not in panel.app_state._all_labels_df["labeling_method"].tolist()

    def test_confirm_asks_before_deleting(self, panel, monkeypatch):
        asked = []
        monkeypatch.setattr(
            panel,
            "_confirm_bulk_delete",
            lambda which, label_ids, total, n: asked.append((which, total, n)) or False,
        )
        panel.app_state.trials = ["0"]
        assert panel.delete_trial_labels("hidden", confirm=True) == 0
        assert asked == [("hidden", 1, 1)]
        assert len(panel.app_state._all_labels_df) == 4  # declined — nothing removed

    def test_a_workflow_step_does_not_ask(self, panel, monkeypatch):
        monkeypatch.setattr(panel, "_confirm_bulk_delete", lambda *a: pytest.fail("must not ask"))
        panel.app_state.trials = ["0"]
        assert panel.delete_trial_labels("hidden") == 1

    def test_deletion_is_undoable_per_trial(self, panel):
        """Unlike curation, deleting is a genuine edit — Ctrl+Z can take it back."""
        panel.app_state.trials = ["0"]
        panel.delete_trial_labels("hidden")
        assert len(panel.app_state._all_labels_df) == 3
        edit = panel.app_state.undo_label_edit()
        assert edit is not None
        assert len(panel.app_state._all_labels_df) == 4
        assert _row(panel.app_state, "1", 4, 0.5)["labeling_method"] == LABELING_AUTOMATED


class TestPurgeScoped:
    def test_purges_short_state_labels_only(self, panel):
        # Add a very short state interval to trial 0 alongside the existing labels.
        rows = add_interval(panel.app_state.get_trial_intervals("0"), 5.0, 5.005, 8, "a")
        panel.app_state.set_trial_intervals("0", rows)
        panel.app_state.label_intervals = rows
        assert panel.purge_trial_labels("filtered", min_duration_s=0.01) == 1
        df = panel.app_state._all_labels_df
        assert not ((df["trial"] == "0") & (df["labels"] == 8) & (df["onset_s"] == 5.0)).any()
        # The original manual state label (1.0 s long) survives.
        assert _row(panel.app_state, "0", 8, 2.0)["labeling_method"] == LABELING_MANUAL

    def test_purge_never_touches_point_events(self, panel):
        assert panel.purge_trial_labels("all", min_duration_s=1000.0) == 1  # the one state label
        df = panel.app_state._all_labels_df
        assert set(df["event_type"]) == {"point"}
        assert len(df) == 3

    def test_purge_is_undoable_per_trial(self, panel):
        panel.purge_trial_labels("all", min_duration_s=1000.0)
        assert len(panel.app_state._all_labels_df) == 3
        assert panel.app_state.undo_label_edit() is not None
        assert len(panel.app_state._all_labels_df) == 4


class TestCorrectOffsetsScoped:
    """``correct_offsets_trial`` looks at *every* row of a subject's sequence,
    points included, so these replace trial 0 with a clean, controlled pair
    of states rather than building on the shared fixture's point events."""

    def _set_two_states(self, panel, gap: float):
        df = add_interval(empty_intervals(), 2.0, 3.0, 8, "a")
        df = add_interval(df, 3.0 + gap, 5.0, 8, "a")
        panel.app_state.set_trial_intervals("0", df)
        panel.app_state.label_intervals = df

    def test_pulls_back_the_offset_across_a_near_zero_gap(self, panel):
        self._set_two_states(panel, gap=0.00003)  # under the 1e-4 s eps
        assert panel.correct_offsets("filtered") == 1
        df = panel.app_state._all_labels_df
        first = df[(df["trial"] == "0") & (df["onset_s"] == 2.0)].iloc[0]
        assert first["offset_s"] == pytest.approx(3.00003 - 1e-4)

    def test_no_gaps_is_a_no_op(self, panel):
        self._set_two_states(panel, gap=1.0)  # well clear of the eps
        assert panel.correct_offsets("filtered") == 0

    def test_confirm_asks_before_correcting(self, panel, monkeypatch):
        self._set_two_states(panel, gap=0.00003)
        asked = []
        monkeypatch.setattr(panel, "_confirm_bulk_correct", lambda which, n: asked.append((which, n)) or False)
        assert panel.correct_offsets("filtered", confirm=True) == 0
        assert asked == [("filtered", 2)]
        # Declined: nothing moved.
        df = panel.app_state._all_labels_df
        first = df[(df["trial"] == "0") & (df["onset_s"] == 2.0)].iloc[0]
        assert first["offset_s"] == 3.0

    def test_a_workflow_step_does_not_ask(self, panel, monkeypatch):
        monkeypatch.setattr(panel, "_confirm_bulk_correct", lambda *a: pytest.fail("must not ask"))
        self._set_two_states(panel, gap=0.00003)
        assert panel.correct_offsets("filtered") == 1

    def test_correction_is_undoable_per_trial(self, panel):
        self._set_two_states(panel, gap=0.00003)
        panel.correct_offsets("filtered")
        assert panel.app_state.undo_label_edit() is not None
        df = panel.app_state._all_labels_df
        first = df[(df["trial"] == "0") & (df["onset_s"] == 2.0)].iloc[0]
        assert first["offset_s"] == 3.0


# ---------------------------------------------------------------------------
# Frame-by-frame review
# ---------------------------------------------------------------------------


class TestFrameReview:
    def test_queue_visits_each_trial_once_in_time_order(self, panel):
        queue = panel.build_queue()
        got = [(t.inst["trial"], t.inst["labels"], t.field) for t in queue]
        assert got == [("0", 4, "point"), ("0", 8, "start"), ("0", 8, "end"), ("0", 6, "point"), ("1", 4, "point")]

    def test_start_installs_the_verdict_keys_and_stop_removes_them(self, panel):
        _set_mode(panel, "frame")
        assert panel.start_review()
        keys = {s.key().toString() for s in panel._session_shortcuts}
        assert {"Return", "Enter", "Backspace", "Del", "B", "N"} <= keys
        panel._stop()
        assert panel._session_shortcuts == []
        assert not panel.session_active

    def test_start_needs_a_video(self, panel):
        panel.app_state.video = None
        assert panel.start_review() is False

    def test_confirm_moves_start_and_sibling_end_follows(self, panel):
        _set_mode(panel, "frame")
        panel.start_review(idx=1)  # trial 0, label 8, START at 2.0
        target = panel.targets[panel.current_index]
        assert target.field == "start"
        panel.app_state.current_frame = panel.app_state.video.time_to_frame(2.2, round_nearest=True)
        panel._confirm()
        row = _row(panel.app_state, "0", 8, 2.2)
        assert row["labeling_method"] == LABELING_MANUAL  # it moved: a hand placed it
        assert row["confidence"] == HUMAN_CONFIDENCE
        QApplication.processEvents()
        # The END target shares the instance, so it looks the row up at 2.2.
        end = panel.targets[2]
        assert end.inst["onset_s"] == 2.2 and end.inst is target.inst

    def test_info_label_shows_confidence(self, panel):
        _set_mode(panel, "frame")
        panel.start_review(idx=0)  # trial 0, label 4, point at 1.0 (automated, conf 0.3)
        assert "conf 0.30" in panel.info_label.text()
        panel.start_review(idx=1)  # trial 0, label 8 START (manual, conf 1.0)
        assert "conf 1.00" in panel.info_label.text()

    def test_confirm_in_place_curates_an_automated_label(self, panel):
        _set_mode(panel, "frame")
        panel.start_review(idx=0)  # trial 0, label 4, point at 1.0 (automated, conf 0.3)
        panel.app_state.current_frame = panel.app_state.video.time_to_frame(1.0, round_nearest=True)
        panel._confirm()
        row = _row(panel.app_state, "0", 4, 1.0)
        assert row["labeling_method"] == LABELING_CURATED
        assert row["confidence"] == 0.3  # the model's score stays meaningful

    def test_confirm_refuses_start_past_end(self, panel):
        _set_mode(panel, "frame")
        panel.start_review(idx=1)
        panel.app_state.current_frame = panel.app_state.video.time_to_frame(3.5, round_nearest=True)
        panel._confirm()
        assert _row(panel.app_state, "0", 8, 2.0)["onset_s"] == 2.0
        assert panel.current_index == 1

    def test_next_curates_the_boundary_it_leaves_when_ticked(self, panel):
        _set_mode(panel, "frame")
        panel.start_review(idx=0)
        panel.next_curates_cb.setChecked(True)
        panel._next()
        assert _row(panel.app_state, "0", 4, 1.0)["labeling_method"] == LABELING_CURATED
        assert panel.current_index == 1
        panel.next_curates_cb.setChecked(False)
        panel._next()  # leaves START of label 8 (manual) — nothing to curate
        panel._next()  # leaves END
        assert panel.current_index == 3
        panel._next()  # leaves label 6 unticked → still automated
        assert _row(panel.app_state, "0", 6, 2.5)["labeling_method"] == LABELING_AUTOMATED
        assert panel.app_state.curation_next_curates is False

    def test_back_walks_the_queue_backwards(self, panel):
        _set_mode(panel, "frame")
        panel.start_review(idx=2)
        panel._back()
        assert panel.current_index == 1
        panel._back()
        panel._back()
        assert panel.current_index == 0

    def test_next_past_the_end_stops_the_session(self, panel):
        _set_mode(panel, "frame")
        panel.start_review(idx=4)
        panel._next()
        assert not panel.session_active

    def test_delete_removes_the_row_and_skips_its_other_boundary(self, panel):
        _set_mode(panel, "frame")
        panel.start_review(idx=1)  # START of label 8
        panel._delete_current()
        df = panel.app_state._all_labels_df
        assert not ((df["trial"] == "0") & (df["labels"] == 8)).any()
        # Skipped the END target of the deleted label; now on label 6.
        assert panel.targets[panel.current_index].inst["labels"] == 6

    def test_start_review_at_enters_frame_mode_at_that_label(self, panel):
        assert panel.mode() == "manual"
        inst = {
            "trial": "0",
            "labels": 6,
            "onset_s": 2.5,
            "offset_s": float("nan"),
            "individual": "a",
            "individual_rec": "",
            "event_type": "point",
        }
        assert panel.start_review_at(inst, "point")
        assert panel.mode() == "frame"
        assert panel.session_active
        assert panel.targets[panel.current_index].inst["labels"] == 6

    def test_start_review_at_a_label_outside_the_scope_reviews_just_it(self, panel):
        panel.app_state.curation_label_ids = [4]
        inst = {
            "trial": "0",
            "labels": 6,
            "onset_s": 2.5,
            "offset_s": float("nan"),
            "individual": "a",
            "individual_rec": "",
            "event_type": "point",
        }
        panel.start_review_at(inst, "point")
        assert [t.inst["labels"] for t in panel.targets] == [6]

    def test_leaving_frame_mode_stops_the_session(self, panel):
        _set_mode(panel, "frame")
        panel.start_review()
        _set_mode(panel, "manual")
        assert not panel.session_active
        assert panel._session_shortcuts == []

    def test_view_window_is_seed_centred(self, panel):
        panel.app_state.refine_window_s = 0.2
        view = panel._view_rel(2.0)
        assert (view.start_s, view.end_s) == pytest.approx((1.9, 2.1))

    def test_normal_trial_navigation_pulls_the_session_along(self, panel):
        _set_mode(panel, "frame")
        panel.start_review(idx=0)
        panel.nav.navigate_to_trial("1")
        QApplication.processEvents()
        assert panel.targets[panel.current_index].inst["trial"] == "1"

    def test_verdict_keys_are_disabled_while_typing(self, panel, monkeypatch):
        _set_mode(panel, "frame")
        panel.start_review()
        monkeypatch.setattr("ethograph.gui.widgets_curation.typing_in_text_field", lambda: True)
        panel._sync_session_shortcuts()
        assert all(not s.isEnabled() for s in panel._session_shortcuts)
        monkeypatch.setattr("ethograph.gui.widgets_curation.typing_in_text_field", lambda: False)
        panel._sync_session_shortcuts()
        assert all(s.isEnabled() for s in panel._session_shortcuts)

    def test_restart_review_is_a_noop_when_inactive(self, panel):
        panel.restart_review()
        assert not panel.session_active

    def test_restart_review_keeps_position_on_the_same_target(self, panel):
        _set_mode(panel, "frame")
        panel.start_review(idx=3)  # trial 0, label 6, point at 2.5
        before = panel.targets[panel.current_index].inst
        panel.restart_review()
        assert panel.session_active
        after = panel.targets[panel.current_index].inst
        assert (after["trial"], after["labels"]) == (before["trial"], before["labels"])

    def test_restart_review_drops_a_target_a_grid_done_just_curated(self, panel):
        """A Done elsewhere may curate the label under review — restarting
        must rebuild the queue against the new state, not keep the stale one."""
        panel.automated_only_cb.setChecked(True)
        _set_mode(panel, "frame")
        panel.start_review(idx=0)  # trial 0, label 4, point at 1.0 (automated)
        target = panel.targets[panel.current_index]
        panel.curate_labels([target.inst])  # what GridModeBar.apply_done does
        panel.restart_review()
        assert panel.session_active
        assert panel.targets[panel.current_index].inst["labels"] == 6

    def test_restart_review_stops_when_the_queue_empties(self, panel):
        panel.automated_only_cb.setChecked(True)
        panel.app_state.curation_label_ids = [6]
        _set_mode(panel, "frame")
        panel.start_review()
        target = panel.targets[panel.current_index]
        panel.curate_labels([target.inst])
        panel.restart_review()
        assert not panel.session_active


class TestAutomatedOnlyFilter:
    """Frame-by-frame review skips manual/curated boundaries by default."""

    def test_checked_by_default_on_a_fresh_state(self, qapp, tmp_path):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state._all_labels_df = _labels_df()
        state.trials = ["0", "1"]
        state.ready = True
        state.video = _FakeVideo()
        nav = NavigationWidget(QWidget(), state)
        widget = CurationPanel(state, _LabelsStub(MAPPINGS))
        widget.set_meta(_Meta(state, nav, _LabelsStub(MAPPINGS)))
        try:
            assert widget.automated_only_cb.isChecked() is True
            queue = widget.build_queue()
            # Label 8 (manual) is left out; only the automated point events remain.
            got = [(t.inst["trial"], t.inst["labels"], t.field) for t in queue]
            assert got == [("0", 4, "point"), ("0", 6, "point"), ("1", 4, "point")]
        finally:
            widget.close()
            nav.close()

    def test_unticking_reveals_manual_and_curated_boundaries_too(self, panel):
        panel.automated_only_cb.setChecked(True)
        assert panel.app_state.frame_review_automated_only is True
        queue = panel.build_queue()
        assert 8 not in [t.inst["labels"] for t in queue]
        panel.automated_only_cb.setChecked(False)
        assert panel.app_state.frame_review_automated_only is False
        queue = panel.build_queue()
        assert 8 in [t.inst["labels"] for t in queue]


class _ContainerStub:
    """Records what the review asks the plots to draw."""

    def __init__(self):
        self.shown: list[tuple] = []
        self.hidden = 0

    def show_onset_curves(self, time, curves, colors=None):
        self.shown.append((np.asarray(time), dict(curves), dict(colors or {})))
        return len(curves)

    def hide_onset_curves(self):
        self.hidden += 1

    def schedule_labels_redraw(self):
        pass

    def restyle_label(self, *_args):
        return 0


def _write_curves(state, tmp_path, trials=("0",), timestamp="20260824_120000"):
    """A session path to hang a prediction run off, plus curves for two classes."""
    session = tmp_path / "Trial_data.nc"
    state.nc_file_path = str(session)
    time = np.linspace(0.0, 5.0, 51)
    onset_curves.write_curves(
        onset_curves.run_dir(session, timestamp) / onset_curves.CURVES_FILE,
        {t: (time, {4: np.full_like(time, 0.8), 6: np.full_like(time, 0.2)}) for t in trials},
    )
    return time


class TestOnsetCurves:
    """The model's probability curves, drawn under the label being reviewed."""

    def test_scope_decides_which_classes_are_drawn(self, panel, tmp_path):
        _write_curves(panel.app_state, tmp_path)
        container = _ContainerStub()
        panel.set_plot_container(container)
        panel.set_scope([4], reason="test")
        _set_mode(panel, "frame")
        assert panel.start_review()
        time, curves, colors = container.shown[-1]
        assert set(curves) == {4}
        assert colors[4] == "#ff0000"  # the class's own mapping colour
        assert len(time) == 51

    def test_empty_scope_draws_every_class(self, panel, tmp_path):
        _write_curves(panel.app_state, tmp_path)
        container = _ContainerStub()
        panel.set_plot_container(container)
        _set_mode(panel, "frame")
        assert panel.start_review()
        assert set(container.shown[-1][1]) == {4, 6}

    def test_a_trial_without_curves_draws_nothing(self, panel, tmp_path):
        _write_curves(panel.app_state, tmp_path, trials=("9",))
        container = _ContainerStub()
        panel.set_plot_container(container)
        _set_mode(panel, "frame")
        assert panel.start_review()
        assert container.shown == []
        assert container.hidden > 0

    def test_no_sidecar_is_not_an_error(self, panel):
        """Labels placed by hand have no curves — review still runs."""
        container = _ContainerStub()
        panel.set_plot_container(container)
        _set_mode(panel, "frame")
        assert panel.start_review()
        assert container.shown == []

    def test_stopping_takes_the_curves_down(self, panel, tmp_path):
        _write_curves(panel.app_state, tmp_path)
        container = _ContainerStub()
        panel.set_plot_container(container)
        _set_mode(panel, "frame")
        panel.start_review()
        before = container.hidden
        panel._stop()
        assert container.hidden > before

    def test_one_run_is_used_without_asking(self, panel, tmp_path):
        _write_curves(panel.app_state, tmp_path)
        container = _ContainerStub()
        panel.set_plot_container(container)
        _set_mode(panel, "frame")
        assert panel.start_review()
        assert set(container.shown[-1][1]) == {4, 6}

    def test_several_runs_with_none_chosen_show_nothing(self, panel, tmp_path):
        """Review never asks — that popped up on every restart_review(). With
        several runs and no app_state.curve_run_path set, it just shows none."""
        _write_curves(panel.app_state, tmp_path, timestamp="20260824_120000")
        _write_curves(panel.app_state, tmp_path, timestamp="20260825_090000")
        container = _ContainerStub()
        panel.set_plot_container(container)
        _set_mode(panel, "frame")
        assert panel.start_review()
        assert container.shown == []

    def test_curve_run_path_picks_among_several_runs(self, panel, tmp_path):
        """The pick lives on app_state, set once elsewhere (the onset-model
        Predict dialog) — review just reads it."""
        _write_curves(panel.app_state, tmp_path, timestamp="20260824_120000")
        _write_curves(panel.app_state, tmp_path, timestamp="20260825_090000")
        session = panel.app_state.nc_file_path
        chosen = onset_curves.run_dirs(session)[0] / onset_curves.CURVES_FILE
        panel.app_state.curve_run_path = str(chosen)
        container = _ContainerStub()
        panel.set_plot_container(container)
        _set_mode(panel, "frame")
        assert panel.start_review()
        assert set(container.shown[-1][1]) == {4, 6}

    def test_restart_review_does_not_re_resolve_a_dialog(self, panel, tmp_path, monkeypatch):
        """restart_review() calls _begin() again on every label commit — it must
        never pop anything up (the bug this whole mechanism replaced)."""
        _write_curves(panel.app_state, tmp_path, timestamp="20260824_120000")
        _write_curves(panel.app_state, tmp_path, timestamp="20260825_090000")

        def _fail_if_called(*_a, **_k):
            pytest.fail("CurationPanel must never pop up a dialog to pick curves")

        monkeypatch.setattr("ethograph.gui.widgets_curation.QMessageBox", _fail_if_called)
        container = _ContainerStub()
        panel.set_plot_container(container)
        _set_mode(panel, "frame")
        assert panel.start_review()
        panel.restart_review()
        panel.restart_review()


def test_qt_key_names(qapp):
    """Sanity: the key names the shortcut test compares against are Qt's."""
    from qtpy.QtGui import QKeySequence

    assert QKeySequence(Qt.Key_Backspace).toString() == "Backspace"


class TestHardTrialsAndReview:
    """Ctrl+T writes the difficulty column; finishing the curation scores the
    session exactly once and flags what scored badly (labels/review_metrics.py)."""

    def test_toggle_difficulty_writes_the_column_and_arms_curation(self, panel):
        state = panel.app_state
        state.metadata_df = pd.DataFrame({"trial": ["0", "1"]})
        panel.toggle_difficulty()
        assert state.curation_active
        assert panel._trials_stub.written[-1] == (rm.DIFFICULTY_COLUMN, {"0": rm.DIFFICULTY_HARD})
        # The table now says hard (the stub does not write back) → the box follows, and the next press clears.
        state.metadata_df[rm.DIFFICULTY_COLUMN] = ["hard", ""]
        panel._sync_hard_checkbox()
        assert panel.hard_cb.isChecked()
        panel.toggle_difficulty()
        assert panel._trials_stub.written[-1] == (rm.DIFFICULTY_COLUMN, {"0": rm.DIFFICULTY_NORMAL})

    def test_the_review_fires_once_when_the_last_trial_is_curated(self, panel, monkeypatch):
        calls = []
        monkeypatch.setattr(panel, "run_review", lambda: calls.append(1) or {})
        panel.activate("test")
        panel.sync_metadata()
        panel.curate_current_trial()  # trial 0 done, trial 1 still automated
        panel.sync_metadata()
        assert calls == []
        panel.curate_trial_labels("filtered")
        panel.sync_metadata()
        panel.sync_metadata()
        assert calls == [1]
        # A new prediction run re-arms it.
        panel.app_state._all_labels_df.loc[0, "labeling_method"] = LABELING_AUTOMATED
        panel.sync_metadata()
        panel.curate_trial_labels("filtered")
        panel.sync_metadata()
        assert calls == [1, 1]

    def test_a_session_already_curated_when_curation_arms_is_not_scored(self, panel, monkeypatch):
        calls = []
        monkeypatch.setattr(panel, "run_review", lambda: calls.append(1) or {})
        panel.app_state._all_labels_df["labeling_method"] = LABELING_MANUAL
        panel.activate("test")
        panel.sync_metadata()
        assert calls == []

    def test_run_review_writes_scores_and_flags_hard(self, panel, tmp_path):
        state = panel.app_state
        session = tmp_path / "s" / "s.nc"
        session.parent.mkdir()
        session.write_bytes(b"")
        state.nc_file_path = str(session)
        state.metadata_df = pd.DataFrame({"trial": ["0", "1"]})
        folder = onset_curves.run_dir(session, "20260101_000001")
        folder.mkdir(parents=True)
        # Trial 0's run said: 4@1.0 (kept), 6@2.5 (kept), 4@9.0 (deleted by the
        # reviewer); it never predicted the state label 8 the reviewer drew.
        predicted = pd.DataFrame(
            {
                "trial": [0, 0, 0],
                "labels": [4, 6, 4],
                "onset_s": [1.0, 2.5, 9.0],
                "offset_s": [np.nan] * 3,
                "event_type": ["point"] * 3,
                "individual": ["a"] * 3,
                "individual_rec": [""] * 3,
                "confidence": [0.5] * 3,
                "labeling_method": [LABELING_AUTOMATED] * 3,
            }
        )
        save_labels_tsv(folder / f"s{onset_curves.PREDICTIONS_SUFFIX}", predicted)
        onset_curves.write_provenance(folder, model_config={}, inference={"tolerance_s": 0.05})
        reviews = panel.run_review()
        assert set(reviews) == {"0"}
        assert reviews["0"].f1_point == pytest.approx(0.8)
        assert reviews["0"].f1_state == 0.0
        written = dict(panel._trials_stub.written)
        assert written[rm.REVIEW_F1_POINT] == {"0": pytest.approx(0.8)}
        assert written[rm.REVIEW_F1_STATE] == {"0": 0.0}
        assert written[rm.DIFFICULTY_COLUMN] == {"0": rm.DIFFICULTY_HARD}
        assert "1 of 1" in panel.review_label.text()
