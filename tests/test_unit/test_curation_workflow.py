"""Curation workflows: the stored model, and the runner that replays it.

The model (``labels/workflow.py``) is Qt-free and tested as data; the runner
(``gui/dialog_curation_workflow.py``) is tested against a stub GUI, so nothing
here fits a classifier or decodes a video.
"""

from __future__ import annotations

import pandas as pd
import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication

from ethograph.gui import dialog_curation_workflow as dcw
from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.widget_trials import TrialsWidget
from ethograph.labels import workflow as wf

METADATA = pd.DataFrame(
    {
        "trial": [1, 2, 3, 4],
        "genotype": ["wt", "wt", "ko", "ko"],
        "score": [0.1, 0.9, 0.4, 0.8],
    }
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point ~/.ethograph at a temp dir so tests never touch the real store."""
    monkeypatch.setenv("ETHOGRAPH_HOME", str(tmp_path / ".ethograph"))


# ----------------------------------------------------------------------
# The stored model
# ----------------------------------------------------------------------


class TestStorage:
    def test_round_trip(self):
        workflow = wf.CurationWorkflow(
            name="nightly",
            description="the usual",
            steps=[
                wf.WorkflowStep("filter_trials", {"filters": [{"column": "genotype", "values": ["wt"]}]}),
                wf.WorkflowStep("save_labels"),
            ],
        )
        wf.save_workflow(workflow)
        assert wf.list_workflows() == ["nightly"]

        loaded = wf.load_workflow("nightly")
        assert loaded.description == "the usual"
        assert [s.kind for s in loaded.steps] == ["filter_trials", "save_labels"]
        assert loaded.steps[0].value("filters") == [{"column": "genotype", "values": ["wt"]}]

    def test_delete(self):
        wf.save_workflow(wf.CurationWorkflow(name="gone"))
        wf.delete_workflow("gone")
        assert wf.list_workflows() == []

    def test_rename_leaves_no_orphan(self):
        wf.save_workflow(wf.CurationWorkflow(name="old", description="keep me"))
        wf.rename_workflow("old", "new")
        assert wf.list_workflows() == ["new"]
        assert wf.load_workflow("new").description == "keep me"

    def test_an_open_project_stores_its_own_workflows(self, tmp_path):
        project = tmp_path / "study"
        wf.save_workflow(wf.CurationWorkflow(name="home"))
        wf.save_workflow(wf.CurationWorkflow(name="study"), project)
        assert (project / "workflows" / "study.yaml").is_file()
        assert wf.list_workflows(project) == ["study"]
        assert wf.list_workflows() == ["home"]

    def test_the_dialog_lists_the_open_projects_workflows(self, qapp, tmp_path):
        project = tmp_path / "study"
        wf.save_workflow(wf.CurationWorkflow(name="study"), project)
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        state.project_path = str(project)
        trials = TrialsWidget(state)
        trials.setup(METADATA)
        dialog = dcw.CurationWorkflowDialog(_DialogMeta(state, trials))
        try:
            assert dialog.workflow_list.item(0).text() == "study"
            assert dialog.workflow_list.count() == 1
        finally:
            dialog.close()
            trials.deleteLater()

    def test_a_name_becomes_a_usable_file_stem(self):
        assert wf.safe_name("wt / tone review") == "wt _ tone review"
        with pytest.raises(ValueError):
            wf.safe_name("///")


class TestDefaults:
    """A parameter left out of the YAML falls back to the kind's declaration,
    so an older stored workflow keeps running when a kind gains a knob."""

    def test_value_falls_back_to_the_declared_default(self):
        step = wf.WorkflowStep("video_grid", {"columns": 4})
        assert step.value("columns") == 4
        assert step.value("per_page") == wf.STEP_KINDS["video_grid"].spec("per_page").default

    def test_an_unknown_parameter_raises(self):
        with pytest.raises(KeyError):
            wf.WorkflowStep("save_labels").value("nonsense")

    def test_a_mutable_default_is_never_shared(self):
        first = wf.WorkflowStep("scope").value("label_ids")
        first.append(7)
        assert wf.WorkflowStep("scope").value("label_ids") == []


class TestValidate:
    def test_a_complete_workflow_has_no_problems(self):
        workflow = wf.CurationWorkflow(
            name="ok",
            steps=[wf.WorkflowStep("predict", {"model": "m"}), wf.WorkflowStep("curate_trials")],
        )
        assert wf.validate(workflow) == []

    def test_no_steps_is_a_problem(self):
        assert wf.validate(wf.CurationWorkflow(name="empty"))

    def test_predict_without_a_model_is_a_problem(self):
        problems = wf.validate(wf.CurationWorkflow(name="x", steps=[wf.WorkflowStep("predict")]))
        assert any("no model chosen" in p for p in problems)

    def test_a_filter_with_no_condition_is_a_problem(self):
        step = wf.WorkflowStep("filter_trials", {"filters": [{"column": "genotype"}]})
        problems = wf.validate(wf.CurationWorkflow(name="x", steps=[step]))
        assert any("neither allowed values nor a comparison" in p for p in problems)

    def test_an_unknown_kind_is_a_problem(self):
        problems = wf.validate(wf.CurationWorkflow(name="x", steps=[wf.WorkflowStep("teleport")]))
        assert any("unknown step kind" in p for p in problems)


def test_every_step_kind_has_a_handler():
    """The contract between the model and the runner — an added kind cannot
    be forgotten on the GUI side."""
    assert set(dcw._HANDLERS) == set(wf.STEP_KINDS)


# ----------------------------------------------------------------------
# Replaying the trials-table filters
# ----------------------------------------------------------------------


class TestTrialFilters:
    """A workflow's *Filter trials* step drives the trials table itself —
    the one trial filter every later step runs over."""

    @pytest.fixture()
    def trials(self, qapp, tmp_path):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        widget = TrialsWidget(state)
        widget.setup(METADATA)
        yield widget
        widget.deleteLater()

    def test_categorical_filter_narrows_the_visible_trials(self, trials):
        assert trials.apply_column_filters([{"column": "genotype", "values": ["wt"]}]) == []
        assert trials.app_state.trials == [1, 2]

    def test_numeric_filter_narrows_the_visible_trials(self, trials):
        assert trials.apply_column_filters([{"column": "score", "op": ">=", "value": 0.5}]) == []
        assert trials.app_state.trials == [2, 4]

    def test_filters_round_trip_through_the_stored_form(self, trials):
        applied = [{"column": "genotype", "values": ["ko"]}, {"column": "score", "op": "<=", "value": 0.5}]
        trials.apply_column_filters(applied)
        assert trials.column_filters() == applied
        trials.clear_column_filters()
        assert trials.column_filters() == []
        assert trials.app_state.trials == [1, 2, 3, 4]

    def test_an_unknown_column_is_reported_not_raised(self, trials):
        skipped = trials.apply_column_filters([{"column": "nope", "values": ["x"]}])
        assert skipped and "nope" in skipped[0]

    def test_a_value_this_dataset_lacks_is_reported(self, trials):
        skipped = trials.apply_column_filters([{"column": "genotype", "values": ["het"]}])
        assert skipped and "het" in skipped[0]

    def test_clear_first_replaces_rather_than_accumulates(self, trials):
        trials.apply_column_filters([{"column": "genotype", "values": ["wt"]}])
        trials.apply_column_filters([{"column": "score", "op": ">=", "value": 0.5}], clear_first=True)
        assert trials.column_filters() == [{"column": "score", "op": ">=", "value": 0.5}]

    def test_all_trials_stays_the_full_set_once_filtered(self, trials):
        """The basis a "hidden trials" delete scope takes its complement against."""
        assert trials.all_trials() == [1, 2, 3, 4]
        trials.apply_column_filters([{"column": "genotype", "values": ["wt"]}])
        assert trials.app_state.trials == [1, 2]
        assert trials.all_trials() == [1, 2, 3, 4]


# ----------------------------------------------------------------------
# The runner
# ----------------------------------------------------------------------


class _Gate(QObject):
    """Stands in for a grid dialog's ``finished`` — the runner waits on it."""

    finished = Signal()


class _RecordingRunner(dcw.WorkflowRunner):
    """A runner whose steps are recorded instead of performed.

    Every kind is redirected, so the walk itself — order, hand-over, stop —
    is what is under test, not what a step does to the GUI.
    """

    def __init__(self, waits_on: dict[str, _Gate] | None = None):
        super().__init__(_MetaStub())
        self.performed: list[str] = []
        self._waits_on = waits_on or {}

    def _perform(self, step):
        self.performed.append(step.kind)
        gate = self._waits_on.get(step.kind)
        if gate is None:
            return False
        self._wait_for(gate.finished, "waiting")
        return True


class _MetaStub:
    def __init__(self):
        self.app_state = ObservableAppState()


@pytest.fixture()
def patched_handlers(monkeypatch):
    """Point every handler at the runner's recorder."""
    for kind in wf.STEP_KINDS:
        monkeypatch.setitem(dcw._HANDLERS, kind, lambda runner, step: runner._perform(step))


def _drain(qapp):
    """Let the runner's ``singleShot(0)`` hops run."""
    for _ in range(40):
        qapp.processEvents()


class TestRunner:
    def test_walks_every_step_in_order(self, qapp, patched_handlers):
        runner = _RecordingRunner()
        done = []
        runner.finished.connect(done.append)
        workflow = wf.CurationWorkflow(
            name="w",
            steps=[wf.WorkflowStep("curate_trials"), wf.WorkflowStep("save_labels")],
        )
        assert runner.run(workflow)
        _drain(qapp)
        assert runner.performed == ["curate_trials", "save_labels"]
        assert done == [True]
        assert not runner.running

    def test_an_interactive_step_holds_the_walk_until_it_ends(self, qapp, patched_handlers):
        gate = _Gate()
        runner = _RecordingRunner({"label_grid": gate})
        workflow = wf.CurationWorkflow(
            name="w",
            steps=[wf.WorkflowStep("label_grid"), wf.WorkflowStep("save_labels")],
        )
        runner.run(workflow)
        _drain(qapp)
        assert runner.performed == ["label_grid"]
        assert runner.running

        gate.finished.emit()
        _drain(qapp)
        assert runner.performed == ["label_grid", "save_labels"]
        assert not runner.running

    def test_stopping_mid_wait_abandons_the_rest(self, qapp, patched_handlers):
        gate = _Gate()
        runner = _RecordingRunner({"label_grid": gate})
        done = []
        runner.finished.connect(done.append)
        workflow = wf.CurationWorkflow(name="w", steps=[wf.WorkflowStep("label_grid"), wf.WorkflowStep("save_labels")])
        runner.run(workflow)
        _drain(qapp)
        runner.stop("stopped")
        gate.finished.emit()  # the closed dialog must not restart the walk
        _drain(qapp)
        assert runner.performed == ["label_grid"]
        assert done == [False]

    def test_a_step_that_cannot_run_stops_the_workflow(self, qapp, monkeypatch):
        def refuse(runner, step):
            raise dcw.WorkflowError("no trials table")

        monkeypatch.setitem(dcw._HANDLERS, "filter_trials", refuse)
        monkeypatch.setitem(dcw._HANDLERS, "save_labels", lambda runner, step: runner._perform(step))
        runner = _RecordingRunner()
        done = []
        runner.finished.connect(done.append)
        workflow = wf.CurationWorkflow(
            name="w",
            steps=[wf.WorkflowStep("filter_trials"), wf.WorkflowStep("save_labels")],
        )
        runner.run(workflow)
        _drain(qapp)
        assert runner.performed == []
        assert done == [False]

    def test_an_invalid_workflow_never_starts(self, qapp, patched_handlers):
        runner = _RecordingRunner()
        assert not runner.run(wf.CurationWorkflow(name="w"))
        assert not runner.running

    def test_after_shown_defers_work_by_one_event_loop_turn(self, qapp):
        """Generating a grid runs nested event loops; starting that in the
        same turn as the dialog's show() drives a window that is still being
        exposed, and it visibly disappears and comes back."""

        class _Shown:
            def isVisible(self):
                return True

        runner = dcw.WorkflowRunner(_MetaStub())
        done = []
        runner.after_shown(_Shown(), lambda: done.append(True))
        assert done == []  # not in this turn
        _drain(qapp)
        assert done == [True]

    def test_after_shown_skips_a_dialog_the_user_already_closed(self, qapp):
        class _Closed:
            def isVisible(self):
                return False

        runner = dcw.WorkflowRunner(_MetaStub())
        done = []
        runner.after_shown(_Closed(), lambda: done.append(True))
        _drain(qapp)
        assert done == []

    def test_a_failing_step_stops_the_workflow_instead_of_raising(self, qapp, monkeypatch):
        """A step raising anything must stop the run cleanly — an exception
        escaping into the event loop leaves the GUI half-configured."""

        def boom(runner, step):
            raise RuntimeError("kaboom")

        monkeypatch.setitem(dcw._HANDLERS, "curate_trials", boom)
        monkeypatch.setitem(dcw._HANDLERS, "save_labels", lambda runner, step: runner._perform(step))
        runner = _RecordingRunner()
        done = []
        runner.finished.connect(done.append)
        runner.run(
            wf.CurationWorkflow(
                name="w",
                steps=[wf.WorkflowStep("curate_trials"), wf.WorkflowStep("save_labels")],
            )
        )
        _drain(qapp)
        assert runner.performed == []
        assert done == [False]
        assert not runner.running

    def test_scope_falls_back_to_what_the_last_prediction_wrote(self, qapp):
        """The one thing carried between steps: an empty scope step reviews
        exactly the classes the prediction produced."""
        runner = dcw.WorkflowRunner(_MetaStub())
        runner.predicted_label_ids = [3, 7]
        recorded = {}

        class _PanelStub:
            def set_scope(self, label_ids, *, reason):
                recorded["ids"] = list(label_ids)

        runner.meta.labels_widget = type("L", (), {"curation_panel": _PanelStub()})()
        dcw._run_scope(runner, wf.WorkflowStep("scope"))
        assert recorded["ids"] == [3, 7]
        assert runner.app_state.curation_mode == "manual"


# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------


MAPPINGS = {
    1: {"name": "approach", "event_type": "point", "color": (1.0, 0.0, 0.0)},
    2: {"name": "groom", "event_type": "state", "color": (0.0, 1.0, 0.0)},
}


class _LabelsStub:
    def __init__(self, app_state):
        from ethograph.gui.widgets_curation import CurationPanel

        self._mappings = MAPPINGS
        self.curation_panel = CurationPanel(app_state, self)


class _DialogMeta:
    def __init__(self, app_state, trials_widget):
        self.app_state = app_state
        self.labels_widget = _LabelsStub(app_state)
        self.trials_widget = trials_widget
        self.data_widget = None
        self.navigation_widget = None
        self.io_widget = None


class TestDialog:
    """Every step kind must be addable and editable — the form is generated
    from the kind's parameter specs, so a spec the editor cannot render would
    only show up here."""

    @pytest.fixture()
    def dialog(self, qapp, tmp_path):
        state = ObservableAppState()
        state._yaml_path = str(tmp_path / "gui_settings.yaml")
        trials = TrialsWidget(state)
        trials.setup(METADATA)
        dialog = dcw.CurationWorkflowDialog(_DialogMeta(state, trials))
        yield dialog
        dialog.close()
        trials.deleteLater()

    def test_every_kind_can_be_added_and_edited(self, dialog):
        dialog._new_named("all-kinds")
        for i, kind in enumerate(wf.STEP_KINDS):
            dialog.kind_combo.setCurrentIndex(i)
            dialog._add_step()
        assert [s.kind for s in dialog._workflow.steps] == list(wf.STEP_KINDS)
        assert dialog.step_list.count() == len(wf.STEP_KINDS)
        for row in range(dialog.step_list.count()):
            dialog.step_list.setCurrentRow(row)
            assert dialog.editor.title.text() == dialog._workflow.steps[row].title()

    def test_a_step_survives_a_save_and_reload(self, dialog):
        dialog._new_named("round-trip")
        dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("video_grid"))
        dialog._add_step()
        dialog._workflow.steps[0].params["per_page"] = 9
        dialog._save()
        dialog._flush_save()
        assert wf.load_workflow("round-trip").steps[0].value("per_page") == 9

    def test_deleting_a_workflow_removes_it_for_good(self, dialog):
        """A pending edit must not write the file back after the delete —
        every selection change flushes, and the deleted one is still selected
        at that moment."""
        dialog._new_named("keep")
        dialog._new_named("doomed")
        dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("save_labels"))
        dialog._add_step()  # leaves "doomed" dirty
        assert dialog._workflow.name == "doomed"

        dialog._delete_selected()
        assert wf.list_workflows() == ["keep"]
        dialog._flush_save()
        assert wf.list_workflows() == ["keep"]

    def test_renaming_moves_the_workflow_rather_than_copying_it(self, dialog):
        dialog._new_named("before")
        dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("curate_trials"))
        dialog._add_step()
        dialog._rename_to("after")

        assert wf.list_workflows() == ["after"]
        assert [s.kind for s in wf.load_workflow("after").steps] == ["curate_trials"]
        dialog._flush_save()
        assert wf.list_workflows() == ["after"]

    def test_renaming_carries_unsaved_edits_across(self, dialog):
        dialog._new_named("before")
        dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("save_labels"))
        dialog._add_step()  # not yet flushed
        dialog._rename_to("after")
        assert [s.kind for s in wf.load_workflow("after").steps] == ["save_labels"]

    def test_a_grid_step_carries_its_cameras(self, dialog):
        dialog._new_named("cams")
        dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("label_grid"))
        dialog._add_step()
        dialog._workflow.steps[0].params["cameras"] = ["cam-top", "cam-side"]
        dialog._save()
        dialog._flush_save()
        assert wf.load_workflow("cams").steps[0].value("cameras") == ["cam-top", "cam-side"]

    def test_a_grid_step_defaults_to_every_camera(self, dialog):
        step = wf.WorkflowStep("video_grid")
        assert step.value("cameras") == []
        assert "all cameras" in dcw.describe_step(step)

    def test_moving_and_removing_steps(self, dialog):
        dialog._new_named("ordering")
        for kind in ("curate_trials", "save_labels"):
            dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData(kind))
            dialog._add_step()
        dialog.step_list.setCurrentRow(0)
        dialog._move(1)
        assert [s.kind for s in dialog._workflow.steps] == ["save_labels", "curate_trials"]
        dialog._remove_step()
        assert [s.kind for s in dialog._workflow.steps] == ["save_labels"]

    def test_a_predict_step_opens_at_the_predict_dialog_s_own_threshold(self, dialog):
        """A prediction step must behave like pressing Predict by hand, so
        the two defaults are pinned to each other, not just to a number."""
        from ethograph.gui.dialog_onset_model import PredictOnsetDialog

        predict = PredictOnsetDialog(dialog.meta)
        try:
            assert wf.STEP_KINDS["predict"].spec("min_confidence").default == predict.min_conf_edit.value()
        finally:
            predict.close()

    def test_capture_takes_the_filters_the_trials_table_has(self, dialog):
        dialog.meta.trials_widget.apply_column_filters([{"column": "genotype", "values": ["ko"]}])
        params = dcw.capture_params("filter_trials", dialog.meta)
        assert params["filters"] == [{"column": "genotype", "values": ["ko"]}]
