"""Curation workflows: record a curation routine once, replay it per session.

Reviewing a model's predictions is the same handful of moves every time —
narrow the trials table to one condition, predict, drop the predicted classes
into the curation scope, open a grid laid out the way that behaviour needs,
walk the boundaries, save. This dialog is where that sequence is written down
(:mod:`ethograph.labels.workflow` holds the Qt-free model and the YAML store
under ``~/.ethograph/defaults/workflows``) and replayed.

Two halves:

* :class:`CurationWorkflowDialog` — the workflows on the left, the selected
  one's steps on the right. A step is added by kind, configured from a form
  built out of the kind's :class:`~ethograph.labels.workflow.ParamSpec` list,
  and **Capture current** fills that form from whatever the GUI is set to
  right now — the fastest way to record a routine is to do it once by hand
  and capture each step as you go.
* :class:`WorkflowRunner` — walks the steps. Most run and return; an
  *interactive* step (a grid to work through, a review to walk) hands over
  and waits for the reviewer to close it before the next step starts. The
  runner carries one thing between steps: the label classes the last
  prediction wrote, so a *Set curation scope* step with no explicit classes
  reviews exactly what was just predicted.

Every step drives the same widgets a user would: the trials table's own
filters, the Predict dialog's own prediction, the Curation section's own
scope and review. A workflow is a recording of the GUI, never a second way
of doing any of it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.dialog_label_gridview import ConfidenceEdit, format_confidence, open_gui_panels
from ethograph.gui.dialog_onset_model import predict_onsets
from ethograph.gui.notify import notify
from ethograph.gui.project import project_dir_of
from ethograph.labels import onset_model as om
from ethograph.labels import workflow as wf

logger = logging.getLogger(__name__)

#: Editing a step writes the workflow YAML this long after the last change.
_SAVE_DELAY_MS = 400


class WorkflowError(RuntimeError):
    """A step cannot run — the workflow stops and says why."""


def _curation_panel(meta):
    return getattr(getattr(meta, "labels_widget", None), "curation_panel", None)


def _require_panel(meta):
    panel = _curation_panel(meta)
    if panel is None:
        raise WorkflowError("No Curation section in this window.")
    return panel


def _require_trials_widget(meta):
    trials = getattr(meta, "trials_widget", None)
    if trials is None:
        raise WorkflowError("No trials table — load a dataset with trial metadata first.")
    return trials


# ======================================================================
# Running
# ======================================================================


class WorkflowRunner(QObject):
    """Walks a workflow's steps, waiting on the interactive ones.

    One runner runs one workflow at a time; :meth:`run` returns immediately
    and the walk continues on the event loop, so the GUI stays live while a
    grid or a review is on screen.
    """

    step_started = Signal(int, str)
    note = Signal(str)
    finished = Signal(bool)

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.app_state = meta.app_state
        self._workflow: wf.CurationWorkflow | None = None
        self._index = 0
        self._running = False
        #: Carried between steps: the classes the last prediction step wrote.
        self.predicted_label_ids: list[int] = []
        #: What the current interactive step is waiting on, so it is
        #: disconnected exactly once however the step ends.
        self._waiting_on: tuple[Any, Callable[..., None]] | None = None

    @property
    def running(self) -> bool:
        return self._running

    def run(self, workflow: wf.CurationWorkflow) -> bool:
        """Start *workflow*; returns whether it started."""
        if self._running:
            notify("A workflow is already running.", severity="warning")
            return False
        problems = wf.validate(workflow)
        if problems:
            notify(problems[0], severity="warning")
            return False
        self._workflow = workflow
        self._index = 0
        self.predicted_label_ids = []
        self._running = True
        self._next()
        return True

    def stop(self, message: str | None = None) -> None:
        """Abandon the run, whatever it is waiting on."""
        if not self._running:
            return
        self._disconnect_wait()
        self._running = False
        if message:
            self.note.emit(message)
        self.finished.emit(False)

    # ------------------------------------------------------------------

    def _next(self) -> None:
        if not self._running or self._workflow is None:
            return
        if self._index >= len(self._workflow.steps):
            self._running = False
            self.note.emit("Workflow finished.")
            self.finished.emit(True)
            return
        step = self._workflow.steps[self._index]
        self._index += 1
        self.step_started.emit(self._index - 1, step.title())
        # Logged, not only shown: the run log lives in a dialog, so without
        # this a session that ends badly cannot say which step it was on.
        logger.info(
            "Workflow %r step %d/%d: %s (%s)",
            self._workflow.name,
            self._index,
            len(self._workflow.steps),
            step.title(),
            describe_step(step),
        )
        handler = _HANDLERS[step.kind]
        try:
            waits = handler(self, step)
        except WorkflowError as e:
            self._running = False
            self.note.emit(f"Stopped: {e}")
            notify(str(e), severity="warning")
            self.finished.emit(False)
            return
        except Exception:
            # A step failing must stop the workflow, not tear down the GUI's
            # event loop mid-run with a half-applied state.
            logger.exception("Workflow %r failed on step: %s", self._workflow.name, step.title())
            self._running = False
            self.note.emit(f"Stopped: {step.title()} failed — see the log.")
            notify(f"Workflow stopped: {step.title()} failed. See the log for the traceback.", severity="error")
            self.finished.emit(False)
            return
        if not waits:
            # Off the current call stack: a handler that opened a dialog must
            # get its event loop back before the next step touches the GUI.
            QTimer.singleShot(0, self._next)

    def after_shown(self, dialog, work) -> None:
        """Run *work* once *dialog* has actually been painted.

        Generating a grid runs nested event loops (progress dialogs, the panel
        capture navigating the GUI). Starting that in the same turn as the
        dialog's ``show()`` drives the window before it has finished being
        exposed, which on Windows makes it visibly disappear and come back.
        One event-loop turn later it is a normal, settled window.
        """

        def run():
            if dialog.isVisible():
                work()

        QTimer.singleShot(0, run)

    def _wait_for(self, signal, note: str) -> None:
        """Hand over to the user: continue when *signal* fires."""
        self.note.emit(note)

        def resume(*_args):
            self._disconnect_wait()
            QTimer.singleShot(0, self._next)

        signal.connect(resume)
        self._waiting_on = (signal, resume)

    def _disconnect_wait(self) -> None:
        if self._waiting_on is None:
            return
        signal, slot = self._waiting_on
        self._waiting_on = None
        signal.disconnect(slot)


# ----------------------------------------------------------------------
# Step handlers: each returns whether the runner should wait for the user
# ----------------------------------------------------------------------


def _run_filter_trials(runner: WorkflowRunner, step: wf.WorkflowStep) -> bool:
    trials = _require_trials_widget(runner.meta)
    skipped = trials.apply_column_filters(step.value("filters"), clear_first=bool(step.value("clear_first")))
    for message in skipped:
        runner.note.emit(f"Filter skipped — {message}.")
    n = len(runner.app_state.trials or [])
    runner.note.emit(f"{n} trial(s) now visible.")
    return False


def _run_predict(runner: WorkflowRunner, step: wf.WorkflowStep) -> bool:
    name = str(step.value("model")).strip()
    if name not in om.list_models():
        raise WorkflowError(f"No model named {name!r} in ~/.ethograph/defaults/runs/lightgbm.")
    if not om.is_trained(name):
        raise WorkflowError(f"Model {name!r} has not been trained yet.")
    individual = str(step.value("individual")).strip() or (runner.app_state.selected_individual() or "")
    outcome = predict_onsets(
        runner.meta,
        name,
        individual=individual,
        min_confidence=float(step.value("min_confidence")),
    )
    if outcome is None:
        raise WorkflowError(f"Model {name!r} could not be loaded.")
    runner.predicted_label_ids = outcome.label_ids()
    runner.note.emit(outcome.message())
    return False


def _run_scope(runner: WorkflowRunner, step: wf.WorkflowStep) -> bool:
    panel = _require_panel(runner.meta)
    label_ids = [int(i) for i in step.value("label_ids") or []] or runner.predicted_label_ids
    panel.set_scope(label_ids, reason="curation workflow")
    runner.app_state.curation_mode = str(step.value("mode"))
    named = ", ".join(str(i) for i in label_ids) if label_ids else "every class"
    runner.note.emit(f"Scope: {named}.")
    return False


def _apply_cameras(runner: WorkflowRunner, setup, cameras) -> None:
    """Tick exactly *cameras* on a grid's setup page; empty leaves it alone.

    The list is stored by camera name, so a workflow moves between datasets;
    a name this one does not have is reported and skipped, exactly as a trial
    filter's unknown column is.
    """
    wanted = {str(c) for c in cameras or []}
    if not wanted or setup.camera_list is None:
        return
    available = set()
    for i in range(setup.camera_list.count()):
        item = setup.camera_list.item(i)
        name = str(item.data(Qt.UserRole))
        available.add(name)
        item.setCheckState(Qt.Checked if name in wanted else Qt.Unchecked)
    missing = wanted - available
    if missing:
        runner.note.emit(f"Camera(s) not in this dataset, skipped: {', '.join(sorted(missing))}.")
    if not wanted & available:
        # Every tile would be dropped; better the grid's own choice than none.
        runner.note.emit("None of the step's cameras exist here — keeping the grid's own selection.")
        for i in range(setup.camera_list.count()):
            setup.camera_list.item(i).setCheckState(Qt.Checked)


def _apply_methods(setup, choice: str) -> None:
    """Put a grid's setup page on the step's labeling-method filter; a choice
    the page does not offer leaves it on its own."""
    index = setup.method_combo.findData(str(choice))
    if index >= 0:
        setup.method_combo.setCurrentIndex(index)


def _apply_grid_mode(mode_bar, mode: str, mark_flagged: bool) -> None:
    """Put a grid's mode bar into *mode*, optionally pre-clicking the flagged.

    A mode the bar does not offer — a workflow stored when ``navigate`` was
    still one — leaves the grid on its own default rather than failing.
    """
    index = mode_bar.mode_combo.findData(mode)
    if index >= 0:
        mode_bar.mode_combo.setCurrentIndex(index)
    # Through the button, so the bar's own rule holds: pre-clicking the
    # flagged tiles only means anything where a click means "uncurated".
    if mark_flagged and mode_bar.mark_flagged_btn.isEnabled():
        mode_bar.mark_flagged_btn.click()


def _panel_title(entry: Any) -> str:
    return str(entry.get("title", "")) if isinstance(entry, dict) else str(entry)


def _panel_entry(title: str, widget: QWidget) -> dict[str, Any]:
    """What a step needs to reopen this panel later: its title, and — for a
    feature panel (lineplot/heatmap) — the feature and dim selections that
    pin e.g. which keypoint or individual it shows. A non-feature panel
    (ephys/raster) carries only its title; it is not reopenable by a step."""
    entry: dict[str, Any] = {"title": title}
    if getattr(widget, "panel_group", None) == "feature" and hasattr(widget, "panel_settings"):
        settings = widget.panel_settings()
        if settings.get("feature"):
            entry["feature"] = settings["feature"]
            entry["selections"] = settings.get("selections") or {}
    return entry


def _ensure_panels_open(runner: WorkflowRunner, entries: list) -> None:
    """Reopen every named panel that names a feature and is not already open.

    Mirrors what Shift+N's add-panel popup does, called directly rather than
    driven through the popup's UI — the same way every other step drives its
    widget. A closed panel's *selections* (which keypoint, which individual)
    are restored too, since that is what makes the reopened panel the same
    one the step recorded, not just a panel on the same feature.

    An entry with no feature on record (a plain title from a workflow saved
    before this existed, or a non-feature panel like ephys/raster) cannot be
    reopened this way — it is left for the caller to report as missing.
    """
    container = getattr(getattr(runner.meta, "data_widget", None), "plot_container", None)
    if container is None:
        return
    # Same set add_panel() itself gates on; checked here too so a feature this
    # session lacks is reported rather than left as a blank, mis-set panel.
    available = set(container._available_features())
    open_titles = {title for title, _w in open_gui_panels(runner.meta)}
    opened, unavailable = [], []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "")
        feature = entry.get("feature")
        if not title or not feature or title in open_titles:
            continue
        if feature not in available:
            unavailable.append(title)
            continue
        plot = container.add_panel("lineplot", feature=feature)
        if plot is None:
            continue
        plot.apply_panel_settings({"feature": feature, "selections": entry.get("selections") or {}})
        container.set_panel_title(plot, title)
        plot.update_plot()
        opened.append(title)
    if opened:
        runner.note.emit(f"Reopened panel(s): {', '.join(opened)}.")
    if unavailable:
        runner.note.emit(f"Panel(s) name a feature this session does not have: {', '.join(unavailable)}.")


def _run_label_grid(runner: WorkflowRunner, step: wf.WorkflowStep) -> bool:
    panel = _require_panel(runner.meta)
    state = runner.app_state
    # The grid's own spins seed from these, so they are set before it opens.
    state.label_grid_columns = int(step.value("columns"))
    state.label_grid_window_s = float(step.value("window_s"))
    state.grid_confidence_threshold = float(step.value("threshold"))
    # Before the grid opens, so its setup page's panel list already sees them.
    _ensure_panels_open(runner, step.value("panels"))
    dialog = panel.open_grid_view()
    if dialog is None:
        raise WorkflowError("The label grid could not be opened.")
    _apply_cameras(runner, dialog.setup, step.value("cameras"))
    _apply_methods(dialog.setup, step.value("methods"))
    wanted = {_panel_title(e) for e in step.value("panels") or []}
    if dialog.panel_list is not None:
        for i in range(dialog.panel_list.count()):
            item = dialog.panel_list.item(i)
            item.setCheckState(Qt.Checked if item.text() in wanted else Qt.Unchecked)
        missing = wanted - {dialog.panel_list.item(i).text() for i in range(dialog.panel_list.count())}
        if missing:
            runner.note.emit(f"Panel(s) not open, skipped: {', '.join(sorted(missing))}.")
    if dialog.window_spin is not None:
        dialog.window_spin.setValue(float(step.value("window_s")))
    if dialog.axis_auto_cb is not None:
        dialog.axis_auto_cb.setChecked(bool(step.value("axis_auto")))
    if dialog.skip_video_cb is not None:
        dialog.skip_video_cb.setChecked(bool(step.value("skip_video")))
    # Waiting is armed before generating: generating runs nested event loops
    # (progress dialogs, panel capture), so the dialog can be closed inside it.
    runner._wait_for(dialog.finished, "Label grid open — close it to continue.")
    if step.value("generate"):
        if wanted:
            runner.note.emit(
                f"Capturing {len(wanted)} panel(s) — the GUI navigates through the labels' "
                "trials and returns. Nothing is lost; let it finish."
            )

        def build():
            if dialog.grid_view is None and dialog.isVisible():
                dialog.generate()
            if dialog.grid_view is not None:
                _apply_grid_mode(
                    dialog.grid_view.mode_bar, str(step.value("grid_mode")), bool(step.value("mark_flagged"))
                )

        runner.after_shown(dialog, build)
    return True


def _run_video_grid(runner: WorkflowRunner, step: wf.WorkflowStep) -> bool:
    panel = _require_panel(runner.meta)
    state = runner.app_state
    state.video_grid_point_window_s = float(step.value("point_window_s"))
    state.video_grid_per_page = int(step.value("per_page"))
    state.video_grid_columns = int(step.value("columns"))
    state.video_grid_speed_pct = float(step.value("speed_pct"))
    state.grid_confidence_threshold = float(step.value("threshold"))
    dialog = panel.open_video_grid()
    if dialog is None:
        raise WorkflowError("The video grid could not be opened.")
    _apply_cameras(runner, dialog.setup, step.value("cameras"))
    _apply_methods(dialog.setup, step.value("methods"))
    dialog.point_window_spin.setValue(float(step.value("point_window_s")))
    dialog.per_page_spin.setValue(int(step.value("per_page")))
    dialog.columns_spin.setValue(int(step.value("columns")))
    runner._wait_for(dialog.finished, "Video grid open — close it to continue.")
    if step.value("generate"):

        def build():
            if dialog.player is None and dialog.isVisible():
                dialog.generate()
            if dialog.player is not None:
                _apply_grid_mode(dialog.player.mode_bar, str(step.value("grid_mode")), False)

        runner.after_shown(dialog, build)
    return True


def _run_frame_review(runner: WorkflowRunner, step: wf.WorkflowStep) -> bool:
    panel = _require_panel(runner.meta)
    state = runner.app_state
    state.refine_window_s = float(step.value("window_s"))
    state.frame_review_automated_only = bool(step.value("automated_only"))
    state.curation_next_curates = bool(step.value("next_curates"))
    state.curation_mode = "frame"
    panel.window_spin.setValue(float(step.value("window_s")))
    panel.automated_only_cb.setChecked(bool(step.value("automated_only")))
    panel.next_curates_cb.setChecked(bool(step.value("next_curates")))
    if not panel.start_review():
        # Nothing to review is a legitimate outcome, not a broken workflow.
        runner.note.emit("Frame-by-frame review had nothing to walk — skipped.")
        return False
    runner._wait_for(panel.review_finished, "Frame-by-frame review running — finish or stop it to continue.")
    return True


def _label_ids_kwargs(step: wf.WorkflowStep) -> dict:
    """``{"label_ids": {...}}`` when the step names explicit classes, else
    ``{}`` — omitting the keyword lets the panel method's own default fall
    back to the curation scope (the scope area, or an earlier ``scope`` step),
    exactly as the bulk-editing dialog's own *All* checkbox would."""
    ids = [int(i) for i in step.value("label_ids") or []]
    return {"label_ids": set(ids)} if ids else {}


def _run_curate_trials(runner: WorkflowRunner, step: wf.WorkflowStep) -> bool:
    panel = _require_panel(runner.meta)
    n = panel.curate_trial_labels(str(step.value("which")), **_label_ids_kwargs(step))
    runner.note.emit(f"Curated {n} label(s).")
    return False


def _run_delete_labels(runner: WorkflowRunner, step: wf.WorkflowStep) -> bool:
    panel = _require_panel(runner.meta)
    n = panel.delete_trial_labels(str(step.value("which")), **_label_ids_kwargs(step))
    runner.note.emit(f"Deleted {n} label(s).")
    return False


def _run_purge_labels(runner: WorkflowRunner, step: wf.WorkflowStep) -> bool:
    panel = _require_panel(runner.meta)
    n = panel.purge_trial_labels(
        str(step.value("which")), float(step.value("min_duration_s")), **_label_ids_kwargs(step)
    )
    runner.note.emit(f"Purged {n} label(s).")
    return False


def _run_save_labels(runner: WorkflowRunner, step: wf.WorkflowStep) -> bool:
    io_widget = getattr(runner.meta, "io_widget", None)
    if io_widget is None:
        raise WorkflowError("No I/O panel to save through.")
    io_widget._save_labels()
    runner.note.emit("Labels saved.")
    return False


#: Step kind → handler. Every kind in :data:`ethograph.labels.workflow.STEP_KINDS`
#: has exactly one entry (checked below), so an added kind cannot be forgotten.
_HANDLERS: dict[str, Callable[[WorkflowRunner, wf.WorkflowStep], bool]] = {
    "filter_trials": _run_filter_trials,
    "predict": _run_predict,
    "scope": _run_scope,
    "label_grid": _run_label_grid,
    "video_grid": _run_video_grid,
    "frame_review": _run_frame_review,
    "curate_trials": _run_curate_trials,
    "delete_labels": _run_delete_labels,
    "purge_labels": _run_purge_labels,
    "save_labels": _run_save_labels,
}

_unhandled = set(wf.STEP_KINDS) - set(_HANDLERS)
if _unhandled:
    raise RuntimeError(f"workflow step kinds with no handler: {sorted(_unhandled)}")


# ======================================================================
# Capturing the GUI's current settings into a step
# ======================================================================


def _current_cameras(state) -> list[str]:
    """The cameras a grid would tick right now: the reviewer's last selection,
    else every camera the session has."""
    saved = getattr(state, "grid_selected_cameras", None)
    if saved:
        return [str(c) for c in saved]
    return [str(c) for c in (getattr(getattr(state, "nwb_alignment", None), "cameras", None) or [])]


def capture_params(kind: str, meta) -> dict[str, Any]:
    """A step of *kind* configured the way the GUI is configured right now.

    The fast way to record a routine: do it once by hand, then capture each
    step. A setting the GUI has no live equivalent of (which model to run)
    keeps the kind's declared default.
    """
    state = meta.app_state
    params = wf.STEP_KINDS[kind].defaults()
    if kind == "filter_trials":
        trials = getattr(meta, "trials_widget", None)
        params["filters"] = trials.column_filters() if trials is not None else []
    elif kind == "predict":
        params["individual"] = state.selected_individual() or ""
    elif kind == "scope":
        params["label_ids"] = [int(i) for i in (state.curation_label_ids or [])]
        params["mode"] = str(state.get_with_default("curation_mode"))
    elif kind == "label_grid":
        params["cameras"] = _current_cameras(state)
        params["columns"] = int(state.get_with_default("label_grid_columns"))
        params["window_s"] = float(state.get_with_default("label_grid_window_s"))
        params["threshold"] = float(state.get_with_default("grid_confidence_threshold"))
    elif kind == "video_grid":
        params["cameras"] = _current_cameras(state)
        params["point_window_s"] = float(state.get_with_default("video_grid_point_window_s"))
        params["per_page"] = int(state.get_with_default("video_grid_per_page"))
        params["columns"] = int(state.get_with_default("video_grid_columns"))
        params["speed_pct"] = float(state.get_with_default("video_grid_speed_pct"))
        params["threshold"] = float(state.get_with_default("grid_confidence_threshold"))
    elif kind == "frame_review":
        params["window_s"] = float(state.get_with_default("refine_window_s"))
        params["automated_only"] = bool(state.get_with_default("frame_review_automated_only"))
        params["next_curates"] = bool(state.get_with_default("curation_next_curates"))
    return params


def _cameras_phrase(step: wf.WorkflowStep) -> str:
    cameras = step.value("cameras") or []
    return ", ".join(str(c) for c in cameras) if cameras else "all cameras"


def describe_step(step: wf.WorkflowStep) -> str:
    """The one-line summary the step list shows under a step's title."""
    if step.kind == "filter_trials":
        filters = step.value("filters") or []
        return " · ".join(wf.describe_filter(f) for f in filters) if filters else "no filters (every trial)"
    if step.kind == "predict":
        model = str(step.value("model")) or "no model chosen"
        return f"{model}, min confidence {format_confidence(float(step.value('min_confidence')))}"
    if step.kind == "scope":
        ids = step.value("label_ids") or []
        classes = ", ".join(str(i) for i in ids) if ids else "what the last prediction wrote"
        return f"{classes} · {wf.CURATION_MODE_CHOICES.get(str(step.value('mode')), '?')}"
    if step.kind == "label_grid":
        panels = step.value("panels") or []
        extra = f" + {len(panels)} panel(s)" if panels else ""
        mode = wf.GRID_MODE_CHOICES.get(str(step.value("grid_mode")), "?")
        return f"{_cameras_phrase(step)} · {int(step.value('columns'))} columns{extra} · {mode}"
    if step.kind == "video_grid":
        return (
            f"{_cameras_phrase(step)} · {int(step.value('per_page'))} clips, "
            f"{int(step.value('columns'))} columns, {float(step.value('speed_pct')):.0f}% speed"
        )
    if step.kind == "frame_review":
        scope = "automated only" if step.value("automated_only") else "every label in scope"
        return f"{float(step.value('window_s')):.2f} s window · {scope}"
    if step.kind in ("curate_trials", "delete_labels"):
        noun = wf.TRIAL_SCOPE_CHOICES.get(str(step.value("which")), "?")
        ids = step.value("label_ids") or []
        classes = ", ".join(str(i) for i in ids) if ids else "the curation scope"
        return f"{noun} · {classes}"
    if step.kind == "purge_labels":
        noun = wf.TRIAL_SCOPE_CHOICES.get(str(step.value("which")), "?")
        ids = step.value("label_ids") or []
        classes = ", ".join(str(i) for i in ids) if ids else "the curation scope"
        return f"{noun} · {classes} · shorter than {float(step.value('min_duration_s')):g} s"
    return step.spec().summary


# ======================================================================
# Editing one step's parameters
# ======================================================================


class FilterEditor(QWidget):
    """The ``filters`` parameter: metadata column conditions, as a list."""

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.meta = meta
        self._filters: list[dict] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setMaximumHeight(90)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        add_btn = QPushButton("Add…")
        add_btn.setAutoDefault(False)
        add_btn.clicked.connect(self._add)
        row.addWidget(add_btn)
        capture_btn = QPushButton("From trials table")
        capture_btn.setAutoDefault(False)
        capture_btn.setToolTip("Take the filters the trials table has active right now.")
        capture_btn.clicked.connect(self._capture)
        row.addWidget(capture_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.setAutoDefault(False)
        remove_btn.clicked.connect(self._remove)
        row.addWidget(remove_btn)
        lay.addLayout(row)

    def value(self) -> list[dict]:
        return [dict(f) for f in self._filters]

    def set_value(self, filters) -> None:
        self._filters = [dict(f) for f in (filters or [])]
        self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        for entry in self._filters:
            self.list.addItem(wf.describe_filter(entry))

    def _columns(self) -> dict[str, list[str] | None]:
        trials = getattr(self.meta, "trials_widget", None)
        return trials.filterable_columns() if trials is not None else {}

    def _capture(self) -> None:
        trials = getattr(self.meta, "trials_widget", None)
        if trials is None:
            notify("No trials table loaded.", severity="warning")
            return
        self.set_value(trials.column_filters())

    def _remove(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self._filters):
            del self._filters[row]
            self._refresh()

    def _add(self) -> None:
        columns = self._columns()
        if not columns:
            notify("No trial metadata columns to filter on.", severity="warning")
            return
        dialog = _FilterDialog(columns, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        entry = dialog.entry()
        if entry is None:
            return
        self._filters = [f for f in self._filters if f.get("column") != entry["column"]]
        self._filters.append(entry)
        self._refresh()


class _FilterDialog(QDialog):
    """One column condition: allowed values, or a numeric comparison."""

    def __init__(self, columns: dict[str, list[str] | None], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Column filter")
        self._columns = columns

        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.column_combo = QComboBox()
        for name in columns:
            self.column_combo.addItem(name)
        self.column_combo.currentTextChanged.connect(self._on_column)
        form.addRow("Column:", self.column_combo)
        lay.addLayout(form)

        self.values_list = QListWidget()
        self.values_list.setMaximumHeight(140)
        lay.addWidget(self.values_list)

        self.numeric_row = QWidget()
        num_lay = QHBoxLayout(self.numeric_row)
        num_lay.setContentsMargins(0, 0, 0, 0)
        self.op_combo = QComboBox()
        self.op_combo.addItem("at least (≥)", ">=")
        self.op_combo.addItem("at most (≤)", "<=")
        num_lay.addWidget(self.op_combo)
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(-1e9, 1e9)
        self.value_spin.setDecimals(4)
        num_lay.addWidget(self.value_spin, stretch=1)
        lay.addWidget(self.numeric_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._on_column(self.column_combo.currentText())

    def _on_column(self, name: str) -> None:
        values = self._columns.get(name)
        numeric = values is None
        self.numeric_row.setVisible(numeric)
        self.values_list.setVisible(not numeric)
        self.values_list.clear()
        for value in values or []:
            item = QListWidgetItem(str(value))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.values_list.addItem(item)

    def entry(self) -> dict | None:
        column = self.column_combo.currentText()
        if self._columns.get(column) is None:
            return {"column": column, "op": self.op_combo.currentData(), "value": float(self.value_spin.value())}
        values = [
            self.values_list.item(i).text()
            for i in range(self.values_list.count())
            if self.values_list.item(i).checkState() == Qt.Checked
        ]
        if not values:
            notify("Tick at least one value.", severity="warning")
            return None
        return {"column": column, "values": values}


class StepEditor(QWidget):
    """A form for one step, built from its kind's parameter specs."""

    changed = Signal()

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.app_state = meta.app_state
        self._step: wf.WorkflowStep | None = None
        self._widgets: dict[str, QWidget] = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel("")
        self.title.setStyleSheet("font-weight: bold;")
        lay.addWidget(self.title)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(self.summary)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        lay.addWidget(self.form_host)
        self.capture_btn = QPushButton("Capture current GUI settings")
        self.capture_btn.setAutoDefault(False)
        self.capture_btn.setToolTip(
            "Fill this step in from how the GUI is set up right now — do the step\nby hand once, then capture it."
        )
        self.capture_btn.clicked.connect(self._capture)
        lay.addWidget(self.capture_btn)
        lay.addStretch()

    def set_step(self, step: wf.WorkflowStep | None) -> None:
        self._step = step
        self._widgets.clear()
        while self.form.count():
            item = self.form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if step is None:
            self.title.setText("")
            self.summary.setText("Select a step, or add one.")
            self.capture_btn.setEnabled(False)
            return
        kind = step.spec()
        self.title.setText(kind.title)
        self.summary.setText(kind.summary)
        self.capture_btn.setEnabled(bool(kind.params))
        for spec in kind.params:
            widget = self._build(spec, step.value(spec.key))
            if spec.tooltip:
                widget.setToolTip(spec.tooltip)
            self._widgets[spec.key] = widget
            self.form.addRow(spec.label + ":", widget)

    def _build(self, spec: wf.ParamSpec, value: Any) -> QWidget:
        if spec.type == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.toggled.connect(self._commit)
            return widget
        if spec.type == "int":
            widget = QSpinBox()
            widget.setRange(int(spec.minimum), int(spec.maximum))
            widget.setValue(int(value))
            widget.valueChanged.connect(self._commit)
            return widget
        if spec.type == "float":
            widget = QDoubleSpinBox()
            widget.setRange(spec.minimum, spec.maximum)
            widget.setDecimals(2)
            widget.setSingleStep(0.05)
            widget.setValue(float(value))
            widget.valueChanged.connect(self._commit)
            return widget
        if spec.type == "confidence":
            widget = ConfidenceEdit(float(value))
            widget.valueChanged.connect(self._commit)
            return widget
        if spec.type == "choice":
            widget = QComboBox()
            for key, text in (spec.choices or self._dynamic_choices(spec)).items():
                widget.addItem(text, key)
            index = widget.findData(value)
            widget.setCurrentIndex(max(0, index))
            widget.currentIndexChanged.connect(self._commit)
            return widget
        if spec.type == "labels":
            widget = QListWidget()
            widget.setMaximumHeight(110)
            chosen = {int(i) for i in value or []}
            for label_id, name in self._label_classes().items():
                item = QListWidgetItem(f"{label_id} — {name}")
                item.setData(Qt.UserRole, label_id)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if label_id in chosen else Qt.Unchecked)
                widget.addItem(item)
            widget.itemChanged.connect(self._commit)
            return widget
        if spec.type == "panels":
            open_entries = [_panel_entry(title, w) for title, w in open_gui_panels(self.meta)]
            return self._panel_list(open_entries, value)
        if spec.type == "cameras":
            cameras = getattr(getattr(self.app_state, "nwb_alignment", None), "cameras", None) or []
            return self._name_list([str(c) for c in cameras], value, absent="not in this dataset")
        if spec.type == "filters":
            widget = FilterEditor(self.meta)
            widget.set_value(value)
            widget.list.model().rowsInserted.connect(self._commit)
            widget.list.model().rowsRemoved.connect(self._commit)
            return widget
        widget = QLineEdit(str(value or ""))
        widget.editingFinished.connect(self._commit)
        return widget

    def _panel_list(self, open_entries: list[dict], value) -> QListWidget:
        """Checkable list of panels, one entry per currently open panel plus
        whatever the step already names that is not open right now.

        Unlike :meth:`_name_list`, the item data is the whole entry — title,
        feature and selections — not just a name, so a closed panel this step
        already ticked keeps enough to be reopened by :func:`_ensure_panels_open`
        even while it stays unavailable to look at here.
        """
        widget = QListWidget()
        widget.setMaximumHeight(90)
        stored = [e if isinstance(e, dict) else {"title": str(e)} for e in (value or [])]
        chosen_titles = {e["title"] for e in stored if e.get("title")}
        open_titles = {e["title"] for e in open_entries}
        by_title = {e["title"]: e for e in open_entries}
        for entry in stored:
            by_title.setdefault(entry.get("title", ""), entry)
        for title in sorted(set(by_title) | chosen_titles):
            entry = by_title.get(title, {"title": title})
            item = QListWidgetItem(title if title in open_titles else f"{title}  (not open)")
            item.setData(Qt.UserRole, entry)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if title in chosen_titles else Qt.Unchecked)
            widget.addItem(item)
        widget.itemChanged.connect(self._commit)
        return widget

    def _name_list(self, available: list[str], value, *, absent: str) -> QListWidget:
        """A checkable list of names, ticked to *value*.

        A stored name the GUI cannot offer right now (a closed panel, a camera
        another dataset had) is still listed and marked *absent* rather than
        dropped — editing this step on one session must not silently rewrite
        what it does on another.
        """
        widget = QListWidget()
        widget.setMaximumHeight(90)
        chosen = {str(v) for v in value or []}
        for name in sorted(set(available) | chosen):
            item = QListWidgetItem(name if name in available else f"{name}  ({absent})")
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in chosen else Qt.Unchecked)
            widget.addItem(item)
        widget.itemChanged.connect(self._commit)
        return widget

    def _dynamic_choices(self, spec: wf.ParamSpec) -> dict[str, str]:
        """Choices that only exist at runtime — currently the trained models."""
        if spec.key == "model":
            return {name: name for name in om.list_models()} or {"": "no models trained yet"}
        return {}

    def _label_classes(self) -> dict[int, str]:
        mappings = getattr(getattr(self.meta, "labels_widget", None), "_mappings", None) or {}
        return {
            int(label_id): str(info.get("name", label_id))
            for label_id, info in sorted(mappings.items())
            if isinstance(label_id, int) and label_id != 0
        }

    def _read(self, spec: wf.ParamSpec) -> Any:
        widget = self._widgets[spec.key]
        if spec.type == "bool":
            return bool(widget.isChecked())
        if spec.type == "int":
            return int(widget.value())
        if spec.type in ("float", "confidence"):
            return float(widget.value())
        if spec.type == "choice":
            return widget.currentData()
        if spec.type in ("labels", "panels", "cameras"):
            return [
                widget.item(i).data(Qt.UserRole)
                for i in range(widget.count())
                if widget.item(i).checkState() == Qt.Checked
            ]
        if spec.type == "filters":
            return widget.value()
        return widget.text().strip()

    def _commit(self, *_args) -> None:
        if self._step is None:
            return
        for spec in self._step.spec().params:
            self._step.params[spec.key] = self._read(spec)
        self.changed.emit()

    def _capture(self) -> None:
        if self._step is None:
            return
        captured = capture_params(self._step.kind, self.meta)
        # Only what the GUI actually holds: a model name has no live source.
        self._step.params.update({k: v for k, v in captured.items() if v not in ("", None)})
        self.set_step(self._step)
        self.changed.emit()


# ======================================================================
# The dialog
# ======================================================================


class CurationWorkflowDialog(QDialog):
    """Manage the saved curation workflows, and run one."""

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Curation workflows")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)
        self.meta = meta
        self.app_state = meta.app_state
        self.runner = WorkflowRunner(meta, parent=self)
        self.runner.step_started.connect(self._on_step_started)
        self.runner.note.connect(self._log)
        self.runner.finished.connect(self._on_finished)
        self._workflow: wf.CurationWorkflow | None = None
        #: Whether the selected workflow has edits not yet on disk.
        self._dirty = False

        # Editing a step is a stream of small changes (every spin tick); the
        # YAML is written once per burst, and always before it matters.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(_SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self._flush_save)

        outer = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # ── Left: the stored workflows ──────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(QLabel("Workflows"))
        self.workflow_list = QListWidget()
        self.workflow_list.currentTextChanged.connect(self._on_workflow_selected)
        left_lay.addWidget(self.workflow_list, stretch=1)
        # Two rows: four buttons side by side do not fit the workflow column.
        self._selection_buttons: list[QPushButton] = []
        for row in (
            (
                ("New", self._new, "Start an empty workflow", False),
                ("Rename…", self._rename, "Rename the selected workflow", True),
            ),
            (
                ("Copy", self._duplicate, "Copy the selected workflow under a new name", True),
                ("Delete", self._delete, "Delete the selected workflow", True),
            ),
        ):
            row_lay = QHBoxLayout()
            for text, slot, tip, needs_selection in row:
                btn = QPushButton(text)
                btn.setAutoDefault(False)
                btn.setToolTip(tip)
                btn.clicked.connect(slot)
                row_lay.addWidget(btn)
                if needs_selection:
                    self._selection_buttons.append(btn)
            left_lay.addLayout(row_lay)
        splitter.addWidget(left)

        # ── Right: the selected workflow's steps ────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("What this workflow is for")
        self.description_edit.editingFinished.connect(self._on_description)
        right_lay.addWidget(self.description_edit)

        steps_group = QGroupBox("Steps")
        steps_lay = QVBoxLayout(steps_group)
        self.step_list = QListWidget()
        self.step_list.currentRowChanged.connect(self._on_step_selected)
        self.step_list.setMinimumHeight(140)
        steps_lay.addWidget(self.step_list)
        add_row = QHBoxLayout()
        self.kind_combo = QComboBox()
        for key, kind in wf.STEP_KINDS.items():
            self.kind_combo.addItem(kind.title, key)
        add_row.addWidget(self.kind_combo, stretch=1)
        add_btn = QPushButton("Add step")
        add_btn.setAutoDefault(False)
        add_btn.clicked.connect(self._add_step)
        add_row.addWidget(add_btn)
        steps_lay.addLayout(add_row)
        move_row = QHBoxLayout()
        for text, slot in (("↑", lambda: self._move(-1)), ("↓", lambda: self._move(1)), ("Remove", self._remove_step)):
            btn = QPushButton(text)
            btn.setAutoDefault(False)
            btn.clicked.connect(slot)
            move_row.addWidget(btn)
        steps_lay.addLayout(move_row)
        right_lay.addWidget(steps_group)

        self.editor = StepEditor(meta)
        self.editor.changed.connect(self._on_step_edited)
        editor_scroll = QScrollArea()
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setWidget(self.editor)
        right_lay.addWidget(editor_scroll, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([180, 460])

        # ── Bottom: run + log ───────────────────────────────────────
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run workflow")
        self.run_btn.setAutoDefault(False)
        self.run_btn.setToolTip("Walk the steps in order; interactive steps wait for you.")
        self.run_btn.clicked.connect(self._run)
        run_row.addWidget(self.run_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setAutoDefault(False)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(lambda: self.runner.stop("Stopped."))
        run_row.addWidget(self.stop_btn)
        outer.addLayout(run_row)

        self.log_list = QListWidget()
        self.log_list.setMaximumHeight(110)
        outer.addWidget(self.log_list)

        self.resize(720, 640)
        self._refresh_workflows()

    # ------------------------------------------------------------------
    # Workflow list
    # ------------------------------------------------------------------

    def _project(self) -> Path | None:
        return project_dir_of(self.app_state)

    def _refresh_workflows(self, select: str | None = None) -> None:
        self.workflow_list.blockSignals(True)
        self.workflow_list.clear()
        names = wf.list_workflows(self._project())
        self.workflow_list.addItems(names)
        self.workflow_list.blockSignals(False)
        if not names:
            self._set_workflow(None)
            return
        target = select if select in names else names[0]
        self.workflow_list.setCurrentRow(names.index(target))

    def _on_workflow_selected(self, name: str) -> None:
        self._flush_save()  # the one being left behind, before it is replaced
        self._set_workflow(wf.load_workflow(name, self._project()) if name else None)

    def _set_workflow(self, workflow: wf.CurationWorkflow | None) -> None:
        self._workflow = workflow
        self.description_edit.setText(workflow.description if workflow else "")
        self.description_edit.setEnabled(workflow is not None)
        self.run_btn.setEnabled(workflow is not None)
        for btn in self._selection_buttons:
            btn.setEnabled(workflow is not None)
        self._refresh_steps()

    def _save(self) -> None:
        """Write the selected workflow, one file write per burst of edits."""
        if self._workflow is not None:
            self._dirty = True
            self._save_timer.start()

    def _flush_save(self) -> None:
        """Write pending edits now; a no-op when there are none.

        The dirty flag is what stops a deleted or renamed workflow coming
        back: every selection change flushes, and an unconditional write here
        would recreate the file that was just removed.
        """
        self._save_timer.stop()
        if self._workflow is not None and self._dirty:
            wf.save_workflow(self._workflow, self._project())
        self._dirty = False

    def _new(self) -> None:
        name, ok = QInputDialog.getText(self, "New workflow", "Name:")
        if ok and name.strip():
            self._new_named(name)

    def _new_named(self, name: str) -> None:
        """Create an empty workflow called *name* and select it."""
        workflow = wf.CurationWorkflow(name=wf.safe_name(name))
        wf.save_workflow(workflow, self._project())
        self._refresh_workflows(select=workflow.name)

    def _duplicate(self) -> None:
        if self._workflow is None:
            return
        name, ok = QInputDialog.getText(self, "Copy workflow", "New name:", text=f"{self._workflow.name} copy")
        if not ok or not name.strip():
            return
        copy = wf.CurationWorkflow.from_dict(self._workflow.to_dict())
        copy.name = wf.safe_name(name)
        wf.save_workflow(copy, self._project())
        self._refresh_workflows(select=copy.name)

    def _rename(self) -> None:
        if self._workflow is None:
            return
        old = self._workflow.name
        text, ok = QInputDialog.getText(self, "Rename workflow", "New name:", text=old)
        if ok:
            self._rename_to(text)

    def _rename_to(self, text: str) -> None:
        """Store the selected workflow under *text* and drop the old file."""
        if self._workflow is None:
            return
        old = self._workflow.name
        try:
            new = wf.safe_name(text)
        except ValueError as e:
            notify(str(e), severity="warning")
            return
        if new == old:
            return
        self._flush_save()  # pending edits belong to the old file
        wf.rename_workflow(old, new, self._project())
        # So a later flush targets the new file rather than resurrecting the old.
        self._workflow.name = new
        self._refresh_workflows(select=new)

    def _delete(self) -> None:
        if self._workflow is None:
            return
        name = self._workflow.name
        confirm = QMessageBox.question(self, "Delete workflow", f"Delete the workflow {name!r}?")
        if confirm == QMessageBox.Yes:
            self._delete_selected()

    def _delete_selected(self) -> None:
        """Remove the selected workflow's file and leave nothing behind."""
        if self._workflow is None:
            return
        wf.delete_workflow(self._workflow.name, self._project())
        # Dropped before the selection changes: the flush that a selection
        # change triggers must have nothing left to write back.
        self._workflow = None
        self._dirty = False
        self._refresh_workflows()

    def _on_description(self) -> None:
        if self._workflow is None:
            return
        self._workflow.description = self.description_edit.text().strip()
        self._save()

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _refresh_steps(self, select: int | None = None) -> None:
        self.step_list.blockSignals(True)
        self.step_list.clear()
        if self._workflow is not None:
            for i, step in enumerate(self._workflow.steps, start=1):
                self.step_list.addItem(f"{i}. {step.title()} — {describe_step(step)}")
        self.step_list.blockSignals(False)
        if self._workflow and self._workflow.steps:
            row = min(max(select if select is not None else 0, 0), len(self._workflow.steps) - 1)
            self.step_list.setCurrentRow(row)
        else:
            self.editor.set_step(None)

    def _current_step(self) -> wf.WorkflowStep | None:
        row = self.step_list.currentRow()
        if self._workflow is None or not (0 <= row < len(self._workflow.steps)):
            return None
        return self._workflow.steps[row]

    def _on_step_selected(self, _row: int) -> None:
        self.editor.set_step(self._current_step())

    def _on_step_edited(self) -> None:
        """A parameter changed: restate that one row and save, nothing more.

        Rebuilding the step list here would re-select the row and rebuild the
        form under the widget being typed into, so only its text is redone.
        """
        row = self.step_list.currentRow()
        step = self._current_step()
        if step is not None and 0 <= row < self.step_list.count():
            self.step_list.item(row).setText(f"{row + 1}. {step.title()} — {describe_step(step)}")
        self._save()

    def _add_step(self) -> None:
        if self._workflow is None:
            notify("Create a workflow first.", severity="warning")
            return
        kind = str(self.kind_combo.currentData())
        step = wf.WorkflowStep(kind=kind, params=capture_params(kind, self.meta))
        self._workflow.steps.append(step)
        self._save()
        self._refresh_steps(select=len(self._workflow.steps) - 1)

    def _remove_step(self) -> None:
        row = self.step_list.currentRow()
        if self._workflow is None or not (0 <= row < len(self._workflow.steps)):
            return
        del self._workflow.steps[row]
        self._save()
        self._refresh_steps(select=row - 1)

    def _move(self, delta: int) -> None:
        row = self.step_list.currentRow()
        if self._workflow is None:
            return
        target = row + delta
        if not (0 <= row < len(self._workflow.steps)) or not (0 <= target < len(self._workflow.steps)):
            return
        steps = self._workflow.steps
        steps[row], steps[target] = steps[target], steps[row]
        self._save()
        self._refresh_steps(select=target)

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def _run(self) -> None:
        if self._workflow is None:
            return
        self._flush_save()
        problems = wf.validate(self._workflow)
        if problems:
            QMessageBox.warning(self, "Workflow not runnable", "\n".join(problems))
            return
        self.log_list.clear()
        if self.runner.run(self._workflow):
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    def _log(self, message: str) -> None:
        self.log_list.addItem(message)
        self.log_list.scrollToBottom()

    def _on_step_started(self, index: int, title: str) -> None:
        self._log(f"Step {index + 1}: {title}")
        self.step_list.setCurrentRow(index)

    def _on_finished(self, completed: bool) -> None:
        self.run_btn.setEnabled(self._workflow is not None)
        self.stop_btn.setEnabled(False)
        notify("Curation workflow finished." if completed else "Curation workflow stopped.")
        # A workflow can predict hundreds of labels and never save them; say
        # so at the end rather than leaving it to the close prompt.
        if not self.app_state.changes_saved:
            self._log("Labels are NOT saved yet — press Ctrl+S, or add a Save labels step.")
            notify("Workflow done, but the labels are not saved yet — Ctrl+S.", severity="warning")

    def closeEvent(self, event):
        self._flush_save()
        self.runner.stop()
        super().closeEvent(event)
