"""LightGBM onset-model dialogs (Model menu).

Two non-modal dialogs around :mod:`ethograph.labels.onset_model`:

* **Train** — the user names a model, ticks one or more point-event classes
  to predict (state events are out of scope; the model assumes at most one
  event per class per trial), ticks features and, per dim (keypoints,
  individuals, space, …), which values to include — plus, per feature,
  whether its rate of change comes along as an extra input, and which of the
  session's **existing label classes** are read as inputs too (a state class as
  its on/off vector, a point class as a Laplacian bump; a class the model
  predicts is greyed out, since it cannot be its own input). The current session's
  existing point events become training trials stored under
  ``~/.ethograph/defaults/runs/lightgbm/{name}/train_data``; more sessions can be added by
  reopening the dialog there, and Train fits one classifier per class from
  everything collected so far.
* **Predict** — pick a trained model and apply it to the current session. All
  of the model's classes are predicted in one pass; a trial that already
  carries a class is never overridden for that class. Each predicted event
  carries the model's own confidence into the labels TSV. The chosen
  **individual** is both whose data is read and whose events are written, so a
  model trained on one animal runs on another's session with the same rig
  (``onset_model.retarget_individual``) — the classifier is fitted on numbers,
  and the individual is only the key that selects them.

Both dialogs run over **the trials the trials table shows** — its filters are
the one place trials are included or excluded for every operation (see
``docs/source/advanced/metadata.md``); neither dialog has filters of its own.

Extraction runs per trial through the same ``DataLoader.select`` path the
plots use, but on throwaway loaders holding no display offset, so it never
disturbs the GUI's navigation state.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ethograph.gui.dialog_label_gridview import ConfidenceEdit, confidence_display
from ethograph.gui.file_dialogs import browse_open_file
from ethograph.gui.notify import notify
from ethograph.io.catalog import PynappleLoader, XarrayLoader
from ethograph.labels import onset_curves
from ethograph.labels import onset_model as om
from ethograph.labels.curve_confidence import DESCRIPTIONS
from ethograph.labels.intervals import (
    EVENT_TYPE_POINT,
    EVENT_TYPE_STATE,
    LABELING_AUTOMATED,
    NO_RECIPIENT,
    add_point,
)
from ethograph.labels.label_inputs import POINT_SIGMAS_S, LabelInput
from ethograph.labels.tsv_store import get_trial_from_tsv
from ethograph.labels.workflow import DEFAULT_CONFIDENCE
from ethograph.utils.paths import session_id

logger = logging.getLogger(__name__)

_CAVEAT = (
    "Only point events can be predicted, and the model assumes each trial "
    "contains each ticked event at most once. Trials that do not carry a "
    "ticked event simply contribute nothing to that event's classifier."
)


# ---------------------------------------------------------------------------
# Session access helpers
# ---------------------------------------------------------------------------


def _base_loader(app_state):
    """The real DataLoader behind the console's DerivedLoader wrapper."""
    loader = getattr(app_state, "data_loader", None)
    return getattr(loader, "base", loader)


def _iter_trial_windows(app_state) -> Iterator[tuple[int | str, object, float | None, float | None, float]]:
    """Yield ``(trial_id, loader, t0, t1, shift)`` per visible trial.

    ``loader.select(feature, sel, t0, t1)`` returns times on the loader's own
    clock; subtracting *shift* makes them trial-relative — the clock labels
    are stored in. Loaders are throwaway (no display-offset provider), so the
    GUI's navigation state is untouched.
    """
    base = _base_loader(app_state)
    if base is None:
        return
    if base.backend == "xarray":
        dt = app_state.dt
        for tid in app_state.trials:
            yield tid, XarrayLoader(dt.trial(tid)), None, None, 0.0
    else:
        sc = getattr(app_state, "source_collection", None)
        fresh = PynappleLoader(base.data, base.catalog)
        for tid in app_state.trials:
            idx = sc.trial_index(tid) if sc is not None else None
            if idx is None:
                continue
            trial_range = sc.trial_range(idx)
            yield tid, fresh, trial_range.start_s, trial_range.end_s, trial_range.start_s


def _point_rows(df: pd.DataFrame, label_id: int) -> pd.DataFrame:
    """Rows of *df* that are the target point event."""
    if df is None or df.empty:
        return pd.DataFrame()
    mask = df["labels"] == label_id
    if "event_type" in df.columns:
        mask &= df["event_type"] == EVENT_TYPE_POINT
    else:
        mask &= df["offset_s"].isna()
    return df[mask]


@dataclass
class PredictionOutcome:
    """What one prediction run wrote, and what it deliberately did not.

    Returned by :func:`predict_onsets` so the Predict dialog and a curation
    workflow report the same run the same way, and so a workflow's later
    steps can pick up exactly the classes and trials it produced.
    """

    model: str
    per_target: dict[int, int]
    target_names: dict[int, str]
    n_predicted: int = 0
    n_existing: int = 0
    n_low: int = 0
    n_absent: int = 0
    trials: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    #: Set only when the model was trained on a different individual than the
    #: one it just read — worth saying, since it is the run's biggest caveat.
    trained_on: str = ""

    def label_ids(self) -> list[int]:
        """The classes this run actually wrote at least one event for."""
        return [label for label, n in self.per_target.items() if n]

    def message(self) -> str:
        per_target_text = ", ".join(f"{self.target_names[label]}: {n}" for label, n in self.per_target.items())
        parts = [f"Predicted {self.n_predicted} onsets ({per_target_text})."]
        if self.n_existing:
            parts.append(f"{self.n_existing} trial/class pairs already labelled (untouched).")
        if self.n_low:
            parts.append(f"{self.n_low} below the confidence threshold.")
        if self.n_absent:
            parts.append(f"{self.n_absent} the model produced no prediction for.")
        if self.errors:
            parts.append(f"{len(self.errors)} trials failed — first: {self.errors[0]}")
        if self.trained_on:
            parts.append(f"The model was trained on {self.trained_on}.")
        return " ".join(parts)


def _save_curves(app_state, curves_by_trial: dict, config: om.OnsetModelConfig, applied: dict) -> Path | None:
    """Write this run's probability curves to its own folder under ``labels/``.

    An aid to review, not part of the prediction: a session with no path on
    disk yet simply keeps none, and a failed write is logged rather than
    losing the predictions it belongs to. *config* is the trained bundle's —
    what actually ran, not the model folder's editable ``config.yaml`` — and
    *applied* how it was run; both land beside the curves so the folder
    says what produced it.
    """
    if not curves_by_trial or not app_state.nc_file_path:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = onset_curves.run_dir(app_state.nc_file_path, timestamp)
    try:
        written = onset_curves.write_curves(folder / onset_curves.CURVES_FILE, curves_by_trial)
        onset_curves.write_provenance(folder, model_config=asdict(config), inference=applied)
        return written
    except OSError as exc:
        logger.warning("Could not write onset curves to %s: %s", folder, exc)
        return None


def ensure_curve_run_chosen(parent, app_state) -> None:
    """After a predict run, make sure ``app_state.curve_run_path`` names one run.

    Frame-by-frame review (``CurationPanel._resolve_curve_source``) never
    asks — it only reads this. Asked here instead, once: with at most one run
    to choose from it resolves silently; with several and nothing picked yet,
    this prompts and the answer sticks in ``curve_run_path`` (session-only)
    for the rest of the session, however many more predict runs follow.
    """
    session = app_state.nc_file_path
    if not session:
        return
    current = app_state.curve_run_path
    if current and Path(current).is_file():
        return
    folders = onset_curves.run_dirs(session)
    if not folders:
        return
    if len(folders) == 1:
        app_state.curve_run_path = str(folders[0] / onset_curves.CURVES_FILE)
        return

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle("Multiple prediction runs")
    box.setText(
        f"This session now has {len(folders)} prediction runs with confidence curves.\n"
        "Pick which run's curves frame-by-frame review should show — remembered for "
        "the rest of this session."
    )
    pick_btn = box.addButton("Pick file…", QMessageBox.AcceptRole)
    box.addButton("Don't show curves", QMessageBox.RejectRole)
    box.exec()
    if box.clickedButton() is not pick_btn:
        return
    chosen = browse_open_file(
        parent,
        app_state,
        "Choose onset_curves.npz",
        f"Onset curves ({onset_curves.CURVES_FILE})",
        preferred_dir=onset_curves.labels_dir(session),
    )
    if chosen:
        app_state.curve_run_path = chosen


def predict_onsets(
    meta,
    name: str,
    *,
    individual: str,
    min_confidence: float,
) -> PredictionOutcome | None:
    """Apply the trained model *name* to every trial the trials table shows.

    *individual* is both whose features are read and whose events are written:
    the model's own individual pinning is re-pointed at them
    (:func:`~ethograph.labels.onset_model.retarget_individual`), which is what
    lets one animal's model run on another's session. Each class is filled
    independently and a trial already carrying one is never overridden for it.
    Returns ``None`` when the model cannot be loaded, when it reads several
    individuals at once, or when *individual* names somebody this session does
    not have — the reason is notified.
    """
    app_state = meta.app_state
    known = app_state.label_individuals()
    if individual not in known:
        # A label belongs to one (actor, receiver) pair and the overlay draws
        # only the selected pair, so an event written for somebody this
        # session has never heard of is stored and never seen. Say so instead
        # of producing invisible labels — the usual cause is a workflow copied
        # between animals with its predict step's individual left behind.
        notify(
            f"The model would write its events for {individual!r}, but this session labels "
            f"{', '.join(known)} — those events would never be drawn. "
            "Set the individual in the Predict dialog, or in the workflow's Predict onsets step.",
            severity="warning",
        )
        return None
    try:
        bundle = om.load_bundle(name)
    except ValueError as e:
        notify(str(e), severity="warning")
        return None
    config = om.bundle_config(bundle)
    trained_on = om.config_individuals(config)
    try:
        # The columns the classifier was fitted on, re-pointed at this
        # session's individual — same features, same order, another animal.
        read_config = om.retarget_individual(config, individual)
    except ValueError as e:
        notify(str(e), severity="warning")
        return None
    outcome = PredictionOutcome(
        model=name,
        per_target={label: 0 for label in config.targets},
        target_names={label: config.target_name(label) for label in config.targets},
        trained_on=trained_on[0] if trained_on and trained_on != [individual] else "",
    )

    df = getattr(app_state, "_all_labels_df", None)
    #: Every trial's probability curves, written beside the labels once the
    #: run is over — what frame-by-frame review draws under each label.
    curves_by_trial: dict[object, tuple[np.ndarray, dict[int, np.ndarray]]] = {}
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        for tid, loader, t0, t1, shift in _iter_trial_windows(app_state):
            trial_rows = df[df["trial"] == tid] if df is not None else None
            # Each class is filled independently: a trial already carrying
            # one of them can still receive the others.
            wanted = [label for label in config.targets if trial_rows is None or _point_rows(trial_rows, label).empty]
            outcome.n_existing += len(config.targets) - len(wanted)
            if not wanted:
                continue
            try:
                time, data = om.extract_model_features(loader, read_config, t0, t1, labels=trial_rows, shift=shift)
                trial_time = time - shift
                result = om.predict_trial(bundle, trial_time, data)
                predictions = result.events
            except ValueError as e:
                outcome.errors.append(f"trial {tid}: {e}")
                continue
            curves_by_trial[tid] = (trial_time, result.curves)
            written = False
            for label in wanted:
                prediction = predictions.get(label)
                if prediction is None:
                    outcome.n_absent += 1
                    continue
                if prediction.confidence < min_confidence:
                    outcome.n_low += 1
                    continue
                trial_df = get_trial_from_tsv(app_state._all_labels_df, tid)
                trial_df = add_point(
                    trial_df,
                    prediction.time,
                    label,
                    individual,
                    NO_RECIPIENT,
                    confidence=prediction.confidence,
                    labeling_method=LABELING_AUTOMATED,
                )
                app_state.set_trial_intervals(tid, trial_df)
                outcome.per_target[label] += 1
                outcome.n_predicted += 1
                written = True
                outcome.trials.add(str(tid))
                df = app_state._all_labels_df
            if written:
                app_state.set_trial_meta_attr(tid, "prediction_source", f"lightgbm:{name}")
    finally:
        QApplication.restoreOverrideCursor()

    _save_curves(
        app_state,
        curves_by_trial,
        config,
        {
            "model": onset_curves.LIGHTGBM,
            "name": name,
            "individual": individual,
            "min_confidence": float(min_confidence),
            "session": str(app_state.nc_file_path),
        },
    )
    if outcome.n_predicted:
        refresh_after_prediction(meta)
    if outcome.errors:
        logger.warning("Onset prediction failures: %s", "; ".join(outcome.errors))
    notify(outcome.message(), severity="warning" if outcome.errors else "info")
    if outcome.n_predicted:
        notify("Review the predictions and save with Ctrl+S.")
    return outcome


def refresh_after_prediction(meta) -> None:
    """Show freshly written predictions: current trial, plots, label shapes."""
    app_state = meta.app_state
    app_state.changes_saved = False
    current = getattr(app_state, "trials_sel", None)
    if current is not None:
        app_state.label_intervals = get_trial_from_tsv(app_state._all_labels_df, current)
    data_widget = getattr(meta, "data_widget", None)
    if data_widget is not None:
        data_widget.update_main_plot(preserve_x_range=True)
    labels_widget = getattr(meta, "labels_widget", None)
    if labels_widget is not None:
        labels_widget.refresh_labels_shapes_layer()


def _all_mappings(app_state) -> dict[int, tuple[str, str]]:
    """Every label class: ``{label_id: (name, event_type)}``."""
    mappings = getattr(app_state, "_label_mappings", None) or {}
    return {
        int(label_id): (str(info.get("name", label_id)), str(info.get("event_type", EVENT_TYPE_STATE)))
        for label_id, info in sorted(mappings.items())
        if isinstance(label_id, int) and label_id != 0
    }


def _point_mappings(app_state) -> dict[int, str]:
    """Point-event label classes: ``{label_id: name}``."""
    return {
        label_id: name
        for label_id, (name, event_type) in _all_mappings(app_state).items()
        if event_type == EVENT_TYPE_POINT
    }


# ---------------------------------------------------------------------------
# Feature tree — feature ▸ dim ▸ value, all checkable
# ---------------------------------------------------------------------------


#: Role holding a feature item's implicit dims: those with only one possible
#: value, so there is nothing to choose and no row is drawn for them at all.
_IMPLICIT_DIMS_ROLE = Qt.UserRole

#: Column carrying each feature's "include its derivative too" tick. It is a
#: checkbox *widget*, not the item's own check state: an auto-tristate item
#: with children reports its children's state in every column, which would
#: leave the box unrendered on any feature that has dim rows.
_DERIVATIVE_COLUMN = 1

_DERIVATIVE_TOOLTIP = (
    "Also feed the classifier this feature's rate of change (np.gradient —\n"
    "central differences, centred on the frame, so a turn in the signal shows\n"
    "up at the frame it happened). The window taps are seen in isolation, so a\n"
    "boosted tree cannot difference them itself."
)


class FeatureTree(QTreeWidget):
    """Checkable tree of features, their dims, and each dim's values.

    Auto-tristate makes the feature row an "All" toggle and dim rows show the
    partial state — the checked values per dim are exactly the subset the
    model config stores (see ``onset_model.enumerate_columns``). A dim with
    only one possible value (e.g. a single-individual dataset's "individual")
    is never drawn as a row — it is included automatically whenever its
    feature is checked, since there is nothing to choose between.

    Each feature row also carries a **d/dt** tick: that feature's time
    derivative then joins its value as an extra input column
    (``OnsetModelConfig.derivatives``).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["Feature", "d/dt"])
        self.headerItem().setToolTip(_DERIVATIVE_COLUMN, _DERIVATIVE_TOOLTIP)
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(_DERIVATIVE_COLUMN, QHeaderView.ResizeToContents)

    def _add_derivative_check(self, item: QTreeWidgetItem, checked: bool = False) -> None:
        box = QCheckBox()
        box.setChecked(checked)
        box.setToolTip(_DERIVATIVE_TOOLTIP)
        self.setItemWidget(item, _DERIVATIVE_COLUMN, box)

    def populate_from_loader(self, app_state) -> None:
        """One top-level item per catalog feature (derived features excluded —
        console recipes live for one trial and cannot travel between sessions)."""
        self.clear()
        loader = getattr(app_state, "data_loader", None)
        base = _base_loader(app_state)
        if base is None:
            return
        derived = getattr(loader, "derived", None) or {}
        for feature in base.catalog.feature_choices():
            if feature in derived:
                continue
            item = QTreeWidgetItem(self, [feature])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            item.setCheckState(0, Qt.Unchecked)
            self._add_derivative_check(item)
            implicit: dict[str, list[str]] = {}
            for dim, values in (base.feature_dims(feature) or {}).items():
                if len(values) == 1:
                    implicit[dim] = [str(values[0])]
                    continue
                dim_item = QTreeWidgetItem(item, [dim])
                dim_item.setFlags(dim_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                for value in values:
                    value_item = QTreeWidgetItem(dim_item, [str(value)])
                    value_item.setFlags(value_item.flags() | Qt.ItemIsUserCheckable)
                    value_item.setCheckState(0, Qt.Unchecked)
            if implicit:
                item.setData(0, _IMPLICIT_DIMS_ROLE, implicit)

    def populate_from_config(self, config: om.OnsetModelConfig) -> None:
        """Read-only view of a frozen config (existing model)."""
        self.clear()
        for feature, dims in config.features.items():
            item = QTreeWidgetItem(self, [feature])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            item.setCheckState(0, Qt.Checked)
            self._add_derivative_check(item, feature in config.derivatives)
            implicit: dict[str, list[str]] = {}
            for dim, values in dims.items():
                if len(values) == 1:
                    implicit[dim] = list(values)
                    continue
                dim_item = QTreeWidgetItem(item, [dim])
                for value in values:
                    value_item = QTreeWidgetItem(dim_item, [str(value)])
                    value_item.setFlags(value_item.flags() | Qt.ItemIsUserCheckable)
                    value_item.setCheckState(0, Qt.Checked)
            if implicit:
                item.setData(0, _IMPLICIT_DIMS_ROLE, implicit)
        self.expandAll()

    def selected_features(self) -> dict[str, dict[str, list[str]]]:
        """The checked subset as a model config. Raises ``ValueError`` when a
        checked feature leaves one of its dims with no values."""
        out: dict[str, dict[str, list[str]]] = {}
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if item.checkState(0) == Qt.Unchecked:
                continue
            feature = item.text(0)
            dims: dict[str, list[str]] = dict(item.data(0, _IMPLICIT_DIMS_ROLE) or {})
            for j in range(item.childCount()):
                dim_item = item.child(j)
                values = [
                    dim_item.child(k).text(0)
                    for k in range(dim_item.childCount())
                    if dim_item.child(k).checkState(0) == Qt.Checked
                ]
                if not values:
                    raise ValueError(f"Feature {feature!r}: no values ticked for dim {dim_item.text(0)!r}.")
                dims[dim_item.text(0)] = values
            out[feature] = dims
        if not out:
            raise ValueError("Tick at least one feature.")
        return out

    def selected_derivatives(self) -> list[str]:
        """Ticked features whose d/dt is ticked too, in tree order."""
        out: list[str] = []
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            box = self.itemWidget(item, _DERIVATIVE_COLUMN)
            if item.checkState(0) != Qt.Unchecked and box is not None and box.isChecked():
                out.append(item.text(0))
        return out


# ---------------------------------------------------------------------------
# Label-input tree — label class ▸ individual, all checkable
# ---------------------------------------------------------------------------


#: Role holding a top-level item's ``(label_id, name, event_type)``.
_LABEL_ROLE = Qt.UserRole

_LABEL_INPUT_HINT = (
    "Feed the classifier the labels this session already has: a state class as "
    "its on/off vector, a point class as a Laplacian bump centred on it "
    f"(sigma {', '.join(f'{s:g} s' for s in POINT_SIGMAS_S)} — one column each). "
    "Tick the individuals whose labels count; a class the model predicts cannot "
    "be one of its own inputs."
)


class LabelInputTree(QTreeWidget):
    """Checkable tree of existing label classes and, per class, whose labels to read.

    Same shape as :class:`FeatureTree`: the class row is auto-tristate, so it
    doubles as that class's "all individuals" toggle, and its children are the
    individuals. A single-individual session draws no children at all — there
    is nothing to choose, exactly as a feature's single-valued dim is never
    drawn as a row.

    A class ticked as a **target** is unchecked and disabled here: at training
    its label is present and at inference it is not, so feeding it back would
    hand the classifier a column that means opposite things on the two sides.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)

    @staticmethod
    def _row(parent, label: int, name: str, event_type: str, individuals: list[str], checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        item = QTreeWidgetItem(parent, [f"{name} ({label}) — {event_type}"])
        item.setData(0, _LABEL_ROLE, (label, name, event_type))
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
        item.setCheckState(0, state)
        for who in individuals:
            child = QTreeWidgetItem(item, [who])
            child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
            child.setCheckState(0, state)

    def populate(self, app_state) -> None:
        """One row per label class this session knows, nothing ticked."""
        self.clear()
        individuals = list(app_state.label_individuals())
        nested = individuals if len(individuals) > 1 else []
        for label, (name, event_type) in _all_mappings(app_state).items():
            self._row(self, label, name, event_type, nested, checked=False)

    def populate_from_config(self, config: om.OnsetModelConfig) -> None:
        """Read-only view of a frozen config's label inputs."""
        self.clear()
        for inp in config.label_inputs:
            self._row(self, inp.label, inp.name, inp.event_type, list(inp.individuals), checked=True)
        self.expandAll()

    def set_excluded(self, label_ids: set[int]) -> None:
        """Uncheck and grey out the classes in *label_ids* (the model's targets)."""
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            label, name, event_type = item.data(0, _LABEL_ROLE)
            excluded = label in label_ids
            if excluded and item.checkState(0) != Qt.Unchecked:
                item.setCheckState(0, Qt.Unchecked)
            item.setDisabled(excluded)
            suffix = "  (predicted — cannot be its own input)" if excluded else ""
            item.setText(0, f"{name} ({label}) — {event_type}{suffix}")

    def set_all_checked(self, checked: bool) -> None:
        """Tick (or clear) every class that is not a target."""
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if not item.isDisabled():
                item.setCheckState(0, state)

    def selected_inputs(self) -> list[LabelInput]:
        """The ticked classes as config entries. Raises ``ValueError`` when a
        ticked class leaves no individual ticked."""
        out: list[LabelInput] = []
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if item.isDisabled() or item.checkState(0) == Qt.Unchecked:
                continue
            label, name, event_type = item.data(0, _LABEL_ROLE)
            individuals = [
                item.child(j).text(0) for j in range(item.childCount()) if item.child(j).checkState(0) == Qt.Checked
            ]
            if item.childCount() and not individuals:
                raise ValueError(f"Label input {name!r}: no individuals ticked.")
            out.append(LabelInput(label=label, name=name, event_type=event_type, individuals=individuals))
        return out


# ---------------------------------------------------------------------------
# Train dialog
# ---------------------------------------------------------------------------


class TrainOnsetDialog(QDialog):
    """Create a LightGBM lightgbm model, collect training data, and train it."""

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LightGBM: Train onset detector")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)
        self.app_state = meta.app_state
        self._config: om.OnsetModelConfig | None = None

        layout = QVBoxLayout(self)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("New model…")
        for name in om.list_models():
            self.model_combo.addItem(name)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo, stretch=1)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("model name")
        model_row.addWidget(self.name_edit, stretch=1)
        layout.addLayout(model_row)

        self.copy_widget = QGroupBox()
        self.copy_widget.setFlat(True)
        copy_row = QHBoxLayout(self.copy_widget)
        copy_row.setContentsMargins(0, 0, 0, 0)
        copy_row.addWidget(QLabel("Copy config from:"))
        self.copy_combo = QComboBox()
        self.copy_combo.addItem("")
        for name in om.list_models():
            self.copy_combo.addItem(name)
        copy_row.addWidget(self.copy_combo, stretch=1)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setAutoDefault(False)
        self.copy_btn.setToolTip(
            "Load another model's targets, features and parameters as a\n"
            "starting point for this new model — freely editable afterwards,\n"
            "since nothing is saved until Add training data or Train runs."
        )
        self.copy_btn.clicked.connect(self._copy_config)
        copy_row.addWidget(self.copy_btn)
        layout.addWidget(self.copy_widget)

        # Two columns: what the model *is* on the left (what it predicts, what
        # it reads), what the run *does* on the right (the extra inputs, the
        # parameters, the sessions). The feature tree is the one control that
        # wants all the height it can get, so it takes the left column's slack.
        columns = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        columns.addLayout(left, stretch=1)
        columns.addLayout(right, stretch=1)
        layout.addLayout(columns, stretch=1)

        target_group = QGroupBox("1 — Point events to predict")
        target_lay = QVBoxLayout(target_group)
        self.target_list = QListWidget()
        self.target_list.setMaximumHeight(110)
        self.target_list.setToolTip("Tick the classes to predict.")
        for label_id, name in _point_mappings(self.app_state).items():
            item = QListWidgetItem(f"{name} ({label_id})")
            item.setData(Qt.UserRole, label_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.target_list.addItem(item)
        target_lay.addWidget(self.target_list)
        caveat = QLabel(_CAVEAT)
        caveat.setWordWrap(True)
        caveat.setStyleSheet("color: grey; font-size: 10px;")
        target_lay.addWidget(caveat)
        left.addWidget(target_group)

        feature_group = QGroupBox("2 — Features (tick dims/values; d/dt adds the rate of change)")
        feature_lay = QVBoxLayout(feature_group)
        self.tree = FeatureTree()
        self.tree.populate_from_loader(self.app_state)
        feature_lay.addWidget(self.tree)
        left.addWidget(feature_group, stretch=1)

        label_group = QGroupBox("3 — Existing labels as inputs (optional)")
        label_lay = QVBoxLayout(label_group)
        label_hint = QLabel(_LABEL_INPUT_HINT)
        label_hint.setWordWrap(True)
        label_hint.setStyleSheet("color: grey; font-size: 10px;")
        label_lay.addWidget(label_hint)
        self.label_tree = LabelInputTree()
        self.label_tree.setMaximumHeight(140)
        self.label_tree.populate(self.app_state)
        label_lay.addWidget(self.label_tree)
        label_buttons = QHBoxLayout()
        self.label_all_btn = QPushButton("Select all")
        self.label_all_btn.setAutoDefault(False)
        self.label_all_btn.clicked.connect(lambda: self.label_tree.set_all_checked(True))
        label_buttons.addWidget(self.label_all_btn)
        self.label_none_btn = QPushButton("Clear")
        self.label_none_btn.setAutoDefault(False)
        self.label_none_btn.clicked.connect(lambda: self.label_tree.set_all_checked(False))
        label_buttons.addWidget(self.label_none_btn)
        label_buttons.addStretch(1)
        label_lay.addLayout(label_buttons)
        right.addWidget(label_group)
        # A class cannot be both predicted and read back, so the target ticks
        # drive what this tree offers.
        self.target_list.itemChanged.connect(self._sync_label_inputs)
        self._sync_label_inputs()

        params_group = QGroupBox("4 — Parameters")
        params_lay = QFormLayout(params_group)
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.05, 30.0)
        self.window_spin.setSingleStep(0.1)
        self.window_spin.setSuffix(" s")
        self.window_spin.setValue(0.5)
        self.window_spin.setToolTip("Width of the feature window the classifier sees around each frame")
        params_lay.addRow("Window size:", self.window_spin)
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.005, 5.0)
        self.tolerance_spin.setDecimals(3)
        self.tolerance_spin.setSingleStep(0.01)
        self.tolerance_spin.setSuffix(" s")
        self.tolerance_spin.setValue(0.05)
        self.tolerance_spin.setToolTip(
            "Frames within this distance of the labelled event count as positive;\n"
            "their sample weight is a Gaussian bump peaking at the event."
        )
        params_lay.addRow("Tolerance:", self.tolerance_spin)
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 2000)
        self.max_iter_spin.setValue(200)
        params_lay.addRow("Boosting iterations:", self.max_iter_spin)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.01, 1.0)
        self.lr_spin.setSingleStep(0.05)
        self.lr_spin.setValue(0.1)
        params_lay.addRow("Learning rate:", self.lr_spin)
        right.addWidget(params_group)

        data_group = QGroupBox("5 — Training data")
        data_lay = QVBoxLayout(data_group)
        self.session_list = QListWidget()
        self.session_list.setMaximumHeight(90)
        data_lay.addWidget(self.session_list)
        self.add_btn = QPushButton("Add current session's events")
        self.add_btn.setAutoDefault(False)
        self.add_btn.setToolTip("Only trials visible in the trials table contribute.")
        self.add_btn.clicked.connect(self._add_session)
        data_lay.addWidget(self.add_btn)
        right.addWidget(data_group)
        right.addStretch(1)

        self.train_btn = QPushButton("Train")
        self.train_btn.setAutoDefault(False)
        self.train_btn.clicked.connect(self._train)
        layout.addWidget(self.train_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.resize(1040, 720)
        self._on_model_changed(self.model_combo.currentText())

    # ------------------------------------------------------------------

    def _config_widgets(self):
        return (
            self.name_edit,
            self.target_list,
            self.tree,
            self.label_tree,
            self.label_all_btn,
            self.label_none_btn,
            self.window_spin,
            self.tolerance_spin,
            self.max_iter_spin,
            self.lr_spin,
        )

    def _on_model_changed(self, text: str):
        """New model: everything editable. Existing model: config is frozen —
        the stored feature layout defines the classifier's input columns, so
        editing it would invalidate every stored training trial."""
        is_new = self.model_combo.currentIndex() == 0
        for widget in self._config_widgets():
            widget.setEnabled(is_new)
        self.copy_widget.setVisible(is_new)
        self._config = None
        self.session_list.clear()
        if is_new:
            self.tree.populate_from_loader(self.app_state)
            self.label_tree.populate(self.app_state)
            self._sync_label_inputs()
            self.status_label.setText("")
            return
        self._config = om.load_config(text)
        self.name_edit.setText(self._config.name)
        self._show_config_targets(self._config)
        self.tree.populate_from_config(self._config)
        self.label_tree.populate_from_config(self._config)
        self.window_spin.setValue(self._config.window_s)
        self.tolerance_spin.setValue(self._config.tolerance_s)
        self.max_iter_spin.setValue(self._config.max_iter)
        self.lr_spin.setValue(self._config.learning_rate)
        self._refresh_sessions()
        trained = "trained" if om.is_trained(text) else "not trained yet"
        self.status_label.setText(f"Config is frozen for an existing model ({trained}).")

    def _show_config_targets(self, config: om.OnsetModelConfig):
        """Read-only view of a frozen model's targets."""
        self.target_list.clear()
        for label_id, name in config.targets.items():
            item = QListWidgetItem(f"{name} ({label_id})")
            item.setData(Qt.UserRole, label_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.target_list.addItem(item)

    def _sync_label_inputs(self, *_args):
        """Keep the target ticks and the label-input tree consistent."""
        self.label_tree.set_excluded(set(self._selected_targets()))

    def _selected_targets(self) -> dict[int, str]:
        """The ticked point-event classes as ``{label_id: name}``."""
        targets: dict[int, str] = {}
        for i in range(self.target_list.count()):
            item = self.target_list.item(i)
            if item.checkState() == Qt.Checked:
                targets[int(item.data(Qt.UserRole))] = item.text().rsplit(" (", 1)[0]
        return targets

    def _refresh_sessions(self):
        self.session_list.clear()
        if self._config is None:
            return
        for session, meta_dict in om.list_sessions(self._config.name).items():
            n = meta_dict.get("n_trials", "?")
            self.session_list.addItem(QListWidgetItem(f"{session} — {n} trials"))

    def _copy_config(self):
        """Load another model's targets/features/parameters as an editable
        starting point for this *new* model. Nothing is saved until Add
        training data or Train runs, so the copy can be freely reshaped."""
        name = self.copy_combo.currentText()
        if not name:
            return
        try:
            source = om.load_config(name)
        except ValueError as e:
            notify(str(e), severity="warning")
            return

        available = {int(self.target_list.item(i).data(Qt.UserRole)): i for i in range(self.target_list.count())}
        for label_id, i in available.items():
            self.target_list.item(i).setCheckState(Qt.Checked if label_id in source.targets else Qt.Unchecked)
        missing = [n for label_id, n in source.targets.items() if label_id not in available]

        self.tree.populate_from_config(source)
        self.label_tree.populate_from_config(source)
        self._sync_label_inputs()
        self.window_spin.setValue(source.window_s)
        self.tolerance_spin.setValue(source.tolerance_s)
        self.max_iter_spin.setValue(source.max_iter)
        self.lr_spin.setValue(source.learning_rate)

        msg = f"Copied config from {name!r} — edit freely, then Train to save as a new model."
        if missing:
            msg += f" Not available in this session's point events: {', '.join(missing)}."
        self.status_label.setText(msg)

    def _ensure_config(self) -> om.OnsetModelConfig | None:
        """The active config, creating + saving a new model's on first use."""
        if self._config is not None:
            return self._config
        name = self.name_edit.text().strip()
        if not name:
            notify("Give the model a name first.", severity="warning")
            return None
        if name in om.list_models():
            notify(f"A model named {name!r} already exists — pick it from the combo to extend it.", severity="warning")
            return None
        targets = self._selected_targets()
        if not targets:
            notify(
                "Tick at least one point-event class — mark a label as a point event first if none are listed.",
                severity="warning",
            )
            return None
        try:
            features = self.tree.selected_features()
            label_inputs = self.label_tree.selected_inputs()
        except ValueError as e:
            notify(str(e), severity="warning")
            return None
        derivatives = self.tree.selected_derivatives()
        self._config = om.OnsetModelConfig(
            name=name,
            targets=targets,
            features=features,
            derivatives=derivatives,
            label_inputs=label_inputs,
            window_s=float(self.window_spin.value()),
            tolerance_s=float(self.tolerance_spin.value()),
            max_iter=int(self.max_iter_spin.value()),
            learning_rate=float(self.lr_spin.value()),
        )
        om.save_config(self._config)
        self.model_combo.addItem(name)
        self.model_combo.setCurrentText(name)  # re-enters _on_model_changed → frozen view
        return self._config

    # ------------------------------------------------------------------

    def _add_session(self):
        config = self._ensure_config()
        if config is None:
            return
        source_path = getattr(self.app_state, "nc_file_path", None)
        if not source_path:
            notify("No dataset loaded.", severity="warning")
            return
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is None or df.empty:
            notify("This session has no labels.", severity="warning")
            return

        session = session_id(source_path)
        n_written = 0
        n_multi = 0
        per_target: dict[int, int] = {label: 0 for label in config.targets}
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for tid, loader, t0, t1, shift in _iter_trial_windows(self.app_state):
                trial_rows = df[df["trial"] == tid]
                y_times: dict[int, float] = {}
                for label in config.targets:
                    rows = _point_rows(trial_rows, label)
                    if rows.empty:
                        continue
                    if len(rows) > 1:
                        n_multi += 1
                    y_times[label] = float(rows.iloc[0]["onset_s"])
                    per_target[label] += 1
                if not y_times:
                    continue
                time, data = om.extract_model_features(loader, config, t0, t1, labels=trial_rows, shift=shift)
                om.write_trial_training_data(config.name, session, tid, time - shift, data, y_times)
                n_written += 1
        except ValueError as e:
            notify(f"Extraction failed: {e}", severity="error")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not n_written:
            notify(f"No trials carry any of: {config.describe_targets()}.", severity="warning")
            return
        om.write_session_meta(
            config.name,
            session,
            {
                "source_path": str(source_path),
                "n_trials": n_written,
                "columns": config.column_names(),
                "added": datetime.now().isoformat(timespec="seconds"),
            },
        )
        self._refresh_sessions()
        per_target_text = ", ".join(f"{config.target_name(label)}: {n}" for label, n in per_target.items())
        msg = f"Stored {n_written} training trials from this session ({per_target_text})."
        if n_multi:
            msg += f" {n_multi} trials had multiple events of one class — only the first was used."
        self.status_label.setText(msg)
        notify(msg)

    def _train(self):
        config = self._ensure_config()
        if config is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            summary = om.train_model(config.name)
        except ValueError as e:
            notify(f"Training failed: {e}", severity="error")
            return
        finally:
            QApplication.restoreOverrideCursor()
        per_target = ", ".join(
            f"{config.target_name(label)}: {stats['n_trials']} trials / {stats['n_positive']} positive frames"
            for label, stats in summary["targets"].items()
        )
        msg = (
            f"Trained {config.name!r} on {summary['n_trials']} trials "
            f"from {summary['n_sessions']} session(s) — {per_target}."
        )

        # The held-out record is what every predicted confidence is scaled by,
        # so it belongs in the one message the user reads after training.
        def _held_out(label: int, cal: dict) -> str:
            ceiling = confidence_display((cal["n_hits"] + 1) / (cal["n_trials"] + 2))
            statistic = cal.get("statistic", "peak")
            return (
                f"{config.target_name(label)}: {cal['n_hits']}/{cal['n_trials']} "
                f"within {config.tolerance_s:g} s (confidence ceiling {ceiling}; "
                f"confidence = {DESCRIPTIONS[statistic]})"
            )

        held_out = ", ".join(_held_out(label, cal) for label, cal in summary.get("calibration", {}).items())
        if held_out:
            msg += f" On trials it did not see — {held_out}."
        self.status_label.setText(msg)
        notify(msg)


# ---------------------------------------------------------------------------
# Predict dialog
# ---------------------------------------------------------------------------


class PredictOnsetDialog(QDialog):
    """Apply a trained lightgbm model to the current session.

    Every class the model was trained on is predicted in one pass, over the
    trials the trials table shows. A trial that already carries a class is
    skipped for that class — the model only fills gaps, it never overrides.
    Each written event carries the model's confidence in the label's
    ``confidence`` column.
    """

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LightGBM: Predict onsets")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)
        self.meta = meta
        self.app_state = meta.app_state

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.model_combo = QComboBox()
        for name in om.list_models():
            self.model_combo.addItem(name)
        self.model_combo.currentTextChanged.connect(self._refresh_info)
        self.model_combo.setToolTip(
            "Trained models from ~/.ethograph/defaults/runs/lightgbm. The line below says which\n"
            "classes this one predicts and what it was trained on."
        )
        form.addRow("Model:", self.model_combo)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: grey; font-size: 10px;")
        form.addRow("", self.info_label)

        self.individual_combo = QComboBox()
        for name in self.app_state.label_individuals():
            self.individual_combo.addItem(name)
        current = self.app_state.selected_individual()
        if current is not None:
            idx = self.individual_combo.findText(current)
            if idx >= 0:
                self.individual_combo.setCurrentIndex(idx)
        self.individual_combo.setToolTip(
            "Whose events these are — and whose data the model reads.\n"
            "\n"
            "A model trained on another animal is re-pointed at this individual:\n"
            "same features, same order, this session's columns. One model per\n"
            "rig, not per animal — as long as the feature layout is the same."
        )
        form.addRow("Individual:", self.individual_combo)

        self.min_conf_edit = ConfidenceEdit(DEFAULT_CONFIDENCE)
        self.min_conf_edit.setToolTip(
            "A prediction scoring below this is not written at all: the trial is\n"
            "left unlabelled for that class rather than given a doubtful label.\n"
            "\n"
            "Confidence is the height of the tallest peak of the model's probability\n"
            "curve for that class — a point on the curve frame-by-frame review draws,\n"
            "so you can see what a good one and a bad one look like before choosing.\n"
            "\n"
            "Leave it at 0 to write every prediction and triage afterwards with\n"
            "Review predictions — nothing is lost that way, and the confidence is\n"
            "on each label either way."
        )
        form.addRow("Min confidence:", self.min_conf_edit)
        layout.addLayout(form)

        # Which trials: the trials table's filters, and nothing else — the one
        # place trials are included or excluded for every operation.
        self.trials_note = QLabel("")
        self.trials_note.setWordWrap(True)
        self.trials_note.setStyleSheet("color: grey; font-size: 10px;")
        layout.addWidget(self.trials_note)

        self.run_btn = QPushButton("Predict missing onsets")
        self.run_btn.setAutoDefault(False)
        self.run_btn.setToolTip(
            "Predict every class of this model, for every trial the trials table\n"
            "shows that does not already carry that class. Existing labels are never\n"
            "overwritten. Nothing is saved to disk until you press Ctrl+S."
        )
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        #: What the last run wrote — the label classes and the trials that got
        #: one — so review drops exactly those classes into curation scope.
        self._reviewable: tuple[list[int], set[str]] = ([], set())
        self.review_btn = QPushButton("Review predictions…")
        self.review_btn.setAutoDefault(False)
        self.review_btn.setEnabled(False)
        self.review_btn.setToolTip(
            "Open the Labels tab with the classes just predicted dropped into the\n"
            "Curation section's scope area, ready to review from there."
        )
        self.review_btn.clicked.connect(self._review)
        layout.addWidget(self.review_btn)

        self.resize(420, 480)
        self._refresh_info(self.model_combo.currentText())
        self._refresh_trials_note()
        self.app_state.trials_changed.connect(self._refresh_trials_note)

    # ------------------------------------------------------------------

    def _refresh_info(self, name: str):
        """Describe the config that will actually run — the trained one."""
        if not name:
            self.info_label.setText("No models found in ~/.ethograph/defaults/runs/lightgbm.")
            self.run_btn.setEnabled(False)
            return
        drifted = om.config_drifted(name)
        config = drifted if drifted is not None else om.load_config(name)
        trained = "trained" if om.is_trained(name) else "NOT trained yet"
        n_sessions = len(om.list_sessions(name))
        text = (
            f"Predicts {config.describe_targets()} · "
            f"{len(config.column_names())} input columns · "
            f"{n_sessions} training session(s) · {trained}."
        )
        if config.label_inputs:
            text += f" Reads existing {', '.join(i.name for i in config.label_inputs)} labels as inputs."
        read_from = om.config_individuals(config)
        if read_from:
            text += f" Trained reading {', '.join(read_from)} — Individual below re-points it at this session."
        if drifted is not None:
            text += (
                " NOTE: config.yaml has been edited since training and is not what runs — "
                "a trained model reads the layout it was fitted on, shown here."
            )
        self.info_label.setText(text)
        self.run_btn.setEnabled(om.is_trained(name))

    def _refresh_trials_note(self, *_args):
        """Say which trials the run covers: the ones the trials table shows."""
        n = len(getattr(self.app_state, "trials", None) or [])
        self.trials_note.setText(
            f"Runs over the {n} trial(s) the trials table currently shows — filter there "
            "(Navigation section) to include or exclude trials."
        )

    # ------------------------------------------------------------------

    def _run(self):
        name = self.model_combo.currentText()
        if not name:
            return
        outcome = predict_onsets(
            self.meta,
            name,
            individual=self.individual_combo.currentText(),
            min_confidence=float(self.min_conf_edit.value()),
        )
        if outcome is None:
            return
        self._reviewable = (outcome.label_ids(), outcome.trials)
        self.review_btn.setEnabled(bool(outcome.n_predicted))
        self.status_label.setText(outcome.message())
        ensure_curve_run_chosen(self, self.app_state)

    def _review(self):
        """Drop what was just predicted into curation scope and open the Labels tab."""
        label_ids, _trials = self._reviewable
        if not label_ids:
            return
        labels_widget = getattr(self.meta, "labels_widget", None)
        curation_panel = getattr(labels_widget, "curation_panel", None)
        if curation_panel is not None:
            curation_panel.set_scope(label_ids, reason="labels predicted by lightgbm")
        collapsible_widgets = getattr(self.meta, "collapsible_widgets", None)
        if collapsible_widgets:
            collapsible_widgets[1].expand()  # "Labels" — see grid_section_container._SHORT_LABELS
