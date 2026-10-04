"""Read-only labels made from the data: one dialog, opened from two places in Tools.

**Labels: Create from a feature…** thresholds a feature of the session — the
periods it spends above or below a value. **Neural: Neuronal firing / burst
detection…** (``neural=True``) is the same form for the loaded units: the
threshold reads a rate off the spikes instead (``firing_rate``,
``instantaneous_rate``), and a **Method** row adds the burst detectors
(MaxInterval, logISI), which read the spike trains themselves — events too
short for a rate to resolve — and show their own parameters in place of the
threshold's.

For a threshold: pick a feature, pin each of its dims to one value — or read one dim value by
value (every unit, every keypoint), which gives one class per value named
``{dim}_{value}`` — and a threshold to be above or below. The periods are
found in every trial the trials table shows, short blips cleaned up on the
way (gaps stitched, short periods dropped), by
:mod:`ethograph.labels.feature_events`.

The result is a **read-only label source**, never the working labels: it
joins ``app_state.prediction_sets`` with a panel of its own (one row per
class, since classes may overlap in time) and becomes what label navigation
and the review grids read (``app_state.label_source``), so its periods can be
walked, played back and looked at in bulk. Nothing is written to disk.

The threshold reads the feature's own units, or a scale of each series' own
across the trials shown — standard deviations from its mean, or a percentile
of its values — which is what lets one number serve every unit or keypoint of
a dim. **Show distribution** draws the values the threshold is compared
against, with the threshold as a line that can be dragged, and says what
share of them pass.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyqtgraph as pg
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
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ethograph.features.neural import BURST_INSTANTANEOUS_RATE_HZ
from ethograph.gui.dialog_onset_model import base_loader, iter_trial_windows
from ethograph.gui.notify import notify
from ethograph.gui.widgets_ephys import (
    FIRING_RATE_UNIT_DIM,
    INSTANT_RATE_FEATURE,
    SPIKE_RATE_FEATURES,
)
from ethograph.io.catalog import INDIVIDUAL_DIMS, SPACE_DIM
from ethograph.labels.feature_events import (
    ABOVE,
    BELOW,
    LOG_ISI,
    MAX_INTERVAL,
    PERCENTILE,
    RAW,
    ZSCORE,
    BurstRule,
    FeatureQuery,
    SeriesSample,
    ThresholdRule,
    burst_events,
    burst_parameters,
    classes_along,
    event_mappings,
    feature_events,
    sample_series,
)
from ethograph.labels.predictions import PredictionSet

logger = logging.getLogger(__name__)

#: Combo data of the entry that reads a dim value by value (a value is a string).
_EVERY_VALUE = None

#: The scales a threshold can be read on: key → (combo text, spin suffix, value it opens at).
#: A raw threshold has no sensible default; the other two are the usual starting points.
_SCALES: dict[str, tuple[str, str, float | None]] = {
    RAW: ("the feature's own units", "", None),
    ZSCORE: ("standard deviations from each series' mean (z-score)", " z", 2.0),
    PERCENTILE: ("a percentile of each series' own values", " %", 95.0),
}

#: Combo data of the method that thresholds a feature; every other method is a burst detector.
THRESHOLD = "threshold"

#: The methods in the order they are offered: key → (combo text, tooltip).
_METHODS: dict[str, tuple[str, str]] = {
    THRESHOLD: (
        "Threshold: a firing rate above or below a value",
        "The periods a unit's rate spends past a value. Right for activity lasting\nhundreds of milliseconds or more.",
    ),
    MAX_INTERVAL: (
        "Bursts: MaxInterval",
        "Bursts read off the spike times with fixed limits on the intervals between spikes:\n"
        "one to start a burst, one to stay in it, then bursts close together are merged and\n"
        "small ones dropped. One set of numbers for every unit.\n"
        "Ranked first of eight detectors by Cotterill et al. (2016, J Neurophysiol 116:306).",
    ),
    LOG_ISI: (
        "Bursts: logISI",
        "Bursts read off the spike times with a limit each unit sets for itself: the valley\n"
        "between the within-burst and between-burst peaks of its own ISI histogram, over the\n"
        "trials shown. Units with different firing patterns need no tuning.\n"
        "Pasquale et al. (2010); ranked second by Cotterill et al. (2016).",
    ),
}

#: A burst detector's parameters as the form shows them: name → (row label, tooltip).
_BURST_PARAMETERS: dict[str, tuple[str, str]] = {
    "max_isi_start": (
        "Start when two spikes are closer than:",
        "A burst starts at the first of two spikes closer together than this.",
    ),
    "max_isi_burst": (
        "Continue while spikes are within:",
        "The burst goes on while consecutive spikes are no further apart than this.",
    ),
    "min_inter_burst_interval": (
        "Merge bursts closer than:",
        "Two bursts separated by less than this (last spike to first spike) become one.",
    ),
    "min_burst_duration": (
        "Drop bursts shorter than:",
        "A burst shorter than this, first spike to last, is dropped (after merging).",
    ),
    "min_spikes_in_burst": ("Fewest spikes in a burst:", "A burst with fewer spikes than this is dropped."),
    "max_cutoff": (
        "Longest within-burst interval:",
        "The within-burst peak of a unit's ISI histogram must lie below this.\n"
        "It is also the limit used for a unit whose histogram has no clear valley.",
    ),
    "fallback": (
        "Units with no clear valley:",
        "Ticked: such a unit is read with the limit above, as the published method does.\n"
        "Unticked: it gets no bursts — the careful choice over units nobody has looked at.",
    ),
}

#: The fewest and most spikes a burst may be asked to have.
_SPIKE_COUNT_RANGE = (2, 10_000)

#: The widest a raw or z-scored threshold may be typed.
_THRESHOLD_LIMIT = 1e12

#: Bars of the distribution plot.
_HISTOGRAM_BINS = 80

#: Characters a set's name may not carry: it doubles as the set's path-like key.
_NAME_FORBIDDEN = str.maketrans({"/": "_", "\\": "_"})


def default_set_name(query: FeatureQuery, rule: ThresholdRule) -> str:
    """``"speed >20"`` / ``"firing_rate >20 (per unit)"`` — what the set is called until renamed."""
    pinned = ", ".join(f"{dim}={value}" for dim, value in query.selections.items())
    name = f"{query.feature} {rule.describe()}"
    if pinned:
        name += f" [{pinned}]"
    if query.dim is not None:
        name += f" (per {query.dim})"
    return name.translate(_NAME_FORBIDDEN)


def default_burst_name(rule: BurstRule, unit: str | None) -> str:
    """``"bursts max_interval (per unit)"`` / ``"bursts log_isi [unit=12]"`` — a burst set until renamed."""
    scope = f"(per {FIRING_RATE_UNIT_DIM})" if unit is None else f"[{FIRING_RATE_UNIT_DIM}={unit}]"
    return f"{rule.describe()} {scope}".translate(_NAME_FORBIDDEN)


def default_every_dim(dims: dict[str, list[str]]) -> str | None:
    """The dim a feature opens read value by value: its units or keypoints, not its x/y or its animals.

    Only one dim can be, so the first with several values that is not the
    space axis or the individuals; failing that, the first with several
    values at all. ``None`` when every dim has a single value.
    """
    several = [dim for dim, values in dims.items() if len(values) > 1]
    preferred = [dim for dim in several if dim != SPACE_DIM and dim not in INDIVIDUAL_DIMS]
    return next(iter(preferred or several), None)


def has_firing_rates(meta) -> bool:
    """Whether units are loaded and at least one passes the neuron table's filters."""
    ephys = getattr(meta, "ephys_widget", None)
    return ephys is not None and bool(ephys.firing_rate_units())


def trial_windows(meta, feature: str, trials: list | None = None, *, neural: bool = False):
    """``(trial, loader, t0, t1, shift)`` per trial for *feature* — every visible trial unless *trials* is given.

    *neural* reads a rate off the spikes, computed trial by trial (the
    feature the plots show is a snapshot of one trial); anything else is
    read from the session's own loader.
    """
    if neural and feature in SPIKE_RATE_FEATURES and has_firing_rates(meta):
        return meta.ephys_widget.firing_rate_windows(meta.app_state.trials if trials is None else trials)
    return iter_trial_windows(meta.app_state, trials)


def create_feature_labels(
    meta, query: FeatureQuery, rule: ThresholdRule, name: str, *, neural: bool = False
) -> PredictionSet | None:
    """Find *query*'s periods in every visible trial and load them as a read-only label source.

    Returns the set, or ``None`` (with the reason notified) when the feature
    never passes the threshold or cannot be read as asked.
    """
    app_state = meta.app_state
    individual = app_state.selected_individual() or "default"
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        events = feature_events(trial_windows(meta, query.feature, neural=neural), query, rule, individual)
    except ValueError as e:
        notify(str(e), severity="warning")
        return None
    finally:
        QApplication.restoreOverrideCursor()
    if events.empty:
        notify(f"{query.feature} is never {rule.direction} {rule.threshold:g} in the trials shown.", severity="warning")
        return None
    return _load_events(meta, events, query.class_names(rule), name)


def create_burst_labels(meta, units: list[str], rule: BurstRule, name: str) -> PredictionSet | None:
    """Find the bursts of *units* in every visible trial and load them as a read-only label source.

    One class per unit, in the neuron table's order. Returns the set, or
    ``None`` (with the reason notified) when no unit bursts or the
    detector refuses its parameters.
    """
    app_state = meta.app_state
    ephys = meta.ephys_widget
    trains = {unit: train for unit, train in ephys.spike_trains().items() if unit in units}
    spans = [(trial, t0, t1, shift) for trial, _reader, t0, t1, shift in ephys.firing_rate_windows(app_state.trials)]
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        events = burst_events(trains, spans, rule, app_state.selected_individual() or "default")
    except ValueError as e:
        notify(str(e), severity="warning")
        return None
    finally:
        QApplication.restoreOverrideCursor()
    if events.empty:
        notify(f"No bursts found by {rule.method} in the trials shown.", severity="warning")
        return None
    return _load_events(meta, events, [f"{FIRING_RATE_UNIT_DIM}_{unit}" for unit in trains], name)


def _load_events(meta, events, class_names: list[str], name: str) -> PredictionSet:
    """Show *events* as a read-only label source in a panel of its own, and make the tools read it."""
    app_state = meta.app_state
    prediction_set = PredictionSet(Path(name.translate(_NAME_FORBIDDEN)), events, mappings=event_mappings(class_names))
    meta.labels_widget.show_prediction_set(prediction_set)
    app_state.label_source_path = str(prediction_set.path)
    n_classes = events["labels"].nunique()
    notify(
        f"{len(events)} periods in {events['trial'].nunique()} trial(s), {n_classes} class(es) — read-only, "
        "in a panel of their own. Label navigation and the grids now read them: open the grids from the "
        + (
            "area under the neuron table, where units can be dragged in to narrow them."
            if classes_along(prediction_set.mappings or {}, FIRING_RATE_UNIT_DIM)
            else "Curation section."
        )
    )
    return prediction_set


class FeatureLabelsDialog(QDialog):
    """A threshold on a feature — or, for the units, on a rate or a burst detector — → a read-only label source.

    *neural* makes it the units' dialog: the features are the rates read off
    the spikes and the burst detectors are offered; without it the features
    are the session's own and a threshold is the only method.
    """

    def __init__(self, meta, parent=None, *, neural: bool = False):
        super().__init__(parent)
        self.meta = meta
        self.app_state = meta.app_state
        self.neural = neural
        self.loader = base_loader(self.app_state)
        self.setWindowTitle("Neuronal firing / burst detection" if neural else "Create labels from a feature")
        self.setMinimumWidth(440)
        #: dim → its combo, rebuilt whenever another feature is picked.
        self._dim_combos: dict[str, QComboBox] = {}
        self._name_edited = False
        #: parameter → its input, rebuilt whenever another burst detector is picked.
        self._burst_inputs: dict[str, QCheckBox | QSpinBox | QDoubleSpinBox] = {}
        #: The values last read for the distribution, and the query they answer.
        self._sample: SeriesSample | None = None
        self._sample_query: FeatureQuery | None = None
        self._build()
        self._on_feature_changed()
        self._on_method_changed()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        subject = (
            "a unit's firing rate spends above or below a value, or a unit spends bursting,"
            if self.neural
            else ("a feature spends above or below a value")
        )
        intro = QLabel(
            f"Find the periods {subject} in every trial the trials table shows. They become a read-only "
            "label source — walk them with Navigate by: Label, play them back, or open the grids — and "
            "never touch your own labels."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(intro)

        # A feature has one method, the threshold; the units have the burst detectors as well.
        self.method_row = QWidget()
        method_row = QHBoxLayout(self.method_row)
        method_row.setContentsMargins(0, 0, 0, 0)
        method_row.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        for key, (text, tooltip) in _METHODS.items():
            if key == THRESHOLD or self.neural:
                self.method_combo.addItem(text, key)
                self.method_combo.setItemData(self.method_combo.count() - 1, tooltip, Qt.ToolTipRole)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_row.addWidget(self.method_combo, stretch=1)
        self.method_row.setVisible(self.neural)
        lay.addWidget(self.method_row)

        self.feature_group = feature_group = QGroupBox("Feature")
        self._feature_form = QFormLayout(feature_group)
        self.feature_combo = QComboBox()
        derived = getattr(getattr(self.app_state, "data_loader", None), "derived", None) or {}
        # The units' dialog reads rates off the spikes, computed again for every trial.
        if self.neural:
            self.feature_combo.addItems(SPIKE_RATE_FEATURES)
            self.feature_combo.setItemData(
                0, "Spikes counted in bins and smoothed, as the Firing rates panel does.", Qt.ToolTipRole
            )
            self.feature_combo.setItemData(
                1,
                "1 / inter-spike interval: from each spike to the next, the rate that interval implies.\n"
                "No smoothing — a burst reads as high the moment it starts.\n"
                f"Opens at the birdsong literature's burst criterion, above {BURST_INSTANTANEOUS_RATE_HZ:g} Hz\n"
                "(Leonardo & Fee 2005) — a starting point, set for songbird RA neurons.",
                Qt.ToolTipRole,
            )
        else:
            # A console feature lives for one trial: it cannot be read across trials.
            self.feature_combo.addItems([f for f in self.loader.catalog.feature_choices() if f not in derived])
        self.feature_combo.currentIndexChanged.connect(self._on_feature_changed)
        self._feature_form.addRow("Feature:", self.feature_combo)
        lay.addWidget(feature_group)
        lay.addWidget(self._build_burst_group())

        self.rule_group = rule_group = QGroupBox("Threshold")
        rule_form = QFormLayout(rule_group)
        threshold_row = QHBoxLayout()
        self.direction_combo = QComboBox()
        self.direction_combo.addItem("Above", ABOVE)
        self.direction_combo.addItem("Below", BELOW)
        self.direction_combo.currentIndexChanged.connect(self._sync_name)
        threshold_row.addWidget(self.direction_combo)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-_THRESHOLD_LIMIT, _THRESHOLD_LIMIT)
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.valueChanged.connect(self._sync_name)
        threshold_row.addWidget(self.threshold_spin, stretch=1)
        rule_form.addRow("Feature is:", threshold_row)
        self.scale_combo = QComboBox()
        for key, (text, _suffix, _default) in _SCALES.items():
            self.scale_combo.addItem(text, key)
        self.scale_combo.setToolTip(
            "What the threshold is measured in.\n"
            "The feature's own units: one number for every series — right when it has a meaning.\n"
            "Z-score: how unusual a value is for its series, over every trial the trials table shows.\n"
            "Percentile: a fixed share of each series' own time (95 = its top 5 %), whatever its\n"
            "distribution — so a series that never does much gets periods too.\n"
            "Use one of the last two when reading a dim value by value: units with different\n"
            "baseline rates, or keypoints that move at different speeds, share no raw threshold."
        )
        self.scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        rule_form.addRow("Measured in:", self.scale_combo)
        self.range_label = QLabel("")
        self.range_label.setWordWrap(True)
        self.range_label.setStyleSheet("color: grey; font-size: 10px;")
        rule_form.addRow("", self.range_label)
        lay.addWidget(rule_group)
        self.distribution_group = self._build_distribution_group()
        lay.addWidget(self.distribution_group)

        self.clean_group = clean_group = QGroupBox("Short blips")
        clean_form = QFormLayout(clean_group)
        self.stitch_spin = self._seconds_spin(
            "Two periods closer together than this become one — a brief dip under the threshold\n"
            "does not split a period. 0 stitches nothing."
        )
        clean_form.addRow("Stitch gaps shorter than:", self.stitch_spin)
        self.min_duration_spin = self._seconds_spin(
            "A period shorter than this is dropped (after stitching). 0 keeps every period."
        )
        clean_form.addRow("Drop periods shorter than:", self.min_duration_spin)
        lay.addWidget(clean_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip("What the set is called in its panel title and in the label-source list.")
        self.name_edit.textEdited.connect(self._on_name_edited)
        name_row.addWidget(self.name_edit, stretch=1)
        lay.addLayout(name_row)

        self.trials_note = QLabel("")
        self.trials_note.setWordWrap(True)
        self.trials_note.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(self.trials_note)
        self._refresh_trials_note()
        self.app_state.trials_changed.connect(self._refresh_trials_note)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.create_btn = QPushButton("Create labels")
        self.create_btn.setAutoDefault(False)
        self.create_btn.clicked.connect(self.create_labels)
        buttons.addWidget(self.create_btn)
        close_btn = QPushButton("Close")
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        lay.addLayout(buttons)

    def _build_burst_group(self) -> QGroupBox:
        """The units to read; the picked detector's parameters are added below them."""
        self.burst_group = QGroupBox("Burst detection")
        self._burst_form = QFormLayout(self.burst_group)
        self.unit_combo = QComboBox()
        units = self.meta.ephys_widget.firing_rate_units() if has_firing_rates(self.meta) else []
        if len(units) > 1:
            self.unit_combo.addItem(f"Every {FIRING_RATE_UNIT_DIM} — one class each", _EVERY_VALUE)
        for unit in units:
            self.unit_combo.addItem(unit, unit)
        self.unit_combo.setToolTip(
            "One unit, or every unit the neuron table's filters let through: each a class of its own,\n"
            f"named {FIRING_RATE_UNIT_DIM}_<id>."
        )
        self.unit_combo.currentIndexChanged.connect(self._sync_name)
        self._burst_form.addRow(f"{FIRING_RATE_UNIT_DIM}:", self.unit_combo)
        return self.burst_group

    def _build_distribution_group(self) -> QGroupBox:
        group = QGroupBox("Distribution of the values")
        lay = QVBoxLayout(group)
        row = QHBoxLayout()
        self.distribution_btn = QPushButton("Show distribution")
        self.distribution_btn.setAutoDefault(False)
        self.distribution_btn.setToolTip(
            "Read the selection in every trial the trials table shows and draw how its values\n"
            "are distributed — all series pooled, on a log count axis so the tail is visible.\n"
            "Drag the line to set the threshold."
        )
        self.distribution_btn.clicked.connect(self.show_distribution)
        row.addWidget(self.distribution_btn)
        self.passing_label = QLabel("")
        self.passing_label.setStyleSheet("color: grey; font-size: 10px;")
        row.addWidget(self.passing_label, stretch=1)
        lay.addLayout(row)

        self.histogram = pg.PlotWidget()
        self.histogram.setMinimumHeight(150)
        self.histogram.setMaximumHeight(200)
        self.histogram.setMouseEnabled(x=True, y=False)
        self.histogram.setLabel("left", "log10(samples + 1)")
        self._histogram_bars = pg.BarGraphItem(x=[], height=[], width=1.0, brush=(120, 160, 220))
        self.histogram.addItem(self._histogram_bars)
        self.threshold_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#ffe066", width=2))
        self.threshold_line.sigPositionChanged.connect(self._on_threshold_line_moved)
        self.histogram.addItem(self.threshold_line)
        self.threshold_spin.valueChanged.connect(self._sync_threshold_line)
        self.direction_combo.currentIndexChanged.connect(self._sync_passing)
        self.histogram.setVisible(False)
        lay.addWidget(self.histogram)
        return group

    @staticmethod
    def _seconds_spin(tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 3600.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        spin.setSuffix(" s")
        spin.setToolTip(tooltip)
        return spin

    # ------------------------------------------------------------------
    # Method
    # ------------------------------------------------------------------

    def method(self) -> str:
        return self.method_combo.currentData()

    def _on_method_changed(self, *_args) -> None:
        """Show the picked method's inputs: a feature and its threshold, or units and a detector's parameters."""
        bursts = self.method() != THRESHOLD
        for group in (self.feature_group, self.rule_group, self.distribution_group, self.clean_group):
            group.setVisible(not bursts)
        self.burst_group.setVisible(bursts)
        if bursts:
            self._rebuild_burst_parameters()
        self._sync_name()

    def _rebuild_burst_parameters(self) -> None:
        """One row per parameter of the picked detector, opening at the detector's own default."""
        for stale in self._burst_inputs.values():
            self._burst_form.removeRow(stale)
        self._burst_inputs = {}
        for name, default in burst_parameters(self.method()).items():
            label, tooltip = _BURST_PARAMETERS[name]
            widget: QCheckBox | QSpinBox | QDoubleSpinBox
            if isinstance(default, bool):
                widget = QCheckBox("read them with that limit")
                widget.setChecked(default)
            elif isinstance(default, int):
                widget = QSpinBox()
                widget.setRange(*_SPIKE_COUNT_RANGE)
                widget.setValue(default)
            else:
                widget = self._seconds_spin(tooltip)
                widget.setValue(default)
            widget.setToolTip(tooltip)
            self._burst_form.addRow(label, widget)
            self._burst_inputs[name] = widget

    def burst_rule(self) -> BurstRule:
        """The detector and parameters the form selects."""
        params = {
            name: widget.isChecked() if isinstance(widget, QCheckBox) else widget.value()
            for name, widget in self._burst_inputs.items()
        }
        return BurstRule(self.method(), params)

    def burst_units(self) -> list[str]:
        """The units read: every one listed, or the one picked."""
        picked = self.unit_combo.currentData()
        if picked is not _EVERY_VALUE:
            return [picked]
        return [self.unit_combo.itemData(i) for i in range(1, self.unit_combo.count())]

    # ------------------------------------------------------------------
    # Feature + dims
    # ------------------------------------------------------------------

    def _on_feature_changed(self, *_args) -> None:
        """Rebuild the dim rows for the picked feature: one combo per dim."""
        for combo in self._dim_combos.values():
            self._feature_form.removeRow(combo)
        self._dim_combos = {}
        dims = self._feature_dims()
        every = default_every_dim(dims)
        for dim, values in dims.items():
            combo = QComboBox()
            if len(values) > 1:
                combo.addItem(f"Every {dim} — one class each", _EVERY_VALUE)
            for value in values:
                combo.addItem(str(value), str(value))
            # One dim opens read value by value; the others pinned to their first value.
            combo.setCurrentIndex(0 if dim == every or len(values) == 1 else 1)
            combo.setToolTip(
                f"One value of {dim}, or every value: each then becomes a class of its own, named {dim}_<value>."
            )
            combo.currentIndexChanged.connect(lambda _i, d=dim: self._on_dim_changed(d))
            self._feature_form.addRow(f"{dim}:", combo)
            self._dim_combos[dim] = combo
        if self._is_firing_rate() and self.feature_combo.currentText() == INSTANT_RATE_FEATURE:
            self._open_at_burst_criterion()
        self._refresh_range()
        self._sync_name()
        self._drop_stale_sample()

    def _open_at_burst_criterion(self) -> None:
        """The instantaneous rate has a criterion from the literature: bursts are above it, in Hz."""
        self.scale_combo.setCurrentIndex(self.scale_combo.findData(RAW))
        self.direction_combo.setCurrentIndex(self.direction_combo.findData(ABOVE))
        self.threshold_spin.setValue(BURST_INSTANTANEOUS_RATE_HZ)

    def _is_firing_rate(self) -> bool:
        return self.neural and has_firing_rates(self.meta)

    def _feature_dims(self) -> dict[str, list[str]]:
        """The picked feature's dims and their values — the firing rate's is its units."""
        if self._is_firing_rate():
            return {FIRING_RATE_UNIT_DIM: self.meta.ephys_widget.firing_rate_units()}
        return self.loader.feature_dims(self.feature_combo.currentText()) or {}

    def _on_dim_changed(self, changed: str) -> None:
        """At most one dim is read value by value: picking a second one pins the first."""
        if self._dim_combos[changed].currentData() is _EVERY_VALUE:
            for dim, combo in self._dim_combos.items():
                if dim != changed and combo.currentData() is _EVERY_VALUE:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(1)
                    combo.blockSignals(False)
        self._refresh_range()
        self._sync_name()
        self._drop_stale_sample()

    def query(self) -> FeatureQuery:
        """What the form selects."""
        selections: dict[str, str] = {}
        every: str | None = None
        values: tuple[str, ...] = ()
        for dim, combo in self._dim_combos.items():
            if combo.currentData() is _EVERY_VALUE:
                every = dim
                values = tuple(combo.itemData(i) for i in range(1, combo.count()))
            else:
                selections[dim] = combo.currentData()
        return FeatureQuery(self.feature_combo.currentText(), selections, every, values)

    def rule(self) -> ThresholdRule:
        return ThresholdRule(
            threshold=self.threshold_spin.value(),
            direction=self.direction_combo.currentData(),
            stitch_gap_s=self.stitch_spin.value(),
            min_duration_s=self.min_duration_spin.value(),
            scale=self.scale(),
        )

    def scale(self) -> str:
        return self.scale_combo.currentData()

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    def show_distribution(self) -> None:
        """Read the selection in every visible trial and draw its values."""
        if not self.feature_combo.count():
            return
        query = self.query()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            sample = sample_series(trial_windows(self.meta, query.feature, neural=self.neural), query)
        except ValueError as e:
            notify(str(e), severity="warning")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._sample, self._sample_query = sample, query
        self._draw_distribution()

    def _drop_stale_sample(self) -> None:
        """Another selection: the values on screen are no longer what the threshold is compared against."""
        if self._sample is not None and self._sample_query != self.query():
            self._sample = self._sample_query = None
            self.histogram.setVisible(False)
            self.passing_label.setText("")

    def _plotted_sample(self) -> np.ndarray:
        """The sampled values on the plot's axis, finite ones only, pooled.

        Z-scored when the threshold is; in the feature's units otherwise — a
        percentile has no axis of its own to pool series on.
        """
        assert self._sample is not None
        values = self._sample.zscored_values() if self.scale() == ZSCORE else self._sample.values
        return values[np.isfinite(values)]

    def _draw_distribution(self) -> None:
        if self._sample is None:
            return
        values = self._plotted_sample()
        if values.size == 0:
            self.histogram.setVisible(False)
            self.passing_label.setText("No values to show for this selection.")
            return
        counts, edges = np.histogram(values, bins=_HISTOGRAM_BINS)
        self._histogram_bars.setOpts(
            x=(edges[:-1] + edges[1:]) / 2, height=np.log10(counts + 1.0), width=edges[1] - edges[0]
        )
        unit = "standard deviations from each series' mean" if self.scale() == ZSCORE else "feature units"
        self.histogram.setLabel("bottom", f"value ({unit})")
        self.histogram.setXRange(float(edges[0]), float(edges[-1]), padding=0.02)
        self.histogram.enableAutoRange(axis="y")
        self.histogram.setVisible(True)
        self._sync_threshold_line()

    def _sync_threshold_line(self, *_args) -> None:
        """Put the line where the threshold falls on the plot's axis.

        A percentile is a different value for every series: the line shows
        it for a single series, where it cannot be dragged (the axis is not
        in percent), and is hidden for several.
        """
        if self._sample is not None:
            position: float | None = self.threshold_spin.value()
            if self.scale() == PERCENTILE:
                cuts = self._sample.cuts(self.rule())
                position = float(cuts[0]) if len(cuts) == 1 and np.isfinite(cuts[0]) else None
            self.threshold_line.setMovable(self.scale() != PERCENTILE)
            self.threshold_line.setVisible(position is not None)
            if position is not None:
                self.threshold_line.blockSignals(True)
                self.threshold_line.setValue(position)
                self.threshold_line.blockSignals(False)
        self._sync_passing()

    def _on_threshold_line_moved(self, *_args) -> None:
        self.threshold_spin.setValue(float(self.threshold_line.value()))

    def _sync_passing(self, *_args) -> None:
        """Say what the threshold lets through: the share of samples, or each series' cut."""
        if self._sample is None:
            return
        rule = self.rule()
        side = self.direction_combo.currentText().lower()
        if rule.scale == PERCENTILE:
            cuts = self._sample.cuts(rule)
            cuts = cuts[np.isfinite(cuts)]
            if cuts.size == 0:
                return
            share = (100.0 - rule.threshold if rule.direction == ABOVE else rule.threshold) / 100.0
            where = f"{cuts[0]:.4g}" if cuts.size == 1 else f"{cuts.min():.4g} to {cuts.max():.4g} across series"
            self.passing_label.setText(f"{share:.1%} of each series' samples are {side} its own cut ({where}).")
            return
        values = self._plotted_sample()
        if values.size == 0:
            return
        share = float(rule.passes(values).mean())
        self.passing_label.setText(f"{share:.1%} of the samples are {side} the threshold.")

    def _on_scale_changed(self, *_args) -> None:
        """Another scale: the number means something else, and the distribution is redrawn on it."""
        _text, suffix, default = _SCALES[self.scale()]
        limit = (0.0, 100.0) if self.scale() == PERCENTILE else (-_THRESHOLD_LIMIT, _THRESHOLD_LIMIT)
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setRange(*limit)
        self.threshold_spin.setSuffix(suffix)
        if default is not None:
            self.threshold_spin.setValue(default)
        self.threshold_spin.blockSignals(False)
        self._draw_distribution()
        self._sync_name()

    # ------------------------------------------------------------------
    # Hints
    # ------------------------------------------------------------------

    def _refresh_range(self) -> None:
        """Show the selection's range in the current trial — something to set the threshold against."""
        self.range_label.setText("")
        trial = getattr(self.app_state, "trials_sel", None)
        if trial is None or not self.feature_combo.count():
            return
        query = self.query()
        for _tid, loader, t0, t1, _shift in trial_windows(self.meta, query.feature, [trial], neural=self.neural):
            plot = loader.select(query.feature, query.selections, t0, t1)
            if plot is None:
                return
            data = np.asarray(plot.data, dtype=np.float64)
            if not np.isfinite(data).any():
                return
            text = f"In this trial it ranges from {np.nanmin(data):.4g} to {np.nanmax(data):.4g}."
            if query.feature == INSTANT_RATE_FEATURE and self._is_firing_rate():
                text = f"Spikes/s, 1 / inter-spike interval, read every bin of the Firing rates panel. {text}"
            elif self._is_firing_rate():
                # Bin, smoothing and units are the Firing rates panel's and the neuron table's.
                text = f"Spikes/s, binned as in Tools ▸ Neural: Compute firing rates. {text}"
            self.range_label.setText(text)

    def _refresh_trials_note(self, *_args) -> None:
        n = len(getattr(self.app_state, "trials", None) or [])
        self.trials_note.setText(
            f"Runs over the {n} trial(s) the trials table currently shows — filter there to include or exclude trials."
        )

    def _on_name_edited(self, *_args) -> None:
        self._name_edited = True

    def _sync_name(self, *_args) -> None:
        """Follow the form until the user types a name of their own."""
        if self._name_edited:
            return
        if self.method() != THRESHOLD:
            self.name_edit.setText(default_burst_name(self.burst_rule(), self.unit_combo.currentData()))
        elif self.feature_combo.count():
            self.name_edit.setText(default_set_name(self.query(), self.rule()))

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_labels(self) -> PredictionSet | None:
        """Build the set as if *Create labels* were pressed."""
        if self.method() != THRESHOLD:
            burst_rule = self.burst_rule()
            name = self.name_edit.text().strip() or default_burst_name(burst_rule, self.unit_combo.currentData())
            return create_burst_labels(self.meta, self.burst_units(), burst_rule, name)
        if not self.feature_combo.count():
            notify("This session has no features to read.", severity="warning")
            return None
        query, rule = self.query(), self.rule()
        name = self.name_edit.text().strip() or default_set_name(query, rule)
        return create_feature_labels(self.meta, query, rule, name, neural=self.neural)
