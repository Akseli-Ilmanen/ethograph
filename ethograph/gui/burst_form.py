"""The burst detectors as a form: the method combo's texts and one input row per parameter.

Shared by the dialogs that run a detector — the one that makes labels of the
bursts (``dialog_feature_labels``) and the one that shows only the bursting
spikes in the raster (``dialog_burst_raster``) — so a parameter reads the
same in both.
"""

from __future__ import annotations

from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QPushButton, QSpinBox

from ethograph.labels.feature_events import (
    INSTANTANEOUS_RATE,
    LOG_ISI,
    MAX_INTERVAL,
    POISSON_SURPRISE,
    BurstRule,
    burst_parameters,
)

#: The detectors in the order they are offered: key → (combo text, tooltip).
BURST_METHODS: dict[str, tuple[str, str]] = {
    LOG_ISI: (
        "Bursts: logISI",
        "Bursts read off the spike times with a limit each unit sets for itself: the valley\n"
        "between the within-burst and between-burst peaks of its own ISI histogram, over the\n"
        "trials shown. Units with different firing patterns need no tuning.\n"
        "Pasquale et al. (2010); ranked second by Cotterill et al. (2016).",
    ),
    MAX_INTERVAL: (
        "Bursts: MaxInterval",
        "Bursts read off the spike times with fixed limits on the intervals between spikes:\n"
        "one to start a burst, one to stay in it, then bursts close together are merged and\n"
        "small ones dropped. One set of numbers for every unit.\n"
        "Ranked first of eight detectors by Cotterill et al. (2016, J Neurophysiol 116:306).",
    ),
    INSTANTANEOUS_RATE: (
        "Bursts: instantaneous rate",
        "A burst is a run of spikes whose instantaneous rate (one over the interval between\n"
        "two spikes) stays above a threshold — the birdsong field's definition, 125 Hz in\n"
        "Leonardo & Fee (2005).",
    ),
    POISSON_SURPRISE: (
        "Bursts: Poisson surprise",
        "A burst is a run of spikes too dense to be chance for a unit firing at its own mean\n"
        "rate: grown and trimmed to make that improbability (its surprise) as large as it gets,\n"
        "and kept above min_surprise. Relative to each unit's rate, so it needs no tuning per unit.\n"
        "Legendy & Salcman (1985), as implemented by Cotterill & Eglen (2018).",
    ),
}

#: A burst detector's parameters, each with its meaning, worded as in pynapple's user guide
#: (doc/user_guide/15_isi_bursts.md). The form labels a row with the argument's own name and shows
#: this, with the default, on hover.
BURST_PARAMETERS: dict[str, str] = {
    "max_isi_start": "A burst starts at the first of two spikes closer than this.",
    "max_isi_burst": "A burst continues while consecutive spikes are no further apart than this.",
    "min_inter_burst_interval": "Bursts closer than this, last spike to first spike, are merged.",
    "min_burst_duration": "Shorter bursts are discarded.",
    "min_spikes_in_burst": "Bursts with fewer spikes are discarded.",
    "max_cutoff": (
        "Largest ISI at which the intra-burst peak may lie. Also the threshold used when no clear\n"
        "minimum between the peaks is found."
    ),
    "fallback": (
        "Whether to use max_cutoff as the threshold when no clear minimum is found, as the published\n"
        "method does. Unticked, such a train returns no bursts."
    ),
    "rate_threshold": "A burst is a run of spikes whose instantaneous firing rate (1 / ISI) stays above this.",
    "min_surprise": (
        "The smallest surprise of a burst: -log of the probability of that many spikes in that time\n"
        "under a Poisson process at the train's mean rate."
    ),
}

#: Parameters whose 0 stands for a value worked out from each train: the text the spin shows at 0.
_AUTO_AT_ZERO = {
    POISSON_SURPRISE: {"max_isi_burst": "2 × mean ISI"},
}

#: The unit each kind of parameter is entered in, as the tooltip states its default.
_UNIT_OF = {"rate_threshold": " Hz", "min_surprise": ""}

#: A parameter whose meaning depends on the detector, by (method, name).
_MEANING_FOR = {
    (POISSON_SURPRISE, "max_isi_burst"): (
        "The ISI beyond which spikes are no longer added to a burst.\n"
        "0 = twice the mean ISI of the train (pynapple's default, None)."
    ),
}


def _tooltip(name: str, default: float | int | bool, method: str = "") -> str:
    """The argument's meaning and its default, as the row shows on hover."""
    meaning = _MEANING_FOR.get((method, name), BURST_PARAMETERS[name])
    auto = _AUTO_AT_ZERO.get(method, {}).get(name)
    if auto is not None and default == 0:
        shown = auto
    elif isinstance(default, bool):
        shown = "ticked" if default else "unticked"
    elif isinstance(default, int):
        shown = str(default)
    else:
        shown = f"{default:g}{_UNIT_OF.get(name, ' s')}"
    return f"{name}\n{meaning}\nDefault: {shown}"


#: The fewest and most spikes a burst may be asked to have.
SPIKE_COUNT_RANGE = (2, 10_000)

#: The highest rate threshold that may be typed, in Hz.
_RATE_LIMIT = 100_000.0

BurstInput = QCheckBox | QSpinBox | QDoubleSpinBox

#: Object name of the form's revert button, how a rebuild finds the one to replace.
_REVERT_BUTTON = "burstRevertToDefaults"


def seconds_spin(tooltip: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 3600.0)
    spin.setDecimals(3)
    spin.setSingleStep(0.05)
    spin.setSuffix(" s")
    spin.setToolTip(tooltip)
    return spin


def _rate_spin(tooltip: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.001, _RATE_LIMIT)
    spin.setDecimals(1)
    spin.setSingleStep(5.0)
    spin.setSuffix(" Hz")
    spin.setToolTip(tooltip)
    return spin


def saved_burst_parameters(app_state, method: str) -> dict[str, float | int | bool]:
    """The values the user last chose for *method*'s parameters — only those still valid for it.

    A saved value of another type than the detector's default (a hand-edited
    file, a detector that changed) is left out, so the default stands.
    """
    saved = (app_state.get_with_default("burst_detector_params") or {}).get(method) or {}
    defaults = burst_parameters(method)
    return {
        name: type(defaults[name])(value)
        for name, value in saved.items()
        if name in defaults
        and isinstance(value, bool) == isinstance(defaults[name], bool)
        and isinstance(value, (bool, int, float))
    }


def _set_input(widget: BurstInput, value: float | int | bool) -> None:
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QSpinBox):
        widget.setValue(int(value))
    else:
        widget.setValue(float(value))


def _remember(app_state, method: str, name: str, value: float | int | bool) -> None:
    """Keep *value* as the user's choice for *method*'s *name* — or forget it when it is the default."""
    saved = {m: dict(p) for m, p in (app_state.get_with_default("burst_detector_params") or {}).items()}
    chosen = saved.setdefault(method, {})
    if value == burst_parameters(method)[name]:
        chosen.pop(name, None)
    else:
        chosen[name] = value
    if not chosen:
        del saved[method]
    app_state.burst_detector_params = saved


def build_burst_inputs(form: QFormLayout, method: str, app_state=None) -> dict[str, BurstInput]:
    """Add one row per parameter of *method* to *form*, and a button that reverts them to the detector's defaults.

    With *app_state* each row opens at the value the user last chose for this
    detector (the detector's own default otherwise) and every change is
    remembered, per detector, across sessions; the button puts this detector's
    rows — and what is remembered of them — back to its defaults. The button
    of an earlier build of the same form is replaced.
    """
    parent = form.parentWidget()
    stale = parent.findChild(QPushButton, _REVERT_BUTTON) if parent is not None else None
    if stale is not None:
        form.removeRow(stale)
    saved = saved_burst_parameters(app_state, method) if app_state is not None else {}
    inputs: dict[str, BurstInput] = {}
    for name, detector_default in burst_parameters(method).items():
        default = saved.get(name, detector_default)
        tooltip = _tooltip(name, detector_default, method)
        widget: BurstInput
        if isinstance(default, bool):
            widget = QCheckBox()
            widget.setChecked(default)
        elif isinstance(default, int):
            widget = QSpinBox()
            widget.setRange(*SPIKE_COUNT_RANGE)
            widget.setValue(default)
        elif name == "rate_threshold":
            widget = _rate_spin(tooltip)
            widget.setValue(default)
        elif name == "min_surprise":
            widget = QDoubleSpinBox()
            widget.setRange(0.0, 1000.0)
            widget.setDecimals(1)
            widget.setSingleStep(0.5)
            widget.setValue(default)
        else:
            widget = seconds_spin(tooltip)
            auto = _AUTO_AT_ZERO.get(method, {}).get(name)
            if auto is not None:  # 0 is not "no limit" but a value worked out from each train
                widget.setSpecialValueText(auto)
            widget.setValue(default)
        widget.setToolTip(tooltip)
        form.addRow(f"{name}:", widget)
        row_label = form.labelForField(widget)
        if row_label is not None:
            row_label.setToolTip(tooltip)
        inputs[name] = widget
        if app_state is not None:
            changed = widget.toggled if isinstance(widget, QCheckBox) else widget.valueChanged
            changed.connect(lambda value, name=name: _remember(app_state, method, name, value))

    revert = QPushButton("Revert to defaults")
    revert.setObjectName(_REVERT_BUTTON)
    revert.setAutoDefault(False)
    revert.setToolTip(f"Put the parameters of {BURST_METHODS[method][0]} back to the detector's own defaults")
    revert.clicked.connect(lambda: revert_burst_inputs(method, inputs))
    form.addRow("", revert)
    return inputs


def revert_burst_inputs(method: str, inputs: dict[str, BurstInput]) -> None:
    """Set every row of *method* back to the detector's own default."""
    for name, default in burst_parameters(method).items():
        _set_input(inputs[name], default)


def read_burst_rule(method: str, inputs: dict[str, BurstInput]) -> BurstRule:
    """The detector and the parameters the rows hold."""
    params = {
        name: widget.isChecked() if isinstance(widget, QCheckBox) else widget.value() for name, widget in inputs.items()
    }
    return BurstRule(method, params)
