"""The burst detectors as a form: the method combo's texts and one input row per parameter.

Shared by the dialogs that run a detector — the one that makes labels of the
bursts (``dialog_feature_labels``) and the one that shows only the bursting
spikes in the raster (``dialog_burst_raster``) — so a parameter reads the
same in both.
"""

from __future__ import annotations

from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QSpinBox

from ethograph.labels.feature_events import INSTANTANEOUS_RATE, LOG_ISI, MAX_INTERVAL, BurstRule, burst_parameters

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
}

#: A burst detector's parameters as the form shows them: name → (row label, tooltip).
BURST_PARAMETERS: dict[str, tuple[str, str]] = {
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
    "rate_threshold": (
        "Rate above:",
        "Two spikes belong to a burst while one over the interval between them exceeds this rate.",
    ),
}

#: The fewest and most spikes a burst may be asked to have.
SPIKE_COUNT_RANGE = (2, 10_000)

#: The highest rate threshold that may be typed, in Hz.
_RATE_LIMIT = 100_000.0

BurstInput = QCheckBox | QSpinBox | QDoubleSpinBox


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


def build_burst_inputs(form: QFormLayout, method: str) -> dict[str, BurstInput]:
    """Add one row per parameter of *method* to *form*, each opening at the detector's own default."""
    inputs: dict[str, BurstInput] = {}
    for name, default in burst_parameters(method).items():
        label, tooltip = BURST_PARAMETERS[name]
        widget: BurstInput
        if isinstance(default, bool):
            widget = QCheckBox("read them with that limit")
            widget.setChecked(default)
        elif isinstance(default, int):
            widget = QSpinBox()
            widget.setRange(*SPIKE_COUNT_RANGE)
            widget.setValue(default)
        elif name == "rate_threshold":
            widget = _rate_spin(tooltip)
            widget.setValue(default)
        else:
            widget = seconds_spin(tooltip)
            widget.setValue(default)
        widget.setToolTip(tooltip)
        form.addRow(label, widget)
        inputs[name] = widget
    return inputs


def read_burst_rule(method: str, inputs: dict[str, BurstInput]) -> BurstRule:
    """The detector and the parameters the rows hold."""
    params = {
        name: widget.isChecked() if isinstance(widget, QCheckBox) else widget.value() for name, widget in inputs.items()
    }
    return BurstRule(method, params)
