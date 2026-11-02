"""The raster of the bursting spikes alone: pick a burst detector, and the raster draws only what it keeps.

**Tools ▸ Neural: Raster of burst spikes only…** opens this dialog. A detector
and its parameters become a *burst filter*: the raster shows it at once, and
**➕ Add panel** lists it under *Ephys: spikes* as *Spike raster — bursts
<method>* beside the plain raster, so the two views can be switched between
later. Every filter is kept with the dataset's local settings.

The detection is the one **Neural: Neuronal firing / burst detection…** runs
to make labels of the bursts; this dialog changes nothing but what the raster
draws.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ethograph.gui.burst_form import BURST_METHODS, BurstInput, build_burst_inputs, read_burst_rule
from ethograph.gui.notify import notify
from ethograph.labels.feature_events import LOG_ISI, BurstRule


class BurstRasterDialog(QDialog):
    """A burst detector → the raster of only the spikes it puts in a burst, and a source in ➕ Add panel."""

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.app_state = meta.app_state
        self.setWindowTitle("Raster of burst spikes only")
        self.setMinimumWidth(440)
        #: parameter → its input, rebuilt whenever another detector is picked.
        self._burst_inputs: dict[str, BurstInput] = {}
        self._build()
        self._on_method_changed()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        intro = QLabel(
            "Draw the raster from the bursting spikes alone: a detector reads every unit the neuron table "
            "lets through, over the trials the trials table shows, and the raster keeps the spikes it puts in "
            "a burst. The filter is added to ➕ Add panel beside the plain raster, so the two can be switched "
            "between. The spikes themselves are untouched."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: grey; font-size: 10px;")
        lay.addWidget(intro)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        for key, (text, tooltip) in BURST_METHODS.items():
            self.method_combo.addItem(text, key)
            self.method_combo.setItemData(self.method_combo.count() - 1, tooltip, 3)  # Qt.ToolTipRole
        self.method_combo.setCurrentIndex(self.method_combo.findData(LOG_ISI))
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_row.addWidget(self.method_combo, 1)
        lay.addLayout(method_row)

        self.burst_group = QGroupBox("Parameters")
        self._burst_form = QFormLayout(self.burst_group)
        lay.addWidget(self.burst_group)

        buttons = QHBoxLayout()
        self.show_btn = QPushButton("Show burst spikes only")
        self.show_btn.setAutoDefault(False)
        self.show_btn.setToolTip("Filter the raster by this detector now, and keep the filter in ➕ Add panel.")
        self.show_btn.clicked.connect(self.show_bursts)
        buttons.addWidget(self.show_btn)
        self.all_btn = QPushButton("Show every spike")
        self.all_btn.setAutoDefault(False)
        self.all_btn.setToolTip("Back to the plain raster. The filters stay in ➕ Add panel.")
        self.all_btn.clicked.connect(self.show_all)
        buttons.addWidget(self.all_btn)
        buttons.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        lay.addLayout(buttons)

    # ------------------------------------------------------------------

    def method(self) -> str:
        return self.method_combo.currentData()

    def _on_method_changed(self, *_args) -> None:
        """One row per parameter of the picked detector, opening at the detector's own default."""
        for stale in self._burst_inputs.values():
            self._burst_form.removeRow(stale)
        self._burst_inputs = build_burst_inputs(self._burst_form, self.method())

    def burst_rule(self) -> BurstRule:
        """The detector and parameters the form selects."""
        return read_burst_rule(self.method(), self._burst_inputs)

    def show_bursts(self) -> BurstRule:
        """Keep the rule as a raster source and draw the raster through it."""
        rule = self.burst_rule()
        saved = list(self.app_state.raster_burst_filters)
        if rule.to_dict() not in saved:
            self.app_state.raster_burst_filters = [*saved, rule.to_dict()]
        self.meta._add_raster_panel(rule)
        notify(f"Raster: {rule.label()} — also in ➕ Add panel, beside the plain spike raster.")
        return rule

    def show_all(self) -> None:
        """The plain raster again."""
        self.meta._add_raster_panel(None)
