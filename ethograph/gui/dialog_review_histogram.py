"""The per-trial F1 histogram: where the human decides which trials were bad.

**Score now** (``widgets_curation``) writes each trial's F1 against the run
that predicted it into the metadata table and nothing else. Which of those
trials the next training run should see more often is a decision, not a
rule: this popup draws the distribution, the reviewer looks for the split,
sets the threshold where the bad trials separate, and presses **Flag** —
once. Everything below the line becomes ``hard`` in the ``difficulty``
column (``labels/review_metrics.py``); a hand-set flag is never removed.

The count under the histogram warns when the cut takes more than a quarter
of the trials: with ``train.oversample`` on, that many hard trials would be
most of what the model sees, and the flag would stop meaning anything.
"""

from __future__ import annotations

from typing import Callable

import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ethograph.gui.dialog_label_gridview import LOW_CONFIDENCE_COLOR, split_histogram
from ethograph.labels import review_metrics as rm

_BAR_COLOR = "#8a9099"
_TITLES = {
    rm.REVIEW_F1_STATE: "State events — F1 at IoU ≥ 0.5",
    rm.REVIEW_F1_POINT: "Point events — F1 within tolerance",
}


class ReviewHistogramDialog(QDialog):
    """Per-trial F1 scores as histograms, a threshold, and one **Flag** press.

    *apply_fn* receives the set of trial ids (as strings) below the threshold
    when the reviewer presses Flag; it is the curation panel's
    ``flag_trials_hard``. The threshold is remembered in
    ``app_state.review_flag_threshold``.
    """

    def __init__(self, app_state, apply_fn: Callable[[set[str]], int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Where was the model bad? — F1 per trial")
        self.setMinimumSize(520, 420)
        self.app_state = app_state
        self._apply_fn = apply_fn
        self._scores = rm.f1_scores(getattr(app_state, "metadata_df", None))
        self._plots: list[tuple[str, pg.PlotWidget]] = []

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Each trial's curated labels scored against what its run predicted. Look for the split "
            "between trials the model handled and trials it did not, put the threshold there, and flag "
            "once — they become <b>hard</b> in the metadata table, and a training run with "
            "<code>train.oversample</code> draws them more often."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #bbb; font-size: 10px;")
        lay.addWidget(intro)

        if not any(self._scores.values()):
            empty = QLabel(
                "No scores yet — press <b>Score now</b> first (it also runs by itself once every trial is curated)."
            )
            empty.setWordWrap(True)
            lay.addWidget(empty)
        for column, values in self._scores.items():
            if not values:
                continue
            plot = pg.PlotWidget()
            plot.setBackground("#1a1d21")
            plot.setXRange(0.0, 1.0, padding=0.02)
            plot.setLabel("bottom", "F1")
            plot.setLabel("left", "trials")
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            lay.addWidget(plot, stretch=1)
            self._plots.append((column, plot))

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Flag trials below F1:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(float(getattr(app_state, "review_flag_threshold", rm.DEFAULT_FLAG_THRESHOLD)))
        self.threshold_spin.valueChanged.connect(self._on_threshold)
        controls.addWidget(self.threshold_spin)
        controls.addWidget(QLabel("Bins:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 50)
        self.bins_spin.setValue(20)
        self.bins_spin.valueChanged.connect(self._redraw)
        controls.addWidget(self.bins_spin)
        controls.addStretch(1)
        lay.addLayout(controls)

        self.count_label = QLabel("")
        self.count_label.setWordWrap(True)
        self.count_label.setTextFormat(Qt.RichText)
        lay.addWidget(self.count_label)

        buttons = QDialogButtonBox()
        self.flag_btn = QPushButton("Flag")
        self.flag_btn.setAutoDefault(False)
        self.flag_btn.clicked.connect(self._flag)
        buttons.addButton(self.flag_btn, QDialogButtonBox.ActionRole)
        buttons.addButton(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._redraw()

    # ------------------------------------------------------------------

    def threshold(self) -> float:
        return float(self.threshold_spin.value())

    def flagged(self) -> set[str]:
        """The trials the current threshold would flag."""
        return rm.trials_below_f1(getattr(self.app_state, "metadata_df", None), self.threshold())

    def _on_threshold(self, value: float) -> None:
        self.app_state.review_flag_threshold = float(value)
        self._redraw()

    def _redraw(self, *_args) -> None:
        threshold = self.threshold()
        bins = self.bins_spin.value()
        for column, plot in self._plots:
            edges, below, above = split_histogram(self._scores[column], threshold, bins)
            centers = (edges[:-1] + edges[1:]) / 2.0
            width = (edges[1] - edges[0]) * 0.9
            plot.clear()
            plot.addItem(pg.BarGraphItem(x=centers, height=above, y0=below, width=width, brush=_BAR_COLOR, pen=None))
            plot.addItem(pg.BarGraphItem(x=centers, height=below, width=width, brush=LOW_CONFIDENCE_COLOR, pen=None))
            if threshold > 0.0:
                plot.addItem(
                    pg.InfiniteLine(pos=threshold, angle=90, pen=pg.mkPen(LOW_CONFIDENCE_COLOR, style=Qt.DashLine))
                )
            plot.setTitle(f"{_TITLES[column]} — {int(below.sum())} of {len(self._scores[column])} below", size="10pt")
        flagged = self.flagged()
        total = rm.scored_trial_count(getattr(self.app_state, "metadata_df", None))
        self.count_label.setText(rm.flag_share_note(len(flagged), total, threshold))
        self.flag_btn.setText(f"Flag {len(flagged)} trial(s) hard" if flagged else "Flag")
        self.flag_btn.setEnabled(bool(flagged))

    def _flag(self) -> None:
        flagged = self.flagged()
        if not flagged:
            return
        self._apply_fn(flagged)
        self.accept()
