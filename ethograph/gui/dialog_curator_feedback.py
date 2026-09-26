"""Model ▸ Curator feedback…: score the curated trials against the model.

Where the curator overruled the model, the next training run should pay extra
attention (``docs/source/models/curation/difficulty.md``). That is rare enough
to live in a popup rather than the Curation section: the tolerance the point
events are judged at, **Score now**, and the F1 **Histogram…** the reviewer
flags hard trials from. The work itself is the curation panel's
(:meth:`~ethograph.gui.widgets_curation.CurationPanel.run_review`,
:meth:`~ethograph.gui.widgets_curation.CurationPanel.flag_trials_hard`); this
dialog only presses it.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ethograph.gui.dialog_review_histogram import ReviewHistogramDialog


class CuratorFeedbackDialog(QDialog):
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Curator feedback")
        self.panel = panel
        self.app_state = panel.app_state

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Where you overruled the model, it should pay extra attention next time. Score each "
            "trial's curated labels against the run that predicted it, then flag the worst as "
            "<b>hard</b> in the metadata table's <code>difficulty</code> column — a training run "
            "with <code>train.oversample</code> draws them more often. <b>Ctrl+T</b> flags the "
            "current trial by hand."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        intro.setStyleSheet("color: #bbb; font-size: 10px;")
        lay.addWidget(intro)

        form = QFormLayout()
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.0, 10.0)
        self.tolerance_spin.setDecimals(3)
        self.tolerance_spin.setSingleStep(0.01)
        self.tolerance_spin.setSpecialValueText("run's own")
        self.tolerance_spin.setSuffix(" s")
        self.tolerance_spin.setToolTip(
            'Point-event tolerance for the review. Left at "run\'s own", each run is judged\n'
            "at the tolerance its model was trained to (read from the run folder). Set it to\n"
            "compare runs trained at different tolerances, or for a run folder that carries none."
        )
        override = self.app_state.get_with_default("review_tolerance_s")
        self.tolerance_spin.setValue(float(override) if override else 0.0)
        self.tolerance_spin.valueChanged.connect(
            lambda v: setattr(self.app_state, "review_tolerance_s", float(v) if v > 0 else None)
        )
        self.tolerance_spin.editingFinished.connect(self.tolerance_spin.clearFocus)
        form.addRow("Tolerance:", self.tolerance_spin)
        lay.addLayout(form)

        row = QHBoxLayout()
        self.score_btn = QPushButton("Score now")
        self.score_btn.setAutoDefault(False)
        self.score_btn.setToolTip(
            "Score the trials the table shows against each one's prediction run — an F1 per\n"
            "trial and event type into the metadata table (state labels at IoU ≥ 0.5, point\n"
            "labels within the run's own tolerance). Measures only; runs by itself once the\n"
            "last trial is curated."
        )
        self.score_btn.clicked.connect(lambda: self.panel.run_review())
        row.addWidget(self.score_btn)
        self.histogram_btn = QPushButton("Histogram…")
        self.histogram_btn.setAutoDefault(False)
        self.histogram_btn.setToolTip(
            "The scored trials' F1 as a histogram. Look for the split, set the threshold in\n"
            "the gap, and flag everything below it hard — once, by your decision."
        )
        self.histogram_btn.clicked.connect(self._open_histogram)
        row.addWidget(self.histogram_btn)
        row.addStretch(1)
        lay.addLayout(row)

        self.status_label = QLabel(panel.review_message)
        self.status_label.setTextFormat(Qt.PlainText)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 10px; color: #bbb;")
        lay.addWidget(self.status_label)
        panel.review_scored.connect(self.status_label.setText)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _open_histogram(self) -> None:
        ReviewHistogramDialog(self.app_state, self.panel.flag_trials_hard, parent=self).exec()
