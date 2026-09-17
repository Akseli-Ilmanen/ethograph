"""A label timeline: a panel that plots nothing but the labels.

Labels are drawn as an overlay on the panels that exist, so a session that
opens with no panel at all (a video and nothing else) has no place to show
them: a label placed from the video would be invisible and impossible to
click again. This panel is that place — an empty time axis over the trial,
on which the ordinary label overlay, pending-label preview, time marker and
click handling all work unchanged. Added like any other panel from the
add-panel popup (**Label timeline**), or automatically on load when the
Labels tab asks for one and nothing else is open.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt

from .plots_base import BasePlot


class LabelRibbonPlot(BasePlot):
    """An empty time axis for the label overlay; y is fixed to ``[0, 1]``."""

    panel_type = "labels"
    panel_group = "labels"

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)
        self.label_items: list = []
        self.plot_item.hideAxis("left")
        self.vb.setYRange(0.0, 1.0, padding=0)
        self.vb.setMouseEnabled(x=True, y=False)

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        """Nothing to render — the label overlay is drawn by the container."""

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        """The ribbon has no y scale to apply."""

    def autoscale(self):
        self.vb.setYRange(0.0, 1.0, padding=0)

    def _apply_y_constraints(self):
        self.vb.setLimits(yMin=0.0, yMax=1.0)


class PredictionPanelPlot(LabelRibbonPlot):
    """One imported prediction file drawn on its own axis, so several can be compared stacked.

    The file's frame-by-frame confidence curve (a run folder's ``.npz``) is
    drawn here too, dashed on the panel's own 0–1 axis, so what a model
    believed sits under the labels it produced — never on a feature plot.
    """

    panel_type = "predictions"
    panel_group = "predictions"

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)
        self.prediction_path: Path | None = None
        # A strip, not a plot: no time axis of its own (the panels below
        # carry one, and it is x-linked to them) and no margins, so the
        # labels and the curve fill the whole panel.
        self.plot_item.hideAxis("bottom")
        self.plot_item.layout.setContentsMargins(0, 0, 0, 0)
        self._confidence_item = pg.PlotCurveItem(pen=pg.mkPen(color="k", width=2, style=Qt.PenStyle.DashLine))
        self._confidence_item.setZValue(5)
        self.plot_item.addItem(self._confidence_item)

    def set_confidence(self, time: np.ndarray, confidence: np.ndarray) -> None:
        """Draw *confidence* over *time* (display clock, values in 0–1)."""
        self._confidence_item.setData(np.asarray(time, dtype=np.float64), np.asarray(confidence, dtype=np.float64))
        self._confidence_item.show()

    def clear_confidence(self) -> None:
        self._confidence_item.hide()
