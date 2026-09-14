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
    """One imported prediction file drawn on its own axis, so several can be compared stacked."""

    panel_type = "predictions"
    panel_group = "predictions"

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)
        self.prediction_path: Path | None = None
