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

#: Above this many rows the names no longer fit beside them.
MAX_NAMED_LANES = 24

#: How opaque the band behind a highlighted row is: light enough to leave the labels on it readable.
_ROW_HIGHLIGHT_ALPHA = 60
#: Behind the labels (``Z_INDEX_LABELS``), so a highlighted row's periods stay on top.
_Z_ROW_HIGHLIGHT = -20


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
        # A confidence curve fits the ribbon's own 0–1 axis: a prediction
        # panel draws its file's, a label timeline the run imported as labels.
        self._confidence_item = pg.PlotCurveItem(pen=pg.mkPen(color="k", width=2, style=Qt.PenStyle.DashLine))
        self._confidence_item.setZValue(5)
        self.plot_item.addItem(self._confidence_item)

    def set_confidence(self, time: np.ndarray, confidence: np.ndarray) -> None:
        """Draw *confidence* over *time* (display clock, values in 0–1)."""
        self._confidence_item.setData(np.asarray(time, dtype=np.float64), np.asarray(confidence, dtype=np.float64))
        self._confidence_item.show()

    def clear_confidence(self) -> None:
        self._confidence_item.hide()

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
        #: ``{label_id: (y0, y1)}`` when each class has a row of its own.
        self.lanes: dict[int, tuple[float, float]] | None = None
        self._lane_names: list[str] = []
        self._row_highlights: list[pg.LinearRegionItem] = []

    def set_lanes(self, mappings: dict[int, dict] | None) -> bool:
        """Give every class of *mappings* its own row, in its order, first on top; ``None`` is one shared row.

        Classes that overlap in time (units active together) would hide each
        other on one row. The rows are named on the left axis while there are
        few enough of them to read. Returns whether the rows changed — the
        left axis did too, so the panels need lining up again.
        """
        ids = [lid for lid in (mappings or {}) if isinstance(lid, int) and lid != 0]
        names = [str(mappings[lid].get("name", lid)) for lid in ids] if mappings else []
        if names == self._lane_names and (self.lanes is None) == (not ids):
            return False
        self._lane_names = names
        axis = self.plot_item.getAxis("left")
        if not ids:
            self.lanes = None
            axis.setTicks(None)
            self.plot_item.hideAxis("left")
            return True
        height = 1.0 / len(ids)
        self.lanes = {lid: (1.0 - (i + 1) * height, 1.0 - i * height) for i, lid in enumerate(ids)}
        if len(ids) <= MAX_NAMED_LANES:
            axis.setTicks([[(sum(self.lanes[lid]) / 2, name) for lid, name in zip(ids, names)]])
            # The container may have shown this axis as an empty gutter to
            # line the panels up; it carries the row names now.
            axis.setStyle(showValues=True)
            self._align_left_forced = False
            self.plot_item.showAxis("left")
        return True

    def highlight_rows(self, colors: dict[int, tuple]) -> None:
        """Mark each class's row with a band in its RGB(A) colour, behind its labels.

        The colour is the one the row's unit has in the raster and the firing
        rates, so a row here is told apart the same way. A class without a row
        is skipped.
        """
        for band in self._row_highlights:
            self.plot_item.removeItem(band)
        self._row_highlights = []
        for label_id, color in colors.items():
            if self.lanes is None or label_id not in self.lanes:
                continue
            band = pg.LinearRegionItem(
                values=self.lanes[label_id],
                orientation="horizontal",
                movable=False,
                brush=(*color[:3], _ROW_HIGHLIGHT_ALPHA),
                pen=pg.mkPen(color[:3], width=2),
            )
            band.setZValue(_Z_ROW_HIGHLIGHT)
            self.plot_item.addItem(band, ignoreBounds=True)
            self._row_highlights.append(band)

    def lane_label_at(self, y: float) -> int | None:
        """The class whose row holds *y*, or ``None`` on a panel without rows."""
        if self.lanes is None:
            return None
        return next((lid for lid, (y0, y1) in self.lanes.items() if y0 <= y <= y1), None)
