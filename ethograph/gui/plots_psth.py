"""Trial-aligned raster + PSTH histogram plot widget (reusable, no Ethograph deps)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal

_BG = "#ffffff"
_AXIS_PEN = pg.mkPen(40, 40, 40, 255)
_TICK_PEN = pg.mkPen(30, 30, 30, 210, width=1)  # single-condition raster: near-black
_SEL_BRUSH = pg.mkBrush(255, 180, 0, 60)
_SEL_PEN = pg.mkPen(200, 120, 0, 200)
_HOVER_PEN = pg.mkPen(0, 100, 200, 160, width=1, style=Qt.DashLine)
_CURSOR_PEN = pg.mkPen(80, 80, 80, 120, width=1, style=Qt.DotLine)
_REF_PEN = pg.mkPen("#CC4400", width=1.5, style=Qt.DashLine)
_BAR_BRUSH = pg.mkBrush(50, 50, 50, 190)  # single-condition PSTH: dark gray
_SEP_PEN = pg.mkPen(140, 140, 140, 180, width=1, style=Qt.DashLine)

_TICK_HALF = 0.4  # half-height of each raster tick in trial-row units

# One color per condition group — enough for typical experimental designs
_CONDITION_PALETTE: list[tuple[int, int, int]] = [
    (228, 26, 28),  # red
    (55, 126, 184),  # blue
    (77, 175, 74),  # green
    (152, 78, 163),  # purple
    (255, 127, 0),  # orange
    (0, 180, 185),  # cyan
    (240, 60, 100),  # magenta
    (150, 180, 20),  # lime
    (160, 100, 0),  # amber
    (80, 140, 220),  # sky
]


def _style_axis(ax: pg.AxisItem, label: str | None = None):
    """Apply dark-on-white styling to a pyqtgraph axis."""
    ax.setPen(_AXIS_PEN)
    ax.setTextPen(_AXIS_PEN)
    ax.setStyle(tickFont=None)
    if label is not None:
        ax.setLabel(label, color="#222222")


class TrialAxisItem(pg.AxisItem):
    """Y-axis that maps integer tick positions to trial ID strings."""

    def __init__(self, trial_ids: list[str]):
        super().__init__(orientation="left")
        self._trial_ids = trial_ids

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            idx = int(round(v))
            out.append(self._trial_ids[idx] if 0 <= idx < len(self._trial_ids) else "")
        return out


class PSTHPlot(pg.GraphicsLayoutWidget):
    """Trial raster (top) + PSTH histogram (bottom), X-axes linked.

    Y-axis: display_row index (0 = top of current sort order).
    Signals emit the *original* trial index (before any sorting).

    When condition_group is provided to set_data():
    - Raster dots are colored per condition group.
    - Thin separator lines are drawn between groups in the raster.
    - PSTH shows one filled line curve per group with a legend.

    Signals
    -------
    trial_hovered(int)          : original trial index under cursor, or -1
    trial_selected(int)         : original trial index that was single-clicked
    hover_info(int, float)      : (trial_idx, rel_time_s) on every raster mouse move
    trial_time_requested(int, float) : (trial_idx, rel_time_s) on double-click
    """

    trial_hovered = Signal(int)
    trial_selected = Signal(int)
    hover_info = Signal(int, float)  # trial_idx, rel_time_s
    trial_time_requested = Signal(int, float)  # trial_idx, rel_time_s (double-click)

    def __init__(self, trial_ids: list[str], parent=None):
        super().__init__(parent)
        self.setBackground(_BG)
        self._trial_ids = list(trial_ids)
        self._n_trials = len(trial_ids)
        self._selected = -1
        self._sort_order: list[int] = list(range(self._n_trials))

        self._spike_width: float = 1.0

        # ---- raster ----
        self._axis = TrialAxisItem(self._trial_ids)
        self.raster = self.addPlot(row=0, col=0, axisItems={"left": self._axis})
        self.raster.addItem(pg.InfiniteLine(pos=0, angle=90, pen=_REF_PEN))
        _style_axis(self.raster.getAxis("bottom"), "Time from event (s)")
        _style_axis(self.raster.getAxis("left"))

        self._highlight = pg.LinearRegionItem(
            values=[0, 1],
            orientation="horizontal",
            brush=_SEL_BRUSH,
            pen=_SEL_PEN,
            movable=False,
        )
        self._highlight.setVisible(False)
        self.raster.addItem(self._highlight)

        self._hover_line = pg.InfiniteLine(pos=-1, angle=0, pen=_HOVER_PEN)
        self._hover_line.setVisible(False)
        self.raster.addItem(self._hover_line)

        self._cursor_vline = pg.InfiniteLine(pos=0, angle=90, pen=_CURSOR_PEN)
        self._cursor_vline.setVisible(False)
        self.raster.addItem(self._cursor_vline)

        # ---- psth ----
        self.psth = self.addPlot(row=1, col=0)
        self.psth.showGrid(x=True, y=True, alpha=0.25)
        self.psth.setMaximumHeight(160)
        self.psth.setXLink(self.raster)
        self.psth.addItem(pg.InfiniteLine(pos=0, angle=90, pen=_REF_PEN))
        _style_axis(self.psth.getAxis("left"), "Rate (Hz)")
        _style_axis(self.psth.getAxis("bottom"), "Time (s)")

        # Dynamic items — cleared and rebuilt on each set_data call
        self._scatter_items: list[pg.ScatterPlotItem] = []
        self._psth_items: list[pg.PlotDataItem | pg.BarGraphItem] = []
        self._sep_items: list[pg.InfiniteLine] = []
        self._legend: pg.LegendItem | None = None

        self.raster.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.raster.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(
        self,
        perievent: dict[int, np.ndarray],
        sort_order: list[int],
        pre_s: float,
        post_s: float,
        bin_s: float,
        condition_group: dict[int, int] | None = None,
        condition_labels: list[str] | None = None,
    ):
        """Render raster + PSTH.

        Parameters
        ----------
        perievent        : {trial_idx: relative_spike_times}
        sort_order       : display_row → trial_idx mapping
        condition_group  : trial_idx → group_index (0-based).  None = single color.
        condition_labels : name for each group (len == n_groups)
        """
        self._sort_order = sort_order
        self._n_trials = len(sort_order)
        self._highlight.setVisible(False)
        self._selected = -1
        self._axis._trial_ids = [self._trial_ids[i] for i in sort_order]

        self._clear_dynamic_items()

        bins = np.arange(-pre_s, post_s + bin_s, bin_s)
        centers = (bins[:-1] + bins[1:]) / 2

        if condition_group:
            n_groups = max(condition_group.values()) + 1
            self._draw_conditioned(
                perievent,
                sort_order,
                bins,
                centers,
                bin_s,
                condition_group,
                condition_labels or [str(g) for g in range(n_groups)],
                n_groups,
            )
        else:
            self._draw_single(perievent, sort_order, bins, centers, bin_s)

        self.raster.setYRange(-0.5, self._n_trials - 0.5, padding=0)
        self.raster.setXRange(-pre_s, post_s, padding=0.02)
        self.psth.setXRange(-pre_s, post_s, padding=0.02)

    def select_row(self, display_row: int):
        if not (0 <= display_row < self._n_trials):
            return
        self._selected = display_row
        self._highlight.setRegion([display_row - 0.5, display_row + 0.5])
        self._highlight.setVisible(True)
        self.trial_selected.emit(self._sort_order[display_row])

    def select_trial_idx(self, trial_idx: int):
        if trial_idx in self._sort_order:
            self.select_row(self._sort_order.index(trial_idx))

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _clear_dynamic_items(self):
        for item in self._scatter_items:
            self.raster.removeItem(item)
        for item in self._sep_items:
            self.raster.removeItem(item)
        for item in self._psth_items:
            self.psth.removeItem(item)
        if self._legend is not None:
            self._legend.clear()
            self.psth.removeItem(self._legend)
            self._legend = None
        self._scatter_items.clear()
        self._sep_items.clear()
        self._psth_items.clear()

    @staticmethod
    def _make_tick_arrays(
        spikes_per_row: list[tuple[int, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build NaN-delimited x/y arrays for a single PlotCurveItem raster.

        Each spike becomes a vertical tick: two points + NaN sentinel.
        One draw call renders all ticks regardless of spike count.
        """
        n_spikes = sum(len(s) for _, s in spikes_per_row)
        if n_spikes == 0:
            return np.array([]), np.array([])
        # 3 points per spike: (t, row-half), (t, row+half), (nan, nan)
        xs = np.empty(n_spikes * 3)
        ys = np.empty(n_spikes * 3)
        pos = 0
        for row, spikes in spikes_per_row:
            n = len(spikes)
            xs[pos : pos + n * 3 : 3] = spikes
            xs[pos + 1 : pos + n * 3 : 3] = spikes
            xs[pos + 2 : pos + n * 3 : 3] = np.nan
            ys[pos : pos + n * 3 : 3] = row - _TICK_HALF
            ys[pos + 1 : pos + n * 3 : 3] = row + _TICK_HALF
            ys[pos + 2 : pos + n * 3 : 3] = np.nan
            pos += n * 3
        return xs, ys

    def _draw_single(self, perievent, sort_order, bins, centers, bin_s):
        """Single color raster rendered as one PlotCurveItem (one draw call)."""
        rows_spikes = [
            (display_row, perievent.get(trial_idx, np.array([]))) for display_row, trial_idx in enumerate(sort_order)
        ]
        xs, ys = self._make_tick_arrays([(r, s) for r, s in rows_spikes if len(s)])
        if len(xs):
            pen = pg.mkPen(30, 30, 30, 210, width=self._spike_width)
            curve = pg.PlotCurveItem(x=xs, y=ys, pen=pen, connect="finite")
            self.raster.addItem(curve)
            self._scatter_items.append(curve)

        all_spikes = [s for _, s in rows_spikes if len(s)]
        if all_spikes:
            counts, _ = np.histogram(np.concatenate(all_spikes), bins=bins)
            n_inc = len(all_spikes)
            rate = counts / max(n_inc, 1) / (bins[1] - bins[0])
            bar = pg.BarGraphItem(
                x=centers,
                height=rate,
                width=(bins[1] - bins[0]) * 0.88,
                brush=_BAR_BRUSH,
                pen=pg.mkPen(None),
            )
            self.psth.addItem(bar)
            self._psth_items.append(bar)

    def _draw_conditioned(
        self,
        perievent,
        sort_order,
        bins,
        centers,
        bin_s,
        condition_group,
        condition_labels,
        n_groups,
    ):
        """Multi-color raster: one PlotCurveItem per group (one draw call each)."""
        bin_w = bins[1] - bins[0]

        group_rows: list[list] = [[] for _ in range(n_groups)]
        group_spikes: list[list] = [[] for _ in range(n_groups)]

        for display_row, trial_idx in enumerate(sort_order):
            g = condition_group.get(trial_idx, 0)
            spikes = perievent.get(trial_idx, np.array([]))
            if len(spikes):
                group_rows[g].append((display_row, spikes))
                group_spikes[g].append(spikes)

        # Raster: one PlotCurveItem per group
        for g in range(n_groups):
            r, gr, b = _CONDITION_PALETTE[g % len(_CONDITION_PALETTE)]
            if group_rows[g]:
                xs, ys = self._make_tick_arrays(group_rows[g])
                curve = pg.PlotCurveItem(
                    x=xs,
                    y=ys,
                    pen=pg.mkPen(r, gr, b, 210, width=self._spike_width),
                    connect="finite",
                )
                self.raster.addItem(curve)
                self._scatter_items.append(curve)

        # Separator lines between condition groups
        prev_g = None
        for display_row, trial_idx in enumerate(sort_order):
            g = condition_group.get(trial_idx, 0)
            if prev_g is not None and g != prev_g:
                line = pg.InfiniteLine(
                    pos=display_row - 0.5,
                    angle=0,
                    pen=_SEP_PEN,
                )
                self.raster.addItem(line)
                self._sep_items.append(line)
            prev_g = g

        # PSTH: one filled step curve per group + legend
        self._legend = self.psth.addLegend(offset=(10, 5))
        self._legend.setLabelTextColor("#111111")

        for g in range(n_groups):
            r, gr, b = _CONDITION_PALETTE[g % len(_CONDITION_PALETTE)]
            if not group_spikes[g]:
                continue
            flat = np.concatenate(group_spikes[g])
            counts, _ = np.histogram(flat, bins=bins)
            n_inc = len(group_spikes[g])
            rate = counts / max(n_inc, 1) / bin_w

            curve = pg.PlotDataItem(
                x=centers,
                y=rate,
                pen=pg.mkPen(r, gr, b, width=2),
                fillLevel=0,
                fillBrush=pg.mkBrush(r, gr, b, 55),
                stepMode="center",
            )
            self.psth.addItem(curve)
            self._psth_items.append(curve)
            self._legend.addItem(curve, condition_labels[g])

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def _raster_view_pos(self, scene_pos):
        """Return (x, y) in raster view-coordinates, or None if outside."""
        if not self.raster.vb.sceneBoundingRect().contains(scene_pos):
            return None
        pt = self.raster.vb.mapSceneToView(scene_pos)
        return pt.x(), pt.y()

    def _on_mouse_moved(self, scene_pos):
        result = self._raster_view_pos(scene_pos)
        if result is None:
            self._hover_line.setVisible(False)
            self._cursor_vline.setVisible(False)
            self.trial_hovered.emit(-1)
            return

        x, y = result
        row = int(round(y))

        self._hover_line.setValue(row)
        self._hover_line.setVisible(True)
        self._cursor_vline.setValue(x)
        self._cursor_vline.setVisible(True)

        trial_idx = self._sort_order[row] if 0 <= row < self._n_trials else -1
        self.trial_hovered.emit(trial_idx)
        self.hover_info.emit(trial_idx, x)

    def _on_mouse_clicked(self, event):
        if event.button() != Qt.LeftButton:
            return
        result = self._raster_view_pos(event.scenePos())
        if result is None:
            return
        x, y = result
        row = int(round(y))
        if not (0 <= row < self._n_trials):
            return

        if event.double():
            trial_idx = self._sort_order[row]
            self.trial_time_requested.emit(trial_idx, x)
        else:
            self.select_row(row)


# ---------------------------------------------------------------------------
# Standalone helpers (shared with demo + dialog)
# ---------------------------------------------------------------------------


def sort_trials(
    perievent: dict[int, np.ndarray],
    sort_by: str,
    post_s: float,
    condition_group: dict[int, int] | None = None,
) -> list[int]:
    """Return trial indices in display order.

    If condition_group is provided, trials are first sorted by condition
    (so same-condition trials are adjacent), then within each group by
    sort_by.  The within-group sort is applied first so the condition
    sort (stable) preserves it.
    """
    keys = list(perievent)
    if sort_by == "spike_count":
        keys.sort(key=lambda i: len(perievent[i]))
    elif sort_by == "rate (0→post)":
        window = max(post_s, 1e-6)
        keys.sort(key=lambda i: np.sum(perievent[i] >= 0) / window)
    if condition_group:
        keys.sort(key=lambda i: condition_group.get(i, 0))
    return keys
