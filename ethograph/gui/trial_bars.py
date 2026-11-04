"""Trial-extent bars along the bottom of a plot under session scope.

Session scope puts every trial on one axis, so the bars say which stretch
of the axis is which trial: one coloured strip per visible trial, its id
written above it. Qt-free layout here; drawing lives on the container.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import pyqtgraph as pg

from ethograph.io.time_model import SourceCollection, TimeRange

from .app_constants import PREDICTION_FALLBACK_Y_TOP, SPECTROGRAM_FALLBACK_Y_HEIGHT, Z_INDEX_LABELS

#: Height of a bar as a fraction of the plot's y-range.
TRIAL_BAR_HEIGHT_RATIO = 0.06
_PALETTE: list[tuple[int, int, int]] = [
    (int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _a in matplotlib.colormaps.get_cmap("tab10")(range(10))
]


@dataclass(frozen=True)
class TrialBar:
    trial_id: int | str
    start_s: float
    end_s: float
    color: tuple[int, int, int]


def trial_bars(
    source_collection: SourceCollection,
    trials: list[int | str],
    visible: TimeRange | None = None,
) -> list[TrialBar]:
    """One bar per trial in *trials*, on the session clock.

    Colours cycle through the palette by position in *trials*, so
    neighbouring trials never share one. *visible* culls bars outside the
    viewport (padded half a window each side) — a long session can hold
    thousands of trials and a text item each is what the GUI would feel.
    """
    pad = visible.duration / 2 if visible is not None else 0.0
    bars: list[TrialBar] = []
    for pos, trial_id in enumerate(trials):
        idx = source_collection.trial_index(trial_id)
        if idx is None:
            continue
        rng = source_collection.trial_range(idx)
        if visible is not None and (rng.end_s < visible.start_s - pad or rng.start_s > visible.end_s + pad):
            continue
        bars.append(TrialBar(trial_id, rng.start_s, rng.end_s, _PALETTE[pos % len(_PALETTE)]))
    return bars


def draw_trial_bars(plot, bars: list[TrialBar], *, inverted_y: bool = False) -> list:
    """Draw *bars* as a bottom strip on *plot*; returns the items added.

    *inverted_y* (the heatmap) puts the strip at the top of the y-range,
    which is the bottom of the screen. Items ignore bounds, like label
    rectangles: they are sized from the y-range, so feeding autoscale would
    move the range they were sized for.
    """
    y_lo, y_hi = plot.plot_item.getViewBox().viewRange()[1]
    degenerate = y_hi <= y_lo
    if degenerate:
        y_lo, y_hi, height = 0.0, PREDICTION_FALLBACK_Y_TOP, SPECTROGRAM_FALLBACK_Y_HEIGHT
    else:
        height = (y_hi - y_lo) * TRIAL_BAR_HEIGHT_RATIO
    if inverted_y:
        y0, y1 = y_hi - height, y_hi
        text_anchor, text_y = (0.5, 0.0), y0
    else:
        y0, y1 = y_lo, y_lo + height
        text_anchor, text_y = (0.5, 1.0), y1
    items: list[pg.GraphicsObject] = []
    for bar in bars:
        rect = pg.PlotDataItem(
            [bar.start_s, bar.end_s, bar.end_s, bar.start_s, bar.start_s],
            [y0, y0, y1, y1, y0],
            fillLevel=y0,
            brush=(*bar.color, 200),
            pen=pg.mkPen((255, 255, 255, 180), width=0),
        )
        rect.setZValue(Z_INDEX_LABELS + 1)
        plot.plot_item.addItem(rect, ignoreBounds=True)
        text = pg.TextItem(str(bar.trial_id), color=bar.color, anchor=text_anchor)
        text.setPos((bar.start_s + bar.end_s) / 2, text_y)
        text.setZValue(Z_INDEX_LABELS + 2)
        plot.plot_item.addItem(text, ignoreBounds=True)
        items.extend((rect, text))
    return items
