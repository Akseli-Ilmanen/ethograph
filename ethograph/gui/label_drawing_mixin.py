"""Mixin providing label and changepoint drawing methods for plot containers."""

from functools import partial
from typing import Any, Dict

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt

from ethograph.labels.curation import subject_str
from ethograph.labels.intervals import EVENT_TYPE_POINT, LABELING_AUTOMATED

from .app_constants import (
    CP_COLOR_OSC_EVENT,
    CP_COLOR_SPECTROGRAM,
    CP_COLOR_WAVEFORM,
    CP_LINE_WIDTH_MEDIUM,
    CP_LINE_WIDTH_THICK,
    CP_LINE_WIDTH_THIN,
    CP_ZOOM_MEDIUM_THRESHOLD,
    CP_ZOOM_VERY_OUT_THRESHOLD,
    DEFAULT_LABEL_OVERLAY_MODES,
    LABEL_OVERLAY_MODE_FULL,
    LABEL_OVERLAY_MODE_NONE,
    PREDICTION_FALLBACK_Y_HEIGHT,
    PREDICTION_FALLBACK_Y_TOP,
    PREDICTION_LABELS_HEIGHT_RATIO,
    SPECTROGRAM_FALLBACK_Y_HEIGHT,
    SPECTROGRAM_LABELS_HEIGHT_RATIO,
    Z_INDEX_CHANGEPOINTS,
    Z_INDEX_LABELS,
    Z_INDEX_PREDICTIONS,
)

# Point events render as a vertical line in the label class's color.  Thicker
# than CP lines (so they read as user-coded data, not reference markers) and
# drawn above the state-event rectangles but below the changepoint markers.
_POINT_EVENT_LINE_WIDTH = 4.0
_POINT_EVENT_Z_INDEX = Z_INDEX_CHANGEPOINTS - 1

# A state label being drawn: dashed anchor on the placed onset plus a faint
# region tracking the cursor. Drawn at the point-event depth so it sits above
# the committed rectangles it is about to join.
_PENDING_LABEL_LINE_WIDTH = 2.0
_PENDING_LABEL_ALPHA = 70

# A label's outline says who vouches for it: an automated label (a model's
# output nobody has looked at) is drawn dotted, a manual or curated one solid.
# Restyling one label swaps these pens on its existing items — no redraw.
_BOUNDARY_COLOR = (255, 255, 255, 180)
_AUTOMATED_BOUNDARY_WIDTH = 2.0


def _method_style(automated: bool):
    return Qt.PenStyle.DotLine if automated else Qt.PenStyle.SolidLine


def _point_pen(color_rgb, automated: bool):
    return pg.mkPen(color=(*color_rgb, 230), width=_POINT_EVENT_LINE_WIDTH, style=_method_style(automated))


def _boundary_pen(automated: bool, width: float):
    width = _AUTOMATED_BOUNDARY_WIDTH if automated else width
    return pg.mkPen(color=_BOUNDARY_COLOR, width=width, style=_method_style(automated))


def draw_key(labels, onset_s, individual=None, individual_rec=None) -> tuple:
    """The identity a drawn label is registered under (display-clock onset)."""
    return (int(labels), round(float(onset_s), 4), subject_str(individual), subject_str(individual_rec))


class LabelDrawingMixin:
    """Mixin that provides label and changepoint drawing on plot widgets.

    Requires the host class to have:
      - app_state (for label_overlay_modes)
      - label_mappings: Dict[int, Dict[str, Any]]
      - audio_cp_items: list
      - osc_event_items: list
      - _pending_label_items, _pending_label_regions, _pending_hover_conns:
        lists, and _pending_label_anchor: float | None
      - spectrogram_plots, audio_trace_plots, heatmap_plots, neo_trace_plots
        (instance lists), ephys_trace_plot
      - current_plot (property or attribute)
    """

    # Fixed-panel attribute -> plot-type key in label_overlay_modes.
    # Dynamic panels are instance lists; any other plot is a line-plot instance.
    _PLOT_TYPE_ATTRS = {
        "ephys_trace_plot": "ephys",
    }

    def set_label_mappings(self, mappings: Dict[int, Dict[str, Any]]):
        self.label_mappings = mappings

    def set_active_label_ids(self, ids: set[int]):
        self._active_label_ids = ids

    def _get_all_plots(self) -> list:
        """Return all plot widgets that exist on this container."""
        candidates = list(getattr(self, "spectrogram_plots", ()) or ())
        candidates += list(getattr(self, "audio_trace_plots", ()) or ())
        candidates += list(getattr(self, "heatmap_plots", ()) or ())
        candidates += list(getattr(self, "neo_trace_plots", ()) or ())
        plot = getattr(self, "ephys_trace_plot", None)
        if plot is not None:
            candidates.append(plot)
        return candidates

    def _plot_type_key(self, plot) -> str:
        if getattr(plot, "panel_type", None) == "labels":
            return "labels"
        if plot in (getattr(self, "spectrogram_plots", ()) or ()):
            return "spectrogram"
        if plot in (getattr(self, "audio_trace_plots", ()) or ()):
            return "audio"
        if plot in (getattr(self, "heatmap_plots", ()) or ()):
            return "heatmap"
        if plot in (getattr(self, "neo_trace_plots", ()) or ()):
            return "neo"
        for attr, type_key in self._PLOT_TYPE_ATTRS.items():
            if plot is getattr(self, attr, None):
                return type_key
        return "lineplot"

    def _label_overlay_mode(self, plot) -> str:
        """Rendering mode ("full" | "bottom" | "none") for this plot's type."""
        type_key = self._plot_type_key(plot)
        if type_key == "labels":
            # A label timeline exists only to show labels; no per-type
            # setting may hide them there.
            return LABEL_OVERLAY_MODE_FULL
        modes = getattr(self.app_state, "label_overlay_modes", None) or {}
        return modes.get(type_key, DEFAULT_LABEL_OVERLAY_MODES[type_key])

    def draw_all_labels(self, slots):
        """Render label slots on every plot whose type's overlay mode isn't "none".

        slots: list of dicts ``{"df", "label_ids", "position"}``.
          - ``df``: DataFrame with onset_s, offset_s, labels (and optional event_type).
          - ``label_ids``: filter set; if None, every non-zero label is drawn (used
            for the predictions slot, since it isn't gated by branch membership).
          - ``position``: ``"main"``, ``"top1"`` or ``"top2"``.
        """
        if not self.label_mappings:
            return

        #: draw_key → [(item, color_rgb, base_width)] for every item drawn
        #: for that label on any plot, so one label can be restyled in place.
        self._label_item_index: dict[tuple, list] = {}

        # "Full" (main) must not visually cover the top1/top2 strips — reserve
        # their height so the main rectangle stops right below them.
        top_positions_present = {slot["position"] for slot in (slots or []) if slot["position"] in ("top1", "top2")}

        # Every plot draws the labels of the individual *it* shows: a pinned
        # panel its own, an unpinned one the sidebar's. `subject_filter` is
        # installed by DataWidget (`_subject_intervals`); without it every
        # row is drawn everywhere.
        subject_filter = getattr(self, "subject_filter", None)
        for plot in self._get_all_plots():
            self._clear_labels_on_plot(plot)
            mode = self._label_overlay_mode(plot)
            if mode == LABEL_OVERLAY_MODE_NONE:
                continue
            for slot in slots or []:
                df = slot["df"] if subject_filter is None else subject_filter(slot["df"], plot)
                self._draw_intervals_on_plot(
                    plot,
                    df,
                    label_ids=slot.get("label_ids"),
                    position=slot["position"],
                    mode=mode,
                    top_positions_present=top_positions_present,
                )

    def _clear_labels_on_plot(self, plot):
        if not hasattr(plot, "label_items"):
            plot.label_items = []
            return
        for item in plot.label_items:
            try:
                plot.plot_item.removeItem(item)
            except (RuntimeError, AttributeError, ValueError):
                pass
        plot.label_items.clear()

    def _draw_intervals_on_plot(
        self,
        plot,
        intervals_df,
        label_ids=None,
        position="main",
        mode=LABEL_OVERLAY_MODE_FULL,
        top_positions_present: frozenset = frozenset(),
    ):
        if not hasattr(plot, "label_items"):
            plot.label_items = []
        if intervals_df is None or intervals_df.empty:
            return
        has_event_type = "event_type" in intervals_df.columns
        has_method = "labeling_method" in intervals_df.columns
        index = getattr(self, "_label_item_index", None)
        if index is None:
            index = self._label_item_index = {}
        for _, row in intervals_df.iterrows():
            labels = int(row["labels"])
            if labels == 0:
                continue
            if label_ids is not None and labels not in label_ids:
                continue
            is_point = has_event_type and row["event_type"] == EVENT_TYPE_POINT
            automated = has_method and row["labeling_method"] == LABELING_AUTOMATED
            if is_point:
                items = self._draw_single_point(plot, row["onset_s"], labels, automated)
            else:
                items = self._draw_single_label(
                    plot, row["onset_s"], row["offset_s"], labels, position, mode, top_positions_present, automated
                )
            if items:
                key = draw_key(labels, row["onset_s"], row.get("individual"), row.get("individual_rec"))
                index.setdefault(key, []).extend(items)
                receiver = subject_str(row.get("individual_rec"))
                if receiver and not is_point:
                    self._draw_receiver_tag(plot, row["onset_s"], receiver, items[0][1])

    def _draw_receiver_tag(self, plot, onset_s: float, receiver: str, color_rgb) -> None:
        """A small ``→ name`` at a directed label's onset: the receiver is a tag, not a lane."""
        tag = pg.TextItem(f"\u2192 {receiver}", color=(*color_rgb, 230), anchor=(0, 0))
        tag.setZValue(_POINT_EVENT_Z_INDEX)
        y_hi = plot.plot_item.getViewBox().viewRange()[1][1]
        tag.setPos(float(onset_s), float(y_hi))
        try:
            plot.plot_item.addItem(tag, ignoreBounds=True)
        except (RuntimeError, AttributeError):
            return
        plot.label_items.append(tag)

    def _draw_single_point(self, plot, time_s, labels, automated=False) -> list:
        """Draw a point event as a thick vertical line in the label's color.

        Dotted while the label is automated (see :func:`_point_pen`).
        """
        if labels not in self.label_mappings:
            return []
        color_rgb = tuple(int(c * 255) for c in self.label_mappings[labels]["color"])
        line = pg.InfiniteLine(pos=time_s, angle=90, pen=_point_pen(color_rgb, automated), movable=False)
        line.setZValue(_POINT_EVENT_Z_INDEX)
        plot.plot_item.addItem(line)
        plot.label_items.append(line)
        return [(line, color_rgb, _POINT_EVENT_LINE_WIDTH)]

    def _is_inverted_y_plot(self, plot) -> bool:
        return plot in (getattr(self, "heatmap_plots", ()) or ())

    def _draw_single_label(
        self,
        plot,
        start_time,
        end_time,
        labels,
        position="main",
        mode=LABEL_OVERLAY_MODE_FULL,
        top_positions_present: frozenset = frozenset(),
        automated=False,
    ) -> list:
        """Draw a single label rectangle; returns its restylable items.

        position: ``"main"`` -> standard full-plot rectangle (or, when top1/top2
        are also shown, a rectangle stopping short of those strips so it never
        covers them) — or, when the plot type's overlay mode is ``"bottom"``, a
        bottom strip (top strip on the inverted-Y heatmap). ``"top1"``/``"top2"``
        -> stacked thin top strips, drawn over the main rectangles. Top2 sits
        directly under Top1 so two prediction-like sources can co-exist visibly.
        *automated* draws the outline dotted (see :func:`_boundary_pen`).
        """
        if labels not in self.label_mappings:
            return []
        color_rgb = tuple(int(c * 255) for c in self.label_mappings[labels]["color"])

        is_main = position == "main"

        if is_main and mode == LABEL_OVERLAY_MODE_FULL and not top_positions_present:
            return self._draw_standard_label(plot, start_time, end_time, color_rgb, automated)

        inverted_y = self._is_inverted_y_plot(plot)
        y_lo, y_hi = plot.plot_item.getViewBox().viewRange()[1]
        degenerate = y_hi <= y_lo

        if is_main and mode == LABEL_OVERLAY_MODE_FULL:
            # Full, but top1/top2 strips are also shown: fill everything
            # below them instead of the whole plot.
            strip_height = (
                PREDICTION_FALLBACK_Y_HEIGHT if degenerate else (y_hi - y_lo) * PREDICTION_LABELS_HEIGHT_RATIO
            )
            reserved = strip_height * len(top_positions_present)
            if inverted_y:
                # Top1/Top2 occupy the y_lo side on inverted plots; leave room there.
                y_bottom = 0 if degenerate else y_lo
                y_top = PREDICTION_FALLBACK_Y_TOP if degenerate else y_hi
                y0, y1 = y_bottom + reserved, y_top
            else:
                y_bottom = 0 if degenerate else y_lo
                y_top = PREDICTION_FALLBACK_Y_TOP if degenerate else y_hi
                y0, y1 = y_bottom, y_top - reserved
            return self._draw_label_region(
                plot, start_time, end_time, color_rgb, y0, y1, Z_INDEX_LABELS, alpha=180, automated=automated
            )

        if is_main:
            # Main in "bottom" mode: bottom strip (or top strip when y is inverted)
            height = SPECTROGRAM_FALLBACK_Y_HEIGHT if degenerate else (y_hi - y_lo) * SPECTROGRAM_LABELS_HEIGHT_RATIO
            if inverted_y:
                y_top = PREDICTION_FALLBACK_Y_TOP if degenerate else y_hi
                y0, y1 = y_top - height, y_top
            else:
                y_bottom = 0 if degenerate else y_lo
                y0, y1 = y_bottom, y_bottom + height
            z, alpha = Z_INDEX_LABELS, 220
        else:
            # Top1 / Top2: stacked thin strips at the y_hi side. On heatmaps
            # (inverted_y) the strips visually appear at the bottom of the
            # screen, mirroring how the existing main bar already works there.
            height = PREDICTION_FALLBACK_Y_HEIGHT if degenerate else (y_hi - y_lo) * PREDICTION_LABELS_HEIGHT_RATIO
            slot_idx = 0 if position == "top1" else 1
            if inverted_y:
                y_bottom = 0 if degenerate else y_lo
                y0 = y_bottom + slot_idx * height
                y1 = y0 + height
            else:
                y_top = PREDICTION_FALLBACK_Y_TOP if degenerate else y_hi
                y1 = y_top - slot_idx * height
                y0 = y1 - height
            # Top2 a touch dimmer than Top1 so they're distinguishable when
            # they show overlapping classes.
            alpha = 200 if slot_idx == 0 else 170
            z = Z_INDEX_PREDICTIONS + slot_idx

        return self._draw_label_region(plot, start_time, end_time, color_rgb, y0, y1, z, alpha, automated=automated)

    def _draw_standard_label(self, plot, start_time, end_time, color_rgb, automated=False) -> list:
        rect = pg.LinearRegionItem(
            values=(start_time, end_time),
            orientation="vertical",
            brush=(*color_rgb, 180),
            pen=pg.mkPen(None),
            movable=False,
        )
        sep_pen = _boundary_pen(automated, 1)
        for line in rect.lines:
            line.setPen(sep_pen)
        rect.setZValue(Z_INDEX_LABELS)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)
        return [(rect, color_rgb, 1)]

    def _draw_label_region(
        self, plot, start_time, end_time, color_rgb, y0, y1, z_value, alpha=220, automated=False
    ) -> list:
        rect = pg.PlotDataItem(
            [start_time, end_time, end_time, start_time, start_time],
            [y0, y0, y1, y1, y0],
            fillLevel=y0,
            brush=(*color_rgb, alpha),
            pen=_boundary_pen(automated, 0),
        )
        rect.setZValue(z_value)
        plot.plot_item.addItem(rect)
        plot.label_items.append(rect)
        return [(rect, color_rgb, 0)]

    def restyle_label(self, key: tuple, automated: bool) -> int:
        """Swap the outline pens of one drawn label (dotted ⇄ solid), in place.

        *key* is :func:`draw_key` of the label as it was drawn (display-clock
        onset). Returns how many items changed; 0 means the label is not on
        screen, and the caller falls back to a full redraw. A whole trial
        changing method (Ctrl+C) is a full redraw anyway — only the one-label
        transitions of a review pass come through here.
        """
        entries = (getattr(self, "_label_item_index", None) or {}).get(key, [])
        n = 0
        for item, color_rgb, width in entries:
            try:
                if isinstance(item, pg.InfiniteLine):
                    item.setPen(_point_pen(color_rgb, automated))
                elif isinstance(item, pg.LinearRegionItem):
                    for line in item.lines:
                        line.setPen(_boundary_pen(automated, width))
                else:
                    item.setPen(_boundary_pen(automated, width))
            except RuntimeError:
                continue  # the plot it sat on was closed
            n += 1
        return n

    # --- State label in progress (between its two clicks) ---

    def _plots_of_subject(self) -> list:
        """The plots showing the labelling subject's individual — where a new label will land."""
        state = getattr(self, "app_state", None)
        if state is None or not hasattr(state, "panel_individual"):
            return list(self._get_all_plots())
        subject = state.selected_individual()
        return [p for p in self._get_all_plots() if state.panel_individual(p) == subject]

    def show_pending_label(self, t_display: float, color_rgb) -> None:
        """Mark where a state label started, until its second click lands.

        Without this the first click has no visible effect anywhere: the user
        picks the end time blind, with nothing on screen saying where the
        interval began or even that one is being drawn. The anchor is dashed
        (so it never reads as a committed label) and a faint region follows the
        cursor to preview the interval on every panel that shows the subject
        being labelled — a panel pinned to another individual will never
        receive this label, so it gets no preview of it either.
        """
        self.clear_pending_label()
        color_rgb = tuple(int(c) for c in color_rgb)
        self._pending_label_anchor = float(t_display)

        for plot in self._plots_of_subject():
            line = pg.InfiniteLine(
                pos=t_display,
                angle=90,
                pen=pg.mkPen(
                    color=(*color_rgb, 255),
                    width=_PENDING_LABEL_LINE_WIDTH,
                    style=Qt.PenStyle.DashLine,
                ),
                movable=False,
            )
            line.setZValue(_POINT_EVENT_Z_INDEX)
            region = pg.LinearRegionItem(
                values=(t_display, t_display),
                orientation="vertical",
                brush=(*color_rgb, _PENDING_LABEL_ALPHA),
                pen=pg.mkPen(None),
                movable=False,
            )
            region.setZValue(_POINT_EVENT_Z_INDEX - 1)
            try:
                # ignoreBounds: a preview stretching past the data must never
                # rescale the axis the user is aiming on.
                plot.plot_item.addItem(region, ignoreBounds=True)
                plot.plot_item.addItem(line, ignoreBounds=True)
            except (RuntimeError, AttributeError):
                continue
            self._pending_label_items.append((plot, line))
            self._pending_label_items.append((plot, region))
            self._pending_label_regions.append(region)
            self._connect_pending_hover(plot)

    def _connect_pending_hover(self, plot) -> None:
        """Track the cursor on *plot* while a state label is half-placed.

        Connected only for the life of the pending label, so hover traffic
        costs nothing during normal review.
        """
        try:
            scene = plot.plot_item.scene()
        except (RuntimeError, AttributeError):
            return
        if scene is None:
            return
        slot = partial(self._on_pending_hover, plot)
        scene.sigMouseMoved.connect(slot)
        self._pending_hover_conns.append((scene, slot))

    def _on_pending_hover(self, plot, scene_pos) -> None:
        if self._pending_label_anchor is None:
            return
        try:
            view_pos = plot.plot_item.vb.mapSceneToView(scene_pos)
        except (RuntimeError, AttributeError):
            return
        t_display = float(view_pos.x())
        bounds = (self._pending_label_anchor, t_display)
        for region in self._pending_label_regions:
            try:
                region.setRegion(bounds)
            except RuntimeError:
                continue
        self._pending_hover_moved(t_display)

    def _pending_hover_moved(self, t_display: float) -> None:
        """Hook: the cursor moved to *t_display* while a state label is half-placed.

        The host overrides this to let the video follow the cursor, so the
        second click can be aimed by watching the frames rather than the trace.
        """

    def clear_pending_label(self) -> None:
        """Drop the anchor + preview (label committed, cancelled or disarmed)."""
        for scene, slot in self._pending_hover_conns:
            try:
                scene.sigMouseMoved.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._pending_hover_conns.clear()
        for plot, item in self._pending_label_items:
            try:
                plot.plot_item.removeItem(item)
            except (RuntimeError, AttributeError, ValueError):
                pass
        self._pending_label_items.clear()
        self._pending_label_regions.clear()
        self._pending_label_anchor = None

    # --- Audio changepoints ---

    def draw_audio_changepoints(self, onsets: np.ndarray, offsets: np.ndarray):
        self.clear_audio_changepoints()
        audio_traces = list(getattr(self, "audio_trace_plots", ()) or ())
        plots_to_draw = list(getattr(self, "spectrogram_plots", ()) or ()) + audio_traces
        line_style = self._get_changepoint_line_style()
        for plot in plots_to_draw:
            color = CP_COLOR_WAVEFORM if plot in audio_traces else CP_COLOR_SPECTROGRAM
            for onset_t in onsets:
                line = pg.InfiniteLine(
                    pos=onset_t,
                    angle=90,
                    pen=pg.mkPen(
                        color=color,
                        width=line_style["width"],
                        style=line_style["style"],
                    ),
                    movable=False,
                )
                line.setZValue(Z_INDEX_CHANGEPOINTS)
                plot.plot_item.addItem(line)
                self.audio_cp_items.append((plot, line, "onset"))
            for offset_t in offsets:
                line = pg.InfiniteLine(
                    pos=offset_t,
                    angle=90,
                    pen=pg.mkPen(
                        color=color,
                        width=line_style["width"],
                        style=line_style["style"],
                    ),
                    movable=False,
                )
                line.setZValue(Z_INDEX_CHANGEPOINTS)
                plot.plot_item.addItem(line)
                self.audio_cp_items.append((plot, line, "offset"))

    def _get_changepoint_line_style(self):
        try:
            xmin, xmax = self.current_plot.get_current_xlim()
            visible_range = xmax - xmin
            if visible_range > CP_ZOOM_VERY_OUT_THRESHOLD:
                return {"style": Qt.PenStyle.DotLine, "width": CP_LINE_WIDTH_THIN}
            elif visible_range > CP_ZOOM_MEDIUM_THRESHOLD:
                return {"style": Qt.PenStyle.DashLine, "width": CP_LINE_WIDTH_MEDIUM}
            else:
                return {"style": Qt.PenStyle.SolidLine, "width": CP_LINE_WIDTH_THICK}
        except (AttributeError, TypeError, ValueError):
            return {"style": Qt.PenStyle.DashLine, "width": CP_LINE_WIDTH_MEDIUM}

    def update_audio_changepoint_styles(self):
        if not self.audio_cp_items:
            return
        line_style = self._get_changepoint_line_style()
        audio_traces = getattr(self, "audio_trace_plots", ()) or ()
        for item in self.audio_cp_items:
            plot, line, _ = item
            color = CP_COLOR_WAVEFORM if plot in audio_traces else CP_COLOR_SPECTROGRAM
            line.setPen(pg.mkPen(color=color, width=line_style["width"], style=line_style["style"]))

    def clear_audio_changepoints(self):
        for item in self.audio_cp_items:
            plot, line = item[0], item[1]
            try:
                plot.plot_item.removeItem(line)
            except (RuntimeError, AttributeError, ValueError):
                pass
        self.audio_cp_items.clear()

    # --- Oscillatory events ---

    def update_oscillatory_event_styles(self):
        if not self.osc_event_items:
            return
        line_style = self._get_changepoint_line_style()
        for plot, line, _ in self.osc_event_items:
            line.setPen(
                pg.mkPen(
                    color=CP_COLOR_OSC_EVENT,
                    width=line_style["width"],
                    style=line_style["style"],
                )
            )

    def clear_oscillatory_events(self):
        for item in self.osc_event_items:
            plot, line = item[0], item[1]
            try:
                plot.plot_item.removeItem(line)
            except (RuntimeError, AttributeError, ValueError):
                pass
        self.osc_event_items.clear()
