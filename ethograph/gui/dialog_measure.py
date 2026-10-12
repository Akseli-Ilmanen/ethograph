"""Tools ▸ *Data: Measure duration / y-range…* — read off two clicked points.

A one-shot, standalone tool: click two places on any time-axis panel and get
back both points (time and y), the duration between them and the y-range as
a tuple. The y value is whatever the clicked panel draws on its left axis —
frequency on a spectrogram, the feature's own value on a line plot — and is
reported with that axis's label and unit, so the number means what the panel
says it means. Times are read in the plot's clock (`app_state.display_basis`)
and, where it differs, also as trial-relative time through
`app_state.from_display`.

Both points may sit on different panels: the duration is still the duration,
but the y-range is only reported when both share a panel, as y on one axis
has nothing to do with y on another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ethograph.gui.notify import notify

_ARM_MESSAGE = "Click two points on any time panel to measure between them."
_MARKER_PEN = pg.mkPen((255, 170, 0), width=1, style=Qt.PenStyle.DashLine)


@dataclass(frozen=True)
class MeasuredPoint:
    """One clicked point: where on the axis, and on which panel."""

    t_display: float
    y: float
    plot: Any

    def y_label(self) -> str:
        """The clicked panel's left-axis label, with its unit when it has one."""
        axis = self.plot.plot_item.getAxis("left")
        text = axis.labelText or "y"
        return f"{text} [{axis.labelUnits}]" if axis.labelUnits else text


def measure_report(a: MeasuredPoint, b: MeasuredPoint, app_state) -> str:
    """The two points, their duration and y-range as plain text."""
    lines = []
    for name, p in (("Point 1", a), ("Point 2", b)):
        resolved = app_state.from_display(p.t_display)
        trial_note = ""
        if resolved is not None and resolved[1] != p.t_display:
            trial_note = f"  (trial {resolved[0]}: {resolved[1]:.6f} s)"
        lines.append(f"{name}: t = {p.t_display:.6f} s{trial_note}, {p.y_label()} = {p.y:.6g}")
    t0, t1 = sorted((a.t_display, b.t_display))
    lines.append(f"Duration: {t1 - t0:.6f} s  (t-range: ({t0:.6f}, {t1:.6f}))")
    if a.plot is b.plot:
        y0, y1 = sorted((a.y, b.y))
        lines.append(f"y-range: ({y0:.6g}, {y1:.6g})  dy = {y1 - y0:.6g}  [{a.y_label()}]")
    else:
        lines.append("y-range: n/a — the points are on different panels")
    return "\n".join(lines)


def open_measure_dialog(plot_container, app_state, parent=None) -> QDialog:
    dlg = MeasureDialog(plot_container, app_state, parent=parent)
    dlg.show()
    notify(_ARM_MESSAGE)
    return dlg


class MeasureDialog(QDialog):
    """Collects two plot clicks and shows the duration and y-range between them."""

    def __init__(self, plot_container, app_state, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measure duration / y-range")
        self.setModal(False)
        self._container = plot_container
        self._app_state = app_state
        self._points: list[MeasuredPoint] = []
        self._markers: list[tuple[Any, pg.InfiniteLine]] = []
        self._hooked: list[Any] = []

        layout = QVBoxLayout(self)
        instructions = QLabel(
            _ARM_MESSAGE + " The y value is read off the clicked panel's own axis "
            "(Hz on a spectrogram, the feature's value elsewhere)."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self.status_label = QLabel("Armed — waiting for the first point.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.report_edit = QPlainTextEdit()
        self.report_edit.setReadOnly(True)
        self.report_edit.setPlaceholderText("Both points, the duration and the y-range appear here.")
        layout.addWidget(self.report_edit)

        button_row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setAutoDefault(False)
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._copy)
        button_row.addWidget(self.copy_btn)
        self.again_btn = QPushButton("Measure again")
        self.again_btn.setAutoDefault(False)
        self.again_btn.clicked.connect(self._reset)
        button_row.addWidget(self.again_btn)
        close_btn = QPushButton("Close")
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.close)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        self.resize(520, 260)

        for plot in self._current_plots():
            self._hook(plot)
        plot_container.panel_added.connect(self._hook)

    # ------------------------------------------------------------------

    def _current_plots(self) -> list[Any]:
        pc = self._container
        plots = [
            *pc.spectrogram_plots,
            *pc.audio_trace_plots,
            *pc.heatmap_plots,
            *pc.neo_trace_plots,
            pc.ephys_trace_plot,
            *pc.line_plots,
        ]
        return [p for p in plots if p is not None]

    def _hook(self, plot) -> None:
        if plot in self._hooked or not hasattr(plot, "plot_clicked"):
            return
        plot.plot_clicked.connect(self._on_plot_clicked)
        self._hooked.append(plot)

    def _on_plot_clicked(self, click_info: dict) -> None:
        plot = click_info.get("plot")
        x, y = click_info.get("x"), click_info.get("y")
        if plot is None or x is None or y is None or click_info["button"] != Qt.MouseButton.LeftButton:
            return
        if len(self._points) >= 2:
            self._reset()
        point = MeasuredPoint(float(x), float(y), plot)
        self._points.append(point)
        marker = pg.InfiniteLine(pos=point.t_display, angle=90, pen=_MARKER_PEN, movable=False)
        plot.plot_item.addItem(marker)
        self._markers.append((plot, marker))
        if len(self._points) == 1:
            self.status_label.setText(f"First point at t = {point.t_display:.6f} s — click the second.")
            return
        self.status_label.setText("Measured — click again to start over, or “Measure again”.")
        self.report_edit.setPlainText(measure_report(self._points[0], self._points[1], self._app_state))
        self.copy_btn.setEnabled(True)

    def _reset(self) -> None:
        self._points.clear()
        self._clear_markers()
        self.report_edit.clear()
        self.copy_btn.setEnabled(False)
        self.status_label.setText("Armed — waiting for the first point.")

    def _clear_markers(self) -> None:
        for plot, marker in self._markers:
            try:
                plot.plot_item.removeItem(marker)
            except RuntimeError:  # the panel was closed meanwhile
                pass
        self._markers.clear()

    def _copy(self) -> None:
        clipboard = QApplication.clipboard()
        assert clipboard is not None
        clipboard.setText(self.report_edit.toPlainText())
        notify("Copied the measurement to the clipboard.")

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._clear_markers()
        for plot in self._hooked:
            try:
                plot.plot_clicked.disconnect(self._on_plot_clicked)
            except (RuntimeError, TypeError):  # panel gone, or never connected
                pass
        self._hooked.clear()
        try:
            self._container.panel_added.disconnect(self._hook)
        except (RuntimeError, TypeError):
            pass
        super().closeEvent(event)
