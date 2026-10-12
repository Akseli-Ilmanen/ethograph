"""Tools ▸ Data: Find smoothing parameters… against a loaded session.

The dialog previews on the active line plot through the real container and
active-panel manager, follows a click to another line plot, puts a runnable
line on the console, and leaves the panel raw when it closes.
"""

import numpy as np
from qtpy.QtWidgets import QApplication

from ethograph.gui.dialog_smoothing import STEPS, open_smoothing_dialog
from ethograph.gui.plots_console import to_identifier
from ethograph.gui.plots_lineplot import PREVIEW_COLOR


def _is_preview(item) -> bool:
    pen = getattr(item, "opts", {}).get("pen")
    return pen is not None and pen.color().getRgb()[:3] == PREVIEW_COLOR


def _preview_curves(plot) -> list:
    return [item for item in plot.plot_items if _is_preview(item)]


def _registration(meta, plot):
    return next(reg for reg in meta.active_panels._regs if reg.plot is plot)


def test_preview_draws_on_the_active_line_plot_and_leaves_when_closed(moll2025_gui, qtbot):
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.active_panels.set_active(_registration(meta, plot))
    raw_items = len(plot.plot_items)

    dlg = open_smoothing_dialog(meta.plot_container, meta.app_state)
    qtbot.addWidget(dlg)
    QApplication.processEvents()

    assert dlg.plot is plot
    assert plot.preview is not None
    feature = str(plot._effective_feature())
    assert f'["{feature}_clean"]' in dlg.code_edit.toPlainText()
    assert "Hz on this panel" in dlg.status_label.text(), dlg.status_label.text()
    assert _preview_curves(plot), "no orange curve drawn"
    assert len(plot.plot_items) > raw_items

    dlg.close()
    QApplication.processEvents()
    assert plot.preview is None
    assert not _preview_curves(plot)
    assert len(plot.plot_items) == raw_items


def test_preview_follows_the_clicked_line_plot(moll2025_gui, qtbot):
    _, meta = moll2025_gui
    first = meta.plot_container.line_plots[0]
    meta.active_panels.set_active(_registration(meta, first))
    second = meta.plot_container.add_lineplot()
    QApplication.processEvents()

    dlg = open_smoothing_dialog(meta.plot_container, meta.app_state)
    qtbot.addWidget(dlg)
    assert dlg.plot is first

    meta.active_panels.set_active(_registration(meta, second))
    QApplication.processEvents()
    assert dlg.plot is second
    assert first.preview is None and second.preview is not None
    dlg.close()
    assert second.preview is None


def test_sent_console_line_runs_as_pasted(moll2025_gui, qtbot):
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.active_panels.set_active(_registration(meta, plot))
    dlg = open_smoothing_dialog(meta.plot_container, meta.app_state)
    qtbot.addWidget(dlg)

    dlg._send_to_console()
    console = meta.plot_container.console_panel
    line = console.input.toPlainText()
    name = to_identifier(str(plot._effective_feature()))
    assert line.startswith(
        f"{name}_clean = apply_along_time({name}, nan_aware(gaussian), sigma_s=0.05, fs=sampling_rate(t))"
    )

    console._on_submitted(line)
    result = console.ns[f"{name}_clean"]
    assert isinstance(result, np.ndarray)
    assert result.shape == console.ns[name].shape
    assert "Error" not in console.output.toPlainText(), console.output.toPlainText()
    dlg.close()


def test_every_stage_draws_its_own_curve(moll2025_gui, qtbot):
    _, meta = moll2025_gui
    plot = meta.plot_container.line_plots[0]
    meta.active_panels.set_active(_registration(meta, plot))
    dlg = open_smoothing_dialog(meta.plot_container, meta.app_state)
    qtbot.addWidget(dlg)
    dlg._add_step("rolling_filter")
    dlg._add_step("bridge_gaps")
    QApplication.processEvents()

    assert "3 step(s) drawn" in dlg.status_label.text(), dlg.status_label.text()
    assert len(_preview_curves(plot)) >= 1, "the final stage is orange"
    named = [item.opts.get("name") for item in plot.plot_items if getattr(item, "opts", {}).get("name")]
    # changepoint scatters carry names too; the stage curves are the last ones drawn
    assert named[-3:] == [STEPS[s.kind].label for s in dlg.pipeline.steps], named
    dlg.close()
