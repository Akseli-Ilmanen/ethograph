"""The *Find smoothing parameters* dialog and its preview for the docs.

Loads movement's wasp sample (``DLC_single-wasp.predictions.h5``, the one in
movement's *filter and interpolate* and *smooth* examples) as a one-trial
session, shows the stinger's x position as a line plot, opens
**Tools ▸ Data: Find smoothing parameters…** and captures the two examples
of ``docs/source/advanced/smoothing.md`` through :mod:`screenshot_gui`::

    python scripts/docs_smoothing_screenshots.py

Writes into ``docs/source/_static/media``:

- ``smoothing_filter_dialog.png`` / ``smoothing_filter_preview.png`` — a
  single ``filter by confidence`` step: the dialog, and the panel with the
  low-confidence samples blanked.
- ``smoothing_dialog_annotated.png`` / ``smoothing_preview.png`` — movement's
  combined pipeline (rolling median, interpolated gaps, Savitzky-Golay): the
  dialog with its parts boxed, as the sidebar figures are, and the panel
  with every stage drawn.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from movement import sample_data
from qtpy.QtWidgets import QPushButton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from docs_sidebar_screenshots import (  # noqa: E402
    BLUE,
    GREEN,
    ORANGE,
    OUT_DIR,
    PURPLE,
    SCALE,
    annotate,
    launch_like_cli,
    panel_to_click,
    qimage_to_pil,
    rect_in,
    union,
    wait,
)
from screenshot_gui import capture_widget  # noqa: E402

import ethograph as eto  # noqa: E402
from ethograph.gui import notify  # noqa: E402
from ethograph.gui.active_panel import PanelKind  # noqa: E402
from ethograph.gui.dialog_smoothing import Pipeline, SmoothingDialog  # noqa: E402

WINDOW = (1500, 520)
WASP_FILE = "DLC_single-wasp.predictions.h5"
#: The stinger jumps around the 14th second: the filter view is the whole recording, the smoothing view zooms on it.
FILTER_VIEW_S = (0.0, 27.1)
SMOOTH_VIEW_S = (13.5, 15.5)


def wasp_session(folder: Path) -> Path:
    """movement's wasp sample written as a one-trial session: ``{folder}/wasp.nc``."""
    ds = sample_data.fetch_dataset(WASP_FILE)
    ds.attrs["trial"] = 1
    nc = folder / "wasp.nc"
    eto.from_datasets([ds]).save(str(nc))
    return nc


def button(dialog, text: str) -> QPushButton:
    return next(b for b in dialog.findChildren(QPushButton) if b.text() == text)


def capture_panel(shell, reg, name: str) -> None:
    # pyqtgraph views do not render into a pixmap on their own: shoot the window, crop the dock.
    full = qimage_to_pil(capture_widget(shell, SCALE))
    r = rect_in(reg.widget.parentWidget(), shell)  # the dock: plot plus the legend gutter
    full.crop((r.left() * SCALE, r.top() * SCALE, (r.right() + 1) * SCALE, (r.bottom() + 1) * SCALE)).save(
        OUT_DIR / name
    )


def set_pipeline(dialog: SmoothingDialog, steps: list[tuple[str, dict]]) -> None:
    dialog._pipeline = Pipeline(())
    for kind, params in steps:
        dialog._add_step(kind)
        for name, value in params.items():
            dialog._set_param(name, value)
    dialog._rebuild_param_form()  # the sliders follow the values, as they do after a click
    wait(800)


def main() -> None:
    app, shell, meta = launch_like_cli()
    notify.SUPPRESS = True
    meta.app_state._layout_snapshot_provider = None
    folder = Path(tempfile.mkdtemp(prefix="ethograph_wasp_"))
    nc = wasp_session(folder)
    io = meta.io_widget
    io._clear_all_line_edits()
    io.nc_file_path_edit.setText(str(nc))
    meta.app_state.nc_file_path = str(nc)
    meta.data_widget.on_load_clicked()
    shell.show()
    wait(3000)
    shell.resize(*WINDOW)  # after the load: loading restores the saved window size
    wait(500)

    pc = meta.plot_container
    for ribbon in pc.label_ribbons():
        pc.remove_panel(ribbon)
    for plot in list(pc.line_plots):
        pc.remove_panel(plot)
    meta._create_panel_for_source("feature", "position", "Lineplot")
    wait(500)
    reg = panel_to_click(meta, PanelKind.LINEPLOT, "position")
    plot = reg.plot
    plot.set_panel_control("keypoint", "stinger")
    plot.set_panel_control("space", "x")
    plot.update_plot()
    meta.active_panels.set_active(reg)
    meta.app_state.show_changepoints = False
    plot.plot_item.setXRange(*FILTER_VIEW_S, padding=0)
    wait(500)

    dialog = SmoothingDialog(pc, meta.app_state, parent=shell)
    dialog.show()

    # Example 1: only the confidence filter, as movement's filter_and_interpolate example (whole recording).
    set_pipeline(dialog, [("filter_by_confidence", {"threshold": 0.6})])
    qimage_to_pil(capture_widget(dialog, SCALE)).save(OUT_DIR / "smoothing_filter_dialog.png")
    capture_panel(shell, reg, "smoothing_filter_preview.png")

    # Example 2: movement's combined pipeline from its smooth example, zoomed on the jumps.
    plot.plot_item.setXRange(*SMOOTH_VIEW_S, padding=0)
    wait(500)
    set_pipeline(
        dialog,
        [
            ("rolling_filter", {"window_s": 0.1, "statistic": "median"}),
            ("interpolate_over_time", {"max_gap_s": 1.0}),
            ("savgol", {"window_s": 0.2, "polyorder": 2}),
        ],
    )
    boxes = [
        (union([rect_in(b, dialog) for b in dialog.boxes.values()]), BLUE, "Three stages: Add a step, drag to reorder"),
        (
            union([rect_in(dialog.step_title, dialog), rect_in(dialog.status_label, dialog)]),
            ORANGE,
            "The selected step's parameters, in seconds or Hz",
        ),
        (rect_in(dialog.code_edit, dialog), GREEN, "The script that reproduces the preview"),
        (
            union([rect_in(dialog.copy_btn, dialog), rect_in(dialog.console_btn, dialog)]),
            PURPLE,
            "Copy code, or send the result to the console",
        ),
    ]
    annotate(qimage_to_pil(capture_widget(dialog, SCALE)), boxes).save(OUT_DIR / "smoothing_dialog_annotated.png")
    capture_panel(shell, reg, "smoothing_preview.png")

    dialog.close()
    app.quit()


if __name__ == "__main__":
    main()
