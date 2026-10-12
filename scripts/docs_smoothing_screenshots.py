"""The *Find smoothing parameters* dialog and its preview for the docs.

Loads the crow template, clicks the speed line plot, opens **Tools ▸ Data:
Find smoothing parameters…** with a bridge-gaps step and a Gaussian step,
and captures both the dialog and the previewed panel through
:mod:`screenshot_gui`::

    python scripts/docs_smoothing_screenshots.py

Writes into ``docs/source/_static/media``:

- ``smoothing_dialog_annotated.png`` — the three stages, the selected step's
  sliders and the generated code boxed, as the sidebar figures are.
- ``smoothing_preview.png`` — the speed panel while the dialog is open: raw
  faded, each step thin, the result thick in orange.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    load_template,
    panel_to_click,
    qimage_to_pil,
    rect_in,
    union,
    wait,
)
from screenshot_gui import capture_widget  # noqa: E402

from ethograph.gui import notify  # noqa: E402
from ethograph.gui.dialog_smoothing import SmoothingDialog  # noqa: E402
from ethograph.gui.active_panel import PanelKind  # noqa: E402

WINDOW = (1500, 900)


def button(dialog, text: str) -> QPushButton:
    return next(b for b in dialog.findChildren(QPushButton) if b.text() == text)


def main() -> None:
    app, shell, meta = launch_like_cli()
    notify.SUPPRESS = True
    meta.app_state._layout_snapshot_provider = None
    real_apply = meta.apply_saved_panel_layout
    meta.apply_saved_panel_layout = lambda: setattr(meta.app_state, "panel_layout", None) or real_apply()
    load_template(meta, "moll2025")
    shell.resize(*WINDOW)
    shell.show()
    wait(3000)
    meta.reset_panels()
    wait(2000)

    pc = meta.plot_container
    for ribbon in pc.label_ribbons():
        pc.remove_panel(ribbon)
    has_speed = any(
        reg.kind == PanelKind.LINEPLOT and reg.plot is not None and reg.plot.panel_state.get("feature") == "speed"
        for reg in meta.active_panels._regs
    )
    if not has_speed:
        meta._create_panel_for_source("feature", "speed", "Lineplot")
    wait(500)
    speed = panel_to_click(meta, PanelKind.LINEPLOT, "speed")
    meta.active_panels.set_active(speed)
    # Only the movement is of interest: no changepoint marks, and the view starts where the crow moves.
    meta.app_state.show_changepoints = False
    wait(500)

    dialog = SmoothingDialog(pc, meta.app_state, parent=shell)
    dialog._pipeline = dialog._pipeline.__class__(())
    dialog.show()
    dialog._add_step("bridge_gaps")
    dialog._add_step("gaussian")
    dialog._set_param("sigma_s", 0.008)
    dialog._rebuild_param_form()  # the slider follows the value, as it does after a click
    wait(800)
    _x0, x1 = speed.plot.plot_item.viewRange()[0]
    speed.plot.plot_item.setXRange(1.0, x1, padding=0)
    wait(800)

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
    img = qimage_to_pil(capture_widget(dialog, SCALE))
    annotate(img, boxes).save(OUT_DIR / "smoothing_dialog_annotated.png")

    # pyqtgraph views do not render into a pixmap on their own: shoot the window, crop the panel.
    full = qimage_to_pil(capture_widget(shell, SCALE))
    r = rect_in(speed.widget.parentWidget(), shell)  # the dock: plot plus the legend gutter
    full.crop((r.left() * SCALE, r.top() * SCALE, (r.right() + 1) * SCALE, (r.bottom() + 1) * SCALE)).save(
        OUT_DIR / "smoothing_preview.png"
    )

    dialog.close()
    app.quit()


if __name__ == "__main__":
    main()
