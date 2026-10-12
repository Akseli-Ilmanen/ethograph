"""The kinematic changepoint parameter dialog for the user manual.

Opens the **Configure...** dialog the way **Changepoints ▸ Detect
changepoints… ▸ Kinematic CPs** does for the ``troughs`` method, with two
parameters already filled in, and captures it through :mod:`screenshot_gui`::

    python scripts/docs_changepoint_screenshots.py

Writes ``docs/source/_static/media/changepoint_troughs_configure_annotated.png``
into ``docs/source/_static/media``: the parameter lines boxed, the buttons
that read and apply them boxed, as the sidebar figures are.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qtpy.QtCore import QPoint, QRect
from qtpy.QtWidgets import QPushButton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from docs_sidebar_screenshots import (  # noqa: E402
    BLUE,
    GREEN,
    ORANGE,
    OUT_DIR,
    SCALE,
    annotate,
    launch_like_cli,
    qimage_to_pil,
    rect_in,
    union,
    wait,
)
from screenshot_gui import capture_widget  # noqa: E402

from ethograph.gui.dialog_function_params import FunctionParamsDialog  # noqa: E402

#: Values a user might try first: the ``None`` defaults replaced on two lines.
EXAMPLE_PARAMS = {"prominence": 0.5, "distance": 10}


def line_rect(editor, block_number: int, root) -> QRect:
    block = editor.document().findBlockByNumber(block_number)
    geo = editor.blockBoundingGeometry(block).translated(editor.contentOffset()).toRect()
    top_left = editor.viewport().mapTo(root, geo.topLeft())
    return QRect(top_left, QPoint(top_left.x() + editor.viewport().width() - 1, top_left.y() + geo.height() - 1))


def button(dialog, text: str) -> QPushButton:
    return next(b for b in dialog.findChildren(QPushButton) if b.text() == text)


def main() -> None:
    app, shell, _meta = launch_like_cli()
    dialog = FunctionParamsDialog("find_troughs", EXAMPLE_PARAMS, parent=shell)
    dialog.show()
    wait(400)

    editor = dialog._editor
    n_auto = len(dialog._spec.auto_params) + len(dialog._spec.fixed_params)
    param_lines = [line_rect(editor, 1 + n_auto + i, dialog) for i in range(len(dialog._param_infos))]
    boxes = [
        (union(param_lines), ORANGE, "Replace None with a value, one parameter per line"),
        (rect_in(button(dialog, "Apply"), dialog), GREEN, "Apply, then Detect: the marks on the plot update"),
        (rect_in(button(dialog, "Documentation"), dialog), BLUE, "Documentation; Copy code when the marks look right"),
        (rect_in(button(dialog, "Copy code to clipboard"), dialog), BLUE, ""),
    ]
    img = qimage_to_pil(capture_widget(dialog, SCALE))
    annotate(img, boxes).save(OUT_DIR / "changepoint_troughs_configure_annotated.png")
    dialog.close()
    app.quit()


if __name__ == "__main__":
    main()
