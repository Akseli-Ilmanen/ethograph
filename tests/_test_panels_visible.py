from pathlib import Path

import yaml
from qtpy.QtWidgets import QApplication, QDockWidget

LOCAL = Path.home() / ".ethograph/cache/example_data/Moll2025/.ethograph/local_settings.yaml"


def test_panels_visible_with_saved_layout(moll2025_gui, qtbot):
    shell, meta = moll2025_gui
    shell.show()
    shell.resize(1600, 1000)
    pc = meta.plot_container
    layout = yaml.safe_load(LOCAL.read_text())["panel_layout"]
    print("saved layout keys:", list(layout))
    print(
        {k: v for k, v in layout.items() if k != "panels"}
        and str({k: v for k, v in layout.items() if k != "panels"})[:1500]
    )
    pc.apply_layout_state(layout)
    for _ in range(40):
        QApplication.processEvents()
        qtbot.wait(50)
    for dock in pc.findChildren(QDockWidget):
        w = dock.widget()
        img = getattr(getattr(w, "image_item", None), "image", None)
        print(
            type(w).__name__, "dock visible", dock.isVisible(), dock.size(), "img", None if img is None else img.shape
        )
