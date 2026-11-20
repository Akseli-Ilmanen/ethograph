from pathlib import Path

from qtpy.QtCore import Qt

OUT = Path(r"C:\Users\aksel\AppData\Local\Temp\claude\11168233fe28aef1\5a5a398a-6760-4c36-8e3e-13e669789d69\scratchpad")


def test_debug(moll2025_gui, qtbot):
    shell, meta = moll2025_gui
    dw = meta.data_widget
    shell.resize(1500, 950)
    shell.show()
    kp = dw.add_skeleton_plot(view_3d=True)
    qtbot.wait(500)
    qtbot.mouseClick(kp.canvas, Qt.LeftButton)
    qtbot.wait(500)
    OUT.mkdir(parents=True, exist_ok=True)
    shell.grab().save(str(OUT / "skel.png"))
    kp.geometry_combo.window().findChild(type(kp.geometry_combo))
    sidebar = kp.controls_widget.parentWidget().parentWidget()
    sidebar.grab().save(str(OUT / "sidebar.png"))
