from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox

from ethograph.gui.plots_space import LibraryGeometryCombo, geometry_dirs_of


def click_last(combo, qtbot, real):
    if real:
        qtbot.mouseClick(combo, Qt.LeftButton)
    else:
        combo.showPopup()
    qtbot.wait(900)
    view = combo.view()
    idx = view.model().index(combo.count() - 1, 0)
    print("  rect", view.visualRect(idx), "viewport", view.viewport().size(), "popup", view.window().isVisible())
    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=view.visualRect(idx).center())
    qtbot.wait(100)


def test_bare(app_state, qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("ETHOGRAPH_HOME", str(tmp_path / "home"))
    library = geometry_dirs_of(app_state)[-1]
    library.mkdir(parents=True)
    (library / "arena.yaml").write_text(
        "references:\n- name: box\n  vertices: [[0, 0], [1, 0]]\n  edges: [[0, 1]]\n", encoding="utf-8"
    )
    for real in (False, True):
        for n in range(2):
            combo = LibraryGeometryCombo(app_state)
            qtbot.addWidget(combo)
            combo.show()
            qtbot.wait(100)
            app_state.space_library_geometry = None
            click_last(combo, qtbot, real)
            print("real", real, "try", n, "picked", combo.currentText(), app_state.space_library_geometry)
            combo.hide()

    plain = QComboBox()
    plain.addItems(["None", "arena"])
    qtbot.addWidget(plain)
    plain.show()
    qtbot.wait(100)
    click_last(plain, qtbot, True)
    print("real plain picked", plain.currentText())
