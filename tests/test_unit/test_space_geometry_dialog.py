"""Edit space geometry: a spin-box edit reaches the plots at once, and Save writes exactly what they drew."""

from pathlib import Path

from qtpy.QtWidgets import QInputDialog

from ethograph.gui.dialog_space_geometry import SpaceGeometryDialog
from ethograph.gui.plots_space import (
    LibraryGeometryCombo,
    geometry_dirs_of,
    load_library_geometries,
    load_reference_geometries,
)


def test_an_edit_previews_live_and_saves_what_was_drawn(app_state, qtbot, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ETHOGRAPH_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(QInputDialog, "getText", lambda *_a, **_k: ("arena", True))
    dialog = SpaceGeometryDialog(app_state)
    qtbot.addWidget(dialog)

    dialog._new_geometry()
    dialog._add_type.setCurrentText("cylinder")
    dialog._add_shape()
    dialog._param_spins["radius"].setValue(3.0)

    drawn = load_reference_geometries(app_state)
    assert app_state.space_library_geometry == "arena"
    assert [r.name for r in drawn] == ["cylinder1"]
    assert drawn[0].vertices[:, 0].max() == 3.0, "the plots draw the spin box's radius before any save"

    dialog._save()
    saved = load_library_geometries([Path(dialog._target_combo.currentData())])["arena"]
    assert saved[0].vertices.tolist() == drawn[0].vertices.tolist()

    dialog.reject()
    assert app_state.space_geometry_preview is None, "closing hands the plots back to the file"


def test_every_geometry_combo_shows_the_one_choice_and_none_draws_nothing(
    app_state, qtbot, tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("ETHOGRAPH_HOME", str(tmp_path / "home"))
    library = geometry_dirs_of(app_state)[-1]
    library.mkdir(parents=True)
    (library / "arena.yaml").write_text(
        "references:\n- name: box\n  vertices: [[0, 0], [1, 0]]\n  edges: [[0, 1]]\n", encoding="utf-8"
    )
    space, skeleton = LibraryGeometryCombo(app_state), LibraryGeometryCombo(app_state)
    qtbot.addWidget(space)
    qtbot.addWidget(skeleton)

    skeleton.setCurrentText("arena")
    assert space.currentText() == "arena"
    assert [r.name for r in load_reference_geometries(app_state)] == ["box"]

    space.setCurrentText(LibraryGeometryCombo.NONE)
    assert skeleton.currentText() == LibraryGeometryCombo.NONE
    assert load_reference_geometries(app_state) == []
