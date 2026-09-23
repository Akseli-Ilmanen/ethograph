"""The box labelling dialog opens over the loaded GUI and binds every open camera view.

No SAM is built here (that needs the checkpoint and a GPU); what is checked is
the wiring a user hits first: the Tools entry, one dialog instance, the
OCTRON-style pages, a click mode on the primary camera, and the individual /
camera tables reflecting the dataset.
"""

from __future__ import annotations

import pytest

pytest.importorskip("octron")


def test_dialog_opens_once_and_binds_the_primary_camera(birdpark_gui, qtbot, monkeypatch, tmp_path):
    from ethograph.gui.dialog_box_labelling import BoxLabellingDialog

    shell, meta = birdpark_gui
    data_widget = meta.data_widget
    meta.app_state.project_path = str(tmp_path)

    dialog = data_widget.open_box_labelling()
    qtbot.addWidget(dialog)
    assert isinstance(dialog, BoxLabellingDialog)
    assert data_widget.open_box_labelling() is dialog  # raised, not duplicated

    titles = [dialog.toolbox.itemText(i) for i in range(dialog.toolbox.count())]
    assert titles == ["Manage project", "Generate annotation data", "Train model", "Analyze videos"]
    assert dialog.project.root == tmp_path / "octron"
    assert dialog.project.root.is_dir()

    primary = shell.video_area.primary
    if getattr(primary, "has_video", False):
        cam = next(iter(dialog._cameras.values()))
        assert primary._label_mode is cam.mode
        assert dialog.frame_table.rowCount() == len(dialog._cameras)
    assert dialog.frame_table.columnCount() == 1 + len(dialog._individuals)

    dialog.close()
    assert getattr(primary, "_label_mode", None) is None


def test_right_click_reaches_a_mode_that_declares_it(qapp):
    """The camera view forwards an unmodified right press only to a mode with handle_right_click."""
    from types import SimpleNamespace

    from ethograph.gui.pygfx_video import CameraView

    view = CameraView()
    calls: list[tuple[str, float, float]] = []

    class Mode:
        locked = False

        def handle_click(self, x, y):
            calls.append(("left", x, y))

        def handle_right_click(self, x, y):
            calls.append(("right", x, y))

        def handle_move(self, x, y):
            pass

        def handle_release(self, x, y):
            pass

    view._label_mode = Mode()
    view.screen_to_image = lambda x, y: (x * 2, y * 2)
    view._dispatch_label(SimpleNamespace(x=1.0, y=2.0, button=2, modifiers=()), "handle_click")
    view._dispatch_label(SimpleNamespace(x=1.0, y=2.0, button=2, modifiers=("Shift",)), "handle_click")
    view._dispatch_label(SimpleNamespace(x=3.0, y=4.0, button=1, modifiers=()), "handle_click")
    assert calls == [("right", 2.0, 4.0), ("left", 6.0, 8.0)]


def test_motion_panel_opens_once_from_a_cached_trace(birdpark_gui, qtbot, monkeypatch, tmp_path):
    """Show pixel motion registers the trace as a derived feature and opens one line plot per camera."""
    import numpy as np

    from ethograph.io.derived import derived_loader_for

    shell, meta = birdpark_gui
    data_widget = meta.data_widget
    meta.app_state.project_path = str(tmp_path)
    dialog = data_widget.open_box_labelling()
    qtbot.addWidget(dialog)
    if not dialog._cameras:
        pytest.skip("no camera view with video")
    cam = next(iter(dialog._cameras.values()))
    n = int(getattr(cam.view, "n_frames", 0) or meta.app_state.num_frames or 0)
    monkeypatch.setattr(dialog, "_motion_trace", lambda cam, fps: np.linspace(0, 1, max(n, 2), dtype=np.float32))

    container = data_widget.plot_container
    before = len(container.feature_plots) if hasattr(container, "feature_plots") else None
    dialog._show_motion_panel(cam)
    dialog._show_motion_panel(cam)  # a second run must not open a second panel

    loader = derived_loader_for(meta.app_state)
    assert loader.is_derived("video_motion")
    assert loader._derived["video_motion"].n_columns == len(dialog._cameras)
    if before is not None:
        assert len(container.feature_plots) == before + 1
    dialog.close()
