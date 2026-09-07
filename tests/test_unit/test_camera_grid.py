"""Several cameras with no saved layout tile into a grid, never one long row.

Every extra camera is a dock added to the same (top) area, so Qt lines them
up side by side; four dropped videos became four slivers. With no layout to
honour, the load arranges them as ``camera_grid_shape`` picks.
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication

from ethograph.gui.video_manager import camera_grid_shape


@pytest.mark.parametrize(
    ("n", "shape"),
    [(1, (1, 1)), (2, (1, 2)), (3, (1, 3)), (4, (2, 2)), (5, (2, 3)), (6, (2, 3)), (8, (3, 3)), (9, (3, 3))],
)
def test_grid_shape(n, shape):
    assert camera_grid_shape(n) == shape


def test_grid_shape_rejects_nothing():
    with pytest.raises(ValueError):
        camera_grid_shape(0)


def _cells(docks):
    return {(d.y(), d.x()) for d in docks}


class TestArrangeGrid:
    def _add(self, shell, n):
        area = shell.video_area
        views = [area.add_extra(f"cam-{i + 2}") for i in range(n)]
        QApplication.processEvents()
        return views

    def test_four_cameras_fill_two_rows(self, gui, qtbot):
        shell, _meta = gui
        shell.show()
        self._add(shell, 3)  # + the primary = 4 docks
        assert shell.video_area.arrange_grid() == (2, 2)
        qtbot.wait(50)
        docks = shell.video_area.camera_docks()
        assert len(docks) == 4
        assert len({d.y() for d in docks}) == 2, "two rows"
        assert len({d.x() for d in docks}) == 2, "two columns"
        assert len(_cells(docks)) == 4, "every dock in its own cell"

    def test_three_cameras_stay_in_one_row(self, gui, qtbot):
        shell, _meta = gui
        shell.show()
        self._add(shell, 2)
        assert shell.video_area.arrange_grid() is None
        qtbot.wait(50)
        docks = shell.video_area.camera_docks()
        assert len({d.y() for d in docks}) == 1

    def test_six_cameras_make_two_by_three(self, gui, qtbot):
        shell, _meta = gui
        shell.show()
        self._add(shell, 5)
        assert shell.video_area.arrange_grid() == (2, 3)
        qtbot.wait(50)
        docks = shell.video_area.camera_docks()
        assert len({d.y() for d in docks}) == 2
        assert len({d.x() for d in docks}) == 3
        assert len(_cells(docks)) == 6


class TestLoadPath:
    """Deciding to tile and tiling are two steps, and the split is the point:
    the layout is applied before the trial creates the camera views.

    The ``gui`` fixture wraps ``apply_saved_panel_layout`` to wipe the layout
    first, so these call the class method directly."""

    def _apply_layout(self, meta):
        type(meta).apply_saved_panel_layout(meta)

    def test_no_saved_layout_defers_the_tiling(self, gui, monkeypatch):
        _shell, meta = gui
        calls = []
        monkeypatch.setattr(meta.shell.video_area, "arrange_grid", lambda: calls.append(1))
        meta.app_state.panel_layout = None

        self._apply_layout(meta)
        assert calls == [], "no camera view exists yet — tiling here would see one dock"
        assert meta._camera_grid_pending

        meta.arrange_camera_grid_if_default()
        assert calls == [1]

    def test_tiling_happens_once_per_load(self, gui, monkeypatch):
        _shell, meta = gui
        calls = []
        monkeypatch.setattr(meta.shell.video_area, "arrange_grid", lambda: calls.append(1))
        meta.app_state.panel_layout = None
        self._apply_layout(meta)

        meta.arrange_camera_grid_if_default()
        meta.arrange_camera_grid_if_default()
        assert calls == [1], "a later trial change must not re-tile the user's arrangement"

    def test_saved_layout_is_honoured_instead(self, gui, monkeypatch):
        _shell, meta = gui
        calls = []
        monkeypatch.setattr(meta.shell.video_area, "arrange_grid", lambda: calls.append(1))
        monkeypatch.setattr(meta.plot_container, "apply_layout_state", lambda layout: None)
        monkeypatch.setattr(meta.data_widget, "apply_space_layout_state", lambda s: None)
        monkeypatch.setattr(meta.data_widget, "apply_radial_layout_state", lambda s: None)
        meta.app_state.panel_layout = {"panels": []}

        self._apply_layout(meta)
        meta.arrange_camera_grid_if_default()
        assert calls == []
