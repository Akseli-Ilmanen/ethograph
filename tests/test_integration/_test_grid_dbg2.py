import sys

sys.path.insert(0, "tests/test_integration")
from test_camera_grid_load import _write_video, _load_dropped_videos  # noqa


def test_dbg(gui, tmp_path, qtbot):
    shell, meta = gui
    shell.show()
    from ethograph.gui import video_manager, widgets_meta

    real_apply = widgets_meta.MetaWidget.arrange_camera_grid_if_default
    real_grid = video_manager.VideoArea.arrange_grid

    def spy_apply(self):
        va = self.shell.video_area
        print(
            "APPLY pending=",
            self._camera_grid_pending,
            "docks=",
            len(va.camera_docks()),
            "extras=",
            list(va.extras),
            "hidden=",
            [getattr(v, "dock_widget", None) is None or v.dock_widget.isHidden() for v in va.extras.values()],
        )
        return real_apply(self)

    def spy_grid(self):
        r = real_grid(self)
        print("GRID ->", r, "docks", len(self.camera_docks()))
        return r

    real_add = video_manager.VideoArea.add_extra

    def spy_add(self, name):
        import traceback

        print(
            "ADD_EXTRA",
            name,
            "|",
            " <- ".join(
                f"{fr.name}:{fr.lineno}" for fr in traceback.extract_stack()[-6:-1] if "ethograph" in fr.filename
            ),
        )
        return real_add(self, name)

    video_manager.VideoArea.add_extra = spy_add
    widgets_meta.MetaWidget.arrange_camera_grid_if_default = spy_apply
    video_manager.VideoArea.arrange_grid = spy_grid
    vids = [_write_video(tmp_path / f"cam{i}.mp4") for i in range(1, 5)]
    _load_dropped_videos(shell, meta, vids)
    print("LAYOUT", type(meta.app_state.panel_layout), bool(meta.app_state.panel_layout))
    qtbot.wait(200)
    docks = shell.video_area.camera_docks()
    print("GEOM", [(d.objectName(), d.x(), d.y(), d.width(), d.height()) for d in docks])
    widgets_meta.MetaWidget.arrange_camera_grid_if_default = real_apply
    video_manager.VideoArea.arrange_grid = real_grid
