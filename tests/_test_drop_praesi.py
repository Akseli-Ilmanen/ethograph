"""Ad-hoc: the real drag & drop of a folder whose path holds a non-ASCII character."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import QApplication

from ethograph.gui.cover_page import CoverPage

FOLDER = Path(r"C:\Users\aksel\Desktop\präsi\octron tracking")
DROPPED = [
    "BP_2021-05-25_08-12-51_655154_0380000.wav",
    "tracks_overview.mp4",
    "tracks_zoom.mp4",
    "tracks_zoom_snippet.mp4",
    "BP_2021-05-25_08-12-51_655154_0380000.nc",
]


def test_drop_praesi(gui, qtbot):
    shell, meta = gui
    shell.show()
    page = CoverPage(shell, meta.io_widget)
    page._drop.paths = [str(FOLDER / name) for name in DROPPED]
    assert page._prepare_dropped()
    print("nc_file_path:", meta.app_state.nc_file_path)
    meta.data_widget.on_load_clicked()
    QApplication.processEvents()
    qtbot.wait(500)
    assert meta.app_state.ready, "the drop must load"
    print("features:", [c for c in meta.data_widget.catalog.feature_choices()][:10])
    print("cameras:", len(shell.video_area.camera_docks()))
