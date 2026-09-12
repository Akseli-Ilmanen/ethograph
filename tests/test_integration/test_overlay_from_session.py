"""A dropped bounding-box .nc is drawn from the loaded dataset, never re-read from disk.

Regression: once a single dropped ``.nc`` became the session file in place, the
overlay's own ``open_dataset`` of that same file collided with the loader's
handles and crashed natively. The overlay now reads the session dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from qtpy.QtWidgets import QApplication

av = pytest.importorskip("av")

FPS = 10
N = 10  # frames


def _write_video(path: Path) -> str:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=FPS)
    stream.width, stream.height = 64, 64
    stream.pix_fmt = "yuv420p"
    for i in range(N):
        img = np.full((64, 64, 3), i * 8 % 256, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return str(path)


def _write_bboxes(path: Path) -> str:
    position = np.zeros((N, 2, 2))
    position[:, 0, :] = np.arange(N)[:, None] + 20  # x
    position[:, 1, :] = 30  # y
    xr.Dataset(
        {
            "position": (("time", "space", "individual"), position),
            "shape": (("time", "space", "individual"), np.full((N, 2, 2), 8.0)),
            "confidence": (("time", "individual"), np.full((N, 2), 0.9)),
        },
        coords={"time": np.arange(N) / FPS, "space": ["x", "y"], "individual": ["m", "f"]},
        attrs={"ds_type": "bboxes", "source_software": "OCTRON", "fps": float(FPS)},
    ).to_netcdf(path)
    return str(path)


def test_boxes_come_from_the_session_dataset(gui, tmp_path: Path, qtbot):
    from ethograph.gui.cover_page import CoverPage, classify_files

    shell, meta = gui
    shell.show()
    paths = [_write_video(tmp_path / "cam.mp4"), _write_bboxes(tmp_path / "cam.nc")]
    page = CoverPage(shell, meta.io_widget)
    page._populate_io_from_buckets(
        classify_files(paths),
        {"data_sr": None, "source_software": None, "pose_fps": None, "extract_audio": False, "audio_track_videos": []},
    )
    meta.data_widget.on_load_clicked()
    QApplication.processEvents()
    qtbot.wait(200)
    assert meta.app_state.ready

    pm = meta.data_widget.pose_mgr
    pr = pm._primary_pr
    assert pr is not None and pr.bbox_data is not None, "boxes are drawn"
    assert pr.file_name == "position", "read off the dataset, not the file"
    assert pm.pose_kind() == "bboxes"

    combo = meta.data_widget.overlay_feature_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["position"]

    # No click needed: the sidebar re-reads what is drawn after every pose refresh.
    assert meta.context_panel.current_context() == "video"
    assert meta.context_panel._sections["bbox"].isVisibleTo(meta.context_panel)
    assert not meta.context_panel._sections["pose"].isVisibleTo(meta.context_panel)
