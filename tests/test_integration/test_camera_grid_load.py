"""Dropping several videos tiles the camera panels, end to end.

The grid is decided while the saved layout is applied, but the camera views do
not exist yet at that point — they are created with the first trial, one step
later — so tiling there saw a single dock and did nothing. Four dropped videos
arrived as one row of slivers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication

av = pytest.importorskip("av")


def _write_video(path: Path, *, seconds: float = 1.0, fps: int = 10) -> str:
    """A tiny real .mp4 — the drop path probes it for its rate and duration."""
    import numpy as np

    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height = 64, 64
    stream.pix_fmt = "yuv420p"
    for i in range(int(seconds * fps)):
        img = np.full((64, 64, 3), i * 8 % 256, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return str(path)


def _load_dropped_videos(shell, meta, paths: list[str]) -> None:
    from ethograph.gui.cover_page import CoverPage, classify_files

    page = CoverPage(shell, meta.io_widget)
    page._populate_io_from_buckets(
        classify_files(paths),
        {"data_sr": None, "source_software": None, "pose_fps": None, "extract_audio": False, "audio_track_videos": []},
    )
    meta.data_widget.on_load_clicked()
    QApplication.processEvents()


@pytest.fixture
def four_videos(tmp_path):
    return [_write_video(tmp_path / f"cam{i}.mp4") for i in range(1, 5)]


def test_four_dropped_videos_tile_two_by_two(gui, four_videos, qtbot):
    shell, meta = gui
    shell.show()
    _load_dropped_videos(shell, meta, four_videos)
    assert meta.app_state.ready, "the drop must load"

    docks = shell.video_area.camera_docks()
    assert len(docks) == 4, "one panel per dropped video, primary included"
    qtbot.wait(100)
    assert len({d.y() for d in docks}) == 2, "two rows, not one long row of slivers"
    assert len({d.x() for d in docks}) == 2, "two columns"
    assert len({(d.y(), d.x()) for d in docks}) == 4, "every panel in its own cell"


def test_two_dropped_videos_stay_side_by_side(gui, four_videos, qtbot):
    shell, meta = gui
    shell.show()
    _load_dropped_videos(shell, meta, four_videos[:2])
    assert meta.app_state.ready

    docks = shell.video_area.camera_docks()
    assert len(docks) == 2
    qtbot.wait(100)
    assert len({d.y() for d in docks}) == 1, "two panels need no second row"
