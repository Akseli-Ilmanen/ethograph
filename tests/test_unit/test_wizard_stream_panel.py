"""StreamPanel: the extension filter over a folder that mixes file types."""

from __future__ import annotations

from pathlib import Path

from ethograph.gui.wizard_media_files import StreamPanel


def _mixed_pose_folder(tmp_path: Path) -> Path:
    d = tmp_path / "dlc"
    d.mkdir()
    for name in ("t1_camA.h5", "t1_camA.csv", "t2_camA.h5", "t2_camA.csv"):
        (d / name).touch()
    return d


def test_single_extension_folder_offers_no_choice(qtbot, tmp_path: Path):
    d = tmp_path / "video"
    d.mkdir()
    (d / "a.mp4").touch()
    panel = StreamPanel("video")
    qtbot.addWidget(panel)
    panel.set_folder(str(d))
    assert panel.extension is None
    assert panel._ext_row.isVisibleTo(panel) is False
    assert panel.get_config().extension is None


def test_mixed_folder_filters_files_by_the_chosen_extension(qtbot, tmp_path: Path):
    d = _mixed_pose_folder(tmp_path)
    panel = StreamPanel("pose")
    qtbot.addWidget(panel)
    panel.set_folder(str(d))

    assert panel._ext_row.isVisibleTo(panel) is True
    assert panel.extension == ".csv"  # first alphabetically; the user picks otherwise
    assert {f.suffix for f in panel._all_files} == {".csv"}

    panel.set_extension(".h5")
    assert {f.suffix for f in panel._all_files} == {".h5"}
    assert panel.get_config().extension == ".h5"
    assert panel.first_file() == str(d / "t1_camA.h5")


def test_config_round_trips_the_extension(qtbot, tmp_path: Path):
    d = _mixed_pose_folder(tmp_path)
    panel = StreamPanel("pose")
    qtbot.addWidget(panel)
    panel.set_folder(str(d))
    panel.set_extension(".h5")
    cfg = panel.get_config()

    other = StreamPanel("pose")
    qtbot.addWidget(other)
    other.apply_config(cfg)
    assert other.extension == ".h5"
