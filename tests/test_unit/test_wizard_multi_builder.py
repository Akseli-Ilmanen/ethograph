"""The multi-trial builder must find media the trial table names by basename only."""

from pathlib import Path

import pandas as pd
import pytest

from ethograph.gui import wizard_multi_builder as builder
from ethograph.gui.wizard_media_files import FilePattern
from ethograph.gui.wizard_state import ModalityConfig, WizardState


def _video_cfg(**kwargs) -> ModalityConfig:
    return ModalityConfig(enabled=True, file_mode="aligned_to_trial", fps=30, **kwargs)


def test_resolve_by_pattern_files(tmp_path: Path):
    video = tmp_path / "vids" / "2024-12-17_001_Crow1-cam-1.mp4"
    video.parent.mkdir()
    video.touch()
    cfg = _video_cfg(pattern=FilePattern(segments=[], files=[video], suffix=".mp4"))
    assert builder.resolve_media_path(cfg, video.name) == video


def test_resolve_by_folder_nested(tmp_path: Path):
    video = tmp_path / "day1" / "sub" / "t1-cam-1.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    cfg = _video_cfg(folder_path=str(tmp_path), nested_subfolders=True)
    assert builder.resolve_media_path(cfg, video.name) == video
    assert builder.resolve_media_path(_video_cfg(folder_path=str(tmp_path)), video.name) is None


def test_with_media_paths_resolves_basenames_and_keeps_blanks(tmp_path: Path):
    files = [tmp_path / f"t{i}-cam-1.mp4" for i in (1, 2)]
    for f in files:
        f.touch()
    state = WizardState(video=_video_cfg(pattern=FilePattern(segments=[], files=files, suffix=".mp4")))
    table = pd.DataFrame({"trial": [1, 2, 3], "video_cam-1": [files[0].name, str(files[1]), ""]})

    out = builder.with_media_paths(table, state)
    assert out["video_cam-1"].tolist() == [str(files[0]), str(files[1]), ""]
    assert table["video_cam-1"].tolist()[0] == files[0].name  # input untouched


def test_with_media_paths_unresolvable_names_the_trial():
    state = WizardState(video=_video_cfg())
    table = pd.DataFrame({"trial": [7], "video_cam-1": ["missing.mp4"]})
    with pytest.raises(ValueError, match="Trial 7.*video_cam-1.*missing.mp4"):
        builder.with_media_paths(table, state)
