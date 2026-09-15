"""Video features from files: matched by video name, sampled onto the trial
clock, plain features downstream, named after their folder, never saved back;
a session's ``video_feature_folders`` attach the same way at open."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from ethograph.io.data_loader import LoadResult
from ethograph.io.trialtree import TrialTree
from ethograph.io.video_feature_files import (
    ATTACHED_FROM,
    VideoFeatureFiles,
    attach_video_features,
    detect_device,
    feature_name_for,
    sample_on_trial,
    video_feature_sources,
    video_feature_vars,
)
from ethograph.segment.config import SessionSpec, load_config
from ethograph.segment.sessions import Session, attach_video_feature_folders

FPS = 100.0
N = 30
FILES = {1: "trial001-cam-1.mp4", 2: "trial002-cam-1.mp4"}
SIDE_FILES = {1: "trial001-cam-2.mp4", 2: "trial002-cam-2.mp4"}


class _Alignment:
    """Names each trial's video per camera; sample 0 of the video sits at *offset*."""

    cameras: list[str] = []
    mics: list[str] = []

    def __init__(self, rate: float | None = FPS, offset: float = 0.0) -> None:
        self.rate, self.offset = rate, offset
        self.by_device = {"cam-1": FILES, "cam-2": SIDE_FILES}

    def media_filename(self, trial, stream, device=None):
        return self.by_device.get(device or "cam-1", {}).get(trial)

    def devices(self, stream):
        return list(self.by_device)

    def get_stream_rate(self, stream, device=None):
        return self.rate

    def stream_offset_for_trial(self, trial, stream, device=None):
        return self.offset


def _tree(tmp_path: Path) -> tuple[TrialTree, Path]:
    time = np.arange(N) / FPS
    coords = {"time": time, "individual": ["A"]}
    trials = [
        xr.Dataset({"speed": (("time", "individual"), np.ones((N, 1)))}, coords=coords, attrs={"trial": t})
        for t in FILES
    ]
    source = tmp_path / "session.nc"
    TrialTree.from_datasets(trials).save(source)
    return TrialTree.open(str(source)), source


def _write_arrays(folder: Path, n_frames: int = N, width: int = 4, trials=FILES) -> Path:
    """Column 0 counts frames, so a sampled value says which frame was read."""
    folder.mkdir(parents=True, exist_ok=True)
    for trial, name in trials.items():
        data = np.zeros((n_frames, width), dtype=np.float32)
        data[:, 0] = np.arange(n_frames)
        data[:, 1] = trial
        np.save(folder / f"{Path(name).stem}.npy", data)
    return folder


def _feral(*paths: Path, **extra) -> VideoFeatureFiles:
    return VideoFeatureFiles.from_paths("feral", paths, **extra)


def test_attached_feature_is_a_plain_variable_on_every_trial(tmp_path: Path):
    dt, _ = _tree(tmp_path)
    folder = _write_arrays(tmp_path / "emb")

    result = attach_video_features(dt, _Alignment(), _feral(folder))

    assert (result.name, result.device, result.width, result.matched, result.n_trials) == ("feral", "cam-1", 4, 2, 2)
    assert result.missing == () and result.coverage == 1.0
    for trial in FILES:
        da = dt.trial(trial)["feral"]
        assert da.dims == ("time", "feral_dims")
        assert da.shape == (N, 4)
        assert da.attrs["kind"] == "video_feature"
        assert da.attrs[ATTACHED_FROM] == str(folder)
        np.testing.assert_allclose(da.values[:, 1], trial)
    np.testing.assert_allclose(dt.trial(1)["feral"].values[:, 0], np.arange(N))
    assert video_feature_vars(dt.trial(1)) == ["feral"]
    assert video_feature_sources(dt.trial(1)) == {"feral": str(folder)}


def test_loose_files_are_one_feature_named_after_their_folder(tmp_path: Path):
    dt, _ = _tree(tmp_path)
    folder = _write_arrays(tmp_path / "emb")
    spec = _feral(folder / "trial001-cam-1.npy")
    result = attach_video_features(dt, _Alignment(), spec)
    assert spec.source == f"1 files in {folder}"
    assert (result.matched, result.missing) == (1, (2,))


def test_a_path_that_is_neither_folder_nor_npy_is_refused(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(FileNotFoundError, match="notes.txt"):
        _feral(tmp_path / "notes.txt")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No .npy files"):
        _feral(tmp_path / "empty")


def test_video_offset_reads_the_earlier_frame(tmp_path: Path):
    """Trial t = 0.20 s with the video starting 0.10 s in → frame 10 (trial = video + offset)."""
    dt, _ = _tree(tmp_path)
    folder = _write_arrays(tmp_path / "emb", n_frames=60)
    attach_video_features(dt, _Alignment(offset=0.10), _feral(folder))
    frame = dt.trial(1)["feral"].values[:, 0]
    assert frame[20] == pytest.approx(10.0)
    assert frame[10] == pytest.approx(0.0)


def test_video_rate_above_trial_rate_strides_the_frames(tmp_path: Path):
    dt, _ = _tree(tmp_path)
    folder = _write_arrays(tmp_path / "emb", n_frames=2 * N)
    attach_video_features(dt, _Alignment(rate=2 * FPS), _feral(folder))
    np.testing.assert_allclose(dt.trial(1)["feral"].values[:, 0], 2 * np.arange(N))


def test_identity_sampling_keeps_the_array_itself():
    array = np.arange(12, dtype=np.float32).reshape(6, 2)
    assert sample_on_trial(array, np.arange(6) / 10.0, 10.0, 0.0) is array
    assert sample_on_trial(array, np.arange(6) / 10.0, 10.0, 0.1) is not array


def test_a_subset_is_fine_and_the_coverage_is_logged(tmp_path: Path, caplog):
    dt, _ = _tree(tmp_path)
    folder = _write_arrays(tmp_path / "emb", trials={1: FILES[1]})
    with caplog.at_level(logging.INFO):
        result = attach_video_features(dt, _Alignment(), _feral(folder))
    assert result.missing == (2,) and result.coverage == 0.5
    assert np.isnan(dt.trial(2)["feral"].values).all()
    assert not np.isnan(dt.trial(1)["feral"].values).any()
    assert "1 of 2 videos matched (50%)" in caplog.text


def test_files_matching_no_video_are_an_error(tmp_path: Path):
    dt, _ = _tree(tmp_path)
    folder = _write_arrays(tmp_path / "emb", trials={1: "unrelated.mp4"})
    with pytest.raises(FileNotFoundError, match="trial001-cam-1.npy"):
        attach_video_features(dt, _Alignment(), _feral(folder))


def test_alignment_without_a_rate_is_an_error(tmp_path: Path):
    dt, _ = _tree(tmp_path)
    with pytest.raises(ValueError, match="declares no rate"):
        attach_video_features(dt, _Alignment(rate=None), _feral(_write_arrays(tmp_path / "emb")))


def test_attaching_again_replaces_the_variable(tmp_path: Path):
    dt, _ = _tree(tmp_path)
    attach_video_features(dt, _Alignment(), _feral(_write_arrays(tmp_path / "a", width=4)))
    attach_video_features(dt, _Alignment(), _feral(_write_arrays(tmp_path / "b", width=6)))
    assert dt.trial(1)["feral"].shape == (N, 6)
    assert dt.trial(1).sizes["feral_dims"] == 6


class TestDetectDevice:
    def test_the_camera_whose_names_match_wins(self, tmp_path: Path):
        dt, _ = _tree(tmp_path)
        assert detect_device(dt, _Alignment(), _feral(_write_arrays(tmp_path / "side", trials=SIDE_FILES))) == "cam-2"
        assert detect_device(dt, _Alignment(), _feral(_write_arrays(tmp_path / "top"))) == "cam-1"

    def test_files_matching_two_cameras_equally_are_refused(self, tmp_path: Path):
        dt, _ = _tree(tmp_path)
        folder = _write_arrays(tmp_path / "both")
        _write_arrays(folder, trials=SIDE_FILES)
        with pytest.raises(ValueError, match="cam-1"):
            detect_device(dt, _Alignment(), _feral(folder))

    def test_a_given_device_is_not_second_guessed(self, tmp_path: Path):
        dt, _ = _tree(tmp_path)
        folder = _write_arrays(tmp_path / "side", trials=SIDE_FILES)
        with pytest.raises(FileNotFoundError):
            attach_video_features(dt, _Alignment(), _feral(folder, device="cam-1"))


class TestFeatureName:
    def test_named_after_the_folder(self):
        assert feature_name_for(Path("D:/emb/feral cfg-A")) == "feral_cfg_A"
        assert feature_name_for("/x/2025_run") == "f_2025_run"

    def test_unique_against_existing_variables(self):
        assert feature_name_for("/x/feral", taken=["feral", "feral_2"]) == "feral_3"


@pytest.mark.parametrize("incremental", [False, True])
def test_save_never_writes_the_video_feature(tmp_path: Path, incremental: bool):
    """Both save paths drop it from the file and keep it on the tree."""
    dt, _ = _tree(tmp_path)
    attach_video_features(dt, _Alignment(), _feral(_write_arrays(tmp_path / "emb")))

    if incremental:
        dt.set_trial_attr(1, "note", "edited")  # one dirty trial → in-place group write
        dt.save()
    else:
        dt.save(tmp_path / "copy.nc")

    written = TrialTree.open(str(tmp_path / ("session.nc" if incremental else "copy.nc")))
    for trial in FILES:
        assert "feral" not in written.trial(trial)
        assert "feral" in dt.trial(trial)
    if incremental:
        assert written.trial(1).attrs["note"] == "edited"


# ---------------------------------------------------------------------------
# The pipeline: a session's video_feature_folders
# ---------------------------------------------------------------------------


def test_config_resolves_video_feature_folders_against_the_file(tmp_path: Path):
    (tmp_path / "s.nc").write_bytes(b"")
    (tmp_path / "mapping.txt").write_text("0 background\n1 flap\n", encoding="utf-8")
    config = {
        "sessions": [{"source": "s.nc", "video_feature_folders": {"feral": "emb/cfgA"}}],
        "features": {"columns": {"feral": {"feral_dims": [0, 1]}}, "labels": {"mapping": "mapping.txt", "branch": 0}},
    }
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    spec = load_config(tmp_path / "project.yaml", []).sessions[0]
    assert spec.video_feature_folders == {"feral": (tmp_path / "emb" / "cfgA").resolve()}


def test_session_open_attaches_the_folders_and_the_catalog_lists_them(tmp_path: Path):
    """What ``open_session`` does after loading: the variable exists for materialise."""
    from ethograph.features.columns import extract_features
    from ethograph.io.catalog import XarrayLoader, catalog_from_xarray

    dt, source = _tree(tmp_path)
    folder = _write_arrays(tmp_path / "emb", width=3)
    alignment = _Alignment()
    catalog = catalog_from_xarray(dt.itrial(0), dt, nwb_alignment=alignment)
    spec = SessionSpec(source=source, video_feature_folders={"feral": folder})
    session = Session(spec=spec, id="s", result=LoadResult(dt=dt, nwb_alignment=alignment, catalog=catalog))

    results = attach_video_feature_folders(session)

    assert [r.name for r in results] == ["feral"]
    assert "feral" in session.result.catalog.feature_choices()
    _, data = extract_features(XarrayLoader(dt.trial(1)), {"feral": {"feral_dims": ["0", "2"]}})
    assert data.shape == (N, 2)
    plain = Session(spec=SessionSpec(source=source), id="t", result=LoadResult(dt=dt))
    assert attach_video_feature_folders(plain) == []
