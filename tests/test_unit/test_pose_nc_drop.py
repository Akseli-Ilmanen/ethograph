"""A movement dataset saved as ``.nc`` is a pose file as well as a feature source.

Dropped on the cover page it pairs with a camera as a ``pose_cam-N`` stream
like a DLC ``.h5`` does, so the overlay draws it — points for a poses
dataset, boxes for a bboxes one — without asking which tracking tool wrote
it; and, like every dropped ``.nc``, it feeds the session dataset too
(``tests/test_unit/test_nc_drop.py`` for the combining rules).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import xarray as xr

from ethograph.gui.cover_page import CoverPage, _FeatureEntry, _open_pose_dataset, _pose_file_fps, classify_files
from ethograph.gui.pose_overlay import PoseOverlayData
from ethograph.gui.pose_render import load_pose_from_file
from ethograph.io.validation import movement_dataset_info

FPS = 25.0
N = 10


def _bboxes(fps: float | None = FPS) -> xr.Dataset:
    position = np.zeros((N, 2, 2))
    position[:, 0, :] = np.arange(N)[:, None] + 100  # x
    position[:, 1, :] = 200
    position[3, :, 1] = np.nan
    return xr.Dataset(
        {
            "position": (("time", "space", "individual"), position),
            "shape": (("time", "space", "individual"), np.full((N, 2, 2), 40.0)),
            "confidence": (("time", "individual"), np.full((N, 2), 0.9)),
        },
        coords={
            "time": np.arange(N) / fps if fps else np.arange(N),
            "space": ["x", "y"],
            "individual": ["m", "f"],
        },
        attrs={"ds_type": "bboxes", "source_software": "OCTRON", **({"fps": fps} if fps else {})},
    )


def _poses() -> xr.Dataset:
    position = np.random.default_rng(0).random((N, 2, 2, 1)) * 100
    return xr.Dataset(
        {
            "position": (("time", "space", "keypoint", "individual"), position),
            "confidence": (("time", "keypoint", "individual"), np.ones((N, 2, 1))),
        },
        coords={"time": np.arange(N) / FPS, "space": ["x", "y"], "keypoint": ["nose", "tail"], "individual": ["a"]},
        attrs={"ds_type": "poses", "fps": FPS},
    )


def test_a_movement_nc_is_a_pose_file_and_a_features_nc_is_a_session(tmp_path):
    boxes, poses, feats = tmp_path / "boxes.nc", tmp_path / "poses.nc", tmp_path / "feats.nc"
    _bboxes().to_netcdf(boxes)
    _poses().to_netcdf(poses)
    xr.Dataset({"speed": ("time", np.zeros(N))}, coords={"time": np.arange(N) / FPS}).to_netcdf(feats)

    assert movement_dataset_info(boxes) == ("bboxes", FPS)
    assert movement_dataset_info(poses) == ("poses", FPS)
    assert movement_dataset_info(feats) is None
    assert movement_dataset_info(tmp_path / "missing.nc") is None

    buckets = classify_files([str(boxes), str(poses), str(feats), str(tmp_path / "missing.nc")])
    assert buckets["pose"] == [str(boxes), str(poses)]
    assert buckets["session"] == [str(feats), str(tmp_path / "missing.nc")]


def test_the_frame_rate_comes_from_the_file_when_it_has_one(tmp_path):
    with_fps, frames = tmp_path / "a.nc", tmp_path / "b.nc"
    _bboxes().to_netcdf(with_fps)
    _bboxes(fps=None).to_netcdf(frames)
    assert _pose_file_fps(str(with_fps)) == FPS
    assert _pose_file_fps(str(frames)) is None, "frames-only: the drop dialog must ask"


def test_a_bboxes_nc_renders_as_boxes_without_a_source_software(tmp_path):
    path = tmp_path / "boxes.nc"
    _bboxes().to_netcdf(path)

    pr = load_pose_from_file(str(path), source_software=None, fps=FPS)
    assert pr.bbox_data is not None
    assert pr.keypoints == []

    overlay = PoseOverlayData(pr)
    assert overlay.n_tracks == 2
    assert overlay.bbox_shown is not None
    assert overlay.bbox_shown.sum() == 2 * N - 1, "one box is NaN on one frame"
    corners = overlay.bbox_corners[0, 0]  # frame 0, individual m: x 80..120, y 180..220
    assert np.isclose(corners[:, 0].min(), 80) and np.isclose(corners[:, 0].max(), 120)
    assert np.isclose(corners[:, 1].min(), 180) and np.isclose(corners[:, 1].max(), 220)


def test_standalone_nc_poses_become_features_with_their_own_fps(tmp_path):
    """No video, no fps asked: the file's rate is used and the stacked
    features .nc keeps it."""
    a, b = tmp_path / "a.nc", tmp_path / "b.nc"
    _bboxes().to_netcdf(a)
    _bboxes().to_netcdf(b)

    assert _open_pose_dataset(str(a), None, None).attrs["ds_type"] == "bboxes"
    entries = [_FeatureEntry("cam-1", str(a), None), _FeatureEntry("cam-2", str(b), None)]
    out = CoverPage._build_session_nc(entries, None, None, tmp_path)
    with xr.open_dataset(out) as ds:
        assert ds.attrs["fps"] == FPS
        assert list(ds.coords["camera"].values) == ["cam-1", "cam-2"]
        assert "position" in ds.data_vars


def _dlc_csv(path, fps: float = FPS):
    from movement.io import save_poses

    save_poses.to_dlc_file(_poses().assign_attrs(source_software="DeepLabCut", fps=fps), path, split_individuals=False)


def test_a_tracking_tools_file_with_a_video_is_a_feature_read_at_the_videos_rate(tmp_path, monkeypatch):
    """DLC/SLEAP files used to be overlay-only beside a video; now their
    position/confidence plot too, and the rate is the video's, which the
    drop dialog never asked for (no pose_fps)."""
    import ethograph.gui.cover_page as cover_page

    dlc = tmp_path / "trial_DLC.csv"
    _dlc_csv(dlc)
    monkeypatch.setattr(
        cover_page, "probe_video", lambda _p: SimpleNamespace(fps=50.0, width=640, height=480, nframes=N), raising=False
    )
    monkeypatch.setattr("ethograph.gui.video_manager.probe_video", cover_page.probe_video)
    cam_map = [(str(tmp_path / "trial.mp4"), str(dlc))]
    entries = CoverPage._feature_entries(SimpleNamespace(), cam_map, [])
    assert entries == [_FeatureEntry("cam-1", str(dlc), 50.0)]
    assert cam_map == [(str(tmp_path / "trial.mp4"), str(dlc))], "still the camera's overlay"

    out = CoverPage._build_session_nc(entries, "DeepLabCut", None, tmp_path)
    with xr.open_dataset(out) as ds:
        assert {"position", "confidence"} <= set(ds.data_vars)
        assert ds.attrs["fps"] == 50.0
        assert ds["time"].values[1] == 1 / 50.0


def test_speed_is_added_once_and_only_when_asked(tmp_path):
    poses, feats = tmp_path / "poses.nc", tmp_path / "feats.nc"
    _poses().to_netcdf(poses)
    _poses().assign(speed=(("time", "keypoint", "individual"), np.full((N, 2, 1), 3.0))).to_netcdf(feats)
    entries = [_FeatureEntry("cam-1", str(poses), None), _FeatureEntry("cam-2", str(feats), None)]

    (tmp_path / "plain").mkdir()
    with xr.open_dataset(CoverPage._build_session_nc(entries, None, None, tmp_path / "plain")) as ds:
        assert np.isnan(ds["speed"].sel(camera="cam-1")).all(), "only the file's own speed, padded"
    with xr.open_dataset(CoverPage._build_session_nc(entries, None, None, tmp_path, compute_speed=True)) as ds:
        assert ds["speed"].dims == ("camera", "time", "keypoint", "individual")
        assert (ds["speed"].sel(camera="cam-2") == 3.0).all(), "a file's own speed is kept"
        assert np.isfinite(ds["speed"].sel(camera="cam-1").values[1:-1]).all()
