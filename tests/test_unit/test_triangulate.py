"""Calibration import, array triangulation and the world frame (ethograph/triangulate)."""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import xarray as xr
import yaml

pytest.importorskip("aniposelib")

from aniposelib.cameras import Camera, CameraGroup  # noqa: E402

import ethograph as eto
from ethograph.gui.plots_space import parse_geometry
from ethograph.io.overlay_source import overlay_candidates
from ethograph.io.trialtree import TrialTree
from ethograph.triangulate.calibration import (
    CalibrationError,
    import_dlc_calibration,
    load_calibration,
    resolve_calibration,
)
from ethograph.triangulate.frame import WorldFrame, load_frame, save_frame
from ethograph.triangulate.geometry import landmark_positions, write_geometry
from ethograph.triangulate.points import reproject_points, triangulate_points
from ethograph.triangulate.session import POSITION_3D, REPROJECTION_ERROR, triangulate_tree

_MATRIX = np.array([[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]])
_DIST = np.array([-0.2, 0.1, 0.001, -0.002, 0.03])
_R = cv2.Rodrigues(np.array([0.05, -0.4, 0.02]))[0]
_T = np.array([-6.0, 0.3, 1.2])


def _rig() -> CameraGroup:
    return CameraGroup(
        [
            Camera(name="left", size=[640, 480], matrix=_MATRIX, dist=_DIST),
            Camera(name="right", size=[640, 480], matrix=_MATRIX, dist=_DIST, rvec=cv2.Rodrigues(_R)[0], tvec=_T),
        ]
    )


def _scene(n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform([-2.0, -2.0, 18.0], [2.0, 2.0, 24.0], size=(n, 3))


def test_triangulation_recovers_projected_points():
    cgroup, scene = _rig(), _scene()
    p3d, error = triangulate_points(cgroup, cgroup.project(scene))
    np.testing.assert_allclose(p3d, scene, atol=1e-3)
    assert np.all(error < 1e-2)


def test_low_confidence_view_leaves_the_point_unseen():
    cgroup, scene = _rig(), _scene(4).reshape(2, 2, 3)
    points = reproject_points(cgroup, scene)
    scores = np.ones(points.shape[:-1])
    scores[1, 0, 1] = 0.2
    p3d, error = triangulate_points(cgroup, points, scores=scores, min_confidence=0.9)
    assert np.isnan(p3d[0, 1]).all() and np.isnan(error[0, 1])
    assert np.isfinite(p3d).all(axis=-1).sum() == 3


def _write_dlc_project(folder, rect: np.ndarray) -> None:
    (folder / "camera_matrix").mkdir(parents=True)
    with open(folder / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"camera_names": ["left", "right"]}, f)
    stereo = {
        "left-right": {
            "cameraMatrix1": _MATRIX,
            "cameraMatrix2": _MATRIX,
            "distCoeffs1": _DIST[None],
            "distCoeffs2": _DIST[None],
            "R": _R,
            "T": _T[:, None],
            "R1": rect,
            "image_shape": [(640, 480), (640, 480)],
        }
    }
    with open(folder / "camera_matrix" / "stereo_params.pickle", "wb") as f:
        pickle.dump(stereo, f)


def test_dlc_import_triangulates_in_deeplabcuts_rectified_frame(tmp_path):
    rect = cv2.Rodrigues(np.array([0.1, 0.2, -0.05]))[0]
    _write_dlc_project(tmp_path / "dlc3d", rect)
    path = import_dlc_calibration(tmp_path / "dlc3d", tmp_path / "project")
    assert path == resolve_calibration(tmp_path / "project")

    scene = _scene()
    p3d, _ = triangulate_points(load_calibration(path, ["left", "right"]), _rig().project(scene))
    np.testing.assert_allclose(p3d, scene @ rect.T, atol=1e-3)


def test_calibration_must_be_named_once_there_are_several(tmp_path):
    folder = tmp_path / "calibration"
    folder.mkdir()
    for name in ("rig_a", "rig_b"):
        _rig().dump(str(folder / f"{name}.toml"))
    with pytest.raises(CalibrationError, match="several"):
        resolve_calibration(tmp_path)
    assert resolve_calibration(tmp_path, "rig_b").stem == "rig_b"
    with pytest.raises(CalibrationError, match="no camera"):
        load_calibration(folder / "rig_a.toml", ["left", "top"])


def test_world_frame_is_rigid_and_survives_its_file(tmp_path):
    tilt = cv2.Rodrigues(np.array([0.3, -0.2, 0.5]))[0]
    room = {"o": np.zeros(3), "px": np.array([2.0, 0.0, 0.0]), "pz": np.array([0.4, 0.0, 3.0])}
    seen = {name: 5.0 * (p @ tilt.T) + np.array([1.0, 2.0, 3.0]) for name, p in room.items()}

    frame = WorldFrame(axes=(("x", "o", "px"), ("z", "o", "pz")), reference_point="o", scale=("o", "px", 2.0))
    transform = frame.fit(seen)
    np.testing.assert_allclose(transform.rotation @ transform.rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(transform.rotation) == pytest.approx(1.0)
    for name, p in room.items():
        np.testing.assert_allclose(transform.apply(seen[name]), p, atol=1e-9)

    calibration = tmp_path / "rig.toml"
    save_frame(calibration, WorldFrame(frame.axes, frame.reference_point, frame.scale, "m", transform))
    loaded = load_frame(calibration)
    assert loaded is not None and loaded.transform is not None
    np.testing.assert_allclose(loaded.transform.apply(seen["pz"]), room["pz"], atol=1e-9)


# -- a session: 2D points per camera → position_3d ---------------------------------

_KEYPOINTS = ["corner_a", "corner_b", "corner_c", "beak"]


def _tracks(n_frames: int = 12) -> np.ndarray:
    """``(time, keypoint, individual, 3)``: three fixed corners and a moving beak."""
    scene = np.empty((n_frames, len(_KEYPOINTS), 1, 3))
    scene[:, 0, 0] = [0.0, 0.0, 20.0]
    scene[:, 1, 0] = [1.5, 0.0, 20.5]
    scene[:, 2, 0] = [0.0, 1.0, 20.0]
    scene[:, 3, 0] = np.linspace([-1.0, -1.0, 19.0], [1.0, 1.0, 22.0], n_frames)
    return scene


def _camera_dataset(scene: np.ndarray, fps: float = 10.0) -> xr.Dataset:
    pixels = reproject_points(_rig(), scene)  # (camera, time, keypoint, individual, 2)
    coords = {
        "time": np.arange(len(scene)) / fps,
        "camera": ["left", "right"],
        "keypoint": _KEYPOINTS,
        "individual": ["bird"],
    }
    return xr.Dataset(
        {
            "position": (("camera", "time", "keypoint", "individual", "space"), pixels),
            "confidence": (("camera", "time", "keypoint", "individual"), np.ones(pixels.shape[:-1])),
        },
        coords={**coords, "space": ["x", "y"]},
    )


def _project_with_rig(tmp_path) -> Path:
    path = tmp_path / "project" / "calibration" / "rig.toml"
    path.parent.mkdir(parents=True)
    _rig().dump(str(path))
    return path


def test_camera_dim_feature_becomes_position_3d_and_stays_an_overlay(tmp_path):
    scene = _tracks()
    dt = eto.from_datasets([_camera_dataset(scene).assign_attrs(trial=1)])
    done = triangulate_tree(dt, None, _project_with_rig(tmp_path))
    assert done == [1]

    ds = dt.trial(1)
    got = ds[POSITION_3D].transpose("time", "keypoint", "individual", "space").values
    np.testing.assert_allclose(got, scene, atol=1e-3)
    assert float(ds[REPROJECTION_ERROR].max()) < 1e-2
    assert ds[POSITION_3D].attrs["calibration"] == "rig" and ds[POSITION_3D].attrs["world_frame"] == 0
    # The 2D points now sit beside x, y, z and must still be what the video overlays.
    assert [c.name for c in overlay_candidates(ds, None, "left", (640, 480))] == ["position"]


def test_first_run_fits_the_world_frame_and_keeps_it_for_the_rig(tmp_path):
    calibration = _project_with_rig(tmp_path)
    frame = WorldFrame(
        axes=(("x", "corner_a", "corner_b"), ("y", "corner_a", "corner_c")),
        reference_point="corner_a",
        scale=("corner_a", "corner_c", 0.5),
        unit="m",
    )
    save_frame(calibration, frame)
    dt = eto.from_datasets([_camera_dataset(_tracks()).assign_attrs(trial=1)])
    triangulate_tree(dt, None, calibration)

    position = dt.trial(1)[POSITION_3D]
    assert position.attrs["world_frame"] == 1 and position.attrs["space_unit"] == "m"
    corner_c = position.sel(keypoint="corner_c", individual="bird").median("time").values
    np.testing.assert_allclose(corner_c, [0.0, 0.5, 0.0], atol=1e-3)
    assert load_frame(calibration).transform is not None


class _TwoRateAlignment:
    """Two cameras with one pose file each; the right one runs at twice the rate and starts 0.2 s early."""

    cameras = ["left", "right"]
    pose_keys = ["left", "right"]
    _rate = {"left": 10.0, "right": 20.0}
    _offset = {"left": 0.0, "right": -0.2}

    def __init__(self, folder: Path):
        self._folder = folder

    def get_stream_rate(self, stream, device=None):
        return self._rate[device]

    def stream_offset_for_trial(self, trial, stream, device=None):
        return self._offset[device]

    def resolve_media_path(self, trial, stream, device=None, fallback_folder=None):
        return str(self._folder / f"{device}.nc")

    def start_time(self, trial):
        return 0.0

    def stop_time(self, trial):
        return None


def test_pose_files_are_paired_on_the_trial_clock_not_by_frame_index(tmp_path):
    def beak_at(t: np.ndarray) -> np.ndarray:
        return np.stack([np.sin(t), np.cos(t), 20.0 + t], axis=-1)[:, None, None, :]

    for c, cam in enumerate(_TwoRateAlignment.cameras):
        rate, offset = _TwoRateAlignment._rate[cam], _TwoRateAlignment._offset[cam]
        video_time = np.arange(40) / rate
        pixels = reproject_points(_rig(), beak_at(video_time + offset))[c]
        xr.Dataset(
            {"position": (("time", "keypoint", "individual", "space"), pixels)},
            coords={"time": video_time, "keypoint": ["beak"], "individual": ["bird"], "space": ["x", "y"]},
        ).to_netcdf(tmp_path / f"{cam}.nc")

    dt = TrialTree.from_datasets([xr.Dataset(attrs={"trial": 1})], validate=False)
    triangulate_tree(dt, _TwoRateAlignment(tmp_path), _project_with_rig(tmp_path))

    position = dt.trial(1)[POSITION_3D].transpose("time", "keypoint", "individual", "space")
    time = position["time"].values
    seen = time <= 1.7  # the right camera's file ends at 1.75 s on the trial clock
    np.testing.assert_allclose(position.values[seen], beak_at(time[seen]), atol=1e-3)
    assert np.isnan(position.values[time > 1.8]).all()


def test_exported_geometry_is_one_the_space_plot_reads(tmp_path):
    dt = eto.from_datasets([_camera_dataset(_tracks()).assign_attrs(trial=1)])
    triangulate_tree(dt, None, _project_with_rig(tmp_path))
    points = landmark_positions([dt.trial(1)], _KEYPOINTS[:3])
    path = write_geometry(tmp_path / "space" / "arena.yaml", points, [("corner_a", "corner_b"), ("corner_a", "beak")])

    (geometry,) = parse_geometry(yaml.safe_load(path.read_text(encoding="utf-8")))
    np.testing.assert_allclose(geometry.vertices[1], [1.5, 0.0, 20.5], atol=1e-3)
    assert geometry.edges == [(0, 1)]


def test_missing_aniposelib_names_the_extra(monkeypatch, tmp_path: Path) -> None:
    """Without aniposelib, loading a calibration fails as a CalibrationError that says how to install it."""
    from ethograph.triangulate.calibration import INSTALL_HINT, TriangulationUnavailableError, load_calibration

    monkeypatch.setitem(sys.modules, "aniposelib.cameras", None)
    with pytest.raises(TriangulationUnavailableError, match=re.escape(INSTALL_HINT)):
        load_calibration(tmp_path / "rig.toml")
