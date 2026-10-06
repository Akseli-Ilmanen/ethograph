"""Triangulating a session: every camera's 2D points → ``position_3d`` in the session's dataset.

The 2D points are read the way the video overlay reads them: a dataset
variable with a ``camera`` dim, else one pose file per camera through the
alignment. Either way they reach aniposelib as one array per trial.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from aniposelib.cameras import CameraGroup
from movement.io import load_dataset

from ethograph.io import schema
from ethograph.io.catalog import INDIVIDUAL_DIMS, KEYPOINT_DIMS
from ethograph.io.data_loader import load_features_dataset
from ethograph.io.netcdf import netcdf_engine
from ethograph.io.overlay_source import DEFAULT_OVERLAY, frames_for_times, overlay_dataset
from ethograph.io.trialtree import TrialTree
from ethograph.triangulate.calibration import load_calibration, resolve_calibration
from ethograph.triangulate.frame import WorldFrame, WorldTransform, load_frame, save_frame
from ethograph.triangulate.points import Method, triangulate_points
from ethograph.utils.paths import global_setting

logger = logging.getLogger(__name__)

POSITION_3D = "position_3d"
REPROJECTION_ERROR = "reprojection_error"
SPACE_3D = ["x", "y", "z"]
#: ``space_unit`` of a triangulation no world frame has scaled.
CALIBRATION_UNITS = "calibration units"

_CAMERA_DIM = "camera"


class NoViewsError(ValueError):
    """A trial whose cameras do not offer the 2D points to triangulate."""


@dataclass
class TrialViews:
    """One trial's 2D points from every camera, on one clock."""

    time: np.ndarray
    time_dim: str
    points: np.ndarray  # (cameras, time, keypoint, individual, 2)
    scores: np.ndarray | None  # (cameras, time, keypoint, individual)
    keypoints: list[str]
    individuals: list[str]


def _named(ds: xr.Dataset | xr.DataArray, spellings: tuple[str, ...]) -> str | None:
    return next((d for d in spellings if d in ds.dims), None)


def _as_arrays(pose: xr.Dataset) -> tuple[np.ndarray, np.ndarray | None, list[str], list[str]]:
    """A movement-shaped *pose* as ``(time, keypoint, individual, 2)`` points and matching scores."""
    position = pose["position"]
    kp_dim, ind_dim = _named(position, KEYPOINT_DIMS), _named(position, INDIVIDUAL_DIMS)
    if kp_dim is None:
        kp_dim = "keypoint"
        pose = pose.expand_dims({kp_dim: ["point"]})
    if ind_dim is None:
        ind_dim = "individual"
        pose = pose.expand_dims({ind_dim: ["individual_0"]})
    points = pose["position"].sel(space=["x", "y"]).transpose("time", kp_dim, ind_dim, "space").values
    scores = pose["confidence"].transpose("time", kp_dim, ind_dim).values if "confidence" in pose else None
    return (
        np.asarray(points, dtype=np.float64),
        scores,
        [str(v) for v in pose[kp_dim].values],
        [str(v) for v in pose[ind_dim].values],
    )


def _views_from_dataset(ds: xr.Dataset, feature: str, cameras: list[str]) -> TrialViews:
    da = ds[feature]
    known = [str(v) for v in da.coords[_CAMERA_DIM].values]
    missing = [c for c in cameras if c not in known]
    if missing:
        raise NoViewsError(f"{feature!r} has no camera {missing}; it holds {known}.")
    time_name = next(str(d) for d in da.dims if "time" in str(d))
    per_camera = [_as_arrays(overlay_dataset(ds, None, feature, cam)) for cam in cameras]
    _, _, keypoints, individuals = per_camera[0]
    all_scores = [s for _, s, _, _ in per_camera if s is not None]
    scores = np.stack(all_scores) if len(all_scores) == len(per_camera) else None
    return TrialViews(
        time=np.asarray(da[time_name].values, dtype=np.float64),
        time_dim=time_name,
        points=np.stack([p for p, _, _, _ in per_camera]),
        scores=scores,
        keypoints=keypoints,
        individuals=individuals,
    )


def _open_pose_file(path: str, source_software: str | None, fps: float) -> xr.Dataset:
    if Path(path).suffix.lower() == ".nc":
        with xr.open_dataset(path, engine=netcdf_engine(path)) as opened:
            return opened.load()
    if source_software is None:
        raise NoViewsError(f"{Path(path).name}: name the software that wrote the pose files (source_software).")
    return load_dataset(path, source_software, fps)


def _pose_device(alignment, camera: str, index: int) -> str | None:
    """The pose stream paired with *camera*: its namesake, else the stream at the camera's position."""
    devices = [str(d) for d in alignment.pose_keys]
    if camera in devices:
        return camera
    return devices[index] if index < len(devices) else None


def _views_from_files(
    ds: xr.Dataset,
    alignment,
    trial: int | str,
    cameras: list[str],
    pose_folder: Path | str | None,
    source_software: str | None,
) -> TrialViews:
    all_cameras = [str(c) for c in alignment.cameras]
    loaded = []
    for cam in cameras:
        if cam not in all_cameras:
            raise NoViewsError(f"The session has no camera {cam!r}; its alignment names {all_cameras}.")
        fps = alignment.get_stream_rate("video", cam)
        if fps is None:
            raise NoViewsError(f"Camera {cam!r} has no frame rate in the alignment.")
        path = alignment.resolve_media_path(
            trial,
            "pose",
            device=_pose_device(alignment, cam, all_cameras.index(cam)),
            fallback_folder=str(pose_folder) if pose_folder is not None else None,
        )
        if not path:
            raise NoViewsError(f"Trial {trial}: no pose file for camera {cam!r}.")
        offset = float(alignment.stream_offset_for_trial(trial, "video", device=cam))
        loaded.append((_as_arrays(_open_pose_file(path, source_software, fps)), float(fps), offset))

    (first_points, _, keypoints, individuals), fps0, offset0 = loaded[0]
    if "time" in ds.coords:
        time = np.asarray(ds["time"].values, dtype=np.float64)
    else:
        time = np.arange(len(first_points)) / fps0 + offset0
        stop = alignment.stop_time(trial)
        duration = np.inf if stop is None else stop - alignment.start_time(trial)
        time = time[(time >= 0.0) & (time < duration)]

    points = np.full((len(cameras), len(time), len(keypoints), len(individuals), 2), np.nan)
    scores = np.full(points.shape[:-1], np.nan)
    for c, ((cam_points, cam_scores, cam_keypoints, cam_individuals), fps, offset) in enumerate(loaded):
        if cam_keypoints != keypoints or cam_individuals != individuals:
            raise NoViewsError(
                f"Trial {trial}: camera {cameras[c]!r} tracks {cam_keypoints} of {cam_individuals}, "
                f"camera {cameras[0]!r} tracks {keypoints} of {individuals}."
            )
        frames = frames_for_times(time, fps, offset)
        inside = (frames >= 0) & (frames < len(cam_points))
        points[c, inside] = cam_points[frames[inside]]
        scores[c, inside] = 1.0 if cam_scores is None else cam_scores[frames[inside]]
    has_scores = any(cam_scores is not None for (_, cam_scores, _, _), _, _ in loaded)
    return TrialViews(time, "time", points, scores if has_scores else None, keypoints, individuals)


def gather_views(
    ds: xr.Dataset,
    alignment,
    trial: int | str,
    cameras: list[str],
    *,
    feature: str = DEFAULT_OVERLAY,
    pose_folder: Path | str | None = None,
    source_software: str | None = None,
) -> TrialViews:
    """Every camera's 2D points for *trial*: the dataset's own when *feature* has a camera dim, else pose files."""
    if feature in ds.data_vars and _CAMERA_DIM in ds[feature].dims:
        return _views_from_dataset(ds, feature, cameras)
    return _views_from_files(ds, alignment, trial, cameras, pose_folder, source_software)


def _triangulate_views(
    cgroup: CameraGroup,
    views: TrialViews,
    method: Method,
    min_confidence: float | None,
    constraints: Sequence[tuple[str, str]],
    constraints_weak: Sequence[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """``(time, keypoint, individual, 3)`` positions and ``(time, keypoint, individual)`` errors."""
    if min_confidence is not None and views.scores is None:
        raise NoViewsError("min_confidence was given but the 2D points carry no confidence.")
    index = {name: i for i, name in enumerate(views.keypoints)}
    pairs = [[(index[a], index[b]) for a, b in group] for group in (constraints, constraints_weak)]
    if method != "optim":
        return triangulate_points(
            cgroup, views.points, method=method, scores=views.scores, min_confidence=min_confidence
        )
    p3d = np.empty((*views.points.shape[1:-1], 3))
    error = np.empty(views.points.shape[1:-1])
    for i in range(len(views.individuals)):
        p3d[:, :, i], error[:, :, i] = triangulate_points(
            cgroup,
            views.points[:, :, :, i],
            method=method,
            scores=None if views.scores is None else views.scores[:, :, :, i],
            min_confidence=min_confidence,
            constraints=pairs[0],
            constraints_weak=pairs[1],
        )
    return p3d, error


def fit_world_frame(frame: WorldFrame, positions: Sequence[np.ndarray], keypoints: list[str]) -> WorldTransform:
    """Fit *frame* to its landmarks' median positions across *positions* ``(time, keypoint, individual, 3)``."""
    stacked = np.concatenate([p.transpose(1, 0, 2, 3).reshape(len(keypoints), -1, 3) for p in positions], axis=1)
    medians = {}
    for name in frame.landmarks:
        if name in keypoints:
            seen = stacked[keypoints.index(name)]
            seen = seen[np.isfinite(seen).all(axis=1)]
            if len(seen):
                medians[name] = np.median(seen, axis=0)
    return frame.fit(medians)


def _assign_3d(
    ds: xr.Dataset, views: TrialViews, p3d: np.ndarray, error: np.ndarray, attrs: dict[str, object]
) -> xr.Dataset:
    kp_dim = _named(ds, KEYPOINT_DIMS) or "keypoint"
    ind_dim = _named(ds, INDIVIDUAL_DIMS) or "individual"
    for dim, values in ((kp_dim, views.keypoints), (ind_dim, views.individuals)):
        if dim in ds.coords and [str(v) for v in ds[dim].values] != values:
            raise ValueError(f"The dataset's {dim!r} is {list(ds[dim].values)}, the triangulated points' is {values}.")
    ds = ds.drop_vars([POSITION_3D, REPROJECTION_ERROR], errors="ignore")
    if "space" in ds.coords and [str(v) for v in ds["space"].values] != SPACE_3D:
        if not set(map(str, ds["space"].values)) <= set(SPACE_3D):
            raise ValueError(f"The dataset's space is {list(ds['space'].values)}; cannot hold x, y, z beside it.")
        ds = ds.reindex(space=SPACE_3D)
    coords = {views.time_dim: views.time, kp_dim: views.keypoints, ind_dim: views.individuals}
    position = xr.DataArray(
        p3d, dims=[views.time_dim, kp_dim, ind_dim, "space"], coords={**coords, "space": SPACE_3D}, attrs=attrs
    ).transpose(views.time_dim, "space", kp_dim, ind_dim)
    schema.describe(position, schema.KINEMATIC_FEATURE, is_egocentric=False)
    reprojection = xr.DataArray(
        error, dims=[views.time_dim, kp_dim, ind_dim], coords=coords, attrs={"units": "pixels", schema.NORMALISE: 0}
    )
    return ds.assign({POSITION_3D: position, REPROJECTION_ERROR: reprojection})


def triangulate_tree(
    dt: TrialTree,
    alignment,
    calibration_path: Path | str,
    *,
    trials: Sequence[int | str] | None = None,
    feature: str = DEFAULT_OVERLAY,
    method: Method = "triangulate",
    min_confidence: float | None = None,
    constraints: Sequence[tuple[str, str]] = (),
    constraints_weak: Sequence[tuple[str, str]] = (),
    pose_folder: Path | str | None = None,
    source_software: str | None = None,
    progress: Callable[[int, int], bool] | None = None,
) -> list[int | str]:
    """Write ``position_3d`` and ``reprojection_error`` into *trials* of *dt*; returns the trials written.

    Every camera of the calibration must offer the 2D points. A trial whose
    cameras do not is skipped with a warning; a run that triangulates no trial
    at all raises the first reason. *progress* is called with ``(done,
    total)`` and stops the run by returning ``False``.
    """
    calibration_path = Path(calibration_path)
    cgroup = load_calibration(calibration_path)
    cameras = [str(name) for name in cgroup.get_names()]
    wanted = list(dt.trials if trials is None else trials)

    results: dict[int | str, tuple[TrialViews, np.ndarray, np.ndarray]] = {}
    first_error: NoViewsError | None = None
    for done, trial in enumerate(wanted):
        if progress is not None and not progress(done, len(wanted)):
            break
        try:
            views = gather_views(
                dt.trial(trial),
                alignment,
                trial,
                cameras,
                feature=feature,
                pose_folder=pose_folder,
                source_software=source_software,
            )
        except NoViewsError as err:
            logger.warning("Trial %s not triangulated: %s", trial, err)
            first_error = first_error or err
            continue
        results[trial] = (
            views,
            *_triangulate_views(cgroup, views, method, min_confidence, constraints, constraints_weak),
        )
    if not results:
        raise first_error or NoViewsError("No trial to triangulate.")

    frame = load_frame(calibration_path)
    transform = None
    if frame is not None:
        transform = frame.transform
        if transform is None:
            keypoints = next(iter(results.values()))[0].keypoints
            transform = fit_world_frame(frame, [p3d for _, p3d, _ in results.values()], keypoints)
            save_frame(
                calibration_path, WorldFrame(frame.axes, frame.reference_point, frame.scale, frame.unit, transform)
            )
    attrs: dict[str, object] = {
        "calibration": calibration_path.stem,
        "triangulation_method": method,
        "source_feature": feature,
        "world_frame": int(transform is not None),
        "space_unit": (frame.unit if frame is not None and frame.unit else CALIBRATION_UNITS),
    }
    if min_confidence is not None:
        attrs["min_confidence"] = float(min_confidence)

    for trial, (views, p3d, error) in results.items():
        if transform is not None:
            p3d = transform.apply(p3d)

        def assign(ds: xr.Dataset, v: TrialViews = views, p: np.ndarray = p3d, e: np.ndarray = error) -> xr.Dataset:
            return _assign_3d(ds, v, p, e, attrs)

        dt.update_trial(trial, assign)
    return list(results)


def triangulate(
    source: Path | str,
    project: Path | str,
    *,
    calibration: str | None = None,
    trials: Sequence[int | str] | None = None,
    feature: str = DEFAULT_OVERLAY,
    method: Method = "triangulate",
    min_confidence: float | None = None,
    constraints: Sequence[tuple[str, str]] = (),
    constraints_weak: Sequence[tuple[str, str]] = (),
    pose_folder: Path | str | None = None,
    source_software: str | None = None,
) -> Path:
    """Triangulate a session and save ``position_3d`` into its ``.nc``; returns that file.

    *source* is the session folder (or its ``.nc``), *project* the project
    folder holding ``calibration/``; *calibration* names one when the project
    has several. ``constraints`` / ``constraints_weak`` are keypoint-name
    pairs for ``method="optim"``. A re-run replaces the variables of the
    trials it covers and leaves the others alone.
    """
    calibration_path = resolve_calibration(project, calibration)
    result = load_features_dataset(str(source), ignore=tuple(global_setting("ignore_files", [])))
    if result.dt is None:
        raise ValueError(f"{source}: triangulation writes into an xarray (.nc) session.")
    done = triangulate_tree(
        result.dt,
        result.nwb_alignment,
        calibration_path,
        trials=trials,
        feature=feature,
        method=method,
        min_confidence=min_confidence,
        constraints=constraints,
        constraints_weak=constraints_weak,
        pose_folder=pose_folder,
        source_software=source_software,
    )
    result.dt.save()
    target = Path(result.dt._source_path)
    logger.info("Triangulated %d trials with %s -> %s", len(done), calibration_path.name, target)
    return target
