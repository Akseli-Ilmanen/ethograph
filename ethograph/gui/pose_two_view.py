"""Two cameras' keypoint stores as one pose: pairing frames, and the 2D + 3D datasets (Qt-free).

Each camera keeps its own :class:`~ethograph.gui.pose_annotate.KeypointStore`
and its own sidecar — the label model has no camera axis. What makes two
stores a pair is here: the same moment on the trial clock, the same keypoint
and individual names, and a calibration that turns the two pixel positions
into one 3D point (:mod:`ethograph.triangulate`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from aniposelib.cameras import CameraGroup

from ethograph.gui.pose_annotate import KeypointStore, store_to_movement_ds
from ethograph.triangulate.frame import WorldTransform
from ethograph.triangulate.points import triangulate_points
from ethograph.triangulate.session import CALIBRATION_UNITS, REPROJECTION_ERROR, SPACE_3D

#: Every camera's 2D keypoints of one clip, beside the first camera's video.
VIEWS_DATASET_SUFFIX = ".keypoints_views.nc"
#: The clip's triangulated keypoints, beside the first camera's video.
POSES_3D_SUFFIX = ".keypoints_3d.nc"


class ViewMismatchError(ValueError):
    """Two views whose stores do not label the same keypoints of the same individuals."""


@dataclass
class ViewSource:
    """One camera's labels, with what places them on the trial clock and in source pixels."""

    camera: str
    store: KeypointStore
    fps: float
    #: Trial time of the view's first frame.
    time_offset: float
    #: Store pixels per source pixel: 1 on the source video, <1 on a proxy.
    scale: float = 1.0


def views_dataset_path(video_path: str | Path) -> Path:
    video = Path(video_path)
    return video.with_name(video.name + VIEWS_DATASET_SUFFIX)


def poses_3d_path(video_path: str | Path) -> Path:
    video = Path(video_path)
    return video.with_name(video.name + POSES_3D_SUFFIX)


def paired_frames(frames: np.ndarray, view: ViewSource, other: ViewSource) -> np.ndarray:
    """*other*'s frame at the moment of each of *view*'s *frames*; ``-1`` where *other* has none."""
    trial_time = np.asarray(frames, dtype=np.float64) / view.fps + view.time_offset
    paired = np.rint((trial_time - other.time_offset) * other.fps).astype(int)
    paired[(paired < 0) | (paired >= other.store.n_frames)] = -1
    return paired


def _check_same_schema(views: Sequence[ViewSource]) -> None:
    first = views[0].store
    for view in views[1:]:
        store = view.store
        if store.keypoint_names != first.keypoint_names or store.individual_names != first.individual_names:
            raise ViewMismatchError(
                f"{view.camera} labels {store.keypoint_names} of {store.individual_names}; "
                f"{views[0].camera} labels {first.keypoint_names} of {first.individual_names}."
            )


def frame_points(views: Sequence[ViewSource], frame: int) -> np.ndarray:
    """Every view's points at the first view's *frame*: ``(cameras, individual, keypoint, 2)`` source pixels."""
    _check_same_schema(views)
    out = []
    for view in views:
        own = frame if view is views[0] else int(paired_frames(np.array([frame]), views[0], view)[0])
        if own < 0:
            out.append(np.full((view.store.n_individuals, view.store.n_keypoints, 2), np.nan))
        else:
            out.append(view.store.positions(own) / view.scale)
    return np.stack(out)


def _dense(view: ViewSource) -> tuple[np.ndarray, np.ndarray]:
    """``(time, keypoint, individual, 2)`` source pixels and ``(time, keypoint, individual)`` confidence."""
    ds = store_to_movement_ds(view.store, view.fps)
    points = ds["position"].transpose("time", "keypoint", "individual", "space").values / view.scale
    return points, ds["confidence"].transpose("time", "keypoint", "individual").values


def pair_datasets(
    views: Sequence[ViewSource],
    cgroup: CameraGroup | None = None,
    transform: WorldTransform | None = None,
    attrs: dict[str, object] | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None, np.ndarray | None]:
    """The clip's 2D dataset (one ``camera`` dim), its 3D dataset, and the untransformed 3D points.

    Both are movement-shaped and on the first view's clock; the other views
    are sampled at the same moments, never at the same frame index. Without
    *cgroup* (its cameras in *views*' order) only the 2D dataset is built.
    The 3D confidence of a point is the lowest any camera gave it.
    """
    _check_same_schema(views)
    first = views[0]
    frames = np.arange(first.store.n_frames)
    points = np.full((len(views), len(frames), first.store.n_keypoints, first.store.n_individuals, 2), np.nan)
    confidence = np.full(points.shape[:-1], np.nan)
    for c, view in enumerate(views):
        own = frames if view is first else paired_frames(frames, first, view)
        seen = own >= 0
        view_points, view_confidence = _dense(view)
        points[c, seen] = view_points[own[seen]]
        confidence[c, seen] = view_confidence[own[seen]]

    coords = {
        "time": frames / first.fps,
        "keypoint": list(first.store.keypoint_names),
        "individual": list(first.store.individual_names),
    }
    base_attrs = {"ds_type": "poses", "fps": float(first.fps), "source_software": "ethograph"}
    ds_2d = xr.Dataset(
        {
            "position": (("camera", "time", "keypoint", "individual", "space"), points, {"space_unit": "pixels"}),
            "confidence": (("camera", "time", "keypoint", "individual"), confidence),
        },
        coords={**coords, "camera": [view.camera for view in views], "space": ["x", "y"]},
        attrs={**base_attrs, "space_unit": "pixels"},
    ).transpose("camera", "time", "space", "keypoint", "individual")
    if cgroup is None:
        return ds_2d, None, None

    raw, error = triangulate_points(cgroup, points)
    position = raw if transform is None else transform.apply(raw)
    lowest = np.where(np.isfinite(raw[..., 0]), np.nanmin(np.nan_to_num(confidence, nan=np.inf), axis=0), np.nan)
    ds_3d = xr.Dataset(
        {
            "position": (("time", "keypoint", "individual", "space"), position),
            "confidence": (("time", "keypoint", "individual"), lowest),
            REPROJECTION_ERROR: (("time", "keypoint", "individual"), error, {"units": "pixels"}),
        },
        coords={**coords, "space": SPACE_3D},
        attrs={
            **base_attrs,
            "space_unit": CALIBRATION_UNITS,
            "world_frame": int(transform is not None),
            **(attrs or {}),
        },
    ).transpose("time", "space", "keypoint", "individual")
    return ds_2d, ds_3d, raw


def stash_fill(stash: dict[str, tuple], video: str | None, store: KeypointStore) -> None:
    """Remember *store*'s fill under *video*: a fill is never written to a sidecar."""
    if video and store.filled is not None:
        stash[video] = (store.filled, store.confidence, store.fill_range)


def restore_fill(stash: dict[str, tuple], video: str | None, store: KeypointStore) -> None:
    """Give *store* back the fill stashed for *video*, when it still fits the schema."""
    if not video or video not in stash or store.filled is not None:
        return
    filled, confidence, fill_range = stash[video]
    if filled.shape == (store.n_frames, store.n_individuals, store.n_keypoints, 2):
        store.filled, store.confidence, store.fill_range = filled, confidence, fill_range
