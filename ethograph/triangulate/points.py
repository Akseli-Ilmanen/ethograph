"""Arrays in, arrays out: the one place aniposelib's triangulation is called.

Batch triangulation, the labelling dialog's live 3D point and the geometry
export all hand their 2D points to :func:`triangulate_points`; nothing else
in Ethograph does multi-view geometry.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

if TYPE_CHECKING:
    from aniposelib.cameras import CameraGroup

Method = Literal["triangulate", "ransac", "optim"]
METHODS: tuple[str, ...] = get_args(Method)


def triangulate_points(
    cgroup: CameraGroup,
    points: np.ndarray,
    *,
    method: Method = "triangulate",
    scores: np.ndarray | None = None,
    min_confidence: float | None = None,
    constraints: Sequence[tuple[int, int]] = (),
    constraints_weak: Sequence[tuple[int, int]] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate *points* ``(cameras, ..., 2)`` → 3D ``(..., 3)`` and reprojection error ``(...)`` in pixels.

    A point with *scores* below *min_confidence* in a camera is not seen by
    that camera. ``optim`` needs *points* shaped ``(cameras, frames,
    keypoints, 2)``; its constraints are keypoint index pairs whose distance
    is rigid (``constraints``) or nearly so (``constraints_weak``).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown triangulation method {method!r}; choose one of {METHODS}.")
    n_cams = len(cgroup.cameras)
    if points.shape[0] != n_cams or points.shape[-1] != 2:
        raise ValueError(f"Expected points shaped ({n_cams}, ..., 2) for this calibration, got {points.shape}.")

    points = np.array(points, dtype=np.float64)
    if min_confidence is not None:
        if scores is None:
            raise ValueError("min_confidence needs the points' scores.")
        points[scores < min_confidence] = np.nan
    lead = points.shape[1:-1]
    flat = points.reshape(n_cams, -1, 2)

    if method == "optim":
        if points.ndim != 4:
            raise ValueError(f"method='optim' needs points shaped (cameras, frames, keypoints, 2), got {points.shape}.")
        p3d = cgroup.triangulate_optim(
            points,
            constraints=[list(pair) for pair in constraints],
            constraints_weak=[list(pair) for pair in constraints_weak],
        ).reshape(-1, 3)
    elif method == "ransac":
        p3d, _, flat, _ = cgroup.triangulate_ransac(flat)
    else:
        p3d = cgroup.triangulate(flat)

    error = np.full(flat.shape[1], np.nan)
    seen = np.isfinite(p3d[:, 0])
    if seen.any():
        error[seen] = cgroup.reprojection_error(p3d[seen], flat[:, seen], mean=True)
    return p3d.reshape(*lead, 3), error.reshape(lead)


def reproject_points(cgroup: CameraGroup, points_3d: np.ndarray) -> np.ndarray:
    """*points_3d* ``(..., 3)`` in every camera's pixels, ``(cameras, ..., 2)``."""
    lead = points_3d.shape[:-1]
    return cgroup.project(np.asarray(points_3d, dtype=np.float64)).reshape(len(cgroup.cameras), *lead, 2)
