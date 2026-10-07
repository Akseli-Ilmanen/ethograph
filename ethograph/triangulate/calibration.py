"""The project's camera calibrations: ``{project}/calibration/{name}.toml``.

Each file is aniposelib's own ``calibration.toml``, untouched, so it stays
usable in Anipose itself. A project holding exactly one needs no choosing.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import yaml

if TYPE_CHECKING:
    from aniposelib.cameras import Camera, CameraGroup

#: The folder's name inside a project folder.
CALIBRATION_DIRNAME = "calibration"

#: How to get aniposelib: it is the ``triangulate`` extra, not part of ``gui``.
INSTALL_HINT = 'uv pip install "ethograph[triangulate]"'


class CalibrationError(ValueError):
    """A calibration that is missing, ambiguous, or does not fit the cameras asked for."""


class TriangulationUnavailableError(CalibrationError):
    """aniposelib is not installed, so nothing can be calibrated or triangulated."""


def aniposelib_cameras() -> tuple[type[Camera], type[CameraGroup]]:
    """``(Camera, CameraGroup)``: the one import of aniposelib, deferred because it is an optional extra."""
    try:
        from aniposelib.cameras import Camera, CameraGroup
    except ImportError as err:
        raise TriangulationUnavailableError(f"Triangulation needs aniposelib: {INSTALL_HINT}") from err
    return Camera, CameraGroup


def calibration_dir(project: Path | str) -> Path:
    return Path(project) / CALIBRATION_DIRNAME


def calibration_names(project: Path | str) -> list[str]:
    """The calibrations a project holds, by file stem."""
    folder = calibration_dir(project)
    return sorted(p.stem for p in folder.glob("*.toml")) if folder.is_dir() else []


def resolve_calibration(project: Path | str, name: str | None = None) -> Path:
    """The calibration file *name* refers to; the only one when *name* is ``None``."""
    names = calibration_names(project)
    if not names:
        raise CalibrationError(
            f"No calibration in {calibration_dir(project)}. Import one from a DeepLabCut 3D project "
            "(import_dlc_calibration) or write an aniposelib calibration.toml there."
        )
    if name is None:
        if len(names) > 1:
            raise CalibrationError(f"{calibration_dir(project)} holds several calibrations {names}; name one.")
        name = names[0]
    elif name not in names:
        raise CalibrationError(f"No calibration {name!r} in {calibration_dir(project)}; it holds {names}.")
    return calibration_dir(project) / f"{name}.toml"


def load_calibration(path: Path | str, cameras: list[str] | None = None) -> CameraGroup:
    """The camera group in *path*, restricted to and ordered as *cameras* when given."""
    _, CameraGroup = aniposelib_cameras()
    cgroup = CameraGroup.load(str(path))
    if cameras is None:
        return cgroup
    known = cgroup.get_names()
    missing = [c for c in cameras if c not in known]
    if missing:
        raise CalibrationError(f"{Path(path).name} has no camera {missing}; it calibrates {known}.")
    return cgroup.subset_cameras_names(cameras)


def import_dlc_calibration(dlc_folder: Path | str, project: Path | str, name: str | None = None) -> Path:
    """Convert a DeepLabCut 3D project's stereo calibration into ``{project}/calibration/{name}.toml``.

    The parameters are copied, not re-estimated, and the world frame is
    DeepLabCut's own (camera 1, rectified), so triangulating with the result
    reproduces the coordinates DeepLabCut 3D wrote. *name* defaults to the
    DeepLabCut folder's name.
    """
    dlc_folder = Path(dlc_folder)
    config_path = dlc_folder / "config.yaml"
    stereo_path = dlc_folder / "camera_matrix" / "stereo_params.pickle"
    for path in (config_path, stereo_path):
        if not path.is_file():
            raise CalibrationError(f"Not a calibrated DeepLabCut 3D project: {path} is missing.")

    with open(config_path, encoding="utf-8") as f:
        cam_names = [str(c) for c in yaml.safe_load(f)["camera_names"]]
    if len(cam_names) != 2:
        raise CalibrationError(f"DeepLabCut 3D calibrates a camera pair; {config_path} names {cam_names}.")

    with open(stereo_path, "rb") as f:
        stereo = pickle.load(f)
    pair = "-".join(cam_names)
    if pair not in stereo:
        raise CalibrationError(f"{stereo_path} has no pair {pair!r}; it holds {sorted(stereo)}.")
    params = stereo[pair]

    Camera, CameraGroup = aniposelib_cameras()
    # DeepLabCut triangulates in camera 1's rectified frame: X_world = R1 @ X_cam1.
    to_cam1 = np.asarray(params["R1"], dtype=np.float64).T
    to_cam2 = np.asarray(params["R"], dtype=np.float64) @ to_cam1
    sizes = params["image_shape"]
    cameras = [
        Camera(
            name=cam_names[0],
            size=list(sizes[0]),
            matrix=params["cameraMatrix1"],
            dist=params["distCoeffs1"],
            rvec=cv2.Rodrigues(to_cam1)[0].ravel(),
        ),
        Camera(
            name=cam_names[1],
            size=list(sizes[1]),
            matrix=params["cameraMatrix2"],
            dist=params["distCoeffs2"],
            rvec=cv2.Rodrigues(to_cam2)[0].ravel(),
            tvec=np.asarray(params["T"], dtype=np.float64).ravel(),
        ),
    ]
    cgroup = CameraGroup(cameras, metadata={"source": "deeplabcut_3d", "source_folder": str(dlc_folder)})

    out = calibration_dir(project) / f"{name or dlc_folder.name}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    cgroup.dump(str(out))
    return out
