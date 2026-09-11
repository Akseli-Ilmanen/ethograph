"""What the video panel can overlay: any feature in the camera's pixels (Qt-free).

The overlay is not "the pose". It is an instance bound to a catalog feature,
the way a line plot is bound to one. A feature qualifies when it has a time
dim and a ``space`` dim holding ``x`` and ``y``, its other dims are at most
``keypoint(s)`` / ``individual(s)`` (a ``camera`` dim is selected away first),
and its values fit the camera's frame. ``position`` is the default; a value
the console computed (``position_smooth = smooth(position)``) qualifies the
moment it is named.

Companions are found by name, movement's convention: ``confidence`` beside
``position`` drives the threshold filter and ``shape`` beside it means boxes;
for any other feature the companions are ``{name}_confidence`` and
``{name}_shape``. No ``ds_type`` switch — a ``shape`` variable is the marker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

DEFAULT_OVERLAY = "position"
#: ``space_unit`` attr spellings that say "pixels" outright.
PIXEL_UNITS = frozenset({"px", "pixel", "pixels"})
_TRACK_DIMS = ("keypoint", "keypoints", "individual", "individuals")
_CAMERA_DIM = "camera"


@dataclass(frozen=True)
class OverlayCandidate:
    name: str
    has_shape: bool
    has_confidence: bool


def companions(name: str) -> tuple[str, str]:
    """``(confidence, shape)`` variable names that belong to *name*."""
    if name == DEFAULT_OVERLAY:
        return "confidence", "shape"
    return f"{name}_confidence", f"{name}_shape"


def time_dim(da: xr.DataArray) -> str | None:
    return next((str(d) for d in da.dims if "time" in str(d)), None)


def _is_pixel_shaped(da: xr.DataArray) -> bool:
    """A time dim, a space dim with x and y, and nothing else but track dims."""
    if time_dim(da) is None or "space" not in da.dims or not np.issubdtype(da.dtype, np.number):
        return False
    space = [str(v) for v in da.coords["space"].values] if "space" in da.coords else []
    if "x" not in space or "y" not in space:
        return False
    extra = {str(d) for d in da.dims} - {time_dim(da), "space", _CAMERA_DIM, *_TRACK_DIMS}
    return not extra


def _is_companion(name: str, arrays: dict[str, xr.DataArray]) -> bool:
    """A ``shape`` (or ``{base}_shape``) beside its feature is not a feature of its own."""
    if name == "shape":
        return DEFAULT_OVERLAY in arrays
    return name.endswith("_shape") and name[: -len("_shape")] in arrays


def fits_frame(da: xr.DataArray, shape: xr.DataArray | None, width: float, height: float) -> bool:
    """Whether *da* plausibly lives in a *width* × *height* pixel frame.

    A ``space_unit`` attr settles it when present (pixels: yes, anything
    else: no). Otherwise every finite x must lie in ``[0, width]`` and every
    finite y in ``[0, height]``, box edges (``da ± shape/2``) included; an
    array with no finite value at all cannot be placed and is refused.
    """
    unit = str(da.attrs.get("space_unit", "")).strip().lower()
    if unit:
        return unit in PIXEL_UNITS
    if width <= 0 or height <= 0:
        return False
    lo, hi = da, da
    if shape is not None:
        half = shape / 2
        lo, hi = da - half, da + half
    xs = np.concatenate([lo.sel(space="x").values.ravel(), hi.sel(space="x").values.ravel()])
    ys = np.concatenate([lo.sel(space="y").values.ravel(), hi.sel(space="y").values.ravel()])
    xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
    if len(xs) == 0 or len(ys) == 0:
        return False
    return bool(xs.min() >= 0 and xs.max() <= width and ys.min() >= 0 and ys.max() <= height)


def _for_camera(da: xr.DataArray, camera: str | None) -> xr.DataArray | None:
    """*da* with the camera dim selected away; ``None`` when the camera is not in it."""
    if _CAMERA_DIM not in da.dims:
        return da
    if camera is None:
        return None
    cams = [str(v) for v in da.coords[_CAMERA_DIM].values]
    if camera not in cams:
        return None
    return da.sel({_CAMERA_DIM: camera}, drop=True)


def _derived_arrays(derived: dict[str, Any]) -> dict[str, xr.DataArray]:
    """Console snapshots with two columns become ``(time, space)`` arrays."""
    out: dict[str, xr.DataArray] = {}
    for name, feature in derived.items():
        values = getattr(feature, "values", None)
        time = getattr(feature, "time", None)
        if values is None or time is None or values.ndim != 2 or values.shape[1] != 2:
            continue
        labels = [str(label).lower() for label in (getattr(feature, "dim_labels", None) or ["x", "y"])]
        if sorted(labels) != ["x", "y"]:
            continue
        out[name] = xr.DataArray(
            np.asarray(values, dtype=float),
            dims=["time", "space"],
            coords={"time": np.asarray(time, dtype=float), "space": labels},
        )
    return out


def _arrays(ds: xr.Dataset | None, derived: dict[str, Any] | None) -> dict[str, xr.DataArray]:
    arrays: dict[str, xr.DataArray] = {}
    if ds is not None:
        arrays.update({str(k): v for k, v in ds.data_vars.items()})
    arrays.update(_derived_arrays(derived or {}))
    return arrays


def overlay_candidates(
    ds: xr.Dataset | None,
    derived: dict[str, Any] | None,
    camera: str | None,
    frame_size: tuple[float, float] | None,
) -> list[OverlayCandidate]:
    """Every feature the video panel can draw, ``position`` first when it qualifies.

    *frame_size* is the camera's ``(width, height)`` in source pixels; when
    unknown the pixel test is skipped and shape alone decides.
    """
    arrays = _arrays(ds, derived)
    found: list[OverlayCandidate] = []
    for name, da in arrays.items():
        if not _is_pixel_shaped(da) or _is_companion(name, arrays):
            continue
        conf_name, shape_name = companions(name)
        da_cam = _for_camera(da, camera)
        if da_cam is None:
            continue
        shape = arrays.get(shape_name)
        shape_cam = _for_camera(shape, camera) if shape is not None and _is_pixel_shaped(shape) else None
        if frame_size is not None and not fits_frame(da_cam, shape_cam, *frame_size):
            continue
        found.append(OverlayCandidate(name, has_shape=shape_cam is not None, has_confidence=conf_name in arrays))
    found.sort(key=lambda c: (c.name != DEFAULT_OVERLAY, c.name))
    return found


def overlay_dataset(
    ds: xr.Dataset | None,
    derived: dict[str, Any] | None,
    name: str,
    camera: str | None,
) -> xr.Dataset:
    """A movement-shaped dataset for *name*: ``position`` (+ ``confidence``, + ``shape``).

    The camera dim is selected away and an array without an individual dim
    gets one, so the points converter sees one shape whatever the source.
    Raises ``KeyError`` when *name* is not a feature.
    """
    arrays = _arrays(ds, derived)
    da = _for_camera(arrays[name], camera)
    if da is None:
        raise KeyError(f"{name!r} has no camera {camera!r}")
    conf_name, shape_name = companions(name)
    out = {"position": da}
    if conf_name in arrays:
        conf = _for_camera(arrays[conf_name], camera)
        if conf is not None:
            out["confidence"] = conf
    if shape_name in arrays and _is_pixel_shaped(arrays[shape_name]):
        shape = _for_camera(arrays[shape_name], camera)
        if shape is not None:
            out["shape"] = shape
    result = xr.Dataset(out)
    tdim = time_dim(da)
    if tdim != "time":
        result = result.rename({tdim: "time"})
    if not any(d in result.dims for d in ("individual", "individuals")):
        result = result.expand_dims({"individuals": ["ind_0"]})
    # The points converter indexes axes by position: (time, space, [keypoint], individual).
    ind = next(d for d in result.dims if d in ("individual", "individuals"))
    kp = next((d for d in result.dims if d in ("keypoint", "keypoints")), None)
    order = ["time", "space", *([kp] if kp else []), ind]
    return result.transpose(*order, missing_dims="ignore")


def frames_for_times(times: np.ndarray, fps: float, offset_s: float) -> np.ndarray:
    """Video frame index of each sample: ``round((t - offset) * fps)``.

    *offset_s* is the trial-relative time of the video's first frame
    (``stream_offset_for_trial``): 0 for a per-trial file, negative when the
    trial starts partway into a session-long video.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    return np.rint((np.asarray(times, dtype=float) - offset_s) * fps).astype(int)
