"""Validation utilities for TrialTree datasets."""

from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ethograph.io import schema

if TYPE_CHECKING:
    from ethograph.io.trialtree import TrialTree


# ---------------------------------------------------------------------------
# Supported file extensions (single source of truth)
# ---------------------------------------------------------------------------

# Not all tested
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

#: Standalone audio files the waveform/spectrogram/playback loader
#: (``audioio.AudioLoader`` → libsndfile) can decode directly.  Video
#: containers (``.mp4``/``.mov``/``.avi``) are deliberately excluded here
#: because libsndfile reads no container format: an embedded AAC/etc. track is
#: decoded once to a cached ``.wav`` by
#: :func:`ethograph.io.audio_extract.resolve_audio_path`, which every audio
#: reader goes through.  See docs/source/advanced/troubleshooting.md
#: ("Audio formats").
AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".ogg",
    ".mp3",
}

POSE_EXTENSIONS = {".h5", ".hdf5", ".csv", ".slp", ".nwb"}

#: ``ds_type`` values of a ``movement`` dataset saved as NetCDF — a pose file
#: like a DLC ``.h5``, except that it needs no conversion.
MOVEMENT_DS_TYPES = frozenset({"poses", "bboxes"})


def movement_dataset_info(path: str | Path) -> tuple[str, float | None] | None:
    """``(ds_type, fps)`` of a ``.nc`` holding a ``movement`` poses/bboxes
    dataset, else ``None`` (any other ``.nc``, or one that cannot be opened).

    Attrs only: the file is opened and closed without reading data. ``fps``
    is the dataset's own, ``None`` when its time axis is in frames.
    """
    p = Path(path)
    if p.suffix.lower() != ".nc" or not p.is_file():
        return None
    try:
        with xr.open_dataset(p) as ds:
            if ds.attrs.get("ds_type") not in MOVEMENT_DS_TYPES or "position" not in ds.data_vars:
                return None
            fps = ds.attrs.get("fps")
            return str(ds.attrs["ds_type"]), (float(fps) if fps else None)
    except (OSError, ValueError):
        return None


# Still images shown as static media views (arena photo, reference frame).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

EPHYS_EXTENSIONS_RAW = {".dat", ".bin", ".raw"}

EPHYS_EXTENSIONS = {
    ".abf",
    ".axgd",
    ".axgx",
    ".bdf",
    ".ccf",
    ".continuous",
    ".edr",
    ".edf",
    ".events",
    ".medd",
    ".meta",
    ".ncs",
    ".nev",
    ".nrd",
    ".nse",
    ".ns1",
    ".ns2",
    ".ns3",
    ".ns4",
    ".ns5",
    ".ns6",
    ".ntt",
    ".nvt",
    ".nwb",
    ".oebin",
    ".openephys",
    ".pl2",
    ".plx",
    ".rdat",
    ".rec",
    ".rhd",
    ".rhs",
    ".ridx",
    ".sev",
    ".sif",
    ".smr",
    ".smrx",
    ".spikes",
    ".tbk",
    ".tdx",
    ".tev",
    ".tin",
    ".tnt",
    ".trc",
    ".tsq",
    ".vhdr",
    ".wcp",
    ".xdat",
}


def _fmt_extensions(exts: set[str]) -> str:
    return ", ".join(sorted(exts))


EPHYS_EXTENSIONS_STR = _fmt_extensions(EPHYS_EXTENSIONS)


def _qt_filter(label: str, exts: set[str]) -> str:
    globs = " ".join(f"*{e}" for e in sorted(exts))
    return f"{label} ({globs});;All files (*)"


VIDEO_FILE_FILTER = _qt_filter("Video files", VIDEO_EXTENSIONS)
AUDIO_FILE_FILTER = _qt_filter("Audio files", AUDIO_EXTENSIONS)
EPHYS_FILE_FILTER = _qt_filter("Ephys files", EPHYS_EXTENSIONS)
IMAGE_FILE_FILTER = _qt_filter("Image files", IMAGE_EXTENSIONS)


def find_temporal_dims(ds: xr.Dataset) -> set[str]:
    """Identify non-time dimensions that co-occur with a time dimension.

    Returns the set of dimension names that appear alongside at least one
    time-like dimension (any dim whose name contains ``"time"``) in the
    same data variable.  Used to discover extra selection dimensions such
    as ``space``, ``keypoint``, or ``individual``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to inspect.

    Returns
    -------
    set[str]
        Dimension names (excluding time dims themselves).
    """
    temporal = set()
    time_dims = set()

    for var in ds.data_vars.values():
        var_time_dims = {d for d in var.dims if "time" in d}
        if var_time_dims:
            time_dims.update(var_time_dims)
            temporal.update(var.dims)

    temporal -= time_dims
    return temporal


def is_integer_array(arr: np.ndarray) -> bool:
    """Check if array contains only integer values (no fractional part)."""
    if np.issubdtype(arr.dtype, np.floating):
        return np.all(np.mod(arr, 1) == 0)
    return np.issubdtype(arr.dtype, np.integer)


def validate_required_attrs(
    ds: xr.Dataset,
) -> list[str]:
    """Validate required dataset attributes.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate.

    Returns
    -------
    list[str]
        Validation error messages (empty if valid).
    """
    errors = []

    if "fps" in ds.attrs:
        if not isinstance(ds.attrs["fps"], Number) or ds.attrs["fps"] <= 0:
            errors.append("'fps' must be a positive number")

    if "trial" not in ds.attrs:
        errors.append("Xarray dataset ('ds') must have 'trial' attribute")

    return errors


def validate_changepoints(ds: xr.Dataset) -> list[str]:
    """Validate changepoint variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing changepoint variables.

    Returns
    -------
    list[str]
        Validation error messages (empty if valid).
    """
    errors = []
    cp_ds = schema.filter_changepoints(ds)

    for var_name, var in cp_ds.data_vars.items():
        arr = var.values

        if not is_integer_array(arr):
            errors.append(f"Changepoint '{var_name}' must contain only integer values")

        if arr.min() < 0 or arr.max() > 1:
            errors.append(f"Changepoint '{var_name}' must have values in range [0, 1]")

        target = var.attrs.get("target_feature")
        if target and target not in ds.data_vars:
            errors.append(f"Changepoint '{var_name}' references non-existent target_feature '{target}'")

    return errors


def validate_dataset(
    ds: xr.Dataset,
    catalog,
) -> list[str]:
    """Validate dataset structure and data types.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to validate.
    catalog : DataCatalog
        Feature/dimension catalog (from ``catalog_from_xarray``).

    Returns
    -------
    list[str]
        Validation error messages (empty if valid).
    """
    errors = []

    errors.extend(validate_required_attrs(ds))

    _ind_coord = next((n for n in ("individuals", "individual") if n in ds.coords), None)
    if _ind_coord is None or len(ds.coords[_ind_coord]) == 0:
        errors.append("Xarray dataset ('ds') must have 'individuals' or 'individual' coordinate")

    for feat_name in catalog.features:
        try:
            feat_var = ds[feat_name]
        except KeyError:
            print(
                f"Warning: Feature '{feat_name}' listed in catalog but not found in dataset. Skipping validation for this feature."  # noqa: E501
            )
            continue

        has_time_coord = any("time" in str(dim).lower() for dim in feat_var.dims)
        if not has_time_coord:
            errors.append(
                f"Feature variable '{feat_name}' must have a coordinate containing 'time'. E.g. 'time', 'time_labels', 'time_aux', etc."  # noqa: E501
            )
        if not isinstance(feat_var.values, np.ndarray):
            errors.append(f"Feature '{feat_name}' must be an array")

    if catalog.changepoints:
        errors.extend(validate_changepoints(ds))

    return errors


def _extract_trial_datasets(dt: "TrialTree") -> list[xr.Dataset]:
    """Extract all trial datasets from a TrialTree."""
    return [ds for _, ds in dt.trial_items()]


def _possible_trial_conditions(ds: xr.Dataset, dt: "TrialTree") -> list[str]:
    """Identify possible trial condition attributes."""
    common_extensions = (
        VIDEO_EXTENSIONS
        | AUDIO_EXTENSIONS
        | POSE_EXTENSIONS
        | EPHYS_EXTENSIONS
        | EPHYS_EXTENSIONS_RAW
        | {".csv", ".h5", ".hdf5", ".npy"}
    )

    common_attrs = dt.get_common_attrs().keys()

    cond_attrs = []
    for key, value in ds.attrs.items():
        if key in ["trial"] or key in common_attrs:
            continue

        if isinstance(value, str):
            if Path(value).suffix.lower() in common_extensions:
                continue

        cond_attrs.append(key)

    return cond_attrs


def validate_datatree(
    dt: "TrialTree",
) -> list[str]:
    """Validate a TrialTree for consistency and data integrity.

    Performs two levels of validation:
    1. Cross-trial consistency: Ensures all trials have the same structure
       (coords, data_vars, attrs keys and optionally values)
    2. Single-dataset validation: Validates data content on first trial
       (array types, RGBA format, changepoints, etc.)

    Parameters
    ----------
    dt : TrialTree
        Tree to validate.
        When False, missing ``fps`` is not an error.

    Returns
    -------
    list[str]
        Validation error messages (empty if valid).
    """
    from ethograph.io.catalog import catalog_from_xarray

    ds = dt.itrial(0)
    catalog = catalog_from_xarray(ds, dt)
    datasets = _extract_trial_datasets(dt)

    if not datasets:
        return ["No trial datasets found in TrialTree"]

    errors = []

    sample_size = min(5, len(datasets))
    sample_indices = np.random.choice(len(datasets), size=sample_size, replace=False)
    for idx in sample_indices:
        errors.extend(
            validate_dataset(
                datasets[idx],
                catalog,
            )
        )

    return list(set(errors))
