"""Pynapple / NWB loading and feature extraction utilities."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d

from ethograph.features.movement import get_angle_rgb
from ethograph.io import schema

## IO

NWB_TYPE_MAP = {
    "TsGroup": nap.TsGroup,
    "TsdFrame": nap.TsdFrame,
    "Tsd": nap.Tsd,
    "Ts": nap.Ts,
    "IntervalSet": nap.IntervalSet,
}


def parse_nwb_types(nwb: nap.NWBFile) -> dict[str, type]:
    result = {}
    for line in str(nwb).split("\n"):
        line = line.strip("│┝┕┍ ━┯┑┿┥┷┙")
        if not line or "Keys" in line or "━" in line:
            continue
        parts = [p.strip() for p in line.split("│") if p.strip()]
        if len(parts) == 2:
            result[parts[0]] = NWB_TYPE_MAP.get(parts[1], parts[1])
    return result


def get_metadata(obj) -> dict:
    meta = {"type": type(obj).__name__}
    if isinstance(obj, nap.TsGroup):
        meta["n_units"] = len(obj)
        meta["metadata_columns"] = list(obj.metadata_columns)
    elif isinstance(obj, nap.TsdFrame):
        meta["columns"] = list(obj.columns)
        meta["shape"] = obj.shape
    elif isinstance(obj, (nap.Tsd, nap.Ts)):
        meta["shape"] = obj.shape
    elif isinstance(obj, nap.IntervalSet):
        meta["n_intervals"] = len(obj)
    return meta


def flatten_nap_folder(data, load_metadata: bool = False) -> dict[str, dict]:
    flat = {}
    for key, val in data.items():
        if isinstance(val, nap.NWBFile):
            type_map = parse_nwb_types(val)
            for nwb_key, nwb_type in type_map.items():
                entry = {"type": nwb_type}
                if load_metadata:
                    try:
                        entry.update(get_metadata(val[nwb_key]))
                    except Exception:
                        entry["error"] = "failed to load"
                flat[nwb_key] = entry
        elif isinstance(val, dict):
            flat.update(flatten_nap_folder(val, load_metadata))
        else:
            entry = {"type": type(val)}
            if load_metadata:
                entry.update(get_metadata(val))
            flat[key] = entry
    return flat


### Features


def _iter_units(
    data: nap.Tsd | nap.TsdFrame | nap.TsGroup,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    match data:
        case nap.Tsd():
            return {"0": (data.t, data.values)}
        case nap.TsdFrame():
            return {col: (data.t, data[col].values) for col in data.columns}
        case nap.TsGroup():
            return {uid: (data[uid].t, data[uid].values) for uid in data.index}
        case _:
            raise TypeError(f"Unsupported type: {type(data)}")


def _apply_changepoint_func(
    t: np.ndarray,
    x: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    result = np.asarray(f(x))
    if len(result) == len(x):
        return t[result.astype(bool)]
    return t[result.astype(int)]


def add_changepoints_to_nap(
    data: nap.Tsd | nap.TsdFrame | nap.TsGroup,
    target_feature: str,
    changepoint_func: Callable[[np.ndarray], np.ndarray],
    **func_kwargs,
) -> nap.TsGroup:
    """Detect changepoints in a pynapple time series and return them as a TsGroup.

    Applies *changepoint_func* independently to each unit or column in *data*.
    Returns a :class:`nap.TsGroup` where each unit contains the changepoint
    timestamps for one source series.  Source metadata (label, feature name,
    type) is stored as group metadata columns; if *data* is itself a
    ``TsGroup``, its metadata columns are forwarded as well.

    Parameters
    ----------
    data : nap.Tsd | nap.TsdFrame | nap.TsGroup
        Input time series.  For a ``Tsd`` a single unit is produced; for a
        ``TsdFrame`` one unit per column; for a ``TsGroup`` one unit per
        neuron/unit.
    target_feature : str
        Human-readable label recorded in the output metadata
        (e.g. ``"speed"``).
    changepoint_func : callable
        A function ``f(x, **kwargs) -> array`` that accepts a 1-D numpy
        array of values and returns an array of changepoint times (or a
        binary indicator of the same length).
    **func_kwargs
        Forwarded to *changepoint_func*.

    Returns
    -------
    nap.TsGroup
        One unit per input series, containing changepoint timestamps.

    Examples
    --------
    >>> import ethograph as eto
    >>> from ethograph.features.changepoints import find_troughs_binary
    >>> data = eto.load_nap_data("experiment.nwb")
    >>> cp_group = eto.add_changepoints_to_nap(
    ...     data["speed"],
    ...     target_feature="speed",
    ...     changepoint_func=find_troughs_binary,
    ...     prominence=0.3,
    ... )
    """
    f = partial(changepoint_func, **func_kwargs)
    units = _iter_units(data)

    ts_dict = {}
    for i, (_, (t, x)) in enumerate(units.items()):
        cp_times = _apply_changepoint_func(t, x, f)
        ts_dict[i] = nap.Ts(t=cp_times, time_support=data.time_support)

    group = nap.TsGroup(ts_dict, time_support=data.time_support)
    group.set_info(
        source_label=list(units.keys()),
        **schema.changepoint_metadata(len(units), target_feature=target_feature),
    )

    if isinstance(data, nap.TsGroup) and data.metadata is not None:
        for col in data.metadata.columns:
            group.set_info(**{col: data.metadata[col].values})

    return group


def add_angle_rgb_to_nap(
    tsdframe: nap.TsdFrame,
    smoothing_params: dict,
    position_key: str = "position",
    xy_columns: list[str] = ["x", "y"],
) -> nap.TsdFrame:
    """Compute heading-angle RGB colour coding from 2-D position data.

    Calculates the heading angle from consecutive (x, y) positions in
    *tsdframe* and maps each angle to an RGB triplet via
    :func:`~ethograph.features.movement.get_angle_rgb`.  Gaussian smoothing
    is applied before angle computation.

    Parameters
    ----------
    tsdframe : nap.TsdFrame
        Position data with at least two columns (x and y).  If more than two
        columns are present, *xy_columns* selects which two to use.
    smoothing_params : dict
        Keyword arguments forwarded to
        :func:`scipy.ndimage.gaussian_filter1d` (e.g. ``{"sigma": 3}``).
    position_key : str, optional
        Input type passed to ``get_angle_rgb`` (default ``"position"``).
    xy_columns : list[str], optional
        Column names to use as x and y when *tsdframe* has more than two
        columns (default ``["x", "y"]``).

    Returns
    -------
    nap.TsdFrame
        A new ``TsdFrame`` with three columns ``["R", "G", "B"]`` on the
        same time support as *tsdframe*.

    Examples
    --------
    >>> import ethograph as eto
    >>> data = eto.load_nap_data("experiment.nwb")
    >>> rgb = eto.add_angle_rgb_to_nap(
    ...     data["position"],
    ...     smoothing_params={"sigma": 3},
    ... )
    >>> rgb.columns
    ['R', 'G', 'B']
    """
    if tsdframe.shape[1] > 2:
        tsdframe = tsdframe[xy_columns]

    rgb, _ = get_angle_rgb(
        tsdframe.values,
        smooth_func=gaussian_filter1d,
        smoothing_params=smoothing_params,
        input_type=position_key,
    )

    return nap.TsdFrame(
        t=tsdframe.t,
        d=rgb,
        columns=["R", "G", "B"],
        time_support=tsdframe.time_support,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

PYNAPPLE_EXTENSIONS = {".nwb", ".npz"}


def find_trials_intervalset(folder: str | Path) -> nap.IntervalSet | None:
    """The trials IntervalSet stored in a pynapple folder, or ``None``.

    Cheap scan: only ``.npz`` files whose arrays include ``start`` and ``end``
    are actually loaded. Used by the cover page to offer converting trial
    timing into ``.ethograph/alignment.nwb`` — the loader itself never reads
    trial timing from the data.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None
    for npz in sorted(folder.glob("*.npz")):
        try:
            with np.load(npz, allow_pickle=True) as raw:
                if not {"start", "end"}.issubset(raw.keys()):
                    continue
        except (OSError, ValueError):
            continue
        obj = nap.load_file(str(npz))
        if isinstance(obj, nap.IntervalSet):
            return obj
    return None


def detect_trials(data: dict) -> nap.IntervalSet | None:
    """Find a trials IntervalSet in a loaded pynapple data dict."""
    for key in ("trials", "epochs", "intervals"):
        obj = data.get(key)
        if isinstance(obj, nap.IntervalSet):
            return obj
    for key, obj in data.items():
        if isinstance(obj, nap.IntervalSet) and "trial" in key.lower():
            return obj
    return None


def label_intervalsets(data: dict) -> dict[str, nap.IntervalSet]:
    """Every IntervalSet in *data* except the one :func:`detect_trials` reads as trials."""
    trials = detect_trials(data)
    return {key: obj for key, obj in data.items() if isinstance(obj, nap.IntervalSet) and obj is not trials}


def _nwb_to_dict(nwb: nap.NWBFile) -> dict:
    """Extract all objects from an NWBFile into a flat dict."""
    data = {}
    for key in parse_nwb_types(nwb):
        try:
            data[key] = nwb[key]
        except Exception:
            pass
    return data


def _load_folder_as_dict(folder_path: Path) -> dict:
    """Load all .npz and .nwb files in a folder into a flat dict."""
    data = {}
    for key, val in nap.load_folder(str(folder_path)).items():
        if isinstance(val, nap.NWBFile):
            data.update(_nwb_to_dict(val))
        else:
            data[key] = val
    return data


def load_nap_data(path: str) -> tuple[dict, nap.IntervalSet | None]:
    """Load pynapple data from a file or folder.

    Supports ``.nwb``, ``.npz``, or a directory (pynapple folder).
    When loading a single ``.npz``, sibling files in the same directory
    are also loaded (e.g. ``trials.npz`` alongside ``speed.npz``).

    Returns
    -------
    data : dict
        Loaded pynapple objects keyed by name.
    trials_ep : IntervalSet or None
        Trial intervals if found in the data.
    """
    p = Path(path)
    if p.is_dir():
        data = _load_folder_as_dict(p)
    elif p.suffix == ".nwb":
        data = _nwb_to_dict(nap.load_file(str(p)))
    elif p.suffix == ".npz":
        data = _load_folder_as_dict(p.parent)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    trials_ep = detect_trials(data)
    return data, trials_ep
