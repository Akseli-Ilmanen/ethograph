from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks

from ethograph.io import schema
from ethograph.labels.intervals import (
    purge_short_intervals,
    snap_boundaries,
    stitch_intervals,
)
from ethograph.utils.xr_utils import get_time_coord

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nan_boundary_indices(arr: np.ndarray) -> np.ndarray:
    """Return indices where NaN transitions occur (NaN->valid and valid->NaN)."""
    arr = np.asarray(arr)
    is_valid = ~np.isnan(arr)
    transitions = np.diff(np.concatenate(([0], is_valid.astype(int), [0])))
    nan_to_val = np.where(transitions == 1)[0]
    val_to_nan = np.where(transitions == -1)[0] - 1
    return np.concatenate((nan_to_val, val_to_nan)).astype(int)


def _to_binary(indices: np.ndarray, length: int) -> np.ndarray:
    """Convert sparse index array to dense binary mask.

    If the only marked positions are the first and last sample, returns all zeros
    (boundary-only case treated as empty).
    """
    mask = np.zeros(length, dtype=np.int8)
    if len(indices) == 0:
        return mask
    valid = indices[(indices >= 0) & (indices < length)]
    mask[valid] = 1
    if np.sum(mask) == 2 and mask[0] == 1 and mask[-1] == 1:
        mask[:] = 0
    return mask


# ---------------------------------------------------------------------------
# NaN boundary handling (refactored to use helpers)
# ---------------------------------------------------------------------------


def add_NaN_boundaries(arr, changepoints):
    """Merge NaN-transition boundaries with other changepoints -> binary mask."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("add_NaN_boundaries only supports 1D (N,) input arrays.")
    nan_bounds = _nan_boundary_indices(arr)
    merged = np.unique(np.concatenate((np.asarray(changepoints, dtype=int), nan_bounds)))
    return _to_binary(merged, len(arr))


# ---------------------------------------------------------------------------
# Binary detection (dense binary masks for apply_ufunc / dataset storage)
# ---------------------------------------------------------------------------


def find_peaks_binary(x, **kwargs):
    """scipy.signal.find_peaks + NaN boundaries -> binary mask."""
    peaks, _ = find_peaks(np.asarray(x), **kwargs)
    return add_NaN_boundaries(x, peaks)


def find_troughs_binary(x, **kwargs):
    """Find troughs (local minima) + NaN boundaries -> binary mask."""
    troughs, _ = find_peaks(-np.asarray(x), **kwargs)
    return add_NaN_boundaries(x, troughs)


def find_nearest_turning_points_binary(x, threshold=1, max_value=None, prominence=0.5, distance=2, **kwargs):
    """Convert a 1D signal into a binary mask marking boundaries of peak regions.

    Identifies peaks in the signal, then finds the nearest "turning points"
    (where the gradient is near zero) on either side of each peak. These
    turning points define the boundaries of peak regions. The result is a
    binary mask where 1 indicates a turning-point boundary.

    The algorithm works in four steps:
        1. Compute the gradient of x and find indices where |gradient| < threshold,
           treating these as candidate turning points (near-stationary regions).
        2. Find peaks in x using scipy.signal.find_peaks with any additional kwargs.
        3. For each peak, select the closest turning point to its left and right.
        4. Add boundaries at NaN transitions in the original signal.

    Args:
        x: Input 1D signal.
        threshold: Maximum absolute gradient value to qualify as a turning point.
            Lower values select only very flat regions. Default is 1.
        max_value: If set, discard turning points where x exceeds this value.
            Useful for ignoring turning points on high plateaus.
        **kwargs: Passed to scipy.signal.find_peaks (e.g. height, distance,
            prominence).

    Returns:
        Binary array of same length as x, with 1 at turning-point boundaries
        and NaN-transition boundaries, 0 elsewhere.
    """
    x = np.asarray(x, dtype=float)
    grad = np.gradient(x)
    turning_points = np.where((grad > -threshold) & (grad < threshold))[0]

    if max_value is not None:
        turning_points = turning_points[x[turning_points] < max_value]

    peaks, _ = find_peaks(x, prominence=prominence, distance=distance, **kwargs)
    turning_points = np.setdiff1d(turning_points, peaks)

    nearest = []
    for peak in peaks:
        left = turning_points[turning_points < peak]
        right = turning_points[turning_points > peak]
        if len(left) > 0:
            nearest.append(left[-1])
        if len(right) > 0:
            nearest.append(right[0])

    return add_NaN_boundaries(x, np.array(nearest, dtype=int))


# ---------------------------------------------------------------------------
# Changepoint time extraction
# ---------------------------------------------------------------------------


def changepoint_fired(mask: xr.DataArray, selections: Mapping[str, Any] | None = None) -> np.ndarray:
    """Where *mask* fires at *selections*: a boolean ``(T,)`` array on the mask's own time axis.

    This is the one reading of a changepoint mask the GUI has. The lineplot
    draws it (``XarrayLoader.select``) and a click snaps to it
    (``XarrayLoader.get_cp_times``), so what is drawn and what is snapped to
    are the same set by construction.

    A selection key that is a dim of the mask pins that dim — ``.sel`` where
    the dim has a coordinate, ``.isel`` where it does not (the rule of
    :func:`ethograph.utils.xr_utils.sel_valid`). Every other key is ignored,
    and every non-time dim left free is OR'd across.
    """
    da = mask
    for dim, value in (selections or {}).items():
        if dim not in da.dims:
            continue
        if dim in da.coords:
            coord = da.coords[dim]
            if isinstance(value, str) and coord.dtype.kind in "iuf":
                value = coord.dtype.type(value)
            da = da.sel({dim: value})
        else:
            da = da.isel({dim: int(value)})
    time_dim = next(d for d in da.dims if "time" in str(d).lower())
    free = [d for d in da.dims if d != time_dim]
    fired = da.any(dim=free) if free else da
    return np.asarray(fired.transpose(time_dim).values, dtype=bool)


def changepoint_mask_times(mask: xr.DataArray, selections: Mapping[str, Any] | None = None) -> np.ndarray:
    """Times at which *mask* fires at *selections*, read off the mask's own time coordinate."""
    time = get_time_coord(mask)
    if time is None:
        raise ValueError(f"changepoint mask {mask.name!r} has no time coordinate")
    fired = changepoint_fired(mask, selections)
    return np.asarray(time.values, dtype=np.float64)[np.flatnonzero(fired)]


def dataset_changepoint_times(
    ds: xr.Dataset,
    feature: str | None = None,
    selections: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Sorted, unique times of every changepoint mask in *ds* — of *feature* only when given.

    Masks that target different features are simply unioned; there is no
    merge step to refuse them. Empty when nothing matches.
    """
    times = [
        changepoint_mask_times(ds[name], selections)
        for name in schema.changepoint_vars(ds)
        if feature is None or ds[name].attrs.get("target_feature") == feature
    ]
    if not times:
        return np.array([], dtype=np.float64)
    return np.unique(np.concatenate(times))


def correct_changepoints(
    df: pd.DataFrame,
    cp_times: np.ndarray,
    min_duration_s: float,
    stitch_gap_s: float,
    max_expansion_s: float,
    max_shrink_s: float,
    label_thresholds_s: dict[int, float] | None = None,
    do_purge: bool = True,
    do_stitch: bool = True,
    do_snap: bool = True,
    do_purge_after: bool = True,
) -> pd.DataFrame:
    """Full interval-native correction pipeline.

    Steps:
        1. purge_short_intervals — pre-cleanup (do_purge)
        2. stitch_intervals — merge same-label across small gaps (do_stitch)
        3. snap_boundaries — snap to changepoint times (do_snap)
        4. purge_short_intervals — post-cleanup (do_purge_after)
    """
    if df.empty:
        return df.copy()

    result = df
    if do_purge:
        result = purge_short_intervals(result, min_duration_s, label_thresholds_s)
    if do_stitch:
        result = stitch_intervals(result, stitch_gap_s)
    if do_snap:
        result = snap_boundaries(result, cp_times, max_expansion_s, max_shrink_s)
    if do_purge_after:
        result = purge_short_intervals(result, min_duration_s, label_thresholds_s)

    return result


def correct_changepoints_automatic(
    df: pd.DataFrame,
    min_duration_s: float = 1e-3,
    stitch_gap_s: float = 0.0,
) -> pd.DataFrame:
    """Lightweight cleanup used while manually creating labels."""
    if df.empty:
        return df.copy()

    result = purge_short_intervals(df, min_duration_s)
    return stitch_intervals(result, stitch_gap_s)


# ---------------------------------------------------------------------------
# Merge changepoints
# ---------------------------------------------------------------------------


def merge_changepoints(ds, vars: Sequence[str] | None = None, keep_dims: Sequence[str] | None = None):
    """Merge changepoint variables in a dataset into a single boolean mask.

    Combines every raw changepoint mask (``attrs["changepoint_mask"]``; see
    :func:`ethograph.io.schema.is_changepoint`) using logical OR across all
    non-time dimensions — the smooth expansions of a mask are ordinary
    features and are not merged. All masks must share the same
    ``target_feature`` attribute.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing one or more changepoint variables.
    vars : sequence of str, optional
        Which changepoint masks to merge; default merges every one
        :func:`~ethograph.io.schema.changepoint_vars` finds. Naming a
        variable that is not a changepoint mask is an error.
    keep_dims : sequence of str, optional
        Dims to leave standing instead of ORing across — typically the
        individual dim, so one animal's changepoints do not leak into
        another's. Default collapses every non-time dim.

    Returns
    -------
    ds : xr.Dataset
        Copy of the input with a new ``"changepoints"`` DataArray
        (float 0/1) replacing the merged changepoint variables.
    target_feature : str
        The shared ``target_feature`` attribute from the input variables.

    Raises
    ------
    ValueError
        If changepoint variables reference different target features.
    """
    ds = ds.copy()
    if vars is None:
        cp_ds = schema.filter_changepoints(ds)
    else:
        unknown = [v for v in vars if v not in ds.data_vars or not schema.is_changepoint(ds[v])]
        if unknown:
            raise ValueError(
                f"{unknown} are not changepoint masks in this dataset (available: {schema.changepoint_vars(ds)})"
            )
        cp_ds = ds[list(vars)]

    target_feature = []
    for var in cp_ds.data_vars:
        target_feature.append(cp_ds[var].attrs["target_feature"])

    if np.unique(target_feature).size > 1:
        raise ValueError(
            f"Not allowed to merge changepoints for different target features: {np.unique(target_feature)}"
        )

    keep = set(keep_dims or ())
    dims = [dim for dim in cp_ds.dims if dim not in ["trials", "time"] and dim not in keep]

    ds["changepoints"] = cp_ds.to_array().any(dim=["variable"] + dims).astype(float)
    ds["changepoints"].attrs.update(schema.changepoint_attrs(target_feature=target_feature[0]))

    ds = ds.drop_vars(list(cp_ds.data_vars))

    return ds, target_feature[0]


# ---------------------------------------------------------------------------
# ML feature engineering
# ---------------------------------------------------------------------------


def _proximity(changepoint_indices: np.ndarray, seq_length: int, sigma: float, distribution: str) -> np.ndarray:
    """Sum of one kernel per changepoint; an isolated changepoint reads exactly 1 at its frame."""
    x = np.arange(seq_length)
    peak = np.zeros(seq_length)
    for idx in changepoint_indices:
        if distribution == "gaussian":
            peak += np.exp(-0.5 * ((x - idx) / sigma) ** 2)
        else:
            peak += np.exp(-np.abs(x - idx) / sigma)
    return peak


def _offsets(changepoint_indices: np.ndarray, seq_length: int, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """``(since, until)``: samples since the previous / until the next changepoint, clipped at *horizon*, in ``[0, 1]``.

    Before the first changepoint ``since`` is saturated, after the last
    ``until`` is — "no candidate in reach".
    """
    idx = np.arange(seq_length)
    n = len(changepoint_indices)
    if n == 0:
        return np.ones(seq_length), np.ones(seq_length)
    prev = np.searchsorted(changepoint_indices, idx, side="right") - 1
    nxt = np.searchsorted(changepoint_indices, idx, side="left")
    since = np.where(prev >= 0, idx - changepoint_indices[np.maximum(prev, 0)], horizon)
    until = np.where(nxt < n, changepoint_indices[np.minimum(nxt, n - 1)] - idx, horizon)
    return np.minimum(since, horizon) / horizon, np.minimum(until, horizon) / horizon


def _segment_lengths(changepoint_indices: np.ndarray, seq_length: int, max_length: float) -> np.ndarray:
    """Length of the candidate segment each frame sits in, on a log scale saturating at *max_length*, in ``[0, 1]``.

    Segments are cut at every changepoint and at both ends of the trial.
    ``log1p`` so that 1, 10 and 100 samples are evenly spread — the linear
    offsets already resolve everything shorter than two horizons, this
    column's job is the long range.
    """
    edges = np.unique(np.concatenate([[0], changepoint_indices, [seq_length]]))
    out = np.zeros(seq_length)
    for start, end in zip(edges[:-1], edges[1:]):
        out[start:end] = end - start
    return np.minimum(np.log1p(out) / np.log1p(max_length), 1.0)


#: ``max_length`` default, in horizons: the length column saturates at ``16 * horizon`` samples.
LENGTH_HORIZONS = 16.0

#: The short end of the label durations the horizon is read off.
SHORT_PERCENTILE = 5.0

#: The long end of the label durations ``max_length`` is read off.
LONG_PERCENTILE = 95.0

#: ``horizon = HORIZON_FRACTION * p5(duration)``: the largest radius that keeps the shortest label's edges apart.
HORIZON_FRACTION = 0.5

#: The derived kernel ladder, ``horizon / k`` — the widest fades at the horizon (``e^-4``).
SIGMA_DIVISORS: tuple[float, ...] = (16.0, 8.0, 4.0)


@dataclass(frozen=True)
class ChangepointScales:
    """The three temporal scales of the expansion, in samples."""

    horizon: float
    max_length: float
    sigmas: tuple[float, ...]


def scales_from_durations(durations_s: Sequence[float] | np.ndarray, fs: float) -> ChangepointScales:
    """Read the expansion's scales off the labelled behaviours' durations.

    Two percentiles, two knobs, no taste: the offsets must resolve the
    shortest labelled behaviour, so the horizon is half its duration
    (:data:`HORIZON_FRACTION` × the :data:`SHORT_PERCENTILE` percentile) — any larger and
    both edges of that label fall inside one horizon; the length column
    stops being informative where almost nothing labelled is longer, so
    ``max_length`` is the :data:`LONG_PERCENTILE` percentile. Kernel widths only mean
    something relative to the boundary radius, so the sigmas are the ladder
    ``horizon / (16, 8, 4)``. Point events carry no duration and must be
    left out by the caller.
    """
    d = np.asarray(durations_s, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size < 2:
        raise ValueError(
            f"scales_from_durations needs at least two positive label durations, got {d.size} — "
            "label some behaviours, or spell sigmas/horizon/max_length in the config"
        )
    if not fs > 0:
        raise ValueError(f"fs must be positive, got {fs}")
    short_s = float(np.percentile(d, SHORT_PERCENTILE))
    long_s = float(np.percentile(d, LONG_PERCENTILE))
    horizon = round(max(1.0, HORIZON_FRACTION * short_s * fs), 4)
    max_length = round(max(long_s * fs, 2.0 * horizon), 4)
    return ChangepointScales(horizon, max_length, tuple(round(horizon / k, 4) for k in SIGMA_DIVISORS))


def default_horizon(sigmas: Sequence[float]) -> float:
    """Where the widest proximity kernel has faded: ``4 * max(sigmas)`` samples (``e^-4``, about 2 % of its peak).

    The one default the offset columns have, so the two families reach
    equally far.
    """
    if not sigmas:
        raise ValueError("horizon must be given explicitly when no sigmas are set (the default is 4 * max(sigmas))")
    return 4.0 * float(max(sigmas))


def more_changepoint_features(
    changepoint_binary: np.ndarray,
    sigmas: Sequence[float],
    distribution: Literal["gaussian", "laplacian"] = "laplacian",
    horizon: float | None = None,
    scale: np.ndarray | None = None,
    max_length: float | None = None,
) -> np.ndarray:
    """Create changepoint-based features from a binary changepoint array.

    Four column groups, in this order (see :data:`CP_TRANSFORMS`):

    - **Binary** — the exact changepoint positions (0/1 mask).
      Example: ``0 0 0 0 1 0 0 0 1 0 0 0 0 0``
    - **Proximity** — one column per sigma: a Laplacian (or Gaussian) kernel
      centred at each changepoint, summed. An isolated changepoint reads 1 at
      its frame; overlapping kernels add, so a cluster of candidates reads
      above 1. Nothing is normalised per trial.
      Example (one sigma): ``0 0 0 .3 1 .3 0 .3 1 .3 0 0 0 0``
    - **Offset** — two columns: samples *since* the previous changepoint and
      *until* the next, each clipped at *horizon* and scaled to ``[0, 1]``.
      A frame reads which side of its nearest candidate it is on, which the
      symmetric kernels cannot say; before the first / after the last
      changepoint the column is saturated (no candidate in reach).
      Example (horizon 4): since ``1 1 1 1 0 .25 .5 .75 0 .25 .5 .75 1 1``,
      until ``1 .75 .5 .25 0 .75 .5 .25 0 1 1 1 1 1``
    - **Length** — one column: the length of the candidate segment the frame
      sits in (cut at every changepoint and at the trial's ends), as
      ``log1p(length) / log1p(max_length)``, clipped at 1. The offsets
      already resolve anything shorter than two horizons; this is the long
      range, where a fragment-prone short segment and a whole bout of rest
      should not read alike.

    Proximity uses a Laplacian kernel by default:

    .. math::

        \\text{prox}(t) = \\sum_i \\exp\\!\\left(-\\frac{|t - i|}{\\sigma}\\right)

    where :math:`i` are the changepoint indices and :math:`\\sigma` controls
    the peak width. Laplacians have a narrow peak that points directly at the
    changepoint while their long tails remain visible from far away. Passing
    multiple ``sigmas`` (e.g. ``[0.5, 3, 5]``) yields features at several
    scales.

    *scale* multiplies every proximity column by :math:`\\exp(-x / \\bar{x})`
    of that signal, emphasising changepoints where it is low — with speed,
    the troughs before and after a movement rather than a dip inside one;
    with an amplitude envelope, the silences between calls. Frames where
    *scale* is NaN read 0.

    Args:
        changepoint_binary: Binary (0/1) array marking changepoint locations.
        sigmas: Kernel widths (samples) for the proximity columns.
        distribution: ``"laplacian"`` (default) or ``"gaussian"`` kernel.
        horizon: Reach of the offset columns, in samples; ``None`` is
            :func:`default_horizon`.
        scale: Optional signal on the same time axis that scales the
            proximity columns.
        max_length: Where the length column saturates, in samples; ``None``
            is :data:`LENGTH_HORIZONS` horizons.

    Returns:
        2D array of shape ``(T, 1 + len(sigmas) + 3)``: the binary mask, one
        proximity column per sigma, ``since``, ``until``, then ``length``.
    """
    changepoint_binary = np.asarray(changepoint_binary)
    seq_length = len(changepoint_binary)
    changepoint_indices = np.flatnonzero(changepoint_binary)
    horizon = default_horizon(sigmas) if horizon is None else float(horizon)
    if not horizon > 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    max_length = LENGTH_HORIZONS * horizon if max_length is None else float(max_length)
    if not max_length > 0:
        raise ValueError(f"max_length must be positive, got {max_length}")

    proximity = [_proximity(changepoint_indices, seq_length, sigma, distribution) for sigma in sigmas]
    if scale is not None:
        scale = np.asarray(scale, dtype=float)
        if scale.shape != (seq_length,):
            raise ValueError(f"scale must have shape ({seq_length},), got {scale.shape}")
        if np.all(np.isnan(scale)):
            multiplier = np.ones(seq_length)
        else:
            multiplier = np.exp(-scale / (np.nanmean(scale) + 1e-8))
        proximity = [np.nan_to_num(col * multiplier, nan=0.0) for col in proximity]

    since, until = _offsets(changepoint_indices, seq_length, horizon)
    length = _segment_lengths(changepoint_indices, seq_length, max_length)
    return np.column_stack([changepoint_binary.astype(float), *proximity, since, until, length])


#: The four column groups ``more_changepoint_features`` produces — the
#: vocabulary ``add_changepoint_features``'s ``transforms`` and
#: ``segment.config.ChangepointFeaturesConfig.transforms`` both read.
#: ``proximity`` names the *shape* generically because which kernel it uses
#: is a separate choice (``distribution``), and whether a signal scales it
#: another (``scale_by``). Its columns are named by rank (``_cp_prox0`` is
#: the narrowest), never by the sigma value: the values may be derived from
#: the labels after the column names are fixed.
CP_BINARY = "binary"
#: How mask ``var``'s ``binary`` column is named (``{var}_cp_binary``): what a layout looks for to find candidates.
CP_BINARY_SUFFIX = "_cp_binary"
CP_PROXIMITY = "proximity"
CP_OFFSET = "offset"
CP_LENGTH = "length"
CP_TRANSFORMS: tuple[str, ...] = (CP_BINARY, CP_PROXIMITY, CP_OFFSET, CP_LENGTH)


def _n_sigmas(sigmas: int | Sequence[float]) -> int:
    return int(sigmas) if isinstance(sigmas, int) else len(sigmas)


def _cp_feature_groups(var: str, sigmas: int | Sequence[float]) -> list[tuple[str, str]]:
    """``(name, transform_group)`` pairs, in ``more_changepoint_features``'s column order.

    *sigmas* is the list or just its length — the names depend on nothing else.
    """
    pairs = [(f"{var}{CP_BINARY_SUFFIX}", CP_BINARY)]
    pairs += [(f"{var}_cp_prox{i}", CP_PROXIMITY) for i in range(_n_sigmas(sigmas))]
    pairs += [(f"{var}_cp_since", CP_OFFSET), (f"{var}_cp_until", CP_OFFSET)]
    pairs.append((f"{var}_cp_length", CP_LENGTH))
    return pairs


def cp_feature_names(var: str, sigmas: int | Sequence[float], transforms: Sequence[str] = CP_TRANSFORMS) -> list[str]:
    """Column names ``add_changepoint_features(..., transforms=transforms)`` writes for *var*.

    Use this to name the exact columns a ``changepoint_features`` config will
    produce without materialising anything — e.g. to paste into
    ``features.columns`` by hand, or to check a config's generated layout.
    *sigmas* may be the list or just how many there are.
    """
    unknown = set(transforms) - set(CP_TRANSFORMS)
    if unknown:
        raise ValueError(f"transforms must be a subset of {CP_TRANSFORMS}, got {sorted(unknown)}")
    wanted = set(transforms)
    return [name for name, group in _cp_feature_groups(var, sigmas) if group in wanted]


def add_changepoint_features(
    ds: xr.Dataset,
    sigmas: Sequence[float],
    distribution: Literal["gaussian", "laplacian"] = "laplacian",
    transforms: Sequence[str] = CP_TRANSFORMS,
    vars: Sequence[str] | None = None,
    horizon: float | None = None,
    scale_by: str | None = None,
    max_length: float | None = None,
) -> xr.Dataset:
    """Expand changepoint variables into the ML features of ``more_changepoint_features``.

    For each changepoint data_var (see :func:`ethograph.io.schema.is_changepoint`; binary 0/1 over
    ``time`` plus any subset of ``keypoint``/``individual``) the requested
    *transforms* are computed per column and added as data_vars with the
    changepoint var's dims — see :data:`CP_TRANSFORMS` / :func:`cp_feature_names`
    for the exact names:

    - ``binary`` → ``{var}_cp_binary`` — the mask itself
    - ``proximity`` → ``{var}_cp_prox{i}`` — one kernel column per sigma, in
      the order given, scaled by *scale_by* when given; the sigma value is in
      the column's ``description``
    - ``offset`` → ``{var}_cp_since`` / ``{var}_cp_until`` — samples since the
      previous / until the next changepoint, clipped at *horizon*
    - ``length`` → ``{var}_cp_length`` — log length of the candidate segment
      the frame sits in, saturating at *max_length*

    All of them carry ``attrs["normalise"] = 0`` (already on a fixed scale)
    and none is marked as a changepoint variable — neither the ``kind`` nor
    the legacy ``type`` — so only the raw binary masks remain changepoints:
    they are the only ones that can be OR-merged, snapped to or drawn as lines.

    Parameters
    ----------
    ds : xr.Dataset
        Trial dataset holding changepoint vars (and *scale_by*, if given).
    sigmas : sequence of float
        Kernel widths (samples) for the proximity columns.
    distribution : ``"laplacian"`` or ``"gaussian"``
        Kernel shape, see ``more_changepoint_features``.
    transforms : subset of :data:`CP_TRANSFORMS`
        Which column groups to keep; default is all four.
    vars : sequence of str, optional
        Which changepoint variables to expand; default expands every one
        :func:`~ethograph.io.schema.changepoint_vars` finds. Naming a
        variable that is not a changepoint mask is an error.
    horizon : float, optional
        Reach of the offset columns in samples; default
        :func:`default_horizon` (``4 * max(sigmas)``).
    scale_by : str, optional
        Feature whose values scale the proximity columns by
        ``exp(-x / mean x)`` — e.g. ``"speed"`` to emphasise changepoints at
        rest. It must be on the changepoint var's time axis and carry no dim
        the changepoint var lacks. Default: unscaled.
    max_length : float, optional
        Where the length column saturates, in samples; default
        :data:`LENGTH_HORIZONS` × *horizon*.

    Returns
    -------
    xr.Dataset
        Copy of ``ds`` with the new data_vars.
    """
    sigmas = list(sigmas)
    ds = ds.copy()
    cp_vars = schema.changepoint_vars(ds) if vars is None else list(vars)
    if not cp_vars:
        raise ValueError("Dataset has no changepoint data_var (kind='changepoint_feature' or type='changepoints')")
    for var in cp_vars:
        if var not in ds.data_vars or not schema.is_changepoint(ds[var]):
            raise ValueError(
                f"{var!r} is not a changepoint mask in this dataset (available: {schema.changepoint_vars(ds)})"
            )
    unknown = set(transforms) - set(CP_TRANSFORMS)
    if unknown:
        raise ValueError(f"transforms must be a subset of {CP_TRANSFORMS}, got {sorted(unknown)}")

    for var in cp_vars:
        cp = ds[var]
        time = get_time_coord(cp).name
        if scale_by is None:
            stacked = xr.apply_ufunc(
                lambda mask: more_changepoint_features(mask, sigmas, distribution, horizon, max_length=max_length),
                cp,
                input_core_dims=[[time]],
                output_core_dims=[[time, "cp_feature"]],
                vectorize=True,
                output_dtypes=[np.float64],
            )
        else:
            scale = ds[scale_by]
            extra = [d for d in scale.dims if d not in cp.dims]
            if extra:
                raise ValueError(
                    f"scale_by feature {scale_by!r} has dims {extra} that changepoint var {var!r} "
                    f"({cp.dims}) lacks; pin them first"
                )
            if time not in scale.dims:
                raise ValueError(f"scale_by feature {scale_by!r} is not on the {time!r} axis of {var!r}")
            stacked = xr.apply_ufunc(
                lambda mask, vals: more_changepoint_features(mask, sigmas, distribution, horizon, vals, max_length),
                cp,
                scale,
                input_core_dims=[[time], [time]],
                output_core_dims=[[time, "cp_feature"]],
                vectorize=True,
                output_dtypes=[np.float64],
            )
        all_pairs = _cp_feature_groups(var, sigmas)
        assert stacked.sizes["cp_feature"] == len(all_pairs)
        wanted = set(transforms)
        scaled = f", scaled by {scale_by}" if scale_by is not None else ""
        details = {
            f"{var}_cp_prox{i}": f": proximity, sigma {sigma:g} samples{scaled}" for i, sigma in enumerate(sigmas)
        }
        for i, (name, group) in enumerate(all_pairs):
            if group not in wanted:
                continue
            feature = stacked.isel(cp_feature=i, drop=True).transpose(*cp.dims)
            feature.attrs = {
                "description": f"Changepoint feature of {var}{details.get(name, '')}",
                schema.KIND: schema.CHANGEPOINT_FEATURE,
                "normalise": 0,
            }
            ds[name] = feature

    return ds
