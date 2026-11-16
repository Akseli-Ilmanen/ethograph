from __future__ import annotations

from functools import partial

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from ethograph.features.movement import get_angle_rgb
from ethograph.io import schema
from ethograph.io.trialtree import TrialTree
from ethograph.utils.xr_utils import get_time_coord


def downsample_trialtree(dt: TrialTree, factor: int) -> TrialTree:
    """Downsample every trial in a TrialTree using a min-max envelope.

    For each contiguous block of *factor* samples, two values are kept:
    the block minimum and maximum.  This preserves peaks and troughs
    (important for spike-like signals) while reducing the number of
    samples by roughly ``factor / 2``.

    All time-like dimensions in every trial are downsampled independently,
    so datasets with mixed sampling rates (e.g. 30 Hz pose + 44 kHz audio)
    are handled correctly.

    Parameters
    ----------
    dt : TrialTree
        Input tree.  Not modified in place.
    factor : int
        Number of raw samples per envelope block.  For example, ``factor=10``
        on a 30 Hz signal produces ~6 Hz output (2 points per block).

    Returns
    -------
    TrialTree
        New tree with downsampled data.  Each trial's ``attrs`` includes
        ``downsample_factor`` and ``downsample_method``.

    Examples
    --------
    >>> import ethograph as eto
    >>> dt = eto.open("high_rate_experiment.nc")
    >>> dt_small = eto.downsample_trialtree(dt, factor=20)
    >>> dt_small.save("experiment_downsampled.nc")
    """
    return dt.map_trials(lambda ds: _downsample_dataset(ds, factor))


def _minmax_envelope(values: np.ndarray, n_segments: int, factor: int) -> np.ndarray:
    shape_suffix = values.shape[1:]
    reshaped = values[: n_segments * factor].reshape(n_segments, factor, *shape_suffix)
    interleaved = np.empty((n_segments * 2, *shape_suffix), dtype=values.dtype)
    interleaved[0::2] = reshaped.min(axis=1)
    interleaved[1::2] = reshaped.max(axis=1)
    return interleaved


def _find_time_dims(ds: xr.Dataset) -> set[str]:
    """Find all time-like dimension names in a dataset."""
    return {dim for dim in ds.dims if "time" in dim.lower()}


def _downsample_along_time_dim(
    ds: xr.Dataset,
    time_dim: str,
    factor: int,
) -> tuple[dict, dict]:
    """Downsample variables along a single time dimension.

    Returns (new_coord_entry, downsampled_vars) for the given time_dim.
    """
    n_time = ds.sizes[time_dim]
    n_segments = n_time // factor
    if n_segments < 2:
        return {}, {}

    usable_len = n_segments * factor
    time_vals = ds.coords[time_dim].values[:usable_len]
    time_downsampled = time_vals[::factor][:n_segments]
    step = (time_vals[-1] - time_vals[0]) / len(time_vals) if len(time_vals) > 1 else 1.0
    half_step = step * factor / 2

    time_interleaved = np.empty(n_segments * 2)
    time_interleaved[0::2] = time_downsampled
    time_interleaved[1::2] = time_downsampled + half_step

    coord_entry = {time_dim: time_interleaved}
    data_vars = {}

    for var_name, var_data in ds.data_vars.items():
        if time_dim not in var_data.dims:
            continue
        time_axis = var_data.dims.index(time_dim)
        values = var_data.values
        if time_axis != 0:
            values = np.moveaxis(values, time_axis, 0)

        other_dims = [d for d in var_data.dims if d != time_dim]
        interleaved = _minmax_envelope(values, n_segments, factor)
        data_vars[var_name] = xr.DataArray(interleaved, dims=[time_dim] + other_dims, attrs=var_data.attrs)

    return coord_entry, data_vars


def _downsample_dataset(ds: xr.Dataset, factor: int) -> xr.Dataset:
    time_dims = _find_time_dims(ds)
    if not time_dims:
        return ds

    new_coords: dict = {}
    all_data_vars: dict = {}
    downsampled_var_names: set[str] = set()

    for time_dim in sorted(time_dims):
        coord_entry, data_vars = _downsample_along_time_dim(ds, time_dim, factor)
        new_coords.update(coord_entry)
        all_data_vars.update(data_vars)
        downsampled_var_names.update(data_vars.keys())

    for coord_name, coord_val in ds.coords.items():
        if coord_name not in new_coords:
            new_coords[coord_name] = coord_val

    for var_name, var_data in ds.data_vars.items():
        if var_name not in downsampled_var_names:
            all_data_vars[var_name] = var_data

    new_attrs = ds.attrs.copy()
    new_attrs["downsample_factor"] = factor
    new_attrs["downsample_method"] = "minmax_envelope"
    return xr.Dataset(all_data_vars, coords=new_coords, attrs=new_attrs)


def add_changepoints_to_ds(ds, target_feature, changepoint_name, changepoint_func, **func_kwargs):
    """Detect changepoints in a feature and store them in the dataset.

    Applies *changepoint_func* independently along every non-time
    dimension (e.g. per keypoint, per individual) using
    :func:`xarray.apply_ufunc` with ``vectorize=True``.  The result is an
    ``int8`` binary array (1 = changepoint, 0 = not) stored as
    ``ds["{target_feature}_{changepoint_name}"]``.


    Parameters
    ----------
    ds : xarray.Dataset
        Trial dataset containing *target_feature*.
    target_feature : str
        Name of the variable to run detection on (e.g. ``"speed"``).
    changepoint_name : str
        Suffix for the output variable name.  The stored variable will be
        called ``"{target_feature}_{changepoint_name}"`` (e.g.
        ``"speed_troughs"``).
    changepoint_func : callable
        A function ``f(x, **kwargs) -> array[int8]`` that takes a 1-D
        numpy array and returns a same-length binary indicator.
    **func_kwargs
        Forwarded to *changepoint_func*.

    Returns
    -------
    xarray.Dataset
        The input dataset with the changepoint variable added in place.

    Examples
    --------
    >>> import ethograph as eto
    >>> from ethograph.features.changepoints import find_troughs_binary
    >>> dt = eto.open("experiment.nc")
    >>> ds = dt.itrial(0)
    >>> ds = eto.add_changepoints_to_ds(
    ...     ds,
    ...     target_feature="speed",
    ...     changepoint_name="troughs",
    ...     changepoint_func=find_troughs_binary,
    ...     prominence=0.3,
    ... )
    >>> ds["speed_troughs"]
    <xarray.DataArray 'speed_troughs' (time: 9000, keypoint: 7)>
    """
    feature_data = ds[target_feature]
    func = partial(changepoint_func, **func_kwargs)

    time_dim = get_time_coord(feature_data).dims[0]
    changepoints = xr.apply_ufunc(
        func,
        feature_data,
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int8],
    )

    changepoints.attrs.update(schema.changepoint_attrs(target_feature=target_feature))

    ds[f"{target_feature}_{changepoint_name}"] = changepoints
    return ds


def add_angle_rgb_to_ds(ds: xr.Dataset, smoothing_params: dict | None) -> xr.Dataset:
    """Compute heading angles and RGB color-coding from 2-D position data.

    For each individual/keypoint combination, calculates the heading angle
    from consecutive (x, y) positions and maps it to an RGB color via
    :func:`~ethograph.features.movement.get_angle_rgb`. Gaussian smoothing
    is applied before angle computation.

    Adds two variables to *ds*:

    * ``angles`` -- heading angle in degrees.
    * ``angle_rgb`` -- ``(R, G, B)`` triplet per time-step

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing a ``position`` variable with at least
        ``space=["x", "y"]`` and a time dimension.
    smoothing_params : dict or None
        Keyword arguments forwarded to
        :func:`scipy.ndimage.gaussian_filter1d` (e.g. ``{"sigma": 3}``);
        ``None`` computes the angles from the positions as they are, without smoothing.

    Returns
    -------
    xarray.Dataset
        The input dataset with ``angles`` and ``angle_rgb`` added in-place.
    """
    xy_pos = ds.position.sel(space=["x", "y"])
    time_dim = get_time_coord(xy_pos).dims[0]
    smooth_func = gaussian_filter1d if smoothing_params is not None else None

    def process_angles(xy):
        _, angles = get_angle_rgb(xy, smooth_func=smooth_func, smoothing_params=smoothing_params)
        return angles

    angles = xr.apply_ufunc(
        process_angles,
        xy_pos,
        input_core_dims=[[time_dim, "space"]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    ds["angles"] = angles
    ds["angles_sin"] = np.sin(np.deg2rad(angles))
    ds["angles_cos"] = np.cos(np.deg2rad(angles))

    def process_rgb(xy):
        rgb, _ = get_angle_rgb(xy, smooth_func=smooth_func, smoothing_params=smoothing_params)
        return rgb

    angle_rgb = xr.apply_ufunc(
        process_rgb,
        xy_pos,
        input_core_dims=[[time_dim, "space"]],
        output_core_dims=[[time_dim, "RGB"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"RGB": 3}},
    )

    ds["angle_rgb"] = angle_rgb

    return ds
