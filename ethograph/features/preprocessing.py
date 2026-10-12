import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import butter, savgol_filter, sosfiltfilt

from ethograph.features.timeaxis import Kernel, apply_along_time


def downsample_with_antialiasing(time: np.ndarray, data: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Downsample array using moving-average low-pass filter before subsampling.

    Parameters
    ----------
    time : numpy.ndarray
        Time values of shape ``(T,)``.
    data : numpy.ndarray
        Data of shape ``(T,)`` or ``(T, D)``.
    factor : int
        Downsample factor.

    Returns
    -------
    time_ds : numpy.ndarray
        Downsampled time values.
    data_ds : numpy.ndarray
        Downsampled data.
    """
    smoothed = uniform_filter1d(data, size=factor, axis=0)
    return time[::factor], smoothed[::factor]


def resample_to_frames(
    data: np.ndarray,
    time_original: np.ndarray,
    time_target: np.ndarray,
) -> np.ndarray:
    """Resample data to target time points with anti-aliasing before interpolation.

    Parameters
    ----------
    data : numpy.ndarray
        Source data array.
    time_original : numpy.ndarray
        Time axis of the source data.
    time_target : numpy.ndarray
        Desired output time axis.

    Returns
    -------
    numpy.ndarray
        Resampled data at ``time_target`` points.
    """
    factor = len(time_original) / len(time_target)
    if factor > 1:
        data = uniform_filter1d(data, size=int(round(factor)), axis=0)
    f = interp1d(time_original, data, axis=0, fill_value="extrapolate")
    return f(time_target)


def interpolate_nans(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Interpolate NaNs using NumPy's interp, with leading/trailing NaNs set to zero.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array possibly containing NaNs.
    axis : int
        Axis along which to interpolate. Defaults to 0.

    Returns
    -------
    numpy.ndarray
        Array with NaNs interpolated and leading/trailing NaNs set to zero.

    Examples
    --------
    >>> arr = np.array([np.NaN, np.NaN, 2.0, np.NaN, 8.0, np.NaN, 10.0, np.NaN])
    >>> interpolate_nans(arr)
    array([ 0.,  0.,  2.,  5.,  8.,  9., 10.,  0.])
    """
    arr = np.asarray(arr)

    def interpolate_1d(x: np.ndarray) -> np.ndarray:
        mask = ~np.isnan(x)
        if not mask.any():
            return np.zeros_like(x)
        indices = np.arange(len(x))
        result = np.interp(indices, indices[mask], x[mask])
        # Set leading/trailing NaNs to zero
        first_valid, last_valid = np.where(mask)[0][[0, -1]]
        result[:first_valid] = 0
        result[last_valid + 1 :] = 0
        return result

    if arr.ndim == 1:
        return interpolate_1d(arr)
    return np.apply_along_axis(interpolate_1d, axis, arr)


def z_normalize(data: np.ndarray) -> np.ndarray:
    """Apply z-score normalization to each feature (column) independently.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(T, F)`` where F is the number of features.

    Returns
    -------
    numpy.ndarray
        Z-normalized array with zero mean and unit variance per column.
    """
    std = np.nanstd(data, axis=0)
    std[std == 0] = 1
    return (data - np.nanmean(data, axis=0)) / std


def clip_by_percentiles(features: np.ndarray, percentile_range: tuple[float, float] = (1, 99)) -> np.ndarray:
    """Clip feature values to percentile bounds.

    Parameters
    ----------
    features : numpy.ndarray
        Array of shape ``(T, F)`` — T time samples, F features.
    percentile_range : tuple[float, float]
        ``(lower, upper)`` percentiles for clipping.

    Returns
    -------
    numpy.ndarray
        Clipped features of shape ``(T, F)``.
    """
    # Compute percentiles per feature dimension
    lower = np.nanpercentile(features, percentile_range[0], axis=0, keepdims=True)
    upper = np.nanpercentile(features, percentile_range[1], axis=0, keepdims=True)

    features_clipped = np.clip(features, lower, upper)

    return features_clipped


def gaussian_smoothing(da, **smoothing_params):
    """
    Apply Gaussian smoothing to position data across DataArray dimensions.

    NaNs are bridged by interpolation before filtering and stay bridged.
    ``smoothing_params`` are ``gaussian_filter1d``'s own, so ``sigma`` is in
    samples; :func:`gaussian` takes it in seconds.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with position data.
    **smoothing_params
        Keyword arguments for gaussian_filter1d.

    Returns
    -------
    xarray.DataArray
        Smoothed position data.
    """

    def in_samples(x, fs, *, axis=-1, **params):
        return gaussian_filter1d(x, axis=axis, **params)

    return apply_along_time(da, nan_aware(in_samples, max_gap_s=None), **smoothing_params)


# ---------------------------------------------------------------------------
# Smoothing kernels — ``f(x, fs, *, axis, **params) -> same shape``
#
# Every parameter is a duration in seconds or a frequency in Hz, converted
# with the rate the caller passes in; a kernel never knows a rate itself.
# All are zero-phase, so an onset keeps its time.
# ---------------------------------------------------------------------------


def _window_samples(window_s: float, fs: float, *, minimum: int) -> int:
    """An odd sample count for a window of *window_s* seconds, refused when too short."""
    n = int(round(window_s * fs))
    if n < minimum:
        raise ValueError(f"a window of {window_s:g} s is {n} sample(s) at {fs:.4g} Hz; it needs at least {minimum}")
    return n if n % 2 else n + 1


def gaussian(x: np.ndarray, fs: float, *, sigma_s: float, axis: int = -1) -> np.ndarray:
    """Gaussian kernel with standard deviation *sigma_s* seconds."""
    sigma = sigma_s * fs
    if sigma < 0.5:
        raise ValueError(f"sigma of {sigma_s:g} s is {sigma:.2f} samples at {fs:.4g} Hz; it needs at least 0.5")
    return gaussian_filter1d(x, sigma=sigma, axis=axis)


def savgol(x: np.ndarray, fs: float, *, window_s: float, polyorder: int = 3, axis: int = -1) -> np.ndarray:
    """Savitzky-Golay: a local polynomial fit, which keeps peak heights."""
    n = _window_samples(window_s, fs, minimum=polyorder + 2)
    return savgol_filter(x, window_length=n, polyorder=polyorder, axis=axis)


def median(x: np.ndarray, fs: float, *, window_s: float, axis: int = -1) -> np.ndarray:
    """Running median: removes single-sample spikes without smearing them."""
    n = _window_samples(window_s, fs, minimum=3)
    size = [1] * x.ndim
    size[axis] = n
    return median_filter(x, size=tuple(size), mode="nearest")


def moving_average(x: np.ndarray, fs: float, *, window_s: float, axis: int = -1) -> np.ndarray:
    """Centred boxcar of *window_s* seconds."""
    n = _window_samples(window_s, fs, minimum=2)
    return uniform_filter1d(x, size=n, axis=axis, mode="nearest")


def lowpass(x: np.ndarray, fs: float, *, cutoff_hz: float, order: int = 2, axis: int = -1) -> np.ndarray:
    """Butterworth low-pass run forward and backward (zero-phase)."""
    if cutoff_hz >= fs / 2:
        raise ValueError(f"a cutoff of {cutoff_hz:g} Hz is not below the Nyquist rate of {fs / 2:.4g} Hz")
    sos = butter(order, cutoff_hz, fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=axis)


#: The smoothing kernels by name, as a dialog or a snippet spells them.
SMOOTHERS: dict[str, Kernel] = {
    "gaussian": gaussian,
    "savgol": savgol,
    "median": median,
    "moving_average": moving_average,
    "lowpass": lowpass,
}


def _fill_nans(x: np.ndarray, axis: int) -> np.ndarray:
    """Linear interpolation across NaNs, holding the nearest valid value at the edges."""

    def fill_1d(v: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(v)
        if valid.all():
            return v
        if not valid.any():
            return np.zeros_like(v)
        idx = np.arange(len(v))
        return np.interp(idx, idx[valid], v[valid])

    return np.apply_along_axis(fill_1d, axis, x)


def _long_gaps(mask: np.ndarray, axis: int, min_len: int) -> np.ndarray:
    """The NaN runs in *mask* of at least *min_len* samples."""

    def runs_1d(m: np.ndarray) -> np.ndarray:
        out = np.zeros_like(m)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(int), [0]))))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start >= min_len:
                out[start:stop] = True
        return out

    return np.apply_along_axis(runs_1d, axis, mask)


def bridge_gaps(x: np.ndarray, fs: float, *, max_gap_s: float | None = None, axis: int = -1) -> np.ndarray:
    """Interpolate across NaN gaps of at most *max_gap_s* seconds (``None``: every gap).

    The gap-handling step of a pipeline; the smoothing kernels then leave
    whatever gaps remain exactly where they are.
    """
    x = np.asarray(x, dtype=float)
    mask = np.isnan(x)
    if not mask.any():
        return x
    filled = _fill_nans(x, axis)
    if max_gap_s is not None:
        filled[_long_gaps(mask, axis, int(round(max_gap_s * fs)) + 1)] = np.nan
    return filled


def nan_aware(f: Kernel, *, max_gap_s: float | None = 0.0) -> Kernel:
    """Wrap kernel *f* so it tolerates NaNs.

    Gaps are bridged by interpolation before *f* runs. Afterwards every gap
    longer than *max_gap_s* seconds is put back as NaN: ``0.0`` (the default)
    keeps every gap, ``None`` bridges them all.
    """

    def wrapped(x: np.ndarray, fs: float, *, axis: int = -1, **params) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mask = np.isnan(x)
        if not mask.any():
            return f(x, fs, axis=axis, **params)
        y = f(_fill_nans(x, axis), fs, axis=axis, **params)
        if max_gap_s is None:
            return y
        min_len = max(1, int(round(max_gap_s * fs)) + 1)
        y[_long_gaps(mask, axis, min_len)] = np.nan
        return y

    wrapped.__name__ = f.__name__
    wrapped.__doc__ = f.__doc__
    return wrapped
