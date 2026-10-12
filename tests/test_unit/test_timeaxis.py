"""One kernel, three containers: ``apply_along_time`` gives the same numbers on
numpy, xarray and pynapple, and the NaN policy and rate checks hold."""

import numpy as np
import pynapple as nap
import pytest
import xarray as xr

from ethograph.features.preprocessing import SMOOTHERS, gaussian, lowpass, median, nan_aware, savgol
from ethograph.features.timeaxis import apply_along_time, sampling_rate

FS = 50.0


def _signal(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = np.sin(2 * np.pi * 0.7 * t)[:, None] * np.array([1.0, 2.0, -0.5]) + rng.normal(0, 0.2, (n, 3))
    x[120:126, 1] = np.nan  # a short gap
    x[200:240, 2] = np.nan  # a long one
    return t, x


@pytest.mark.parametrize("name", sorted(SMOOTHERS))
def test_every_kernel_agrees_across_containers(name):
    t, x = _signal()
    params = {
        "gaussian": {"sigma_s": 0.05},
        "savgol": {"window_s": 0.3, "polyorder": 3},
        "median": {"window_s": 0.2},
        "moving_average": {"window_s": 0.2},
        "lowpass": {"cutoff_hz": 3.0},
    }[name]
    kernel = nan_aware(SMOOTHERS[name])

    on_numpy = apply_along_time(x, kernel, fs=FS, **params)
    da = xr.DataArray(x, dims=("time", "space"), coords={"time": t, "space": ["x", "y", "z"]}, name="pos")
    on_xarray = apply_along_time(da, kernel, **params)
    frame = nap.TsdFrame(t=t, d=x, columns=["x", "y", "z"])
    on_pynapple = apply_along_time(frame, kernel, **params)

    assert on_xarray.dims == ("time", "space")
    assert on_xarray.attrs["smoothing"].startswith(f"{name}(")
    np.testing.assert_allclose(on_xarray.values, on_numpy, equal_nan=True)
    np.testing.assert_allclose(on_pynapple.values, on_numpy, equal_nan=True)
    assert list(on_pynapple.columns) == ["x", "y", "z"]
    # the gaps are still gaps, the rest is finite
    assert np.isnan(on_numpy[120:126, 1]).all() and np.isnan(on_numpy[200:240, 2]).all()
    assert np.isfinite(on_numpy[:120]).all()


def test_nan_policy_bridges_short_gaps_only():
    t, x = _signal()
    keep = nan_aware(gaussian, max_gap_s=0.0)(x, FS, axis=0, sigma_s=0.05)
    short = nan_aware(gaussian, max_gap_s=0.2)(x, FS, axis=0, sigma_s=0.05)
    bridge = nan_aware(gaussian, max_gap_s=None)(x, FS, axis=0, sigma_s=0.05)
    assert np.isnan(keep[122, 1]) and np.isnan(keep[220, 2])
    assert np.isfinite(short[122, 1]) and np.isnan(short[220, 2])  # 6 samples = 0.12 s bridged, 40 samples kept
    assert np.isfinite(bridge).all()


def test_pynapple_epochs_are_filtered_apart():
    """A kernel never reaches across the gap between two epochs of the time support."""
    t = np.concatenate([np.arange(100) / FS, 10.0 + np.arange(100) / FS])
    x = np.concatenate([np.zeros(100), np.full(100, 5.0)])
    tsd = nap.Tsd(t=t, d=x, time_support=nap.IntervalSet([0.0, 10.0], [2.0, 12.0]))
    out = apply_along_time(tsd, gaussian, sigma_s=0.1)
    assert isinstance(out, nap.Tsd)
    np.testing.assert_allclose(out.values[:100], 0.0)
    np.testing.assert_allclose(out.values[100:], 5.0)


def test_rate_comes_from_the_axis_and_refuses_jitter():
    assert sampling_rate(np.arange(10) / 30.0) == pytest.approx(30.0)
    jittered = np.cumsum(np.r_[0.0, np.tile([0.02, 0.05], 20)])
    with pytest.raises(ValueError, match="not uniformly sampled"):
        sampling_rate(jittered)
    with pytest.raises(ValueError, match="at least two"):
        sampling_rate(np.array([0.0]))


def test_too_short_for_the_rate_is_refused_naming_it():
    x = np.zeros(50)
    with pytest.raises(ValueError, match="30 Hz"):
        savgol(x, 30.0, window_s=0.05, polyorder=3)
    with pytest.raises(ValueError, match="Nyquist"):
        lowpass(x, 30.0, cutoff_hz=20.0)
    with pytest.raises(ValueError, match="at least 3"):
        median(x, 30.0, window_s=0.01)


def test_xarray_needs_a_time_coordinate():
    da = xr.DataArray(np.zeros((5, 2)), dims=("frame", "space"), name="pos")
    with pytest.raises(ValueError, match="no time coordinate"):
        apply_along_time(da, gaussian, sigma_s=0.1)
