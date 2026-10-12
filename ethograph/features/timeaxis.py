"""Apply a numpy kernel along the time axis of whatever holds the data.

A *kernel* is a plain numpy function ``f(x, fs, *, axis=-1, **params)`` that
returns an array of the same shape. It never knows which container the data
came from and never knows the rate on its own: the adapter reads the rate off
the time axis and passes it in, so every parameter a kernel takes is a
duration or a frequency, never a sample count.

:func:`apply_along_time` dispatches on the container — a numpy array, an
``xarray.DataArray`` or a pynapple ``Tsd``/``TsdFrame``/``TsdTensor`` — and
is the one place the three meet. The GUI preview runs a kernel on the numpy
it already plots; the snippet it copies runs the same kernel on the dataset.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Callable

import numpy as np
import pynapple as nap
import xarray as xr

from ethograph.utils.xr_utils import get_time_coord

Kernel = Callable[..., np.ndarray]

#: Relative spread of sample intervals above which a time axis is not uniform.
JITTER_TOLERANCE = 0.02


def sampling_rate(time: np.ndarray, *, tolerance: float = JITTER_TOLERANCE, name: str = "the time axis") -> float:
    """The rate in Hz of a uniformly sampled *time* vector.

    Raises ``ValueError`` when fewer than two samples exist or when the
    intervals disagree with their median by more than *tolerance*: a kernel
    whose width is a duration has no meaning on an irregular axis.
    """
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError(f"{name} needs at least two samples to have a rate")
    dt = np.diff(t)
    median = float(np.median(dt))
    if median <= 0:
        raise ValueError(f"{name} is not increasing")
    spread = float(np.max(np.abs(dt - median)) / median)
    if spread > tolerance:
        raise ValueError(f"{name} is not uniformly sampled (intervals vary by {spread:.0%} around {median:.4g} s)")
    return 1.0 / median


def describe_call(f: Kernel, params: dict[str, Any]) -> str:
    """``gaussian(sigma_s=0.05)`` — the kernel and its parameters as one string."""
    args = ", ".join(f"{k}={v!r}" for k, v in params.items())
    return f"{getattr(f, '__name__', 'kernel')}({args})"


@singledispatch
def apply_along_time(obj: Any, f: Kernel, **params: Any) -> Any:
    """Run kernel *f* along the time axis of *obj*, returning the same kind of object.

    ``params`` are the kernel's own keyword arguments. A bare numpy array has
    no time axis of its own, so it also needs ``fs=`` and is taken time-first.
    """
    raise TypeError(f"apply_along_time: unsupported container {type(obj).__name__}")


@apply_along_time.register
def _(obj: np.ndarray, f: Kernel, *, fs: float, **params: Any) -> np.ndarray:
    return f(np.asarray(obj, dtype=float), fs, axis=0, **params)


@apply_along_time.register
def _(obj: xr.DataArray, f: Kernel, **params: Any) -> xr.DataArray:
    time = get_time_coord(obj)
    if time is None:
        raise ValueError(f"{obj.name!r} has no time coordinate")
    time_dim = time.dims[0]
    fs = sampling_rate(time.values, name=f"{obj.name!r}'s {time_dim}")
    out = xr.apply_ufunc(
        f,
        obj,
        kwargs={"fs": fs, "axis": -1, **params},
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        keep_attrs=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    out = out.transpose(time_dim, ...)
    out.attrs["smoothing"] = describe_call(f, params)
    return out


def _apply_per_epoch(obj: nap.Tsd | nap.TsdFrame | nap.TsdTensor, f: Kernel, params: dict[str, Any]) -> np.ndarray:
    """Filter each epoch of the time support on its own, so a kernel never
    reaches across the gap between two trials."""
    parts = []
    for start, end in zip(obj.time_support.start, obj.time_support.end):
        part = obj.restrict(nap.IntervalSet(start, end))
        values = np.asarray(part.values, dtype=float)
        if len(values) < 2:
            parts.append(values)
            continue
        fs = sampling_rate(part.t, name=f"epoch {start:.3f}-{end:.3f} s")
        parts.append(f(values, fs, axis=0, **params))
    out = np.concatenate(parts) if parts else np.asarray(obj.values, dtype=float)
    if len(out) != len(obj):
        raise ValueError("the time support does not cover every sample; restrict the series to it first")
    return out


@apply_along_time.register
def _(obj: nap.Tsd, f: Kernel, **params: Any) -> nap.Tsd:
    return nap.Tsd(t=obj.t, d=_apply_per_epoch(obj, f, params), time_support=obj.time_support)


@apply_along_time.register
def _(obj: nap.TsdFrame, f: Kernel, **params: Any) -> nap.TsdFrame:
    return nap.TsdFrame(t=obj.t, d=_apply_per_epoch(obj, f, params), columns=obj.columns, time_support=obj.time_support)


@apply_along_time.register
def _(obj: nap.TsdTensor, f: Kernel, **params: Any) -> nap.TsdTensor:
    return nap.TsdTensor(t=obj.t, d=_apply_per_epoch(obj, f, params), time_support=obj.time_support)
