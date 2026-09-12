"""Feature columns — the one definition of a model's input layout.

A model config names its inputs as ``{feature: {dim: [values]}}``: for every
feature, the explicit values to take along each of its dims. Every
combination of values is one **feature column** — a feature with all its
dims pinned to a single value — selected through
:meth:`~ethograph.io.catalog.DataLoader.select` (the same path the plots
use), so the same spelling works for xarray, pynapple and NWB sessions.

Pinning explicit values (never "all") is what keeps the column set, and
therefore the model's input layout, identical across sessions. Both the
onset model (``labels/onset_model.py``) and the segmentation pipeline
(``ethograph/segment``) read their inputs through this module.

A feature can additionally contribute its **time derivative** as an extra
column per combination (:func:`time_derivative`) — how fast the value is
changing is often what marks an event, and a boosted tree cannot difference
its own inputs.

An **angle** is instead replaced by its ``(sin, cos)`` pair
(:func:`sin_cos_values`): a circular quantity read as a plain number lies to
every model about the distance between its two ends, and no amount of
z-scoring repairs the jump at the wrap. The units are the variable's own
``units`` attr where it declares one and are otherwise read off the values
(:func:`angle_units`), so degrees and radians both arrive as the same two
bounded columns.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

#: Relative tolerance for "same sampling rate" comparisons.
FS_RTOL = 1e-3

#: Suffix marking a derivative column, appended to the name of the column it
#: is taken from.
DERIVATIVE_SUFFIX = "d/dt"

#: The two components an angle is replaced by, in the order they are laid out.
SIN_COMPONENT = "sin"
COS_COMPONENT = "cos"
CIRCULAR_COMPONENTS: tuple[str, str] = (SIN_COMPONENT, COS_COMPONENT)

#: The two units an angle may be written in.
RADIANS = "radians"
DEGREES = "degrees"

#: How a ``units`` attr may spell each of them.
_UNIT_SPELLINGS: dict[str, str] = {
    "rad": RADIANS,
    "radian": RADIANS,
    "radians": RADIANS,
    "deg": DEGREES,
    "degree": DEGREES,
    "degrees": DEGREES,
    "°": DEGREES,
}

#: Largest magnitude a radian-valued angle reaches. Above it, the values are
#: degrees — a full turn is 6.28 in radians and 360 in degrees, so there is no
#: near miss to worry about (see :func:`angle_units`).
_RADIAN_CEILING = 2.0 * np.pi + 1e-6

#: A dim's value written ``"a..b"`` — sugar for the explicit inclusive range,
#: never a session-dependent "all" (see :func:`expand_dim_values`).
_RANGE_RE = re.compile(r"^(\d+)\.\.(\d+)$")


class FeatureColumn(NamedTuple):
    """One input column: a feature with every dim pinned to one value."""

    feature: str
    selections: dict[str, str]
    name: str
    #: This column is the pinned series' time derivative, not the series
    #: itself (see :func:`enumerate_columns`).
    derivative: bool = False
    #: Which component of the angle's ``(sin, cos)`` encoding this column is
    #: (``None`` = the value as the dataset stores it).
    circular: str | None = None


def column_name(
    feature: str,
    selections: dict[str, str],
    derivative: bool = False,
    circular: str | None = None,
) -> str:
    name = feature
    if selections:
        name += "|" + ",".join(f"{d}={v}" for d, v in selections.items())
    if circular:
        name += f"|{circular}"
    return f"{name}|{DERIVATIVE_SUFFIX}" if derivative else name


def parse_column_name(name: str) -> FeatureColumn:
    """The inverse of :func:`column_name`: what a materialised column name selects."""
    feature, *parts = name.split("|")
    derivative = bool(parts) and parts[-1] == DERIVATIVE_SUFFIX
    if derivative:
        parts.pop()
    circular = parts.pop() if parts and parts[-1] in CIRCULAR_COMPONENTS else None
    if len(parts) > 1:
        raise ValueError(f"Column name {name!r} is not one column_name() writes")
    selections = dict(pair.split("=", 1) for pair in parts[0].split(",")) if parts else {}
    return FeatureColumn(feature, selections, name, derivative, circular)


def time_derivative(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    """``d(values)/dt`` by second-order central differences (``np.gradient``).

    Centred on the sample, so a turn in the signal shows up in the derivative
    *at* the frame it happened rather than half a frame late — which is what a
    one-sided difference would do, and the whole point of these columns is to
    time an event. *time* is passed through, so non-uniform sampling is handled
    and the units are the feature's own per second.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        raise ValueError("Need at least 2 samples to take a time derivative.")
    return np.gradient(values, np.asarray(time, dtype=np.float64))


def parse_angle_units(declared: Any, what: str) -> str | None:
    """*declared* — a ``units`` attr — as :data:`RADIANS` / :data:`DEGREES`.

    ``None`` when the variable says nothing, which leaves the units to
    :func:`angle_units`. A unit that is not angular is an error: it says this
    column is not an angle, and encoding it as one would silently feed the
    model the sine of a distance.
    """
    if declared is None:
        return None
    spelling = str(declared).strip().lower().rstrip(".")
    if not spelling:
        return None
    if spelling in _UNIT_SPELLINGS:
        return _UNIT_SPELLINGS[spelling]
    raise ValueError(
        f"{what} is configured as an angle (sin/cos), but the dataset declares "
        f"units={declared!r} — expected one of {sorted(_UNIT_SPELLINGS)}."
    )


def angle_units(values: np.ndarray, declared: Any = None, what: str = "an angle") -> str:
    """Whether *values* are radians or degrees.

    The variable's own ``units`` attr decides where it has one; otherwise the
    values do, and the decision is logged. The two scales are a factor of ~57
    apart, so the reading is only ambiguous for an angle that never leaves
    ``±2pi`` — a signal that spends its life within a third of a degree of
    zero, which is not an angle anyone segments behaviour on. Declare
    ``units`` on the variable to settle it.
    """
    unit = parse_angle_units(declared, what)
    if unit is not None:
        return unit
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    unit = DEGREES if peak > _RADIAN_CEILING else RADIANS
    return unit


def sin_cos_values(values: np.ndarray, units: str) -> tuple[np.ndarray, np.ndarray]:
    """*values* as ``(sin, cos)``, both in ``[-1, 1]``.

    NaNs pass through as NaNs — a gap in the angle stays a gap, for the
    interpolation step to fill rather than this one to invent.
    """
    radians = np.asarray(values, dtype=np.float64)
    if units == DEGREES:
        radians = np.deg2rad(radians)
    elif units != RADIANS:
        raise ValueError(f"Unknown angle units {units!r}; expected {RADIANS!r} or {DEGREES!r}.")
    return np.sin(radians), np.cos(radians)


def expand_column(col: FeatureColumn, derivative: bool = False, circular: bool = False) -> list[FeatureColumn]:
    """The columns one pinned series contributes, in layout order.

    The one definition of that order, read by :func:`enumerate_columns` and by
    :func:`extract_features`, so the names and the numbers cannot disagree: an
    angle becomes ``sin`` then ``cos``, and a derivative follows each value it
    is taken from (the derivative of the *components*, never of the angle
    itself — that one jumps by ``2pi`` at the wrap).
    """
    parts = (
        [
            col._replace(name=column_name(col.feature, col.selections, circular=c), circular=c)
            for c in CIRCULAR_COMPONENTS
        ]
        if circular
        else [col]
    )
    out: list[FeatureColumn] = []
    for part in parts:
        out.append(part)
        if derivative:
            out.append(
                part._replace(
                    name=column_name(part.feature, part.selections, True, part.circular),
                    derivative=True,
                )
            )
    return out


def expand_dim_values(values: Any) -> list[str]:
    """A dim's configured value(s) as an explicit ``list[str]``.

    Accepts the literal list (unchanged, values coerced to ``str``) or a
    compact inclusive range for a purely numeric dim, written ``"a..b"``
    (e.g. ``s3d_dims: 0..1023`` instead of spelling out 1024 indices) — sugar
    for the same explicit list, expanded here and nowhere else. This is not
    "all": the range is a closed, session-independent literal, exactly like
    writing every value out by hand.
    """
    if isinstance(values, str):
        m = _RANGE_RE.match(values.strip())
        if m:
            start, stop = int(m.group(1)), int(m.group(2))
            return [str(i) for i in range(start, stop + 1)]
        return [values]
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return [str(values)]


def enumerate_columns(
    features: dict[str, dict[str, Any]],
    derivatives: Iterable[str] | None = None,
    sin_cos: Iterable[str] | None = None,
) -> list[FeatureColumn]:
    """Expand the config's per-dim value lists into pinned columns.

    Order is deterministic (config order, then the cartesian product in the
    values' stored order) — it defines the model's input layout. A feature
    named in *derivatives* contributes a second column per combination, its
    time derivative, directly after the value it is taken from; a feature
    named in *sin_cos* is replaced by the two components of its angle
    (:func:`expand_column`).
    """
    wants_derivative = set(derivatives or ())
    wants_sin_cos = set(sin_cos or ())
    columns: list[FeatureColumn] = []
    for feature, dims in features.items():
        dim_names = list(dims)
        value_lists = [expand_dim_values(dims[d]) for d in dim_names]
        for combo in itertools.product(*value_lists):
            selections = dict(zip(dim_names, combo))
            col = FeatureColumn(feature, selections, column_name(feature, selections))
            columns.extend(expand_column(col, feature in wants_derivative, feature in wants_sin_cos))
    return columns


def sampling_rate(time: np.ndarray) -> float:
    """Sampling rate implied by a time vector (median spacing)."""
    time = np.asarray(time, dtype=np.float64)
    if time.size < 2:
        raise ValueError("Need at least 2 samples to determine a sampling rate.")
    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        raise ValueError("Time vector is not increasing.")
    return 1.0 / dt


def check_same_fs(fs_ref: float, fs: float, what: str) -> None:
    if not np.isclose(fs_ref, fs, rtol=FS_RTOL):
        raise ValueError(
            f"Sampling-rate mismatch: {what} runs at {fs:.6g} Hz but the model's "
            f"features run at {fs_ref:.6g} Hz. All selected features must share "
            "one sampling rate."
        )


def extract_features(
    loader: Any,
    features: dict[str, dict[str, list[str]]],
    t0: float | None = None,
    t1: float | None = None,
    derivatives: Iterable[str] | None = None,
    sin_cos: Iterable[str] | None = None,
    units: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select every configured column over ``[t0, t1]`` and stack to ``(T, D)``.

    *loader* is any :class:`~ethograph.io.catalog.DataLoader`; times are in the
    loader's native clock. Columns come out in :func:`enumerate_columns` order,
    a feature named in *derivatives* contributing its :func:`time_derivative`
    right after each value it is taken from and a feature named in *sin_cos*
    arriving as its two angle components. *units* is the ``units`` attr of the
    features that declare one (feature → attr), read only for *sin_cos*; a
    feature missing from it has its units read off its values
    (:func:`angle_units`). Raises ``ValueError`` when a feature is missing, a
    selection does not pin down to one column, or sampling rates differ.
    """
    wants_derivative = set(derivatives or ())
    wants_sin_cos = set(sin_cos or ())
    declared = dict(units or {})
    missing = wants_sin_cos - set(features)
    if missing:
        raise ValueError(f"sin_cos names {sorted(missing)}, which is not among the selected features.")
    columns = enumerate_columns(features)
    if not columns:
        raise ValueError("The model config selects no features.")

    time_ref: np.ndarray | None = None
    fs_ref = 0.0
    arrays: list[np.ndarray] = []
    for col in columns:
        plot_data = loader.select(col.feature, col.selections, t0, t1)
        if plot_data is None:
            raise ValueError(f"Feature {col.feature!r} is not available in this session.")
        data = np.asarray(plot_data.data, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError(
                f"Column {col.name!r} did not pin down to a single series "
                f"(got shape {data.shape}) — the dataset has a dim the model "
                "config does not cover. Recreate the model on this dataset."
            )
        time = np.asarray(plot_data.time, dtype=np.float64)
        fs = sampling_rate(time)
        if time_ref is None:
            time_ref, fs_ref = time, fs
        else:
            check_same_fs(fs_ref, fs, f"feature {col.feature!r}")
            if abs(float(time[0]) - float(time_ref[0])) > 0.5 / fs_ref:
                raise ValueError(
                    f"Feature {col.feature!r} starts {abs(time[0] - time_ref[0]):.4g} s "
                    "away from the other features — their samples cannot be aligned."
                )
        if col.feature in wants_sin_cos:
            values = list(sin_cos_values(data, angle_units(data, declared.get(col.feature), f"Column {col.name!r}")))
        else:
            values = [data]
        for series in values:
            arrays.append(series)
            if col.feature in wants_derivative:
                arrays.append(time_derivative(series, time))

    assert time_ref is not None
    n = min(len(time_ref), *(len(a) for a in arrays))
    stacked = np.column_stack([a[:n] for a in arrays])
    return time_ref[:n], stacked


def _compact_dim_value(values: list[str]) -> Any:
    """*values* collapsed to ``"start..stop"`` when it is exactly that
    ascending contiguous digit run (see :func:`expand_dim_values`); otherwise
    the explicit list, unchanged."""
    if len(values) >= 2 and all(v.isdigit() for v in values):
        start = int(values[0])
        if values == [str(i) for i in range(start, start + len(values))]:
            return f"{start}..{start + len(values) - 1}"
    return list(values)


def columns_to_yaml(columns: dict[str, dict[str, list[str]]], *, indent: int = 2) -> str:
    """Render a ``features.columns`` dict as a paste-ready block, ``columns:``
    heading included, one feature per line in the project config's inline-dict
    style (e.g. ``position: {space: [x, y], keypoint: [beak, tail]}``).

    A dim whose values are one ascending contiguous run of digits (e.g. a
    video-feature's numbered channels) is written as the compact ``a..b``
    range :func:`expand_dim_values` reads back, instead of the full list.

    *indent* is the ``columns:`` heading's own indent — 2 to paste directly
    under a top-level ``features:`` key, as in ``configs/project.yaml``.
    """
    import yaml

    pad = " " * indent
    lines = [f"{pad}columns:"]
    for feature, dims in columns.items():
        compact = {dim: _compact_dim_value(values) for dim, values in dims.items()}
        inline = yaml.safe_dump(compact, default_flow_style=True, sort_keys=False).strip()
        lines.append(f"{pad}  {feature}: {inline}")
    return "\n".join(lines) + "\n"
