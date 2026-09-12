"""Unified combo catalog and loader for xarray, pynapple backends.

Replaces the old type_vars_dict pattern with:
- DataCatalog: what dimensions/features are available (builds combo boxes)
- DataLoader: how to load data (select by feature + combo dims + time window → PlotData)

Three backends, same interface. Differs in how combo dims are discovered,
how selection works (sel_valid principle: overspecified combos are OK), and
how time slicing works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

import numpy as np

from ethograph.io import schema

if TYPE_CHECKING:
    import xarray as xr

    from ethograph.io.trialtree import TrialTree


#: Dim spellings that select an individual animal, most-preferred first.
#: movement (v0.17+) is singular; datasets built by the older wizard wrote the
#: plural. **A combo is always named after the dim it selects from** — neither
#: spelling is renamed to the other, so a selection key is always a real dim
#: and ``.sel()`` needs no translation. Renaming one of them is what let a
#: panel's selections look complete to ``_sanitize_selections`` while leaving
#: a dim free in the loader.
INDIVIDUAL_DIMS = ("individual", "individuals")

#: Dim spellings that select a keypoint, most-preferred first — movement's
#: singular and the older plural, read as they are (see ``INDIVIDUAL_DIMS``).
KEYPOINT_DIMS = ("keypoint", "keypoints")

#: movement's name for the x/y/z axis. Pynapple columns spelling exactly that
#: get the same dim name, so a space plot, a panel combo and a saved selection
#: mean the same thing whichever backend the session came from.
SPACE_DIM = "space"
_SPACE_COLUMNS = frozenset({"x", "y", "z"})


# ---------------------------------------------------------------------------
# PlotData — universal rendering output: (T,) or (T, D)
# ---------------------------------------------------------------------------


@dataclass
class PlotData:
    """Source-agnostic data ready for rendering."""

    time: np.ndarray
    data: np.ndarray  # (T,) or (T, D)
    dim_labels: list[str] | None = None
    title: str = ""
    ylabel: str = ""
    color_data: np.ndarray | None = None
    changepoints: dict[str, np.ndarray] | None = None


# ---------------------------------------------------------------------------
# ColumnAxis — a pynapple feature's non-time axis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ColumnAxis:
    """The dim a pynapple feature's columns are selected by, plus its labels.

    The pynapple counterpart of an xarray dim and its coord: ``dim`` is the
    combo/selection key, ``labels`` are the raw column labels in data order.
    """

    dim: str
    labels: tuple[Any, ...]

    def match(self, value: Any) -> Any | None:
        """The raw column label *value* names, or ``None`` if it names none.

        ``feature_dims()`` stringifies labels for the combo UI, so a numeric
        column comes back as ``"0"``/``"43"`` — the same coercion
        :func:`_selections_for_var` does on the xarray side, in the one
        direction pynapple needs it.
        """
        wanted = str(value)
        return next((c for c in self.labels if str(c) == wanted), None)

    def index(self, label: Any) -> int:
        """Positional index of *label* — how every backend slices a column."""
        return list(self.labels).index(label)


# ---------------------------------------------------------------------------
# ComboSpec + DataCatalog — what's available
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComboSpec:
    """A selectable dimension: name + allowed values."""

    name: str
    values: tuple[str, ...]


@dataclass
class DataCatalog:
    """Unified catalog of available features, dimensions, and streams.

    Built by ``catalog_from_xarray`` / ``catalog_from_pynapple`` /
    ``catalog_from_nwb``.  The GUI creates combo boxes from ``combos``
    and uses the loader returned alongside this catalog for data access.
    """

    combos: dict[str, ComboSpec] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    changepoints: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    mics: list[str] = field(default_factory=list)
    trial_conditions: list[str] = field(default_factory=list)

    def combo_values(self, name: str) -> tuple[str, ...]:
        spec = self.combos.get(name)
        return spec.values if spec else ()

    @property
    def individual_combo(self) -> str | None:
        """Name of the combo selecting an individual, in this dataset's own
        spelling — ``None`` when it has no individual dim. Ask for it rather
        than hardcoding either spelling."""
        return next((n for n in INDIVIDUAL_DIMS if n in self.combos), None)

    def feature_choices(self) -> list[str]:
        """The canonical GUI feature list — the SINGLE source used by the
        features combo, the add-panel popup, and panel creation, so a feature
        offered anywhere is displayable everywhere."""
        spec = self.combos.get("features")
        if spec is not None and spec.values:
            return [str(v) for v in spec.values]
        return [str(f) for f in self.features]

    def to_type_vars_dict(self) -> dict:
        """Backwards-compatible dict for existing GUI combo creation."""
        tvd: dict[str, Any] = {}
        for name, spec in self.combos.items():
            tvd[name] = np.array(spec.values) if spec.values else []
        tvd["features"] = self.features
        if self.changepoints:
            tvd["changepoints"] = self.changepoints
        if self.cameras:
            tvd["cameras"] = np.array(self.cameras)
        if self.mics:
            tvd["mics"] = np.array(self.mics)
        tvd["trial_conditions"] = self.trial_conditions
        return tvd


# ---------------------------------------------------------------------------
# NWB data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimeSeriesRecord:
    """Metadata for one TimeSeries discovered inside an NWB file."""

    path: str
    name: str
    neurodata_type: str
    description: str
    shape: tuple[int, ...]
    dtype: str
    unit: str
    rate: float | None
    starting_time: float | None
    n_timestamps: int | None
    timestamps_range: tuple[float, float] | None

    @property
    def duration(self) -> float | None:
        if self.rate and self.starting_time is not None:
            return self.shape[0] / self.rate
        if self.timestamps_range:
            return self.timestamps_range[1] - self.timestamps_range[0]
        return None

    @property
    def time_range(self) -> tuple[float, float] | None:
        if self.timestamps_range:
            return self.timestamps_range
        if self.rate and self.starting_time is not None:
            end = self.starting_time + self.shape[0] / self.rate
            return (self.starting_time, end)
        return None

    @property
    def is_regularly_sampled(self) -> bool:
        return self.rate is not None

    def time_to_slice(self, t_start: float, t_stop: float) -> slice:
        if self.rate and self.starting_time is not None:
            i0 = int((t_start - self.starting_time) * self.rate)
            i1 = int((t_stop - self.starting_time) * self.rate)
            return slice(max(i0, 0), min(i1, self.shape[0]))
        raise ValueError("Irregular timestamps: use np.searchsorted on the timestamps dataset")


@dataclass(frozen=True, slots=True)
class TimeIntervalsRecord:
    """Metadata for one TimeIntervals table discovered inside an NWB file."""

    path: str
    name: str
    neurodata_type: str
    description: str
    n_rows: int
    column_names: tuple[str, ...]
    time_range: tuple[float, float] | None


# ---------------------------------------------------------------------------
# DataLoader protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DataLoader(Protocol):
    """Backend-agnostic data access."""

    @property
    def backend(self) -> str: ...

    @property
    def catalog(self) -> DataCatalog: ...

    def select(
        self,
        feature: str,
        selections: dict[str, str],
        t0: float,
        t1: float,
        color_variable: str | None = None,
    ) -> PlotData | None: ...

    def feature_dims(self, feature: str) -> dict[str, list[str]]: ...

    # Convenience properties — delegate to catalog
    @property
    def features(self) -> list[str]: ...

    @property
    def dims(self) -> dict[str, np.ndarray]: ...

    @property
    def changepoint_names(self) -> list[str]: ...

    def get_type_vars(self) -> dict: ...


# ---------------------------------------------------------------------------
# Shared mixin: catalog-backed convenience properties
# ---------------------------------------------------------------------------


class _CatalogMixin:
    """Delegates metadata properties to a DataCatalog."""

    _catalog: DataCatalog

    @property
    def catalog(self) -> DataCatalog:
        return self._catalog

    @property
    def features(self) -> list[str]:
        return self._catalog.features

    @property
    def dims(self) -> dict[str, np.ndarray]:
        return {n: np.array(s.values) for n, s in self._catalog.combos.items() if n != "features"}

    @property
    def changepoint_names(self) -> list[str]:
        return self._catalog.changepoints

    def get_type_vars(self) -> dict:
        return self._catalog.to_type_vars_dict()


# ---------------------------------------------------------------------------
# XarrayLoader
# ---------------------------------------------------------------------------


def _selections_for_var(selections: dict[str, str], var: xr.DataArray) -> dict:
    """Adapt combo-named *selections* to the dims of *var*.

    A selection key **is** a dim name or it is ignored — combos are named after
    their dim, so there is nothing to translate. Resist adding a plural→singular
    fallback here: ``PanelStateMixin._sanitize_selections`` decides which dims
    are still free on exactly these terms, and any rule applied on one side but
    not the other leaves a dim free that the other thinks is pinned, which is
    how ``sel_valid`` ends up with more than ``(time,)``/``(time, dim)``. A
    stale plural key from an older settings file is simply inert.

    The one real adjustment: ``feature_dims()`` stringifies coord labels for the
    combo UI, so a numeric coordinate (``component=0``, ``unit=43``) arrives as
    ``"0"``/``"43"``. Coerce back to the coordinate's dtype so ``.sel()`` matches.
    """
    out = dict(selections)
    for dim, val in list(out.items()):
        if isinstance(val, str) and dim in var.coords and var.coords[dim].dtype.kind in "iuf":
            try:
                out[dim] = var.coords[dim].dtype.type(val)
            except (ValueError, TypeError):
                pass
    return out


class XarrayLoader(_CatalogMixin):
    """Feature access from an ``xr.Dataset``.  Uses ``sel_valid`` for selection.

    The dataset's time coord is trial-local (0-based). Like
    :class:`PynappleLoader`, a display-offset provider bridges to the plot
    axis: it returns the shift from display time to the loader's native
    clock, so in session basis (axis session-absolute) the provider returns
    ``-trial_start`` and the current trial renders at its true session
    position — previously session scope silently selected nothing for any
    trial not starting near 0.
    """

    def __init__(self, ds: xr.Dataset, catalog: DataCatalog | None = None) -> None:
        self._ds = ds
        if catalog is None:
            catalog = _auto_catalog_xarray(ds)
        self._catalog = catalog
        self._display_offset_provider: Callable[[], float] | None = None

    def set_display_offset_provider(self, provider: Callable[[], float] | None) -> None:
        """Install the callable mapping display time to native (trial-local) time."""
        self._display_offset_provider = provider

    def display_offset(self) -> float:
        """Current display→native offset in seconds (0.0 without a provider)."""
        if self._display_offset_provider is None:
            return 0.0
        return float(self._display_offset_provider())

    @property
    def backend(self) -> str:
        return "xarray"

    def update_ds(self, ds: xr.Dataset) -> None:
        """Swap the backing dataset (called on trial change)."""
        self._ds = ds

    def feature_dims(self, feature: str) -> dict[str, list[str]]:
        if self._ds is None or feature not in self._ds.data_vars:
            return {}
        var = self._ds[feature]
        result: dict[str, list[str]] = {}
        for d in var.dims:
            if "time" in d.lower():
                continue
            if d in var.coords:
                result[d] = [str(v) for v in var.coords[d].values]
            else:
                result[d] = [str(i) for i in range(self._ds.sizes[d])]
        return result

    def select(
        self,
        feature: str,
        selections: dict[str, str],
        t0: float | None = None,
        t1: float | None = None,
        color_variable: str | None = None,
    ) -> PlotData | None:
        import ethograph as eto

        ds = self._ds
        if ds is None or feature not in ds.data_vars:
            return None

        var = ds[feature]
        time_coord = eto.get_time_coord(var)
        if time_coord is None:
            return None

        # Shift the display-time query into the trial-local coord; the
        # returned PlotData is shifted back so it lands on the axis drawn.
        offset = self.display_offset()
        if t0 is not None and t1 is not None:
            ds = ds.sel({time_coord.name: slice(t0 + offset, t1 + offset)})
            var = ds[feature]

        time = eto.get_time_coord(var).values
        _sel = _selections_for_var(selections, var)
        data, filt_kwargs = eto.sel_valid(var, _sel)
        var_sel = var.sel(**filt_kwargs)

        dim_labels = None
        if data.ndim == 2:
            non_time_dim = next((d for d in var_sel.dims if "time" not in d.lower()), None)
            if non_time_dim and non_time_dim in var_sel.coords:
                dim_labels = [str(c) for c in var_sel.coords[non_time_dim].values]
            else:
                dim_labels = [str(i) for i in range(data.shape[1])]

        color_data = None
        if data.ndim == 1 and color_variable and color_variable in ds.data_vars:
            color_var = ds[color_variable]
            color_kwargs = {k: v for k, v in _selections_for_var(selections, color_var).items() if k != "RGB"}
            # The panel's selections are sanitized against its FEATURE's dims
            # only, so a dim the colour var alone carries reaches here unjudged:
            # free (blowing sel_valid's (time,)/(time, D) shape assertion) or
            # pinned to a value from another dataset (a KeyError out of .sel).
            # Both are settled the way _sanitize_selections settles extra feature
            # dims — pin to the first value. A value the colour var does not have
            # is stale whatever the dim's size: a session with a single individual
            # is exactly where another dataset's name lands.
            for d in color_var.dims:
                if "time" in str(d).lower() or d == "RGB":
                    continue
                if d in color_var.coords:
                    coord_vals = color_var.coords[d].values
                    stale = d in color_kwargs and color_kwargs[d] not in coord_vals
                    if stale or (d not in color_kwargs and color_var.sizes[d] > 1):
                        color_kwargs[d] = coord_vals[0]
                elif d not in color_kwargs and color_var.sizes[d] > 1:
                    color_kwargs[d] = 0
            color_data, _ = eto.sel_valid(color_var, color_kwargs)

        changepoints = None
        if data.ndim == 1:
            # The same reading a click snaps to (get_cp_times): the masks
            # targeting this feature, at this panel's selections.
            from ethograph.features.changepoints import changepoint_fired

            cp_dict: dict[str, np.ndarray] = {
                cp_name: changepoint_fired(ds[cp_name], selections)
                for cp_name in schema.changepoint_vars(ds)
                if ds[cp_name].attrs.get("target_feature") == feature
            }
            if cp_dict:
                changepoints = cp_dict

        ylabel = var.attrs.get("ylabel", feature)
        title = feature

        return _shift_plot_time(
            PlotData(
                time=time,
                data=data,
                dim_labels=dim_labels,
                title=title,
                ylabel=ylabel,
                color_data=color_data,
                changepoints=changepoints,
            ),
            offset,
        )

    def time_range(self, feature: str | None = None) -> tuple[float, float]:
        import ethograph as eto

        if self._ds is None:
            return (0.0, 0.0)
        if feature and feature in self._ds.data_vars:
            tc = eto.get_time_coord(self._ds[feature])
        else:
            tc = None
            for var in self._ds.data_vars.values():
                tc = eto.get_time_coord(var)
                if tc is not None:
                    break
        if tc is None:
            return (0.0, 0.0)
        vals = tc.values
        return (float(vals[0]), float(vals[-1]))

    def get_cp_times(
        self,
        feature: str | None = None,
        selections: dict[str, Any] | None = None,
        t0: float | None = None,
        t1: float | None = None,
    ) -> np.ndarray:
        """Changepoint times of *feature*'s masks at *selections*, in display coordinates.

        Exactly the marks :meth:`select` draws for that feature at those
        selections — both read :func:`~ethograph.features.changepoints.changepoint_fired`.
        ``t0``/``t1`` (display clock) restrict the answer to a window.
        """
        from ethograph.features.changepoints import dataset_changepoint_times

        if self._ds is None:
            return np.array([], dtype=np.float64)
        offset = self.display_offset()
        cp_times = dataset_changepoint_times(self._ds, feature, selections) - offset
        if t0 is not None and t1 is not None:
            cp_times = cp_times[(cp_times >= t0) & (cp_times <= t1)]
        return cp_times


# ---------------------------------------------------------------------------
# PynappleLoader
# ---------------------------------------------------------------------------


class PynappleLoader(_CatalogMixin):
    """Feature access backed by raw pynapple objects, rendered in display time.

    Pynapple objects live in absolute session time, but callers pass ``t0, t1``
    in the plot's x-axis coordinates — which are trial-local whenever the GUI's
    window is trial-based (trial / label / sequence scope) and absolute only in
    session scope.  The bridge is ``display_offset``: queries are shifted into
    absolute time before ``restrict()`` and returned times are shifted back, so
    the PlotData always lands on the axis the caller drew.  The offset is
    *pulled* per call from a provider installed by the GUI (the loader itself
    stays free of trial state); with no provider it is 0.0 and the loader
    behaves as pure absolute-time access.
    """

    def __init__(
        self,
        data: dict,
        catalog: DataCatalog | None = None,
    ) -> None:
        import pynapple as nap

        self._data = data
        if catalog is None:
            catalog = catalog_from_pynapple(data)
        self._catalog = catalog

        self._feature_objs: dict = {
            k: v for k, v in data.items() if isinstance(v, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))
        }
        self._axes = _column_axes(self._feature_objs)
        self._display_offset_provider: Callable[[], float] | None = None

    def set_display_offset_provider(self, provider: Callable[[], float] | None) -> None:
        """Install the callable that maps display time to absolute time.

        The provider returns the amount to ADD to a caller's ``t0/t1`` to reach
        absolute session time (a trial's session start when the window is
        trial-local, 0.0 in session scope).
        """
        self._display_offset_provider = provider

    def display_offset(self) -> float:
        """Current display→absolute offset in seconds (0.0 without a provider)."""
        if self._display_offset_provider is None:
            return 0.0
        return float(self._display_offset_provider())

    @property
    def backend(self) -> str:
        return "pynapple"

    @property
    def data(self) -> dict:
        """The raw pynapple objects — lets callers build a second loader over
        the same session (e.g. an offset-free one for whole-session sweeps)."""
        return self._data

    def _keypoint_names(self) -> list[str]:
        """The keypoints backing the synthetic ``pose_estimation`` feature."""
        kp_spec = self._catalog.combos.get("keypoint") if self._catalog else None
        return list(kp_spec.values) if kp_spec and kp_spec.values else []

    def _column_axis(self, feature: str) -> ColumnAxis | None:
        """The axis *feature* is selected on — the keypoints' own for pose."""
        if feature == "pose_estimation":
            keypoints = self._keypoint_names()
            return self._axes.get(keypoints[0]) if keypoints else None
        return self._axes.get(feature)

    def _pinned_column(self, feature: str, selections: dict[str, str]) -> Any | None:
        """The column *selections* pins for *feature*, or ``None`` if free."""
        return _pinned_column(self._column_axis(feature), selections)

    def feature_dims(self, feature: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        # Columns first — _build_axis_items expands the first dim into axis
        # items (pose_estimation · x, etc.)
        axis = self._column_axis(feature)
        if axis is not None:
            result[axis.dim] = [str(c) for c in axis.labels]
        if feature == "pose_estimation":
            # Keypoints second — picked up by _populate_keypoint_combo
            keypoints = self._keypoint_names()
            if not keypoints:
                return {}
            result["keypoint"] = keypoints
        return result

    def _restrict(self, obj: Any, t0: float, t1: float):
        """Restrict to ``[t0, t1]`` (absolute session times)."""
        import pynapple as nap

        return obj.restrict(nap.IntervalSet(start=t0, end=t1))

    def _select_all_keypoints(
        self,
        selections: dict[str, str],
        t0: float,
        t1: float,
    ) -> PlotData | None:
        """Stack all keypoints into one (T, N) array, sliced by the pinned column."""
        keypoints = self._keypoint_names()
        if not keypoints:
            return None

        arrays: list[np.ndarray] = []
        labels: list[str] = []
        time = None

        for kp_name in keypoints:
            obj = self._feature_objs.get(kp_name)
            if obj is None:
                continue
            obj = self._restrict(obj, t0, t1)
            if len(obj) == 0:
                continue
            if time is None:
                time = obj.t

            axis = self._axes.get(kp_name)
            if axis is None:
                arrays.append(obj.values)
                labels.append(kp_name)
                continue

            column = _pinned_column(axis, selections)
            values = _flat_values(obj, len(obj))
            if column is not None:
                arrays.append(values[:, axis.index(column)])
                labels.append(kp_name)
            else:
                for i, c in enumerate(axis.labels):
                    arrays.append(values[:, i])
                    labels.append(f"{kp_name}_{c}")

        if not arrays or time is None:
            return None

        stacked = np.column_stack(arrays) if len(arrays) > 1 else arrays[0]
        dim_labels = labels if stacked.ndim == 2 else None

        return PlotData(
            time=time,
            data=stacked,
            dim_labels=dim_labels,
            title="pose_estimation",
            ylabel="pose_estimation",
        )

    def select(
        self,
        feature: str,
        selections: dict[str, str],
        t0: float,
        t1: float,
        color_variable: str | None = None,
    ) -> PlotData | None:
        import pynapple as nap

        # Shift the display-time query into absolute session time; everything
        # below runs absolute, and the returned PlotData is shifted back so it
        # lands on the axis the caller drew.
        offset = self.display_offset()
        t0 += offset
        t1 += offset

        actual_key = feature
        if feature == "pose_estimation":
            kp = selections.get("keypoint")
            if kp and kp in self._feature_objs:
                actual_key = kp
            else:
                return _shift_plot_time(self._select_all_keypoints(selections, t0, t1), offset)

        if actual_key not in self._feature_objs:
            return None

        obj = self._feature_objs[actual_key]
        obj = self._restrict(obj, t0, t1)

        if len(obj) == 0:
            return None

        time = obj.t

        # --- extract numpy by the feature's own column dim, exactly as
        # `sel_valid` selects by dim on the xarray side ---
        axis = self._axes.get(actual_key)
        if axis is None:
            data = obj.values
            dim_labels = None
        else:
            column = _pinned_column(axis, selections)
            values = _flat_values(obj, len(time))
            if column is not None:
                data = values[:, axis.index(column)]
                dim_labels = None
            else:
                data = values
                dim_labels = [str(c) for c in axis.labels]

        # Color data
        color_data = None
        if data.ndim == 1 and color_variable and color_variable in self._feature_objs:
            color_obj = self._feature_objs[color_variable]
            color_obj = self._restrict(color_obj, t0, t1)
            if isinstance(color_obj, nap.TsdFrame) and len(color_obj) > 0:
                color_data = color_obj.values

        # Changepoints (1-D features only)
        changepoints = None
        if data.ndim == 1:
            cp_dict: dict[str, np.ndarray] = {}
            for key, raw_obj in self._data.items():
                if not isinstance(raw_obj, nap.TsGroup):
                    continue
                if not (hasattr(raw_obj, "metadata") and raw_obj.metadata is not None):
                    continue
                meta = raw_obj.metadata
                cp_units = schema.changepoint_units(meta)
                if not cp_units:
                    continue
                target = meta["target_feature"].iloc[0] if "target_feature" in meta.columns else None
                if target != feature:
                    continue
                for uid in cp_units:
                    ts_obj = raw_obj[uid]
                    ts_obj = self._restrict(ts_obj, t0, t1)
                    if len(ts_obj) == 0:
                        continue
                    if isinstance(ts_obj, nap.Tsd):
                        mask = ts_obj.values.astype(bool)
                        cp_times = ts_obj.t[mask]
                    else:
                        cp_times = ts_obj.t
                    if len(cp_times) == 0:
                        continue
                    binary = np.zeros(len(time), dtype=np.int8)
                    idxs = np.searchsorted(time, cp_times)
                    idxs = np.clip(idxs, 0, len(time) - 1)
                    binary[idxs] = 1
                    cp_dict[key] = binary
            if cp_dict:
                changepoints = cp_dict

        return _shift_plot_time(
            PlotData(
                time=time,
                data=data,
                dim_labels=dim_labels,
                title=feature,
                ylabel=feature,
                color_data=color_data,
                changepoints=changepoints,
            ),
            offset,
        )

    def get_cp_times(
        self,
        feature: str | None = None,
        selections: dict[str, Any] | None = None,
        t0: float | None = None,
        t1: float | None = None,
    ) -> np.ndarray:
        """Changepoint event times of *feature*'s ``TsGroup``s, in display coordinates.

        *selections* is accepted for parity with :class:`XarrayLoader` — a
        ``TsGroup`` has no keypoint/individual dims to pin. ``t0``/``t1``
        (display clock) restrict the answer to a window; without them the
        whole session is returned.
        """
        import pynapple as nap

        offset = self.display_offset()
        windowed = t0 is not None and t1 is not None
        if windowed:
            t0 += offset
            t1 += offset

        all_times: list[np.ndarray] = []
        for key, obj in self._data.items():
            if not isinstance(obj, nap.TsGroup):
                continue
            if not (hasattr(obj, "metadata") and obj.metadata is not None):
                continue
            meta = obj.metadata
            cp_units = schema.changepoint_units(meta)
            if not cp_units:
                continue
            if feature and "target_feature" in meta.columns:
                if meta["target_feature"].iloc[0] != feature:
                    continue
            for uid in cp_units:
                ts_obj = obj[uid]
                if windowed:
                    ts_obj = self._restrict(ts_obj, t0, t1)
                if len(ts_obj) == 0:
                    continue
                if isinstance(ts_obj, nap.Tsd):
                    mask = ts_obj.values.astype(bool)
                    all_times.append(ts_obj.t[mask])
                else:
                    all_times.append(ts_obj.t)
        if not all_times:
            return np.array([], dtype=np.float64)
        return np.unique(np.concatenate(all_times)).astype(np.float64) - offset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shift_plot_time(plot_data: PlotData | None, offset: float) -> PlotData | None:
    """Rebase a PlotData's time axis from absolute back to display coordinates."""
    if plot_data is not None and offset:
        plot_data.time = np.asarray(plot_data.time, dtype=np.float64) - offset
    return plot_data


def _flat_values(obj: Any, n_time: int) -> np.ndarray:
    """A pynapple object's values as ``(T, D)`` — tensors flattened past time."""
    values = np.asarray(obj.values)
    if values.ndim == 1:
        return values.reshape(n_time, 1)
    if values.ndim > 2:
        return values.reshape(n_time, -1)
    return values


def _pinned_column(axis: ColumnAxis | None, selections: dict[str, Any]) -> Any | None:
    """The column *selections* pins on *axis*, or ``None`` if it is left free.

    A selection key **is** the axis's dim or it is inert — the rule
    :func:`_selections_for_var` and ``PanelStateMixin._sanitize_selections``
    already follow. Scanning every selection's *value* for one that happens to
    name a column (what this used to do) breaks it in both directions: a dim
    left on "All" stays pinned by an unrelated key's value, and a key for a
    dim the feature does not have silently selects.
    """
    if axis is None or axis.dim not in selections:
        return None
    return axis.match(selections[axis.dim])


def _column_axes(feature_objs: dict) -> dict[str, ColumnAxis]:
    """Map each multi-column pynapple feature to the axis it is selected on.

    This is the **single authority** for that dim's name: the catalog builds
    its combo from it, ``feature_dims()`` reports it, and ``select()`` reads
    the selection stored under it. Three separate namings (a ``_dim_map`` for
    ``select``, a hardcoded ``"space"`` for pose in ``feature_dims``, and the
    catalog's own rule) is how a panel ended up with a combo the loader never
    consulted — clicking "All" on it changed nothing.

    Naming follows xarray so a panel behaves the same on either backend:
    ``x``/``y``/``z`` columns are movement's ``space`` dim; other objects
    sharing a column tuple share one dim; a lone object gets
    ``{name}_columns``.
    """
    import pynapple as nap

    groups: dict[tuple, list[str]] = {}
    labels: dict[tuple, tuple] = {}
    for name, obj in feature_objs.items():
        if isinstance(obj, nap.TsdFrame):
            cols = tuple(obj.columns)
        elif isinstance(obj, nap.TsdTensor):
            # Rendering flattens everything past the time axis, so the axis the
            # user picks from is that flattened one.
            cols = tuple(str(i) for i in range(int(np.prod(obj.shape[1:]))))
        else:
            continue
        groups.setdefault(cols, []).append(name)
        labels[cols] = cols

    axes: dict[str, ColumnAxis] = {}
    used: set[str] = set()

    for cols, names in groups.items():
        if set(str(c) for c in cols) <= _SPACE_COLUMNS:
            dim_name = SPACE_DIM
        elif len(names) == 1:
            dim_name = f"{names[0]}_columns"
        else:
            dim_name = "columns"
        base, suffix = dim_name, 2
        while dim_name in used:
            dim_name = f"{base}_{suffix}"
            suffix += 1
        used.add(dim_name)
        axis = ColumnAxis(dim=dim_name, labels=labels[cols])
        for name in names:
            axes[name] = axis

    return axes


# Dimensions that are internal to color variables — never show as user combos
_HIDDEN_DIMS = frozenset({"RGB", "RGBA"})

# Variables that are never user-selectable features
_EXCLUDED_VARS = frozenset({"onset_s", "offset_s", "labels", "individual", "boundary_events"})


def _feature_vars(ds: xr.Dataset) -> list[str]:
    """All data_vars with a time dimension, minus changepoints and internal vars."""
    features = []
    for name, var in ds.data_vars.items():
        if name in _EXCLUDED_VARS:
            continue
        if schema.is_changepoint(var):
            continue
        if any("time" in str(d).lower() for d in var.dims):
            features.append(name)
    return features


def _auto_catalog_xarray(ds: xr.Dataset) -> DataCatalog:
    """Quick catalog from a Dataset when no TrialTree is available."""
    from ethograph.io.validation import find_temporal_dims

    combos: dict[str, ComboSpec] = {}

    # Named first so the individual leads the combo row; the name is the dim's
    # own, never renamed (see INDIVIDUAL_DIMS).
    _ind_dim = next((n for n in INDIVIDUAL_DIMS if n in ds.coords), None)
    if _ind_dim is not None:
        vals = tuple(ds.coords[_ind_dim].values.astype(str))
        combos[_ind_dim] = ComboSpec(_ind_dim, vals)

    features_list = _feature_vars(ds)
    changepoints_list = schema.changepoint_vars(ds)

    if features_list:
        combos["features"] = ComboSpec("features", tuple(features_list))

    for name in find_temporal_dims(ds):
        if name in combos or name.upper() in _HIDDEN_DIMS:
            continue
        if name in ds.coords:
            coord = ds.coords[name]
            if coord.dtype.kind in ("U", "S", "O"):
                vals = tuple(coord.values.astype(str))
            else:
                vals = tuple(str(v) for v in coord.values)
        else:
            vals = tuple(str(i) for i in range(ds.sizes[name]))
        combos[name] = ComboSpec(name, vals)

    return DataCatalog(
        combos=combos,
        features=features_list,
        changepoints=changepoints_list,
    )


# ---------------------------------------------------------------------------
# Catalog builders
# ---------------------------------------------------------------------------


def catalog_from_xarray(ds: xr.Dataset, dt: TrialTree, nwb_alignment=None) -> DataCatalog:
    """Build a DataCatalog from an xarray Dataset + TrialTree."""
    from ethograph.io.validation import (
        _possible_trial_conditions,
        find_temporal_dims,
    )

    combos: dict[str, ComboSpec] = {}

    # Named first so the individual leads the combo row; the name is the dim's
    # own, never renamed (see INDIVIDUAL_DIMS).
    _ind_dim = next((n for n in INDIVIDUAL_DIMS if n in ds.coords), None)
    if _ind_dim is not None:
        vals = tuple(ds.coords[_ind_dim].values.astype(str))
        combos[_ind_dim] = ComboSpec(_ind_dim, vals)

    features_list = _feature_vars(ds)
    changepoints_list = schema.changepoint_vars(ds)

    if features_list:
        combos["features"] = ComboSpec("features", tuple(features_list))

    extra_dims = find_temporal_dims(ds)
    for name in extra_dims:
        if name in combos or name.upper() in _HIDDEN_DIMS:
            continue
        if name in ds.coords:
            coord = ds.coords[name]
            if coord.dtype.kind in ("U", "S", "O"):
                vals = tuple(coord.values.astype(str))
            else:
                vals = tuple(str(v) for v in coord.values)
        else:
            vals = tuple(str(i) for i in range(ds.sizes[name]))
        combos[name] = ComboSpec(name, vals)

    sio = nwb_alignment or getattr(dt, "nwb_alignment", None)
    cameras = list(sio.cameras) if sio and sio.cameras else []
    mics = list(sio.mics) if sio and sio.mics else []
    trial_conditions = _possible_trial_conditions(ds, dt)

    return DataCatalog(
        combos=combos,
        features=features_list,
        changepoints=changepoints_list,
        cameras=cameras,
        mics=mics,
        trial_conditions=trial_conditions,
    )


def _find_pose_series_names(nwb_path: str | Path) -> list[str]:
    """Scan an NWB file with h5py and return PoseEstimationSeries leaf names."""
    import h5py

    hits: list[str] = []

    def _visit(name: str, obj: Any) -> None:
        ndt = obj.attrs.get("neurodata_type", "")
        if isinstance(ndt, bytes):
            ndt = ndt.decode()
        if ndt == "PoseEstimationSeries":
            hits.append(name.rsplit("/", 1)[-1])

    with h5py.File(nwb_path, "r") as f:
        f.visititems(_visit)
    return hits


def _discover_pose_keypoints(source_path: str | Path) -> set[str]:
    """Find PoseEstimationSeries names from NWB files at *source_path*.

    *source_path* can be a ``.nwb`` file, a ``.npz`` file (scans sibling
    NWB files), or a folder (scans contained NWB files).
    """
    p = Path(source_path)
    nwb_files: list[Path] = []
    if p.suffix == ".nwb" and p.is_file():
        nwb_files.append(p)
    elif p.is_dir():
        nwb_files.extend(p.glob("*.nwb"))
    elif p.is_file():
        nwb_files.extend(p.parent.glob("*.nwb"))

    keypoint_names: set[str] = set()
    for nwb in nwb_files:
        try:
            keypoint_names.update(_find_pose_series_names(nwb))
        except Exception:
            continue
    return keypoint_names


def catalog_from_pynapple(
    data: dict,
    *,
    source_path: str | Path | None = None,
) -> DataCatalog:
    """Build a DataCatalog from pynapple objects.

    Parameters
    ----------
    source_path
        Path to the original data source (``.nwb``, ``.npz``, or folder).
        When provided, NWB files are scanned for ``PoseEstimationSeries``
        to create a ``keypoint`` combo from matching pynapple keys.
    """
    import pynapple as nap

    pose_names: set[str] = set()
    if source_path is not None:
        pose_names = _discover_pose_keypoints(source_path)

    feature_objs = {k: v for k, v in data.items() if isinstance(v, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))}
    axes = _column_axes(feature_objs)

    combos: dict[str, ComboSpec] = {}
    combos["individual"] = ComboSpec("individual", ("individual_0",))

    def _add_column_combo(key: str) -> None:
        """Register the combo for *key*'s column axis, named as the loader
        names it — one authority, so every combo the sidebar shows is one
        ``select()`` actually reads."""
        axis = axes.get(key)
        if axis is not None and axis.dim not in combos:
            combos[axis.dim] = ComboSpec(axis.dim, tuple(str(c) for c in axis.labels))

    features: list[str] = []
    changepoints: list[str] = []
    keypoint_names: list[str] = []

    for key, obj in data.items():
        if isinstance(obj, nap.IntervalSet):
            continue

        if isinstance(obj, nap.TsGroup):
            if hasattr(obj, "metadata") and obj.metadata is not None:
                if schema.changepoint_units(obj.metadata):
                    changepoints.append(key)
            continue

        if isinstance(obj, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
            if key in pose_names:
                keypoint_names.append(key)
            else:
                features.append(key)
                _add_column_combo(key)

    if keypoint_names:
        keypoint_names = sorted(keypoint_names)
        combos["keypoint"] = ComboSpec("keypoint", tuple(keypoint_names))
        features.insert(0, "pose_estimation")
        # The keypoints' columns are an axis like any other — and when a plain
        # feature carries the same ones (x/y/z), it is literally the same axis,
        # so both share the one combo instead of fighting over the selection.
        _add_column_combo(keypoint_names[0])

    if features:
        combos["features"] = ComboSpec("features", tuple(features))

    return DataCatalog(
        combos=combos,
        features=features,
        changepoints=changepoints,
        trial_conditions=[],
    )
