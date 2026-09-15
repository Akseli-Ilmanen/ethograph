from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pynapple as nap
import xarray as xr

from ethograph.io.nwb_alignment import (
    EmpytAlignment,
    discover_nwb,
    make_nwb_alignment,
)
from ethograph.io.validation import validate_datatree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attrs_equal(a: Any, b: Any) -> bool:
    try:
        result = a == b
        if isinstance(result, np.ndarray):
            return result.all()
        return bool(result)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# TrialTree
# ---------------------------------------------------------------------------


class TrialTree(xr.DataTree):
    def __init__(self, data=None, children=None, name=None):
        if isinstance(data, xr.DataTree):
            super().__init__(dataset=data.ds, children=children, name=name)
            for child_name, child_node in data.children.items():
                self[child_name] = child_node
        else:
            super().__init__(dataset=data, children=children, name=name)
        # Always initialize nwb_alignment to a null-object (empty EmpytAlignment)
        if not hasattr(self, "nwb_alignment"):
            self.nwb_alignment = EmpytAlignment()
        # Dirty-tracking for incremental saves (see `_try_incremental_save`).
        if not hasattr(self, "_dirty_trials"):
            self._dirty_trials: set[str] = set()
        if not hasattr(self, "_saved_node_names"):
            self._saved_node_names: frozenset[str] | None = None
        if not hasattr(self, "_saved_var_shapes"):
            self._saved_var_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
        if not hasattr(self, "_extra_file_handle"):
            self._extra_file_handle: xr.DataTree | None = None

    # ------------------------------------------------------------------
    # Node name resolution
    # ------------------------------------------------------------------

    def _trial_node_name(self, trial) -> str:
        trial_str = str(trial)
        if trial_str in self.children:
            return trial_str
        sanitized = trial_str.replace("/", "_")
        if sanitized != trial_str and sanitized in self.children:
            return sanitized
        legacy = f"trial_{trial_str}"
        if legacy in self.children:
            return legacy
        raise KeyError(f"No node found for trial {trial!r}")

    def _has_trial_node(self, trial) -> bool:
        trial_str = str(trial)
        sanitized = trial_str.replace("/", "_")
        return trial_str in self.children or sanitized in self.children or f"trial_{trial_str}" in self.children

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(self._trial_node_name(key))
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = self._trial_node_name(key) if self._has_trial_node(key) else str(key)
        if isinstance(value, xr.Dataset):
            value = xr.DataTree(value)
        super().__setitem__(key, value)

    # ------------------------------------------------------------------
    # Continuous mode support
    # ------------------------------------------------------------------

    @property
    def _is_continuous(self) -> bool:
        """True when backed by a single shared dataset + epoch boundaries."""
        return getattr(self, "_continuous_ds", None) is not None

    def _slice_continuous(self, trial_id) -> xr.Dataset:
        """Slice the continuous dataset for a single trial, shifting time to 0."""
        start, stop = self._trial_epochs[trial_id]
        ds = self._continuous_ds

        time_dims = [d for d in ds.dims if "time" in d.lower()]

        sel = {dim: slice(start, stop) for dim in time_dims}
        sliced = ds.sel(sel)

        for dim in time_dims:
            if dim in sliced.coords:
                sliced = sliced.assign_coords({dim: sliced.coords[dim].values - start})

        sliced.attrs = dict(ds.attrs)
        sliced.attrs["trial"] = trial_id
        return sliced

    # ------------------------------------------------------------------
    # Trial list & iteration
    # ------------------------------------------------------------------

    @property
    def trials(self) -> list[int | str]:
        """List of trial identifiers."""
        if self._is_continuous:
            return sorted(self._trial_epochs.keys())

        raw = [
            node.ds.attrs["trial"]
            for node in self.children.values()
            if node.ds is not None and "trial" in node.ds.attrs
        ]
        trials = [val.item() if hasattr(val, "item") else val for val in raw]
        if not trials:
            raise ValueError("No datasets with 'trial' attribute found in the tree.")
        return trials

    def trial_items(self):
        """Iterate over ``(trial_id, dataset)`` pairs for all trial nodes."""
        if self._is_continuous:
            for trial_id in sorted(self._trial_epochs.keys()):
                yield trial_id, self._slice_continuous(trial_id)
            return

        for node in self.children.values():
            if node.ds is not None and "trial" in node.ds.attrs:
                trial_id = node.ds.attrs["trial"]
                if hasattr(trial_id, "item"):
                    trial_id = trial_id.item()
                yield trial_id, node.ds

    def map_trials(self, func: Callable[[xr.Dataset], xr.Dataset]) -> TrialTree:
        """Apply *func* to every trial dataset and return a new TrialTree.

        For continuous trees, materialises sliced datasets into a
        standard per-node TrialTree so *func* results can be stored.
        """
        if self._is_continuous:
            datasets = [func(self._slice_continuous(t)) for t in sorted(self._trial_epochs.keys())]
            return TrialTree.from_datasets(datasets, validate=False)

        def _apply(ds):
            if ds is None or "trial" not in ds.attrs:
                return ds
            return func(ds)

        return self.from_datatree(
            self.map_over_datasets(_apply),
            attrs=self.attrs,
            source=self,
        )

    def update_trial(self, trial, func: Callable[[xr.Dataset], xr.Dataset]) -> None:
        """Read-modify-write a single trial's dataset.

        Not supported for continuous trees — call :meth:`materialise`
        first if you need per-trial mutation. Marks *trial* dirty so
        :meth:`save` can write just this trial back instead of the whole tree.
        """
        if self._is_continuous:
            raise TypeError("Cannot update trials in a continuous TrialTree. ")
        node_name = self._trial_node_name(trial)
        self[node_name] = xr.DataTree(func(self[node_name].ds))
        self._dirty_trials.add(node_name)

    def set_trial_attr(self, trial, key: str, value: Any) -> None:
        """Set a single attr on *trial*'s dataset, tracked for incremental save.

        Equivalent to mutating ``dt.trial(trial).attrs[key]`` directly, except
        that direct mutation bypasses dirty-tracking and forces a full rewrite
        on the next :meth:`save`.
        """

        def _set(ds: xr.Dataset) -> xr.Dataset:
            ds = ds.copy()
            ds.attrs[key] = value
            return ds

        self.update_trial(trial, _set)

    # ------------------------------------------------------------------
    # Trial data access
    # ------------------------------------------------------------------

    def trial(self, trial) -> xr.Dataset:
        """Return the dataset for the given trial ID.

        Parameters
        ----------
        trial : int or str
            Trial identifier matching ``ds.attrs["trial"]``.

        Examples
        --------
        >>> import ethograph as eto
        >>> dt = eto.open("session.nc")
        >>> ds = dt.trial(1)
        >>> ds.attrs["trial"]
        1
        >>> ds["speed"]  # access a feature variable
        <xarray.DataArray 'speed' (time: 9000, keypoint: 4)>
        """
        if self._is_continuous:
            if trial not in self._trial_epochs:
                raise KeyError(f"No epoch found for trial {trial!r}")
            return self._slice_continuous(trial)
        ds = self[self._trial_node_name(trial)].ds
        if ds is None:
            raise ValueError(f"Trial {trial} has no dataset")
        return ds

    def itrial(self, trial_idx: int) -> xr.Dataset:
        """Return the dataset at an integer index (0-based).

        Parameters
        ----------
        trial_idx : int
            Zero-based index into the list of trials.

        Examples
        --------
        >>> import ethograph as eto
        >>> dt = eto.open("session.nc")
        >>> dt.trials
        [1, 2, 3]
        >>> ds = dt.itrial(0)  # same as dt.trial(1)
        >>> ds.attrs["trial"]
        1
        >>> ds = dt.itrial(2)  # same as dt.trial(3)
        >>> ds.attrs["trial"]
        3
        """
        if self._is_continuous:
            trial_ids = sorted(self._trial_epochs.keys())
            if trial_idx >= len(trial_ids):
                raise IndexError(f"Trial index {trial_idx} out of range")
            return self._slice_continuous(trial_ids[trial_idx])

        trial_nodes = [
            k for k in self.children if self.children[k].ds is not None and "trial" in self.children[k].ds.attrs
        ]
        if trial_idx >= len(trial_nodes):
            raise IndexError(f"Trial index {trial_idx} out of range")
        ds = self[trial_nodes[trial_idx]].ds
        if ds is None:
            raise ValueError(f"Trial at index {trial_idx} has no dataset")
        return ds

    def get_all_trials(self) -> dict[int, xr.Dataset]:
        """Return a dict mapping trial ID to Dataset for all trials."""
        return {num: self.trial(num) for num in self.trials}

    def get_common_attrs(self) -> dict[str, Any]:
        """Return attributes that are identical across all trials."""
        trials_dict = self.get_all_trials()
        if not trials_dict:
            return {}
        common = dict(next(iter(trials_dict.values())).attrs)
        for ds in trials_dict.values():
            common = {k: v for k, v in common.items() if k in ds.attrs and _attrs_equal(ds.attrs[k], v)}
        return common

    # ------------------------------------------------------------------
    # Label operations (labels are now stored in TSV files, not in .nc)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_attr(self, attr_name: str, attr_value: Any) -> TrialTree:
        """Return a new TrialTree containing only trials that match an attribute.

        Checks the metadata table first; falls back to ``ds.attrs``.
        """

        def values_match(stored: Any, target: Any) -> bool:
            if stored == target:
                return True
            for coerce in (str, int, float):
                try:
                    return coerce(stored) == coerce(target)
                except (ValueError, TypeError):
                    continue
            return False

        mdf = getattr(self, "_metadata_df", None)
        if mdf is not None and attr_name in mdf.columns:
            matching_trials = set()
            for _, row in mdf.iterrows():
                if values_match(row[attr_name], attr_value):
                    matching_trials.add(row["trial"])
            new_tree = xr.DataTree()
            for name, node in self.children.items():
                if node.ds and node.ds.attrs.get("trial") in matching_trials:
                    new_tree[name] = node
            result = TrialTree(new_tree)
            result._metadata_df = mdf[mdf["trial"].isin(result.trials)].reset_index(drop=True)
            return result

        new_tree = xr.DataTree()
        for name, node in self.children.items():
            if node.ds and attr_name in node.ds.attrs:
                if values_match(node.ds.attrs[attr_name], attr_value):
                    new_tree[name] = node
        return TrialTree(new_tree)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: str) -> TrialTree:
        """Load a TrialTree from a NetCDF file.

        Auto-discovers ``.ethograph/alignment.nwb`` next to the file.
        """

        tree = xr.open_datatree(path, engine="netcdf4")

        tree.__class__ = cls
        tree._source_path = path
        nwb = discover_nwb(path)
        tree.nwb_alignment = make_nwb_alignment(nwb)
        tree._dirty_trials = set()
        tree._extra_file_handle = None
        tree._record_saved_state()
        return tree

    @classmethod
    def from_datasets(
        cls,
        datasets: list[xr.Dataset],
        validate: bool = True,
    ) -> TrialTree:
        """Build a TrialTree from a list of xarray Datasets.

        Parameters
        ----------
        datasets
            Each dataset must have a unique ``attrs["trial"]`` key.
        validate
            Run validation after construction.
        """
        tree = cls()
        seen: set = set()
        for ds in datasets:
            trial_num = ds.attrs.get("trial")
            if trial_num is None:
                raise ValueError("Each dataset must have 'trial' attribute")
            if trial_num in seen:
                raise ValueError(f"Duplicate trial number: {trial_num}")
            seen.add(trial_num)
            node_name = str(trial_num).replace("/", "_")
            tree[node_name] = xr.DataTree(ds)
        # Ensure nwb_alignment is always set (inherited from __init__, but explicit)
        if not hasattr(tree, "nwb_alignment") or tree.nwb_alignment is None:
            tree.nwb_alignment = EmpytAlignment()
        if validate:
            tree._validate_tree()
        return tree

    @classmethod
    def from_continuous(
        cls,
        ds: xr.Dataset,
        epochs: "pd.DataFrame | nap.IntervalSet",
    ) -> TrialTree:
        """Build a TrialTree from a single continuous recording + trial epochs.

        Unlike :meth:`from_datasets` which requires pre-split data, this
        stores one shared dataset and slices on demand when :meth:`trial`
        is called.  Time coordinates are shifted to 0 per trial.

        Parameters
        ----------
        ds
            Full recording dataset.  Must have at least one dimension
            whose name contains ``"time"``.
        epochs
            Trial boundaries.  Accepts:

            - ``pd.DataFrame`` with columns ``trial``, ``start_time``,
              ``stop_time``.
            - ``nap.IntervalSet`` — trial IDs taken from a ``"trial"``
              metadata column, or ``1, 2, …`` if absent.

        Examples
        --------
        >>> import pandas as pd
        >>> epochs = pd.DataFrame(
        ...     {
        ...         "trial": [1, 2, 3],
        ...         "start_time": [0.0, 60.0, 120.0],
        ...         "stop_time": [60.0, 120.0, 180.0],
        ...     }
        ... )
        >>> dt = TrialTree.from_continuous(ds, epochs)
        >>> dt.trial(2)  # returns 60-120s slice, time shifted to 0
        """
        tree = cls()

        if isinstance(epochs, nap.IntervalSet):
            epoch_dict = {}
            has_trial = "trial" in epochs.metadata.columns
            for i in range(len(epochs)):
                trial_id = epochs.metadata["trial"].iloc[i] if has_trial else i + 1
                if hasattr(trial_id, "item"):
                    trial_id = trial_id.item()
                epoch_dict[trial_id] = (float(epochs.start[i]), float(epochs.end[i]))

        elif isinstance(epochs, pd.DataFrame):
            epoch_dict = {}

            if "trial" in epochs.columns:
                trials_col = epochs["trial"].values
            else:
                trials_col = None

            starts_col = epochs["start_time"].values
            stops_col = epochs["stop_time"].values
            for i in range(len(epochs)):
                trial_id = trials_col[i] if trials_col is not None else i + 1
                if hasattr(trial_id, "item"):
                    trial_id = trial_id.item()
                epoch_dict[trial_id] = (float(starts_col[i]), float(stops_col[i]))
        else:
            raise ValueError("epochs must be a pandas DataFrame or pynapple IntervalSet")

        tree._continuous_ds = ds
        tree._trial_epochs = epoch_dict

        # Ensure nwb_alignment is always set (inherited from __init__, but explicit)
        if not hasattr(tree, "nwb_alignment") or tree.nwb_alignment is None:
            tree.nwb_alignment = EmpytAlignment()
        return tree

    @classmethod
    def from_datatree(
        cls,
        dt: xr.DataTree,
        attrs: dict | None = None,
        *,
        source: "TrialTree | None" = None,
    ) -> TrialTree:
        """Wrap an existing DataTree as a TrialTree.

        Parameters
        ----------
        source
            Original TrialTree to copy ``nwb_alignment`` and ``_source_path`` from.
        """
        tree = cls()
        for name, child in dt.children.items():
            tree[name] = child
        if dt.ds is not None:
            tree.ds = dt.ds
        tree.attrs = (attrs if attrs is not None else dt.attrs).copy()
        if source is not None:
            tree.nwb_alignment = source.nwb_alignment
            sp = getattr(source, "_source_path", None)
            if sp is not None:
                tree._source_path = sp
            mdf = getattr(source, "_metadata_df", None)
            if mdf is not None:
                tree._metadata_df = mdf
        return tree

    def save(self, path: str | Path | None = None) -> None:
        """Write the TrialTree to a NetCDF file.

        Continuous trees are materialised into per-trial nodes before
        saving so the file can be re-opened with :meth:`open`.

        When *path* is the file this tree was opened from (or last saved to)
        and only some trials were touched via :meth:`update_trial` /
        :meth:`set_trial_attr` since then, only those trials are written back
        in place (see :meth:`_try_incremental_save`) — a one-attr edit on one
        trial does not rewrite the whole file. Anything else (a new path, an
        added/removed trial, a variable whose shape changed) falls back to a
        full atomic write (temp file then rename) to avoid partial writes.
        If an alignment NWB exists and the save directory differs from where
        the NWB lives, a copy is placed in ``<save_dir>/.ethograph/`` so that
        :meth:`open` can discover it.
        """
        if self._is_continuous:
            self.materialise().save(path)
            return

        source_path = getattr(self, "_source_path", None)
        if path is None and source_path is None:
            raise ValueError("No path provided and no source path stored.")

        path = Path(path) if path else Path(source_path)

        detached = self._detach_external()
        try:
            if self._try_incremental_save(path):
                self._source_path = str(path)
                self._ensure_alignment_nwb(path.parent)
                return

            temp_path = path.with_suffix(".tmp.nc")

            try:
                self.load()
                self.to_netcdf(temp_path, mode="w")
                self._close_all()
                temp_path.replace(path)
                self._source_path = str(path)
                self._ensure_alignment_nwb(path.parent)
                self._dirty_trials.clear()
                self._record_saved_state()
            finally:
                self._close_all()
        finally:
            self._reattach_external(detached)

    def _detach_external(self) -> dict[str, dict[str, xr.DataArray]]:
        """Take every externally attached variable off the trial nodes.

        A video feature attached in memory from files
        (``io/video_feature_files.py``, ``attrs["attached_from"]``) is not the
        session's data and is never written; :meth:`save` detaches them,
        writes, and puts them back with :meth:`_reattach_external`.
        """
        from ethograph.io.video_feature_files import video_feature_vars

        detached: dict[str, dict[str, xr.DataArray]] = {}
        for name in self._current_trial_node_names():
            ds = self[name].ds
            names = video_feature_vars(ds)
            if names:
                detached[name] = {var: ds[var] for var in names}
                self[name] = xr.DataTree(ds.drop_vars(names))
        return detached

    def _reattach_external(self, detached: dict[str, dict[str, xr.DataArray]]) -> None:
        for name, variables in detached.items():
            self[name] = xr.DataTree(self[name].ds.assign(variables))

    def _close_all(self) -> None:
        """Release every file handle this tree holds.

        :meth:`_try_incremental_save` grafts subtrees from a freshly
        re-opened ``DataTree`` onto this one; closing ``self`` alone doesn't
        reach the close hook that lives on that other tree's root, so it's
        tracked separately (``_extra_file_handle``) and closed here too.
        """
        self.close()
        if self._extra_file_handle is not None:
            self._extra_file_handle.close()
            self._extra_file_handle = None

    def _current_trial_node_names(self) -> list[str]:
        return [name for name, node in self.children.items() if node.ds is not None and "trial" in node.ds.attrs]

    def _record_saved_state(self) -> None:
        """Snapshot the trial/variable layout as "known to be on disk"."""
        names = self._current_trial_node_names()
        self._saved_node_names = frozenset(names)
        self._saved_var_shapes = {
            name: {var: tuple(da.shape) for var, da in self[name].ds.variables.items()} for name in names
        }

    def _try_incremental_save(self, path: Path) -> bool:
        """Write only dirty trials back into an existing file, in place.

        Returns ``True`` once *path* reflects every pending change (including
        the case where nothing changed and no I/O was needed at all).
        Returns ``False`` when an in-place write isn't safe and the caller
        should do a full rewrite instead: no prior save/open to diff against,
        *path* isn't the file this tree was last synced with, a trial was
        added/removed, or a dirty trial's variable shape no longer matches
        what's on disk (HDF5 can't resize a dimension in place).

        HDF5/netCDF4 won't open a file for writing while this tree still
        holds it open for reading, so a successful in-place write closes the
        tree first, writes the dirty groups, then re-opens the file and
        re-points every untouched trial at the fresh handle — the dirty
        trials keep the in-memory dataset that was just verified on disk.
        """
        source_path = getattr(self, "_source_path", None)
        if self._saved_node_names is None or source_path is None:
            return False
        if Path(source_path).resolve() != path.resolve() or not path.exists():
            return False

        current_names = frozenset(self._current_trial_node_names())
        if current_names != self._saved_node_names:
            return False  # a trial was added/removed since the last save

        if not self._dirty_trials:
            return True  # nothing changed — skip all I/O

        dirty = sorted(self._dirty_trials & current_names)
        for name in dirty:
            ds = self[name].ds
            for var_name, shape in self._saved_var_shapes.get(name, {}).items():
                if var_name not in ds.variables or tuple(ds.variables[var_name].shape) != shape:
                    return False  # variable removed, or its shape changed in place

        loaded = {name: self[name].ds.load() for name in dirty}
        self._close_all()
        for name, ds in loaded.items():
            ds.to_netcdf(path, mode="a", group=f"/{name}")

        fresh = xr.open_datatree(str(path), engine="netcdf4")
        for name in current_names:
            self[name] = xr.DataTree(loaded[name]) if name in loaded else fresh[name]
        self._extra_file_handle = fresh

        self._dirty_trials.clear()
        self._record_saved_state()
        return True

    def _ensure_alignment_nwb(self, save_dir: Path) -> None:
        """Copy alignment NWB next to the save location if needed."""
        import shutil

        from ethograph.io.nwb_alignment import (
            _NWB_FILENAME,
            _SETTINGS_DIR,
            NWBAlignment,
        )

        sio = self.nwb_alignment
        if not isinstance(sio, NWBAlignment):
            return
        nwb = str(sio._path)
        if not Path(nwb).exists():
            return

        target = save_dir / _SETTINGS_DIR / _NWB_FILENAME
        if target.exists() or Path(nwb).resolve() == target.resolve():
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nwb, target)
        self.nwb_alignment = make_nwb_alignment(target)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_tree(self) -> list[str]:
        errors = validate_datatree(self)
        if errors:
            raise ValueError("TrialTree validation failed:\n" + "\n".join(f"• {e}" for e in errors))
        return errors

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    @property
    def session(self) -> xr.Dataset | None:
        """Legacy session access. Returns None for NWB-backed trees."""
        if "session" in self.children:
            return self["session"].ds
        return None

    def get_trial_metadata(self, trial) -> dict:
        """Return condition metadata for a single trial as a dict."""
        df = getattr(self, "_metadata_df", None)
        if df is None:
            return {}
        if df.empty:
            return {}
        row = df[df["trial"] == trial]
        if row.empty:
            row = df[df["trial"] == str(trial)]
        if row.empty and isinstance(trial, (int, float)):
            row = df[df["trial"].astype(str) == str(int(trial))]
        if row.empty:
            return {}
        return {k: v for k, v in row.iloc[0].to_dict().items() if k != "trial" and pd.notna(v)}
