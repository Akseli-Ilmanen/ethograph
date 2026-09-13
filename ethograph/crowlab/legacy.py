"""Custom lab-internal legacy code"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ethograph import TrialTree
from ethograph.labels.intervals import empty_intervals
from ethograph.labels.tsv_store import _empty_all_labels


def _xr_to_intervals(ds: xr.Dataset) -> pd.DataFrame:
    """Convert an xarray Dataset back to an intervals DataFrame (legacy)."""
    if "onset_s" not in ds.data_vars:
        return empty_intervals()
    return pd.DataFrame(
        {
            "onset_s": ds["onset_s"].values.astype(np.float64),
            "offset_s": ds["offset_s"].values.astype(np.float64),
            "labels": ds["labels"].values.astype(np.int32),
            "individual": ds["individual"].values.astype(str),
        }
    )


def trees_to_df(
    trees: dict[str, "TrialTree"],
    keep_attrs: list[str],
    correct_offsets_enabled: bool = False,
) -> pd.DataFrame:
    """Flatten labelled segments from one or more TrialTrees into a tidy pd.DataFrame.

    Each non-background interval (``labels > 0``) becomes one row. This is
    the standard way to export ethograph labels for analysis.

    Parameters
    ----------
    trees : dict[str, TrialTree] | TrialTree | Path | str | list
        Flexible input — pass any of:

        * a single :class:`~ethograph.io.trialtree.TrialTree`
        * a dict mapping arbitrary keys to TrialTrees
        * a file path (or list of paths) to saved ``.nc`` files
        * a list of TrialTree objects
    keep_attrs : list[str]
        Trial-level ``ds.attrs`` keys to carry over as extra columns
        (e.g. ``["stimulus", "num_pellets"]``).

    Returns
    -------
    pandas.DataFrame
        One row per labelled segment.  Always-present columns:

        ============== =================================================
        Column         Description
        ============== =================================================
        session        ``ds.attrs["session"]`` (empty string if absent)
        trial          trial identifier
        session_trial  ``"{session}_{trial}"`` for grouping
        individual     subject identifier
        labels         integer action-label class
        onset_s        segment start in trial-relative seconds
        offset_s       segment end in trial-relative seconds
        duration       ``offset_s - onset_s``
        sequence_idx   zero-based position in the trial's label order
        sequence       dash-joined label IDs (e.g. ``"1-3-2"``)
        ============== =================================================

        Columns present only when a session table with ``start_time`` exists:

        =============== ================================================
        trial_onset     absolute trial start (seconds)
        onset_global    ``trial_onset + onset_s``
        offset_global   ``trial_onset + offset_s``
        trial_offset    absolute trial end (if ``stop_time`` available)
        =============== ================================================

    Examples
    --------
    >>> import ethograph as eto
    >>> dt = eto.open("experiment.nc")
    >>> df = eto.trees_to_df(dt, keep_attrs=["stimulus"])
    >>> df[["trial", "labels", "onset_s", "offset_s", "duration"]].head()
       trial  labels  onset_s  offset_s  duration  stimulus
    0      1       2     0.50      1.23      0.73    tone_A
    1      1       1     1.23      2.10      0.87    tone_A
    2      2       3     0.00      0.95      0.95    tone_B

    Multiple files at once:

    >>> df = eto.trees_to_df(
    ...     ["bird_A.nc", "bird_B.nc"],
    ...     keep_attrs=["session"],
    ... )
    """
    from ethograph.io.trialtree import TrialTree

    xr_to_intervals = _xr_to_intervals

    if isinstance(trees, TrialTree):
        trees = {"_single": trees}
    elif isinstance(trees, (str, Path)):
        trees = {"tree_0": TrialTree.open(Path(trees))}
    elif isinstance(trees, list):
        if trees and isinstance(trees[0], (str, Path)):
            trees = {f"tree_{i}": TrialTree.open(Path(p)) for i, p in enumerate(trees)}
        else:
            trees = {str(i): t for i, t in enumerate(trees)}

    rows = []

    for dt in trees.values():
        for trial_id, ds in dt.trial_items():
            intervals = xr_to_intervals(ds)

            attrs = {}
            for attr in keep_attrs:
                if attr in ds.attrs:
                    attrs[attr] = ds.attrs[attr]

            valid = intervals[intervals["labels"] > 0].sort_values("onset_s")

            sequence = valid["labels"].tolist()

            for idx, (_, seg) in enumerate(valid.iterrows()):
                row = {
                    "session": ds.attrs.get("session", ""),  # optional
                    "trial": trial_id,
                    "session_trial": f"{ds.attrs.get('session', '')}_{trial_id}",
                    "individual": seg["individual"],
                    "labels": int(seg["labels"]),
                    "onset_s": seg["onset_s"],
                    "offset_s": seg["offset_s"],
                }
                t_start = None
                t_stop = None
                try:
                    t_start = dt.start_time(trial_id)
                except (AttributeError, KeyError, ValueError):
                    pass
                try:
                    t_stop = dt.stop_time(trial_id)
                except (AttributeError, KeyError, ValueError):
                    pass

                if t_start is None and "pulse_onsets" in ds:
                    t_start = float(ds.pulse_onsets.values[0]) / 30_000  # Legacy crow lab

                if t_start is not None:
                    row["trial_onset"] = t_start
                    row["onset_global"] = t_start + seg["onset_s"]
                    row["offset_global"] = t_start + seg["offset_s"]
                if t_stop is not None:
                    row["trial_offset"] = t_stop

                row.update(
                    {
                        "duration": seg["offset_s"] - seg["onset_s"],
                        "sequence_idx": idx,  # zero-indexing
                        "sequence": "-".join(str(s) for s in sequence),
                    }
                )
                row.update(attrs)
                rows.append(row)

    df = pd.DataFrame(rows)

    if correct_offsets_enabled:
        df = correct_offsets(df)

    return df


def correct_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """Insert artificial gaps where consecutive intervals are too close.

    Pynapple cannot resolve intervals separated by less than ~1e-6 s. When
    an offset and the next onset are within ``eps`` of each other (or
    exactly equal), pynapple raises or silently merges them. This function
    pulls back the earlier offset by ``eps`` so every pair of intervals has
    a resolvable gap.

    Columns updated: ``offset_s``, ``offset_global`` (if present),
    ``duration``.
    """
    df = df.copy().sort_values(["session", "trial", "individual", "sequence_idx"])

    # Pynapple can resolve up to 1e-6 intervals, so we must set lower.
    eps = 1e-3

    idx = df.index.tolist()
    for i in range(len(idx)):
        for j in range(len(idx)):
            if i == j:
                continue
            row_i = idx[i]
            row_j = idx[j]
            if abs(df.loc[row_i, "offset_s"] - df.loc[row_j, "onset_s"]) < eps:
                print(
                    f"Corrected gap (size: {abs(df.loc[row_i, 'offset_s'] - df.loc[row_j, 'onset_s'])}), at labels: {df.loc[row_i, 'labels']}, {df.loc[row_j, 'labels']}"  # noqa: E501
                )
                df.loc[row_i, "offset_s"] = df.loc[row_j, "onset_s"] - eps
                df.loc[row_i, "offset_global"] = df.loc[row_j, "onset_global"] - eps
                df.loc[row_i, "duration"] = df.loc[row_i, "offset_s"] - df.loc[row_i, "onset_s"]

    delta = df["onset_s"] - df["offset_s"].shift(1)
    assert delta.min() >= eps

    return df


# ---------------------------------------------------------------------------
# Migration from label_dt (xr.DataTree) to TSV
# ---------------------------------------------------------------------------


def migrate_label_dt_to_tsv(label_dt) -> pd.DataFrame:
    """Extract all intervals + per-trial metadata from a label DataTree.

    Returns all_labels_df with meta columns.
    """
    xr_to_intervals = _xr_to_intervals

    rows = []

    for trial_id, ds in label_dt.trial_items():
        intervals = xr_to_intervals(ds)
        if not intervals.empty:
            trial_rows = intervals.copy()
            trial_rows.insert(0, "trial", trial_id)

            # Migrate per-trial attrs to columns
            trial_rows["human_verified"] = int(ds.attrs.get("human_verified", 0))
            trial_rows["changepoint_corrected"] = int(ds.attrs.get("changepoint_corrected", 0))
            trial_rows["prediction_source"] = ""
            rows.append(trial_rows)

    if rows:
        all_df = pd.concat(rows, ignore_index=True)
    else:
        all_df = _empty_all_labels()

    return all_df


def init_empty_labels(trials: list) -> pd.DataFrame:
    """Create empty labels DataFrame."""
    return _empty_all_labels()


def convert_session_to_nwb(dt: TrialTree, output_path: str | Path | None = None) -> Path:
    """Convert an old .nc file with xarray session node to an alignment.nwb file.

    Reads ``dt.session`` (the legacy xarray session node) and writes an
    equivalent NWB file with trials table and acquisition ImageSeries.

    Parameters
    ----------
    dt
        TrialTree loaded from an old ``.nc`` file that has a ``"session"``
        child node with media DataArrays.
    output_path
        Where to write the NWB file. Defaults to ``.ethograph/alignment.nwb``
        relative to the tree's source path.

    Returns
    -------
    Path to the created NWB file.
    """
    from ethograph.io.pairing import pair_media

    sess = dt.session
    if sess is None:
        raise ValueError("TrialTree has no legacy session node to convert.")

    trials = dt.trials
    rows = []
    for trial_id in trials:
        row = {"trial": trial_id}
        if "start_time" in sess:
            try:
                row["start_time"] = float(sess["start_time"].sel(trial=trial_id))
            except (KeyError, ValueError):
                row["start_time"] = 0.0
        else:
            row["start_time"] = 0.0

        row["stop_time"] = row["start_time"] + dt.trial(trial_id).time.values[-1]

        # Extract media columns
        for stream in ("video", "audio", "pose"):
            if stream not in sess:
                continue
            da = sess[stream]
            if "trial" in da.dims:
                # Per-trial media
                device_dim = None
                for dim in da.dims:
                    if dim != "trial":
                        device_dim = dim
                        break
                if device_dim and device_dim in da.coords:
                    for dev in da.coords[device_dim].values:
                        try:
                            val = str(da.sel(trial=trial_id, **{device_dim: dev}).values)
                            row[f"{stream}_{dev}"] = val if val != "nan" else ""
                        except (KeyError, ValueError):
                            row[f"{stream}_{dev}"] = ""
                else:
                    try:
                        val = str(da.sel(trial=trial_id).values)
                        row[f"{stream}_0"] = val if val != "nan" else ""
                    except (KeyError, ValueError):
                        row[f"{stream}_0"] = ""
            else:
                # Session-wide media
                device_dim = None
                for dim in da.dims:
                    device_dim = dim
                    break
                if device_dim and device_dim in da.coords:
                    for dev in da.coords[device_dim].values:
                        try:
                            val = str(da.sel(**{device_dim: dev}).values)
                            row[f"{stream}_{dev}"] = val if val != "nan" else ""
                        except (KeyError, ValueError):
                            row[f"{stream}_{dev}"] = ""

        rows.append(row)

    trial_df = pd.DataFrame(rows)

    try:
        ds = dt.itrial(0)
        if "fps" in ds.attrs:
            fps = float(ds.attrs["fps"])
    except (StopIteration, IndexError):
        pass

    if output_path is None:
        source = getattr(dt, "_source_path", None)
        if source:
            output_path = Path(source).parent / ".ethograph" / "alignment.nwb"
        else:
            output_path = Path.cwd() / ".ethograph" / "alignment.nwb"

    output_path = Path(output_path)
    stream_rates = {"video": fps, "pose": fps}

    pair_media(trial_df, stream_rates=stream_rates, output_path=output_path)
    return output_path


def migrate_attrs_to_metadata_tsv(
    dt: TrialTree,
    output_path: str | Path | None = None,
) -> Path:
    """Extract per-trial condition attrs from ds.attrs and save as metadata TSV.

    Parameters
    ----------
    dt
        TrialTree with condition metadata in ``ds.attrs``.
    output_path
        Where to write.  Defaults to ``{nc_stem}_metadata.tsv``.

    Returns
    -------
    Path to the created TSV file.
    """
    from ethograph.io.metadata_table import (
        metadata_from_attrs,
        metadata_tsv_path,
        save_metadata_tsv,
    )

    df = metadata_from_attrs(dt)

    if output_path is None:
        source = getattr(dt, "_source_path", None)
        if source:
            output_path = metadata_tsv_path(source)
        else:
            output_path = Path.cwd() / "metadata.tsv"

    output_path = Path(output_path)
    save_metadata_tsv(output_path, df)
    dt.metadata_df = df
    return output_path
