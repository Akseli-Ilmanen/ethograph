"""Trial metadata table: TSV-based per-trial condition metadata.

The metadata table is a plain TSV file at ``{nc_stem}_metadata.tsv``,
one row per trial.  Required column: ``trial`` (join key).  All other
columns are user-defined conditions (genotype, treatment, etc.).

Three source scenarios (checked in priority order):
1. TSV file exists → editable, user-owned
2. NWB trials table has metadata columns → read-only
3. Legacy ds.attrs have conditions → one-time migration to TSV
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ethograph.io.nwb_alignment import make_nwb_alignment
from ethograph.io.pynapple import load_nap_data

logger = logging.getLogger(__name__)

#: Media streams whose per-trial filename the trials table shows.
MEDIA_STREAMS = ("video", "pose", "audio", "ephys")

_STREAM_COL_RE = re.compile(rf"^({'|'.join(MEDIA_STREAMS)})_(.+)$")

TABULAR_METADATA_EXTS = frozenset({".tsv", ".csv", ".xlsx", ".xls"})

_NWB_STRUCTURAL_COLUMNS = frozenset(
    {
        "trial",
        "start_time",
        "stop_time",
    }
)


def validate_metadata_timing(df: pd.DataFrame, path: str | Path | None = None) -> None:
    """Validate a metadata DataFrame for use as trial timing source.

    Raises ValueError with a specific message for every detectable problem.
    Call this when the user *explicitly* provides a metadata file.

    Required columns: ``trial``, ``start_time``, ``stop_time``.
    """
    label = f" in {Path(path).name}" if path else ""

    if df.empty:
        raise ValueError(f"Metadata table{label} is empty")

    missing = [c for c in ("start_time", "stop_time") if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata table{label} missing column(s): {', '.join(missing)}")

    # Check trial column
    if "trial" not in df.columns:
        raise ValueError(f"Metadata table{label} missing required 'trial' column")

    # Check numeric types
    for col in ("start_time", "stop_time"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}'{label} must be numeric, got {df[col].dtype}")

    starts = df["start_time"].values
    stops = df["stop_time"].values

    # Check for NaN
    nan_starts = np.isnan(starts)
    nan_stops = np.isnan(stops)
    if nan_starts.any():
        rows = np.where(nan_starts)[0] + 1
        raise ValueError(f"NaN in 'start_time'{label} at row(s): {list(rows)}")
    if nan_stops.any():
        rows = np.where(nan_stops)[0] + 1
        raise ValueError(f"NaN in 'stop_time'{label} at row(s): {list(rows)}")

    # Check start < stop per row
    inverted = starts >= stops
    if inverted.any():
        rows = np.where(inverted)[0] + 1
        raise ValueError(f"start_time >= stop_time{label} at row(s): {list(rows)}")

    # Check monotonic starts
    if len(starts) > 1 and not np.all(np.diff(starts) > 0):
        raise ValueError(f"'start_time' values{label} are not strictly increasing")

    # Check duplicate trial IDs
    trials = df["trial"].values
    uniq, counts = np.unique(trials, return_counts=True)
    dups = uniq[counts > 1]
    if len(dups) > 0:
        raise ValueError(f"Duplicate trial IDs{label}: {list(dups)}")


def trials_ep_from_metadata_df(df: pd.DataFrame):
    """Build a pynapple IntervalSet from a metadata DataFrame with timing columns.

    Returns None if the DataFrame lacks ``start_time`` / ``stop_time``.
    """
    if df.empty or "start_time" not in df.columns or "stop_time" not in df.columns:
        return None

    from ethograph.io.nwb_alignment import _build_trials_ep

    return _build_trials_ep(df)


def metadata_tsv_path(nc_path: str | Path) -> Path:
    """Derive the metadata TSV path from a dataset file path."""
    p = Path(nc_path).resolve()
    return p.parent / f"{p.stem}_metadata.tsv"


def load_metadata_tsv(path: str | Path) -> pd.DataFrame:
    """Load a metadata TSV, CSV, or Excel file.

    The delimiter is chosen from the extension (comma for ``.csv``, tab
    otherwise) rather than sniffed: a single-column table has no delimiter
    character to sniff, and ``csv.Sniffer`` then picks a letter out of the
    header text itself (e.g. splitting ``trial`` on its own ``t``).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    if "trial" not in df.columns:
        raise ValueError(f"Metadata table {path} missing required 'trial' column")
    return df


def save_metadata_tsv(path: str | Path, df: pd.DataFrame) -> None:
    """Save a metadata TSV file (atomic write), minus the media filenames.

    See :func:`stored_columns` — a filename is the alignment's to state.
    """
    path = Path(path)
    df = stored_columns(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tsv.tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def condition_columns(df: pd.DataFrame) -> list[str]:
    """Return user-defined condition column names — the editable ones.

    Excludes structural NWB timing columns and the media filename columns:
    a filename is the alignment's to state, not the metadata file's.
    """
    return [c for c in df.columns if c != "trial" and not _is_nwb_infrastructure_col(c)]


def stored_columns(df: pd.DataFrame) -> pd.DataFrame:
    """*df* without its media filename columns — what a metadata file holds.

    Filenames are joined on from the alignment at load
    (:func:`attach_media_columns`); a copy in the metadata file would only go
    stale the moment the media is re-aligned or moved.
    """
    return df.drop(columns=media_columns(df))


def is_media_column(col: str) -> bool:
    """True if *col* names a trial's media file (``video_cam-1``, ``pose_cam-1``)."""
    return bool(_STREAM_COL_RE.match(col)) and not col.endswith("_start")


def media_columns(df: pd.DataFrame) -> list[str]:
    """Media filename columns present in *df*, in the order it holds them."""
    return [c for c in df.columns if is_media_column(str(c))]


def _media_sort_key(col: str) -> tuple[int, str]:
    match = _STREAM_COL_RE.match(col)
    assert match is not None
    return MEDIA_STREAMS.index(match.group(1)), match.group(2)


def order_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """``trial``, then the conditions, then the media filenames.

    Filenames are the least discriminating column in the table, so they sit on
    the right — visible (and filterable) without pushing a condition out of
    view.
    """
    media = media_columns(df)
    lead = ["trial"] if "trial" in df.columns else []
    rest = [c for c in df.columns if c != "trial" and c not in media]
    return df.loc[:, lead + rest + sorted(media, key=_media_sort_key)]


def media_columns_from_alignment(alignment, trial_ids: list[int | str]) -> pd.DataFrame:
    """One ``{stream}_{device}`` column per media device, a filename per trial.

    Read off the alignment rather than a metadata file: the alignment is the
    one holder of which file a trial plays, and it names that file whether it
    lives in a trials-table column or in an acquisition's ``external_file``.
    """
    columns: dict[str, list[str | None]] = {}
    for stream in MEDIA_STREAMS:
        for device in alignment.devices(stream):
            names = [alignment.media_filename(t, stream, device) for t in trial_ids]
            if any(names):
                columns[f"{stream}_{device}"] = names
    return pd.DataFrame(columns, index=pd.RangeIndex(len(trial_ids)))


def attach_media_columns(df: pd.DataFrame, alignment=None) -> pd.DataFrame:
    """Add the alignment's media filenames to *df* and order the table.

    An existing column of the same name is replaced: the alignment wins over a
    copy that ended up in a metadata file.
    """
    if alignment is None or df.empty or "trial" not in df.columns:
        return order_metadata_columns(df)

    media = media_columns_from_alignment(alignment, list(df["trial"]))
    result = df.drop(columns=[c for c in media.columns if c in df.columns])
    for col in media.columns:
        result[col] = media[col].to_numpy()
    return order_metadata_columns(result)


def _is_nwb_infrastructure_col(col: str) -> bool:
    """True if column is structural or a filename rather than a condition."""
    return _is_structural_col(col) or is_media_column(col)


def _is_structural_col(col: str) -> bool:
    """True for timing/offset columns — dropped from the metadata table."""
    return col in _NWB_STRUCTURAL_COLUMNS or col.endswith("_start")


def empty_metadata_df(trials: list[int | str]) -> pd.DataFrame:
    """Build an empty metadata table with one row per trial."""
    return pd.DataFrame({"trial": list(trials)})


def _normalise_trial_column(df: pd.DataFrame, trial_ids: list[int | str] | None) -> pd.DataFrame:
    if df.empty:
        if trial_ids is None:
            return pd.DataFrame({"trial": []})
        return pd.DataFrame({"trial": list(trial_ids)})

    result = df.copy()
    if "trial" not in result.columns:
        if trial_ids is None:
            trial_ids = list(range(1, len(result) + 1))
        result.insert(0, "trial", list(trial_ids)[: len(result)])
    else:
        # Older alignment NWBs / TSVs may store integral trial IDs as floats
        # (e.g. 1.0); app_state.trials requires int or str.
        col = result["trial"]
        if pd.api.types.is_float_dtype(col):
            non_nan = col.dropna()
            if not non_nan.empty and (non_nan == non_nan.astype("int64")).all():
                result["trial"] = col.astype("Int64")
    return result


def metadata_from_nwb_trials(trials_df: pd.DataFrame, trial_ids: list[int | str] | None = None) -> pd.DataFrame:
    """Extract user metadata columns from an NWB trials table."""
    if trials_df is None or trials_df.empty:
        return empty_metadata_df(trial_ids or [])

    df = _normalise_trial_column(trials_df, trial_ids)
    keep_cols = ["trial"] + [c for c in df.columns if c != "trial" and not _is_structural_col(c)]
    return order_metadata_columns(df.loc[:, [c for c in keep_cols if c in df.columns]].copy())


def metadata_from_intervalset(trials_ep, trial_ids: list[int | str] | None = None) -> pd.DataFrame:
    """Extract metadata columns from a pynapple IntervalSet."""
    if trials_ep is None:
        return empty_metadata_df(trial_ids or [])

    data = (
        pd.DataFrame(trials_ep.metadata).copy() if getattr(trials_ep, "metadata", None) is not None else pd.DataFrame()
    )
    df = _normalise_trial_column(data, trial_ids)
    if df.empty:
        return empty_metadata_df(trial_ids or [])

    keep_cols = ["trial"] + [c for c in df.columns if c != "trial" and not _is_structural_col(c)]
    return order_metadata_columns(df.loc[:, [c for c in keep_cols if c in df.columns]].copy())


def load_metadata_df(
    source_path: str | Path | None = None,
    *,
    metadata_path: str | Path | None = None,
    nwb_alignment=None,
    trials_ep=None,
    trial_ids: list[int | str] | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Load a metadata table from TSV or structured NWB/pynapple metadata.

    Priority order:
    1. Explicit ``metadata_path`` (``.tsv``, ``.nwb``, ``.npz``, or pynapple folder).
    2. ``source_path`` as direct metadata source (``.nwb``/``.npz``/folder).
    3. Sidecar ``{stem}_metadata.tsv`` next to ``source_path``.
    4. Metadata embedded in the loaded NWB alignment object.
    5. Metadata stored on a pynapple IntervalSet.
    6. Empty table with one row per trial.

    Whatever the source, the alignment's media filenames are joined on as the
    rightmost columns (:func:`attach_media_columns`).
    """
    df, path, opened = _resolve_metadata_source(
        source_path,
        metadata_path=metadata_path,
        nwb_alignment=nwb_alignment,
        trials_ep=trials_ep,
        trial_ids=trial_ids,
    )
    return attach_media_columns(df, nwb_alignment or opened), path


def _resolve_metadata_source(
    source_path: str | Path | None = None,
    *,
    metadata_path: str | Path | None = None,
    nwb_alignment=None,
    trials_ep=None,
    trial_ids: list[int | str] | None = None,
) -> tuple[pd.DataFrame, str | None, object | None]:
    """The metadata table, its path, and any alignment opened to read it."""
    opened = None
    if metadata_path is not None:
        path = Path(metadata_path)
        if path.suffix.lower() in TABULAR_METADATA_EXTS and path.exists():
            return _normalise_trial_column(load_metadata_tsv(path), trial_ids), str(path), None
        if path.suffix.lower() == ".nwb" and path.exists():
            opened = make_nwb_alignment(path)
            return metadata_from_nwb_trials(opened.trials_df, trial_ids), str(path), opened
        if (path.suffix.lower() == ".npz" or path.is_dir()) and path.exists():
            _, trials_ep = load_nap_data(str(path))
            return metadata_from_intervalset(trials_ep, trial_ids), str(path), None

    if source_path is not None:
        source = Path(source_path)
        if source.suffix.lower() in TABULAR_METADATA_EXTS and source.exists():
            return _normalise_trial_column(load_metadata_tsv(source), trial_ids), str(source), None
        if source.suffix.lower() == ".nwb" and source.exists():
            opened = make_nwb_alignment(source)
            if opened.trials_df is not None and not opened.trials_df.empty:
                return metadata_from_nwb_trials(opened.trials_df, trial_ids), str(source), opened
        if source.is_file():
            sidecar = metadata_tsv_path(source)
            if sidecar.exists():
                return _normalise_trial_column(load_metadata_tsv(sidecar), trial_ids), str(sidecar), opened
        if source.suffix.lower() == ".npz" and source.exists():
            _, trials_ep = load_nap_data(str(source))
            return metadata_from_intervalset(trials_ep, trial_ids), str(source), opened
        if source.is_dir() and source.exists():
            _, trials_ep = load_nap_data(str(source))
            return metadata_from_intervalset(trials_ep, trial_ids), str(source), opened

    if nwb_alignment is not None:
        return metadata_from_nwb_trials(nwb_alignment.trials_df, trial_ids), None, None

    if trials_ep is not None:
        return metadata_from_intervalset(trials_ep, trial_ids), None, None

    return empty_metadata_df(trial_ids or []), None, opened
