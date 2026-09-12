"""Write an edited trial metadata table back to the source it was read from.

:func:`~ethograph.io.metadata_table.load_metadata_df` picks a metadata source
by priority; an edit made in the GUI has to land back in *that* source, or the
next load silently discards it.  :func:`resolve_metadata_target` names the file
and how to write it, and :func:`write_metadata` writes it.

NWB trials tables are written in append mode (``NWBHDF5IO(path, "a")``): the
export path used by :func:`~ethograph.io.nwb_alignment.edit_nwb` copies
already-written datasets verbatim, so a changed value never reaches the file.

:data:`DERIVED_COLUMNS` — state EthoGraph works out for itself, the curation
verdict — never goes into an NWB. That write happens in place
(there is no atomic replace to fall back on), and for a non-NWB dataset the
alignment NWB is the sole holder of the trial timing, so a crash mid-write
would cost far more than the column. :func:`ensure_tabular_target` hands out a
sidecar TSV instead, and :func:`write_trials_metadata` refuses those columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ethograph.io.metadata_table import (
    TABULAR_METADATA_EXTS,
    _normalise_trial_column,
    condition_columns,
    empty_metadata_df,
    metadata_tsv_path,
    stored_columns,
)
from ethograph.labels.curation import CURATED_COLUMN

logger = logging.getLogger(__name__)

TARGET_TABULAR = "tabular"
TARGET_NWB = "nwb"

#: Columns EthoGraph derives rather than reads. Written to a tabular file
#: only — see the module docstring.
DERIVED_COLUMNS = frozenset({CURATED_COLUMN})


@dataclass(frozen=True)
class MetadataTarget:
    """The file an edited metadata table belongs to, and how to write it."""

    path: Path
    kind: str  # TARGET_TABULAR | TARGET_NWB


def default_sidecar(source: str | Path) -> Path:
    """Sidecar metadata TSV for a data source that cannot be written in place."""
    source = Path(source)
    if source.is_dir():
        return source / f"{source.name}_metadata.tsv"
    return metadata_tsv_path(source)


def resolve_metadata_target(
    source_path: str | Path | None,
    *,
    metadata_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
) -> MetadataTarget | None:
    """Where edits to the loaded metadata table have to be written.

    Mirrors the load priority so a written value is the one read back:

    1. An explicit tabular ``metadata_path`` — write that file.
    2. An explicit ``.nwb`` ``metadata_path``, or an ``.nwb`` data source —
       write its trials table.
    3. The alignment NWB (``.ethograph/alignment.nwb``) — write its trials
       table, because it outranks a sidecar TSV for the columns it carries.
    4. Otherwise a sidecar ``{stem}_metadata.tsv``, which outranks every
       remaining source (pynapple ``IntervalSet`` metadata) on the next load.

    Returns ``None`` when there is no data source to hang a metadata file off.
    """
    if metadata_path:
        path = Path(metadata_path)
        if path.suffix.lower() in TABULAR_METADATA_EXTS:
            return MetadataTarget(path, TARGET_TABULAR)
        if path.suffix.lower() == ".nwb":
            return MetadataTarget(path, TARGET_NWB)
        # .npz / pynapple folder: not editable in place — fall through.

    if source_path and Path(source_path).suffix.lower() == ".nwb" and Path(source_path).exists():
        return MetadataTarget(Path(source_path), TARGET_NWB)

    if alignment_path and Path(alignment_path).exists():
        return MetadataTarget(Path(alignment_path), TARGET_NWB)

    if source_path:
        return MetadataTarget(default_sidecar(source_path), TARGET_TABULAR)

    return None


def ensure_tabular_target(
    source_path: str | Path | None,
    metadata_df: pd.DataFrame | None,
    *,
    metadata_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    trials: list | None = None,
) -> MetadataTarget | None:
    """A tabular metadata target for :data:`DERIVED_COLUMNS`, written if missing.

    An NWB target becomes the sidecar TSV, seeded with the loaded metadata
    table (one row per trial when there is none) — so the derived column has a
    file to live in, and the NWB is left alone. The sidecar then becomes the
    metadata table for this dataset: the caller sets ``app_state.metadata_path``
    so the next load reads it first, ahead of the NWB it was copied from. An
    existing file is never overwritten. Returns ``None`` when there is no data
    source to hang a metadata file off.
    """
    target = resolve_metadata_target(source_path, metadata_path=metadata_path, alignment_path=alignment_path)
    if target is None:
        return None
    if target.kind == TARGET_NWB:
        if not source_path:
            return None
        target = MetadataTarget(default_sidecar(source_path), TARGET_TABULAR)

    if not target.path.exists():
        df = metadata_df
        if df is None or df.empty:
            df = empty_metadata_df(list(trials or []))
        else:
            df = _normalise_trial_column(df, list(trials) if trials else None)
        save_metadata_table(target.path, df)
        logger.info("Curation metadata lives in %s.", target.path.name)
    return target


def coerce_value(text: str, series: pd.Series | None):
    """Typed value for *text*, matching the column it is going into.

    An edited cell arrives as text; the column may be integer-valued (trial
    quality 1–5) or float, and storing the text as a string would break both
    the numeric filter in the trials table and the NWB column's dtype.
    """
    text = text.strip()
    if not text:
        return ""
    if series is not None and pd.api.types.is_numeric_dtype(series):
        try:
            return int(text) if pd.api.types.is_integer_dtype(series) else float(text)
        except ValueError:
            return text
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def blank_column(df: pd.DataFrame) -> pd.Series:
    """An empty column that later accepts any type.

    Explicitly ``object``: pandas gives a bare ``df[c] = ""`` a string dtype
    that then refuses the numbers a hand-scored column is likely to hold.
    """
    return pd.Series([""] * len(df), index=df.index, dtype=object)


def fits_dtype(series: pd.Series, value) -> bool:
    """Whether *value* can go into *series* without changing its dtype."""
    if series.dtype == object:
        return True
    if pd.api.types.is_numeric_dtype(series):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def save_metadata_table(path: str | Path, df: pd.DataFrame) -> None:
    """Write a metadata table, honouring the target's format (atomic).

    Media filename columns are left out (:func:`stored_columns`).
    """
    path = Path(path)
    df = stored_columns(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    tmp = path.with_name(path.name + ".tmp")
    if suffix in (".xlsx", ".xls"):
        df.to_excel(tmp, index=False)
    elif suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def write_metadata(
    target: MetadataTarget,
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> None:
    """Write *df* to *target*.

    A tabular target is rewritten whole — the file *is* the metadata table. An
    NWB target only gets *columns* (the ones actually edited), so scoring one
    condition never rewrites the rest of somebody's recording.
    """
    if target.kind == TARGET_NWB:
        write_trials_metadata(target.path, df, columns=columns)
    else:
        save_metadata_table(target.path, df)


# ---------------------------------------------------------------------------
# NWB trials table
# ---------------------------------------------------------------------------


def write_trials_metadata(
    nwb_path: str | Path,
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> list[str]:
    """Write metadata *columns* of *df* into an NWB trials table.

    Rows are joined on the trials table's ``trial`` column when it has one, and
    positionally otherwise. Existing columns are updated in place; unknown ones
    are appended. Returns the column names written.

    :data:`DERIVED_COLUMNS` are dropped: a curation verdict is ours, not the
    recording's, and this write is not atomic.

    Raises ``ValueError`` when a value cannot be stored in an existing column's
    dtype (a word in a numeric column, say) — HDF5 datasets keep the dtype they
    were written with.
    """
    from pynwb import NWBHDF5IO

    nwb_path = Path(nwb_path)
    wanted = list(columns) if columns is not None else list(df.columns)
    editable = set(condition_columns(df))
    refused = [c for c in wanted if c in DERIVED_COLUMNS]
    if refused:
        logger.warning(
            "Not writing derived column(s) %s to %s — they belong in a metadata TSV",
            ", ".join(refused),
            nwb_path.name,
        )
    cols = [c for c in wanted if c in df.columns and c in editable and c not in DERIVED_COLUMNS]
    if not cols:
        return []

    with NWBHDF5IO(str(nwb_path), "a", load_namespaces=True) as io:
        nwbfile = io.read()
        table = nwbfile.trials
        if table is None:
            raise ValueError(f"{nwb_path.name} has no trials table to write metadata into")

        order = _row_order(table, df)
        for col in cols:
            values = _column_values(df[col], order)
            if col in table.colnames:
                _update_column(table[col], values, col, nwb_path)
            else:
                table.add_column(name=col, description=col, data=values)
        io.write(nwbfile)

    logger.info("Wrote metadata column(s) %s to %s", ", ".join(cols), nwb_path.name)
    return cols


def _row_order(table, df: pd.DataFrame) -> list[int | None]:
    """Positional index into *df* for every row of the trials *table*."""
    n_rows = len(table)
    if "trial" not in table.colnames or "trial" not in df.columns:
        return [i if i < len(df) else None for i in range(n_rows)]

    lookup = {str(t): i for i, t in enumerate(df["trial"])}
    return [lookup.get(str(t)) for t in list(table["trial"][:])]


def _column_values(series: pd.Series, order: list[int | None]) -> list:
    """Native Python values for *series*, one per trials-table row."""
    numeric = pd.api.types.is_numeric_dtype(series)
    values: list = []
    for idx in order:
        value = series.iloc[idx] if idx is not None else None
        if value is None or pd.isna(value):
            values.append(float("nan") if numeric else "")
        elif numeric:
            values.append(float(value))
        else:
            values.append(str(value))
    return values


def _update_column(vector_data, values: list, name: str, nwb_path: Path) -> None:
    data = vector_data.data
    dtype = getattr(data, "dtype", None)
    if dtype is not None:
        values = _cast_to_dtype(values, dtype, name, nwb_path)
    for i, value in enumerate(values):
        data[i] = value


def _cast_to_dtype(values: list, dtype, name: str, nwb_path: Path) -> list:
    kind = getattr(dtype, "kind", "O")
    try:
        if kind in "iu":
            return [int(v) for v in values]
        if kind == "f":
            return [float(v) for v in values]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Column {name!r} in {nwb_path.name} is numeric, so a non-numeric value cannot be "
            f"stored in it. Use a new column, or a TSV metadata file, for free-text values."
        ) from e
    if kind in "OSU":
        return [str(v) for v in values]
    return values
