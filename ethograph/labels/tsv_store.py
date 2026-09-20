"""TSV-based label storage for Ethograph.

File format:
    trial                   - trial identifier
    individual              - individual performing the behaviour (actor)
    individual_rec          - recipient of a dyadic behaviour ("" if none)
    labels                  - integer label class ID
    onset_s                 - start time in seconds (trial-relative)
    offset_s                - end time in seconds (trial-relative)
    confidence              - 1.0 for a human label, the model's own score for
                              a predicted one
    labeling_method         - per label: "manual" | "automated" | "curated"
                              (see ethograph.labels.curation)
    n_samples               - per-trial sample count for dense conversion (int, 0 if unknown)
    changepoint_corrected   - per-trial flag (0/1), repeated per row
    prediction_source       - path to prediction file that produced this label (empty if human)

A ``human_verified`` column in an older file is carried along untouched and
never read: whether a trial is reviewed is a question answered per label by
``labeling_method`` now.

Label names are managed centrally in mapping.txt.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ethograph.io.session_layout import labels_path
from ethograph.labels.intervals import (
    INTERVAL_COLUMNS,
    INTERVAL_DTYPES,
    KIND_KEY,
    KIND_LABEL,
    KIND_TRIAL_META,
    LABEL_SCHEMA,
    empty_intervals,
    ensure_confidence,
    ensure_event_type,
    ensure_individual_rec,
    ensure_labeling_method,
    schema_columns,
    write_order,
)

logger = logging.getLogger(__name__)

#: Every stored column, in file order. Derived from the schema, so a column
#: added there needs no edit here.
TSV_COLUMNS = schema_columns(KIND_KEY, KIND_LABEL, KIND_TRIAL_META)

# Per-trial metadata columns (same value for all rows in a trial)
TRIAL_META_COLUMNS = schema_columns(KIND_TRIAL_META)

TRIAL_META_DEFAULTS = {col.name: col.default for col in LABEL_SCHEMA if col.kind == KIND_TRIAL_META}


def _ensure_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in every per-label column an older file may lack (in place)."""
    ensure_event_type(df)
    ensure_individual_rec(df)
    ensure_confidence(df)
    ensure_labeling_method(df)
    return df


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def labels_tsv_path(source: str | Path, suffix: str = "") -> Path:
    """The session's labels file: ``labels{suffix}.tsv`` in the session folder.

    *source* is the session folder or any file in it (``session.nc``, an ``.nwb``);
    one folder is one session, so the name never depends on the file.

    Examples
    --------
    >>> labels_tsv_path("experiment/data.nc").name
    'labels.tsv'
    >>> labels_tsv_path("experiment", suffix="_downsampled_100x").name
    'labels_downsampled_100x.tsv'
    """
    return labels_path(source, suffix)


# ---------------------------------------------------------------------------
# Load / save labels
# ---------------------------------------------------------------------------

#: The minimum a file must have to be a labels file at all.
REQUIRED_COLUMNS = {col.name for col in LABEL_SCHEMA if col.required}

#: Columns that must carry a value on *every* row. ``offset_s`` is excluded —
#: a point event legitimately stores it as NaN.
REQUIRED_NONNULL_COLUMNS = tuple(col.name for col in LABEL_SCHEMA if col.nonnull)


def validate_labels_tsv(df: pd.DataFrame, path: str | Path = "") -> None:
    """Validate that a labels DataFrame has all required columns and values.

    Raises
    ------
    ValueError
        If any of ``onset_s``, ``offset_s``, ``labels``, ``individual``, ``trial``
        are missing from the DataFrame columns, or if any row is missing a
        value in one of :data:`REQUIRED_NONNULL_COLUMNS` (``offset_s`` aside,
        since a point event's is legitimately blank).
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Labels file {path} is missing required columns: {sorted(missing)}. Required: {sorted(REQUIRED_COLUMNS)}"
        )

    for col in REQUIRED_NONNULL_COLUMNS:
        # astype(str) covers both plain-object and pandas' newer StringDtype
        # columns; a numeric column's values never stringify to "".
        blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        n_blank = int(blank.sum())
        if n_blank:
            # +2: 1 for the header row, 1 for 1-indexing.
            rows = (blank[blank].index[:10] + 2).tolist()
            raise ValueError(
                f"Labels file {path} has {n_blank} row(s) with a missing '{col}' value "
                f"(e.g. line(s) {rows}). Every row must have a '{col}'."
            )


def load_labels_tsv(path: str | Path) -> pd.DataFrame:
    """Load labels from a TSV file.

    Parameters
    ----------
    path : str or Path
        Path to a ``_labels.tsv`` file.

    Returns
    -------
    pd.DataFrame
        Columns: ``trial``, ``onset_s``, ``offset_s``, ``labels`` (int),
        ``individual``, ``labeling_method``, ``changepoint_corrected``,
        ``prediction_source``.

    Examples
    --------
    >>> df = load_labels_tsv("experiment/data_labels.tsv")
    >>> df[["trial", "onset_s", "offset_s", "labels", "individual"]].head()
       trial  onset_s  offset_s  labels individual
    0      1     0.41     0.505       1      crow1
    1      1     0.51     0.620       2      crow1
    """
    path = Path(path)
    if not path.exists():
        return _empty_all_labels()

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        # comment="#": files saved while Ethograph briefly wrote a leading
        # "# time_basis:" line still read back as a normal table.
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", comment="#")
    validate_labels_tsv(df, path)

    _ensure_label_columns(df)
    for col in INTERVAL_COLUMNS:
        if col in df.columns and col in INTERVAL_DTYPES:
            df[col] = df[col].astype(INTERVAL_DTYPES[col])

    for col, default in TRIAL_META_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    df["prediction_source"] = df["prediction_source"].fillna("").astype(str)
    df["n_samples"] = df["n_samples"].fillna(0).astype(int)
    return df


def save_labels_tsv(path: str | Path, df: pd.DataFrame) -> None:
    """Save labels DataFrame to TSV. Uses atomic write (tmp + rename).

    Parameters
    ----------
    path : str or Path
        Destination path.
    df : pd.DataFrame
        Labels DataFrame with required columns (see :data:`REQUIRED_COLUMNS`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if "onset_global" in out.columns:
        out = out.sort_values("onset_global").reset_index(drop=True)
    else:
        out = out.sort_values(["trial", "onset_s"]).reset_index(drop=True)

    out = out[write_order(out.columns)]

    # Drop negative-duration state intervals; keep point events (NaN offset).
    duration = out["offset_s"] - out["onset_s"]
    out = out[duration.isna() | (duration >= 0.0)]

    tmp = path.with_suffix(".tsv.tmp")
    out.to_csv(tmp, sep="\t", index=False, encoding="utf-8-sig")
    tmp.replace(path)


def labels_equal(left: pd.DataFrame | None, right: pd.DataFrame | None) -> bool:
    """Whether two label tables hold the same labels.

    Compared on :data:`TSV_COLUMNS` only, sorted: a saved file carries computed
    extras (duration, global timing, session) that say nothing about what was
    labelled, and row order is not content either. Used to answer "are there
    really unsaved labels?" honestly, instead of trusting a flag that any
    label-adjacent action clears.
    """
    a, b = _comparable_labels(left), _comparable_labels(right)
    if len(a) != len(b):
        return False
    if a.empty:
        return True
    for col in TSV_COLUMNS:
        lhs, rhs = a[col], b[col]
        if pd.api.types.is_numeric_dtype(lhs) and pd.api.types.is_numeric_dtype(rhs):
            if not np.allclose(lhs.to_numpy(float), rhs.to_numpy(float), equal_nan=True):
                return False
        elif not lhs.astype(str).equals(rhs.astype(str)):
            return False
    return True


def _comparable_labels(df: pd.DataFrame | None) -> pd.DataFrame:
    """*df* reduced to the canonical columns, in a canonical order."""
    if df is None or df.empty:
        return _empty_all_labels()[TSV_COLUMNS]
    out = _ensure_label_columns(df.copy())
    for col in TSV_COLUMNS:
        if col not in out.columns:
            out[col] = TRIAL_META_DEFAULTS.get(col, "")
    out = out[TSV_COLUMNS]
    return out.sort_values(["trial", "onset_s", "labels"], kind="stable").reset_index(drop=True)


def _empty_all_labels() -> pd.DataFrame:
    df = empty_intervals()
    df.insert(0, "trial", pd.Series(dtype=object))
    for col, default in TRIAL_META_DEFAULTS.items():
        df[col] = pd.Series(dtype=type(default))
    return df


# ---------------------------------------------------------------------------
# Per-trial access
# ---------------------------------------------------------------------------


def get_trial_from_tsv(all_df: pd.DataFrame, trial) -> pd.DataFrame:
    """Extract all rows for a single trial from the all-labels DataFrame.

    Returns a DataFrame with the full ``TSV_COLUMNS`` set: trial, individual,
    labels, onset_s, offset_s, plus per-trial metadata columns.  Callers that
    only need interval data can ignore the extras; ``set_trial_in_tsv`` already
    discards everything except ``INTERVAL_COLUMNS`` when writing back.
    """
    if all_df is None or all_df.empty:
        return empty_intervals()
    mask = all_df["trial"] == trial
    cols = [c for c in TSV_COLUMNS if c in all_df.columns]
    trial_df = all_df.loc[mask, cols].reset_index(drop=True)
    if trial_df.empty:
        return empty_intervals()
    # A table built before a per-label column existed still hands out rows
    # carrying every INTERVAL_COLUMN, so callers can index them blindly.
    return _ensure_label_columns(trial_df)


def set_trial_in_tsv(
    all_df: pd.DataFrame,
    trial,
    trial_df: pd.DataFrame,
) -> pd.DataFrame:
    """Replace all rows for a trial in the all-labels DataFrame.

    Preserves per-trial metadata columns from the existing rows.  The columns
    added to files written before they existed (``event_type``,
    ``individual_rec``, ``confidence``, ``labeling_method``) are filled in on
    the *whole* table first: the untouched trials must not end up with NaN
    where the rewritten trial has a value.  Any other per-trial column a
    loaded file carries (a legacy ``human_verified``, say) rides along with
    the trial's previous value, so rewriting one trial never blanks it.
    """
    if all_df is None:
        all_df = _empty_all_labels()
    all_df = _ensure_label_columns(all_df.copy())

    # Preserve existing meta values for this trial
    old_meta = get_trial_meta(all_df, trial)
    old_rows = all_df[all_df["trial"] == trial]
    other = all_df[all_df["trial"] != trial]

    trial_df = _ensure_label_columns(trial_df.copy())
    new_rows = trial_df[INTERVAL_COLUMNS].copy()
    new_rows.insert(0, "trial", trial)
    for col, default in TRIAL_META_DEFAULTS.items():
        new_rows[col] = old_meta.get(col, default)
    for col in all_df.columns:
        if col in new_rows.columns:
            continue
        new_rows[col] = old_rows[col].iloc[0] if not old_rows.empty else None

    result = pd.concat([other, new_rows], ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Per-trial metadata (stored as columns, same value per trial)
# ---------------------------------------------------------------------------


def get_trial_meta(all_df: pd.DataFrame, trial) -> dict:
    """Read per-trial metadata from columns. Returns dict with defaults for missing trials."""
    if all_df is None or all_df.empty:
        return dict(TRIAL_META_DEFAULTS)
    mask = all_df["trial"] == trial
    rows = all_df.loc[mask]
    if rows.empty:
        return dict(TRIAL_META_DEFAULTS)
    first = rows.iloc[0]
    result = {}
    for col, default in TRIAL_META_DEFAULTS.items():
        val = first.get(col, default)
        if col == "prediction_source":
            result[col] = str(val) if pd.notna(val) else ""
        else:
            result[col] = int(val) if pd.notna(val) else default
    return result


def set_trial_meta_attr(all_df: pd.DataFrame, trial, key: str, value) -> pd.DataFrame:
    """Set a per-trial metadata column value for all rows of a trial."""
    if all_df is None:
        return all_df
    mask = all_df["trial"] == trial
    if mask.any():
        all_df.loc[mask, key] = value
    return all_df


def init_empty_labels(trials: list) -> pd.DataFrame:
    """Create empty labels DataFrame."""
    return _empty_all_labels()


# ---------------------------------------------------------------------------
# Undo history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelEdit:
    """One trial's labels as they stood *before* a single edit.

    ``rows`` carries :data:`~ethograph.labels.intervals.INTERVAL_COLUMNS` only
    (so each row's ``labeling_method`` is restored with it); ``meta`` the
    per-trial flags, so undoing an edit also takes back the flags it set.
    """

    trial: object
    description: str
    rows: pd.DataFrame
    meta: dict


class LabelHistory:
    """Bounded undo stack over the label table.

    A snapshot holds **one trial's rows**, never the whole table: every trial's
    labels share one DataFrame, so copying all of them per click would cost
    with the size of the dataset instead of the size of the edit. Depth is
    bounded, so a long labelling session cannot grow without limit.

    One entry is one *user action*: the caller records once, before the action,
    and everything that action goes on to do (interval trimming, sliver purge,
    changepoint correction) is inside that single step.
    """

    def __init__(self, max_depth: int = 200) -> None:
        self._stack: deque[LabelEdit] = deque(maxlen=max_depth)

    def __len__(self) -> int:
        return len(self._stack)

    def clear(self) -> None:
        self._stack.clear()

    def peek(self) -> LabelEdit | None:
        """What the next :meth:`undo` would take back, without taking it back."""
        return self._stack[-1] if self._stack else None

    def record(self, all_df: pd.DataFrame | None, trial, description: str) -> None:
        """Snapshot *trial*'s rows as they are now, before an edit changes them."""
        rows = get_trial_from_tsv(all_df, trial)
        cols = [c for c in INTERVAL_COLUMNS if c in rows.columns]
        self._stack.append(
            LabelEdit(
                trial=trial,
                description=description,
                rows=rows[cols].copy(),
                meta=get_trial_meta(all_df, trial),
            )
        )

    def undo(self, all_df: pd.DataFrame | None) -> tuple[pd.DataFrame, LabelEdit] | None:
        """Put the newest snapshot back.

        Returns the restored table and the edit it took back, or ``None`` when
        there is nothing left to undo.
        """
        if not self._stack:
            return None
        edit = self._stack.pop()
        restored = set_trial_in_tsv(all_df, edit.trial, edit.rows)
        for key, value in edit.meta.items():
            restored = set_trial_meta_attr(restored, edit.trial, key, value)
        return restored, edit
