"""Computed columns added to the labels table on save.

Everything here holds for any dataset: a label's duration, its place in the
trial's sequence, the trial's timing on the session clock, and whatever the
trial metadata table carries.

Columns that only mean something for one lab do not belong here. They live in
``labels/exporters/`` as a named exporter the user picks in the export
settings; this module hands it the enriched table and the session's context.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ethograph.labels.exporters import ExportContext, Exporter, run
from ethograph.labels.intervals import TRACK_COLUMNS

logger = logging.getLogger(__name__)

#: Metadata columns describing the *curation* rather than the trial, so
#: exporting them would say nothing about the data.
_SKIP_METADATA_COLUMNS = frozenset(
    {
        "trial",
        "offsets_corrected",
        "small_labels_purged",
        "model_confidence",
        "model_confidence_level",
    }
)


def _trial_timing(nwb_alignment, trial_id) -> tuple[float | None, float | None]:
    """The trial's start and stop on the session clock, or ``(None, None)``.

    A session with no alignment has no session clock, so the global columns are
    simply absent. A trial the alignment does not know is worth a warning: the
    labels and the alignment disagree about which trials exist.
    """
    if nwb_alignment is None:
        return None, None
    try:
        return nwb_alignment.start_time(trial_id), nwb_alignment.stop_time(trial_id)
    except (KeyError, ValueError):
        logger.warning("Alignment has no timing for trial %r; exporting it without session times.", trial_id)
        return None, None


def _trial_metadata(metadata_df: pd.DataFrame | None, trial_id) -> dict[str, object]:
    """The metadata row for *trial_id*, keyed by column; empty when there is none.

    The table joins on a ``trial`` column when it has one, and on the index
    otherwise.
    """
    if metadata_df is None or metadata_df.empty:
        return {}
    if "trial" in metadata_df.columns:
        rows = metadata_df[metadata_df["trial"] == trial_id]
    elif trial_id in metadata_df.index:
        rows = metadata_df.loc[[trial_id]]
    else:
        return {}
    if rows.empty:
        return {}
    first = rows.iloc[0]
    return {col: first[col] for col in rows.columns if col not in _SKIP_METADATA_COLUMNS}


def _enrich_trial(group: pd.DataFrame, trial_id, nwb_alignment, metadata_df) -> pd.DataFrame:
    """One trial's labels, with its sequence, its timing and its metadata columns."""
    group = group.sort_values("onset_s").reset_index(drop=True)

    group["sequence_idx"] = range(len(group))
    group["sequence"] = "-".join(str(label) for label in group["labels"])

    t_start, t_stop = _trial_timing(nwb_alignment, trial_id)
    if t_start is not None:
        group["trial_onset"] = t_start
        group["onset_global"] = t_start + group["onset_s"]
        group["offset_global"] = t_start + group["offset_s"]
    if t_stop is not None:
        group["trial_offset"] = t_stop

    for col, value in _trial_metadata(metadata_df, trial_id).items():
        group[col] = value

    return group


def enrich_labels_df(
    all_labels_df: pd.DataFrame,
    nwb_alignment=None,
    dt=None,
    metadata_df: pd.DataFrame | None = None,
    exporter: Exporter | None = None,
    session_dir: Path | None = None,
) -> pd.DataFrame:
    """Add the computed analysis columns to a raw labels table.

    Adds ``duration``, ``sequence``, ``sequence_idx``, the trial's timing on the
    session clock (``trial_onset``, ``trial_offset``, ``onset_global``,
    ``offset_global``) and every trial metadata column.

    Parameters
    ----------
    all_labels_df : pd.DataFrame
        Raw labels with required columns: onset_s, offset_s, labels, individual, trial.
    nwb_alignment
        Session alignment, for trial timing. Without one the global columns are
        left out.
    dt : TrialTree, optional
        Handed to *exporter*; unused otherwise.
    metadata_df : pd.DataFrame, optional
        Trial metadata table, joined per trial.
    exporter : callable, optional
        The lab's own columns (``labels/exporters/``). Without one, only the
        columns above are added.
    session_dir : Path, optional
        Handed to *exporter*, for a sidecar file beside the session.

    Returns
    -------
    pd.DataFrame
        One row per non-background segment. Empty when there is nothing to export.
    """
    if all_labels_df is None or all_labels_df.empty:
        return pd.DataFrame()

    valid = all_labels_df[all_labels_df["labels"] > 0].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["duration"] = valid["offset_s"] - valid["onset_s"]

    enriched = [
        _enrich_trial(group, trial_id, nwb_alignment, metadata_df)
        for trial_id, group in valid.groupby("trial", sort=False)
    ]
    df = pd.concat(enriched, ignore_index=True)

    if exporter is not None:
        context = ExportContext(dt=dt, alignment=nwb_alignment, metadata_df=metadata_df, session_dir=session_dir)
        df = run(exporter, df, context)

    return df.sort_values(["trial", "onset_s"]).reset_index(drop=True)


def correct_offsets_trial(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Apply gap correction to a single trial's interval DataFrame.

    For each actor (``TRACK_COLUMNS`` --- the recipient is an attribute, not a
    second track), pulls back ``offset_s`` when the gap to the next onset is
    smaller than ``eps`` so pynapple can resolve all intervals.

    Works on the per-trial format (columns: trial, onset_s, offset_s, labels,
    individual) returned by ``app_state.get_trial_intervals()``.

    Returns
    -------
    tuple[pd.DataFrame, int, int]
        Corrected DataFrame, number of offsets corrected, number of negative gaps found.
    """
    if df.empty:
        return df, 0, 0
    eps = 1e-4
    subject = [c for c in TRACK_COLUMNS if c in df.columns]
    df = df.copy().sort_values([*subject, "onset_s"]).reset_index(drop=True)

    corrected = 0
    negative_gaps = 0
    for _, group in df.groupby(subject):
        idx = group.index.tolist()
        for i in range(len(idx) - 1):
            gap = df.loc[idx[i + 1], "onset_s"] - df.loc[idx[i], "offset_s"]

            if gap < 0:
                negative_gaps += 1

            if gap < eps:
                corrected += 1
                df.loc[idx[i], "offset_s"] = df.loc[idx[i + 1], "onset_s"] - eps
                df.loc[idx[i], "duration"] = df.loc[idx[i], "offset_s"] - df.loc[idx[i], "onset_s"]

                if "offset_global" in df.columns and "onset_global" in df.columns:
                    df.loc[idx[i], "offset_global"] = df.loc[idx[i + 1], "onset_global"] - eps
                    df.loc[idx[i], "duration"] = df.loc[idx[i], "offset_s"] - df.loc[idx[i], "onset_s"]

    if "onset_global" in df.columns:
        df.sort_values([*subject, "onset_global"], inplace=True)

    return df, corrected, negative_gaps
