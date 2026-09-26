"""Label curation: who vouches for a label, and what "reviewed" means per trial.

Every label row carries a ``labeling_method`` (the ndx-ethogram vocabulary):

* :data:`~ethograph.labels.intervals.LABELING_MANUAL` — a human placed or
  last edited it.
* :data:`~ethograph.labels.intervals.LABELING_AUTOMATED` — a model produced it
  and nobody has looked at it.
* :data:`~ethograph.labels.intervals.LABELING_CURATED` — automated output a
  human looked at and let stand.

Curation is the transition *automated → curated*. It never touches a manual
label (a human already vouched for it, more strongly), and it never runs
backwards: a curated label goes back to *automated* only by re-running a model
over a trial that lost it, and to *manual* only through an edit.

A trial is **curated** when none of its labels is still automated — a trial
with no labels at all is trivially curated. That per-trial answer is what the
metadata table's :data:`CURATED_COLUMN` and the trial colouring in the GUI
show. Everything here is pandas-only so the rules are unit-testable without
Qt; the GUI (``gui/widgets_curation.py``) decides *when* to apply them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ethograph.labels.intervals import (
    EVENT_TYPE_POINT,
    EVENT_TYPE_STATE,
    LABELING_AUTOMATED,
    LABELING_CURATED,
    LABELING_MANUAL,
    LABELING_METHODS,
    SUBJECT_COLUMNS,
    ensure_labeling_method,
    stitch_intervals,
)

#: Metadata-table column holding the per-trial verdict: "yes" when every
#: label of the trial is manual or curated, "no" while any is still
#: automated. String-valued (not 0/1) so the trials table's funnel filter
#: treats it as categorical (a yes/no checklist) rather than a numeric range.
CURATED_COLUMN = "curated"
CURATED_YES = "yes"
CURATED_NO = "no"

#: Visit order of the boundaries of one label: START before END. A "label"
#: target is the whole label (segment review), one per row.
FIELD_LABEL = "label"
FIELD_RANK = {"point": 0, "start": 0, "end": 1, FIELD_LABEL: 0}

#: How :func:`build_review_queue` orders the boundaries in scope.
#: "trial" walks every boundary of a trial before moving to the next trial
#: (time order within, arbitrary class order); "label" walks every instance
#: of one class, across every trial, before moving to the next class.
REVIEW_ORDER_TRIAL = "trial"
REVIEW_ORDER_LABEL = "label"
REVIEW_ORDERS = (REVIEW_ORDER_TRIAL, REVIEW_ORDER_LABEL)


# ---------------------------------------------------------------------------
# Row identity
# ---------------------------------------------------------------------------


def subject_str(value) -> str:
    """Subject columns compare as text; ``None`` and NaN are the same blank."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def label_key(row) -> tuple:
    """The identity of one label row: (trial, class, onset, actor, recipient).

    Row indices do not survive an edit (every ``set_trial_intervals`` rebuilds
    the trial's rows), but these values do — they are what every consumer
    that has to find "this label again" matches on.
    """
    return (
        str(row["trial"]),
        int(row["labels"]),
        round(float(row["onset_s"]), 6),
        subject_str(row.get("individual")),
        subject_str(row.get("individual_rec")),
    )


def row_mask(df: pd.DataFrame, inst: dict) -> pd.Series:
    """Locate *inst*'s row in a labels DataFrame (class + subject + onset).

    *inst* carries ``labels``, ``onset_s`` and the subject columns; ``trial``
    is not compared, so this works on the current trial's
    ``label_intervals`` as well as on the whole table.
    """
    mask = df["labels"] == inst["labels"]
    mask &= np.isclose(df["onset_s"].astype(float), float(inst["onset_s"]), atol=1e-6)
    for col in SUBJECT_COLUMNS:
        if col not in df.columns:
            continue
        val = inst.get(col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            mask &= df[col].isna() | (df[col].astype(str) == "")
        else:
            mask &= df[col].astype(str) == str(val)
    return mask


def trial_mask(df: pd.DataFrame, trial) -> pd.Series:
    """Rows of *trial*, compared as text (tables disagree on int vs str ids)."""
    return df["trial"].astype(str) == str(trial)


def scope_mask(df: pd.DataFrame, label_ids: set[int] | None) -> pd.Series:
    """Rows whose class is in *label_ids*; ``None`` (or empty) means every class."""
    if not label_ids:
        return pd.Series(True, index=df.index)
    return df["labels"].isin({int(i) for i in label_ids})


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def set_method(all_df: pd.DataFrame | None, mask: pd.Series, method: str) -> tuple[pd.DataFrame | None, int]:
    """Stamp *method* on the rows *mask* selects. Returns (table, rows changed)."""
    if method not in LABELING_METHODS:
        raise ValueError(f"Unknown labeling_method {method!r}; expected one of {LABELING_METHODS}")
    if all_df is None or all_df.empty:
        return all_df, 0
    df = ensure_labeling_method(all_df.copy())
    change = mask & (df["labeling_method"] != method)
    n = int(change.sum())
    if n:
        df.loc[change, "labeling_method"] = method
    return df, n


def curate_rows(all_df: pd.DataFrame | None, mask: pd.Series) -> tuple[pd.DataFrame | None, int]:
    """Automated → curated on the rows *mask* selects; manual rows stay manual."""
    if all_df is None or all_df.empty:
        return all_df, 0
    df = ensure_labeling_method(all_df.copy())
    return set_method(df, mask & (df["labeling_method"] == LABELING_AUTOMATED), LABELING_CURATED)


def curate_trial(
    all_df: pd.DataFrame | None,
    trial,
    label_ids: set[int] | None = None,
) -> tuple[pd.DataFrame | None, int]:
    """Curate every automated label of *trial* (within *label_ids*, if given).

    This is what "the human has seen this trial" means — Ctrl+C in the GUI,
    or merely opening the trial in *Inspect is enough* mode. Manual labels are
    never rewritten as curated: a human made them, which says more.
    """
    if all_df is None or all_df.empty:
        return all_df, 0
    return curate_rows(all_df, trial_mask(all_df, trial) & scope_mask(all_df, label_ids))


def curate_trials(
    all_df: pd.DataFrame | None,
    trials,
    label_ids: set[int] | None = None,
) -> tuple[pd.DataFrame | None, int]:
    """Curate every automated label of *trials* (within *label_ids*, if given).

    The multi-trial sibling of :func:`curate_trial`, in the same shape as
    :func:`delete_labels`/:func:`purge_short_labels` — one call over the set
    rather than a loop.
    """
    if all_df is None or all_df.empty:
        return all_df, 0
    trial_strs = {str(t) for t in trials}
    mask = all_df["trial"].astype(str).isin(trial_strs) & scope_mask(all_df, label_ids)
    return curate_rows(all_df, mask)


def curate_label(all_df: pd.DataFrame | None, inst: dict) -> tuple[pd.DataFrame | None, int]:
    """Curate one label (identified as :func:`row_mask` does, within its trial)."""
    if all_df is None or all_df.empty:
        return all_df, 0
    return curate_rows(all_df, trial_mask(all_df, inst["trial"]) & row_mask(all_df, inst))


def mark_manual(all_df: pd.DataFrame | None, inst: dict) -> tuple[pd.DataFrame | None, int]:
    """A human edited this label: whatever it was, it is manual now."""
    if all_df is None or all_df.empty:
        return all_df, 0
    return set_method(all_df, trial_mask(all_df, inst["trial"]) & row_mask(all_df, inst), LABELING_MANUAL)


# ---------------------------------------------------------------------------
# Bulk deletion
# ---------------------------------------------------------------------------


def delete_rows(all_df: pd.DataFrame | None, mask: pd.Series) -> tuple[pd.DataFrame | None, int]:
    """Drop the rows *mask* selects. Returns (table, rows deleted)."""
    if all_df is None or all_df.empty:
        return all_df, 0
    n = int(mask.sum())
    if not n:
        return all_df, 0
    return all_df.loc[~mask].reset_index(drop=True), n


def delete_labels(
    all_df: pd.DataFrame | None,
    trials,
    label_ids: set[int] | None = None,
) -> tuple[pd.DataFrame | None, int]:
    """Delete every label of *trials* (within *label_ids*, if given).

    Unlike :func:`curate_trial`, this drops the rows outright regardless of
    ``labeling_method`` — the event is gone, not marked reviewed. Used for
    bulk clean-up (e.g. every trial where a metadata column has a given
    value never got a real recording of this behaviour).
    """
    if all_df is None or all_df.empty:
        return all_df, 0
    trial_strs = {str(t) for t in trials}
    mask = all_df["trial"].astype(str).isin(trial_strs) & scope_mask(all_df, label_ids)
    return delete_rows(all_df, mask)


def purge_short_labels(
    all_df: pd.DataFrame | None,
    trials,
    min_duration_s: float,
    label_ids: set[int] | None = None,
) -> tuple[pd.DataFrame | None, int]:
    """Drop state-interval labels of *trials* shorter than *min_duration_s*.

    Point events have no duration and are never touched. The multi-trial,
    scoped sibling of :func:`~ethograph.labels.intervals.purge_short_intervals`
    (which purges one trial's own rows); like :func:`delete_labels`, this
    drops rows regardless of ``labeling_method``.
    """
    if all_df is None or all_df.empty:
        return all_df, 0
    trial_strs = {str(t) for t in trials}
    in_scope = all_df["trial"].astype(str).isin(trial_strs) & scope_mask(all_df, label_ids)
    is_state = all_df["event_type"] == EVENT_TYPE_STATE
    durations = all_df["offset_s"].astype(float) - all_df["onset_s"].astype(float)
    short = in_scope & is_state & (durations < min_duration_s)
    return delete_rows(all_df, short)


def stitch_labels(
    trial_df: pd.DataFrame,
    max_gap_s: float,
    label_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Merge same-class state labels of one trial whose gap is under *max_gap_s*.

    Only rows whose class is in *label_ids* are candidates; every other row
    (other classes, point events) passes through untouched. The merge rule
    itself is :func:`~ethograph.labels.intervals.stitch_intervals`, the one
    the changepoint correction uses. Returns (table, rows absorbed).
    """
    if trial_df is None or trial_df.empty:
        return trial_df, 0
    candidate = scope_mask(trial_df, label_ids) & (trial_df["event_type"] == EVENT_TYPE_STATE)
    if not candidate.any():
        return trial_df, 0
    stitched = stitch_intervals(trial_df[candidate].copy(), max_gap_s)
    n = int(candidate.sum()) - len(stitched)
    if not n:
        return trial_df, 0
    out = pd.concat([trial_df[~candidate], stitched], ignore_index=True)
    out.sort_values("onset_s", inplace=True, kind="stable", na_position="last")
    return out.reset_index(drop=True), n


# ---------------------------------------------------------------------------
# Per-trial verdicts
# ---------------------------------------------------------------------------


def method_counts(all_df: pd.DataFrame | None, trial) -> dict[str, int]:
    """How many labels of *trial* are manual / automated / curated."""
    counts = dict.fromkeys(LABELING_METHODS, 0)
    if all_df is None or all_df.empty:
        return counts
    df = ensure_labeling_method(all_df.copy())
    rows = df[trial_mask(df, trial)]
    for method, n in rows["labeling_method"].value_counts().items():
        counts[str(method)] = int(n)
    return counts


def trial_curation_status(all_df: pd.DataFrame | None, trials) -> dict[str, bool]:
    """``{str(trial): curated?}`` for every trial in *trials*.

    A trial is curated when none of its labels is automated; a trial without
    labels has nothing left to look at and counts as curated.
    """
    status = {str(t): True for t in trials}
    if all_df is None or all_df.empty:
        return status
    df = ensure_labeling_method(all_df.copy())
    pending = df.loc[df["labeling_method"] == LABELING_AUTOMATED, "trial"].astype(str).unique()
    for trial in pending:
        if trial in status:
            status[trial] = False
    return status


def curated_column(metadata_df: pd.DataFrame | None, status: dict[str, bool]) -> pd.DataFrame | None:
    """*metadata_df* with :data:`CURATED_COLUMN` set from *status* (a copy).

    ``None`` in, ``None`` out: there is no table to put the verdict in. A
    trial the status does not mention keeps whatever the column held (or
    gets :data:`CURATED_NO` when the column is new).
    """
    if metadata_df is None:
        return None
    df = metadata_df.copy()
    if CURATED_COLUMN not in df.columns:
        df[CURATED_COLUMN] = CURATED_NO
    values = df["trial"].astype(str).map(lambda t: (CURATED_YES if status[t] else CURATED_NO) if t in status else None)
    df[CURATED_COLUMN] = values.where(values.notna(), df[CURATED_COLUMN]).fillna(CURATED_NO)
    return df


def _is_curated_value(value) -> bool:
    """Read a (possibly legacy int-valued) curated cell as a bool."""
    if isinstance(value, str):
        return value.strip().lower() == CURATED_YES
    if pd.isna(value):
        return False
    return bool(value)


def curated_column_differs(metadata_df: pd.DataFrame | None, status: dict[str, bool]) -> bool:
    """Whether writing *status* into *metadata_df* would change anything."""
    if metadata_df is None:
        return False
    if CURATED_COLUMN not in metadata_df.columns:
        return True
    current = dict(zip(metadata_df["trial"].astype(str), metadata_df[CURATED_COLUMN]))
    for trial, curated in status.items():
        if trial not in current:
            continue
        value = current[trial]
        if pd.isna(value) or _is_curated_value(value) != bool(curated):
            return True
    return False


# ---------------------------------------------------------------------------
# Frame-by-frame review queue
# ---------------------------------------------------------------------------


@dataclass
class ReviewTarget:
    """One boundary to review frame by frame, or one whole label to review as
    a segment.

    ``inst`` is shared between the start and end targets of the same state
    event, so committing a new start updates the onset the end target (and
    the row lookup) will use.
    """

    inst: dict
    field: str  # "point" | "start" | "end" | "label"


def _inst_from_row(row) -> dict:
    return {
        "trial": row["trial"],
        "labels": int(row["labels"]),
        "onset_s": float(row["onset_s"]),
        "offset_s": float(row["offset_s"]),
        "individual": row.get("individual"),
        "individual_rec": row.get("individual_rec"),
        "event_type": row.get("event_type", "state"),
    }


def _targets_for_inst(inst: dict, *, whole: bool = False) -> list[ReviewTarget]:
    if whole:
        return [ReviewTarget(inst, FIELD_LABEL)]
    is_point = inst["event_type"] == EVENT_TYPE_POINT or not math.isfinite(inst["offset_s"])
    if is_point:
        return [ReviewTarget(inst, "point")]
    return [ReviewTarget(inst, "start"), ReviewTarget(inst, "end")]


def build_review_queue(
    all_df: pd.DataFrame | None,
    label_ids: set[int] | None,
    *,
    individual: str | None = None,
    allowed_trials: set[str] | None = None,
    automated_only: bool = False,
    order: str = REVIEW_ORDER_TRIAL,
    whole_labels: bool = False,
) -> list[ReviewTarget]:
    """Every boundary of the labels in scope, sorted per *order*.

    One target per point event, a start then an end target per state event;
    with *whole_labels* one ``"label"`` target per row instead (the segment
    review edits both boundaries of a label in one go). *order*
    (:data:`REVIEW_ORDERS`) is "trial" (each trial visited once, in time
    order — the default) or "label" (each class visited once, across every
    trial in time order, before the next class). *allowed_trials* (as
    strings) is the trials-table filter. *automated_only* skips manual and
    already-curated labels — a human already vouched for those, so a
    from-scratch review has nothing to add.
    """
    if all_df is None or all_df.empty:
        return []
    if order not in REVIEW_ORDERS:
        raise ValueError(f"Unknown review order {order!r}; expected one of {REVIEW_ORDERS}")
    mask = scope_mask(all_df, label_ids)
    if individual is not None and "individual" in all_df.columns:
        mask &= all_df["individual"].astype(str) == str(individual)
    if allowed_trials is not None:
        mask &= all_df["trial"].astype(str).isin(allowed_trials)
    df = all_df
    if automated_only:
        df = ensure_labeling_method(all_df.copy())
        mask &= df["labeling_method"] == LABELING_AUTOMATED
    sort_cols = ["labels", "trial", "onset_s"] if order == REVIEW_ORDER_LABEL else ["trial", "onset_s"]
    rows = df[mask].sort_values(sort_cols)
    targets: list[ReviewTarget] = []
    for _, row in rows.iterrows():
        targets.extend(_targets_for_inst(_inst_from_row(row), whole=whole_labels))
    return targets


def targets_from_seeds(seeds: list[dict]) -> list[ReviewTarget]:
    """Build a queue from boundaries chosen elsewhere (the label grid).

    Each seed is a label row (``trial``, ``labels``, ``onset_s``, ``offset_s``,
    the subject columns, ``event_type``) plus the ``field`` to edit. Sorted
    (trial, onset) with START before END; the two boundaries of one state
    event share a single ``inst``, exactly as in :func:`build_review_queue`.
    """
    ordered = sorted(
        seeds,
        key=lambda s: (str(s["trial"]), float(s["onset_s"]), FIELD_RANK.get(s.get("field", "point"), 0)),
    )
    insts: dict[tuple, dict] = {}
    targets: list[ReviewTarget] = []
    for seed in ordered:
        key = (
            str(seed["trial"]),
            int(seed["labels"]),
            round(float(seed["onset_s"]), 6),
            subject_str(seed.get("individual")),
            subject_str(seed.get("individual_rec")),
        )
        inst = insts.get(key)
        if inst is None:
            offset = seed.get("offset_s")
            inst = {
                "trial": seed["trial"],
                "labels": int(seed["labels"]),
                "onset_s": float(seed["onset_s"]),
                "offset_s": float(offset) if offset is not None else float("nan"),
                "individual": seed.get("individual"),
                "individual_rec": seed.get("individual_rec"),
                "event_type": seed.get("event_type", "state"),
            }
            insts[key] = inst
        targets.append(ReviewTarget(inst, seed.get("field", "point")))
    return targets


def queue_index_of(targets: list[ReviewTarget], inst: dict, field: str | None = None) -> int | None:
    """Position of *inst* (and *field*, when given) in *targets*, else None."""
    for i, target in enumerate(targets):
        same = (
            str(target.inst["trial"]) == str(inst["trial"])
            and target.inst["labels"] == int(inst["labels"])
            and abs(float(target.inst["onset_s"]) - float(inst["onset_s"])) <= 1e-6
            and subject_str(target.inst.get("individual")) == subject_str(inst.get("individual"))
            and subject_str(target.inst.get("individual_rec")) == subject_str(inst.get("individual_rec"))
        )
        if same and (field is None or target.field == field):
            return i
    return None
