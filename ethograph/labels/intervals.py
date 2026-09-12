"""Interval-based label representation and core primitives for EthoGraph.

Labels are stored as a pandas DataFrame with columns:
    onset_s        (float64) - start time in seconds
    offset_s       (float64) - end time in seconds (NaN for point events)
    labels         (int32)   - label class ID (nonzero)
    individual     (str)     - the individual performing the behaviour (actor)
    individual_rec (str)     - the recipient of a dyadic behaviour, "" if none
    event_type     (str)     - "state" (interval) or "point" (instantaneous)
    confidence     (float64) - how sure the label is: 1.0 for a human label,
                               the model's own score for a predicted one
    labeling_method (str)    - who vouches for the label: "manual" (placed or
                               edited by hand), "automated" (a model's output
                               nobody has looked at) or "curated" (automated,
                               then approved by a human) — the vocabulary of
                               ndx-ethogram

``individual`` and ``individual_rec`` together are the **subject** of a label:
each (actor, recipient) pair is its own independent track, exactly as each
individual was on its own before recipients existed.  ``NO_RECIPIENT`` ("")
means a solo behaviour — the default, and what every pre-recipient file reads
back as.

Use ``split_by_kind(df)`` at the top of every interval operation so points
pass through untouched. ``states_only(df)`` / ``points_only(df)`` are for
read-only consumers (dense conversion, plot rendering).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────

EVENT_TYPE_STATE = "state"
EVENT_TYPE_POINT = "point"
EVENT_TYPES = (EVENT_TYPE_STATE, EVENT_TYPE_POINT)

#: Value of ``individual_rec`` for a behaviour with no recipient.
NO_RECIPIENT = ""

#: Confidence of a label placed by a human — certain by definition. Every
#: other producer (a model) writes its own score in ``[0, 1]``.
HUMAN_CONFIDENCE = 1.0

#: ``labeling_method`` values (the ndx-ethogram vocabulary). A label is
#: *manual* when a human placed or last edited it, *automated* when a model
#: produced it and nobody has looked at it yet, and *curated* once a human has
#: looked at an automated label and let it stand.
LABELING_MANUAL = "manual"
LABELING_AUTOMATED = "automated"
LABELING_CURATED = "curated"
LABELING_METHODS = (LABELING_MANUAL, LABELING_AUTOMATED, LABELING_CURATED)

INTERVAL_COLUMNS = [
    "onset_s",
    "offset_s",
    "labels",
    "individual",
    "individual_rec",
    "event_type",
    "confidence",
    "labeling_method",
]

INTERVAL_DTYPES = {
    "onset_s": np.float64,
    "offset_s": np.float64,
    "labels": np.int32,
    "individual": object,
    "individual_rec": object,
    "event_type": object,
    "confidence": np.float64,
    "labeling_method": object,
}

#: The columns identifying whose label a row is — the actor and the recipient.
#: What identifies a label row's subject: who acted, and towards whom.
SUBJECT_COLUMNS = ["individual", "individual_rec"]

#: The unit of exclusivity: one actor never does two things at once (within a
#: branch). The receiver is an attribute of the label, not a second track — a
#: label directed at another animal and a solo one of the same actor exclude
#: each other exactly like two solo ones do.
TRACK_COLUMNS = ["individual"]


# ── Empty DataFrame ──────────────────────────────────────────────────────


def empty_intervals() -> pd.DataFrame:
    """Create an empty intervals DataFrame with the correct columns and dtypes.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with the :data:`INTERVAL_COLUMNS`.

    Examples
    --------
    >>> from ethograph.labels.intervals import empty_intervals
    >>> df = empty_intervals()
    >>> df.columns.tolist()[:3]
    ['onset_s', 'offset_s', 'labels']
    >>> len(df)
    0
    """
    return pd.DataFrame({col: pd.Series(dtype=INTERVAL_DTYPES[col]) for col in INTERVAL_COLUMNS})


# ── Event-kind helpers (use these to avoid forgetting guards) ───────────


def ensure_event_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``event_type`` column defaulting to ``"state"`` if missing.

    Mutates and returns *df*.  Use on freshly-loaded DataFrames so downstream
    code can rely on the column existing.
    """
    if "event_type" not in df.columns:
        df["event_type"] = EVENT_TYPE_STATE
    else:
        df["event_type"] = df["event_type"].fillna(EVENT_TYPE_STATE).astype(object)
    return df


def ensure_individual_rec(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``individual_rec`` column defaulting to :data:`NO_RECIPIENT`.

    Mutates and returns *df*.  Every label file written before recipients
    existed reads back as a table of solo behaviours, which is what it is.
    """
    if "individual_rec" not in df.columns:
        df["individual_rec"] = NO_RECIPIENT
    else:
        df["individual_rec"] = df["individual_rec"].fillna(NO_RECIPIENT).astype(object)
    return df


def ensure_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``confidence`` column defaulting to :data:`HUMAN_CONFIDENCE`.

    Mutates and returns *df*.  A row with no confidence is a row nobody
    expressed a doubt about — every label file written before models scored
    their own output reads back as fully confident, which is what it is.
    """
    if "confidence" not in df.columns:
        df["confidence"] = HUMAN_CONFIDENCE
    else:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(HUMAN_CONFIDENCE)
    return df


def ensure_labeling_method(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``labeling_method`` column where it is missing or blank.

    Mutates and returns *df*.  A row with no method is read off its
    ``confidence``: a label written before methods existed is a hand-made one
    unless a model scored it below :data:`HUMAN_CONFIDENCE`, in which case it
    is automated output nobody has curated yet — the conservative reading, so
    a file from an older EthoGraph never claims more review than it got.
    """
    ensure_confidence(df)
    derived = pd.Series(
        np.where(df["confidence"].to_numpy(dtype=float) < HUMAN_CONFIDENCE, LABELING_AUTOMATED, LABELING_MANUAL),
        index=df.index,
        dtype=object,
    )
    if "labeling_method" not in df.columns:
        df["labeling_method"] = derived
    else:
        method = df["labeling_method"].astype(object)
        blank = method.isna() | ~method.isin(LABELING_METHODS)
        df["labeling_method"] = method.where(~blank, derived).astype(object)
    return df


def subject_mask(
    df: pd.DataFrame,
    individual: str | None,
    individual_rec: str | None = None,
) -> pd.Series:
    """Boolean mask selecting the rows belonging to one (actor, recipient) pair.

    ``None`` on either side means "any" — used as the selection fallback when
    the names in a loaded file don't match the current selection.  Passing
    :data:`NO_RECIPIENT` for *individual_rec* selects solo behaviours only,
    which is what makes each pair an independent track.
    """
    mask = pd.Series(True, index=df.index)
    if individual is not None:
        mask &= df["individual"].astype(str) == str(individual)
    if individual_rec is not None and "individual_rec" in df.columns:
        rec = df["individual_rec"].fillna(NO_RECIPIENT).astype(str)
        mask &= rec == str(individual_rec)
    return mask


def select_subject(
    df: pd.DataFrame,
    individual: str | None,
    individual_rec: str | None = None,
) -> pd.DataFrame:
    """The rows of *df* belonging to one (actor, recipient) pair."""
    if df is None or df.empty:
        return df
    return df[subject_mask(df, individual, individual_rec)]


def states_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are state events (intervals).

    Backwards-compatible: a DataFrame without ``event_type`` is treated as
    all-states.
    """
    if df.empty or "event_type" not in df.columns:
        return df
    return df[df["event_type"] == EVENT_TYPE_STATE]


def points_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are point events (instantaneous)."""
    if df.empty or "event_type" not in df.columns:
        return df.iloc[0:0]
    return df[df["event_type"] == EVENT_TYPE_POINT]


def split_by_kind(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into ``(states, points)``.

    Use at the top of any interval-aware operation::

        states, points = split_by_kind(df)
        # ... transform states ...
        return pd.concat([transformed_states, points], ignore_index=True)

    This is the canonical way to keep point events unaffected by interval
    operations (purge, stitch, snap, etc.).
    """
    return states_only(df), points_only(df)


def _recombine(transformed_states: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Concat transformed states with untouched points, sorted by onset."""
    if points.empty:
        return transformed_states.reset_index(drop=True)
    out = pd.concat([transformed_states, points], ignore_index=True)
    out.sort_values("onset_s", inplace=True, kind="stable", na_position="last")
    out.reset_index(drop=True, inplace=True)
    return out


# ── Mapping loaders ─────────────────────────────────────────────────────


def load_mapping(mapping_file: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load a class-name ↔ index mapping file.

    The file is whitespace-delimited with lines ``<index> <name>``.

    Parameters
    ----------
    mapping_file : str or Path
        Path to the mapping file.

    Returns
    -------
    class_to_idx : dict[str, int]
    idx_to_class : dict[int, str]

    Examples
    --------
    >>> class_to_idx, idx_to_class = load_mapping("mapping.txt")
    >>> class_to_idx["walk"]
    1
    >>> idx_to_class[1]
    'walk'
    """
    class_to_idx: dict[str, int] = {}
    idx_to_class: dict[int, str] = {}
    with open(mapping_file, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                idx = int(parts[0])
                class_name = parts[1]
                class_to_idx[class_name] = idx
                idx_to_class[idx] = class_name
    return class_to_idx, idx_to_class


_LABEL_COLORS = [
    [1, 1, 1],
    [255, 102, 178],
    [102, 158, 255],
    [153, 51, 255],
    [255, 51, 51],
    [102, 255, 102],
    [255, 153, 102],
    [0, 153, 0],
    [0, 0, 128],
    [255, 255, 0],
    [0, 204, 204],
    [128, 128, 0],
    [255, 0, 255],
    [255, 165, 0],
    [0, 128, 255],
    [7, 7, 215],
    [128, 0, 255],
    [255, 215, 0],
    [73, 113, 233],
    [255, 128, 0],
    [138, 34, 34],
    [188, 82, 223],
    [103, 176, 29],
    [220, 20, 60],
    [3, 243, 3],
    [147, 24, 147],
    [178, 111, 44],
    [16, 166, 166],
    [71, 197, 238],
    [255, 149, 114],
    [16, 89, 162],
    [26, 195, 68],
    [254, 216, 103],
    [0, 237, 118],
    [177, 177, 36],
    [73, 243, 200],
]

_GAP_COLOR = [128 / 255.0, 128 / 255.0, 128 / 255.0]


def load_label_mapping(
    mapping_file: str | Path = "mapping.txt",
    order: list[int] | None = None,
) -> Dict[int, Dict]:
    """Load a label mapping with colors for visualization.

    Parameters
    ----------
    mapping_file : str or Path
        Path to the mapping file. Each line is
        ``<id> <name> [<branch>] [<event_type>]`` where *branch* is an
        optional integer (default 0) grouping labels into branches for
        independent labeling, and *event_type* is ``"state"`` (default) or
        ``"point"``.
    order : list[int] or None
        Label IDs in the desired display sequence. If provided, overrides
        the default order (which follows label ID). Any ID not listed
        retains its default position.

    Returns
    -------
    dict[int, dict]
        ``{label_id: {"name": str, "color": ndarray(3,), "order": int,
        "branch": int, "event_type": str}}``.

    Raises
    ------
    FileNotFoundError
        If *mapping_file* does not exist.

    Examples
    --------
    >>> mappings = load_label_mapping("mapping.txt")
    >>> mappings[1]["name"]
    'walk'

    Reorder labels for display without changing the file::

        mappings = load_label_mapping("mapping.txt", order=[0, 3, 1, 2])

    Draw labelled intervals on a plot::

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        mappings = load_label_mapping("mapping.txt")
        fig, ax = plt.subplots()
        ax.plot(time, signal)
        for _, row in intervals_df.iterrows():
            color = mappings[int(row["labels"])]["color"]
            ax.axvspan(row["onset_s"], row["offset_s"], alpha=0.5, color=color)
        handles = [mpatches.Patch(color=m["color"], label=m["name"]) for m in mappings.values()]
        ax.legend(handles=handles)
        plt.show()
    """
    mapping_file = Path(mapping_file)
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    label_mappings: dict = {}
    with open(mapping_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if parts[0].startswith("("):
                nums = parts[0].strip("()").split(",")
                label_id = (int(nums[0]), int(nums[1]))
                label_mappings[label_id] = {
                    "name": parts[1],
                    "color": _GAP_COLOR,
                    "branch": 0,
                    "event_type": EVENT_TYPE_STATE,
                }
            else:
                label_id = int(parts[0])
                branch = int(parts[2]) if len(parts) >= 3 else 0
                event_type = parts[3] if len(parts) >= 4 else EVENT_TYPE_STATE
                if event_type not in EVENT_TYPES:
                    event_type = EVENT_TYPE_STATE
                label_mappings[label_id] = {
                    "name": parts[1],
                    "color": np.array(_LABEL_COLORS[label_id % len(_LABEL_COLORS)]) / 255.0,
                    "branch": branch,
                    "event_type": event_type,
                }

    if order is not None:
        label_mappings = {k: label_mappings[k] for k in order if k in label_mappings}

    return label_mappings


def save_label_mapping(mapping_file: str | Path, mappings: Dict[int, Dict]) -> None:
    """Write a label mapping back to disk, preserving branch and event_type.

    Lines have the form ``<id> <name> <branch> <event_type>`` for scalar IDs.
    The ``event_type`` column is omitted when it equals the default
    (``"state"``) so files stay backward-compatible with older readers.

    Parameters
    ----------
    mapping_file : str or Path
        Destination path.
    mappings : dict[int, dict]
        The mapping dict as returned by :func:`load_label_mapping`.
    """
    mapping_file = Path(mapping_file)
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for label_id, data in sorted(mappings.items(), key=lambda kv: kv[0] if isinstance(kv[0], int) else kv[0][0]):
        if isinstance(label_id, tuple):
            lines.append(f"({label_id[0]},{label_id[1]}) {data['name']} {data['order']}")
        else:
            event_type = data.get("event_type", EVENT_TYPE_STATE)
            base = f"{label_id} {data['name']} {data.get('branch', 0)}"
            if event_type != EVENT_TYPE_STATE:
                base = f"{base} {event_type}"
            lines.append(base)
    mapping_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Interval operations ─────────────────────────────────────────────────


def add_point(
    df: pd.DataFrame,
    time_s: float,
    labels: int,
    individual: str,
    individual_rec: str = NO_RECIPIENT,
    confidence: float = HUMAN_CONFIDENCE,
    labeling_method: str = LABELING_MANUAL,
) -> pd.DataFrame:
    """Add a point event (instantaneous label) at *time_s*.

    Point events are stored with ``offset_s = NaN`` and
    ``event_type = "point"``.  They are never trimmed, split, or merged by
    :func:`add_interval`, :func:`purge_short_intervals`,
    :func:`stitch_intervals`, or :func:`snap_boundaries`.

    *confidence* defaults to :data:`HUMAN_CONFIDENCE` and *labeling_method* to
    :data:`LABELING_MANUAL` — a hand-placed label; a model passes its own
    score and :data:`LABELING_AUTOMATED` so a later curation pass knows what
    still needs a look.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame sorted by ``onset_s``.
    """
    new_row = {
        "onset_s": float(time_s),
        "offset_s": float("nan"),
        "labels": int(labels),
        "individual": individual,
        "individual_rec": individual_rec,
        "event_type": EVENT_TYPE_POINT,
        "confidence": float(confidence),
        "labeling_method": str(labeling_method),
    }
    new_df = pd.DataFrame([new_row])
    df = ensure_labeling_method(ensure_individual_rec(df.copy()))
    for col, dtype in INTERVAL_DTYPES.items():
        new_df[col] = new_df[col].astype(dtype)
    result = pd.concat([df, new_df], ignore_index=True)
    result.sort_values("onset_s", inplace=True, kind="stable", na_position="last")
    result.reset_index(drop=True, inplace=True)
    return result


def add_interval(
    df: pd.DataFrame,
    onset_s: float,
    offset_s: float,
    labels: int,
    individual: str,
    protected_label_ids: set[int] | None = None,
    individual_rec: str = NO_RECIPIENT,
    confidence: float = HUMAN_CONFIDENCE,
    labeling_method: str = LABELING_MANUAL,
) -> pd.DataFrame:
    """Add an interval, resolving overlaps for the same subject.

    If the new interval overlaps existing intervals for the same
    (actor, recipient) pair, the existing intervals are trimmed or split —
    unless their label ID is in *protected_label_ids*, in which case they are
    kept untouched.  Another pair's intervals are never touched: the same
    animal mounting bird A and preening bird B are two independent tracks.

    Parameters
    ----------
    df : pd.DataFrame
        Current intervals DataFrame.
    onset_s, offset_s : float
        Start and end times in seconds.
    labels : int
        Label class ID.
    individual : str
        Individual performing the behaviour (actor).
    protected_label_ids : set[int] | None
        Label IDs that must not be trimmed or split (e.g. labels belonging
        to inactive branches).  ``None`` means no protection.
    individual_rec : str
        Recipient of the behaviour; :data:`NO_RECIPIENT` for a solo one.
    confidence : float
        How sure this label is; :data:`HUMAN_CONFIDENCE` for a hand-placed one.
    labeling_method : str
        Who vouches for it; :data:`LABELING_MANUAL` for a hand-placed one.
        A remnant trimmed off an existing interval keeps that interval's
        method — nobody re-judged it.

    Returns
    -------
    pd.DataFrame
        Updated intervals DataFrame sorted by ``onset_s``.

    Examples
    --------
    >>> df = empty_intervals()
    >>> df = add_interval(df, 0.0, 1.0, 1, "crow_A")
    >>> df = add_interval(df, 0.5, 1.5, 2, "crow_A")
    >>> len(df)
    2
    >>> float(df.iloc[0]["offset_s"])  # first interval trimmed
    0.499
    """
    if onset_s > offset_s:
        onset_s, offset_s = offset_s, onset_s

    states_df, points_df = split_by_kind(ensure_labeling_method(ensure_individual_rec(df.copy())))
    # The actor's whole track, whatever each row's receiver: a new label
    # trims every label of this actor it overlaps.
    mask_same_ind = subject_mask(states_df, individual)
    other = states_df[~mask_same_ind]
    same = states_df[mask_same_ind].copy()

    kept: list[dict] = []
    for _, row in same.iterrows():
        ro, rf = row["onset_s"], row["offset_s"]
        rid = row["labels"]

        # Never trim/split intervals from protected (inactive) branches
        if protected_label_ids is not None and int(rid) in protected_label_ids:
            kept.append(row.to_dict())
            continue

        if rf <= onset_s or ro >= offset_s:
            kept.append(row.to_dict())
            continue

        eps = 1e-3
        # A trimmed remnant is the same label as before — it keeps the
        # confidence of the row it was cut from, not the new label's.
        row_conf = row.get("confidence", HUMAN_CONFIDENCE)
        row_method = row.get("labeling_method", LABELING_MANUAL)
        row_rec = row.get("individual_rec", NO_RECIPIENT)
        if ro < onset_s:
            kept.append(
                {
                    "onset_s": ro,
                    "offset_s": onset_s - eps,
                    "labels": rid,
                    "individual": individual,
                    "individual_rec": row_rec,
                    "confidence": row_conf,
                    "labeling_method": row_method,
                }
            )
        if rf > offset_s:
            kept.append(
                {
                    "onset_s": offset_s + eps,
                    "offset_s": rf,
                    "labels": rid,
                    "individual": individual,
                    "individual_rec": row_rec,
                    "confidence": row_conf,
                    "labeling_method": row_method,
                }
            )

    kept.append(
        {
            "onset_s": onset_s,
            "offset_s": offset_s,
            "labels": labels,
            "individual": individual,
            "individual_rec": individual_rec,
            "confidence": float(confidence),
            "labeling_method": str(labeling_method),
        }
    )

    new_same = _rows_to_df(kept)
    transformed_states = pd.concat([other, new_same], ignore_index=True)
    return _recombine(transformed_states, points_df)


def delete_interval(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    """Drop interval by DataFrame index."""
    return df.drop(index=idx).reset_index(drop=True)


def find_interval_at(
    df: pd.DataFrame,
    time_s: float,
    individual: str | None,
    label_ids: set[int] | None = None,
    individual_rec: str | None = None,
) -> int | None:
    """Return DataFrame index of state interval containing *time_s* for one subject.

    Point events are never returned here — use :func:`find_point_at` for those.

    Parameters
    ----------
    individual : str | None
        ``None`` matches any individual — used as a selection fallback when
        the current individual name doesn't match what a loaded file stores.
    label_ids : set[int] | None
        When given, only match intervals whose ``labels`` value is in this set.
        Useful for restricting to the active branch.
    individual_rec : str | None
        Recipient to match; ``None`` matches any (same fallback role).

    Returns ``None`` if no non-background interval contains the time.
    """
    mask = (df["onset_s"] <= time_s) & (df["offset_s"] >= time_s) & (df["labels"] != 0)
    mask = mask & subject_mask(df, individual, individual_rec)
    if "event_type" in df.columns:
        mask = mask & (df["event_type"] == EVENT_TYPE_STATE)
    if label_ids is not None:
        mask = mask & df["labels"].isin(label_ids)
    matches = df.index[mask]
    if len(matches) == 0:
        return None
    return int(matches[0])


def find_point_at(
    df: pd.DataFrame,
    time_s: float,
    individual: str | None,
    tolerance_s: float,
    label_ids: set[int] | None = None,
    individual_rec: str | None = None,
) -> int | None:
    """Return DataFrame index of point event near *time_s* for one subject.

    A point matches if ``|onset_s - time_s| <= tolerance_s``.  Caller chooses
    the tolerance — typically a few pixels' worth of plot time.
    ``individual=None`` / ``individual_rec=None`` match any (selection fallback).
    """
    if "event_type" not in df.columns or df.empty:
        return None
    mask = (
        (df["event_type"] == EVENT_TYPE_POINT) & ((df["onset_s"] - time_s).abs() <= tolerance_s) & (df["labels"] != 0)
    )
    mask = mask & subject_mask(df, individual, individual_rec)
    if label_ids is not None:
        mask = mask & df["labels"].isin(label_ids)
    matches = df.index[mask]
    if len(matches) == 0:
        return None
    return int(matches[0])


def get_interval_bounds(df: pd.DataFrame, idx: int) -> tuple[float, float, int]:
    """Return ``(onset_s, offset_s, labels)`` for interval at *idx*."""
    row = df.loc[idx]
    return float(row["onset_s"]), float(row["offset_s"]), int(row["labels"])


def purge_short_intervals(
    df: pd.DataFrame,
    min_duration_s: float,
    label_thresholds_s: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Drop intervals shorter than a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Intervals DataFrame.
    min_duration_s : float
        Default minimum duration in seconds.
    label_thresholds_s : dict[int, float], optional
        Per-label minimum durations (overrides *min_duration_s*).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.

    Examples
    --------
    >>> df = add_interval(empty_intervals(), 0.0, 0.01, 1, "A")
    >>> df = add_interval(df, 1.0, 2.0, 2, "A")
    >>> purged = purge_short_intervals(df, min_duration_s=0.1)
    >>> len(purged)
    1
    """
    if label_thresholds_s is None:
        label_thresholds_s = {}

    states_df, points_df = split_by_kind(df)
    durations = states_df["offset_s"] - states_df["onset_s"]
    thresholds = states_df["labels"].map(lambda lid: label_thresholds_s.get(lid, min_duration_s))
    keep = durations >= thresholds
    return _recombine(states_df[keep], points_df)


def stitch_intervals(
    df: pd.DataFrame,
    max_gap_s: float,
    individual: str | None = None,
) -> pd.DataFrame:
    """Merge adjacent same-label intervals where gap <= *max_gap_s*.

    Parameters
    ----------
    df : pd.DataFrame
        Intervals DataFrame.
    max_gap_s : float
        Maximum gap (seconds) between intervals to merge.
    individual : str, optional
        If given, only stitch intervals for this individual.

    Returns
    -------
    pd.DataFrame
        Stitched intervals DataFrame.

    Examples
    --------
    >>> df = add_interval(empty_intervals(), 0.0, 1.0, 1, "A")
    >>> df = add_interval(df, 1.05, 2.0, 1, "A")
    >>> stitched = stitch_intervals(df, max_gap_s=0.1)
    >>> len(stitched)
    1
    >>> float(stitched.iloc[0]["offset_s"])
    2.0
    """
    if df.empty:
        return df.copy()

    states_df, points_df = split_by_kind(ensure_individual_rec(df.copy()))

    if individual is not None:
        mask = states_df["individual"] == individual
        other = states_df[~mask]
        target = states_df[mask].copy()
    else:
        other = empty_intervals()
        target = states_df.copy()

    target.sort_values([*SUBJECT_COLUMNS, "onset_s"], inplace=True)
    target.reset_index(drop=True, inplace=True)

    merged: list[dict] = []
    i = 0
    while i < len(target):
        row = target.iloc[i]
        current = row.to_dict()
        j = i + 1
        while j < len(target):
            nxt = target.iloc[j]
            if (
                all(nxt[col] == current[col] for col in SUBJECT_COLUMNS)
                and nxt["labels"] == current["labels"]
                and (nxt["onset_s"] - current["offset_s"]) < max_gap_s
            ):
                current["offset_s"] = nxt["offset_s"]
                # A merged interval is only as trustworthy as its weakest part.
                current["confidence"] = min(
                    float(current.get("confidence", HUMAN_CONFIDENCE)),
                    float(nxt.get("confidence", HUMAN_CONFIDENCE)),
                )
                j += 1
            else:
                break
        merged.append(current)
        i = j

    transformed_states = pd.concat([other, _rows_to_df(merged)], ignore_index=True)
    return _recombine(transformed_states, points_df)


def snap_boundaries(
    df: pd.DataFrame,
    cp_times: np.ndarray,
    max_expansion_s: float,
    max_shrink_s: float,
) -> pd.DataFrame:
    """Snap interval onset/offset to nearest changepoint times.

    Parameters
    ----------
    df : pd.DataFrame
        Intervals DataFrame.
    cp_times : np.ndarray
        Candidate changepoint times.
    max_expansion_s : float
        Maximum allowed expansion (seconds).
    max_shrink_s : float
        Maximum allowed shrinkage (seconds).

    Returns
    -------
    pd.DataFrame
        Snapped intervals with overlaps resolved.
    """
    if df.empty or len(cp_times) == 0:
        return df.copy()

    states_df, points_df = split_by_kind(ensure_individual_rec(df.copy()))
    cp_times = np.sort(cp_times)
    rows = []

    for _, row in states_df.iterrows():
        onset = row["onset_s"]
        offset = row["offset_s"]

        snap_onset = _snap_onset(onset, cp_times, max_expansion_s, max_shrink_s)
        snap_offset = _snap_offset(offset, cp_times, max_expansion_s, max_shrink_s)

        if snap_onset >= snap_offset:
            snap_onset = onset
            snap_offset = offset

        # Only the boundaries move: confidence, labeling_method and the rest
        # of the row are the label's provenance and survive the snap.
        rows.append({**row.to_dict(), "onset_s": snap_onset, "offset_s": snap_offset})

    transformed = _rows_to_df(rows)
    transformed.sort_values([*TRACK_COLUMNS, "onset_s"], inplace=True)
    transformed.reset_index(drop=True, inplace=True)
    transformed = _resolve_overlaps(transformed)
    return _recombine(transformed, points_df)


# ── Private helpers ──────────────────────────────────────────────────────


def _snap_onset(boundary, cp_times, max_expansion_s, max_shrink_s):
    # cp before boundary -> expansion (onset moves earlier); cp after -> shrink (onset moves later)
    lo = boundary - max_expansion_s
    hi = boundary + max_shrink_s
    candidates = cp_times[(cp_times >= lo) & (cp_times <= hi)]
    if len(candidates) == 0:
        return boundary
    nearest_idx = np.argmin(np.abs(candidates - boundary))
    return float(candidates[nearest_idx])


def _snap_offset(boundary, cp_times, max_expansion_s, max_shrink_s):
    # cp after boundary -> expansion (offset moves later); cp before -> shrink (offset moves earlier)
    lo = boundary - max_shrink_s
    hi = boundary + max_expansion_s
    candidates = cp_times[(cp_times >= lo) & (cp_times <= hi)]
    if len(candidates) == 0:
        return boundary
    nearest_idx = np.argmin(np.abs(candidates - boundary))
    return float(candidates[nearest_idx])


def _resolve_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Clip a label that runs into the next label of another class on the same subject.

    Two labels that *meet* — offset equal to the next onset, which is what
    both snapping to the same changepoint produces — are left exactly where
    they are; only a true overlap is clipped, and the clipped edge lands on
    the next onset itself, never a millisecond short of it.
    """
    if df.empty:
        return df
    df = ensure_individual_rec(df.copy())

    groups = []
    for _actor, group in df.groupby(TRACK_COLUMNS, sort=False):
        group = group.sort_values("onset_s").reset_index(drop=True)
        for i in range(len(group) - 1):
            if group.at[i, "offset_s"] > group.at[i + 1, "onset_s"]:
                if group.at[i, "labels"] != group.at[i + 1, "labels"]:
                    group.at[i, "offset_s"] = group.at[i + 1, "onset_s"]
        groups.append(group)

    result = pd.concat(groups, ignore_index=True)
    result.sort_values("onset_s", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return empty_intervals()
    df = pd.DataFrame(rows)
    if "event_type" not in df.columns:
        df["event_type"] = EVENT_TYPE_STATE
    else:
        df["event_type"] = df["event_type"].fillna(EVENT_TYPE_STATE).astype(object)
    df = df.reindex(columns=INTERVAL_COLUMNS)
    ensure_individual_rec(df)
    ensure_confidence(df)
    ensure_labeling_method(df)
    for col, dtype in INTERVAL_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df
