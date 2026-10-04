"""How a run's predictions held up under review, and which trials were hard.

Once every trial of a session is curated, each trial's final labels are
compared with what the run that predicted into it wrote — the
``*_predictions.tsv`` in its folder under ``labels/`` (:mod:`onset_curves`).
The comparison is an F1 per trial and event type:

* a **state** label is a hit when a final label of the same class and subject
  overlaps it at IoU ≥ :data:`STATE_IOU`;
* a **point** label is a hit when one lies within the run's own tolerance —
  the precision the model was trained to (:func:`run_tolerance_s`), never a
  number the reviewer has to pick, unless they override it on purpose.

The scores land in the metadata table as :data:`REVIEW_F1_STATE` /
:data:`REVIEW_F1_POINT`, so the trials table's own filters are the criterion
for anything further. A trial scoring below the flag threshold is marked
:data:`DIFFICULTY_HARD` in :data:`DIFFICULTY_COLUMN` — the same column the
reviewer sets by hand for a trial that is simply difficult. Auto-flagging only
ever sets ``hard``; a flag a human set is never cleared by a score
(:func:`difficulty_values`). Training pipelines read the column to show hard
trials more often (``train.oversample`` in :mod:`ethograph.segment`).

Pandas only, so the rules are unit-testable without Qt; the GUI
(``gui/widgets_curation.py``) decides *when* to run them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ethograph.labels import onset_curves
from ethograph.labels.curation import subject_str
from ethograph.labels.intervals import EVENT_TYPE_POINT, EVENT_TYPE_STATE
from ethograph.labels.tsv_store import get_trial_from_tsv, get_trial_meta, load_labels_tsv

#: Metadata-table column: how hard a trial is. String-valued (not 0/1) so the
#: trials table's funnel treats it as a checklist, like ``curated``.
DIFFICULTY_COLUMN = "difficulty"
DIFFICULTY_NORMAL = "normal"
DIFFICULTY_HARD = "hard"
DIFFICULTIES = (DIFFICULTY_NORMAL, DIFFICULTY_HARD)

#: Metadata-table columns holding the per-trial review scores (NaN when the
#: trial has no labels of that event type on either side).
REVIEW_F1_STATE = "review_f1_state"
REVIEW_F1_POINT = "review_f1_point"
REVIEW_COLUMNS = (REVIEW_F1_STATE, REVIEW_F1_POINT)

#: A trial's mean frame confidence over its run's curve (1 − normalised
#: entropy), written by the Curation section's *Confidence curves…*. A
#: number to sort and filter trials on — which ones to open first, which
#: to curate in bulk — and never an input to ``difficulty``.
MODEL_CONFIDENCE_COLUMN = "model_confidence"
#: A state label matches at this IoU — the ``f1@50`` convention the
#: segmentation pipeline selects its checkpoints on.
STATE_IOU = 0.5

#: A trial whose F1 (either event type) falls below this is flagged hard.
DEFAULT_FLAG_THRESHOLD = 0.5

#: Above this share of the scored trials, flagging from the histogram is
#: warned against: with ``train.oversample`` on they would be most of what
#: the model sees, and a flag that most trials carry means nothing.
FLAG_SHARE_WARNING = 0.25


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Counts:
    """True positives, false positives and false negatives of one comparison."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn

    @property
    def f1(self) -> float | None:
        """``None`` when there was nothing on either side to compare."""
        if self.total == 0:
            return None
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def _subject_key(row: pd.Series) -> tuple[int, str, str]:
    return int(row["labels"]), subject_str(row.get("individual")), subject_str(row.get("individual_rec"))


def _iou(a_on: float, a_off: float, b_on: float, b_off: float) -> float:
    inter = max(0.0, min(a_off, b_off) - max(a_on, b_on))
    union = max(a_off, b_off) - min(a_on, b_on)
    return inter / union if union > 0 else 0.0


def _match(pred: pd.DataFrame, final: pd.DataFrame, score) -> Counts:
    """Greedy one-to-one matching: every predicted row, in onset order, takes
    the best still-unmatched final row of its class and subject when *score*
    (a similarity in [0, 1], or ``None`` for "too far") allows it."""
    if pred.empty and final.empty:
        return Counts()
    final_keys = [_subject_key(r) for _, r in final.iterrows()]
    taken = np.zeros(len(final), dtype=bool)
    tp = fp = 0
    for _, p in pred.sort_values("onset_s").iterrows():
        key = _subject_key(p)
        best, best_j = None, -1
        for j, (_, f) in enumerate(final.iterrows()):
            if taken[j] or final_keys[j] != key:
                continue
            s = score(p, f)
            if s is not None and (best is None or s > best):
                best, best_j = s, j
        if best_j >= 0:
            taken[best_j] = True
            tp += 1
        else:
            fp += 1
    return Counts(tp, fp, int((~taken).sum()))


def match_intervals(pred: pd.DataFrame, final: pd.DataFrame, *, iou: float = STATE_IOU) -> Counts:
    """State labels: a hit at IoU ≥ *iou* with a final label of the same class and subject."""

    def score(p, f):
        v = _iou(float(p["onset_s"]), float(p["offset_s"]), float(f["onset_s"]), float(f["offset_s"]))
        return v if v >= iou else None

    return _match(pred, final, score)


def match_points(pred: pd.DataFrame, final: pd.DataFrame, *, tolerance_s: float) -> Counts:
    """Point labels: a hit within *tolerance_s* of a final label of the same class and subject."""
    if tolerance_s <= 0:
        raise ValueError(f"A point tolerance is a positive number of seconds, got {tolerance_s}")

    def score(p, f):
        d = abs(float(p["onset_s"]) - float(f["onset_s"]))
        return 1.0 - d / tolerance_s if d <= tolerance_s else None

    return _match(pred, final, score)


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "event_type" not in df.columns:
        return df, df.iloc[0:0]
    kind = df["event_type"].fillna(EVENT_TYPE_STATE).astype(str)
    return df[kind != EVENT_TYPE_POINT], df[kind == EVENT_TYPE_POINT]


@dataclass(frozen=True)
class TrialReview:
    """One trial's comparison: what the run said against what it ended up as."""

    trial: str
    run: Path
    state: Counts
    #: ``None`` when the trial had point labels but no tolerance to judge them at.
    point: Counts | None
    tolerance_s: float | None

    @property
    def f1_state(self) -> float | None:
        return self.state.f1

    @property
    def f1_point(self) -> float | None:
        return None if self.point is None else self.point.f1

    def below(self, threshold: float) -> bool:
        return any(v is not None and v < threshold for v in (self.f1_state, self.f1_point))


def review_trial(
    trial, run: Path, pred: pd.DataFrame, final: pd.DataFrame, *, tolerance_s: float | None
) -> TrialReview:
    """Compare one trial's predicted rows with its final rows."""
    pred_state, pred_point = _split(pred)
    final_state, final_point = _split(final)
    state = match_intervals(pred_state, final_state)
    if pred_point.empty and final_point.empty:
        point: Counts | None = Counts()
    elif tolerance_s is None:
        point = None
    else:
        point = match_points(pred_point, final_point, tolerance_s=tolerance_s)
    return TrialReview(str(trial), run, state, point, tolerance_s)


# ---------------------------------------------------------------------------
# Which run, at what tolerance
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def run_source(folder: Path) -> str | None:
    """The ``prediction_source`` a run stamped into its rows, from its ``inference.yaml``."""
    value = _read_yaml(folder / onset_curves.INFERENCE_FILE).get("prediction_source")
    return str(value) if value else None


def run_tolerance_s(folder: Path) -> float | None:
    """The point tolerance the run's model was trained to, in seconds.

    Read from the run's ``inference.yaml`` (``tolerance_s``, written by every
    model since it was introduced) and, for older folders, worked out from
    what the model kept: the lightgbm model's ``config.yaml`` ``tolerance_s``,
    spot's ``infer.focus_window_ms`` (a window *around* the peak, so half of
    it either side). ``None`` when the run says nothing — a segmentation
    model, which predicts no point events.
    """
    inference = _read_yaml(folder / onset_curves.INFERENCE_FILE)
    if inference.get("tolerance_s") is not None:
        return float(inference["tolerance_s"])
    model = inference.get("model")
    if model == onset_curves.LIGHTGBM:
        value = _read_yaml(folder / onset_curves.CONFIG_FILE).get("tolerance_s")
        return float(value) if value is not None else None
    if model == "spot":
        window_ms = (inference.get("infer") or {}).get("focus_window_ms")
        return float(window_ms) / 2000.0 if window_ms is not None else None
    return None


def resolve_run(session_path: str | Path, trial, source: str | None) -> Path | None:
    """The run folder whose predictions *trial* should be judged against.

    The run that declares *source* as its ``prediction_source`` wins; a
    folder from before runs declared one falls back to the newest run whose
    predictions include the trial. ``None`` when no run on disk predicted
    into the trial — nothing to compare, so nothing is scored.
    """
    runs = onset_curves.prediction_run_dirs(session_path)
    if source:
        for folder in reversed(runs):
            if run_source(folder) == source:
                return folder
    trial = str(trial)
    for folder in reversed(runs):
        df = read_predictions(folder)
        if df is not None and (df["trial"].astype(str) == trial).any():
            return folder
    return None


def read_predictions(folder: Path) -> pd.DataFrame | None:
    """The labels a run wrote, or ``None`` for a folder without a predictions file."""
    path = onset_curves.predictions_file(folder)
    return load_labels_tsv(path) if path is not None else None


def review_session(
    session_path: str | Path,
    all_labels_df: pd.DataFrame | None,
    trials,
    *,
    tolerance_override_s: float | None = None,
) -> dict[str, TrialReview]:
    """Every trial of *trials* a run predicted into, compared with its final labels.

    A trial with no ``prediction_source`` and no run holding it is left out:
    a human labelled it from scratch, and there is nothing to grade.
    *tolerance_override_s* replaces every run's own point tolerance.
    """
    if all_labels_df is None or all_labels_df.empty:
        return {}
    predictions: dict[Path, pd.DataFrame | None] = {}
    tolerances: dict[Path, float | None] = {}
    out: dict[str, TrialReview] = {}
    for trial in trials:
        source = str(get_trial_meta(all_labels_df, trial).get("prediction_source") or "")
        run = resolve_run(session_path, trial, source or None)
        if run is None:
            continue
        if run not in predictions:
            predictions[run] = read_predictions(run)
            tolerances[run] = run_tolerance_s(run)
        pred = predictions[run]
        if pred is None:
            continue
        pred = pred[pred["trial"].astype(str) == str(trial)]
        final = get_trial_from_tsv(all_labels_df, trial)
        tolerance = tolerance_override_s if tolerance_override_s is not None else tolerances[run]
        out[str(trial)] = review_trial(trial, run, pred, final, tolerance_s=tolerance)
    return out


# ---------------------------------------------------------------------------
# Into the metadata table
# ---------------------------------------------------------------------------


def trial_confidence_means(confidence_map: dict[str, np.ndarray | None]) -> dict[str, float]:
    """``{trial: mean frame confidence}`` for every trial with a curve; a trial
    without one is left out rather than written as 0."""
    means: dict[str, float] = {}
    for trial, curve in confidence_map.items():
        if curve is None or len(curve) == 0:
            continue
        mean = float(np.nanmean(curve))
        if not np.isnan(mean):
            means[str(trial)] = mean
    return means


def is_hard(value) -> bool:
    return isinstance(value, str) and value.strip().lower() == DIFFICULTY_HARD


def f1_scores(metadata_df: pd.DataFrame | None) -> dict[str, list[float]]:
    """The scored values per review column, ``{column: [f1, ...]}`` — what the
    histogram draws. A column the table lacks, or a blank cell, contributes
    nothing."""
    out: dict[str, list[float]] = {column: [] for column in REVIEW_COLUMNS}
    if metadata_df is None or metadata_df.empty:
        return out
    for column in REVIEW_COLUMNS:
        if column in metadata_df.columns:
            values = pd.to_numeric(metadata_df[column], errors="coerce")
            out[column] = [float(v) for v in values if not pd.isna(v)]
    return out


def _scored_rows(metadata_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Rows with at least one review score, the scores as floats (NaN elsewhere)."""
    if metadata_df is None or metadata_df.empty or "trial" not in metadata_df.columns:
        return None
    present = [c for c in REVIEW_COLUMNS if c in metadata_df.columns]
    if not present:
        return None
    df = metadata_df[["trial", *present]].copy()
    for column in present:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[df[present].notna().any(axis=1)]


def scored_trial_count(metadata_df: pd.DataFrame | None) -> int:
    """How many trials carry a review score at all."""
    rows = _scored_rows(metadata_df)
    return 0 if rows is None else len(rows)


def trials_below_f1(metadata_df: pd.DataFrame | None, threshold: float) -> set[str]:
    """The scored trials whose state *or* point F1 is below *threshold* — the
    histogram's flag set. A threshold of 0 flags nothing."""
    rows = _scored_rows(metadata_df)
    if rows is None or threshold <= 0.0:
        return set()
    present = [c for c in REVIEW_COLUMNS if c in rows.columns]
    low = (rows[present] < threshold).any(axis=1)
    return set(rows.loc[low, "trial"].astype(str))


def flag_share_note(n_flagged: int, n_scored: int, threshold: float) -> str:
    """The line under the histogram: the count, and a warning past
    :data:`FLAG_SHARE_WARNING` of the scored trials."""
    if not n_scored:
        return "No scored trials."
    share = n_flagged / n_scored
    text = f"Below {threshold:.2f}: <b>{n_flagged} of {n_scored}</b> scored trial(s) ({share:.0%})."
    if share > FLAG_SHARE_WARNING:
        text += (
            f" <span style='color:#d94040'>That is more than {FLAG_SHARE_WARNING:.0%} of them — with "
            "train.oversample on, hard trials would be most of what the model sees, and the flag would "
            "stop meaning anything. Move the threshold to the gap, or flag by hand.</span>"
        )
    return text


def difficulty_values(metadata_df: pd.DataFrame | None, hard: set[str], trials) -> dict[str, str]:
    """``{trial: difficulty}`` to write for *trials*: ``hard`` for *hard* and
    for anything already hard (a human's flag outranks a score), ``normal``
    for the rest. Only what the table does not already say."""
    current: dict[str, object] = {}
    if metadata_df is not None and DIFFICULTY_COLUMN in metadata_df.columns:
        current = dict(zip(metadata_df["trial"].astype(str), metadata_df[DIFFICULTY_COLUMN]))
    out: dict[str, str] = {}
    for trial in (str(t) for t in trials):
        held = current.get(trial)
        value = DIFFICULTY_HARD if trial in hard or is_hard(held) else DIFFICULTY_NORMAL
        if held is None or pd.isna(held) or str(held) != value:
            out[trial] = value
    return out


def review_columns(reviews: dict[str, TrialReview]) -> dict[str, dict[str, float]]:
    """The score columns to write: ``{column: {trial: f1}}``, NaN where a
    trial had nothing of that event type."""
    state = {t: (np.nan if r.f1_state is None else r.f1_state) for t, r in reviews.items()}
    point = {t: (np.nan if r.f1_point is None else r.f1_point) for t, r in reviews.items()}
    return {REVIEW_F1_STATE: state, REVIEW_F1_POINT: point}


def summary(reviews: dict[str, TrialReview]) -> str:
    """One line for the reviewer: pooled F1 per event type over the scored trials."""
    if not reviews:
        return "Review: no trial had a prediction run to compare against."
    state = sum((r.state for r in reviews.values()), Counts())
    point = sum((r.point for r in reviews.values() if r.point is not None), Counts())
    parts = []
    if state.total:
        parts.append(f"state F1@{int(STATE_IOU * 100)}: {state.f1:.2f}")
    if point.total:
        parts.append(f"point F1: {point.f1:.2f}")
    unjudged = sum(1 for r in reviews.values() if r.point is None)
    if unjudged:
        parts.append(f"{unjudged} trial(s) with point events but no tolerance to judge them at")
    scored = ", ".join(parts) if parts else "nothing to compare"
    return f"Scored {len(reviews)} trial(s) against their runs — {scored}. Histogram… to flag the worst as hard."
