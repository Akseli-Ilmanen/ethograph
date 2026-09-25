"""Read a :mod:`ethograph.segment.inference` prediction folder.

One run, one folder, beside the session's own ``labels/`` (see
:mod:`ethograph.labels.onset_curves` for the sibling convention the LightGBM
lightgbm model uses)::

    labels/
        predictions_{run_name}_{timestamp}/
            {stem}_predictions.tsv   # real intervals, per-segment confidence, labeling_method
            {stem}_probs.npz         # per (trial, individual): "{key}" -> (T, C) probs, "{key}_time" -> (T,)

The TSV needs no reconstruction — it is already the GUI's native labels
format. Only the frame-by-frame confidence *curve* (for the review overlay)
is read from the ``.npz``, one array at a time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ethograph.labels.confidence import entropy_confidence
from ethograph.labels.intervals import SUBJECT_COLUMNS

logger = logging.getLogger(__name__)


def prediction_to_labels_and_confidence(
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert prediction array to dense labels and optional confidence.

    Parameters
    ----------
    pred : np.ndarray
        Shape (T, n_classes) for softmax probabilities, or (T,) for dense labels.

    Returns
    -------
    labels : np.ndarray, shape (T,)
        Dense integer labels (argmax for softmax input).
    confidence : np.ndarray or None
        Shape (T,) confidence scores. For softmax input: 1 - normalized_entropy.
        None if input is already dense labels.
    """
    pred = np.asarray(pred)

    if pred.ndim == 2:
        return np.argmax(pred, axis=1), entropy_confidence(pred)

    # Shape (T,) — already dense labels
    return pred.astype(int), None


def merge_as_labels(existing: pd.DataFrame | None, predicted: pd.DataFrame) -> pd.DataFrame:
    """*predicted* rows added onto *existing*, ground truth left untouched.

    A row is skipped when *existing* already has an interval for the same
    ``(trial, labels, individual, individual_rec)`` — the lightgbm model's own
    rule (:func:`~ethograph.gui.dialog_onset_model.predict_onsets`): a trial
    already carrying an event for a class is never overridden, whichever
    pipeline predicted it. The alternative to merging is a plain replace,
    which the caller does itself — this function is only the "keep what I
    have, add what's missing" half.
    """
    if existing is None or existing.empty:
        return predicted.reset_index(drop=True)
    if predicted.empty:
        return existing.reset_index(drop=True)
    key_cols = ["trial", "labels", *SUBJECT_COLUMNS]
    existing_keys = set(existing[key_cols].astype(str).itertuples(index=False, name=None))
    predicted_keys = predicted[key_cols].astype(str).apply(tuple, axis=1)
    new_rows = predicted[~predicted_keys.isin(existing_keys)]
    return pd.concat([existing, new_rows], ignore_index=True)


@dataclass
class PredictionSet:
    """One imported prediction file: its rows, and the run's store when it came from a run folder."""

    path: Path
    labels_df: pd.DataFrame
    store: PredictionsStore | None = None

    @property
    def name(self) -> str:
        """What the GUI calls this set — panel title, ➕ popup, Predictions list.

        Every run folder writes the same ``{stem}_predictions.tsv``, so a
        run's file is named with its folder in front; a plain ``.tsv`` picked
        by hand is its own name.
        """
        if self.store is None:
            return self.path.name
        return f"{self.path.parent.name}/{self.path.name}"


def add_prediction_set(sets: list[PredictionSet], new: PredictionSet) -> list[PredictionSet]:
    """*sets* with *new* appended — re-importing the same file replaces it in place."""
    out = list(sets)
    for i, existing in enumerate(out):
        if existing.path == new.path:
            out[i] = new
            return out
    out.append(new)
    return out


def remove_prediction_set(sets: list[PredictionSet], path: Path) -> list[PredictionSet]:
    """*sets* without the file at *path* — a path not in the list is a no-op."""
    return [s for s in sets if s.path != path]


class PredictionsStore:
    """Read one :mod:`ethograph.segment.inference` prediction folder.

    Example
    -------
    ::

        store = PredictionsStore("labels/predictions_mstcn_20260101_000000")
        labels_df, _ = store.load_all(dt)
        confidence = store.get_confidence(trial=5, dt=dt, individual="A")
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        # *_labels.tsv is the legacy spelling — a run written before predictions
        # were renamed to *_predictions.tsv to read distinctly from a curated
        # labels file of the same session.
        tsvs = sorted(self.folder.glob("*_predictions.tsv")) or sorted(self.folder.glob("*_labels.tsv"))
        if not tsvs:
            raise FileNotFoundError(f"No *_predictions.tsv in {self.folder}")
        self.tsv_path = tsvs[0]
        npzs = sorted(self.folder.glob("*_probs.npz"))
        self.npz_path = npzs[0] if npzs else None

    def load_all(self, dt, individual: str | None = None, **_ignored) -> tuple[pd.DataFrame, dict]:
        """Every trial's predictions, already postprocessed by the run itself."""
        from ethograph.labels.tsv_store import load_labels_tsv

        return load_labels_tsv(self.tsv_path), {}

    def get_confidence(self, trial, dt, individual: str | None = None) -> np.ndarray | None:
        """Per-frame confidence for one (trial, individual), from the run's own probabilities.

        Returns ``None`` when the run has no ``.npz`` (e.g. a hand-edited
        folder) or no key matches — an aid to review, never something a
        caller depends on.
        """
        curve = self.get_confidence_curve(trial, dt, individual)
        return None if curve is None else curve[1]

    def get_confidence_curve(self, trial, dt, individual: str | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """``(time, confidence)`` for one (trial, individual), on the run's own clock.

        The run may have been trained subsampled, so its curve is sparser
        than the session's time coordinate: the ``{key}_time`` array written
        beside the probabilities is the only clock the curve can be drawn on.
        """
        if self.npz_path is None:
            return None
        marker = f"_trial{trial}_"
        with np.load(self.npz_path) as npz:
            keys = [k for k in npz.files if marker in k and not k.endswith("_time") and not k.endswith("_boundary")]
            if not keys:
                return None
            match = keys[0]
            if individual is not None and len(keys) > 1:
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(individual))
                match = next((k for k in keys if k.endswith(f"_{safe}")), match)
            if f"{match}_time" not in npz.files:
                raise ValueError(f"{self.npz_path.name} has no '{match}_time' array: written by an older run")
            probs = np.asarray(npz[match], dtype=np.float64)
            time = np.asarray(npz[f"{match}_time"], dtype=np.float64)
        _, confidence = prediction_to_labels_and_confidence(probs)
        if confidence is None:
            return None
        return time, confidence
