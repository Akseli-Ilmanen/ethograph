"""A prediction set on disk — the GUI's labels TSV plus its curves, whichever model decoded them.

Torch-free: FERAL's predictions are written through it without a model in sight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ethograph.labels.intervals import LABELING_AUTOMATED, NO_RECIPIENT
from ethograph.labels.onset_curves import labels_dir, write_provenance
from ethograph.labels.tsv_store import save_labels_tsv

PREDICTIONS_PREFIX = "predictions"


def prediction_run_dir(session_path: Path, run_name: str, timestamp: str) -> Path:
    """Where one inference run's outputs for one session are written.

    One folder per call, so cross-validation folds and repeated ad-hoc
    ``infer()`` calls never overwrite each other's predictions.
    """
    return labels_dir(session_path) / f"{PREDICTIONS_PREFIX}_{run_name}_{timestamp}"


def _segment_confidence(conf: np.ndarray, time: np.ndarray, onset: float, offset: float) -> float:
    m = (time >= onset) & (time <= offset)
    return float(conf[m].mean()) if m.any() else float(conf.max())


PREDICTION_COLUMNS = [
    "trial",
    "individual",
    "individual_rec",
    "labels",
    "onset_s",
    "offset_s",
    "event_type",
    "confidence",
    "labeling_method",
    "changepoint_corrected",
    "prediction_source",
    "n_samples",
]


def label_rows(
    intervals: pd.DataFrame,
    curves: dict[int, np.ndarray],
    time: np.ndarray,
    trial: int | str,
    individual: str,
    source: str,
    corrected: bool,
) -> list[dict]:
    """One sample's decoded intervals as rows of a prediction set, whichever model decoded them."""
    rows = []
    for _, seg in intervals.iterrows():
        onset, offset, lid = float(seg["onset_s"]), float(seg["offset_s"]), int(seg["labels"])
        rows.append(
            {
                "trial": trial,
                "individual": individual,
                "individual_rec": NO_RECIPIENT,
                "labels": lid,
                "onset_s": onset,
                "offset_s": offset,
                "event_type": "state",
                "confidence": _segment_confidence(curves[lid], time, onset, offset),
                "labeling_method": LABELING_AUTOMATED,
                "changepoint_corrected": int(corrected),
                "prediction_source": source,
                "n_samples": int(len(time)),
            }
        )
    return rows


def write_prediction_set(
    out_dir: Path,
    stem: str,
    rows: list[dict],
    arrays: dict[str, np.ndarray],
    *,
    model_config: Path | dict,
    inference_note: dict,
) -> tuple[Path, Path]:
    """One session's prediction set in the layout the GUI reads; returns the TSV and the ``.npz``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / f"{stem}_predictions.tsv"
    npz_path = out_dir / f"{stem}_probs.npz"
    save_labels_tsv(tsv_path, pd.DataFrame(rows) if rows else pd.DataFrame(columns=PREDICTION_COLUMNS))
    np.savez_compressed(npz_path, **arrays)
    write_provenance(out_dir, model_config=model_config, inference=inference_note)
    return tsv_path, npz_path
