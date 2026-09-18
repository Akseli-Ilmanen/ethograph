"""A run's scores become the GUI's labels, plus the curves they were read off.

E2E-Spot writes one file per epoch holding every frame that scored above a low
threshold — the curve with its zeros left out. This module turns that into the
two things the GUI already knows how to read:

* a labels TSV (``labeling_method=automated``, one row per event — one per
  class per trial unless ``infer.max_events_per_trial`` says more),
* an ``onset_curves.npz`` beside the session, so frame-by-frame review draws
  the curve under the label it is on.

Both formats are the ones the LightGBM lightgbm model writes, unchanged. The
curves file is model-agnostic by design (``(time, {label: curve})``, numpy
only), and every model's run folder follows
:data:`~ethograph.labels.onset_curves.RUN_PREFIX` — ``predictions_{model}_{timestamp}``.

**Nothing is dropped for being uncertain.** A low-confidence prediction is
written and flagged; a missing label cannot be reviewed, and review is the
point.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ethograph.labels.intervals import LABELING_AUTOMATED, NO_RECIPIENT
from ethograph.spot.confidence import DEFAULT_RULE, CurveStats, confidence_of, curve_events, densify, window_samples
from ethograph.spot.config import ResolvedClip, SpotConfig

logger = logging.getLogger(__name__)

#: What ``event_type`` a spotted event is. Point events have no offset.
POINT = "point"


@dataclass(frozen=True)
class SpottedEvent:
    """One class's predicted moment in one trial."""

    video_id: str
    label: int
    #: Frame on the video's full-rate clock.
    frame: float
    #: Seconds on the video's clock (``frame / fps``).
    video_s: float
    stats: CurveStats
    #: The rule ``confidence`` is read by (``infer.confidence`` / ``confidence_alpha``).
    rule: str = DEFAULT_RULE
    alpha: float = 0.5

    @property
    def confidence(self) -> float:
        return confidence_of(self.stats, self.rule, self.alpha)


def read_predictions(path: Path) -> list[dict]:
    """E2E-Spot's per-video predictions, from either file it writes.

    ``pred-{split}.{epoch}.json`` holds only the argmax — a class appears
    where its score beats background — while ``.recall.json.gz`` holds every
    frame above a low threshold. For one event per class per trial the tallest
    peak of the class's own curve is the answer whatever background says, so
    prefer the recall file; the argmax one still reads.
    """
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


def _curve_length(entry: dict, clip: ResolvedClip, num_frames: int | None, frames: np.ndarray) -> int:
    """How long this trial's curve is, on the prediction's (strided) clock.

    The trial's own length first — ``num_frames`` on the full-rate clock,
    floored by the stride as the dataset floors it — so every class's curve
    spans the whole trial and shares one time axis. Upstream's recall files
    carry no length, so without one the curve would end at its last candidate
    and two classes of one trial would disagree about how long it is.
    """
    if "num_frames" in entry:
        return int(entry["num_frames"])
    if num_frames is not None:
        return int(num_frames) // clip.stride
    return int(frames.max()) + 1 if frames.size else 1


def spot_entry(
    entry: dict, config: SpotConfig, clip: ResolvedClip, num_frames: int | None = None
) -> tuple[list[SpottedEvent], dict[int, np.ndarray]]:
    """One video's events and dense curves, on the video's full-rate clock.

    A strided run predicts on a downsampled clock and says so through its own
    ``fps``; the peak index comes back through
    :meth:`~ethograph.spot.config.ResolvedClip.to_frame`, which lands on the
    **centre** of the bin. *num_frames* is the trial's length on the full-rate
    clock, for entries that do not state their own.

    A class's curve is read as ``infer.max_events_per_trial`` events at most
    (:func:`~ethograph.labels.curve_confidence.curve_events`), in time order.
    """
    by_class: dict[str, list[tuple[int, float]]] = {}
    for event in entry.get("events", []):
        by_class.setdefault(str(event["label"]), []).append((int(event["frame"]), float(event.get("score", 1.0))))

    pred_fps = float(entry["fps"])
    window = window_samples(config.infer.focus_window_ms / 1000.0, pred_fps)
    gap = window_samples(config.infer.min_event_gap_s, pred_fps)
    events: list[SpottedEvent] = []
    curves: dict[int, np.ndarray] = {}
    for name, candidates in by_class.items():
        try:
            label = config.class_label(name)
        except ValueError:
            logger.warning("%s: prediction names class %r which this config does not spot", entry["video"], name)
            continue
        frames = np.array([c[0] for c in candidates], dtype=int)
        scores = np.array([c[1] for c in candidates], dtype=np.float64)
        curve = densify(frames, scores, _curve_length(entry, clip, num_frames, frames))
        curves[label] = curve
        for stats in curve_events(curve, window, gap, config.infer.max_events_per_trial):
            full_frame = clip.to_frame(stats.index)
            events.append(
                SpottedEvent(
                    video_id=str(entry["video"]),
                    label=label,
                    frame=full_frame,
                    video_s=full_frame / clip.fps,
                    stats=stats,
                    rule=config.infer.confidence,
                    alpha=config.infer.confidence_alpha,
                )
            )
    return events, curves


def to_labels_frame(
    events: list[SpottedEvent],
    trials: dict[str, tuple[int | str, float]],
    source: str,
    individual: str,
) -> pd.DataFrame:
    """Predicted events as label rows, on the **trial-relative** clock.

    *trials* maps ``video_id`` to ``(trial id, stream offset)``. The offset is
    the one ``VideoSync`` convention — ``trial = video + offset`` — so a video
    time becomes a trial time by adding it. Getting this backwards shifts
    every prediction by the offset and is invisible in the result.

    *individual* is stamped into every row alike. It is never blank: a labels
    TSV with a blank actor is refused on load, and the GUI draws a label only
    on the panels of the individual it names.
    """
    rows = []
    for event in events:
        trial, offset = trials[event.video_id]
        rows.append(
            {
                "trial": trial,
                "onset_s": event.video_s + offset,
                "offset_s": np.nan,
                "labels": int(event.label),
                "individual": individual,
                "individual_rec": NO_RECIPIENT,
                "event_type": POINT,
                "confidence": event.confidence,
                "labeling_method": LABELING_AUTOMATED,
                "changepoint_corrected": 0,
                "prediction_source": source,
                "n_samples": 0,
            }
        )
    return pd.DataFrame(rows)


def flagged(events: list[SpottedEvent], config: SpotConfig) -> list[SpottedEvent]:
    """The events whose confidence sits below ``infer.flag_confidence_below``.

    Reported, never removed — see this module's docstring.
    """
    return [e for e in events if e.confidence < config.infer.flag_confidence_below]
