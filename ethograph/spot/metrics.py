"""How well a run spots on a labelled split — the test summary, in durations.

The trainer leaves ``pred-{split}.{epoch}.recall.json.gz`` files behind and
``loss.json`` carries a ``val_mAP`` that the ladder found too strict and too
noisy to rank runs by. What the proof of principle asks is simpler: for each
class, how many labelled events were spotted at all, and how far off they
were — read at the tolerances a curator cares about, in milliseconds, so a
run at 200 fps and one at 25 fps answer the same question.

:func:`evaluate_run` scores a run's chosen epoch on ``dataset/{split}.json``,
producing the split's predictions through upstream's ``test_e2e.py`` when
the trainer did not, and writes ``test_metrics.yaml`` beside the checkpoints
— the same file name the segmentation pipeline uses for the same purpose.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ethograph.spot.config import ResolvedClip, SpotConfig
from ethograph.spot.predict import read_predictions, spot_entry

logger = logging.getLogger(__name__)

TEST_METRICS_FILE = "test_metrics.yaml"
#: A run that reads ``features:``, scored with them zeroed.
NOFEATURES_METRICS_FILE = "test_metrics_nofeatures.yaml"

#: The tolerances a hit is counted at — durations, never frames.
TOLERANCES_MS = (10, 20, 50, 100)


@dataclass
class ClassScore:
    """One class's record on one split."""

    label: int
    name: str
    #: Labelled events of this class in the split.
    n_truth: int = 0
    #: Labelled events the run put nothing on (no candidate at all).
    n_missing: int = 0
    #: Predictions in trials the split labels but where this class did not happen.
    n_spurious: int = 0
    #: ``|predicted - labelled|`` in milliseconds, one per labelled event that got a prediction.
    errors_ms: list[float] = field(default_factory=list)

    @property
    def n_predicted(self) -> int:
        return len(self.errors_ms)

    def hit_rate(self, tolerance_ms: float) -> float:
        """Fraction of the labelled events landing within *tolerance_ms* (a miss counts against)."""
        if self.n_truth == 0:
            return float("nan")
        return float(sum(e <= tolerance_ms for e in self.errors_ms)) / self.n_truth

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.errors_ms)) if self.errors_ms else float("nan")

    @property
    def median_ms(self) -> float:
        return float(np.median(self.errors_ms)) if self.errors_ms else float("nan")

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n_truth": self.n_truth,
            "n_predicted": self.n_predicted,
            "n_missing": self.n_missing,
            "n_spurious": self.n_spurious,
            "mean_error_ms": _finite(self.mean_ms),
            "median_error_ms": _finite(self.median_ms),
            "hit_rate": {f"{t}ms": _finite(self.hit_rate(t)) for t in TOLERANCES_MS},
        }


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def score_predictions(
    entries: list[dict], truth: list[dict], config: SpotConfig, clip: ResolvedClip
) -> dict[int, ClassScore]:
    """Every class's record: *entries* (upstream's recall file) against *truth* (the split's JSON).

    A prediction comes back on the full-rate clock through :func:`spot_entry`
    — the bin's centre, so a strided run is not read as systematically early
    — and the error is in milliseconds of that clock.
    """
    scores = {label: ClassScore(label, config.class_name(label)) for label in config.labels.classes}
    predicted = {str(entry["video"]): entry for entry in entries}
    for gt in truth:
        video = str(gt["video"])
        fps = float(gt["fps"])
        entry = predicted.get(video)
        events = spot_entry(entry, config, clip, num_frames=int(gt["num_frames"]))[0] if entry else []
        by_label = {e.label: e for e in events}
        labelled = {config.class_label(ev["label"]): int(ev["frame"]) for ev in gt["events"]}
        for label, frame in labelled.items():
            score = scores[label]
            score.n_truth += 1
            hit = by_label.get(label)
            if hit is None:
                score.n_missing += 1
            else:
                score.errors_ms.append(abs(hit.frame - frame) / fps * 1000.0)
        for label in by_label:
            if label not in labelled and label in scores:
                scores[label].n_spurious += 1
    return scores


def format_table(metrics: dict) -> str:
    """The summary as one readable block, the way the ladder printed it."""
    header = "  ".join(f"<={t:<3}ms" for t in TOLERANCES_MS)
    lines = [
        f"{metrics['run']} epoch {metrics['epoch']} on {metrics['split']}: "
        f"{metrics['n_trials']} trials, {metrics['n_events']} labelled events",
        f"{'class':<12} {'n':>4} {'miss':>5} {'spur':>5} {'mean':>7} {'med':>7}   {header}",
    ]
    for name, row in metrics["classes"].items():
        mean = f"{row['mean_error_ms']:7.1f}" if row["mean_error_ms"] is not None else f"{'-':>7}"
        med = f"{row['median_error_ms']:7.1f}" if row["median_error_ms"] is not None else f"{'-':>7}"
        hits = "  ".join(
            f"{row['hit_rate'][f'{t}ms']:>7.0%}" if row["hit_rate"][f"{t}ms"] is not None else f"{'-':>7}"
            for t in TOLERANCES_MS
        )
        lines.append(
            f"{name:<12} {row['n_truth']:>4} {row['n_missing']:>5} {row['n_spurious']:>5} {mean} {med}   {hits}"
        )
    lines.append(
        "miss = labelled events with no prediction; spur = predictions where the class did not happen; "
        "mean/med = absolute error of the rest; <=k = share of labelled events within k."
    )
    return "\n".join(lines)


def split_predictions(config: SpotConfig, run_dir: Path, epoch: int, split: str, zero_features: bool = False) -> Path:
    """``pred-{split}.{epoch}[.nofeatures].recall.json.gz`` in *run_dir*, made through ``test_e2e.py`` if absent."""
    from ethograph.spot.inference import predict_split, stage_checkpoint

    stem = f"pred-{split}.{epoch}" + (".nofeatures" if zero_features else "")
    path = run_dir / f"{stem}.recall.json.gz"
    if path.is_file():
        return path
    logger.info("%s: no %s predictions for epoch %d — running test_e2e.py", run_dir.name, stem, epoch)
    with tempfile.TemporaryDirectory(prefix="spot_eval_") as tmp:
        staged = stage_checkpoint(run_dir, epoch, Path(tmp) / "model")
        predict_split(config, staged, split, run_dir / stem, run_dir / "evaluate.log", zero_features=zero_features)
    if not path.is_file():
        raise FileNotFoundError(f"test_e2e.py finished but wrote no {path}")
    return path


COMPARE_FILE = "compare.tsv"


def compare_runs(config: SpotConfig) -> pd.DataFrame:
    """One row per trained run that has a ``test_metrics.yaml``: written to ``runs/compare.tsv`` and returned.

    The columns are the per-class summary flattened — ``{class}.miss``,
    ``.spur``, ``.med_ms``, ``.mean_ms`` and ``.hit{tolerance}ms`` — so two
    runs are read side by side. A run not yet scored is left out, not
    scored here: :func:`evaluate_run` is the one place that produces a test
    prediction.
    """
    from ethograph.spot.inference import trained_runs

    rows = []
    for run_dir in trained_runs(config):
        path = run_dir / TEST_METRICS_FILE
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        row: dict[str, object] = {"run": data["run"], "epoch": data["epoch"]}
        for name, cls in data["classes"].items():
            row[f"{name}.miss"] = cls["n_missing"]
            row[f"{name}.spur"] = cls["n_spurious"]
            row[f"{name}.med_ms"] = cls["median_error_ms"]
            row[f"{name}.mean_ms"] = cls["mean_error_ms"]
            for key, value in cls["hit_rate"].items():
                row[f"{name}.hit{key}"] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        config.runs_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.runs_dir / COMPARE_FILE, sep="\t", index=False, float_format="%.3f")
    return df


def evaluate_run(
    config: SpotConfig, run_dir: Path, split: str = "test", epoch: int | None = None, zero_features: bool = False
) -> dict:
    """Score *run_dir* on ``dataset/{split}.json`` and write ``test_metrics.yaml`` beside its checkpoints.

    *epoch* defaults to the one the sweep ranks first on the run's own
    validation predictions — the epoch inference uses — so the number
    reported is the number the GUI will see. *zero_features* scores a run
    that reads ``features:`` with zeros in their place, into
    ``test_metrics_nofeatures.yaml``.
    """
    from ethograph.spot.inference import best_epoch, run_clip, run_label, run_reads_features

    if zero_features and not run_reads_features(run_dir):
        raise ValueError(f"{run_dir.name} reads no features — there is nothing to zero")

    truth_path = config.dataset_dir / f"{split}.json"
    if not truth_path.is_file():
        raise FileNotFoundError(f"No {truth_path} — materialise() writes the splits")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if not truth:
        raise ValueError(f"{truth_path} lists no trials")
    if epoch is None:
        epoch = best_epoch(run_dir, config)
    fps = float(truth[0]["fps"])
    clip = run_clip(run_dir, fps)
    entries = read_predictions(split_predictions(config, run_dir, epoch, split, zero_features=zero_features))
    scores = score_predictions(entries, truth, config, clip)
    metrics = {
        "run": run_label(run_dir) + (" (features zeroed)" if zero_features else ""),
        "run_dir": str(run_dir),
        "epoch": epoch,
        "split": split,
        "features": (None if not run_reads_features(run_dir) else "zeroed" if zero_features else "on"),
        "n_trials": len(truth),
        "n_events": sum(len(t["events"]) for t in truth),
        "fps": fps,
        "tolerances_ms": list(TOLERANCES_MS),
        "classes": {score.name: score.to_dict() for score in scores.values()},
    }
    out = run_dir / (NOFEATURES_METRICS_FILE if zero_features else TEST_METRICS_FILE)
    out.write_text(yaml.safe_dump(metrics, sort_keys=False), encoding="utf-8")
    logger.info("\n%s\n-> %s", format_table(metrics), out)
    return metrics
