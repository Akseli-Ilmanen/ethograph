"""The pose side of ``materialise``: the features a config lists, one file per trial.

A point event is a relation between things — a beak, a stick, a pellet — and
the user knows which relations matter better than any architecture can
discover from three hundred events. So the pose side of this pipeline is
**nothing but the variables you list**, in the segmentation pipeline's
column spelling (:func:`~ethograph.features.columns.extract_features`, the
same path the lightgbm model reads through)::

    features:
      velocity: {space: [x, y], keypoint: [stickTip, pellet]}
      pellet_stickClosest_dist: {}

Every one of them is a variable in the session's ``.nc`` — built with
``features/geometry.py`` or your own code, plottable in the GUI before a
model ever sees it. No graph, no adjacency, no learned geometry.

One file per trial, ``features/{video_id}.npz``::

    time      (T,)      trial-relative seconds, the features' own clock
    x         (T, F)    the listed columns, NaN where a value is missing
    events    (E,)      frame index on this clock, per target event
    labels    (E,)      class id per event
    fps       ()        the features' sampling rate

plus ``features/features.json`` — the column names, in order — so the block
and the model reading it cannot disagree about which column is which.

The **pixel model** reads them as a second input beside the CNN features; for
that they are written once more, z-scored on the training split, under
``features/block/`` (:func:`export_block`), with the statistics saved so a
session predicted later is put on the training scale rather than its own.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ethograph.features.columns import enumerate_columns, extract_features, sampling_rate
from ethograph.segment.sessions import Session
from ethograph.spot.config import ResolvedClip, SpotConfig
from ethograph.spot.dataset import TrialRecord

logger = logging.getLogger(__name__)

NAMES_FILE = "features.json"
BLOCK_INFO_FILE = "block.json"
STATS_FILE = "stats.npz"


def feature_names(features: dict[str, dict]) -> list[str]:
    """The column names *features* expands to, in the order they are assembled."""
    return [c.name for c in enumerate_columns(features)]


def trial_features(session: Session, trial: int | str, features: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    """``(time, x (T, F))`` for one trial, on the features' own clock.

    Every column comes out of ``extract_features`` with the dims the config
    pins; a session with several individuals names one per feature
    (``{individual: [name]}``) or reads them all, segment-style.
    """
    if not features:
        raise ValueError("features: is empty — nothing for the pose side to read")
    window = next(w for w in session.trial_windows([trial]))
    time, data = extract_features(window.loader, features, window.t0, window.t1)
    time = np.asarray(time, dtype=np.float64) - window.shift
    return time, np.asarray(data, dtype=np.float32)


def write_trial_features(path: Path, time: np.ndarray, x: np.ndarray, events: dict[int, list[float]]) -> Path:
    """Write one trial's features plus its events as indices on this clock (``labels`` and ``events`` run parallel)."""
    fs = sampling_rate(time)
    labels = [label for label in sorted(events) for _ in events[label]]
    frames = [int(np.searchsorted(time, t)) for label in sorted(events) for t in events[label]]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time=time.astype(np.float64),
        x=x.astype(np.float32),
        events=np.asarray(frames, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        fps=np.float64(fs),
    )
    return path


def read_trial_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {key: np.asarray(npz[key]) for key in ("time", "x", "events", "labels", "fps")}


def read_names(features_dir: Path) -> list[str]:
    return list(json.loads((features_dir / NAMES_FILE).read_text(encoding="utf-8"))["names"])


def _events_on_trial_clock(record: TrialRecord, config: SpotConfig) -> dict[int, list[float]]:
    """The record's events back as trial-relative seconds.

    The record holds frames of its trial's window; ``trial = video + offset``
    (the ``VideoSync`` convention, the window's start folded into
    ``offset_s``) takes them back to the clock the features are on.
    """
    return {
        config.class_label(name): [f / record.fps + record.offset_s for f in frames]
        for name, frames in record.events.items()
    }


def export_features(config: SpotConfig, sessions: list[Session], records: list[TrialRecord]) -> Path:
    """Write every materialised trial's features under ``features/``.

    *records* are the trials the frame export kept, keyed by ``video_id``, so
    the pose and pixel sides cover exactly the same trials.
    """
    if not config.features:
        raise ValueError("features: is empty — nothing to export")
    by_source = {str(s.source): s for s in sessions}
    out_dir = config.features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    names = feature_names(config.features)
    (out_dir / NAMES_FILE).write_text(json.dumps({"names": names}, indent=2), encoding="utf-8")
    for record in records:
        session = by_source[str(record.source)]
        time, x = trial_features(session, record.trial, config.features)
        if x.shape[1] != len(names):
            raise ValueError(f"{record.video_id}: {x.shape[1]} columns for {len(names)} names — the layout drifted")
        write_trial_features(out_dir / f"{record.video_id}.npz", time, x, _events_on_trial_clock(record, config))
    logger.info("features/: %d trials x %d columns %s", len(records), len(names), names)
    return out_dir


# ---------------------------------------------------------------------------
# The block: the same columns, z-scored, as the pixel model's second input
# ---------------------------------------------------------------------------


class Stats:
    """Per-column mean/std from the training trials; a missing value reads as 0 after scaling."""

    def __init__(self, xs: list[np.ndarray]) -> None:
        stacked = np.concatenate(xs, axis=0)
        self.mean = np.nanmean(stacked, axis=0)
        self.std = np.nanstd(stacked, axis=0)
        self.std[~np.isfinite(self.std) | (self.std < 1e-6)] = 1.0
        self.mean[~np.isfinite(self.mean)] = 0.0

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - self.mean) / self.std, nan=0.0).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> Stats:
        obj = cls.__new__(cls)
        with np.load(path) as npz:
            obj.mean, obj.std = npz["mean"], npz["std"]
        return obj


def load_split(config: SpotConfig, split: str) -> list[str]:
    path = config.dataset_dir / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No {split}.json under {config.dataset_dir} — run project.materialise() first")
    return [entry["video"] for entry in json.loads(path.read_text(encoding="utf-8"))]


def strided(raw: dict[str, np.ndarray], stride: int, fps: float) -> np.ndarray:
    """The trial's columns on the strided clock the models read."""
    if not np.isclose(float(raw["fps"]), fps, rtol=1e-3):
        raise ValueError(f"features at {float(raw['fps']):g} Hz but the clip was resolved at {fps:g}")
    return raw["x"][::stride].astype(np.float32)


def export_block(
    config: SpotConfig,
    video_ids: list[str] | None = None,
    *,
    stats: Stats | None = None,
    clip: "ResolvedClip | None" = None,
) -> Path:
    """Write ``features/block/`` for *video_ids* (default: every trial of every split).

    *stats* ``None`` fits the scaling on the training split and saves it;
    passing the saved one is how a session predicted later is put on the
    training split's scale rather than its own. *clip* is the run's, for a
    session predicted later — its stride, not the one this card would pick
    (:meth:`~ethograph.spot.config.SpotConfig.resolve_clip`). Each file is
    ``features (T', F)`` on the strided clock plus ``stride``/``fps``, the
    shape the vendored trainer's ``--fuse_dir`` loader reads.
    """
    if not config.features:
        raise ValueError("features: is empty — there is no block to export")
    if video_ids is None:
        video_ids = [v for split in ("train", "val", "test") for v in load_split(config, split)]
    if not video_ids:
        raise ValueError("block: no trials to export")
    raw = {v: read_trial_features(config.features_dir / f"{v}.npz") for v in video_ids}
    fps = float(next(iter(raw.values()))["fps"])
    if clip is None:
        clip = config.resolve_clip(fps)
    elif not np.isclose(clip.fps, fps, rtol=1e-3):
        raise ValueError(f"the run's clip is at {clip.fps:g} fps but these features are at {fps:g}")
    out_dir = config.block_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if stats is None:
        train = load_split(config, "train")
        if not train:
            raise ValueError("block: train.json lists no trials to fit the statistics on")
        stats = Stats([strided(read_trial_features(config.features_dir / f"{v}.npz"), clip.stride, fps) for v in train])
        stats.save(out_dir / STATS_FILE)
    names = read_names(config.features_dir)
    for video_id, trial in raw.items():
        block = stats.apply(strided(trial, clip.stride, fps))
        np.savez_compressed(out_dir / f"{video_id}.npz", features=block, stride=clip.stride, fps=clip.fps)
    info = {"dim": len(names), "stride": clip.stride, "names": names}
    (out_dir / BLOCK_INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("features/block/: %d trials x %d columns -> %s", len(video_ids), len(names), out_dir)
    return out_dir


def block_dim(config: SpotConfig) -> int:
    """The block's width, from ``block.json``."""
    return int(json.loads((config.block_dir / BLOCK_INFO_FILE).read_text(encoding="utf-8"))["dim"])


def export_block_for_inference(config: SpotConfig, video_ids: list[str], clip: "ResolvedClip | None" = None) -> Path:
    """The block of trials predicted after training, on the training split's scale and the run's stride."""
    stats_path = config.block_dir / STATS_FILE
    if not stats_path.is_file():
        raise FileNotFoundError(f"{stats_path} missing — the run read features but its statistics are gone")
    return export_block(config, video_ids, stats=Stats.load(stats_path), clip=clip)
