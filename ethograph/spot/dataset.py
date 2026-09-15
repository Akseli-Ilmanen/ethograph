"""Stage 1: sessions become the layout E2E-Spot reads.

This is the pixel pipeline's :func:`~ethograph.segment.materialise.materialise`
— the one stage that turns a config's sessions into files the model can be
pointed at, and like it, **role-agnostic**: one export serves every run and
every cross-validation fold, because roles live in the run.

What it writes, under ``config.root``::

    frames/{video_id}/000000.jpg ...      # one folder per trial, resized
    dataset/train.json                    # E2E-Spot's per-video schema
    dataset/val.json
    dataset/test.json
    dataset/class.txt                     # class name per line, 1-indexed
    dataset/index.tsv                     # our own provenance: key -> trial

``video_id`` is ``{session name}_trial{trial}`` (``SessionSpec.label``: the
config's ``name``, else the source's stem) — the key the split, the index
and every prediction file agree on.

Decoding is the expensive part and it resumes: a trial whose folder already
holds the expected frame count *and* was exported at the same size and crop
(``export.json`` beside the frames) is left alone.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ethograph.io.video_decode import iter_rgb_frames
from ethograph.labels.intervals import LABELING_AUTOMATED
from ethograph.labels.tsv_store import get_trial_from_tsv
from ethograph.segment.config import SessionSpec
from ethograph.segment.sessions import Session, filter_trials, open_session
from ethograph.segment.train import assign_roles
from ethograph.spot.config import SpotConfig

logger = logging.getLogger(__name__)

#: JPEG quality for exported frames. Upstream's own.
JPEG_QUALITY = 90

#: Written beside a trial's frames: the size and crop they were exported at,
#: so a config that changes either re-decodes instead of reusing the folder.
EXPORT_FILE = "export.json"

CLASS_FILE = "class.txt"
INDEX_FILE = "index.tsv"


@dataclass
class TrialRecord:
    """One trial as E2E-Spot's per-video schema wants it."""

    video_id: str
    source: Path
    trial: int | str
    video_path: Path
    num_frames: int
    fps: float
    width: int
    height: int
    #: class name -> frame index on the video's own clock.
    events: dict[str, int]
    #: ``(x0, y0, x1, y1)`` source pixels cut out before the resize; ``None``
    #: = the whole frame.
    crop: tuple[int, int, int, int] | None = None

    def export_spec(self) -> dict:
        """What decides the pixels on disk -- the content of ``export.json``."""
        return {"width": self.width, "height": self.height, "crop": list(self.crop) if self.crop else None}

    def to_json(self) -> dict:
        return {
            "video": self.video_id,
            "num_frames": self.num_frames,
            "num_events": len(self.events),
            "events": [{"frame": frame, "label": name, "comment": ""} for name, frame in sorted(self.events.items())],
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }


def probe_video(video: Path) -> tuple[float, int, int, int]:
    """``(fps, n_frames, width, height)``, read from the container itself.

    The rate is never a setting: a clip length in seconds means nothing
    without the rate the video was actually recorded at.
    """
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None:
            raise ValueError(f"Cannot determine frame rate of {video}")
        return float(rate), int(stream.frames), stream.codec_context.width, stream.codec_context.height


def point_events(session: Session, trial: int | str, classes: Iterable[int]) -> dict[int, float]:
    """``{label: onset_s}`` for the target point events of one trial.

    Only ``manual``/``curated`` rows: an automated label is a model's own
    output and training on it would be training on a prediction. Onsets are
    trial-relative, as every labels TSV is.
    """
    df = get_trial_from_tsv(session.result.all_labels_df, trial)
    if df.empty:
        return {}
    wanted = {int(c) for c in classes}
    rows = df[
        (df["event_type"] == "point")
        & (df["labels"].astype(int).isin(wanted))
        & (df["labeling_method"] != LABELING_AUTOMATED)
    ]
    return {int(row.labels): float(row.onset_s) for row in rows.itertuples()}


def event_frame(onset_s: float, offset_s: float, fps: float) -> int:
    """A trial-relative onset as a frame index on the video's clock.

    ``VideoSync``'s convention is ``trial = video + offset``, so a trial time
    samples the video at ``t - offset``. Getting this backwards shifts every
    label by the stream offset and is invisible in the result, which is why
    the conversion is written once, here.
    """
    return int(round((onset_s - offset_s) * fps))


def plan_session(session: Session, config: SpotConfig, *, require_events: bool = True) -> list[TrialRecord]:
    """What :func:`export_frames` would write for one session, decoding nothing.

    Training wants only trials that carry a target event; inference wants
    every trial that has video, which is what ``require_events=False`` gives.
    """
    alignment = session.result.nwb_alignment
    camera = session.video_device(config.labels.camera)
    records: list[TrialRecord] = []
    for trial in filter_trials(session, config.trials):
        events = point_events(session, trial, config.labels.classes)
        if not events and require_events:
            continue
        video = session.media_path(trial, "video", device=camera)
        if video is None:
            logger.warning("%s trial %s: no %s video, skipped", session.spec.label, trial, camera or "default")
            continue
        fps, n_frames, width, height = probe_video(video)
        crop = config.labels.crop
        if crop is not None:
            crop.check_fits(width, height, f"{session.spec.label} trial {trial} ({video.name})")
            width, height = crop.width, crop.height
        offset = float(alignment.stream_offset_for_trial(trial, "video", device=camera))
        frames = {config.class_name(label): event_frame(t, offset, fps) for label, t in events.items()}
        outside = {name: f for name, f in frames.items() if not 0 <= f < n_frames}
        if outside:
            logger.warning("%s trial %s: events %s fall outside the video, skipped", session.spec.label, trial, outside)
            continue
        scale = config.labels.frame_height / height
        records.append(
            TrialRecord(
                video_id=f"{session.spec.label}_trial{trial}",
                source=session.source,
                trial=trial,
                video_path=video,
                num_frames=n_frames,
                fps=fps,
                width=int(round(width * scale)),
                height=config.labels.frame_height,
                events=frames,
                crop=None if crop is None else crop.as_tuple(),
            )
        )
    return records


def _iter_frames(video: Path, decode_threads: int | None = None) -> Iterator[np.ndarray]:
    """Decoded RGB frames of *video*, in order (:func:`ethograph.io.video_decode.iter_rgb_frames`)."""
    return iter_rgb_frames(video, threads=decode_threads)


def export_is_current(out_dir: Path, record: TrialRecord) -> bool:
    """Whether *out_dir* already holds what *record* would write.

    The frame count must match, and so must the size and crop recorded in
    ``export.json`` -- a folder decoded before crops existed carries no such
    file and is a full-frame export, so it counts as current only when
    *record* asks for the whole frame. Anything else is re-decoded, with the
    reason logged: silently reusing a folder cut at a different box would
    train on pixels the config never named.
    """
    if not out_dir.is_dir() or len(list(out_dir.glob("*.jpg"))) != record.num_frames:
        return False
    spec_path = out_dir / EXPORT_FILE
    if not spec_path.is_file():
        return record.crop is None
    stored = json.loads(spec_path.read_text(encoding="utf-8"))
    if stored == record.export_spec():
        return True
    logger.warning(
        "%s: frames on disk were exported as %s, the config asks for %s -- re-decoding",
        record.video_id,
        stored,
        record.export_spec(),
    )
    return False


def export_frames(record: TrialRecord, frames_dir: Path) -> int:
    """Write one trial's frames as ``{frames_dir}/{video_id}/%06d.jpg``.

    Returns the number written, which the caller trusts over the container's
    claimed frame count. A folder that is already current
    (:func:`export_is_current`) is left alone, so an interrupted export
    resumes rather than starting over. The crop, if any, is cut from the
    decoded frame first and the resize applies to what is left.
    """
    from PIL import Image

    out_dir = frames_dir / record.video_id
    if export_is_current(out_dir, record):
        return record.num_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (record.width, record.height)
    crop = record.crop
    written = 0
    for i, frame in enumerate(_iter_frames(record.video_path)):
        if crop is not None:
            x0, y0, x1, y1 = crop
            frame = frame[y0:y1, x0:x1]
        Image.fromarray(frame).resize(size, Image.BILINEAR).save(out_dir / f"{i:06d}.jpg", quality=JPEG_QUALITY)
        written += 1
    (out_dir / EXPORT_FILE).write_text(json.dumps(record.export_spec()), encoding="utf-8")
    return written


def _export_one(job: tuple[TrialRecord, Path]) -> tuple[str, int]:
    record, frames_dir = job
    return record.video_id, export_frames(record, frames_dir)


#: Decoders left idle so the GUI, the trainer and the disk still get a core each.
_SPARE_CPUS = 2
#: More threads than this only queue on the disk the JPEGs land on.
MAX_WORKERS = 16


def default_workers() -> int:
    """How many trials to decode at once: the machine's cores minus a couple, capped.

    Like ``resolve_device()``, decided at runtime rather than written into
    the config — a worker count is a property of the machine, and a YAML
    that moves between machines must not carry one.
    """
    return max(1, min(MAX_WORKERS, (os.cpu_count() or 1) - _SPARE_CPUS))


def export_all(records: list[TrialRecord], frames_dir: Path, workers: int | None = None) -> list[TrialRecord]:
    """Decode every record, in parallel, and return the ones that came out whole.

    The cost is the resize and the JPEG encode rather than the H.264 decode,
    which is why this spends cores on it — with **threads**: the
    conversion, the resize and the encode all run in C and release the GIL,
    and a process pool would spawn workers that re-import the caller's
    script, which on Windows re-runs a ``run.py`` without a ``__main__`` guard
    inside every worker. A trial whose written frame count disagrees with the
    container's claim is dropped with a warning — its labels would point at
    frames that are not there.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(record, frames_dir) for record in records]
    counts: dict[str, int] = {}
    if workers is None:
        workers = default_workers()
    if workers <= 1:
        for job in jobs:
            video_id, n = _export_one(job)
            counts[video_id] = n
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for video_id, n in pool.map(_export_one, jobs):
                counts[video_id] = n
    kept: list[TrialRecord] = []
    for record in records:
        n = counts.get(record.video_id, 0)
        if n != record.num_frames:
            logger.warning(
                "%s: wrote %d frames but the container claims %d — dropped", record.video_id, n, record.num_frames
            )
            continue
        kept.append(record)
    return kept


def _index_frame(records: list[TrialRecord]) -> pd.DataFrame:
    """The provenance table ``assign_roles`` splits and we read back."""
    return pd.DataFrame(
        [
            {
                "key": r.video_id,
                "source": str(r.source),
                "trial": str(r.trial),
                "fps": r.fps,
                "num_frames": r.num_frames,
                "num_events": len(r.events),
            }
            for r in records
        ]
    )


def split_records(records: list[TrialRecord], config: SpotConfig) -> dict[str, list[TrialRecord]]:
    """Deal records into train/val/test by whole trial, via ``train.split``.

    Delegates to :func:`~ethograph.segment.train.assign_roles` — the same
    draw, the same seed, and the same ``holdout_sessions`` meaning, so a
    cross-validation fold is written the same way in both pipelines.
    """
    index = _index_frame(records)
    if index.empty:
        raise ValueError("Nothing to split — no trial carried a target point event with video")
    roles = assign_roles(config, index)  # type: ignore[arg-type]
    out: dict[str, list[TrialRecord]] = {"train": [], "val": [], "test": []}
    for record in records:
        out[roles[record.video_id]].append(record)
    return out


def write_dataset(splits: dict[str, list[TrialRecord]], config: SpotConfig) -> Path:
    """Write ``{split}.json``, ``class.txt`` and ``index.tsv``."""
    dataset_dir = config.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        path = dataset_dir / f"{split}.json"
        path.write_text(json.dumps([r.to_json() for r in records], indent=2), encoding="utf-8")
        logger.info("%s: %d trials, %d events", path.name, len(records), sum(len(r.events) for r in records))
    names = [config.class_name(label) for label in config.labels.classes]
    (dataset_dir / CLASS_FILE).write_text("\n".join(names) + "\n", encoding="utf-8")
    every = [r for records in splits.values() for r in records]
    _index_frame(every).to_csv(dataset_dir / INDEX_FILE, sep="\t", index=False)
    return dataset_dir


def open_spot_session(config: SpotConfig, spec: SessionSpec) -> Session:
    """Open *spec* the way this pipeline reads it: its ``label_inputs`` rendered for ``config.individual``.

    The one opener ``materialise`` and ``inference`` share, so the columns a
    run trained on are rendered identically for a session predicted later.
    """
    return open_session(spec, label_inputs=config.label_inputs, actor=config.individual)


@dataclass
class MaterialiseResult:
    """What the export produced: the index the model reads, and what fed it."""

    dataset_dir: Path
    sessions: list[Session]
    #: The trials the frame export kept, in plan order — the same set every
    #: other modality must cover.
    records: list[TrialRecord]


def materialise(
    config: SpotConfig, workers: int | None = None, sessions: list[SessionSpec] | None = None
) -> MaterialiseResult:
    """Run the whole stage: open, plan, decode, split, index.

    *sessions* defaults to every session in the config.
    """
    specs = config.sessions if sessions is None else sessions
    opened: list[Session] = []
    records: list[TrialRecord] = []
    for spec in specs:
        session = open_spot_session(config, spec)
        opened.append(session)
        planned = plan_session(session, config)
        logger.info("%s: %d trials with a target event and video", session.spec.label, len(planned))
        records.extend(planned)
    if not records:
        raise ValueError(
            "No trial in any session carries one of "
            f"{config.labels.classes} as a manual/curated point event with video — "
            "check labels.classes, labels.camera and trials.where."
        )
    records = export_all(records, config.frames_dir, workers)
    dataset_dir = write_dataset(split_records(records, config), config)
    return MaterialiseResult(dataset_dir=dataset_dir, sessions=opened, records=records)
