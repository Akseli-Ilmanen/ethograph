"""FERAL's inputs from a project's sessions — the export half of ``extractor: feral``.

FERAL (Skovorodnikov & Razzauti, 2025) learns per-frame classes straight
from video files and runs in its own environment (its pins clash with the
GUI's — ADR 0009). What crosses the boundary is files, in FERAL's own
layout, under ``{root}/feral/``:

* ``labels.json`` — FERAL's label file: one class per frame of every video,
  the class names, and the ``train`` / ``val`` / ``test`` / ``inference``
  splits.
* ``config.yaml`` — a complete FERAL config (``feral train-config`` reads
  it verbatim): its packaged defaults, the chosen ``preset`` laid over
  them, and what only this project knows — where the videos and labels are,
  and ``chunk_step`` resolved from ``video_features.context_s`` against the
  video's own rate.
* ``videos.tsv`` — which session and trial each video is, for reading
  results back; ``export.yaml`` — the FERAL class → label id mapping, the
  rate, the sessions FERAL never trains on.

Two things come back. FERAL's own per-frame predictions (``feral infer
--output`` → ``{root}/feral/predictions/``) become one prediction set per
session, in the GUI's labels format, through
:func:`import_feral_predictions` — FERAL alone, no segmentation model. And
its ``(frames, D)`` embeddings attach to every session as the variable
``feral`` from ``{root}/feral/embeddings/`` —
:func:`ethograph.segment.sessions.attach_feral_embeddings`, run when a
session opens under a config whose extractor is ``feral``.

**A FERAL video is one trial of one individual.** Labels are rasterised on
the video's own frame clock through the alignment (trial = video +
offset), so a video that starts before its trial is fine; a video that
holds several trials is not — the frames outside the trial would be
labelled background — and is refused once more than half a video lies
outside its trial.

**Leakage.** A FERAL fine-tuned on a session's labels embeds that session
better than an unseen one; a downstream model trained on those embeddings
then scores that session too well. ``train.split.holdout_sessions`` is the
one guard: those sessions go to FERAL as ``test`` and ``inference`` only,
never ``train`` or ``val``, so their embeddings come from a model that
never saw their labels — and downstream they are the test set, as always.
Without it, every downstream number over FERAL features is optimistic.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ethograph.io.video_probe import probe_video
from ethograph.labels.ml import dense_to_intervals
from ethograph.segment.config import FERAL, FERAL_EMBEDDINGS_DIR, SegmentConfig, config_to_dict
from ethograph.segment.inference import label_rows, prediction_run_dir, write_prediction_set
from ethograph.segment.postprocess import postprocess_intervals
from ethograph.segment.samples import ClassTable, class_table, dense_targets, sample_key
from ethograph.segment.sessions import Session, changepoint_times, filter_trials, open_session
from ethograph.segment.train import assign_roles
from ethograph.utils.logging import log_to_file
from ethograph.utils.xr_utils import get_time_coord

logger = logging.getLogger(__name__)

#: FERAL's packaged defaults, copied (see ``feral_defaults/NOTICE.md``).
DEFAULTS_DIR = Path(__file__).parent / "feral_defaults"
#: The FERAL release the copied defaults come from — named in the config header.
FERAL_VERSION = "1.0.1"

LABELS_JSON = "labels.json"
CONFIG_YAML = "config.yaml"
VIDEOS_TSV = "videos.tsv"
EXPORT_YAML = "export.yaml"
#: Where ``feral infer --save_embeddings`` is told to write, under ``feral_dir``.
EMBEDDINGS_DIR = FERAL_EMBEDDINGS_DIR
#: Where ``feral infer --output`` is told to write, under ``feral_dir``: one JSON per video folder.
PREDICTIONS_DIR = "predictions"
#: The variable FERAL's embeddings become on every trial (the extractor's name).
VARIABLE = FERAL
#: FERAL's name for class 0.
BACKGROUND = "other"
#: FERAL's own partitions, in the order its label file lists them.
ROLES = ("train", "val", "test")
#: A video with more of its frames outside its trial than this is refused.
MAX_OUTSIDE_FRACTION = 0.5

__all__ = [
    "BACKGROUND",
    "CONFIG_YAML",
    "EMBEDDINGS_DIR",
    "EXPORT_YAML",
    "FERAL_VERSION",
    "LABELS_JSON",
    "PREDICTIONS_DIR",
    "VARIABLE",
    "VIDEOS_TSV",
    "FeralExport",
    "VideoRecord",
    "export_feral",
    "feral_config",
    "import_feral_predictions",
    "load_defaults",
    "plan_videos",
    "predictions_file",
    "resolve_chunk_step",
]


@dataclass(frozen=True)
class VideoRecord:
    """One FERAL video: a trial's camera file, its labels on the video's frame clock."""

    key: str
    session_id: str
    source: Path
    trial: int | str
    individual: str
    video: Path
    fps: float
    n_frames: int
    #: FERAL's role for this video; ``inference`` is every video and not a role.
    role: str
    #: Class *indices* (the project's ``ClassTable``) per frame.
    labels: np.ndarray = field(repr=False, compare=False)
    #: Frames the trial does not cover, labelled background for want of better.
    outside: int = 0


@dataclass(frozen=True)
class FeralExport:
    """What :func:`export_feral` wrote, and the numbers the log line names."""

    folder: Path
    prefix: Path
    fps: float
    chunk_step: int
    n_videos: int
    #: FERAL class index → the project's label id (``0`` is background).
    label_ids: list[int]
    #: Label ids of the target that no train video holds, mapped to background.
    dropped: list[int]
    #: Sessions FERAL never trains on — ``test`` + ``inference`` only.
    holdout_sessions: list[str]

    @property
    def files(self) -> list[Path]:
        return [self.folder / name for name in (LABELS_JSON, CONFIG_YAML, VIDEOS_TSV, EXPORT_YAML)]


# ---------------------------------------------------------------------------
# FERAL's defaults
# ---------------------------------------------------------------------------


def load_defaults(preset: str | None = None) -> dict[str, Any]:
    """FERAL's ``default_config.yaml`` with *preset* deep-merged over it, as ``feral train --mode`` does."""
    cfg = yaml.safe_load((DEFAULTS_DIR / "default_config.yaml").read_text(encoding="utf-8"))
    if preset is None:
        return cfg
    presets = yaml.safe_load((DEFAULTS_DIR / "presets.yaml").read_text(encoding="utf-8"))
    if preset not in presets:
        raise ValueError(f"video_features.preset={preset!r} is not a FERAL preset; choose from {sorted(presets)}")
    return _deep_merge(cfg, presets[preset])


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_chunk_step(context_s: float | None, fps: float, chunk_length: int) -> int:
    """Upstream's ``chunk_step`` for a chunk of *chunk_length* frames to span *context_s* at *fps*.

    ``None`` is every frame — upstream's own chunk, whatever the rate. A
    duration lands on a whole number of frames, so the span is what the
    rate can carry, never exactly what was asked for.
    """
    if fps <= 0:
        raise ValueError(f"Frame rate must be positive, got {fps!r}")
    if context_s is None:
        return 1
    return max(1, int(round(context_s * fps / (chunk_length - 1))))


def feral_config(config: SegmentConfig, prefix: Path, fps: float, folder: Path) -> tuple[dict[str, Any], int]:
    """The complete FERAL config for this export, and the ``chunk_step`` it resolved.

    Everything is FERAL's own default (plus the preset) except what only the
    project knows: the video folder, the label file, the run name, where
    FERAL writes, and the chunk stride. The shifts are multiplied by the
    stride so overlapping chunks keep predicting the same frames — the
    preset's overlap ratio holds at every rate.
    """
    vf = config.video_features
    cfg = load_defaults(vf.preset)
    data = cfg["data"]
    step = resolve_chunk_step(vf.context_s, fps, int(data["chunk_length"]))
    data["chunk_step"] = step
    data["chunk_shift"] = int(data["chunk_shift"]) * step
    if data.get("eval_chunk_shift") is not None:
        data["eval_chunk_shift"] = int(data["eval_chunk_shift"]) * step
    data["prefix"] = str(prefix)
    data["label_json"] = str(folder / LABELS_JSON)
    cfg["run_name"] = config.train.run_name or "feral"
    cfg["output_dir"] = str(folder)
    return cfg, step


# ---------------------------------------------------------------------------
# Videos and labels
# ---------------------------------------------------------------------------


def _trial_extent(session: Session, trial: int | str) -> tuple[float, float]:
    """``(start, end)`` of the trial on its own clock — what the labels can cover."""
    ds = session.trial_dataset(trial)
    if ds is not None:
        coord = get_time_coord(next(iter(ds.data_vars.values())))
        if coord is None:
            raise ValueError(f"{session.id} trial {trial}: no time coord to place the video against")
        time = np.asarray(coord.values, dtype=float)
        return float(time[0]), float(time[-1])
    window = next(session.trial_windows([trial]))
    assert window.t0 is not None and window.t1 is not None
    return 0.0, float(window.t1 - window.t0)


def plan_videos(config: SegmentConfig, sessions: list[Session], classes: ClassTable) -> list[VideoRecord]:
    """One record per (trial, individual) of every session, labels on the video's frames.

    A session with several individuals is refused: FERAL labels the whole
    frame, so a group video needs one crop per animal first.
    """
    device_name = config.video_features.camera
    records: list[VideoRecord] = []
    for session in sessions:
        individuals = session.individuals(config)
        if len(individuals) != 1:
            raise ValueError(
                f"{session.spec.label}: {len(individuals)} individuals {individuals} — FERAL labels one class per "
                "frame of the whole video, so a group recording needs one cropped video per animal first. "
                "Set config.individual to export one of them."
            )
        individual = individuals[0]
        device = session.video_device(device_name)
        alignment = session.result.nwb_alignment
        trials = filter_trials(session, config.trials)
        missing = []
        for trial in trials:
            video = session.media_path(trial, "video", device)
            if video is None or not video.is_file():
                missing.append(trial)
                continue
            probe = probe_video(str(video))
            offset = float(alignment.stream_offset_for_trial(trial, "video", device))
            # The video's frame i sits at trial time offset + i / fps.
            time = offset + np.arange(probe.nframes, dtype=float) / probe.fps
            labels = session.curated_labels(trial)
            y, _ = dense_targets(labels, time, individual, classes)
            t0, t1 = _trial_extent(session, trial)
            half = 0.5 / probe.fps
            outside = int(np.count_nonzero((time < t0 - half) | (time > t1 + half)))
            if outside > MAX_OUTSIDE_FRACTION * probe.nframes:
                raise ValueError(
                    f"{session.spec.label} trial {trial}: {outside} of {probe.nframes} frames of {video.name} lie "
                    f"outside the trial ({t0:.2f}–{t1:.2f} s) — FERAL labels whole files, so a session-long "
                    "video needs cutting into one clip per trial first"
                )
            records.append(
                VideoRecord(
                    key=f"{session.id}_trial{trial}_{individual}",
                    session_id=session.id,
                    source=session.source,
                    trial=trial,
                    individual=individual,
                    video=video.resolve(),
                    fps=probe.fps,
                    n_frames=probe.nframes,
                    role="",
                    labels=y,
                    outside=outside,
                )
            )
        if missing:
            raise FileNotFoundError(
                f"{session.spec.label}: no video for trials {missing} — check the alignment's file names "
                "and the session's video_dir"
            )
    if not records:
        raise ValueError("No videos to export — check trials.where and the sessions' trials.")
    return records


def _with_roles(config: SegmentConfig, records: list[VideoRecord]) -> list[VideoRecord]:
    """Each record with its role from ``train.split`` — the same draw ``train()`` makes."""
    index = pd.DataFrame(
        {
            "key": [r.key for r in records],
            "source": [str(r.source) for r in records],
            "trial": [r.trial for r in records],
        }
    )
    roles = assign_roles(config, index)
    return [replace(r, role=roles[r.key]) for r in records]


def _compact_classes(records: list[VideoRecord], classes: ClassTable) -> tuple[dict[int, int], list[int], list[int]]:
    """Project class index → FERAL class, keeping only classes some train video holds.

    FERAL weights a class by its inverse train frequency, and a class with no
    train frames gets a clamped weight that swamps the loss; so those classes
    become background for FERAL and are listed as ``dropped``.
    """
    in_train: set[int] = set()
    for r in records:
        if r.role == "train":
            in_train.update(int(v) for v in np.unique(r.labels))
    used = sorted(in_train | {0})
    compact = {index: i for i, index in enumerate(used)}
    dropped = [classes.label_ids[i] for i in range(classes.n_classes) if i not in compact]
    for i in range(classes.n_classes):
        compact.setdefault(i, 0)
    return compact, [classes.label_ids[i] for i in used], dropped


def _common_prefix(videos: list[Path]) -> Path:
    try:
        return Path(os.path.commonpath([str(v) for v in videos]))
    except ValueError as exc:
        raise ValueError(
            "The sessions' videos share no common folder (different drives) — FERAL reads every video "
            "under one prefix, so keep them on one drive"
        ) from exc


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------


def export_feral(config: SegmentConfig, sessions: list[Session] | None = None) -> FeralExport:
    """Write FERAL's inputs for every session of *config* under ``config.feral_dir``.

    Opens the sessions without the session-open expansions (FERAL sees
    pixels, not features), rasterises each trial's curated labels on its
    video's frames, draws the roles ``train()`` would, and writes
    ``labels.json``, ``config.yaml``, ``videos.tsv`` and ``export.yaml``.
    Logs the commands to run in the FERAL environment.
    """
    if config.video_features.extractor != VARIABLE:
        raise ValueError(
            f"video_features.extractor is {config.video_features.extractor!r}; export_feral writes for extractor: feral"
        )
    folder = config.feral_dir
    with log_to_file(folder / "export.log"):
        return _export(config, folder, sessions)


def _export(config: SegmentConfig, folder: Path, sessions: list[Session] | None) -> FeralExport:
    if sessions is None:
        sessions = [open_session(spec, None, expand_changepoints=False) for spec in config.sessions]
    classes = class_table(config)
    records = _with_roles(config, plan_videos(config, sessions, classes))

    rates = sorted({r.fps for r in records})
    if len(rates) != 1:
        raise ValueError(
            f"The videos disagree on frame rate: {rates} — one FERAL config has one chunk stride, so every video "
            "must share a rate"
        )
    fps = rates[0]
    prefix = _common_prefix([r.video for r in records])
    compact, label_ids, dropped = _compact_classes(records, classes)
    names = {
        str(i): (BACKGROUND if lid == 0 else classes.names[classes.id_to_index[lid]]) for i, lid in enumerate(label_ids)
    }

    holdout = {str(p) for p in config.train.split.holdout_sessions}
    key_of = {r.key: r.video.relative_to(prefix).as_posix() for r in records}
    splits: dict[str, list[str]] = {role: [] for role in ROLES}
    for r in records:
        splits[r.role].append(key_of[r.key])
    # Every video is inferred: the embeddings are what the project trains on.
    splits["inference"] = [key_of[r.key] for r in records]
    labels_json = {
        "class_names": names,
        "is_multilabel": False,
        "labels": {key_of[r.key]: [compact[int(v)] for v in r.labels] for r in records},
        "splits": splits,
    }
    cfg, step = feral_config(config, prefix, fps, folder)

    folder.mkdir(parents=True, exist_ok=True)
    (folder / LABELS_JSON).write_text(json.dumps(labels_json), encoding="utf-8")
    preset = config.video_features.preset
    stride = "every frame" if step == 1 else f"{config.video_features.context_s} s at {fps:g} fps"
    header = (
        f"# Written by ethograph for FERAL {FERAL_VERSION}: its default_config.yaml"
        + (f" with preset {preset!r}" if preset else "")
        + ", plus data.prefix, data.label_json, run_name, output_dir and the chunk stride\n"
        + f"# (chunk_step {step} = {stride}; chunk_shift / eval_chunk_shift are multiplied by it).\n"
        + f"# Train with:  feral train-config {folder / CONFIG_YAML}\n"
    )
    (folder / CONFIG_YAML).write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "video": key_of[r.key],
                "key": r.key,
                "role": r.role,
                "session_id": r.session_id,
                "source": str(r.source),
                "trial": r.trial,
                "individual": r.individual,
                "n_frames": r.n_frames,
                "outside_trial_frames": r.outside,
            }
            for r in records
        ]
    ).to_csv(folder / VIDEOS_TSV, sep="\t", index=False)
    span_s = ((int(cfg["data"]["chunk_length"]) - 1) * step + 1) / fps
    export_info = {
        "feral_version": FERAL_VERSION,
        "prefix": str(prefix),
        "video_fps": fps,
        "chunk_step": step,
        "chunk_span_s": span_s,
        "embeddings_dir": str(folder / EMBEDDINGS_DIR),
        "class_label_ids": label_ids,  # FERAL class i is the project's label id class_label_ids[i]
        "dropped_label_ids": dropped,
        "holdout_sessions": sorted(holdout),
    }
    (folder / EXPORT_YAML).write_text(yaml.safe_dump(export_info, sort_keys=False), encoding="utf-8")

    result = FeralExport(
        folder=folder,
        prefix=prefix,
        fps=fps,
        chunk_step=step,
        n_videos=len(records),
        label_ids=label_ids,
        dropped=dropped,
        holdout_sessions=sorted(holdout),
    )
    _log_summary(config, result, records, classes, span_s)
    return result


def _log_summary(
    config: SegmentConfig, result: FeralExport, records: list[VideoRecord], classes: ClassTable, span_s: float
) -> None:
    by_role = {role: sum(r.role == role for r in records) for role in ROLES}
    outside = sum(r.outside for r in records)
    logger.info(
        "FERAL export: %d videos (%s), %d classes, %g fps → chunk_step %d (%.2f s per chunk) → %s",
        result.n_videos,
        ", ".join(f"{role} {n}" for role, n in by_role.items()),
        len(result.label_ids),
        result.fps,
        result.chunk_step,
        span_s,
        result.folder,
    )
    if result.dropped:
        names = [classes.names[classes.id_to_index[lid]] for lid in result.dropped]
        logger.warning("%d classes occur in no train video and are background for FERAL: %s", len(names), names)
    if outside:
        logger.warning(
            "%d frames lie outside their trial and are labelled background (videos.tsv: outside_trial_frames)",
            outside,
        )
    if result.holdout_sessions:
        logger.info(
            "Held out of FERAL's training (test + inference only): %s — their embeddings are clean",
            result.holdout_sessions,
        )
    else:
        logger.warning(
            "No train.split.holdout_sessions: FERAL trains on every session, so every downstream score over its "
            "embeddings is inflated. Name the sessions to hold out for a number you can report."
        )
    checkpoint = result.folder / "checkpoints" / f"{config.train.run_name or 'feral'}_best_checkpoint.pt"
    folders = sorted({r.video.parent for r in records})
    infer = "\n".join(
        f"  feral infer {checkpoint} {f} --output {predictions_file(result.folder, result.prefix, f)}" for f in folders
    )
    logger.info(
        "In the FERAL environment:\n  feral train-config %s\n%s\n"
        "then back here: project.import_feral_predictions() for FERAL's labels. For its embeddings as a video "
        "feature, add to every feral infer:  --save_embeddings %s  — then project.materialise()",
        result.folder / CONFIG_YAML,
        infer,
        result.folder / EMBEDDINGS_DIR,
    )


# ---------------------------------------------------------------------------
# FERAL's predictions, back as labels
# ---------------------------------------------------------------------------


def predictions_file(folder: Path, prefix: Path, video_folder: Path) -> Path:
    """Where ``feral infer --output`` writes for one video folder.

    ``feral infer`` keys its predictions by bare file name, so two sessions
    may both hold a ``trial1.mp4``; one JSON per video folder, named after
    the folder's path under the export's prefix, keeps them apart.
    """
    rel = video_folder.relative_to(prefix)
    name = "__".join(rel.parts) if rel.parts else prefix.name
    return folder / PREDICTIONS_DIR / f"{name}.json"


def import_feral_predictions(config: SegmentConfig, sessions: list[Session] | None = None) -> list[Path]:
    """FERAL's per-frame predictions → one prediction set per session; returns the written TSVs.

    Reads the JSON files ``feral infer --output`` wrote under
    ``{feral_dir}/predictions/`` (the export's log names the commands),
    places every video's frames on its trial's clock through the alignment,
    takes the most probable class per frame, and runs the intervals through
    ``infer.postprocess`` — the same steps a segmentation run's predictions
    take. Frames outside the trial are dropped. The sets are written beside
    each session's labels like every model's, ``labeling_method=automated``.
    """
    folder = config.feral_dir
    if not (folder / EXPORT_YAML).is_file():
        raise FileNotFoundError(f"No FERAL export in {folder} — run export_feral first")
    with log_to_file(folder / "import.log"):
        return _import_predictions(config, folder, sessions)


def _import_predictions(config: SegmentConfig, folder: Path, sessions: list[Session] | None) -> list[Path]:
    info = yaml.safe_load((folder / EXPORT_YAML).read_text(encoding="utf-8"))
    prefix, fps = Path(info["prefix"]), float(info["video_fps"])
    label_ids = np.asarray(info["class_label_ids"], dtype=int)
    videos = pd.read_csv(folder / VIDEOS_TSV, sep="\t", dtype={"trial": str})
    videos["json"] = [predictions_file(folder, prefix, (prefix / v).parent) for v in videos["video"]]
    found = {path: path.is_file() for path in videos["json"].unique()}
    if not any(found.values()):
        raise FileNotFoundError(
            f"None of FERAL's prediction files exist: {[str(p) for p in found]} — run the feral infer "
            f"commands in {folder / 'export.log'} first"
        )
    for path, exists in found.items():
        if not exists:
            logger.warning("No predictions at %s — its videos are skipped", path)

    if sessions is None:
        sessions = [open_session(spec, None, expand_changepoints=False) for spec in config.sessions]
    by_source = {str(session.source): session for session in sessions}
    name = f"{VARIABLE}_{config.train.run_name}" if config.train.run_name else VARIABLE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pcfg = config.infer.postprocess
    written = []
    for source, group in videos[videos["json"].map(found)].groupby("source", sort=False):
        if source not in by_source:
            raise ValueError(f"{VIDEOS_TSV} names session {source}, which this config no longer holds — export again")
        session = by_source[source]
        device = session.video_device(config.video_features.camera)
        alignment = session.result.nwb_alignment
        trial_of = {str(t): t for t in session.result.trial_ids}
        rows: list[dict] = []
        arrays: dict[str, np.ndarray] = {}
        for path, of_json in group.groupby("json", sort=False):
            preds = json.loads(Path(path).read_text(encoding="utf-8"))["preds"]
            for record in of_json.itertuples(index=False):
                video = Path(record.video).name
                if video not in preds:
                    raise KeyError(f"{path} holds no predictions for {video} — it was written for another folder")
                probs = np.asarray(preds[video], dtype=float)
                if probs.shape != (record.n_frames, len(label_ids)):
                    raise ValueError(
                        f"{video}: predictions of shape {probs.shape}, the export has {record.n_frames} frames and "
                        f"{len(label_ids)} classes — the checkpoint was trained on another export"
                    )
                trial = trial_of[record.trial]
                offset = float(alignment.stream_offset_for_trial(trial, "video", device))
                time = offset + np.arange(record.n_frames, dtype=float) / fps
                t0, t1 = _trial_extent(session, trial)
                inside = (time >= t0 - 0.5 / fps) & (time <= t1 + 0.5 / fps)
                time, probs = time[inside], probs[inside]
                key = sample_key(session.id, trial, record.individual)
                arrays[key] = probs.astype(np.float16)
                arrays[f"{key}_time"] = time
                raw = dense_to_intervals(label_ids[probs.argmax(axis=1)], [record.individual], time_coord=time)
                cp = changepoint_times(session, trial, {**pcfg.changepoints}) if pcfg.changepoint_correction else None
                intervals = postprocess_intervals(raw, pcfg, cp)
                curves = {int(lid): probs[:, i] for i, lid in enumerate(label_ids) if lid != 0}
                corrected = bool(pcfg.changepoint_correction and cp is not None and len(cp) > 0)
                rows.extend(label_rows(intervals, curves, time, trial, record.individual, name, corrected))
        tsv, _ = write_prediction_set(
            prediction_run_dir(session.source, name, timestamp),
            session.stem,
            rows,
            arrays,
            model_config=folder / CONFIG_YAML,
            inference_note={
                "model": VARIABLE,
                "run": name,
                "run_dir": str(folder),
                "session": str(session.source),
                "infer": config_to_dict(config)["infer"],
            },
        )
        logger.info("%s: %d labels from FERAL → %s", session.id, len(rows), tsv)
        written.append(tsv)
    return written
