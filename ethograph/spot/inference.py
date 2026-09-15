"""Stage 3: a run's predictions written into the GUI's labels, per session.

The same shape as :func:`ethograph.segment.inference.inference`: pick a run,
pick the sessions, and each session gets one folder beside it under
``labels/`` — ``predictions_spot_{run}_{timestamp}/`` — holding

* ``{stem}_predictions.tsv`` — one automated point event per class per trial,
  with the shape-based ``confidence`` (:mod:`ethograph.spot.confidence`);
* ``onset_curves.npz`` — the per-frame curves those events were read off,
  in the format frame-by-frame review already draws.

Inference covers **every trial with video**, labelled or not — that is what
predicting into a session means. Trials whose frames were never decoded are
decoded now, into the same ``frames/`` the training used.

The epoch is chosen the way the ladder was read: by the tallest-peak sweep on
the run's own validation predictions (fewest misses, then most within 20 ms)
— never by ``val_mAP``, which is unfit for it. Upstream's ``test_e2e.py`` is
driven as a subprocess and always loads the run's *last* checkpoint, so the
chosen one is staged alone in a temporary model directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from ethograph.labels import onset_curves
from ethograph.labels.tsv_store import save_labels_tsv
from ethograph.segment.sessions import Session, open_session
from ethograph.spot import dataset as dataset_stage
from ethograph.spot.config import ResolvedClip, SpotConfig, config_to_dict
from ethograph.spot.predict import SpottedEvent, flagged, read_predictions, spot_entry, to_labels_frame
from ethograph.spot.vendored import run_logged, script_command

logger = logging.getLogger(__name__)

#: The split name upstream's CLI accepts that no training stage uses.
INFER_SPLIT = "challenge"

#: Tolerance the epoch choice counts a hit within, in the video's frames at
#: the run's full rate — the ladder's ``<=4`` column (20 ms at 200 fps).
EPOCH_HIT_FRAMES = 4

MODEL_NAME = "spot"


def run_label(run_dir: Path) -> str:
    """How a run is named in what it writes: ``ctx2s_res10ms`` or ``ctx2s_res10ms_distil_ab12cd34``.

    A distilled student lives in ``runs/{baseline}_distil_{fingerprint}/stage3/``;
    its label is the run's, not the stage folder's.
    """
    return run_dir.parent.name if run_dir.name in ("stage2", "stage3") else run_dir.name


def trained_runs(config: SpotConfig) -> list[Path]:
    """Every run under ``runs/`` with upstream's ``config.json``: label-only runs and distilled ``stage3`` students."""
    if not config.runs_dir.is_dir():
        return []
    found = [p for p in config.runs_dir.glob("*") if (p / "config.json").is_file()]
    found += [p for p in config.runs_dir.glob("*/stage3") if (p / "config.json").is_file()]
    return sorted(found, key=lambda p: p.stat().st_mtime)


def teacher_runs(config: SpotConfig) -> list[Path]:
    """Every pose teacher under ``teacher/``, oldest first — scored by ``evaluate()`` like any run.

    A folder left behind by an architecture this package no longer builds is
    not a teacher: it is skipped rather than handed to the pixel model's
    ``test_e2e.py``, which reads keys its ``config.json`` never carried.
    """
    from ethograph.spot.teacher import is_teacher_run

    if not config.teacher_dir.is_dir():
        return []
    found = [p for p in config.teacher_dir.glob("*") if is_teacher_run(p)]
    return sorted(found, key=lambda p: p.stat().st_mtime)


def resolve_run_dir(config: SpotConfig, run: str | Path | None) -> Path:
    """A run by name under ``runs/``, by path, or the newest trained one (student or baseline) when *run* is None."""
    if run is None:
        candidates = trained_runs(config)  # never a teacher: inference means pixels
        if not candidates:
            raise FileNotFoundError(f"No trained run under {config.runs_dir} — run project.train() first")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    path = Path(run)
    if (path / "config.json").is_file():
        return path.resolve()
    named = config.run_dir(str(run))
    if (named / "config.json").is_file():
        return named
    teacher = config.teacher_dir / str(run)
    if (teacher / "config.json").is_file():  # a teacher by name, for evaluate()
        return teacher
    if (named / "stage3" / "config.json").is_file():  # a distilled student, named by its run folder
        return named / "stage3"
    raise FileNotFoundError(f"No run {run!r}: neither {path} nor {named} (or its stage3/) holds a config.json")


def run_config_file(run_dir: Path) -> Path:
    """The config a run was trained from, to copy beside its predictions.

    Our ``config.yaml`` — a distilled student's sits one level up, beside its
    ``stage2``/``stage3`` — else upstream's ``config.json``, which every run
    has, for one trained before this project wrote configs.
    """
    for candidate in (run_dir / "config.yaml", run_dir.parent / "config.yaml"):
        if candidate.is_file():
            return candidate
    return run_dir / "config.json"


def run_reads_features(run_dir: Path) -> bool:
    """Whether the run reads the feature block beside the frames (trained with ``features:`` as input)."""
    stored = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return bool(stored.get("fuse_dim"))


def run_clip(run_dir: Path, fps: float) -> ResolvedClip:
    """The clip a run was trained with, on a video at *fps*.

    Read from the run's own ``config.json`` — the frame counts upstream
    stored — so inference cannot disagree with training about the stride.
    """
    stored = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return ResolvedClip(
        fps=float(fps),
        stride=int(stored.get("stride", 1)),
        clip_len=int(stored["clip_len"]),
        dilate_len=int(stored.get("dilate_len", 0)),
    )


def _sweep(entries: list[dict], truth: dict[str, dict], config: SpotConfig, clip: ResolvedClip) -> tuple[int, int]:
    """``(misses, hits within EPOCH_HIT_FRAMES)`` of one prediction file against *truth*."""
    misses = hits = 0
    for entry in entries:
        gt = truth.get(entry["video"])
        if gt is None:
            continue
        events, _ = spot_entry(entry, config, clip)
        by_label = {e.label: e for e in events}
        for gt_event in gt["events"]:
            label = config.class_label(gt_event["label"])
            if label not in by_label:
                misses += 1
            elif abs(by_label[label].frame - int(gt_event["frame"])) <= EPOCH_HIT_FRAMES:
                hits += 1
    return misses, hits


def val_truth_path(run_dir: Path, config: SpotConfig) -> Path | None:
    """The ``val.json`` a run's validation predictions are scored against.

    The project's own first; else the dataset directory the run's
    ``config.json`` names, so a run trained before this project existed still
    picks its epoch by the sweep rather than by its last checkpoint.
    """
    candidates = [config.dataset_dir / "val.json"]
    stored = json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("dataset")
    if stored:
        candidates.append(Path(stored) / "val.json")
    return next((c for c in candidates if c.is_file()), None)


def best_epoch(run_dir: Path, config: SpotConfig) -> int:
    """The epoch the sweep ranks first on the run's validation predictions.

    Falls back to the last checkpoint when the run wrote no validation
    predictions at all (a run started with ``start_val_epoch`` past its end),
    or when no ``val.json`` can be found to score them against.
    """
    val_path = val_truth_path(run_dir, config)
    files = sorted(run_dir.glob("pred-val.*.recall.json.gz"))
    if files and val_path is not None:
        truth_list = json.loads(val_path.read_text(encoding="utf-8"))
        truth = {v["video"]: v for v in truth_list}
        fps = float(truth_list[0]["fps"]) if truth_list else 0.0
        clip = run_clip(run_dir, fps)
        scored = []
        for path in files:
            epoch = int(path.name.split(".")[1])
            misses, hits = _sweep(read_predictions(path), truth, config, clip)
            scored.append((misses, -hits, epoch))
        misses, neg_hits, epoch = min(scored)
        logger.info(
            "Epoch %d chosen on val: %d misses, %d within %d frames", epoch, misses, -neg_hits, EPOCH_HIT_FRAMES
        )
        return epoch
    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"{run_dir} holds no checkpoint")
    epoch = int(checkpoints[-1].stem.split("_")[1])
    logger.warning(
        "%s: no validation predictions%s — using its last checkpoint (epoch %d), which the ladder found is "
        "rarely the best one",
        run_dir.name,
        " to score" if (files and val_path is None) else "",
        epoch,
    )
    return epoch


def stage_checkpoint(run_dir: Path, epoch: int, staging: Path) -> Path:
    """A model directory holding only *epoch*, for a CLI that loads the last one.

    ``test_e2e.py`` finds "the last epoch" by the ``optim_*.pt`` names present
    and loads the matching ``checkpoint_*.pt``; it never reads the optimiser
    state, so an empty ``optim_000.pt`` beside the one checkpoint is enough.
    """
    source = run_dir / f"checkpoint_{epoch:03d}.pt"
    if not source.is_file():
        raise FileNotFoundError(f"{run_dir} has no checkpoint for epoch {epoch}")
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "config.json", staging / "config.json")
    shutil.copy2(source, staging / "checkpoint_000.pt")
    (staging / "optim_000.pt").touch()
    return staging


def _write_infer_split(config: SpotConfig, records: list[dataset_stage.TrialRecord]) -> Path:
    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    class_file = config.dataset_dir / dataset_stage.CLASS_FILE
    if not class_file.is_file():
        names = [config.class_name(label) for label in config.labels.classes]
        class_file.write_text("\n".join(names) + "\n", encoding="utf-8")
    path = config.dataset_dir / f"{INFER_SPLIT}.json"
    path.write_text(json.dumps([r.to_json() for r in records], indent=2), encoding="utf-8")
    return path


def predict_split(
    config: SpotConfig, model_dir: Path, split: str, out_prefix: Path, log_path: Path, zero_features: bool = False
) -> list[dict]:
    """Run ``test_e2e.py`` on *split* of the project's dataset and read the recall file it writes.

    *zero_features* hands a model that reads the feature block zeros for it
    — the ablation behind ``evaluate(zero_features=True)``.
    """
    command = script_command("test_e2e") + [
        str(model_dir.resolve()),
        str(config.frames_dir.resolve()),
        "-s",
        split,
        "--save_as",
        str(out_prefix.resolve()),
        "-d",
        str(config.dataset_dir.resolve()),
    ] + (["--zero_fuse"] if zero_features else [])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    code = run_logged(command, log_path, cwd=log_path.parent)
    if code != 0:
        raise RuntimeError(f"test_e2e.py exited with {code}; see {log_path}")
    recall = Path(f"{out_prefix}.recall.json.gz")
    if not recall.is_file():
        raise FileNotFoundError(f"test_e2e.py finished but wrote no {recall}")
    return read_predictions(recall)


def flag_out_of_order(events: list[SpottedEvent], config: SpotConfig) -> list[SpottedEvent]:
    """Events of a trial that break the class order get confidence 0 — flagged, kept.

    ``labels.classes`` is read as the order the events happen in (first
    contact before last contact). A structural rule fires with near-perfect
    precision and costs nothing; repairing the order instead would silently
    file the most suspicious trial as clean.
    """
    order = {label: i for i, label in enumerate(config.labels.classes)}
    by_video: dict[str, list[SpottedEvent]] = {}
    for e in events:
        by_video.setdefault(e.video_id, []).append(e)
    out: list[SpottedEvent] = []
    for video_id, found in by_video.items():
        ranked = sorted(found, key=lambda e: order[e.label])
        frames = [e.frame for e in ranked]
        if frames != sorted(frames):
            logger.warning(
                "%s: events out of class order (%s) — flagged", video_id, [(e.label, e.frame) for e in ranked]
            )
            found = [replace(e, stats=replace(e.stats, peak=0.0, focus=0.0, ratio=0.0, found=False)) for e in found]
        out.extend(found)
    return out


def _curve_time(curve: np.ndarray, clip: ResolvedClip, offset: float) -> np.ndarray:
    """Trial-relative seconds of each strided prediction bin (bin centres)."""
    bins = np.arange(len(curve), dtype=np.float64)
    return clip.to_frame(bins) / clip.fps + offset


def infer_session(
    config: SpotConfig,
    run_dir: Path,
    epoch: int,
    session: Session,
    out_dir: Path,
    workers: int | None = None,
    loaded: tuple[object, dict] | None = None,
) -> Path:
    """Predict one session with one epoch of one run; returns the labels TSV.

    The video is decoded straight into the model (:mod:`ethograph.spot.stream`);
    no frame folder is written or read — that is training's. *loaded* is the
    run's model already in memory (:func:`~ethograph.spot.stream.load_run_model`),
    passed by :func:`inference` so one checkpoint read serves every session.
    """
    records = dataset_stage.plan_session(session, config, require_events=False)
    if not records:
        raise ValueError(
            f"{session.spec.label}: no trial has a {config.labels.camera or 'default'} video to predict on"
        )
    rates = {round(r.fps, 6) for r in records}
    if len(rates) != 1:
        raise ValueError(f"{session.spec.label}: videos at several rates {sorted(rates)}; one clip cannot fit them all")
    clip = run_clip(run_dir, records[0].fps)
    blocks: dict[str, np.ndarray] = {}
    if run_reads_features(run_dir):
        # A model that reads the features needs every predicted trial's
        # block, on the training split's scale.
        from ethograph.spot.features import export_block_for_inference, export_features

        export_features(config, [session], records)
        # The block on the run's own stride: the features' rate is the pose's,
        # and the run recorded its stride on the video's clock.
        block_dir = export_block_for_inference(config, [r.video_id for r in records], clip=clip)
        for r in records:
            with np.load(block_dir / f"{r.video_id}.npz") as npz:
                blocks[r.video_id] = np.asarray(npz["features"], dtype=np.float32)

    alignment = session.result.nwb_alignment
    camera = session.video_device(config.labels.camera)
    trials = {
        r.video_id: (r.trial, float(alignment.stream_offset_for_trial(r.trial, "video", device=camera)))
        for r in records
    }
    from ethograph.spot.stream import predict_records

    class_names = [config.class_name(label) for label in config.labels.classes]
    entries, lengths = predict_records(
        run_dir,
        epoch,
        records,
        class_names,
        blocks=blocks,
        jpeg_roundtrip=config.infer.jpeg_roundtrip,
        workers=workers,
        loaded=loaded,
    )

    events = []
    per_trial: dict[object, onset_curves.TrialCurves] = {}
    for entry in entries:
        found, curves = spot_entry(entry, config, clip, num_frames=lengths.get(str(entry["video"])))
        events.extend(found)
        trial, offset = trials[str(entry["video"])]
        if curves:
            first = next(iter(curves.values()))
            per_trial[trial] = (_curve_time(first, clip, offset), curves)
    if config.infer.flag_out_of_order:
        events = flag_out_of_order(events, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = to_labels_frame(
        events, trials, source=f"{MODEL_NAME}:{run_label(run_dir)}@{epoch}", individual=config.individual
    )
    tsv_path = out_dir / f"{session.stem}_predictions.tsv"
    save_labels_tsv(tsv_path, df)
    onset_curves.write_curves(out_dir / onset_curves.CURVES_FILE, per_trial)
    onset_curves.write_provenance(
        out_dir,
        model_config=run_config_file(run_dir),
        inference={
            "model": MODEL_NAME,
            "run": run_label(run_dir),
            "run_dir": str(run_dir),
            "epoch": int(epoch),
            "checkpoint": f"checkpoint_{epoch:03d}.pt",
            "session": str(session.source),
            "trials": len(records),
            "infer": config_to_dict(config)["infer"],
        },
    )
    low = flagged(events, config)
    logger.info(
        "%s: %d predicted events over %d trials, %d flagged below %.2f -> %s",
        session.spec.label,
        len(events),
        len(records),
        len(low),
        config.infer.flag_confidence_below,
        tsv_path,
    )
    return tsv_path


def inference(
    config: SpotConfig,
    run: str | Path | None = None,
    sessions: Iterable[str | Path] | None = None,
    workers: int | None = None,
) -> list[Path]:
    """Predict every session of the config with *run*; *sessions* narrows it.

    A session is named by its ``name``, its full ``source`` path or the
    file's stem — how a cross-validation fold asks for the one it held out.
    """
    from ethograph.spot.stream import load_run_model
    from ethograph.utils.device import resolve_device

    run_dir = resolve_run_dir(config, run)
    epoch = best_epoch(run_dir, config)
    loaded = load_run_model(run_dir, epoch, len(config.labels.classes), resolve_device())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    written: list[Path] = []
    for spec in config.select_sessions(sessions):
        session = open_session(spec)
        out_dir = onset_curves.run_dir(session.source, timestamp, model=f"{MODEL_NAME}_{run_label(run_dir)}")
        written.append(infer_session(config, run_dir, epoch, session, out_dir, workers=workers, loaded=loaded))
    return written
