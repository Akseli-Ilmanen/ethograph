"""Inference: a trained run over sessions → one prediction set per session.

Beside each session's own ``labels/`` folder (the same one label backups and
LightGBM onset-model runs use, see :mod:`ethograph.labels.onset_curves`), one
folder per call to :func:`infer`, named ``predictions_{run_name}_{timestamp}``
so a re-run never overwrites an earlier one::

    labels/
        predictions_{run_name}_{timestamp}/
            {stem}_predictions.tsv   # the GUI's native labels format, labeling_method=automated
            {stem}_probs.npz    # per sample: "{key}" → (T, C) float16, "{key}_time" → (T,)
            config.yaml         # the run's own config, as trained
            inference.yaml      # the run's name and the infer: settings applied here

The TSV is what the GUI loads and compares; the ``.npz`` exists only for the
confidence overlay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml

from ethograph.labels.ml import dense_to_intervals
from ethograph.labels.onset_curves import write_provenance
from ethograph.labels.tsv_store import load_labels_tsv, save_labels_tsv
from ethograph.segment.config import SegmentConfig, config_to_dict, load_config
from ethograph.segment.materialise import COLUMNS_FILE, read_target_table
from ethograph.segment.models import as_output, build_model
from ethograph.segment.postprocess import postprocess_intervals
from ethograph.segment.prediction_sets import (
    label_rows,
    prediction_run_dir,
    write_prediction_set,
)
from ethograph.segment.preprocess import NormStats
from ethograph.segment.samples import (
    SELF_TOKEN,
    ChannelTable,
    ColumnLayout,
    TargetTable,
    build_sample_features,
    channels_to_track,
    is_multilabel,
    sample_key,
)
from ethograph.segment.sessions import Session, changepoint_times, filter_trials, open_session
from ethograph.segment.train import BEST_FILE, STATS_FILE
from ethograph.utils.device import resolve_device
from ethograph.utils.logging import log_to_file

logger = logging.getLogger(__name__)


@dataclass
class Run:
    run_dir: Path
    config: SegmentConfig
    layout: ColumnLayout
    classes: TargetTable
    stats: NormStats
    model: torch.nn.Module
    device: torch.device
    #: The ablation this run was trained with (``None`` = every column).
    keep: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self.run_dir.name


def resolve_run_dir(config: SegmentConfig, run: str | Path | None) -> Path:
    """Resolve *run* (a path, an exact run dir name, or a base name) to a trained run directory.

    ``train()`` names each run ``{base_name}_{timestamp}`` so it never
    overwrites another run. A *run* (or, when ``None``, ``config.infer.run``
    / the config's own base name) that does not match a directory exactly is
    tried as that base name, resolving to the most recently trained run for
    it.
    """
    from ethograph.segment.train import run_name_for

    run = run or config.infer.run or run_name_for(config)
    candidate = Path(run)
    if candidate.is_dir() and (candidate / BEST_FILE).is_file():
        return candidate.resolve()
    run_dir = config.run_dir(str(run))
    if (run_dir / BEST_FILE).is_file():
        return run_dir
    return _latest_run_dir(config, str(run))


def _latest_run_dir(config: SegmentConfig, base_name: str) -> Path:
    """The most recently trained ``{base_name}_{timestamp}`` run directory."""
    candidates = sorted(p.name for p in config.runs_dir.glob(f"{base_name}_*") if (p / BEST_FILE).is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No trained run named {base_name!r} (or {base_name}_<timestamp>) under {config.runs_dir}"
        )
    return config.run_dir(candidates[-1])


def load_run(run_dir: Path, device: str | None = None) -> Run:
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    layout = ColumnLayout.from_dict(yaml.safe_load((run_dir / COLUMNS_FILE).read_text(encoding="utf-8")))
    classes = read_target_table(run_dir / "classes.yaml")
    stats = NormStats.load(run_dir / STATS_FILE)
    dev = torch.device(resolve_device(device or config.train.device))
    # columns.yaml is the *full* layout, so a freshly built sample can be
    # checked against it; the run's own drop_kinds then re-derives the
    # ablation it was trained with.
    keep = layout.keep_mask(config.train.drop_kinds)
    n_features = int(keep.sum())
    model = build_model(config.model.architecture, config.model.params, n_features, classes.n_outputs)
    model.load_state_dict(torch.load(run_dir / BEST_FILE, map_location=dev, weights_only=True))
    model.to(dev).eval()
    return Run(run_dir, config, layout, classes, stats, model, dev, keep=None if keep.all() else keep)


def predict_probabilities(run: Run, x: np.ndarray) -> np.ndarray:
    """``(F, T)`` preprocessed features → ``(T, C)`` probabilities.

    A softmax over the classes of an exclusive run; a sigmoid per channel of
    a multi-label one (the columns then do not sum to 1).
    """
    if run.keep is not None:
        x = x[run.keep]
    xn = torch.from_numpy(np.ascontiguousarray(run.stats.apply(x))).unsqueeze(0).to(run.device)
    mask = torch.ones(1, 1, xn.shape[-1], device=run.device)
    with torch.no_grad():
        output = as_output(run.model(xn, mask))
        logits = output.logits[-1]
        probs = torch.sigmoid(logits) if is_multilabel(run.classes) else torch.softmax(logits, dim=1)
        return probs[0].T.cpu().numpy()


def decode_sample(
    run: Run, probs: np.ndarray, time: np.ndarray, individual: str, threshold: float
) -> list[tuple[pd.DataFrame, dict[int, np.ndarray]]]:
    """``(T, C)`` probabilities → the sample's raw interval sets, one per exclusive track.

    An exclusive run is one track: argmax over the classes. A multi-label
    run decodes each of the sample's own (``self``) tracks on its own —
    channels above *threshold*, the most probable winning a frame where
    several are on (:func:`~ethograph.segment.samples.channels_to_track`) —
    so labels of different branches overlap and labels of one branch never
    do, exactly the GUI's rule. The other individuals' channels (``subjects:
    all``) are training targets only: each animal's labels come from its own
    sample. Beside each interval set comes the probability curve per label
    id, for the segment confidence.
    """
    classes = run.classes
    if not isinstance(classes, ChannelTable):
        indices = probs.argmax(axis=1)
        ids = classes.ids(indices)
        curves = {int(lid): probs[:, i] for i, lid in enumerate(classes.label_ids) if lid != 0}
        return [(dense_to_intervals(ids, [individual], time_coord=time), curves)]
    on = probs.T > threshold  # (C, T)
    out = []
    for track in classes.tracks(subject=SELF_TOKEN):
        idx = channels_to_track(on, probs.T, track)
        ids = track.classes.ids(idx)
        curves = {classes.channels[c].label_id: probs[:, c] for c in track.channels}
        out.append((dense_to_intervals(ids, [individual], time_coord=time), curves))
    return out


def infer_session(
    config: SegmentConfig,
    run: Run,
    session: Session,
    out_dir: Path | None = None,
    trials: Iterable[int | str] | None = None,
) -> tuple[Path, Path]:
    """Predict every (trial, individual) of *session*; returns the two written paths.

    *trials* narrows the trials passing the config's filter to the ids it
    names — how a trial-level cross-validation fold predicts only what it
    held out.
    """
    chosen = filter_trials(session, config.trials)
    if trials is not None:
        wanted = {str(t) for t in trials}
        chosen = [t for t in chosen if str(t) in wanted]
    return _infer_trials(config, run, session, chosen, out_dir)


def _infer_trials(
    config: SegmentConfig, run: Run, session: Session, trials: list[int | str], out_dir: Path | None
) -> tuple[Path, Path]:
    individuals = session.individuals(config)
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    pcfg = config.infer.postprocess
    step = int(run.config.train.subsample)
    trials_without_cp: list[int | str] = []
    for window in session.trial_windows(trials):
        for individual in individuals:
            time, x, layout = build_sample_features(config, session, window, individual, individuals)
            run.layout.check(layout, f"{session.id} trial {window.trial} individual {individual}")
            if step > 1:  # the rate this run was trained at, so its receptive field means the same thing
                time, x = time[::step], x[:, ::step]
            probs = predict_probabilities(run, x)
            cp = (
                changepoint_times(session, window.trial, {**pcfg.changepoints}) if pcfg.changepoint_correction else None
            )
            if cp is not None and len(cp) == 0:
                trials_without_cp.append(window.trial)
            key = sample_key(session.id, window.trial, individual)
            arrays[key] = probs.astype(np.float16)
            arrays[f"{key}_time"] = time.astype(np.float64)
            for raw, curves in decode_sample(run, probs, time, individual, float(config.infer.threshold)):
                intervals = postprocess_intervals(raw, pcfg, cp)
                corrected = bool(pcfg.changepoint_correction and cp is not None and len(cp) > 0)
                rows.extend(label_rows(intervals, curves, time, window.trial, individual, run.name, corrected))
    if trials_without_cp:
        logger.warning(
            "%s: changepoint_correction is on but %d/%d trials have no changepoints — those boundaries "
            "were not snapped. Check that the session's changepoint variables are described "
            "(ethograph.io.schema.changepoint_attrs) and that infer.postprocess.changepoints selects them.",
            session.id,
            len(trials_without_cp),
            len(trials) * len(individuals),
        )
    out_dir = (
        out_dir
        if out_dir is not None
        else prediction_run_dir(session.source, run.name, datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    tsv_path, npz_path = write_prediction_set(
        out_dir,
        session.stem,
        rows,
        arrays,
        model_config=run.run_dir / "config.yaml",
        inference_note={
            "model": "segment",
            "run": run.name,
            "run_dir": str(run.run_dir),
            "session": str(session.source),
            "infer": config_to_dict(config)["infer"],
        },
    )
    logger.info("%s: %d predicted labels → %s", session.id, len(rows), tsv_path)
    return tsv_path, npz_path


def check_neural_units(session: Session, run_config: SegmentConfig) -> None:
    """Refuse a session whose units are not the ones the run's neural feature was trained on.

    A neural decoder only runs on the recording it was trained on; the
    same file under another alignment (a sleep epoch) is fine, another
    recording's units are not, and the generic "did not pin down" error a
    missing column would otherwise raise says nothing about why.
    """
    from ethograph.segment.sessions import neural_columns

    cfg = run_config.features.neural
    if cfg is None:
        return
    trained = run_config.features.columns.get(cfg.name, {})
    present = neural_columns(session, cfg)
    missing = {dim: sorted(set(values) - set(present.get(dim, []))) for dim, values in trained.items()}
    missing = {dim: values for dim, values in missing.items() if values}
    if missing:
        raise ValueError(
            f"{session.spec.label}: the run was trained on units this session does not have — missing {missing}; "
            f"the session's are {present}. A neural decoder only runs on the recording it was trained on."
        )


def inherit_neural_columns(config: SegmentConfig, run_config: SegmentConfig) -> SegmentConfig:
    """The project config with its unit columns taken from the run that recorded them.

    ``features.neural`` leaves the unit columns to be read off the session
    at materialise; the run's saved config carries the result, so inference
    needs neither the materialised dataset nor a second reading of the
    spikes. A project that spells the entry itself is left alone — and held
    to the run's layout like any other column, by the layout check.
    """
    cfg = config.features.neural
    if cfg is None or cfg.name in config.features.columns:
        return config
    recorded = run_config.features.columns.get(cfg.name)
    if recorded is None:
        raise ValueError(
            f"features.neural names {cfg.name!r}, but the run was trained without it (its columns: "
            f"{sorted(run_config.features.columns)}) — pick a run trained on this config, or spell "
            f"features.columns.{cfg.name} to say which units to feed it."
        )
    columns = {**config.features.columns, cfg.name: dict(recorded)}
    return replace(config, features=replace(config.features, columns=columns))


def inference(
    config: SegmentConfig,
    run: str | Path | None = None,
    sessions: Iterable[str | Path] | None = None,
    trials: Iterable[int | str] | None = None,
) -> list[Path]:
    """Predict every session of the config; *sessions* narrows that to a few.

    A session is named by its full ``source`` path or just the file's stem,
    which is how a cross-validation fold asks for the one session it held
    out; *trials* narrows every session to the trial ids it names, which is
    how a trial-level fold asks for the trials it held out.
    """
    loaded = load_run(resolve_run_dir(config, run))
    config = inherit_neural_columns(config, loaded.config)
    specs = config.select_sessions(sessions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with log_to_file(loaded.run_dir / "infer.log"):
        written = []
        for spec in specs:
            # The run's config carries the scales it was trained at; the
            # project's may still be waiting to derive them.
            session = open_session(spec, loaded.config)
            check_neural_units(session, loaded.config)
            out_dir = prediction_run_dir(session.source, loaded.name, timestamp)
            tsv, _ = infer_session(config, loaded, session, out_dir=out_dir, trials=trials)
            written.append(tsv)
        return written


def merge_prediction_sets(
    paths: list[Path], out_dir: Path, stem: str, *, model_config: Path, inference_note: dict
) -> Path:
    """Concatenate prediction sets written for *disjoint* trials of one session into one.

    What a trial-level cross-validation ends with: each fold predicted the
    trials it held out, and this stitches them into the one set you open in
    the GUI against the curated labels — every trial predicted exactly once,
    by a model that never saw it. Rows keep the ``prediction_source`` of the
    fold that wrote them, and the ``_probs.npz`` arrays are merged by sample
    key. Returns the merged TSV's path.
    """
    if not paths:
        raise ValueError("Nothing to merge — no prediction sets were written.")
    frames = [load_labels_tsv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if not df.empty:
        df = df.sort_values(["trial", "onset_s"], kind="stable").reset_index(drop=True)
    arrays: dict[str, np.ndarray] = {}
    for p in paths:
        with np.load(p.with_name(f"{stem}_probs.npz")) as npz:
            arrays.update({k: npz[k] for k in npz.files})
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / f"{stem}_predictions.tsv"
    save_labels_tsv(tsv_path, df)
    np.savez_compressed(out_dir / f"{stem}_probs.npz", **arrays)
    write_provenance(out_dir, model_config=model_config, inference=inference_note)
    return tsv_path
