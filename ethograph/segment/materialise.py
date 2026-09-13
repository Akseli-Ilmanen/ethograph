"""Feature engineering: write the materialised dataset.

Layout (the action-segmentation literature's, so third-party models plug in
directly), under ``{root}/data/{features.name}/``::

    features / {key}.npy  # (F, T) float32, session-level preprocessed
    groundTruth / {key}.txt  # one class name per frame
    groundTruth / {key}.npy  # multi-label targets instead: (C, T) uint8, one row per channel
    mapping.txt  # "{index} {name}" — contiguous, 0 = background (channel names for multi-label)
    index.tsv  # key, session_id, source, trial, individual, n_frames, fs, n_labelled
    columns.yaml  # the column layout (names, normalise flags, vector groups)
    classes.yaml  # class index ↔ label id, or the multi-label channels (``target: multilabel``)

``key`` is ``{session_id}_trial{trial}_{individual}``. Role assignment and
normalisation statistics belong to a run, not to the dataset.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ethograph.io import schema
from ethograph.labels.intervals import states_only
from ethograph.segment.config import SegmentConfig
from ethograph.segment.samples import (
    ChannelTable,
    ColumnLayout,
    TargetTable,
    build_sample_features,
    dense_channel_targets,
    dense_targets,
    is_multilabel,
    others_of,
    sample_key,
    subject_tokens,
    target_label_ids,
    target_table,
    target_table_from_dict,
)
from ethograph.segment.sessions import (
    Session,
    expand_changepoint_features,
    filter_trials,
    neural_columns,
    open_session,
)
from ethograph.utils.logging import log_to_file
from ethograph.utils.xr_utils import get_time_coord

logger = logging.getLogger(__name__)

INDEX_FILE = "index.tsv"
COLUMNS_FILE = "columns.yaml"
CLASSES_FILE = "classes.yaml"
MAPPING_FILE = "mapping.txt"


def materialise(config: SegmentConfig, sessions: list[Session] | None = None) -> Path:
    """Write the materialised dataset for every session in the config."""
    data_dir = config.data_dir
    with log_to_file(data_dir / "materialise.log"):
        return _materialise_run(config, data_dir, sessions)


def _materialise_run(config: SegmentConfig, data_dir: Path, sessions: list[Session] | None) -> Path:
    for sub in ("features", "groundTruth"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    layout: ColumnLayout | None = None
    if sessions is None:
        sessions = [open_session(s, config, expand_changepoints=False) for s in config.sessions]
    if not sessions:
        raise ValueError("No sessions to materialise — config.sessions is empty.")
    classes = target_table(config, len(sessions[0].individuals(config)))
    config = derive_changepoint_scales(config, sessions)
    config = resolve_neural_columns(config, sessions)
    for session in sessions:
        expand_changepoint_features(session, config)
    for session in sessions:
        trials = filter_trials(session, config.trials)
        individuals = session.individuals(config)
        logger.info("%s: %d trials × %d individuals", session.id, len(trials), len(individuals))
        if not session.declares_schema():
            logger.warning(
                "%s declares no `kind` on any variable, so train.drop_kinds has nothing to drop and "
                "normalise=0 columns will be z-scored. Describe the variables when you build the session "
                "(ethograph.io.schema.describe), or write %s for a backend without attrs.",
                session.id,
                schema.sidecar_path(session.source),
            )
        for window in session.trial_windows(trials):
            labels = session.curated_labels(window.trial)
            for individual in individuals:
                key = sample_key(session.id, window.trial, individual)
                time, x, sample_layout = build_sample_features(config, session, window, individual, individuals)
                if layout is None:
                    layout = sample_layout
                else:
                    layout.check(sample_layout, f"{session.id} trial {window.trial} individual {individual}")
                    _check_fs(layout.fs, sample_layout.fs, key)
                if isinstance(classes, ChannelTable):
                    subjects = subject_tokens(individual, others_of(individual, individuals))
                    y, n_labelled = dense_channel_targets(labels, time, subjects, classes)
                else:
                    y, n_labelled = dense_targets(labels, time, individual, classes)
                np.save(data_dir / "features" / f"{key}.npy", x)
                _write_ground_truth(data_dir / "groundTruth", key, y, classes)
                rows.append(
                    {
                        "key": key,
                        "session_id": session.id,
                        "source": str(session.source),
                        "trial": window.trial,
                        "individual": individual,
                        "n_frames": int(x.shape[1]),
                        "fs": float(sample_layout.fs),
                        "n_labelled": int(n_labelled),
                    }
                )
    if layout is None or not rows:
        raise ValueError("No samples were materialised — check trials.where and the sessions' trials.")

    cpf = config.features.changepoint_features
    if cpf is not None:
        layout.changepoint_features = cpf.scales()
    neural = config.features.neural
    if neural is not None:
        layout.neural_columns = {neural.name: dict(config.features.columns[neural.name])}
    pd.DataFrame(rows).to_csv(data_dir / INDEX_FILE, sep="\t", index=False)
    (data_dir / COLUMNS_FILE).write_text(yaml.safe_dump(layout.to_dict(), sort_keys=False), encoding="utf-8")
    (data_dir / CLASSES_FILE).write_text(yaml.safe_dump(classes.to_dict(), sort_keys=False), encoding="utf-8")
    (data_dir / MAPPING_FILE).write_text(
        "".join(f"{i} {name}\n" for i, name in enumerate(classes.names)), encoding="utf-8"
    )
    logger.info("Materialised %d samples × %d columns → %s", len(rows), layout.n_features, data_dir)
    return data_dir


def derive_changepoint_scales(config: SegmentConfig, sessions: list[Session]) -> SegmentConfig:
    """The config with ``features.changepoint_features``'s scales read off the labels.

    Every unset one of ``sigmas`` / ``horizon`` / ``max_length`` is derived
    from the durations of the curated state events of the branch's classes,
    over the trials the config selects in every session, at the rate of the
    first mask being expanded (:func:`~ethograph.features.changepoints.scales_from_durations`).
    A config that is already resolved, or has no changepoint section, comes
    back unchanged.
    """
    cpf = config.features.changepoint_features
    if cpf is None or not cpf.unresolved:
        return config
    ids = target_label_ids(config)
    durations: list[np.ndarray] = []
    n_sessions = 0
    for session in sessions:
        n_sessions += 1
        for tid in filter_trials(session, config.trials):
            labels = session.curated_labels(tid)
            if labels.empty:
                continue
            df = states_only(labels)
            df = df[df["labels"].astype(int).isin(ids)]
            durations.append((df["offset_s"] - df["onset_s"]).to_numpy(dtype=float))
    d = np.concatenate(durations) if durations else np.array([], dtype=float)
    fs = _mask_rate(sessions[0], next(iter(cpf.inputs)))
    context = f"{d.size} manual/curated state events of {n_sessions} session(s)"
    resolved = cpf.resolve(d, fs, context)
    logger.info("features.changepoint_features: %s", resolved.note)
    return replace(config, features=replace(config.features, changepoint_features=resolved))


def resolve_neural_columns(config: SegmentConfig, sessions: list[Session]) -> SegmentConfig:
    """The config with ``features.columns.{neural.name}`` spelled as the session's unit ids.

    The units are the session's own, so they are read off the opened
    session rather than written in the YAML; a config that already spells
    the entry (a subset, or a run config re-read) comes back unchanged.
    Every session must carry the same units — which in practice means one
    session, since units are not consistent across recordings.
    """
    cfg = config.features.neural
    if cfg is None or cfg.name in config.features.columns:
        return config
    if not sessions:
        raise ValueError("features.neural needs a session to read the unit ids off")
    dims = neural_columns(sessions[0], cfg)
    for session in sessions[1:]:
        other = neural_columns(session, cfg)
        if other != dims:
            raise ValueError(
                f"{session.source}: its units {other} differ from {sessions[0].source}'s {dims} — units are only "
                "consistent within one session, so a neural project lists one session (or spells "
                f"features.columns.{cfg.name} to pin a shared subset)."
            )
    n_units = sum(len(v) for v in dims.values())
    logger.info("features.neural: %r resolved to %d unit column(s): %s", cfg.name, n_units, dims)
    columns = {**config.features.columns, cfg.name: dims}
    return replace(config, features=replace(config.features, columns=columns))


def _mask_rate(session: Session, var: str) -> float:
    """Sampling rate of changepoint mask *var*, off its own time coordinate in the first trial."""
    ds = session.trial_dataset(session.trial_ids[0])
    if ds is None:
        raise ValueError(f"{session.source}: changepoint expansion needs an xarray session")
    if var not in ds.data_vars:
        raise ValueError(f"{session.source}: no variable {var!r} to read a sampling rate off")
    time = get_time_coord(ds[var])
    if time is None or time.size < 2:
        raise ValueError(f"{session.source}: {var!r} has no time coordinate to read a sampling rate off")
    return float(1.0 / np.median(np.diff(np.asarray(time.values, dtype=float))))


def resolved_config(config: SegmentConfig) -> SegmentConfig:
    """The config with what the materialised dataset recorded: changepoint scales and unit columns.

    What every stage after ``materialise`` calls before it opens a session or
    saves a run config, so the numbers a run trains and predicts with are
    the ones ``columns.yaml`` holds, never re-derived. A config that is
    already resolved comes back unchanged; one that is unresolved with no
    materialised dataset to read from is an error naming the fix.
    """
    return _resolved_changepoints(_resolved_neural(config))


def _resolved_neural(config: SegmentConfig) -> SegmentConfig:
    cfg = config.features.neural
    if cfg is None or cfg.name in config.features.columns:
        return config
    path = config.data_dir / COLUMNS_FILE
    if not path.is_file():
        raise ValueError(
            f"features.neural leaves the unit columns of {cfg.name!r} to be read off the session, which happens "
            f"at materialise — and {config.data_dir} holds no materialised dataset yet. Run Project.materialise() "
            f"first, or spell features.columns.{cfg.name} in the config."
        )
    recorded = read_layout(config.data_dir).neural_columns or {}
    if cfg.name not in recorded:
        raise ValueError(
            f"{path} records no unit columns for {cfg.name!r} — it was materialised by a config without "
            "features.neural (or with another name). Re-materialise, or spell the columns in the config."
        )
    columns = {**config.features.columns, cfg.name: dict(recorded[cfg.name])}
    return replace(config, features=replace(config.features, columns=columns))


def _resolved_changepoints(config: SegmentConfig) -> SegmentConfig:
    cpf = config.features.changepoint_features
    if cpf is None or not cpf.unresolved:
        return config
    path = config.data_dir / COLUMNS_FILE
    if not path.is_file():
        raise ValueError(
            "features.changepoint_features leaves sigmas/horizon/max_length to be derived from the labels, "
            f"which happens at materialise — and {config.data_dir} holds no materialised dataset yet. "
            "Run Project.materialise() first, or spell the three values (samples) in the config."
        )
    scales = read_layout(config.data_dir).changepoint_features
    if scales is None:
        raise ValueError(
            f"{path} records no changepoint scales — it was materialised by a config without "
            "features.changepoint_features. Re-materialise, or spell sigmas/horizon/max_length in the config."
        )
    return replace(config, features=replace(config.features, changepoint_features=cpf.with_scales(scales)))


def _check_fs(fs_ref: float, fs: float, key: str) -> None:
    if not np.isclose(fs_ref, fs, rtol=1e-3):
        raise ValueError(f"{key}: sampling rate {fs:.6g} Hz differs from the dataset's {fs_ref:.6g} Hz")


def _write_ground_truth(folder: Path, key: str, y: np.ndarray, classes: TargetTable) -> None:
    if is_multilabel(classes):
        np.save(folder / f"{key}.npy", np.asarray(y, dtype=np.uint8))
        return
    names = classes.names
    (folder / f"{key}.txt").write_text("".join(f"{names[int(i)]}\n" for i in y), encoding="utf-8")


# ---------------------------------------------------------------------------
# Reading back
# ---------------------------------------------------------------------------


def read_index(data_dir: Path) -> pd.DataFrame:
    path = data_dir / INDEX_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{data_dir} holds no materialised dataset — call Project.materialise() first")
    return pd.read_csv(path, sep="\t", dtype={"individual": str})


def read_layout(data_dir: Path) -> ColumnLayout:
    return ColumnLayout.from_dict(yaml.safe_load((data_dir / COLUMNS_FILE).read_text(encoding="utf-8")))


def read_classes(data_dir: Path) -> TargetTable:
    return read_target_table(data_dir / CLASSES_FILE)


def read_target_table(path: Path) -> TargetTable:
    """A ``classes.yaml`` (of a dataset or a run) → its table."""
    return target_table_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_sample(data_dir: Path, key: str, classes: TargetTable) -> tuple[np.ndarray, np.ndarray]:
    """``(x (F, T) float32, y)`` of one materialised sample.

    ``y`` is ``(T,)`` class indices for an exclusive target and ``(C, T)``
    0/1 for a multi-label one — the time axis is the last one either way.
    """
    x = np.load(data_dir / "features" / f"{key}.npy")
    if is_multilabel(classes):
        y = np.load(data_dir / "groundTruth" / f"{key}.npy").astype(np.int64)
        n = min(x.shape[1], y.shape[1])
        return x[:, :n], y[:, :n]
    names = (data_dir / "groundTruth" / f"{key}.txt").read_text(encoding="utf-8").splitlines()
    index_of = {name: i for i, name in enumerate(classes.names)}
    y = np.fromiter((index_of[n] for n in names), dtype=np.int64, count=len(names))
    n = min(x.shape[1], len(y))
    return x[:, :n], y[:n]
