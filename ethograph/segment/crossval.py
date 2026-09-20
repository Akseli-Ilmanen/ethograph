"""Cross-validation: one fold per session — or per group of trials — each predicting what it never saw.

Stage 2 of the workflow, once a {mod}`search <ethograph.segment.search>` has
settled the hyperparameters. Leave-one-session-out: fold *i* trains on every
session but the *i*-th and then runs inference over that held-out session, so
each session ends up with a prediction set produced by a model that never saw
one of its frames.

That last part is the point. Predictions land in the GUI's own labels format
beside the session, under its own ``labels/`` folder
(``labels/predictions_{run}_{timestamp}/{stem}_predictions.tsv``), so you load them
next to the curated labels and look at *where* the model is still wrong —
which classes, which trials, which boundaries — rather than at one aggregate
number. A random trial split cannot give you that: its test trials share a
recording day, lighting and animal with the trials the model trained on, and
they are scattered across sessions rather than making up one you can open.

**Trial folds** (``n_folds=k``) are for the project whose sessions cannot be
held out: one session of neural decoding, whose units exist in that recording
only. Every trial is dealt into exactly one of *k* folds, fold *i* trains on
the others and predicts its own, and the fold predictions are merged into one
prediction set per session — so every trial is still predicted exactly once,
by a model that never saw it, and the set still opens in the GUI as a whole.

Everything is written under ``{root}/cross_validation/{name}/``::

    folds.tsv  # one row per fold: what was held out, run, test metrics, prediction path
    predictions.tsv  # trial folds only: the merged prediction set per session
    eval_comparison.pdf  # class-wise F1 / IoU / boundary deltas across folds, one dot per fold
    crossval.log

The folds themselves are ordinary runs, nested under ``runs/{name}/`` so they
do not bury the runs you trained by hand.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ethograph.segment.config import (
    FERAL,
    SegmentConfig,
    SessionSpec,
    config_from_dict,
    config_to_dict,
)
from ethograph.segment.inference import inference, merge_prediction_sets
from ethograph.segment.materialise import COLUMNS_FILE, materialise, read_index, read_target_table
from ethograph.segment.metrics import EVAL_ARRAYS_FILE
from ethograph.segment.prediction_sets import prediction_run_dir
from ethograph.segment.train import RunResult, run_name_for, train
from ethograph.utils.configkit import apply_overrides, as_overrides
from ethograph.utils.logging import log_to_file
from ethograph.utils.paths import session_id

logger = logging.getLogger(__name__)

FOLDS_FILE = "folds.tsv"
MERGED_FILE = "predictions.tsv"


@dataclass
class Fold:
    """One held-out session (or group of trials) and what training on the rest produced."""

    #: The held-out session's id, or ``fold-{k}`` for a trial fold.
    session: str
    source: Path | None
    run_dir: Path
    best_epoch: int
    #: ``test_metrics.yaml`` of the fold: what it held out, raw and post-processed.
    metrics: dict[str, Any] | None
    #: The prediction set written beside the held-out session (``None`` if ``predict=False``).
    predictions: Path | None
    #: Trial fold only: the trial ids held out.
    trials: list[str] | None = None
    #: Trial fold only: the fold's prediction set per session, in config order.
    prediction_sets: list[Path] = field(default_factory=list)


def cross_validation_name_for(config: SegmentConfig) -> str:
    """The directory a cross-validation writes into — ``{root}/cross_validation/{name}``."""
    return f"cv_{run_name_for(config)}"


def _rebuild(config: SegmentConfig, overrides: dict[str, Any]) -> SegmentConfig:
    base_dir = config.config_path.parent if config.config_path else config.root
    return config_from_dict(
        apply_overrides(config_to_dict(config), as_overrides(overrides)), base_dir, config_path=config.config_path
    )


def _fold_config(config: SegmentConfig, held_out: SessionSpec, name: str, val_fraction: float) -> SegmentConfig:
    """*config* with *held_out* pinned as the test session and the fold's run name."""
    return _rebuild(
        config,
        {
            "train.split.holdout_sessions": [str(held_out.source)],
            "train.split.train_fraction": 1.0 - val_fraction,
            "train.split.val_fraction": val_fraction,
            "train.split.test_fraction": 0.0,
            # The session *id*, not its stem: sessions of one project are
            # routinely files of the same name in different session folders,
            # and a fold has to be tellable from the other three.
            "train.run_name": f"{name}/fold-{session_id(held_out.source)}",
        },
    )


def _trial_fold_config(
    config: SegmentConfig, held_out: list[str], k: int, name: str, val_fraction: float
) -> SegmentConfig:
    """*config* with *held_out* trials pinned as the test set and the fold's run name."""
    return _rebuild(
        config,
        {
            "train.split.holdout_trials": list(held_out),
            "train.split.train_fraction": 1.0 - val_fraction,
            "train.split.val_fraction": val_fraction,
            "train.split.test_fraction": 0.0,
            "train.run_name": f"{name}/fold-{k}",
        },
    )


def trial_folds(trials: Iterable[int | str], n_folds: int, seed: int) -> list[list[str]]:
    """Deal *trials* into *n_folds* disjoint groups, deterministically.

    Shuffled once with *seed* and dealt round-robin, so the folds differ in
    size by at most one and every trial sits in exactly one. Ids come back
    as strings, the spelling ``train.split.holdout_trials`` compares by.
    """
    ids = sorted({str(t) for t in trials}, key=lambda s: (len(s), s))
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if n_folds > len(ids):
        raise ValueError(f"n_folds={n_folds} but the sessions have only {len(ids)} trials to deal")
    order = list(ids)
    random.Random(seed).shuffle(order)
    return [sorted(order[k::n_folds], key=lambda s: (len(s), s)) for k in range(n_folds)]


def cross_validate(
    config: SegmentConfig,
    folds: Iterable[str | Path] | None = None,
    val_fraction: float = 0.0,
    predict: bool = True,
    n_folds: int | None = None,
) -> pd.DataFrame:
    """Leave-one-session-out over *folds* — or *n_folds* trial folds — returning one row per fold.

    *folds* names the sessions to hold out — by path or by source stem —
    defaulting to every session, which is the full cross-validation. Naming
    two of six is how you compare parameter sets without paying for all six.

    *n_folds* switches to trial folds: every trial of every session is dealt
    into exactly one of *n_folds* folds (:func:`trial_folds`, seeded by
    ``train.split.seed``), each fold trains on the rest and predicts its
    own, and the folds' predictions are merged into one prediction set per
    session (``predictions.tsv`` lists them). Required when the config has
    one session; exclusive with *folds*.

    *val_fraction* carves a validation slice out of the *training* part of
    each fold, to select ``best.pt``. It defaults to ``0``: after a search
    the hyperparameters (``epochs`` included) are settled, every remaining
    trial is worth training on, and ``best.pt`` is the last epoch.

    With *predict*, each fold also writes a prediction set beside its
    held-out session — the reason to run this rather than a random split.
    """
    split = config.train.split
    if split.holdout_sessions or split.holdout_trials:
        raise ValueError(
            "train.split.holdout_sessions / holdout_trials is already set — that is one fold, pinned by hand. "
            "Cross-validation writes it per fold; leave it out of the config."
        )
    if config.video_features.extractor == FERAL and FERAL in config.features.columns:
        logger.warning(
            "Cross-validating over FERAL embeddings: FERAL was fine-tuned on the labels of every session it "
            "trained on, so a fold that holds out one of those sessions scores a model whose input already knows "
            "the answer. These fold scores are inflated. For a number you can report, name the true test "
            "sessions in train.split.holdout_sessions before project.video_features() (they stay out of FERAL's "
            "training) and read train()'s test score."
        )
    if n_folds is not None and folds is not None:
        raise ValueError("Pass either folds (sessions to hold out) or n_folds (trial folds), not both.")
    if n_folds is None and len(config.sessions) < 2:
        raise ValueError(
            f"Leave-one-session-out needs at least two sessions, this config has {len(config.sessions)}. "
            "Fold by trial instead: cross_validate(n_folds=5)."
        )

    name = cross_validation_name_for(config)
    out_dir = config.cross_validation_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (config.data_dir / COLUMNS_FILE).is_file():
        logger.info("No materialised dataset at %s — materialising once for every fold", config.data_dir)
        materialise(config)

    with log_to_file(out_dir / "crossval.log"):
        if n_folds is not None:
            results = _run_trial_folds(config, name, n_folds, val_fraction, predict)
        else:
            results = _run_session_folds(config, name, config.select_sessions(folds), val_fraction, predict)

        table = _fold_table(results)
        table.to_csv(out_dir / FOLDS_FILE, sep="\t", index=False)
        logger.info("Wrote %s", out_dir / FOLDS_FILE)
        if n_folds is not None and predict:
            merged = _merge_trial_fold_predictions(config, results, name)
            merged.to_csv(out_dir / MERGED_FILE, sep="\t", index=False)
            logger.info("Wrote %s", out_dir / MERGED_FILE)
        _write_fold_comparison(out_dir, results)
        return table


def _run_session_folds(
    config: SegmentConfig, name: str, held_out: list[SessionSpec], val_fraction: float, predict: bool
) -> list[Fold]:
    logger.info(
        "Cross-validation %r: %d of %d sessions held out, val_fraction=%g",
        name,
        len(held_out),
        len(config.sessions),
        val_fraction,
    )
    results: list[Fold] = []
    for n, spec in enumerate(held_out, start=1):
        logger.info("[fold %d/%d] holding out %s", n, len(held_out), session_id(spec.source))
        fold_config = _fold_config(config, spec, name, val_fraction)
        result: RunResult = train(fold_config)
        predictions = None
        if predict:
            written = inference(fold_config, run=result.run_dir, sessions=[str(spec.source)])
            predictions = written[0]
        results.append(
            Fold(
                session=session_id(spec.source),
                source=spec.source,
                run_dir=result.run_dir,
                best_epoch=result.best_epoch,
                metrics=result.test_metrics,
                predictions=predictions,
            )
        )
    return results


def _run_trial_folds(config: SegmentConfig, name: str, n_folds: int, val_fraction: float, predict: bool) -> list[Fold]:
    index = read_index(config.data_dir)
    groups = trial_folds(list(index["trial"]), n_folds, config.train.split.seed)
    logger.info(
        "Cross-validation %r: %d trial folds over %d trials of %d session(s), val_fraction=%g",
        name,
        n_folds,
        sum(len(g) for g in groups),
        len(config.sessions),
        val_fraction,
    )
    results: list[Fold] = []
    for k, held_out in enumerate(groups):
        logger.info("[fold %d/%d] holding out %d trials: %s", k + 1, n_folds, len(held_out), held_out)
        fold_config = _trial_fold_config(config, held_out, k, name, val_fraction)
        result: RunResult = train(fold_config)
        written: list[Path] = []
        if predict:
            written = inference(fold_config, run=result.run_dir, trials=held_out)
        results.append(
            Fold(
                session=f"fold-{k}",
                source=None,
                run_dir=result.run_dir,
                best_epoch=result.best_epoch,
                metrics=result.test_metrics,
                predictions=written[0] if written else None,
                trials=held_out,
                prediction_sets=written,
            )
        )
    return results


def _merge_trial_fold_predictions(config: SegmentConfig, folds: list[Fold], name: str) -> pd.DataFrame:
    """One merged prediction set per session, from every fold's held-out trials."""
    from ethograph.segment.sessions import open_session

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for i, spec in enumerate(config.sessions):
        session = open_session(spec, config)
        paths = [f.prediction_sets[i] for f in folds if len(f.prediction_sets) > i]
        out_dir = prediction_run_dir(spec.source, name, timestamp)
        merged = merge_prediction_sets(
            paths,
            out_dir,
            session.stem,
            model_config=folds[0].run_dir / "config.yaml",
            inference_note={
                "model": "segment",
                "cross_validation": name,
                "session": str(spec.source),
                "folds": [{"run_dir": str(f.run_dir), "trials": f.trials} for f in folds],
                "infer": config_to_dict(config)["infer"],
            },
        )
        logger.info("%s: merged %d fold prediction sets → %s", session.id, len(paths), merged)
        rows.append({"session": session.id, "source": str(spec.source), "predictions": str(merged)})
    return pd.DataFrame(rows)


def _write_fold_comparison(out_dir: Path, folds: list[Fold]) -> None:
    """The cross-fold comparison figure — one dot per fold's held-out test."""
    from ethograph.segment.plotting import load_run_eval, write_comparison_pdf

    evals = [load_run_eval(f.run_dir, name=f.session) for f in folds if (f.run_dir / EVAL_ARRAYS_FILE).is_file()]
    if len(evals) < 2:
        logger.info(
            "Fewer than 2 folds wrote %s (test_fraction may be 0, or a fold never reached a test evaluation) — "
            "skipping the comparison figure",
            EVAL_ARRAYS_FILE,
        )
        return
    classes = read_target_table(folds[0].run_dir / "classes.yaml")
    path = write_comparison_pdf(
        out_dir / "eval_comparison.pdf", evals, classes, title=f"Cross-validation — {len(evals)} folds"
    )
    logger.info("Wrote %s", path)


def _fold_table(folds: list[Fold]) -> pd.DataFrame:
    """One row per fold: what it held out, its run, and the metrics on what it never saw."""
    rows = []
    for fold in folds:
        row: dict[str, Any] = {
            "session": fold.session,
            "run": fold.run_dir.name,
            "run_dir": str(fold.run_dir),
            "best_epoch": fold.best_epoch,
            "predictions": str(fold.predictions) if fold.predictions else None,
        }
        if fold.trials is not None:
            row["trials"] = ",".join(fold.trials)
        for stage in ("raw", "postprocessed"):
            for key, value in (fold.metrics or {}).get(stage, {}).items():
                if key != "classwise":
                    row[f"{stage}.{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)
