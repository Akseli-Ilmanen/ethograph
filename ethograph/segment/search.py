"""Hyperparameter search: Optuna over the config, scored on the validation trials.

Stage 1 of the workflow. Every trial of the study is one full training run
with a different draw of ``search.params``, and its objective is
:attr:`~ethograph.segment.train.RunResult.best_score` — ``train.select_on``
measured on the ``val`` trials of ``train.split``. That is the whole reason
the validation split exists: it is the only thing that chooses settings, and
``test`` stays untouched so the number you report at the end is still honest.

A search parameter is keyed by the same dotted path an override uses
(``train.learning_rate``, ``model.params.num_f_maps``), so a space and a hand
override are the same spelling, and the winning draw is written back out as a
config you can point ``Project`` at::

    result = project.search()
    best = eto.segment.Project(result.config_path)  # searches/{name}/best.yaml

``best.yaml`` is a two-line config: ``base:`` the file you searched, plus the
parameters that won. Diffable, re-runnable, and it names its provenance.

Everything the study writes lives under ``{root}/searches/{name}/``:

* ``study.db`` — the Optuna storage; re-running resumes it rather than restarting
* ``trials.tsv`` — one row per trial: value, parameters, run directory, state
* ``best.yaml`` — ``base:`` your config, plus the winning parameters
* ``eval_comparison.pdf`` — class-wise F1 / IoU / boundary deltas across every
  finished trial's test split, one dot per trial
* ``search.log`` — everything logged during the study

The runs themselves are ordinary runs, nested under ``runs/{name}/`` so a
study does not bury the runs you trained by hand. Their weights are pruned
when the study ends (``search.keep_weights``); their config, split and
metrics are always kept.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import yaml

from ethograph.segment.config import SegmentConfig, config_from_dict, config_to_dict
from ethograph.segment.materialise import COLUMNS_FILE, materialise, read_target_table
from ethograph.segment.metrics import EVAL_ARRAYS_FILE
from ethograph.segment.train import BEST_FILE, LAST_FILE, run_name_for, train
from ethograph.utils.configkit import apply_overrides, as_overrides, deep_merge
from ethograph.utils.logging import log_to_file

logger = logging.getLogger(__name__)

STUDY_FILE = "study.db"
TRIALS_FILE = "trials.tsv"
BEST_CONFIG_FILE = "best.yaml"


@dataclass
class SearchResult:
    """What a study produced."""

    search_dir: Path
    #: dotted config key → the winning value.
    best_params: dict[str, Any]
    #: ``train.select_on`` on the validation trials, for the winning draw.
    best_score: float
    best_run_dir: Path
    #: Wall-clock training time of the winning trial, in seconds.
    best_train_seconds: float
    #: ``searches/{name}/best.yaml`` — a config that inherits yours and pins those params.
    config_path: Path
    #: One row per trial: number, state, value and every parameter.
    trials: pd.DataFrame

    @property
    def overrides(self) -> list[str]:
        """The winning draw as dotted ``key=value`` strings, for :meth:`Project.update`."""
        return as_overrides(dict(sorted(self.best_params.items())))


def search_name_for(config: SegmentConfig) -> str:
    """The configured (or derived) study name — ``{root}/searches/{name}``."""
    return config.search.name or f"search_{run_name_for(config)}"


def _nest(params: dict[str, Any]) -> dict[str, Any]:
    """``{"train.lr": 1e-4}`` → ``{"train": {"lr": 1e-4}}``."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def _config_with(config: SegmentConfig, params: dict[str, Any]) -> SegmentConfig:
    """*config* rebuilt with *params* (dotted keys) applied — so a bad draw fails here."""
    data = config_to_dict(config)
    base_dir = config.config_path.parent if config.config_path else config.root
    return config_from_dict(apply_overrides(data, as_overrides(params)), base_dir, config_path=config.config_path)


def _write_best_config(config: SegmentConfig, params: dict[str, Any], path: Path) -> Path:
    """A config that inherits the searched one and pins the winning parameters."""
    data: dict[str, Any] = _nest(params)
    if config.config_path is not None:
        # `base:` resolves relative to the file that names it.
        data = {"base": os.path.relpath(config.config_path, path.parent).replace("\\", "/"), **data}
    else:
        # No file to inherit from: write the whole resolved config instead.
        data = deep_merge(config_to_dict(config), data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _prune_weights(run_dirs: list[Path], keep: Path) -> None:
    """Delete the checkpoints of every run but *keep* — a study is dozens of runs."""
    freed = 0
    dropped = 0
    for run_dir in run_dirs:
        if run_dir == keep:
            continue
        for name in (BEST_FILE, LAST_FILE):
            path = run_dir / name
            if path.is_file():
                freed += path.stat().st_size
                path.unlink()
                dropped += 1
    if dropped:
        logger.info(
            "Dropped the weights of %d losing trials, %.1f MB (search.keep_weights is off; "
            "their config, split and metrics are kept)",
            dropped,
            freed / 1e6,
        )


def _write_trial_comparison(search_dir: Path, run_dirs: list[Path], name: str) -> None:
    """The cross-trial comparison figure over every finished trial's held-out test.

    Every finished trial shares the same ``train.split``, so this is the same
    test set scored under each draw of ``search.params`` — different from the
    study's own objective, which only ever sees ``val``.
    """
    from ethograph.segment.plotting import load_run_eval, write_comparison_pdf

    evals = [load_run_eval(d, name=d.name) for d in run_dirs if (d / EVAL_ARRAYS_FILE).is_file()]
    if len(evals) < 2:
        logger.info(
            "Fewer than 2 finished trials of %r wrote %s (train.split.test_fraction may be 0) — "
            "skipping the comparison figure",
            name,
            EVAL_ARRAYS_FILE,
        )
        return
    classes = read_target_table(run_dirs[0] / "classes.yaml")
    path = write_comparison_pdf(
        search_dir / "eval_comparison.pdf", evals, classes, title=f"Search {name} — {len(evals)} trials"
    )
    logger.info("Wrote %s", path)


def search(config: SegmentConfig, n_trials: int | None = None) -> SearchResult:
    """Run the configured Optuna study and return its best draw.

    Materialises the dataset once up front, so every trial reads the same
    features — a search tunes the *model*, never the feature engineering.
    """
    scfg = config.search
    if not scfg.params:
        raise ValueError(
            "search.params is empty — name at least one hyperparameter to search, e.g.\n"
            "  search:\n"
            "    params:\n"
            "      train.learning_rate: {type: float, low: 1.0e-5, high: 1.0e-2, log: true}"
        )
    if not config.train.split.val_fraction:
        raise ValueError(
            "A search is scored on the validation trials, and train.split.val_fraction is 0. "
            "Set the three fractions to something like 0.6 / 0.2 / 0.2."
        )
    if config.train.split.holdout_sessions:
        raise ValueError(
            "train.split.holdout_sessions is set, which is a cross-validation fold, not a search. "
            "Search over the ratio split first, then cross-validate with the parameters it found."
        )

    if not (config.data_dir / COLUMNS_FILE).is_file():
        logger.info("No materialised dataset at %s — materialising once for the whole study", config.data_dir)
        materialise(config)

    name = search_name_for(config)
    search_dir = config.searches_dir / name
    search_dir.mkdir(parents=True, exist_ok=True)
    total = int(n_trials if n_trials is not None else scfg.n_trials)

    with log_to_file(search_dir / "search.log"):
        # Optuna installs its own stream handler, which would print every trial
        # a second time next to ours and bypass the search.log entirely.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            study_name=name,
            storage=f"sqlite:///{(search_dir / STUDY_FILE).as_posix()}",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=scfg.seed),
            pruner=optuna.pruners.MedianPruner() if scfg.prune else optuna.pruners.NopPruner(),
        )
        run_dirs: dict[int, Path] = {}
        logger.info(
            "Search %r: %d trials over %s, maximising val %s",
            name,
            total,
            sorted(scfg.params),
            config.train.select_on,
        )

        def objective(trial: Any) -> float:
            params = {key: space.suggest(trial, key) for key, space in scfg.params.items()}
            trial_config = _config_with(config, {**params, "train.run_name": f"{name}/trial{trial.number:03d}"})
            logger.info("[trial %d] %s", trial.number, params)

            def report(epoch: int, score: float) -> None:
                trial.report(score, epoch)
                if trial.should_prune():
                    logger.info("[trial %d] behind the median at epoch %d — abandoned", trial.number, epoch)
                    raise optuna.TrialPruned(f"trial {trial.number} pruned at epoch {epoch}")

            result = train(trial_config, on_eval=report)
            run_dirs[trial.number] = result.run_dir
            trial.set_user_attr("run_dir", str(result.run_dir))
            trial.set_user_attr("best_epoch", result.best_epoch)
            trial.set_user_attr("train_seconds", result.train_seconds)
            logger.info(
                "[trial %d] val %s = %.4f at epoch %d",
                trial.number,
                config.train.select_on,
                result.best_score,
                result.best_epoch,
            )
            return result.best_score

        study.optimize(objective, n_trials=total, timeout=scfg.timeout)

        rows = [
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "best_epoch": t.user_attrs.get("best_epoch"),
                "train_seconds": t.user_attrs.get("train_seconds"),
                "run_dir": t.user_attrs.get("run_dir"),
                **t.params,
            }
            for t in study.trials
        ]
        trials = pd.DataFrame(rows)
        trials.to_csv(search_dir / TRIALS_FILE, sep="\t", index=False)

        finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not finished:
            states = sorted({t.state.name for t in study.trials})
            raise RuntimeError(
                f"No trial of {name!r} finished (states: {states}). "
                "With every trial pruned, lower search.prune or raise train.epochs; "
                f"the per-trial logs are under {config.runs_dir / name}."
            )
        best = study.best_trial
        best_run_dir = Path(best.user_attrs["run_dir"])
        config_path = _write_best_config(config, dict(best.params), search_dir / BEST_CONFIG_FILE)
        if not scfg.keep_weights:
            _prune_weights(list(run_dirs.values()), keep=best_run_dir)
        _write_trial_comparison(search_dir, [Path(t.user_attrs["run_dir"]) for t in finished], name)
        logger.info(
            "Best trial %d: val %s = %.4f  %s", best.number, config.train.select_on, best.value, dict(best.params)
        )
        logger.info("Train with it: eto.segment.Project(%r)", str(config_path))
        return SearchResult(
            search_dir=search_dir,
            best_params=dict(best.params),
            best_score=float(best.value),
            best_run_dir=best_run_dir,
            best_train_seconds=float(best.user_attrs["train_seconds"]),
            config_path=config_path,
            trials=trials,
        )
