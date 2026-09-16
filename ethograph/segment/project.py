"""``Project`` — a segmentation project and everything you can do with it.

One object, built from one YAML config, with a method per stage. There is no
command line: a pipeline run is a script, so it is reviewable, diffable and
re-runnable, and there is exactly one way to express it.

    import ethograph as eto

    project = eto.segment.Project("project.yaml")
    project.video_features(merge=True)     # video features once per video, merged into the sessions
    project.materialise()                  # feature engineering → the materialised dataset

    # Stage 1 — settle the hyperparameters on the 60/20/20 split
    best = project.search()

    # Stage 2 — cross-validate them, one fold per session, and look at the
    # predictions in the GUI
    eto.segment.Project(best.config_path).cross_validate()

Anything in the config can be overridden without editing the file, using the
same dotted keys the YAML has::

    project.update("model.architecture=mstcn", "train.run_name=mstcn")

which is how a benchmark is written — a loop over overrides, then
:meth:`compare`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import yaml

from ethograph.segment.config import FERAL, FERAL_EMBEDDINGS_DIR, SegmentConfig, load_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from ethograph.segment.feral import FeralExport
    from ethograph.segment.samples import ColumnLayout
    from ethograph.segment.search import SearchResult
    from ethograph.segment.sessions import Session
    from ethograph.segment.train import RunResult
    from ethograph.video_features.select import FeatureRanking

logger = logging.getLogger(__name__)


class Project:
    """A segmentation project: its config, its data, its runs."""

    def __init__(self, config: str | Path | SegmentConfig, *overrides: str) -> None:
        """Open the project described by *config*.

        *config* is a path to a YAML file (the usual case) or an already-built
        :class:`~ethograph.segment.config.SegmentConfig`. Positional
        *overrides* are dotted ``key.path=value`` strings applied on top.
        """
        if isinstance(config, SegmentConfig):
            if overrides:
                raise ValueError("Overrides need a config file to re-read; pass the path instead.")
            self._config = config
            self._path: Path | None = config.config_path
            self._overrides: list[str] = []
        else:
            self._path = Path(config)
            self._overrides = list(overrides)
            self._config = load_config(self._path, self._overrides)

    def __repr__(self) -> str:
        name = self._path.name if self._path else "<in-memory>"
        return (
            f"Project({name!r}, sessions={len(self.config.sessions)}, "
            f"architecture={self.config.model.architecture!r}, root={str(self.root)!r})"
        )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @property
    def config(self) -> SegmentConfig:
        """The resolved config. Read it freely; change it with :meth:`update`."""
        return self._config

    @property
    def root(self) -> Path:
        """The project directory — ``data/``, ``runs/`` and ``video_features/`` live here."""
        return self._config.root

    def update(self, *overrides: str) -> Project:
        """Apply more dotted overrides, in place, and return self.

        The config is rebuilt from the file each time, so overrides accumulate
        and a typo is caught here rather than half-way through a run.
        """
        if self._path is None:
            raise ValueError("This project was built from a SegmentConfig object; there is no file to re-read.")
        self._overrides.extend(overrides)
        self._config = load_config(self._path, self._overrides)
        return self

    def sessions(self) -> list[Session]:
        """Every session of the config, opened (no Qt involved)."""
        from ethograph.segment.sessions import open_session

        return [open_session(spec, self._config) for spec in self._config.sessions]

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def video_features(self, merge: bool = False, overwrite: bool = False, in_place: bool = False) -> list[Path]:
        """Run the configured extractor over every video the sessions use → ``{root}/video_features``.

        With *merge*, each session is also written out carrying the feature
        (named after ``video_features.extractor``) on its trials' own time
        axis — a sibling ``{stem}_{extractor}.nc``, never the source file
        unless *in_place*. Returns the sidecars written (the merged sessions
        are logged).
        """
        from ethograph.segment.video_features import extract_video_features, merge_video_features

        if in_place and not merge:
            raise ValueError("in_place only means something with merge=True — it says where the merge is written.")
        if self._config.video_features.extractor == FERAL:
            if merge:
                raise ValueError(
                    "FERAL's embeddings are never merged into the session file: they attach from "
                    f"{self._config.feral_dir / FERAL_EMBEDDINGS_DIR} whenever a session opens under this config."
                )
            return self.export_feral().files
        written = extract_video_features(self._config, overwrite=overwrite)
        if merge:
            for session in self.sessions():
                merge_video_features(session, self._config, in_place=in_place)
        return written

    def export_feral(self) -> FeralExport:
        """Write FERAL's inputs for this project's sessions → ``{root}/feral/``.

        ``labels.json`` and a complete ``config.yaml`` for ``feral
        train-config``, with every trial's curated labels rasterised on its
        video's frames and the roles ``train()`` would draw —
        ``train.split.holdout_sessions`` are FERAL's ``test`` and
        ``inference`` only, so their embeddings come from a model that never
        saw their labels. FERAL then runs in its own environment; the log
        names the commands. See :mod:`ethograph.segment.feral`.
        """
        from ethograph.segment.feral import export_feral

        return export_feral(self._config)

    def materialise(self) -> Path:
        """Feature engineering: write the materialised dataset, and return its path."""
        from ethograph.segment.materialise import materialise

        return materialise(self._config)

    def train(self) -> RunResult:
        """Fit the configured architecture; materialises first if needed."""
        from ethograph.segment.train import train

        return train(self._config)

    def search(self, n_trials: int | None = None) -> SearchResult:
        """Optuna over ``search.params``, scored on the validation trials.

        Stage 1: find the hyperparameters. Every trial is a full run judged by
        ``train.select_on`` on ``val``; the winning draw is written to
        ``searches/{name}/best.yaml`` as a config inheriting this one, so
        stage 2 is ``Project(result.config_path).cross_validate()``.

        *n_trials* overrides ``search.n_trials`` for this call. A study is
        stored in ``searches/{name}/study.db``, so calling this again adds
        trials to it rather than starting over.
        """
        from ethograph.segment.search import search

        return search(self._config, n_trials=n_trials)

    def cross_validate(
        self,
        folds: Iterable[str | Path] | None = None,
        val_fraction: float = 0.0,
        predict: bool = True,
        n_folds: int | None = None,
    ) -> pd.DataFrame:
        """Leave-one-session-out, predicting each held-out session for the GUI.

        Stage 2, once :meth:`search` has settled the hyperparameters: fold *i*
        trains on every session but the *i*-th and predicts that one, so every
        prediction set comes from a model that never saw the session. Load it
        beside the curated labels to see *where* the model is still wrong.

        *folds* names the sessions to hold out (by path or source stem);
        default is all of them. Naming two of six compares parameter sets
        without paying for the full sweep. *val_fraction* carves a validation
        slice out of each fold's training sessions (default ``0`` — the
        hyperparameters are already fixed).

        *n_folds* folds by **trial** instead: every trial is dealt into
        exactly one of *n_folds* folds, each fold trains on the others and
        predicts its own, and the fold predictions are merged into one
        prediction set per session. The cross-validation for a project whose
        sessions cannot be held out — one session of neural decoding, whose
        units exist in that recording only.
        """
        from ethograph.segment.crossval import cross_validate

        return cross_validate(self._config, folds=folds, val_fraction=val_fraction, predict=predict, n_folds=n_folds)

    def inference(
        self,
        run: str | Path | None = None,
        sessions: Iterable[str | Path] | None = None,
        trials: Iterable[int | str] | None = None,
    ) -> list[Path]:
        """Write a prediction set beside every session of the config.

        *sessions* narrows that to the ones it names (full path or source
        stem), *trials* to the trial ids it names; *run* names a run other
        than the config's own.
        """
        from ethograph.segment.inference import inference

        return inference(self._config, run=run, sessions=sessions, trials=trials)

    # ------------------------------------------------------------------
    # Looking at results
    # ------------------------------------------------------------------

    def runs(self) -> list[str]:
        """Names of the runs trained in this project, oldest first."""
        runs_dir = self._config.runs_dir
        if not runs_dir.is_dir():
            return []
        return sorted(p.name for p in runs_dir.iterdir() if (p / "config.yaml").is_file())

    def compare(self) -> pd.DataFrame:
        """One row per finished run: its test metrics, raw and post-processed."""
        from ethograph.segment.train import compare_runs

        return compare_runs(self._config.runs_dir)

    def load_run(self, run: str | Path | None = None):
        """A trained run, ready to predict with (see :meth:`infer` for the usual path)."""
        from ethograph.segment.inference import load_run, resolve_run_dir

        return load_run(resolve_run_dir(self._config, run))

    # ------------------------------------------------------------------
    # Choosing video features
    # ------------------------------------------------------------------

    def rank_video_features(self, min_frames: int = 0) -> tuple[FeatureRanking, list[str]]:
        """Rank the materialised video-feature columns by Cohen's d.

        Reads the materialised dataset, so it ranks exactly the columns a
        model would see, and costs no re-extraction. Returns the ranking and
        the column names it indexes, so ``names[i] for i in ranking.top(20)``
        names the twenty most behaviour-discriminating video dimensions.

        Raises ``ValueError`` when no column declares
        ``kind="video_feature"`` — that is what tells this apart from the
        kinematic columns (see :mod:`ethograph.io.schema`).
        """
        from ethograph.io.schema import VIDEO_FEATURE
        from ethograph.segment.dataset import MaterialisedStore
        from ethograph.video_features.select import rank_features

        store = MaterialisedStore.open(self._config.data_dir)
        columns = _video_columns(store.layout)
        if not columns:
            raise ValueError(
                f"No column of {self._config.data_dir} declares kind={VIDEO_FEATURE!r}. "
                "Describe the video features when you build the session "
                "(ethograph.io.schema.describe), then materialise again."
            )
        trials = []
        for key in store.keys:
            x, y = store.load(key)
            trials.append((x[columns].T, y))
        ranking = rank_features(trials, min_frames=min_frames)
        return ranking, [store.layout.names[i] for i in columns]


def _video_columns(layout: ColumnLayout) -> list[int]:
    from ethograph.io.schema import VIDEO_FEATURE

    return [i for i, kind in enumerate(layout.kinds) if kind == VIDEO_FEATURE]


def architectures() -> list[str]:
    """Every registered architecture name."""
    from ethograph.segment.models import available_architectures

    return available_architectures()


def tunable_params(architecture: str) -> dict[str, Any]:
    """What ``model.params`` accepts for *architecture*, and each key's default.

    The architectures share almost no hyperparameter names — ``mlp`` takes
    ``f_maps_list``, ``mstcn`` takes ``num_f_maps`` — so a benchmark over
    several of them needs a search space per architecture, and this is where
    the names come from::

        for name in eto.segment.architectures():
            print(name, sorted(eto.segment.tunable_params(name)))

    A setting upstream leaves required — ``motionbert``'s ``num_joints`` — has
    no default and so is not listed; the builder names it if it is missing.
    """
    from ethograph.segment.models import DEFAULTS_FILES, available_architectures, skeleton_graph
    from ethograph.segment.models.vendored import tunable_params as _tunable

    available_architectures()  # registers the built-ins
    if architecture in DEFAULTS_FILES:
        return yaml.safe_load(DEFAULTS_FILES[architecture].read_text(encoding="utf-8")) or {}
    if architecture in skeleton_graph._DEFAULTS_FILE:
        return skeleton_graph.tunable_params(architecture)
    return _tunable(architecture)


def extract_videos(
    videos: Iterable[str | Path],
    out_dir: str | Path,
    stack_s: float | None = None,
    analysis_fps: float | None = None,
    overwrite: bool = False,
    include: Iterable[str] | None = None,
    **kwargs: Any,
) -> list[Path]:
    """An extractor over a bare folder of videos, with no project — videos in, sidecars out.

    The scripted counterpart of :meth:`Project.video_features` for footage
    that is not yet part of a session. Defaults are
    :class:`~ethograph.segment.config.VideoFeaturesConfig`'s — the ``s3d``
    extractor with its 0.5 s window — and any field of it may be passed as a
    keyword (``stack_s=0.3``; ``extractor="timm", model_name=...``).

    *include* is a list of regular expressions; only videos whose path
    matches one are extracted, which is how you take one camera out of a
    folder holding several::

        eto.segment.extract_videos(["/data/videos"], "/data/features", include=["cam-1"])
    """
    from ethograph.segment.config import VideoFeaturesConfig
    from ethograph.segment.video_features import extract_videos as _extract

    fields: dict[str, Any] = {"analysis_fps": analysis_fps, "stack_s": stack_s, **kwargs}
    return _extract(videos, out_dir, VideoFeaturesConfig(**fields), overwrite=overwrite, include=include)
