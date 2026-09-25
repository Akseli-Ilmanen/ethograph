"""The one entry point: a config plus a method per stage.

Modelled on :class:`ethograph.segment.project.Project`, deliberately — the two
pipelines differ in what they read, not in how a run is expressed.

    project = eto.spot.Project("spot.yaml", "clip.context_s=4")
    project.materialise()
    project.train()
    project.inference()
    project.cross_validate()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ethograph.segment.sessions import open_session
from ethograph.spot import dataset as dataset_stage
from ethograph.spot.config import (
    ResolvedClip,
    SpotConfig,
    config_to_dict,
    load_config,
    save_config,
)
from ethograph.spot.vendored import (
    check_vram,
    describe_architecture,
    feature_architectures,
    run_with_retries,
    script_command,
)

logger = logging.getLogger(__name__)

CONFIG_FILE = "config.yaml"


@dataclass(frozen=True)
class RunResult:
    """Where a finished run's outputs are."""

    name: str
    run_dir: Path
    clip: ResolvedClip


def _dataset_fps(config: SpotConfig) -> float:
    """The one frame rate the export produced, refusing a mixed set.

    Every temporal setting is a duration resolved against this, so two rates
    in one project would mean two different clip lengths for one config — a
    silent inconsistency rather than an error.
    """
    import pandas as pd

    index_path = config.dataset_dir / dataset_stage.INDEX_FILE
    if not index_path.is_file():
        raise FileNotFoundError(f"No exported dataset at {config.dataset_dir} — run project.materialise() first")
    rates = sorted(set(pd.read_csv(index_path, sep="\t")["fps"].round(6)))
    if not rates:
        raise ValueError(f"{index_path} lists no trials")
    if len(rates) > 1:
        raise ValueError(
            f"The exported trials have {len(rates)} different frame rates ({rates}). "
            "clip.context_s and clip.resolution_ms resolve against one rate; split the project, "
            "or restrict trials.where until one rate remains."
        )
    return float(rates[0])


class Project:
    """A pixel event-spotting project: one config, a method per stage."""

    def __init__(self, config: str | Path | SpotConfig, *overrides: str) -> None:
        if isinstance(config, SpotConfig):
            if overrides:
                raise ValueError("Pass overrides with a config path, not with an already-built SpotConfig")
            self._config = config
        else:
            self._config = load_config(config, list(overrides))

    def __repr__(self) -> str:
        cfg = self._config
        where = cfg.config_path.name if cfg.config_path else "<in memory>"
        return f"Project({where}, {len(cfg.sessions)} sessions, root={cfg.root})"

    @property
    def config(self) -> SpotConfig:
        return self._config

    @property
    def root(self) -> Path:
        return self._config.root

    def update(self, *overrides: str) -> Project:
        """A new project with dotted ``key=value`` overrides applied."""
        if self._config.config_path is None:
            raise ValueError("This project was built from an in-memory config; overrides need a config file")
        return Project(self._config.config_path, *overrides)

    # -- stages ----------------------------------------------------------

    def materialise(self, workers: int | None = None, sessions: Iterable[str | Path] | None = None) -> Path:
        """Stage 1: sessions to frames plus E2E-Spot's own index.

        With ``graph:`` set, the same trials' entity graphs land under
        ``keypoints/`` — one split, one set of trials, two modalities.
        """
        specs = self._config.select_sessions(sessions)
        result = dataset_stage.materialise(self._config, workers=workers, sessions=specs)
        if self._config.features:
            from ethograph.spot.features import export_features

            export_features(self._config, result.sessions, result.records)
        return result.dataset_dir

    def fill_poses(
        self,
        backend: str = "spline",
        sessions: Iterable[str | Path] | None = None,
        overwrite: bool = False,
        checkpoint: str | Path | None = None,
    ) -> list[Path]:
        """Stage 0a: every labelled clip's sidecar, filled and exported headless.

        The keypoint-labelling dialog leaves ``<video>.keypoints.json`` beside
        each clip you labelled; this runs the fill backend over all of them
        and writes ``<video>.keypoints.nc`` — the dialog's own export, for
        hundreds of clips at once. Clips never labelled are skipped.
        """
        from ethograph.spot.pose_batch import fill_and_export_session

        written: list[Path] = []
        for spec in self._config.select_sessions(sessions):
            session = open_session(spec)
            written += fill_and_export_session(
                session, self._config, backend, overwrite=overwrite, checkpoint=checkpoint
            )
        return written

    def merge_poses(
        self,
        var: str = "position",
        sessions: Iterable[str | Path] | None = None,
        in_place: bool = False,
    ) -> list[Path]:
        """Stage 0b: each clip's keypoints onto its trial's clock, into the session.

        Writes a sibling ``{stem}_pose2d.nc`` per session (or the session
        itself with ``in_place``) carrying *var* as ``(time, space, keypoint,
        individual)`` — what ``graph.feature`` then names. Anything that
        writes ``<video>.keypoints.nc`` (a tracker, the dialog) merges alike.
        """
        from ethograph.spot.pose_batch import merge_keypoints

        return [
            merge_keypoints(open_session(spec), self._config, var=var, in_place=in_place)
            for spec in self._config.select_sessions(sessions)
        ]

    def resolved_clip(self) -> ResolvedClip:
        """The frame counts this config's durations imply for the exported rate."""
        return self._config.resolve_clip(_dataset_fps(self._config))

    def run_name(self) -> str:
        """The run's name: ``train.run_name``, else the clip's durations, ``_features`` when the pose is fed in."""
        cfg = self._config
        if cfg.train.run_name:
            return cfg.train.run_name
        clip = self.resolved_clip()
        name = f"ctx{clip.context_s:g}s_res{clip.resolution_ms:g}ms"
        return f"{name}_features" if cfg.fusing else name

    def train(self) -> RunResult:
        """Stage 2: upstream's training loop, driven by the resolved config.

        With ``features:`` listed the block
        is exported first (:func:`~ethograph.spot.features.export_block`) and
        handed to the trainer beside the frames.
        """
        cfg = self._config
        known = architectures()
        if cfg.model.architecture not in known:
            raise ValueError(
                f"model.architecture={cfg.model.architecture!r} is not one the vendored trainer accepts. "
                f"eto.spot.architectures() lists them: {known}"
            )
        clip = self.resolved_clip()
        check_vram(clip.frames_per_batch)
        name = self.run_name()
        run_dir = cfg.run_dir(name)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / CONFIG_FILE)
        logger.info(
            "Run %s: %.2f s context, %.1f ms resolution -> stride %d, clip_len %d, dilate_len %d at %g fps",
            name,
            clip.context_s,
            clip.resolution_ms,
            clip.stride,
            clip.clip_len,
            clip.dilate_len,
            clip.fps,
        )
        command = self._train_command(
            clip,
            run_dir,
            epochs=cfg.train.epochs,
            epoch_frames=cfg.train.epoch_frames,
            learning_rate=cfg.train.learning_rate,
            criterion="map",
        ) + self._block_flags(export=True)
        run_with_retries(command, run_dir, cfg.train.retries)
        return RunResult(name=name, run_dir=run_dir, clip=clip)

    def _block_flags(self, export: bool = False) -> list[str]:
        """The trainer's feature-block flags, empty unless fusing; *export* rebuilds ``features/block/``."""
        cfg = self._config
        if not cfg.fusing:
            return []
        from ethograph.spot.features import block_dim, export_block

        if export:
            export_block(cfg)
        return [
            "--fuse_dir",
            str(cfg.block_dir.resolve()),
            "--fuse_dim",
            str(block_dim(cfg)),
            "--fuse_dropout",
            str(cfg.train.features_dropout),
        ]

    def cross_validate(
        self, sessions: Iterable[str | Path] | None = None, workers: int | None = None
    ) -> list[RunResult]:
        """Stage 3: one fold per session, each predicting the session it held out.

        Every fold is a project of its own under ``cross_validation/{session}/``
        (see :mod:`ethograph.spot.crossval`), so what the GUI opens was never
        trained on.
        """
        from ethograph.spot.crossval import cross_validate

        return cross_validate(self._config, sessions=sessions, workers=workers)

    def inference(
        self, run: str | Path | None = None, sessions: Iterable[str | Path] | None = None, workers: int | None = None
    ) -> list[Path]:
        """Stage 4: a run's predictions into each session's ``labels/`` folder.

        *run* is a name under ``runs/``, a path, or ``None`` for the newest.
        Every trial with video is predicted, labelled or not; the epoch is the
        one the sweep ranks first on the run's own validation predictions.
        Returns the labels TSVs written, one per session.
        """
        from ethograph.spot.inference import inference

        return inference(self._config, run=run, sessions=sessions, workers=workers)

    def evaluate(
        self,
        run: str | Path | None = None,
        split: str = "test",
        epoch: int | None = None,
        zero_features: bool = False,
    ) -> dict:
        """The test summary: a run's chosen epoch scored on ``dataset/{split}.json``.

        Per class — labelled events, misses, spurious predictions, mean and
        median error in ms, and the hit rate at each of
        :data:`~ethograph.spot.metrics.TOLERANCES_MS`. Written to the run's
        ``test_metrics.yaml`` and returned. *run* as for :meth:`inference`;
        *epoch* defaults to the one inference would use. *zero_features*
        scores a run that reads ``features:`` with them zeroed
        (``test_metrics_nofeatures.yaml``) — the difference to the plain score
        is what the pose contributes.
        """
        from ethograph.spot.inference import resolve_run_dir
        from ethograph.spot.metrics import evaluate_run

        run_dir = resolve_run_dir(self._config, run)
        return evaluate_run(self._config, run_dir, split=split, epoch=epoch, zero_features=zero_features)

    def compare(self) -> pd.DataFrame:
        """Every scored run side by side — one row per ``test_metrics.yaml`` — written to ``runs/compare.tsv``."""
        from ethograph.spot.metrics import compare_runs

        return compare_runs(self._config)

    def runs(self) -> list[str]:
        """Names of the runs under ``runs/``, newest last."""
        root = self._config.runs_dir
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir() if (p / CONFIG_FILE).is_file())

    # -- internals -------------------------------------------------------

    def _train_command(
        self,
        clip: ResolvedClip,
        run_dir: Path,
        *,
        epochs: int,
        epoch_frames: int,
        learning_rate: float,
        criterion: str,
    ) -> list[str]:
        """Upstream's CLI, spelled from the resolved config.

        ``train_e2e.py`` builds its paths as ``os.path.join('data', dataset)``,
        and an absolute second argument wins — so the dataset directory is
        passed as-is and nothing needs a ``data/`` folder.
        """
        cfg = self._config
        if epochs <= cfg.train.warm_up_epochs:
            # Upstream's schedule is linear warm-up then cosine over the rest;
            # with nothing left for the cosine it divides by zero mid-epoch.
            raise ValueError(
                f"{epochs} epoch(s) with train.warm_up_epochs={cfg.train.warm_up_epochs}: the cosine schedule "
                "needs at least one epoch after the warm-up. Raise the epochs or lower train.warm_up_epochs."
            )
        command = script_command("train_e2e") + [
            str(cfg.dataset_dir.resolve()),
            str(cfg.frames_dir.resolve()),
            "-s",
            str(run_dir.resolve()),
            "-m",
            cfg.model.architecture,
            "-t",
            cfg.model.head,
            "--num_epochs",
            str(epochs),
            "--clip_len",
            str(clip.clip_len),
            "--stride",
            str(clip.stride),
            "--dilate_len",
            str(clip.dilate_len),
            "--batch_size",
            str(cfg.train.batch_size),
            "-ag",
            str(cfg.train.acc_grad),
            "--learning_rate",
            str(learning_rate),
            "--warm_up_epochs",
            str(cfg.train.warm_up_epochs),
            "--start_val_epoch",
            str(cfg.train.start_val_epoch),
            "--epoch_num_frames",
            str(epoch_frames),
            "--criterion",
            criterion,
        ]
        if cfg.model.multiscale:
            dilations = cfg.model.shift_dilations(clip.fps / clip.stride)
            command += [
                "--shift_dilations",
                ",".join(str(d) for d in dilations),
                "--attention_groups",
                str(cfg.model.attention_groups),
            ]
        return command


def architectures() -> list[str]:
    """Every architecture ``model.architecture`` may name, from the vendored trainer's own CLI.

    ``eto.spot.describe_architecture(name)`` says what each one is::

        for name in eto.spot.architectures():
            print(name, "-", eto.spot.describe_architecture(name))
    """
    return feature_architectures()


__all__ = ["Project", "RunResult", "architectures", "config_to_dict", "describe_architecture"]
