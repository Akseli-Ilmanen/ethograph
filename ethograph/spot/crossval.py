"""Cross-validation: one fold per session, each predicting the session it held out.

The fold's split is written the way the segmentation pipeline writes it —
``train.split.holdout_sessions`` — so "held out" means the same thing in both
(:mod:`ethograph.segment.crossval`). Every fold is a project of its own under
``cross_validation/{session}/``: the same frames (never decoded twice), its own
``dataset/`` with the held-out session as the whole test split, its own
``runs/fold_{session}``. A fold ends by scoring that test split
(``test_metrics.yaml`` — the trained-on-the-others number) and writing its
predictions into the held-out session's ``labels/``, so what the GUI opens was
never trained on.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from ethograph.spot.config import SpotConfig
from ethograph.spot.project import Project, RunResult

logger = logging.getLogger(__name__)


def cross_validate(
    config: SpotConfig, sessions: Iterable[str | Path] | None = None, workers: int | None = None
) -> list[RunResult]:
    """One fold per session of *config* (or per session *sessions* names)."""
    folds: list[RunResult] = []
    specs = config.select_sessions(sessions)
    for spec in specs:
        stem = spec.label
        split = replace(config.train.split, holdout_sessions=[spec.source])
        train = replace(config.train, split=split, run_name=f"fold_{stem}")
        fold_cfg = replace(config, train=train, root=config.cross_validation_dir / stem, frames=config.frames_dir)
        fold = Project(fold_cfg)
        logger.info("Fold %d/%d: holding out %s", len(folds) + 1, len(specs), stem)
        fold.materialise(workers=workers)
        result = fold.train()
        fold.evaluate(run=result.run_dir)
        fold.inference(run=result.run_dir, sessions=[spec.source], workers=workers)
        folds.append(result)
    return folds
