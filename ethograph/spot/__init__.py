"""Precise event spotting from pixels — point events learned from video.

Where :mod:`ethograph.segment` learns state events from feature columns you
choose, this learns a single moment per trial from the frames themselves. The
model is E2E-Spot (Hong et al., ECCV 2022), vendored the way DLC2Action is.

    import ethograph as eto

    project = eto.spot.Project("spot.yaml")
    project.materialise()            # sessions -> frames + the model's own index
    project.train()             # one run under runs/
    project.cross_validate()    # one fold per session

Sessions, the trial filter and the split are the segmentation pipeline's,
imported unchanged — combining sessions into one training set is a property of
how the data is organised, not of the model. What is this pipeline's own is
how video becomes clips, and there **every temporal setting is a duration**
resolved against the video's own rate (:class:`~ethograph.spot.config.ClipConfig`).

Docs: ``docs/source/models/spot/index.md``.
"""

from __future__ import annotations

from ethograph.spot.confidence import CurveStats, curve_stats, densify
from ethograph.spot.config import (
    ClipConfig,
    CropConfig,
    InferConfig,
    LabelsConfig,
    ModelConfig,
    ResolvedClip,
    SpotConfig,
    TrainConfig,
    config_from_dict,
    load_config,
    save_config,
)
from ethograph.spot.dataset import TrialRecord, materialise
from ethograph.spot.features import export_block, export_features
from ethograph.spot.pose_batch import fill_and_export_video, merge_keypoints
from ethograph.spot.predict import SpottedEvent, read_predictions, spot_entry
from ethograph.spot.project import Project, RunResult, architectures
from ethograph.spot.vendored import describe_architecture
from ethograph.utils.logging import enable_console_logging

# Importing the pipeline turns on INFO logging, as `ethograph.segment` does:
# every stage's progress is meant to be visible without configuring anything.
enable_console_logging(__name__)

__all__ = [
    "ClipConfig",
    "CropConfig",
    "CurveStats",
    "InferConfig",
    "LabelsConfig",
    "ModelConfig",
    "ResolvedClip",
    "RunResult",
    "SpotConfig",
    "Project",
    "SpottedEvent",
    "TrainConfig",
    "TrialRecord",
    "architectures",
    "describe_architecture",
    "config_from_dict",
    "curve_stats",
    "densify",
    "export_block",
    "export_features",
    "fill_and_export_video",
    "materialise",
    "load_config",
    "merge_keypoints",
    "read_predictions",
    "save_config",
    "spot_entry",
]
