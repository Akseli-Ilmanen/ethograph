"""Pretrained video features for action segmentation.

One registry of **extractors** (:data:`EXTRACTORS`, built by name with
:func:`build_extractor`), every one writing the same sidecar — a
``(time_video, {name}_dims)`` DataArray on the video's own clock:

* ``s3d`` — clip-wise: each frame's feature is the Kinetics-400 S3D
  embedding of the ``stack_s`` window centred on it (:class:`S3DConfig`,
  :func:`plan_s3d`, :func:`extract_s3d`).
* ``timm`` — frame-wise: any ``timm`` image backbone, DINOv2 by default,
  embedding each frame on its own (``timm_extract``; needs ``ethograph[timm]``).

Configured in seconds, resolved against the video's own rate, extracted
streaming. :mod:`ethograph.segment.video_features` runs an extractor over a
project's videos and merges the sidecars onto the trial clock.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ethograph.video_features.base import (
    EXTRACTORS,
    TIME_DIM,
    CropBox,
    Extractor,
    build_extractor,
    check_extractor_name,
    extractor_module,
    feature_dim,
    feature_dim_of,
    sidecar_path,
    time_dim_of,
)
from ethograph.video_features.plan import MIN_STACK, FramePlan, S3DConfig, S3DPlan, plan_frames, plan_s3d
from ethograph.video_features.select import FeatureRanking, cohens_d, rank_features

if TYPE_CHECKING:
    from ethograph.video_features.extract import extract_s3d
    from ethograph.video_features.s3d import FULL_STAGE, S3D_STAGES

# The torch-backed extractors load on first use, so configuring a project never needs torch.
_LAZY = {"extract_s3d": "extract", "FULL_STAGE": "s3d", "S3D_STAGES": "s3d"}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"ethograph.video_features.{_LAZY[name]}"), name)


__all__ = [
    "EXTRACTORS",
    "FULL_STAGE",
    "CropBox",
    "Extractor",
    "FeatureRanking",
    "FramePlan",
    "MIN_STACK",
    "S3DConfig",
    "S3DPlan",
    "S3D_STAGES",
    "TIME_DIM",
    "build_extractor",
    "check_extractor_name",
    "cohens_d",
    "extract_s3d",
    "extractor_module",
    "feature_dim",
    "feature_dim_of",
    "plan_frames",
    "plan_s3d",
    "rank_features",
    "sidecar_path",
    "time_dim_of",
]
