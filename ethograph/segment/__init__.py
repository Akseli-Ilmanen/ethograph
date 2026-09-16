"""Segmentation pipeline — learn state events from trial-structured sessions.

Everything is scripted: one YAML config becomes a
:class:`~ethograph.segment.project.Project`, and each stage is a method on
it. There is no command line, so there is exactly one way to express a run,
and that way is a file you can diff and re-run.

    import ethograph as eto

    project = eto.segment.Project("project.yaml")
    project.video_features(merge=True)   # video features once per video, merged into the sessions
    project.materialise()                # feature engineering → the materialised dataset

    best = project.search()              # stage 1: Optuna on the 60/20/20 split
    eto.segment.Project(best.config_path).cross_validate()   # stage 2: one fold per session

The workflow has two stages and they use the trials differently. **Search**
splits trials 60/20/20 and lets Optuna maximise the validation score — that
is the only thing validation is for. **Cross-validation** then takes the
settings it found and holds out one whole *session* per fold, writing a
prediction set beside each held-out session so you can open it in the GUI
against the curated labels and see where the model is still wrong.

The vocabulary (session, trial, sample, feature column, materialised dataset,
architecture, run, role, prediction set) is defined in the repository's
``CONTEXT.md``; the design is documented in ``docs/source/advanced/segment/``.
"""

from ethograph.segment.config import SegmentConfig, as_overrides, load_config
from ethograph.segment.feral import export_feral
from ethograph.segment.project import Project, architectures, extract_videos, tunable_params
from ethograph.segment.sessions import (
    discover_columns,
    discover_columns_from_source,
    feature_sampling_rates,
    feature_sampling_rates_from_source,
)
from ethograph.utils.logging import enable_console_logging

enable_console_logging()

__all__ = [
    "Project",
    "SegmentConfig",
    "architectures",
    "as_overrides",
    "discover_columns",
    "discover_columns_from_source",
    "export_feral",
    "extract_videos",
    "feature_sampling_rates",
    "feature_sampling_rates_from_source",
    "load_config",
    "tunable_params",
]
