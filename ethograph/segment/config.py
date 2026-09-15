"""One YAML per project: sessions, features, model, train, infer.

A config is a plain YAML file built into typed dataclasses. Two conveniences
and nothing more:

* ``base: other.yaml`` — the file is deep-merged *over* ``other.yaml``
  (relative to itself), so a benchmark is a base file plus small overrides.
* dotlist overrides — ``Project("cfg.yaml", "model.architecture=mstcn",
  "train.loss.focal_gamma=2")`` — values are parsed as YAML.

Relative paths resolve against the config file's folder. Unknown keys are an
error: a typo must not silently become a default. Derived values (column
layout, normalisation statistics, class table) are never part of the config;
they are outputs written into the run directory.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Iterable

import yaml

from ethograph.labels.tsv_store import labels_tsv_path
from ethograph.utils.paths import defaults_dir, ethograph_home
from ethograph.video_features.base import CropBox, Extractor, check_extractor_name, extractor_module

logger = logging.getLogger(__name__)

DLC2ACTION_CONFIG = Path(__file__).parent / "dlc2action" / "config"
"""The vendored DLC2Action config tree (see ``dlc2action/NOTICE.md``).

Reached by path, not by import: importing the package here would re-enter
``ethograph.__init__``'s lazy loader while this module is still being built.
"""

TRAINING_CONFIG = DLC2ACTION_CONFIG / "training.yaml"
"""Upstream's ``dlc2action/config/training.yaml``.

Unlike ``config/model/*.yaml`` and ``config/losses.yaml``, this file has no
constructor to feed: it configures DLC2Action's *own* training loop, and we
run ours. Only the settings that carry over unchanged are read from it — see
:func:`upstream_training_default`.
"""


GUI_SETTINGS_FILENAME = "gui_settings.yaml"
"""The GUI's global settings file, under :func:`~ethograph.utils.paths.ethograph_home`."""

GUI_POSTPROCESS_KEYS: dict[str, tuple[str, Any]] = {
    "min_duration_s": ("cp_min_label_length_s", 0.05),
    "label_thresholds": ("cp_label_thresholds", {}),
    "stitch_gap_s": ("cp_stitch_gap_len_s", 0.015),
    "max_expansion_s": ("cp_max_expansion_s", 0.05),
    "max_shrink_s": ("cp_max_shrink_s", 0.05),
}
"""``PostprocessConfig`` field → (``gui_settings.yaml`` key, the GUI's default).

The GUI's changepoint-correction section and ``infer.postprocess`` are the
same four steps under different names; this is the translation, and the one
place it is written. The defaults are the GUI's own (``AppStateSpec.VARS``)
for a key the file has not saved yet — covered by
``tests/test_unit/test_segment_gui_postprocess.py``, which checks both
against the spec.
"""

GUI_POSTPROCESS_STEPS: dict[str, tuple[str, bool]] = {
    "cp_step_purge": ("cp_step_purge", True),
    "cp_step_stitch": ("cp_step_stitch", True),
    "cp_step_snap": ("cp_step_snap", True),
    "cp_step_purge_after": ("cp_step_purge_after", True),
}
"""The GUI's step checkboxes. The pipeline derives its steps from the values
(``postprocess.py``), so an unticked step reads as its parameter zeroed:
purge (either box) off → ``min_duration_s = 0`` and no thresholds, stitch off
→ ``stitch_gap_s = 0``, snap off → ``changepoint_correction: false``."""


def gui_settings_path(value: str | bool, base_dir: Path) -> Path:
    """Where ``infer.postprocess.gui_settings`` points: ``true`` = the ethograph home's file."""
    if value is True:
        return ethograph_home() / GUI_SETTINGS_FILENAME
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def read_gui_postprocess(path: Path) -> dict[str, Any]:
    """The ``infer.postprocess`` values the GUI's ``gui_settings.yaml`` at *path* expresses.

    Only the correction keys (:data:`GUI_POSTPROCESS_KEYS` + the step boxes);
    the ``changepoints`` selection has no GUI counterpart and is left to the
    config. Missing: a config that asks for a GUI file that is not there is an
    error, not a silent default.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"infer.postprocess.gui_settings points at {path}, which does not exist — "
            "open the GUI once (it writes the file) or spell the postprocess values in the config"
        )
    saved = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(saved, dict):
        raise ValueError(f"{path} is not a mapping")
    values = {field_name: saved.get(key, default) for field_name, (key, default) in GUI_POSTPROCESS_KEYS.items()}
    steps = {name: bool(saved.get(key, default)) for name, (key, default) in GUI_POSTPROCESS_STEPS.items()}
    if not (steps["cp_step_purge"] or steps["cp_step_purge_after"]):
        values["min_duration_s"] = 0.0
        values["label_thresholds"] = {}
    if not steps["cp_step_stitch"]:
        values["stitch_gap_s"] = 0.0
    values["changepoint_correction"] = steps["cp_step_snap"]
    return values


def upstream_training_defaults() -> dict[str, Any]:
    """DLC2Action's own training defaults, straight from ``config/training.yaml``."""
    if not TRAINING_CONFIG.is_file():
        raise FileNotFoundError(f"No vendored DLC2Action training config at {TRAINING_CONFIG}")
    return yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8")) or {}


def upstream_training_default(key: str, cast: Any) -> Any:
    """One value from that file, coerced.

    The coercion is not cosmetic: YAML 1.1 reads ``lr: 1e-3`` as the *string*
    ``"1e-3"``, which would silently reach the optimizer.
    """
    defaults = upstream_training_defaults()
    if key not in defaults:
        raise KeyError(f"{TRAINING_CONFIG} has no {key!r}; found {sorted(defaults)}")
    return cast(defaults[key])


ROLES = ("train", "val", "test")
"""The three roles a *sample* can have inside a run.

A role is never declared per session in the config file. It is drawn by
:class:`SplitConfig` from ``train_fraction`` / ``val_fraction`` /
``test_fraction`` (stage 1 — the ratios you tune a search against), or pinned
whole-session by ``train.split.holdout_sessions`` (stage 2 — the
cross-validation folds that
:meth:`ethograph.segment.project.Project.cross_validate` writes per fold).
"""

#: The variable :func:`~ethograph.features.changepoints.merge_changepoints`
#: writes, and therefore the stem of the columns a merged expansion generates.
MERGED_CHANGEPOINTS = "changepoints"


@dataclass
class SessionSpec:
    """One session: its source file, and where its labels and video live.

    ``labels_path`` defaults to ``{stem}_labels.tsv`` beside ``source`` (the
    GUI's own convention, :func:`~ethograph.labels.tsv_store.labels_tsv_path`)
    when left unset — :func:`config_from_dict` fills it in and logs what it
    assumed. Nothing ever creates that file: a session's ``_labels.tsv`` is
    the user's curated labels, and only the user (or the GUI, on an explicit
    save/curate) should ever write it. A ``labels_path`` naming a file that
    does not exist yet just means this session has none.
    ``video_dir``, when the session has video,
    is the one folder searched for it (no project-level list to fall
    through).

    A session carries no role. Every listed session is material for the
    model; which of its *trials* end up train / val / test is drawn by
    ``train.split``, and holding a whole session out is a cross-validation
    fold (``train.split.holdout_sessions``), not something you write per
    session in the config.

    ``alignment`` replaces the trials the source's own ``.ethograph/alignment.nwb``
    would give: the same file listed twice, once with its behaviour trials
    and once with windows tiled over a sleep epoch
    (:func:`~ethograph.segment.windows.write_windows_alignment`), is two
    sessions of one recording. Give the second a ``name``.
    """

    source: Path
    labels_path: Path | None = None
    video_dir: Path | None = None
    #: An alignment NWB to read the trials from instead of the sidecar the
    #: source would find on its own.
    alignment: Path | None = None
    #: How the session is referred to in outputs — fold names, prediction
    #: keys, log lines. Defaults to the source's stem, which is fine until
    #: every session's file is called ``Trial_data.nc``.
    name: str | None = None
    #: Video features from files, ``{variable: folder}``: one ``{video stem}.npy``
    #: per trial's camera file, attached in memory when the session opens
    #: (:mod:`ethograph.io.video_feature_files`), so ``features.columns`` can
    #: select ``feral: {feral_dims: 0..767}`` like any variable of the file.
    video_feature_folders: dict[str, Path] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        return self.source.name if self.source.is_dir() else self.source.stem


def name_colliding_sessions(specs: list[SessionSpec]) -> None:
    """Sessions whose default labels collide get a ``name`` from the folder that tells them apart.

    ``Trial_data3.nc`` in every session's ``behav/`` folder is the normal
    layout, not a mistake, so the nearest ancestor whose name differs within
    the group (``ses-000_date-20250309_01``) is prefixed to the stem. A
    session named explicitly is never renamed — two explicit names that
    collide are refused, as is a source listed twice, which no folder can
    tell apart. Two sessions with one label would otherwise write their
    trials under the same ids and silently overwrite each other.
    """
    groups: dict[str, list[SessionSpec]] = {}
    for spec in specs:
        groups.setdefault(spec.label, []).append(spec)
    taken = set(groups)
    for label, group in groups.items():
        if len(group) < 2:
            continue
        if any(s.name for s in group):
            raise ValueError(f"config.sessions: {label!r} names more than one session — give each a distinct `name:`")
        paths = [s.source.resolve() for s in group]
        for depth in range(min(len(p.parents) for p in paths)):
            names = [f"{p.parents[depth].name}_{s.label}" for p, s in zip(paths, group)]
            if len(set(names)) == len(names) and not taken.intersection(names):
                for spec, name in zip(group, names):
                    spec.name = name
                taken.update(names)
                logger.info(
                    "config.sessions: %d sessions are called %r; named by their folders: %s", len(group), label, names
                )
                break
        else:
            raise ValueError(f"config.sessions: {label!r} is listed more than once from the same place: {paths}")


@dataclass
class TrialsConfig:
    """Trial filter applied in every stage: metadata column → allowed values."""

    where: dict[str, list[Any]] = field(default_factory=dict)
    #: Keep only the first N trials that pass ``where`` (in session order) —
    #: a smoke run on a few trials before committing a night of GPU.
    #: ``None`` = all of them.
    limit: int | None = None


@dataclass
class PreprocessConfig:
    """The fixed preprocessing chain, in the order it runs."""

    #: Frames whose keypoint confidence is below this become NaN (then interpolated).
    likelihood_threshold: float | None = None
    #: The per-keypoint confidence feature used by ``likelihood_threshold``.
    likelihood_feature: str = "confidence"
    interpolate: bool = True
    clip_percentiles: tuple[float, float] | None = (2.0, 98.0)
    #: Z-score with statistics computed over the *training* samples of a run.
    zscore: bool = True
    #: Features never z-scored (in addition to any carrying ``attrs["normalise"] = 0``).
    zscore_exclude: list[str] = field(default_factory=list)


TARGET_SUBJECTS = ("self", "all")
"""``features.labels.subjects``: whose labels are targets — the sample's own
individual, or every individual of the cast (``self``, ``other1``, …)."""


@dataclass
class LabelsConfig:
    """Which labels are the targets: one branch of a ``mapping.txt``, or several.

    One ``branch`` is the exclusive case — one class per frame, softmax.
    Listing ``branches`` (or asking for every subject's labels with
    ``subjects: all``) makes the target **multi-label**: one binary channel
    per (subject, class), a sigmoid each, so labels of different branches —
    or of different animals — coexist on a frame. Within one (subject,
    branch) *track* classes stay exclusive, exactly as the GUI draws them.
    """

    #: ``mapping.txt`` path; ``None`` defaults to ``~/.ethograph/defaults/mapping.txt``.
    mapping: Path | None = None
    branch: int = 0
    #: Several branches at once → a multi-label target. Spell either this or
    #: ``branch``, never both.
    branches: list[int] | None = None
    #: Label ids to predict; ``None`` = every state class of the branch(es).
    classes: list[int] | None = None
    #: ``self``: the sample's individual only. ``all``: also ``other1``,
    #: ``other2``, … in layout order — a multi-label target.
    subjects: str = "self"

    def __post_init__(self) -> None:
        if self.subjects not in TARGET_SUBJECTS:
            raise ValueError(f"features.labels.subjects={self.subjects!r} — one of {list(TARGET_SUBJECTS)}")
        if self.branches is not None:
            if not self.branches:
                raise ValueError("features.labels.branches is empty — list at least one branch, or spell branch:")
            if self.branch != 0:
                raise ValueError(
                    f"features.labels spells both branch={self.branch} and branches={self.branches} — use one"
                )
            self.branches = [int(b) for b in self.branches]

    @property
    def branch_list(self) -> list[int]:
        """The branches whose state classes are targets."""
        return list(self.branches) if self.branches is not None else [int(self.branch)]

    @property
    def multilabel(self) -> bool:
        """Whether the target is one binary channel per (subject, class)."""
        return self.branches is not None or self.subjects == "all"


@dataclass
class ChangepointFeaturesConfig:
    """Expand named changepoint masks into :func:`~ethograph.features.changepoints.more_changepoint_features`.

    Applied once per session, at ``open_session`` time (materialise and
    infer both go through it): each ``inputs`` entry names a raw changepoint
    mask and pins its dims exactly like a ``features.columns`` entry, and
    ``transforms`` picks which of the four column groups
    (:data:`~ethograph.features.changepoints.CP_TRANSFORMS`) to keep. The
    generated columns are merged straight into ``features.columns`` at
    config-load time (see :func:`config_from_dict`), so nothing downstream
    needs to know this section exists, and you never spell out
    ``{var}_cp_sigma2_weighted`` yourself.

    This is the one exception to "features are built with the session,
    never by the pipeline": it is a deterministic expansion of a mask
    already in the file, not a new modelling choice, so there is nothing to
    decide beyond which columns to keep — see
    ``examples/segment_changepoint_features.ipynb`` for what each one looks
    like on real data before turning it on here.

    **The temporal scales are read off the labels unless spelled.** Leave
    ``sigmas``, ``horizon`` and ``max_length`` out and ``materialise``
    derives them from the durations of the curated state events
    (:func:`~ethograph.features.changepoints.scales_from_durations`),
    records them in the dataset's ``columns.yaml`` with a ``note`` saying
    what happened, and every later stage reads them back from there, so a
    session predicted later — labelled or not — is expanded at exactly the
    training scale. A config with any of the three unset is *unresolved*
    (:attr:`unresolved`) until then; spell a value, in samples, to pin it.
    """

    #: Kernel widths of the proximity columns, in samples; ``None`` derives
    #: the ladder ``horizon / (16, 8, 4)``. The columns are named by rank
    #: (``_cp_prox0``…), so the names never depend on the values.
    sigmas: list[float] | None = None
    distribution: str = "laplacian"
    #: OR every mask named in ``inputs`` into one ``changepoints`` mask
    #: (across every non-time dim) and expand *that* — one block of columns
    #: instead of one per mask. Use it when "something changed here" is the
    #: signal and which detector fired is not. All merged masks must share a
    #: ``target_feature``.
    merge: bool = False
    #: feature → dim → values, the same shape as ``features.columns``: which
    #: raw changepoint masks to expand, and which dims to pin (typically
    #: ``keypoint``; the individual dim is still pinned per sample).
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Which of the four ``more_changepoint_features`` column groups become
    #: real columns; default is all four. ``binary`` duplicates the raw mask
    #: (just marked ``normalise=0``), so drop it if you already select the
    #: mask itself.
    transforms: list[str] | None = None
    #: Reach of the ``offset`` columns, in samples: the distance to the
    #: previous / next candidate saturates here. ``None`` derives it from the
    #: shortest labelled behaviour (half its 5th-percentile duration).
    horizon: float | None = None
    #: Feature whose values scale the proximity columns by ``exp(-x / mean x)``,
    #: emphasising candidates where it is low — ``speed`` picks the troughs at
    #: rest over a dip inside a movement, an amplitude envelope the silences
    #: between calls. Pinned like the mask (it may carry no dim the mask
    #: lacks). ``None`` leaves the proximity unscaled.
    scale_by: str | None = None
    #: Where the ``length`` column saturates, in samples. ``None`` derives it
    #: from the longest labelled behaviours (the 95th-percentile duration).
    max_length: float | None = None
    #: Written by ``materialise`` when it derived any of the three: what was
    #: read off which labels. Carried into the run's ``config.yaml`` so a
    #: run says where its numbers came from. Never set it by hand.
    note: str | None = None

    def __post_init__(self) -> None:
        from ethograph.features.changepoints import CP_TRANSFORMS

        if self.distribution not in ("laplacian", "gaussian"):
            raise ValueError(
                f"features.changepoint_features.distribution must be 'laplacian' or 'gaussian', "
                f"got {self.distribution!r}"
            )
        if self.sigmas is not None:
            self.sigmas = [float(s) for s in self.sigmas]
            if not self.sigmas:
                raise ValueError(
                    "features.changepoint_features.sigmas must name at least one sigma, or be left out to be derived"
                )
            if any(not s > 0 for s in self.sigmas):
                raise ValueError(f"features.changepoint_features.sigmas must be positive (samples), got {self.sigmas}")
        if not self.inputs:
            raise ValueError("features.changepoint_features.inputs must name at least one changepoint variable")
        if self.transforms is None:
            self.transforms = list(CP_TRANSFORMS)
        unknown = set(self.transforms) - set(CP_TRANSFORMS)
        if unknown:
            raise ValueError(
                f"features.changepoint_features.transforms must be a subset of {CP_TRANSFORMS}, got {sorted(unknown)}"
            )
        if self.horizon is not None:
            self.horizon = float(self.horizon)
            if not self.horizon > 0:
                raise ValueError(
                    f"features.changepoint_features.horizon must be positive (samples), got {self.horizon}"
                )
        if self.max_length is not None:
            self.max_length = float(self.max_length)
            if not self.max_length > 0:
                raise ValueError(
                    f"features.changepoint_features.max_length must be positive (samples), got {self.max_length}"
                )

    @property
    def unresolved(self) -> bool:
        """``True`` while any of the three scales still has to be read off the labels."""
        return self.sigmas is None or self.horizon is None or self.max_length is None

    @property
    def n_sigmas(self) -> int:
        """How many proximity columns each mask expands to — fixed before the values are."""
        from ethograph.features.changepoints import SIGMA_DIVISORS

        return len(self.sigmas) if self.sigmas is not None else len(SIGMA_DIVISORS)

    SCALE_KEYS: ClassVar[tuple[str, ...]] = ("sigmas", "horizon", "max_length", "note")

    def scales(self) -> dict[str, Any]:
        """The resolved scales as the plain block ``columns.yaml`` records."""
        if self.unresolved:
            raise ValueError("features.changepoint_features is unresolved — materialise first")
        return {
            "sigmas": [float(s) for s in self.sigmas or []],
            "horizon": float(self.horizon),  # type: ignore[arg-type]
            "max_length": float(self.max_length),  # type: ignore[arg-type]
            "note": self.note,
        }

    def with_scales(self, scales: dict[str, Any]) -> ChangepointFeaturesConfig:
        """A copy resolved from a recorded block — a materialised dataset's, or a run's."""
        return replace(
            self,
            sigmas=[float(s) for s in scales["sigmas"]],
            horizon=float(scales["horizon"]),
            max_length=float(scales["max_length"]),
            note=scales.get("note"),
        )

    def resolve(self, durations_s: Any, fs: float, context: str) -> ChangepointFeaturesConfig:
        """A copy with every unset scale read off *durations_s* (seconds) at *fs* Hz.

        *context* says which labels those were, for the note. A config that
        is already resolved comes back unchanged.
        """
        from ethograph.features.changepoints import (
            HORIZON_FRACTION,
            LONG_PERCENTILE,
            SHORT_PERCENTILE,
            SIGMA_DIVISORS,
            scales_from_durations,
        )

        if not self.unresolved:
            return self
        derived = scales_from_durations(durations_s, fs)
        horizon = derived.horizon if self.horizon is None else float(self.horizon)
        max_length = derived.max_length if self.max_length is None else float(self.max_length)
        sigmas = [horizon / k for k in SIGMA_DIVISORS] if self.sigmas is None else list(self.sigmas)
        parts = []
        if self.horizon is None:
            parts.append(
                f"horizon = {HORIZON_FRACTION:g} x p{SHORT_PERCENTILE:g}(duration) = {horizon:g} samples "
                f"({horizon / fs:.3f} s)"
            )
        if self.max_length is None:
            parts.append(
                f"max_length = p{LONG_PERCENTILE:g}(duration) = {max_length:g} samples ({max_length / fs:.3f} s)"
            )
        if self.sigmas is None:
            divisors = ", ".join(f"{k:g}" for k in SIGMA_DIVISORS)
            parts.append(f"sigmas = horizon / ({divisors}) = [{', '.join(f'{s:g}' for s in sigmas)}] samples")
        note = (
            f"Derived at materialise from {context} at {fs:g} Hz: " + "; ".join(parts) + ". "
            "Spell sigmas, horizon or max_length (samples) under features.changepoint_features to pin one."
        )
        return replace(self, sigmas=sigmas, horizon=horizon, max_length=max_length, note=note)

    def expanded_columns(self) -> dict[str, dict[str, Any]]:
        """The ``features.columns`` entries this config generates — known before the scales are."""
        from ethograph.features.changepoints import cp_feature_names

        out: dict[str, dict[str, Any]] = {}
        if self.merge:
            # The merge ORs across every non-time dim, so the one surviving
            # mask has no dim left to pin.
            for name in cp_feature_names(MERGED_CHANGEPOINTS, self.n_sigmas, self.transforms):
                out[name] = {}
            return out
        for var, dims in self.inputs.items():
            for name in cp_feature_names(var, self.n_sigmas, self.transforms):
                out[name] = dict(dims or {})
        return out


@dataclass
class NeuralFeaturesConfig:
    """Turn the session's spike trains into one dense feature, at session open.

    A pynapple session's units are a ``TsGroup`` — event times, which no
    loader can read as a feature — so this is the second exception to
    "features are built with the session, never by the pipeline": the
    binning is a modelling choice worth sweeping (bin size, smoothing, rate
    versus count), and it is cheap enough to redo on every open rather than
    save. ``transform`` is a list of pynapple expressions evaluated in
    order, ``x`` being the previous result (the ``TsGroup`` at first), with
    ``nap``, ``np`` and :func:`~ethograph.features.neural.sliding_window`
    in scope; the last one must leave a ``TsdFrame`` with one column per
    unit. See :mod:`ethograph.features.neural`.

    The result is the feature ``name``, declared ``kind: neural_feature``
    and z-scored per run like any kinematic column. Its columns are the
    session's own unit ids, so they are not spelled in ``features.columns``:
    ``materialise`` reads them off the session, records them in
    ``columns.yaml``, and every later stage takes them from there — which
    is also why a neural project is one session: two sessions do not share
    units. Spell ``features.columns.{name}`` yourself to pin a subset.
    """

    #: Key of the ``TsGroup`` in the session (``units.npz`` → ``units``).
    units: str = "units"
    #: The feature the transform produces.
    name: str = "rate"
    #: pynapple expressions applied in order to ``x``; the last leaves a ``TsdFrame``.
    transform: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.units:
            raise ValueError("features.neural.units must name the session's TsGroup")
        if not self.name:
            raise ValueError("features.neural.name must name the feature the transform produces")
        if self.name == self.units:
            raise ValueError(f"features.neural.name {self.name!r} is the TsGroup's own key — pick another name")
        self.transform = [str(s) for s in self.transform]
        if not self.transform:
            raise ValueError("features.neural.transform must list at least one step, e.g. 'x.count(0.005)'")


@dataclass
class FeaturesConfig:
    """What a sample is made of."""

    #: Materialised dataset name → ``{root}/data/{name}``.
    name: str = "default"
    #: feature → dim → values. The individual dim is never listed (it is
    #: pinned per sample); a second individual dim may be ``other: "*"``.
    columns: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Individuals that become samples; ``None`` = the dataset's individual coord.
    #: For a single-animal project, prefer the top-level ``config.individual``
    #: instead — a one-item list here is what it resolves to.
    individuals: list[str] | None = None
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    labels: LabelsConfig | None = None
    #: Set to expand raw changepoint masks into proximity/segment-ID features
    #: at session-open time; ``None`` = sessions keep only what their own
    #: ``.nc``/sidecar already declares.
    changepoint_features: ChangepointFeaturesConfig | None = None
    #: Set to bin the session's spike trains into a dense feature at
    #: session-open time (pynapple sessions only); ``None`` = no spikes.
    neural: NeuralFeaturesConfig | None = None
    #: Features in ``columns`` that are **angles**: each is replaced by the
    #: two components of its ``(sin, cos)`` encoding, in radians or degrees
    #: as the variable's ``units`` attr says (or as its values imply). A
    #: circular quantity read as a plain number puts its two ends maximally
    #: far apart, and the components are bounded, so they are never z-scored.
    sin_cos: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown = [name for name in self.sin_cos if name not in self.columns]
        if unknown:
            raise ValueError(
                f"features.sin_cos names {unknown}, which features.columns does not select "
                f"(it has {sorted(self.columns)})."
            )


@dataclass
class ModelConfig:
    architecture: str = "c2f_tcn"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentConfig:
    """Training-time augmentation; geometric ones act on vector groups only."""

    noise_std: float = 0.0
    #: Random temporal stretch factor range, e.g. ``[0.8, 1.2]``.
    stretch: tuple[float, float] | None = None
    mirror: bool = False
    rotate_deg: float = 0.0


@dataclass
class SplitConfig:
    """Three ratios, drawn by whole trial. Nothing else to decide.

    The trials of every session (after ``trials.where``) are pooled, shuffled
    once with ``seed`` and cut into the three roles — 60/20/20 by default.
    Splitting is by whole trial, never mid-trial, so no trial is ever in two
    roles.

    ``holdout_sessions`` is the cross-validation escape hatch: name one or
    more sessions and *all* of their trials become ``test``, whatever the
    fractions say, with ``val_fraction`` (renormalised against
    ``train_fraction``) carved out of the sessions that remain. That is what
    :meth:`ethograph.segment.project.Project.cross_validate` writes per fold,
    and it is the only place a whole session gets a role.

    ``holdout_trials`` is the same thing one level down, for a project whose
    sessions cannot be held out — a single-session neural decoding project,
    whose units exist in one recording only. The named trial ids (in every
    session) become ``test``; ``cross_validate(n_folds=k)`` deals every
    trial into exactly one fold and writes this per fold. The two holdouts
    are exclusive: a fold is by session or by trial, never both.

    **These three defaults are ours, deliberately not upstream's.**
    DLC2Action's ``config/training.yaml`` sets ``val_frac: 0.2`` and
    ``test_frac: 0`` over fixed-length 128-frame windows; a sample here is a
    whole trial, and its ``test_frac: 0`` assumes a separate held-out project.
    60/20/20 is what the two-stage workflow needs: a validation set big enough
    that the score it hands Optuna means something, and a test set that stays
    untouched underneath it.
    """

    #: Fraction of trials the model learns from.
    train_fraction: float = 0.6
    #: Fraction held back to score the model *during* development — the
    #: objective an Optuna search maximises, and what selects ``best.pt``.
    val_fraction: float = 0.2
    #: Fraction touched once, at the end, for a number you can report.
    test_fraction: float = 0.2
    seed: int = 0
    #: Sessions held out whole as ``test`` (a cross-validation fold). Each
    #: entry is a session ``source`` path, matched against ``sessions``.
    holdout_sessions: list[Path] = field(default_factory=list)
    #: Trials held out whole as ``test`` (a trial-level cross-validation
    #: fold), by trial id, in every session. Written per fold by
    #: ``cross_validate(n_folds=...)``.
    holdout_trials: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("train_fraction", "val_fraction", "test_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"train.split.{name} must be between 0 and 1, got {value}")
            setattr(self, name, value)
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "train.split fractions must sum to 1, got "
                f"train_fraction={self.train_fraction} + val_fraction={self.val_fraction} + "
                f"test_fraction={self.test_fraction} = {total:g}"
            )
        if not self.train_fraction:
            raise ValueError("train.split.train_fraction is 0 — there would be nothing to learn from")
        self.holdout_sessions = [Path(p) for p in self.holdout_sessions]
        self.holdout_trials = [str(t) for t in self.holdout_trials]
        if self.holdout_sessions and self.holdout_trials:
            raise ValueError(
                "train.split names both holdout_sessions and holdout_trials — a fold holds out whole "
                "sessions or whole trials, never both."
            )


@dataclass
class TrainConfig:
    #: Base run name; ``None`` derives one from architecture + features name.
    #: Every call to :func:`~ethograph.segment.train.train` creates its own,
    #: never-overwritten run directory by appending the creation timestamp
    #: (to the minute) — see ``train._new_run_dir``.
    run_name: str | None = None
    #: Upstream's ``num_epochs``, ``lr``, ``weight_decay``.
    epochs: int = field(default_factory=lambda: upstream_training_default("num_epochs", int))
    learning_rate: float = field(default_factory=lambda: upstream_training_default("lr", float))
    weight_decay: float = field(default_factory=lambda: upstream_training_default("weight_decay", float))
    #: **Ours, deliberately not upstream's.** Upstream's ``batch_size: 64``
    #: counts fixed-length 128-frame windows (``general.yaml: len_segment``);
    #: a sample here is a whole trial, so 64 of them is a different quantity
    #: entirely. One trial per step also keeps ASFormer's sliding attention
    #: and C2F's BatchNorm on their intended footing.
    batch_size: int = 1
    #: Ours: upstream does not clip gradients. ``0`` disables.
    grad_clip: float = 1.0
    #: Validate every N epochs. Every run trains its full ``epochs`` budget;
    #: validation records the metric curve and keeps the best checkpoint, it
    #: never cuts the run short.
    eval_every: int = 5
    #: Validation metric ``best.pt`` is selected on, and the objective a
    #: hyperparameter search would read off :class:`RunResult.best_score`.
    select_on: str = "f1@50"
    f1_thresholds: list[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    seed: int = 0
    device: str | None = None
    #: Feature categories to leave out of this run — the ablation axis
    #: (``[video_feature]`` trains the same model without S3D). Applied to
    #: the materialised dataset's columns, so an ablation costs a run, not a
    #: re-materialisation. Columns whose kind is undeclared are always kept.
    drop_kinds: list[str] = field(default_factory=list)
    #: Train and predict at ``fs / subsample`` — the temporal-resolution axis,
    #: run-level like :attr:`drop_kinds`, so one materialised dataset serves
    #: every rate. ``1`` is the dataset's own rate. Every metric a run reports
    #: is then in that run's frames; to compare rates, score the predictions
    #: back on the full-rate grid (``scripts/experiment2_smoothing.py``).
    subsample: int = 1
    #: Overrides on DLC2Action's ``config/losses.yaml`` ``ms_tcn`` block —
    #: the same shape as ``model.params``. The loss is upstream's
    #: ``MS_TCN_Loss``; we write no default for it. See
    #: :func:`ethograph.segment.losses.build_loss`.
    loss: dict[str, Any] = field(default_factory=dict)
    #: Weight of the frame-wise loss above in the total.
    frame_weight: float = 1.0
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class SearchSpace:
    """One hyperparameter's range, keyed in ``search.params`` by its dotted config path.

    ``type`` mirrors Optuna's three suggest calls::

        train.learning_rate: {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
        model.params.num_f_maps: {type: int, low: 32, high: 256, step: 32}
        train.augment.mirror: {type: categorical, choices: [true, false]}
    """

    type: str = "float"
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.type not in ("float", "int", "categorical"):
            raise ValueError(f"search space type must be 'float', 'int' or 'categorical', got {self.type!r}")
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("a categorical search space needs a non-empty 'choices' list")
            return
        if self.low is None or self.high is None:
            raise ValueError(f"a {self.type} search space needs 'low' and 'high'")
        self.low, self.high = float(self.low), float(self.high)
        if self.low >= self.high:
            raise ValueError(f"search space low ({self.low}) must be below high ({self.high})")
        if self.log and self.low <= 0:
            raise ValueError("a log-scaled search space needs low > 0")

    def suggest(self, trial: Any, name: str) -> Any:
        """Ask *trial* for a value of this space."""
        if self.type == "categorical":
            return trial.suggest_categorical(name, list(self.choices or []))
        if self.type == "int":
            return trial.suggest_int(
                name, int(self.low or 0), int(self.high or 0), step=int(self.step or 1), log=self.log
            )
        return trial.suggest_float(name, float(self.low or 0.0), float(self.high or 0.0), step=self.step, log=self.log)


@dataclass
class SearchConfig:
    """Optuna hyperparameter search — stage 1 of the workflow.

    Every trial trains one run with a different draw of ``params`` and is
    scored by ``train.select_on`` **on the validation trials**
    (:attr:`~ethograph.segment.train.RunResult.best_score`). That is what the
    validation split is for; nothing else reads it to make a decision.

    Keys are the same dotted paths an override uses, so a space and a manual
    override are the same spelling::

        search:
          n_trials: 30
          params:
            train.learning_rate: {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
            model.params.num_f_maps: {type: categorical, choices: [64, 128, 256]}
    """

    #: Number of configurations to try.
    n_trials: int = 20
    #: Stop the study after this many seconds, however many trials are left.
    timeout: float | None = None
    #: dotted config key → :class:`SearchSpace`.
    params: dict[str, SearchSpace] = field(default_factory=dict)
    #: Search name → ``{root}/searches/{name}``, and ``runs/{name}/trial000_…``.
    #: ``None`` derives one from the run name.
    name: str | None = None
    seed: int = 0
    #: Abandon a trial whose validation curve is below the running median at
    #: the same epoch (Optuna's ``MedianPruner``).
    prune: bool = True
    #: Keep every trial's weights. Off by default: a study is dozens of runs,
    #: and only the best one's ``best.pt``/``last.pt`` is worth the disk. The
    #: config, split and metrics of a pruned or losing trial are always kept.
    keep_weights: bool = False

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError(f"search.n_trials must be at least 1, got {self.n_trials}")
        spaces: dict[str, SearchSpace] = {}
        for key, space in self.params.items():
            if isinstance(space, SearchSpace):
                spaces[key] = space
            elif isinstance(space, dict):
                unknown = set(space) - {f.name for f in fields(SearchSpace)}
                if unknown:
                    raise ValueError(f"search.params.{key}: unknown key(s) {sorted(unknown)}")
                spaces[key] = SearchSpace(**space)
            else:
                raise ValueError(f"search.params.{key}: expected a mapping, got {type(space).__name__}")
        self.params = spaces


@dataclass
class PostprocessConfig:
    """Purge → stitch → (snap to changepoints) → purge.

    The interval steps are the GUI's changepoint correction, and
    ``gui_settings`` lets a config *take* the GUI's numbers instead of
    spelling them: ``true`` reads ``gui_settings.yaml`` from the ethograph
    home, a path reads that file (see :data:`GUI_POSTPROCESS_KEYS` for which
    keys). Anything spelled explicitly beside it still wins, so an override
    such as ``infer.postprocess.max_shrink_s=0.1`` composes with it. The
    values are resolved when the config is loaded, and a saved run config
    carries them explicitly — the run does not change when the GUI does.
    """

    #: ``true`` / a path: read the correction settings from the GUI's
    #: ``gui_settings.yaml`` (:func:`read_gui_postprocess`); ``None``: as spelled.
    gui_settings: str | bool | None = None
    min_duration_s: float = 0.0
    label_thresholds: dict[int, float] = field(default_factory=dict)
    stitch_gap_s: float = 0.0
    changepoint_correction: bool = False
    #: Selections pinning the changepoint variables (e.g. ``keypoint: beakTip``);
    #: the individual is pinned per sample.
    changepoints: dict[str, str] = field(default_factory=dict)
    max_expansion_s: float = 0.05
    max_shrink_s: float = 0.05

    def __post_init__(self) -> None:
        self.label_thresholds = {int(k): float(v) for k, v in self.label_thresholds.items()}


@dataclass
class InferConfig:
    #: Run name (or path) under ``{root}/runs``; ``None`` = the most recently
    #: trained run for ``train.run_name`` (see ``train.run_name_for`` /
    #: ``infer.resolve_run_dir``'s ``{base_name}_{timestamp}`` naming).
    run: str | None = None
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    #: Multi-label targets only: a channel's sigmoid must exceed this to
    #: count as on. Ours — upstream decodes multi-label through its own
    #: metric layer, which is not vendored. Exclusive targets argmax and
    #: never read it.
    threshold: float = 0.5


#: The project default for the S3D window when ``stack_s`` is not spelled:
#: 15 frames at 30 fps, so it works down to 26 fps.
S3D_DEFAULT_STACK_S = 0.5


@dataclass
class VideoFeaturesConfig:
    """Which extractor, the choices that change its features, and the camera.

    ``extractor`` names an entry of :data:`ethograph.video_features.EXTRACTORS`:
    ``s3d`` (clip-wise, a ``stack_s`` window of motion per frame — the
    default) or ``timm`` (frame-wise, any timm image backbone, DINOv2 unless
    ``model_name`` says otherwise). A setting
    that belongs to the other extractor is refused by name rather than
    ignored — ``stack_s`` means nothing to a frame-wise model, ``model_name``
    nothing to S3D.

    Everything else about the extraction (batch size, decode chunk, fp16,
    device, S3D's ``dense`` ablation mode) is a performance detail with one
    sensible answer, so it is not a project setting; build the extractor's
    own config yourself in the rare case you need one.
    """

    #: Registry name of the network.
    extractor: str = "s3d"
    #: ``timm`` only: the backbone. ``None`` = the registry default.
    model_name: str | None = None
    #: ``s3d`` only: temporal extent of one window, in seconds — how much
    #: motion context each frame's feature sees. Must be at least 13 frames at
    #: the effective rate; ``None`` = :data:`S3D_DEFAULT_STACK_S`.
    stack_s: float | None = None
    #: Rate the network sees; ``None`` = every frame. Frames are skipped,
    #: never interpolated up, so halving this roughly halves the cost.
    analysis_fps: float | None = None
    #: Which camera's video to take, when the alignment holds several.
    camera: str | None = None
    #: One pixel box cut from every frame before the network sees it — the
    #: individual's part of the frame, in the GUI crop tool's numbers.
    crop: CropBox | None = None

    def __post_init__(self) -> None:
        check_extractor_name(self.extractor)
        if isinstance(self.crop, dict):
            # The YAML loader builds nested dataclasses; the Python helper
            # (`extract_videos(crop={...})`) hands the mapping straight here.
            self.crop = CropBox(**self.crop)
        if self.stack_s is not None and self.extractor != "s3d":
            raise ValueError(
                f"video_features.stack_s is the s3d window; the {self.extractor!r} extractor is "
                "frame-wise and has none — remove it or set extractor: s3d"
            )
        if self.model_name is not None and self.extractor != "timm":
            raise ValueError(
                f"video_features.model_name chooses a timm backbone; it means nothing to {self.extractor!r}"
            )

    @property
    def name(self) -> str:
        """The extractor's name: the sidecar suffix and the merged variable."""
        return self.extractor

    def build(self) -> Extractor:
        """The configured :class:`~ethograph.video_features.Extractor` (imports its package)."""
        module = extractor_module(self.extractor)
        if self.extractor == "s3d":
            stack_s = S3D_DEFAULT_STACK_S if self.stack_s is None else self.stack_s
            return module.S3DExtractor(module.S3DConfig(analysis_fps=self.analysis_fps, stack_s=stack_s), self.crop)
        params = {"analysis_fps": self.analysis_fps}
        if self.model_name is not None:
            params["model_name"] = self.model_name
        return module.TimmExtractor(module.TimmConfig(**params), self.crop)


@dataclass
class SegmentConfig:
    sessions: list[SessionSpec]
    #: Project directory: ``data/`` and ``runs/`` live here. Default: the config's folder.
    root: Path = Path(".")
    #: The one individual this project's samples belong to — the single-animal
    #: spelling, stamped into every exported label row's ``individual`` column.
    #: Equivalent to ``features.individuals: [name]``; set only one of them
    #: (:func:`config_from_dict` fills the other in and refuses a mismatch).
    individual: str | None = None
    trials: TrialsConfig = field(default_factory=TrialsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    video_features: VideoFeaturesConfig = field(default_factory=VideoFeaturesConfig)
    #: Where this config was loaded from (not part of the YAML).
    config_path: Path | None = None

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / self.features.name

    @property
    def video_features_dir(self) -> Path:
        return self.root / "video_features"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def searches_dir(self) -> Path:
        return self.root / "searches"

    @property
    def cross_validation_dir(self) -> Path:
        return self.root / "cross_validation"

    def run_dir(self, run_name: str) -> Path:
        return self.runs_dir / run_name

    def select_sessions(self, selector: Iterable[str | Path] | None) -> list[SessionSpec]:
        """The sessions *selector* names, in config order; ``None`` = all of them.

        An entry matches a session by full path or by the source file's stem,
        so a fold can be named ``"ses-03"`` rather than spelled out.
        """
        if selector is None:
            return list(self.sessions)
        chosen: list[SessionSpec] = []
        wanted = [str(s) for s in selector]
        for item in wanted:
            matches = [
                s for s in self.sessions if str(s.source) == item or s.source.stem == item or s.source.name == item
            ]
            if not matches:
                raise ValueError(
                    f"No session matches {item!r}; this config has {[s.source.stem for s in self.sessions]}"
                )
            for spec in matches:
                if spec not in chosen:
                    chosen.append(spec)
        return [s for s in self.sessions if s in chosen]


# ---------------------------------------------------------------------------
# Building from dicts
# ---------------------------------------------------------------------------

_PATH_FIELDS = {"source", "labels_path", "video_dir", "alignment", "mapping", "root", "frames"}
_PATH_LIST_FIELDS = {"holdout_sessions"}
_TUPLE_FIELDS = {"clip_percentiles", "stretch"}


#: Settings that were removed, spelled as the dotted path :func:`_build` sees.
#: A key here is dropped with a log line instead of being refused as unknown:
#: a run's ``config.yaml`` is the record of what it trained with and is read
#: back by :func:`~ethograph.segment.inference.inference`, so retiring a
#: setting must not make every run trained before it unloadable. Removing an
#: entry from this table is what finally breaks those runs.
RETIRED_KEYS: dict[str, str] = {
    "config.train.circle": "the circle metric-learning term was removed; runs that set it trained with it, "
    "but nothing reads it now",
}


def _build(cls: type, data: Any, where: str, base_dir: Path, nested: dict[str, type] | None = None) -> Any:
    """Build dataclass *cls* from *data*, failing on unknown keys.

    *nested* maps a field name to the dataclass its mapping builds, and
    defaults to this module's own :data:`_NESTED`. A sibling pipeline passes
    its own so that a field name both of them use — ``train``, ``model``,
    ``split`` — builds the right type for the config being read.
    """
    nested = _NESTED if nested is None else nested
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    retired = [name for name in data if f"{where}.{name}" in RETIRED_KEYS]
    if retired:
        data = {k: v for k, v in data.items() if k not in retired}
        for name in retired:
            logger.info("%s.%s is a retired setting, ignored: %s", where, name, RETIRED_KEYS[f"{where}.{name}"])
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; valid keys: {sorted(known)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        if value is None and known[name].default_factory is not MISSING:
            # `params:` with nothing after it — or only a comment — is YAML
            # null, and for a field whose default is built by a factory (every
            # dict, list and nested config here) that plainly means "leave it
            # at the default". Passing the None on would hand a `None` to code
            # expecting a mapping, far from the line that wrote it.
            continue
        kwargs[name] = _convert(name, known[name].type, value, f"{where}.{name}", base_dir, nested)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"{where}: {exc}") from exc


_NESTED: dict[str, type] = {
    "preprocess": PreprocessConfig,
    "labels": LabelsConfig,
    "changepoint_features": ChangepointFeaturesConfig,
    "neural": NeuralFeaturesConfig,
    "features": FeaturesConfig,
    "model": ModelConfig,
    "augment": AugmentConfig,
    "split": SplitConfig,
    "train": TrainConfig,
    "search": SearchConfig,
    "postprocess": PostprocessConfig,
    "infer": InferConfig,
    "trials": TrialsConfig,
    "video_features": VideoFeaturesConfig,
    "crop": CropBox,
}


def _convert(
    name: str, annotation: Any, value: Any, where: str, base_dir: Path, nested: dict[str, type] | None = None
) -> Any:
    nested = _NESTED if nested is None else nested
    if value is None:
        return None
    if name in nested:
        return _build(nested[name], value, where, base_dir, nested)
    if name == "sessions":
        if not isinstance(value, list):
            raise ValueError(f"{where}: 'sessions' must be a list")
        return [_session(v, f"{where}[{i}]", base_dir) for i, v in enumerate(value)]
    if name in _PATH_FIELDS:
        return _path(value, base_dir)
    if name == "video_feature_folders":
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a mapping of variable name -> folder, got {type(value).__name__}")
        return {str(k): _path(v, base_dir) for k, v in value.items()}
    if name in _PATH_LIST_FIELDS:
        if not isinstance(value, list):
            raise ValueError(f"{where}: expected a list of session paths, got {type(value).__name__}")
        return [_path(v, base_dir) for v in value]
    if name in _TUPLE_FIELDS:
        if len(value) != 2:
            raise ValueError(f"{where}: expected two numbers, got {value!r}")
        return (float(value[0]), float(value[1]))
    cast = _numeric_cast(annotation)
    if cast is not None and isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return _as_number(cast, value, where)
    return value


_NUMERIC: dict[str, type] = {"float": float, "int": int}


def _numeric_cast(annotation: Any) -> type | None:
    """The coercion a ``float``/``int`` (optionally ``| None``) field needs, else ``None``.

    Not cosmetic: YAML 1.1 reads ``learning_rate: 1e-4`` as the *string*
    ``"1e-4"`` — no dot, no sign — so an unconverted value would reach the
    optimizer as text. Covered by ``tests/test_unit/test_segment_pipeline.py``.
    """
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    return _NUMERIC.get(text.replace(" ", "").removesuffix("|None"))


def _as_number(cast: type, value: Any, where: str) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: expected a number, got {value!r}") from exc
    if cast is int and not float(number).is_integer():
        raise ValueError(f"{where}: expected a whole number, got {value!r}")
    return cast(number)


def _session(value: Any, where: str, base_dir: Path) -> SessionSpec:
    if isinstance(value, (str, Path)):
        return SessionSpec(source=_path(value, base_dir))
    if isinstance(value, dict) and "role" in value:
        raise ValueError(
            f"{where}: sessions no longer carry a 'role'. Set the ratios in train.split "
            "(train_fraction / val_fraction / test_fraction) to split trials, and hold whole "
            "sessions out with Project.cross_validate() (train.split.holdout_sessions)."
        )
    spec = _build(SessionSpec, value, where, base_dir)
    if spec.name is not None and not isinstance(spec.name, str):
        # YAML 1.1 reads `name: 20260304_01` as the integer 2026030401 — the
        # underscore is a digit separator — and the original spelling is gone.
        raise ValueError(
            f"{where}.name: got {spec.name!r}, not a string. Quote it — name: '{spec.name}' — YAML reads "
            "digits with underscores as one number."
        )
    return spec


def _path(value: Any, base_dir: Path) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


# ---------------------------------------------------------------------------
# YAML in / out
# ---------------------------------------------------------------------------


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge *over* onto *base*, returning a new dict."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml_chain(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    base_ref = raw.pop("base", None)
    if base_ref is None:
        return raw
    base_path = _path(base_ref, path.parent)
    if not base_path.is_file():
        raise FileNotFoundError(f"{path}: base config {base_path} does not exist")
    return deep_merge(_read_yaml_chain(base_path), raw)


#: The generic config machinery, for a sibling pipeline that shares the
#: session/split dataclasses but has its own stage graph (``ethograph.spot``).
build_dataclass = _build
read_yaml_chain = _read_yaml_chain
resolve_path = _path


def as_overrides(params: dict[str, Any]) -> list[str]:
    """``{"train.epochs": 40}`` → ``["train.epochs=40"]``, the spelling :func:`apply_overrides` takes.

    Values go through YAML, so a dict, a list, a path or a bool round-trips
    exactly as the file would have spelled it — which is what a script
    building overrides programmatically wants, rather than ``str()`` and its
    Python-repr quoting.
    """
    return [f"{key}={_dump_value(value)}" for key, value in params.items()]


def _dump_value(value: Any) -> str:
    """*value* as a one-line YAML scalar/flow collection.

    PyYAML ends a scalar *document* with a ``...`` marker on its own line
    (``1e-05`` dumps as ``"1.0e-05\\n...\\n"``), which would travel into the
    override string and out again into anything that prints or reuses it.
    """
    text = yaml.safe_dump(value, default_flow_style=True).strip()
    lines = [line for line in text.splitlines() if line.strip() != "..."]
    return " ".join(lines)


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` dotlist overrides (values parsed as YAML)."""
    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} is not of the form key.path=value")
        key, _, raw_value = item.partition("=")
        value = yaml.safe_load(raw_value) if raw_value != "" else None
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            if node.get(part) is None:
                # `params:` with nothing after it is YAML null — "the default",
                # exactly as _build reads it — so an override may descend into it.
                node[part] = {}
            node = node[part]
            if not isinstance(node, dict):
                raise ValueError(f"Override {item!r}: {part!r} is not a mapping")
        node[parts[-1]] = value
    return out


def _resolve_gui_postprocess(data: dict, base_dir: Path) -> dict:
    """Fill ``infer.postprocess`` from the GUI's settings file when it asks for that.

    The GUI's values are the base; every key spelled in the config beside
    ``gui_settings`` (or arriving as an override) wins over them. The path
    is recorded in place of ``true`` so a saved run config says where the
    numbers came from — and, carrying them explicitly, no longer depends on
    that file.
    """
    infer = data.get("infer")
    postprocess = infer.get("postprocess") if isinstance(infer, dict) else None
    if not isinstance(postprocess, dict) or not postprocess.get("gui_settings"):
        return data
    path = gui_settings_path(postprocess["gui_settings"], base_dir)
    explicit = {k: v for k, v in postprocess.items() if k != "gui_settings"}
    resolved = {**read_gui_postprocess(path), **explicit, "gui_settings": str(path)}
    return {**data, "infer": {**infer, "postprocess": resolved}}


def _default_labels_path(spec: SessionSpec) -> None:
    """Fill in ``spec.labels_path`` with the GUI's own ``{stem}_labels.tsv`` convention.

    Only resolves the path — :func:`~ethograph.segment.sessions.open_session`
    is what creates the file if nothing is there yet, since a config may be
    built (or re-read on every :meth:`Project.update`) without ever opening
    a session.
    """
    spec.labels_path = labels_tsv_path(spec.source)
    logger.info("%s: no labels_path set — defaulting to %s", spec.source, spec.labels_path)


def config_from_dict(data: dict, base_dir: Path, config_path: Path | None = None) -> SegmentConfig:
    data = _resolve_gui_postprocess(dict(data), base_dir)
    data.setdefault("root", ".")
    cfg = _build(SegmentConfig, data, "config", base_dir)
    cfg.config_path = config_path
    if not cfg.sessions:
        raise ValueError("config.sessions is empty — list at least one session")
    for spec in cfg.sessions:
        if spec.labels_path is None:
            _default_labels_path(spec)
    name_colliding_sessions(cfg.sessions)
    if cfg.individual is not None:
        if cfg.features.individuals is None:
            cfg.features.individuals = [cfg.individual]
        elif list(cfg.features.individuals) != [cfg.individual]:
            raise ValueError(
                f"config.individual={cfg.individual!r} conflicts with config.features.individuals="
                f"{cfg.features.individuals!r} — set only one of them"
            )
    if cfg.features.labels is None:
        raise ValueError("config.features.labels is required (at least a branch of the mapping.txt naming the classes)")
    if cfg.features.labels.mapping is None:
        cfg.features.labels.mapping = defaults_dir("mapping.txt")
    if cfg.features.changepoint_features is not None:
        generated = cfg.features.changepoint_features.expanded_columns()
        collisions = set(generated) & set(cfg.features.columns)
        if collisions:
            raise ValueError(
                f"config.features.columns already names {sorted(collisions)}, which "
                "config.features.changepoint_features also generates — remove the explicit "
                "entries, or drop them from changepoint_features.inputs/transforms"
            )
        cfg.features.columns.update(generated)
    if not cfg.features.columns and cfg.features.neural is None:
        raise ValueError("config.features.columns is empty — name at least one feature")
    known_sources = {str(s.source) for s in cfg.sessions}
    unknown_holdout = [str(p) for p in cfg.train.split.holdout_sessions if str(p) not in known_sources]
    if unknown_holdout:
        raise ValueError(
            f"train.split.holdout_sessions names {unknown_holdout}, which config.sessions does not list; "
            f"it holds {sorted(known_sources)}"
        )
    if len(cfg.train.split.holdout_sessions) == len(cfg.sessions):
        raise ValueError("train.split.holdout_sessions holds out every session — nothing would be left to train on")
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> SegmentConfig:
    """Read a config file (following ``base:``), apply overrides, build."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    data = _read_yaml_chain(path)
    if overrides:
        data = apply_overrides(data, list(overrides))
    return config_from_dict(data, path.parent, config_path=path)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj) if f.name != "config_path"}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def config_to_dict(cfg: SegmentConfig) -> dict:
    """The fully resolved config as plain YAML-able data (absolute paths).

    Round-trips: the columns ``features.changepoint_features`` generated are
    left out, because :func:`config_from_dict` merges them back in and would
    otherwise read them as explicit entries colliding with its own expansion.
    """
    data = _to_plain(cfg)
    if cfg.features.changepoint_features is not None:
        generated = cfg.features.changepoint_features.expanded_columns()
        columns = data["features"]["columns"]
        data["features"]["columns"] = {k: v for k, v in columns.items() if k not in generated}
    return data


def save_config(cfg: SegmentConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
    return path


def with_overrides(cfg: SegmentConfig, **changes: Any) -> SegmentConfig:
    """A copy of *cfg* with top-level fields replaced."""
    return dataclasses.replace(cfg, **changes)
