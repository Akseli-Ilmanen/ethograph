"""Configuration for the pixel event-spotting pipeline.

One YAML file becomes a :class:`SpotConfig`, exactly as the segmentation
pipeline's config becomes a ``SegmentConfig`` — same ``base:`` chaining, same
dotted ``key=value`` overrides, same generic dataclass builder
(:func:`~ethograph.segment.config.build_dataclass`). What differs is the stage
graph: there is no ``features:`` section, because this model reads pixels.

**Every temporal setting is a duration.** Upstream E2E-Spot expresses clip
length, temporal stride and label dilation in *frames*, tuned at 25 fps; the
identical numbers at 200 fps give the model an eighth of the real-time
aperture and it collapses to background. So the config asks for
:attr:`ClipConfig.context_s` and :attr:`ClipConfig.resolution_ms` and derives
the frame counts from each video's own rate (:meth:`ClipConfig.resolve`).
Docs: ``docs/source/models/spot/index.md``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ethograph.labels.tsv_store import labels_tsv_path
from ethograph.segment.config import (
    SessionSpec,
    SplitConfig,
    TrialsConfig,
    apply_overrides,
    build_dataclass,
    name_colliding_sessions,
    read_yaml_chain,
)

logger = logging.getLogger(__name__)

#: Frames per loader batch a **10 GB** card holds without paging — the one
#: measured point (200 frames trains at ~3.7 it/s; 400 frames pages at
#: 11.4 GB and crawls). The budget of the card actually present scales from
#: it (:func:`ethograph.spot.vendored.frame_budget`); this constant is what
#: :meth:`ClipConfig.resolve` assumes when no card is asked. ``clip_len *
#: batch_size / acc_grad`` must stay at or below the budget.
MAX_FRAMES_PER_BATCH = 200

#: Shortest clip E2E-Spot's GRU head is worth running; below this the temporal
#: model has nothing to integrate over.
MIN_CLIP_LEN = 8


@dataclass
class CropConfig:
    """The part of the camera's frame the model reads, in source pixels.

    ``(x0, y0)`` is the top-left corner and ``(x1, y1)`` the exclusive
    bottom-right, y down — the box the GUI's crop tool reports (Tools ▸
    *Video: Pick a crop for a config…*), so the numbers copy straight
    across. The crop is cut from the decoded frame *before* it is resized to
    :attr:`LabelsConfig.frame_height`, so a tight crop spends the model's
    pixels on less scene rather than shrinking the frame.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x0), int(self.y0), int(self.x1), int(self.y1))

    def validate(self) -> None:
        if min(self.x0, self.y0) < 0:
            raise ValueError(f"labels.crop: the top-left corner ({self.x0}, {self.y0}) is outside the frame")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(
                f"labels.crop: ({self.x0}, {self.y0})-({self.x1}, {self.y1}) is empty — "
                "x1 must exceed x0 and y1 must exceed y0"
            )

    def check_fits(self, width: int, height: int, what: str) -> None:
        """Refuse a crop that reaches outside a ``width`` x ``height`` frame.

        A crop was picked on one video and applies to every trial's, so a
        camera whose videos change size between sessions is caught here, on
        the trial it fails, rather than exporting a smaller box in silence.
        """
        if self.x1 > width or self.y1 > height:
            raise ValueError(
                f"labels.crop ({self.x0}, {self.y0})-({self.x1}, {self.y1}) reaches outside {what}, "
                f"which is {width}x{height} px"
            )


@dataclass
class LabelsConfig:
    """Which labels are learned, and which camera saw them."""

    #: Point-event class ids, as they appear in the labels TSV's ``labels``
    #: column. Background is implicit and is never listed.
    classes: list[int] = field(default_factory=list)
    #: The camera whose video the model reads — one per project, because a
    #: model trained on two viewpoints at once has no way to say which it is
    #: looking at. Resolved per trial through the alignment.
    camera: str | None = None
    #: The region of the frame the model reads; ``None`` = the whole frame.
    #: Picked in the GUI, which reports it in this spelling.
    crop: CropConfig | None = None
    #: Height every exported frame (after the crop) is resized to (width
    #: follows the aspect ratio). Upstream's RegNetY-008 configuration.
    frame_height: int = 224


@dataclass
class ResolvedClip:
    """:class:`ClipConfig` in the units upstream's CLI takes, for one rate."""

    fps: float
    stride: int
    clip_len: int
    dilate_len: int

    @property
    def context_s(self) -> float:
        """Seconds of video one clip actually spans at this rate."""
        return self.clip_len * self.stride / self.fps

    @property
    def resolution_ms(self) -> float:
        """Milliseconds one strided frame actually spans at this rate."""
        return 1000.0 * self.stride / self.fps

    @property
    def frames_per_batch(self) -> int:
        return self.clip_len

    def to_frame(self, index: float) -> float:
        """A predicted strided bin back on the full-rate clock.

        The **centre** of the bin: the dataset bins a truth frame as
        ``floor(frame / stride)``, so the expected full-rate frame for a bin is
        ``bin * stride + (stride - 1) / 2``. Reading it as ``bin * stride``
        makes every strided run look early by half a stride — 7.5 ms at
        stride 4, against a 20 ms budget.
        """
        return index * self.stride + (self.stride - 1) / 2.0


@dataclass
class ClipConfig:
    """How video becomes clips, in durations rather than frame counts."""

    #: Seconds of video the model sees at once. The ladder's dominant axis:
    #: below ~2 s the model misses events outright.
    context_s: float = 2.0
    #: Milliseconds one model frame spans — the grid a label can land on.
    #: ``null`` (the default) = **as fine as the frame budget allows** for
    #: ``context_s``: every frame when it fits, else the smallest stride that
    #: does. Spell it to pin the grid across machines (a 10 GB card and a
    #: 24 GB card would otherwise choose different strides for a 200 fps
    #: video); buying context by coarsening it stopped paying at ~10 ms on the
    #: rig this was measured on.
    resolution_ms: float | None = None
    #: Milliseconds either side of the event that count as positive during
    #: training. Held as a duration so dilation is not confounded with
    #: resolution when the latter changes.
    positive_window_ms: float = 10.0

    def resolve(self, fps: float, max_frames: int | None = None) -> ResolvedClip:
        """The frame counts *fps* implies, refused if they cannot be trained.

        *max_frames* is the frame budget of one loader batch — the card's
        (:func:`~ethograph.spot.vendored.frame_budget`), else the 10 GB
        measurement :data:`MAX_FRAMES_PER_BATCH`. With ``resolution_ms``
        unset the stride is the smallest that fits ``context_s`` in it; spelled,
        it is honoured and refused if it does not fit. Raises ``ValueError``
        naming the duration to change — never the frame count, which is not
        something the config spells.
        """
        if fps <= 0:
            raise ValueError(f"Frame rate must be positive, got {fps!r}")
        budget = int(max_frames) if max_frames is not None else MAX_FRAMES_PER_BATCH
        if self.resolution_ms is None:
            stride = max(1, int(math.ceil(self.context_s * fps / budget - 1e-9)))
        else:
            stride = max(1, int(round(self.resolution_ms / 1000.0 * fps)))
        clip_len = int(round(self.context_s * fps / stride))
        spelled = f"clip.resolution_ms={self.resolution_ms} ms" if self.resolution_ms is not None else "every frame"
        if clip_len < MIN_CLIP_LEN:
            raise ValueError(
                f"clip.context_s={self.context_s} s at {fps:g} fps and {spelled} is only {clip_len} model frames. "
                f"Raise context_s or lower resolution_ms until it reaches {MIN_CLIP_LEN}."
            )
        if clip_len > budget:
            needed = budget * stride / fps
            coarser = 1000.0 * self.context_s / budget
            raise ValueError(
                f"clip.context_s={self.context_s} s at {fps:g} fps and {spelled} needs {clip_len} frames per batch, "
                f"above the {budget} this card holds. "
                f"Either drop context_s to {needed:.2f} s or raise resolution_ms to {coarser:.1f} (or unset it)."
            )
        # dilate_len is counted in *strided* frames, so stride and dilation
        # multiply. Deriving it from the duration is what keeps the positive
        # window the same width in real time across every resolution.
        dilate_len = int(round(self.positive_window_ms / 1000.0 * fps / stride))
        resolved = ResolvedClip(fps=float(fps), stride=stride, clip_len=clip_len, dilate_len=dilate_len)
        # A duration only ever lands on a whole number of frames. Say so when
        # the rate cannot carry what was asked for, rather than reporting a
        # precision the grid does not have.
        if self.resolution_ms is not None and abs(resolved.resolution_ms - self.resolution_ms) > 0.5:
            logger.info(
                "clip.resolution_ms=%g is %g ms at %g fps (stride %d) — the rate cannot divide it finer",
                self.resolution_ms,
                resolved.resolution_ms,
                fps,
                stride,
            )
        return resolved


@dataclass
class TeacherConfig:
    """The pose-only teacher (:mod:`~ethograph.spot.pose_model`): the listed ``features:``, shifts, a bi-GRU.

    Every temporal setting is a duration, like :class:`ClipConfig`. The
    shift scales are UMEG-Net's ``{1, 2, 4}`` frames at 25 fps — 40/80/160 ms
    — and would mean 5/10/20 ms at 200 fps, the same trap the clip length
    fell into.
    """

    #: Temporal shift scales of the shift blocks, in milliseconds. Resolved
    #: against the features' own rate; a scale below one sample rounds up.
    shift_scales_ms: list[float] = field(default_factory=lambda: [40.0, 80.0, 160.0])
    #: Hidden width of every shift block.
    hidden: int = 64
    #: Number of stacked shift blocks.
    depth: int = 4
    #: Channels shifted forward and backward, as a fraction of ``hidden``.
    shift_fraction: float = 0.125
    #: Bi-GRU head width.
    head_hidden: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 8
    #: Foreground class weight in the per-frame cross-entropy. E2E-Spot's own.
    fg_weight: float = 5.0
    seed: int = 0

    def validate(self) -> None:
        if self.depth < 1:
            raise ValueError(f"teacher.depth must be >= 1, got {self.depth}")

    def shift_samples(self, fs: float) -> list[int]:
        """The scales as whole samples of a curve at *fs*, deduplicated, ascending."""
        if fs <= 0:
            raise ValueError(f"Frame rate must be positive, got {fs!r}")
        scales = sorted({max(1, int(round(ms / 1000.0 * fs))) for ms in self.shift_scales_ms})
        if not scales:
            raise ValueError("teacher.shift_scales_ms is empty — give at least one scale")
        return scales


@dataclass
class DistilConfig:
    """Teaching the student the teacher's representation, then its head the labels.

    One act with two steps, as UMEG-Net does it: (2) the student's trunk and
    GRU learn to reproduce the frozen teacher's per-frame embedding on every
    clip that has pose and video — no labels; (3) the CNN is frozen and the
    GRU + head learn the labels. Both steps are the vendored trainer with a
    stage flag, so they produce ordinary runs.
    """

    #: The teacher run under ``teacher/`` to distil from; ``None`` = the one
    #: whose embeddings are under ``features/embeddings``. (Not ``teacher``:
    #: the config builder resolves nesting by field name, and that one names
    #: the teacher's own section.)
    teacher_run: str | None = None
    #: The run under ``runs/`` the student starts from; ``None`` = newest.
    #: Warm-starting from the label-only baseline is what makes "student
    #: beats baseline" a statement about the representation, not the epochs.
    init_run: str | None = None
    epochs: int = 6
    epoch_frames: int = 250_000
    learning_rate: float = 1e-4
    #: The head step.
    head_epochs: int = 4
    head_learning_rate: float = 1e-4
    retries: int = 2


@dataclass
class ModelConfig:
    """The backbone and its temporal module."""

    #: Upstream's ``--feature_arch``. ``rny008_gsm`` is E2E-Spot's own;
    #: ``rny008_msagsm`` swaps its Gate Shift Module for the multi-scale one
    #: (:mod:`ethograph.spot.msagsm`), the rest of the network unchanged.
    architecture: str = "rny008_gsm"
    #: Upstream's ``--temporal_arch``.
    head: str = "gru"
    #: MSAGSM only: how far each gated-shift branch reaches, in milliseconds.
    #: Resolved against the *strided* clock the backbone sees. The paper's
    #: ``{1, 2, 3}`` frames at 25 fps are 40/80/120 ms.
    shift_scales_ms: list[float] = field(default_factory=lambda: [40.0, 80.0, 120.0])
    #: MSAGSM only: channel groups of the spatial attention (the paper's 2).
    attention_groups: int = 2

    @property
    def multiscale(self) -> bool:
        return self.architecture.endswith("_msagsm")

    def shift_dilations(self, fps_strided: float) -> list[int]:
        """The branches' reach in strided frames, deduplicated, ascending."""
        if fps_strided <= 0:
            raise ValueError(f"Frame rate must be positive, got {fps_strided!r}")
        dilations = sorted({max(1, int(round(ms / 1000.0 * fps_strided))) for ms in self.shift_scales_ms})
        if not dilations:
            raise ValueError("model.shift_scales_ms is empty — give at least one scale")
        return dilations


@dataclass
class TrainConfig:
    """One training run. Deliberately small — upstream owns the loop."""

    run_name: str | None = None
    #: With ``features:`` listed: hand them to the pixel model beside the CNN
    #: features, before the GRU (the run is named ``{clip}_features``).
    #: ``false`` keeps the list for the teacher only — the distillation
    #: recipe, where inference is video alone.
    features_as_input: bool = True
    #: Share of training clips whose feature block is zeroed (modality
    #: dropout), so the pixels are trained to carry the event on their own
    #: too — which is what keeps ``evaluate(zero_features=True)`` meaningful.
    features_dropout: float = 0.3
    epochs: int = 8
    #: Frames pushed per epoch, independent of dataset size. Upstream ties an
    #: epoch to a fixed frame budget rather than to a pass over the data, so
    #: this — not the number of trials — is what decides an epoch's cost.
    epoch_frames: int = 250_000
    learning_rate: float = 1e-3
    warm_up_epochs: int = 1
    #: Validate from this epoch on. Every ladder run peaked at epoch 1-3 and
    #: came apart after, so measuring late is measuring the wrong thing.
    start_val_epoch: int = 1
    batch_size: int = 4
    #: Gradient accumulation. ``batch_size / acc_grad`` clips reach the card at
    #: once, so this is the knob that keeps ``clip_len`` within
    #: :data:`MAX_FRAMES_PER_BATCH`.
    acc_grad: int = 4
    #: Retries of a crashed run, resuming from its last checkpoint.
    retries: int = 2
    seed: int = 0
    device: str | None = None
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class InferConfig:
    """Turning a run's per-frame scores into labels the GUI reads."""

    #: Milliseconds around the tallest peak that count as the same event when
    #: reading ``focus``/``ratio`` off a curve — the timescale you care about
    #: (twice the precision you believe your labels to, as the lightgbm model
    #: takes it from its ``tolerance_s``). Flat across 50-200 ms on a 200 fps
    #: rig; set it for yours.
    focus_window_ms: float = 100.0
    #: Below this the prediction is written anyway and flagged, never dropped:
    #: a missing label cannot be reviewed, and review is the point.
    flag_confidence_below: float = 0.01
    #: Written into every predicted row's ``prediction_source``.
    source: str | None = None
    #: A trial whose predicted events are not in ``labels.classes`` order has
    #: every event's confidence set to 0 — flagged for review, never reordered
    #: or dropped. A repaired sequence would hide exactly the trial that most
    #: needs a look.
    flag_out_of_order: bool = False
    #: Inference decodes the video straight into the model; each frame is
    #: passed through JPEG in memory first, so the model sees what training
    #: saw (the export writes JPEGs). Off = an ablation.
    jpeg_roundtrip: bool = True
    #: Which reading of a prediction's curve is written as its ``confidence``
    #: (:data:`ethograph.labels.rescore.RULES`): ``product`` (focus × ratio),
    #: ``ratio``, ``focus``, ``peak``, or ``custom`` = ratio × (α + (1 − α)·focus)
    #: with ``confidence_alpha``. The grids' histogram popup previews these on
    #: a session's curves and copies the choice as these lines.
    confidence: str = "product"
    confidence_alpha: float = 0.5

    def validate(self) -> None:
        from ethograph.labels.rescore import RULES

        if self.confidence not in RULES:
            raise ValueError(f"infer.confidence must be one of {list(RULES)}, got {self.confidence!r}")
        if not 0.0 <= self.confidence_alpha <= 1.0:
            raise ValueError(f"infer.confidence_alpha must be in [0, 1], got {self.confidence_alpha!r}")
        if self.focus_window_ms <= 0:
            raise ValueError(f"infer.focus_window_ms must be positive, got {self.focus_window_ms!r}")


@dataclass
class SpotConfig:
    """A whole pixel event-spotting project."""

    sessions: list[SessionSpec]
    #: Project directory: ``frames/``, ``dataset/`` and ``runs/`` live here.
    root: Path = Path(".")
    #: The individual these events belong to — this pipeline predicts one
    #: event stream per trial, not per individual (there is no individual dim
    #: in a pixel model's sample), so this single value is stamped into every
    #: exported label row's ``individual`` column. ``None`` leaves it blank.
    individual: str | None = None
    #: Where exported frames live, when not ``{root}/frames``. Decoding is the
    #: expensive stage, so a folder another project already filled is reused —
    #: a trial whose folder holds the right frame count is never re-decoded.
    frames: Path | None = None
    trials: TrialsConfig = field(default_factory=TrialsConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    #: The keypoint side, for a graph teacher; ``None`` = pixels only.
    #: The pose side, **optional**: session variables in ``segment``'s
    #: ``features.columns`` spelling (``velocity: {space: [x, y], keypoint:
    #: [stickTip]}``, ``pellet_stickClosest_dist: {}``), on the pose's rate —
    #: position, velocity, the distances you wrote down and can plot. Listed,
    #: they ride beside the CNN features into the pixel model's GRU
    #: (``train.features_as_input``) and are what the pose teacher reads.
    #: Absent, the model is E2E-Spot on pixels alone.
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    distil: DistilConfig = field(default_factory=DistilConfig)
    #: Where this config was loaded from (not part of the YAML).
    config_path: Path | None = None

    def resolve_clip(self, fps: float) -> ResolvedClip:
        """The clip at *fps* under the frame budget of the card present (:func:`~ethograph.spot.vendored.frame_budget`).

        The one resolver every stage uses, so the student, the teacher, the
        feature block and the export agree about the stride. A trained run
        records its own in ``config.json`` and is read back from there
        (:func:`~ethograph.spot.inference.run_clip`), so a session predicted on
        another card still uses the run's stride.
        """
        from ethograph.spot.vendored import frame_budget

        return self.clip.resolve(fps, max_frames=frame_budget())

    @property
    def frames_dir(self) -> Path:
        """Exported JPEG frames, one folder per trial. The expensive artefact.

        One folder per project, whatever the crop: a trial's ``export.json``
        records the size and crop it was decoded at, and
        :func:`~ethograph.spot.dataset.export_is_current` re-decodes a trial
        whose record disagrees with the config, so changing the crop rewrites
        the frames in place rather than growing a folder per variant.
        """
        return self.frames if self.frames is not None else self.root / "frames"

    @property
    def features_dir(self) -> Path:
        """The listed features, one ``.npz`` per trial (``features:`` only)."""
        return self.root / "features"

    @property
    def block_dir(self) -> Path:
        """The same columns z-scored on the training split — the pixel model's second input."""
        return self.features_dir / "block"

    @property
    def embeddings_dir(self) -> Path:
        """The teacher's per-clip embeddings, what the student distils from."""
        return self.features_dir / "embeddings"

    @property
    def fusing(self) -> bool:
        """Whether the pixel model reads the features beside the frames: listed, and ``train.features_as_input``."""
        return bool(self.features) and self.train.features_as_input

    @property
    def teacher_dir(self) -> Path:
        return self.root / "teacher"

    @property
    def dataset_dir(self) -> Path:
        """E2E-Spot's own index: ``{split}.json`` plus ``class.txt``."""
        return self.root / "dataset"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def cross_validation_dir(self) -> Path:
        return self.root / "cross_validation"

    def run_dir(self, run_name: str) -> Path:
        return self.runs_dir / run_name

    def class_name(self, label: int) -> str:
        """The name E2E-Spot knows a class by. Its schema is string-keyed and
        ours is integer-keyed, so the mapping is written down once, here."""
        return f"label_{int(label)}"

    def class_label(self, name: str) -> int:
        """Inverse of :meth:`class_name`, refusing anything it did not write."""
        for label in self.labels.classes:
            if self.class_name(label) == name:
                return label
        raise ValueError(f"{name!r} is not one of this config's classes {self.labels.classes}")

    def select_sessions(self, selector: Any) -> list[SessionSpec]:
        """The sessions *selector* names, in config order; ``None`` = all.

        Matches by ``name``, full path, or the source's stem, so a fold can
        be named ``"20260307_01"`` rather than spelled out — the same rule
        ``SegmentConfig.select_sessions`` follows.
        """
        if selector is None:
            return list(self.sessions)
        chosen: list[SessionSpec] = []
        for item in [str(s) for s in selector]:
            matches = [
                s
                for s in self.sessions
                if s.label == item or str(s.source) == item or s.source.stem == item or s.source.name == item
            ]
            if not matches:
                raise ValueError(f"No session matches {item!r}; this config has {[s.label for s in self.sessions]}")
            for spec in matches:
                if spec not in chosen:
                    chosen.append(spec)
        return [s for s in self.sessions if s in chosen]


#: Field name -> the dataclass its mapping builds. Passed to the shared
#: builder so that ``train``, ``model`` and ``labels`` build *this* pipeline's
#: types rather than the segmentation pipeline's same-named ones.
_NESTED: dict[str, type] = {
    "trials": TrialsConfig,
    "labels": LabelsConfig,
    "clip": ClipConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "split": SplitConfig,
    "infer": InferConfig,
    "teacher": TeacherConfig,
    "distil": DistilConfig,
    "crop": CropConfig,
}


def config_from_dict(data: dict, base_dir: Path, config_path: Path | None = None) -> SpotConfig:
    data = dict(data)
    if "graph" in data:
        raise ValueError(
            "graph: is gone — there is no graph model any more. Compute the distances and angles you care "
            "about as variables in the session file (features/geometry.py) and list them under features:."
        )
    if "fuse" in data:
        raise ValueError(
            "fuse: is gone — listed features: ride into the pixel model's GRU by default; "
            "train.features_as_input and train.features_dropout are the two settings that remain."
        )
    if isinstance(data.get("features"), dict) and "columns" in data["features"]:
        raise ValueError(
            "features: in a spot config lists the pose variables directly (segment's features.columns "
            "spelling, e.g. `velocity: {space: [x, y]}`), not a section with a columns: key — this pipeline "
            "reads pixels plus what you list here, it does not materialise feature columns"
        )
    if isinstance(data.get("teacher"), dict) and ("features" in data["teacher"] or "extra_features" in data["teacher"]):
        raise ValueError("the pose features are spelled at the top level — `features:` — one list for every model")
    data.setdefault("root", ".")
    cfg = build_dataclass(SpotConfig, data, "config", base_dir, _NESTED)
    cfg.config_path = config_path
    if not cfg.sessions:
        raise ValueError("config.sessions is empty — list at least one session")
    if not cfg.labels.classes:
        raise ValueError("config.labels.classes is empty — name at least one point-event class to spot")
    duplicates = {c for c in cfg.labels.classes if cfg.labels.classes.count(c) > 1}
    if duplicates:
        raise ValueError(f"config.labels.classes lists {sorted(duplicates)} more than once")
    for spec in cfg.sessions:
        if spec.labels_path is None:
            spec.labels_path = labels_tsv_path(spec.source)
            logger.info("%s: labels_path not set, assuming %s", spec.source, spec.labels_path)
    cfg.teacher.validate()
    cfg.infer.validate()
    for name, dims in cfg.features.items():
        if not isinstance(dims, dict):
            raise ValueError(f"features.{name}: expected a mapping of dim -> values, got {dims!r}")
    if not 0.0 <= cfg.train.features_dropout < 1.0:
        raise ValueError(f"train.features_dropout must be in [0, 1), got {cfg.train.features_dropout!r}")
    if cfg.labels.crop is not None:
        cfg.labels.crop.validate()
    name_colliding_sessions(cfg.sessions)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> SpotConfig:
    """Read a config file (following ``base:``), apply overrides, build."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    data = read_yaml_chain(path)
    if overrides:
        data = apply_overrides(data, list(overrides))
    return config_from_dict(data, path.parent, config_path=path)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_plain(v) for k, v in asdict(obj).items() if k != "config_path"}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def config_to_dict(cfg: SpotConfig) -> dict:
    """The fully resolved config as plain YAML-able data (absolute paths)."""
    return _to_plain(cfg)


def features_fingerprint(cfg: SpotConfig) -> str:
    """Eight hex digits naming the ``features`` + ``teacher`` sections, so a changed list is a new teacher.

    The teacher's run folder and the distilled student's carry it, so
    re-running after editing the list lands beside the earlier result
    instead of on top of it, and a stage-2 folder can only be skipped for
    the teacher it was matched against.
    """
    data = config_to_dict(cfg)
    key = json.dumps({"features": data.get("features"), "teacher": data.get("teacher")}, sort_keys=True, default=str)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def save_config(cfg: SpotConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
    return path


def with_overrides(cfg: SpotConfig, **changes: Any) -> SpotConfig:
    """A copy of *cfg* with top-level fields replaced."""
    return replace(copy.deepcopy(cfg), **changes)
