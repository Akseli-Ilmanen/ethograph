"""Supervised point-event onset detection ("LightGBM").

A deliberately constrained CPU model: the user labels one or more point-event
classes in a few trials, the features they pick are windowed around every
frame, and one scikit-learn ``HistGradientBoostingClassifier`` per class
learns frame-vs-no-event — scikit-learn's histogram-based gradient boosting,
modelled on Microsoft's LightGBM. The constraints that keep the task tractable
for a small model:

* only **point** events — state events have two boundaries and are out of scope;
* at most **one** event per class per trial, so inference is an argmax over
  that class's smoothed probability curve, not an open-ended peak-picking
  problem;
* every chosen feature column must share **one sampling rate** — windows are
  index-based, so mixed rates would silently misalign and are refused.

A model can predict **several point-event classes at once**: the features,
window and tolerance are shared (extracted once per trial), and each class
gets its own binary classifier trained on the trials that carry it.

Target encoding: each classifier sees a binary target (frames within
``tolerance_s`` of the labelled event are positive) with a Gaussian sample
weight peaking at the event, so a near-miss frame counts less than the exact
frame. At inference the per-frame probability is Gaussian-smoothed with the
same tolerance and its tallest peak becomes the predicted onset.

Confidence is a statistic of that curve around its tallest peak
(:mod:`ethograph.labels.curve_confidence`): the peak's **height** by default,
or — when the model's own held-out record says it separates hits from misses
clearly better — the curve's **shape** (how much of it sits under that one
bump, and whether a rival stands elsewhere). Every candidate is readable
straight off the curve frame-by-frame review draws, so a threshold is still
something the user sets by looking; which one is written is a measured
property of the model, recorded in its calibration. A human label is 1.0 by
definition; the curves are kept beside the labels
(:mod:`ethograph.labels.onset_curves`).

How often the model is *right* is a property of the model, not of one label:
:func:`fit_confidence_calibration` scores every training trial with
classifiers that never saw it, reports that hit rate, and is where the
statistic is chosen.

Every class lands on its own curve's tallest peak, independent of every other
class. A model that jointly decoded the order (this module used to fit a
linear-chain CRF) can move an event away from its own evidence to satisfy the
sequence — which is invisible on the curve, and unarguable when it is wrong.

Model layout (``~/.ethograph/defaults/runs/lightgbm/{name}/``)::

    config.yaml                 # frozen at creation: targets, features, params
    model.joblib                # trained bundle: one clf per target, plus the
                                #   config it was fitted with
    train_data/{session}/       # one folder per contributing session
        meta.yaml               # source path, columns, trials (provenance)
        trial_{id}.npz          # time (T,), data (T, D), y_labels, y_times, fs

**Prediction reads the bundle and nothing else.** ``model.joblib`` carries its
own copy of the config, because that is the layout the classifiers were fitted
on: editing ``config.yaml`` afterwards changes what Train would do next, never
what a trained model reads (:func:`config_drifted` says so out loud).
``train_data`` is training input plus provenance — :func:`train_model` and the
held-out scoring read the ``.npz`` files, inference reads none of them, and
``meta.yaml`` is a summary for the dialogs.

Feature selection reuses the catalog's dim logic: the config stores, per
feature, the explicit values chosen for each of its dims; every combination is
pinned and selected through ``DataLoader.select`` (the same ``sel_valid``
path the plots use), yielding one column per combination. Freezing explicit
values (instead of "all") keeps the column set — and thus the model's input
layout — identical across sessions.

A feature can also contribute its **time derivative** as an extra column
(``config.derivatives``, ``np.gradient`` — centred on the sample, so the
derivative peaks at the frame the signal turned). Boosted trees see each tap
of the window in isolation and cannot difference them, so "how fast is this
changing" has to be handed to them as its own input.

The session's **own labels** can be inputs too (``config.label_inputs``,
:mod:`ethograph.labels.label_inputs`): a state class becomes its on/off
indicator, a point class a Laplacian bump centred on it. They are rendered onto
the feature time base and appended after every feature column, so the input
layout stays one ordered list. A class the model predicts can never be one of
its own inputs — at training the label is there and at inference it is not, by
construction, which is the one way to hand a classifier a column that means
opposite things on the two sides.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ethograph.features.columns import (
    FeatureColumn,
    check_same_fs,
    enumerate_columns,
    expand_dim_values,
    extract_features,
    sampling_rate,
)
from ethograph.io.catalog import INDIVIDUAL_DIMS
from ethograph.labels.curve_confidence import (
    DESCRIPTIONS,
    CurveStats,
    choose_statistic,
    curve_stats,
    focus_window_s,
    rank_statistics,
    tallest_peak,
    window_samples,
)
from ethograph.labels.label_inputs import LabelInput, label_columns, render_label_inputs
from ethograph.utils.paths import defaults_dir

logger = logging.getLogger(__name__)

#: Cap on window taps per input column: beyond this, lags are spread evenly
#: across the window so high-rate features don't explode the design matrix.
MAX_LAGS = 25

#: Deterministic seed for negative-frame subsampling.
_RNG_SEED = 0

#: Keeps a 0 or 1 probability off the log-odds asymptotes.
_PROB_EPS = 1e-6

#: Placeholder label for training trials stored before multi-target support
#: (their npz holds one unlabelled ``y_time``).
_LEGACY_TARGET = -1

#: Folds used to cross-fit the probability curves the held-out score reads.
#: Each fold refits every target, so this multiplies training time; 3 keeps
#: the scoring honest without making Train a coffee break.
_CV_FOLDS = 3

#: Below this many held-out predictions a target's calibration is the identity
#: map under its ceiling: too few points to fit a slope through, and pretending
#: otherwise would be its own kind of overconfidence.
_MIN_CALIBRATION_TRIALS = 6

#: Platt scaling's L2 strength. Small on purpose — with a handful of held-out
#: trials a steep map is noise, and the ceiling is what carries the honesty.
_CALIBRATION_C = 0.5

_MODEL_FILE = "model.joblib"
_CONFIG_FILE = "config.yaml"
_TRAIN_DIR = "train_data"


# ---------------------------------------------------------------------------
# Config + storage layout
# ---------------------------------------------------------------------------


@dataclass
class OnsetModelConfig:
    """Frozen description of one onset model (written once, at creation)."""

    name: str
    #: label id -> label name, one entry per point-event class the model
    #: predicts. Each gets its own binary classifier over the shared features.
    targets: dict[int, str] = field(default_factory=dict)
    #: feature -> dim -> explicit values to include (every combination becomes
    #: one input column). A feature with no dims maps to ``{}``.
    features: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    #: Features whose time derivative is included as an extra column beside
    #: every value column they produce (see :func:`.columns.time_derivative`).
    derivatives: list[str] = field(default_factory=list)
    #: Existing label classes fed to the classifier as extra input columns,
    #: appended after every feature column (see
    #: :mod:`ethograph.labels.label_inputs`).
    label_inputs: list[LabelInput] = field(default_factory=list)
    window_s: float = 0.5
    tolerance_s: float = 0.05
    max_iter: int = 200
    learning_rate: float = 0.1
    #: Negative frames kept per positive frame (hard negatives near the event
    #: are always kept; the rest are subsampled deterministically).
    neg_per_pos: int = 20

    def __post_init__(self) -> None:
        # YAML round-trips keys as ints already, but a config built from GUI
        # widgets may carry numpy ints or strings.
        self.targets = {int(label): str(name) for label, name in self.targets.items()}
        self.derivatives = [str(feature) for feature in self.derivatives]
        # A YAML round trip hands back plain dicts; the GUI hands back the
        # dataclass. One shape reaches the rest of the module either way.
        self.label_inputs = [i if isinstance(i, LabelInput) else LabelInput(**i) for i in self.label_inputs]
        self.validate()

    def validate(self) -> None:
        """Raise unless this config describes a model that can exist.

        Called on construction and again by :func:`save_config`, so a config
        assembled field by field is refused before it reaches disk rather than
        at the first trial it trains on.
        """
        clash = [i.name for i in self.label_inputs if i.label in self.targets]
        if clash:
            # At training the class is labelled and at inference it is not —
            # prediction only ever runs on trials that lack the target. Such a
            # column means opposite things on the two sides.
            raise ValueError(
                f"Model {self.name!r}: {', '.join(clash)} is both predicted and fed back as an input. "
                "A model cannot read the class it is asked to place."
            )

    def columns(self) -> list[FeatureColumn]:
        """The catalog columns: one per pinned combination, plus the
        derivative columns the config asks for."""
        return enumerate_columns(self.features, self.derivatives)

    def label_columns(self) -> list[str]:
        """The label-input columns, appended after every catalog column."""
        return label_columns(self.label_inputs)

    def column_names(self) -> list[str]:
        """This model's whole input layout, in the order it is assembled."""
        return [c.name for c in self.columns()] + self.label_columns()

    @property
    def target_labels(self) -> list[int]:
        return list(self.targets)

    def target_name(self, label: int) -> str:
        return self.targets.get(int(label), str(label))

    def describe_targets(self) -> str:
        return ", ".join(f"{name} ({label})" for label, name in self.targets.items())


def models_root() -> Path:
    """The model store used while no project is open, ``~/.ethograph/defaults/runs/lightgbm``."""
    return defaults_dir("runs") / "lightgbm"


def model_dir(name: str) -> Path:
    return models_root() / name


def list_models() -> list[str]:
    """Names of all models that have a config on disk."""
    root = models_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / _CONFIG_FILE).is_file())


def save_config(config: OnsetModelConfig) -> Path:
    config.validate()
    d = model_dir(config.name)
    d.mkdir(parents=True, exist_ok=True)
    path = d / _CONFIG_FILE
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    return path


def _upgrade_config_dict(raw: dict) -> dict:
    """A stored config as the current dataclass expects it.

    Single-target models written before multi-target support carry
    ``target_label``/``target_name`` instead of ``targets``.
    """
    raw = dict(raw)
    if "targets" not in raw and "target_label" in raw:
        raw["targets"] = {int(raw.pop("target_label")): str(raw.pop("target_name", ""))}
    raw.pop("target_label", None)
    raw.pop("target_name", None)
    # A config that asked for the removed sequence CRF, or declared an
    # expected event order, still loads — those keys just no longer exist.
    raw.pop("use_crf", None)
    raw.pop("expected_order", None)
    raw.pop("expect_together", None)
    return raw


def load_config(name: str) -> OnsetModelConfig:
    raw = yaml.safe_load((model_dir(name) / _CONFIG_FILE).read_text(encoding="utf-8"))
    return OnsetModelConfig(**_upgrade_config_dict(raw))


def individual_dim(config: OnsetModelConfig) -> str | None:
    """The individual dim this config pins, in the spelling it was frozen with.

    ``None`` when no feature selects along one — a session with a single
    individual has nothing to choose, so the dim never reaches the config.
    """
    for dims in config.features.values():
        for dim in dims:
            if dim in INDIVIDUAL_DIMS:
                return dim
    return None


def config_individuals(config: OnsetModelConfig) -> list[str]:
    """Every individual *config* reads features from, in config order."""
    dim = individual_dim(config)
    if dim is None:
        return []
    out: list[str] = []
    for dims in config.features.values():
        for value in expand_dim_values(dims.get(dim, [])):
            if value not in out:
                out.append(value)
    return out


def retarget_individual(config: OnsetModelConfig, individual: str | None) -> OnsetModelConfig:
    """*config* with its individual dim re-pinned to *individual*.

    A classifier is fitted on numbers; the individual in ``features`` is only
    the key that selects those numbers out of a session. Re-pinning it is what
    lets a model trained on one animal read another's session, as long as the
    rig and the feature layout are the same — the column *order* is untouched,
    so the classifier still sees its own input layout.

    Its :attr:`~OnsetModelConfig.label_inputs` are re-pointed by the same
    rule, so a model that reads one animal's approach to time its peck reads
    the other animal's approach when run on them.

    Returns *config* unchanged when it pins no individual dim, when
    *individual* is empty, or when it **already reads** that individual — a
    model built on two animals at once (an actor and a partner) keeps reading
    both when asked for one of them, because that is the model it is; the
    combo then only says whose labels these are.

    Raises ``ValueError`` only for the undecidable case: a model reading
    several individuals, asked for one it does not read. Collapsing two
    columns onto one animal would hand the classifier the same data in the
    slots it learned as two different animals — wrong, and invisibly so.
    """
    if not individual:
        return config
    features = config.features
    dim = individual_dim(config)
    if dim is not None:
        pinned = config_individuals(config)
        if individual not in pinned:
            if len(pinned) > 1:
                raise ValueError(
                    f"Model {config.name!r} reads {len(pinned)} individuals ({', '.join(pinned)}), "
                    f"so it cannot be re-pinned to {individual!r} alone. "
                    "Train a model on this session instead."
                )
            features = {
                feature: {
                    d: ([individual] if d == dim else list(expand_dim_values(values))) for d, values in dims.items()
                }
                for feature, dims in config.features.items()
            }
    label_inputs = [inp.retarget(individual) for inp in config.label_inputs]
    if features is config.features and label_inputs == config.label_inputs:
        return config
    return replace(config, features=features, label_inputs=label_inputs)


def session_dir(name: str, session: str) -> Path:
    return model_dir(name) / _TRAIN_DIR / session


def list_sessions(name: str) -> dict[str, dict]:
    """Contributed training sessions: ``{session_id: meta dict}``."""
    train_root = model_dir(name) / _TRAIN_DIR
    if not train_root.is_dir():
        return {}
    out: dict[str, dict] = {}
    for p in sorted(train_root.iterdir()):
        meta_path = p / "meta.yaml"
        if meta_path.is_file():
            out[p.name] = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    return out


# ---------------------------------------------------------------------------
# Feature columns — the catalog-selection view of the config
# ---------------------------------------------------------------------------
# Defined once in ``ethograph.features.columns`` and shared with the
# segmentation pipeline; re-exported here so existing callers keep working.

_check_same_fs = check_same_fs

__all__ = [
    "FeatureColumn",
    "enumerate_columns",
    "extract_features",
    "extract_model_features",
    "sampling_rate",
]


def extract_model_features(
    loader,
    config: OnsetModelConfig,
    t0: float | None = None,
    t1: float | None = None,
    *,
    labels: pd.DataFrame | None = None,
    shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One trial's design inputs as *config* defines them: ``(time, (T, D))``.

    The one way a model's inputs are assembled — training-data collection and
    inference both go through it, so neither the derivative columns nor the
    label columns can be present on one side and missing on the other.

    *time* is on the loader's own clock. *labels* is this trial's label rows
    and *shift* the offset from that clock to the trial clock labels are stored
    on (``0`` for xarray, the trial start for pynapple), so the label columns
    are drawn where the events actually are.
    """
    time, data = extract_features(loader, config.features, t0, t1, config.derivatives)
    if not config.label_inputs:
        return time, data
    if labels is None:
        raise ValueError(
            f"Model {config.name!r} reads existing labels as inputs "
            f"({', '.join(i.name for i in config.label_inputs)}), but none were passed."
        )
    return time, np.column_stack([data, render_label_inputs(config.label_inputs, labels, time - shift)])


# ---------------------------------------------------------------------------
# Training-data storage
# ---------------------------------------------------------------------------


def write_trial_training_data(
    name: str,
    session: str,
    trial_id: int | str,
    time: np.ndarray,
    data: np.ndarray,
    y_times: dict[int, float],
) -> Path:
    """Persist one trial's assembled features + its event times under the model.

    *y_times* maps target label id to that event's time in this trial. A
    target the trial does not carry is simply absent: the trial then
    contributes nothing to that target's classifier, because an unlabelled
    trial is not evidence that the event never happened.
    """
    d = session_dir(name, session)
    d.mkdir(parents=True, exist_ok=True)
    safe_trial = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(trial_id))
    path = d / f"trial_{safe_trial}.npz"
    labels = sorted(y_times)
    np.savez_compressed(
        path,
        time=np.asarray(time, dtype=np.float64),
        data=np.asarray(data, dtype=np.float64),
        y_labels=np.asarray(labels, dtype=np.int64),
        y_times=np.asarray([y_times[label] for label in labels], dtype=np.float64),
        fs=np.float64(sampling_rate(time)),
    )
    return path


def read_trial_training_data(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, float], float]:
    """Read one stored training trial: ``(time, data, {label: y_time}, fs)``.

    A trial written before multi-target support stores a bare ``y_time`` with
    no label; it reads back under :data:`_LEGACY_TARGET`, which
    :func:`_resolve_y_time` maps onto a single-target model's only target.
    """
    with np.load(path) as npz:
        time = np.asarray(npz["time"], dtype=np.float64)
        data = np.asarray(npz["data"], dtype=np.float64)
        fs = float(npz["fs"])
        if "y_labels" in npz:
            y_times = {int(label): float(t) for label, t in zip(npz["y_labels"], npz["y_times"])}
        else:
            y_times = {_LEGACY_TARGET: float(npz["y_time"])}
    return time, data, y_times, fs


def _resolve_y_time(y_times: dict[int, float], label: int, config: OnsetModelConfig) -> float | None:
    """This trial's event time for *label*, or ``None`` if it carries none."""
    if label in y_times:
        return y_times[label]
    if _LEGACY_TARGET in y_times and len(config.targets) == 1:
        return y_times[_LEGACY_TARGET]
    return None


def write_session_meta(name: str, session: str, meta: dict) -> Path:
    d = session_dir(name, session)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "meta.yaml"
    path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Windowing + targets
# ---------------------------------------------------------------------------


def lag_offsets(fs: float, window_s: float, max_lags: int = MAX_LAGS) -> np.ndarray:
    """Sample offsets (in frames) of the centred window seen at each frame.

    At most *max_lags* taps, spread evenly across ``window_s`` — identical for
    training and inference because both derive it from (fs, window_s).
    """
    half = max(1, int(round(window_s * fs / 2)))
    n = min(2 * half + 1, max_lags)
    return np.unique(np.round(np.linspace(-half, half, n)).astype(int))


def build_windows(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per-frame design matrix ``(T, D * len(offsets))``.

    Frames whose window reaches past the trial edge get NaN taps —
    ``HistGradientBoostingClassifier`` handles NaN natively.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    n_time, n_dims = data.shape
    x = np.full((n_time, n_dims * len(offsets)), np.nan)
    for j, off in enumerate(offsets):
        src0, src1 = max(0, off), min(n_time, n_time + off)
        dst0, dst1 = max(0, -off), min(n_time, n_time - off)
        x[dst0:dst1, j * n_dims : (j + 1) * n_dims] = data[src0:src1]
    return x


def make_targets(
    time: np.ndarray,
    y_time: float,
    tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary per-frame target + Gaussian sample weight around the event.

    Frames within ``tolerance_s`` of the event are positive; their weight is a
    Gaussian bump (sigma = tolerance/2) peaking at the event, so the exact
    frame counts most. Negative frames get weight 1.
    """
    time = np.asarray(time, dtype=np.float64)
    delta = time - float(y_time)
    y = (np.abs(delta) <= tolerance_s).astype(np.int8)
    if not y.any():
        raise ValueError(
            f"The event at {y_time:.3f} s falls outside the feature time base ({time[0]:.3f}–{time[-1]:.3f} s)."
        )
    sigma = tolerance_s / 2.0
    weights = np.ones_like(time)
    pos = y.astype(bool)
    weights[pos] = np.exp(-0.5 * (delta[pos] / sigma) ** 2)
    return y, weights


def _subsample_mask(
    y: np.ndarray,
    delta: np.ndarray,
    tolerance_s: float,
    neg_per_pos: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep all positives, all hard negatives near the event, and a random
    subset of the remaining negatives (``neg_per_pos`` per positive)."""
    keep = y.astype(bool) | (np.abs(delta) <= 5.0 * tolerance_s)
    n_pos = int(y.sum())
    far_idx = np.flatnonzero(~keep)
    n_far = max(0, neg_per_pos * n_pos - int((keep & ~y.astype(bool)).sum()))
    if len(far_idx) > n_far:
        keep[rng.choice(far_idx, size=n_far, replace=False)] = True
    else:
        keep[far_idx] = True
    return keep


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------


def _iter_trial_files(name: str) -> list[Path]:
    train_root = model_dir(name) / _TRAIN_DIR
    if not train_root.is_dir():
        return []
    return sorted(train_root.glob("*/trial_*.npz"))


def _fit_target(
    label: int,
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    ws: list[np.ndarray],
    config: OnsetModelConfig,
) -> tuple[HistGradientBoostingClassifier, dict]:
    """Fit one target's binary classifier from its per-trial slices."""
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    w_train = np.concatenate(ws)
    # Balance the classes: scale positive weights so both classes carry the
    # same total mass (HistGradientBoostingClassifier has no class_weight).
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0:
        raise ValueError(f"Target {config.target_name(label)!r} has no positive frames in its training data.")
    w_train = w_train.copy()
    w_train[y_train == 1] *= n_neg / max(1, n_pos)

    clf = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        random_state=_RNG_SEED,
    )
    clf.fit(x_train, y_train, sample_weight=w_train)
    return clf, {"n_trials": len(xs), "n_frames": len(y_train), "n_positive": n_pos}


class TrainingTrial(NamedTuple):
    """One stored training trial, as training and cross-fitting see it."""

    time: np.ndarray
    data: np.ndarray
    y_times: dict[int, float]
    fs: float


def load_training_trials(name: str, config: OnsetModelConfig) -> list[TrainingTrial]:
    """Every stored trial of a model, checked to share one sampling rate."""
    trials: list[TrainingTrial] = []
    fs_ref: float | None = None
    for path in _iter_trial_files(name):
        time, data, y_times, fs = read_trial_training_data(path)
        if fs_ref is None:
            fs_ref = fs
        else:
            _check_same_fs(fs_ref, fs, f"training trial {path.name!r}")
        trials.append(TrainingTrial(time, data, y_times, fs))
    return trials


def _fit_targets(
    trials: list[TrainingTrial],
    offsets: np.ndarray,
    config: OnsetModelConfig,
    targets: list[int] | None = None,
) -> tuple[dict[int, HistGradientBoostingClassifier], dict[int, dict]]:
    """Fit one binary classifier per target over *trials*.

    The design matrix is built once per trial and reused for every target —
    only the binary labels and the negative subsampling differ.
    """
    rng = np.random.default_rng(_RNG_SEED)
    buckets: dict[int, tuple[list, list, list]] = {
        label: ([], [], []) for label in (targets if targets is not None else config.targets)
    }
    for trial in trials:
        x = build_windows(trial.data, offsets)
        for label, (xs, ys, ws) in buckets.items():
            y_time = _resolve_y_time(trial.y_times, label, config)
            if y_time is None:
                continue
            y, w = make_targets(trial.time, y_time, config.tolerance_s)
            keep = _subsample_mask(y, trial.time - y_time, config.tolerance_s, config.neg_per_pos, rng)
            xs.append(x[keep])
            ys.append(y[keep])
            ws.append(w[keep])

    models: dict[int, HistGradientBoostingClassifier] = {}
    per_target: dict[int, dict] = {}
    for i, (label, (xs, ys, ws)) in enumerate(buckets.items(), start=1):
        if not xs:
            raise ValueError(
                f"No training trial carries {config.target_name(label)!r} — add a session "
                "that labels it, or create the model without that target."
            )
        logger.info("Fitting target %d/%d: %s...", i, len(buckets), config.target_name(label))
        models[label], per_target[label] = _fit_target(label, xs, ys, ws, config)
        stats = per_target[label]
        logger.info(
            "  %s: %d trials, %d frames, %d positive",
            config.target_name(label),
            stats["n_trials"],
            stats["n_frames"],
            stats["n_positive"],
        )
    return models, per_target


def train_model(name: str) -> dict:
    """Fit one classifier per target.

    Returns a summary dict: ``n_sessions``, ``n_trials``, ``targets`` (each
    label's ``n_trials``/``n_frames``/``n_positive`` plus its held-out record).
    Raises ``ValueError`` when there is no training data, a target has no
    labelled trial, or the stored trials disagree on sampling rate.
    """
    config = load_config(name)
    if not config.targets:
        raise ValueError(f"Model {name!r} has no target point events configured.")
    trials = load_training_trials(name, config)
    if not trials:
        raise ValueError("No training data yet — add at least one session first.")

    logger.info("Training model %r on %d trials, %d target(s)...", name, len(trials), len(config.targets))
    fs_ref = trials[0].fs
    offsets = lag_offsets(fs_ref, config.window_s)
    models, per_target = _fit_targets(trials, offsets, config)

    # What the model scores on trials it did not see: the model's report card.
    logger.info("Scoring held-out trials...")
    cross_fitted = _cross_fit_curves(trials, offsets, config)
    calibration = fit_confidence_calibration(trials, cross_fitted, config)

    bundle = {
        "models": models,
        "fs": fs_ref,
        "columns": config.column_names(),
        "config": asdict(config),
        "calibration": {label: cal.to_dict() for label, cal in calibration.items()},
    }
    summary = {
        "n_sessions": len(list_sessions(name)),
        "n_trials": len(trials),
        "targets": per_target,
        "calibration": {label: cal.to_dict() for label, cal in calibration.items()},
    }
    for label, stats in per_target.items():
        cal = calibration.get(label)
        if cal is not None:
            stats["held_out"] = f"{cal.n_hits}/{cal.n_trials}"
            stats["confidence_ceiling"] = cal.hit_rate
            stats["confidence_statistic"] = cal.statistic
    joblib.dump(bundle, model_dir(name) / _MODEL_FILE)
    logger.info("Training done: %r", name)
    return summary


def is_trained(name: str) -> bool:
    return (model_dir(name) / _MODEL_FILE).is_file()


def load_bundle(name: str) -> dict:
    """The trained bundle: ``{"models", "fs", "columns", "config"}``.

    A bundle written by a single-target model (one ``"model"``) is presented
    as a one-entry ``"models"`` mapping, so callers only see one shape.
    """
    path = model_dir(name) / _MODEL_FILE
    if not path.is_file():
        raise ValueError(f"Model {name!r} has not been trained yet.")
    bundle = joblib.load(path)
    bundle["config"] = _upgrade_config_dict(bundle["config"])
    if "models" not in bundle:
        bundle["models"] = {OnsetModelConfig(**bundle["config"]).target_labels[0]: bundle.pop("model")}
    bundle["models"] = {int(label): clf for label, clf in bundle["models"].items()}
    # A bundle from when the sequence model existed still loads; its CRF is
    # simply not consulted, because nothing decodes an order any more.
    bundle.pop("crf", None)
    # Stored as plain dicts so a bundle survives this dataclass changing shape.
    bundle["calibration"] = {
        int(label): TargetCalibration(**cal) for label, cal in (bundle.get("calibration") or {}).items()
    }
    return bundle


def bundle_config(bundle: dict) -> OnsetModelConfig:
    """The config a trained bundle was fitted with."""
    return OnsetModelConfig(**_upgrade_config_dict(bundle["config"]))


def _predictive_config(config: OnsetModelConfig) -> tuple:
    """The parts of a config that decide what a *trained* model reads.

    The name is left out on purpose: a model folder copied under a new name
    predicts exactly as its original did.
    """
    return (
        config.features,
        config.derivatives,
        config.label_inputs,
        config.targets,
        config.window_s,
        config.tolerance_s,
    )


def config_drifted(name: str) -> OnsetModelConfig | None:
    """The trained bundle's config, when ``config.yaml`` no longer matches it.

    ``None`` when the model is untrained or the two agree. Hand-editing
    ``config.yaml`` is the obvious-looking way to point a model at another
    animal's session and it does nothing — the bundle carries the layout the
    classifiers were fitted on. Returning the trained config lets the Predict
    dialog say which layout will actually be read, instead of leaving the run
    to fail once per trial.
    """
    if not is_trained(name):
        return None
    trained = bundle_config(load_bundle(name))
    return trained if _predictive_config(trained) != _predictive_config(load_config(name)) else None


def _smoothing_kernel(tolerance_s: float, fs: float) -> np.ndarray:
    """Gaussian kernel matching the training tolerance, normalised to sum 1."""
    sigma = max(1.0, tolerance_s * fs / 2.0)
    half = int(np.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    return kernel / kernel.sum()


#: The one reading of a curve every consumer shares (re-exported).
__all__ += ["tallest_peak"]


def read_curve(curve: np.ndarray, fs: float, tolerance_s: float) -> CurveStats:
    """The curve's statistics around its tallest peak.

    The focus window is the model's own ``tolerance_s`` twice over
    (:func:`~ethograph.labels.curve_confidence.focus_window_s`) — the
    timescale the user declared, in this clock's samples.
    """
    return curve_stats(curve, window_samples(focus_window_s(tolerance_s), fs))


def _logit(p) -> np.ndarray | float:
    """Log-odds, clipped off the asymptotes so a 0 or 1 reading stays finite."""
    p = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class TargetCalibration:
    """What one target's held-out predictions said about its reliability.

    This is a verdict on the **model**, reported when it trains — not a
    transform applied to any label. Cross-fitting
    (:func:`fit_confidence_calibration`) predicts every training trial with
    classifiers that never saw it and counts the predictions landing within
    ``tolerance_s`` of the labelled event, so a model that gets a third of its
    held-out trials right says so out loud however loud its curves are. A
    label's own ``confidence`` stays the height of its curve's tallest peak,
    which is what the user sees drawn and thresholds by eye.

    ``slope``/``intercept`` are a Platt fit of that peak height against the
    hits, kept so the mapping is on record; nothing reads them today.

    ``statistic`` is the one thing here that *is* applied to a label: which
    reading of the curve is written as its confidence. Chosen on the held-out
    record (:func:`~ethograph.labels.curve_confidence.choose_statistic`) —
    ``peak`` unless a shape statistic separates hits from misses clearly
    better; ``aucs`` is the record of every candidate, so the choice can be
    read.
    """

    #: Held-out predictions scored, and how many landed within the tolerance.
    n_trials: int
    n_hits: int
    #: Platt scaling of the raw reading's log-odds (identity: 1, 0).
    slope: float
    intercept: float
    #: The curve statistic written as this target's confidence.
    statistic: str = "peak"
    #: AUC per candidate statistic on the held-out record.
    aucs: dict[str, float] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        """Laplace-smoothed held-out accuracy — the model's report card.

        A model right on 8 of 8 held-out trials scores 9/10, not 1.0: eight
        trials cannot support certainty. A model with nothing held out at all
        (one training trial) reads 1/2 — nothing is known about how often it
        is right, and that is what a half says.
        """
        return (self.n_hits + 1) / (self.n_trials + 2)

    def apply(self, raw: float) -> float:
        """The stored mapping of a peak height onto P(hit), capped at the
        record. Kept on record; a label's ``confidence`` does not go through
        it — see this class's own docstring."""
        return float(min(_sigmoid(self.slope * float(_logit(raw)) + self.intercept), self.hit_rate))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OnsetPrediction:
    """One target's predicted event in one trial."""

    label: int
    name: str
    time: float
    #: The number written to the label's ``confidence`` column: the curve's
    #: ``statistic`` around its tallest peak, readable off the drawn curve.
    confidence: float
    statistic: str = "peak"


def target_curves(
    models: dict[int, HistGradientBoostingClassifier],
    x: np.ndarray,
    kernel: np.ndarray,
) -> dict[int, np.ndarray]:
    """Each target's smoothed per-frame event probability over a trial.

    Smoothing with the training tolerance makes a plateau of near-hits beat a
    single spurious spike.
    """
    return {label: np.convolve(clf.predict_proba(x)[:, 1], kernel, mode="same") for label, clf in models.items()}


class TrialPrediction(NamedTuple):
    """One trial's predicted events **and** the curves they were read off.

    The curves are what the events' confidences compress: keeping them lets
    the GUI show *why* a score is low (a rival moment) instead of only that
    it is. See :mod:`ethograph.labels.onset_curves`.
    """

    events: dict[int, OnsetPrediction]
    #: label -> smoothed per-frame event probability, on the caller's clock.
    curves: dict[int, np.ndarray]


def predict_events(
    bundle: dict,
    time: np.ndarray,
    data: np.ndarray,
) -> dict[int, OnsetPrediction]:
    """The events :func:`predict_trial` predicts, for callers wanting only those."""
    return predict_trial(bundle, time, data).events


def predict_trial(
    bundle: dict,
    time: np.ndarray,
    data: np.ndarray,
) -> TrialPrediction:
    """Predict one event per target in one trial's assembled features.

    Times are on *time*'s clock. Every target is read independently off its
    own smoothed probability curve: the tallest peak is the event, and that
    peak's height is the confidence. No class's evidence is allowed to move
    another's.
    """
    config = bundle_config(bundle)
    fs = sampling_rate(time)
    _check_same_fs(float(bundle["fs"]), fs, "this session's features")

    offsets = lag_offsets(float(bundle["fs"]), config.window_s)
    x = build_windows(data, offsets)
    curves = target_curves(bundle["models"], x, _smoothing_kernel(config.tolerance_s, fs))

    calibration = bundle.get("calibration") or {}
    events: dict[int, OnsetPrediction] = {}
    for label, curve in curves.items():
        stats = read_curve(curve, fs, config.tolerance_s)
        cal = calibration.get(label)
        statistic = cal.statistic if cal is not None else "peak"
        events[label] = OnsetPrediction(
            label=label,
            name=config.target_name(label),
            time=float(time[stats.index]),
            confidence=stats.statistic(statistic),
            statistic=statistic,
        )
    return TrialPrediction(events, curves)


def _cross_fit_curves(
    trials: list[TrainingTrial],
    offsets: np.ndarray,
    config: OnsetModelConfig,
) -> list[dict[int, np.ndarray]]:
    """Out-of-fold probability curves, one dict per trial.

    A classifier scores its own training trials almost perfectly, so reading
    its record off them would flatter it. Each trial is therefore scored by
    classifiers that never saw it: the trials are split into folds and each
    fold is scored by a model fitted on the rest.
    """
    n_splits = min(_CV_FOLDS, len(trials))
    curves: list[dict[int, np.ndarray]] = [{} for _ in trials]
    for fold in range(n_splits):
        logger.info("  cross-fitting fold %d/%d...", fold + 1, n_splits)
        held = [i for i in range(len(trials)) if i % n_splits == fold]
        rest = [trials[i] for i in range(len(trials)) if i % n_splits != fold]
        covered = [
            label
            for label in config.targets
            if any(_resolve_y_time(trial.y_times, label, config) is not None for trial in rest)
        ]
        missing = [label for label in config.targets if label not in covered]
        if missing:
            # Too few labelled trials for this target to be held out fairly.
            # The sequence model still trains; its emissions for these trials
            # are simply the ones the final model produces.
            logger.warning(
                "Cross-fitting fold %d has no training trial for %s — those curves are not out-of-fold.",
                fold,
                ", ".join(config.target_name(label) for label in missing),
            )
        models, _ = _fit_targets(rest, offsets, config, targets=covered) if covered else ({}, {})
        fallback, _ = _fit_targets(trials, offsets, config, targets=missing) if missing else ({}, {})
        for i in held:
            trial = trials[i]
            x = build_windows(trial.data, offsets)
            kernel = _smoothing_kernel(config.tolerance_s, trial.fs)
            curves[i] = target_curves({**models, **fallback}, x, kernel)
    return curves


def _fit_platt(raws: np.ndarray, hits: np.ndarray) -> tuple[float, float]:
    """Map a reading's log-odds to the probability of a hit.

    With too few held-out predictions, or only one outcome among them, there
    is nothing to fit a slope through: the map is the identity and the
    ceiling (:attr:`TargetCalibration.hit_rate`) carries the honesty on its
    own. Fitting is what sharpens a good model's readings apart; the identity
    keeps them in the order the curves put them in.
    """
    if raws.size < _MIN_CALIBRATION_TRIALS or np.unique(hits).size < 2:
        return 1.0, 0.0
    fit = LogisticRegression(C=_CALIBRATION_C)
    fit.fit(np.asarray(_logit(raws)).reshape(-1, 1), hits)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def fit_confidence_calibration(
    trials: list[TrainingTrial],
    curves: list[dict[int, np.ndarray]],
    config: OnsetModelConfig,
) -> dict[int, TargetCalibration]:
    """Each target's held-out record, read off the cross-fitted curves.

    A prediction is a **hit** when it lands within ``tolerance_s`` of the
    labelled event — the same tolerance the classifier was trained to, and the
    one the review nudges within. This says how good the *model* is; a
    label's own ``confidence`` is its curve's peak height and is not touched
    by what is fitted here.
    """
    calibrations: dict[int, TargetCalibration] = {}
    for label in config.targets:
        readings: list[CurveStats] = []
        hits: list[int] = []
        for trial, curve in zip(trials, curves):
            y_time = _resolve_y_time(trial.y_times, label, config)
            if y_time is None or label not in curve:
                continue
            stats = read_curve(curve[label], trial.fs, config.tolerance_s)
            readings.append(stats)
            hits.append(int(abs(float(trial.time[stats.index]) - y_time) <= config.tolerance_s))
        raw_arr = np.asarray([s.peak for s in readings], dtype=np.float64)
        hit_arr = np.asarray(hits, dtype=int)
        slope, intercept = _fit_platt(raw_arr, hit_arr)
        aucs = rank_statistics(readings, hit_arr.astype(bool)) if readings else {}
        calibrations[label] = TargetCalibration(
            n_trials=int(hit_arr.size),
            n_hits=int(hit_arr.sum()),
            slope=slope,
            intercept=intercept,
            statistic=choose_statistic(aucs) if aucs else "peak",
            aucs={k: (float(v) if np.isfinite(v) else float("nan")) for k, v in aucs.items()},
        )
        logger.info(
            "  %s: %d/%d held-out predictions within %.3f s (confidence ceiling %.2f); confidence = %s",
            config.target_name(label),
            calibrations[label].n_hits,
            calibrations[label].n_trials,
            config.tolerance_s,
            calibrations[label].hit_rate,
            DESCRIPTIONS[calibrations[label].statistic],
        )
    return calibrations
