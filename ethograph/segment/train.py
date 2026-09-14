"""Train: fit an architecture for a fixed epoch budget.

A run never overwrites another: each call to :func:`train` creates its own
directory, the configured (or derived) run name plus the creation timestamp
to the minute (``{run_name}_{YYYYmmdd-HHMM}``, deduplicated with a ``-2``,
``-3``, ... suffix on a same-minute collision). A run directory
``{root}/runs/{run_name}_{timestamp}/`` holds everything needed to apply or
compare the run::

    config.yaml         # the resolved config this run was trained with
    columns.yaml        # the input layout (copied from the materialised dataset)
    classes.yaml        # class index ↔ label id
    stats.npz           # normalisation statistics of the training samples
    splits/*.bundle     # which sample keys the split drew into which role
    metrics.tsv         # one row per validation: val metrics (selects best.pt) + test raw/postprocessed
    best.pt / last.pt   # best validation epoch / the final epoch
    test_metrics.yaml   # test metrics of best.pt, raw and post-processed
    test_eval.npz       # matched-segment IoUs + onset/offset deltas behind those metrics
    eval.pdf            # class-wise F1 + boundary-delta histograms

Checkpoint selection (``best.pt``) is keyed on validation only, never test —
the periodic test readout in ``metrics.tsv`` is a training-time diagnostic,
not a selection signal, so ``test_metrics.yaml`` stays a clean held-out
report of whichever epoch validation actually picked.
"""

from __future__ import annotations

import logging
import random
import shutil
import time as _time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from ethograph.segment.config import PostprocessConfig, SegmentConfig, TrainConfig, save_config
from ethograph.segment.dataset import MaterialisedStore, SampleDataset, collate
from ethograph.segment.losses import build_objective
from ethograph.segment.materialise import COLUMNS_FILE, materialise, read_target_table, resolved_config
from ethograph.segment.metrics import (
    EVAL_ARRAYS_FILE,
    METRICS_FILE,
    TEST_METRICS_FILE,
    evaluate,
    flatten_channels,
    save_eval_arrays,
    scalar_metrics,
)
from ethograph.segment.models import as_output, build_model
from ethograph.segment.postprocess import postprocess_channels, postprocess_dense
from ethograph.segment.preprocess import NormStats
from ethograph.segment.samples import ChannelTable, TargetTable, is_multilabel
from ethograph.utils.device import resolve_device
from ethograph.utils.logging import log_to_file

logger = logging.getLogger(__name__)

BEST_FILE = "best.pt"
LAST_FILE = "last.pt"
STATS_FILE = "stats.npz"


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


def _trial_split(index: pd.DataFrame, keys: list[str]) -> dict[str, tuple[str, str]]:
    """Sample key → (source, trial), for grouping a random split by whole trial."""
    by_key = index.set_index("key")
    return {k: (str(by_key.loc[k, "source"]), str(by_key.loc[k, "trial"])) for k in keys}


def _draw(trials: list[tuple[str, str]], fractions: dict[str, float], seed: int) -> dict[tuple[str, str], str]:
    """Deal whole trials into roles by fraction, deterministically.

    Every fraction gets its rounded share off the front of one shuffle, and
    whatever is left over is ``train`` — so a shortage is always taken out of
    the largest pool rather than silently emptying ``val``.
    """
    order = sorted(trials)
    random.Random(seed).shuffle(order)
    out: dict[tuple[str, str], str] = {}
    cut = 0
    for role in ("test", "val"):
        n = int(round(len(order) * fractions.get(role, 0.0)))
        for trial in order[cut : cut + n]:
            out[trial] = role
        cut += n
    for trial in order[cut:]:
        out[trial] = "train"
    return out


def assign_roles(config: SegmentConfig, index: pd.DataFrame) -> dict[str, str]:
    """Sample key → role, drawn by whole trial from ``train.split``.

    Two shapes, and the config says which by whether it names holdouts:

    * **Ratios** (the default) — every trial of every session is pooled,
      shuffled once with ``split.seed`` and cut 60/20/20 (or whatever the
      three fractions say). This is stage 1: ``val`` is the objective an
      Optuna search maximises, ``test`` the number you report.
    * **Held-out sessions** — every trial of a session named in
      ``split.holdout_sessions`` is ``test``, whatever the fractions say, and
      the remaining sessions are split train/val by ``val_fraction``
      renormalised against ``train_fraction``. This is stage 2: one
      cross-validation fold, written by
      :meth:`~ethograph.segment.project.Project.cross_validate`.
    * **Held-out trials** — the same, one level down: every sample of a trial
      id named in ``split.holdout_trials`` is ``test``, in every session.
      A trial-level fold, for the one-session project whose sessions cannot
      be held out; naming a trial no session has is an error, since a fold
      that holds out nothing would score the training set.
    """
    split = config.train.split
    holdout = {str(p) for p in split.holdout_sessions}
    holdout_trials = set(split.holdout_trials)
    trial_of = _trial_split(index, list(index["key"]))
    roles: dict[str, str] = {}

    if holdout_trials:
        known = {trial for _, trial in trial_of.values()}
        missing = sorted(holdout_trials - known, key=str)
        if missing:
            raise ValueError(
                f"train.split.holdout_trials names {missing}, which no session's materialised trials include "
                f"(they are {sorted(known, key=str)})"
            )
    if holdout or holdout_trials:
        rest = []
        for key, (source, trial) in trial_of.items():
            if source in holdout or trial in holdout_trials:
                roles[key] = "test"
            else:
                rest.append(key)
        pool = split.train_fraction + split.val_fraction
        val_share = (split.val_fraction / pool) if pool else 0.0
        by_trial = _draw(sorted({trial_of[k] for k in rest}), {"val": val_share}, split.seed)
        for key in rest:
            roles[key] = by_trial[trial_of[key]]
        return roles

    by_trial = _draw(
        sorted(set(trial_of.values())),
        {"test": split.test_fraction, "val": split.val_fraction},
        split.seed,
    )
    for key, trial in trial_of.items():
        roles[key] = by_trial[trial]
    return roles


def write_bundles(run_dir: Path, roles: dict[str, str]) -> None:
    splits = run_dir / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    for role in ("train", "val", "test"):
        keys = [k for k, r in roles.items() if r == role]
        (splits / f"{role}.bundle").write_text("".join(f"{k}\n" for k in keys), encoding="utf-8")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    run_dir: Path
    best_epoch: int
    best_score: float
    #: Wall-clock time spent in the training loop (excludes materialisation, final test eval, plotting).
    train_seconds: float
    test_metrics: dict[str, Any] | None


def run_name_for(config: SegmentConfig) -> str:
    """The configured (or derived) base run name — before the timestamp :func:`train` appends."""
    if config.train.run_name:
        return config.train.run_name
    return f"{config.model.architecture}_{config.features.name}"


def _new_run_dir(config: SegmentConfig, base_name: str) -> Path:
    """A fresh run directory for *base_name*, never an existing one.

    ``{base_name}_{YYYYmmdd-HHMM}``; a same-minute collision (e.g. a
    benchmark loop) gets ``-2``, ``-3``, ... appended.
    """
    stamped = f"{base_name}_{datetime.now():%Y%m%d-%H%M}"
    run_dir = config.run_dir(stamped)
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = config.run_dir(f"{stamped}-{suffix}")
    return run_dir


@dataclass
class DensePredictions:
    """What one pass over a loader produced, per sample key."""

    pred: dict[str, np.ndarray]
    gt: dict[str, np.ndarray]
    conf: dict[str, np.ndarray]


def _predict_dense(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float | None = None
) -> DensePredictions:
    """Predictions, ground truth and confidence per sample.

    Exclusive (*threshold* ``None``): argmax class indices ``(T,)`` and the
    winning probability. Multi-label: ``(C, T)`` 0/1 — each channel's sigmoid
    against *threshold* — and ``conf`` holds the ``(C, T)`` probabilities
    themselves, which the per-track post-processing reads to break ties.
    """
    model.eval()
    out = DensePredictions({}, {}, {})
    with torch.no_grad():
        for x, y, mask, _candidates, keys in loader:
            result = as_output(model(x.to(device), mask.to(device)))
            if threshold is None:
                probs = torch.softmax(result.logits[-1], dim=1)
                p_max, p_arg = probs.max(dim=1)
                for i, key in enumerate(keys):
                    n = int(mask[i, 0].sum().item())
                    out.pred[key] = p_arg[i, :n].cpu().numpy()
                    out.conf[key] = p_max[i, :n].cpu().numpy()
                    out.gt[key] = y[i, :n].numpy()
            else:
                probs = torch.sigmoid(result.logits[-1])
                for i, key in enumerate(keys):
                    n = int(mask[i, 0].sum().item())
                    p = probs[i, :, :n].cpu().numpy()
                    out.pred[key] = (p > threshold).astype(np.int64)
                    out.conf[key] = p
                    out.gt[key] = y[i, :, :n].numpy()
    model.train()
    return out


def _postprocess_all(
    dense: DensePredictions, fs: float, classes: TargetTable, cfg: PostprocessConfig
) -> dict[str, np.ndarray]:
    """Post-process every sample of *dense*."""
    if isinstance(classes, ChannelTable):
        return {
            key: postprocess_channels(value, fs, classes, cfg, probs=dense.conf.get(key))
            for key, value in dense.pred.items()
        }
    return {key: postprocess_dense(value, fs, classes, cfg) for key, value in dense.pred.items()}


def evaluate_dense(
    classes: TargetTable, gt: dict[str, np.ndarray], pred: dict[str, np.ndarray], thresholds: list[float], fs: float
) -> dict[str, Any]:
    """:func:`evaluate`, with a multi-label run's channels flattened into exclusive samples first."""
    if is_multilabel(classes):
        return evaluate(flatten_channels(gt), flatten_channels(pred), thresholds, fs)
    return evaluate(gt, pred, thresholds, fs)


def _frame_accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> tuple[int, int]:
    """``(correct, total)`` frames of one batch — per class channel for a multi-label target."""
    frame_mask = mask[:, 0, :] > 0
    if y.dim() == 3:
        predicted = logits > 0
        hit = (predicted == (y == 1)) & frame_mask.unsqueeze(1)
        return int(hit.sum()), int(frame_mask.sum()) * int(y.shape[1])
    predicted = logits.argmax(dim=1)
    return int(((predicted == y) & frame_mask).sum()), int(frame_mask.sum())


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: SegmentConfig, on_eval: Callable[[int, float], None] | None = None) -> RunResult:
    """Train one run; materialises the dataset first if it is missing.

    *on_eval* is called with ``(epoch, validation score)`` after every
    validation, before the next epoch starts. It exists for a hyperparameter
    search to report the curve and abandon a hopeless trial — raising from it
    (Optuna's ``TrialPruned``) stops the run where it stands, leaving the run
    directory as a record of how far it got. Nothing else uses it.
    """
    data_dir = config.data_dir
    if not (data_dir / COLUMNS_FILE).is_file():
        logger.info("No materialised dataset at %s — materialising first", data_dir)
        materialise(config)
    config = resolved_config(config)
    tcfg = config.train
    store = MaterialisedStore.open(data_dir, tcfg.subsample)
    if tcfg.subsample > 1:
        logger.info(
            "train.subsample=%d: %.4g Hz -> %.4g Hz",
            tcfg.subsample,
            store.layout.fs * tcfg.subsample,
            store.layout.fs,
        )
    _seed(tcfg.seed)
    device = torch.device(resolve_device(tcfg.device))

    run_dir = _new_run_dir(config, run_name_for(config))
    run_dir.mkdir(parents=True)
    with log_to_file(run_dir / "train.log"):
        return _train_run(config, tcfg, store, device, run_dir, on_eval)


def _train_run(
    config: SegmentConfig,
    tcfg: TrainConfig,
    store: MaterialisedStore,
    device: torch.device,
    run_dir: Path,
    on_eval: Callable[[int, float], None] | None = None,
) -> RunResult:
    save_config(config, run_dir / "config.yaml")
    shutil.copy(config.data_dir / COLUMNS_FILE, run_dir / COLUMNS_FILE)
    (run_dir / "classes.yaml").write_text(yaml.safe_dump(store.classes.to_dict(), sort_keys=False), encoding="utf-8")

    roles = assign_roles(config, store.index)
    keys = {role: [k for k, r in roles.items() if r == role] for role in ("train", "val", "test")}
    if not keys["train"]:
        raise ValueError(
            "No training samples — train.split left nothing in 'train' (or trials.where filtered every trial out)."
        )
    labelled = set(store.index.loc[store.index["n_labelled"] > 0, "key"])
    unlabelled_train = [k for k in keys["train"] if k not in labelled]
    if unlabelled_train:
        logger.warning(
            "%d of %d training samples carry no curated label for this branch (all background)",
            len(unlabelled_train),
            len(keys["train"]),
        )
    write_bundles(run_dir, roles)
    logger.info("Samples — train: %d, val: %d, test: %d", *(len(keys[r]) for r in ("train", "val", "test")))

    # An ablation drops whole feature categories from the columns the model
    # sees; the materialised dataset is untouched, so `drop_kinds=[video_feature]`
    # costs a run rather than a re-materialisation.
    if tcfg.drop_kinds and not any(store.layout.kinds):
        raise ValueError(
            f"train.drop_kinds={tcfg.drop_kinds} but no column of {config.data_dir} declares a kind, "
            "so the ablation would silently train the full model. Describe the session's variables "
            "and materialise again (see docs/source/advanced/variable_schema.md)."
        )
    keep = store.layout.keep_mask(tcfg.drop_kinds)
    layout = store.layout if keep.all() else store.layout.subset(keep)
    keep_mask = None if keep.all() else keep
    if tcfg.drop_kinds and keep_mask is None:
        logger.warning(
            "train.drop_kinds=%s dropped nothing — the columns declare %s",
            tcfg.drop_kinds,
            sorted({k for k in store.layout.kinds if k}),
        )
    if keep_mask is not None:
        logger.info("Ablation %s: %d of %d columns kept", tcfg.drop_kinds, layout.n_features, store.layout.n_features)
        if layout.n_features == 0:
            raise ValueError(f"train.drop_kinds={tcfg.drop_kinds} drops every column — nothing left to train on.")

    train_raw = [store.load(k) for k in keys["train"]]
    train_x = [x if keep_mask is None else x[keep_mask] for x, _ in train_raw]
    if config.features.preprocess.zscore:
        stats = NormStats.compute(train_x, np.asarray(layout.normalise))
    else:
        stats = NormStats.identity(layout.n_features)
    stats.save(run_dir / STATS_FILE)

    n_outputs = store.classes.n_outputs
    exclusive = not is_multilabel(store.classes)
    threshold = None if exclusive else float(config.infer.threshold)
    model = build_model(config.model.architecture, config.model.params, layout.n_features, n_outputs).to(device)
    objective, loss_settings = build_objective(config, n_outputs, layout=store.layout, exclusive=exclusive)
    objective = objective.to(device)
    logger.info("Objective: %s", yaml.safe_dump(loss_settings, sort_keys=False, default_flow_style=True).strip())
    # Constant learning rate, as upstream trains these models. A schedule would
    # have to key off the validation metric to mean anything: annealing on the
    # training loss reads "the model has memorised the training trials" as
    # "learning has stalled", which are opposite situations.
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)

    def _dataset(subset: list[str], augment_cfg=None) -> SampleDataset:
        return SampleDataset(store, subset, stats, augment_cfg, seed=tcfg.seed, keep=keep_mask, layout=layout)

    train_loader = DataLoader(
        _dataset(keys["train"], tcfg.augment),
        batch_size=tcfg.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(_dataset(keys["val"]), batch_size=1, collate_fn=collate) if keys["val"] else None
    if val_loader is None:
        if config.train.split.val_fraction:
            logger.warning(
                "train.split.val_fraction=%g but the draw produced no validation trials (%d trials in total) — "
                "no metrics curve, and best.pt is the last epoch.",
                config.train.split.val_fraction,
                len(store.index),
            )
        else:
            logger.warning("No validation samples — no metrics curve, and best.pt is the last epoch.")
    test_loader = DataLoader(_dataset(keys["test"]), batch_size=1, collate_fn=collate) if keys["test"] else None

    metrics_rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_epoch = 0
    select_on = tcfg.select_on
    t_start = _time.time()
    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_parts: dict[str, float] = {}
        correct = total = 0
        for x, y, mask, candidates, _ in train_loader:
            x, y, mask, candidates = x.to(device), y.to(device), mask.to(device), candidates.to(device)
            optimizer.zero_grad()
            output = as_output(model(x, mask))
            loss, parts = objective(output, y, candidates)
            loss.backward()
            if tcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tcfg.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())
            for name, value in parts.items():
                epoch_parts[name] = epoch_parts.get(name, 0.0) + value
            hit, seen = _frame_accuracy(output.logits[-1], y, mask)
            correct += hit
            total += seen
        n_batches = max(len(train_loader), 1)
        epoch_loss /= n_batches
        epoch_parts = {k: v / n_batches for k, v in epoch_parts.items() if k != "total"}
        logger.info(
            "[epoch %d] loss %.4f (%s)  train acc %.1f%%",
            epoch,
            epoch_loss,
            ", ".join(f"{k} {v:.4f}" for k, v in sorted(epoch_parts.items())) or "frame only",
            100 * correct / max(total, 1),
        )

        is_last = epoch == tcfg.epochs
        if val_loader is not None and (epoch % tcfg.eval_every == 0 or is_last):
            val = _predict_dense(model, val_loader, device, threshold)
            m = evaluate_dense(store.classes, val.gt, val.pred, tcfg.f1_thresholds, store.layout.fs)
            if select_on not in m:
                raise ValueError(
                    f"train.select_on={select_on!r} is not a metric; available: {sorted(scalar_metrics(m))}"
                )
            row = {"epoch": epoch, "loss": epoch_loss, **scalar_metrics(m)}
            score = float(m[select_on])
            if score > best_score:
                best_score, best_epoch = score, epoch
                torch.save(model.state_dict(), run_dir / BEST_FILE)
                logger.info("  new best.pt (val %s = %.2f)", select_on, score)

            if test_loader is not None:
                held = _predict_dense(model, test_loader, device, threshold)
                test_raw = evaluate_dense(store.classes, held.gt, held.pred, tcfg.f1_thresholds, store.layout.fs)
                test_processed_pred = _postprocess_all(held, store.layout.fs, store.classes, config.infer.postprocess)
                test_processed = evaluate_dense(
                    store.classes, held.gt, test_processed_pred, tcfg.f1_thresholds, store.layout.fs
                )
                row.update({f"test_raw_{k}": v for k, v in scalar_metrics(test_raw).items()})
                row.update({f"test_post_{k}": v for k, v in scalar_metrics(test_processed).items()})
                logger.info(
                    "  test %s = %.2f -> %.2f postprocessed  (acc %.1f -> %.1f, edit %.1f -> %.1f)",
                    select_on,
                    test_raw[select_on],
                    test_processed[select_on],
                    test_raw["acc"],
                    test_processed["acc"],
                    test_raw["edit"],
                    test_processed["edit"],
                )

            metrics_rows.append(row)
            pd.DataFrame(metrics_rows).to_csv(run_dir / METRICS_FILE, sep="\t", index=False)
            if on_eval is not None:
                on_eval(epoch, score)
    torch.save(model.state_dict(), run_dir / LAST_FILE)
    if not (run_dir / BEST_FILE).is_file():
        shutil.copy(run_dir / LAST_FILE, run_dir / BEST_FILE)
        best_epoch = epoch
    train_seconds = _time.time() - t_start
    logger.info("Trained %d epochs in %.0f s; best epoch %d", epoch, train_seconds, best_epoch)

    test_metrics = None
    if test_loader is not None:
        model.load_state_dict(torch.load(run_dir / BEST_FILE, map_location=device, weights_only=True))
        held = _predict_dense(model, test_loader, device, threshold)
        raw = evaluate_dense(store.classes, held.gt, held.pred, tcfg.f1_thresholds, store.layout.fs)
        processed_pred = _postprocess_all(held, store.layout.fs, store.classes, config.infer.postprocess)
        processed = evaluate_dense(store.classes, held.gt, processed_pred, tcfg.f1_thresholds, store.layout.fs)
        test_metrics = {
            "best_epoch": best_epoch,
            "train_seconds": train_seconds,
            "select_on": select_on,
            "objective": loss_settings,
            "thresholds": list(tcfg.f1_thresholds),
            "raw": {**scalar_metrics(raw), "classwise": raw["classwise"]},
            "postprocessed": {**scalar_metrics(processed), "classwise": processed["classwise"]},
        }
        (run_dir / TEST_METRICS_FILE).write_text(yaml.safe_dump(test_metrics, sort_keys=False), encoding="utf-8")
        save_eval_arrays(run_dir / EVAL_ARRAYS_FILE, raw, processed)
        from ethograph.segment.plotting import write_eval_pdf

        write_eval_pdf(run_dir / "eval.pdf", raw, processed, store.classes, tcfg.f1_thresholds)
        logger.info("Test %s: raw %.2f, post-processed %.2f", select_on, raw[select_on], processed[select_on])
    return RunResult(
        run_dir=run_dir,
        best_epoch=best_epoch,
        best_score=float(best_score),
        train_seconds=train_seconds,
        test_metrics=test_metrics,
    )


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def compare_runs(runs_dir: Path) -> pd.DataFrame:
    """One row per run with test metrics: written to ``runs/compare.tsv`` and returned.

    Also writes ``runs/compare.pdf``, the run-vs-run comparison figure —
    class-wise F1, IoU distribution, boundary deltas — whenever at least two
    runs also wrote ``test_eval.npz``. A run trained before that file existed
    still counts in the table; it is just left out of the figure.
    """
    rows = []
    eval_dirs: list[Path] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        test_path = run_dir / TEST_METRICS_FILE
        if not test_path.is_file():
            continue
        data = yaml.safe_load(test_path.read_text(encoding="utf-8"))
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        row = {
            "run": run_dir.name,
            "architecture": cfg["model"]["architecture"],
            "best_epoch": data["best_epoch"],
            "train_seconds": data.get("train_seconds"),
        }
        for stage in ("raw", "postprocessed"):
            for k, v in data[stage].items():
                if k != "classwise":
                    row[f"{stage}.{k}"] = v
        rows.append(row)
        if (run_dir / EVAL_ARRAYS_FILE).is_file():
            eval_dirs.append(run_dir)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(runs_dir / "compare.tsv", sep="\t", index=False)
    if len(eval_dirs) >= 2:
        _write_run_comparison(runs_dir, eval_dirs)
    return df


def _write_run_comparison(runs_dir: Path, run_dirs: list[Path]) -> None:
    from ethograph.segment.plotting import load_run_eval, write_comparison_pdf

    evals = [load_run_eval(d) for d in run_dirs]
    classes = read_target_table(run_dirs[0] / "classes.yaml")
    path = write_comparison_pdf(runs_dir / "compare.pdf", evals, classes, title=f"{len(evals)} runs compared")
    logger.info("Wrote %s", path)
