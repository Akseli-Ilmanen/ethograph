"""Which role each sample plays in a run — torch-free, so every stage can ask."""

from __future__ import annotations

import random

import pandas as pd

from ethograph.segment.config import SegmentConfig


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
