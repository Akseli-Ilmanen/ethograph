"""Existing labels as model inputs — one column per class of *another* branch.

Labelling is expensive, and the labels of an earlier question often say a
great deal about *when* the answer to a new one can happen: a peck rarely
comes before the head has turned, a landing never before the approach. This
module lets a project feed those labels to its model as ordinary input
columns, rendered onto the clock of the features they join, so the
segmentation and spotting pipelines can exploit the work already done.

Two renderings, one per event type, decided by the ``mapping.txt`` entry and
frozen in the config so training and inference agree on the column layout:

* a **state** label becomes its on/off indicator — ``1`` inside every
  interval of that class, ``0`` outside;
* a **point** event becomes a Laplacian bump centred on it,
  ``max_i exp(-|t - t_i| / sigma)``, at each configured width. The kernel is
  the one :func:`~ethograph.features.changepoints.more_changepoint_features`
  draws around a changepoint, for the same reason: the narrow peak points at
  the moment, the long tails stay readable from far away. The maximum, not
  the sum, keeps the column in ``[0, 1]`` whatever the events do.

A class the trial does not carry renders as zeros. That is the honest reading
— "no such label here" — and exactly the state a trial is in when a model
runs on one nobody has labelled yet.

**An input branch is never a target branch.** The rule lives in the configs
(:func:`check_branches_disjoint`), not here: a model fed the labels it is
asked to predict learns to copy its input, scores perfectly under
cross-validation and has learned nothing about the other inputs at all.

The rendering is xarray-only: the variable is added per trial to the
session's tree (:func:`add_label_inputs`), carrying the individual dim when
the dataset has one so every sample reads its own animal's labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ethograph.io import schema
from ethograph.io.catalog import INDIVIDUAL_DIMS
from ethograph.labels.intervals import EVENT_TYPE_POINT, EVENT_TYPES, load_label_mapping
from ethograph.utils.xr_utils import get_time_coord

#: The variable the rendered columns are added under, and the dim naming them.
VARIABLE = "label_inputs"
DIM = "label_input"


@dataclass(frozen=True)
class LabelInputClass:
    """One class of ``mapping.txt`` fed to the model as input."""

    label: int
    name: str
    branch: int
    event_type: str

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Label {self.name!r} ({self.label}): unknown event type {self.event_type!r}")

    @property
    def is_point(self) -> bool:
        return self.event_type == EVENT_TYPE_POINT

    def column_values(self, sigmas_s: Sequence[float]) -> list[str]:
        """This class's values along :data:`DIM`: its name, or one per bump width."""
        if self.is_point:
            return [f"{self.name}@{float(s):g}s" for s in sigmas_s]
        return [self.name]


def classes_of_branches(
    mapping: str | Path, branches: Sequence[int], classes: Sequence[int] | None = None
) -> list[LabelInputClass]:
    """Every class of *branches* in *mapping* (background excluded), in id order.

    *classes* narrows to those ids; an id it names outside the branches is an
    error, since the branch is what keeps an input from being a target.
    """
    wanted = {int(b) for b in branches}
    out: list[LabelInputClass] = []
    for lid, info in sorted((k, v) for k, v in load_label_mapping(mapping).items() if isinstance(k, int)):
        if lid == 0 or int(info.get("branch", 0)) not in wanted:
            continue
        out.append(LabelInputClass(int(lid), str(info["name"]), int(info.get("branch", 0)), str(info["event_type"])))
    if classes is not None:
        chosen = {int(c) for c in classes}
        missing = chosen - {c.label for c in out}
        if missing:
            raise ValueError(
                f"label_inputs.classes names ids {sorted(missing)} that are not classes of "
                f"branch(es) {sorted(wanted)} in {mapping}"
            )
        out = [c for c in out if c.label in chosen]
    if not out:
        raise ValueError(f"Branch(es) {sorted(wanted)} of {mapping} have no classes to feed as inputs")
    return out


def branches_of(mapping: str | Path, labels: Sequence[int]) -> dict[int, int]:
    """``{label id: branch}`` for *labels*, read off *mapping*."""
    table = load_label_mapping(mapping)
    out: dict[int, int] = {}
    for lid in labels:
        if int(lid) not in table:
            raise ValueError(f"Label {lid} is not in {mapping}")
        out[int(lid)] = int(table[int(lid)].get("branch", 0))
    return out


def check_branches_disjoint(inputs: Sequence[int], targets: Sequence[int], what: str) -> None:
    """Refuse an input branch that is also a target branch.

    *what* names the target setting for the message. The rule that keeps the
    feature honest: a model handed the labels it predicts learns to copy
    them, and cross-validation cannot tell — the same labels are on both
    sides of every fold.
    """
    shared = sorted(set(int(b) for b in inputs) & set(int(b) for b in targets))
    if shared:
        raise ValueError(
            f"label_inputs.branches lists branch(es) {shared}, which {what} predicts — a model fed the labels "
            "it is asked to predict learns to copy them. Labels of one branch are inputs to another branch only."
        )


def column_values(classes: Sequence[LabelInputClass], sigmas_s: Sequence[float]) -> list[str]:
    """Every value along :data:`DIM`, in the order :func:`render` stacks the columns."""
    return [v for c in classes for v in c.column_values(sigmas_s)]


def laplacian_bump(time: np.ndarray, events: np.ndarray, sigma_s: float) -> np.ndarray:
    """``max_i exp(-|t - t_i| / sigma)`` over *time*: 1 at each event, decaying away."""
    if sigma_s <= 0:
        raise ValueError(f"A bump width must be positive, got {sigma_s!r} s")
    time = np.asarray(time, dtype=np.float64)
    events = np.asarray(events, dtype=np.float64)
    if events.size == 0:
        return np.zeros(time.size, dtype=np.float64)
    return np.exp(-np.abs(time[:, None] - events[None, :]) / float(sigma_s)).max(axis=1)


def state_indicator(time: np.ndarray, onsets: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """``1`` inside every ``[onset, offset]``, ``0`` elsewhere."""
    time = np.asarray(time, dtype=np.float64)
    out = np.zeros(time.size, dtype=np.float64)
    for onset, offset in zip(np.asarray(onsets, dtype=np.float64), np.asarray(offsets, dtype=np.float64)):
        if np.isfinite(onset) and np.isfinite(offset):
            out[(time >= onset) & (time <= offset)] = 1.0
    return out


def render(
    labels: pd.DataFrame,
    time: np.ndarray,
    classes: Sequence[LabelInputClass],
    sigmas_s: Sequence[float],
    actor: str | None = None,
) -> np.ndarray:
    """*classes* rendered onto *time* (the clock *labels* are on) as ``(T, C)``.

    *actor* keeps only that individual's labels; ``None`` reads everyone's.
    The event type is the class's own (from the mapping), never the row's,
    so the layout cannot change with what a trial happens to contain.
    """
    time = np.asarray(time, dtype=np.float64)
    if actor is not None and not labels.empty:
        labels = labels[labels["individual"].astype(str) == str(actor)]
    columns: list[np.ndarray] = []
    for cls in classes:
        rows = labels[labels["labels"].astype(int) == cls.label] if not labels.empty else labels
        onsets = np.asarray(rows["onset_s"], dtype=np.float64) if not rows.empty else np.array([], dtype=np.float64)
        if cls.is_point:
            columns.extend(laplacian_bump(time, onsets[np.isfinite(onsets)], sigma) for sigma in sigmas_s)
        else:
            offsets = (
                np.asarray(rows["offset_s"], dtype=np.float64) if not rows.empty else np.array([], dtype=np.float64)
            )
            columns.append(state_indicator(time, onsets, offsets))
    return np.column_stack(columns) if columns else np.zeros((time.size, 0), dtype=np.float64)


def add_label_inputs(
    ds: xr.Dataset,
    labels: pd.DataFrame,
    classes: Sequence[LabelInputClass],
    sigmas_s: Sequence[float],
    clock: str,
    *,
    individuals: Sequence[str] | None = None,
    actor: str | None = None,
) -> xr.Dataset:
    """*ds* with :data:`VARIABLE` added on *clock*'s time coordinate.

    *labels* are the trial's rows, trial-relative like the dataset. With
    *individuals* given, the variable carries an individual dim — the
    dataset's own (its coordinate values win) or a new ``individual`` when
    the dataset has none — one slice per animal, rendered from that animal's
    rows; a pipeline that pins the individual per sample then reads its own.
    Without it the variable is flat ``(time, label_input)``, rendered from
    *actor*'s rows (``None`` = everyone's) — the single-stream case.

    Declared ``kind="label_input"``, ``normalise=0``: the columns already live
    in ``[0, 1]`` and mean what they say there.
    """
    if VARIABLE in ds.data_vars:
        raise ValueError(f"{VARIABLE!r} is already in the dataset")
    if clock not in ds.data_vars:
        raise ValueError(f"label_inputs.clock names {clock!r}, which the dataset does not have")
    time_coord = get_time_coord(ds[clock])
    if time_coord is None:
        raise ValueError(f"label_inputs.clock {clock!r} has no time coordinate to render onto")
    time_dim = str(time_coord.dims[0])
    time = np.asarray(time_coord.values, dtype=np.float64)
    values = column_values(classes, sigmas_s)

    ind_dim = next((d for d in INDIVIDUAL_DIMS if d in ds.dims), None)
    if individuals is None and ind_dim is None:
        data = render(labels, time, classes, sigmas_s, actor)
        da = xr.DataArray(data, dims=(time_dim, DIM), coords={time_dim: time_coord, DIM: values})
    else:
        if ind_dim is not None:
            names = [str(v) for v in ds.coords[ind_dim].values]
            coord = ds.coords[ind_dim]
        else:
            ind_dim = INDIVIDUAL_DIMS[0]
            names = [str(v) for v in (individuals or [])]
            coord = names
        stacked = np.stack([render(labels, time, classes, sigmas_s, name) for name in names], axis=1)
        da = xr.DataArray(
            stacked, dims=(time_dim, ind_dim, DIM), coords={time_dim: time_coord, ind_dim: coord, DIM: values}
        )
    schema.describe(da, schema.LABEL_INPUT, normalise=False)
    return ds.assign({VARIABLE: da})


__all__ = [
    "DIM",
    "VARIABLE",
    "LabelInputClass",
    "add_label_inputs",
    "branches_of",
    "check_branches_disjoint",
    "classes_of_branches",
    "column_values",
    "laplacian_bump",
    "render",
    "state_indicator",
]
