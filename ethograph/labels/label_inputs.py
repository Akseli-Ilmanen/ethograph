"""Existing labels as classifier inputs.

The lightgbm model reads a session's own labels the same way it reads a feature:
one input column per (class, individual), rendered onto the feature time base.
What a class already tells you about *when* is often the strongest evidence
there is — a peck rarely happens before the head has turned, a landing never
before the approach — and until now none of it reached the classifier.

Two renderings, one per event type, frozen in the config at creation so
training and inference cannot disagree about the shape of a column:

* a **state** label becomes its on/off indicator — ``1`` inside every interval
  of that class, ``0`` outside. That is the whole of what a state says.
* a **point** event becomes a Laplacian bump centred on it, at each of
  :data:`POINT_SIGMAS_S`. The kernel is the one
  :func:`ethograph.features.changepoints.more_changepoint_features` uses on
  changepoints, and for the same reason: a Laplacian's narrow peak points
  straight at the moment while its long tails stay readable from far away, so
  one column carries both "it is *here*" and "it was a while ago". The two
  widths are deliberately hard-coded — a sharp one for timing, a wide one for
  reach — because a kernel width is not something a user has any way to choose
  by looking.

A class the trial does not carry renders as zeros. That is the honest reading:
the column says "no such label here", which is exactly the state a trial is in
when the model runs on it.

Times are **trial-relative**, the clock labels are stored on.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ethograph.labels.intervals import EVENT_TYPE_POINT, EVENT_TYPE_STATE

#: Laplacian widths (seconds) every point-event input column is rendered at —
#: one column per sigma. Hard-coded on purpose; see this module's docstring.
POINT_SIGMAS_S: tuple[float, ...] = (0.1, 1.0)

#: Prefix marking an input column as a label channel rather than a catalog
#: feature, so the two can never be confused in a stored column list.
COLUMN_PREFIX = "label:"


@dataclass
class LabelInput:
    """One existing label class fed to the classifier as input columns.

    *event_type* is frozen here rather than re-read from ``mapping.txt`` at
    prediction time: it decides how many columns the class contributes and how
    they are drawn, so a mapping edited after training must not silently
    change the model's input layout.

    *individuals* pins whose labels of this class are read, one column each (a
    point event: one per sigma per individual). An empty list means "whoever
    labelled it" — a single-individual session has nothing to choose, exactly
    as a feature's single-valued dim is never drawn as a row.
    """

    label: int
    name: str
    event_type: str = EVENT_TYPE_STATE
    individuals: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.label = int(self.label)
        self.name = str(self.name)
        self.event_type = str(self.event_type)
        if self.event_type not in (EVENT_TYPE_STATE, EVENT_TYPE_POINT):
            raise ValueError(f"Label input {self.name!r}: unknown event type {self.event_type!r}.")
        self.individuals = [str(v) for v in self.individuals]

    # ------------------------------------------------------------------

    @property
    def is_point(self) -> bool:
        return self.event_type == EVENT_TYPE_POINT

    def _keys(self) -> list[str | None]:
        """The individuals this input reads; ``[None]`` means "whoever"."""
        return list(self.individuals) if self.individuals else [None]

    def columns(self) -> list[str]:
        """This input's column names, in the order :meth:`render` stacks them."""
        names: list[str] = []
        for who in self._keys():
            base = f"{COLUMN_PREFIX}{self.name}({self.label})"
            if who is not None:
                base += f"|individual={who}"
            if self.is_point:
                names += [f"{base}|sigma={sigma:g}" for sigma in POINT_SIGMAS_S]
            else:
                names.append(base)
        return names

    def retarget(self, individual: str) -> LabelInput:
        """This input with its single pinned individual swapped for *individual*.

        Unchanged when it pins nobody (nothing to choose), when it already
        reads *individual*, or when it reads several at once — the same rule
        :func:`~ethograph.labels.onset_model.retarget_individual` applies to
        features, for the same reason: two columns collapsed onto one animal
        would hand the classifier the same data in slots it learned as two.
        """
        if not individual or len(self.individuals) != 1 or self.individuals == [individual]:
            return self
        return replace(self, individuals=[individual])

    # ------------------------------------------------------------------

    def render(self, df: pd.DataFrame | None, time: np.ndarray) -> np.ndarray:
        """This input's columns over *time* (trial-relative), ``(T, n)``."""
        time = np.asarray(time, dtype=np.float64)
        rows = _rows_of_class(df, self.label, self.event_type)
        return np.concatenate([_render_one(rows, who, time, self.is_point) for who in self._keys()], axis=1)


def _rows_of_class(df: pd.DataFrame | None, label: int, event_type: str) -> pd.DataFrame:
    """Rows of *df* that are this class, in the event type the config froze."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["onset_s", "offset_s", "individual"])
    mask = df["labels"] == label
    if "event_type" in df.columns:
        mask &= df["event_type"] == event_type
    elif event_type == EVENT_TYPE_POINT:
        mask &= df["offset_s"].isna()
    return df[mask]


def _render_one(rows: pd.DataFrame, who: str | None, time: np.ndarray, is_point: bool) -> np.ndarray:
    """One individual's columns of one class: ``(T, len(POINT_SIGMAS_S))`` for
    a point event, ``(T, 1)`` for a state."""
    if who is not None and "individual" in rows.columns:
        rows = rows[rows["individual"].astype(str) == who]
    if is_point:
        onsets = np.asarray(rows["onset_s"], dtype=np.float64)
        return np.column_stack([_laplacian(time, onsets[np.isfinite(onsets)], sigma) for sigma in POINT_SIGMAS_S])
    channel = np.zeros(time.size, dtype=np.float64)
    for onset, offset in zip(rows["onset_s"], rows["offset_s"]):
        if np.isfinite(onset) and np.isfinite(offset):
            channel[(time >= float(onset)) & (time <= float(offset))] = 1.0
    return channel[:, None]


def _laplacian(time: np.ndarray, onsets: np.ndarray, sigma: float) -> np.ndarray:
    """``max_i exp(-|t - onset_i| / sigma)`` — 1 at each event, decaying away.

    The maximum, not the sum: the channel stays in ``[0, 1]`` whatever the
    events do, so two events close together read as one strong moment rather
    than as an impossibly strong one, and a trial's values never depend on how
    many other events it happens to contain.
    """
    if onsets.size == 0:
        return np.zeros(time.size, dtype=np.float64)
    return np.exp(-np.abs(time[:, None] - onsets[None, :]) / float(sigma)).max(axis=1)


def label_columns(inputs: list[LabelInput]) -> list[str]:
    """Every column name *inputs* contributes, in render order."""
    return [name for inp in inputs for name in inp.columns()]


def render_label_inputs(inputs: list[LabelInput], df: pd.DataFrame | None, time: np.ndarray) -> np.ndarray:
    """*inputs* rendered onto *time* (trial-relative) as ``(T, D)``.

    The one place a label becomes a model input, so a column cannot be present
    at training and absent at inference.
    """
    time = np.asarray(time, dtype=np.float64)
    if not inputs:
        return np.zeros((time.size, 0), dtype=np.float64)
    return np.concatenate([inp.render(df, time) for inp in inputs], axis=1)
