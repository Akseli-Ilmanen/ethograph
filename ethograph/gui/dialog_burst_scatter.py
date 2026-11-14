"""Burst scatter: every state label in the video grid as a point, two features of the user's choice as axes.

Opened from the video grid's **Scatter…**. Similar bursts sit near each
other, so the plot is a way to *choose what to watch*:

* a **click** shows the clips nearest the click (of the clicked point's
  class), as many as the grid shows at once;
* a **region** — drag on the plot to draw the rectangle, or tick **Region**;
  drag again anywhere to start a new one — limits the grid to the bursts
  inside it — with a keypoint's x and y as the axes, that is "only bursts
  while the keypoint was here". **Keep region** makes that a filter that
  outlives the axes: the scatter itself is limited to those bursts, so they
  can be compared on any other pair of features until **Show all**.
  **Map x/y** does the setup in one step: the picked feature's x and y
  become the axes, drawn to scale like the space plot, ready to drag over;
  **Arena** draws a geometry of the space plot's library (the box, the
  perches) behind the map, in white.

The axes are one number per burst (:mod:`ethograph.features.burst_stats`):

* **timing** — duration, and the burst's onset measured from the trial's
  start and end, from the trial's first and last label, and from the nearest
  onset or offset of one label class — picked in the box beside the axis,
  which is greyed out until the axis is one of those two;
* **spikes** — count, rate, first ISI, irregularity, rate trend, peak
  position — when the classes are units whose spike trains are loaded;
* **behaviour** — any catalog feature, every dim pinned, summarised over
  the burst (mean, peak, value at onset, change from just before).

* **behaviour PCA** — every ticked feature, averaged over a sliding window
  and decomposed across all trials at once
  (:mod:`ethograph.features.behaviour_pca`); each burst sits at the
  behavioural state it happened in, so a region on two components is
  "bursts while the animal was doing this".

Every timing axis is *burst onset minus the reference*: negative is before it.

**Baseline** draws, faintly behind the bursts, the same bursts shifted in
time within their trial (``pynapple.shift_timestamps``): where the bursts
would fall if their timing had nothing to do with the axes. Bursts sitting
off that cloud are the ones saying something.

**Colour by** paints the points by their own class (the unit), by the label
that takes up most of the burst, by a value (any feature with a single
series, averaged over the burst, or any axis column — spread over a colour
scale between two percentiles of the values shown, 0–95 unless changed, 5–95
for values that go negative), by a colour the
dataset already holds (a feature with ``rgb`` in its name, read at the middle
of the burst), or not at all. Three kinds of mark keep the
greys apart: a burst is a filled disc, a burst that is mostly in no label a
hollow ring, and a baseline (shifted) burst a small faint blue cross.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import QRectF, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ethograph.features.behaviour_pca import BehaviourPCA, fit_behaviour_pca, window_means, window_starts
from ethograph.features.burst_stats import (
    SPIKE_FEATURES,
    SUMMARIES,
    default_percentiles,
    nearest_points,
    points_in_rect,
    scale_between_percentiles,
    shift_onsets,
    spike_features,
    time_from_nearest,
    window_summary,
)
from ethograph.gui.dialog_label_gridview import _mapping_color_hex, entry_key
from ethograph.gui.dialog_onset_model import base_loader, iter_trial_windows
from ethograph.gui.notify import notify
from ethograph.gui.plots_space import ReferenceGeometry, geometry_dirs_of, load_library_geometries
from ethograph.gui.widgets_ephys import FIRING_RATE_UNIT_DIM, SPIKE_RATE_FEATURES
from ethograph.labels.feature_events import classes_along

_BG = "#1a1d21"
#: Slack when cutting a unit's spikes to a burst: the burst's bounds are its
#: first and last spike, and must survive the round trip through trial time.
_EDGE_S = 1e-6

DURATION = "Duration (s)"
FROM_TRIAL_START = "Time from trial start (s)"
FROM_TRIAL_END = "Time from trial end (s)"
FROM_FIRST_LABEL = "Time from the trial's first label onset (s)"
FROM_LAST_LABEL = "Time from the trial's last label offset (s)"

EDGES = {"onset": "onset_s", "offset": "offset_s"}

#: The axes that need a label class chosen beside them: axis text -> edge.
LABEL_AXES = {
    "Time from a label's nearest onset (s)": "onset",
    "Time from a label's nearest offset (s)": "offset",
}


#: How many rows a dropdown shows before it scrolls.
COMBO_VISIBLE_ROWS = 20


def scrolling_combo() -> QComboBox:
    """A dropdown whose list scrolls past :data:`COMBO_VISIBLE_ROWS` rows.

    The style's default popup is a menu as tall as its items: with a few
    dozen units or features it runs off the screen and the rest cannot be
    reached. As a list it is capped and gets a scroll bar.
    """
    combo = QComboBox()
    combo.setMaxVisibleItems(COMBO_VISIBLE_ROWS)
    combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
    view = combo.view()
    if view is not None:
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    return combo


# ----------------------------------------------------------------------
# Columns: one value per burst
# ----------------------------------------------------------------------


def unique_bursts(entries: list) -> list:
    """The state labels among *entries*, each once whatever camera saw it."""
    rows: dict[tuple, object] = {}
    for entry in entries:
        if not entry.is_point:
            rows.setdefault(entry_key(entry), entry)
    return list(rows.values())


#: Column of :func:`reference_labels` naming each row's class as the label box shows it.
REF = "ref"
#: Column of :func:`reference_labels` holding that class's colour.
REF_COLOR = "ref_color"

#: How the points can be coloured: key -> combo text.
COLOUR_MODES = {
    "class": "Class (e.g. unit)",
    "label": "Label it occurs in",
    "feature": "Feature value (1D)",
    "rgb": "Precomputed colour (RGB)",
    "none": "No colouring",
}
#: The dims of a colour feature that hold its channels, never something to pin.
_CHANNEL_DIMS = frozenset({"RGB", "RGBA"})
#: Steps of the colour scale a feature value is drawn on.
_SCALE_STEPS = 64
#: The colour scale's name in pyqtgraph: dark = low, bright = high, readable on a dark background.
_SCALE_MAP = "viridis"
#: Every burst when nothing is coloured, and the outline of one that is mostly in no label.
_NEUTRAL = "#c8ccd2"
#: The baseline's crosses: a hue no burst is drawn in by default, and faint.
_BASELINE_RGBA = (110, 165, 255, 140)


def reference_labels(app_state) -> pd.DataFrame | None:
    """Every label a burst can be timed against: the working labels and each
    loaded prediction set, except the set the grid itself shows and any set of
    units (burst or firing labels, ``unit_<id>``) — those are neural events, not
    behaviour, and colouring a burst by another unit's burst means nothing.

    Two sources may reuse an id for different classes, so a class is named by
    its :data:`REF` text — ``name (id)``, with the set's name after it for a
    prediction set. ``None`` when nothing but the grid's own labels is loaded.
    """
    shown = getattr(app_state, "label_source_path", None)
    sources = [("", getattr(app_state, "_all_labels_df", None), getattr(app_state, "_label_mappings", None) or {})]
    for prediction_set in getattr(app_state, "prediction_sets", None) or []:
        if str(prediction_set.path) == shown:
            continue
        mappings = prediction_set.mappings if prediction_set.mappings is not None else sources[0][2]
        if classes_along(mappings, FIRING_RATE_UNIT_DIM):
            continue
        sources.append((prediction_set.path.name, prediction_set.labels_df, mappings))
    parts = []
    for source, df, mappings in sources:
        if df is None or df.empty:
            continue
        part = df[["trial", "labels", "onset_s", "offset_s"]].copy()
        suffix = f" · {source}" if source else ""
        part[REF] = [f"{mappings.get(int(i), {}).get('name', int(i))} ({int(i)}){suffix}" for i in part["labels"]]
        part[REF_COLOR] = [_mapping_color_hex(mappings.get(int(i), {})) for i in part["labels"]]
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else None


def timing_columns(app_state, bursts: list) -> dict[str, np.ndarray]:
    """The timing axes every dataset has: duration, and onset from the trial's
    bounds and from its first and last label (NaN in a trial without labels)."""
    onsets = np.array([b.onset_s for b in bursts], dtype=float)
    columns = {DURATION: np.array([b.duration for b in bursts], dtype=float), FROM_TRIAL_START: onsets}

    collection = getattr(app_state, "source_collection", None)
    if collection is not None:
        durations: dict[str, float] = {}
        for trial in {str(b.trial): b.trial for b in bursts}.values():
            idx = collection.trial_index(trial)
            durations[str(trial)] = float(collection.trial_range(idx).duration) if idx is not None else float("nan")
        columns[FROM_TRIAL_END] = onsets - np.array([durations[str(b.trial)] for b in bursts])

    df = reference_labels(app_state)
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    if df is not None:
        trial = df["trial"].astype(str)
        first = df.groupby(trial)["onset_s"].min().to_dict()
        # A point label has no offset: it ends where it starts.
        last = df["offset_s"].fillna(df["onset_s"]).groupby(trial).max().to_dict()
    nan = float("nan")
    columns[FROM_FIRST_LABEL] = onsets - np.array([first.get(str(b.trial), nan) for b in bursts])
    columns[FROM_LAST_LABEL] = onsets - np.array([last.get(str(b.trial), nan) for b in bursts])
    return columns


#: How many shifted copies of every burst make the baseline cloud.
BASELINE_REPEATS = 5


@dataclass
class ShiftedBurst:
    """A burst moved in time: what a column is read for to draw the baseline."""

    trial: object
    label_id: int
    name: str
    onset_s: float
    offset_s: float

    @property
    def duration(self) -> float:
        return self.offset_s - self.onset_s


def trial_durations(app_state, bursts: list) -> dict[str, float]:
    """Each burst trial's length in seconds; a trial the session has no timing for is left out."""
    collection = getattr(app_state, "source_collection", None)
    out: dict[str, float] = {}
    if collection is None:
        return out
    for trial in {str(b.trial): b.trial for b in bursts}.values():
        idx = collection.trial_index(trial)
        if idx is not None:
            out[str(trial)] = float(collection.trial_range(idx).duration)
    return out


def shifted_bursts(bursts: list, durations: dict[str, float], repeats: int = BASELINE_REPEATS) -> list[ShiftedBurst]:
    """Every burst moved by a random offset shared with the others of its trial, *repeats* times over.

    Each keeps its class and its length; one the shift leaves too little
    room for before the trial's end is left out rather than cut short.
    """
    by_trial: dict[str, list] = {}
    for burst in bursts:
        if str(burst.trial) in durations:
            by_trial.setdefault(str(burst.trial), []).append(burst)
    out: list[ShiftedBurst] = []
    for trial, members in by_trial.items():
        length = durations[trial]
        onsets = np.array([b.onset_s for b in members], dtype=float)
        for _ in range(repeats):
            for burst, onset in zip(members, shift_onsets(onsets, length)):
                offset = float(onset) + burst.duration
                if offset <= length:
                    out.append(ShiftedBurst(burst.trial, burst.label_id, burst.name, float(onset), offset))
    return out


def covering_labels(app_state, bursts: list) -> list[tuple[str, str] | None]:
    """Per burst, the ``(ref, colour)`` of the label class that takes up most of it.

    The burst's time is shared out between the label classes it overlaps and
    the gap — the time in no label; whichever has the largest share wins, and
    a burst the gap wins is ``None``. Every instance of a class in the
    burst's trial counts towards that class; a point label lasts no time.
    """
    out: list[tuple[str, str] | None] = [None] * len(bursts)
    df = reference_labels(app_state)
    if df is None:
        return out
    states = df[df["offset_s"].notna()]
    by_trial = {trial: group for trial, group in states.groupby(states["trial"].astype(str))}
    for i, burst in enumerate(bursts):
        rows = by_trial.get(str(burst.trial))
        if rows is None or burst.duration <= 0.0:
            continue
        overlap = np.minimum(rows["offset_s"].to_numpy(), burst.offset_s) - np.maximum(
            rows["onset_s"].to_numpy(), burst.onset_s
        )
        covered = pd.Series(np.clip(overlap, 0.0, None)).groupby(rows[REF].to_numpy()).sum()
        gap = max(0.0, burst.duration - float(covered.sum()))
        if covered.max() > gap:
            ref = str(covered.idxmax())
            out[i] = (ref, str(rows.loc[rows[REF] == ref, REF_COLOR].iloc[0]))
    return out


def label_timing_column(app_state, bursts: list, ref: str, edge: str) -> np.ndarray:
    """Each burst's onset minus the nearest *edge* (onset / offset) of the class *ref* names in its trial."""
    df = reference_labels(app_state)
    out = np.full(len(bursts), np.nan)
    if df is None:
        return out
    rows = df[df[REF] == ref]
    refs = {trial: group[EDGES[edge]].to_numpy(dtype=float) for trial, group in rows.groupby(rows["trial"].astype(str))}
    for i, burst in enumerate(bursts):
        if str(burst.trial) in refs:
            out[i] = time_from_nearest(burst.onset_s, refs[str(burst.trial)])
    return out


def spike_columns(meta, bursts: list) -> dict[str, np.ndarray]:
    """The spike features of each burst, read off its unit's train.

    Empty when no spike trains are loaded or no class is a unit
    (``unit_<id>``, as burst detection names them).
    """
    ephys = getattr(meta, "ephys_widget", None)
    if ephys is None:
        return {}
    trains = ephys.spike_trains()
    if not trains:
        return {}
    wanted = {str(b.trial) for b in bursts}
    trials = [t for t in meta.app_state.trials if str(t) in wanted]
    shifts = {str(trial): shift for trial, _reader, _t0, _t1, shift in ephys.firing_rate_windows(trials)}
    times = {unit: np.asarray(train.t, dtype=float) for unit, train in trains.items()}
    prefix = f"{FIRING_RATE_UNIT_DIM}_"

    columns = {title: np.full(len(bursts), np.nan) for title in SPIKE_FEATURES.values()}
    found = False
    for i, burst in enumerate(bursts):
        name = str(burst.name)
        spikes = times.get(name.removeprefix(prefix)) if name.startswith(prefix) else None
        shift = shifts.get(str(burst.trial))
        if spikes is None or shift is None:
            continue
        lo = np.searchsorted(spikes, burst.onset_s + shift - _EDGE_S, side="left")
        hi = np.searchsorted(spikes, burst.offset_s + shift + _EDGE_S, side="right")
        for key, value in spike_features(spikes[lo:hi]).items():
            columns[SPIKE_FEATURES[key]][i] = value
        found = True
    return columns if found else {}


def feature_column(app_state, bursts: list, feature: str, selections: dict[str, str], how: str) -> np.ndarray:
    """*feature* summarised over each burst (:func:`window_summary`), read trial by trial.

    Raises ``ValueError`` when *selections* leave the feature with more than
    one series — a point on the scatter is one number.
    """
    by_trial: dict[str, list[int]] = {}
    for i, burst in enumerate(bursts):
        by_trial.setdefault(str(burst.trial), []).append(i)
    trials = [t for t in app_state.trials if str(t) in by_trial]
    out = np.full(len(bursts), np.nan)
    for trial, loader, t0, t1, shift in iter_trial_windows(app_state, trials):
        data = loader.select(feature, selections, t0, t1)
        if data is None:
            continue
        values = np.asarray(data.data, dtype=float)
        if values.ndim > 1:
            if values.shape[1] != 1:
                raise ValueError(f"'{feature}' still has {values.shape[1]} series — pin every dimension to one value.")
            values = values[:, 0]
        time = np.asarray(data.time, dtype=float) - shift
        for i in by_trial[str(trial)]:
            out[i] = window_summary(time, values, bursts[i].onset_s, bursts[i].offset_s, how)
    return out


def channel_dim(dims: dict[str, list[str]]) -> str | None:
    """Which dim of a colour feature holds its channels: one of three or four values.

    The dim is found by its size, not its name — datasets call it ``RGB``,
    ``channel``, ``c``… With several that size, one named like a colour dim
    wins, else the last (channels are stored innermost).
    """
    sized = [d for d, values in dims.items() if len(values) in (3, 4)]
    named = [d for d in sized if d.upper() in _CHANNEL_DIMS or any(w in d.lower() for w in ("rgb", "col", "chan"))]
    candidates = named or sized
    return candidates[-1] if candidates else None


def rgb_choices(app_state) -> list[tuple[str, str, dict[str, str]]]:
    """The precomputed colours on offer, as ``(text, feature, selections)``.

    A colour feature is one with ``rgb`` in its name (the dataset convention)
    and a dim of three or four channels, which is never pinned: the feature
    is read whole, one colour per sample. One that has further dims — a
    colour per keypoint — is offered once per combination of their values.
    """
    loader = base_loader(app_state)
    out: list[tuple[str, str, dict[str, str]]] = []
    for feature in behavioural_features(app_state):
        if "rgb" not in feature.lower() or loader is None:
            continue
        all_dims = loader.feature_dims(feature)
        channels = channel_dim(all_dims)
        if channels is None:
            continue
        dims = {d: v for d, v in all_dims.items() if d != channels}
        for combo in itertools.product(*dims.values()):
            text = f"{feature} [{', '.join(combo)}]" if combo else feature
            out.append((text, feature, dict(zip(dims, combo))))
    return out


def rgb_column(app_state, bursts: list, feature: str, selections: dict[str, str]) -> np.ndarray:
    """Each burst's colour ``(N, 3)`` in 0–1: *feature* at the sample nearest the burst's middle.

    A colour is read at one moment, never averaged — the mean of two colours
    of a map is a colour that means neither. NaN where the trial has none.
    """
    by_trial: dict[str, list[int]] = {}
    for i, burst in enumerate(bursts):
        by_trial.setdefault(str(burst.trial), []).append(i)
    trials = [t for t in app_state.trials if str(t) in by_trial]
    out = np.full((len(bursts), 3), np.nan)
    for trial, loader, t0, t1, shift in iter_trial_windows(app_state, trials):
        data = loader.select(feature, selections, t0, t1)
        if data is None or len(data.time) == 0:
            continue
        colours = np.asarray(data.data, dtype=float)
        if colours.ndim != 2 or colours.shape[1] < 3:
            raise ValueError(f"'{feature}' is not a colour: expected three channels per sample.")
        time = np.asarray(data.time, dtype=float) - shift
        for i in by_trial[str(trial)]:
            middle = (bursts[i].onset_s + bursts[i].offset_s) / 2.0
            out[i] = colours[int(np.argmin(np.abs(time - middle))), :3]
    # Colours stored as 0–255 are brought onto 0–1.
    if np.nanmax(out, initial=0.0) > 1.0:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0)


def behavioural_features(app_state) -> list[str]:
    """The features a behaviour PCA can read: the catalog's, less the spike
    rates and the console's (those live for one trial only)."""
    loader = base_loader(app_state)
    if loader is None:
        return []
    derived = getattr(getattr(app_state, "data_loader", None), "derived", None) or {}
    return [f for f in loader.catalog.feature_choices() if f not in derived and f not in SPIKE_RATE_FEATURES]


def feature_series(
    loader, feature: str, t0: float | None, t1: float | None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every single series of *feature* — one per combination of its dims' values — as ``name -> (time, values)``.

    A loader reads one series at a time when a feature has several dims, so
    each combination is pinned in turn; a combination with no data is left out.
    """
    dims = loader.feature_dims(feature)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for combo in itertools.product(*dims.values()):
        data = loader.select(feature, dict(zip(dims, combo)), t0, t1)
        if data is None or len(data.time) == 0:
            continue
        values = np.asarray(data.data, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"'{feature}' still has {values.shape[1]} series with every dimension pinned.")
        name = f"{feature} [{', '.join(combo)}]" if combo else feature
        out[name] = (np.asarray(data.time, dtype=float), values)
    return out


def behaviour_pca(app_state, features: list[str], window_s: float, step_s: float, n_components: int) -> BehaviourPCA:
    """Fit the trial-concatenated PCA of *features* over every trial the trials table shows.

    Each feature contributes all its series (every keypoint, coordinate,
    individual), matched across trials by name; a trial lacking a series
    keeps its windows, with that column empty. Raises ``ValueError`` when
    nothing could be read.
    """
    columns: dict[str, None] = {}  # ordered: first seen first
    per_trial: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for trial, loader, t0, t1, shift in iter_trial_windows(app_state, list(app_state.trials)):
        series = per_trial.setdefault(str(trial), {})
        for feature in features:
            for name, (time, values) in feature_series(loader, feature, t0, t1).items():
                series[name] = (time - shift, values)
                columns.setdefault(name)
    blocks: dict[str, np.ndarray] = {}
    centers: dict[str, np.ndarray] = {}
    for trial, series in per_trial.items():
        if not series:
            continue
        starts = window_starts(max(float(time[-1]) for time, _ in series.values()), window_s, step_s)
        empty = np.full((len(starts), 1), np.nan)
        blocks[trial] = np.hstack(
            [window_means(*series[name], starts, window_s) if name in series else empty for name in columns]
        )
        centers[trial] = starts + window_s / 2.0
    return fit_behaviour_pca(blocks, centers, list(columns), n_components)


# ----------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------


class _RegionViewBox(pg.ViewBox):
    """A view box whose left drag draws a rectangle instead of panning.

    The rectangle is announced in view coordinates when the drag ends. An
    item under the cursor (the region itself) takes its own drags first, and
    a drag on an axis still scales it; right-drag zooms and the wheel too.
    """

    region_drawn = Signal(float, float, float, float)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != Qt.LeftButton or axis is not None:
            super().mouseDragEvent(ev, axis=axis)
            return
        ev.accept()
        if not ev.isFinish():
            self.updateScaleBox(ev.buttonDownPos(), ev.pos())
            return
        self.rbScaleBox.hide()
        rect = self.childGroup.mapRectFromParent(QRectF(ev.buttonDownPos(ev.button()), ev.pos()).normalized())
        self.region_drawn.emit(rect.left(), rect.right(), rect.top(), rect.bottom())


class BurstScatterDialog(QDialog):
    """Two features per burst as a scatter; a click or a region chooses what the grid plays."""

    #: ``(set of entry keys, description)`` — the bursts the grid should show.
    selected = Signal(object, str)
    #: The selection was dropped: show everything again.
    cleared = Signal()

    def __init__(self, meta, entries: list, n_nearest_fn: Callable[[], int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Burst scatter")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.meta = meta
        self.app_state = meta.app_state
        self._n_nearest_fn = n_nearest_fn
        self.bursts = unique_bursts(entries)
        self.keys = [entry_key(b) for b in self.bursts]
        self._class_ids = np.array([b.label_id for b in self.bursts], dtype=int)
        self.columns: dict[str, np.ndarray] = {}
        #: Axis title -> how to read that column for any list of bursts: what
        #: lets the baseline be drawn on it. An axis without one (the spike
        #: features: a shifted burst has no spikes of its own) has no baseline.
        self._recipes: dict[str, Callable[[list], np.ndarray]] = {}
        #: The shifted bursts while the baseline is on, and their columns as read.
        self._baseline: list[ShiftedBurst] = []
        self._baseline_columns: dict[str, np.ndarray] = {}
        self.columns.update(timing_columns(self.app_state, self.bursts))
        for timing_title in self.columns:
            self._recipes[timing_title] = self._timing_recipe(timing_title)
        self.columns.update(spike_columns(meta, self.bursts))
        #: Keys of the bursts on the grid's screen, ringed on the plot.
        self._marked: set[tuple] = set()
        #: Indices into ``bursts`` of the points drawn (class filter, finite on both axes).
        self._shown = np.empty(0, dtype=int)
        self._dim_combos: dict[str, QComboBox] = {}
        self._roi: pg.RectROI | None = None
        #: A click picks bursts only once it is known not to be half a double click.
        self._pending_click = (0.0, 0.0)
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self._on_single_click)
        #: (label id, edge) -> its timing column, read once.
        self._label_timing: dict[tuple[str, str], np.ndarray] = {}
        #: The single-series features (found once) and their per-burst means as read.
        self._one_series: list[str] | None = None
        self._colour_features: dict[str, np.ndarray] = {}
        #: The precomputed colours on offer (found once) and each one's per-burst colours as read.
        self._rgb_choices: list[tuple[str, str, dict[str, str]]] | None = None
        self._rgb_columns: dict[str, np.ndarray] = {}
        #: What the colour scale currently spans, said under the plot.
        self._scale_text = ""
        #: Per burst, the label covering most of it (read once, on first use).
        self._covering: list[tuple[str, str] | None] | None = None
        #: Titles of the axes the behaviour PCA added, replaced by the next run.
        self._pca_titles: list[str] = []
        #: Titles of the axes **Map x/y** added: the plot is drawn to scale while both are shown.
        #: Each to the coordinate it holds (0 = x, 1 = y): which column of a geometry it is drawn against.
        self._map_titles: dict[str, int] = {}
        #: The space plot's geometry library, by name.
        self._geometries: dict[str, list[ReferenceGeometry]] = load_library_geometries(geometry_dirs_of(self.app_state))
        #: Which bursts a kept region lets through (``None``: all), and what it was.
        self._kept: np.ndarray | None = None
        self._kept_text = ""

        layout = QVBoxLayout(self)
        axes = QHBoxLayout()
        axes.addWidget(QLabel("Class:"))
        self.class_combo = scrolling_combo()
        self.class_combo.addItem("All classes", None)
        seen: dict[int, str] = {}
        for burst in self.bursts:
            seen.setdefault(burst.label_id, burst.name)
        for label_id in sorted(seen):
            self.class_combo.addItem(f"{seen[label_id]} ({label_id})", label_id)
        self.class_combo.currentIndexChanged.connect(self._redraw)
        axes.addWidget(self.class_combo)
        colours = QHBoxLayout()
        colours.addWidget(QLabel("Colour by:"))
        self.colour_combo = scrolling_combo()
        for key, text in COLOUR_MODES.items():
            self.colour_combo.addItem(text, key)
        self.colour_combo.setToolTip(
            "Class: each point in its own class's colour (the unit).\n"
            "Label it occurs in: the colour of the label taking up most of the burst; a hollow ring when\n"
            "most of it is in no label.\n"
            "Feature value (1D): a colour scale over the value picked in the box beside this one.\n"
            "Precomputed colour (RGB): a colour the dataset already holds (a feature with 'rgb' in its name,\n"
            "picked in the box beside this one), read at the middle of the burst.\n"
            "No colouring: every point the same."
        )
        self.colour_combo.currentIndexChanged.connect(self._on_colour_mode_changed)
        colours.addWidget(self.colour_combo)
        self.colour_feature_combo = scrolling_combo()
        self.colour_feature_combo.setToolTip(
            "What the colour shows — used when Colour by is 'Feature value (1D)' or 'Precomputed colour (RGB)',\n"
            "greyed out otherwise. For a feature value:\n"
            "Any feature with a single series (its mean over the burst), or any of the axis columns.\n"
            "Dark is low, bright is high, between the two percentiles beside this box."
        )
        self.colour_feature_combo.setEnabled(False)
        self.colour_feature_combo.currentIndexChanged.connect(self._on_colour_feature_changed)
        colours.addWidget(self.colour_feature_combo, stretch=1)
        self.percentile_spins: list[QDoubleSpinBox] = []
        for tip in ("lowest", "highest"):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setSuffix(" %")
            spin.setToolTip(
                f"The percentile of the values shown that takes the {tip} colour; values past it are clipped\n"
                "onto it. Picking a value resets both: 0 to 95, or 5 to 95 when the value goes negative."
            )
            spin.setEnabled(False)
            spin.setValue(95.0 if tip == "highest" else 0.0)  # before the signal: nothing to redraw yet
            spin.valueChanged.connect(self._redraw)
            colours.addWidget(spin)
            self.percentile_spins.append(spin)
        self.baseline_check = QCheckBox("Baseline")
        self.baseline_check.setToolTip(
            "Draw, faintly behind the bursts, the same bursts shifted in time within their trial\n"
            f"({BASELINE_REPEATS} random shifts): where they would fall if their timing had nothing to do\n"
            "with these axes. Tick again for a fresh shuffle. Not drawn on the spike axes."
        )
        self.baseline_check.toggled.connect(self._on_baseline_toggled)
        colours.addSpacing(12)
        colours.addWidget(self.baseline_check)
        colours.addSpacing(12)
        colours.addWidget(QLabel("Arena:"))
        self.geometry_combo = scrolling_combo()
        self.geometry_combo.addItem("None", None)
        for geometry_name in sorted(self._geometries):
            self.geometry_combo.addItem(geometry_name, geometry_name)
        # Opens on the geometry the space plot shows, when it shows one.
        chosen = self.geometry_combo.findData(getattr(self.app_state, "space_library_geometry", None))
        self.geometry_combo.setCurrentIndex(max(0, chosen))
        self.geometry_combo.setToolTip(
            "A geometry of the space plot's library, drawn in white behind a map — shown while the axes\n"
            "are a feature's x and y (Map x/y), where its coordinates mean something."
        )
        self.geometry_combo.setEnabled(bool(self._geometries))
        self.geometry_combo.currentIndexChanged.connect(self._redraw)
        colours.addWidget(self.geometry_combo)
        colours.addStretch(1)
        axes.addSpacing(12)
        axes.addWidget(QLabel("X:"))
        self.x_combo = scrolling_combo()
        axes.addWidget(self.x_combo, stretch=1)
        self.x_label_combo = self._make_label_combo()
        axes.addWidget(self.x_label_combo)
        axes.addWidget(QLabel("Y:"))
        self.y_combo = scrolling_combo()
        axes.addWidget(self.y_combo, stretch=1)
        self.y_label_combo = self._make_label_combo()
        axes.addWidget(self.y_label_combo)
        self.log_checks: list[QCheckBox] = []
        for axis_name in ("X", "Y"):
            check = QCheckBox(f"Log {axis_name}")
            check.setToolTip(
                f"Draw the {axis_name} axis on a log scale — a proportional relationship is then a straight line,\n"
                "its slope the exponent. Bursts whose value is zero or negative have no place on it and are left out."
            )
            check.toggled.connect(self._on_log_toggled)
            axes.addWidget(check)
            self.log_checks.append(check)
        layout.addLayout(axes)
        layout.addLayout(colours)
        self._fill_axis_combos()
        self.y_combo.setCurrentIndex(min(1, self.y_combo.count() - 1))
        for combo in (self.x_combo, self.y_combo, self.x_label_combo, self.y_label_combo):
            combo.currentIndexChanged.connect(self._redraw)

        layout.addWidget(self._build_feature_box())
        layout.addWidget(self._build_pca_box())

        select = QHBoxLayout()
        self.region_check = QCheckBox("Region")
        self.region_check.setToolTip(
            "Show a rectangle on the plot: the video grid is limited to the bursts inside it.\n"
            "With a keypoint's x and y as the axes, that is 'only bursts while it was here'."
        )
        self.region_check.toggled.connect(self._on_region_toggled)
        select.addWidget(self.region_check)
        self.keep_btn = QPushButton("Keep region")
        self.keep_btn.setAutoDefault(False)
        self.keep_btn.setEnabled(False)
        self.keep_btn.setToolTip(
            "Limit the scatter and the video grid to the bursts in the region, whatever the axes show next.\n"
            "Put a keypoint's x and y on the axes, draw where it should be, keep it — then change the axes\n"
            "to compare only those bursts. Keeping a second region narrows the first."
        )
        self.keep_btn.clicked.connect(self.keep_region)
        select.addWidget(self.keep_btn)
        self.center_btn = QPushButton("Center")
        self.center_btn.setAutoDefault(False)
        self.center_btn.setToolTip("Fit the view around the points shown (also: double-click the plot)")
        self.center_btn.clicked.connect(self.center)
        select.addWidget(self.center_btn)
        self.clear_btn = QPushButton("Show all")
        self.clear_btn.setAutoDefault(False)
        self.clear_btn.setToolTip("Drop the selection: the video grid shows every burst again")
        self.clear_btn.clicked.connect(self._clear)
        select.addWidget(self.clear_btn)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: grey;")
        select.addWidget(self.count_label, stretch=1)
        layout.addLayout(select)

        hint = QLabel(
            "One point per burst · click to play the bursts nearest the click · drag to draw a region · "
            "double-click to centre · "
            "white rings: what the grid shows now · "
            "timing axes are burst onset minus the reference (negative = before)."
        )
        hint.setStyleSheet("color: grey; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        view_box = _RegionViewBox()
        view_box.region_drawn.connect(self.draw_region)
        self.plot = pg.PlotWidget(viewBox=view_box)
        self.plot.setBackground(_BG)
        self.plot.setMenuEnabled(False)
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        layout.addWidget(self.plot, stretch=1)
        self.resize(760, 720)
        self._redraw()

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------

    def _fill_axis_combos(self) -> None:
        for combo in (self.x_combo, self.y_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([*self.columns, *LABEL_AXES])
            if combo.findText(current) >= 0:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        self._fill_colour_feature_combo()

    def _fill_colour_feature_combo(self) -> None:
        """What the box beside Colour by offers: values for a colour scale, or the precomputed colours."""
        combo = self.colour_feature_combo
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if self.colour_combo.currentData() == "rgb":
            if self._rgb_choices is None:
                self._rgb_choices = rgb_choices(self.app_state)
            for text, feature, selections in self._rgb_choices:
                combo.addItem(text, ("rgb", (feature, selections)))
        else:
            for feature in self._single_series_features():
                combo.addItem(feature, ("feature", feature))
            for title in self.columns:
                combo.addItem(title, ("column", title))
        if combo.findText(current) >= 0:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _on_colour_mode_changed(self, *_args) -> None:
        self._fill_colour_feature_combo()
        self._on_colour_feature_changed()

    def _on_colour_feature_changed(self, *_args) -> None:
        """Another value on the colour scale: open its percentiles at the default for its sign, and redraw."""
        if self.colour_combo.currentData() == "feature":
            picked = self.colour_values()
            if picked is not None:
                for spin, percentile in zip(self.percentile_spins, default_percentiles(picked[1])):
                    spin.blockSignals(True)
                    spin.setValue(percentile)
                    spin.blockSignals(False)
        self._redraw()

    def _single_series_features(self) -> list[str]:
        """The behavioural features that are one series — nothing to pin before reading them."""
        if self._one_series is None:
            loader = base_loader(self.app_state)
            self._one_series = (
                [f for f in behavioural_features(self.app_state) if not loader.feature_dims(f)]
                if loader is not None
                else []
            )
        return self._one_series

    def colour_values(self) -> tuple[str, np.ndarray] | None:
        """The name and the per-burst values the colour scale shows, or ``None`` with nothing picked."""
        picked = self.colour_feature_combo.currentData()
        if picked is None or picked[0] == "rgb":
            return None
        kind, name = picked
        if kind == "column":
            return name, self.columns[name]
        if name not in self._colour_features:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._colour_features[name] = feature_column(self.app_state, self.bursts, name, {}, "mean")
            finally:
                QApplication.restoreOverrideCursor()
        return name, self._colour_features[name]

    def _make_label_combo(self) -> QComboBox:
        """The label class a :data:`LABEL_AXES` axis measures from — greyed out on any other axis."""
        combo = scrolling_combo()
        df = reference_labels(self.app_state)
        if df is not None:
            combo.addItems(sorted(df[REF].unique()))
        combo.setToolTip(
            'Which label the axis measures from — used by the two "Time from a label\'s nearest…" axes,\n'
            "greyed out on every other."
            if combo.count()
            else "No labels to measure from: load the session's labels (or a prediction set)\n"
            "besides the ones this grid shows."
        )
        combo.setEnabled(False)
        return combo

    def _axis(self, combo: QComboBox, label_combo: QComboBox) -> tuple[str, np.ndarray]:
        """The title and the values of the axis *combo* names."""
        text = combo.currentText()
        if text not in LABEL_AXES:
            return text, self.columns[text]
        ref = label_combo.currentText()
        if not ref:
            return text, np.full(len(self.bursts), np.nan)
        edge = LABEL_AXES[text]
        if (ref, edge) not in self._label_timing:
            self._label_timing[(ref, edge)] = label_timing_column(self.app_state, self.bursts, ref, edge)
        return f"Time from nearest {ref} {edge} (s)", self._label_timing[(ref, edge)]

    def add_column(
        self,
        title: str,
        values: np.ndarray,
        axis: QComboBox | None = None,
        recipe: Callable[[list], np.ndarray] | None = None,
    ) -> None:
        """Offer *values* as an axis named *title*, and put it on *axis* when given.

        *recipe* reads the same column for any bursts — what the baseline needs.
        """
        self.columns[title] = np.asarray(values, dtype=float)
        self._baseline_columns.pop(title, None)
        if recipe is not None:
            self._recipes[title] = recipe
        self._fill_axis_combos()
        if axis is not None:
            axis.setCurrentText(title)
        self._redraw()

    def _timing_recipe(self, title: str) -> Callable[[list], np.ndarray]:
        def read(bursts: list) -> np.ndarray:
            return timing_columns(self.app_state, bursts)[title]

        return read

    def _feature_recipe(self, feature: str, selections: dict[str, str], how: str) -> Callable[[list], np.ndarray]:
        def read(bursts: list) -> np.ndarray:
            return feature_column(self.app_state, bursts, feature, selections, how)

        return read

    @staticmethod
    def _pca_recipe(pca: BehaviourPCA, component: int) -> Callable[[list], np.ndarray]:
        def read(bursts: list) -> np.ndarray:
            return np.array([pca.at(b.trial, b.onset_s, b.offset_s, component) for b in bursts])

        return read

    def _next_axis(self) -> QComboBox:
        """Where a freshly added column goes: Y, unless Y already shows an added one and X does not."""
        builtin = set(timing_columns(self.app_state, self.bursts)) | set(SPIKE_FEATURES.values())
        if self.y_combo.currentText() not in builtin and self.x_combo.currentText() in builtin:
            return self.x_combo
        return self.y_combo

    def _build_pca_box(self) -> QGroupBox:
        box = QGroupBox("Behaviour PCA — where in behaviour space each burst happened")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.pca_features = QListWidget()
        self.pca_features.setMaximumHeight(84)
        self.pca_features.setToolTip(
            "The features the PCA reads, each with all its series (every keypoint, coordinate, individual).\n"
            "Untick anything that is not behaviour."
        )
        for feature in behavioural_features(self.app_state):
            item = QListWidgetItem(feature)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.pca_features.addItem(item)
        row.addWidget(self.pca_features, stretch=1)
        settings = QVBoxLayout()
        self.pca_window_spin = QDoubleSpinBox()
        self.pca_window_spin.setRange(0.001, 60.0)
        self.pca_window_spin.setDecimals(3)
        self.pca_window_spin.setSingleStep(0.05)
        self.pca_window_spin.setValue(float(self.app_state.get_with_default("burst_pca_window_s")))
        self.pca_window_spin.valueChanged.connect(lambda v: setattr(self.app_state, "burst_pca_window_s", float(v)))
        self.pca_window_spin.setSuffix(" s")
        self.pca_window_spin.setToolTip("Each feature is averaged over a window this long")
        self.pca_step_spin = QDoubleSpinBox()
        self.pca_step_spin.setRange(0.001, 60.0)
        self.pca_step_spin.setDecimals(3)
        self.pca_step_spin.setSingleStep(0.05)
        self.pca_step_spin.setValue(float(self.app_state.get_with_default("burst_pca_step_s")))
        self.pca_step_spin.valueChanged.connect(lambda v: setattr(self.app_state, "burst_pca_step_s", float(v)))
        self.pca_step_spin.setSuffix(" s")
        self.pca_step_spin.setToolTip("How far the window slides each time — shorter than the window, and they overlap")
        self.pca_components_spin = QSpinBox()
        self.pca_components_spin.setRange(2, 20)
        self.pca_components_spin.setValue(int(self.app_state.get_with_default("burst_pca_components")))
        self.pca_components_spin.valueChanged.connect(lambda v: setattr(self.app_state, "burst_pca_components", int(v)))
        self.pca_components_spin.setToolTip("How many components become axes")
        for text, spin in (
            ("Window:", self.pca_window_spin),
            ("Step:", self.pca_step_spin),
            ("Components:", self.pca_components_spin),
        ):
            line = QHBoxLayout()
            line.addWidget(QLabel(text))
            line.addWidget(spin)
            settings.addLayout(line)
        row.addLayout(settings)
        self.pca_btn = QPushButton("Run PCA")
        self.pca_btn.setAutoDefault(False)
        self.pca_btn.setToolTip(
            "Window every ticked feature over every trial shown, stack the trials, z-score each series\n"
            "(a gap becomes 0, the series' mean) and decompose. The components become axes: each burst\n"
            "sits at the mean of the windows inside it."
        )
        self.pca_btn.clicked.connect(self.run_pca)
        row.addWidget(self.pca_btn)
        lay.addLayout(row)
        self.pca_label = QLabel("")
        self.pca_label.setStyleSheet("color: grey; font-size: 10px;")
        self.pca_label.setWordWrap(True)
        lay.addWidget(self.pca_label)
        box.setEnabled(self.pca_features.count() > 0)
        return box

    def run_pca(self) -> None:
        """Fit the behaviour PCA and put its first two components on the axes."""
        items = [self.pca_features.item(i) for i in range(self.pca_features.count())]
        features = [item.text() for item in items if item is not None and item.checkState() == Qt.Checked]
        if not features:
            notify("Tick at least one feature for the PCA.", severity="warning")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pca = behaviour_pca(
                self.app_state,
                features,
                self.pca_window_spin.value(),
                self.pca_step_spin.value(),
                self.pca_components_spin.value(),
            )
        except ValueError as exc:
            notify(str(exc), severity="warning")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.show_pca(pca)

    def show_pca(self, pca: BehaviourPCA) -> None:
        """Offer *pca*'s components as axes (replacing an earlier run's) and show the first two."""
        for title in self._pca_titles:
            self.columns.pop(title, None)
            self._recipes.pop(title, None)
            self._baseline_columns.pop(title, None)
        self._pca_titles = []
        lines = []
        for k, fraction in enumerate(pca.explained):
            title = f"Behaviour PC{k + 1} ({fraction:.0%} of variance)"
            self._recipes[title] = self._pca_recipe(pca, k)
            self.columns[title] = self._recipes[title](self.bursts)
            self._pca_titles.append(title)
            weights = ", ".join(f"{'+' if w >= 0 else '−'}{name}" for name, w in pca.top_loadings(k))
            lines.append(f"PC{k + 1} {fraction:.0%}: {weights}")
        self.pca_label.setText("   ·   ".join(lines))
        self._fill_axis_combos()
        self.x_combo.setCurrentText(self._pca_titles[0])
        self.y_combo.setCurrentText(self._pca_titles[min(1, len(self._pca_titles) - 1)])
        self._redraw()

    def _build_feature_box(self) -> QGroupBox:
        box = QGroupBox("Add a behaviour feature as an axis")
        row = QHBoxLayout(box)
        self.feature_combo = scrolling_combo()
        loader = base_loader(self.app_state)
        self.feature_combo.addItems(loader.catalog.feature_choices() if loader is not None else [])
        self.feature_combo.currentIndexChanged.connect(self._rebuild_dim_combos)
        row.addWidget(self.feature_combo)
        self._dims_row = QHBoxLayout()
        row.addLayout(self._dims_row)
        self.summary_combo = scrolling_combo()
        for key, text in SUMMARIES.items():
            self.summary_combo.addItem(text, key)
        self.summary_combo.setToolTip(
            "How the feature becomes one number per burst: its mean or peak while the burst lasts,\n"
            "its value at burst onset, or its mean during the burst minus the mean just before."
        )
        row.addWidget(self.summary_combo)
        self.add_feature_btn = QPushButton("Add")
        self.add_feature_btn.setAutoDefault(False)
        self.add_feature_btn.clicked.connect(self._add_feature)
        row.addWidget(self.add_feature_btn)
        self.map_btn = QPushButton("Map x/y")
        self.map_btn.setAutoDefault(False)
        self.map_btn.setToolTip(
            "Put this feature's x and y on the axes, to scale: the scatter becomes a map of where it was\n"
            "during each burst. Drag over the part of space you want, then Keep region to limit to it."
        )
        self.map_btn.clicked.connect(self.add_map)
        row.addWidget(self.map_btn)
        row.addStretch()
        box.setEnabled(self.feature_combo.count() > 0)
        self._rebuild_dim_combos()
        return box

    def _rebuild_dim_combos(self, *_args) -> None:
        while self._dims_row.count():
            taken = self._dims_row.takeAt(0)
            widget = taken.widget() if taken is not None else None
            if widget is not None:
                widget.deleteLater()
        self._dim_combos = {}
        loader = base_loader(self.app_state)
        feature = self.feature_combo.currentText()
        if loader is None or not feature:
            return
        for dim, values in loader.feature_dims(feature).items():
            combo = scrolling_combo()
            combo.addItems([str(v) for v in values])
            combo.setToolTip(f"Which {dim} of {feature}")
            self._dims_row.addWidget(combo)
            self._dim_combos[dim] = combo
        self.map_btn.setEnabled(self._space_dim() is not None)

    def _space_dim(self) -> str | None:
        """The dim of the picked feature that holds an x and a y, if it has one."""
        for dim, combo in self._dim_combos.items():
            if combo.findText("x") >= 0 and combo.findText("y") >= 0:
                return dim
        return None

    def add_map(self) -> None:
        """The picked feature's x on X and y on Y — where it was during each burst."""
        dim = self._space_dim()
        if dim is None:
            return
        feature = self.feature_combo.currentText()
        how = self.summary_combo.currentData()
        pinned = {d: combo.currentText() for d, combo in self._dim_combos.items() if d != dim}
        where = f" [{', '.join(pinned.values())}]" if pinned else ""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            values = {
                axis: feature_column(self.app_state, self.bursts, feature, {**pinned, dim: axis}, how)
                for axis in ("x", "y")
            }
        except ValueError as exc:
            notify(str(exc), severity="warning")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if not (np.isfinite(values["x"]) & np.isfinite(values["y"])).any():
            notify(f"'{feature}' has no position during any of these bursts.", severity="warning")
            return
        for axis, combo in (("x", self.x_combo), ("y", self.y_combo)):
            title = f"{feature}{where} {axis} · {SUMMARIES[how]}"
            self._map_titles[title] = ("x", "y").index(axis)
            self.add_column(
                title,
                values[axis],
                combo,
                recipe=self._feature_recipe(feature, {**pinned, dim: axis}, how),
            )

    def _add_feature(self) -> None:
        feature = self.feature_combo.currentText()
        selections = {dim: combo.currentText() for dim, combo in self._dim_combos.items()}
        how = self.summary_combo.currentData()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            values = feature_column(self.app_state, self.bursts, feature, selections, how)
        except ValueError as exc:
            notify(str(exc), severity="warning")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if not np.isfinite(values).any():
            notify(f"'{feature}' has no values during any of these bursts.", severity="warning")
            return
        pinned = f" [{', '.join(selections.values())}]" if selections else ""
        self.add_column(
            f"{feature}{pinned} · {SUMMARIES[how]}",
            values,
            self._next_axis(),
            recipe=self._feature_recipe(feature, selections, how),
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _on_log_toggled(self, *_args) -> None:
        """A scale changed: a region drawn on the other one means nothing now, and the view is refitted."""
        self.region_check.setChecked(False)
        self._redraw()
        self.center()

    @staticmethod
    def _on_scale(values: np.ndarray, log: bool) -> np.ndarray:
        """*values* as the plot places them: themselves, or their log10 (NaN where that has none)."""
        if not log:
            return values
        values = np.asarray(values, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(values > 0.0, np.log10(values), np.nan)

    def _plot_axes(self) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]:
        """Both axes as ``(title, values)``, the values where the plot places them (log or not)."""
        x_title, x = self._axis(self.x_combo, self.x_label_combo)
        y_title, y = self._axis(self.y_combo, self.y_label_combo)
        log_x, log_y = (check.isChecked() for check in self.log_checks)
        return (x_title, self._on_scale(x, log_x)), (y_title, self._on_scale(y, log_y))

    def _xy(self) -> tuple[np.ndarray, np.ndarray]:
        (_x_title, x), (_y_title, y) = self._plot_axes()
        return x, y

    def _redraw(self, *_args) -> None:
        for combo, label_combo in ((self.x_combo, self.x_label_combo), (self.y_combo, self.y_label_combo)):
            label_combo.setEnabled(combo.currentText() in LABEL_AXES and label_combo.count() > 0)
        self.colour_feature_combo.setEnabled(self.colour_combo.currentData() in ("feature", "rgb"))
        for spin in self.percentile_spins:
            spin.setEnabled(self.colour_combo.currentData() == "feature")
        (x_title, x), (y_title, y) = self._plot_axes()
        # The axis writes its ticks as the values themselves; the points sit at their log10.
        for side, check in zip(("bottom", "left"), self.log_checks):
            self.plot.getPlotItem().getAxis(side).setLogMode(check.isChecked())
        mask = np.isfinite(x) & np.isfinite(y)
        if self._kept is not None:
            mask &= self._kept
        label_id = self.class_combo.currentData()
        if label_id is not None:
            mask &= self._class_ids == label_id
        self._shown = np.flatnonzero(mask)

        item = self.plot.getPlotItem()
        for old in [i for i in item.items if isinstance(i, pg.ScatterPlotItem) or getattr(i, "_is_reference", False)]:
            item.removeItem(old)
        item.setLabel("bottom", x_title)
        item.setLabel("left", y_title)
        # A map is drawn to scale, as the space plot draws it.
        item.setAspectLocked(x_title in self._map_titles and y_title in self._map_titles)
        self._draw_geometry(x_title, y_title)
        shown = self._shown
        if item.legend is None:
            item.addLegend(offset=(-10, 10))
        item.legend.clear()
        colours = self.point_colours()
        # Points the legend does not name are one item, each in its own colour: a colour scale or a
        # precomputed colour has as many colours as points.
        plain = np.array([i for i in shown if colours[i][0] is None and colours[i][1] is not None], dtype=int)
        if len(plain):
            brushes = [pg.mkBrush(colours[i][1]) for i in plain]
            item.addItem(pg.ScatterPlotItem(x=x[plain], y=y[plain], brush=brushes, pen=pg.mkPen(None), size=7))
        for name, color, members in self._colour_groups(shown, colours):
            # No colour: a burst mostly in no label, drawn hollow so it is neither a grey burst nor baseline.
            brush = pg.mkBrush(color) if color is not None else pg.mkBrush(None)
            pen = pg.mkPen(None) if color is not None else pg.mkPen(_NEUTRAL, width=1.5)
            item.addItem(pg.ScatterPlotItem(x=x[members], y=y[members], brush=brush, pen=pen, size=7, name=name))
        baseline_text = self._draw_baseline(label_id)
        ringed = np.array([i for i in shown if self.keys[i] in self._marked], dtype=int)
        if len(ringed):
            rings = pg.ScatterPlotItem(
                x=x[ringed], y=y[ringed], brush=pg.mkBrush(None), pen=pg.mkPen("w", width=2), size=13
            )
            rings.setZValue(5)
            item.addItem(rings)
        dropped = int(len(self.bursts) if label_id is None else (self._class_ids == label_id).sum()) - len(shown)
        text = f"{len(shown)} bursts"
        if self._kept is not None:
            text += f" · kept: {self._kept_text}"
        elif dropped:
            text += f" · {dropped} without a value on these axes"
        if baseline_text:
            text += f" · {baseline_text}"
        if self._scale_text:
            text += f" · {self._scale_text}"
        self.count_label.setText(text)

    def geometry_segments(self, x_title: str, y_title: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """The chosen geometry's edges as ``(xs, ys)`` on the current axes — none unless they are a map."""
        name = self.geometry_combo.currentData()
        if name is None or x_title not in self._map_titles or y_title not in self._map_titles:
            return []
        column_x, column_y = self._map_titles[x_title], self._map_titles[y_title]
        segments = []
        for ref in self._geometries[name]:
            vertices = ref.vertices
            for start, end in ref.edges:
                segments.append(
                    (
                        self._on_scale(vertices[[start, end], column_x], self.log_checks[0].isChecked()),
                        self._on_scale(vertices[[start, end], column_y], self.log_checks[1].isChecked()),
                    )
                )
        return segments

    def _draw_geometry(self, x_title: str, y_title: str) -> None:
        item = self.plot.getPlotItem()
        for xs, ys in self.geometry_segments(x_title, y_title):
            # White whatever the file says: its own colour is chosen for the space plot's background.
            line = pg.PlotCurveItem(x=xs, y=ys, pen=pg.mkPen("w", width=1.5))
            line._is_reference = True
            line.setZValue(-8)
            item.addItem(line)

    def _on_baseline_toggled(self, on: bool) -> None:
        """Shift the bursts afresh (or drop the shifted ones) and redraw."""
        self._baseline = []
        self._baseline_columns = {}
        if on:
            durations = trial_durations(self.app_state, self.bursts)
            if not durations:
                notify("The session has no trial timing to shift the bursts within.", severity="warning")
                self.baseline_check.blockSignals(True)
                self.baseline_check.setChecked(False)
                self.baseline_check.blockSignals(False)
                return
            self._baseline = shifted_bursts(self.bursts, durations)
        self._redraw()

    def _baseline_axis(self, combo: QComboBox, label_combo: QComboBox) -> np.ndarray | None:
        """The axis *combo* names, read for the shifted bursts — ``None`` when it cannot be."""
        text = combo.currentText()
        key = text
        if text in LABEL_AXES:
            ref, edge = label_combo.currentText(), LABEL_AXES[text]
            if not ref:
                return None
            key = f"{text} · {ref}"
            if key not in self._baseline_columns:
                self._baseline_columns[key] = label_timing_column(self.app_state, self._baseline, ref, edge)
        elif key not in self._baseline_columns:
            recipe = self._recipes.get(text)
            if recipe is None:
                return None
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._baseline_columns[key] = np.asarray(recipe(self._baseline), dtype=float)
            finally:
                QApplication.restoreOverrideCursor()
        return self._baseline_columns[key]

    def baseline_points(self, label_id: int | None) -> tuple[np.ndarray, np.ndarray] | None:
        """The shifted bursts on the current axes (of *label_id*'s class when given), else ``None``."""
        if not self._baseline:
            return None
        x = self._baseline_axis(self.x_combo, self.x_label_combo)
        y = self._baseline_axis(self.y_combo, self.y_label_combo)
        if x is None or y is None:
            return None
        x = self._on_scale(x, self.log_checks[0].isChecked())
        y = self._on_scale(y, self.log_checks[1].isChecked())
        mask = np.isfinite(x) & np.isfinite(y)
        if label_id is not None:
            mask &= np.array([b.label_id == label_id for b in self._baseline])
        return x[mask], y[mask]

    def _draw_baseline(self, label_id: int | None) -> str:
        """Draw the shifted bursts on the current axes; returns what to say about them (nothing when off)."""
        if not self._baseline:
            return ""
        points = self.baseline_points(label_id)
        if points is None:
            return "no baseline on these axes"
        cloud = pg.ScatterPlotItem(
            x=points[0],
            y=points[1],
            symbol="+",
            brush=pg.mkBrush(*_BASELINE_RGBA),
            pen=pg.mkPen(*_BASELINE_RGBA),
            size=9,
            name="baseline (shifted bursts)",
        )
        cloud.setZValue(-5)
        self.plot.getPlotItem().addItem(cloud)
        return f"baseline: {len(points[0])} shifted bursts"

    def point_colours(self) -> list[tuple[str | None, str | None]]:
        """Per burst, its ``(legend name, colour)`` under the chosen colouring.

        No name: not in the legend. No colour: drawn hollow (mostly in no label).
        """
        mode = self.colour_combo.currentData()
        self._scale_text = ""
        if mode == "class":
            return [(None, b.color_hex) for b in self.bursts]
        if mode == "none":
            return [(None, _NEUTRAL)] * len(self.bursts)
        if mode == "feature":
            return self._scale_colours()
        if mode == "rgb":
            return self._rgb_colours()
        if self._covering is None:
            self._covering = covering_labels(self.app_state, self.bursts)
        return [("no label", None) if c is None else c for c in self._covering]

    def _scale_colours(self) -> list[tuple[str | None, str | None]]:
        """Each burst's colour on the scale of the picked value; hollow where it has none.

        The scale spans the bursts drawn, so one unit's range is not flattened by another's.
        """
        hollow: list[tuple[str | None, str | None]] = [("no value", None)] * len(self.bursts)
        picked = self.colour_values()
        if picked is None or len(self._shown) == 0:
            return hollow
        name, values = picked
        low, high = (spin.value() for spin in self.percentile_spins)
        scaled, lo, hi = scale_between_percentiles(values[self._shown], low, high)
        if not np.isfinite(scaled).any():
            return hollow
        table = pg.colormap.get(_SCALE_MAP).getLookupTable(nPts=_SCALE_STEPS)
        self._scale_text = f"colour: {name} from {lo:.3g} (dark) to {hi:.3g} (bright)"
        out = hollow
        for i, value in zip(self._shown, scaled):
            if np.isfinite(value):
                r, g, b = (int(c) for c in table[int(round(value * (_SCALE_STEPS - 1)))][:3])
                out[int(i)] = (None, f"#{r:02x}{g:02x}{b:02x}")
        return out

    def _rgb_colours(self) -> list[tuple[str | None, str | None]]:
        """Each burst in the dataset's own colour for it; hollow where the feature has none."""
        hollow: list[tuple[str | None, str | None]] = [("no value", None)] * len(self.bursts)
        picked = self.colour_feature_combo.currentData()
        if picked is None or picked[0] != "rgb":
            return hollow
        text = self.colour_feature_combo.currentText()
        if text not in self._rgb_columns:
            feature, selections = picked[1]
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._rgb_columns[text] = rgb_column(self.app_state, self.bursts, feature, selections)
            except ValueError as exc:
                notify(str(exc), severity="warning")
                return hollow
            finally:
                QApplication.restoreOverrideCursor()
        out = hollow
        for i, colour in enumerate(self._rgb_columns[text]):
            if np.isfinite(colour).all():
                r, g, b = (int(round(c * 255)) for c in colour)
                out[i] = (None, f"#{r:02x}{g:02x}{b:02x}")
        return out

    def _colour_groups(
        self, shown: np.ndarray, colours: list[tuple[str | None, str | None]]
    ) -> list[tuple[str | None, str | None, np.ndarray]]:
        """The drawn bursts that share a mark: ``(legend name, colour, indices)``.

        Only the named ones (a legend entry each) and the hollow ones; the
        unnamed filled points are one item with a colour per point.
        """
        groups: dict[tuple[str | None, str | None], list[int]] = {}
        for i in shown:
            if colours[i][0] is not None or colours[i][1] is None:
                groups.setdefault(colours[i], []).append(int(i))
        return [(name, color, np.array(members, dtype=int)) for (name, color), members in groups.items()]

    def mark(self, keys: set[tuple]) -> None:
        """Ring the bursts whose clips the grid shows now."""
        self._marked = set(keys)
        self._redraw()

    def center(self) -> None:
        """Fit the view around the points shown."""
        x, y = self._xy()
        shown = self._shown
        if len(shown) == 0:
            return

        def span(values: np.ndarray) -> tuple[float, float]:
            lo, hi = float(values.min()), float(values.max())
            # A single value has no extent to fit: give it room either side.
            return (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)

        self.plot.getPlotItem().getViewBox().setRange(xRange=span(x[shown]), yRange=span(y[shown]), padding=0.05)

    def _on_click(self, event) -> None:
        view = self.plot.getPlotItem().getViewBox()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        if event.double():
            # The click that began the double click must not also pick bursts.
            self._click_timer.stop()
            self.center()
            return
        if self.region_check.isChecked() or len(self._shown) == 0:
            return
        point = view.mapSceneToView(event.scenePos())
        self._pending_click = (float(point.x()), float(point.y()))
        self._click_timer.start(QApplication.doubleClickInterval())

    def _on_single_click(self) -> None:
        self.select_nearest(*self._pending_click)

    def select_nearest(self, x0: float, y0: float) -> None:
        """Select the bursts nearest ``(x0, y0)``: the nearest point's class, as many as the grid shows."""
        x, y = self._xy()
        shown = self._shown
        closest = nearest_points(x[shown], y[shown], x0, y0, 1)
        if len(closest) == 0:
            return
        same = shown[self._class_ids[shown] == self._class_ids[shown[closest[0]]]]
        picked = same[nearest_points(x[same], y[same], x0, y0, self._n_nearest_fn())]
        self.selected.emit({self.keys[i] for i in picked}, f"{len(picked)} bursts nearest the scatter click")

    def _on_region_toggled(self, on: bool) -> None:
        self.keep_btn.setEnabled(on)
        item = self.plot.getPlotItem()
        if self._roi is not None:
            item.removeItem(self._roi)
            self._roi = None
        if not on:
            return
        (x0, x1), (y0, y1) = item.getViewBox().viewRange()
        width, height = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        self._place_region(x0 + width / 2.0, x0 + 1.5 * width, y0 + height / 2.0, y0 + 1.5 * height)

    def _place_region(self, x0: float, x1: float, y0: float, y1: float) -> None:
        self._roi = pg.RectROI([x0, y0], [x1 - x0, y1 - y0], pen=pg.mkPen("w", width=2))
        self._roi.addScaleHandle([0, 0], [1, 1])
        self._roi.setZValue(10)
        self._roi.sigRegionChangeFinished.connect(self._on_region_changed)
        self.plot.getPlotItem().addItem(self._roi)
        self._on_region_changed()

    def draw_region(self, x0: float, x1: float, y0: float, y1: float) -> None:
        """A rectangle drawn on the plot becomes the region, replacing the one there."""
        if x0 == x1 or y0 == y1:
            return
        self.region_check.blockSignals(True)
        self.region_check.setChecked(True)
        self.region_check.blockSignals(False)
        self.keep_btn.setEnabled(True)
        if self._roi is not None:
            self.plot.getPlotItem().removeItem(self._roi)
        self._place_region(x0, x1, y0, y1)

    def _on_region_changed(self, *_args) -> None:
        if self._roi is None:
            return
        pos, size = self._roi.pos(), self._roi.size()
        self.select_region(pos.x(), pos.x() + size.x(), pos.y(), pos.y() + size.y())

    def select_region(self, x0: float, x1: float, y0: float, y1: float) -> None:
        """Select every drawn burst inside the rectangle (none inside leaves the grid as it is)."""
        x, y = self._xy()
        shown = self._shown
        inside = shown[points_in_rect(x[shown], y[shown], x0, x1, y0, y1)]
        if len(inside) == 0:
            self.count_label.setText("No bursts in the region.")
            return
        self.selected.emit({self.keys[i] for i in inside}, f"{len(inside)} bursts in the scatter region")

    def keep_region(self) -> None:
        """Make the region a filter: only its bursts stay on the scatter and in the grid."""
        if self._roi is None:
            return
        pos, size = self._roi.pos(), self._roi.size()
        x0, x1, y0, y1 = pos.x(), pos.x() + size.x(), pos.y(), pos.y() + size.y()
        (x_title, x), (y_title, y) = self._plot_axes()
        # Said in the values' own units, whatever scale the rectangle was drawn on.
        if self.log_checks[0].isChecked():
            x0, x1 = 10.0**x0, 10.0**x1
        if self.log_checks[1].isChecked():
            y0, y1 = 10.0**y0, 10.0**y1
        shown = self._shown
        inside = shown[points_in_rect(x[shown], y[shown], x0, x1, y0, y1)]
        if len(inside) == 0:
            self.count_label.setText("No bursts in the region.")
            return
        self._kept = np.zeros(len(self.bursts), dtype=bool)
        self._kept[inside] = True
        described = f"{x_title} {x0:.3g} to {x1:.3g}, {y_title} {y0:.3g} to {y1:.3g}"
        self._kept_text = f"{self._kept_text}; {described}" if self._kept_text else described
        self.region_check.setChecked(False)
        self._redraw()
        self.selected.emit({self.keys[i] for i in inside}, f"{len(inside)} bursts kept from the scatter region")

    def _clear(self) -> None:
        self._kept = None
        self._kept_text = ""
        self.region_check.setChecked(False)
        self._redraw()
        self.cleared.emit()
