"""Tools ▸ *Data: Find smoothing parameters…* — try a preprocessing pipeline, copy the call.

The GUI never transforms data. This dialog previews a pipeline over the
active line plot while it is open and writes the Python that reproduces it.
Closing the dialog puts the panel back as it was; the only thing remembered
is the last pipeline, in ``app_state.function_params_cache``.

The pipeline is three stages, run in this order:

1. **Filters** — movement's ``filter_by_confidence`` and ``rolling_filter``.
2. **Gaps** — what to do with NaNs: our ``bridge_gaps`` (seconds) or
   movement's ``interpolate_over_time``.
3. **Smoothing** — the kernels in ``features/preprocessing.SMOOTHERS``, run
   through ``features/timeaxis.apply_along_time``; they never create or
   remove a gap.

Each stage holds any number of steps, reordered by drag and drop. The plot
draws the raw data faded, every step's output thin in its own colour and the
final result thick, so the effect of each step is visible on its own. Every
step links to the documentation of what it calls and to its source. The
preview and the snippet run the same maths; parameters are seconds or Hz,
resolved against the plotted data's own rate, and a step too short for that
rate says so in the status line instead of drawing.
"""

from __future__ import annotations

import inspect
import webbrowser
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.features.preprocessing import SMOOTHERS, bridge_gaps, nan_aware
from ethograph.features.timeaxis import sampling_rate
from ethograph.gui.active_panel import PanelKind
from ethograph.gui.notify import notify
from ethograph.gui.plots_lineplot import LinePreview
from ethograph.gui.theme import monospace_font
from ethograph.io.catalog import PynappleLoader

CACHE_KEY = "smoothing"
_SLIDER_STEPS = 1000
_SCIPY = "https://docs.scipy.org/doc/scipy/reference/generated/"
_MOVEMENT = "https://movement.neuroinformatics.dev/latest/api/"

STAGES: tuple[tuple[str, str], ...] = (("filter", "1. Filters"), ("gaps", "2. Gaps"), ("smooth", "3. Smoothing"))
STAGE_ORDER = {key: i for i, (key, _) in enumerate(STAGES)}


# ---------------------------------------------------------------------------
# Parameter specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """A continuous parameter on a slider, in seconds, Hz or a ratio."""

    name: str
    unit: str
    low: float
    high: float
    default: float
    decimals: int = 3
    log: bool = True


@dataclass(frozen=True)
class IntSpec:
    name: str
    low: int
    high: int
    default: int


@dataclass(frozen=True)
class ChoiceSpec:
    name: str
    choices: tuple[str, ...]
    default: str


AnySpec = ParamSpec | IntSpec | ChoiceSpec


@dataclass(frozen=True)
class Companions:
    """What a step may need besides the plotted values."""

    confidence: np.ndarray | None = None


@dataclass(frozen=True)
class StepSpec:
    """One kind of step: its stage, how it is previewed, how it is spelled, where it is documented."""

    label: str
    stage: str
    doc_url: str
    params: tuple[AnySpec, ...]
    #: ``run(x, fs, params, companions) -> same shape``
    run: Callable[[np.ndarray, float, dict[str, Any], Companions], np.ndarray]
    #: ``expression(params) -> the right-hand side applied to ``x``
    expression: Callable[[dict[str, Any]], str]
    imports: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()
    #: How far past the view to fetch, in seconds, for the given params.
    pad_s: Callable[[dict[str, Any]], float] = lambda params: 0.0
    #: Whether the snippet line needs ``fs`` (a window in samples).
    needs_fs: bool = False
    source: Callable[[], str] | None = None
    #: movement's functions take a DataArray: not for pynapple, not for the console.
    movement_only: bool = False

    def default_params(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


def _fmt(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _kwargs(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={_fmt(v)}" for k, v in params.items())


# --- smoothing kernels: nan_aware with the default policy keeps every remaining gap ---


def _kernel_step(kind: str, label: str, doc_url: str, params: tuple[AnySpec, ...], pad: Callable) -> StepSpec:
    kernel = SMOOTHERS[kind]
    return StepSpec(
        label=label,
        stage="smooth",
        doc_url=doc_url,
        params=params,
        run=lambda x, fs, p, _c: nan_aware(kernel)(x, fs, axis=0, **p),
        expression=lambda p: f"eto.apply_along_time(x, nan_aware({kind}), {_kwargs(p)})",
        imports=(f"from ethograph.features.preprocessing import nan_aware, {kind}",),
        pad_s=pad,
        source=lambda: inspect.getfile(kernel),
    )


# --- gaps -------------------------------------------------------------------


def _run_bridge_gaps(x, fs, p, _c):
    max_gap = None if p["which"] == "every gap" else p["max_gap_s"]
    return bridge_gaps(x, fs, max_gap_s=max_gap, axis=0)


def _bridge_gaps_expression(p) -> str:
    max_gap = "None" if p["which"] == "every gap" else _fmt(p["max_gap_s"])
    return f"eto.apply_along_time(x, bridge_gaps, max_gap_s={max_gap})"


# --- movement filtering -----------------------------------------------------


def _samples(seconds: float, fs: float) -> int:
    return max(1, int(round(seconds * fs)))


def _run_filter_by_confidence(x, _fs, p, companions: Companions):
    if companions.confidence is None:
        raise ValueError("filter_by_confidence needs a 'confidence' variable aligned with this panel's feature")
    conf = companions.confidence
    if x.ndim == 2 and conf.ndim == 1:
        conf = conf[:, None]
    return np.where(conf >= p["threshold"], x, np.nan)


def _run_rolling(x, fs, p, _c):
    """movement.filtering.rolling_filter: reflect-padded, centred, NaN-skipping."""
    window = _samples(p["window_s"], fs)
    half = window // 2
    if half == 0:
        return x.astype(float)
    padded = np.pad(x.astype(float), [(half, half)] + [(0, 0)] * (x.ndim - 1), mode="reflect")
    frame = pd.DataFrame(padded.reshape(len(padded), -1))
    rolled = getattr(frame.rolling(window, center=True, min_periods=1), p["statistic"])().to_numpy()
    return rolled[half:-half].reshape(x.shape)


def _run_interpolate(x, fs, p, _c):
    """movement.filtering.interpolate_over_time: fill gaps of at most max_gap samples."""
    return bridge_gaps(x, fs, max_gap_s=_samples(p["max_gap_s"], fs) / fs, axis=0)


def _movement_source() -> str:
    import movement.filtering

    return inspect.getfile(movement.filtering)


STEPS: dict[str, StepSpec] = {
    # --- 1. filters
    "filter_by_confidence": StepSpec(
        label="filter by confidence (movement)",
        stage="filter",
        doc_url=_MOVEMENT + "movement.filtering.filter_by_confidence.html",
        params=(ParamSpec("threshold", "", 0.0, 1.0, 0.6, decimals=2, log=False),),
        run=_run_filter_by_confidence,
        expression=lambda p: f'filter_by_confidence(x, ds["confidence"], threshold={_fmt(p["threshold"])})',
        imports=("from movement.filtering import filter_by_confidence",),
        needs=("confidence",),
        source=_movement_source,
        movement_only=True,
    ),
    "rolling_filter": StepSpec(
        label="rolling filter (movement)",
        stage="filter",
        doc_url=_MOVEMENT + "movement.filtering.rolling_filter.html",
        params=(
            ParamSpec("window_s", "s", 0.01, 2.0, 0.2),
            ChoiceSpec("statistic", ("median", "mean", "max", "min"), "median"),
        ),
        run=_run_rolling,
        expression=lambda p: (
            f"rolling_filter(x, window=round({_fmt(p['window_s'])} * fs), statistic={p['statistic']!r})"
        ),
        imports=("from movement.filtering import rolling_filter",),
        pad_s=lambda p: 0.5 * p["window_s"],
        needs_fs=True,
        source=_movement_source,
        movement_only=True,
    ),
    # --- 2. gaps
    "bridge_gaps": StepSpec(
        label="bridge gaps",
        stage="gaps",
        doc_url="https://numpy.org/doc/stable/reference/generated/numpy.interp.html",
        params=(
            ChoiceSpec("which", ("gaps up to max_gap_s", "every gap"), "gaps up to max_gap_s"),
            ParamSpec("max_gap_s", "s", 0.01, 5.0, 0.2),
        ),
        run=_run_bridge_gaps,
        expression=_bridge_gaps_expression,
        imports=("from ethograph.features.preprocessing import bridge_gaps",),
        pad_s=lambda p: 0.0 if p["which"] == "every gap" else p["max_gap_s"],
        source=lambda: inspect.getfile(bridge_gaps),
    ),
    "interpolate_over_time": StepSpec(
        label="interpolate over time (movement)",
        stage="gaps",
        doc_url=_MOVEMENT + "movement.filtering.interpolate_over_time.html",
        params=(ParamSpec("max_gap_s", "s", 0.01, 5.0, 0.2),),
        run=_run_interpolate,
        expression=lambda p: f"interpolate_over_time(x, max_gap=round({_fmt(p['max_gap_s'])} * fs))",
        imports=("from movement.filtering import interpolate_over_time",),
        pad_s=lambda p: p["max_gap_s"],
        needs_fs=True,
        source=_movement_source,
        movement_only=True,
    ),
    # --- 3. smoothing
    "gaussian": _kernel_step(
        "gaussian",
        "Gaussian",
        _SCIPY + "scipy.ndimage.gaussian_filter1d.html",
        (ParamSpec("sigma_s", "s", 0.002, 0.5, 0.05),),
        lambda p: 3.0 * p["sigma_s"],
    ),
    "savgol": _kernel_step(
        "savgol",
        "Savitzky-Golay",
        _SCIPY + "scipy.signal.savgol_filter.html",
        (ParamSpec("window_s", "s", 0.01, 2.0, 0.3), IntSpec("polyorder", 1, 6, 3)),
        lambda p: 0.5 * p["window_s"],
    ),
    "median": _kernel_step(
        "median",
        "Median",
        _SCIPY + "scipy.ndimage.median_filter.html",
        (ParamSpec("window_s", "s", 0.01, 2.0, 0.2),),
        lambda p: 0.5 * p["window_s"],
    ),
    "moving_average": _kernel_step(
        "moving_average",
        "Moving average",
        _SCIPY + "scipy.ndimage.uniform_filter1d.html",
        (ParamSpec("window_s", "s", 0.01, 2.0, 0.2),),
        lambda p: 0.5 * p["window_s"],
    ),
    "lowpass": _kernel_step(
        "lowpass",
        "Butterworth low-pass",
        _SCIPY + "scipy.signal.sosfiltfilt.html",
        (ParamSpec("cutoff_hz", "Hz", 0.1, 100.0, 3.0, decimals=2), IntSpec("order", 1, 8, 2)),
        lambda p: 4.0 / p["cutoff_hz"],
    ),
}
assert set(SMOOTHERS) <= set(STEPS), "every kernel is a step"
assert {spec.stage for spec in STEPS.values()} == set(STAGE_ORDER), "every step belongs to a stage"


# ---------------------------------------------------------------------------
# The pipeline: what the dialog has dialled in
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    kind: str
    params: dict[str, Any]

    @property
    def spec(self) -> StepSpec:
        return STEPS[self.kind]

    def label(self) -> str:
        return f"{self.spec.label}  ({_kwargs(self.params)})"

    def with_param(self, name: str, value: Any) -> Step:
        return Step(self.kind, {**self.params, name: value})

    @classmethod
    def default(cls, kind: str) -> Step:
        return cls(kind, STEPS[kind].default_params())


@dataclass(frozen=True)
class Pipeline:
    """Steps in execution order: every filter, then every gap step, then every kernel."""

    steps: tuple[Step, ...] = (Step("gaussian", {"sigma_s": 0.05}),)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.steps, key=lambda s: STAGE_ORDER[s.spec.stage]))
        object.__setattr__(self, "steps", ordered)

    def stage(self, stage: str) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if s.spec.stage == stage)

    def with_stage(self, stage: str, steps: tuple[Step, ...]) -> Pipeline:
        others = tuple(s for s in self.steps if s.spec.stage != stage)
        return Pipeline(others + steps)

    def replace(self, old: Step, new: Step) -> Pipeline:
        return Pipeline(tuple(new if s is old else s for s in self.steps))

    def run_stages(self, x: np.ndarray, fs: float, companions: Companions) -> list[tuple[str, np.ndarray]]:
        """Every step's output in order, labelled, the last being the result."""
        out = []
        for step in self.steps:
            x = np.asarray(step.spec.run(np.asarray(x, dtype=float), fs, step.params, companions))
            out.append((step.spec.label, x))
        return out

    def run(self, x: np.ndarray, fs: float, companions: Companions) -> np.ndarray:
        stages = self.run_stages(x, fs, companions)
        return stages[-1][1] if stages else np.asarray(x, dtype=float)

    def pad_s(self) -> float:
        return sum(step.spec.pad_s(step.params) for step in self.steps)

    def needs(self) -> set[str]:
        return {n for step in self.steps for n in step.spec.needs}

    def has_movement_step(self) -> bool:
        return any(step.spec.movement_only for step in self.steps)

    def imports(self) -> list[str]:
        by_module: dict[str, set[str]] = {}
        for step in self.steps:
            for line in step.spec.imports:
                module, names = line.split(" import ")
                by_module.setdefault(module, set()).update(n.strip() for n in names.split(","))
        return [f"{module} import {', '.join(sorted(names))}" for module, names in by_module.items()]

    def to_cache(self) -> dict[str, Any]:
        return {"steps": [{"kind": s.kind, "params": dict(s.params)} for s in self.steps]}

    @classmethod
    def from_cache(cls, cached: dict[str, Any] | None) -> Pipeline:
        if not cached or not isinstance(cached.get("steps"), list):
            return cls()
        steps = []
        for entry in cached["steps"]:
            kind = entry.get("kind")
            if kind not in STEPS:
                continue
            params = STEPS[kind].default_params()
            given = entry.get("params", {})
            for spec in STEPS[kind].params:
                if spec.name in given:
                    params[spec.name] = given[spec.name]
            steps.append(Step(kind, params))
        return cls(tuple(steps))


def smoothing_snippet(pipeline: Pipeline, feature: str, selections: dict[str, str], backend: str) -> str:
    """The script that reproduces the preview on the whole variable.

    *backend* is ``"xarray"`` or ``"pynapple"``; the steps are the same, only
    the way the session is opened differs. movement's functions take a
    DataArray, so on pynapple they are left as a comment. The panel's
    selections are quoted: the pipeline runs over every keypoint and
    individual at once.
    """
    if backend == "pynapple":
        opener = 'data, _ = eto.load_nap_data("session.nwb")'
        holder = "data"
        fs_line = "fs = x.rate"
    else:
        opener = 'ds = eto.open("data.nc")'
        holder = "ds"
        fs_line = "fs = eto.sampling_rate(eto.get_time_coord(x).values)"
    pinned = ", ".join(f"{k}={v}" for k, v in sorted(selections.items()))
    comment = "# runs over the whole variable"
    if pinned:
        comment = f"# previewed on {pinned}; this runs over the whole variable"

    lines = ["import ethograph as eto", *pipeline.imports(), "", opener, comment, f'x = {holder}["{feature}"]']
    if any(step.spec.needs_fs for step in pipeline.steps):
        lines.append(fs_line)
    for step in pipeline.steps:
        rhs = step.spec.expression(step.params)
        if backend == "pynapple" and step.spec.movement_only:
            lines.append(f"# x = {rhs}  # movement filters take an xarray DataArray")
        else:
            lines.append(f"x = {rhs}")
    lines.append(f'{holder}["{feature}_clean"] = x')
    return "\n".join(lines)


def console_line(pipeline: Pipeline, bound_name: str) -> str | None:
    """The same chain on the console's bound variable and its time vector ``t``.

    ``None`` when a step is movement's: those take a DataArray, and the
    console holds numpy.
    """
    if pipeline.has_movement_step() or not pipeline.steps:
        return None
    expr = bound_name
    for step in pipeline.steps:
        rhs = step.spec.expression(step.params)
        # the snippet's ``eto.apply_along_time(x, …)`` becomes the console's bare name with the panel's rate
        inner = rhs[len("eto.apply_along_time(x, ") : -1]
        expr = f"apply_along_time({expr}, {inner}, fs=sampling_rate(t))"
    return f"{bound_name}_clean = {expr}"


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


def open_smoothing_dialog(plot_container, app_state, parent=None) -> QDialog:
    dlg = SmoothingDialog(plot_container, app_state, parent=parent)
    dlg.show()
    return dlg


class _StageBox(QGroupBox):
    """One stage: its steps as a drag-reorderable list, with Add and Remove."""

    def __init__(self, stage: str, title: str, dialog: SmoothingDialog):
        super().__init__(title)
        self.stage = stage
        self._dialog = dialog
        layout = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list.setMinimumHeight(90)
        model = self.list.model()
        assert model is not None
        model.rowsMoved.connect(lambda *_: dialog._on_rows_moved(self))
        self.list.currentRowChanged.connect(lambda _row: dialog._on_selection_changed(self))
        layout.addWidget(self.list, 1)
        row = QHBoxLayout()
        add = QToolButton()
        add.setText("Add ▾")
        add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(add)
        for kind, spec in STEPS.items():
            if spec.stage == stage:
                menu.addAction(spec.label, lambda _checked=False, k=kind: dialog._add_step(k))
        add.setMenu(menu)
        row.addWidget(add)
        remove = QPushButton("Remove")
        remove.setAutoDefault(False)
        remove.clicked.connect(lambda: dialog._remove_step(self))
        row.addWidget(remove)
        row.addStretch()
        layout.addLayout(row)

    def steps_in_list_order(self, steps: tuple[Step, ...]) -> tuple[Step, ...]:
        order = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            assert item is not None
            order.append(steps[item.data(Qt.ItemDataRole.UserRole)])
        return tuple(order)


class SmoothingDialog(QDialog):
    """Three stage boxes across the top, the selected step's parameters and the code below."""

    def __init__(self, plot_container, app_state, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find smoothing parameters")
        self.setModal(False)
        self._container = plot_container
        self._app_state = app_state
        self._plot = None
        self._updating = False
        self._selected: Step | None = None

        cache = getattr(app_state, "function_params_cache", None) or {}
        self._pipeline = Pipeline.from_cache(cache.get(CACHE_KEY))

        root = QVBoxLayout(self)
        self.target_label = QLabel()
        self.target_label.setWordWrap(True)
        root.addWidget(self.target_label)

        boxes = QHBoxLayout()
        self.boxes: dict[str, _StageBox] = {}
        for i, (stage, title) in enumerate(STAGES):
            if i:
                arrow = QLabel("→")
                arrow.setStyleSheet("font-size: 20px; color: gray;")
                boxes.addWidget(arrow)
            box = _StageBox(stage, title, self)
            self.boxes[stage] = box
            boxes.addWidget(box, 1)
        root.addLayout(boxes)

        lower = QHBoxLayout()
        root.addLayout(lower, 1)

        params_col = QVBoxLayout()
        lower.addLayout(params_col, 0)
        self.step_title = QLabel("Select a step to edit its parameters.")
        self.step_title.setStyleSheet("font-weight: bold;")
        params_col.addWidget(self.step_title)
        self.param_form = QFormLayout()
        params_col.addLayout(self.param_form)
        links = QHBoxLayout()
        self.docs_btn = QPushButton("Documentation")
        self.docs_btn.setAutoDefault(False)
        self.docs_btn.setStyleSheet(
            "QPushButton { color: #4da6ff; border: none; font-weight: bold; }"
            "QPushButton:hover { color: #80bfff; text-decoration: underline; }"
        )
        self.docs_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.docs_btn.clicked.connect(self._open_docs)
        links.addWidget(self.docs_btn)
        self.source_btn = QPushButton("Open source code")
        self.source_btn.setAutoDefault(False)
        self.source_btn.setStyleSheet(self.docs_btn.styleSheet())
        self.source_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.source_btn.clicked.connect(self._open_source)
        links.addWidget(self.source_btn)
        links.addStretch()
        params_col.addLayout(links)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumWidth(380)
        params_col.addWidget(self.status_label)
        footer = QLabel(
            "Raw draws faded, each step thin in its own colour, the result thick in orange. Kernels are "
            "zero-phase and never touch a gap. Labels and changepoint snapping keep using the raw data. "
            "Nothing is saved; the panel returns to raw when this closes."
        )
        footer.setWordWrap(True)
        footer.setStyleSheet("color: gray;")
        params_col.addWidget(footer)
        params_col.addStretch()

        code_col = QVBoxLayout()
        lower.addLayout(code_col, 1)
        self.code_edit = QPlainTextEdit()
        self.code_edit.setReadOnly(True)
        self.code_edit.setFont(monospace_font())
        self.code_edit.setMinimumWidth(560)
        code_col.addWidget(self.code_edit, 1)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.copy_btn = QPushButton("Copy code")
        self.copy_btn.setAutoDefault(False)
        self.copy_btn.clicked.connect(self._copy)
        buttons.addWidget(self.copy_btn)
        self.console_btn = QPushButton("Send to console")
        self.console_btn.setAutoDefault(False)
        self.console_btn.clicked.connect(self._send_to_console)
        buttons.addWidget(self.console_btn)
        close_btn = QPushButton("Close")
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        code_col.addLayout(buttons)
        self.resize(1250, 640)

        self._rebuild_lists(select=self._pipeline.steps[-1] if self._pipeline.steps else None)

        active = getattr(plot_container, "active_panels", None)
        if active is not None:
            active.active_changed.connect(self._on_active_changed)
        self._retarget(self._initial_plot())

    # ------------------------------------------------------------------ target panel

    def _initial_plot(self):
        active = getattr(self._container, "active_panels", None)
        reg = active.active if active is not None else None
        if reg is not None and reg.kind == PanelKind.LINEPLOT and reg.plot is not None:
            return reg.plot
        plots = list(getattr(self._container, "line_plots", []))
        return plots[0] if plots else None

    def _on_active_changed(self, reg) -> None:
        if reg.kind == PanelKind.LINEPLOT and reg.plot is not None and reg.plot is not self._plot:
            self._retarget(reg.plot)

    def _retarget(self, plot) -> None:
        if self._plot is not None:
            self._plot.set_preview(None)
        self._plot = plot
        self._refresh()

    @property
    def plot(self):
        return self._plot

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def selected_step(self) -> Step | None:
        return self._selected

    # ------------------------------------------------------------------ steps

    def _rebuild_lists(self, select: Step | None) -> None:
        self._updating = True
        try:
            for stage, box in self.boxes.items():
                box.list.clear()
                for i, step in enumerate(self._pipeline.stage(stage)):
                    item = QListWidgetItem(step.label())
                    item.setData(Qt.ItemDataRole.UserRole, i)
                    box.list.addItem(item)
                    if step is select:
                        box.list.setCurrentRow(box.list.count() - 1)
                if select is None or select.spec.stage != stage:
                    box.list.setCurrentRow(-1)
        finally:
            self._updating = False
        self._select(select)

    def _select(self, step: Step | None) -> None:
        self._selected = step
        self.step_title.setText(step.spec.label if step else "Select a step to edit its parameters.")
        self._rebuild_param_form()
        self.docs_btn.setEnabled(step is not None)
        self.docs_btn.setToolTip(step.spec.doc_url if step else "")
        self.source_btn.setEnabled(step is not None and step.spec.source is not None)

    def _on_selection_changed(self, box: _StageBox) -> None:
        if self._updating:
            return
        row = box.list.currentRow()
        steps = self._pipeline.stage(box.stage)
        if not 0 <= row < len(steps):
            return
        self._updating = True
        try:
            for other in self.boxes.values():
                if other is not box:
                    other.list.setCurrentRow(-1)
        finally:
            self._updating = False
        self._select(steps[row])

    def _on_rows_moved(self, box: _StageBox) -> None:
        reordered = box.steps_in_list_order(self._pipeline.stage(box.stage))
        self._pipeline = self._pipeline.with_stage(box.stage, reordered)
        self._rebuild_lists(select=self._selected)
        self._refresh()

    def _add_step(self, kind: str) -> None:
        step = Step.default(kind)
        stage = step.spec.stage
        self._pipeline = self._pipeline.with_stage(stage, (*self._pipeline.stage(stage), step))
        self._rebuild_lists(select=step)
        self._refresh()

    def _remove_step(self, box: _StageBox) -> None:
        """Remove the box's selected step, or its last one when none is selected."""
        steps = self._pipeline.stage(box.stage)
        if not steps:
            return
        row = box.list.currentRow()
        if not 0 <= row < len(steps):
            row = len(steps) - 1
        remaining = steps[:row] + steps[row + 1 :]
        self._pipeline = self._pipeline.with_stage(box.stage, remaining)
        self._rebuild_lists(select=remaining[min(row, len(remaining) - 1)] if remaining else None)
        self._refresh()

    # ------------------------------------------------------------------ the selected step's parameters

    def _rebuild_param_form(self) -> None:
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        step = self._selected
        if step is None:
            return
        for spec in step.spec.params:
            if isinstance(spec, ParamSpec):
                label = f"{spec.name} [{spec.unit}]" if spec.unit else spec.name
                self.param_form.addRow(label, self._slider_row(spec, step))
            elif isinstance(spec, IntSpec):
                self.param_form.addRow(spec.name, self._int_spin(spec, step))
            else:
                self.param_form.addRow(spec.name, self._choice_combo(spec, step))

    def _slider_row(self, spec: ParamSpec, step: Step) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, _SLIDER_STEPS)
        slider.setMinimumWidth(200)
        spin = QDoubleSpinBox()
        spin.setKeyboardTracking(False)
        spin.setDecimals(spec.decimals)
        spin.setRange(spec.low, spec.high * 10 if spec.log else spec.high)
        spin.setSingleStep((spec.high - spec.low) / 100)
        value = float(step.params[spec.name])
        spin.setValue(value)
        slider.setValue(_to_slider(spec, value))

        def from_slider(pos: int) -> None:
            if self._updating:
                return
            v = round(_from_slider(spec, pos), spec.decimals)
            self._updating = True
            spin.setValue(v)
            self._updating = False
            self._set_param(spec.name, v)

        def from_spin(v: float) -> None:
            if self._updating:
                return
            self._updating = True
            slider.setValue(_to_slider(spec, v))
            self._updating = False
            self._set_param(spec.name, float(v))

        slider.valueChanged.connect(from_slider)
        spin.valueChanged.connect(from_spin)
        layout.addWidget(slider, 1)
        layout.addWidget(spin)
        return row

    def _int_spin(self, spec: IntSpec, step: Step) -> QSpinBox:
        spin = QSpinBox()
        spin.setKeyboardTracking(False)
        spin.setRange(spec.low, spec.high)
        spin.setValue(int(step.params[spec.name]))
        spin.valueChanged.connect(lambda v: None if self._updating else self._set_param(spec.name, int(v)))
        return spin

    def _choice_combo(self, spec: ChoiceSpec, step: Step) -> QComboBox:
        combo = QComboBox()
        combo.addItems(list(spec.choices))
        combo.setCurrentText(str(step.params[spec.name]))
        combo.currentTextChanged.connect(lambda s: None if self._updating else self._set_param(spec.name, s))
        return combo

    def _set_param(self, name: str, value: Any) -> None:
        old = self._selected
        if old is None:
            return
        new = old.with_param(name, value)
        self._pipeline = self._pipeline.replace(old, new)
        self._selected = new
        box = self.boxes[new.spec.stage]
        item = box.list.currentItem()
        if item is not None:
            item.setText(new.label())
        self._refresh()

    # ------------------------------------------------------------------ preview + code

    def _backend(self) -> str:
        loader = getattr(self._app_state, "data_loader", None)
        return "pynapple" if isinstance(loader, PynappleLoader) else "xarray"

    def _feature_and_selections(self) -> tuple[str | None, dict[str, str]]:
        if self._plot is None:
            return None, {}
        feature = self._plot._effective_feature()
        selections = {str(k): str(v) for k, v in self._plot._effective_selections().items()}
        return (str(feature) if feature else None), selections

    def _refresh(self) -> None:
        self._remember()
        feature, selections = self._feature_and_selections()
        if feature is None:
            self.target_label.setText("No line plot to preview on. Add one, or click one.")
            self.code_edit.setPlainText("")
            self.status_label.setText("")
            self.copy_btn.setEnabled(False)
            self.console_btn.setEnabled(False)
            return
        pinned = ", ".join(f"{k}={v}" for k, v in sorted(selections.items()))
        self.target_label.setText(f"Previewing on: {feature}" + (f"  ({pinned})" if pinned else ""))
        self.code_edit.setPlainText(smoothing_snippet(self._pipeline, feature, selections, self._backend()))
        self.copy_btn.setEnabled(bool(self._pipeline.steps))
        console_ok = bool(self._pipeline.steps) and not self._pipeline.has_movement_step()
        self.console_btn.setEnabled(console_ok)
        self.console_btn.setToolTip(
            "Put the same chain on the console's bound panel variable"
            if console_ok
            else "movement's functions take a DataArray; the console holds numpy"
        )
        self.status_label.setText("")
        if self._pipeline.steps:
            self._plot.set_preview(LinePreview(apply=self._apply, pad_s=self._pipeline.pad_s()))
        else:
            self._plot.set_preview(None)

    def _companions(self, time: np.ndarray, selections: dict[str, str]) -> Companions:
        """The aligned companion variables the pipeline's steps ask for."""
        if "confidence" not in self._pipeline.needs():
            return Companions()
        loader = getattr(self._app_state, "data_loader", None)
        plot_data = loader.select("confidence", selections, t0=float(time[0]), t1=float(time[-1])) if loader else None
        if plot_data is None or len(plot_data.time) != len(time):
            raise ValueError(
                "no 'confidence' variable on this feature's clock; filter_by_confidence has nothing to read"
            )
        return Companions(confidence=np.asarray(plot_data.data, dtype=float))

    def _apply(self, data: np.ndarray, time: np.ndarray) -> list[tuple[str, np.ndarray]] | None:
        """Run the pipeline on what the panel plots; a refusal becomes the status line."""
        _feature, selections = self._feature_and_selections()
        try:
            fs = sampling_rate(time, name="this panel's time axis")
            stages = self._pipeline.run_stages(data, fs, self._companions(time, selections))
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return None
        self.status_label.setText(f"{fs:.4g} Hz on this panel; {len(stages)} step(s) drawn.")
        return stages

    def _remember(self) -> None:
        cache = dict(getattr(self._app_state, "function_params_cache", None) or {})
        cache[CACHE_KEY] = self._pipeline.to_cache()
        self._app_state.function_params_cache = cache

    # ------------------------------------------------------------------ outputs

    def _open_docs(self) -> None:
        if self._selected is not None:
            webbrowser.open(self._selected.spec.doc_url)

    def _open_source(self) -> None:
        from ethograph.gui.dialog_function_params import _do_open_source

        if self._selected is not None and self._selected.spec.source is not None:
            _do_open_source(self._selected.spec.source(), self)

    def _copy(self) -> None:
        clipboard = QApplication.clipboard()
        assert clipboard is not None
        clipboard.setText(self.code_edit.toPlainText())
        notify("Copied the preprocessing script to the clipboard.")

    def _send_to_console(self) -> None:
        from ethograph.gui.plots_console import to_identifier

        feature, _ = self._feature_and_selections()
        line = console_line(self._pipeline, to_identifier(feature)) if feature else None
        if line is None or self._plot is None:
            return
        console = self._container.add_console_panel()
        console.bind_panel(self._plot)
        console.input.setPlainText(line)
        console.input.setFocus()

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._plot is not None:
            try:
                self._plot.set_preview(None)
            except RuntimeError:  # the panel was closed meanwhile
                pass
        active = getattr(self._container, "active_panels", None)
        if active is not None:
            try:
                active.active_changed.disconnect(self._on_active_changed)
            except (RuntimeError, TypeError):
                pass
        super().closeEvent(event)


def _to_slider(spec: ParamSpec, value: float) -> int:
    if spec.log:
        frac = (np.log(value) - np.log(spec.low)) / (np.log(spec.high) - np.log(spec.low))
    else:
        frac = (value - spec.low) / (spec.high - spec.low)
    return int(round(float(np.clip(frac, 0.0, 1.0)) * _SLIDER_STEPS))


def _from_slider(spec: ParamSpec, pos: int) -> float:
    frac = pos / _SLIDER_STEPS
    if spec.log:
        return float(np.exp(np.log(spec.low) + frac * (np.log(spec.high) - np.log(spec.low))))
    return spec.low + frac * (spec.high - spec.low)
