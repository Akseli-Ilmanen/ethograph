"""Curation workflows: a named, replayable sequence of curation steps.

Curating a session is the same handful of moves over and over — narrow the
trials table to one condition, run the onset model over them, drop the
predicted classes into the curation scope, open a grid laid out the way that
behaviour needs, walk the boundaries, save. A *workflow* is that sequence
written down once and replayed with one click.

The model here is deliberately dumb: a workflow is a name and an ordered list
of :class:`WorkflowStep`, each a ``kind`` plus a flat ``params`` mapping.
Nothing in this module touches Qt or the GUI — :mod:`ethograph.gui.dialog_curation_workflow`
owns the handlers that actually drive the widgets, and :data:`STEP_KINDS` is
the contract between the two: it declares which parameters a kind takes,
their types and defaults, so the editor builds its form and the runner reads
its arguments from one place.

Workflows live in ``~/.ethograph/defaults/workflows/{name}.yaml`` — the same global
store as the onset models they usually invoke, so a workflow written on one
dataset is there for the next.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ethograph.utils.paths import defaults_dir

logger = logging.getLogger(__name__)

#: Parameter value kinds the editor knows how to render and the YAML can hold.
#: ``confidence`` is a probability in [0, 1] typed in full rather than
#: stepped — a model's scores live at the bottom of the range, where 0.0002
#: and 0.001 are different answers and a spin box's fixed decimals are not.
PARAM_TYPES = (
    "bool",
    "int",
    "float",
    "confidence",
    "text",
    "choice",
    "labels",
    "filters",
    "panels",
    "cameras",
)

#: What a confidence parameter opens at: low enough to keep or flag almost
#: nothing, high enough to be a starting point that can be typed upwards.
DEFAULT_CONFIDENCE = 0.001


@dataclass(frozen=True)
class ParamSpec:
    """One parameter of a step kind: how it is stored, shown and defaulted."""

    key: str
    label: str
    type: str
    default: Any
    tooltip: str = ""
    #: For ``type="choice"``: mapping of stored value to the text shown.
    choices: dict[str, str] = field(default_factory=dict)
    #: For ``type="int"``/``"float"``: inclusive bounds for the spin box.
    minimum: float = 0.0
    maximum: float = 1e6

    def __post_init__(self) -> None:
        if self.type not in PARAM_TYPES:
            raise ValueError(f"unknown parameter type {self.type!r} for {self.key!r}")


@dataclass(frozen=True)
class StepKind:
    """One kind of step: its title, what it does, and the parameters it takes.

    *interactive* marks a step the user finishes by hand (a grid to work
    through, a review to walk): the runner hands over and waits for it rather
    than moving straight on to the next step.
    """

    key: str
    title: str
    summary: str
    params: tuple[ParamSpec, ...] = ()
    interactive: bool = False

    def defaults(self) -> dict[str, Any]:
        return {spec.key: copy_default(spec.default) for spec in self.params}

    def spec(self, key: str) -> ParamSpec | None:
        return next((s for s in self.params if s.key == key), None)


def copy_default(value: Any) -> Any:
    """A default that callers may mutate without editing the spec."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


#: Curation modes, mirroring ``gui.widgets_curation.CURATION_MODES`` — spelled
#: here so this module stays importable without Qt.
CURATION_MODE_CHOICES = {
    "manual": "Manual (trial level)",
    "inspect": "Inspect is enough (trial level)",
    "frame": "Frame-by-frame review",
}

#: What a *single* tile click means, mirroring
#: ``gui.dialog_label_gridview.GRID_MODES``. A double click always navigates,
#: in every mode, so there is no mode for it — a workflow written before that
#: (``grid_mode: navigate``) names no choice, and the grid keeps its default.
GRID_MODE_CHOICES = {
    "curate": "Click = curated",
    "uncurate": "Click = uncurated, rest = curated",
}


#: Which trials a bulk label operation (curate / delete / purge) acts on.
#: Canonical here, not mirrored from the GUI — it names no Qt object, so
#: ``gui.widgets_curation`` and ``gui.dialog_bulk_labels`` import it directly
#: instead of keeping their own copy. "single" is the current trial; "all"
#: is every trial regardless of the table's filters; "filtered" is what the
#: table currently shows (after any earlier ``filter_trials`` step); "hidden"
#: is everything its filters currently exclude.
TRIAL_SCOPE_SINGLE = "single"
TRIAL_SCOPE_ALL = "all"
TRIAL_SCOPE_FILTERED = "filtered"
TRIAL_SCOPE_HIDDEN = "hidden"
TRIAL_SCOPE_CHOICES = {
    TRIAL_SCOPE_SINGLE: "Current trial",
    TRIAL_SCOPE_ALL: "All trials",
    TRIAL_SCOPE_FILTERED: "Filtered trials (shown by the table)",
    TRIAL_SCOPE_HIDDEN: "Hidden trials (filtered out)",
}

#: The grids' "Labeling method" filter, mirroring
#: ``gui.dialog_label_gridview.GRID_METHOD_FILTERS`` — spelled here so this
#: module stays importable without Qt.
GRID_METHOD_CHOICES = {
    "automated": "Automated only",
    "manual": "Manual only",
    "curated": "Curated only",
    "human": "Manual or curated",
    "all": "All labels",
}

#: The ``methods`` parameter both grid steps take: which half of the labels
#: the grid is about. Reviewing a model's output means its own labels and
#: nothing else.
_METHODS_PARAM = ParamSpec(
    "methods",
    "Labeling method",
    "choice",
    "all",
    "Which labels of the scope's classes the grid shows.",
    choices=GRID_METHOD_CHOICES,
)

#: The ``which`` parameter every bulk label step takes, spelled once.
_TRIALS_PARAM = ParamSpec(
    "which",
    "Trials",
    "choice",
    TRIAL_SCOPE_FILTERED,
    "Which trials this step runs over.",
    choices=TRIAL_SCOPE_CHOICES,
)

#: The ``label_ids`` parameter curate/delete/purge share: an explicit class
#: list, like the bulk-editing dialog's own checklist. Empty means "don't
#: override — use whatever the curation scope area holds" (the drag-and-drop
#: scope, or an earlier ``scope`` step), exactly as leaving the dialog's own
#: checklist on *All* would; it is never "every class" for its own sake.
_LABEL_IDS_PARAM = ParamSpec(
    "label_ids",
    "Label classes",
    "labels",
    [],
    "Empty = whatever the curation scope currently holds (the scope area, or an earlier 'Set curation "
    "scope' step) — the same fallback the scope step itself uses.",
)

#: The ``cameras`` parameter both grid steps take, spelled once: each grid
#: shows one tile per (label, camera). An empty list means "leave the cameras
#: as the grid would tick them itself" — every camera, or the last selection
#: the reviewer made (``app_state.grid_selected_cameras``).
_CAMERAS_PARAM = ParamSpec(
    "cameras",
    "Cameras",
    "cameras",
    [],
    "Which cameras each label is shown from. Empty = whatever the grid ticks by itself.\n"
    "A camera this dataset does not have is skipped.",
)


STEP_KINDS: dict[str, StepKind] = {
    kind.key: kind
    for kind in (
        StepKind(
            key="filter_trials",
            title="Filter trials",
            summary="Set the trials table's column filters — the one trial filter every later step runs over.",
            params=(
                ParamSpec(
                    "filters",
                    "Column filters",
                    "filters",
                    [],
                    "One entry per metadata column: a set of allowed values, or a numeric comparison.",
                ),
                ParamSpec(
                    "clear_first",
                    "Clear existing filters first",
                    "bool",
                    True,
                    "Start from every trial rather than adding to whatever is filtered now.",
                ),
            ),
        ),
        StepKind(
            key="predict",
            title="Predict onsets",
            summary="Run a trained LightGBM onset model over the visible trials, filling classes they lack.",
            params=(
                ParamSpec("model", "Model", "choice", "", "A trained model from ~/.ethograph/defaults/runs/lightgbm."),
                ParamSpec(
                    "individual",
                    "Individual",
                    "text",
                    "",
                    "Whose data the model reads and who the predicted events belong to — a model "
                    "trained on another animal is re-pointed at this one. Empty = whichever "
                    "individual the GUI has selected.",
                ),
                ParamSpec(
                    "min_confidence",
                    "Min confidence",
                    "confidence",
                    # The Predict dialog's own default: low enough to write
                    # essentially everything and triage afterwards on the
                    # confidence, high enough to drop a score of nothing.
                    DEFAULT_CONFIDENCE,
                    "Predictions scoring below this are not written at all.\n"
                    "Typed in full, to as many decimals as the scores need (0.0002).",
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
        ),
        StepKind(
            key="scope",
            title="Set curation scope",
            summary="Drop label classes into the Curation section's scope area and pick the curation mode.",
            params=(
                ParamSpec(
                    "label_ids",
                    "Label classes",
                    "labels",
                    [],
                    "Empty = whatever the workflow's last prediction step wrote (every class if none did).",
                ),
                ParamSpec(
                    "mode",
                    "Curation mode",
                    "choice",
                    "manual",
                    "How a label gets curated from here on.",
                    choices=CURATION_MODE_CHOICES,
                ),
            ),
        ),
        StepKind(
            key="label_grid",
            title="Label grid view",
            summary="Open the frame grid on the scope, laid out and generated as configured.",
            interactive=True,
            params=(
                _CAMERAS_PARAM,
                _METHODS_PARAM,
                ParamSpec("columns", "Columns", "int", 3, "Tiles per row.", minimum=1, maximum=12),
                ParamSpec(
                    "window_s",
                    "Panel time window",
                    "float",
                    1.0,
                    "Seconds of each ticked GUI panel shown under a frame.",
                    minimum=0.01,
                    maximum=600.0,
                ),
                ParamSpec(
                    "panels",
                    "GUI panels",
                    "panels",
                    [],
                    "Panels screenshotted under each frame. A closed panel that named a feature\n"
                    "(and, for keypoint/individual data, which selection) is reopened with that\n"
                    "restored before the grid runs; one with no feature on record is skipped.",
                ),
                ParamSpec(
                    "axis_auto",
                    "Autoscale y per window",
                    "bool",
                    True,
                    "Fit each capture's y-range to its own time window.",
                ),
                ParamSpec(
                    "skip_video",
                    "Skip video loading during capture",
                    "bool",
                    True,
                    "Trial switches during capture skip the video decoder — much faster.",
                ),
                ParamSpec(
                    "threshold",
                    "Flag confidence below",
                    "confidence",
                    DEFAULT_CONFIDENCE,
                    "Tiles scoring below this are outlined red.\n"
                    "Typed in full, to as many decimals as the scores need (0.0002); 0 flags nothing.",
                    minimum=0.0,
                    maximum=1.0,
                ),
                ParamSpec("grid_mode", "Single click means", "choice", "curate", choices=GRID_MODE_CHOICES),
                ParamSpec(
                    "mark_flagged",
                    "Pre-click the low-confidence tiles",
                    "bool",
                    False,
                    "Only meaningful in 'Click = uncurated, rest = curated'.",
                ),
                ParamSpec(
                    "generate",
                    "Generate straight away",
                    "bool",
                    True,
                    "Untick to open on the Setup tab so the layout can be tweaked first.",
                ),
            ),
        ),
        StepKind(
            key="video_grid",
            title="Video grid",
            summary="Open the clip player on the scope, one label class at a time.",
            interactive=True,
            params=(
                _CAMERAS_PARAM,
                _METHODS_PARAM,
                ParamSpec(
                    "point_window_s",
                    "Window around point events",
                    "float",
                    0.5,
                    "A point event's clip spans this much either side of the instant.",
                    minimum=0.05,
                    maximum=30.0,
                ),
                ParamSpec("per_page", "Clips on screen", "int", 6, minimum=1, maximum=24),
                ParamSpec("columns", "Columns", "int", 3, minimum=1, maximum=8),
                ParamSpec(
                    "speed_pct",
                    "Playback speed",
                    "float",
                    100.0,
                    "Per cent of the recording's own speed — the grid's own, not the GUI's.",
                    minimum=5.0,
                    maximum=400.0,
                ),
                ParamSpec(
                    "threshold",
                    "Flag confidence below",
                    "confidence",
                    DEFAULT_CONFIDENCE,
                    "Tiles scoring below this are outlined red.\n"
                    "Typed in full, to as many decimals as the scores need (0.0002); 0 flags nothing.",
                    minimum=0.0,
                    maximum=1.0,
                ),
                ParamSpec("grid_mode", "Single click means", "choice", "curate", choices=GRID_MODE_CHOICES),
                ParamSpec(
                    "generate",
                    "Generate straight away",
                    "bool",
                    True,
                    "Untick to open on the Setup tab so the layout can be tweaked first.",
                ),
            ),
        ),
        StepKind(
            key="frame_review",
            title="Frame-by-frame review",
            summary="Start the boundary review over the scope and wait for it to finish.",
            interactive=True,
            params=(
                ParamSpec(
                    "window_s",
                    "View window",
                    "float",
                    0.5,
                    "Seconds of time series shown around the boundary being reviewed.",
                    minimum=0.02,
                    maximum=600.0,
                ),
                ParamSpec(
                    "automated_only",
                    "Automated only",
                    "bool",
                    True,
                    "Leave manual and curated boundaries out of the queue.",
                ),
                ParamSpec(
                    "next_curates",
                    "N (next) marks curated",
                    "bool",
                    True,
                    "Moving on with N means the boundary was seen and is fine.",
                ),
            ),
        ),
        StepKind(
            key="curate_trials",
            title="Curate trials' labels",
            summary="Mark every automated label of the chosen classes, in the chosen trials, as curated.",
            params=(_TRIALS_PARAM, _LABEL_IDS_PARAM),
        ),
        StepKind(
            key="delete_labels",
            title="Delete trials' labels",
            summary="Delete every label of the chosen classes, in the chosen trials — every "
            "labeling_method, not just automated.",
            params=(_TRIALS_PARAM, _LABEL_IDS_PARAM),
        ),
        StepKind(
            key="purge_labels",
            title="Purge short labels",
            summary="Delete state-interval labels of the chosen classes shorter than a threshold, in the "
            "chosen trials. Point events are never touched.",
            params=(
                _TRIALS_PARAM,
                _LABEL_IDS_PARAM,
                ParamSpec(
                    "min_duration_s",
                    "Shorter than",
                    "float",
                    0.010,
                    "Labels below this duration (seconds) are dropped.",
                    minimum=0.0,
                    maximum=600.0,
                ),
            ),
        ),
        StepKind(
            key="correct_offsets",
            title="Correct offsets",
            summary="Pull back each label's offset across a near-zero gap to the next onset of the same "
            "subject, in the chosen trials — makes every interval strictly separated so pynapple can "
            "resolve them. Not scoped by label class: a subject's whole sequence has to be seen together.",
            params=(_TRIALS_PARAM,),
        ),
        StepKind(
            key="save_labels",
            title="Save labels",
            summary="Write the labels TSV, exactly as Ctrl+S does.",
        ),
    )
}


@dataclass
class WorkflowStep:
    """One step: a kind from :data:`STEP_KINDS` plus its parameters."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def spec(self) -> StepKind:
        try:
            return STEP_KINDS[self.kind]
        except KeyError:
            raise KeyError(f"unknown workflow step kind {self.kind!r}") from None

    def value(self, key: str) -> Any:
        """The parameter's value, falling back to the kind's declared default."""
        if key in self.params:
            return self.params[key]
        spec = self.spec().spec(key)
        if spec is None:
            raise KeyError(f"step {self.kind!r} has no parameter {key!r}")
        return copy_default(spec.default)

    def title(self) -> str:
        return self.spec().title


@dataclass
class CurationWorkflow:
    """A named, ordered list of steps."""

    name: str
    description: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CurationWorkflow:
        steps = [
            WorkflowStep(kind=str(s["kind"]), params=dict(s.get("params") or {})) for s in (raw.get("steps") or [])
        ]
        return cls(name=str(raw["name"]), description=str(raw.get("description") or ""), steps=steps)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(workflow: CurationWorkflow) -> list[str]:
    """Everything wrong with *workflow*, as sentences; empty means runnable."""
    problems: list[str] = []
    if not workflow.name.strip():
        problems.append("The workflow has no name.")
    if not workflow.steps:
        problems.append("The workflow has no steps.")
    for i, step in enumerate(workflow.steps, start=1):
        kind = STEP_KINDS.get(step.kind)
        if kind is None:
            problems.append(f"Step {i}: unknown step kind {step.kind!r}.")
            continue
        known = {spec.key for spec in kind.params}
        problems.extend(
            f"Step {i} ({kind.title}): unknown parameter {key!r}." for key in step.params if key not in known
        )
        if step.kind == "predict" and not str(step.value("model")).strip():
            problems.append(f"Step {i} ({kind.title}): no model chosen.")
        if step.kind == "filter_trials":
            problems.extend(f"Step {i} ({kind.title}): {msg}" for msg in filter_problems(step.value("filters")))
    return problems


def filter_problems(filters: Any) -> list[str]:
    """What is wrong with a ``filter_trials`` step's column filters."""
    if not isinstance(filters, list):
        return ["the column filters must be a list."]
    out: list[str] = []
    for entry in filters:
        if not isinstance(entry, dict) or not str(entry.get("column", "")).strip():
            out.append("a column filter names no column.")
            continue
        column = entry["column"]
        has_values = isinstance(entry.get("values"), list) and bool(entry["values"])
        has_numeric = entry.get("op") in (">=", "<=") and entry.get("value") is not None
        if not (has_values or has_numeric):
            out.append(f"the filter on {column!r} states neither allowed values nor a comparison.")
    return out


def describe_filter(entry: dict[str, Any]) -> str:
    """One column filter as the phrase the step list shows."""
    column = str(entry.get("column", "?"))
    values = entry.get("values")
    if isinstance(values, list) and values:
        return f"{column} in {{{', '.join(str(v) for v in values)}}}"
    if entry.get("op") in (">=", "<=") and entry.get("value") is not None:
        return f"{column} {entry['op']} {entry['value']}"
    return f"{column} (no condition)"


# ---------------------------------------------------------------------------
# Storage: ~/.ethograph/defaults/workflows/{name}.yaml
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"[^A-Za-z0-9 _.-]+")


def workflows_root() -> Path:
    """The workflow store used while no project is open, ``~/.ethograph/defaults/workflows``."""
    return defaults_dir("workflows")


def safe_name(name: str) -> str:
    """*name* as a file stem: the characters a workflow name may use."""
    cleaned = _NAME_RE.sub("_", (name or "").strip()).strip()
    if not any(ch.isalnum() for ch in cleaned):
        raise ValueError("a workflow name must contain at least one letter or digit")
    return cleaned


def workflow_path(name: str) -> Path:
    return workflows_root() / f"{safe_name(name)}.yaml"


def list_workflows() -> list[str]:
    """Names of every stored workflow, alphabetically."""
    root = workflows_root()
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.yaml"))


def load_workflow(name: str) -> CurationWorkflow:
    path = workflow_path(name)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw.setdefault("name", path.stem)
    return CurationWorkflow.from_dict(raw)


def save_workflow(workflow: CurationWorkflow) -> Path:
    path = workflow_path(workflow.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(workflow.to_dict(), sort_keys=False), encoding="utf-8")
    return path


def delete_workflow(name: str) -> None:
    workflow_path(name).unlink(missing_ok=True)


def rename_workflow(old: str, new: str) -> Path:
    """Store *old* under *new* and drop the old file."""
    workflow = load_workflow(old)
    workflow.name = safe_name(new)
    path = save_workflow(workflow)
    if safe_name(old) != workflow.name:
        delete_workflow(old)
    return path
