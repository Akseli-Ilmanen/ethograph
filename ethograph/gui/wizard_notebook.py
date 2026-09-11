"""Generate the Data wizard's per-rig setup notebook (Qt-free).

One `.ipynb` per rig: the first cell (tagged ``parameters``, the papermill
convention) is the only thing a user edits next session; boring work is a
library call (`eto.discover_media`, `eto.pair_media`); only neuroconv tuning
is left explicit for the two modes that carry a recording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jinja2
import nbformat

_TEMPLATE_DIR = Path(__file__).parent / "templates" / "notebook"

Mode = Literal["pair", "free_running", "triggered"]
Timing = Literal["none", "offset", "pulses", "onsets"]


@dataclass
class RigSource:
    stream: str  # video | pose | audio
    folder: str  # relative to session_dir
    pattern: str | None = None  # regex over stem, named groups trial/camera/mic; None = natsorted
    n_devices: int = 1
    rate: float | None = None  # audio rate; video/pose read from header
    software: str | None = None  # pose
    session_wide: bool = False  # one file for the whole session; `folder` is then that file
    offset_s: float | None = None  # session_wide: when the file starts, relative to trial 1


@dataclass
class RigSpec:
    rig_name: str
    session_dir: str
    mode: Mode
    sources: list[RigSource] = field(default_factory=list)
    timing: Timing = "none"
    split_files: bool = False  # free_running: recorder rotated files (storage detail)
    offset_s: float | None = None  # timing == offset
    recording_file: str | None = None  # relative to session_dir, e.g. "ephys/session.rhd"; modes 2/3
    recording_interface: str = "IntanRecordingInterface"  # neuroconv class name; digital twin derived
    frame_line: str | None = None  # digital line with a pulse per frame (timing == pulses)
    trigger_line: str | None = None  # digital line with a pulse per trial (triggered)
    trial_table_file: str | None = None  # optional metadata TSV/CSV with a `trial` column
    burst_gap: float = 10.0  # x median frame interval, triggered + pulses


_DEVICE_PREFIX = {"video": "cam", "pose": "cam", "audio": "mic"}


# ---------------------------------------------------------------------------
# Literal formatting: the `rig` dict a user sees in the parameters cell
# ---------------------------------------------------------------------------


def _raw_pattern(pattern: str) -> str:
    """Render a regex as a raw string literal, so `re.compile` sees it unescaped."""
    if '"' in pattern:
        return repr(pattern)
    return f'r"{pattern}"'


def _format_source_literal(src: RigSource) -> str:
    parts = [f'"stream": {src.stream!r}', f'"folder": {src.folder!r}']
    if src.pattern:
        parts.append(f'"pattern": {_raw_pattern(src.pattern)}')
    if src.n_devices != 1:
        parts.append(f'"n_devices": {src.n_devices}')
    if src.rate is not None:
        parts.append(f'"rate": {src.rate!r}')
    if src.software is not None:
        parts.append(f'"software": {src.software!r}')
    if src.session_wide:
        parts.append('"session_wide": True')
        parts.append(f'"offset_s": {src.offset_s!r}')
    return "{" + ", ".join(parts) + "}"


def _format_rig_literal(spec: RigSpec) -> str:
    lines = ["{"]
    lines.append(f'    "rig_name": {spec.rig_name!r},')
    lines.append(f'    "mode": {spec.mode!r},')
    lines.append(f'    "timing": {spec.timing!r},')
    lines.append('    "sources": [')
    for src in spec.sources:
        lines.append(f"        {_format_source_literal(src)},")
    lines.append("    ],")
    if spec.mode != "pair":
        lines.append(f'    "recording_file": {spec.recording_file!r},')
        lines.append(f'    "recording_interface": {spec.recording_interface!r},')
    if spec.frame_line is not None:
        lines.append(f'    "frame_line": {spec.frame_line!r},')
    if spec.trigger_line is not None:
        lines.append(f'    "trigger_line": {spec.trigger_line!r},')
    if spec.frame_line is not None or spec.trigger_line is not None:
        lines.append(f'    "burst_gap": {spec.burst_gap!r},')
    if spec.timing == "offset":
        lines.append(f'    "offset_s": {spec.offset_s!r},')
    if spec.trial_table_file is not None:
        lines.append(f'    "trial_table_file": {spec.trial_table_file!r},')
    lines.append("}")
    return "\n".join(lines)


def _stream_rates_literal(spec: RigSpec) -> str:
    rates = {src.stream: src.rate for src in spec.sources if src.rate is not None and not src.session_wide}
    if not rates:
        return "{}"
    return "{" + ", ".join(f'"{k}": {v!r}' for k, v in rates.items()) + "}"


def _camera_columns(video: RigSource) -> list[str]:
    prefix = _DEVICE_PREFIX[video.stream]
    return [f"{video.stream}_{prefix}-{i + 1}" for i in range(video.n_devices)]


def _video_source(spec: RigSpec) -> RigSource:
    for src in spec.sources:
        if src.stream == "video":
            return src
    raise ValueError(f"rig {spec.rig_name!r} has timing {spec.timing!r} but no video source")


def _session_wide_entries(spec: RigSpec) -> list[tuple[str, str, str, str]]:
    """Sources recorded once for the whole session: (column, file, rate, offset) as notebook text."""
    entries: list[tuple[str, str, str, str]] = []
    for src in spec.sources:
        if not src.session_wide:
            continue
        col = f"{src.stream}_{_DEVICE_PREFIX[src.stream]}-1"
        rate_text = repr(src.rate) if src.rate is not None else "None  # TODO: set the sample rate"
        offset_text = repr(src.offset_s) if src.offset_s is not None else 'rig["offset_s"]'
        entries.append((col, src.folder, rate_text, offset_text))
    return entries


def _digital_interface_name(recording_interface: str) -> str:
    """The neuroconv digital-line twin of a recording interface: Intan -> IntanDigitalInterface."""
    vendor = recording_interface.removesuffix("RecordingInterface")
    return f"{vendor}DigitalInterface"


# ---------------------------------------------------------------------------
# Jinja2 environment
# ---------------------------------------------------------------------------


def _create_jinja_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
        undefined=jinja2.StrictUndefined,
    )


_env: jinja2.Environment | None = None


def _get_env() -> jinja2.Environment:
    global _env
    if _env is None:
        _env = _create_jinja_env()
    return _env


def _render(name: str, ctx: dict) -> str:
    rendered = _get_env().get_template(name).render(ctx)
    return _clean_blank_lines(rendered)


def _clean_blank_lines(text: str) -> str:
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n") + "\n"


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------


def _md_cell(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text)


def _code_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)


def _parameters_cell(spec: RigSpec) -> nbformat.NotebookNode:
    ctx = {"session_dir": spec.session_dir, "rig_literal": _format_rig_literal(spec)}
    cell = _code_cell(_render("parameters.py.j2", ctx))
    cell.metadata["tags"] = ["parameters"]
    return cell


def _discover_cell() -> nbformat.NotebookNode:
    return _code_cell(_render("discover.py.j2", {}))


def _timing_cells(spec: RigSpec) -> tuple[list[nbformat.NotebookNode], bool]:
    """Return the timing cells (if any) and whether a `session_wide` dict was built."""
    if spec.timing == "none":
        return [], False

    if spec.timing == "offset":
        video = _video_source(spec)
        entries = _session_wide_entries(spec)
        ctx = {
            "video_folder": video.folder,
            "camera_columns": repr(_camera_columns(video)),
            "session_wide_entries": entries,
        }
        return [_md_cell("## Timing"), _code_cell(_render("timing_offset.py.j2", ctx))], True

    if spec.timing == "pulses":
        video = _video_source(spec)
        ctx = {
            "digital_interface": _digital_interface_name(spec.recording_interface),
            "video_folder": video.folder,
            "camera_columns": repr(_camera_columns(video)),
        }
        if spec.mode == "free_running":
            ctx["split_files"] = spec.split_files
            template = "timing_pulses_free.py.j2"
        else:
            template = "timing_pulses_triggered.py.j2"
        return [_md_cell("## Timing"), _code_cell(_render(template, ctx))], False

    # spec.timing == "onsets"
    video = _video_source(spec)
    camera_columns = _camera_columns(video)
    ctx = {
        "digital_interface": _digital_interface_name(spec.recording_interface),
        "video_folder": video.folder,
        "camera_columns": repr(camera_columns),
        "primary_camera_column": camera_columns[0],
    }
    return [_md_cell("## Timing"), _code_cell(_render("timing_onsets.py.j2", ctx))], False


def _write_cell(spec: RigSpec, session_wide: bool) -> nbformat.NotebookNode:
    stream_rates_literal = _stream_rates_literal(spec)
    if spec.mode == "pair":
        ctx = {
            "stream_rates_literal": stream_rates_literal,
            "session_wide": session_wide,
            "session_wide_entries": _session_wide_entries(spec),
        }
        return _code_cell(_render("write_pair.py.j2", ctx))

    ctx = {
        "recording_interface": spec.recording_interface,
        "stream_rates_literal": stream_rates_literal,
        "session_wide": session_wide,
        "has_digital": spec.timing in ("pulses", "onsets"),
        "has_trial_metadata": spec.timing == "onsets",
    }
    return _code_cell(_render("write_neuroconv.py.j2", ctx))


def _open_cell(spec: RigSpec) -> nbformat.NotebookNode:
    target = f"{spec.session_dir}/.ethograph" if spec.mode == "pair" else f"{spec.session_dir}/session.nwb"
    return _md_cell(f"## Open\n\nStart page → select `{target}`")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_notebook(spec: RigSpec) -> nbformat.NotebookNode:
    """Turn a rig's wizard answers into one setup notebook."""
    nb = nbformat.v4.new_notebook()
    cells: list[nbformat.NotebookNode] = [
        _md_cell(f"## {spec.rig_name}: parameters"),
        _parameters_cell(spec),
        _md_cell("## Discover files"),
        _discover_cell(),
    ]

    timing_cells, session_wide = _timing_cells(spec)
    cells.extend(timing_cells)

    cells.append(_md_cell("## Write"))
    cells.append(_write_cell(spec, session_wide))
    cells.append(_open_cell(spec))

    nb["cells"] = cells
    return nb


def write_notebook(spec: RigSpec, path: str | Path) -> Path:
    """Build and write the rig's setup notebook to `path`."""
    nb = build_notebook(spec)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(out_path))
    return out_path
