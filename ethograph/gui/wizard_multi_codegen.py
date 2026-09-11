"""Generate executable Python code for temporal alignment setup."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jinja2

from ethograph.io.validation import (
    AUDIO_EXTENSIONS,
    EPHYS_EXTENSIONS,
    POSE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)

if TYPE_CHECKING:
    from ethograph.gui.wizard_media_files import FilePattern
    from ethograph.gui.wizard_state import ModalityConfig, WizardState

_TEMPLATE_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModalityContext:
    name: str
    file_mode: str
    folder_path: str = ""
    single_file_path: str = ""
    nested_subfolders: bool = False
    extensions: list[str] = field(default_factory=list)
    regex: str | None = None  # non-anchored pattern for .search()
    device_role: str | None = None  # named group that carries device: "camera" | "mic"
    constant_offset: float = 0.0
    fps: float | None = None
    audio_sr: float | None = None


# ---------------------------------------------------------------------------
# Regex: non-anchored pattern from wizard segments
# ---------------------------------------------------------------------------

_TOKEN = r"[^/._\-\s]+"


def _segments_to_regex(pattern: FilePattern) -> str | None:
    """Return a non-anchored regex with named groups for role-bearing segments."""
    from ethograph.gui.wizard_media_files import (
        FOLDER_POSITION,
        _find_token_spans,
        _tokenize,
    )

    has_roles = any(
        seg.varying and seg.role and seg.role != "ignore" for seg in pattern.segments if seg.position != FOLDER_POSITION
    )
    if not has_roles or not pattern.files:
        return None

    ref = pattern.files[0].stem
    tokens = _tokenize(ref, pattern.tokenize_mode)
    spans = _find_token_spans(ref, tokens)

    parts: list[str] = []
    prev_end = 0
    for idx, seg in enumerate(pattern.segments):
        if seg.position == FOLDER_POSITION:
            continue
        if idx < len(spans) and prev_end < spans[idx][0]:
            parts.append(re.escape(ref[prev_end : spans[idx][0]]))
        if idx < len(spans):
            prev_end = spans[idx][1]
        if seg.varying and seg.role and seg.role != "ignore":
            parts.append(f"(?P<{seg.role}>{_TOKEN})")
        elif not seg.varying:
            parts.append(re.escape(seg.text))
        else:
            parts.append(_TOKEN)

    return "".join(parts)


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def _escape_path(path: str) -> str:
    return path.replace("\\", "\\\\")


def _get_extensions(name: str) -> list[str]:
    mapping = {
        "video": VIDEO_EXTENSIONS,
        "audio": AUDIO_EXTENSIONS,
        "pose": POSE_EXTENSIONS,
        "ephys": EPHYS_EXTENSIONS,
    }
    return sorted(mapping.get(name, []))


def _build_modality_context(name: str, cfg: ModalityConfig, state: WizardState) -> ModalityContext:
    extensions = (
        sorted({f.suffix.lower() for f in cfg.pattern.files if f.suffix})
        if cfg.pattern and cfg.pattern.files
        else _get_extensions(name)[:3]
    )

    regex = None
    if cfg.pattern and cfg.pattern.segments:
        if cfg.pattern.regex_pattern:
            regex = cfg.pattern.regex_pattern.replace('"', '\\"')
        else:
            raw = _segments_to_regex(cfg.pattern)
            if raw:
                regex = raw.replace('"', '\\"')

    device_role = "mic" if name == "audio" else "camera" if name in ("video", "pose") else None

    return ModalityContext(
        name=name,
        file_mode=cfg.file_mode,
        folder_path=_escape_path(cfg.folder_path or ""),
        single_file_path=_escape_path(cfg.single_file_path or ""),
        nested_subfolders=cfg.nested_subfolders,
        extensions=extensions,
        regex=regex,
        device_role=device_role,
        constant_offset=cfg.constant_offset,
        fps=cfg.fps,
        audio_sr=cfg.audio_sr if name == "audio" else None,
    )


def _stream_rates_literal(modalities: list[ModalityContext]) -> str:
    rates: dict[str, float] = {}
    for mod in modalities:
        if mod.name in ("video", "pose") and mod.fps:
            rates[mod.name] = float(mod.fps)
        elif mod.name == "audio" and mod.audio_sr:
            rates[mod.name] = float(mod.audio_sr)
    if not rates:
        return '{}  # TODO: add sampling rates, e.g. {"video": 30.0}'
    return "{" + ", ".join(f'"{k}": {v}' for k, v in rates.items()) + "}"


def _offsets(modalities: list[ModalityContext]) -> list[tuple[str, float]]:
    return [(m.name, m.constant_offset) for m in modalities if m.constant_offset != 0.0]


def _build_template_context(state: WizardState) -> dict:
    modalities = [
        _build_modality_context(name, getattr(state, name), state)
        for name in ("video", "pose", "audio", "ephys")
        if getattr(state, name).enabled
    ]
    media_modalities = [m for m in modalities if m.name in ("video", "pose", "audio")]
    pose_mod = next((m for m in modalities if m.name == "pose"), None)

    trial_table_source = None
    trial_table_sep = ","
    if state.trial_table_path:
        trial_table_source = state.trial_table_path
        trial_table_sep = "\\t" if state.trial_table_path.endswith(".tsv") else ","

    return {
        "modalities": modalities,
        "media_modalities": media_modalities,
        "any_pose": pose_mod is not None,
        "pose_fps": float(pose_mod.fps) if pose_mod and pose_mod.fps else None,
        "stream_rates_literal": _stream_rates_literal(modalities),
        "trial_table_source": _escape_path(trial_table_source) if trial_table_source else None,
        "trial_table_sep": trial_table_sep,
        "output_path": _escape_path(state.output_path) if state.output_path else None,
        "offsets": _offsets(modalities),
    }


# ---------------------------------------------------------------------------
# Jinja2 environment
# ---------------------------------------------------------------------------


def _create_jinja_env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
        undefined=jinja2.StrictUndefined,
    )

    def format_glob(ext: str, prefix: str) -> str:
        return f'"{prefix}{ext}"'

    env.filters["format_glob"] = format_glob
    return env


_env: jinja2.Environment | None = None


def _get_env() -> jinja2.Environment:
    global _env
    if _env is None:
        _env = _create_jinja_env()
    return _env


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_alignment_code(state: WizardState) -> str:
    """Generate executable Python code that reproduces the user's alignment setup."""
    ctx = _build_template_context(state)
    template = _get_env().get_template("wizard_nwb_codegen.j2")
    rendered = template.render(ctx)
    return _clean_blank_lines(rendered)


def _clean_blank_lines(text: str) -> str:
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n"))
