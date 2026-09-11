"""
Napari dock widget for media file discovery and pattern matching.

Run standalone:  python media_discovery_widget.py
Dock in napari:  viewer.window.add_dock_widget(MediaDiscoveryWidget(viewer))
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.io.validation import (
    AUDIO_EXTENSIONS,
    POSE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from ethograph.utils.qt import mono_font

logger = logging.getLogger(__name__)

BG = "#1a1d21"
BG_PANEL = "#22262c"
BG_INPUT = "#2c3038"
BORDER = "#383e4a"
TEXT = "#e0e0e0"
TEXT_MID = "#9aa0ac"
TEXT_DIM = "#5e6470"
ACCENT = "#50c8b4"

COLOR_TRIAL = "#50c8b4"
COLOR_CAMERA = "#e8737a"
COLOR_MIC = "#e8c75a"

# Roles are filename-segment labels used by the pattern parser to identify
# what each part of a filename represents (e.g. "camera" = this segment
# identifies which camera).  These are NOT the xarray dimension names —
# the actual dims ("cameras", "mics") are defined in trialtree.STREAMS.
ROLE_COLORS = {
    "trial": COLOR_TRIAL,
    "camera": COLOR_CAMERA,
    "mic": COLOR_MIC,
    "ignore": TEXT_DIM,
    None: TEXT_DIM,
}

# Convert extension sets to regex patterns for classification
STREAM_RULES: dict[str, list[str]] = {
    "video": [rf"\{ext}$" for ext in VIDEO_EXTENSIONS],
    "pose": [rf"\{ext}$" for ext in POSE_EXTENSIONS],
    "audio": [rf"\{ext}$" for ext in AUDIO_EXTENSIONS],
}

FS = 13
MAX_PREVIEW = 20
FOLDER_POSITION = -1


# ─── pattern analysis ────────────────────────────────────────────────────────


@dataclass
class Segment:
    position: int
    text: str
    varying: bool
    values: list[str] = field(default_factory=list)
    role: str | None = None


@dataclass
class FilePattern:
    segments: list[Segment]
    files: list[Path]
    suffix: str
    tokenize_mode: str = "smart"
    regex_pattern: str | None = None

    def summary(self) -> dict[str, list[str]]:
        return {s.role: s.values for s in self.segments if s.varying and s.role and s.role != "ignore"}


_TIMESTAMP_RE = re.compile(
    r"^\d{4}[-/.]\d{2}[-/.]\d{2}$"
    r"|^\d{2}[-/.]\d{2}[-/.]\d{4}$"
    r"|^\d{2}[-/.]\d{2}[-/.]\d{2}$"
)


def _is_timestamp(s: str) -> bool:
    return bool(_TIMESTAMP_RE.match(s))


def _tokenize(name: str, mode: str = "smart") -> list[str]:
    if mode == "smart":
        primary = [t for t in re.split(r"[_\s]+", name) if t]
        result: list[str] = []
        for part in primary:
            if _is_timestamp(part):
                result.append(part)
            else:
                sub = [s for s in part.split("-") if s]
                result.extend(sub)
        return result
    if mode == "_":
        return [t for t in re.split(r"[_\s]+", name) if t]
    if mode == "-":
        return [t for t in re.split(r"[-\s]+", name) if t]
    return name.split("_")


def _find_token_spans(stem: str, tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        idx = stem.find(tok, pos)
        if idx >= 0:
            spans.append((idx, idx + len(tok)))
            pos = idx + len(tok)
        else:
            spans.append((pos, pos))
    return spans


def _build_segments(tokenized: list[list[str]]) -> list[Segment]:
    segments: list[Segment] = []
    for i, grp in enumerate(zip(*tokenized)):
        uniq = sorted(set(grp))
        if len(uniq) == 1:
            segments.append(Segment(i, uniq[0], False))
        else:
            segments.append(Segment(i, "", True, uniq))
    return segments


def analyze_filenames_with_regex(files: list[Path], pattern: str) -> FilePattern | None:
    """Build a FilePattern from a regex with named groups (trial, camera, mic)."""
    try:
        rx = re.compile(pattern)
    except re.error:
        return None

    group_names = list(rx.groupindex.keys())
    if not group_names:
        return None

    matched_files: list[Path] = []
    values_by_group: dict[str, set[str]] = {g: set() for g in group_names}

    for f in files:
        m = rx.search(f.stem)
        if not m:
            continue
        matched_files.append(f)
        for g in group_names:
            val = m.group(g)
            if val is not None:
                values_by_group[g].add(val)

    if not matched_files:
        return None

    segments: list[Segment] = []
    for i, g in enumerate(group_names):
        vals = sorted(values_by_group[g])
        role = g if g in ("trial", "camera", "mic") else "ignore"
        segments.append(
            Segment(
                position=i,
                text="",
                varying=len(vals) > 1,
                values=vals,
                role=role,
            )
        )

    suffix = matched_files[0].suffix
    return FilePattern(
        segments,
        matched_files,
        suffix,
        tokenize_mode="regex",
        regex_pattern=pattern,
    )


def classify_stream(filename: str) -> str | None:
    lower = filename.lower()
    for stream, pats in STREAM_RULES.items():
        if any(re.search(p, lower) for p in pats):
            return stream
    return None


def extract_file_row(
    filepath: Path,
    segments: list[Segment],
    tokenize_mode: str = "smart",
    *,
    regex_pattern: str | None = None,
) -> dict[str, str]:
    row: dict[str, str] = {"path": str(filepath)}
    if tokenize_mode == "regex" and regex_pattern:
        m = re.search(regex_pattern, filepath.stem)
        if m:
            for key, val in m.groupdict().items():
                if val is not None and key in ("trial", "camera", "mic"):
                    row[key] = val
        return row
    tokens = _tokenize(filepath.stem, tokenize_mode)
    for seg in segments:
        if not (seg.varying and seg.role and seg.role != "ignore"):
            continue
        if seg.position == FOLDER_POSITION:
            row[seg.role] = filepath.parent.name
        elif seg.position < len(tokens):
            row[seg.role] = tokens[seg.position]
    return row


# ─── config persistence ──────────────────────────────────────────────────────

CONFIG_FILENAME = ".media_discovery.json"


@dataclass
class StreamConfig:
    folder: str
    role_map: dict[int, str]  # segment position → role name
    nested: bool = False
    regex_pattern: str | None = None


@dataclass
class MediaConfig:
    streams: dict[str, StreamConfig]

    def to_dict(self) -> dict:
        return {
            "streams": {
                name: {
                    "folder": sc.folder,
                    "roles": {str(k): v for k, v in sc.role_map.items()},
                    "nested": sc.nested,
                    **({"regex": sc.regex_pattern} if sc.regex_pattern else {}),
                }
                for name, sc in self.streams.items()
            }
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MediaConfig":
        streams = {}
        for name, sc in d.get("streams", {}).items():
            streams[name] = StreamConfig(
                folder=sc["folder"],
                role_map={int(k): v for k, v in sc.get("roles", {}).items()},
                nested=sc.get("nested", False),
                regex_pattern=sc.get("regex"),
            )
        return cls(streams=streams)

    def save(self, path: str | Path) -> None:
        import json

        path = Path(path)
        if path.is_dir():
            path = path / CONFIG_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MediaConfig":
        import json

        path = Path(path)
        if path.is_dir():
            path = path / CONFIG_FILENAME
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def exists(cls, path: str | Path) -> bool:
        path = Path(path)
        if path.is_dir():
            path = path / CONFIG_FILENAME
        return path.is_file()


def _apply_roles(pattern: FilePattern, role_map: dict[int, str]) -> None:
    for seg in pattern.segments:
        if seg.varying and seg.position in role_map:
            seg.role = role_map[seg.position]


# ─── filename list ───────────────────────────────────────────────────────────


class FilenameList(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(20, 10, 20, 10)
        self._lay.setSpacing(4)
        self._rows: list[QLabel] = []

    def refresh(self, pattern: FilePattern | None):
        for w in self._rows:
            w.deleteLater()
        self._rows.clear()
        if not pattern:
            return

        mono = mono_font(FS)

        if pattern.tokenize_mode == "regex" and pattern.regex_pattern:
            self._refresh_regex(pattern, mono)
            return

        segs = [s for s in pattern.segments if s.position != FOLDER_POSITION]
        folder_seg = next((s for s in pattern.segments if s.position == FOLDER_POSITION), None)
        for fp in pattern.files[:MAX_PREVIEW]:
            stem = fp.stem
            tokens = _tokenize(stem, pattern.tokenize_mode)
            spans = _find_token_spans(stem, tokens)
            parts: list[str] = []
            if folder_seg:
                tok = fp.parent.name
                if folder_seg.varying and folder_seg.role and folder_seg.role != "ignore":
                    c = ROLE_COLORS.get(folder_seg.role, TEXT_DIM)
                    parts.append(f"<span style='color:{c};font-weight:700'>{tok}</span>")
                else:
                    parts.append(f"<span style='color:{TEXT_DIM}'>{tok}</span>")
                parts.append(f"<span style='color:{TEXT_DIM}'>/</span>")
            prev_end = 0
            for idx, seg in enumerate(segs):
                if idx >= len(spans):
                    break
                start, end = spans[idx]
                if start > prev_end:
                    delim_text = stem[prev_end:start]
                    parts.append(f"<span style='color:{TEXT_DIM}'>{delim_text}</span>")
                tok_text = stem[start:end]
                if seg.varying and seg.role and seg.role != "ignore":
                    c = ROLE_COLORS.get(seg.role, TEXT_DIM)
                    parts.append(f"<span style='color:{c};font-weight:700'>{tok_text}</span>")
                else:
                    parts.append(f"<span style='color:{TEXT_DIM}'>{tok_text}</span>")
                prev_end = end
            parts.append(f"<span style='color:{TEXT_DIM}'>{pattern.suffix}</span>")

            lbl = QLabel("".join(parts))
            lbl.setFont(mono)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("background:transparent; padding:2px 0;")
            self._lay.addWidget(lbl)
            self._rows.append(lbl)

        self._add_overflow_label(pattern, mono)

    def _refresh_regex(self, pattern: FilePattern, mono: QFont):
        rx = re.compile(pattern.regex_pattern)
        for fp in pattern.files[:MAX_PREVIEW]:
            stem = fp.stem
            m = rx.search(stem)
            if not m:
                html = f"<span style='color:{TEXT_DIM}'>{stem}{pattern.suffix}</span>"
            else:
                group_spans = []
                for name in rx.groupindex:
                    s, e = m.span(name)
                    if s >= 0:
                        role = name if name in ("trial", "camera", "mic") else None
                        group_spans.append((s, e, role))
                group_spans.sort()

                parts: list[str] = []
                prev = 0
                for s, e, role in group_spans:
                    if s > prev:
                        parts.append(f"<span style='color:{TEXT_DIM}'>{stem[prev:s]}</span>")
                    c = ROLE_COLORS.get(role, TEXT_DIM)
                    parts.append(f"<span style='color:{c};font-weight:700'>{stem[s:e]}</span>")
                    prev = e
                if prev < len(stem):
                    parts.append(f"<span style='color:{TEXT_DIM}'>{stem[prev:]}</span>")
                parts.append(f"<span style='color:{TEXT_DIM}'>{pattern.suffix}</span>")
                html = "".join(parts)

            lbl = QLabel(html)
            lbl.setFont(mono)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("background:transparent; padding:2px 0;")
            self._lay.addWidget(lbl)
            self._rows.append(lbl)

        self._add_overflow_label(pattern, mono)

    def set_plain_files(self, files: list[Path]):
        """Show filenames without any coloring."""
        for w in self._rows:
            w.deleteLater()
        self._rows.clear()
        mono = mono_font(FS)
        for fp in files[:MAX_PREVIEW]:
            lbl = QLabel(f"<span style='color:{TEXT_DIM}'>{fp.name}</span>")
            lbl.setFont(mono)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("background:transparent; padding:2px 0;")
            self._lay.addWidget(lbl)
            self._rows.append(lbl)
        self._add_overflow(len(files), mono)

    def _add_overflow_label(self, pattern: FilePattern, mono: QFont):
        self._add_overflow(len(pattern.files), mono)

    def _add_overflow(self, total: int, mono: QFont):
        rest = total - MAX_PREVIEW
        if rest > 0:
            lbl = QLabel(f"<span style='color:{TEXT_DIM};font-style:italic'>… {rest} more</span>")
            lbl.setFont(mono)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("background:transparent;")
            self._lay.addWidget(lbl)
            self._rows.append(lbl)


# ─── pattern editor ─────────────────────────────────────────────────────────


class PatternEditor(QWidget):
    """Visual filename-pattern editor.

    Workflow — "pick colour, then paint":
      1. Click a role button (Trial / Camera / Mic) to activate it.
      2. Drag-select text in the reference filename — it immediately
         highlights in that role's colour.
      3. Pick another role and highlight another part.
      4. The file list and pattern bar below update automatically.

    Internally the widget builds a regex from the marked regions and
    emits ``pattern_changed`` so the ``StreamPanel`` can re-analyse.
    """

    pattern_changed = Signal(str)  # emits regex string, or "" when cleared

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._marks: list[tuple[int, int, str]] = []  # (start, end, role)
        self._reference = ""
        self._override = ""
        self._active_role: str | None = None
        self._role_buttons: dict[str, QPushButton] = {}
        self._build()

    # ── build ────────────────────────────────────────────────────────────

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 6, 14, 2)
        layout.setSpacing(4)

        # reference file selector
        ref_row = QHBoxLayout()
        ref_row.setSpacing(6)
        ref_lbl = QLabel("Example file:")
        ref_lbl.setStyleSheet(f"color:{TEXT_MID}; font-size:{FS - 1}px;")
        ref_row.addWidget(ref_lbl)
        self._file_combo = QComboBox()
        self._file_combo.setStyleSheet(
            f"QComboBox {{ background:{BG_INPUT}; color:{TEXT}; "
            f"border:1px solid {BORDER}; border-radius:4px; "
            f"padding:3px 8px; font-size:{FS - 1}px; }}"
            f"QComboBox::drop-down {{ border:none; width:18px; }}"
            f"QComboBox QAbstractItemView {{ background:{BG_PANEL}; "
            f"color:{TEXT}; selection-background-color:{BORDER}; }}"
        )
        self._file_combo.currentTextChanged.connect(self._on_reference_changed)
        ref_row.addWidget(self._file_combo, stretch=1)
        layout.addLayout(ref_row)

        # role buttons — pick a colour first, then paint on the filename
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        hint = QLabel("Pick role, then highlight:")
        hint.setStyleSheet(f"color:{TEXT_DIM}; font-size:{FS - 1}px;")
        btn_row.addWidget(hint)
        for role, color, label in [
            ("trial", COLOR_TRIAL, "Trial"),
            ("camera", COLOR_CAMERA, "Camera"),
            ("mic", COLOR_MIC, "Mic"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, r=role: self._activate_role(r))
            self._role_buttons[role] = btn
            self._style_role_btn(btn, color, active=False)
            btn_row.addWidget(btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{TEXT_DIM}; "
            f"border:1px solid {BORDER}; border-radius:4px; "
            f"padding:3px 10px; font-size:{FS - 1}px; }}"
            f"QPushButton:hover {{ color:{TEXT}; border-color:{TEXT_MID}; }}"
        )
        clear_btn.clicked.connect(self._clear_marks)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # filename text — read-only QLineEdit for mouse-selection
        self._edit = QLineEdit()
        self._edit.setReadOnly(True)
        self._edit.setFont(mono_font(FS))
        self._edit.setStyleSheet(
            f"QLineEdit {{ background:{BG_INPUT}; color:{TEXT}; "
            f"border:1px solid {BORDER}; border-radius:4px; "
            f"padding:6px 10px; font-size:{FS}px; }}"
            f"QLineEdit:focus {{ border-color:{ACCENT}; }}"
        )
        self._edit.selectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._edit)

        # coloured preview
        self._preview = QLabel()
        self._preview.setFont(mono_font(FS))
        self._preview.setTextFormat(Qt.TextFormat.RichText)
        self._preview.setStyleSheet("background:transparent; padding:2px 0;")
        layout.addWidget(self._preview)

        # debounce timer — emit pattern_changed after selection stabilises
        self._emit_timer = QTimer()
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(200)
        self._emit_timer.timeout.connect(lambda: self.pattern_changed.emit(self._build_regex()))

    @staticmethod
    def _style_role_btn(btn: QPushButton, color: str, *, active: bool):
        if active:
            btn.setStyleSheet(
                f"QPushButton {{ background:{color}; color:{BG}; "
                f"border:1.5px solid {color}; border-radius:4px; "
                f"padding:3px 12px; font-size:{FS - 1}px; font-weight:700; }}"
            )
        else:
            btn.setStyleSheet(
                f"QPushButton {{ background:{BG_INPUT}; color:{color}; "
                f"border:1.5px solid {color}; border-radius:4px; "
                f"padding:3px 12px; font-size:{FS - 1}px; font-weight:700; }}"
                f"QPushButton:hover {{ background:{color}; color:{BG}; }}"
            )

    # ── public API ───────────────────────────────────────────────────────

    def set_files(self, files: list[Path]):
        """Populate the reference-file combo.  First file becomes the default."""
        stems = [f.stem for f in files[:60]]
        self._file_combo.blockSignals(True)
        self._file_combo.clear()
        self._file_combo.addItems(stems)
        self._file_combo.blockSignals(False)
        if stems:
            self._set_reference(stems[0])
        else:
            self._set_reference("")

    def set_regex(self, regex: str):
        """Restore a previously-saved regex (from config)."""
        self._marks.clear()
        self._override = regex
        self._update_preview()
        self.pattern_changed.emit(regex)

    @property
    def regex(self) -> str:
        """Current regex string, or ``""`` when no marks are set."""
        return self._build_regex()

    # ── internals ────────────────────────────────────────────────────────

    def _set_reference(self, stem: str):
        self._reference = stem
        self._edit.setText(stem)
        self._marks.clear()
        self._override = ""
        self._update_preview()
        self.pattern_changed.emit("")

    def _on_reference_changed(self, text: str):
        self._set_reference(text)

    def _activate_role(self, role: str):
        """Toggle a role as the active paint brush."""
        role_colors = {"trial": COLOR_TRIAL, "camera": COLOR_CAMERA, "mic": COLOR_MIC}
        if self._active_role == role:
            # Deactivate
            self._active_role = None
            for r, btn in self._role_buttons.items():
                btn.setChecked(False)
                self._style_role_btn(btn, role_colors[r], active=False)
        else:
            self._active_role = role
            for r, btn in self._role_buttons.items():
                is_active = r == role
                btn.setChecked(is_active)
                self._style_role_btn(btn, role_colors[r], active=is_active)

    def _on_selection_changed(self):
        """Live-paint: when the user drags a selection while a role is active,
        immediately mark that region and update the preview."""
        if not self._active_role or not self._reference:
            return
        sel = self._edit.selectedText()
        if not sel:
            return
        start = self._edit.selectionStart()
        end = start + len(sel)
        role = self._active_role
        # Replace any existing mark for this role + remove overlaps
        self._marks = [(s, e, r) for s, e, r in self._marks if r != role and (e <= start or s >= end)]
        self._marks.append((start, end, role))
        self._marks.sort()
        self._override = ""
        self._update_preview()
        # Debounced: let the selection stabilise before triggering analysis
        self._emit_timer.start()

    def _clear_marks(self):
        self._marks.clear()
        self._active_role = None
        self._override = ""
        role_colors = {"trial": COLOR_TRIAL, "camera": COLOR_CAMERA, "mic": COLOR_MIC}
        for r, btn in self._role_buttons.items():
            btn.setChecked(False)
            self._style_role_btn(btn, role_colors[r], active=False)
        self._update_preview()
        self.pattern_changed.emit("")

    def _update_preview(self):
        if not self._reference:
            self._preview.setText("")
            return
        if not self._marks:
            self._preview.setText("")
            return
        parts: list[str] = []
        prev = 0
        for start, end, role in self._marks:
            if start > prev:
                parts.append(f"<span style='color:{TEXT_DIM}'>{self._reference[prev:start]}</span>")
            c = ROLE_COLORS.get(role, TEXT_DIM)
            parts.append(
                f"<span style='color:{c};font-weight:bold;"
                f"text-decoration:underline'>"
                f"{self._reference[start:end]}</span>"
                f"<sub style='color:{c}'> {role}</sub>"
            )
            prev = end
        if prev < len(self._reference):
            parts.append(f"<span style='color:{TEXT_DIM}'>{self._reference[prev:]}</span>")
        self._preview.setText("".join(parts))

    def _build_regex(self) -> str:
        if self._override:
            return self._override
        if not self._marks or not self._reference:
            return ""
        parts: list[str] = []
        prev = 0
        for start, end, role in self._marks:
            if start > prev:
                parts.append(re.escape(self._reference[prev:start]))
            # Infer a character class from the delimiter that follows
            if end < len(self._reference):
                nxt = self._reference[end]
                parts.append(f"(?P<{role}>[^{re.escape(nxt)}]+)")
            else:
                parts.append(f"(?P<{role}>.+)")
            prev = end
        if prev < len(self._reference):
            parts.append(re.escape(self._reference[prev:]))
        return "".join(parts)


# ─── stream panel ────────────────────────────────────────────────────────────


class StreamPanel(QWidget):
    changed = Signal()

    def __init__(
        self,
        stream: str,
        parent: QWidget | None = None,
        allowed_roles: list[str] | None = None,
    ):
        super().__init__(parent)
        self._stream = stream
        self._allowed_roles = allowed_roles
        self._pattern: FilePattern | None = None
        self._all_files: list[Path] = []
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 14, 0, 0)
        outer.setSpacing(10)

        # folder row
        row = QHBoxLayout()
        row.setContentsMargins(14, 0, 14, 0)
        self._folder = QLineEdit()
        self._folder.setPlaceholderText(f"select {self._stream} folder …")
        self._folder.setStyleSheet(
            f"QLineEdit {{ background:{BG_INPUT}; color:{TEXT}; "
            f"border:1px solid {BORDER}; border-radius:5px; "
            f"padding:9px 14px; font-size:{FS}px; }}"
            f"QLineEdit:focus {{ border-color:{ACCENT}; }}"
        )
        self._folder.textChanged.connect(self._on_folder)
        btn = QPushButton("…")
        btn.setFixedSize(38, 38)
        btn.setStyleSheet(
            f"QPushButton {{ background:{BG_INPUT}; color:{TEXT_MID}; "
            f"border:1px solid {BORDER}; border-radius:5px; "
            f"font-size:18px; font-weight:bold; }}"
            f"QPushButton:hover {{ border-color:{ACCENT}; color:{TEXT}; }}"
        )
        btn.clicked.connect(self._browse)
        row.addWidget(self._folder, stretch=1)
        row.addWidget(btn)
        outer.addLayout(row)

        # nested subfolder checkbox
        self._nested_cb = QCheckBox("Scan subfolders (e.g. one folder per camera)")
        self._nested_cb.setStyleSheet(
            f"QCheckBox {{ color:{TEXT_MID}; font-size:{FS - 1}px; padding:0 14px; }}"
            f"QCheckBox::indicator {{ width:14px; height:14px; }}"
        )
        self._nested_cb.toggled.connect(lambda: self._on_folder(self._folder.text()))
        outer.addWidget(self._nested_cb)

        # pattern editor
        self._pattern_editor = PatternEditor()
        self._pattern_editor.pattern_changed.connect(lambda _: self._apply_analysis())
        outer.addWidget(self._pattern_editor)

        # summary
        self._summary = QLabel()
        self._summary.setStyleSheet(f"color:{TEXT_MID}; font-size:{FS}px; padding:0 20px;")
        outer.addWidget(self._summary)

        # scrollable filename list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border:none; background:transparent; }}"
            f"QScrollBar:vertical {{ background:{BG}; width:7px; border:none; }}"
            f"QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:3px; min-height:30px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}"
        )
        self._flist = FilenameList()
        scroll.setWidget(self._flist)
        outer.addWidget(scroll, stretch=1)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, f"select {self._stream} folder")
        if d:
            self._folder.setText(d)

    def set_folder(self, path: str) -> None:
        """Seed the folder from what an earlier wizard page already picked.

        Without this, a source already chosen on the Sources page (folder or
        explicit files) would have to be re-browsed here from scratch —
        this is the same folder, just re-typed into a different widget.
        """
        if path:
            self._folder.setText(path)

    def _on_folder(self, text: str):
        p = Path(text)
        if not p.is_dir():
            self._all_files = []
            self._set_pattern(None)
            return
        if self._nested_cb.isChecked():
            all_files: list[Path] = []
            for sd in sorted(d for d in p.iterdir() if d.is_dir()):
                files = sorted(f for f in sd.iterdir() if f.is_file())
                relevant = [f for f in files if classify_stream(f.name) == self._stream]
                all_files.extend(relevant or files)
            self._all_files = all_files
        else:
            files = sorted(f for f in p.iterdir() if f.is_file())
            relevant = [f for f in files if classify_stream(f.name) == self._stream]
            self._all_files = relevant or files
        self._pattern_editor.set_files(self._all_files)
        self._apply_analysis()

    def _apply_analysis(self):
        if not self._all_files:
            self._set_pattern(None)
            return
        regex_text = self._pattern_editor.regex
        if not regex_text:
            self._flist.set_plain_files(self._all_files)
            self._summary.setText(f"{len(self._all_files)} files")
            self._pattern = None
            self.changed.emit()
            return
        pat = analyze_filenames_with_regex(self._all_files, regex_text)
        self._set_pattern(pat)

    def _set_pattern(self, pat: FilePattern | None):
        self._pattern = pat
        if pat is None:
            self._flist.refresh(None)
            self._summary.setText("")
            self.changed.emit()
            return
        self._flist.refresh(pat)
        info = pat.summary()
        bits = [f"{len(pat.files)} files"]
        for key, label in [("trial", "trials"), ("camera", "cameras"), ("mic", "mics")]:
            if key in info:
                vals = info[key]
                extra = f" ({', '.join(vals)})" if len(vals) <= 6 else ""
                bits.append(f"{len(vals)} {label}{extra}")
        self._summary.setText("  ·  ".join(bits))
        self.changed.emit()

    @property
    def pattern(self) -> FilePattern | None:
        return self._pattern

    def first_file(self) -> str | None:
        """The first matching file found in the folder, pattern or not — for
        probing a rate/fps that a single representative file can answer."""
        return str(self._all_files[0]) if self._all_files else None

    def get_config(self) -> StreamConfig | None:
        """The chosen folder, ready to pair — a drawn pattern is optional.

        No pattern (the user never painted a trial/camera/mic role) is a
        valid single-device answer: the files pair to trials by natural sort,
        exactly like a single-camera drop. A pattern is only needed to name
        more than one device or to disambiguate trial numbering.
        """
        folder = self._folder.text()
        if not folder or not self._all_files:
            return None
        if self._pattern is None:
            return StreamConfig(folder=folder, role_map={}, nested=self._nested_cb.isChecked(), regex_pattern=None)
        role_map = {
            seg.position: seg.role
            for seg in self._pattern.segments
            if seg.varying and seg.role and seg.role != "ignore"
        }
        return StreamConfig(
            folder=folder,
            role_map=role_map,
            nested=self._nested_cb.isChecked(),
            regex_pattern=self._pattern.regex_pattern,
        )

    def apply_config(self, cfg: StreamConfig) -> None:
        self._nested_cb.setChecked(cfg.nested)
        if cfg.regex_pattern:
            self._pattern_editor.set_regex(cfg.regex_pattern)
        self._folder.setText(cfg.folder)
        if self._pattern is not None and not cfg.regex_pattern:
            _apply_roles(self._pattern, cfg.role_map)
            self._set_pattern(self._pattern)


# ─── session table ───────────────────────────────────────────────────────────


class SessionPreview(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 10, 0, 0)
        lay.setSpacing(6)

        hdr = QLabel("session table")
        hdr.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px; padding:0 14px;")
        lay.addWidget(hdr)

        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            f"QTableWidget {{ gridline-color:{BORDER}; background:{BG}; "
            f"alternate-background-color:{BG_PANEL}; color:{TEXT}; "
            f"font-size:{FS}px; border:none; }}"
            f"QHeaderView::section {{ background:{BG_PANEL}; color:{TEXT_MID}; "
            f"padding:6px 10px; border:none; border-bottom:1px solid {BORDER}; "
            f"font-size:{FS - 1}px; }}"
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        lay.addWidget(self._table, stretch=1)

        self._status = QLabel()
        self._status.setStyleSheet(f"color:{TEXT_MID}; font-size:{FS}px; padding:4px 14px;")
        lay.addWidget(self._status)

    def update_from_panels(self, panels: list[StreamPanel]):
        import pandas as pd

        dfs: list[pd.DataFrame] = []
        for panel in panels:
            pat = panel.pattern
            if not pat:
                continue
            stream = panel._stream
            rows = [
                extract_file_row(f, pat.segments, pat.tokenize_mode, regex_pattern=pat.regex_pattern) for f in pat.files
            ]
            df = pd.DataFrame(rows)
            if "trial" not in df.columns:
                continue
            dev = next((c for c in ("camera", "mic") if c in df.columns), None)
            if dev:
                piv = df.pivot(index="trial", columns=dev, values="path")
                piv.columns = [f"{stream}_{c}" for c in piv.columns]
                piv = piv.reset_index()
            else:
                piv = df[["trial", "path"]].rename(columns={"path": f"{stream}_0"})
            dfs.append(piv)

        if not dfs:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._status.setText("assign 'trial' in at least one stream")
            return

        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.merge(d, on="trial", how="outer")

        # Sort naturally: numeric if all trial IDs are digits, else alphabetical
        trials = merged["trial"]
        if trials.apply(lambda v: str(v).isdigit()).all():
            merged = merged.assign(_sort=merged["trial"].astype(int))
        else:
            merged = merged.assign(_sort=merged["trial"].astype(str).str.lower())
        merged = merged.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

        cols = list(merged.columns)
        show = min(len(merged), 10)
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(show)

        stream_bg = {"video": COLOR_TRIAL, "pose": COLOR_CAMERA, "audio": COLOR_MIC}
        for r in range(show):
            for c, col in enumerate(cols):
                val = merged.iloc[r][col]
                txt = Path(str(val)).name if pd.notna(val) and str(val) else ""
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                for sn, clr in stream_bg.items():
                    if col.startswith(sn):
                        qc = QColor(clr)
                        qc.setAlpha(25)
                        item.setBackground(qc)
                self._table.setItem(r, c, item)

        self._table.resizeColumnsToContents()
        nt = merged["trial"].nunique()
        mc = len([c for c in cols if c != "trial"])
        extra = f"  (showing {show}/{len(merged)})" if len(merged) > show else ""
        self._status.setText(f"{nt} trials  ·  {mc} streams{extra}")


# ─── main widget ─────────────────────────────────────────────────────────────


class MediaDiscoveryWidget(QWidget):
    def __init__(self, napari_viewer=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._panels: list[StreamPanel] = []
        self._build()

    def _build(self):
        self.setWindowTitle("media discovery")
        self.setMinimumWidth(500)
        self.setStyleSheet(f"background:{BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 10)
        root.setSpacing(0)

        # ── tab labels (plain text, spaced by font size) ──
        self._tab_labels: list[QLabel] = []
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(16, 12, 16, 10)
        tab_row.setSpacing(FS * 2)

        for i, name in enumerate(("Video", "Pose", "Audio")):
            lbl = QLabel(name)
            lbl.setStyleSheet(
                f"color:{TEXT_DIM}; font-size:{FS + 1}px; padding:0; background:transparent; border:none;"
            )
            lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            lbl.mousePressEvent = lambda _, idx=i: self._show_tab(idx)
            tab_row.addWidget(lbl)
            self._tab_labels.append(lbl)

        tab_row.addStretch()
        tab_bg = QWidget()
        tab_bg.setStyleSheet(f"background:{BG_PANEL};")
        tab_bg.setLayout(tab_row)
        root.addWidget(tab_bg)

        # ── stream panels (stacked) ──
        for stream in ("video", "pose", "audio"):
            p = StreamPanel(stream)
            p.changed.connect(self._rebuild_session)
            p.setVisible(False)
            self._panels.append(p)
            root.addWidget(p, stretch=3)

        # ── divider ──
        div = QWidget()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background:{BORDER};")
        root.addWidget(div)

        # ── session table (always visible, bottom) ──
        self._session = SessionPreview()
        root.addWidget(self._session, stretch=2)

        # ── save / load row ──
        bar = QHBoxLayout()
        bar.setContentsMargins(14, 8, 14, 0)

        self._config_status = QLabel()
        self._config_status.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        bar.addWidget(self._config_status)
        bar.addStretch()

        load_btn = QPushButton("Load config")
        load_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{TEXT_MID}; "
            f"border:1px solid {BORDER}; border-radius:4px; "
            f"padding:7px 16px; font-size:{FS}px; }}"
            f"QPushButton:hover {{ border-color:{TEXT_MID}; color:{TEXT}; }}"
        )
        load_btn.clicked.connect(self._load_config)
        bar.addWidget(load_btn)

        save_btn = QPushButton("Save config")
        save_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{ACCENT}; "
            f"border:1px solid {ACCENT}; border-radius:4px; "
            f"padding:7px 16px; font-size:{FS}px; }}"
            f"QPushButton:hover {{ background:{ACCENT}; color:{BG}; }}"
        )
        save_btn.clicked.connect(self._save_config)
        bar.addWidget(save_btn)

        root.addLayout(bar)

        self._show_tab(0)

    def _show_tab(self, idx: int):
        for i, p in enumerate(self._panels):
            p.setVisible(i == idx)
        for i, lbl in enumerate(self._tab_labels):
            if i == idx:
                lbl.setStyleSheet(
                    f"color:{TEXT}; font-size:{FS + 1}px; font-weight:bold; "
                    f"padding:0; background:transparent; border:none;"
                )
            else:
                lbl.setStyleSheet(
                    f"color:{TEXT_DIM}; font-size:{FS + 1}px; padding:0; background:transparent; border:none;"
                )

    def _rebuild_session(self):
        self._session.update_from_panels(self._panels)

    # ── config persistence ──

    def get_config(self) -> MediaConfig:
        streams = {}
        stream_names = ("video", "pose", "audio")
        for panel, name in zip(self._panels, stream_names):
            cfg = panel.get_config()
            if cfg is not None:
                streams[name] = cfg
        return MediaConfig(streams=streams)

    def apply_config(self, config: MediaConfig) -> None:
        stream_names = ("video", "pose", "audio")
        for panel, name in zip(self._panels, stream_names):
            if name in config.streams:
                panel.apply_config(config.streams[name])
        self._rebuild_session()

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save media config", CONFIG_FILENAME, "JSON (*.json)")
        if not path:
            return
        config = self.get_config()
        config.save(path)
        self._config_status.setText(f"saved → {Path(path).name}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load media config", "", "JSON (*.json)")
        if not path:
            return
        config = MediaConfig.load(path)
        self.apply_config(config)
        self._config_status.setText(f"loaded ← {Path(path).name}")
