"""Reusable dialog for configuring function parameters as Python kwargs."""

import ast
import inspect
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

import ruptures as rpt
from qtpy.QtCore import QRegularExpression, Qt
from qtpy.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
)

from ethograph.gui.notify import notify_dialog

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ParamInfo:
    name: str
    type_hint: type | str
    default: Any
    description: str = ""


@dataclass
class FunctionSpec:
    func: Callable
    doc_url: str = ""
    auto_params: tuple[str, ...] = ("data", "sr")
    return_hint: str = ""
    params: list[ParamInfo] | None = None
    fixed_params: dict[str, Any] = None
    copy_preamble: str = ""
    import_path: str = ""
    copy_template: str = ""

    def __post_init__(self):
        if self.fixed_params is None:
            self.fixed_params = {}


# ---------------------------------------------------------------------------
# Ruptures thin wrappers (so inspect.signature works)
# ---------------------------------------------------------------------------


def _ruptures_pelt(
    data,
    model: str = "l2",
    min_size: int = 2,
    jump: int = 5,
    pen: float = 1.0,
) -> list[int]:
    """Penalized changepoint detection (PELT algorithm).

    Detects an unknown number of changepoints by minimizing a penalized cost.
    Faster than optimal methods for large signals.

    Args:
        data: 1-D or 2-D input signal.
        model: Cost function — 'l1', 'l2', 'rbf', 'linear', 'normal', 'ar'.
        min_size: Minimum segment length (samples).
        jump: Subsampling factor (higher = faster but less precise).
        pen: Penalty for adding a changepoint. Higher = fewer changepoints.
    """
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(data)
    return algo.predict(pen=pen)


def _ruptures_binseg(
    data,
    model: str = "l2",
    min_size: int = 2,
    jump: int = 5,
    n_bkps: int = 5,
    pen: float | None = None,
) -> list[int]:
    """Binary segmentation changepoint detection.

    Greedy sequential algorithm. Supports both penalty-based and
    fixed-number-of-breakpoints modes.

    Args:
        data: 1-D or 2-D input signal.
        model: Cost function — 'l1', 'l2', 'rbf', 'linear', 'normal', 'ar'.
        min_size: Minimum segment length (samples).
        jump: Subsampling factor.
        n_bkps: Number of breakpoints to detect (used when pen is None).
        pen: Penalty value. When set, overrides n_bkps.
    """
    algo = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(data)
    if pen is not None:
        return algo.predict(pen=pen)
    return algo.predict(n_bkps=n_bkps)


def _ruptures_bottomup(
    data,
    model: str = "l2",
    min_size: int = 2,
    jump: int = 5,
    n_bkps: int = 5,
) -> list[int]:
    """Bottom-up segmentation changepoint detection.

    Starts with many segments and merges adjacent ones greedily.

    Args:
        data: 1-D or 2-D input signal.
        model: Cost function — 'l1', 'l2', 'rbf', 'linear', 'normal', 'ar'.
        min_size: Minimum segment length (samples).
        jump: Subsampling factor.
        n_bkps: Number of breakpoints to detect.
    """
    algo = rpt.BottomUp(model=model, min_size=min_size, jump=jump).fit(data)
    return algo.predict(n_bkps=n_bkps)


def _ruptures_window(
    data,
    model: str = "l2",
    min_size: int = 2,
    jump: int = 5,
    n_bkps: int = 5,
    width: int = 100,
) -> list[int]:
    """Sliding-window changepoint detection.

    Slides a window across the signal and detects changepoints where
    the two halves of the window differ most.

    Args:
        data: 1-D or 2-D input signal.
        model: Cost function — 'l1', 'l2', 'rbf', 'linear', 'normal', 'ar'.
        min_size: Minimum segment length (samples).
        jump: Subsampling factor.
        n_bkps: Number of breakpoints to detect.
        width: Window width (samples).
    """
    algo = rpt.Window(width=width, model=model, min_size=min_size, jump=jump).fit(data)
    return algo.predict(n_bkps=n_bkps)


def _ruptures_dynp(
    data,
    model: str = "l2",
    min_size: int = 2,
    jump: int = 5,
    n_bkps: int = 5,
) -> list[int]:
    """Dynamic programming changepoint detection.

    Optimal segmentation — finds the global minimum of the sum of costs.
    Slow for large signals.

    Args:
        data: 1-D or 2-D input signal.
        model: Cost function — 'l1', 'l2', 'rbf', 'linear', 'normal', 'ar'.
        min_size: Minimum segment length (samples).
        jump: Subsampling factor.
        n_bkps: Number of breakpoints to detect.
    """
    algo = rpt.Dynp(model=model, min_size=min_size, jump=jump).fit(data)
    return algo.predict(n_bkps=n_bkps)


# ---------------------------------------------------------------------------
# Doc URLs
# ---------------------------------------------------------------------------

DOC_URLS: dict[str, str] = {
    "meansquared_cp": "https://vocalpy.readthedocs.io/en/latest/api/generated/vocalpy.segment.html",
    "ava_cp": "https://vocalpy.readthedocs.io/en/latest/api/generated/vocalpy.segment.html",
    "vocalseg_cp": "https://github.com/timsainb/vocalization-segmentation",
    "ruptures_pelt": "https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/",
    "ruptures_binseg": "https://centre-borelli.github.io/ruptures-docs/user-guide/detection/binseg/",
    "ruptures_bottomup": "https://centre-borelli.github.io/ruptures-docs/user-guide/detection/bottomup/",
    "ruptures_window": "https://centre-borelli.github.io/ruptures-docs/user-guide/detection/window/",
    "ruptures_dynp": "https://centre-borelli.github.io/ruptures-docs/user-guide/detection/dynp/",
    "energy_lowpass": "https://github.com/Akseli-Ilmanen/Ethograph/blob/main/ethograph/features/energy.py",
    "energy_highpass": "https://github.com/Akseli-Ilmanen/Ethograph/blob/main/ethograph/features/energy.py",
    "energy_band": "https://github.com/Akseli-Ilmanen/Ethograph/blob/main/ethograph/features/energy.py",
    "energy_meansquared": "https://vocalpy.readthedocs.io/",
    "energy_ava": "https://vocalpy.readthedocs.io/",
    "find_troughs": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html",
    "find_turning_points": "https://ethograph.readthedocs.io/en/latest/changepoints/#kinematic-changepoints",
    "continuity_cp": "https://github.com/timsainb/vocalization-segmentation",
    "oscillatory_events": "https://pynapple.org/advanced/12_filtering.html#detecting-oscillatory-events",
}


# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------


def _audio_preamble(*imports: str, data_var: str = "data", rate_var: str = "sr") -> str:
    header = "import vocalpy as voc\n"
    for imp in imports:
        header += f"{imp}\n"
    header += (
        "\n"
        'sound = voc.Sound.read("path/to/audio.wav")\n'
        f"{data_var} = sound.data[0]  # first channel, 1-D\n"
        f"{rate_var} = sound.samplerate\n"
    )
    return header


_AUDIO_PREAMBLE = _audio_preamble()

_VOCALSEG_PREAMBLE = _audio_preamble(
    "from ethograph._vendor.vocalseg.dynamic_thresholding import dynamic_threshold_segmentation",
)

_VOCALSEG_CONTINUITY_PREAMBLE = _audio_preamble(
    "from ethograph._vendor.vocalseg.continuity_filtering import continuity_segmentation",
)

# TODO: rewrite with example for ephys data (e.g. for high-pass)
_ENERGY_AUDIO_PREAMBLE = _audio_preamble(
    "from ethograph.features.energy import {func}",
)


_ENERGY_VOCALPY_PREAMBLE = _audio_preamble(
    "from ethograph.features.energy import {func}",
)

_RUPTURES_PREAMBLE = "import numpy as np\nimport ruptures as rpt\n\ndata = ...  # (N,) 1-D numpy array\n"

_KINEMATIC_PREAMBLE = (
    "from ethograph.features.changepoints import {func}\n"
    "\n"
    "# Select a 1-D feature from your xarray dataset\n"
    'speed = ds["speed"].sel(keypoint="snout", individual="mouse1").values\n'
)

_OSCILLATORY_PREAMBLE = (
    "import numpy as np\n"
    "import pynapple as nap\n"
    "from ethograph.features.oscillatory import detect_oscillatory_events_np\n"
    "\n"
    "data = ...  # 1-D numpy array (ephys, audio, or feature signal)\n"
    "sr = ...    # sampling rate in Hz\n"
)


def _build_registry() -> dict[str, FunctionSpec]:
    import vocalpy as voc

    from ethograph._vendor.vocalseg.continuity_filtering import continuity_segmentation
    from ethograph._vendor.vocalseg.dynamic_thresholding import (
        dynamic_threshold_segmentation,
    )
    from ethograph.features.changepoints import (
        find_nearest_turning_points_binary,
        find_troughs_binary,
    )
    from ethograph.features.energy import (
        bandpass_envelope,
        highpass_envelope,
        lowpass_envelope,
    )
    from ethograph.features.oscillatory import detect_oscillatory_events_np

    return {
        # --- Audio changepoint methods ---
        "meansquared_cp": FunctionSpec(
            func=voc.segment.meansquared,
            doc_url=DOC_URLS["meansquared_cp"],
            return_hint="segments",
            copy_preamble=_AUDIO_PREAMBLE,
            import_path="voc.segment.meansquared",
        ),
        "ava_cp": FunctionSpec(
            func=voc.segment.ava,
            doc_url=DOC_URLS["ava_cp"],
            return_hint="segments",
            copy_preamble=_AUDIO_PREAMBLE,
            import_path="voc.segment.ava",
        ),
        "vocalseg_cp": FunctionSpec(
            func=dynamic_threshold_segmentation,
            doc_url=DOC_URLS["vocalseg_cp"],
            return_hint="results",
            copy_preamble=_VOCALSEG_PREAMBLE,
        ),
        "continuity_cp": FunctionSpec(
            func=continuity_segmentation,
            doc_url=DOC_URLS["continuity_cp"],
            return_hint="results",
            copy_preamble=_VOCALSEG_CONTINUITY_PREAMBLE,
        ),
        # --- Ruptures ---
        "ruptures_pelt": FunctionSpec(
            func=_ruptures_pelt,
            doc_url=DOC_URLS["ruptures_pelt"],
            return_hint="bkps",
            copy_preamble=_RUPTURES_PREAMBLE,
            import_path="rpt.Pelt",
            copy_template="algo = rpt.Pelt(model={model}, min_size={min_size}, jump={jump}).fit(data)\nbkps = algo.predict(pen={pen})",  # noqa: E501
        ),
        "ruptures_binseg": FunctionSpec(
            func=_ruptures_binseg,
            doc_url=DOC_URLS["ruptures_binseg"],
            return_hint="bkps",
            copy_preamble=_RUPTURES_PREAMBLE,
            import_path="rpt.Binseg",
            copy_template="algo = rpt.Binseg(model={model}, min_size={min_size}, jump={jump}).fit(data)\nbkps = algo.predict(n_bkps={n_bkps})  # or algo.predict(pen=<penalty>)",  # noqa: E501
        ),
        "ruptures_bottomup": FunctionSpec(
            func=_ruptures_bottomup,
            doc_url=DOC_URLS["ruptures_bottomup"],
            return_hint="bkps",
            copy_preamble=_RUPTURES_PREAMBLE,
            import_path="rpt.BottomUp",
            copy_template="algo = rpt.BottomUp(model={model}, min_size={min_size}, jump={jump}).fit(data)\nbkps = algo.predict(n_bkps={n_bkps})",  # noqa: E501
        ),
        "ruptures_window": FunctionSpec(
            func=_ruptures_window,
            doc_url=DOC_URLS["ruptures_window"],
            return_hint="bkps",
            copy_preamble=_RUPTURES_PREAMBLE,
            import_path="rpt.Window",
            copy_template="algo = rpt.Window(width={width}, model={model}, min_size={min_size}, jump={jump}).fit(data)\nbkps = algo.predict(n_bkps={n_bkps})",  # noqa: E501
        ),
        "ruptures_dynp": FunctionSpec(
            func=_ruptures_dynp,
            doc_url=DOC_URLS["ruptures_dynp"],
            return_hint="bkps",
            copy_preamble=_RUPTURES_PREAMBLE,
            import_path="rpt.Dynp",
            copy_template="algo = rpt.Dynp(model={model}, min_size={min_size}, jump={jump}).fit(data)\nbkps = algo.predict(n_bkps={n_bkps})",  # noqa: E501
        ),
        # --- Energy envelopes ---
        "energy_lowpass": FunctionSpec(
            func=lowpass_envelope,
            doc_url=DOC_URLS["energy_lowpass"],
            return_hint="env_time, envelope",
            copy_preamble=_ENERGY_AUDIO_PREAMBLE,
        ),
        "energy_highpass": FunctionSpec(
            func=highpass_envelope,
            doc_url=DOC_URLS["energy_highpass"],
            return_hint="env_time, envelope",
            copy_preamble=_ENERGY_AUDIO_PREAMBLE,
        ),
        "energy_band": FunctionSpec(
            func=bandpass_envelope,
            doc_url=DOC_URLS["energy_band"],
            return_hint="env_time, envelope",
            copy_preamble=_ENERGY_AUDIO_PREAMBLE,
        ),
        "energy_meansquared": FunctionSpec(
            func=voc.signal.energy.meansquared,
            doc_url=DOC_URLS["energy_meansquared"],
            return_hint="env_time, envelope",
            copy_preamble=_ENERGY_VOCALPY_PREAMBLE,
        ),
        "energy_ava": FunctionSpec(
            func=voc.signal.energy.ava,
            doc_url=DOC_URLS["energy_ava"],
            return_hint="env_time, envelope",
            copy_preamble=_ENERGY_VOCALPY_PREAMBLE,
        ),
        # --- Kinematic changepoints ---
        "find_troughs": FunctionSpec(
            func=find_troughs_binary,
            doc_url=DOC_URLS["find_troughs"],
            return_hint="binary_mask",
            copy_preamble=_KINEMATIC_PREAMBLE,
        ),
        "find_turning_points": FunctionSpec(
            func=find_nearest_turning_points_binary,
            doc_url=DOC_URLS["find_turning_points"],
            return_hint="binary_mask",
            copy_preamble=_KINEMATIC_PREAMBLE,
        ),
        # --- Oscillatory event detection ---
        "oscillatory_events": FunctionSpec(
            func=detect_oscillatory_events_np,
            doc_url=DOC_URLS["oscillatory_events"],
            return_hint="onsets, offsets",
            copy_preamble=_OSCILLATORY_PREAMBLE,
            import_path="detect_oscillatory_events_np",
        ),
    }


_REGISTRY_CACHE: dict[str, FunctionSpec] | None = None


def get_registry() -> dict[str, FunctionSpec]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _build_registry()
    return _REGISTRY_CACHE


# ---------------------------------------------------------------------------
# Syntax highlighter
# ---------------------------------------------------------------------------


class PythonParamHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules: list[tuple[QRegularExpression, QTextCharFormat]] = []

        fmt_auto = QTextCharFormat()
        fmt_auto.setForeground(QColor(140, 140, 140))
        self._rules.append((QRegularExpression(r"<\w+>"), fmt_auto))

        fmt_keyword = QTextCharFormat()
        fmt_keyword.setForeground(QColor(80, 120, 200))
        fmt_keyword.setFontWeight(QFont.Bold)
        self._rules.append((QRegularExpression(r"\b(True|False|None)\b"), fmt_keyword))

        fmt_number = QTextCharFormat()
        fmt_number.setForeground(QColor(0, 150, 150))
        self._rules.append((QRegularExpression(r"\b-?\d+\.?\d*([eE][+-]?\d+)?\b"), fmt_number))

        fmt_string = QTextCharFormat()
        fmt_string.setForeground(QColor(50, 140, 50))
        self._rules.append((QRegularExpression(r"""("[^"]*"|'[^']*')"""), fmt_string))

        fmt_param = QTextCharFormat()
        fmt_param.setForeground(QColor(150, 80, 150))
        self._rules.append((QRegularExpression(r"\b\w+(?=\s*=)"), fmt_param))

        fmt_comment = QTextCharFormat()
        fmt_comment.setForeground(QColor(117, 113, 94))
        fmt_comment.setFontItalic(True)
        self._rules.append((QRegularExpression(r"#.*$"), fmt_comment))

    def highlightBlock(self, text):
        for pattern, fmt in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


class ProtectedLineEditor(QPlainTextEdit):
    """QPlainTextEdit that prevents editing on specified line numbers."""

    def __init__(self, protected_lines: set[int], parent=None):
        super().__init__(parent)
        self._protected = protected_lines

    def _is_protected(self, block_number: int) -> bool:
        return block_number in self._protected

    def _touches_protected(self) -> bool:
        cursor = self.textCursor()
        if cursor.hasSelection():
            start = self.document().findBlock(cursor.selectionStart()).blockNumber()
            end = self.document().findBlock(cursor.selectionEnd()).blockNumber()
            return any(self._is_protected(i) for i in range(start, end + 1))
        return self._is_protected(cursor.block().blockNumber())

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        nav_keys = {
            Qt.Key_Left,
            Qt.Key_Right,
            Qt.Key_Up,
            Qt.Key_Down,
            Qt.Key_Home,
            Qt.Key_End,
            Qt.Key_PageUp,
            Qt.Key_PageDown,
        }
        if key in nav_keys:
            super().keyPressEvent(event)
            return

        if mods == Qt.ControlModifier and key in (Qt.Key_C, Qt.Key_A):
            super().keyPressEvent(event)
            return

        cursor = self.textCursor()
        line = cursor.block().blockNumber()

        if key == Qt.Key_Backspace and cursor.atBlockStart():
            if line > 0 and self._is_protected(line - 1):
                return

        if key == Qt.Key_Delete and cursor.atBlockEnd():
            if self._is_protected(line + 1):
                return

        if self._touches_protected():
            return

        super().keyPressEvent(event)

    def insertFromMimeData(self, source):
        if self._touches_protected():
            return
        super().insertFromMimeData(source)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KNOWN_EDITORS = [
    ("VS Code", ["code"]),
    ("Cursor", ["cursor"]),
    ("PyCharm", ["pycharm"]),
    ("Sublime Text", ["subl"]),
    ("Notepad++", ["notepad++"]),
    ("Vim", ["vim"]),
    ("Nano", ["nano"]),
    ("Emacs", ["emacs"]),
]

_chosen_editor: str | None = None


def _detect_editors() -> list[tuple[str, str]]:
    import shutil

    found = []
    for label, cmds in _KNOWN_EDITORS:
        for cmd in cmds:
            path = shutil.which(cmd)
            if path:
                found.append((label, path))
                break
    return found


def _open_with_editor(editor_path: str, file_path: str) -> None:
    if sys.platform == "win32":
        subprocess.Popen([editor_path, file_path], creationflags=0x08)
    else:
        subprocess.Popen([editor_path, file_path])


def _open_in_default_editor(file_path: str) -> None:
    if sys.platform == "win32":
        os.startfile(file_path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", file_path])
    else:
        subprocess.Popen(["xdg-open", file_path])


def open_source_file(file_path: str, parent=None) -> None:
    global _chosen_editor

    if _chosen_editor:
        _open_with_editor(_chosen_editor, file_path)
        return

    editors = _detect_editors()

    items = [name for name, _ in editors] + ["Browse..."]
    editor_paths = {name: path for name, path in editors}

    choice, ok = QInputDialog.getItem(
        parent,
        "Open source with...",
        "Select editor:",
        items,
        0,
        False,
    )
    if not ok:
        return

    if choice == "Browse...":
        exe_filter = "Executables (*.exe)" if sys.platform == "win32" else "All files (*)"
        path, _ = QFileDialog.getOpenFileName(parent, "Select editor", "", exe_filter)
        if path:
            _chosen_editor = path
            _open_with_editor(path, file_path)
    else:
        path = editor_paths[choice]
        _chosen_editor = path
        _open_with_editor(path, file_path)


def _do_open_source(file_path: str, parent=None) -> None:
    global _chosen_editor
    if _chosen_editor == "__system__":
        _open_in_default_editor(file_path)
    elif _chosen_editor:
        _open_with_editor(_chosen_editor, file_path)
    else:
        open_source_file(file_path, parent)


def _display_name(spec: FunctionSpec) -> str:
    return spec.import_path or spec.func.__name__


def _get_param_infos(spec: FunctionSpec) -> list[ParamInfo]:
    if spec.params is not None:
        return spec.params

    sig = inspect.signature(spec.func)
    infos = []
    for name, p in sig.parameters.items():
        if name in spec.auto_params or name in spec.fixed_params:
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is p.empty:
            continue
        formatted = _format_default(p.default)
        try:
            ast.literal_eval(formatted)
        except (ValueError, SyntaxError):
            continue
        type_hint = p.annotation if p.annotation is not p.empty else "Any"
        infos.append(ParamInfo(name, type_hint, p.default))
    return infos


def _format_default(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return repr(value)
    return str(value)


def _build_editor_text(spec: FunctionSpec, param_infos: list[ParamInfo], current_params: dict) -> str:
    lines = [f"{_display_name(spec)}("]
    for auto in spec.auto_params:
        lines.append(f"    <{auto}>,  # set by GUI")
    for name, val in spec.fixed_params.items():
        lines.append(f"    {name}={_format_default(val)},  # only supported")
    for pi in param_infos:
        val = current_params.get(pi.name, pi.default)
        lines.append(f"    {pi.name}={_format_default(val)},")
    lines.append(")")
    return "\n".join(lines)


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _rst_inline_to_html(text: str) -> str:
    def _math_repl(m):
        expr = m.group(1)
        expr = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", expr)
        return f'<i><font color="#8250df">{expr}</font></i>'

    text = re.sub(r":math:`([^`]+)`", _math_repl, text)
    text = re.sub(r":class:`([^`]+)`", r'<font color="#0550ae">\1</font>', text)
    text = re.sub(r"``([^`]+)``", r'<font color="#0550ae">\1</font>', text)
    text = re.sub(r"\[(\d+)\]_", r"[\1]", text)
    return text


def _build_help_html(spec: FunctionSpec, font_size: int = 16) -> str:
    target = spec.func
    name = _display_name(spec)

    # --- Build signature text ---
    try:
        sig = inspect.signature(target)
        param_strs: list[str] = []
        for pname, p in sig.parameters.items():
            s = pname
            if p.annotation is not p.empty:
                ann = p.annotation.__name__ if isinstance(p.annotation, type) else str(p.annotation)
                s += f": {ann}"
            if p.default is not p.empty:
                s += f" = {p.default!r}"
            param_strs.append(s)
        ret_ann = ""
        if sig.return_annotation is not sig.empty:
            ra = sig.return_annotation
            ret_ann = f" -> {ra.__name__}" if isinstance(ra, type) else f" -> {ra}"
        if len(param_strs) > 2:
            params_text = ",\n    ".join(param_strs)
            sig_text = f"def {name}(\n    {params_text},\n){ret_ann}:"
        else:
            sig_text = f"def {name}({', '.join(param_strs)}){ret_ann}:"
    except (ValueError, TypeError):
        sig_text = f"def {name}(...):"

    # --- Highlight signature (use <b>/<font> for QTextBrowser compat) ---
    sig_escaped = _html_escape(sig_text)
    # String literals FIRST (before any tags with quotes are inserted)
    sig_html = re.sub(
        r'("(?:[^"\\]|\\.)*?")',
        r'<font color="#0a3069">\1</font>',
        sig_escaped,
    )
    sig_html = sig_html.replace(
        "def ",
        '<b><font color="#cf222e">def</font></b> ',
        1,
    )
    sig_html = re.sub(
        r": ([A-Za-z_][\w|.\[\] ]*?)( =|,|\n|\))",
        r': <b><font color="#0550ae">\1</font></b>\2',
        sig_html,
    )
    sig_html = re.sub(
        r"(-&gt;) (.+?)(:)$",
        r'\1 <b><font color="#0550ae">\2</font></b>\3',
        sig_html,
        flags=re.MULTILINE,
    )
    for kw in ("True", "False", "None"):
        sig_html = re.sub(
            rf"\b{kw}\b",
            f'<b><font color="#cf222e">{kw}</font></b>',
            sig_html,
        )

    # --- Format docstring ---
    docstring = inspect.getdoc(target) or ""
    doc_html = ""
    if docstring:
        doc_lines = docstring.splitlines()
        first_line = doc_lines[0] if doc_lines else ""
        rest_lines = doc_lines[1:]
        indented_rest = []
        for line in rest_lines:
            if line.strip():
                indented_rest.append(f"    {line}")
            else:
                indented_rest.append("")
        if indented_rest:
            doc_text = first_line + "\n" + "\n".join(indented_rest)
        else:
            doc_text = first_line

        doc_escaped = _html_escape(doc_text)

        # Section headers (Parameters, Returns, Args, etc.)
        doc_escaped = re.sub(
            r"^(    )(Parameters|Returns|Raises|Yields|Notes|Examples|"
            r"References|See Also|Example|Args)",
            r'\1<b><font color="#cf222e">\2</font></b>',
            doc_escaped,
            flags=re.MULTILINE,
        )
        # NumPy-style params: "    name : type"
        doc_escaped = re.sub(
            r"^(    )(\w+)( : ?)(.*)$",
            r'\1<b><font color="#953800">\2</font></b>'
            r'\3<b><font color="#0550ae">\4</font></b>',
            doc_escaped,
            flags=re.MULTILINE,
        )
        # Google-style params: "        name: desc"
        doc_escaped = re.sub(
            r"^(        )(\w+)(: )",
            r'\1<b><font color="#953800">\2</font></b>\3',
            doc_escaped,
            flags=re.MULTILINE,
        )
        # RST inline markup
        doc_escaped = _rst_inline_to_html(doc_escaped)

        doc_html = f'\n    <font color="#0a3069">"""{doc_escaped}\n    """</font>'

    # --- Assemble (avoid <pre>: QTextBrowser ignores tags inside it) ---
    code_block = f"{sig_html}{doc_html}"
    lines = code_block.split("\n")
    html_lines = []
    for line in lines:
        content = line.lstrip(" ")
        indent = len(line) - len(content)
        html_lines.append("&nbsp;" * indent + content)
    code_as_div = "<br>".join(html_lines)

    return (
        f'<html><body style="font-family:Consolas,Monaco,monospace; font-size:{font_size}px; '
        f'line-height:1.4; margin:0; background:#fff; color:#000;">'
        f'<div style="background:#f6f8fa; padding:16px; margin:12px; '
        f'border:1px solid #d0d7de;">'
        f"{code_as_div}</div>"
        f"</body></html>"
    )


def _parse_editor_text(
    text: str,
    param_infos: list[ParamInfo],
    auto_params: tuple[str, ...],
    skip_params: set[str] | None = None,
) -> dict:
    params = {}
    skip = set(auto_params) | (skip_params or set())
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("(") or line.startswith(")"):
            continue
        if line.startswith("<") and ">" in line:
            continue
        if "=" not in line:
            continue

        # Strip inline comments
        if "  #" in line:
            line = line[: line.index("  #")].rstrip().rstrip(",")

        # Could be part of the function call header
        if "(" in line:
            # Might be "func_name(" — skip
            if line.endswith("("):
                continue
            # Might be "func_name(\n  param=val" — extract after (
            idx = line.index("(")
            line = line[idx + 1 :].strip().rstrip(",")
            if not line or "=" not in line:
                continue

        name, _, val_str = line.partition("=")
        name = name.strip()
        val_str = val_str.strip()

        if name in skip:
            continue

        params[name] = val_str
    return params


def _validate_and_convert(raw_params: dict, param_infos: list[ParamInfo]) -> tuple[dict, str | None]:
    info_map = {pi.name: pi for pi in param_infos}
    result = {}

    for name, val_str in raw_params.items():
        try:
            value = ast.literal_eval(val_str)
        except (ValueError, SyntaxError):
            return {}, f"Invalid data type: Cannot parse value for '{name}': {val_str}"

        if name in info_map:
            pi = info_map[name]
            th = pi.type_hint
            if isinstance(th, type) and not isinstance(value, (th, type(None))):
                if th is float and isinstance(value, int):
                    value = float(value)
                elif th is int and isinstance(value, float) and value == int(value):
                    value = int(value)
                else:
                    expected = th.__name__
                    return (
                        {},
                        f"'{name}' expects {expected}, got {type(value).__name__}: {value}",
                    )

        result[name] = value

    return result, None


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class FunctionParamsDialog(QDialog):
    def __init__(self, registry_key: str, current_params: dict | None = None, parent=None):
        super().__init__(parent)
        self._registry_key = registry_key
        self._spec = get_registry()[registry_key]
        self._param_infos = _get_param_infos(self._spec)
        self._current_params = dict(current_params) if current_params else {}
        self._result_params: dict | None = None

        self.setWindowTitle(f"Configure: {_display_name(self._spec)}")
        self.setMinimumSize(1000, 650)
        self.resize(1300, 800)

        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left: function documentation
        self._doc_browser = QTextBrowser()
        self._doc_browser.setHtml(_build_help_html(self._spec, font_size=18))
        self._doc_browser.setOpenExternalLinks(True)
        splitter.addWidget(self._doc_browser)

        # Right: editor (header, auto_params, fixed_params, and closing paren are read-only)
        n_auto = len(self._spec.auto_params)
        n_fixed = len(self._spec.fixed_params)
        n_params = len(self._param_infos)
        last_line = 1 + n_auto + n_fixed + n_params
        protected = {0, *range(1, 1 + n_auto + n_fixed), last_line}

        editor_font = QFont("Consolas")
        editor_font.setStyleHint(QFont.Monospace)
        self._editor = ProtectedLineEditor(protected)
        self._editor.setFont(editor_font)
        self._highlighter = PythonParamHighlighter(self._editor.document())
        self._editor.setStyleSheet("QPlainTextEdit { font-family: 'Consolas'; font-size: 18pt; }")
        editor_text = _build_editor_text(
            self._spec,
            self._param_infos,
            self._current_params,
        )
        self._editor.setPlainText(editor_text)
        splitter.addWidget(self._editor)

        splitter.setSizes([700, 500])

        # Buttons: ...stretch...  [Docs]  [Copy code]  [Apply]
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        try:
            source_file = inspect.getfile(self._spec.func)
        except (TypeError, OSError):
            source_file = None

        if source_file:
            source_btn = QPushButton("Open source code in python editor")
            source_btn.setToolTip(source_file)
            source_btn.setStyleSheet(
                "QPushButton { color: #4da6ff; border: none; font-weight: bold; }"
                "QPushButton:hover { color: #80bfff; text-decoration: underline; }"
            )
            source_btn.setCursor(Qt.PointingHandCursor)
            source_btn.clicked.connect(lambda: _do_open_source(source_file, self))
            btn_layout.addWidget(source_btn)

        if self._spec.doc_url:
            import webbrowser

            docs_btn = QPushButton("Documentation")
            docs_btn.setToolTip(self._spec.doc_url)
            docs_btn.setStyleSheet(
                "QPushButton { color: #4da6ff; border: none; font-weight: bold; }"
                "QPushButton:hover { color: #80bfff; text-decoration: underline; }"
            )
            docs_btn.setCursor(Qt.PointingHandCursor)
            docs_btn.clicked.connect(lambda: webbrowser.open(self._spec.doc_url))
            btn_layout.addWidget(docs_btn)

        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(reset_btn)

        copy_btn = QPushButton("Copy code to clipboard")
        copy_btn.clicked.connect(self._on_copy)
        btn_layout.addWidget(copy_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(apply_btn)

        layout.addLayout(btn_layout)

    def _on_reset(self):
        default_params = {pi.name: pi.default for pi in self._param_infos}
        editor_text = _build_editor_text(self._spec, self._param_infos, default_params)
        self._editor.setPlainText(editor_text)

    def _on_apply(self):
        raw = _parse_editor_text(
            self._editor.toPlainText(),
            self._param_infos,
            self._spec.auto_params,
            skip_params=set(self._spec.fixed_params),
        )
        result, error = _validate_and_convert(raw, self._param_infos)
        if error:
            notify_dialog(error, "warning", "Validation Error", self)
            return
        result.update(self._spec.fixed_params)
        self._result_params = result
        self.accept()

    def _on_copy(self):
        raw = _parse_editor_text(
            self._editor.toPlainText(),
            self._param_infos,
            self._spec.auto_params,
            skip_params=set(self._spec.fixed_params),
        )
        result, error = _validate_and_convert(raw, self._param_infos)
        if error:
            notify_dialog(error, "warning", "Validation Error", self)
            return

        # Merge defaults, fixed params, and user values for template formatting
        all_values = {pi.name: pi.default for pi in self._param_infos}
        all_values.update(self._spec.fixed_params)
        all_values.update(result)

        if self._spec.copy_template:
            format_args = {k: _format_default(v) for k, v in all_values.items()}
            call = self._spec.copy_template.format(**format_args)
        else:
            # Build minimal call with only non-default values + fixed params
            info_map = {pi.name: pi for pi in self._param_infos}
            changed = {k: v for k, v in result.items() if k not in info_map or v != info_map[k].default}
            changed.update(self._spec.fixed_params)

            auto_str = ", ".join(self._spec.auto_params)
            if changed:
                param_lines = [f"    {auto_str},"]
                for k, v in changed.items():
                    param_lines.append(f"    {k}={_format_default(v)},")
                body = "\n".join(param_lines)
                fname = _display_name(self._spec)
                call = f"{self._spec.return_hint} = {fname}(\n{body}\n)"
            else:
                fname = _display_name(self._spec)
                call = f"{self._spec.return_hint} = {fname}({auto_str})"

        preamble = self._spec.copy_preamble
        if preamble and "{func}" in preamble:
            preamble = preamble.format(func=self._spec.func.__name__)
        snippet = preamble + "\n" + call if preamble else call

        clipboard = QApplication.clipboard()
        clipboard.setText(snippet)

    def get_params(self) -> dict | None:
        return self._result_params


# ---------------------------------------------------------------------------
# Convenience opener
# ---------------------------------------------------------------------------


def open_function_params_dialog(
    registry_key: str,
    app_state,
    parent=None,
) -> dict | None:
    cache = getattr(app_state, "function_params_cache", None) or {}
    current = cache.get(registry_key, {})

    dialog = FunctionParamsDialog(registry_key, current, parent)
    if dialog.exec_() and dialog.get_params() is not None:
        result = dialog.get_params()
        cache[registry_key] = result
        app_state.function_params_cache = cache
        return result
    return None
