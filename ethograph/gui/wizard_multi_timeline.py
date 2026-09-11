"""Temporal alignment timeline visualization for the NC wizard (Page 4)."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import QRectF, QRegularExpression, Qt
from qtpy.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainterPath,
    QPen,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsPathItem,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.dialog_function_params import _do_open_source
from ethograph.gui.wizard_media_files import extract_file_row
from ethograph.gui.wizard_multi_codegen import generate_alignment_code
from ethograph.gui.wizard_state import ModalityConfig, WizardState
from ethograph.utils.paths import defaults_dir

logger = logging.getLogger(__name__)


class TimeAxis(pg.AxisItem):
    """AxisItem that formats tick labels based on total duration and value.

    Rules:
    - If total duration > 1 hour  → HH:MM:SS
    - Else if value > 5 minutes   → M:SS
    - Else                        → seconds
    """

    def tickStrings(self, values, scale, spacing):
        out = []

        # Get total visible duration
        try:
            min_val, max_val = self.range
            duration = max_val - min_val
        except Exception:
            duration = 0

        for v in values:
            try:
                s = float(v)
            except Exception:
                out.append("")
                continue

            # 🔑 Rule 1: long recordings → always HH:MM:SS
            if duration > 3600:
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                sec = int(s % 60)
                out.append(f"{h:d}:{m:02d}:{sec:02d}")

            # Rule 2: medium values → M:SS
            elif s > 300:
                m = int(s // 60)
                sec = int(s % 60)
                out.append(f"{m:d}:{sec:02d}")

            # Rule 3: short values → seconds
            else:
                if spacing < 1:
                    out.append(f"{s:.2f}")
                elif spacing < 5:
                    out.append(f"{s:.1f}")
                else:
                    out.append(f"{int(round(s))}")

        return out


# Colors per modality (matching dialog_media_files.py palette)
MODALITY_COLORS = {
    "video": "#50c8b4",
    "pose": "#e8737a",
    "audio": "#e8c75a",
    "ephys": "#b07ae8",
    "features": "#7ab0e8",
}


class PythonCodeHighlighter(QSyntaxHighlighter):
    """Simple Python syntax highlighter for the code preview panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules: list[tuple[QRegularExpression, QTextCharFormat]] = []

        keyword_fmt = QTextCharFormat()
        keyword_fmt.setForeground(QColor("#c586c0"))
        for kw in [
            "and",
            "as",
            "assert",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "False",
            "finally",
            "for",
            "from",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "None",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "True",
            "try",
            "while",
            "with",
            "yield",
        ]:
            self._rules.append((QRegularExpression(rf"\\b{kw}\\b"), keyword_fmt))

        number_fmt = QTextCharFormat()
        number_fmt.setForeground(QColor("#b5cea8"))
        self._rules.append((QRegularExpression(r"\b-?\d+(\.\d+)?([eE][+-]?\d+)?\b"), number_fmt))

        string_fmt = QTextCharFormat()
        string_fmt.setForeground(QColor("#ce9178"))
        self._rules.append((QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'), string_fmt))
        self._rules.append((QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), string_fmt))

        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor("#6a9955"))
        self._rules.append((QRegularExpression(r"#.*$"), comment_fmt))

        func_fmt = QTextCharFormat()
        func_fmt.setForeground(QColor("#dcdcaa"))
        self._rules.append((QRegularExpression(r"\b[A-Za-z_][A-Za-z0-9_]*(?=\()"), func_fmt))

    def highlightBlock(self, text: str):
        for pattern, fmt in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


# ---------------------------------------------------------------------------
# Notebook conversion helper
# ---------------------------------------------------------------------------


def _code_to_notebook(code: str) -> dict:
    """Convert Python code with section markers to Jupyter notebook format."""
    import re

    section_pattern = r"^# ─── \d+\. .+ ───$"
    lines = code.split("\n")

    cells = []
    current_cell_lines = []

    for line in lines:
        if re.match(section_pattern, line):
            if current_cell_lines:
                cell_code = "\n".join(current_cell_lines).strip()
                if cell_code:
                    cells.append(
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": cell_code.split("\n"),
                        }
                    )
            current_cell_lines = [line]
        else:
            current_cell_lines.append(line)

    if current_cell_lines:
        cell_code = "\n".join(current_cell_lines).strip()
        if cell_code:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": cell_code.split("\n"),
                }
            )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    return notebook


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _make_rounded_bar(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    brush: QBrush,
    pen: QPen,
    radius: float = 0.08,
) -> QGraphicsPathItem:
    path = QPainterPath()
    rect = QRectF(x0, y0, x1 - x0, y1 - y0)
    path.addRoundedRect(rect, radius, radius)
    item = QGraphicsPathItem(path)
    item.setBrush(brush)
    item.setPen(pen)
    return item


def _collect_ephys_rows(
    nwb_alignment,
    app_state=None,
) -> list[tuple[str, float, float]]:
    """Collect ephys timeline spans from NWB ElectricalSeries or app_state paths.

    Returns list of (label, t_start, t_end) tuples — one per ephys source.
    """
    from ethograph.io.nwb_alignment import NWBAlignment
    from ethograph.utils.stream_durations import (
        get_ephys_duration,
        get_kilosort_duration,
    )

    rows: list[tuple[str, float, float]] = []
    sio = nwb_alignment

    # 1. NWB ElectricalSeries
    if isinstance(sio, NWBAlignment):
        for info in sio.electrical_series():
            rate = info.get("rate")
            name = info.get("name", "ephys")
            if rate and rate > 0:
                try:
                    acq = sio.nwb.acquisition[name]
                    n_samples = acq.data.shape[0]
                    starting_time = float(acq.starting_time) if getattr(acq, "starting_time", None) is not None else 0.0
                    dur = n_samples / rate
                    rows.append((f"ephys: {name}", starting_time, starting_time + dur))
                except Exception:
                    pass

    if rows:
        return rows

    # 2. Fall back to app_state ephys_path / neurons_path
    if app_state is None:
        return rows

    ephys_path = getattr(app_state, "ephys_path", None)
    neurons_path = getattr(app_state, "neurons_path", None)

    dur = None
    label = "ephys"
    if ephys_path:
        dur = get_ephys_duration(ephys_path)
        label = f"ephys: {Path(ephys_path).name}"
    if dur is None and neurons_path:
        dur = get_kilosort_duration(neurons_path)
        label = f"ephys: {Path(neurons_path).name}"

    if dur is not None:
        rows.append((label, 0.0, dur))

    return rows


def draw_session_timeline(
    plot: pg.PlotWidget,
    nwb_alignment,
    items_out: list | None = None,
    extra_streams: list[str] | None = None,
    app_state=None,
) -> float:
    """Draw a timeline from NWB acquisition ImageSeries and ElectricalSeries.

    Each ImageSeries with ``external_file`` gets one row.  Ephys rows are
    drawn from ElectricalSeries or app_state paths.  Trial boundaries from
    the trials table are shown as dotted vertical lines.
    """
    from ethograph.io.nwb_alignment import NWBAlignment

    sio = nwb_alignment
    items: list = items_out if items_out is not None else []
    is_nwb = isinstance(sio, NWBAlignment) and sio.nwb and sio.nwb.acquisition

    # Collect acquisition names that have external files (video/audio/pose)
    acq_names = []
    if is_nwb:
        acq_names = [name for name, series in sio.nwb.acquisition.items() if getattr(series, "external_file", None)]

    # Collect ephys rows
    ephys_rows = _collect_ephys_rows(sio, app_state)

    if not acq_names and not ephys_rows:
        return 0.0

    # Build combined row list: media acquisitions + ephys
    all_row_names = list(acq_names) + [label for label, _, _ in ephys_rows]
    n_rows = len(all_row_names)
    rows_rev = list(reversed(all_row_names))
    y_ticks = [(i + 0.5, rows_rev[i]) for i in range(n_rows)]
    plot.getAxis("left").setTicks([y_ticks])
    plot.setYRange(-0.2, n_rows + 0.2)

    max_t = 0.0

    # Draw media acquisition bars
    for row_idx, row_name in enumerate(rows_rev):
        if row_name not in acq_names:
            continue
        acq_name = row_name
        stream = acq_name.split("_", 1)[0] if "_" in acq_name else acq_name
        device = acq_name.split("_", 1)[1] if "_" in acq_name else None

        color = pg.mkColor(MODALITY_COLORS.get(stream, "#888888"))
        color.setAlpha(160)
        bar_brush = pg.mkBrush(color)
        bar_pen = pg.mkPen(color.lighter(130), width=1)
        y_base = row_idx

        for _filepath, t_start, t_end in sio.file_time_spans(stream, device):
            bar = _make_rounded_bar(t_start, t_end, y_base + 0.3, y_base + 0.7, bar_brush, bar_pen)
            plot.addItem(bar)
            items.append(bar)
            max_t = max(max_t, t_end)

    # Draw ephys bars
    for ephys_label, t_start, t_end in ephys_rows:
        row_idx = rows_rev.index(ephys_label)
        color = pg.mkColor(MODALITY_COLORS.get("ephys", "#b07ae8"))
        color.setAlpha(160)
        bar_brush = pg.mkBrush(color)
        bar_pen = pg.mkPen(color.lighter(130), width=1)
        y_base = row_idx

        bar = _make_rounded_bar(t_start, t_end, y_base + 0.3, y_base + 0.7, bar_brush, bar_pen)
        plot.addItem(bar)
        items.append(bar)
        max_t = max(max_t, t_end)

    # Trial boundary lines
    trial_df = getattr(sio, "trials_df", pd.DataFrame())
    trials = trial_df["trial"].tolist() if not trial_df.empty and "trial" in trial_df.columns else []

    for trial_id in trials:
        t0 = sio.start_time(trial_id)
        line = pg.InfiniteLine(
            pos=t0,
            angle=90,
            pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DotLine),
        )
        plot.addItem(line)
        items.append(line)

        t1 = sio.stop_time(trial_id)
        label_x = (t0 + t1) / 2 if t1 is not None else t0
        lbl = pg.TextItem(str(trial_id), color="#aaaaaa", anchor=(0.5, 1.0))
        lbl.setPos(label_x, n_rows + 0.1)
        plot.addItem(lbl)
        items.append(lbl)

        if t1 is not None:
            max_t = max(max_t, t1)

    if max_t <= 0.0:
        return 0.0
    plot.setXRange(0, min(max_t, 120), padding=0.02)
    return max_t


def _compute_file_durations(state: WizardState) -> dict[str, dict[str, float]]:
    from ethograph.utils.stream_durations import (
        get_kilosort_duration,
        probe_duration,
    )

    durations: dict[str, dict[str, float]] = {}
    for name in ["video", "pose", "audio", "ephys"]:
        cfg: ModalityConfig = getattr(state, name)
        if not cfg.enabled:
            continue

        if cfg.pattern and cfg.pattern.files:
            files = cfg.pattern.files
        elif cfg.single_file_path:
            files = [Path(cfg.single_file_path)]
        elif name == "ephys" and cfg.neurons_path:
            dur = get_kilosort_duration(cfg.neurons_path)
            if dur is not None:
                durations[name] = {cfg.neurons_path: dur}
            continue
        else:
            continue

        durs: dict[str, float] = {}
        for f in files:
            fp = str(f)
            fps = cfg.fps if name == "pose" else None
            dur = probe_duration(fp, name, fps)
            if dur is not None:
                durs[fp] = dur

        # Ephys: fall back to kilosort folder if raw file probing failed
        if not durs and name == "ephys" and cfg.neurons_path:
            dur = get_kilosort_duration(cfg.neurons_path)
            if dur is not None:
                durs[cfg.neurons_path] = dur

        if durs:
            durations[name] = durs

    return durations


def _normalize_trial_key(value: object) -> object:
    """Normalize trial identifiers so numeric strings match integer trial IDs."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)
        return s
    return value


class TimelinePage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Step 4 — Temporal alignment</b>"))

        # Tab widget for visualization and code
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 8px 24px;
                min-width: 120px;
            }
        """)

        # Tab 1: Visualization (stacked: table for aligned, timeline for unaligned)
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)

        self._viz_stack = QStackedWidget()

        # --- Page 0: Aligned table view ---
        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        table_layout.addWidget(
            QLabel("All files are aligned to trials. Each row shows which files belong to each trial.")
        )
        self._aligned_table = QTableWidget()
        self._aligned_table.setStyleSheet(
            "QTableWidget { background-color: #1a1d21; color: #d4d4d4; gridline-color: #3e3e3e; }"
            "QHeaderView::section { background-color: #2d2d2d; color: #d4d4d4; padding: 4px; }"
        )
        self._aligned_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_layout.addWidget(self._aligned_table, stretch=1)
        self._viz_stack.addWidget(table_page)

        # --- Page 1: Timeline view ---
        timeline_page = QWidget()
        timeline_layout = QVBoxLayout(timeline_page)
        timeline_layout.addWidget(
            QLabel(
                "Review how your files align in time. "
                "Colored bars show file durations; dotted lines mark trial boundaries."
            )
        )
        timeline_layout.addSpacing(4)

        self._plot = pg.PlotWidget(axisItems={"bottom": TimeAxis(orientation="bottom")})
        self._plot.setBackground("#1a1d21")
        self._plot.showGrid(x=True, y=False, alpha=0.15)
        self._plot.setLabel("bottom", "Time")
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.getAxis("left").setTicks([])
        timeline_layout.addWidget(self._plot, stretch=1)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Pan:"))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._slider)
        timeline_layout.addLayout(slider_row)
        self._viz_stack.addWidget(timeline_page)

        viz_layout.addWidget(self._viz_stack, stretch=1)

        self._note_label = QLabel("")
        self._note_label.setWordWrap(True)
        self._note_label.setStyleSheet("font-size: 11px; color: #555; padding: 2px 4px;")
        self._note_label.hide()
        viz_layout.addWidget(self._note_label)

        self._tabs.addTab(viz_tab, "1: Visualization")

        # Tab 2: Python code
        code_tab = QWidget()
        code_layout = QVBoxLayout(code_tab)
        code_layout.addWidget(
            QLabel(
                "Executable Python code that reproduces your alignment setup. "
                "Copy this code to customize or debug your workflow."
            )
        )
        code_layout.addSpacing(4)

        self._code_editor = QPlainTextEdit()
        self._code_editor.setReadOnly(True)
        self._code_editor.setFont(QFont("Consolas", 10))
        self._code_editor.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #3e3e3e; }"
        )
        self._code_highlighter = PythonCodeHighlighter(self._code_editor.document())
        code_layout.addWidget(self._code_editor, stretch=1)

        code_btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy code to clipboard")
        copy_btn.clicked.connect(self._on_copy_code)
        code_btn_row.addWidget(copy_btn)

        editor_btn = QPushButton("Open in code editor")
        editor_btn.clicked.connect(self._on_open_in_editor)
        code_btn_row.addWidget(editor_btn)
        code_btn_row.addStretch()
        code_layout.addLayout(code_btn_row)

        self._tabs.addTab(code_tab, "2: Python code")

        layout.addWidget(self._tabs, stretch=1)

        layout.addSpacing(8)

        # Output path (shared across tabs)
        self._out_widget = QWidget()
        _out_row = QHBoxLayout(self._out_widget)
        _out_row.setContentsMargins(0, 0, 0, 0)
        _out_row.addWidget(QLabel("Output path:"))
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Select output location for session.nc...")
        self._output_edit.setReadOnly(True)
        out_browse = QPushButton("Browse")
        out_browse.clicked.connect(self._browse_output)
        _out_row.addWidget(self._output_edit)
        _out_row.addWidget(out_browse)
        layout.addWidget(self._out_widget)

        self._total_duration = 1.0
        self._items: list = []
        self._state: WizardState | None = None

    def populate_from_state(self, state: WizardState):
        self._state = state
        self._clear()
        self._regenerate_code()

        # Switch view mode: table for aligned, timeline for unaligned
        if state.files_aligned_to_trials:
            self._viz_stack.setCurrentIndex(0)
            self._populate_aligned_table(state)
            return
        self._viz_stack.setCurrentIndex(1)

        durations = _compute_file_durations(state)
        state.file_durations = durations

        enabled_modalities = [name for name in ["video", "pose", "audio", "ephys"] if getattr(state, name).enabled]

        # Build rows: each (label, modality_name, device_name_or_None)
        rows: list[tuple[str, str, str | None]] = []
        for name in enabled_modalities:
            devices = self._get_devices(name, state)
            if devices:
                for dev in devices:
                    rows.append((f"{name}: {dev}", name, dev))
            else:
                rows.append((name.capitalize(), name, None))

        n_rows = len(rows)
        rows_reversed = list(reversed(rows))
        y_ticks = [(i + 0.5, rows_reversed[i][0]) for i in range(n_rows)]
        self._plot.getAxis("left").setTicks([y_ticks])
        self._plot.setYRange(-0.2, n_rows + 0.2)

        # Trial start times by row index (for aligned_to_trial file_mode)
        trial_start_by_index: dict[int, float] = {}
        trial_index_by_key: dict[object, int] = {}
        if state.trial_table is not None and "trial" in state.trial_table.columns:
            for idx, tid in enumerate(state.trial_table["trial"].tolist()):
                trial_index_by_key[tid] = idx
                trial_index_by_key[str(tid)] = idx
                norm = _normalize_trial_key(tid)
                if norm is not None:
                    trial_index_by_key[norm] = idx
            if "start_time" in state.trial_table.columns:
                starts = pd.to_numeric(state.trial_table["start_time"], errors="coerce")
                for idx, t0 in enumerate(starts.tolist()):
                    if np.isfinite(t0):
                        trial_start_by_index[idx] = float(t0)

        # Map file path → device string for multi-device streams
        file_device_map: dict[str, dict[str, str]] = {}
        file_trial_index_map: dict[str, dict[str, int]] = {}
        for name in enabled_modalities:
            cfg: ModalityConfig = getattr(state, name)
            if not (cfg.pattern and cfg.pattern.files):
                continue
            dev_role = "mic" if name == "audio" else "camera"
            summary = cfg.pattern.summary()
            if dev_role not in summary:
                continue
            mapping: dict[str, str] = {}
            trial_mapping: dict[str, int] = {}
            for f in cfg.pattern.files:
                row_data = extract_file_row(
                    f,
                    cfg.pattern.segments,
                    cfg.pattern.tokenize_mode,
                    regex_pattern=cfg.pattern.regex_pattern,
                )
                fp = str(f)
                mapping[fp] = row_data.get(dev_role, "")
                trial_val = _normalize_trial_key(row_data.get("trial"))
                if trial_val is not None:
                    idx = trial_index_by_key.get(trial_val) or trial_index_by_key.get(str(trial_val))
                    if idx is not None:
                        trial_mapping[fp] = idx
            file_device_map[name] = mapping
            if trial_mapping:
                file_trial_index_map[name] = trial_mapping

        # Draw file bars
        max_time = 0.0
        for row_idx, (label, name, device) in enumerate(rows_reversed):
            cfg: ModalityConfig = getattr(state, name)
            color = pg.mkColor(MODALITY_COLORS.get(name, "#888888"))
            color.setAlpha(160)
            bar_pen = pg.mkPen(color.lighter(130), width=1)
            bar_brush = pg.mkBrush(color)
            y_base = row_idx
            offset = cfg.constant_offset

            mod_durs = durations.get(name, {})
            filtered = (
                {fp: dur for fp, dur in mod_durs.items() if file_device_map[name].get(fp) == device}
                if device is not None and name in file_device_map
                else mod_durs
            )

            cum = 0.0
            for filepath, dur in filtered.items():
                if not math.isfinite(dur):
                    continue
                if cfg.file_mode == "aligned_to_trial":
                    trial_idx = file_trial_index_map.get(name, {}).get(filepath)
                    x_start = (
                        offset + trial_start_by_index.get(trial_idx, cum) if trial_idx is not None else offset + cum
                    )
                else:
                    x_start = offset + cum
                cum += dur

                if not math.isfinite(x_start):
                    continue
                bar = _make_rounded_bar(
                    x_start,
                    x_start + dur,
                    y_base + 0.3,
                    y_base + 0.7,
                    bar_brush,
                    bar_pen,
                )
                self._plot.addItem(bar)
                self._items.append(bar)
                max_time = max(max_time, x_start + dur)

        # Draw trial boundaries from start_time column
        if state.trial_table is not None and "start_time" in state.trial_table.columns:
            for _, row in state.trial_table.iterrows():
                t0 = float(row["start_time"])
                if not math.isfinite(t0):
                    continue
                line = pg.InfiniteLine(
                    pos=t0,
                    angle=90,
                    pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DotLine),
                )
                self._plot.addItem(line)
                self._items.append(line)

                tid = row.get("trial", "")
                lbl = pg.TextItem(str(tid), color="#aaaaaa", anchor=(0.5, 1.0))
                lbl.setPos(t0, n_rows + 0.1)
                self._plot.addItem(lbl)
                self._items.append(lbl)

                if "stop_time" in state.trial_table.columns:
                    t1 = float(row["stop_time"])
                    if math.isfinite(t1):
                        max_time = max(max_time, t1)

        if max_time > 0.0 and math.isfinite(max_time):
            self._total_duration = max_time
            self._plot.setXRange(0, min(max_time, 120), padding=0.02)

    # ------------------------------------------------------------------
    # Aligned table view
    # ------------------------------------------------------------------

    def _populate_aligned_table(self, state: WizardState):
        """Build a table: rows=trials, columns=stream_device, cells=filenames."""
        if state.trial_table is None:
            return

        trial_ids = state.trial_table["trial"].tolist() if "trial" in state.trial_table.columns else []
        if not trial_ids:
            return

        # Collect stream_device columns from the trial table
        stream_cols = [
            c
            for c in state.trial_table.columns
            if c not in ("trial", "start_time", "stop_time") and not c.endswith("_start")
        ]

        self._aligned_table.setRowCount(len(trial_ids))
        self._aligned_table.setColumnCount(len(stream_cols) + 1)
        self._aligned_table.setHorizontalHeaderLabels(["Trial"] + stream_cols)

        for row, trial_id in enumerate(trial_ids):
            self._aligned_table.setItem(row, 0, QTableWidgetItem(str(trial_id)))
            for col_idx, col_name in enumerate(stream_cols):
                val = state.trial_table.iloc[row].get(col_name, "")
                cell = Path(str(val)).name if val else ""
                self._aligned_table.setItem(row, col_idx + 1, QTableWidgetItem(cell))

        self._aligned_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def _populate_aligned_table_from_alignment(self, nwb_alignment):
        """Build a table from TrialTree's NWB session trials_df."""
        sio = nwb_alignment
        df = sio.trials_df
        if df.empty:
            return

        trial_col = df["trial"].tolist() if "trial" in df.columns else list(range(1, len(df) + 1))

        stream_cols = [
            c for c in df.columns if c not in ("trial", "start_time", "stop_time") and not c.endswith("_start")
        ]

        self._aligned_table.setRowCount(len(trial_col))
        self._aligned_table.setColumnCount(len(stream_cols) + 1)
        self._aligned_table.setHorizontalHeaderLabels(["Trial"] + stream_cols)

        for row, trial_id in enumerate(trial_col):
            self._aligned_table.setItem(row, 0, QTableWidgetItem(str(trial_id)))
            for col_idx, col_name in enumerate(stream_cols):
                val = df.iloc[row].get(col_name, "")
                cell = Path(str(val)).name if val else ""
                self._aligned_table.setItem(row, col_idx + 1, QTableWidgetItem(cell))

        self._aligned_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def _get_devices(self, name: str, state: WizardState) -> list[str]:
        if name == "video" and state.camera_names:
            return state.camera_names
        if name == "audio" and state.mic_names:
            return state.mic_names
        cfg: ModalityConfig = getattr(state, name)
        if cfg.pattern:
            dev_role = "mic" if name == "audio" else "camera"
            summary = cfg.pattern.summary()
            if dev_role in summary:
                return summary[dev_role]
        return []

    def _clear(self):
        for item in self._items:
            self._plot.removeItem(item)
        self._items.clear()

    def _on_slider(self, value: int):
        frac = value / 1000.0
        window = min(self._total_duration, 120)
        center = frac * (self._total_duration - window) + window / 2
        self._plot.setXRange(center - window / 2, center + window / 2, padding=0)

    def collect_state(self, state: WizardState):
        state.output_path = self._output_edit.text()

    def _browse_output(self):
        result = QFileDialog.getSaveFileName(
            self,
            "Save dataset",
            "session.nc",
            "NetCDF files (*.nc);;All files (*)",
        )
        if result and result[0]:
            path = result[0]
            if not path.endswith(".nc"):
                path += ".nc"
            self._output_edit.setText(path)
            if self._state:
                self._state.output_path = path
                self._regenerate_code()

    def _regenerate_code(self):
        """Generate and display Python code for current state."""
        if self._state is None:
            return
        code = generate_alignment_code(self._state)
        self._code_editor.setPlainText(code)

    def _on_copy_code(self):
        """Copy generated code to clipboard."""
        code = self._code_editor.toPlainText()
        clipboard = QApplication.clipboard()
        clipboard.setText(code)

    def _on_open_in_editor(self):
        """Save the code as a notebook under the defaults folder and open it in the user's editor."""
        from datetime import datetime

        code = self._code_editor.toPlainText()
        if not code.strip():
            return

        notebook = _code_to_notebook(code)

        wizard_dir = defaults_dir("wizard")
        wizard_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = wizard_dir / f"ethograph_alignment_setup_{timestamp}.ipynb"

        output_file.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

        _do_open_source(str(output_file), self)

    # ------------------------------------------------------------------
    # Standalone (non-wizard) entry point
    # ------------------------------------------------------------------

    def configure_for_standalone(self):
        """Hide wizard-specific controls; call before populate_from_trialtree()."""
        self._out_widget.hide()

    def populate_from_trialtree(self, dt, app_state):
        """Populate from a loaded TrialTree's NWB session data.

        Aligned mode (table view): trials table has ``{stream}_{device}``
        filename columns.
        Unaligned mode (timeline view): filenames are in acquisition
        ImageSeries, trials table has only timing.
        """
        self._clear()
        self._state = None
        self._code_editor.setPlainText("# Open via the New Dataset Wizard to generate alignment code.")

        # Detect mode: timeline if start_time exists or no filename columns
        sio = getattr(app_state, "nwb_alignment", None)
        if sio is None:
            sio = getattr(dt, "nwb_alignment", None)
        if sio is None:
            from ethograph.io.nwb_alignment import EmpytAlignment

            sio = EmpytAlignment()
        df = sio.trials_df
        has_timing = not df.empty and "start_time" in df.columns
        _STREAM_PREFIXES = ("video_", "pose_", "audio_", "ephys_")
        has_filename_cols = not df.empty and any(col.startswith(_STREAM_PREFIXES) for col in df.columns)

        if has_filename_cols and not has_timing:
            self._viz_stack.setCurrentIndex(0)
            self._populate_aligned_table_from_alignment(sio)
        else:
            self._viz_stack.setCurrentIndex(1)
            self._total_duration = draw_session_timeline(
                self._plot,
                sio,
                items_out=self._items,
                app_state=app_state,
            )
