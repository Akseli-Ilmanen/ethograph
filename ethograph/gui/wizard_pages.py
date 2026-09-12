"""The Data wizard's pages: mode, sources, timing, write.

The wizard opens with one question, answered by picking one of three modes,
each shown with a figure:

1. **Pair my media files** — every file already starts with its trial; the
   user only says which file is which. Runs in the wizard.
2. **Align media to a recording system, free-running camera**
3. **Align media to a recording system, triggered camera**

Modes 2 and 3 need ``neuroconv``: the wizard writes one notebook per rig and
stops; the notebook computes the timing and writes ``session.nwb``.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.file_dialogs import browse_open_file, browse_save_file
from ethograph.gui.make_pretty import styled_link
from ethograph.gui.project import project_dir_of
from ethograph.gui.wizard_state import WizardState
from ethograph.utils.paths import defaults_dir

ASSETS = Path(__file__).parent / "assets" / "wizard"
FIGURES = {
    "free_running": ASSETS / "video_setup_free_running.png",
    "triggered": ASSETS / "video_setup_triggered.png",
}
NEUROCONV_HOWTO = "https://neuroconv.readthedocs.io/en/main/how_to/align_external_video.html"

MODE_TITLES = {
    "pair": "1 · Pair my media files",
    "free_running": "2 · Align media to a recording system, free-running camera",
    "triggered": "3 · Align media to a recording system, triggered camera",
}

#: neuroconv recording interfaces the timing page offers, by recording system.
RECORDING_INTERFACES: list[tuple[str, str, str]] = [
    ("Intan (.rhd / .rhs)", "IntanRecordingInterface", "Intan (*.rhd *.rhs)"),
    ("Open Ephys (.oebin)", "OpenEphysRecordingInterface", "Open Ephys (*.oebin)"),
    ("SpikeGLX (.bin + .meta)", "SpikeGLXRecordingInterface", "SpikeGLX (*.bin)"),
]

_FIGURE_WIDTH = 640


def _figure_label(key: str) -> QLabel:
    label = QLabel()
    pix = QPixmap(str(FIGURES[key]))
    label.setPixmap(pix.scaledToWidth(_FIGURE_WIDTH, Qt.SmoothTransformation))
    label.setAlignment(Qt.AlignLeft)
    return label


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #9aa0ac;")
    return label


# ─── Page 0: mode ─────────────────────────────────────────────────────────────


class _ModeBlock(QFrame):
    """One selectable mode: radio title, description, figure."""

    clicked = Signal(str)

    def __init__(self, key: str, description: str, figure: QWidget, group: QButtonGroup, parent=None):
        super().__init__(parent)
        self.key = key
        self.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(self)
        self.radio = QRadioButton(MODE_TITLES[key])
        self.radio.setStyleSheet("font-weight: bold;")
        group.addButton(self.radio)
        lay.addWidget(self.radio)
        lay.addWidget(_muted(description))
        lay.addWidget(figure)

    def mousePressEvent(self, event):
        self.radio.setChecked(True)
        super().mousePressEvent(event)


def _pairing_table_figure() -> QTableWidget:
    cols = ["trial", "video_cam-1", "video_cam-2", "pose_cam-1", "audio_mic-1"]
    rows = [
        ["1", "cam1_trial001.mp4", "cam2_trial001.mp4", "dlc_cam1_trial001.h5", "mic1_trial001.wav"],
        ["2", "cam1_trial002.mp4", "cam2_trial002.mp4", "dlc_cam1_trial002.h5", "mic1_trial002.wav"],
        ["3", "cam1_trial003.mp4", "cam2_trial003.mp4", "dlc_cam1_trial003.h5", "mic1_trial003.wav"],
    ]
    table = QTableWidget(len(rows), len(cols))
    table.setHorizontalHeaderLabels(cols)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            item = QTableWidgetItem(value)
            item.setFlags(Qt.ItemIsEnabled)
            table.setItem(r, c, item)
    table.verticalHeader().hide()
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.setFocusPolicy(Qt.NoFocus)
    table.setSelectionMode(QTableWidget.NoSelection)
    table.setMaximumHeight(table.horizontalHeader().height() + 3 * table.rowHeight(0) + 6)
    table.setMaximumWidth(_FIGURE_WIDTH)
    return table


class ModePage(QWidget):
    """Page 0: one long page, three modes, one figure each, plus the imports."""

    import_requested = Signal(str)  # "dandi" | "boris" | "nwb"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        lay.addWidget(QLabel("<b>🧙 Data wizard</b>"))
        intro = _muted(
            "You have media files (video, audio, pose), possibly from several cameras or "
            "microphones. If they are already aligned in time, choose <b>Pair my media files</b>. "
            "Otherwise choose one of the two below; they align your media to a recording "
            "system's clock following "
            + styled_link(NEUROCONV_HOWTO, "neuroconv's guide to aligning behaviour video")
            + "."
        )
        intro.setOpenExternalLinks(True)
        lay.addWidget(intro)

        self._group = QButtonGroup(self)
        self._blocks: dict[str, _ModeBlock] = {}
        specs = [
            (
                "pair",
                "Every file already starts with its trial. You only say which file is which: a table "
                "with one row per trial and one column per camera or microphone. Ephys is not offered "
                "here; it lives on a recording system's clock, which is modes 2 and 3.",
                _pairing_table_figure(),
            ),
            (
                "free_running",
                "The camera ran for the whole session on its own clock. The recording system (Intan, "
                "Open Ephys, SpikeGLX) knows about it through a known offset or a pulse per frame; the "
                "recorder may have split the video into several files.",
                _figure_label("free_running"),
            ),
            (
                "triggered",
                "One file per trial with real gaps between them. The recording system triggered each "
                "file, so it knows the trial onsets, and may also have a pulse per frame.",
                _figure_label("triggered"),
            ),
        ]
        for key, description, figure in specs:
            block = _ModeBlock(key, description, figure, self._group)
            self._blocks[key] = block
            lay.addWidget(block)
        self._blocks["pair"].radio.setChecked(True)

        imports = QHBoxLayout()
        imports.addWidget(_muted("Also:"))
        for label, key in [
            ("Download from DANDI…", "dandi"),
            ("Import a BORIS project…", "boris"),
            ("Open an .nwb I already have…", "nwb"),
        ]:
            btn = QPushButton(label)
            btn.setFlat(True)
            btn.setAutoDefault(False)
            if key == "boris":
                btn.setEnabled(False)
                btn.setToolTip("BORIS import is not available yet.")
            btn.clicked.connect(lambda _=False, k=key: self.import_requested.emit(k))
            imports.addWidget(btn)
        imports.addStretch()
        lay.addLayout(imports)
        lay.addStretch()

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def mode(self) -> str:
        for key, block in self._blocks.items():
            if block.radio.isChecked():
                return key
        return "pair"

    def collect_state(self, state: WizardState) -> None:
        state.mode = self.mode()
        if state.mode == "pair":
            state.timing = "none"


# ─── Page 1: sources ──────────────────────────────────────────────────────────
#
# There is no dedicated Sources page any more — video/pose/audio each get a
# folder+pattern tab unconditionally on the next page
# (``wizard_multi_tabs.ModalityConfigPage``), and "enabled" is read back from
# whether that tab has a folder in it. The cover page's own drag-and-drop
# card (``gui/cover_page.py``) is the simple, dialog-free path for a single
# device; this file's remaining pages are mode / timing / write.


# ─── Page 2: timing (modes 2 and 3) ──────────────────────────────────────────


class TimingPage(QWidget):
    """How the recording system knows about the camera. Becomes the notebook's timing cell."""

    def __init__(self, app_state, parent: QWidget | None = None):
        super().__init__(parent)
        self._app_state = app_state
        self._mode = "free_running"
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        self._title = QLabel("<b>How does the recording system know about the camera?</b>")
        lay.addWidget(self._title)
        self._figure = _figure_label("free_running")
        lay.addWidget(self._figure)

        self._group = QButtonGroup(self)
        self._rb_offset = QRadioButton(
            "Known offset · the camera started a known time before or after the recording; one number"
        )
        self._rb_pulses = QRadioButton("Pulse per frame · a frame-out line into a digital input")
        self._rb_onsets = QRadioButton("Trial onsets only · a trigger line with one pulse per trial")
        for rb in (self._rb_offset, self._rb_pulses, self._rb_onsets):
            self._group.addButton(rb)
            rb.toggled.connect(self._refresh)
            lay.addWidget(rb)
        self._split = QCheckBox(
            "The recorder split the video into several files · a storage detail; parts are laid end to end"
        )
        lay.addWidget(self._split)

        self._offset_row = QWidget()
        orow = QHBoxLayout(self._offset_row)
        orow.setContentsMargins(24, 0, 0, 0)
        self._offset = QDoubleSpinBox()
        self._offset.setRange(-1e6, 1e6)
        self._offset.setDecimals(3)
        self._offset.setSuffix(" s")
        orow.addWidget(QLabel("Camera starts at"))
        orow.addWidget(self._offset)
        orow.addWidget(_muted("on the recording system's clock"))
        orow.addStretch()
        lay.addWidget(self._offset_row)

        rec = QGroupBox("Recording system")
        rf = QFormLayout(rec)
        self._interface = QComboBox()
        for label, _cls, _filt in RECORDING_INTERFACES:
            self._interface.addItem(label)
        rf.addRow("System", self._interface)
        file_row = QHBoxLayout()
        self._rec_file = QLineEdit()
        self._rec_file.setPlaceholderText("session.rhd")
        browse = QPushButton("Browse…")
        browse.setAutoDefault(False)
        browse.clicked.connect(self._browse_recording)
        file_row.addWidget(self._rec_file, 1)
        file_row.addWidget(browse)
        rf.addRow("File", file_row)
        self._frame_line = QLineEdit()
        self._frame_line.setPlaceholderText("DIGITAL-IN-02")
        self._frame_line_label = QLabel("Frame line")
        rf.addRow(self._frame_line_label, self._frame_line)
        self._trigger_line = QLineEdit()
        self._trigger_line.setPlaceholderText("DIGITAL-IN-01")
        self._trigger_line_label = QLabel("Trigger line")
        rf.addRow(self._trigger_line_label, self._trigger_line)
        self._burst_gap = QDoubleSpinBox()
        self._burst_gap.setRange(2.0, 1000.0)
        self._burst_gap.setValue(10.0)
        self._burst_gap.setSuffix(" × frame interval")
        self._burst_gap_label = QLabel("Split bursts at")
        rf.addRow(self._burst_gap_label, self._burst_gap)
        lay.addWidget(rec)

        note = QLabel(
            "Everything on this page becomes the timing cell of your notebook. neuroconv reads the line and "
            "puts every frame on the session clock; EthoGraph opens the result. Recipe: "
            + styled_link(NEUROCONV_HOWTO, "How to time-align behavior videos")
        )
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        lay.addWidget(note)
        lay.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._figure.setPixmap(QPixmap(str(FIGURES[mode])).scaledToWidth(_FIGURE_WIDTH, Qt.SmoothTransformation))
        free = mode == "free_running"
        self._rb_offset.setVisible(free)
        self._split.setVisible(free)
        self._rb_onsets.setVisible(not free)
        (self._rb_pulses if free else self._rb_onsets).setChecked(True)
        self._refresh()

    def _refresh(self, *_args) -> None:
        offset = self._rb_offset.isChecked()
        pulses = self._rb_pulses.isChecked()
        onsets = self._rb_onsets.isChecked()
        self._offset_row.setVisible(offset)
        for w in (self._frame_line, self._frame_line_label):
            w.setVisible(pulses)
        for w in (self._trigger_line, self._trigger_line_label):
            w.setVisible(onsets)
        for w in (self._burst_gap, self._burst_gap_label):
            w.setVisible(pulses and self._mode == "triggered")

    def _browse_recording(self) -> None:
        _label, _cls, filt = RECORDING_INTERFACES[self._interface.currentIndex()]
        path = browse_open_file(self, self._app_state, "Select the recording", f"{filt};;All files (*)")
        if path:
            self._rec_file.setText(path)

    def validate(self) -> str | None:
        if not self._rec_file.text().strip():
            return "Pick the recording system's file."
        if self._rb_pulses.isChecked() and not self._frame_line.text().strip():
            return "Name the digital line that carries the frame pulses."
        if self._rb_onsets.isChecked() and not self._trigger_line.text().strip():
            return "Name the digital line that carries the trial triggers."
        return None

    def collect_state(self, state: WizardState) -> None:
        if self._rb_offset.isChecked():
            state.timing = "offset"
        elif self._rb_pulses.isChecked():
            state.timing = "pulses"
        else:
            state.timing = "onsets"
        state.split_files = self._split.isVisible() and self._split.isChecked()
        state.offset_s = float(self._offset.value()) if state.timing == "offset" else None
        state.recording_file = self._rec_file.text().strip()
        state.recording_interface = RECORDING_INTERFACES[self._interface.currentIndex()][1]
        state.frame_line = self._frame_line.text().strip() or None
        state.trigger_line = self._trigger_line.text().strip() or None
        state.burst_gap = float(self._burst_gap.value())


# ─── Last page: write ─────────────────────────────────────────────────────────


class WritePage(QWidget):
    """Rig name, notebook path and, for mode 1, the session file to write."""

    def __init__(self, app_state, parent: QWidget | None = None):
        super().__init__(parent)
        self._app_state = app_state
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("<b>Write</b>"))
        form = QFormLayout()
        self._rig = QLineEdit()
        self._rig.setPlaceholderText("e.g. pair24_rig")
        self._rig.textChanged.connect(self._refresh_notebook_path)
        form.addRow("Rig name", self._rig)
        self._notebook = QLineEdit()
        self._notebook.setReadOnly(True)
        form.addRow("Notebook", self._notebook)
        out_row = QHBoxLayout()
        self._output = QLineEdit()
        self._output.setPlaceholderText("session.nc")
        browse = QPushButton("Browse…")
        browse.setAutoDefault(False)
        browse.clicked.connect(self._browse_output)
        out_row.addWidget(self._output, 1)
        out_row.addWidget(browse)
        self._output_label = QLabel("Session file")
        form.addRow(self._output_label, out_row)
        lay.addLayout(form)
        self._hint = _muted("")
        lay.addWidget(self._hint)
        self._cells = _muted("")
        self._cells.setStyleSheet("color: #9aa0ac; font-family: monospace;")
        lay.addWidget(self._cells)
        lay.addStretch()

    def _wizard_dir(self) -> Path:
        project = project_dir_of(self._app_state)
        return (project / "wizard") if project else defaults_dir("wizard")

    def _refresh_notebook_path(self, _text: str = "") -> None:
        name = self._rig.text().strip() or "rig"
        self._notebook.setText(str(self._wizard_dir() / f"{name}.ipynb"))

    def _browse_output(self) -> None:
        path = browse_save_file(self, self._app_state, "Session file", "NetCDF (*.nc)")
        if path:
            self._output.setText(path)

    def populate_from_state(self, state: WizardState) -> None:
        folders = [c.folder_path for c in (state.video, state.pose, state.audio) if c.enabled and c.folder_path]
        session_dir = Path(folders[0]).parent if folders else Path.home()
        state.session_dir = str(session_dir)
        if not self._rig.text():
            self._rig.setText(session_dir.name or "rig")
        self._refresh_notebook_path()
        pair = state.mode == "pair"
        self._output.setVisible(pair)
        self._output_label.setVisible(pair)
        if pair:
            if not self._output.text():
                self._output.setText(str(session_dir / "session.nc"))
            self._hint.setText(
                "The wizard pairs the files now, writes .ethograph/alignment.nwb beside the session file, "
                "and saves the notebook as the record. Next session: change session_dir in its first cell "
                "and run all."
            )
        else:
            self._hint.setText(
                "The wizard cannot read a digital line; the notebook does. It is written now and the "
                "wizard closes. Run it, then pick the session.nwb it writes on the start page."
            )
        cells = ["1 · parameters", "2 · discover files", "3 · trial table"]
        if state.timing != "none":
            cells.append(f"4 · timing ({state.timing})")
        cells.append(
            ("5 · write .nwb with neuroconv, then pair the rest over it")
            if not pair
            else "4 · pair_media → alignment.nwb"
        )
        cells.append("open in EthoGraph")
        self._cells.setText("  →  ".join(cells))

    def validate(self, state: WizardState) -> str | None:
        if not self._rig.text().strip():
            return "Give the rig a name; it names the notebook."
        if state.mode == "pair" and not self._output.text().strip():
            return "Pick where to write the session file."
        return None

    def collect_state(self, state: WizardState) -> None:
        state.rig_name = self._rig.text().strip()
        state.notebook_path = self._notebook.text()
        if state.mode == "pair":
            state.output_path = self._output.text().strip()
