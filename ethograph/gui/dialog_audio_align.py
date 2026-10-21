"""**More tools → Pair & align audio with video…**: microphone files placed on their videos' clocks.

A camera that records sound and a microphone that was started by hand share
one scene but not one clock. The dialog takes as many audio files as videos,
proposes which file belongs to which video and where each one starts
(:mod:`ethograph.io.audio_align`), and shows every proposal as two spectrograms
cut around the loudest shared moment, the microphone's already shifted by the
offset, so a right answer is transients lining up vertically. Nothing is
written until the user has accepted every row.

The result is an ordinary multi-trial session: one trial per video, the
microphone as ``audio_mic-1`` with its own start, and the camera's soundtrack
beside it as ``audio_cam-1-track`` — the sanity check stays audible in the GUI.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import natsort
import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.dialog_busy_progress import BusyProgressDialog
from ethograph.gui.dialog_session_folder import choose_session_folder
from ethograph.gui.file_dialogs import browse_open_files
from ethograph.gui.notify import notify_dialog
from ethograph.gui.project import chosen_project_dir, is_foreign_session, record_drop, register_session
from ethograph.gui.session_folder import source_folders
from ethograph.io.audio_align import (
    ANALYSIS_RATE,
    GOOD_SCORE,
    LOW_SCORE,
    TRACK_FILE_PREFIX,
    AlignedPair,
    OffsetEstimate,
    load_mono,
    log_spectrogram,
    loudest_time,
    propose_pairs,
    score_matrix,
    session_pairing_table,
)
from ethograph.io.audio_extract import ensure_extracted_audio, has_embedded_audio
from ethograph.io.pairing import pair_media
from ethograph.io.session_layout import alignment_path
from ethograph.io.validation import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from ethograph.utils.paths import SETTINGS_DIR
from ethograph.utils.stream_durations import probe_duration

logger = logging.getLogger(__name__)

#: Where a session keeps the copies of its cameras' soundtracks.
VIDEO_AUDIO_DIRNAME = "video_audio"
#: Half-width of the preview window around the loudest shared moment.
PREVIEW_HALF_WIDTH_S = 4.0

SCORE_EXPLANATION = (
    "<b>How the score is derived.</b> Both recordings are reduced to 13 MFCCs every 10 ms, "
    "each coefficient z-scored over time, so only <i>when</i> the sound changes counts, not its "
    "level or the microphones' colour. The two frame sequences are cross-correlated at every "
    "possible lag; the lag with the highest correlation is the offset. The score is that peak's "
    "<i>standard score</i>: how many standard deviations it stands above the mean of the whole "
    "correlation. Above 10 the peak is unambiguous; below 5 it is barely distinct from the noise. "
    "Longer overlaps give higher scores, so short clips score lower than long ones."
)

_COL_VIDEO, _COL_AUDIO, _COL_OFFSET, _COL_SCORE, _COL_DURATION, _COL_ACCEPT = range(6)


class _PathList(QListWidget):
    """A list that accepts file drops of the given extensions and keeps full paths."""

    def __init__(self, extensions: set[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.extensions = extensions
        self.paths: list[str] = []
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        # A drop target, not a browser: a few rows tall, so the spectrograms get the height.
        self.setMinimumHeight(72)
        self.setMaximumHeight(120)
        self.setStyleSheet("QListWidget { border: 2px dashed rgba(255,255,255,60); border-radius: 8px; }")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.add_paths([url.toLocalFile() for url in event.mimeData().urls()])
        event.acceptProposedAction()

    def add_paths(self, paths: list[str]) -> None:
        for p in paths:
            if p and Path(p).suffix.lower() in self.extensions and p not in self.paths:
                self.paths.append(p)
        self.paths = natsort.natsorted(self.paths)
        self.clear()
        for p in self.paths:
            self.addItem(Path(p).name)

    def clear_paths(self) -> None:
        self.paths = []
        self.clear()


class AudioAlignDialog(QDialog):
    """Pair N audio files with N videos and place each on its video's clock, under the user's eye."""

    def __init__(self, app_state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app_state = app_state
        self.setWindowTitle("Pair & align audio with video")
        self.setMinimumSize(980, 720)
        # Spectrograms are judged by eye, so the window may take the whole screen:
        # a maximize button like the grid views, and F11 for true full screen.
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
        )
        QShortcut(QKeySequence(Qt.Key_F11), self, activated=self.toggle_full_screen)
        self.pairs: list[AlignedPair] = []
        self.session_dir: Path | None = None
        self._tracks: list[str] = []  # the videos' extracted soundtracks, one per video
        self._signals: dict[str, np.ndarray] = {}  # analysis-rate mono per file
        self._matrix: list[list[OffsetEstimate]] = []
        self._updating = False
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "<b>Pair &amp; align audio with video</b><br>"
                "Drop as many audio files as videos. Each video's own soundtrack is compared with every "
                "audio file to propose which belongs to which and where it starts. Check every row "
                "against the spectrograms — the microphone's is already shifted, so a right offset "
                "shows the same sounds one above the other — then accept it."
            )
        )

        lists = QHBoxLayout()
        self._videos = _PathList(VIDEO_EXTENSIONS)
        self._audios = _PathList(AUDIO_EXTENSIONS)
        for title, widget, picker in (
            ("Videos (with a soundtrack)", self._videos, self._browse_videos),
            ("Audio files (external microphone)", self._audios, self._browse_audio),
        ):
            column = QVBoxLayout()
            head = QHBoxLayout()
            head.addWidget(QLabel(f"<b>{title}</b>"))
            head.addStretch(1)
            browse = QPushButton("Add…")
            browse.clicked.connect(picker)
            head.addWidget(browse)
            column.addLayout(head)
            column.addWidget(widget)
            lists.addLayout(column, 1)
        layout.addLayout(lists)

        actions = QHBoxLayout()
        self._analyse_btn = QPushButton("Analyse")
        self._analyse_btn.setToolTip("Extract each video's soundtrack and cross-correlate it with every audio file")
        self._analyse_btn.clicked.connect(self._on_analyse)
        actions.addWidget(self._analyse_btn)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._on_clear)
        actions.addWidget(clear)
        actions.addStretch(1)
        layout.addLayout(actions)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["Video", "Audio file", "Offset (s)", "Score", "Duration", "Accept"])
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.currentCellChanged.connect(lambda *_: self._refresh_preview())
        layout.addWidget(self._table, 1)

        self._preview_title = QLabel("")
        layout.addWidget(self._preview_title)
        self._preview = pg.GraphicsLayoutWidget()
        self._preview.setMinimumHeight(300)
        self._track_plot = self._preview.addPlot(row=0, col=0, title="video soundtrack")
        self._mic_plot = self._preview.addPlot(row=1, col=0, title="microphone, shifted by the offset")
        self._mic_plot.setXLink(self._track_plot)
        self._track_img = pg.ImageItem()
        self._mic_img = pg.ImageItem()
        for plot, img in ((self._track_plot, self._track_img), (self._mic_plot, self._mic_img)):
            plot.addItem(img)
            img.setColorMap(pg.colormap.get("viridis"))
            plot.setLabel("left", "Hz")
            plot.setMenuEnabled(False)
        self._mic_plot.setLabel("bottom", "video time (s)")
        layout.addWidget(self._preview, 3)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.button(QDialogButtonBox.Ok).setText("Create session")
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._refresh_ok()

    def toggle_full_screen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _browse_videos(self) -> None:
        self._videos.add_paths(browse_open_files(self, self.app_state, "Videos", "Videos (*.mp4 *.avi *.mov *.mkv)"))

    def _browse_audio(self) -> None:
        self._audios.add_paths(
            browse_open_files(self, self.app_state, "Audio files", "Audio (*.wav *.flac *.ogg *.mp3)")
        )

    def add_videos(self, paths: list[str]) -> None:
        self._videos.add_paths(paths)

    def add_audio(self, paths: list[str]) -> None:
        self._audios.add_paths(paths)

    def _on_clear(self) -> None:
        self._videos.clear_paths()
        self._audios.clear_paths()
        self.pairs = []
        self._tracks = []
        self._signals = {}
        self._matrix = []
        self._table.setRowCount(0)
        self._refresh_preview()
        self._refresh_ok()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _on_analyse(self) -> None:
        try:
            self._check_inputs()
        except ValueError as exc:
            notify_dialog(str(exc), "error")
            return
        dlg = BusyProgressDialog("Comparing the soundtracks with the audio files…", parent=self)
        _, error = dlg.execute(self.analyse)
        if error is not None:
            notify_dialog(f"Alignment failed:\n{error}", "error")
            return
        self._fill_table()

    def _check_inputs(self) -> None:
        videos, audios = self._videos.paths, self._audios.paths
        if not videos or not audios:
            raise ValueError("Drop at least one video and one audio file.")
        if len(videos) != len(audios):
            raise ValueError(
                f"Pairing needs as many audio files as videos: {len(videos)} videos, {len(audios)} audio files."
            )
        silent = [Path(v).name for v in videos if not has_embedded_audio(v)]
        if silent:
            raise ValueError(
                "These videos carry no soundtrack, so there is nothing to align the audio against:\n"
                + "\n".join(silent)
            )

    def analyse(self) -> list[AlignedPair]:
        """Extract, load, cross-correlate every pair and propose the assignment (no Qt)."""
        videos, audios = self._videos.paths, self._audios.paths
        self._tracks = [str(ensure_extracted_audio(v)) for v in videos]
        for path in [*self._tracks, *audios]:
            if path not in self._signals:
                self._signals[path] = load_mono(path)
        self._matrix = score_matrix([self._signals[t] for t in self._tracks], [self._signals[a] for a in audios])
        assignment = propose_pairs(self._matrix)
        self.pairs = [self._make_pair(i, j, accepted=False) for i, j in enumerate(assignment)]
        return self.pairs

    def _make_pair(self, video_idx: int, audio_idx: int, accepted: bool) -> AlignedPair:
        est = self._matrix[video_idx][audio_idx]
        return AlignedPair(
            video=self._videos.paths[video_idx],
            audio=self._audios.paths[audio_idx],
            offset_s=est.offset_s,
            score=est.score,
            track=self._tracks[video_idx],
            accepted=accepted,
        )

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _fill_table(self) -> None:
        self._updating = True
        self._table.setRowCount(len(self.pairs))
        for row, pair in enumerate(self.pairs):
            self._table.setItem(row, _COL_VIDEO, self._readonly(Path(pair.video).name))

            audio = QComboBox()
            for a in self._audios.paths:
                audio.addItem(Path(a).name, a)
            audio.setCurrentIndex(self._audios.paths.index(pair.audio))
            audio.currentIndexChanged.connect(lambda idx, r=row: self._on_audio_changed(r, idx))
            self._table.setCellWidget(row, _COL_AUDIO, audio)

            offset = QDoubleSpinBox()
            offset.setDecimals(3)
            offset.setSingleStep(0.01)
            offset.setRange(-36000.0, 36000.0)
            offset.setValue(pair.offset_s)
            offset.setToolTip(
                "Where the audio file's first sample falls on the video's clock; negative = it started before the video"
            )
            offset.valueChanged.connect(lambda value, r=row: self._on_offset_changed(r, value))
            self._table.setCellWidget(row, _COL_OFFSET, offset)

            accept = QCheckBox()
            accept.setChecked(pair.accepted)
            accept.toggled.connect(lambda checked, r=row: self._on_accept_toggled(r, checked))
            holder = QWidget()
            box = QHBoxLayout(holder)
            box.setContentsMargins(0, 0, 0, 0)
            box.setAlignment(Qt.AlignCenter)
            box.addWidget(accept)
            self._table.setCellWidget(row, _COL_ACCEPT, holder)
            self._refresh_row(row)
        self._table.resizeColumnsToContents()
        self._updating = False
        if self.pairs:
            self._table.selectRow(0)
        self._refresh_preview()
        self._refresh_ok()

    def _refresh_row(self, row: int) -> None:
        pair = self.pairs[row]
        score = self._readonly(f"{pair.score:.1f}")
        if pair.score < LOW_SCORE:
            score.setForeground(QColor("#e06c75"))
            verdict = "Weak match: the peak barely stands out. Check the spectrograms or set the offset by hand."
        elif pair.score < GOOD_SCORE:
            score.setForeground(QColor("#e5c07b"))
            verdict = "Plausible match; confirm it on the spectrograms."
        else:
            verdict = "Clear match."
        score.setToolTip(f"<b>{verdict}</b><br><br>{SCORE_EXPLANATION}")
        self._table.setItem(row, _COL_SCORE, score)
        self._table.setItem(row, _COL_DURATION, self._readonly(self._duration_text(pair)))
        natural = natsort.natsorted(self._audios.paths)[row] if row < len(self._audios.paths) else None
        video_item = self._table.item(row, _COL_VIDEO)
        if natural is not None and natural != pair.audio:
            video_item.setToolTip("The content match disagrees with the files' sort order.")
            video_item.setForeground(QColor("#e5c07b"))

    @staticmethod
    def _duration_text(pair: AlignedPair) -> str:
        video = probe_duration(pair.video, "video")
        audio = probe_duration(pair.audio, "audio")
        if video is None or audio is None:
            return "?"
        return f"video {video:.1f} s, audio {audio:.1f} s"

    @staticmethod
    def _readonly(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _on_audio_changed(self, row: int, audio_idx: int) -> None:
        """Give *row* the chosen file; the row that held it takes this row's old file (a swap)."""
        if self._updating:
            return
        chosen = self._audios.paths[audio_idx]
        current = self.pairs[row].audio
        other = next((r for r, p in enumerate(self.pairs) if p.audio == chosen and r != row), None)
        self.pairs[row] = self._make_pair(row, audio_idx, accepted=False)
        if other is not None:
            self.pairs[other] = self._make_pair(other, self._audios.paths.index(current), accepted=False)
        self._fill_table()
        self._table.selectRow(row)

    def _on_offset_changed(self, row: int, value: float) -> None:
        if self._updating:
            return
        self.pairs[row].offset_s = float(value)
        self.pairs[row].accepted = False
        self._set_accept_checked(row, False)
        self._refresh_preview()
        self._refresh_ok()

    def _on_accept_toggled(self, row: int, checked: bool) -> None:
        if self._updating:
            return
        self.pairs[row].accepted = checked
        self._refresh_ok()

    def _set_accept_checked(self, row: int, checked: bool) -> None:
        holder = self._table.cellWidget(row, _COL_ACCEPT)
        box = holder.findChild(QCheckBox) if holder is not None else None
        if box is not None:
            self._updating = True
            box.setChecked(checked)
            self._updating = False

    def _refresh_ok(self) -> None:
        ready = bool(self.pairs) and all(p.accepted for p in self.pairs)
        ok = self._buttons.button(QDialogButtonBox.Ok)
        ok.setEnabled(ready)
        ok.setToolTip("" if ready else "Accept every row first")

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _refresh_preview(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self.pairs):
            self._track_img.clear()
            self._mic_img.clear()
            self._preview_title.setText("")
            return
        pair = self.pairs[row]
        track = self._signals[pair.track or ""]
        mic = self._signals[pair.audio]
        # The overlap on the video clock: where both recordings have sound.
        lo = max(0.0, pair.offset_s)
        hi = min(len(track) / ANALYSIS_RATE, pair.offset_s + len(mic) / ANALYSIS_RATE)
        centre = loudest_time(track, ANALYSIS_RATE, lo, hi) if hi > lo else 0.0
        t0, t1 = centre - PREVIEW_HALF_WIDTH_S, centre + PREVIEW_HALF_WIDTH_S
        self._draw(self._track_img, track, t0, t1, shift=0.0)
        self._draw(self._mic_img, mic, t0, t1, shift=pair.offset_s)
        self._track_plot.setXRange(t0, t1, padding=0)
        drift = self._matrix[row][self._audios.paths.index(pair.audio)].drift_ppm if self._matrix else None
        text = f"<b>{Path(pair.video).name}</b> ↔ <b>{Path(pair.audio).name}</b>: {pair.describe_offset()}"
        if drift is not None and abs(drift) > 50:
            text += f" — <span style='color:#e5c07b'>clocks drift by {drift:+.0f} ppm</span>"
        self._preview_title.setText(text)

    @staticmethod
    def _draw(img: pg.ImageItem, signal: np.ndarray, t0: float, t1: float, shift: float) -> None:
        """Cut ``[t0, t1]`` of the video clock out of *signal*, whose own clock starts at *shift*."""
        times, freqs, db = log_spectrogram(signal, ANALYSIS_RATE, t0 - shift, t1 - shift)
        img.setImage(db.T, autoLevels=False, levels=(np.percentile(db, 5), np.percentile(db, 99.5)))
        img.setRect(t0, float(freqs[0]), t1 - t0, float(freqs[-1] - freqs[0]))

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        folders = source_folders({"video": self._videos.paths, "audio": self._audios.paths})
        chosen = choose_session_folder(folders, self.app_state, self)
        if chosen is None:
            return
        if is_foreign_session(chosen) and alignment_path(chosen).is_file():
            answer = QMessageBox.question(
                self,
                "Overwrite this alignment?",
                f"<b>{chosen.name}</b> already holds an alignment no drag &amp; drop made. Replace it?",
            )
            if answer != QMessageBox.Yes:
                return
        try:
            self.write_session(chosen)
        except Exception as exc:  # noqa: BLE001 - outermost GUI boundary
            logger.exception("Writing the aligned session failed")
            notify_dialog(f"Could not write the session:\n{exc}", "error")
            return
        self.accept()

    def write_session(self, session_dir: Path) -> Path:
        """Write *session_dir*'s alignment from the accepted pairs and point ``app_state`` at it.

        The cameras' soundtracks are copied into ``.ethograph/video_audio/`` as
        ``video-track_{video}.wav`` — named for what they are, since their file
        name is what the audio panels show — and the session gets a drop record,
        so it reopens from the start page like any drop.
        """
        if not self.pairs or not all(p.accepted for p in self.pairs):
            raise ValueError("Every pair must be accepted before the session is written")
        track_dir = session_dir / SETTINGS_DIR / VIDEO_AUDIO_DIRNAME
        track_dir.mkdir(parents=True, exist_ok=True)
        placed: list[AlignedPair] = []
        for pair in self.pairs:
            copy = track_dir / f"{TRACK_FILE_PREFIX}{Path(pair.video).stem}.wav"
            if pair.track is not None and not copy.exists():
                shutil.copyfile(pair.track, copy)
            placed.append(AlignedPair(pair.video, pair.audio, pair.offset_s, pair.score, str(copy), True))
        table, rates = session_pairing_table(placed)
        pair_media(table, stream_rates=rates, output_path=alignment_path(session_dir), on_existing="replace")

        app_state = self.app_state
        app_state.nc_file_path = str(session_dir)
        app_state.nwb_file_path = None
        app_state.metadata_path = None
        app_state.labels_import_path = None
        app_state.video_folder = str(Path(placed[0].video).parent)
        app_state.audio_folder = str(Path(placed[0].audio).parent)
        app_state.pose_folder = None
        app_state.ephys_path = None
        app_state.neurons_path = None
        app_state.image_paths = []
        record_drop(session_dir, [*self._videos.paths, *self._audios.paths], app_state)
        project = chosen_project_dir(app_state)
        if project is not None:
            register_session(project, session_dir)
        self.session_dir = session_dir
        return session_dir
