"""Live audio recording window: one oscillogram, one spectrogram, a microphone.

Opened from the start page's tools menu. The window shares nothing with the
main window but the two plot classes and the spectrogram settings
(``nfft``, ``hop_frac``, ``vmin_db``/``vmax_db``, ``spec_colormap`` are the
GUI's own, so what is tuned here is what the session opens with).

Two plot modes:

* **live** — the right edge is "now", the window length steps through
  :data:`~ethograph.io.live_audio.ZOOM_LADDER_S`, the mouse never pans. The
  spectrogram grows column by column (:class:`IncrementalSpectrogram`); a
  setting change greys the panel and recomputes the visible window from the
  ring buffer.
* **file** — after Stop, both plots read the finished WAV through the
  ordinary :class:`FileSource` path with free pan/zoom, so what was just
  recorded can be inspected before the next take.

Every take is a WAV in the chosen folder; "Open in Ethograph" hands the
folder's WAVs to the start page as a multi-trial drop, so the folder becomes a
session with one trial per file and annotation starts at once.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import natsort
import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.file_dialogs import browse_open_dir
from ethograph.gui.notify import notify
from ethograph.gui.plots_audiotrace import AudioTracePlot, _downsample_minmax
from ethograph.gui.plots_spectrogram import SharedAudioCache, SpectrogramPlot, auto_levels_db
from ethograph.io.live_audio import (
    WAV_SUBTYPES,
    ZOOM_LADDER_S,
    IncrementalSpectrogram,
    LiveRecorder,
    Recording,
    check_input_settings,
    day_folder,
    list_input_devices,
)
from ethograph.io.plot_sources import FileSource, SampleSlice

logger = logging.getLogger(__name__)

#: GUI refresh period.
TICK_MS = 33
#: How much of the gap between the smoothed display clock and the stream clock
#: is closed per tick: blocks arrive in bursts, the axis must not.
CLOCK_EASE = 0.15
#: Beyond this lag the display clock snaps instead of easing (a GUI stall).
CLOCK_SNAP_S = 0.25
#: nfft choices offered (powers of two, as the main GUI expects).
NFFT_CHOICES = (128, 256, 512, 1024, 2048, 4096, 8192)
HOP_CHOICES = (0.125, 0.25, 0.5, 0.75)
COLORMAPS = {
    "CET-R4": "jet",
    "CET-L8": "blue-pink-yellow",
    "CET-L16": "black-blue-green-white",
    "CET-CBL2": "black-blue-yellow-white",
    "CET-L1": "black-white",
    "CET-L3": "inferno",
}
COMMON_RATES = (8000.0, 16000.0, 22050.0, 44100.0, 48000.0, 96000.0, 192000.0, 250000.0, 384000.0)


class _SpecColumns:
    """Ring of spectrogram columns ``(capacity, F)``; the newest column's time is known."""

    def __init__(self, capacity: int, n_freqs: int) -> None:
        self._buf = np.full((capacity, n_freqs), -100.0, dtype=np.float32)
        self._write = 0
        self.held = 0
        self.last_centre_s: float | None = None

    def append(self, cols: np.ndarray, centres: np.ndarray) -> None:
        """``cols`` is ``(F, K)`` as scipy returns it."""
        k = cols.shape[1]
        if k == 0:
            return
        cap = len(self._buf)
        block = cols.T[-cap:]
        k = len(block)
        end = self._write + k
        if end <= cap:
            self._buf[self._write : end] = block
        else:
            first = cap - self._write
            self._buf[self._write :] = block[:first]
            self._buf[: k - first] = block[first:]
        self._write = end % cap
        self.held = min(cap, self.held + k)
        self.last_centre_s = float(centres[-1])

    def latest(self, n: int) -> np.ndarray:
        n = max(0, min(n, self.held))
        if n == 0:
            return self._buf[:0]
        cap = len(self._buf)
        start = (self._write - n) % cap
        if start + n <= cap:
            return self._buf[start : start + n]
        return np.concatenate([self._buf[start:], self._buf[: (start + n) % cap]])


class _LiveTracePlot(AudioTracePlot):
    live = True

    def _on_view_range_changed(self):
        if self.live or not self.isVisible():
            return
        self._td.trigger()


class _LiveSpectrogramPlot(SpectrogramPlot):
    live = True

    def _on_view_range_changed(self):
        if self.live or not self.isVisible():
            return
        self._td.trigger()


class LiveRecorderWindow(QDialog):
    """The recorder. ``on_open_session(wav_paths)`` is the start page's hand-off."""

    def __init__(
        self,
        app_state,
        project_dir: Path,
        on_open_session: Callable[[list[str]], None],
        parent: QWidget | None = None,
        *,
        start_stream: bool = True,
    ) -> None:
        super().__init__(parent)
        self.app_state = app_state
        self._project_dir = Path(project_dir)
        self._open_session_cb = on_open_session
        self._start_stream = start_stream
        self.setWindowTitle("ethograph — live audio recording")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
        )
        self.setModal(False)
        self.resize(1100, 700)

        self._recorder: LiveRecorder | None = None
        self._spec: IncrementalSpectrogram | None = None
        self._cols: _SpecColumns | None = None
        self._consumed = 0
        self._recordings: list[Recording] = []
        self._devices = list_input_devices()
        self._disp_t: float | None = None
        self._last_wall: float | None = None
        self._auto_pending = False
        self._capture_note = ""

        self._window_idx = self._nearest_ladder(app_state.get_with_default("live_recording_window_s"))

        outer = QVBoxLayout(self)
        outer.addWidget(self._build_capture_row())
        outer.addWidget(self._build_output_row())
        outer.addWidget(self._build_plots(), 1)
        outer.addWidget(self._build_view_row())
        outer.addWidget(self._build_transport_row())

        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)

        self._refresh_folder_label()
        self._open_stream()

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------
    def _build_capture_row(self) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        for idx, name, n_in, rate in self._devices:
            self._device_combo.addItem(f"{name} ({n_in} in, {rate:g} Hz)", idx)
        wanted = self.app_state.get_with_default("live_recording_device")
        for i, (_idx, name, _n, _r) in enumerate(self._devices):
            if name == wanted:
                self._device_combo.setCurrentIndex(i)
        self._device_combo.setToolTip(
            "Input device; several microphones must be channels of one device, never several devices"
        )
        lay.addWidget(self._device_combo, 3)

        lay.addWidget(QLabel("Channels:"))
        self._channels_spin = QSpinBox()
        self._channels_spin.setRange(1, 64)
        self._channels_spin.setValue(int(self.app_state.get_with_default("live_recording_channels")))
        self._channels_spin.setToolTip("Channels recorded into one multichannel WAV; each opens as its own mic")
        lay.addWidget(self._channels_spin)

        lay.addWidget(QLabel("Rate (Hz):"))
        self._rate_combo = QComboBox()
        self._rate_combo.setEditable(True)
        self._rate_combo.setToolTip(
            "The device's default is preselected; a rate it refuses is reported, never silently changed"
        )
        lay.addWidget(self._rate_combo)

        lay.addWidget(QLabel("Format:"))
        self._subtype_combo = QComboBox()
        self._subtype_combo.addItems(WAV_SUBTYPES)
        self._subtype_combo.setCurrentText(str(self.app_state.get_with_default("live_recording_subtype")))
        lay.addWidget(self._subtype_combo)

        self._fill_rates()
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        self._channels_spin.valueChanged.connect(lambda _v: self._open_stream())
        self._rate_combo.currentTextChanged.connect(lambda _t: self._open_stream())
        self._capture_widgets = (self._device_combo, self._channels_spin, self._rate_combo, self._subtype_combo)
        return row

    def _build_output_row(self) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Save to:"))
        self._folder_label = QLineEdit()
        self._folder_label.setReadOnly(True)
        lay.addWidget(self._folder_label, 3)
        browse = QPushButton("Browse…")
        browse.setToolTip("Record somewhere other than the project's recordings folder; remembered")
        browse.clicked.connect(self._on_browse_folder)
        lay.addWidget(browse)
        today = QPushButton("Today's folder")
        today.setToolTip("Back to {project}/recordings/{YYYY-MM-DD}")
        today.clicked.connect(self._on_today_folder)
        lay.addWidget(today)
        lay.addWidget(QLabel("Prefix:"))
        self._prefix_edit = QLineEdit(str(self.app_state.get_with_default("live_recording_prefix")))
        self._prefix_edit.setToolTip("Files are named {prefix}_{YYYYMMDD}_{HHMMSS}.wav")
        self._prefix_edit.setMaximumWidth(140)
        self._prefix_edit.editingFinished.connect(self._on_prefix_edited)
        lay.addWidget(self._prefix_edit)
        lay.addWidget(QLabel("Roll every (min):"))
        self._roll_spin = QSpinBox()
        self._roll_spin.setRange(1, 24 * 60)
        self._roll_spin.setValue(int(self.app_state.get_with_default("live_recording_roll_min")))
        self._roll_spin.setToolTip("A take longer than this continues in a new file (a new trial), no gap logged")
        self._roll_spin.valueChanged.connect(self._on_roll_changed)
        lay.addWidget(self._roll_spin)
        self._output_widgets = (browse, today, self._prefix_edit)
        return row

    def _build_plots(self) -> QWidget:
        self._trace = _LiveTracePlot(self.app_state)
        self._spectrogram = _LiveSpectrogramPlot(self.app_state)
        self._spectrogram.plot_item.setXLink(self._trace.plot_item)
        for plot in (self._trace, self._spectrogram):
            plot.time_marker.hide()
            plot._interaction_enabled = False
            plot.vb.setMouseEnabled(x=False, y=True)
        self._trace.setLabel("bottom", "Time since stream start", units="s")
        self._trace.scene().sigMouseClicked.connect(self._on_trace_clicked)
        self._set_full_scale()
        self._overlay = QLabel("Recomputing…", self._spectrogram)
        self._overlay.setAlignment(Qt.AlignCenter)
        self._overlay.setStyleSheet("background: rgba(128,128,128,160); color: white; font-size: 16pt;")
        self._overlay.hide()
        split = QSplitter(Qt.Vertical)
        split.addWidget(self._trace)
        split.addWidget(self._spectrogram)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        return split

    def _build_view_row(self) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(QLabel("Show channel:"))
        self._shown_combo = QComboBox()
        self._shown_combo.setToolTip("Which recorded channel the plots draw; every channel is written regardless")
        self._shown_combo.currentIndexChanged.connect(lambda _i: self._on_spec_setting_changed())
        lay.addWidget(self._shown_combo)

        lay.addWidget(QLabel("Window:"))
        minus = QToolButton()
        minus.setText("−")
        minus.setToolTip("Shorter window (Ctrl+wheel also steps)")
        minus.clicked.connect(lambda: self._step_window(-1))
        self._window_label = QLabel()
        plus = QToolButton()
        plus.setText("+")
        plus.setToolTip("Longer window")
        plus.clicked.connect(lambda: self._step_window(+1))
        lay.addWidget(minus)
        lay.addWidget(self._window_label)
        lay.addWidget(plus)
        self._refresh_window_label()
        full = QToolButton()
        full.setText("Full scale")
        full.setToolTip(
            "Amplitude back to ±1 (full scale). Double-click the oscillogram to fit the loudest\n"
            "sample on screen at that moment and hold it."
        )
        full.clicked.connect(self._set_full_scale)
        lay.addWidget(full)

        lay.addSpacing(16)
        lay.addWidget(QLabel("nfft:"))
        self._nfft_combo = QComboBox()
        self._nfft_combo.addItems([str(n) for n in NFFT_CHOICES])
        self._nfft_combo.setCurrentText(str(int(self.app_state.get_with_default("nfft"))))
        self._nfft_combo.currentTextChanged.connect(self._on_nfft_changed)
        lay.addWidget(self._nfft_combo)

        lay.addWidget(QLabel("hop:"))
        self._hop_combo = QComboBox()
        self._hop_combo.addItems([str(h) for h in HOP_CHOICES])
        self._hop_combo.setCurrentText(str(self.app_state.get_with_default("hop_frac")))
        self._hop_combo.currentTextChanged.connect(self._on_hop_changed)
        lay.addWidget(self._hop_combo)

        lay.addWidget(QLabel("dB:"))
        self._vmin_spin = QDoubleSpinBox()
        self._vmin_spin.setRange(-200.0, 100.0)
        self._vmin_spin.setValue(float(self.app_state.get_with_default("vmin_db")))
        self._vmin_spin.valueChanged.connect(self._on_levels_changed)
        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(-200.0, 100.0)
        self._vmax_spin.setValue(float(self.app_state.get_with_default("vmax_db")))
        self._vmax_spin.valueChanged.connect(self._on_levels_changed)
        lay.addWidget(self._vmin_spin)
        lay.addWidget(QLabel("to"))
        lay.addWidget(self._vmax_spin)
        auto = QToolButton()
        auto.setText("Auto")
        auto.setToolTip("Fit the dB range to what is on screen: noise floor at the bottom, loudest at the top")
        auto.clicked.connect(self._auto_levels)
        lay.addWidget(auto)

        lay.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        for name, label in COLORMAPS.items():
            self._cmap_combo.addItem(label, name)
        idx = self._cmap_combo.findData(self.app_state.get_with_default("spec_colormap"))
        self._cmap_combo.setCurrentIndex(max(0, idx))
        self._cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        lay.addWidget(self._cmap_combo)
        lay.addStretch(1)
        return row

    def _build_transport_row(self) -> QWidget:
        row = QFrame()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        self._record_btn = QPushButton("●  Record")
        self._record_btn.setStyleSheet("color: #c0392b; font-weight: bold;")
        self._record_btn.clicked.connect(self._on_record)
        self._pause_btn = QPushButton("⏸  Pause")
        self._pause_btn.clicked.connect(self._on_pause)
        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        for b in (self._record_btn, self._pause_btn, self._stop_btn):
            lay.addWidget(b)
        self._status_label = QLabel()
        lay.addWidget(self._status_label, 1)

        self._meter_box = QWidget()
        self._meter_layout = QHBoxLayout(self._meter_box)
        self._meter_layout.setContentsMargins(0, 0, 0, 0)
        self._meters: list[QProgressBar] = []
        lay.addWidget(self._meter_box, 2)
        self._clip_btn = QToolButton()
        self._clip_btn.setText("CLIP")
        self._clip_btn.setToolTip(
            "Lights when any channel reaches full scale; click to reset. Turn the interface's gain down."
        )
        self._clip_btn.clicked.connect(self._on_reset_clip)
        lay.addWidget(self._clip_btn)

        self._open_btn = QPushButton("Open in Ethograph →")
        self._open_btn.setToolTip("Load this folder as a session (one trial per WAV) and start annotating")
        self._open_btn.clicked.connect(self._on_open_session)
        lay.addWidget(self._open_btn)
        self._set_transport_state("idle")
        return row

    # ------------------------------------------------------------------
    # Capture settings
    # ------------------------------------------------------------------
    def _selected_device(self) -> tuple[int, str, int, float] | None:
        i = self._device_combo.currentIndex()
        return self._devices[i] if 0 <= i < len(self._devices) else None

    def _fill_rates(self) -> None:
        dev = self._selected_device()
        self._rate_combo.blockSignals(True)
        self._rate_combo.clear()
        if dev is not None:
            _idx, _name, n_in, default_rate = dev
            self._channels_spin.setMaximum(max(1, n_in))
            rates = sorted({default_rate, *COMMON_RATES})
            for r in rates:
                self._rate_combo.addItem(f"{r:g}", r)
            saved = self.app_state.get_with_default("live_recording_rate")
            self._rate_combo.setCurrentText(f"{saved if saved else default_rate:g}")
        self._rate_combo.blockSignals(False)

    def _on_device_changed(self, _index: int) -> None:
        dev = self._selected_device()
        if dev is not None:
            self.app_state.live_recording_device = dev[1]
        self._fill_rates()
        self._open_stream()

    def _capture_settings(self) -> tuple[int, float, int] | None:
        dev = self._selected_device()
        if dev is None:
            return None
        try:
            rate = float(self._rate_combo.currentText())
        except ValueError:
            return None
        return dev[0], rate, int(self._channels_spin.value())

    def _open_stream(self) -> None:
        """(Re)open the input stream with the strip's settings; monitoring starts at once."""
        self._close_recorder()
        settings = self._capture_settings()
        if settings is None:
            self._status_label.setText("No input device found." if not self._devices else "Enter a sample rate.")
            self._set_transport_state("idle")
            return
        device, rate, channels = settings
        note = ""
        if self._start_stream:
            problem = check_input_settings(device, rate, channels)
            if problem is not None:
                # WASAPI (shared mode) and CoreAudio take only the endpoint's own
                # layout: retry with the device's native channel count and rate.
                dev = self._selected_device()
                assert dev is not None
                native_rate, native_channels = dev[3], dev[2]
                if check_input_settings(device, native_rate, native_channels) is None:
                    note = f"Device needs {native_channels} ch at {native_rate:g} Hz; adjusted. "
                    rate, channels = native_rate, native_channels
                    self._channels_spin.blockSignals(True)
                    self._channels_spin.setValue(channels)
                    self._channels_spin.blockSignals(False)
                    self._rate_combo.blockSignals(True)
                    self._rate_combo.setCurrentText(f"{rate:g}")
                    self._rate_combo.blockSignals(False)
                else:
                    self._status_label.setText(f"Device refuses {rate:g} Hz × {channels} ch: {problem}")
                    self._set_transport_state("idle")
                    return
        self.app_state.live_recording_rate = rate
        self.app_state.live_recording_channels = channels
        self.app_state.live_recording_subtype = self._subtype_combo.currentText()
        self._recorder = LiveRecorder(
            rate,
            channels,
            ring_s=ZOOM_LADDER_S[-1] + 5.0,
            folder=self._folder(),
            prefix=self._prefix_edit.text().strip() or "rec",
            subtype=self._subtype_combo.currentText(),
            roll_s=self._roll_spin.value() * 60.0,
        )
        self._shown_combo.blockSignals(True)
        self._shown_combo.clear()
        self._shown_combo.addItems([f"Ch {i + 1}" for i in range(channels)])
        shown = int(self.app_state.get_with_default("live_recording_channel_shown"))
        self._shown_combo.setCurrentIndex(min(shown, channels - 1))
        self._shown_combo.blockSignals(False)
        self._rebuild_meters(channels)
        if self._start_stream:
            try:
                self._recorder.start_monitoring(device)
            except Exception as e:  # noqa: BLE001 - PortAudio errors are not typed usefully
                self._status_label.setText(f"Could not open the device: {e}")
                self._recorder = None
                self._set_transport_state("idle")
                return
        self._enter_live_mode()
        self._set_transport_state("monitoring")
        self._capture_note = note
        self._timer.start()

    def _close_recorder(self) -> None:
        self._timer.stop()
        if self._recorder is None:
            return
        finished = self._recorder.close()
        if finished is not None:
            self._recordings.append(finished)
        self._recorder = None

    # ------------------------------------------------------------------
    # Output settings
    # ------------------------------------------------------------------
    def _folder(self) -> Path:
        custom = self.app_state.get_with_default("live_recording_folder")
        if custom:
            return Path(custom)
        return day_folder(self._project_dir / "recordings", datetime.now())

    def _refresh_folder_label(self) -> None:
        self._folder_label.setText(str(self._folder()))
        if self._recorder is not None:
            self._recorder.folder = self._folder()

    def _on_browse_folder(self) -> None:
        path = browse_open_dir(self, self.app_state, "Folder for the recordings", preferred_dir=str(self._folder()))
        if not path:
            return
        self.app_state.live_recording_folder = str(Path(path))
        self._refresh_folder_label()

    def _on_today_folder(self) -> None:
        self.app_state.live_recording_folder = None
        self._refresh_folder_label()

    def _on_prefix_edited(self) -> None:
        prefix = self._prefix_edit.text().strip() or "rec"
        self.app_state.live_recording_prefix = prefix
        if self._recorder is not None:
            self._recorder.prefix = prefix

    def _on_roll_changed(self, minutes: int) -> None:
        self.app_state.live_recording_roll_min = int(minutes)
        if self._recorder is not None:
            self._recorder.roll_s = minutes * 60.0

    # ------------------------------------------------------------------
    # View settings
    # ------------------------------------------------------------------
    @staticmethod
    def _nearest_ladder(seconds: float) -> int:
        return int(np.argmin([abs(math.log(w) - math.log(max(seconds, 1e-3))) for w in ZOOM_LADDER_S]))

    @property
    def window_s(self) -> float:
        return ZOOM_LADDER_S[self._window_idx]

    def _refresh_window_label(self) -> None:
        self._window_label.setText(f"{self.window_s:g} s")

    def _step_window(self, step: int) -> None:
        self._window_idx = int(np.clip(self._window_idx + step, 0, len(ZOOM_LADDER_S) - 1))
        self.app_state.live_recording_window_s = self.window_s
        self._refresh_window_label()
        if not self._trace.live:
            return
        self._tick()

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier and self._trace.live:
            self._step_window(+1 if event.angleDelta().y() < 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def _on_nfft_changed(self, text: str) -> None:
        self.app_state.nfft = int(text)
        self._on_spec_setting_changed()

    def _on_hop_changed(self, text: str) -> None:
        self.app_state.hop_frac = float(text)
        self._on_spec_setting_changed()

    def _on_levels_changed(self, _value: float) -> None:
        self.app_state.vmin_db = float(self._vmin_spin.value())
        self.app_state.vmax_db = float(self._vmax_spin.value())
        self._spectrogram.update_levels()

    def _on_cmap_changed(self, index: int) -> None:
        name = self._cmap_combo.itemData(index)
        self.app_state.spec_colormap = name
        self._spectrogram.update_colormap(name)

    def _on_spec_setting_changed(self) -> None:
        self.app_state.live_recording_channel_shown = max(0, self._shown_combo.currentIndex())
        if self._trace.live:
            self._rebuild_spectrogram()
        else:
            self._enter_file_mode(self._last_file())

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------
    def _shown_channel(self) -> int:
        return max(0, self._shown_combo.currentIndex())

    def _enter_live_mode(self) -> None:
        for plot in (self._trace, self._spectrogram):
            plot.live = True
            plot.set_source(None)
            plot.vb.setMouseEnabled(x=False, y=True)
        self._trace.trace_item.setData([], [])
        self._consumed = 0
        self._disp_t = None
        self._last_wall = None
        self._rebuild_spectrogram()

    def _rebuild_spectrogram(self) -> None:
        """Grey the panel, recompute the visible window from the ring, ungrey."""
        if self._recorder is None:
            return
        self._overlay.setGeometry(self._spectrogram.rect())
        self._overlay.show()
        self._overlay.repaint()
        rec = self._recorder
        nfft = int(self.app_state.get_with_default("nfft"))
        hop_frac = float(self.app_state.get_with_default("hop_frac"))
        self._spec = IncrementalSpectrogram(rec.samplerate, nfft, hop_frac)
        capacity = int(math.ceil(ZOOM_LADDER_S[-1] / self._spec.hop_s)) + 2
        self._cols = _SpecColumns(capacity, len(self._spec.freqs))
        held = rec.ring.latest(int(ZOOM_LADDER_S[-1] * rec.samplerate))
        # The ring's oldest held sample sits at stream time (total - held)/fs:
        # the fresh spectrogram's clock must start there, not at 0.
        offset_samples = rec.ring.total - len(held)
        self._spec._next_frame_start = offset_samples
        centres, cols = self._spec.feed(held[:, self._shown_channel()])
        self._cols.append(cols, centres)
        self._consumed = rec.ring.total
        nyquist = rec.samplerate / 2
        self._spectrogram.vb.setLimits(yMin=0, yMax=nyquist, minYRange=max(100.0, nyquist * 0.02), maxYRange=nyquist)
        self._spectrogram.plot_item.setYRange(0, nyquist, padding=0)
        self._overlay.hide()
        self._auto_pending = True
        self._tick()

    def _tick(self) -> None:
        rec = self._recorder
        if rec is None:
            return
        try:
            rec.poll()
        except RuntimeError as e:
            logger.exception("Recording failed")
            self._status_label.setText(str(e))
            self._set_transport_state("monitoring")
            return
        rolled = rec.take_finished()
        if rolled:
            self._recordings.extend(rolled)
            notify(f"Rolled into a new file: {rec.current_path.name if rec.current_path else ''}")
        if rec.stream_error:
            self._status_label.setText(f"Stream: {rec.stream_error}")
            rec.stream_error = None
        self._update_meters(rec)
        self._update_status(rec)
        if not self._trace.live or self._spec is None or self._cols is None:
            return

        new = rec.ring.since(self._consumed)
        self._consumed = rec.ring.total
        if len(new):
            centres, cols = self._spec.feed(new[:, self._shown_channel()])
            self._cols.append(cols, centres)

        now = self._display_now(rec)
        t0 = now - self.window_s
        fs = rec.samplerate
        n = int(self.window_s * fs)
        width = self._trace.vb.screenGeometry().width() or 400
        # Envelope bins are aligned to absolute sample indices, so a bin keeps
        # its min/max from tick to tick instead of being re-cut as data arrives.
        step = max(1, n // max(width, 400))
        n += (rec.ring.total - n) % step
        held = rec.ring.latest(n)
        if len(held):
            ts = rec.stream_time_s() - len(held) / fs + np.arange(len(held)) / fs
            values = held[:, self._shown_channel()].astype(np.float64)
            result = _downsample_minmax(SampleSlice(ts, values), ts[0], ts[-1], width)
            if result is not None:
                self._trace.trace_item.setData(result[0], result[1])
        self._trace.vb.setXRange(t0, now, padding=0)

        if self._auto_pending and self._cols.held * self._spec.hop_s >= 1.0:
            self._auto_pending = False
            self._auto_levels()
        if self._cols.held and self._cols.last_centre_s is not None:
            hop_s = self._spec.hop_s
            n_cols = min(self._cols.held, int(self.window_s / hop_s) + 2)
            cols = self._cols.latest(n_cols)
            width = self._spectrogram.vb.screenGeometry().width() or 400
            step = max(1, len(cols) // (2 * width))
            shown = cols[::step]
            x0 = self._cols.last_centre_s - (len(cols) - 0.5) * hop_s
            self._spectrogram.spec_item.setImage(shown, autoLevels=False)
            self._spectrogram.spec_item.setRect(x0, 0, len(cols) * hop_s, fs / 2)

    def _set_full_scale(self) -> None:
        self._trace.vb.enableAutoRange(y=False)
        self._trace.vb.setLimits(yMin=-1.05, yMax=1.05)
        self._trace.plot_item.setYRange(-1.0, 1.0, padding=0)

    def _on_trace_clicked(self, event) -> None:
        """Double-click: fit the amplitude to the loudest visible sample, then hold."""
        if not event.double() or event.button() != Qt.LeftButton:
            return
        y = self._trace.trace_item.yData
        if y is None or len(y) == 0:
            return
        peak = float(np.max(np.abs(y)))
        if peak <= 0:
            return
        self._trace.vb.enableAutoRange(y=False)
        self._trace.plot_item.setYRange(-1.1 * peak, 1.1 * peak, padding=0)

    def _display_now(self, rec: LiveRecorder) -> float:
        """The axis's "now": wall-clock smooth, eased onto the stream clock, never ahead of it."""
        wall = time.monotonic()
        ring_t = rec.stream_time_s()
        if self._disp_t is None or self._last_wall is None:
            disp = ring_t
        else:
            disp = self._disp_t + (wall - self._last_wall)
            disp += CLOCK_EASE * (ring_t - disp)
            if abs(ring_t - disp) > CLOCK_SNAP_S:
                disp = ring_t
            disp = min(disp, ring_t)
        self._disp_t, self._last_wall = disp, wall
        return disp

    def _auto_levels(self) -> None:
        """The main GUI's rule (:func:`auto_levels_db`) over what is on screen."""
        if self._trace.live:
            if self._cols is None or self._cols.held == 0:
                return
            levels = auto_levels_db(self._cols.latest(self._cols.held).T)
        else:
            sxx = self._spectrogram.buffer.Sxx_db
            levels = auto_levels_db(sxx) if sxx is not None else None
        if levels is None:
            return
        zmin, zmax = levels
        for spin, value in ((self._vmin_spin, zmin), (self._vmax_spin, zmax)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.app_state.vmin_db = zmin
        self.app_state.vmax_db = zmax
        self._spectrogram.update_levels(zmin, zmax)

    def _update_meters(self, rec: LiveRecorder) -> None:
        for i, bar in enumerate(self._meters):
            peak = float(rec.peak[i]) if i < len(rec.peak) else 0.0
            db = 20 * math.log10(max(peak, 1e-6))
            bar.setValue(int(np.clip(db + 60, 0, 60)))
        clipped = bool(rec.clipped.any())
        self._clip_btn.setStyleSheet("background: #c0392b; color: white; font-weight: bold;" if clipped else "")

    def _rebuild_meters(self, channels: int) -> None:
        for bar in self._meters:
            self._meter_layout.removeWidget(bar)
            bar.deleteLater()
        self._meters = []
        for i in range(channels):
            bar = QProgressBar()
            bar.setRange(0, 60)
            bar.setTextVisible(False)
            bar.setToolTip(f"Ch {i + 1} peak, −60 dBFS to 0")
            bar.setMaximumHeight(12)
            self._meter_layout.addWidget(bar)
            self._meters.append(bar)

    def _update_status(self, rec: LiveRecorder) -> None:
        state = rec.state
        if state == "monitoring":
            n = len(self._recordings)
            self._status_label.setText(f"{self._capture_note}Monitoring. {n} file{'s' if n != 1 else ''} this session.")
        else:
            path = rec.current_path
            self._status_label.setText(
                f"{'Paused' if state == 'paused' else 'Recording'} {path.name if path else ''} — "
                f"{rec.file_duration_s:7.1f} s"
            )

    def _on_reset_clip(self) -> None:
        if self._recorder is not None:
            self._recorder.reset_clip()

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def _set_transport_state(self, state: str) -> None:
        """``idle`` (no stream), ``monitoring``, ``recording`` or ``paused``."""
        self._record_btn.setEnabled(state == "monitoring")
        self._pause_btn.setEnabled(state in ("recording", "paused"))
        self._pause_btn.setText("▶  Resume" if state == "paused" else "⏸  Pause")
        self._stop_btn.setEnabled(state in ("recording", "paused"))
        busy = state in ("recording", "paused")
        for w in (*self._capture_widgets, *self._output_widgets):
            w.setEnabled(not busy)
        self._open_btn.setEnabled(not busy and any(self._folder().glob("*.wav")))

    def _on_record(self) -> None:
        if self._recorder is None:
            return
        self._recorder.folder = self._folder()
        if not self._trace.live:
            self._enter_live_mode()
        try:
            path = self._recorder.record()
        except OSError as e:
            self._status_label.setText(f"Cannot write there: {e}")
            return
        notify(f"Recording {path.name}")
        self._set_transport_state("recording")

    def _on_pause(self) -> None:
        if self._recorder is None:
            return
        if self._recorder.state == "paused":
            self._recorder.resume()
            self._set_transport_state("recording")
        else:
            self._recorder.pause()
            self._set_transport_state("paused")

    def _on_stop(self) -> None:
        if self._recorder is None or self._recorder.state == "monitoring":
            return
        try:
            rec = self._recorder.stop()
        except RuntimeError as e:
            self._status_label.setText(str(e))
            self._set_transport_state("monitoring")
            return
        self._recordings.append(rec)
        self._set_transport_state("monitoring")
        self._enter_file_mode(rec.path)

    # ------------------------------------------------------------------
    # File mode (after Stop)
    # ------------------------------------------------------------------
    def _last_file(self) -> Path | None:
        return self._recordings[-1].path if self._recordings else None

    def _enter_file_mode(self, path: Path | None) -> None:
        """Read the finished WAV through the ordinary buffered path, pan/zoom free."""
        if path is None:
            return
        loader = SharedAudioCache.get_loader(str(path))
        if loader is None:
            return
        source = FileSource("audio", loader, channel=self._shown_channel())
        for plot in (self._trace, self._spectrogram):
            plot.live = False
            plot.set_source(source)
            plot.vb.setMouseEnabled(x=True, y=True)
        extent = source.time_range
        self._trace.vb.setLimits(xMin=extent.start_s, xMax=extent.end_s)
        self._trace.vb.setXRange(extent.start_s, extent.end_s, padding=0)
        self._trace.setLabel("bottom", f"{path.name}", units="s")
        for plot in (self._trace, self._spectrogram):
            plot.update_plot_content(extent.start_s, extent.end_s)
        self._trace.vb.setLimits(xMin=None, xMax=None)
        self._auto_levels()

    # ------------------------------------------------------------------
    # Hand-off + close
    # ------------------------------------------------------------------
    def _on_open_session(self) -> None:
        folder = self._folder()
        wavs = natsort.natsorted(str(p) for p in folder.glob("*.wav"))
        if not wavs:
            return
        self._close_recorder()
        self.accept()
        self._open_session_cb(wavs)

    def closeEvent(self, event) -> None:
        rec = self._recorder
        if rec is not None and rec.state != "monitoring":
            answer = QMessageBox.question(
                self,
                "Recording in progress",
                "Stop the recording and close? The file is finished properly.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        self._close_recorder()
        super().closeEvent(event)
