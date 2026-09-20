"""Bottom playback control bar: play/pause, time slider, FPS, trial navigation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import QRectF, QSize, Qt
from qtpy.QtGui import QColor, QIcon, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ethograph.datasets import is_template_path
from ethograph.utils.ffmpeg import ffmpeg_available

from .app_constants import BOTTOM_BAR_MIN_WIDTH_PX, PLAYBACK_MODE_CHOICES
from .audio_clock import set_master_volume, volume_pct_to_gain

if TYPE_CHECKING:
    from ethograph.gui.app_state import ObservableAppState

logger = logging.getLogger(__name__)

_TIMEBAR_RESOLUTION = 1000


def _playback_icon(kind: str, color: str = "#e6e6e6") -> QIcon:
    """Render a crisp antialiased play/pause icon (font glyphs look bad on Windows)."""
    s = 64  # oversampled; QIcon scales down smoothly
    pm = QPixmap(s, s)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    # Stroke with a round-join pen to give the shapes softly rounded corners.
    painter.setPen(QPen(c, s * 0.09, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(c)
    if kind == "play":
        path = QPainterPath()
        path.moveTo(s * 0.34, s * 0.24)
        path.lineTo(s * 0.78, s * 0.50)
        path.lineTo(s * 0.34, s * 0.76)
        path.closeSubpath()
        painter.drawPath(path)
    else:
        bar_w = s * 0.13
        painter.drawRoundedRect(QRectF(s * 0.30, s * 0.26, bar_w, s * 0.48), 2, 2)
        painter.drawRoundedRect(QRectF(s * 0.57, s * 0.26, bar_w, s * 0.48), 2, 2)
    painter.end()
    return QIcon(pm)


def _speaker_icon(color: str = "#e6e6e6") -> QPixmap:
    """Render a small antialiased speaker glyph for the playback-audio indicator."""
    s = 64
    pm = QPixmap(s, s)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    painter.setBrush(c)
    painter.setPen(QPen(c, s * 0.06, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    cone = QPainterPath()
    cone.moveTo(s * 0.12, s * 0.40)
    cone.lineTo(s * 0.26, s * 0.40)
    cone.lineTo(s * 0.44, s * 0.24)
    cone.lineTo(s * 0.44, s * 0.76)
    cone.lineTo(s * 0.26, s * 0.60)
    cone.lineTo(s * 0.12, s * 0.60)
    cone.closeSubpath()
    painter.drawPath(cone)
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(c, s * 0.07, Qt.SolidLine, Qt.RoundCap))
    painter.drawArc(QRectF(s * 0.40, s * 0.32, s * 0.26, s * 0.36), -60 * 16, 120 * 16)
    painter.drawArc(QRectF(s * 0.40, s * 0.22, s * 0.42, s * 0.56), -55 * 16, 110 * 16)
    painter.end()
    return pm


class _InteractiveSlider(QSlider):
    """QSlider that tracks mouse interaction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_dragging = False

    def mousePressEvent(self, event: QMouseEvent):
        self.is_dragging = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.is_dragging = False
        super().mouseReleaseEvent(event)


class BottomBarScrollHost(QScrollArea):
    """Horizontally scrollable host for the playback bar.

    The bar packs ~900 px of controls, so as a bare dock widget its minimum
    width pinned the whole window layout: on a small screen there was no slack
    left, and the right sidebar's separator could not be dragged at all. Inside
    a scroll area the bar keeps its natural width, the dock's minimum width
    becomes small, and narrow windows scroll to reach the far controls.
    """

    def __init__(self, bar: QWidget, parent=None):
        super().__init__(parent)
        self._bar = bar
        self.setWidget(bar)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(BOTTOM_BAR_MIN_WIDTH_PX)
        self.setStyleSheet("QScrollArea { background-color: #2a2a2a; border: none; }")
        self.horizontalScrollBar().rangeChanged.connect(lambda *_: self._sync_height())
        self._sync_height()

    def _sync_height(self):
        """Grow by the scrollbar's height only while a scrollbar is shown."""
        sb = self.horizontalScrollBar()
        extent = sb.sizeHint().height() if sb.maximum() > 0 else 0
        # minimumHeight, not height(): the bar's fixed height is set in its
        # constructor, long before it is first laid out.
        bar_h = self._bar.minimumHeight() or self._bar.sizeHint().height()
        self.setFixedHeight(bar_h + extent)


class BottomPlaybackBar(QWidget):
    """Bottom control bar with play/pause, time slider, FPS, and trial navigation."""

    def __init__(self, app_state: ObservableAppState, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.setStyleSheet("QWidget { background-color: #2a2a2a; color: white; padding: 4px; }")
        self.setFixedHeight(64)

        outer = QHBoxLayout(self)
        outer.setSpacing(8)
        outer.setContentsMargins(8, 4, 8, 4)

        # Add-panel button spans both rows on the left (opens SourcePopup).
        self.add_panel_btn = QPushButton("➕ Add\npanel")
        self.add_panel_btn.setFixedWidth(90)
        self.add_panel_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.add_panel_btn.setToolTip("Add a panel: drag a source onto the plot area, or press Enter  (Shift+N)")
        outer.addWidget(self.add_panel_btn)

        rows = QVBoxLayout()
        rows.setSpacing(4)
        outer.addLayout(rows, stretch=1)

        # ── Top row: play/pause + time slider ────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)
        self._play_icon = _playback_icon("play")
        self._pause_icon = _playback_icon("pause")
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setIcon(self._play_icon)
        self.play_pause_btn.setIconSize(QSize(16, 16))
        self.play_pause_btn.setFixedWidth(36)
        self.play_pause_btn.setToolTip("Play / Pause  (Space)")
        self.play_pause_btn.clicked.connect(self._on_play_pause_clicked)
        top.addWidget(self.play_pause_btn)

        self.time_slider = _InteractiveSlider(Qt.Horizontal)
        self.time_slider.setRange(0, _TIMEBAR_RESOLUTION)
        self.time_slider.setMinimumHeight(18)
        self.time_slider.setTracking(True)
        self.time_slider.valueChanged.connect(self._on_slider_value_changed)
        top.addWidget(self.time_slider, stretch=1)
        rows.addLayout(top)

        # ── Bottom row: audio channel · mode · FPS · toggles · trials
        bot = QHBoxLayout()
        bot.setSpacing(8)

        # Playback-audio indicator (channel Play will sound); hidden if silent.
        self._audio_ind_icon = QLabel()
        self._audio_ind_icon.setPixmap(_speaker_icon().scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._audio_ind_label = QLabel()
        self._audio_ind_label.setStyleSheet("color: #b8b8b8;")
        for w in (self._audio_ind_icon, self._audio_ind_label):
            w.setToolTip("Channel used for audio playback (follows the last-clicked audio panel)")
            w.hide()
            bot.addWidget(w)

        # Playback mode
        self.playback_mode_combo = QComboBox()
        for label_text, mode_value in PLAYBACK_MODE_CHOICES:
            self.playback_mode_combo.addItem(label_text, mode_value)
        self.playback_mode_combo.setToolTip(
            "Audio-synced: audio + video locked (may drop frames).\n"
            "Smooth: every video frame, may run slower than the set speed, no audio.\n"
            "Real-time (skip frames): approximate the set speed by dropping frames, no audio."
        )
        self.playback_mode_combo.currentIndexChanged.connect(self._on_playback_mode_changed)
        bot.addWidget(self.playback_mode_combo)

        # Playback speed as a % of the original recording (drives video FPS and
        # audio pitch/rate together — see _update_speed_info for the derived
        # fps/kHz readout).
        bot.addWidget(QLabel("Speed:"))
        self.speed_display = QLineEdit()
        self.speed_display.setFixedWidth(46)
        self.speed_display.setText(f"{app_state.get_with_default('playback_speed_pct'):.0f}")
        self.speed_display.setToolTip(
            "Playback speed as a % of the original recording. 100% = native speed.\n"
            "Scales both the video frame rate and the audio pitch/rate together —\n"
            "e.g. 50% plays at half speed, one octave lower."
        )
        self.speed_display.editingFinished.connect(self._on_speed_pct_changed)
        bot.addWidget(self.speed_display)
        bot.addWidget(QLabel("%"))

        self.speed_info_label = QLabel()
        self.speed_info_label.setStyleSheet("color: #999999;")
        self.speed_info_label.setToolTip("Effective frame rate / audio sample rate at the current speed setting.")
        bot.addWidget(self.speed_info_label)

        # Playback volume: gain inside Ethograph's own output stream, so it is
        # independent of the system/device volume. Applied live mid-playback.
        volume_tooltip = (
            "Playback volume within Ethograph (independent of the system volume).\n"
            "Logarithmic scale, like a mixing-desk fader: equal slider steps sound\n"
            "like equal loudness steps (50% ≈ −18 dB, 100% = the recording's own level)."
        )
        self._volume_icon = QLabel()
        self._volume_icon.setPixmap(_speaker_icon().scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._volume_icon.setToolTip(volume_tooltip)
        bot.addWidget(self._volume_icon)
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setFixedWidth(70)
        self.volume_slider.setToolTip(volume_tooltip)
        volume_pct = float(app_state.get_with_default("playback_volume_pct"))
        self.volume_slider.setValue(int(round(volume_pct)))
        set_master_volume(volume_pct_to_gain(volume_pct))
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        bot.addWidget(self.volume_slider)

        # Center playback + Hide label text
        self.center_playback_cb = QCheckBox("Center")
        self.center_playback_cb.setToolTip("Keep the playhead centered in the view during playback")
        self.center_playback_cb.setChecked(bool(app_state.get_with_default("center_playback")))
        self.center_playback_cb.toggled.connect(lambda v: setattr(self.app_state, "center_playback", v))
        bot.addWidget(self.center_playback_cb)

        # Low-res proxy decoding for smooth navigation (generated on first use).
        self.proxy_cb = QCheckBox("Proxy")
        self.proxy_cb.setChecked(app_state.get_with_default("video_quality_mode") == "proxy")
        self.proxy_cb.toggled.connect(self._on_proxy_toggled)
        self._update_proxy_checkbox()
        bot.addWidget(self.proxy_cb)

        bot.addStretch()

        # Trial navigation cluster: ◀ Trial <id> (i/n) ▶
        nav_style = """
            QPushButton {
                background-color: transparent;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                color: #dddddd;
                font-size: 13px;
                padding: 0px;
            }
            QPushButton:hover { background-color: #3d3d3d; border-color: #5a5a5a; }
            QPushButton:pressed { background-color: #505050; }
            QPushButton:disabled { color: #555555; border-color: #383838; }
        """

        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedSize(28, 22)
        self.prev_btn.setStyleSheet(nav_style)
        self.prev_btn.setToolTip("Previous trial")
        self.prev_btn.setFocusPolicy(Qt.NoFocus)
        self.prev_btn.clicked.connect(self._on_prev_trial)
        bot.addWidget(self.prev_btn)

        self.trial_label = QLabel("Trial - / -")
        self.trial_label.setMinimumWidth(100)
        self.trial_label.setAlignment(Qt.AlignCenter)
        bot.addWidget(self.trial_label)

        # Whose labels a click places — the last clicked panel's individual.
        # Only worth saying when there is more than one animal to choose.
        self.subject_label = QLabel()
        self.subject_label.setObjectName("labelling_subject_label")
        self.subject_label.setStyleSheet("color: #e3b341; font-weight: bold;")
        self.subject_label.setToolTip(
            "The individual a new label is about: the pin of the panel you last clicked, else the sidebar's"
        )
        self.subject_label.hide()
        bot.addWidget(self.subject_label)

        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedSize(28, 22)
        self.next_btn.setStyleSheet(nav_style)
        self.next_btn.setToolTip("Next trial")
        self.next_btn.setFocusPolicy(Qt.NoFocus)
        self.next_btn.clicked.connect(self._on_next_trial)
        bot.addWidget(self.next_btn)
        rows.addLayout(bot)

        # Wire app_state signals. trials_sel is a dynamic *_sel attribute with
        # no auto-generated signal — trial changes are announced via trial_changed.
        # The slider position itself follows time_marker_updated (see
        # set_data_widget), not current_frame — one time-based mapping both ways.
        app_state.playback_speed_pct_changed.connect(self._update_speed_display)
        app_state.trial_changed.connect(self._update_trial_label)
        app_state.curation_changed.connect(self._update_trial_label)
        app_state.labelling_subject_changed.connect(self._update_subject_label)
        app_state.trial_changed.connect(self._update_audio_indicator)
        app_state.trial_changed.connect(self._update_playback_mode_combo)
        app_state.trial_changed.connect(self._update_speed_info)
        app_state.playback_mic_key_changed.connect(self._update_audio_indicator)
        app_state.playback_mic_key_changed.connect(self._update_speed_info)
        if hasattr(app_state, "ready_changed"):
            app_state.ready_changed.connect(self._update_trial_label)
            app_state.ready_changed.connect(self._update_audio_indicator)
            app_state.ready_changed.connect(self._update_playback_mode_combo)
            app_state.ready_changed.connect(self._update_speed_info)
            app_state.ready_changed.connect(self._update_proxy_checkbox)

        self._update_trial_label()
        self._update_audio_indicator()
        self._update_playback_mode_combo()
        self._update_speed_info()
        self._sync_play_icon()

    def _on_playback_mode_changed(self, index: int):
        if index < 0:
            return
        self.app_state.playback_mode = self.playback_mode_combo.itemData(index)

    def _update_playback_mode_combo(self):
        """Reflect the effective mode (auto follows audio presence)."""
        effective = self.app_state.effective_playback_mode()
        idx = self.playback_mode_combo.findData(effective)
        if idx >= 0:
            self.playback_mode_combo.blockSignals(True)
            self.playback_mode_combo.setCurrentIndex(idx)
            self.playback_mode_combo.blockSignals(False)

    _PROXY_TOOLTIP = (
        "Play video from a smaller, low-resolution copy so moving through\n"
        "the video stays smooth on large or high-resolution recordings.\n"
        "\n"
        "The copy is generated in the background the first time you enable it\n"
        "(a ⏳ badge shows on the video panel), then reused. It has the same\n"
        "frame rate and length as the original, so labels and timing stay\n"
        "exactly aligned.\n"
        "\n"
        "Copies are cached in a '.ethograph_proxies' folder next to each\n"
        "video and reused across sessions. Uncheck to play the original\n"
        "full-resolution video."
    )
    _PROXY_NO_FFMPEG_TOOLTIP = (
        "Proxy generation requires ffmpeg (optional). Video plays at full\n"
        'resolution without it. Install with: uv pip install "ethograph[proxy]",\n'
        "or conda install -c conda-forge ffmpeg."
    )
    _PROXY_TEMPLATE_TOOLTIP = (
        "Template datasets ship small, already low-resolution videos —\n"
        "a proxy copy would not make navigation any smoother."
    )

    def _update_proxy_checkbox(self):
        """Enable the proxy toggle only where a proxy can and should be built."""
        if not ffmpeg_available():
            self.proxy_cb.setChecked(False)
            self.proxy_cb.setEnabled(False)
            self.proxy_cb.setToolTip(self._PROXY_NO_FFMPEG_TOOLTIP)
        elif self._is_template_dataset():
            self.proxy_cb.setChecked(False)
            self.proxy_cb.setEnabled(False)
            self.proxy_cb.setToolTip(self._PROXY_TEMPLATE_TOOLTIP)
        else:
            self.proxy_cb.setEnabled(True)
            self.proxy_cb.setToolTip(self._PROXY_TOOLTIP)

    def _is_template_dataset(self) -> bool:
        """True when the loaded data comes from the downloaded templates tree."""
        return any(is_template_path(getattr(self.app_state, attr, None)) for attr in ("nc_file_path", "nwb_file_path"))

    def _on_proxy_toggled(self, checked: bool):
        """Switch video decode quality (full-res ⇄ low-res proxy)."""
        dw = getattr(self, "_data_widget", None)
        if dw is not None and hasattr(dw, "set_video_quality"):
            dw.set_video_quality(checked)
        else:
            self.app_state.video_quality_mode = "proxy" if checked else "full"

    def _on_volume_changed(self, value: int):
        """Apply the output gain live and persist it (gui_settings.yaml)."""
        set_master_volume(volume_pct_to_gain(value))
        self.app_state.playback_volume_pct = float(value)

    def _update_audio_indicator(self):
        """Show the channel Play will sound, or hide the indicator when silent."""
        label = self.app_state.playback_audio_label()
        if label:
            self._audio_ind_label.setText(label)
            tooltip = self.app_state.playback_audio_tooltip() or ""
            for w in (self._audio_ind_icon, self._audio_ind_label):
                w.setToolTip(tooltip)
        self._audio_ind_icon.setVisible(bool(label))
        self._audio_ind_label.setVisible(bool(label))

    def _on_play_pause_clicked(self):
        """Toggle playback."""
        if hasattr(self, "_data_widget") and self._data_widget is not None:
            self._data_widget.toggle_pause_resume()
        self._sync_play_icon()

    def _sync_play_icon(self):
        """Update button icon based on playback state."""
        video = getattr(self.app_state, "video", None)
        if video is not None:
            is_playing = video.is_playing
        else:
            audio_player = self._audio_player()
            is_playing = audio_player.playing if audio_player else False
        self.play_pause_btn.setIcon(self._pause_icon if is_playing else self._play_icon)

    def _audio_player(self):
        """No-video playback controller (marker-driven), if available."""
        data_widget = getattr(self, "_data_widget", None)
        plot_container = getattr(data_widget, "plot_container", None)
        return getattr(plot_container, "audio_player", None)

    def _on_slider_value_changed(self):
        """Handle slider value changes — only seek if user is dragging.

        The slider is time-based: its position maps onto ``padded_bounds``
        and seeks the time marker directly, video or not. A loaded video is
        seeked to the matching frame as a side effect.
        """
        if not self.time_slider.is_dragging:
            return
        tr = self.app_state.padded_bounds
        if tr is None or tr.duration <= 0:
            return
        time_s = tr.start_s + self.time_slider.value() / _TIMEBAR_RESOLUTION * tr.duration
        fixed = self.app_state.get_with_default("xlim_mode") == "fixed"
        video = self.app_state.video
        if fixed:
            # Slide BEFORE seeking: the frame_changed handler then finds the
            # marker inside the new window and skips its center-scroll,
            # avoiding a second conflicting setXRange per slider tick.
            self._slide_fixed_window(time_s, move_marker=video is None)
        if video is not None and self.app_state.num_frames > 0:
            video.seek_to_frame(video.time_to_frame(time_s))
        elif not fixed:
            self._seek_marker(time_s)

    def _seek_marker(self, time_s: float):
        """Move the time marker (no video): update it and keep it in view."""
        data_widget = getattr(self, "_data_widget", None)
        plot_container = getattr(data_widget, "plot_container", None)
        if plot_container is not None:
            plot_container._on_slider_time(time_s)

    def _slide_fixed_window(self, t0: float, move_marker: bool = False):
        """Fixed x-limits mode: the slider moves the window's start (t0)."""
        data_widget = getattr(self, "_data_widget", None)
        plot_container = getattr(data_widget, "plot_container", None)
        if plot_container is None:
            return
        span = self.app_state.view_span
        tr = self.app_state.padded_bounds
        if tr is not None:
            t0 = max(tr.start_s, min(t0, tr.end_s - span))
        master = getattr(plot_container, "_xlink_master", None) or getattr(plot_container, "_feature_plot", None)
        if master is not None:
            master.vb.setXRange(t0, t0 + span, padding=0)
        if move_marker:
            plot_container.update_time_marker_by_time(t0)

    def _update_slider_from_time(self, time_s: float):
        """Follow the time marker: every marker move (playback, seek, click)
        emits ``time_marker_updated``; map that time onto the slider."""
        if self.time_slider.is_dragging:
            return
        tr = self.app_state.padded_bounds
        if tr is None or tr.duration <= 0:
            return
        frac = (time_s - tr.start_s) / tr.duration
        self._set_slider_silently(int(max(0.0, min(1.0, frac)) * _TIMEBAR_RESOLUTION))

    def _set_slider_silently(self, slider_pos: int):
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(slider_pos)
        self.time_slider.blockSignals(False)

    def _update_speed_display(self):
        """Reflect ``playback_speed_pct`` in the speed field."""
        pct = self.app_state.playback_speed_pct
        self.speed_display.blockSignals(True)
        self.speed_display.setText(f"{pct:.0f}")
        self.speed_display.blockSignals(False)
        self._update_speed_info()

    def _on_speed_pct_changed(self):
        """Handle speed % field edit."""
        text = self.speed_display.text().strip()
        if not text:
            return
        try:
            pct = float(text)
            if pct > 0:
                self.app_state.playback_speed_pct = pct
            else:
                self.speed_display.setText(f"{self.app_state.playback_speed_pct:.0f}")
        except ValueError:
            self.speed_display.setText(f"{self.app_state.playback_speed_pct:.0f}")
        self._update_speed_info()

    def _native_audio_rate(self) -> float | None:
        """Native sample rate (Hz) of the audio that will actually play, if any."""
        audio_path, _ = self.app_state.get_audio_source(self.app_state.playback_mic_selection())
        audio_path = audio_path or getattr(self.app_state, "audio_path", None)
        if not audio_path:
            return None
        from .plots_spectrogram import SharedAudioCache

        loader = SharedAudioCache.get_loader(audio_path)
        return float(loader.rate) if loader is not None else None

    def _update_speed_info(self):
        """Show the derived fps / sample-rate readout at the current speed %.

        Purely informational — playback always resamples to a fixed output
        rate (see ``audio_clock.OUTPUT_RATE``), so this never clips against
        the sound device's max sample rate, even for high-rate (ultrasonic)
        recordings or very slow/fast speeds.
        """
        pct = self.app_state.playback_speed_pct
        parts = []
        native_fps = getattr(self.app_state, "video_fps", None)
        if native_fps:
            parts.append(f"{native_fps * pct / 100.0:.1f} fps")
        native_sr = self._native_audio_rate()
        if native_sr:
            parts.append(f"{native_sr * pct / 100.0 / 1000.0:.1f} kHz")
        self.speed_info_label.setText(f"({', '.join(parts)})" if parts else "")

    def _update_trial_label(self):
        """Update trial label with the actual trial ID plus positional counter.

        Coloured like the navigation combo: green when every label of the
        trial is manual or curated, red while some are still a model's
        unreviewed output (``labels/curation.py``).
        """
        trials = getattr(self.app_state, "trials", None)
        trials_sel = getattr(self.app_state, "trials_sel", None)
        if trials and trials_sel is not None:
            try:
                idx = trials.index(trials_sel)
                self.trial_label.setText(f"Trial {trials_sel} ({idx + 1}/{len(trials)})")
                self.prev_btn.setEnabled(idx > 0)
                self.next_btn.setEnabled(idx < len(trials) - 1)
            except (ValueError, IndexError):
                self.trial_label.setText(f"Trial {trials_sel}")
                self.prev_btn.setEnabled(True)
                self.next_btn.setEnabled(True)
            curated = self.app_state.trial_is_curated(trials_sel)
            self.trial_label.setStyleSheet(f"color: {'#7ee787' if curated else '#ff7b72'}; font-weight: bold;")
            self.trial_label.setToolTip(
                "Every label of this trial is manual or curated"
                if curated
                else "Some labels of this trial are still automated (not yet curated)"
            )
        else:
            self.trial_label.setText("Trial - / -")
            self.trial_label.setStyleSheet("")
            self.trial_label.setToolTip("")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

    def _update_subject_label(self, *_):
        subject = getattr(self.app_state, "labelling_subject", None)
        names = self.app_state.label_individuals() if getattr(self.app_state, "ready", False) else []
        show = bool(subject) and len(names) > 1
        self.subject_label.setText(f"labelling: {subject}" if show else "")
        self.subject_label.setVisible(show)

    def _on_prev_trial(self):
        """Navigate to previous trial."""
        trials = getattr(self.app_state, "trials", None)
        if not trials:
            return
        try:
            curr_idx = trials.index(self.app_state.trials_sel)
        except (ValueError, IndexError):
            return
        if curr_idx > 0:
            new_trial = trials[curr_idx - 1]
            self.app_state.trials_sel = new_trial
            self.app_state.trial_changed.emit()

    def _on_next_trial(self):
        """Navigate to next trial."""
        trials = getattr(self.app_state, "trials", None)
        if not trials:
            return
        try:
            curr_idx = trials.index(self.app_state.trials_sel)
        except (ValueError, IndexError):
            return
        if curr_idx < len(trials) - 1:
            new_trial = trials[curr_idx + 1]
            self.app_state.trials_sel = new_trial
            self.app_state.trial_changed.emit()

    def set_data_widget(self, data_widget):
        """Reference to DataWidget for toggling playback."""
        self._data_widget = data_widget
        plot_container = getattr(data_widget, "plot_container", None)
        if plot_container is not None:
            plot_container.time_marker_updated.connect(self._update_slider_from_time)
            plot_container.audio_player.on_state_changed = self._sync_play_icon

    def connect_video_sync(self, sync):
        """Connect to video sync to monitor playback state."""
        if getattr(self, "_connected_sync", None) is sync:
            return
        if getattr(self, "_connected_sync", None) is not None:
            try:
                self._connected_sync.playback_stopped.disconnect(self._sync_play_icon)
            except (RuntimeError, TypeError):
                pass
        sync.playback_stopped.connect(self._sync_play_icon)
        self._connected_sync = sync
