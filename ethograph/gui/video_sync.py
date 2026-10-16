"""Video playback + time synchronization over the pygfx camera view.

``VideoSync`` keeps the exact public API the rest of the GUI relies on
(``seek_to_frame``, ``play_segment``, ``frame_to_time`` …) but drives a
:class:`~ethograph.gui.pygfx_video.CameraView` instead of napari dims.

Playback is a QTimer stepping trial frames at ``fps_playback``; there is no
render-rate cap — the async decoder drops stale seek requests, so it is the
natural limiter (no forced frame-skipping, no napari dims.play).
"""

import logging
import time
from typing import Any, Optional

from qtpy.QtCore import QObject, QTimer, Signal
from qtpy.QtGui import QGuiApplication

from ethograph.utils.audio import get_audio_sr

logger = logging.getLogger(__name__)

# The frame timer is capped at the display refresh rate (queried at runtime):
# the monitor can't show more distinct frames than that, and it bounds the
# per-frame storm (frame_changed → plot/marker redraws, pose overlay, GL draws)
# so fast playback can't flood the event loop / GL canvas and crash. When
# fps_playback exceeds the refresh rate, playback steps several source frames
# per tick and the async decoder drops the stale ones. Used only if the screen
# refresh rate can't be queried.
RENDER_FPS_FALLBACK = 60.0


class VideoSync(QObject):
    """Video player synchronized with the plots and audio playback."""

    frame_changed = Signal(int)
    playback_stopped = Signal()

    def __init__(
        self,
        app_state,
        view,
        video_source: str,
        audio_source: Optional[str] = None,
    ):
        super().__init__()
        self.app_state = app_state
        self.view = view
        self.video_source = video_source
        self.audio_source = audio_source
        self._time_offset = view.time_offset

        self._segment_end_frame: Optional[int] = None
        # True (sub-frame) end time of the segment being played, when the
        # caller knows it (see play_segment()).
        self._segment_end_time_s: Optional[float] = None
        self._current_frame: int = 0

        # Phase 3: audio-master clock (regular playback). When set, playback is
        # driven from real audio output; the marker sits on the exact clock time
        # via ``marker_time_override`` (read by plots_container).
        self._audio_clock: Any = None
        self._clock_start_t: float = 0.0  # trial-relative seconds at clock start
        self._clock_start_marker: float = 0.0
        self.marker_time_override: Optional[float] = None
        # Smooth mode: decode-paced (synchronous) stepping, every frame, no audio.
        self._smooth_mode: bool = False
        # Session-basis gap run: after the trial's last frame, playback does
        # NOT stop — the marker keeps advancing on a wall clock (video frozen,
        # views blanked by the gap logic) until the auto-follow loads the next
        # trial's video and resumes, or the session ends.
        self._gap_run: bool = False
        self._gap_wall_start: float = 0.0
        self._gap_t0: float = 0.0

        self._play_timer = QTimer()
        self._play_timer.timeout.connect(self._advance)
        # Playback watchdog: pynaviz's render chain can die silently (see
        # install_animate_guard in pygfx_video.py) — audio and the playhead
        # keep moving while the image freezes. While playing, check the
        # view's animate heartbeat and re-arm a stalled chain.
        self._render_watchdog = QTimer()
        self._render_watchdog.setInterval(1000)
        self._render_watchdog.timeout.connect(self._check_render_stall)
        self._stall_logged = False
        self._step: float = 1.0
        self._frame_accum: float = 0.0
        # Cold-decoder bridge: step with synchronous in-process decode until
        # the async worker proves live (see _seek_playback_frame).
        self._sync_until_ready: bool = False

        self.total_frames = view.n_frames
        self.total_duration = self.total_frames / self.fps if self.fps else 0.0
        self.audio_sr = get_audio_sr(audio_source) if audio_source else None

        view.time_changed.connect(self._on_view_time_changed)
        # A mid-playback channel switch rebuilds the audio clock on the new
        # channel (guarded getattr — plain app-state fakes carry no signal).
        self._mic_signal = getattr(app_state, "playback_mic_key_changed", None)
        if self._mic_signal is not None:
            self._mic_signal.connect(self._on_playback_channel_changed)

    # ------------------------------------------------------------------
    # Time mapping
    # ------------------------------------------------------------------
    # frame_to_time / time_to_frame speak the DISPLAY clock (the plot axis):
    # trial-relative in trial basis, session-absolute in session basis. The
    # video itself stays a per-trial decode; the display offset is pulled per
    # call so scope changes need no re-sync (same pattern as PynappleLoader).

    def _display_offset(self) -> float:
        """Shift from trial-relative time to the plot axis's clock."""
        to_display = getattr(self.app_state, "to_display", None)
        if to_display is None:
            return 0.0
        return float(to_display(getattr(self.app_state, "trials_sel", None), 0.0))

    def frame_to_time(self, frame: int) -> float:
        return frame / self.fps + self._time_offset + self._display_offset()

    def time_to_frame(self, time_s: float, *, round_nearest: bool = False) -> int:
        frames = (time_s - self._display_offset() - self._time_offset) * self.fps
        return round(frames) if round_nearest else int(frames)

    @property
    def fps(self) -> float:
        return self.app_state.video_fps

    @property
    def fps_playback(self) -> float:
        """Render-target FPS: native video FPS scaled by ``playback_speed_pct``."""
        return self.fps * self.app_state.playback_speed_pct / 100.0

    @property
    def is_playing(self) -> bool:
        return self._play_timer.isActive()

    @property
    def current_frame(self) -> int:
        return self._current_frame

    # ------------------------------------------------------------------
    # Seeking
    # ------------------------------------------------------------------

    def seek_to_frame(self, frame: int):
        # An explicit seek always shows the frame-exact time — clear any
        # leftover marker override from a prior continuous-end segment finish.
        self.marker_time_override = None
        frame = max(0, min(int(frame), self.total_frames - 1)) if self.total_frames else 0
        # Same cold-worker tolerance as playback: an async seek against a
        # worker that has not yet served a frame is dropped, so decode this
        # one in-process too — an arrow step must always show its frame.
        self.view.seek_trial_frame(frame)
        if not self.view.decoder_ready():
            self.view.seek_trial_frame(frame, synchronous=True)
        # The async path presents only through pynaviz's self-re-arming
        # animate chain; the playback watchdog is the only other thing that
        # restarts a dead one, and it does not run while stopped.
        nudge = getattr(self.view, "nudge_render_if_stalled", None)
        if nudge is not None and nudge():
            logger.warning("Video render loop was stalled; re-armed the draw chain on seek.")
        self._apply_frame(frame)

    def _apply_frame(self, frame: int):
        """Update state + notify listeners (plots, extra cameras)."""
        self._current_frame = frame
        self.app_state.current_frame = frame
        finishing_segment = self._segment_end_frame is not None and frame >= self._segment_end_frame
        if finishing_segment and self._segment_end_time_s is not None:
            # The playhead is always exact: snap the marker to the segment's
            # true (sub-frame) end time rather than the nearest frame's time.
            self.marker_time_override = self._segment_end_time_s
        self.frame_changed.emit(frame)
        if finishing_segment:
            self._segment_end_frame = None
            self.stop(clear_marker_override=self._segment_end_time_s is None)

    def _on_view_time_changed(self, video_time: float):
        """User interacted with the video canvas (scroll/keys)."""
        if self.fps <= 0:
            return
        self.marker_time_override = None
        frame = int(round(video_time * self.fps)) - self.view.start_frame
        frame = max(0, min(frame, self.total_frames - 1)) if self.total_frames else 0
        self._apply_frame(frame)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def start(self):
        if self.is_playing:
            return
        self._segment_end_frame = None
        self._segment_end_time_s = None
        self._audio_clock = None
        self._smooth_mode = False
        self._gap_run = False
        self.marker_time_override = None
        self._sync_until_ready = not self.view.decoder_ready()
        self._render_watchdog.start()

        from .app_constants import PLAYBACK_MODE_SMOOTH, PLAYBACK_MODE_SYNCED

        mode = self.app_state.effective_playback_mode()

        # Audio-synced: drive playback from real audio output so marker/video
        # can't drift. Audio only ever plays in this mode. There is no speed cap
        # — the audio device's max sample rate is the only limit; beyond it the
        # stream simply fails to open and playback falls through silently.
        if mode == PLAYBACK_MODE_SYNCED and self.fps > 0:
            t0, t1 = self._current_frame / self.fps, self.total_frames / self.fps
            clock = self._build_audio_clock(t0, t1)
            if clock is not None and clock.start():
                self._audio_clock = clock
                self._clock_start_t = t0
                self._clock_start_marker = self.frame_to_time(self._current_frame)
                render_fps = min(self.fps_playback, self._render_cap())
                if render_fps > 0:
                    self._play_timer.start(int(1000 / render_fps))
                return
            # synced requested but no usable audio device → fall through silent.

        if mode == PLAYBACK_MODE_SMOOTH:
            # Decode-paced: show every frame in order (may run slower than fps).
            self._smooth_mode = True
            render_fps = min(self.fps_playback, self._render_cap()) if self.fps_playback > 0 else self._render_cap()
            self._play_timer.start(int(1000 / render_fps))
            return

        # Skip frames (default): approximate real-time fps by dropping frames.
        self._start_timer()

    def stop(self, clear_marker_override: bool = True):
        """Stop playback.

        ``clear_marker_override=False`` lets a finishing segment (see
        ``_apply_frame``) leave the marker frozen on the segment's exact end
        time instead of snapping it back to the last frame's time.
        """
        self._gap_run = False
        self._sync_until_ready = False
        self._play_timer.stop()
        self._render_watchdog.stop()
        self._segment_end_frame = None
        self._segment_end_time_s = None
        clock, self._audio_clock = self._audio_clock, None
        if clock is not None:
            clock.stop()
            self._commit_clock_position(clock)
        self._smooth_mode = False
        if clear_marker_override:
            self.marker_time_override = None
        self.playback_stopped.emit()

    def _commit_clock_position(self, clock) -> None:
        """Leave the playhead where the listener last heard the audio.

        Inside the device-latency window the per-tick frame may never have
        advanced (a sub-0.4 s Play → Stop burst), so the clock's final
        position is committed to ``current_frame`` at teardown — the next
        Play continues from there instead of restarting the burst.
        """
        if self.fps <= 0:
            return
        frame = int(round((self._clock_start_t + clock.elapsed_s()) * self.fps))
        frame = max(0, min(frame, self.total_frames - 1)) if self.total_frames else 0
        if frame != self._current_frame:
            self.view.seek_trial_frame(frame)
            self._apply_frame(frame)

    def _on_playback_channel_changed(self, *_args):
        """Rebuild the audio clock when the playback channel moves mid-playback.

        ``_build_audio_clock`` resolves the mic/channel once, at Play — without
        a rebuild the audible channel only changed after Stop and the next
        Play. The clock anchors advance by the old clock's elapsed time, so
        marker and video carry on seamlessly from the same position.
        """
        if self._audio_clock is None or not self.is_playing or self.fps <= 0:
            return
        old, self._audio_clock = self._audio_clock, None
        elapsed = old.elapsed_s()
        old.stop()
        self._clock_start_t += elapsed
        self._clock_start_marker += elapsed
        clock = self._build_audio_clock(self._clock_start_t, self.total_frames / self.fps)
        if clock is not None and clock.start():
            self._audio_clock = clock
            return
        # No usable stream for the new channel → carry on with silent frames.
        self._start_timer()

    def _playback_speed(self) -> float:
        # Single speed lever for both video FPS and audio pitch/rate.
        return self.app_state.playback_speed_pct / 100.0

    def _audio_file_offset(self) -> float:
        """Trial-relative time of the audio file's first sample.

        0.0 for per-trial audio; the stream offset for session-wide files.
        Subtract from a trial-relative time before indexing the audio file.
        """
        sio = getattr(self.app_state, "nwb_alignment", None)
        if sio is None:
            return 0.0
        trial = getattr(self.app_state, "trials_sel", None)
        return float(sio.stream_offset_for_trial(trial, "audio", None) or 0.0)

    def toggle_pause_resume(self):
        self.stop() if self.is_playing else self.start()

    def _render_cap(self) -> float:
        """Max frame-timer rate: the display refresh rate (queried at runtime)."""
        screen = QGuiApplication.primaryScreen()
        rate = screen.refreshRate() if screen is not None else 0.0
        return rate if rate and rate > 0 else RENDER_FPS_FALLBACK

    def _start_timer(self):
        # Fire at min(fps_playback, display refresh); at higher playback speeds
        # step several frames per tick (the async decoder drops the stale ones)
        # rather than flooding the event loop / GL canvas with timer ticks.
        render_fps = min(self.fps_playback, self._render_cap())
        if render_fps <= 0:
            return
        self._step = self.fps_playback / render_fps
        self._frame_accum = float(self._current_frame)
        self._play_timer.start(int(1000 / render_fps))

    # ------------------------------------------------------------------
    # Session-basis gap run (playback across the trial's end)
    # ------------------------------------------------------------------

    def _session_playback(self) -> bool:
        return getattr(self.app_state, "display_basis", "trial") == "session"

    def _session_end(self) -> float | None:
        sc = getattr(self.app_state, "source_collection", None)
        session = sc.session_range if sc is not None else None
        return session.end_s if session is not None else None

    def _end_of_video(self):
        """The trial's video ran out during regular playback.

        Trial basis (or no session extent known): stop, as always. Session
        basis: keep the play timer and advance the MARKER on a wall clock —
        the video freezes on its last frame (the gap logic blanks the views),
        and the auto-follow picks the marker up in the next trial's span,
        loads that video and resumes playback.
        """

        if not self._session_playback() or self._session_end() is None:
            self.stop()
            return
        self._gap_run = True
        self._gap_t0 = self.frame_to_time(self._current_frame)
        self._gap_wall_start = time.perf_counter()
        if self._audio_clock is not None:
            self._audio_clock.stop()
            self._audio_clock = None
        self._smooth_mode = False
        if not self._play_timer.isActive():
            self._play_timer.start(int(1000 / self._render_cap()))

    def _advance_gap(self):

        speed = self._playback_speed()
        t = self._gap_t0 + (time.perf_counter() - self._gap_wall_start) * speed
        end = self._session_end()
        if end is None or t >= end:
            self.stop()
            return
        # Marker advances on the exact clock time; the video stays on its
        # last frame (plots_container reads the override for the marker).
        self.marker_time_override = t
        self._apply_frame(self._current_frame)

    def _check_render_stall(self):
        """Watchdog tick: re-arm the view's render chain if its heartbeat is
        stale, logging once per stall (not once per second of one stall)."""
        if not self.is_playing:
            self._render_watchdog.stop()
            return
        nudge = getattr(self.view, "nudge_render_if_stalled", None)
        if nudge is None:
            return
        if nudge():
            if not self._stall_logged:
                logger.warning("Video render loop stalled during playback; re-armed the draw chain.")
                self._stall_logged = True
        else:
            self._stall_logged = False

    def _seek_playback_frame(self, frame: int):
        """Seek during playback, tolerating a cold decode worker.

        A freshly spawned worker (trial/camera switch) drops async seeks for
        ~2 s while it re-imports its stack — playback started against it used
        to render nothing and jump to the segment end. Until the worker proves
        live, each tick decodes its frame synchronously in-process so every
        tick renders, and still queues one async request so the waking
        worker's first served frame is current; the tick after it answers
        hands over to the pure async path.
        """
        if self._sync_until_ready:
            if self.view.decoder_ready():
                self._sync_until_ready = False
            else:
                self.view.seek_trial_frame(frame)
                self.view.seek_trial_frame(frame, synchronous=True)
                return
        self.view.seek_trial_frame(frame)

    def _advance(self):
        if self._gap_run:
            self._advance_gap()
            return
        if self._audio_clock is not None:
            self._advance_from_clock()
            return
        if self._smooth_mode:
            self._advance_smooth()
            return
        self._frame_accum += self._step
        next_frame = int(round(self._frame_accum))
        max_frame = self.total_frames - 1
        if self._segment_end_frame is not None:
            max_frame = min(max_frame, self._segment_end_frame)
        if next_frame >= max_frame:
            next_frame = max_frame
            self._seek_playback_frame(next_frame)
            self._apply_frame(next_frame)
            if self._segment_end_frame is None:
                self._end_of_video()
            return
        self._seek_playback_frame(next_frame)
        self._apply_frame(next_frame)

    def _advance_smooth(self):
        """Decode-paced stepping: advance one frame, decoding it synchronously so
        no frame is ever skipped. Under load this runs slower than real-time
        instead of dropping frames (the tradeoff for a smooth review pass)."""
        next_frame = self._current_frame + 1
        max_frame = self.total_frames - 1
        if self._segment_end_frame is not None:
            max_frame = min(max_frame, self._segment_end_frame)
        next_frame = min(next_frame, max_frame)
        self.view.seek_trial_frame(next_frame, synchronous=True)
        self._apply_frame(next_frame)
        if next_frame >= max_frame and self._segment_end_frame is None:
            self._end_of_video()

    def _advance_from_clock(self):
        """Drive video + marker from the audio clock's real playback position."""
        clock = self._audio_clock
        elapsed = clock.elapsed_s()
        max_frame = self.total_frames - 1
        segment_end = self._segment_end_frame
        if segment_end is not None:
            max_frame = min(max_frame, segment_end)
        frame = min(int(round((self._clock_start_t + elapsed) * self.fps)), max_frame)
        if segment_end is not None and clock.finished:
            # The clock's span ends at the segment bounds; make sure the
            # finishing logic in _apply_frame runs even if frame rounding
            # left us one frame short of the segment's end frame.
            frame = segment_end
        # The playhead sits on the exact (sub-frame) clock time; plots_container
        # reads this override so it isn't quantized to the displayed frame.
        self.marker_time_override = self._clock_start_marker + elapsed
        self._seek_playback_frame(frame)
        self._apply_frame(frame)
        if segment_end is not None:
            # _apply_frame finishes the segment itself (exact-end marker +
            # stop); _end_of_video here would wipe that marker override.
            return
        if clock.finished or frame >= max_frame:
            self._end_of_video()

    def _build_audio_clock(self, t0_s: float, t1_s: float):
        """Preload the selected audio span and wrap it in an :class:`AudioClock`.

        Returns ``None`` when audio is unavailable so the caller falls back to
        the frame timer.
        """
        from .audio_clock import AudioClock

        resolved, channel_idx = self.app_state.get_audio_source(self.app_state.playback_mic_selection())
        audio_path = resolved or self.app_state.audio_path or self.audio_source
        if not audio_path:
            return None
        try:
            from audioio import AudioLoader
        except ImportError:
            return None
        try:
            from ..io.audio_extract import resolve_audio_path

            file_off = self._audio_file_offset()
            with AudioLoader(resolve_audio_path(audio_path)) as data:
                fs = data.rate
                s0 = max(0, int((t0_s - file_off) * fs))
                s1 = min(len(data), int((t1_s - file_off) * fs))
                if s1 <= s0:
                    return None
                segment = data[s0:s1]
            if segment.ndim > 1:
                segment = segment[:, min(channel_idx, segment.shape[1] - 1)]
            return AudioClock(segment, fs, speed=self._playback_speed())
        except Exception:
            logger.warning("Could not build audio clock; falling back.", exc_info=True)
            return None

    def play_segment(
        self,
        start_frame: int,
        end_frame: int,
        exact_t0: float | None = None,
        exact_t1: float | None = None,
    ):
        """Play frames ``[start_frame, end_frame]`` with synchronized audio.

        ``exact_t0``/``exact_t1`` are the segment's true (sub-frame) bounds on
        the **display clock**, like every other public time on this class. Given
        them, audio is sliced from them rather than from the video frame grid
        and the red marker stops on ``exact_t1`` itself; the video still shows
        the nearest frames — see docs/advanced/playback.md.
        """
        self.stop()

        start_frame = max(0, min(int(start_frame), self.total_frames - 1))
        end_frame = max(0, min(int(end_frame), self.total_frames - 1))
        if end_frame <= start_frame:
            end_frame = min(start_frame + 1, self.total_frames - 1)

        self._segment_end_frame = end_frame
        # The marker override is read as a display-clock time, which is what
        # the caller handed us.
        self._segment_end_time_s = exact_t1
        # ASYNC initial seek, exactly like regular Play (space): a synchronous
        # decode here blocks the GUI on a far seek, which is the gap the user
        # feels between navigating and playback starting. The first timer tick
        # renders the frame instead (synchronously only while the worker is
        # cold — see _seek_playback_frame).
        self.view.seek_trial_frame(start_frame)
        self._apply_frame(start_frame)
        self._segment_end_frame = end_frame  # _apply_frame may have cleared it
        self._sync_until_ready = not self.view.decoder_ready()
        self._render_watchdog.start()

        if self.fps > 0:
            # _build_audio_clock indexes the audio file on the trial clock.
            audio_t0 = None if exact_t0 is None else self._trial_time(exact_t0)
            audio_t1 = None if exact_t1 is None else self._trial_time(exact_t1)
            if audio_t0 is None or audio_t1 is None:
                audio_t0, audio_t1 = start_frame / self.fps, end_frame / self.fps
            # Same DAC-anchored AudioClock as regular Play (space): segment
            # playback gets the identical loudness, volume control, speed
            # resampling and drift-free marker. (The old audioio PlayAudio
            # path peak-normalized the segment, so V-playback sounded louder
            # than regular playback of the same audio.)
            clock = self._build_audio_clock(audio_t0, audio_t1)
            if clock is not None and clock.start():
                self._audio_clock = clock
                self._clock_start_t = audio_t0
                self._clock_start_marker = exact_t0 if exact_t0 is not None else self.frame_to_time(start_frame)
                render_fps = min(self.fps_playback, self._render_cap())
                if render_fps > 0:
                    self._play_timer.start(int(1000 / render_fps))
                return
            # No usable audio device/span → silent frame-timer playback.

        self._start_timer()

    def _trial_time(self, t_display: float) -> float | None:
        """Display-clock second → trial-relative second (``None`` if unresolved)."""
        resolved = self.app_state.from_display(t_display)
        return None if resolved is None else float(resolved[1])

    def cleanup(self):
        try:
            self.view.time_changed.disconnect(self._on_view_time_changed)
        except (RuntimeError, TypeError):
            pass
        if self._mic_signal is not None:
            try:
                self._mic_signal.disconnect(self._on_playback_channel_changed)
            except (RuntimeError, TypeError):
                pass
            self._mic_signal = None
        self._play_timer.stop()
        self._render_watchdog.stop()
        if self._audio_clock is not None:
            # Teardown mid-playback (trial/camera change) must close the
            # output stream too, or the old span keeps sounding over the new.
            self._audio_clock.stop()
            self._audio_clock = None
        self._segment_end_frame = None
        self.playback_stopped.emit()
