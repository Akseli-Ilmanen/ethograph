"""Playback over a cold decode worker steps synchronously until it warms up.

Regression cover for auto-play on navigate: right after a trial/camera switch
the pynaviz decode worker may still be spawning (~2 s on Windows) and drops
async seeks — playback used to render nothing and jump to the segment end.
``VideoSync`` now starts immediately but routes seeks through
``_seek_playback_frame``: while ``view.decoder_ready()`` is False every tick
decodes its frame synchronously in-process (plus one async request so the
waking worker's first serve is current); once the worker answers, seeks hand
over to the pure async path.
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication

from ethograph.gui.video_sync import VideoSync

FPS = 25.0


class _FakeSignal:
    def connect(self, *_args, **_kwargs):
        pass

    def disconnect(self, *_args, **_kwargs):
        pass


class _FakeView:
    """CameraView stand-in: records seeks, answers the readiness check."""

    def __init__(self, ready: bool = False):
        self.time_offset = 0.0
        self.n_frames = 100
        self.time_changed = _FakeSignal()
        self.ready = ready
        self.seeks: list[tuple[int, bool]] = []

    def seek_trial_frame(self, frame: int, synchronous: bool = False):
        self.seeks.append((frame, synchronous))

    def decoder_ready(self) -> bool:
        return self.ready


class _FakeAppState:
    video_fps = FPS
    playback_speed_pct = 100.0
    current_frame = 0
    audio_path = None

    def get_audio_source(self, _sel):
        return None, 0

    def playback_mic_selection(self):
        return None


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _make_sync(qapp, ready: bool) -> VideoSync:
    return VideoSync(_FakeAppState(), _FakeView(ready=ready), video_source="fake.mp4")


def test_cold_worker_steps_synchronously(qapp):
    sync = _make_sync(qapp, ready=False)
    try:
        sync.play_segment(10, 50)
        assert sync._sync_until_ready
        assert sync._play_timer.isActive()  # playback starts immediately
        # The initial seek is ASYNC, like regular Play (space) — a synchronous
        # decode here blocked the GUI on far seeks.
        assert (10, False) in sync.view.seeks
        assert (10, True) not in sync.view.seeks

        sync.view.seeks.clear()
        sync._advance()
        # One async request (keeps the waking worker current) + the sync
        # decode that actually renders this tick's frame.
        assert sync.view.seeks == [(11, False), (11, True)]
    finally:
        sync.stop()


def test_handover_to_async_when_worker_answers(qapp):
    sync = _make_sync(qapp, ready=False)
    try:
        sync.play_segment(10, 50)
        sync._advance()

        sync.view.ready = True
        sync.view.seeks.clear()
        sync._advance()
        assert sync.view.seeks == [(12, False)]  # pure async from here on
        assert not sync._sync_until_ready
    finally:
        sync.stop()


def test_warm_worker_plays_async_from_the_start(qapp):
    sync = _make_sync(qapp, ready=True)
    try:
        sync.play_segment(10, 50)
        assert not sync._sync_until_ready

        sync.view.seeks.clear()
        sync._advance()
        assert sync.view.seeks == [(11, False)]
    finally:
        sync.stop()


def test_stop_resets_the_cold_bridge(qapp):
    sync = _make_sync(qapp, ready=False)
    sync.play_segment(10, 50)
    sync.stop()
    assert not sync._sync_until_ready
    assert not sync.is_playing


def test_segment_end_frame_renders_while_cold(qapp):
    """The final tick's seek must render too — a dropped async seek would
    leave the view short of the segment end."""
    sync = _make_sync(qapp, ready=False)
    try:
        sync.play_segment(10, 11)
        sync.view.seeks.clear()
        sync._advance()  # reaches end_frame immediately
        assert (11, True) in sync.view.seeks
    finally:
        sync.stop()


def test_single_step_decodes_synchronously_while_cold(qapp):
    """An arrow step must show its frame even before the worker has served
    one — on macOS a forked worker could die at start-up and never serve."""
    sync = _make_sync(qapp, ready=False)
    sync.seek_to_frame(5)
    assert sync.view.seeks == [(5, False), (5, True)]
    assert sync.app_state.current_frame == 5


def test_single_step_is_async_once_warm(qapp):
    sync = _make_sync(qapp, ready=True)
    sync.seek_to_frame(5)
    assert sync.view.seeks == [(5, False)]
