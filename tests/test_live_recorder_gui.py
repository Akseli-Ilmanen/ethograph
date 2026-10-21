"""The recorder window without a sound card: ingest blocks, tick, stop, hand-off."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("soundfile")
pytest.importorskip("sounddevice")

from ethograph.gui import live_recorder as lr  # noqa: E402


@pytest.fixture
def window(qapp, app_state, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(lr, "list_input_devices", lambda: [(0, "Fake mic", 2, 1000.0)])
    opened: list[list[str]] = []
    project = tmp_path / "project"
    project.mkdir()
    app_state.live_recording_channels = 2
    win = lr.LiveRecorderWindow(app_state, project, on_open_session=opened.append, start_stream=False)
    win.opened = opened
    yield win
    win._close_recorder()
    win.close()


def test_monitoring_feeds_both_plots_without_writing(window):
    rec = window._recorder
    assert rec is not None and rec.state == "monitoring"
    rng = np.random.default_rng(1)
    rec.ingest(rng.standard_normal((1500, 2)).astype(np.float32))
    window._tick()
    window._tick()
    assert window._trace.trace_item.xData is not None and len(window._trace.trace_item.xData) > 0
    assert window._cols is not None and window._cols.held > 0
    assert list(window._folder().glob("*.wav")) == []
    assert window._trace.vb.viewRange()[0][1] == pytest.approx(1.5)


def test_stop_enters_file_mode_and_hand_off_lists_the_folder(window):
    rec = window._recorder
    window._on_record()
    assert rec.state == "recording"
    rec.ingest(np.full((2000, 2), 0.1, dtype=np.float32))
    window._tick()
    window._on_stop()
    assert rec.state == "monitoring"
    assert not window._trace.live and window._trace.source is not None
    assert window._trace.source.time_range.end_s == pytest.approx(2.0)
    assert window._open_btn.isEnabled()
    wav = window._recordings[-1].path
    assert wav.parent == window._folder()
    window._on_open_session()
    assert window.opened == [[str(wav)]]


def test_nfft_change_rebuilds_from_the_ring(window):
    rec = window._recorder
    rec.ingest(np.random.default_rng(2).standard_normal((3000, 2)).astype(np.float32))
    window._tick()
    window._nfft_combo.setCurrentText("512")
    assert window.app_state.nfft == 512
    assert window._spec is not None and window._spec.nfft == 512
    assert window._cols is not None and window._cols.held > 0
    assert window._overlay.isHidden()


def test_window_ladder_steps_and_persists(window):
    start = window.window_s
    window._step_window(+1)
    assert window.window_s > start
    assert window.app_state.live_recording_window_s == window.window_s


def test_levels_auto_fit_once_a_second_of_audio_is_on_screen(window):
    rec = window._recorder
    before = (window.app_state.vmin_db, window.app_state.vmax_db)
    t = np.arange(2000) / 1000.0
    tone = (0.5 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    rec.ingest(np.stack([tone, tone], axis=1))
    window._tick()
    after = (window.app_state.vmin_db, window.app_state.vmax_db)
    assert after != before
    assert window._vmin_spin.value() == after[0] and window._vmax_spin.value() == after[1]
    assert 20 <= after[1] - after[0] <= 80


def test_amplitude_is_full_scale_until_a_double_click_fits_it(window):
    rec = window._recorder
    assert window._trace.vb.viewRange()[1] == pytest.approx([-1.0, 1.0])
    rec.ingest(np.full((1500, 2), 0.2, dtype=np.float32))
    window._tick()
    assert window._trace.vb.viewRange()[1] == pytest.approx([-1.0, 1.0])  # no re-fit per tick

    class _Click:
        def double(self):
            return True

        def button(self):
            from qtpy.QtCore import Qt

            return Qt.LeftButton

    window._on_trace_clicked(_Click())
    assert window._trace.vb.viewRange()[1] == pytest.approx([-0.22, 0.22], abs=1e-3)
    window._set_full_scale()
    assert window._trace.vb.viewRange()[1] == pytest.approx([-1.0, 1.0])
