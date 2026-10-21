"""Live capture core: ring buffer, chunk-invariant spectrogram, file lifecycle."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import spectrogram

from ethograph.io.live_audio import (
    IncrementalSpectrogram,
    LiveRecorder,
    RingBuffer,
    list_input_devices,
    next_recording_path,
    pause_log_path,
)

soundfile = pytest.importorskip("soundfile")


def test_ring_latest_survives_wraparound():
    ring = RingBuffer(10, 2)
    ring.append(np.arange(8, dtype=np.float32).reshape(4, 2))  # 4 samples
    ring.append(np.arange(100, 118, dtype=np.float32).reshape(9, 2))  # wraps
    out = ring.latest(6)
    expected = np.arange(106, 118, dtype=np.float32).reshape(6, 2)
    np.testing.assert_array_equal(out, expected)
    assert ring.total == 13
    assert len(ring.latest(50)) == 10  # never more than held


def test_since_returns_only_new_samples():
    ring = RingBuffer(100, 1)
    ring.append(np.ones(5))
    mark = ring.total
    ring.append(np.full(3, 2.0))
    np.testing.assert_array_equal(ring.since(mark)[:, 0], [2.0, 2.0, 2.0])


@pytest.mark.parametrize("chunk", [7, 64, 256, 1000])
def test_incremental_spectrogram_matches_whole_signal(chunk):
    fs, nfft, hop_frac = 8000.0, 128, 0.5
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096).astype(np.float32)
    inc = IncrementalSpectrogram(fs, nfft, hop_frac)
    times, cols = [], []
    for i in range(0, len(x), chunk):
        t, c = inc.feed(x[i : i + chunk])
        times.append(t)
        cols.append(c)
    got_t = np.concatenate(times)
    got = np.concatenate(cols, axis=1)
    hop = int(nfft * hop_frac)
    f, ref_t, sxx = spectrogram(x, fs=fs, nperseg=nfft, noverlap=nfft - hop)
    ref = 10 * np.log10(sxx + 1e-10)
    n = min(got.shape[1], ref.shape[1])
    assert n == ref.shape[1]  # every frame of the whole signal was produced
    np.testing.assert_allclose(got_t[:n], ref_t[:n])
    np.testing.assert_allclose(got[:, :n], ref[:, :n], rtol=1e-4, atol=1e-3)


def test_next_recording_path_never_overwrites(tmp_path: Path):
    when = datetime(2026, 10, 21, 14, 30, 5)
    first = next_recording_path(tmp_path, "rec", when)
    assert first.name == "rec_20261021_143005.wav"
    first.touch()
    assert next_recording_path(tmp_path, "rec", when).name == "rec_20261021_143005_1.wav"


def _recorder(tmp_path: Path, **kw) -> LiveRecorder:
    defaults = dict(ring_s=1.0, folder=tmp_path, prefix="rec", subtype="PCM_16", roll_s=3600.0)
    defaults.update(kw)
    return LiveRecorder(1000.0, 2, **defaults)


def test_record_pause_stop_writes_contiguous_file_and_pause_log(tmp_path: Path):
    clock = [100.0]
    rec = _recorder(tmp_path, clock=lambda: clock[0])
    rec.ingest(np.zeros((100, 2)))  # monitoring: nothing written
    assert rec.state == "monitoring"
    path = rec.record()
    rec.ingest(np.full((500, 2), 0.25))
    rec.pause()
    assert rec.state == "paused"
    rec.ingest(np.ones((300, 2)))  # dropped: paused
    clock[0] += 7.5
    rec.resume()
    rec.ingest(np.full((200, 2), -0.25))
    finished = rec.stop()
    assert rec.state == "monitoring"
    assert finished.path == path
    data, sr = soundfile.read(path, dtype="float32")
    assert sr == 1000 and data.shape == (700, 2)
    assert np.allclose(data[:500], 0.25, atol=1e-4) and np.allclose(data[500:], -0.25, atol=1e-4)
    assert finished.pauses == [(0.5, 7.5)]
    log = pd.read_csv(pause_log_path(path), sep="\t")
    assert log.loc[0, "file_time_s"] == 0.5 and log.loc[0, "wall_gap_s"] == 7.5


def test_poll_rolls_a_long_take_into_a_new_file(tmp_path: Path):
    rec = _recorder(tmp_path, roll_s=0.5)
    first = rec.record()
    rec.ingest(np.zeros((600, 2)))
    rec.poll()
    assert rec.state == "recording" and rec.current_path != first
    rolled = rec.take_finished()
    assert [r.path for r in rolled] == [first] and rolled[0].duration_s == 0.6
    assert rolled[0].pauses == []
    rec.stop()


def test_clip_flag_latches_until_reset(tmp_path: Path):
    rec = _recorder(tmp_path)
    block = np.zeros((10, 2))
    block[3, 1] = 1.0
    rec.ingest(block)
    rec.ingest(np.zeros((10, 2)))
    assert rec.clipped.tolist() == [False, True]
    rec.reset_clip()
    assert not rec.clipped.any()


def test_close_finishes_an_open_file(tmp_path: Path):
    rec = _recorder(tmp_path)
    rec.record()
    rec.ingest(np.zeros((10, 2)))
    finished = rec.close()
    assert finished is not None and soundfile.info(finished.path).frames == 10


def test_input_devices_come_from_one_native_host_api_default_first():
    hostapis = [
        {"name": "MME", "default_input_device": 0},
        {"name": "Windows WASAPI", "default_input_device": 3},
    ]
    devices = [
        {
            "name": "Microsoft Sound Mapper - Input",
            "hostapi": 0,
            "max_input_channels": 2,
            "default_samplerate": 44100.0,
        },
        {"name": "Mic (MME)", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 44100.0},
        {"name": "Mic (WASAPI)", "hostapi": 1, "max_input_channels": 2, "default_samplerate": 48000.0},
        {"name": "Array (WASAPI)", "hostapi": 1, "max_input_channels": 4, "default_samplerate": 48000.0},
        {"name": "Speakers", "hostapi": 1, "max_input_channels": 0, "default_samplerate": 48000.0},
    ]
    got = list_input_devices(devices, hostapis)
    assert [g[1] for g in got] == ["Array (WASAPI)", "Mic (WASAPI)"]
    # No native API at all: every input device is offered.
    got = list_input_devices(devices, hostapis[:1])
    assert [g[1] for g in got] == ["Microsoft Sound Mapper - Input", "Mic (MME)", "Mic (WASAPI)", "Array (WASAPI)"]
