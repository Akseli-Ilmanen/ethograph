"""Live microphone capture: ring buffer, disk writer and incremental spectrogram.

Qt-free. The GUI (``gui/live_recorder.py``) owns a :class:`LiveRecorder`, polls
it from a timer and draws what :meth:`LiveRecorder.ring` holds. The sound card
delivers blocks on its own thread (:meth:`LiveRecorder.ingest`); nothing on
that thread touches the disk — blocks are queued to :class:`WavWriter`'s thread.

Every setting that names a rate comes from the device or the user; nothing in
here assumes one.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.signal import spectrogram

logger = logging.getLogger(__name__)

#: Window lengths (s) the live view steps through; the right edge stays at "now".
ZOOM_LADDER_S: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
#: WAV sample formats offered, in soundfile's spelling.
WAV_SUBTYPES: tuple[str, ...] = ("PCM_16", "PCM_24", "FLOAT")
#: Sample magnitude at or above which a block counts as clipped.
CLIP_LEVEL = 0.999
#: RIFF WAV cannot exceed 4 GB; roll a file well before that.
WAV_MAX_BYTES = 4 * 1024**3


class RingBuffer:
    """Fixed-capacity float32 ring over ``(samples, channels)`` blocks."""

    def __init__(self, capacity: int, n_channels: int) -> None:
        if capacity <= 0 or n_channels <= 0:
            raise ValueError("RingBuffer needs a positive capacity and channel count")
        self._buf = np.zeros((capacity, n_channels), dtype=np.float32)
        self._capacity = capacity
        self._write = 0
        #: Samples ever appended (monotonic; the stream clock in samples).
        self.total = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def n_channels(self) -> int:
        return self._buf.shape[1]

    def append(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        n = len(block)
        if n >= self._capacity:
            block = block[-self._capacity :]
            n = self._capacity
        with self._lock:
            end = self._write + n
            if end <= self._capacity:
                self._buf[self._write : end] = block
            else:
                first = self._capacity - self._write
                self._buf[self._write :] = block[:first]
                self._buf[: n - first] = block[first:]
            self._write = end % self._capacity
            self.total += n

    def latest(self, n: int) -> np.ndarray:
        """The newest ``min(n, held)`` samples, oldest first, as a copy."""
        with self._lock:
            held = min(self.total, self._capacity)
            n = max(0, min(n, held))
            if n == 0:
                return np.zeros((0, self.n_channels), dtype=np.float32)
            start = (self._write - n) % self._capacity
            if start + n <= self._capacity:
                return self._buf[start : start + n].copy()
            return np.concatenate([self._buf[start:], self._buf[: (start + n) % self._capacity]])

    def since(self, total_before: int) -> np.ndarray:
        """Samples appended after the stream clock read ``total_before``."""
        return self.latest(self.total - total_before)


class WavWriter:
    """Writes queued blocks to one WAV on its own thread.

    ``soundfile`` streams to disk, so a long recording never sits in memory.
    """

    def __init__(self, path: Path, samplerate: float, n_channels: int, subtype: str) -> None:
        import soundfile as sf

        self.path = Path(path)
        self._rate = samplerate
        self._file = sf.SoundFile(
            str(self.path), mode="w", samplerate=int(round(samplerate)), channels=n_channels, subtype=subtype
        )
        self._bytes_per_frame = n_channels * {"PCM_16": 2, "PCM_24": 3, "FLOAT": 4}[subtype]
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        #: Frames handed to :meth:`write` (counted on the audio thread).
        self.frames_queued = 0
        self.frames_written = 0
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=f"wav-writer:{self.path.name}", daemon=True)
        self._thread.start()

    @property
    def duration_s(self) -> float:
        return self.frames_queued / self._rate

    @property
    def bytes_queued(self) -> int:
        return self.frames_queued * self._bytes_per_frame

    def write(self, block: np.ndarray) -> None:
        self.frames_queued += len(block)
        self._queue.put(block)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        if self.error is not None:
            raise RuntimeError(f"Writing {self.path} failed") from self.error

    def _run(self) -> None:
        try:
            while True:
                block = self._queue.get()
                if block is None:
                    break
                self._file.write(block)
                self.frames_written += len(block)
        except Exception as e:  # noqa: BLE001 - surfaced by close() on the caller's thread
            self.error = e
        finally:
            self._file.close()


class IncrementalSpectrogram:
    """scipy's spectrogram, one chunk at a time, identical to the whole-signal result.

    Frames start every ``hop`` samples; a chunk's unframed tail is kept for the
    next call, so feeding a signal in any chunking yields the same columns as
    one call over the full signal (``tests/test_live_audio.py`` guards this).
    """

    def __init__(self, fs: float, nfft: int, hop_frac: float) -> None:
        if nfft < 4:
            raise ValueError("nfft must be at least 4")
        self.fs = fs
        self.nfft = nfft
        self.hop = max(1, int(nfft * hop_frac))
        self._tail = np.zeros(0, dtype=np.float32)
        #: Sample index (in the fed signal) where the next frame starts.
        self._next_frame_start = 0
        self.freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    @property
    def hop_s(self) -> float:
        return self.hop / self.fs

    def reset(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)
        self._next_frame_start = 0

    def feed(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Consume 1-D samples; returns ``(column_centre_times_s, Sxx_db (F, K))``.

        Times are in the fed signal's clock (sample 0 = first ever sample).
        """
        data = np.concatenate([self._tail, np.asarray(samples, dtype=np.float32)])
        n_frames = (len(data) - self.nfft) // self.hop + 1 if len(data) >= self.nfft else 0
        if n_frames <= 0:
            self._tail = data
            return np.zeros(0), np.zeros((len(self.freqs), 0), dtype=np.float32)
        consumed = n_frames * self.hop
        with np.errstate(under="ignore"):
            _freqs, _times, sxx = spectrogram(
                data[: consumed + self.nfft - self.hop], fs=self.fs, nperseg=self.nfft, noverlap=self.nfft - self.hop
            )
        sxx_db = 10 * np.log10(sxx + 1e-10).astype(np.float32)
        np.nan_to_num(sxx_db, copy=False, nan=-100.0, posinf=0.0, neginf=-100.0)
        starts = self._next_frame_start + np.arange(n_frames) * self.hop
        centres = (starts + self.nfft / 2) / self.fs
        self._tail = data[consumed:]
        self._next_frame_start += consumed
        return centres, sxx_db


def next_recording_path(folder: Path, prefix: str, when: datetime) -> Path:
    """``{prefix}_{YYYYMMDD}_{HHMMSS}.wav`` in ``folder``, never overwriting."""
    stem = f"{prefix}_{when.strftime('%Y%m%d_%H%M%S')}"
    path = folder / f"{stem}.wav"
    n = 1
    while path.exists():
        path = folder / f"{stem}_{n}.wav"
        n += 1
    return path


def day_folder(root: Path, when: datetime) -> Path:
    return root / when.strftime("%Y-%m-%d")


def pause_log_path(wav: Path) -> Path:
    return wav.with_name(f"{wav.stem}.pauses.tsv")


@dataclass
class Recording:
    """One finished WAV: where it is, how long it is, and the gaps it hides."""

    path: Path
    duration_s: float
    #: ``(file_time_s, wall_gap_s)`` per pause: the file is contiguous, the
    #: wall clock jumped by ``wall_gap_s`` at ``file_time_s``.
    pauses: list[tuple[float, float]] = field(default_factory=list)

    def write_pause_log(self) -> Path | None:
        if not self.pauses:
            return None
        out = pause_log_path(self.path)
        pd.DataFrame(self.pauses, columns=["file_time_s", "wall_gap_s"]).to_csv(out, sep="\t", index=False)
        return out


#: The one host API listed per platform. PortAudio exposes every endpoint once
#: per API; the native one (WASAPI, CoreAudio, ALSA) lists only endpoints that
#: exist now, where MME / DirectSound add virtual "mapper" devices and WDM-KS
#: adds every endpoint the driver has ever registered, plugged in or not.
PREFERRED_HOSTAPIS = ("WASAPI", "Core Audio", "ALSA")


def list_input_devices(devices=None, hostapis=None) -> list[tuple[int, str, int, float]]:
    """``(index, name, max_input_channels, default_samplerate)`` per input device.

    One host API's view (:data:`PREFERRED_HOSTAPIS`), its default input first;
    every API when none of the preferred ones is present. ``devices`` and
    ``hostapis`` default to sounddevice's queries and exist for tests.
    """
    if devices is None or hostapis is None:
        import sounddevice as sd

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    chosen = next((i for i, api in enumerate(hostapis) if any(p in api["name"] for p in PREFERRED_HOSTAPIS)), None)
    default = hostapis[chosen]["default_input_device"] if chosen is not None else -1
    out: list[tuple[int, str, int, float]] = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0 or (chosen is not None and dev["hostapi"] != chosen):
            continue
        entry = (idx, str(dev["name"]), int(dev["max_input_channels"]), float(dev["default_samplerate"]))
        if idx == default:
            out.insert(0, entry)
        else:
            out.append(entry)
    return out


def check_input_settings(device: int, samplerate: float, channels: int) -> str | None:
    """``None`` when the device accepts these settings, else PortAudio's reason."""
    import sounddevice as sd

    try:
        sd.check_input_settings(device=device, samplerate=samplerate, channels=channels, dtype="float32")
    except (sd.PortAudioError, ValueError) as e:
        return str(e)
    return None


class LiveRecorder:
    """Capture state machine: monitoring → recording ⇄ paused → monitoring.

    Audio flows the moment :meth:`start_monitoring` opens the stream, so the
    plots show the room before anything is written. :meth:`record` opens a
    file, :meth:`pause` stops writing without closing it, :meth:`stop` closes
    it and returns the :class:`Recording`. :meth:`poll` (GUI timer) rolls a
    file past ``roll_s`` or the WAV size limit.
    """

    def __init__(
        self,
        samplerate: float,
        n_channels: int,
        *,
        ring_s: float,
        folder: Path,
        prefix: str,
        subtype: str,
        roll_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if samplerate <= 0:
            raise ValueError("samplerate must be positive")
        self.samplerate = samplerate
        self.n_channels = n_channels
        self.ring = RingBuffer(int(ring_s * samplerate), n_channels)
        self.folder = Path(folder)
        self.prefix = prefix
        self.subtype = subtype
        self.roll_s = roll_s
        self._clock = clock
        self._stream: Any = None  # sounddevice.InputStream while monitoring
        self._writer: WavWriter | None = None
        self._paused = False
        self._pause_started: float | None = None
        self._pauses: list[tuple[float, float]] = []
        self.peak = np.zeros(n_channels, dtype=np.float32)
        self.clipped = np.zeros(n_channels, dtype=bool)
        self.stream_error: str | None = None
        #: Files finished since the last :meth:`take_finished`.
        self._finished: list[Recording] = []

    # -- state -----------------------------------------------------------
    @property
    def state(self) -> str:
        if self._writer is None:
            return "monitoring"
        return "paused" if self._paused else "recording"

    @property
    def file_duration_s(self) -> float:
        return self._writer.duration_s if self._writer is not None else 0.0

    @property
    def current_path(self) -> Path | None:
        return self._writer.path if self._writer is not None else None

    def stream_time_s(self) -> float:
        return self.ring.total / self.samplerate

    # -- audio thread ------------------------------------------------------
    def ingest(self, block: np.ndarray) -> None:
        """A block from the sound card (or a test). Audio-thread safe: no disk I/O."""
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        self.ring.append(block)
        peak = np.abs(block).max(axis=0)
        self.peak = peak
        self.clipped |= peak >= CLIP_LEVEL
        writer = self._writer
        if writer is not None and not self._paused:
            writer.write(block.copy())

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status:
            self.stream_error = str(status)
        self.ingest(indata)

    def start_monitoring(self, device: int, blocksize: int = 0) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            device=device,
            samplerate=self.samplerate,
            channels=self.n_channels,
            dtype="float32",
            blocksize=blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> Recording | None:
        """Stop everything; a file still open is finished and returned."""
        finished = self.stop() if self._writer is not None else None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return finished

    # -- transport ---------------------------------------------------------
    def record(self) -> Path:
        if self._writer is not None:
            raise RuntimeError("already recording")
        self.folder.mkdir(parents=True, exist_ok=True)
        path = next_recording_path(self.folder, self.prefix, datetime.now())
        self._writer = WavWriter(path, self.samplerate, self.n_channels, self.subtype)
        self._paused = False
        self._pauses = []
        return path

    def pause(self) -> None:
        if self._writer is None or self._paused:
            return
        self._paused = True
        self._pause_started = self._clock()

    def resume(self) -> None:
        if self._writer is None or not self._paused:
            return
        gap = self._clock() - (self._pause_started or self._clock())
        self._pauses.append((self._writer.duration_s, gap))
        self._paused = False
        self._pause_started = None

    def stop(self) -> Recording:
        writer = self._writer
        if writer is None:
            raise RuntimeError("not recording")
        if self._paused:
            self.resume()
        self._writer = None
        writer.close()
        rec = Recording(writer.path, writer.duration_s, list(self._pauses))
        rec.write_pause_log()
        self._pauses = []
        return rec

    def poll(self) -> None:
        """GUI-timer hook: roll the file when it is long or large enough."""
        writer = self._writer
        if writer is None or self._paused:
            return
        if writer.error is not None:
            self._writer = None
            raise RuntimeError(f"Writing {writer.path} failed") from writer.error
        if writer.duration_s >= self.roll_s or writer.bytes_queued >= WAV_MAX_BYTES - 64 * 1024**2:
            self._finished.append(self.stop())
            self.record()

    def take_finished(self) -> list[Recording]:
        out, self._finished = self._finished, []
        return out

    def reset_clip(self) -> None:
        self.clipped[:] = False
