"""Enhanced spectrogram plot inheriting from BasePlot."""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from qtpy.QtCore import Signal  # noqa: E402
from scipy.signal import spectrogram  # noqa: E402

from ..io.plot_sources import PlotSource  # noqa: E402
from .app_constants import (  # noqa: E402
    BUFFER_COVERAGE_MARGIN,
    DEFAULT_BUFFER_MULTIPLIER_AUDIO,
    DEFAULT_FALLBACK_MAX_FREQUENCY,
    SPECTROGRAM_DEBOUNCE_MS,
    Z_INDEX_BACKGROUND,
)
from .plots_base import BasePlot, ThrottleDebounce  # noqa: E402


def _audio_load_hint(audio_path, err: Exception) -> str:
    """Build a single actionable message for an unreadable audio file."""
    from ..io.audio_extract import is_video_container

    if is_video_container(audio_path):
        return (
            f"Cannot use the audio track of '{audio_path}' ({err}). Embedded audio is "
            "decoded to a cached WAV before use; the container has no readable audio "
            "track, or it could not be decoded. Supply a separate WAV/FLAC/OGG file."
        )
    return (
        f"Failed to load audio file '{audio_path}' ({err}). Supported audio formats "
        "are WAV, FLAC, OGG and MP3; convert other formats before loading."
    )


def auto_levels_db(sxx_db: np.ndarray) -> tuple[float, float] | None:
    """audian's levels rule, shared by every spectrogram: the floor is the 95th
    percentile of the band's top sixteenth (noise), the ceiling the maximum,
    the span clamped to 20–80 dB. ``sxx_db`` is ``(F, T)``."""
    if sxx_db.size == 0:
        return None
    nf = max(1, sxx_db.shape[0] // 16)
    with np.errstate(all="ignore"):
        zmin = float(np.percentile(sxx_db[-nf:, :], 95))
        zmax = float(np.max(sxx_db))
    if not np.isfinite(zmin) or not np.isfinite(zmax):
        return None
    zmax = zmin + 0.95 * (zmax - zmin)
    if zmax - zmin < 20:
        zmax = zmin + 20
    if zmax - zmin > 80:
        zmin = zmax - 80
    return round(zmin, 1), round(zmax, 1)


class SharedAudioCache:
    """Singleton cache for AudioLoader instances.

    Multiple GUI components (waveform plot, spectrogram, heatmap, envelope
    overlay, set_time) all need the same audio loader. Without caching, each
    would open the file independently, wasting file handles and memory.
    """

    _instances = {}
    #: Paths that already failed to load, so we log the reason once instead of
    #: spamming the console on every plot refresh / zoom.
    _failed: set[str] = set()
    _lock = threading.Lock()

    @classmethod
    def get_loader(cls, audio_path, buffer_size=10.0):
        if not audio_path:
            return None

        with cls._lock:
            if audio_path in cls._failed:
                return None
            if audio_path not in cls._instances:
                try:
                    from audioio import AudioLoader

                    from ..io.audio_extract import resolve_audio_path

                    # A video container's track is decoded to a cached WAV first:
                    # audioio's seekable backends read no container format.
                    readable = resolve_audio_path(audio_path)
                    cls._instances[audio_path] = AudioLoader(readable, buffersize=buffer_size)
                except (OSError, IOError, ValueError, RuntimeError) as e:
                    cls._failed.add(audio_path)
                    logger.error("%s", _audio_load_hint(audio_path, e))
                    return None
            return cls._instances[audio_path]

    @classmethod
    def clear_cache(cls):
        with cls._lock:
            cls._instances.clear()
            cls._failed.clear()


class SpectrogramPlot(BasePlot):
    """Spectrogram plot with shared sync and marker functionality from BasePlot."""

    sigFilterChanged = Signal(float, float)
    bufferUpdated = Signal()

    def __init__(self, app_state, parent=None):
        super().__init__(app_state, parent)

        #: Pinned mic/channel (an ``audio_source_map`` key); None follows the
        #: global Mic combo.
        self.mic_name: str | None = None

        self.setLabel("left", "Frequency", units="Hz")

        self.spec_item = pg.ImageItem()
        self.spec_item.setZValue(Z_INDEX_BACKGROUND)
        self.addItem(self.spec_item)

        self.init_colorbar()
        self.buffer = SpectrogramBuffer(app_state)
        self.source: PlotSource | None = None

        self._set_frequency_limits()

        self._td = ThrottleDebounce(
            debounce_ms=SPECTROGRAM_DEBOUNCE_MS,
            debounce_cb=self._do_range_update,
        )

        self.vb.sigRangeChanged.connect(self._on_view_range_changed)

    def init_colorbar(self):
        """Initialize colorbar for spectrogram."""
        vmin = self.app_state.get_with_default("vmin_db")
        vmax = self.app_state.get_with_default("vmax_db")
        self.spec_item.setLevels([vmin, vmax])

        colormap = self.app_state.get_with_default("spec_colormap")
        self.spec_item.setColorMap(colormap)

    def update_colormap(self, colormap_name: str):
        """Update colormap for spectrogram."""
        self.spec_item.setColorMap(colormap_name)

    def update_levels(self, vmin=None, vmax=None):
        """Update dB levels for spectrogram display."""
        if vmin is None:
            vmin = self.app_state.get_with_default("vmin_db")
        if vmax is None:
            vmax = self.app_state.get_with_default("vmax_db")
        self.spec_item.setLevels([vmin, vmax])

    def set_source(self, source: PlotSource | None):
        """Set a PlotSource for spectrogram data. Clears buffer."""
        self.source = source
        self.buffer._clear_buffer()
        if source is not None:
            self._set_frequency_limits()

    def update_plot_content(self, t0: Optional[float] = None, t1: Optional[float] = None):
        """Update the spectrogram content."""
        source = self.source
        if source is None:
            return

        if t0 is None or t1 is None:
            xmin, xmax = self.get_current_xlim()
            t0, t1 = xmin, xmax

        result = self.buffer.get_spectrogram(source, t0, t1)
        if result is None:
            return

        Sxx_db, spec_rect = result
        if Sxx_db is not None and self.buffer.buffer_changed:
            self.spec_item.setImage(Sxx_db.T, autoLevels=False)
            self.spec_item.setRect(*spec_rect)
            self.buffer.buffer_changed = False
            self.bufferUpdated.emit()

        self.current_range = (t0, t1)

    def apply_y_range(self, ymin: Optional[float], ymax: Optional[float]):
        """Apply frequency range for spectrogram."""
        if ymin is not None and ymax is not None:
            self.plot_item.setYRange(ymin, ymax)

    def _set_frequency_limits(self):
        """Set frequency limits based on source sampling rate."""
        nyquist_freq = DEFAULT_FALLBACK_MAX_FREQUENCY

        if self.source is not None:
            nyquist_freq = self.source.sampling_rate / 2
        else:
            audio_path = getattr(self.app_state, "audio_path", None)
            if audio_path:
                try:
                    audio_loader = SharedAudioCache.get_loader(audio_path)
                    if audio_loader:
                        nyquist_freq = audio_loader.rate / 2
                except (OSError, IOError, AttributeError):
                    pass

        spec_ymin = self.app_state.get_with_default("spec_ymin")
        spec_ymax = self.app_state.get_with_default("spec_ymax")
        if spec_ymax is None or spec_ymax <= 0:
            spec_ymax = nyquist_freq
        if spec_ymin is None or spec_ymin < 0:
            spec_ymin = 0

        min_freq_range = max(100.0, nyquist_freq * 0.02)
        self.vb.setLimits(
            yMin=0,
            yMax=nyquist_freq,
            minYRange=min_freq_range,
            maxYRange=nyquist_freq,
        )
        self.plot_item.setYRange(spec_ymin, spec_ymax, padding=0)

    def _apply_y_constraints(self):
        """Apply frequency-based y-axis constraints."""
        self._set_frequency_limits()

    def update_buffer_settings(self):
        """Update buffer settings from app state."""
        self.buffer.update_buffer_size()

    def _on_view_range_changed(self):
        if not self.isVisible():
            return
        if not hasattr(self.app_state, "ds") or self.app_state.ds is None:
            return
        self._td.trigger()

    def _do_range_update(self):
        if not self.isVisible():
            return
        t0, t1 = self.get_current_xlim()
        self.update_plot_content(t0, t1)


class SpectrogramBuffer:
    """Single-buffer spectrogram cache inspired by audian's BufferedData pattern."""

    def __init__(self, app_state):
        self.app_state = app_state
        self.current_identity: str | None = None
        self.buffer_multiplier = self._get_buffer_multiplier()

        self.Sxx_db = None
        self.freqs = None
        self.times = None
        self.buffer_t0 = 0.0
        self.buffer_t1 = 0.0
        self.fs = None
        self.fresolution = 1.0

        self.buffer_changed = False

    def _get_buffer_multiplier(self):
        spec_buffer = getattr(self.app_state, "spec_buffer", None)
        if spec_buffer is not None and spec_buffer > 0:
            return spec_buffer
        try:
            val = self.app_state.get_with_default("buffer_multiplier")
            return val if val is not None else DEFAULT_BUFFER_MULTIPLIER_AUDIO
        except (KeyError, AttributeError):
            return DEFAULT_BUFFER_MULTIPLIER_AUDIO

    def _covers_range(self, t0, t1):
        """Check if current buffer covers requested range with margin."""
        if self.Sxx_db is None:
            return False
        margin = (t1 - t0) * BUFFER_COVERAGE_MARGIN
        return self.buffer_t0 <= t0 - margin and self.buffer_t1 >= t1 + margin

    def get_spectrogram(self, source: PlotSource, t0: float, t1: float):
        """Get spectrogram data, computing only if necessary."""
        if source.identity != self.current_identity:
            self._clear_buffer()
            self.current_identity = source.identity

        if self._covers_range(t0, t1):
            return self.Sxx_db, self._get_spec_rect()

        self._compute_buffer(source, t0, t1)

        if self.Sxx_db is None:
            return None

        return self.Sxx_db, self._get_spec_rect()

    def _compute_buffer(self, source: PlotSource, t0: float, t1: float):
        """Compute spectrogram for buffered range."""
        self.fs = source.sampling_rate

        window_size = t1 - t0
        buffer_size = window_size * self.buffer_multiplier
        # Clamp to the source's span on the display axis — the file does not
        # necessarily start at t=0 there (session basis, stream offsets).
        extent = source.time_range
        self.buffer_t0 = max(extent.start_s, t0 - buffer_size / 2)
        self.buffer_t1 = min(extent.end_s, t1 + buffer_size / 2)

        result = source.get_data(self.buffer_t0, self.buffer_t1)
        audio_data = result.values if hasattr(result, "values") else result

        if len(audio_data) == 0:
            return

        nfft = self.app_state.get_with_default("nfft")
        hop_frac = self.app_state.get_with_default("hop_frac")

        # Clamp nfft to data length (power-of-2) for low sample-rate sources
        max_nfft = 1 << (len(audio_data).bit_length() - 1)  # largest pow2 <= len
        if max_nfft < 4:
            return
        nfft = min(nfft, max_nfft)

        hop = int(nfft * hop_frac)

        if len(audio_data) < nfft:
            return

        with np.errstate(under="ignore"):
            freqs, times, Sxx = spectrogram(audio_data, fs=self.fs, nperseg=nfft, noverlap=nfft - hop)

        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        np.nan_to_num(Sxx_db, copy=False, nan=-100.0, posinf=0.0, neginf=-100.0)
        self.Sxx_db = Sxx_db
        self.freqs = freqs
        self.times = times + self.buffer_t0
        self.fresolution = self.fs / nfft if nfft > 0 else 1.0
        self.buffer_changed = True

    def _get_spec_rect(self):
        """Get rectangle [x, y, width, height] for setRect."""
        if self.Sxx_db is None or self.freqs is None:
            return [0, 0, 1, 1]

        t_duration = self.buffer_t1 - self.buffer_t0
        f_max = self.freqs[-1] + self.fresolution if len(self.freqs) > 0 else self.fs / 2

        return [self.buffer_t0, 0, t_duration, f_max]

    def _clear_buffer(self):
        self.Sxx_db = None
        self.freqs = None
        self.times = None
        self.buffer_t0 = 0.0
        self.buffer_t1 = 0.0
        self.buffer_changed = False
        self.current_identity = None

    def update_buffer_size(self):
        self.buffer_multiplier = self._get_buffer_multiplier()
        self._clear_buffer()
