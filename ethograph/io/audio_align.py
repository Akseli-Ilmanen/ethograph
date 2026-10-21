"""Where an external microphone's recording sits on a camera's clock.

A camera that records its own soundtrack and a separate microphone that
started at some other moment share one acoustic scene. Cross-correlating the
two recordings finds the lag at which they agree, which is the offset the
alignment NWB needs (``audio_mic-1_start``) for the microphone's file to play
in step with the video.

The method follows the BBC's ``audio-offset-finder`` (Apache-2.0): both
signals become standardised MFCC frame sequences, the sequences are
cross-correlated, and the chosen peak is scored by how far it stands above the
rest of the correlation (its *standard score*). Working on MFCCs rather than
the waveform makes the estimate indifferent to the two microphones' very
different frequency responses and to the camera's lossy codec. Nothing is
copied from that project; this file is an independent implementation of the
published idea, listed under "Adapted code" in ``THIRD_PARTY_NOTICES.md``.

Sign convention, used everywhere in this module and by the dialog:
``offset_s`` is the time on the *reference* (video) clock at which sample 0 of
the *query* (microphone) file occurs. A microphone that started five seconds
before the camera has ``offset_s == -5.0``.

Qt-free; the dialog in :mod:`ethograph.gui.dialog_audio_align` drives it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.optimize import linear_sum_assignment
from scipy.signal import correlate, correlation_lags, resample_poly, spectrogram, stft

#: Analysis rate. Speech-band content is what two microphones in one room share;
#: anything above 4 kHz differs too much between a camera's codec and a studio mic.
ANALYSIS_RATE = 8000
#: One MFCC frame every 10 ms: the offset's resolution, finer than any video frame.
HOP_S = 0.01
_N_MELS = 40
_N_MFCC = 13
_LOG_FLOOR = 1e-10

#: A standard score below this is reported as unreliable (the BBC's threshold
#: for "check by hand").
LOW_SCORE = 5.0
#: Above this the peak is unambiguous.
GOOD_SCORE = 10.0


@dataclass(frozen=True)
class OffsetEstimate:
    """One cross-correlation's verdict.

    ``window_offsets`` holds ``(query_centre_s, offset_s)`` for successive
    windows of the query, when it was long enough to split; a trend across
    them means the two devices' clocks run at different rates.
    """

    offset_s: float
    score: float
    window_offsets: list[tuple[float, float]] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return self.score >= LOW_SCORE

    @property
    def drift_ppm(self) -> float | None:
        """Clock-rate disagreement in parts per million, ``None`` without enough windows.

        Negative when the query's clock runs fast (it counts more seconds per
        reference second, so later windows land earlier on the reference).
        """
        if len(self.window_offsets) < 3:
            return None
        centres = np.array([c for c, _ in self.window_offsets])
        offsets = np.array([o for _, o in self.window_offsets])
        slope = np.polyfit(centres, offsets, 1)[0]
        return float(slope * 1e6)


# ── Signals ──


def load_mono(path: str | Path, rate: int = ANALYSIS_RATE) -> np.ndarray:
    """A file's channels averaged and resampled to *rate*, read block-wise."""
    import soundfile as sf

    info = sf.info(str(path))
    ratio = Fraction(rate, int(info.samplerate))
    blocks: list[np.ndarray] = []
    for block in sf.blocks(str(path), blocksize=int(info.samplerate) * 60, dtype="float32", always_2d=True):
        mono = block.mean(axis=1)
        if ratio != 1:
            mono = resample_poly(mono, ratio.numerator, ratio.denominator)
        blocks.append(mono.astype(np.float32))
    if not blocks:
        raise ValueError(f"{Path(path).name} holds no audio samples")
    return np.concatenate(blocks)


def _mel_filterbank(n_fft: int, rate: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges_hz = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(rate / 2), n_mels + 2))
    fft_freqs = np.linspace(0.0, rate / 2, n_fft // 2 + 1)
    bank = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        lo, centre, hi = edges_hz[i : i + 3]
        rising = (fft_freqs - lo) / max(centre - lo, 1e-9)
        falling = (hi - fft_freqs) / max(hi - centre, 1e-9)
        bank[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


def mfcc_frames(signal: np.ndarray, rate: int = ANALYSIS_RATE, hop_s: float = HOP_S) -> np.ndarray:
    """Standardised MFCCs, ``(n_frames, N_MFCC)``, one frame per *hop_s*.

    Each coefficient is z-scored over time so the cross-correlation measures
    the *shape* of the two recordings' spectral trajectories, not their level.
    """
    hop = int(round(rate * hop_s))
    n_fft = 1 << (2 * hop - 1).bit_length()  # ~25 ms window as the next power of two
    if len(signal) < n_fft:
        raise ValueError("Recording shorter than one analysis window")
    _, _, spec = stft(signal, fs=rate, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    power = np.abs(spec) ** 2
    mel = _mel_filterbank(n_fft, rate, _N_MELS) @ power
    cepstra = dct(np.log(mel + _LOG_FLOOR), type=2, norm="ortho", axis=0)[1 : _N_MFCC + 1]
    frames = cepstra.T
    std = frames.std(axis=0)
    std[std == 0] = 1.0
    return (frames - frames.mean(axis=0)) / std


# ── Offsets ──


def _correlate_frames(ref: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation summed over coefficients, with the lag of every bin.

    ``corr[k]`` pairs ``query[n]`` with ``ref[n + lags[k]]``, so ``lags[k]`` is
    where the query's first frame falls on the reference's frame axis.
    """
    corr = np.zeros(len(ref) + len(query) - 1)
    for c in range(ref.shape[1]):
        corr += correlate(ref[:, c], query[:, c], mode="full", method="fft")
    return corr, correlation_lags(len(ref), len(query), mode="full")


def _peak(corr: np.ndarray, lags: np.ndarray, hop_s: float) -> tuple[float, float]:
    k = int(np.argmax(corr))
    std = corr.std()
    score = float((corr[k] - corr.mean()) / std) if std > 0 else 0.0
    return float(lags[k] * hop_s), score


def estimate_offset(
    ref: np.ndarray,
    query: np.ndarray,
    rate: int = ANALYSIS_RATE,
    hop_s: float = HOP_S,
    window_s: float = 60.0,
) -> OffsetEstimate:
    """Where *query* starts on *ref*'s clock, from their MFCC cross-correlation.

    The whole query gives the offset. When it spans at least three *window_s*
    windows, each window is also placed on its own; those per-window offsets
    expose a clock drift the single number would hide.
    """
    ref_frames = mfcc_frames(ref, rate, hop_s)
    query_frames = mfcc_frames(query, rate, hop_s)
    corr, lags = _correlate_frames(ref_frames, query_frames)
    offset_s, score = _peak(corr, lags, hop_s)

    window_frames = int(round(window_s / hop_s))
    windows: list[tuple[float, float]] = []
    if window_frames > 0 and len(query_frames) >= 3 * window_frames:
        for start in range(0, len(query_frames) - window_frames + 1, window_frames):
            piece = query_frames[start : start + window_frames]
            w_corr, w_lags = _correlate_frames(ref_frames, piece)
            w_offset, w_score = _peak(w_corr, w_lags, hop_s)
            if w_score >= LOW_SCORE:
                # The window's own start on the ref clock, minus where it sits in the query.
                windows.append(((start + window_frames / 2) * hop_s, w_offset - start * hop_s))
    return OffsetEstimate(offset_s=offset_s, score=score, window_offsets=windows)


def estimate_offset_between(ref_path: str | Path, query_path: str | Path, **kwargs) -> OffsetEstimate:
    """:func:`estimate_offset` on two audio files."""
    return estimate_offset(load_mono(ref_path), load_mono(query_path), **kwargs)


# ── Correspondence ──


def score_matrix(refs: list[np.ndarray], queries: list[np.ndarray], **kwargs) -> list[list[OffsetEstimate]]:
    """Every reference against every query: ``[i][j]`` places ``queries[j]`` on ``refs[i]``."""
    return [[estimate_offset(ref, query, **kwargs) for query in queries] for ref in refs]


def propose_pairs(matrix: list[list[OffsetEstimate]]) -> list[int]:
    """``result[i]`` is the query that belongs to reference ``i``, maximising the total score.

    One query per reference and one reference per query (the Hungarian
    assignment); the caller shows the proposal, the user has the last word.
    """
    scores = np.array([[est.score for est in row] for row in matrix])
    if scores.shape[0] != scores.shape[1]:
        raise ValueError(f"Pairing needs as many queries as references, got {scores.shape[1]} and {scores.shape[0]}")
    rows, cols = linear_sum_assignment(-scores)
    result = [0] * len(rows)
    for r, c in zip(rows, cols, strict=True):
        result[int(r)] = int(c)
    return result


# ── Preview ──


def log_spectrogram(
    signal: np.ndarray, rate: int, t0: float, t1: float, n_fft: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(times_s, freqs_hz, dB)`` of *signal* between *t0* and *t1* on its own clock.

    Time outside the recording is silence, so two previews cut at the same
    width line up column for column however the files overlap.
    """
    n = int(round((t1 - t0) * rate))
    i0 = int(round(t0 * rate))
    piece = np.zeros(n, dtype=np.float32)
    lo, hi = max(i0, 0), min(i0 + n, len(signal))
    if hi > lo:
        piece[lo - i0 : hi - i0] = signal[lo:hi]
    hop = n_fft // 4
    freqs, times, power = spectrogram(piece, fs=rate, nperseg=n_fft, noverlap=n_fft - hop)
    return times + t0, freqs, 10.0 * np.log10(power + _LOG_FLOOR)


def loudest_time(signal: np.ndarray, rate: int, t0: float, t1: float, window_s: float = 1.0) -> float:
    """Centre of the loudest *window_s* of *signal* inside ``[t0, t1]`` — where a preview should look."""
    i0, i1 = max(int(t0 * rate), 0), min(int(t1 * rate), len(signal))
    win = max(int(window_s * rate), 1)
    if i1 - i0 <= win:
        return (max(t0, 0.0) + min(t1, len(signal) / rate)) / 2
    energy = np.cumsum(np.concatenate([[0.0], signal[i0:i1].astype(np.float64) ** 2]))
    rms = energy[win:] - energy[:-win]
    return (i0 + int(np.argmax(rms)) + win / 2) / rate


# ── Session ──

#: Device name of a camera's own soundtrack in the alignment, so a panel reads
#: ``audio — cam-1-track`` next to the microphone's ``mic-1``.
TRACK_DEVICE_SUFFIX = "-track"
#: Prefix of the soundtrack's WAV inside the session, so its filename says what it is.
TRACK_FILE_PREFIX = "video-track_"


@dataclass
class AlignedPair:
    """One trial: a video, the microphone file placed on its clock, and the camera's own soundtrack."""

    video: str
    audio: str
    offset_s: float
    score: float
    track: str | None = None
    accepted: bool = True

    def describe_offset(self) -> str:
        """``"audio starts 5.00 s before video"`` — the sentence the user checks."""
        if abs(self.offset_s) < 0.0005:
            return "audio and video start together"
        when = "before" if self.offset_s < 0 else "after"
        return f"audio starts {abs(self.offset_s):.2f} s {when} video"


def session_pairing_table(pairs: list[AlignedPair]) -> tuple[pd.DataFrame, dict[str, float]]:
    """The :func:`~ethograph.io.pairing.pair_media` table for *pairs*, one trial per video.

    Trials follow one another on the session clock, each as long as its video.
    The microphone file's ``audio_mic-1_start`` is the trial's start plus the
    offset, which is what :meth:`~ethograph.io.nwb_alignment.NWBAlignment.stream_offset_for_trial`
    hands back to playback. Every rate is read from the files.
    """
    from ethograph.utils.stream_durations import probe_duration

    if not pairs:
        raise ValueError("No pairs to build a session from")
    rows = []
    cursor = 0.0
    rates: dict[str, float] = {}
    for i, pair in enumerate(pairs, start=1):
        duration = probe_duration(pair.video, "video")
        if duration is None:
            raise ValueError(f"Could not read the duration of {Path(pair.video).name}")
        row = {
            "trial": i,
            "start_time": cursor,
            "stop_time": cursor + duration,
            "video_cam-1": pair.video,
            "audio_mic-1": pair.audio,
            "audio_mic-1_start": cursor + pair.offset_s,
        }
        if pair.track is not None:
            row[f"audio_cam-1{TRACK_DEVICE_SUFFIX}"] = pair.track
        rows.append(row)
        cursor += duration
    table = pd.DataFrame(rows)
    rates["video_cam-1"] = _video_fps(pairs[0].video)
    rates["audio_mic-1"] = _audio_rate(pairs[0].audio)
    if pairs[0].track is not None:
        rates[f"audio_cam-1{TRACK_DEVICE_SUFFIX}"] = _audio_rate(pairs[0].track)
    return table, rates


def _video_fps(path: str) -> float:
    from ethograph.gui.video_manager import probe_video

    fps = probe_video(path).fps
    if not fps:
        raise ValueError(f"Could not read the frame rate of {Path(path).name}")
    return float(fps)


def _audio_rate(path: str) -> float:
    import soundfile as sf

    return float(sf.info(str(path)).samplerate)
