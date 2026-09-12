"""Alignment of media streams, feature data and trial timing from alignment.nwb."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pynwb import NWBFile

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPOCH_GAP = 1e-4
_KNOWN_STREAMS = ("video", "pose", "audio", "ephys")
_NWB_FILENAME = "alignment.nwb"
_SETTINGS_DIR = ".ethograph"
_SENTINEL = object()


@dataclass(frozen=True)
class _FileSpan:
    """One external file of an ImageSeries and where it sits in session time."""

    index: int
    path: str
    t_start: float
    t_end: float


def _span_overlapping(spans: list[_FileSpan], t0: float, t1: float | None) -> _FileSpan | None:
    """The span overlapping ``[t0, t1]`` most; ``None`` when none does.

    With no ``t1`` the span containing ``t0`` wins. A trial that starts in a
    gap just before a file (a triggered camera's first frame lands after the
    trigger) still takes that file when ``t1`` reaches into it.
    """
    if t1 is None:
        for s in spans:
            if s.t_start <= t0 <= s.t_end:
                return s
        return None
    best: _FileSpan | None = None
    best_overlap = 0.0
    for s in spans:
        overlap = min(s.t_end, t1) - max(s.t_start, t0)
        if overlap > best_overlap:
            best, best_overlap = s, overlap
    return best


# Extension → stream mapping, checked in order (first match wins).
# Video before audio because .mp4/.avi/.mov are in both sets but are
# primarily video containers.
_STREAM_EXTENSION_MAP: list[tuple[str, set[str]]] = []  # populated lazily


def _get_stream_extension_map() -> list[tuple[str, set[str]]]:
    """Build (stream, extensions) list on first call."""
    if _STREAM_EXTENSION_MAP:
        return _STREAM_EXTENSION_MAP
    from ethograph.io.validation import (
        AUDIO_EXTENSIONS,
        POSE_EXTENSIONS,
        VIDEO_EXTENSIONS,
    )

    _STREAM_EXTENSION_MAP.extend(
        [
            ("video", VIDEO_EXTENSIONS),
            ("pose", POSE_EXTENSIONS),
            ("audio", AUDIO_EXTENSIONS),
        ]
    )
    return _STREAM_EXTENSION_MAP


def _classify_imageseries(nwb) -> dict[str, list[str]]:
    """Classify acquisition ImageSeries by the extensions of their external files.

    Returns ``{stream: [acq_name, ...]}`` for items whose names don't follow
    the ``{stream}_{device}`` convention but whose external files match known
    media extensions.
    """
    from pynwb.image import ImageSeries

    result: dict[str, list[str]] = {}
    if not nwb.acquisition:
        return result

    ext_map = _get_stream_extension_map()

    for name, obj in nwb.acquisition.items():
        if not isinstance(obj, ImageSeries):
            continue
        if obj.external_file is None or len(obj.external_file) == 0:
            continue

        # Already follows {stream}_{device} convention — skip classification
        if any(name.startswith(f"{s}_") for s in _KNOWN_STREAMS):
            continue

        # Check the first non-empty external file extension
        ext = None
        for f in obj.external_file:
            f_str = str(f).strip()
            if f_str:
                ext = Path(f_str).suffix.lower()
                break
        if not ext:
            continue

        for stream, exts in ext_map:
            if ext in exts:
                result.setdefault(stream, []).append(name)
                break

    return result


def discover_nwb(nc_path: str | Path) -> Path | None:
    """Find an NWB session file near a data file.

    Search order:
    1. ``<dir>/.ethograph/alignment.nwb``
    2. Any ``.nwb`` file in ``<dir>/.ethograph/``
    """
    d = Path(nc_path).resolve().parent
    ethograph_dir = d / _SETTINGS_DIR
    if ethograph_dir.is_dir():
        candidate = ethograph_dir / _NWB_FILENAME
        if candidate.exists():
            return candidate
        nwb_files = list(ethograph_dir.glob("*.nwb"))
        if nwb_files:
            return nwb_files[0]
    return None


def make_nwb_alignment(nwb_path: str | Path | None = None):
    """Create a EmpytAlignment from an NWB path, falling back to base EmpytAlignment."""
    if nwb_path is None:
        return EmpytAlignment()
    if Path(nwb_path).exists():
        return NWBAlignment(nwb_path)
    # A path that was asked for but isn't there is bad user input, not an
    # absent alignment: without this the GUI silently behaves as if no
    # alignment file had been named at all.
    logger.warning("Alignment file not found: %s — continuing without alignment", nwb_path)
    return EmpytAlignment()


# ---------------------------------------------------------------------------
# Base class (also serves as null-object when no NWB is loaded)
# ---------------------------------------------------------------------------


class EmpytAlignment:
    """Base session metadata interface with null-object defaults.

    ``NWBAlignment`` overrides these with real NWB-backed implementations.
    When no NWB file is available, the base class is used directly.
    """

    @property
    def trials_df(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_media(self, trial, stream: str, device: str | None = None) -> str | None:
        return None

    def media_filename(self, trial, stream: str, device: str | None = None) -> str | None:
        return None

    def devices(self, stream: str) -> list[str]:
        return []

    @property
    def cameras(self) -> list[str]:
        return self.devices("video")

    @property
    def mics(self) -> list[str]:
        return self.devices("audio")

    def start_time(self, trial) -> float:
        return 0.0

    def stop_time(self, trial) -> float | None:
        return None

    def stream_offset_for_trial(self, trial, stream: str, device: str | None = None) -> float:
        return 0.0

    def get_stream_rate(self, stream: str, device: str | None = None) -> float | None:
        return None

    def set_stream_rate(self, rate: float, stream: str, device: str | None = None) -> None:
        pass

    def resolve_media_path(
        self,
        trial,
        stream: str,
        device: str | None = None,
        fallback_folder: str | None = None,
    ) -> str | None:
        return None

    def electrical_series(self) -> list[dict]:
        """Discover ElectricalSeries in acquisition. Returns list of {name, path, n_channels, rate}."""
        return []

    @property
    def trials_ep(self) -> Any:
        """Pynapple IntervalSet built from trials_df start/stop times."""
        cached = getattr(self, "_trials_ep_cache", _SENTINEL)
        if cached is not _SENTINEL:
            return cached
        ep = _build_trials_ep(self.trials_df)
        self._trials_ep_cache = ep
        return ep

    def trial_epoch(self, trial) -> Any:
        raise ValueError("No timing information available")

    def restrict(self, obj: Any, trial) -> Any:
        raise ValueError("No timing information available")

    def print_session(self) -> None:
        print("No session table.")

    def close(self) -> None:
        pass

    def reload(self) -> None:
        pass


class TableAlignment(EmpytAlignment):
    """Alignment backed by a tabular dataframe with trial timing columns.

    Expected columns are ``trial``, ``start_time``, and ``stop_time``.
    This is used as a fallback when no suitable alignment NWB is available.
    """

    def __init__(self, trials_df: pd.DataFrame) -> None:
        self._trials_df = trials_df.copy()

    @property
    def trials_df(self) -> pd.DataFrame:
        return self._trials_df

    def _trial_row(self, trial) -> pd.Series | None:
        df = self._trials_df
        if df.empty or "trial" not in df.columns:
            return None
        match = df[df["trial"] == trial]
        if match.empty:
            match = df[df["trial"] == str(trial)]
        if match.empty and isinstance(trial, str) and trial.isdigit():
            match = df[df["trial"] == int(trial)]
        if match.empty:
            return None
        return match.iloc[0]

    def start_time(self, trial) -> float:
        row = self._trial_row(trial)
        if row is None or "start_time" not in row.index or pd.isna(row["start_time"]):
            return 0.0
        return float(row["start_time"])

    def stop_time(self, trial) -> float | None:
        row = self._trial_row(trial)
        if row is None or "stop_time" not in row.index or pd.isna(row["stop_time"]):
            return None
        return float(row["stop_time"])


# ---------------------------------------------------------------------------
# Column parsing helpers
# ---------------------------------------------------------------------------


def _parse_stream_columns(columns: list[str], stream: str) -> list[str]:
    """Extract device names from trial table columns for a given stream.

    Columns like ``video_cam_1``, ``pose_cam_2`` -> ``["cam_1", "cam_2"]``.
    Excludes ``_start`` suffix columns (those are timing, not media).
    """
    devices = []
    prefix = f"{stream}_"
    for col in columns:
        if col.startswith(prefix) and not col.endswith("_start"):
            dev = col[len(prefix) :]
            if dev and dev not in devices:
                devices.append(dev)
    return devices


# ---------------------------------------------------------------------------
# NWBAlignment
# ---------------------------------------------------------------------------


class NWBAlignment:
    """Session metadata backed by an NWB file.

    All external media (video, audio, pose) are stored as ImageSeries
    in acquisition with ``{stream}_{device}`` naming.  Timing comes from
    ``rate`` or ``timestamps`` on the ImageSeries.
    """

    def __init__(self, nwb_path: str | Path) -> None:
        self._path = Path(nwb_path)
        self._io: Any = None
        self._nwb: Any = None
        self._trials_df_cache: pd.DataFrame | None = None
        self._rate_dict: dict[tuple[str, str | None], float] = {}
        self._span_cache: dict[str, list[_FileSpan]] = {}

    @classmethod
    def from_nwb_object(cls, nwb_obj) -> "NWBAlignment":
        """Create alignment from an already-opened pynwb NWBFile object."""
        instance = cls.__new__(cls)
        instance._path = Path(".")
        instance._io = None
        instance._nwb = nwb_obj
        instance._trials_df_cache = None
        instance._rate_dict = {}
        instance._span_cache = {}
        return instance

    def _open(self) -> None:
        if self._nwb is not None:
            return
        from pynwb import NWBHDF5IO

        self._io = NWBHDF5IO(str(self._path), "r")
        self._nwb = self._io.read()

    @property
    def nwb(self):
        self._open()
        return self._nwb

    # ── Trials table ──

    @property
    def trials_df(self) -> pd.DataFrame:
        if self._trials_df_cache is not None:
            return self._trials_df_cache
        nwb = self.nwb
        if nwb.trials is None:
            self._trials_df_cache = pd.DataFrame()
            return self._trials_df_cache
        df = nwb.trials.to_dataframe()
        self._trials_df_cache = df
        return self._trials_df_cache

    def _trial_row(self, trial) -> pd.Series | None:
        df = self.trials_df
        if df.empty:
            return None
        if "trial" in df.columns:
            match = df[df["trial"] == trial]
            if match.empty:
                match = df[df["trial"] == str(trial)]
            if match.empty and isinstance(trial, str) and trial.isdigit():
                match = df[df["trial"] == int(trial)]
            if not match.empty:
                return match.iloc[0]
            # Single-row tables (e.g. "session" label) — fall back to the only row
            if len(df) == 1:
                return df.iloc[0]
            return None
        # No trial column: use 1-based integer lookup into row index
        idx = self._trial_to_iloc(trial)
        if idx is not None and 0 <= idx < len(df):
            return df.iloc[idx]
        return None

    def _trial_index(self, trial) -> int | None:
        df = self.trials_df
        if df.empty:
            return None
        if "trial" in df.columns:
            for i, val in enumerate(df["trial"]):
                if val == trial or str(val) == str(trial):
                    return i
            return None
        return self._trial_to_iloc(trial)

    @staticmethod
    def _trial_to_iloc(trial) -> int | None:
        """Convert a trial ID to a 0-based row index (assumes 1-based trial IDs)."""
        try:
            idx = int(trial) - 1
            return idx if idx >= 0 else None
        except (ValueError, TypeError):
            return None

    # ── Acquisition lookup ──

    def _find_acquisition(self, stream: str, device: str | None = None):
        """Find an acquisition item by ``{stream}_{device}`` or by direct name.

        Handles extension-classified items where the device name *is* the
        acquisition name (not prefixed with ``{stream}_``).
        """
        nwb = self.nwb
        if not nwb.acquisition:
            return None
        acq_name = f"{stream}_{device}" if device else stream
        acq = nwb.acquisition.get(acq_name)
        if acq is not None:
            return acq
        # Fallback: device name is the raw acquisition name
        if device:
            return nwb.acquisition.get(device)
        return None

    # ── Media access ──

    def get_media(self, trial, stream: str, device: str | None = None) -> str | None:
        """Return the trial table's media filename for *stream*/*device*.

        A stored value that is an absolute local path (e.g. baked in by a
        template authored on a different machine) is reduced to its
        basename, so callers can always safely join it onto their own
        video/audio/pose folder instead of resolving to a path that only
        existed on the authoring machine.
        """
        row = self._trial_row(trial)
        if row is None:
            return None
        col = f"{stream}_{device}" if device else stream
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val):
                val = str(val)
                if not _is_url(val) and os.path.isabs(val):
                    return os.path.basename(val)
                return val
        return None

    def media_filename(self, trial, stream: str, device: str | None = None) -> str | None:
        """The name of the file *trial* plays for *stream*/*device*.

        Picks the same file :meth:`resolve_media_path` would, but names it
        without resolving it on disk — the trials table reads the same whether
        or not the media folder is currently attached.
        """
        acq = self._find_acquisition(stream, device)
        if acq is not None and getattr(acq, "external_file", None):
            files = list(acq.external_file)
            found = self._file_for_trial(acq, trial)
            if found is not None:
                idx = found.index
            else:
                trial_idx = self._trial_index(trial)
                idx = trial_idx if trial_idx is not None and trial_idx < len(files) else 0
            if idx < len(files):
                return _filename_from_url_or_path(str(files[idx]))

        media = self.get_media(trial, stream, device)
        return _filename_from_url_or_path(media) if media else None

    def devices(self, stream: str) -> list[str]:
        """Discover devices from trials table columns AND acquisition items.

        Three sources, checked in order:

        1. Trials table columns (``video_cam_1`` -> device ``cam_1``).
        2. Acquisition ImageSeries following ``{stream}_{device}`` naming.
        3. Acquisition ImageSeries whose ``external_file`` extensions match
           known media types (e.g. ``.mp4`` -> video, ``.wav`` -> audio).
        """
        devs: list[str] = []

        # 1. From trials table columns
        df = self.trials_df
        if not df.empty:
            devs = _parse_stream_columns(list(df.columns), stream)

        # 2. From acquisition ImageSeries with {stream}_{device} naming
        nwb = self.nwb
        prefix = f"{stream}_"
        if nwb.acquisition:
            for name in nwb.acquisition:
                if name.startswith(prefix):
                    dev = name[len(prefix) :]
                    if dev and dev not in devs:
                        devs.append(dev)

        # 3. From extension-based classification of remaining ImageSeries
        classified = _classify_imageseries(nwb)
        for acq_name in classified.get(stream, []):
            if acq_name not in devs:
                devs.append(acq_name)

        return devs

    @property
    def cameras(self) -> list[str]:
        return self.devices("video")

    @property
    def mics(self) -> list[str]:
        return self.devices("audio")

    def electrical_series(self) -> list[dict]:
        """Discover ElectricalSeries in acquisition. Returns list of {name, path, n_channels, rate}."""
        import pynwb.ecephys

        nwb = self.nwb
        if not nwb.acquisition:
            return []
        results = []
        for name, obj in nwb.acquisition.items():
            if not isinstance(obj, pynwb.ecephys.ElectricalSeries):
                continue
            n_ch = obj.data.shape[1] if hasattr(obj.data, "shape") and obj.data.ndim > 1 else 1
            rate = float(obj.rate) if obj.rate else None
            results.append(
                {
                    "name": name,
                    "path": str(self._path),
                    "n_channels": n_ch,
                    "rate": rate,
                }
            )
        return results

    # ── Timing ──

    @cached_property
    def has_real_timing(self) -> bool:
        """Whether the trials table has meaningful start/stop times.

        Returns False when all trials have identical placeholder timing
        (e.g. start=0.0, stop=1.0 for every row), which indicates the
        NWB was generated without real session timing.
        """
        df = self.trials_df
        if df.empty or "start_time" not in df.columns:
            return False
        starts = df["start_time"].dropna()
        if len(starts) == 0:
            return False
        if len(starts) > 1 and starts.nunique() > 1:
            return True
        # Single unique start: check if stop_time varies or any duration > 1.001
        if "stop_time" in df.columns:
            stops = df["stop_time"].dropna()
            durations = stops - starts.iloc[0]
            if durations.nunique() > 1:
                return True
            if (durations > 1.001).any():
                return True
        return False

    @property
    def pose_keys(self) -> list[str]:
        """Pose estimation container names from acquisition ImageSeries."""
        return self.devices("pose")

    def start_time(self, trial) -> float:
        if not self.has_real_timing:
            return 0.0
        row = self._trial_row(trial)
        if row is not None and "start_time" in row.index:
            val = row["start_time"]
            if pd.notna(val):
                return float(val)
        return 0.0

    def stop_time(self, trial) -> float | None:
        if not self.has_real_timing:
            return None
        row = self._trial_row(trial)
        if row is None:
            return None
        if "stop_time" in row.index:
            val = row["stop_time"]
            if pd.notna(val):
                return float(val)
        return None

    def _session_end_time(self) -> float | None:
        """Largest ``stop_time`` across trials, or ``None`` if unavailable."""
        df = self.trials_df
        if df.empty or "stop_time" not in df.columns:
            return None
        stops = df["stop_time"].dropna()
        return float(stops.max()) if not stops.empty else None

    def stream_offset_for_trial(
        self,
        trial,
        stream: str,
        device: str | None = None,
    ) -> float:
        """Trial-relative time of sample 0 of the file that holds *trial*.

        0.0 for per-trial files, negative for a session-wide file that
        started before the trial. The file is found by time (the trial's
        start falls in its span), never by position -- see ``_file_for_trial``.
        """
        acq = self._find_acquisition(stream, device)
        if acq is None:
            return 0.0
        found = self._file_for_trial(acq, trial)
        if found is None:
            return 0.0
        return found.t_start - self.start_time(trial)

    # ── File time spans ──

    def _acq_spans(self, acq) -> list[_FileSpan]:
        """Every external file of *acq* with its session-time span.

        Handles both NWB timing schemes (``timestamps`` and ``rate``). A last
        file whose end is unknown (rate mode, no ``num_samples``, no trials
        table) gets ``t_end = inf``; a file with no start at all is dropped.

        Cached per acquisition — the trials table asks once per trial for a
        filename; :meth:`reload` clears it.
        """
        cached = self._span_cache.get(acq.name)
        if cached is not None:
            return cached

        files = list(getattr(acq, "external_file", None) or [])
        if not files:
            return []
        raw_sf = getattr(acq, "starting_frame", None)
        starting_frame = [int(f) for f in raw_sf] if raw_sf is not None else [0] * len(files)
        timestamps = getattr(acq, "timestamps", None)
        rate = getattr(acq, "rate", None)
        starting_time = float(acq.starting_time) if getattr(acq, "starting_time", None) is not None else 0.0
        ts = np.asarray(timestamps) if timestamps is not None and len(timestamps) > 0 else None

        spans: list[_FileSpan] = []
        for i, filepath in enumerate(files):
            frame_start = starting_frame[i] if i < len(starting_frame) else 0
            frame_end = starting_frame[i + 1] if i + 1 < len(starting_frame) else None

            if ts is not None:
                t_start = float(ts[frame_start]) if frame_start < len(ts) else float(ts[0])
                if frame_end is not None and frame_end <= len(ts):
                    t_end = float(ts[frame_end - 1])
                else:
                    t_end = float(ts[-1])
            elif rate and rate > 0:
                t_start = starting_time + frame_start / rate
                if frame_end is not None:
                    t_end = starting_time + frame_end / rate
                else:
                    num = getattr(acq, "num_samples", None)
                    if num:
                        t_end = starting_time + int(num) / rate
                    else:
                        session_end = self._session_end_time()
                        t_end = session_end if session_end is not None else float("inf")
            else:
                continue
            spans.append(_FileSpan(i, str(filepath), t_start, t_end))
        self._span_cache[acq.name] = spans
        return spans

    def _file_for_trial(self, acq, trial) -> _FileSpan | None:
        """The external file of *acq* that holds *trial*.

        With real trial timing the file is the one overlapping the trial most
        (the trial's start lies in it; a trial that starts in a gap before a
        triggered file takes that file). Without timing every trial starts at
        0.0 and time says nothing, so the trial's position is used -- one file
        per trial when the counts agree, else the first file.
        """
        spans = self._acq_spans(acq)
        if not spans:
            return None
        if self.has_real_timing:
            best = _span_overlapping(spans, self.start_time(trial), self.stop_time(trial))
            if best is not None:
                return best
        trial_idx = self._trial_index(trial)
        if trial_idx is not None and len(spans) == len(self.trials_df) and trial_idx < len(spans):
            return spans[trial_idx]
        return spans[0]

    def file_time_spans(self, stream: str, device: str | None = None) -> list[tuple[str, float, float]]:
        """Return [(filepath, t_start, t_end), ...] for each external file in the stream.

        Files whose end cannot be determined are skipped.
        """
        acq = self._find_acquisition(stream, device)
        if acq is None:
            return []
        return [
            (s.path, s.t_start, s.t_end) for s in self._acq_spans(acq) if np.isfinite(s.t_end) and s.t_end > s.t_start
        ]

    # ── Stream rate ──

    def get_stream_rate(self, stream: str, device: str | None = None) -> float | None:
        """Read the sampling rate for a stream from its acquisition ImageSeries."""
        key = (stream, device)
        if key in self._rate_dict:
            return self._rate_dict[key]

        from ethograph.utils.nwb import resolve_timeseries_timing

        acq = self._find_acquisition(stream, device)
        if acq is not None:
            rate, _ = resolve_timeseries_timing(acq)
            return rate

        # Fallback: scan all {stream}_* items
        nwb = self.nwb
        if nwb.acquisition:
            for name, acq in nwb.acquisition.items():
                if name.startswith(f"{stream}_"):
                    rate, _ = resolve_timeseries_timing(acq)
                    return rate

        return None

    def set_stream_rate(self, rate: float, stream: str, device: str | None = None) -> None:
        self._rate_dict[(stream, device)] = rate

    def stream_rates(self) -> dict[str, tuple[float, float | None]]:
        """Return ``{acq_name: (starting_time, rate)}`` for all acquisition streams.

        Only returns rate and starting_time from NWB ImageSeries metadata.
        Does not compute duration — external files must be probed for that.
        """
        from ethograph.utils.nwb import resolve_timeseries_timing

        nwb = self.nwb
        if not nwb.acquisition:
            return {}
        result: dict[str, tuple[float, float | None]] = {}
        for name, acq in nwb.acquisition.items():
            rate, _ = resolve_timeseries_timing(acq)
            start = float(acq.starting_time) if acq.starting_time is not None else 0.0
            result[name] = (start, rate)
        return result

    # ── Media path resolution ──

    def resolve_media_path(
        self,
        trial,
        stream: str,
        device: str | None = None,
        fallback_folder: str | None = None,
    ) -> str | None:
        """Resolve the full path for a media file.

        1. Try the ImageSeries ``external_file`` path for this trial (if on disk).
        2. Fallback: trial table filename + ``fallback_folder``.
        3. Returns ``None`` if unresolvable.
        """
        acq = self._find_acquisition(stream, device)

        trial_idx = self._trial_index(trial)
        nwb_base_dir = self._path.parent

        if acq is not None and hasattr(acq, "external_file") and acq.external_file:
            files = list(acq.external_file)
            found = self._file_for_trial(acq, trial)
            if found is not None:
                file_idx = found.index
            elif trial_idx is not None and trial_idx < len(files):
                file_idx = trial_idx
            else:
                file_idx = 0

            if file_idx < len(files):
                raw_path = files[file_idx]
                # URLs returned directly
                if _is_url(raw_path):
                    # If fallback_folder has a local copy, prefer that
                    if fallback_folder:
                        filename = _filename_from_url_or_path(raw_path)
                        candidate = os.path.normpath(os.path.join(fallback_folder, filename))
                        if os.path.isfile(candidate):
                            return candidate
                    return raw_path
                # Try the stored path directly
                if os.path.isfile(raw_path):
                    return raw_path
                # Try relative to NWB file location
                rel = nwb_base_dir / raw_path
                if rel.is_file():
                    return str(rel)
                # Fallback: filename + folder
                filename = _filename_from_url_or_path(raw_path)
                if fallback_folder:
                    candidate = os.path.normpath(os.path.join(fallback_folder, filename))
                    if os.path.isfile(candidate):
                        return candidate

        # Last resort: trial table filename + fallback_folder
        media_file = self.get_media(trial, stream, device)
        if media_file:
            if _is_url(media_file):
                return media_file
            if os.path.isfile(media_file):
                return media_file
            if fallback_folder:
                filename = _filename_from_url_or_path(media_file)
                candidate = os.path.normpath(os.path.join(fallback_folder, filename))
                if os.path.isfile(candidate):
                    return candidate

        return None

    # ── Epochs (pynapple IntervalSet) ──

    @cached_property
    def trials_ep(self):
        """Pynapple IntervalSet built from the NWB trials table."""
        return _build_trials_ep(self.trials_df)

    def _trial_ep_idx(self, trial) -> int:
        ep = self.trials_ep
        if ep is None:
            raise KeyError(f"Trial {trial} not found in timing table")
        for i, t in enumerate(ep.metadata["trial"]):
            if t == trial or str(t) == str(trial):
                return i
        raise KeyError(f"Trial {trial} not found in timing table")

    def trial_epoch(self, trial):
        import pynapple as nap

        ep = self.trials_ep
        if ep is None:
            raise ValueError("No timing information available")
        idx = self._trial_ep_idx(trial)
        return nap.IntervalSet(start=ep.start[idx], end=ep.end[idx])

    def restrict(self, obj, trial):
        return obj.restrict(self.trial_epoch(trial))

    # ── Display ──

    def print_session(self) -> None:
        df = self.trials_df
        if df.empty:
            print("No session table.")
            return
        print(f"\n{'=' * 60}")
        print(f"  NWB Session: {self._path.name}")
        print(f"  {len(df)} trials, columns: {list(df.columns)}")
        print(f"{'=' * 60}")
        print(df.to_string(max_rows=20))

        nwb = self.nwb
        if nwb.acquisition:
            print("\n  Acquisition items:")
            for name, acq in nwb.acquisition.items():
                info = f"    {name}"
                if hasattr(acq, "rate") and acq.rate:
                    info += f" (rate={acq.rate} Hz)"
                if hasattr(acq, "external_file") and acq.external_file:
                    info += f" ({len(acq.external_file)} files)"
                if hasattr(acq, "timestamps") and acq.timestamps is not None:
                    info += f" ({len(acq.timestamps)} timestamps)"
                print(info)

    # ── Cleanup ──

    def close(self) -> None:
        if self._io is not None:
            try:
                self._io.close()
            except Exception:
                pass
            self._io = None
            self._nwb = None

    def reload(self) -> None:
        """Drop the open handle and every cached table.

        Required around writing to this same file: pynwb holds it open for
        reading and HDF5 refuses a second, writable handle — and afterwards the
        cached trials table would still describe the pre-write file.
        """
        self.close()
        self._trials_df_cache = None
        self._rate_dict.clear()
        self._span_cache.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def _filename_from_url_or_path(path: str) -> str:
    """Extract filename from a URL or filesystem path (Windows-safe)."""
    if _is_url(path):
        from urllib.parse import urlparse

        return Path(urlparse(path).path).name
    return Path(path).name


def _coerce_trial_id(value):
    """Coerce a trial ID to a plain Python int or str.

    ``DataFrame.iterrows()`` upcasts rows to the columns' common dtype, so an
    int trial column becomes float64 when timing columns are present — coerce
    integral floats back to int before writing to NWB.
    """
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _build_trials_ep(df: pd.DataFrame, session_end: float | None = None):
    """Build a pynapple IntervalSet from a trials DataFrame with start/stop times.

    Parameters
    ----------
    df
        Trials DataFrame with ``start_time`` and optionally ``stop_time``.
    session_end
        Session end time (seconds). Used as fallback stop for the last trial
        when its stop time is unknown. Pass ``source_collection.union_range.end_s``
        after data loading.

    Returns None when timing data is missing or non-monotonic.
    """
    if df.empty or "start_time" not in df.columns:
        return None

    import pynapple as nap

    starts = df["start_time"].values.astype(np.float64)
    n = len(starts)
    if n == 0:
        return None

    ends = np.full(n, np.nan, dtype=np.float64)
    if "stop_time" in df.columns:
        ends = df["stop_time"].values.astype(np.float64)

    has_monotonic = n == 1 or bool(np.all(np.diff(starts) > 0))
    if not has_monotonic:
        return None

    has_stop = ~np.isnan(ends)
    safe_ends = ends.copy()
    for i in range(n - 1):
        if np.isnan(safe_ends[i]):
            safe_ends[i] = starts[i + 1] - _EPOCH_GAP
        elif safe_ends[i] >= starts[i + 1]:
            safe_ends[i] = starts[i + 1] - _EPOCH_GAP

    trial_ids = df["trial"].values if "trial" in df.columns else np.arange(1, n + 1)

    # Last trial with unknown stop: use session end if available, else drop
    if np.isnan(safe_ends[-1]):
        if session_end is not None and session_end > starts[-1]:
            safe_ends[-1] = session_end
        else:
            mask = np.ones(n, dtype=bool)
            mask[-1] = False
            starts = starts[mask]
            safe_ends = safe_ends[mask]
            has_stop = has_stop[mask]
            trial_ids = trial_ids[mask]
            if len(starts) == 0:
                return None

    return nap.IntervalSet(
        start=starts,
        end=safe_ends,
        metadata={
            "trial": np.array(trial_ids),
            "has_stop": has_stop.astype(float),
        },
    )


def _parse_stream_devices(columns: list[str]) -> dict[str, list[str]]:
    """Detect ``{stream}_{device}`` columns -> ``{stream: [device, ...]}``."""
    result: dict[str, list[str]] = {}
    for col in columns:
        for stream in _KNOWN_STREAMS:
            prefix = f"{stream}_"
            if col.startswith(prefix) and not col.endswith("_start"):
                device = col[len(prefix) :]
                if device:
                    result.setdefault(stream, []).append(device)
    return result


def _segments_contiguous(seg_starts: list[float], seg_nsamples: list[int], rate: float) -> bool:
    """True when every segment starts where the previous one ended (constant rate).

    A contiguous run can be stored as compact ``rate`` + ``starting_time`` metadata;
    a run with gaps needs per-sample ``timestamps`` to preserve those gaps.
    """
    if not rate or rate <= 0:
        return False
    tol = 0.5 / rate
    expected = seg_starts[0]
    for start, n in zip(seg_starts, seg_nsamples):
        if abs(start - expected) > tol:
            return False
        expected = start + int(n) / rate
    return True


def _add_external_series(
    nwbfile: NWBFile,
    name: str,
    files: list[str],
    seg_starts: list[float],
    seg_nsamples: list[int],
    rate: float | None,
) -> None:
    """Add an external-file ImageSeries with compact timing where possible.

    Uses ``rate`` + ``starting_time`` for a single contiguous constant-rate
    timeline (the common case — sidecar stays tiny); falls back to dense
    per-sample ``timestamps`` only when segments have gaps a single rate can't
    express. With no rate, stores one frame per file at its start time.
    """
    from pynwb.image import ImageSeries

    files = list(files)
    seg_starts = [float(s) for s in seg_starts]
    if name in nwbfile.acquisition:
        del nwbfile.acquisition[name]

    if not rate or rate <= 0:
        nwbfile.add_acquisition(
            ImageSeries(
                name=name,
                description=name,
                external_file=files,
                format="external",
                starting_frame=np.arange(len(files), dtype=np.int32),
                timestamps=np.asarray(seg_starts, dtype=np.float64),
            )
        )
        return

    starting_frames = np.zeros(len(files), dtype=np.int32)
    acc = 0
    for i, n in enumerate(seg_nsamples):
        starting_frames[i] = acc
        acc += int(n)

    kwargs = dict(
        name=name,
        description=name,
        external_file=files,
        format="external",
        starting_frame=starting_frames,
    )

    if _segments_contiguous(seg_starts, seg_nsamples, rate):
        # external + rate carries no data array, so num_samples must be given
        # explicitly (this is also what lets readers find the last file's end).
        # uint64 is the dtype the NWB spec stores — a plain int would trigger
        # hdmf's int64→uint64 DtypeConversionWarning on every write.
        nwbfile.add_acquisition(
            ImageSeries(
                rate=float(rate),
                starting_time=seg_starts[0],
                num_samples=np.uint64(sum(int(n) for n in seg_nsamples)),
                **kwargs,
            )
        )
    else:
        ts = np.concatenate([s + np.arange(int(n)) / rate for s, n in zip(seg_starts, seg_nsamples)])
        nwbfile.add_acquisition(ImageSeries(timestamps=ts, **kwargs))


def sync_acquisition_for_streams(
    nwbfile: NWBFile,
    stream_rates: dict[str, float],
) -> None:
    """Create ImageSeries acquisition items for ALL external media streams.

    Reads the trials table to discover ``{stream}_{device}`` columns.
    For each stream+device pair, creates an ``ImageSeries`` in
    ``nwbfile.acquisition`` with ``external_file``, ``starting_frame``,
    and ``rate`` (or ``timestamps`` if offsets are present).

    Parameters
    ----------
    nwbfile
        NWB file with a populated trials table.
    stream_rates
        Mapping of stream name to sampling rate, e.g.
        ``{"video": 30.0, "audio": 44100.0, "pose": 30.0}``.
    """
    df = nwbfile.trials.to_dataframe()
    stream_devices = _parse_stream_devices(list(df.columns))

    for stream, devices in stream_devices.items():
        rate = stream_rates.get(stream)
        if rate is None or rate <= 0:
            continue

        for device in devices:
            col = f"{stream}_{device}"
            if col not in df.columns:
                continue

            valid = df[df[col] != ""]
            if valid.empty:
                continue

            external_files = valid[col].tolist()

            start_col = f"{col}_start"
            if start_col in df.columns:
                starts = valid[start_col].values.astype(float)
            else:
                starts = valid["start_time"].values.astype(float)

            seg_starts: list[float] = []
            seg_nsamples: list[int] = []
            for i, (_, row) in enumerate(valid.iterrows()):
                duration = float(row["stop_time"]) - float(row["start_time"])
                seg_starts.append(float(starts[i]))
                seg_nsamples.append(max(1, int(duration * rate)))

            if device not in [d.name for d in nwbfile.devices.values()]:
                nwbfile.create_device(name=device, description=f"{stream} device {device}")

            _add_external_series(nwbfile, f"{stream}_{device}", external_files, seg_starts, seg_nsamples, rate)


def _infer_times_from_media(
    trial_table: pd.DataFrame,
    video_cols: list[str],
    audio_cols: list[str],
    media_root: Path | None,
    pose_cols: list[str] | None = None,
    pose_fps: float | None = None,
) -> pd.DataFrame:
    if not (video_cols or audio_cols or pose_cols):
        raise ValueError("start_time/stop_time missing and no video_*/audio_*/pose_* columns to infer durations from")

    table = trial_table.copy()
    starts, stops = [], []
    cursor = 0.0
    for _, row in table.iterrows():
        duration = _probe_trial_duration(
            row,
            video_cols,
            audio_cols,
            media_root,
            pose_cols=pose_cols,
            pose_fps=pose_fps,
        )
        starts.append(cursor)
        stops.append(cursor + duration)
        cursor += duration

    table["start_time"] = starts
    table["stop_time"] = stops
    return table


def _probe_trial_duration(
    row: pd.Series,
    video_cols: list[str],
    audio_cols: list[str],
    media_root: Path | None,
    pose_cols: list[str] | None = None,
    pose_fps: float | None = None,
) -> float:
    from ethograph.utils.stream_durations import probe_duration

    for col in video_cols:
        path = _resolve_media_path(row.get(col), media_root)
        if path is not None:
            dur = probe_duration(str(path), "video")
            if dur is not None:
                return dur

    for col in audio_cols:
        path = _resolve_media_path(row.get(col), media_root)
        if path is not None:
            dur = probe_duration(str(path), "audio")
            if dur is not None:
                return dur

    if pose_cols and pose_fps is not None:
        for col in pose_cols:
            path = _resolve_media_path(row.get(col), media_root)
            if path is not None:
                dur = probe_duration(str(path), "pose", pose_fps)
                if dur is not None:
                    return dur

    raise ValueError(f"Could not probe duration for trial row: {row.to_dict()}")


def _resolve_media_path(filename: Any, media_root: Path | None) -> Path | None:
    if not filename or pd.isna(filename):
        return None
    path = Path(filename)
    if media_root and not path.is_absolute():
        path = media_root / path
    return path if path.exists() else None


def align_media_per_trial(
    trial_table: pd.DataFrame,
    stream_rates: dict[str, float] | None = None,
    output_path: str | Path | None = None,
    session_description: str = "NWB file for media alignment (ethograph generated).",
    media_root: str | Path | None = None,
    pose_fps: float | None = None,
) -> NWBFile:
    """Create an alignment.nwb from a trial table.

    .. deprecated::
        Thin wrapper over :func:`ethograph.io.pairing.pair_media` — kept so
        existing callers keep working. ``session_description`` is accepted
        for backward compatibility but is no longer threaded through
        (``pair_media`` has no per-call description); new code should call
        :func:`~ethograph.io.pairing.pair_media` (or
        :func:`~ethograph.io.pairing.discover_media` +
        :func:`~ethograph.io.pairing.pair_media`) directly.

    Parameters
    ----------
    trial_table
        DataFrame with ``trial`` column and ``{stream}_{device}`` filename
        columns.  Rows are trials, in order.  ``start_time`` / ``stop_time``
        are optional -- omit them and each trial's duration is probed from its
        own media, laying trials end to end from ``0.0``.
    stream_rates
        Sampling rate per stream.  Must include every stream that has
        columns in the table.  Example:
        ``{"video": 30.0, "audio": 48000.0, "pose": 30.0}``
    output_path
        Where to write the ``.nwb`` file.
    media_root
        Folder the filename columns are relative to.  Only needed when times
        are inferred, since probing must open the files.
    pose_fps
        Frame rate for probing pose files.  Required to infer times from a
        table whose only media columns are ``pose_*``.

    Returns
    -------
    The in-memory :class:`~pynwb.NWBFile`, written to ``output_path`` when given.

    Examples
    --------
    >>> import pandas as pd, ethograph as eto
    >>> table = pd.DataFrame(
    ...     {
    ...         "trial": [1, 2, 3],
    ...         "video_cam-1": ["t1.mp4", "t2.mp4", "t3.mp4"],
    ...         "pose_cam-1": ["t1.h5", "t2.h5", "t3.h5"],
    ...     }
    ... )
    >>> eto.align_media_per_trial(table, {"video": 30.0, "pose": 30.0}, "out/.ethograph/alignment.nwb")
    """
    from ethograph.io.pairing import pair_media  # local: pairing imports this module at top level

    return pair_media(
        trial_table,
        stream_rates=stream_rates,
        output_path=output_path,
        media_root=media_root,
        pose_fps=pose_fps,
    )


def trials_df_from_trials_ep(ep) -> pd.DataFrame:
    """Trials DataFrame (``trial``/``start_time``/``stop_time`` + metadata
    columns) from a pynapple trials IntervalSet."""
    meta = getattr(ep, "metadata", None)
    if meta is not None and "trial" in meta.columns:
        ids = [_coerce_trial_id(t) for t in meta["trial"]]
    else:
        ids = list(range(1, len(ep) + 1))
    df = pd.DataFrame(
        {
            "trial": ids,
            "start_time": np.asarray(ep.start, dtype=np.float64),
            "stop_time": np.asarray(ep.end, dtype=np.float64),
        }
    )
    if meta is not None:
        for col in meta.columns:
            if col not in df.columns:
                df[col] = list(meta[col])
    return df


def alignment_from_trials_ep(ep, output_path: str | Path) -> Path:
    """Write a bare ``alignment.nwb`` holding only a trials table from *ep*.

    The ONE sanctioned way a pynapple trials IntervalSet (``trials.npz``)
    becomes trial timing: converted once, on the user's explicit say-so
    (cover-page offer) — the loader itself never reads timing from the data.
    The IntervalSet's metadata columns (condition, region, …) travel into the
    trials table, so the alignment file is the single per-trial record.
    """
    from datetime import datetime
    from uuid import uuid4

    import pynwb
    from dateutil.tz import tzlocal
    from pynwb import NWBHDF5IO

    trials = trials_df_from_trials_ep(ep)

    nwbfile = pynwb.NWBFile(
        session_description="NWB file for trial alignment (ethograph generated from a trials IntervalSet).",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )
    extra_cols = [c for c in trials.columns if c not in ("start_time", "stop_time")]
    for col in extra_cols:
        nwbfile.add_trial_column(name=col, description=col)
    for _, row in trials.iterrows():
        values = {}
        for col in extra_cols:
            value = row[col]
            if col == "trial":
                value = _coerce_trial_id(value)
            elif not isinstance(value, (int, float, str, np.integer, np.floating)):
                value = str(value)
            values[col] = value
        nwbfile.add_trial(start_time=float(row["start_time"]), stop_time=float(row["stop_time"]), **values)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(output), "w") as io:
        io.write(nwbfile)
    return output


def align_media_from_streams(
    trials: pd.DataFrame,
    streams: list[dict],
    output_path: str | Path,
) -> Path:
    """Create an alignment.nwb for unaligned / complex scenarios.

    The trials table contains only timing (no filenames).  All file
    references go into ImageSeries acquisition items.

    Parameters
    ----------
    trials
        DataFrame with ``trial``, ``start_time``, ``stop_time``.
    streams
        List of stream dicts, each with::

            {
                "name": "video_cam-1",  # acquisition item name
                "files": ["t1.mp4", ...],  # one per trial (full paths)
                "rate": 30.0,  # sampling rate
            }

        For session-wide files (one file spanning all trials)::

            {
                "name": "audio_mic-1",
                "files": ["session.wav"],
                "rate": 44100.0,
                "starting_time": 0.0,  # when file starts in session time
            }

    A spec carrying a ``"timestamps"`` key is refused: per-sample timestamps are
    neuroconv's job — build the ``.nwb`` with ``ExternalVideoInterface`` (or the
    matching interface for the stream) and pair over it with
    :func:`ethograph.io.pairing.pair_media` instead.

    output_path
        Where to write the ``.nwb`` file.

    Returns
    -------
    Path to the created NWB file.

    Examples
    --------
    Per-trial video + pose, session-wide audio::

        >>> trials = pd.DataFrame({
        ...     "trial": [1, 2, 3],
        ...     "start_time": [0.0, 10.5, 22.3],
        ...     "stop_time": [8.2, 19.1, 30.0],
        ... })
        >>> streams = [
        ...     {"name": "video_cam-1", "files": ["t1.mp4", "t2.mp4", "t3.mp4"], "rate": 30.0},
        ...     {"name": "pose_cam-1", "files": ["t1.h5", "t2.h5", "t3.h5"], "rate": 30.0},
        ...     {"name": "audio_mic-1", "files": ["session.wav"], "rate": 48000.0, "starting_time": 0.0},
        ... ]
        >>> eto.align_media_from_streams(trials, streams, ".ethograph/alignment.nwb")
    """
    from datetime import datetime
    from uuid import uuid4

    import pynwb
    from dateutil.tz import tzlocal
    from pynwb import NWBHDF5IO

    for spec in streams:
        if "timestamps" in spec:
            raise ValueError(
                "Per-sample timestamps are neuroconv's job: build the .nwb with ExternalVideoInterface and pair over it"
            )

    nwbfile = pynwb.NWBFile(
        session_description="NWB file for media alignment (ethograph generated).",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )

    # Trials table: only timing, no filenames
    nwbfile.add_trial_column(name="trial", description="Trial number")
    for _, row in trials.iterrows():
        nwbfile.add_trial(
            trial=_coerce_trial_id(row["trial"]),
            start_time=float(row["start_time"]),
            stop_time=float(row["stop_time"]),
        )

    trial_starts = trials["start_time"].values.astype(float)
    trial_stops = trials["stop_time"].values.astype(float)
    n_trials = len(trials)

    for spec in streams:
        name = spec["name"]
        files = spec["files"]
        rate = spec.get("rate")
        starting_time = spec.get("starting_time", None)

        # Parse stream_device for device creation
        parts = name.split("_", 1)
        device_name = parts[1] if len(parts) > 1 else parts[0]
        if device_name not in [d.name for d in nwbfile.devices.values()]:
            nwbfile.create_device(name=device_name, description=f"Device {device_name}")

        if len(files) == 1 and n_trials > 1:
            # Session-wide: one file spanning all trials
            t0 = starting_time if starting_time is not None else float(trial_starts[0])
            n_samples = max(1, int((float(trial_stops[-1]) - t0) * rate)) if rate else 1
            _add_external_series(nwbfile, name, files, [t0], [n_samples], rate)
        else:
            # Per-trial: one file per trial
            seg_starts = []
            seg_nsamples = []
            for i in range(min(len(files), n_trials)):
                t0 = float(trial_starts[i])
                dur = float(trial_stops[i]) - t0
                seg_starts.append(t0)
                seg_nsamples.append(max(1, int(dur * rate)) if rate else 1)
            _add_external_series(nwbfile, name, files[:n_trials], seg_starts, seg_nsamples, rate)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(output), "w") as io:
        io.write(nwbfile)

    return output


import gc  # noqa: E402
import tempfile  # noqa: E402
from contextlib import contextmanager  # noqa: E402

from pynwb import NWBHDF5IO  # noqa: E402


@contextmanager
def edit_nwb(path):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(suffix=".nwb", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp)
    try:
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as read_io:
            nwbfile = read_io.read()
            yield nwbfile
            with NWBHDF5IO(str(tmp), "w") as write_io:
                write_io.export(src_io=read_io, nwbfile=nwbfile)
        del nwbfile
        gc.collect()
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _video_name_to_device(name: str) -> str:
    """Strip a leading ``video_`` / ``video-`` prefix so an ImageSeries name maps
    to the same device string that :meth:`NWBAlignment.cameras` reports."""
    for prefix in ("video_", "video-"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def pose_video_links_from_nwb(nwb) -> dict[str, str]:
    """Native ndx-pose pose↔video links from an open NWBFile:
    ``{pose_container_name: source_video_name}``.

    Reads ``PoseEstimation.source_video`` (ndx-pose >= 0.2.1), the NWB-native way
    to bind a pose estimation to its source video ``ImageSeries``. Containers with
    no ``source_video`` link are omitted (they fall back to manual matching).
    """
    links: dict[str, str] = {}
    for mod in nwb.processing.values():
        for name, di in mod.data_interfaces.items():
            if not hasattr(di, "pose_estimation_series"):
                continue
            source_video = getattr(di, "source_video", None)
            if source_video is not None:
                links[name] = source_video.name
    return links


def pose_keys_for_cameras(links: dict[str, str], cameras: list[str]) -> list[str | None]:
    """Order pose container names to match ``cameras`` via native source_video links.

    A pose container is assigned to camera ``c`` when its ``source_video`` name
    resolves to ``c`` (after stripping a ``video_`` / ``video-`` prefix). Returns a
    camera-aligned list with ``None`` for any camera that has no linked pose.
    """
    by_device = {_video_name_to_device(video_name): pose for pose, video_name in links.items()}
    return [by_device.get(str(c)) for c in cameras]
