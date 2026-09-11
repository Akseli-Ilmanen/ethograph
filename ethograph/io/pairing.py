"""Pairing media files to trials: the one library the Data wizard and its
generated notebooks call for Route 1 ("Pair my media files").

Route 1 assumes every file already starts with its trial — the recording was
triggered per trial, or a session-wide file's timing is already known — so
the user only has to say *which file is which*. :func:`discover_media` turns
a rig's folders/patterns into the pairing table (``trial`` + one
``{stream}_{device}`` column per source), and :func:`pair_media` writes that
table into ``.ethograph/alignment.nwb`` — or, given a neuroconv-written NWB
that already computed trial timing (Route 2/3), adds the streams to it in
place so it becomes pair-able too.

Qt-free: this module is imported by headless notebooks, not just the GUI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import natsort
import pandas as pd

from ethograph.io.nwb_alignment import (
    _add_external_series,
    _coerce_trial_id,
    _infer_times_from_media,
    edit_nwb,
    sync_acquisition_for_streams,
)

if TYPE_CHECKING:
    from pynwb import NWBFile

#: Default device name per stream, used when a source names no device and
#: (for a pattern) captures no camera/mic group of its own.
_DEFAULT_DEVICE: dict[str, str] = {"video": "cam-1", "pose": "cam-1", "audio": "mic-1"}

#: The named regex group a pattern uses for the device axis, per stream.
_DEVICE_GROUP: dict[str, str] = {"video": "camera", "pose": "camera", "audio": "mic"}


@dataclass(frozen=True)
class SourceSpec:
    """One media source of a rig: where its files are and how they map to trials/devices."""

    stream: str  # "video" | "pose" | "audio"
    folder: str | None = None  # relative to session_dir, or absolute
    files: tuple[str, ...] = ()  # explicit list; wins over folder
    pattern: str | None = None  # regex over the file *stem*: named groups trial (required), camera|mic (optional)
    device: str | None = None  # device name when pattern has no camera/mic group (default "cam-1" / "mic-1")
    software: str | None = None  # pose only, e.g. "DeepLabCut"


def _stream_extensions(stream: str) -> frozenset[str]:
    from ethograph.io.validation import AUDIO_EXTENSIONS, POSE_EXTENSIONS, VIDEO_EXTENSIONS

    by_stream = {"video": VIDEO_EXTENSIONS, "pose": POSE_EXTENSIONS, "audio": AUDIO_EXTENSIONS}
    if stream not in by_stream:
        raise ValueError(f"Unknown stream {stream!r}; expected 'video', 'pose', or 'audio'")
    return frozenset(by_stream[stream])


def _resolve_path(value: str, session_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else session_dir / path


def _source_files(session_dir: Path, source: SourceSpec) -> list[Path]:
    """Every file *source* names, natsorted; extension-filtered when scanned from a folder."""
    if source.files:
        return list(natsort.natsorted(_resolve_path(f, session_dir) for f in source.files))
    if not source.folder:
        raise ValueError(f"SourceSpec for stream {source.stream!r} needs a 'folder' or 'files'")
    folder = _resolve_path(source.folder, session_dir)
    if not folder.is_dir():
        raise ValueError(f"{source.stream} folder does not exist: {folder}")
    exts = _stream_extensions(source.stream)
    files = natsort.natsorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts)
    if not files:
        raise ValueError(f"No {source.stream} files found in {folder} (looked for {sorted(exts)})")
    return files


def _pattern_frame(source: SourceSpec, files: list[Path]) -> pd.DataFrame:
    """One row per file, pivoted to ``trial`` + ``{stream}_{device}`` columns."""
    rx = re.compile(source.pattern)  # type: ignore[arg-type]
    if "trial" not in rx.groupindex:
        raise ValueError(f"Pattern for stream {source.stream!r} has no named 'trial' group: {source.pattern!r}")
    device_group = _DEVICE_GROUP[source.stream]

    rows: list[dict[str, str]] = []
    for f in files:
        m = rx.search(f.stem)
        if not m:
            raise ValueError(f"Pattern {source.pattern!r} does not match file: {f}")
        gd = m.groupdict()
        device = gd.get(device_group) or source.device or _DEFAULT_DEVICE[source.stream]
        rows.append({"trial": gd["trial"], "device": device, "file": f.name})

    df = pd.DataFrame(rows)
    if df["trial"].str.isdigit().all():
        df["trial"] = df["trial"].astype(int)
    piv = df.pivot(index="trial", columns="device", values="file")
    piv.columns = [f"{source.stream}_{d}" for d in piv.columns]
    return piv.reset_index()


def discover_media(session_dir: str | Path, sources: Sequence[SourceSpec]) -> pd.DataFrame:
    """The pairing table: ``trial`` + one ``{stream}_{device}`` column per source/device, basenames only.

    - pattern ``None``: files natsorted; row i is trial i+1; device from ``device`` or the default.
    - pattern set: every file must match; named groups build trial + device columns.
    - Extensions: only those in :mod:`ethograph.io.validation` ``VIDEO_``/``AUDIO_``/``POSE_EXTENSIONS``
      for the source's stream.
    - Fail fast: two pattern-less sources with different file counts -> ``ValueError`` naming both counts;
      a pattern that fails to match a file -> ``ValueError`` naming the file; an empty folder -> ``ValueError``.
    - A trial present in one source but not another leaves ``""`` in the missing cell (like the current
      template).
    """
    session_dir = Path(session_dir)
    if not sources:
        raise ValueError("No sources given")

    frames: list[pd.DataFrame] = []
    patternless_counts: dict[str, int] = {}

    for source in sources:
        files = _source_files(session_dir, source)

        if source.pattern is None:
            device = source.device or _DEFAULT_DEVICE[source.stream]
            col = f"{source.stream}_{device}"
            patternless_counts[col] = len(files)
            frames.append(pd.DataFrame({"trial": range(1, len(files) + 1), col: [f.name for f in files]}))
            continue

        frames.append(_pattern_frame(source, files))

    if len(set(patternless_counts.values())) > 1:
        detail = ", ".join(f"{col}={n}" for col, n in patternless_counts.items())
        raise ValueError(f"pattern-less sources disagree on file count: {detail}")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="trial", how="outer")

    # Natural sort: numeric when every trial id is a digit string, else alphabetical.
    trials = merged["trial"]
    if trials.apply(lambda v: str(v).isdigit()).all():
        merged = merged.assign(_sort=trials.astype(int))
    else:
        merged = merged.assign(_sort=trials.astype(str).str.lower())
    merged = merged.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    return merged.fillna("")


def _existing_nwb_has_trials(path: Path) -> bool:
    """Whether *path* is readable as an NWB file with a populated trials table."""
    import gc

    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        has_trials = nwbfile.trials is not None
    del nwbfile
    gc.collect()
    return has_trials


def _pair_into_existing(
    path: Path,
    trial_table: pd.DataFrame,
    *,
    stream_rates: dict[str, float] | None,
    session_wide: dict[str, tuple[str, float, float]] | None,
) -> NWBFile:
    """Add *trial_table*'s media columns and *session_wide* streams to the NWB at *path*, in place.

    Never rebuilds the file: existing acquisition items (a neuroconv ``ExternalVideoInterface``
    series, say) are left untouched, and every per-trial column is added new.
    """
    reserved = {"trial", "start_time", "stop_time"}
    media_cols = [c for c in trial_table.columns if c not in reserved]

    result: NWBFile
    with edit_nwb(path) as nwbfile:
        table = nwbfile.trials
        if table is None:
            raise ValueError(f"{path.name} has no trials table to pair media into")
        if len(trial_table) != len(table):
            raise ValueError(f"trial_table has {len(trial_table)} rows but {path.name} has {len(table)} trials")

        for col in media_cols:
            values = [str(v) if pd.notna(v) else "" for v in trial_table[col]]
            table.add_column(name=col, description=f"{col} filename", data=values)

        if stream_rates:
            sync_acquisition_for_streams(nwbfile, stream_rates)

        if session_wide:
            stops = [float(s) for s in nwbfile.trials["stop_time"][:]]
            last_stop = max(stops) if stops else None
            for name, (file, rate, starting_time) in session_wide.items():
                _create_device_for(nwbfile, name)
                n_samples = 1
                if rate and last_stop is not None:
                    n_samples = max(1, int((last_stop - starting_time) * rate))
                _add_external_series(nwbfile, name, [file], [starting_time], [n_samples], rate)

        result = nwbfile

    return result


def _create_device_for(nwbfile: NWBFile, stream_name: str) -> None:
    """Ensure a Device exists for the ``{stream}_{device}`` acquisition name."""
    parts = stream_name.split("_", 1)
    device_name = parts[1] if len(parts) > 1 else parts[0]
    if device_name not in [d.name for d in nwbfile.devices.values()]:
        nwbfile.create_device(name=device_name, description=f"Device {device_name}")


def _pair_into_new(
    trial_table: pd.DataFrame,
    *,
    stream_rates: dict[str, float] | None,
    session_wide: dict[str, tuple[str, float, float]] | None,
    output_path: Path | None,
    media_root: str | Path | None,
    pose_fps: float | None,
) -> NWBFile:
    """Build a fresh alignment NWB from *trial_table* (the ``align_media_per_trial`` shape),
    plus one ImageSeries per *session_wide* stream."""
    from datetime import datetime
    from uuid import uuid4

    import pynwb
    from dateutil.tz import tzlocal
    from pynwb import NWBHDF5IO

    nwbfile = pynwb.NWBFile(
        session_description="NWB file for media alignment (ethograph generated).",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )

    reserved = {"trial", "start_time", "stop_time"}
    media_cols = [c for c in trial_table.columns if c not in reserved]
    video_cols = [c for c in media_cols if c.startswith("video_")]
    audio_cols = [c for c in media_cols if c.startswith("audio_")]
    pose_cols = [c for c in media_cols if c.startswith("pose_")]
    has_times = {"start_time", "stop_time"}.issubset(trial_table.columns)

    table = (
        trial_table
        if has_times
        else _infer_times_from_media(
            trial_table,
            video_cols,
            audio_cols,
            Path(media_root) if media_root else None,
            pose_cols=pose_cols,
            pose_fps=pose_fps,
        )
    )

    if "trial" in table.columns:
        nwbfile.add_trial_column(name="trial", description="Trial number")
    for col in media_cols:
        nwbfile.add_trial_column(name=col, description=f"{col} filename")

    for _, row in table.iterrows():
        trial_row: dict = {
            "start_time": float(row["start_time"]),
            "stop_time": float(row["stop_time"]),
        }
        if "trial" in table.columns:
            trial_row["trial"] = _coerce_trial_id(row["trial"])
        for col in media_cols:
            trial_row[col] = str(row[col]) if pd.notna(row[col]) else ""
        nwbfile.add_trial(**trial_row)

    if stream_rates:
        sync_acquisition_for_streams(nwbfile, stream_rates)

    if session_wide:
        stops = table["stop_time"].astype(float).tolist()
        last_stop = max(stops) if stops else None
        for name, (file, rate, starting_time) in session_wide.items():
            _create_device_for(nwbfile, name)
            n_samples = 1
            if rate and last_stop is not None:
                n_samples = max(1, int((last_stop - starting_time) * rate))
            _add_external_series(nwbfile, name, [file], [starting_time], [n_samples], rate)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with NWBHDF5IO(str(output_path), "w") as io:
            io.write(nwbfile)

    return nwbfile


def pair_media(
    trial_table: pd.DataFrame,
    stream_rates: dict[str, float] | None = None,
    session_wide: dict[str, tuple[str, float, float]] | None = None,
    output_path: str | Path | None = None,
    media_root: str | Path | None = None,
    pose_fps: float | None = None,
) -> NWBFile:
    """Write ``.ethograph/alignment.nwb`` from a pairing table.

    Per-trial columns are handled as :func:`~ethograph.io.nwb_alignment.align_media_per_trial`
    always did; ``session_wide`` streams each get one ``ImageSeries`` with a ``starting_time``
    (the ``align_media_from_streams`` behaviour for a single session-wide file).
    ``start_time``/``stop_time`` are optional — omitted, they are inferred from the media
    (needs ``media_root``).

    If ``output_path`` is an existing NWB that *already has a trials table* (a neuroconv-written
    source), the streams are added to that file in place through
    :func:`~ethograph.io.nwb_alignment.edit_nwb`, never rebuilt: per-trial columns from
    *trial_table* are written as new trial columns (row count must equal the file's trials, else
    ``ValueError``) and ``session_wide`` streams as acquisition ``ImageSeries``. Existing
    acquisition names are left alone.

    Parameters
    ----------
    trial_table
        ``trial`` + ``{stream}_{device}`` filename columns (as :func:`discover_media` returns).
        ``start_time``/``stop_time`` optional.
    stream_rates
        Sampling rate per stream, e.g. ``{"video": 30.0, "audio": 48000.0}``.
    session_wide
        ``{"{stream}_{device}": (file, rate_hz, starting_time_s)}`` for streams that are one
        file spanning the whole session rather than one file per trial.
    output_path
        Where to write (or, for an existing NWB with a trials table, extend) the ``.nwb`` file.
    media_root
        Folder the filename columns are relative to; only needed when times are inferred.
    pose_fps
        Frame rate for probing pose files, when inferring times from a table whose only media
        columns are ``pose_*``.

    Returns
    -------
    The in-memory :class:`~pynwb.NWBFile`.
    """
    out = Path(output_path) if output_path is not None else None
    if out is not None and out.exists() and _existing_nwb_has_trials(out):
        return _pair_into_existing(out, trial_table, stream_rates=stream_rates, session_wide=session_wide)
    return _pair_into_new(
        trial_table,
        stream_rates=stream_rates,
        session_wide=session_wide,
        output_path=out,
        media_root=media_root,
        pose_fps=pose_fps,
    )
