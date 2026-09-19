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

import logging
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
    write_individuals,
)

if TYPE_CHECKING:
    from pynwb import NWBFile

logger = logging.getLogger(__name__)

#: Default device name per stream, used when a source names no device and
#: (for a pattern) captures no camera/mic group of its own.
_DEFAULT_DEVICE: dict[str, str] = {"video": "cam-1", "pose": "cam-1", "audio": "mic-1"}

#: The named regex group a pattern uses for the device axis, per stream.
_DEVICE_GROUP: dict[str, str] = {"video": "camera", "pose": "camera", "audio": "mic"}


@dataclass(frozen=True)
class SourceSpec:
    """One media source of a rig: where its files are and how they map to trials/devices."""

    stream: str  # "video" | "pose" | "audio"
    folder: str | None = None  # absolute path of the folder holding the files
    files: tuple[str, ...] = ()  # explicit absolute paths; wins over folder
    pattern: str | None = None  # regex the whole file *stem* must match; groups: trial (required), camera|mic
    device: str | None = None  # device name when pattern has no camera/mic group (default "cam-1" / "mic-1")
    software: str | None = None  # pose only, e.g. "DeepLabCut"
    extension: str | None = None  # keep only files with this suffix (e.g. ".h5"); None = every extension of the stream


def _basename(value: object) -> str:
    """The filename the NWB trials table stores for a cell: a full path's name, ``""`` when empty."""
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return ""
    return Path(str(value)).name


def _stream_extensions(stream: str) -> frozenset[str]:
    from ethograph.io.validation import AUDIO_EXTENSIONS, POSE_EXTENSIONS, VIDEO_EXTENSIONS

    by_stream = {"video": VIDEO_EXTENSIONS, "pose": POSE_EXTENSIONS, "audio": AUDIO_EXTENSIONS}
    if stream not in by_stream:
        raise ValueError(f"Unknown stream {stream!r}; expected 'video', 'pose', or 'audio'")
    return frozenset(by_stream[stream])


def _source_files(source: SourceSpec) -> list[Path]:
    """Every file *source* names, natsorted; extension-filtered when scanned from a folder."""
    if source.files:
        return list(natsort.natsorted(Path(f) for f in source.files))
    if not source.folder:
        raise ValueError(f"SourceSpec for stream {source.stream!r} needs a 'folder' or 'files'")
    folder = Path(source.folder)
    if not folder.is_dir():
        raise ValueError(f"{source.stream} folder does not exist: {folder}")
    exts = _stream_extensions(source.stream)
    if source.extension is not None:
        ext = source.extension.lower()
        if ext not in exts:
            raise ValueError(
                f"Extension {source.extension!r} is not a {source.stream} extension (one of {sorted(exts)})"
            )
        exts = frozenset({ext})
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
        m = rx.fullmatch(f.stem)
        if not m:
            logger.info("Skipping %s: does not match pattern %r", f.name, source.pattern)
            continue
        gd = m.groupdict()
        device = gd.get(device_group) or source.device or _DEFAULT_DEVICE[source.stream]
        rows.append({"trial": gd["trial"], "device": device, "file": str(f)})

    if not rows:
        raise ValueError(f"Pattern {source.pattern!r} matches none of the {len(files)} {source.stream} files")
    df = pd.DataFrame(rows)
    if df["trial"].str.isdigit().all():
        df["trial"] = df["trial"].astype(int)
    dup = df[df.duplicated(["trial", "device"], keep=False)]
    if not dup.empty:
        first = dup.iloc[0]
        names = ", ".join(
            Path(f).name for f in dup[(dup["trial"] == first["trial"]) & (dup["device"] == first["device"])]["file"]
        )
        raise ValueError(
            f"Pattern {source.pattern!r} maps several {source.stream} files to trial {first['trial']} / "
            f"{first['device']}: {names}. Narrow the pattern or pick one extension."
        )
    piv = df.pivot(index="trial", columns="device", values="file")
    piv.columns = [f"{source.stream}_{d}" for d in piv.columns]
    return piv.reset_index()


def discover_media(sources: Sequence[SourceSpec]) -> pd.DataFrame:
    """The pairing table: ``trial`` + one ``{stream}_{device}`` column per source/device, full paths.

    Each cell is the file's full path, so the table says where every file is and
    :func:`pair_media` needs no root; the NWB trials table receives the basenames.

    - pattern ``None``: files natsorted; row i is trial i+1; device from ``device`` or the default.
    - pattern set: named groups build trial + device columns; a file the pattern does not match
      is skipped (logged at INFO).
    - Extensions: only those in :mod:`ethograph.io.validation` ``VIDEO_``/``AUDIO_``/``POSE_EXTENSIONS``
      for the source's stream; ``extension`` narrows a scanned folder to one of them (a DLC folder
      holds ``.h5`` and ``.csv`` twins of every file).
    - Fail fast: two pattern-less sources with different file counts -> ``ValueError`` naming both counts;
      a pattern that matches no file at all -> ``ValueError``; two files on one (trial, device) ->
      ``ValueError`` naming them; an empty folder -> ``ValueError``.
    - A trial present in one source but not another leaves ``""`` in the missing cell (like the current
      template).
    """
    if not sources:
        raise ValueError("No sources given")

    frames: list[pd.DataFrame] = []
    patternless_counts: dict[str, int] = {}

    for source in sources:
        files = _source_files(source)

        if source.pattern is None:
            device = source.device or _DEFAULT_DEVICE[source.stream]
            col = f"{source.stream}_{device}"
            patternless_counts[col] = len(files)
            frames.append(pd.DataFrame({"trial": range(1, len(files) + 1), col: [str(f) for f in files]}))
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
    individuals: Sequence[str] | None = None,
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

        already = [c for c in media_cols if c in table.colnames]
        if already:
            raise ValueError(
                f"{path.name} already pairs {', '.join(already)}. Pairing adds columns, so the same "
                "media cannot be paired in twice; pass on_existing='replace' to rebuild the alignment "
                "from this table instead."
            )
        for col in media_cols:
            values = [_basename(v) for v in trial_table[col]]
            table.add_column(name=col, description=f"{col} filename", data=values)

        if stream_rates:
            sync_acquisition_for_streams(nwbfile, stream_rates, media_paths=trial_table)
        if individuals is not None:
            write_individuals(nwbfile, individuals)

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
    pose_fps: float | None,
    individuals: Sequence[str] | None = None,
) -> NWBFile:
    """Build a fresh alignment NWB from *trial_table*, plus one ImageSeries per *session_wide* stream."""
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
            trial_row[col] = _basename(row[col])
        nwbfile.add_trial(**trial_row)

    if stream_rates:
        sync_acquisition_for_streams(nwbfile, stream_rates, media_paths=table)
    if individuals is not None:
        write_individuals(nwbfile, individuals)

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
    pose_fps: float | None = None,
    individuals: Sequence[str] | None = None,
    on_existing: str = "extend",
) -> NWBFile:
    """Write ``.ethograph/alignment.nwb`` from a pairing table.

        Each per-trial ``{stream}_{device}`` column becomes a trials-table column plus one
        ``ImageSeries`` whose segments start at the trial starts; ``session_wide`` streams each get
        one ``ImageSeries`` with a ``starting_time``.
        ``start_time``/``stop_time`` are optional — omitted, they are inferred from the media
        files the table names.

    **What happens to a file that is already there is the caller's decision, never the
    filesystem's** (``on_existing``). Annotating somebody's NWB and building a session's
    own alignment are different jobs, and a function that guessed between them by
    looking at the disk was eventually called from the path that wanted the other one.

    ``on_existing="extend"`` (the default) adds to an existing NWB that *already has a
    trials table* — a neuroconv-written source — in place through
    :func:`~ethograph.io.nwb_alignment.edit_nwb`, never rebuilding it: per-trial columns
    from *trial_table* become new trial columns (row count must equal the file's trials,
    else ``ValueError``) and ``session_wide`` streams acquisition ``ImageSeries``;
    existing acquisition names are left alone. ``on_existing="replace"`` deletes the file
    and writes it afresh — what a *drop* or the wizard does, because an alignment built
    from dropped files is derived data and rebuilding it must be idempotent.

        Parameters
        ----------
        trial_table
            ``trial`` + ``{stream}_{device}`` columns holding each file's full path (as
            :func:`discover_media` returns); a bare filename is accepted when times are given.
            The NWB trials table receives the basenames. ``start_time``/``stop_time`` optional.
        stream_rates
            Sampling rate per stream, e.g. ``{"video": 30.0, "audio": 48000.0}``; a
            ``{stream}_{device}`` key (``"video_cam-2": 60.0``) overrides its stream's rate.
        session_wide
            ``{"{stream}_{device}": (file, rate_hz, starting_time_s)}`` for streams that are one
            file spanning the whole session rather than one file per trial.
        output_path
            Where to write the ``.nwb`` file.
        on_existing
            What an existing *output_path* means: ``"extend"`` pairs into its trials table,
            ``"replace"`` rebuilds it from scratch. Anything building a session's own
            alignment passes ``"replace"``, so running it twice is the same as once.
        pose_fps
            Frame rate for probing pose files, when inferring times from a table whose only media
            columns are ``pose_*``.
        individuals
            The individuals this session labels. Recorded in the session record, which is their
            one home; a dataset's individual dim is checked against it on load.

        Returns
        -------
        The in-memory :class:`~pynwb.NWBFile`.
    """
    if on_existing not in ("extend", "replace"):
        raise ValueError(f"on_existing must be 'extend' or 'replace', got {on_existing!r}")
    out = Path(output_path) if output_path is not None else None
    if out is not None and out.exists() and on_existing == "replace":
        out.unlink()  # derived data: rebuilt whole, so pairing twice is pairing once
    if out is not None and out.exists() and _existing_nwb_has_trials(out):
        return _pair_into_existing(
            out, trial_table, stream_rates=stream_rates, session_wide=session_wide, individuals=individuals
        )
    return _pair_into_new(
        trial_table,
        stream_rates=stream_rates,
        session_wide=session_wide,
        output_path=out,
        pose_fps=pose_fps,
        individuals=individuals,
    )
