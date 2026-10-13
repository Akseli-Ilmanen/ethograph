"""BORIS project (.boris) import: observations -> ethograph sessions.

A BORIS observation binds one or more media files (concatenated in Player 1)
to a list of events coded in observation-global time. Ethograph has no
"observation": a session is a folder with an alignment NWB and a ``labels.tsv``,
and a label's time is relative to its trial. So one observation becomes one
session folder and each media file in Player 1 becomes one trial (1-indexed);
events are split across file boundaries into trial-local intervals.

Qt-free and scripted, so a whole archive migrates in a loop::

    import ethograph as eto

    results = eto.labels.boris.import_boris_project(
        "study.boris",
        "D:/study",  # the project folder: mapping.txt + one session per observation
        video_folder="D:/videos",  # where the media really is (BORIS stores old absolute paths)
    )

The GUI's **More tools → Import BORIS projects…** drives the same function.
Media never moves: the session only records where each file is.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ethograph.io.pairing import pair_media
from ethograph.io.session_layout import alignment_path, labels_path, settings_dir
from ethograph.labels.converters import extend_mapping
from ethograph.labels.intervals import (
    EVENT_TYPE_POINT,
    EVENT_TYPE_STATE,
    ensure_confidence,
    ensure_individual_rec,
    ensure_labeling_method,
    load_label_mapping,
    save_label_mapping,
)
from ethograph.labels.tsv_store import (
    TRIAL_META_DEFAULTS,
    TSV_COLUMNS,
    init_empty_labels,
    save_labels_tsv,
)

logger = logging.getLogger(__name__)

MAPPING_FILENAME = "mapping.txt"

#: BORIS's event kinds, as spelled in ``behaviors_conf[...]["type"]``.
_BORIS_TYPE_TO_EVENT_TYPE = {
    "Point event": EVENT_TYPE_POINT,
    "State event": EVENT_TYPE_STATE,
}

#: BORIS's video file extensions (``media_info.hasVideo`` is the authority when present).
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v", ".webm", ".flv"}

#: A label with no BORIS subject: BORIS allows "No focal subject" coding.
UNNAMED_SUBJECT = "individual_0"


# ---------------------------------------------------------------------------
# Reading the project file
# ---------------------------------------------------------------------------


def load_boris_project(path: str | Path) -> dict:
    """Parse a .boris project file (JSON, optionally gzipped)."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def list_media_observations(project: dict) -> list[str]:
    """Observation keys for MEDIA-type observations only (LIVE has no media to align to)."""
    obs = project.get("observations", {})
    return [k for k, v in obs.items() if v.get("type") == "MEDIA"]


def observation_media_files(observation: dict) -> list[str]:
    """Files in Player 1 of an observation, in concatenation order."""
    file_field = observation.get("file", {})
    if isinstance(file_field, dict):
        return list(file_field.get("1", []))
    return list(file_field) if isinstance(file_field, list) else []


def unique_behavior_codes(project: dict) -> list[str]:
    """Sorted unique behavior codes declared in ``behaviors_conf``."""
    beh = project.get("behaviors_conf", {}).values()
    return sorted({b.get("code", "") for b in beh if b.get("code")})


def behavior_event_types(project: dict) -> dict[str, str]:
    """``{behavior_code: "state" | "point"}`` from ``behaviors_conf``.

    A behaviour whose BORIS type is missing or unrecognised is a state.
    """
    out: dict[str, str] = {}
    for entry in project.get("behaviors_conf", {}).values():
        code = entry.get("code")
        if not code:
            continue
        out[code] = _BORIS_TYPE_TO_EVENT_TYPE.get(entry.get("type"), EVENT_TYPE_STATE)
    return out


def subject_names(project: dict) -> list[str]:
    """Subjects declared in ``subjects_conf``, in BORIS's order."""
    return [s["name"] for s in project.get("subjects_conf", {}).values() if s.get("name")]


# ---------------------------------------------------------------------------
# Observation time -> trial-local time
# ---------------------------------------------------------------------------


def _media_durations(observation: dict) -> list[float]:
    media_files = observation_media_files(observation)
    lengths = observation.get("media_info", {}).get("length", {})
    missing = [f for f in media_files if f not in lengths]
    if missing:
        raise ValueError(
            f"BORIS media_info.length has no duration for {missing}; events cannot be placed in trial-local time"
        )
    return [float(lengths[f]) for f in media_files]


def _cumulative_offsets(durations: list[float]) -> list[float]:
    offsets: list[float] = []
    running = 0.0
    for dur in durations:
        offsets.append(running)
        running += dur
    return offsets


def _match_event_to_file(t_obs: float, offsets: list[float], durations: list[float]) -> tuple[int, float] | None:
    for i, (t0, dur) in enumerate(zip(offsets, durations)):
        if t0 <= t_obs <= t0 + dur + 1e-6:
            return i, t_obs - t0
    return None


def build_trial_table(observation: dict) -> pd.DataFrame:
    """Per-file trial table: one trial per media file in Player 1.

    Columns: ``trial``, ``start_time``, ``stop_time``, ``video_cam-1`` (the stored path).
    ``start_time`` / ``stop_time`` are cumulative offsets in observation time.
    """
    media_files = observation_media_files(observation)
    if not media_files:
        raise ValueError("Observation has no media files in Player 1")
    durations = _media_durations(observation)
    offsets = _cumulative_offsets(durations)
    return pd.DataFrame(
        {
            "trial": range(1, len(media_files) + 1),
            "start_time": offsets,
            "stop_time": [t0 + d for t0, d in zip(offsets, durations)],
            "video_cam-1": media_files,
        }
    )


def extract_intervals(observation: dict, name_to_id: dict[str, int], event_types: dict[str, str]) -> pd.DataFrame:
    """Convert an observation's events into a labels DataFrame.

    State events use BORIS's toggle convention (odd occurrence = START,
    even = STOP per ``(subject, behavior_code)`` key); a state still open
    at the end is closed at its trial's end, one spanning a file boundary is
    clipped at the boundary. Point events are rows with ``offset_s = NaN``.
    Events whose behaviour is not in ``name_to_id`` are skipped.
    """
    media_files = observation_media_files(observation)
    if not media_files:
        return init_empty_labels([])

    durations = _media_durations(observation)
    offsets = _cumulative_offsets(durations)

    open_starts: dict[tuple[str, str], tuple[int, float]] = {}
    rows: list[dict] = []

    def add_row(onset: float, offset: float, behavior: str, subject: str, file_idx: int, event_type: str) -> None:
        rows.append(
            {
                "onset_s": onset,
                "offset_s": offset,
                "labels": name_to_id[behavior],
                "individual": subject,
                "trial": file_idx + 1,
                "event_type": event_type,
            }
        )

    for ev in observation.get("events", []):
        if len(ev) < 3:
            continue
        t_obs = float(ev[0])
        subject = str(ev[1]) if ev[1] else UNNAMED_SUBJECT
        behavior = str(ev[2])
        if behavior not in name_to_id:
            logger.warning("Behaviour %r is not declared in behaviors_conf — skipping its events", behavior)
            continue

        match = _match_event_to_file(t_obs, offsets, durations)
        if match is None:
            logger.warning("Event t=%.3f lies outside every media file — skipping", t_obs)
            continue
        file_idx, t_local = match

        if event_types.get(behavior, EVENT_TYPE_STATE) == EVENT_TYPE_POINT:
            add_row(t_local, float("nan"), behavior, subject, file_idx, EVENT_TYPE_POINT)
            continue

        key = (subject, behavior)
        if key not in open_starts:
            open_starts[key] = (file_idx, t_local)
            continue
        start_file, start_local = open_starts.pop(key)
        end_local = t_local if file_idx == start_file else durations[start_file]
        add_row(start_local, end_local, behavior, subject, start_file, EVENT_TYPE_STATE)
        if file_idx != start_file:
            logger.warning(
                "State %r spans a file boundary (trial %d -> %d); clipped at the end of trial %d",
                behavior,
                start_file + 1,
                file_idx + 1,
                start_file + 1,
            )

    for (subject, behavior), (start_file, start_local) in open_starts.items():
        add_row(start_local, durations[start_file], behavior, subject, start_file, EVENT_TYPE_STATE)
        logger.warning("Unclosed state %r by %r — closed at the end of trial %d", behavior, subject, start_file + 1)

    if not rows:
        return init_empty_labels([])

    df = pd.DataFrame(rows).sort_values(["trial", "onset_s"]).reset_index(drop=True)
    ensure_individual_rec(df)
    ensure_confidence(df)
    ensure_labeling_method(df)
    for col, default in TRIAL_META_DEFAULTS.items():
        df[col] = default
    return df[TSV_COLUMNS]


# ---------------------------------------------------------------------------
# Media on disk
# ---------------------------------------------------------------------------


def resolve_media_paths(observation: dict, search_dirs: list[Path]) -> list[str]:
    """Absolute paths for the observation's media, by basename when the stored path is stale.

    Tries the stored path first, then each of ``search_dirs`` joined with the
    basename. A file found nowhere keeps its stored string; the caller reports it.
    """
    resolved: list[str] = []
    for stored in observation_media_files(observation):
        p = Path(stored)
        if p.is_absolute() and p.is_file():
            resolved.append(str(p))
            continue
        for d in search_dirs:
            candidate = d / p.name
            if candidate.is_file():
                resolved.append(str(candidate))
                break
        else:
            resolved.append(stored)
    return resolved


def match_pose_files(
    pose_folder: Path,
    video_paths: list[str],
    extensions: tuple[str, ...] = (".csv", ".h5", ".hdf5", ".slp", ".parquet"),
) -> list[str]:
    """For each video, the pose file in ``pose_folder`` whose stem starts with the video's stem.

    Tracking tools append a model suffix to the video's name (``cam1DLC.csv``), so
    the shortest matching stem is the closest: ``cam1`` takes ``cam1DLC``, never
    ``cam10DLC``. Unmatched videos get an empty string; an empty list means
    nothing matched.
    """
    candidates: list[Path] = []
    for ext in extensions:
        candidates.extend(pose_folder.glob(f"*{ext}"))

    matched = []
    for v in video_paths:
        v_stem = Path(v).stem
        hits = sorted((p for p in candidates if p.stem.startswith(v_stem)), key=lambda p: (len(p.stem), p.name))
        matched.append(str(hits[0]) if hits else "")
    return matched if any(matched) else []


def _media_stream(observation: dict, media_files: list[str]) -> str:
    """``"video"`` or ``"audio"``: what Player 1 holds, from BORIS's own probe."""
    has_video = observation.get("media_info", {}).get("hasVideo", {})
    flags = [bool(has_video[f]) if f in has_video else Path(f).suffix.lower() in _VIDEO_EXTENSIONS for f in media_files]
    if all(flags):
        return "video"
    if not any(flags):
        return "audio"
    raise ValueError("Player 1 mixes video and audio-only files; split them into separate observations")


def _single_fps(observation: dict, media_files: list[str]) -> float:
    fps_info = observation.get("media_info", {}).get("fps", {})
    rates = {float(fps_info[f]) for f in media_files if fps_info.get(f)}
    if not rates:
        raise ValueError("BORIS media_info.fps is empty; the alignment needs the media's frame rate")
    if len(rates) > 1:
        raise ValueError(
            f"Media files run at different rates ({sorted(rates)}); one observation is one session, "
            "so split them into one observation per rate"
        )
    return next(iter(rates))


# ---------------------------------------------------------------------------
# Writing sessions
# ---------------------------------------------------------------------------


def session_folder_name(observation_key: str) -> str:
    """A folder name from an observation id: path separators and unsafe characters become ``_``."""
    name = re.sub(r"[^\w.\- ]+", "_", observation_key, flags=re.UNICODE).strip(" ._")
    return name or "observation"


def write_mapping(mapping_path: Path, project: dict) -> dict[str, int]:
    """Add the project's behaviours to ``mapping_path`` (created if missing), keeping existing ids.

    A behaviour new to the mapping takes its BORIS kind (state/point); one already
    there keeps whatever the mapping says, so a second import never flips a class.
    """
    names = unique_behavior_codes(project)
    if not names:
        raise ValueError("BORIS project declares no behaviours")
    name_to_id, added = extend_mapping(names, mapping_path)
    event_types = behavior_event_types(project)
    point_added = [n for n in added if event_types.get(n) == EVENT_TYPE_POINT]
    if point_added:
        mappings = load_label_mapping(mapping_path)
        for name in point_added:
            mappings[name_to_id[name]]["event_type"] = EVENT_TYPE_POINT
        save_label_mapping(mapping_path, mappings)
    return name_to_id


@dataclass
class BorisSessionResult:
    """What one observation became."""

    observation: str
    session_dir: Path
    alignment: Path
    labels: Path
    n_trials: int
    n_labels: int
    #: The media path recorded per trial, resolved where it could be.
    media_paths: list[str] = field(default_factory=list)
    #: Media files found nowhere on disk (the session records the stale path; the GUI will ask).
    missing_media: list[str] = field(default_factory=list)


def import_boris_observation(
    project: dict,
    observation_key: str,
    session_dir: str | Path,
    mapping_path: str | Path,
    *,
    search_dirs: list[Path] | None = None,
    pose_folder: str | Path | None = None,
) -> BorisSessionResult:
    """Write one observation as one session folder.

    ``session_dir/.ethograph/alignment.nwb`` carries the trials (one per media
    file) and their media paths; ``session_dir/labels.tsv`` the events. The
    vocabulary goes to ``mapping_path`` (see :func:`write_mapping`).
    """
    observation = project["observations"][observation_key]
    if observation.get("type") != "MEDIA":
        raise ValueError(
            f"Observation {observation_key!r} is {observation.get('type')!r}; only MEDIA observations carry media"
        )
    # The layout helpers resolve a path that is not a folder yet to its parent.
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = Path(mapping_path)

    name_to_id = write_mapping(mapping_path, project)
    event_types = behavior_event_types(project)
    trials_df = build_trial_table(observation)
    media_files = observation_media_files(observation)
    stream = _media_stream(observation, media_files)
    rate = _single_fps(observation, media_files) if stream == "video" else None

    media_paths = resolve_media_paths(observation, list(search_dirs or []))
    missing = [p for p in media_paths if not Path(p).is_file()]
    for p in missing:
        logger.warning("Media file not found: %s (searched %s)", p, [str(d) for d in search_dirs or []])

    pairing = trials_df[["trial", "start_time", "stop_time"]].assign(**{f"{stream}_cam-1": media_paths})
    stream_rates = {"video": rate} if rate is not None else None

    if pose_folder is not None:
        pose_paths = match_pose_files(Path(pose_folder), media_paths)
        if pose_paths:
            pairing["pose_cam-1"] = pose_paths
        else:
            logger.warning("No pose file in %s matches a video of %r", pose_folder, observation_key)

    settings_dir(session_dir).mkdir(parents=True, exist_ok=True)
    nwb_out = alignment_path(session_dir)
    subjects = subject_names(project)
    pair_media(
        pairing,
        stream_rates=stream_rates,
        output_path=nwb_out,
        individuals=subjects or None,
        on_existing="replace",
    )

    labels_df = extract_intervals(observation, name_to_id, event_types)
    labels_out = labels_path(session_dir)
    save_labels_tsv(labels_out, labels_df)

    logger.info("BORIS %r -> %s (%d trials, %d labels)", observation_key, session_dir, len(trials_df), len(labels_df))
    return BorisSessionResult(
        observation=observation_key,
        session_dir=session_dir,
        alignment=nwb_out,
        labels=labels_out,
        n_trials=len(trials_df),
        n_labels=len(labels_df),
        media_paths=media_paths,
        missing_media=missing,
    )


def import_boris_project(
    boris_path: str | Path,
    output_dir: str | Path,
    *,
    observations: list[str] | None = None,
    video_folder: str | Path | None = None,
    pose_folder: str | Path | None = None,
) -> list[BorisSessionResult]:
    """Convert a ``.boris`` project: every MEDIA observation becomes a session under ``output_dir``.

    ``output_dir`` is the ethograph project folder: it receives ``mapping.txt``
    and one session folder per observation, named after the observation. Several
    ``.boris`` files imported into the same folder share the mapping, so a
    behaviour keeps its id across them.

    ``video_folder`` is searched by basename when BORIS's stored media paths are
    stale (an archive moved, a different machine); ``pose_folder`` attaches a
    tracking file per video by filename stem.
    """
    boris_path = Path(boris_path)
    output_dir = Path(output_dir)
    project = load_boris_project(boris_path)
    available = list_media_observations(project)
    if not available:
        raise ValueError(f"{boris_path.name} has no MEDIA observations (LIVE observations have nothing to align to)")
    keys = observations if observations is not None else available
    unknown = [k for k in keys if k not in available]
    if unknown:
        raise KeyError(f"Not MEDIA observations of {boris_path.name}: {unknown}; available: {available}")

    search_dirs = [Path(video_folder)] if video_folder else []
    search_dirs += [boris_path.parent, output_dir]
    mapping_path = output_dir / MAPPING_FILENAME

    results: list[BorisSessionResult] = []
    for key in keys:
        results.append(
            import_boris_observation(
                project,
                key,
                output_dir / session_folder_name(key),
                mapping_path,
                search_dirs=search_dirs,
                pose_folder=pose_folder,
            )
        )
    return results
