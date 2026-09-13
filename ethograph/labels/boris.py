"""BORIS project (.boris) import: events -> ethograph labels, media -> alignment.

A BORIS observation binds one or more media files (concatenated in Player 1)
to a list of events coded in observation-global time. This module:

- parses the JSON project file,
- splits events across media boundaries into trial-local intervals,
- builds a per-file trial table for ``pair_media``.

Each media file becomes one ethograph trial (1-indexed).
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import pandas as pd

from ethograph.labels.converters import (
    LabelConverter,
    build_mapping_from_labels,
)
from ethograph.labels.intervals import EVENT_TYPE_POINT, EVENT_TYPE_STATE
from ethograph.labels.tsv_store import TRIAL_META_DEFAULTS, init_empty_labels

_BORIS_TYPE_TO_EVENT_TYPE = {
    "Point event": EVENT_TYPE_POINT,
    "State event": EVENT_TYPE_STATE,
}

logger = logging.getLogger(__name__)


def load_boris_project(path: str | Path) -> dict:
    """Parse a .boris project file (JSON, optionally gzipped)."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def list_media_observations(project: dict) -> list[str]:
    """Observation keys for MEDIA-type observations only (LIVE is unsupported)."""
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
    """Map ``{behavior_code: "state" | "point"}`` from ``behaviors_conf``.

    Behaviors whose BORIS type is missing or unrecognised default to ``"state"``.
    Use the result to populate the ``event_type`` column in ``mapping.txt``.
    """
    out: dict[str, str] = {}
    for entry in project.get("behaviors_conf", {}).values():
        code = entry.get("code")
        if not code:
            continue
        out[code] = _BORIS_TYPE_TO_EVENT_TYPE.get(entry.get("type"), EVENT_TYPE_STATE)
    return out


def _cumulative_offsets(media_files: list[str], lengths: dict[str, float]) -> list[float]:
    offsets = []
    running = 0.0
    for f in media_files:
        offsets.append(running)
        if f not in lengths:
            raise ValueError(
                f"BORIS media_info.length missing duration for {f!r}; cannot place events in trial-local time"
            )
        running += float(lengths[f])
    return offsets


def _match_event_to_file(
    t_obs: float,
    offsets: list[float],
    durations: list[float],
) -> tuple[int, float] | None:
    for i, (t0, dur) in enumerate(zip(offsets, durations)):
        if t0 <= t_obs <= t0 + dur + 1e-6:
            return i, t_obs - t0
    return None


def build_trial_table(observation: dict) -> pd.DataFrame:
    """Per-file trial table (1 trial per media file in Player 1).

    Columns: ``trial``, ``start_time``, ``stop_time``, ``video_cam-1`` (basename).
    ``start_time`` / ``stop_time`` are cumulative offsets in observation time.
    """
    media_files = observation_media_files(observation)
    lengths = observation.get("media_info", {}).get("length", {})
    if not media_files:
        raise ValueError("Observation has no media files in Player 1")

    rows = []
    running = 0.0
    for i, path in enumerate(media_files):
        if path not in lengths:
            raise ValueError(f"BORIS media_info.length missing duration for {path!r}")
        dur = float(lengths[path])
        rows.append(
            {
                "trial": i + 1,
                "start_time": running,
                "stop_time": running + dur,
                "video_cam-1": Path(path).name,
            }
        )
        running += dur
    return pd.DataFrame(rows)


def extract_intervals(
    observation: dict,
    name_to_id: dict[str, int],
) -> pd.DataFrame:
    """Convert an observation's events into a labels DataFrame.

    State events use BORIS's toggle convention (odd occurrence = START,
    even = STOP per ``(subject, behavior_code)`` key). Point events become
    zero-duration intervals. Events whose behavior is not in ``name_to_id``
    are skipped.
    """
    media_files = observation_media_files(observation)
    if not media_files:
        return init_empty_labels([])

    lengths = observation.get("media_info", {}).get("length", {})
    offsets = _cumulative_offsets(media_files, lengths)
    durations = [float(lengths[f]) for f in media_files]

    behaviors_conf = observation.get("behaviors_conf")  # rarely per-observation
    beh_type = _build_behavior_type_lookup(behaviors_conf)

    open_starts: dict[tuple[str, str], tuple[int, float]] = {}
    rows: list[dict] = []

    for ev in observation.get("events", []):
        if len(ev) < 3:
            continue
        t_obs = float(ev[0])
        subject = str(ev[1]) if ev[1] else "individual_0"
        behavior = str(ev[2])
        if behavior not in name_to_id:
            continue

        match = _match_event_to_file(t_obs, offsets, durations)
        if match is None:
            logger.warning("Event t=%.3f outside any media file — skipping", t_obs)
            continue
        file_idx, t_local = match

        if beh_type.get(behavior) == "Point event":
            rows.append(
                {
                    "onset_s": t_local,
                    "offset_s": float("nan"),
                    "labels": name_to_id[behavior],
                    "individual": subject,
                    "trial": file_idx + 1,
                    "event_type": EVENT_TYPE_POINT,
                }
            )
            continue

        key = (subject, behavior)
        if key in open_starts:
            start_file, start_local = open_starts.pop(key)
            end_local = t_local if file_idx == start_file else durations[start_file]
            rows.append(
                {
                    "onset_s": start_local,
                    "offset_s": end_local,
                    "labels": name_to_id[behavior],
                    "individual": subject,
                    "trial": start_file + 1,
                    "event_type": EVENT_TYPE_STATE,
                }
            )
            if file_idx != start_file:
                logger.warning(
                    "State %r spans file boundary (trial %d -> %d); clipped at trial %d end",
                    behavior,
                    start_file + 1,
                    file_idx + 1,
                    start_file + 1,
                )
        else:
            open_starts[key] = (file_idx, t_local)

    for (subject, behavior), (start_file, start_local) in open_starts.items():
        rows.append(
            {
                "onset_s": start_local,
                "offset_s": durations[start_file],
                "labels": name_to_id[behavior],
                "individual": subject,
                "trial": start_file + 1,
                "event_type": EVENT_TYPE_STATE,
            }
        )
        logger.warning(
            "Unclosed state %r by subject %r — closed at end of trial %d",
            behavior,
            subject,
            start_file + 1,
        )

    if not rows:
        return init_empty_labels([])

    df = pd.DataFrame(rows).sort_values(["trial", "onset_s"]).reset_index(drop=True)
    for col, default in TRIAL_META_DEFAULTS.items():
        df[col] = default
    return df


def _build_behavior_type_lookup(
    project_or_obs_behaviors: dict | None,
) -> dict[str, str]:
    """Return ``{code: "State event" | "Point event"}``."""
    if not project_or_obs_behaviors:
        return {}
    out: dict[str, str] = {}
    for entry in project_or_obs_behaviors.values():
        code = entry.get("code")
        t = entry.get("type")
        if code and t:
            out[code] = t
    return out


def resolve_media_paths(
    observation: dict,
    search_dirs: list[Path],
) -> list[str]:
    """Resolve stored media filenames to absolute paths.

    Tries, in order: the stored path (if absolute and exists), then each
    ``search_dirs`` entry joined with the basename. Returns the stored
    string unchanged if nothing resolves (caller decides what to do).
    """
    resolved: list[str] = []
    for stored in observation_media_files(observation):
        p = Path(stored)
        if p.is_absolute() and p.exists():
            resolved.append(str(p))
            continue
        for d in search_dirs:
            candidate = d / p.name
            if candidate.exists():
                resolved.append(str(candidate))
                break
        else:
            resolved.append(str(p))
    return resolved


def match_pose_files(
    pose_folder: Path,
    video_paths: list[str],
    extensions: tuple[str, ...] = (".csv", ".h5", ".hdf5", ".slp", ".parquet"),
) -> list[str]:
    """For each video, find a pose file in ``pose_folder`` whose stem contains
    the video's stem. Unmatched videos get an empty string.

    Returns an empty list if no video matches (caller should then skip pose).
    """
    candidates: list[Path] = []
    for ext in extensions:
        candidates.extend(pose_folder.glob(f"*{ext}"))

    matched = []
    for v in video_paths:
        v_stem = Path(v).stem
        hit = next((str(p) for p in candidates if v_stem in p.stem), "")
        matched.append(hit)
    return matched if any(matched) else []


class BorisLabelConverter(LabelConverter):
    """Extract labels from a single BORIS observation."""

    name = "boris"

    def __init__(
        self,
        boris_path: str | Path,
        observation_key: str | None = None,
    ) -> None:
        super().__init__()
        self._path = Path(boris_path)
        self._project = load_boris_project(self._path)
        obs_keys = list_media_observations(self._project)
        if not obs_keys:
            raise ValueError(f"No MEDIA observations in {self._path}")
        self._observation_key = observation_key or obs_keys[0]
        if self._observation_key not in self._project["observations"]:
            raise KeyError(f"Observation {self._observation_key!r} not in {[*self._project['observations']]}")
        self._label_map = build_mapping_from_labels(unique_behavior_codes(self._project))

    @property
    def project(self) -> dict:
        return self._project

    @property
    def observation(self) -> dict:
        return self._project["observations"][self._observation_key]

    def trial_table(self) -> pd.DataFrame:
        return build_trial_table(self.observation)

    def extract(self, trials_df: pd.DataFrame | None = None) -> pd.DataFrame:
        return extract_intervals(self.observation, self._label_map)
