"""A project directory: the study-level home the cover page remembers.

Qt-free. A drag & drop makes a session out of the folder the files came from
(or the folder the user picks when they came from several): that folder gets
``.ethograph/`` with the alignment NWB, any derived ``.nc``, the drop's
:class:`DropRecord` and its ``local_settings.yaml``. Data never moves. The
project keeps a registry of the session folders dropped while it was open, so
the cover page can list and reopen them.

**The project folder holds files, never a settings file.** ``mapping.txt``, the
pipeline configs under ``config/``, the skeleton library, the runs — each is read
from its own path. What used to be ``project.yaml`` is gone: the individuals and
the ignored-file globs are the user's (``gui_settings.yaml``, so they follow the
person rather than a folder that may be copied between machines), the skeleton is
one YAML per skeleton, and the rest were defaults the GUI remembers by itself.
:func:`migrate_project_yaml` folds an old file's lists into the global settings.

Layout::

    my_study/
    ├── mapping.txt                   # the label vocabulary
    ├── config/skeleton/*.yaml        # the skeleton library
    └── sessions.txt                  # one session folder per line, oldest first

    D:/rig/2026-09-06/                # the dropped folder, now a session
    ├── cam0.mp4, cam1.mp4, ...
    └── .ethograph/
        ├── alignment.nwb
        ├── drop.yaml                 # DropRecord
        └── local_settings.yaml
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from ethograph.utils.paths import SETTINGS_DIR

logger = logging.getLogger(__name__)

SESSIONS_REGISTRY_FILENAME = "sessions.txt"
DROP_RECORD_FILENAME = "drop.yaml"
_STAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

#: App-state fields a drop sets and a reopen restores, in that order. Every
#: entry is a plain scalar/list so the record round-trips through YAML.
DROP_STATE_FIELDS: tuple[str, ...] = (
    "nc_file_path",
    "nwb_file_path",
    "video_folder",
    "audio_folder",
    "pose_folder",
    "ephys_path",
    "neurons_path",
    "image_paths",
    "primary_camera",
    "extra_cameras",
    "source_software",
    "labels_import_path",
)


def project_dir_of(app_state) -> Path | None:
    """The chosen project folder, or ``None`` when none is set or it no longer exists."""
    value = getattr(app_state, "project_path", None)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_dir() else None


#: The settings file older versions kept in the project folder. Nothing reads it
#: any more: individuals and ignored files are the user's (``gui_settings.yaml``),
#: the skeleton is a file of its own, and ``rig`` / ``pose.source_software`` were
#: defaults the GUI now remembers by itself. :func:`migrate_project_yaml` folds an
#: existing one into the global settings once, so no study loses its lists.
LEGACY_SETTINGS_FILENAME = "project.yaml"


def migrate_project_yaml(app_state) -> list[str]:
    """Fold a legacy ``project.yaml``'s lists into the global settings, once.

    Returns the names of the settings that gained entries, so the caller can say
    so; ``[]`` when there is nothing to migrate. The file is left on disk — it is
    the user's, and deleting something we no longer own is not our call.
    """
    project = project_dir_of(app_state)
    if project is None:
        return []
    path = project / LEGACY_SETTINGS_FILENAME
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        logger.warning("Could not read the legacy %s; nothing migrated", path)
        return []
    if not isinstance(raw, dict):
        return []

    moved: list[str] = []
    for key, setting in (("individuals", "extra_individuals"), ("ignore", "ignore_files")):
        values = raw.get(key)
        if not isinstance(values, list):
            continue
        current = [str(v) for v in app_state.get_with_default(setting)]
        added = [str(v) for v in values if str(v) not in current]
        if added:
            setattr(app_state, setting, current + added)
            moved.append(setting)
    if moved:
        logger.info("Migrated %s from %s into the global settings", moved, path)
    return moved


def drop_record_path(session_dir: Path | str) -> Path:
    return Path(session_dir) / SETTINGS_DIR / DROP_RECORD_FILENAME


def is_foreign_session(folder: Path | str) -> bool:
    """Whether *folder* is a session someone made on purpose, which a drop must ask about.

    A folder with ``.ethograph/alignment.nwb`` but no drop record was set up by the
    wizard, a notebook or by hand; a drop re-pairing into it would replace their
    alignment, so the cover page asks first. A folder whose alignment came from a
    drop is regenerable and replaced silently, and a bare ``.ethograph/`` (settings
    only) makes nothing a session.
    """
    folder = Path(folder)
    return (folder / SETTINGS_DIR / "alignment.nwb").is_file() and not drop_record_path(folder).is_file()


@dataclass
class DropRecord:
    """What was dropped and the app state that reopens it."""

    created: str
    files: list[str]
    state: dict[str, object] = field(default_factory=dict)

    @property
    def title(self) -> str:
        """``"2026-09-06 21:47 — cam0.mp4, mic.wav"``: the row a reopen list shows."""
        names = ", ".join(Path(f).name for f in self.files)
        try:
            when = datetime.strptime(self.created, _STAMP_FORMAT).strftime("%Y-%m-%d %H:%M")
        except ValueError:  # a hand-edited record
            when = self.created
        return f"{when} — {names}" if names else when

    def save(self, session_dir: Path | str) -> Path:
        path = drop_record_path(session_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, session_dir: Path | str) -> DropRecord:
        raw = yaml.safe_load(drop_record_path(session_dir).read_text(encoding="utf-8"))
        return cls(created=str(raw["created"]), files=list(raw.get("files") or []), state=dict(raw.get("state") or {}))


def record_drop(session_dir: Path, files: list[str], app_state, now: datetime | None = None) -> DropRecord:
    """Write the record for a drop that just populated *app_state* into *session_dir*."""
    state = {name: getattr(app_state, name) for name in DROP_STATE_FIELDS}
    record = DropRecord(created=(now or datetime.now()).strftime(_STAMP_FORMAT), files=list(files), state=state)
    record.save(session_dir)
    return record


def restore_drop(record: DropRecord, app_state) -> None:
    """Put a recorded drop's state back, in the order a fresh drop sets it.

    ``nc_file_path`` goes first so the session folder's own ``local_settings.yaml``
    is the one reloaded; every other field then overrides what that restored,
    exactly as :meth:`CoverPage._populate_io_from_buckets` does for a new drop.
    """
    app_state.metadata_path = None
    for name in DROP_STATE_FIELDS:
        setattr(app_state, name, record.state.get(name))


def register_session(project: Path | str, session_dir: Path | str) -> None:
    """Add *session_dir* to the project's registry (moved to the end when already there)."""
    registry = Path(project) / SESSIONS_REGISTRY_FILENAME
    entry = str(Path(session_dir).resolve())
    lines = [ln for ln in _read_registry(registry) if ln != entry]
    lines.append(entry)
    registry.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_registry(registry: Path) -> list[str]:
    if not registry.is_file():
        return []
    return [ln.strip() for ln in registry.read_text(encoding="utf-8").splitlines() if ln.strip()]


def list_drops(project: Path | str) -> list[tuple[Path, DropRecord]]:
    """Every registered session that still has its drop record, newest first."""
    found: list[tuple[Path, DropRecord]] = []
    for entry in reversed(_read_registry(Path(project) / SESSIONS_REGISTRY_FILENAME)):
        folder = Path(entry)
        if drop_record_path(folder).is_file():
            found.append((folder, DropRecord.load(folder)))
    return found
