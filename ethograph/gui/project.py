"""A project directory: the study-level home the cover page remembers.

Qt-free. A drag & drop makes a session out of the folder the files came from
(or the folder the user picks when they came from several): that folder gets
``.ethograph/`` with the alignment NWB, any derived ``.nc``, the drop's
:class:`DropRecord` and its ``local_settings.yaml``. Data never moves. The
project keeps a registry of the session folders dropped while it was open, so
the cover page can list and reopen them.

The project also holds ``project.yaml``: the study's defaults for what recurs
across its sessions (who the individuals are, which cameras and mics a rig has,
which pose software, the skeleton to draw — written inline, not in a file of its
own). A session's own record or dataset overrides them; they are defaults, never
claims about a session. Machine paths never go in it.

Layout::

    my_study/
    ├── project.yaml                  # ProjectSettings
    └── sessions.txt                  # one session folder per line, oldest first

    D:/rig/2026-09-06/                # the dropped folder, now a session
    ├── cam0.mp4, cam1.mp4, ...
    └── .ethograph/
        ├── alignment.nwb
        ├── drop.yaml                 # DropRecord
        └── local_settings.yaml
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from ethograph.utils.paths import SETTINGS_DIR

SESSIONS_REGISTRY_FILENAME = "sessions.txt"
PROJECT_SETTINGS_FILENAME = "project.yaml"
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


@dataclass(frozen=True)
class ProjectSettings:
    """The study-level defaults ``project.yaml`` holds. Every field optional; empty means "no default"."""

    individuals: tuple[str, ...] = ()
    cameras: tuple[str, ...] = ()
    mics: tuple[str, ...] = ()
    rig: str | None = None  # default rig notebook name under wizard/
    pose_software: str | None = None  # default tracking tool, e.g. "DeepLabCut"
    skeleton: dict | None = None  # the skeleton itself: {keypoints: [...], connections: [{start, end, ...}]}
    #: File-name globs never read from a session folder's root — old versions of a
    #: dataset (``Trial_data.nc`` beside ``Trial_data3.nc``). A file named explicitly
    #: by ``source:`` is still loaded.
    ignore: tuple[str, ...] = ()


_PROJECT_KEYS = {"individuals", "cameras", "mics", "rig", "pose", "ignore"}
_POSE_KEYS = {"source_software", "skeleton"}


def load_project_settings(project: Path | str | None) -> ProjectSettings:
    """Read ``{project}/project.yaml``; no project or no file gives the defaults.

    An unknown key is a typo, not a preference, and is refused by name.
    """
    if project is None:
        return ProjectSettings()
    path = Path(project) / PROJECT_SETTINGS_FILENAME
    if not path.is_file():
        return ProjectSettings()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping of settings, got {type(raw).__name__}")
    unknown = sorted(set(raw) - _PROJECT_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; known: {sorted(_PROJECT_KEYS)}")
    pose = raw.get("pose") or {}
    if not isinstance(pose, dict):
        raise ValueError(f"{path}: 'pose' must be a mapping with {sorted(_POSE_KEYS)}")
    unknown_pose = sorted(set(pose) - _POSE_KEYS)
    if unknown_pose:
        raise ValueError(f"{path}: unknown pose keys {unknown_pose}; known: {sorted(_POSE_KEYS)}")
    skeleton = pose.get("skeleton")
    if skeleton is not None and not (isinstance(skeleton, dict) and "connections" in skeleton):
        raise ValueError(f"{path}: 'pose.skeleton' is the skeleton itself, a mapping with 'connections'")
    return ProjectSettings(
        individuals=_names(raw.get("individuals"), path, "individuals"),
        cameras=_names(raw.get("cameras"), path, "cameras"),
        mics=_names(raw.get("mics"), path, "mics"),
        rig=str(raw["rig"]) if raw.get("rig") else None,
        pose_software=str(pose["source_software"]) if pose.get("source_software") else None,
        skeleton=dict(skeleton) if skeleton else None,
        ignore=_names(raw.get("ignore"), path, "ignore"),
    )


def _names(value, path: Path, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, (str, int)) for v in value):
        raise ValueError(f"{path}: '{key}' must be a list of names")
    names = tuple(str(v) for v in value)
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: '{key}' repeats a name: {list(names)}")
    return names


def project_settings_of(app_state) -> ProjectSettings:
    """The chosen project's settings, or the defaults when no project is set."""
    return load_project_settings(project_dir_of(app_state))


def find_project_dir(start: Path | str) -> Path | None:
    """The nearest folder at or above *start* holding ``project.yaml`` — how a pipeline finds its project."""
    here = Path(start).resolve()
    for folder in (here, *here.parents):
        if (folder / PROJECT_SETTINGS_FILENAME).is_file():
            return folder
    return None


def add_ignore(project: Path | str, name: str) -> Path:
    """Append *name* to the project's ``ignore`` list, creating ``project.yaml`` if needed."""
    path = Path(project) / PROJECT_SETTINGS_FILENAME
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    raw = dict(raw or {})
    names = [str(n) for n in raw.get("ignore") or []]
    if name and name not in names:
        names.append(name)
    raw["ignore"] = names
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def drop_record_path(session_dir: Path | str) -> Path:
    return Path(session_dir) / SETTINGS_DIR / DROP_RECORD_FILENAME


def is_foreign_session(folder: Path | str) -> bool:
    """Whether *folder* is a session someone made on purpose, which a drop must not overwrite.

    A folder with ``.ethograph/alignment.nwb`` but no drop record was set up by the
    wizard, a notebook or by hand; a drop re-pairing into it would replace their
    alignment. A folder whose alignment came from a drop is regenerable and fair
    game, and a bare ``.ethograph/`` (settings only) makes nothing a session.
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
