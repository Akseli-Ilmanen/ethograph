"""A project directory: the study-level home the cover page remembers.

Qt-free. The one thing a project holds today is ``sessions/``: every drag &
drop made while a project is open lands in its own timestamped folder there
(alignment NWB, derived ``.nc``, ``local_settings.yaml``) together with a
:class:`DropRecord` naming what was dropped and the state needed to reopen
it. Without a project a drop is throwaway, as before.

Layout::

    my_study/
    └── sessions/
        └── 2026-09-06_21-47-12/
            ├── drop.yaml                 # DropRecord
            ├── alignment-xxxxxxxx.tmp.nwb
            └── .ethograph/local_settings.yaml
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

SESSIONS_DIRNAME = "sessions"
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


def sessions_dir(project: Path | str) -> Path:
    return Path(project) / SESSIONS_DIRNAME


def new_drop_dir(project: Path | str, now: datetime | None = None) -> Path:
    """Create and return ``{project}/sessions/{timestamp}``; a same-second clash gets a suffix."""
    stamp = (now or datetime.now()).strftime(_STAMP_FORMAT)
    base = sessions_dir(project)
    base.mkdir(parents=True, exist_ok=True)
    path = base / stamp
    n = 2
    while path.exists():
        path = base / f"{stamp}_{n}"
        n += 1
    path.mkdir()
    return path


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
        except ValueError:  # a suffixed clash folder, or a hand-renamed one
            when = self.created
        return f"{when} — {names}" if names else when

    def save(self, drop_dir: Path | str) -> Path:
        path = Path(drop_dir) / DROP_RECORD_FILENAME
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, drop_dir: Path | str) -> DropRecord:
        raw = yaml.safe_load((Path(drop_dir) / DROP_RECORD_FILENAME).read_text(encoding="utf-8"))
        return cls(created=str(raw["created"]), files=list(raw.get("files") or []), state=dict(raw.get("state") or {}))


def record_drop(drop_dir: Path, files: list[str], app_state) -> DropRecord:
    """Write the record for a drop that just populated *app_state*."""
    state = {name: getattr(app_state, name) for name in DROP_STATE_FIELDS}
    record = DropRecord(created=drop_dir.name, files=list(files), state=state)
    record.save(drop_dir)
    return record


def restore_drop(record: DropRecord, app_state) -> None:
    """Put a recorded drop's state back, in the order a fresh drop sets it.

    ``nc_file_path`` goes first so the drop folder's own ``local_settings.yaml``
    is the one reloaded; every other field then overrides what that restored,
    exactly as :meth:`CoverPage._populate_io_from_buckets` does for a new drop.
    """
    app_state.metadata_path = None
    for name in DROP_STATE_FIELDS:
        setattr(app_state, name, record.state.get(name))


def list_drops(project: Path | str) -> list[tuple[Path, DropRecord]]:
    """Every recorded drop under the project, newest first; folders without a record are skipped."""
    base = sessions_dir(project)
    if not base.is_dir():
        return []
    found: list[tuple[Path, DropRecord]] = []
    for folder in sorted(base.iterdir(), reverse=True):
        if folder.is_dir() and (folder / DROP_RECORD_FILENAME).is_file():
            found.append((folder, DropRecord.load(folder)))
    return found
