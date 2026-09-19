"""The skeleton library: one YAML file per skeleton, selected by its file name.

Qt-free. A skeleton is a study-level answer ("this is what a crow looks like"),
not a per-session one, so it lives in a file next to the other study assets —
the same shape as the Space plot's geometry library (``config/space/``), and for
the same reason: a study with a crow rig and a mouse rig has *two* skeletons, and
one inline blob in one settings file cannot hold both.

Two directories are read, nearest first:

* ``{project}/config/skeleton/*.yaml`` — this study's, shareable and diffable;
* ``~/.ethograph/defaults/config/skeleton/*.yaml`` — the user's own, available
  with no project folder at all.

A name that exists in both resolves to the project's. Which one is drawn is
``app_state.skeleton_name``; a skeleton carried by the data (an NWB's own) wins
by default — see ``PoseDisplayManager._resolved_skeleton_config``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from ethograph.utils.paths import defaults_dir

logger = logging.getLogger(__name__)

#: Where a library lives inside a project folder (and inside the home defaults).
SKELETON_DIRNAME = "skeleton"


def library_dirs(project: Path | str | None) -> list[Path]:
    """The directories searched, nearest first: the project's, then the user's own."""
    dirs: list[Path] = []
    if project is not None:
        dirs.append(Path(project) / "config" / SKELETON_DIRNAME)
    dirs.append(defaults_dir("config") / SKELETON_DIRNAME)
    return dirs


def load_skeletons(project: Path | str | None = None) -> dict[str, dict]:
    """Every skeleton in the library, keyed by file stem, nearest directory winning.

    An unparsable file is skipped with a log line, never raised: the library is
    user-supplied input, and one bad file must not cost the others.
    """
    found: dict[str, dict] = {}
    for folder in reversed(library_dirs(project)):  # farthest first, so nearest overwrites
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.y*ml")):
            try:
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                logger.exception("Skipping unreadable skeleton %s", path)
                continue
            if isinstance(config, dict) and "connections" in config:
                found[path.stem] = config
            else:
                logger.warning("%s is not a skeleton (needs a 'connections' list), skipping", path)
    return found


def save_skeleton(name: str, config: dict, project: Path | str | None = None) -> Path:
    """Write *config* as ``{name}.yaml`` in the nearest library, creating it.

    The project's library when a project is set, so the study shares it; the
    user's own otherwise, so drawing a skeleton never requires a project folder.
    """
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"a skeleton's name is one file name, got {name!r}")
    folder = library_dirs(project)[0]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def delete_skeleton(name: str, project: Path | str | None = None) -> bool:
    """Remove ``{name}.yaml`` from the nearest library it is in; ``False`` when it was in none."""
    for folder in library_dirs(project):
        path = folder / f"{name}.yaml"
        if path.is_file():
            path.unlink()
            return True
    return False


def resolve_skeleton(name: str | None, project: Path | str | None = None) -> dict | None:
    """The named skeleton, or — when nothing is named and the library holds exactly one — that one.

    A library of one needs no choosing, which is the common case: a study has a
    single animal and drew its skeleton once.
    """
    library = load_skeletons(project)
    if name and name in library:
        return library[name]
    if not name and len(library) == 1:
        return next(iter(library.values()))
    return None
