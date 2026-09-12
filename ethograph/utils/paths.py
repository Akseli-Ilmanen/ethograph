"""Path utilities with zero internal dependencies (stdlib only)."""

import hashlib
import logging
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SETTINGS_DIR = ".ethograph"

#: Base directory for per-drop throwaway alignment/.nc files. Anything under it
#: is session-scratch: it is deleted on the next drop, so a path pointing into
#: it must never be persisted as a setting (see :func:`is_throwaway_path`).
TMP_ALIGNMENT_DIRNAME = "ethograph_tmp_alignment"


def tmp_alignment_base() -> Path:
    """Return the base directory holding per-drop throwaway alignment files."""
    import tempfile

    return Path(tempfile.gettempdir()) / TMP_ALIGNMENT_DIRNAME


def is_throwaway_path(value: object) -> bool:
    """True when *value* points inside :func:`tmp_alignment_base`.

    Saving such a path would outlive the file it names: each drop wipes the
    previous drop's directory, so a persisted reference either dangles or,
    worse, resolves to a different session's alignment.
    """
    if not isinstance(value, (str, Path)):
        return False
    try:
        return tmp_alignment_base().resolve() in Path(value).resolve().parents
    except (OSError, ValueError):
        return False


#: Environment variable that overrides the location of the global settings
#: directory.  When set, its value is used verbatim as the ``.ethograph``
#: settings directory instead of ``~/.ethograph``.
ETHOGRAPH_HOME_ENV = "ETHOGRAPH_HOME"


def ethograph_home() -> Path:
    """Return the global settings directory (the ``.ethograph`` folder).

    Every global (user-level) config, download, and cache location in ethograph
    is anchored here.  Resolution order:

    1. ``$ETHOGRAPH_HOME`` — if set, used verbatim (``~`` is expanded).  This is
       the escape hatch for CI, automation harnesses, or relocating settings to
       another drive, and avoids having to fake ``USERPROFILE``/``HOME`` so that
       :func:`pathlib.Path.home` resolves correctly.
    2. ``~/.ethograph`` — the default for a normal interactive session.

    The directory is *not* created; callers that write into it create it as
    needed.  On Windows, :func:`pathlib.Path.home` raises ``RuntimeError`` when
    none of ``USERPROFILE``/``HOME``/``HOMEDRIVE`` are set (there is no
    password-database fallback as on POSIX), so setting ``$ETHOGRAPH_HOME`` is
    the robust way to run in a stripped environment.

    Returns
    -------
    Path
        Absolute path to the ``.ethograph`` settings directory.

    Examples
    --------
    >>> import ethograph as eto
    >>> eto.ethograph_home()  # doctest: +SKIP
    PosixPath('/home/user/.ethograph')
    """
    override = os.environ.get(ETHOGRAPH_HOME_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / SETTINGS_DIR


# --- Home layout -------------------------------------------------------------
#
# ~/.ethograph/
#   gui_settings.yaml   the user's viewing habits (layout, paths, playback)
#   logs/               per-machine session logs
#   cache/              derived media keyed by content, safe to wipe
#   defaults/           a project directory's layout, used while no project
#                       is open: mapping.txt, config/space/, runs/lightgbm/,
#                       workflows/, wizard/
#
# Everything about a *study* is meant to live in a project directory; these
# are the fallbacks for a session opened without one, in the same shape.

CACHE_DIRNAME = "cache"
DEFAULTS_DIRNAME = "defaults"
LOGS_DIRNAME = "logs"

#: Old home-level folders/files and where they live now, relative to the home
#: directory.  Read by :func:`migrate_home_layout` once per start.
HOME_LAYOUT_MOVES: dict[str, str] = {
    "proxies": f"{CACHE_DIRNAME}/proxies",
    "audio_tracks": f"{CACHE_DIRNAME}/audio_tracks",
    "example_data": f"{CACHE_DIRNAME}/example_data",
    "dandi": f"{CACHE_DIRNAME}/dandi",
    "models/cotracker": f"{CACHE_DIRNAME}/weights/cotracker",
    "models": f"{DEFAULTS_DIRNAME}/runs/lightgbm",
    "workflows": f"{DEFAULTS_DIRNAME}/workflows",
    "geometries": f"{DEFAULTS_DIRNAME}/config/space",
    "mapping.txt": f"{DEFAULTS_DIRNAME}/mapping.txt",
    "alignment_wizard": f"{DEFAULTS_DIRNAME}/wizard",
}


def cache_dir(name: str | None = None) -> Path:
    """``~/.ethograph/cache[/name]`` — derived media, keyed by content, never by project."""
    base = ethograph_home() / CACHE_DIRNAME
    return base / name if name else base


def defaults_dir(name: str | None = None) -> Path:
    """``~/.ethograph/defaults[/name]`` — study assets used while no project is open."""
    base = ethograph_home() / DEFAULTS_DIRNAME
    return base / name if name else base


def logs_dir() -> Path:
    """``~/.ethograph/logs``."""
    return ethograph_home() / LOGS_DIRNAME


#: The project folder every install starts from, shipped as package data
#: (``ethograph/defaults/``: mapping.txt, config/segment.yaml, config/spot.yaml,
#: config/space/*.yaml).
BUNDLED_DEFAULTS_DIR = Path(__file__).resolve().parents[1] / "defaults"


def seed_defaults(dest: Path | None = None) -> list[Path]:
    """Copy every bundled default into *dest* (``defaults_dir()``) that is not there yet.

    File by file, so an example added in a later release reaches an existing
    install; a file the user edited is never overwritten. ``README.md`` is
    documentation for the repo, not a default, and is skipped.

    Returns
    -------
    list of Path
        The files written.
    """
    import shutil

    dest = defaults_dir() if dest is None else dest
    written: list[Path] = []
    for src in sorted(BUNDLED_DEFAULTS_DIR.rglob("*")):
        if not src.is_file() or src.name == "README.md":
            continue
        target = dest / src.relative_to(BUNDLED_DEFAULTS_DIR)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        written.append(target)
    return written


def migrate_home_layout(home: Path | None = None) -> list[tuple[Path, Path]]:
    """Move pre-layout folders under ``cache/`` and ``defaults/``.

    Each entry of :data:`HOME_LAYOUT_MOVES` is moved only when the old path
    exists and the new one does not; a destination that already exists is left
    alone and the old path is kept, so nothing is ever overwritten.  Entries
    are applied in order, which is why ``models/cotracker`` precedes
    ``models``.

    Returns
    -------
    list of (old, new)
        The moves performed.
    """
    import shutil

    home = ethograph_home() if home is None else home
    moved: list[tuple[Path, Path]] = []
    for old_rel, new_rel in HOME_LAYOUT_MOVES.items():
        old = home / old_rel
        new = home / new_rel
        if not old.exists() or new.exists():
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
        logger.info("Moved %s -> %s", old, new)
        moved.append((old, new))
    return moved


def get_project_root(start: Path | None = None) -> Path:
    """Find the repository root by walking up from *start* until ``pyproject.toml`` is found.

    Parameters
    ----------
    start : Path, optional
        Directory to start searching from.  Defaults to the current
        working directory.

    Returns
    -------
    Path
        Absolute path to the project root.

    Raises
    ------
    FileNotFoundError
        If no ``pyproject.toml`` is found in any ancestor directory.

    Examples
    --------
    >>> import ethograph as eto
    >>> eto.get_project_root()
    PosixPath('/home/user/code/ethograph')
    """
    if start is not None:
        path = start.resolve()
    else:
        path = Path.cwd().resolve()
    for parent in [path] + list(path.parents):
        if (parent / "pyproject.toml").exists():
            if parent.parent.name != "deps":
                return parent
            continue
    fallback = Path(__file__).resolve()
    for parent in fallback.parents:
        if (parent / "pyproject.toml").exists():
            if parent.parent.name != "deps":
                return parent
            continue
    raise FileNotFoundError(f"Could not find project root starting from {path}")


def extract_pattern_groups(
    filenames: list[str | Path], pattern: str, convert_numeric: bool = True
) -> list[dict[str, str | int]]:
    """Extract named groups from filenames using a regex pattern.

    Parameters
    ----------
    filenames
        List of file paths
    pattern
        Regex pattern with named groups, e.g.:
        r'(?P<camera>cam[12])_trial(?P<trial>\\d+)\\.mp4'
    convert_numeric
        If True, automatically convert purely numeric strings to integers
        (e.g., "001" -> 1, "100" -> 100)

    Returns
    -------
    List of dicts mapping group names to extracted values (str or int)

    Examples
    --------
    >>> files = ["cam1_trial001.mp4", "cam2_trial001.mp4"]
    >>> pattern = r"(?P<camera>cam[12])_trial(?P<trial>\\d+)\\.mp4"
    >>> extract_pattern_groups(files, pattern, convert_numeric=True)
    [{'camera': 'cam1', 'trial': 1}, {'camera': 'cam2', 'trial': 1}]

    >>> extract_pattern_groups(files, pattern, convert_numeric=False)
    [{'camera': 'cam1', 'trial': '001'}, {'camera': 'cam2', 'trial': '001'}]
    """
    regex = re.compile(pattern)
    results = []
    for f in filenames:
        fname = Path(f).name
        match = regex.search(fname)
        if match:
            groups = match.groupdict()
            if convert_numeric:
                groups = {k: int(v) if v.isdigit() else v for k, v in groups.items()}
            results.append(groups)
    return results


def check_paths_exist(nc_paths):
    missing_paths = [p for p in nc_paths if not os.path.exists(p)]
    if missing_paths:
        print("Error: The following test_nc_paths do not exist:")
        for p in missing_paths:
            print(f"  {p}")
        exit(1)


def session_id(source_path: str | Path) -> str:
    """Stable, filesystem-safe identifier for one loaded session.

    ``{stem}-{hash}``: the stem keeps the folder recognisable, the hash of the
    resolved path keeps two same-named sessions apart.
    """
    p = Path(source_path)
    digest = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem) or "session"
    return f"{stem}-{digest}"


def media_cache_key(media_path: Path | str, recipe_version: int) -> str:
    """Deterministic cache key from source identity (path, size, mtime).

    Shared by every derived-media cache (video proxies, extracted audio
    tracks): moving, renaming, or re-recording the source yields a new key, so
    a stale derivative is never silently reused for changed media. Bumping the
    caller's *recipe_version* invalidates everything it wrote before.
    """
    media_path = Path(media_path)
    st = media_path.stat()
    raw = f"{media_path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{recipe_version}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{media_path.stem}_{digest}"


def path_exists(value: str, kind: str = "any") -> bool:
    """Whether *value* names something of *kind* ("file", "dir" or "any") on this machine."""
    if not value:
        return False
    path = Path(value).expanduser()
    if kind == "file":
        return path.is_file()
    if kind == "dir":
        return path.is_dir()
    return path.exists()


def sanitize_path_state(state: dict[str, Any], path_kinds: Mapping[str, str]) -> dict[str, Any]:
    """Drop entries of *state* whose path does not exist on this machine.

    Settings files travel: a dataset folder is copied to another pc, an
    external drive is unplugged, a shared ``gui_settings.yaml`` is reused
    elsewhere.  A restored path that names nothing here must never reach the
    media resolvers — they would report a missing video/pose/audio file for
    every trial, blaming the data rather than the stale setting.

    Parameters
    ----------
    state : dict
        Settings mapping; left untouched, the copy is returned.
    path_kinds : Mapping[str, str]
        Keys of *state* holding a path, mapped to what must exist for the
        value to be usable: ``"file"``, ``"dir"`` or ``"any"``.  List values
        are filtered element-wise, and the key dropped when nothing survives.

    Returns
    -------
    dict
        Copy of *state* without the unusable path entries.
    """
    cleaned = dict(state)
    for key, kind in path_kinds.items():
        value = cleaned.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            kept = [v for v in value if path_exists(v, kind)]
            if len(kept) == len(value):
                continue
            logger.info("Setting %r: dropping %d path(s) missing on this machine", key, len(value) - len(kept))
            if kept:
                cleaned[key] = type(value)(kept)
            else:
                del cleaned[key]
        elif not path_exists(value, kind):
            logger.info("Ignoring setting %r: %s does not exist on this machine", key, value)
            del cleaned[key]
    return cleaned


def find_config(name: str, data_dir: Path | str | None = None, project_dir: Path | str | None = None) -> Path | None:
    """Find a config file: beside the data, else the project's, else the bundled default.

    Most specific wins:
    1. Walk up from *data_dir* looking for ``.ethograph/{name}`` in each
       ancestor — a session's own copy overrides the study's.
    2. ``{project_dir}/{name}`` — the study's copy, when a project folder is
       chosen on the cover page.
    3. ``~/.ethograph/defaults/{name}`` — the backup every install ships with.

    Parameters
    ----------
    name
        Config filename, e.g. ``"mapping.txt"`` or ``"arena.yaml"``.
    data_dir
        Directory of the loaded data file.  Pass ``None`` to skip the
        walk-up search (e.g. at application startup before any file is loaded).
    project_dir
        The project folder (``app_state.project_path``), or ``None``.

    Returns
    -------
    First existing path, or ``None`` if none are found.

    Examples
    --------
    >>> find_config("mapping.txt", "/data/session_01", project_dir="/data/my_study")
    PosixPath('/data/my_study/mapping.txt')
    """
    if data_dir is not None:
        d = Path(data_dir).resolve()
        for parent in [d] + list(d.parents):
            candidate = parent / SETTINGS_DIR / name
            if candidate.exists():
                return candidate

    if project_dir is not None:
        candidate = Path(project_dir) / name
        if candidate.is_file():
            return candidate

    global_candidate = defaults_dir(name)
    if global_candidate.exists():
        return global_candidate

    return None


def default_config_dir(data_dir: Path | str | None = None) -> Path:
    """Return the ``.ethograph/`` directory where new configs should be written.

    Uses ``data_dir/.ethograph/`` when a data directory is known, otherwise
    falls back to ``~/.ethograph/defaults/``.
    """
    if data_dir is not None:
        return Path(data_dir) / SETTINGS_DIR
    return defaults_dir()


def find_mapping_file(data_dir: Path | str | None = None, project_dir: Path | str | None = None) -> Path | None:
    """Find mapping.txt. Convenience wrapper around :func:`find_config`."""
    return find_config("mapping.txt", data_dir, project_dir)


def find_nwb_file(data_dir: Path | str | None = None) -> Path | None:
    """Find alignment.nwb in ``.ethograph/`` relative to *data_dir*.

    Search order:
    1. ``<data_dir>/.ethograph/alignment.nwb``
    2. Any ``.nwb`` file in ``<data_dir>/.ethograph/``
    3. Walk up parent directories looking for ``.ethograph/alignment.nwb``

    Returns
    -------
    Path or None
    """
    if data_dir is None:
        return None
    d = Path(data_dir).resolve()

    # Check immediate .ethograph directory
    ethograph_dir = d / SETTINGS_DIR
    if ethograph_dir.is_dir():
        candidate = ethograph_dir / "alignment.nwb"
        if candidate.exists():
            return candidate
        nwb_files = list(ethograph_dir.glob("*.nwb"))
        if nwb_files:
            return nwb_files[0]

    # Check one parent up only
    parent = d.parent
    if parent != d:
        ethograph_dir = parent / SETTINGS_DIR
        if ethograph_dir.is_dir():
            candidate = ethograph_dir / "alignment.nwb"
            if candidate.exists():
                return candidate
            nwb_files = list(ethograph_dir.glob("*.nwb"))
            if nwb_files:
                return nwb_files[0]

    return None


def extract_trial_info_from_filename(path):
    """
    Extract session_date, trial_num, and bird from a DLC filename.
    Expected filename format: YYYY-MM-DD_NNN_Bird_...
    """
    filename = os.path.basename(path)
    parts = filename.split("_")
    if len(parts) >= 3:
        session_date = parts[0]
        trial_num = int(parts[1])
        bird = parts[2]
        return session_date, trial_num, bird
    else:
        raise ValueError(f"Filename format not recognized: {filename}")


def auto_git_commit(label_path: Path) -> None:
    """Auto-commit a label file if it lives inside a git repository."""
    file_dir = str(label_path.parent)
    try:
        root_result = subprocess.run(
            ["git", "-C", file_dir, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise ValueError("git is not installed or not found on PATH.")

    if root_result.returncode != 0:
        raise ValueError(
            f"Remote backup folder is not inside a git repository: {file_dir}\n"
            f"git said: {root_result.stderr.strip()}\n"
            "Run 'git init' in an ancestor folder first, or use 'Save with timestamp' mode."
        )

    repo_root = Path(root_result.stdout.strip())
    rel_path = label_path.relative_to(repo_root)

    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "add", str(rel_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        commit_result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "commit",
                "-m",
                f"Labels updated: {label_path.name}",
            ],
            capture_output=True,
            text=True,
        )
        if commit_result.returncode != 0:
            msg = commit_result.stderr.strip() or commit_result.stdout.strip()
            if "nothing to commit" in msg:
                logger.info("git: nothing to commit for %s", label_path.name)
                return
            raise subprocess.CalledProcessError(
                commit_result.returncode,
                "git commit",
                output=commit_result.stdout,
                stderr=commit_result.stderr,
            )
        subprocess.run(
            ["git", "-C", str(repo_root), "push"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Auto-committed and pushed %s to git", label_path.name)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"git commit/push failed: {e.stderr.strip() or e.stdout.strip() or str(e)}")
