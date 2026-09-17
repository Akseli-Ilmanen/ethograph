"""A session is a folder. What lives in it, and where, is decided here — nowhere else.

A session folder holds ``.ethograph/`` (the alignment record and the GUI's local
settings), ``labels.tsv``, ``metadata.tsv`` and ``labels/`` (backups and prediction
runs). Feature files (``.nc``), a root ``.nwb`` or pynapple ``.npz`` files are
layers over it. A source path handed to the loader may still be a file — the
session is then that file's folder — so every derivation goes through
:func:`session_dir_of` rather than ``Path(...).parent``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SETTINGS_DIRNAME = ".ethograph"
ALIGNMENT_FILENAME = "alignment.nwb"
LABELS_FILENAME = "labels.tsv"
METADATA_FILENAME = "metadata.tsv"

#: Session files that used to be named after the ``.nc`` stem: ``(glob, canonical name)``.
_LEGACY_NAMES: tuple[tuple[str, str], ...] = (("*_labels.tsv", LABELS_FILENAME), ("*_metadata.tsv", METADATA_FILENAME))


def session_dir_of(source: str | Path) -> Path:
    """The session folder a source path belongs to: the folder itself, or a file's parent."""
    path = Path(source).resolve()
    return path if path.is_dir() else path.parent


def settings_dir(source: str | Path) -> Path:
    return session_dir_of(source) / SETTINGS_DIRNAME


def alignment_path(source: str | Path) -> Path:
    """Where the session's alignment record lives, whether or not it exists yet."""
    return settings_dir(source) / ALIGNMENT_FILENAME


def root_nwb_files(folder: str | Path) -> list[Path]:
    """``.nwb`` files at the folder's root — a session backed by its own NWB source."""
    return sorted(p for p in Path(folder).glob("*.nwb") if p.is_file())


def is_session_folder(folder: str | Path) -> bool:
    """A folder is a session iff it holds ``.ethograph/alignment.nwb`` or a root ``.nwb``."""
    folder = Path(folder)
    return folder.is_dir() and (alignment_path(folder).is_file() or bool(root_nwb_files(folder)))


def labels_path(source: str | Path, suffix: str = "") -> Path:
    """``labels{suffix}.tsv`` in the session folder; *suffix* marks a downsampled copy."""
    return session_dir_of(source) / f"labels{suffix}.tsv"


def metadata_path(source: str | Path) -> Path:
    """``metadata.tsv`` in the session folder."""
    return session_dir_of(source) / METADATA_FILENAME


def adopt_legacy_files(source: str | Path, *, labels: bool = True, metadata: bool = True) -> list[tuple[Path, Path]]:
    """Rename a lone ``{stem}_labels.tsv`` / ``{stem}_metadata.tsv`` to the canonical name, once.

    Sessions written before labels were one file per folder named them after the
    ``.nc``. When the canonical file is absent and exactly one legacy file sits in
    the folder, it is renamed and the rename logged; several candidates are left
    alone (that is a folder of sessions, not a session) with a warning naming them.
    A kind the caller was given an explicit path for is skipped (``labels=False``,
    ``metadata=False``): a file someone points at by name must stay where it is.
    Returns the renames made.
    """
    folder = session_dir_of(source)
    renamed: list[tuple[Path, Path]] = []
    wanted = {LABELS_FILENAME: labels, METADATA_FILENAME: metadata}
    for pattern, canonical in _LEGACY_NAMES:
        if not wanted[canonical]:
            continue
        target = folder / canonical
        if target.exists():
            continue
        candidates = sorted(p for p in folder.glob(pattern) if p.is_file() and p.name != canonical)
        if len(candidates) == 1:
            candidates[0].rename(target)
            logger.info("Adopted %s as %s (one session per folder)", candidates[0].name, canonical)
            renamed.append((candidates[0], target))
        elif len(candidates) > 1:
            logger.warning(
                "%s has several %s files (%s) and no %s; none adopted — one session per folder",
                folder,
                pattern,
                ", ".join(p.name for p in candidates),
                canonical,
            )
    return renamed
